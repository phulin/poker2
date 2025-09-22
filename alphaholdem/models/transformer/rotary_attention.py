"""Rotary Position Embedding (RoPE) attention implementation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension by half for RoPE application."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query/key pair."""
    cos = cos.to(dtype=q.dtype)
    sin = sin.to(dtype=q.dtype)

    # Slice cos and sin to match query sequence length
    L_q = q.shape[-2]  # query sequence length
    cos = cos[..., :L_q, :]  # slice to match query length
    sin = sin[..., :L_q, :]  # slice to match query length

    # Apply RoPE
    q_rotated = (q * cos) + (rotate_half(q) * sin)
    k_rotated = (k * cos) + (rotate_half(k) * sin)

    return q_rotated, k_rotated


class RotarySelfAttention(nn.Module):
    """Multi-head self-attention with RoPE positional encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert (
            self.d_head % 2 == 0
        ), "Rotary embeddings require an even projected head dimension"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = dropout

    def create_empty_cache(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create empty KV cache for the given batch size and device."""
        empty_k = torch.zeros(batch_size, self.n_heads, 0, self.d_head, device=device)
        empty_v = torch.zeros(batch_size, self.n_heads, 0, self.d_head, device=device)
        return empty_k, empty_v

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with KV caching support.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Boolean mask of shape (batch_size, seq_len) where True means allow
            cos: Cosine values for RoPE of shape (seq_len, d_head)
            sin: Sine values for RoPE of shape (seq_len, d_head)
            kv_cache: Optional cached key-value pairs from previous forward passes

        Returns:
            Tuple of (output_tensor, new_kv_cache)
            - output_tensor: Attention output of shape (batch_size, seq_len, d_model)
            - new_kv_cache: Updated cache for next forward pass
        """

        # x: [B, L_q, d_model]
        # B: batch size, L_q: sequence length
        B, L_q, _ = x.shape
        H = self.n_heads

        q = (
            self.q_proj(x).view(B, L_q, H, self.d_head).permute(0, 2, 1, 3)
        )  # [B,H,L_q,D]
        k = (
            self.k_proj(x).view(B, L_q, H, self.d_head).permute(0, 2, 1, 3)
        )  # [B,H,L_q,D]
        v = self.v_proj(x).view(B, L_q, H, self.d_head).permute(0, 2, 1, 3)

        # Apply RoPE to current chunk q/k
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        L_cache = 0
        if kv_cache is not None:
            cached_k, cached_v = kv_cache  # [B,H,L_cache,D]
            L_cache = cached_k.shape[2]
            # Concatenate cached keys/values first (key length grows)
            k = torch.cat([cached_k, k], dim=2)  # [B,H,L_cache+L_q,D]
            v = torch.cat([cached_v, v], dim=2)  # [B,H,L_cache+L_q,D]

        # Always return updated cache (the *full* K/V we just built)
        new_kv_cache = (k, v)

        L_k = L_cache + L_q  # total key length

        # ---- Build SDPA boolean attn mask (True = ALLOW) for keys ----
        # Caller supplies a *current-chunk* allow mask: [B, L_q], True = allow attention.
        assert attention_mask.shape == (
            B,
            L_q,
        ), f"attention_mask must be [B,L_q], got {attention_mask.shape} vs L_q={L_q}"

        if L_cache > 0:
            # Cached keys are always valid keys (allow attention): ones
            cached_allow = torch.ones(B, L_cache, dtype=torch.bool, device=x.device)
            key_allow_mask = torch.cat(
                [cached_allow, attention_mask], dim=1
            )  # [B, L_k]
        else:
            key_allow_mask = attention_mask  # [B, L_k == L_q]

        assert key_allow_mask.shape == (
            B,
            L_k,
        ), f"key_allow_mask {key_allow_mask.shape} must match total key length L_k={L_k}"

        # Expand to [BH, L_q, L_k] (same key mask for every query position)
        key_allow_mask_bh = (
            key_allow_mask.unsqueeze(1).expand(B, H, L_k).reshape(B * H, L_k)
        )
        attn_mask_3d = key_allow_mask_bh.unsqueeze(1).expand(B * H, L_q, L_k)

        # Optional: safety check — don't allow fully blocked rows
        # Check if any row has ALL False values (completely blocked)
        if (key_allow_mask_bh.sum(dim=1) == 0).any():
            raise RuntimeError(
                "Attention mask blocks all keys for at least one (B*H) row."
            )

        # Reshape q/k/v for SDPA
        q = q.reshape(B * H, L_q, self.d_head)
        k = k.reshape(B * H, L_k, self.d_head)
        v = v.reshape(B * H, L_k, self.d_head)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_3d,  # True = allow attention
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        context = context.view(B, H, L_q, self.d_head).permute(0, 2, 1, 3).contiguous()
        context = context.view(B, L_q, self.d_model)
        return self.out_proj(context), new_kv_cache
