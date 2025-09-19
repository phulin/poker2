"""Rotary Position Embedding (RoPE) attention implementation."""

from typing import Tuple, Optional

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
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


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
            attention_mask: Boolean mask of shape (batch_size, seq_len) where True means attend
            cos: Cosine values for RoPE of shape (seq_len, d_head)
            sin: Sine values for RoPE of shape (seq_len, d_head)
            kv_cache: Optional cached key-value pairs from previous forward passes

        Returns:
            Tuple of (output_tensor, new_kv_cache)
            - output_tensor: Attention output of shape (batch_size, seq_len, d_model)
            - new_kv_cache: Updated cache for next forward pass
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply rotary position embedding
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Handle KV caching
        if kv_cache is not None:
            # Concatenate with cached keys and values
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # dim=2 is the sequence dimension
            v = torch.cat([cached_v, v], dim=2)

        # Always return updated cache
        new_kv_cache = (k, v)

        # Get the actual sequence length after caching
        actual_seq_len = k.shape[2]

        # collapse batch and heads for SDPA
        q = q.reshape(batch_size * self.n_heads, seq_len, self.d_head)
        k = k.reshape(batch_size * self.n_heads, actual_seq_len, self.d_head)
        v = v.reshape(batch_size * self.n_heads, actual_seq_len, self.d_head)

        # Prepare attention mask for SDPA
        # For KV caching, we need to handle the mask differently
        if kv_cache is not None:
            # When using cache, we need to create a mask that accounts for the full sequence
            # The mask should be True for all cached positions and for current positions
            cached_len = cached_k.shape[2]
            # Create mask for cached positions (all True)
            cached_mask = torch.ones(
                batch_size, cached_len, device=x.device, dtype=torch.bool
            )
            # Concatenate with current mask
            attn_mask = torch.cat([cached_mask, attention_mask], dim=1)
        else:
            attn_mask = attention_mask

        # Expand mask for all heads
        attn_mask = attn_mask.unsqueeze(1).expand(
            batch_size, self.n_heads, actual_seq_len
        )
        attn_mask = attn_mask.reshape(batch_size * self.n_heads, actual_seq_len)

        # For SDPA, we need to create a 3D mask
        # Only attend to positions where mask is True
        attn_mask_3d = attn_mask.unsqueeze(1).expand(
            batch_size * self.n_heads, seq_len, actual_seq_len
        )

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_3d,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        context = context.view(batch_size, self.n_heads, seq_len, self.d_head)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        return self.out_proj(context), new_kv_cache
