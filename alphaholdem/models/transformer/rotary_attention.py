"""Rotary self-attention utilities and module."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.profiling import profile
from .kv_cache import LayerKVCache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension by half for RoPE application."""

    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
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
        self.attn_dropout = nn.Dropout(dropout)

    @profile
    def forward(
        self,
        x_new: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        cache: LayerKVCache | None,
        capture_cache: bool,
        past_lengths: torch.Tensor | None = None,
        valid_new_mask: torch.Tensor,
        new_token_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, LayerKVCache | None, torch.Tensor]:
        """Compute self-attention on the new tokens only."""

        batch_size, max_t_new, _ = x_new.shape
        device = x_new.device

        q = self.q_proj(x_new).view(batch_size, max_t_new, self.n_heads, self.d_head)
        k_new = self.k_proj(x_new).view(
            batch_size, max_t_new, self.n_heads, self.d_head
        )
        v_new = self.v_proj(x_new).view(
            batch_size, max_t_new, self.n_heads, self.d_head
        )

        q = q.permute(0, 2, 1, 3)
        k_new = k_new.permute(0, 2, 1, 3)
        v_new = v_new.permute(0, 2, 1, 3)

        q, k_new = apply_rotary_pos_emb(q, k_new, rotary_cos, rotary_sin)

        valid_new = valid_new_mask.unsqueeze(1).unsqueeze(-1).to(dtype=q.dtype)
        q = q * valid_new
        k_new = k_new * valid_new
        v_new = v_new * valid_new

        trimmed_past = torch.zeros_like(new_token_counts, device=device)
        if past_lengths is not None:
            trimmed_past = past_lengths.to(device=device, dtype=new_token_counts.dtype)

        past_k: torch.Tensor | None = None
        past_v: torch.Tensor | None = None
        if cache is not None:
            typed_cache = cache.to_device_dtype(device=device, dtype=q.dtype)
            past_k = typed_cache.keys
            past_v = typed_cache.values
            cache_lengths = typed_cache.lengths.to(
                device=device, dtype=new_token_counts.dtype
            )
            cache_lengths = cache_lengths.clamp(max=past_k.shape[2])
            trimmed_past = cache_lengths
            if past_k.numel() == 0 or past_k.shape[2] == 0:
                past_k = None
                past_v = None
                trimmed_past = torch.zeros_like(new_token_counts, device=device)

        if past_k is not None and past_v is not None:
            max_cached = past_k.shape[2]
            past_positions = torch.arange(max_cached, device=device)
            past_mask = past_positions.unsqueeze(0) < trimmed_past.unsqueeze(1)
            past_mask = past_mask.unsqueeze(1).unsqueeze(-1).to(dtype=past_k.dtype)
            past_k = past_k[:, :, :max_cached] * past_mask
            past_v = past_v[:, :, :max_cached] * past_mask

        if max_t_new > 0:
            new_mask = valid_new_mask.unsqueeze(1).unsqueeze(-1).to(k_new.dtype)
            k_new = k_new * new_mask
            v_new = v_new * new_mask

        total_lengths = trimmed_past + new_token_counts
        max_total_len = (
            int(total_lengths.max().item()) if total_lengths.numel() > 0 else 0
        )

        if past_k is not None and past_v is not None:
            k_total = torch.cat([past_k, k_new], dim=2)
            v_total = torch.cat([past_v, v_new], dim=2)
        else:
            k_total = k_new
            v_total = v_new

        if k_total.shape[2] > max_total_len:
            k_total = k_total[:, :, :max_total_len]
            v_total = v_total[:, :, :max_total_len]

        attn_dropout = self.attn_dropout.p if self.training else 0.0
        total_positions = torch.arange(max_total_len, device=device).unsqueeze(0)
        valid_keys = total_positions < total_lengths.unsqueeze(1)
        attn_mask = (~valid_keys).unsqueeze(1).unsqueeze(1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k_total,
            v_total,
            attn_mask=attn_mask,
            dropout_p=attn_dropout,
            is_causal=False,
        )

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, max_t_new, self.d_model)
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output * valid_new_mask.unsqueeze(-1).to(attn_output.dtype)

        new_cache: LayerKVCache | None = None
        if capture_cache:
            new_cache = LayerKVCache(
                keys=k_total.detach(),
                values=v_total.detach(),
                lengths=total_lengths.detach(),
            )

        return attn_output, new_cache, total_lengths
