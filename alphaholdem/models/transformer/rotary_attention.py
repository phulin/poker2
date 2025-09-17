"""Rotary self-attention utilities and module."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def build_rope_cos_sin(
    inv_freq: torch.Tensor,
    positions: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct cos/sin rotary caches for the provided absolute positions."""

    freqs = torch.einsum("bs,d->bsd", positions.to(inv_freq.dtype), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos.unsqueeze(1), sin.unsqueeze(1)


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

    def forward(
        self,
        x_new: torch.Tensor,
        inv_freq: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        capture_cache: bool,
        past_lengths: torch.Tensor,
        valid_new_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor]:
        """Compute self-attention on the new tokens only."""

        batch_size, max_t_new, _ = x_new.shape
        device = x_new.device

        past_k = cache.get("k") if cache is not None else None
        past_v = cache.get("v") if cache is not None else None

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

        steps = torch.arange(max_t_new, device=device)
        positions = past_lengths.unsqueeze(1) + steps.unsqueeze(0)
        positions = positions.masked_fill(~valid_new_mask, 0)
        cos, sin = build_rope_cos_sin(inv_freq, positions, q.dtype)
        q, k_new = apply_rotary_pos_emb(q, k_new, cos, sin)

        valid_new = valid_new_mask.unsqueeze(1).unsqueeze(-1).to(dtype=q.dtype)
        q = q * valid_new
        k_new = k_new * valid_new
        v_new = v_new * valid_new

        new_counts = valid_new_mask.sum(dim=1)
        total_lengths = past_lengths + new_counts

        if past_k is not None and past_v is not None:
            past_k = past_k.to(device=device, dtype=q.dtype)
            past_v = past_v.to(device=device, dtype=q.dtype)
            if past_k.shape[2] > 0:
                past_indices = torch.arange(past_k.shape[2], device=device).unsqueeze(0)
                valid_past = past_indices < past_lengths.unsqueeze(1)
                valid_past = valid_past.unsqueeze(1).unsqueeze(-1).to(dtype=q.dtype)
                masked_past_k = past_k * valid_past
                masked_past_v = past_v * valid_past
            else:
                masked_past_k = past_k
                masked_past_v = past_v
            k_total = torch.cat([masked_past_k, k_new], dim=2)
            v_total = torch.cat([masked_past_v, v_new], dim=2)
        else:
            k_total = k_new
            v_total = v_new

        max_total_len = k_total.shape[2]
        total_positions = torch.arange(max_total_len, device=device).unsqueeze(0)
        valid_keys = total_positions < total_lengths.unsqueeze(1)
        attn_mask = (~valid_keys).unsqueeze(1).unsqueeze(1)

        attn_dropout = self.attn_dropout.p if self.training else 0.0
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

        new_cache: dict[str, torch.Tensor] | None = None
        if capture_cache:
            new_cache = {
                "k": k_total.detach(),
                "v": v_total.detach(),
                "lengths": total_lengths.detach(),
            }

        return attn_output, new_cache, total_lengths
