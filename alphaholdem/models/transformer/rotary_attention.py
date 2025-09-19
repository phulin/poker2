"""Rotary Position Embedding (RoPE) attention implementation."""

from typing import Tuple

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

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # collapse batch and heads for SDPA
        q = q.reshape(batch_size * self.n_heads, seq_len, self.d_head)
        k = k.reshape(batch_size * self.n_heads, seq_len, self.d_head)
        v = v.reshape(batch_size * self.n_heads, seq_len, self.d_head)

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False,
            enable_gqa=True,
        )
        context = context.view(batch_size, self.n_heads, seq_len, self.d_head)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        return self.out_proj(context)
