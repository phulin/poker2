"""Main transformer model for poker using variable-length sequences and RoPE."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...core.interfaces import Model
from ...core.registry import register_model
from ...utils.profiling import profile
from .embedding_data import StructuredEmbeddingData
from .embeddings import (
    ActionEmbedding,
    CardEmbedding,
    ContextEmbedding,
    combine_embeddings,
)
from .heads import TransformerPolicyHead, TransformerValueHead


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate last dimension by half for RoPE application."""

    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
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

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        if attention_mask is not None:
            key_padding = (~attention_mask).unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(key_padding, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        context = torch.matmul(attn, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)

        return self.out_proj(context)


class TransformerLayer(nn.Module):
    """Single transformer encoder block with RoPE attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = RotarySelfAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attn_out = self.attn(x, attention_mask, cos, sin)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


@register_model("poker_transformer_v1")
class PokerTransformerV1(nn.Module, Model):
    """Transformer-based poker model with variable-length input encoding."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        num_bet_bins: int = 8,
        dropout: float = 0.1,
        use_auxiliary_loss: bool = True,
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.num_bet_bins = num_bet_bins
        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Embedding modules for different token families
        self.card_embedding = CardEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)
        self.action_embedding = ActionEmbedding(
            num_bet_bins=num_bet_bins, d_model=d_model
        )
        self.context_embedding = ContextEmbedding(
            num_bet_bins=num_bet_bins, d_model=d_model
        )

        self.input_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.layers = nn.ModuleList(
            [TransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.cls_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        self.policy_head = TransformerPolicyHead(d_model, num_bet_bins, dropout)
        self.value_head = TransformerValueHead(d_model, dropout)

        # Precompute rotary frequencies
        head_dim = d_model // n_heads
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize parameters with Xavier uniform where appropriate."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_rope_cache(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.rotary_inv_freq.to(device=device, dtype=dtype)
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        return cos, sin

    @profile
    def forward(
        self, structured_data: StructuredEmbeddingData
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass on structured observation batch."""

        embeddings = combine_embeddings(
            self.card_embedding,
            self.action_embedding,
            self.context_embedding,
            structured_data,
        )

        x = self.input_ffn(embeddings)
        attention_mask = structured_data.attention_mask

        cos, sin = self._build_rope_cache(
            seq_len=x.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        if self.use_gradient_checkpointing and self.training:
            for layer in self.layers:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, cos, sin, use_reentrant=False
                )
        else:
            for layer in self.layers:
                x = layer(x, attention_mask, cos, sin)

        cls_state = x[:, 0]
        cls_state = self.cls_mlp(cls_state)

        policy_output = self.policy_head(cls_state)
        value = self.value_head(cls_state)

        return {
            "policy_logits": policy_output["policy_logits"],
            "value": value,
        }

    def get_model_info(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "model_type": "poker_transformer_v1",
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "num_bet_bins": self.num_bet_bins,
            "use_auxiliary_loss": self.use_auxiliary_loss,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
