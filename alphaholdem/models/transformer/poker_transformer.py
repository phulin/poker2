"""Main transformer model for poker using variable-length sequences and RoPE."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...core.interfaces import Model
from ...core.registry import register_model
from ...utils.profiling import profile
from ..model_outputs import ModelOutput
from .structured_embedding_data import StructuredEmbeddingData
from .embeddings import PokerFusedEmbedding
from .heads import TransformerPolicyHead, TransformerValueHead
from .rotary_attention import RotarySelfAttention


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
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv_cache = self.attn(x, attention_mask, cos, sin, kv_cache)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, new_kv_cache


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

        # Single fused embedding module for all token types
        self.embedding = PokerFusedEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)

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

    def create_empty_cache(
        self, batch_size: int, device: torch.device
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Create empty KV cache for all layers and players."""
        cache = {}
        for layer in self.layers:
            layer_cache = layer.attn.create_empty_cache(batch_size, device)
            cache[id(layer)] = layer_cache
        return cache

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
        self,
        structured_data: StructuredEmbeddingData,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> ModelOutput:
        """
        Run forward pass on structured observation batch.

        Args:
            structured_data: Input structured data
            kv_cache: Optional KV cache dictionary keyed by layer ID

        Returns:
            ModelOutput with policy_logits, value, and updated cache
        """
        embeddings = self.embedding(structured_data)

        x = self.input_ffn(embeddings)
        attention_mask = structured_data.attention_mask

        cos, sin = self._build_rope_cache(
            seq_len=x.size(1),
            device=x.device,
            dtype=x.dtype,
        )

        # Handle KV caching through layers
        new_kv_cache = {}

        if self.use_gradient_checkpointing and self.training:
            # Note: Gradient checkpointing doesn't support KV caching well
            # In training mode, we typically don't use caching anyway
            for layer in self.layers:
                x, _ = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, cos, sin, None, use_reentrant=False
                )
        else:
            for layer in self.layers:
                layer_cache = kv_cache.get(id(layer)) if kv_cache else None
                x, layer_new_cache = layer(x, attention_mask, cos, sin, layer_cache)
                if layer_new_cache is not None:
                    new_kv_cache[id(layer)] = layer_new_cache

        cls_state = x[:, 0]
        cls_state = self.cls_mlp(cls_state)

        policy_output = self.policy_head(cls_state)
        value = self.value_head(cls_state)

        return ModelOutput(
            policy_logits=policy_output["policy_logits"],
            value=value,
            kv_cache=new_kv_cache,
        )

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
