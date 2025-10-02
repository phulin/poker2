"""Main transformer model for poker using variable-length sequences and RoPE."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint

from alphaholdem.core.interfaces import Model
from alphaholdem.core.registry import register_model
from alphaholdem.models.model_outputs import ModelOutput
from alphaholdem.models.transformer.embeddings import PokerFusedEmbedding
from alphaholdem.models.transformer.heads import (
    TransformerPolicyHead,
    TransformerValueHead,
)
from alphaholdem.models.transformer.orthogonal_linear import OrthogonalLinear
from alphaholdem.models.transformer.rotary_attention import RotarySelfAttention
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.tokens import CLS_INDEX, HOLE0_INDEX, HOLE1_INDEX
from alphaholdem.utils.profiling import profile


class TransformerLayer(nn.Module):
    """Single transformer encoder block with RoPE attention (pre-LN)."""

    def __init__(
        self, d_model: int, n_heads: int, dropout: float, residual_scale: float
    ) -> None:
        super().__init__()
        self.attn = RotarySelfAttention(d_model, n_heads, dropout, residual_scale)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            OrthogonalLinear(d_model, d_model * 4, gain=math.sqrt(2.0)),
            nn.GELU(),
            nn.Dropout(dropout),
            OrthogonalLinear(
                d_model * 4,
                d_model,
                gain=1.0,
                output_scale=residual_scale,
            ),
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
        # Pre-LN: apply norm before attention
        x_norm = self.norm1(x)
        attn_out, new_kv_cache = self.attn(x_norm, attention_mask, cos, sin, kv_cache)

        x = x + self.dropout(attn_out)

        # Pre-LN: apply norm before FFN
        x_ffn_norm = self.norm2(x)
        ffn_out = self.ffn(x_ffn_norm)

        x = x + self.dropout(ffn_out)

        return x, new_kv_cache


@register_model("poker_transformer_v1")
class PokerTransformerV1(nn.Module, Model):
    """Transformer-based poker model with variable-length input encoding."""

    def __init__(
        self,
        max_sequence_length: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        num_bet_bins: int,
        dropout: float,
        use_gradient_checkpointing: bool,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.num_bet_bins = num_bet_bins
        self.max_sequence_length = max_sequence_length
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Single fused embedding module for all token types
        self.embedding = PokerFusedEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)

        self.input_ffn = nn.Sequential(
            OrthogonalLinear(d_model, d_model * 2, gain=math.sqrt(2.0)),
            nn.GELU(),
            nn.Dropout(dropout),
            OrthogonalLinear(d_model * 2, d_model, gain=1.0),
            # no normalization as TransformerLayer has pre-LN
        )

        residual_scale = 1.0 / math.sqrt(n_layers) if n_layers > 0 else 1.0
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, n_heads, dropout, residual_scale)
                for _ in range(n_layers)
            ]
        )

        # Layers have pre-LN, so we need to normalize the output
        self.post_norm = nn.LayerNorm(d_model)

        self.cls_mlp = nn.Sequential(
            OrthogonalLinear(d_model * 4, d_model, gain=math.sqrt(2.0)),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.policy_head = TransformerPolicyHead(d_model, num_bet_bins)
        self.value_head = TransformerValueHead(d_model)

        # Precompute rotary frequencies
        head_dim = d_model // n_heads
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("rotary_inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for RoPE
        cos, sin = self._build_rope_cache(
            seq_len=max_sequence_length,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

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
        """Initialize parameters using orthogonal defaults suitable for PPO."""

        for module in self.modules():
            if isinstance(module, OrthogonalLinear):
                module.reset_parameters()
            elif isinstance(module, nn.Linear):
                raise RuntimeError("All linear layers should be OrthogonalLinear")
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

        # Handle KV caching through layers
        new_kv_cache = {}

        if self.use_gradient_checkpointing and self.training:
            # Note: Gradient checkpointing doesn't support KV caching well
            # In training mode, we typically don't use caching anyway
            for layer in self.layers:
                x, _ = torch.utils.checkpoint.checkpoint(
                    layer,
                    x,
                    attention_mask,
                    self.cos,
                    self.sin,
                    None,
                    use_reentrant=False,
                )
        else:
            for layer in self.layers:
                layer_cache = kv_cache.get(id(layer)) if kv_cache else None
                x, layer_new_cache = layer(
                    x, attention_mask, self.cos, self.sin, layer_cache
                )
                if layer_new_cache is not None:
                    new_kv_cache[id(layer)] = layer_new_cache

        x = self.post_norm(x)

        cls_state = x[:, CLS_INDEX]
        hole_mean = (x[:, HOLE0_INDEX] + x[:, HOLE1_INDEX]) / 2
        hole_diff = (x[:, HOLE0_INDEX] - x[:, HOLE1_INDEX]) / 2
        hole_prod = x[:, HOLE0_INDEX] * x[:, HOLE1_INDEX]
        x = torch.cat([cls_state, hole_mean, hole_diff, hole_prod], dim=-1)
        x = self.cls_mlp(x)

        return ModelOutput(
            policy_logits=self.policy_head(x),
            value=self.value_head(x),
            kv_cache=new_kv_cache,
        )

    @torch.no_grad()
    def adjust_scale(self, weight_scale: float, bias_adjustment: float) -> None:
        """
        Apply PopArt scaling adjustments to the value head's last linear layer.

        Args:
            weight_scale: Scaling factor for the weight matrix
            bias_adjustment: Adjustment term for the bias vector
        """
        # Access the last linear layer in the value head
        last_linear = self.value_head.value_head[-1]

        # Apply weight scaling
        last_linear.weight.data.mul_(weight_scale)

        # Apply bias adjustment if bias exists
        if last_linear.bias is not None:
            last_linear.bias.data.mul_(weight_scale).add_(bias_adjustment)

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
