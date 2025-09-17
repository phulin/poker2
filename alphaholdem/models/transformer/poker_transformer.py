"""Main transformer model for poker using variable-length sequences and RoPE."""

from __future__ import annotations

from typing import Any, Dict, Tuple

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
        x_new: torch.Tensor,
        inv_freq: torch.Tensor,
        cache: dict[str, torch.Tensor] | None,
        capture_cache: bool,
        past_lengths: torch.Tensor,
        valid_new_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor]:
        attn_out, attn_cache, total_lengths = self.attn(
            x_new,
            inv_freq,
            cache,
            capture_cache,
            past_lengths,
            valid_new_mask,
        )
        residual = x_new
        x = self.norm1(residual + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        mask = valid_new_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * mask
        return x, attn_cache, total_lengths


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

    @profile
    def forward(
        self,
        structured_data: StructuredEmbeddingData,
        kv_cache: list[dict[str, torch.Tensor]] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Run forward pass on structured observation batch."""

        embeddings = combine_embeddings(
            self.card_embedding,
            self.action_embedding,
            self.context_embedding,
            structured_data,
        )

        hidden_full = self.input_ffn(embeddings)
        batch_size, seq_len, _ = hidden_full.shape
        device = hidden_full.device

        current_lengths = structured_data.lengths.to(device).long()

        cache_inputs = kv_cache
        has_cached_prefix = cache_inputs is not None
        use_checkpoint = self.use_gradient_checkpointing and torch.is_grad_enabled()

        if use_checkpoint and has_cached_prefix:
            raise RuntimeError(
                "KV caching with gradient checkpointing is not supported."
            )

        capture_new_cache = not use_checkpoint
        next_caches: list[dict[str, torch.Tensor]] | None = (
            [] if capture_new_cache else None
        )

        inv_freq = self.rotary_inv_freq.to(device=device, dtype=hidden_full.dtype)

        if use_checkpoint:
            valid_new_mask = torch.arange(seq_len, device=device).unsqueeze(0).expand(
                batch_size, -1
            ) < current_lengths.unsqueeze(1)
            zero_prefix = torch.zeros_like(current_lengths)

            def run_layer_with_checkpoint(
                layer_module: TransformerLayer,
                tensor: torch.Tensor,
            ) -> torch.Tensor:
                def inner(inputs: torch.Tensor) -> torch.Tensor:
                    output, _, _ = layer_module(
                        inputs,
                        inv_freq,
                        None,
                        False,
                        zero_prefix,
                        valid_new_mask,
                    )
                    return output

                return torch.utils.checkpoint.checkpoint(
                    inner, tensor, use_reentrant=False
                )

            layer_output = hidden_full
            for layer in self.layers:
                layer_output = run_layer_with_checkpoint(layer, layer_output)

            final_hidden = layer_output
            new_token_counts = current_lengths
            max_new_tokens = seq_len
            valid_new_mask_out = valid_new_mask
        else:
            start_positions = torch.zeros_like(current_lengths)
            if has_cached_prefix and cache_inputs:
                first_cache = cache_inputs[0]
                if first_cache is not None:
                    cached_lengths = first_cache.get("lengths")
                    if cached_lengths is not None:
                        start_positions = cached_lengths.to(device=device).long()
                        start_positions = torch.clamp(
                            start_positions, min=0, max=seq_len
                        )

            reset_mask = current_lengths <= start_positions
            if reset_mask.any():
                start_positions = start_positions.masked_fill(reset_mask, 0)

            new_token_counts = current_lengths - start_positions
            no_new_mask = new_token_counts <= 0
            if no_new_mask.any():
                start_positions = start_positions.masked_fill(no_new_mask, 0)
                new_token_counts = current_lengths - start_positions

            max_new_tokens = int(new_token_counts.max().item())
            token_offsets = torch.arange(max_new_tokens, device=device)
            valid_new_mask_out = token_offsets.unsqueeze(
                0
            ) < new_token_counts.unsqueeze(1)

            gather_indices = start_positions.unsqueeze(1) + token_offsets.unsqueeze(0)
            gather_indices = gather_indices.clamp(max=seq_len - 1, min=0)

            x_new = torch.gather(
                hidden_full,
                dim=1,
                index=gather_indices.unsqueeze(-1).expand(-1, -1, hidden_full.size(-1)),
            )
            x_new = x_new * valid_new_mask_out.unsqueeze(-1).to(hidden_full.dtype)

            layer_input = x_new
            layer_start_positions = start_positions
            for layer_idx, layer in enumerate(self.layers):
                layer_cache = cache_inputs[layer_idx] if has_cached_prefix else None
                if layer_cache is not None and "lengths" in layer_cache:
                    cached_lengths = layer_cache["lengths"].to(device=device).long()
                    diverged = cached_lengths != layer_start_positions
                    if diverged.any():
                        layer_start_positions = layer_start_positions.masked_fill(
                            diverged, 0
                        )

                layer_output, updated_cache, total_lengths = layer(
                    layer_input,
                    inv_freq,
                    layer_cache,
                    capture_new_cache,
                    layer_start_positions,
                    valid_new_mask_out,
                )

                if (
                    capture_new_cache
                    and next_caches is not None
                    and updated_cache is not None
                ):
                    next_caches.append(updated_cache)

                layer_input = layer_output
                layer_start_positions = total_lengths - new_token_counts

            final_hidden = layer_input

        batch_indices = torch.arange(batch_size, device=device)
        summary_positions = new_token_counts - 1
        summary_positions = summary_positions.clamp(min=0)
        summary_states = final_hidden[batch_indices, summary_positions]
        summary_states = self.cls_mlp(summary_states)

        policy_output = self.policy_head(summary_states)
        value = self.value_head(summary_states)

        outputs = {
            "policy_logits": policy_output["policy_logits"],
            "value": value,
        }
        outputs["kv_cache"] = next_caches

        return outputs

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
