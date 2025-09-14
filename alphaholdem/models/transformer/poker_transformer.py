"""Main transformer model for poker."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ...core.interfaces import Model
from ...core.registry import register_model
from ...utils.profiling import profile
from .embedding_data import StructuredEmbeddingData
from .embeddings import ActionEmbedding, CardEmbedding, ContextEmbedding
from .heads import TransformerPolicyHead, TransformerValueHead
from .state_encoder import TransformerStateEncoder


@register_model("poker_transformer_v1")
class PokerTransformerV1(nn.Module, Model):
    """Transformer-based poker model using token-based state representation.

    This model converts poker game states into sequences of tokens and uses
    a transformer encoder to process them. It outputs policy logits, value
    estimates, and optionally hand range predictions for the opponent.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        num_bet_bins: int = 8,  # Add num_bet_bins parameter
        dropout: float = 0.1,
        use_auxiliary_loss: bool = True,
        use_gradient_checkpointing: bool = False,  # Disabled for performance
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.num_bet_bins = num_bet_bins

        # Structured embeddings
        self.card_embedding = CardEmbedding(num_bet_bins=num_bet_bins, d_model=d_model)
        self.action_embedding = ActionEmbedding(
            num_bet_bins=num_bet_bins, d_model=d_model
        )
        self.context_embedding = ContextEmbedding(
            num_bet_bins=num_bet_bins, d_model=d_model
        )

        # Input feed-forward layers after embeddings
        self.input_ffn = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
        )

        # Transformer encoder with vanilla attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Output feed-forward layer to process entire sequence
        # Calculate max sequence length: CLS + cards + actions + context
        max_seq_len = TransformerStateEncoder.get_max_sequence_length()
        self.output_ffn = nn.Sequential(
            nn.Linear(max_seq_len * d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
        )

        # Output heads
        self.policy_head = TransformerPolicyHead(d_model, num_bet_bins, dropout)
        self.value_head = TransformerValueHead(d_model, dropout)

        # if use_auxiliary_loss:
        #     self.hand_range_head = HandRangeHead(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize transformer layers
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @profile
    def forward(
        self, structured_data: StructuredEmbeddingData
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with structured embeddings.

        Args:
            structured_data: StructuredEmbeddingData containing all embedding components

        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_len = structured_data.batch_size, structured_data.seq_len
        d_model = self.d_model

        # Get structured embeddings first
        # Convert tensors to long for embedding layers
        card_embeddings = self.card_embedding(
            structured_data.token_ids,
            structured_data.card_ranks,
            structured_data.card_suits,
            structured_data.card_streets,
        )
        action_embeddings = self.action_embedding(
            structured_data.token_ids,
            structured_data.action_actors,
            structured_data.action_streets,
            structured_data.action_legal_masks,
        )
        context_embeddings = self.context_embedding(
            structured_data.token_ids,
            structured_data.context_features,
        )

        # Create CLS token (zero vector)
        cls_token = torch.zeros(batch_size, 1, d_model, device=card_embeddings.device)

        # Concatenate: CLS + card + action + context
        x = torch.cat(
            [cls_token, card_embeddings, action_embeddings, context_embeddings], dim=1
        )

        # Apply input feed-forward layers to each token
        x = self.input_ffn(x)  # [batch_size, seq_len, d_model]

        # Create padding mask: 0 for valid tokens, 1 for padding (token_id < 0)
        # padding: [batch_size, seq_len, d_model]
        # structured_data.token_ids: [batch_size, seq_len]
        padding_mask = structured_data.token_ids < 0

        # Transformer encoding with vanilla attention
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            def transformer_forward(x):
                return self.transformer(x, src_key_padding_mask=padding_mask)

            x = torch.utils.checkpoint.checkpoint(
                transformer_forward, x, use_reentrant=False
            )
        else:
            # Standard forward pass
            x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Feed-forward layer to process entire sequence
        # Flatten the sequence dimension
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(batch_size, -1)  # [batch_size, seq_len * d_model]
        x_processed = self.output_ffn(x_flat)  # [batch_size, d_model]

        # Output heads
        policy_output = self.policy_head(x_processed)
        value = self.value_head(x_processed)

        outputs = {
            "policy_logits": policy_output["policy_logits"],
            "value": value,
        }

        # Add auxiliary hand range prediction if enabled
        # if self.use_auxiliary_loss and hasattr(self, "hand_range_head"):
        #     hand_range_logits = self.hand_range_head(x_processed)
        #     outputs["hand_range_logits"] = hand_range_logits

        return outputs

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging/debugging."""
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
