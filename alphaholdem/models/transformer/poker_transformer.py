"""Main transformer model for poker."""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union
from line_profiler import profile
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.interfaces import Model
from ...core.registry import register_model
from .embeddings import CardEmbedding, ActionEmbedding, ContextEmbedding
from .heads import TransformerPolicyHead, TransformerValueHead, HandRangeHead
from .embedding_data import StructuredEmbeddingData
from .attention import PokerTransformerEncoderLayer


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
        vocab_size: int = 60,  # 52 cards + 8 special tokens
        num_actions: int = 8,
        dropout: float = 0.1,
        use_auxiliary_loss: bool = True,
        use_gradient_checkpointing: bool = False,  # Disabled for performance
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.use_auxiliary_loss = use_auxiliary_loss
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Positional embeddings (use large fixed size)
        self.pos_emb = nn.Embedding(42, d_model)

        # Structured embeddings
        self.card_embedding = CardEmbedding(d_model)
        self.action_embedding = ActionEmbedding(d_model)
        self.context_embedding = ContextEmbedding(d_model)

        # Transformer encoder with poker-specific attention
        encoder_layers = [
            PokerTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_model * 4,
                dropout=dropout,
                use_poker_biases=True,
            )
            for _ in range(n_layers)
        ]
        self.transformer = nn.ModuleList(encoder_layers)

        # Output heads
        self.policy_head = TransformerPolicyHead(d_model, num_actions)
        self.value_head = TransformerValueHead(d_model)

        # if use_auxiliary_loss:
        #     self.hand_range_head = HandRangeHead(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        nn.init.normal_(self.pos_emb.weight, mean=0, std=0.02)

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
        device = structured_data.device

        # Create positional IDs
        pos_ids = torch.arange(seq_len, device=device)
        position_indices = pos_ids.unsqueeze(0).expand(
            batch_size, -1
        )  # [batch_size, seq_len]

        # Get structured embeddings
        # Convert tensors to long for embedding layers
        card_embeddings = self.card_embedding(
            structured_data.token_ids.long(), structured_data.card_streets.long()
        )
        action_embeddings = self.action_embedding(
            structured_data.action_actors.long(),
            structured_data.action_streets.long(),
            structured_data.action_legal_masks,
        )
        context_embeddings = self.context_embedding(
            structured_data.context_features,
        )

        # Combine embeddings (simple addition for now)
        # In practice, you might want to use learned combination weights
        x = card_embeddings + action_embeddings + context_embeddings

        # Add positional embeddings
        pos_embeddings = self.pos_emb(pos_ids)  # [seq_len, d_model]
        x = x + pos_embeddings.unsqueeze(0)
        x = self.dropout(x)

        # Create attention mask for padded positions
        # Mask out positions where card_indices == -1 (padding)
        attention_mask = (
            structured_data.token_ids != -1
        ).float()  # [batch_size, seq_len]

        # Transformer encoding with poker-specific attention
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            def transformer_forward(x):
                for layer in self.transformer:
                    x = layer(
                        x,
                        src_mask=None,  # We'll use key padding mask instead
                        card_indices=structured_data.token_ids,
                        position_indices=position_indices,
                        attention_mask=attention_mask,
                    )
                return x

            x = torch.utils.checkpoint.checkpoint(
                transformer_forward, x, use_reentrant=False
            )
        else:
            for layer in self.transformer:
                x = layer(
                    x,
                    src_mask=None,  # We'll use key padding mask instead
                    card_indices=structured_data.token_ids,
                    position_indices=position_indices,
                    attention_mask=attention_mask,
                )

        # Get CLS token representation (first token)
        cls_repr = x[:, 0]  # [batch_size, d_model]

        # Output heads
        policy_output = self.policy_head(cls_repr)
        value = self.value_head(cls_repr)

        outputs = {
            "policy_logits": policy_output["action_logits"],
            "size_params": policy_output["size_params"],
            "value": value,
        }

        # Add auxiliary hand range prediction if enabled
        if self.use_auxiliary_loss and hasattr(self, "hand_range_head"):
            hand_range_logits = self.hand_range_head(cls_repr)
            outputs["hand_range_logits"] = hand_range_logits

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
            "vocab_size": self.vocab_size,
            "num_actions": self.num_actions,
            "use_auxiliary_loss": self.use_auxiliary_loss,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
