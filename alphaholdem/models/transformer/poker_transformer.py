"""Main transformer model for poker."""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.interfaces import Model
from ...core.registry import register_model
from .embeddings import CardEmbedding, ActionEmbedding, ContextEmbedding
from .heads import TransformerPolicyHead, TransformerValueHead, HandRangeHead
from .tokenizer import PokerTokenizer


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
        max_sequence_length: int = 50,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.vocab_size = vocab_size
        self.num_actions = num_actions
        self.use_auxiliary_loss = use_auxiliary_loss
        self.max_sequence_length = max_sequence_length
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Tokenizer for converting states to token sequences
        self.tokenizer = PokerTokenizer(max_sequence_length)

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_sequence_length, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output heads
        self.policy_head = TransformerPolicyHead(d_model, num_actions)
        self.value_head = TransformerValueHead(d_model)

        if use_auxiliary_loss:
            self.hand_range_head = HandRangeHead(d_model)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize token embeddings
        nn.init.normal_(self.token_emb.weight, mean=0, std=0.02)
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

    def forward(
        self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with token-based input.

        Args:
            token_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len] (1=attend, 0=ignore)

        Returns:
            Dictionary containing model outputs
        """
        return self._forward_tokens(token_ids, attention_mask)

    def forward_legacy(
        self, cards_tensor: torch.Tensor, actions_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy forward pass for compatibility with existing training loop.

        Args:
            cards_tensor: Cards tensor (not used in transformer, kept for compatibility)
            actions_tensor: Actions tensor (not used in transformer, kept for compatibility)

        Returns:
            Tuple of (logits, value) for compatibility with training loop
        """
        # For now, we'll use dummy token sequences since the training loop
        # doesn't yet support transformer state encoding
        batch_size = cards_tensor.shape[0]
        seq_len = 20  # Fixed sequence length for now
        device = cards_tensor.device

        # Create dummy token sequences
        token_ids = torch.randint(
            0, self.vocab_size, (batch_size, seq_len), device=device
        )
        attention_mask = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )

        # Get model outputs
        outputs = self._forward_tokens(token_ids, attention_mask)

        # Return in the format expected by training loop
        return outputs["policy_logits"], outputs["value"]

    def _forward_tokens(
        self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Internal forward pass with token inputs.

        Args:
            token_ids: Token IDs tensor [batch_size, seq_len]
            attention_mask: Attention mask tensor [batch_size, seq_len]

        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_len = token_ids.shape

        # Create positional IDs
        pos_ids = torch.arange(seq_len, device=token_ids.device)

        # Token embeddings + positional embeddings
        token_embeddings = self.token_emb(token_ids)  # [batch, seq_len, d_model]
        pos_embeddings = self.pos_emb(pos_ids)  # [seq_len, d_model]

        # Add positional embeddings to token embeddings
        x = token_embeddings + pos_embeddings.unsqueeze(0)
        x = self.dropout(x)

        # Create attention mask (True = ignore, False = attend)
        if attention_mask is not None:
            # Convert to transformer format: True = ignore
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Transformer encoding
        if self.use_gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            def transformer_forward(x, mask):
                return self.transformer(x, src_key_padding_mask=mask)

            x = torch.utils.checkpoint.checkpoint(
                transformer_forward, x, src_key_padding_mask, use_reentrant=False
            )
        else:
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

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

    def encode_state(
        self, tensor_env, env_idx: int, player: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a game state into token sequence.

        Args:
            tensor_env: HUNLTensorEnv instance
            env_idx: Environment index
            player: Player index (0 or 1)

        Returns:
            Tuple of (token_ids, attention_mask)
        """
        return self.tokenizer.tokenize_state(tensor_env, env_idx, player)

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
            "max_sequence_length": self.max_sequence_length,
            "use_auxiliary_loss": self.use_auxiliary_loss,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
