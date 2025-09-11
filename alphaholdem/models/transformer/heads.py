"""Output heads for transformer poker model."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPolicyHead(nn.Module):
    """Policy head for transformer model with mixed discrete-continuous outputs.

    Outputs:
    - Action type logits (fold, check/call, bet, raise, allin)
    - Bet size parameters (Beta distribution parameters for continuous sizing)
    """

    def __init__(self, d_model: int, num_actions: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions

        # Action type head (discrete)
        self.action_type_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_actions),
        )

        # Bet size head (continuous - Beta distribution parameters)
        self.size_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 2),  # Alpha and Beta parameters
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize head weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for policy head.

        Args:
            x: Input representation [batch_size, d_model]

        Returns:
            Dictionary containing:
                - action_logits: Action type logits [batch_size, num_actions]
                - size_params: Beta distribution parameters [batch_size, 2]
        """
        action_logits = self.action_type_head(x)

        # Size parameters for Beta distribution
        size_raw = self.size_head(x)
        size_params = F.softplus(size_raw) + 1.0  # Ensure positive parameters

        return {"action_logits": action_logits, "size_params": size_params}


class TransformerValueHead(nn.Module):
    """Value head for transformer model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize head weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for value head.

        Args:
            x: Input representation [batch_size, d_model]

        Returns:
            Value estimates [batch_size]
        """
        value = self.value_head(x)
        return value.squeeze(-1)


class HandRangeHead(nn.Module):
    """Auxiliary head for predicting opponent's hand range.

    Predicts probability distribution over all possible two-card combinations
    (1,326 combinations total).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.num_combinations = 1326  # All possible two-card combinations

        # Hand range head
        self.range_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, self.num_combinations),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize head weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for hand range head.

        Args:
            x: Input representation [batch_size, d_model]

        Returns:
            Hand range logits [batch_size, 1326]
        """
        range_logits = self.range_head(x)
        return range_logits

    def get_hand_range_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get hand range probabilities with softmax.

        Args:
            x: Input representation [batch_size, d_model]

        Returns:
            Hand range probabilities [batch_size, 1326]
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
