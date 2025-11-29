"""Utility functions for activation functions."""

from __future__ import annotations

import torch.nn as nn

from alphaholdem.core.structured_config import NonlinearityType


class SwiGLU(nn.Module):
    """
    SwiGLU activation with learnable projection matrices and biases.
    y = silu(x @ W + b) * (x @ V + c)
    where silu(x) = x * sigmoid(x)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.W = nn.Linear(in_features, out_features)
        self.V = nn.Linear(in_features, out_features)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.W(x)) * self.V(x)


def get_activation(
    nonlinearity: NonlinearityType, in_features: int, out_features: int
) -> nn.Module:
    """Get activation module from NonlinearityType.

    Args:
        nonlinearity: The type of nonlinearity to use
        inplace: Whether to use inplace operations (for ReLU/SiLU)

    Returns:
        Activation module
    """
    if nonlinearity == NonlinearityType.relu:
        return nn.ReLU()
    elif nonlinearity == NonlinearityType.gelu:
        return nn.GELU()
    elif nonlinearity == NonlinearityType.silu:
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
