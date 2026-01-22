"""Utility functions for activation functions."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from p2.core.structured_config import NonlinearityType


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
    nonlinearity: NonlinearityType, **activation_kwargs: Any
) -> nn.Module:
    """Get activation module from NonlinearityType.

    Args:
        nonlinearity: The type of nonlinearity to use
        activation_kwargs: Extra kwargs forwarded to the activation (e.g. ``inplace``)

    Returns:
        Activation module
    """
    if nonlinearity == NonlinearityType.relu:
        return nn.ReLU(**activation_kwargs)
    if nonlinearity == NonlinearityType.gelu:
        activation_kwargs.pop("inplace", None)
        return nn.GELU(**activation_kwargs)
    if nonlinearity == NonlinearityType.silu:
        return nn.SiLU(**activation_kwargs)
    if nonlinearity == NonlinearityType.swiglu:
        raise ValueError("SwiGLU requires in/out dimensions; construct explicitly.")
    raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
