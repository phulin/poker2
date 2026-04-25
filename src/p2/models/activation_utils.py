"""Utility functions for activation functions."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from p2.core.structured_config import NonlinearityType


class SwiGLU(nn.Module):
    """
    SwiGLU FFN block (Llama-style):
        y = down( silu(gate(x)) * up(x) )
    with an inner expansion at ``hidden_features``.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.gate = nn.Linear(in_features, hidden_features, bias=False)
        self.up = nn.Linear(in_features, hidden_features, bias=False)
        self.down = nn.Linear(hidden_features, out_features, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.down(self.silu(self.gate(x)) * self.up(x))


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
