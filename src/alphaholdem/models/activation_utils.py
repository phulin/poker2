"""Utility functions for activation functions."""

from __future__ import annotations

import torch.nn as nn

from alphaholdem.core.structured_config import NonlinearityType


def get_activation(nonlinearity: NonlinearityType, inplace: bool = False) -> nn.Module:
    """Get activation module from NonlinearityType.

    Args:
        nonlinearity: The type of nonlinearity to use
        inplace: Whether to use inplace operations (for ReLU/SiLU)

    Returns:
        Activation module
    """
    if nonlinearity == NonlinearityType.relu:
        return nn.ReLU(inplace=inplace)
    elif nonlinearity == NonlinearityType.gelu:
        return nn.GELU()
    elif nonlinearity == NonlinearityType.silu:
        return nn.SiLU(inplace=inplace)
