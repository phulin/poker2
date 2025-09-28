"""
Context managers for temporarily changing model training/evaluation mode.

This module provides context managers that can temporarily set a model to train or eval mode
and automatically restore the original state when exiting the context.
"""

from contextlib import contextmanager
from typing import Generator

import torch.nn as nn


@contextmanager
def model_train(model: nn.Module) -> Generator[None, None, None]:
    """
    Context manager to temporarily set a model to training mode.

    Args:
        model: The PyTorch model to set to training mode

    Yields:
        None

    Example:
        >>> with model_train(model):
        ...     # Model is in training mode here
        ...     loss = model(inputs)
        >>> # Model is restored to its original mode here
    """
    original_training = model.training
    try:
        model.train()
        yield
    finally:
        if original_training:
            model.train()
        else:
            model.eval()


@contextmanager
def model_eval(model: nn.Module) -> Generator[None, None, None]:
    """
    Context manager to temporarily set a model to evaluation mode.

    Args:
        model: The PyTorch model to set to evaluation mode

    Yields:
        None

    Example:
        >>> with model_eval(model):
        ...     # Model is in evaluation mode here
        ...     with torch.no_grad():
        ...         output = model(inputs)
        >>> # Model is restored to its original mode here
    """
    original_training = model.training
    try:
        model.eval()
        yield
    finally:
        if original_training:
            model.train()
        else:
            model.eval()
