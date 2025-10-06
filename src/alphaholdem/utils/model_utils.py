"""
Utility functions for model predictions and data processing.

This module provides utility functions for getting predictions from models,
converting between different data formats, and handling legal action masks.
"""

from typing import Optional, Union

import torch
import torch.nn as nn

from alphaholdem.models.cnn.cnn_embedding_data import CNNEmbeddingData
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.rl.vectorized_replay import BatchSample


def compute_masked_logits(
    logits: torch.Tensor,
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """Compute masked logits from a model given data and legal masks."""
    return torch.where(legal_masks, logits, -1e9)


def get_logits_log_probs_values(
    model: nn.Module,
    data: Union[CNNEmbeddingData, StructuredEmbeddingData],
    legal_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Get log probabilities and values from a model given data and legal masks.

    Args:
        model: The PyTorch model to get predictions from
        data: Input data (tensor or StructuredEmbeddingData)
        legal_masks: Legal action masks [batch_size, num_actions]

    Returns:
        Tuple of (logits, log_probs, values, value_quantiles) where:
        - logits: [batch_size, num_actions]
        - log_probs: [batch_size, num_actions]
        - values: [batch_size]
        - value_quantiles: Optional[batch_size, num_quantiles]
    """
    outputs: ModelOutput = model(data)
    logits = outputs.policy_logits
    values = outputs.value
    value_quantiles = outputs.value_quantiles

    # Apply legal mask
    masked_logits = compute_masked_logits(logits, legal_masks)

    return logits, torch.log_softmax(masked_logits, dim=-1), values, value_quantiles


def get_log_probs(
    model: nn.Module,
    data: Union[CNNEmbeddingData, StructuredEmbeddingData],
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """Get log probabilities from a model given data and legal masks.

    Args:
        model: The PyTorch model to get predictions from
        data: Input data (tensor or StructuredEmbeddingData)
        legal_masks: Legal action masks [batch_size, num_actions]

    Returns:
        Log probabilities [batch_size, num_actions]
    """
    return get_logits_log_probs_values(model, data, legal_masks)[1]


def get_log_probs(
    model: nn.Module,
    batch: BatchSample,
) -> torch.Tensor:
    """Get log probabilities from a model given data and legal masks.

    Args:
        model: The PyTorch model to get predictions from
        batch: Batch sample

    Returns:
        Log probabilities [batch_size, num_actions]
    """
    return get_log_probs(model, batch.embedding_data, batch.legal_masks)


def get_probs_and_values(
    model: nn.Module,
    data: Union[CNNEmbeddingData, StructuredEmbeddingData],
    legal_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get probabilities and values from a model given data and legal masks.

    Args:
        model: The PyTorch model to get predictions from
        data: Input data (tensor or StructuredEmbeddingData)
        legal_masks: Legal action masks [batch_size, num_actions]

    Returns:
        Tuple of (probs, values) where:
        - probs: [batch_size, num_actions]
        - values: [batch_size]
    """
    _, log_probs, values, _ = get_logits_log_probs_values(model, data, legal_masks)
    return log_probs, values


def get_probs(
    model: nn.Module,
    data: Union[CNNEmbeddingData, StructuredEmbeddingData],
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Get probabilities from a model given data and legal masks.

    Args:
        model: The PyTorch model to get predictions from
        data: Input data (tensor or StructuredEmbeddingData)
        legal_masks: Legal action masks [batch_size, num_actions]

    Returns:
        Action probabilities [batch_size, num_actions]
    """
    return get_probs_and_values(model, data, legal_masks)[0]


def get_best_action(
    model: nn.Module,
    data: Union[CNNEmbeddingData, StructuredEmbeddingData],
    legal_masks: torch.Tensor,
) -> torch.Tensor:
    """
    Get the best action (argmax) from the model given data and legal masks.

    Args:
        model: The PyTorch model to get predictions from
        data: Input data (tensor or StructuredEmbeddingData)
        legal_masks: Legal action masks [batch_size, num_actions]

    Returns:
        Best action indices [batch_size]
    """
    outputs = model(data)
    logits = outputs.policy_logits
    masked_logits = compute_masked_logits(logits, legal_masks)
    return torch.argmax(masked_logits, dim=-1)
