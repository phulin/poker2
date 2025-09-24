"""Model factory for creating different types of poker models."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from alphaholdem.core.registry import MODELS
from alphaholdem.models.cnn import SiameseConvNetV1
from alphaholdem.models.transformer import PokerTransformerV1


class ModelFactory:
    """Factory for creating poker models and encoders."""

    @staticmethod
    def create_model(
        model_type: str,
        model_config: Dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        """
        Create a model instance based on type and configuration.

        Args:
            model_type: Type of model to create ('cnn', 'transformer', etc.)
            model_config: Model configuration parameters
            device: Device to create model on

        Returns:
            Model instance
        """
        if model_type == "cnn":
            return ModelFactory._create_cnn_model(model_config, device)
        elif model_type == "transformer":
            return ModelFactory._create_transformer_model(model_config, device)
        else:
            # Try to find in registry
            if model_type in MODELS:
                model_class = MODELS[model_type]
                return model_class(**model_config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def _create_cnn_model(
        config: Dict[str, Any], device: torch.device
    ) -> SiameseConvNetV1:
        """Create CNN-based model."""
        return torch.compile(SiameseConvNetV1(**config).to(device))

    @staticmethod
    def _create_transformer_model(
        config: Dict[str, Any], device: torch.device
    ) -> PokerTransformerV1:
        """Create transformer-based model."""
        return torch.compile(PokerTransformerV1(**config).to(device))

    @staticmethod
    def get_available_model_types() -> list[str]:
        """Get list of available model types."""
        return ["cnn", "transformer"]  # Add more as implemented
