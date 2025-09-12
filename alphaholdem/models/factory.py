"""Model factory for creating different types of poker models."""

from __future__ import annotations

from typing import Dict, Any, Type
import torch
import torch.nn as nn

from ..core.interfaces import Model
from ..core.registry import MODELS
from .cnn import SiameseConvNetV1, CardsPlanesV1, ActionsHUEncoderV1
from .transformer import PokerTransformerV1, TransformerStateEncoder
from .state_encoder import CNNStateEncoder


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
        return SiameseConvNetV1(**config)

    @staticmethod
    def _create_transformer_model(
        config: Dict[str, Any], device: torch.device
    ) -> PokerTransformerV1:
        """Create transformer-based model."""
        return PokerTransformerV1(**config)

    @staticmethod
    def create_state_encoder(
        encoder_type: str, device: torch.device, tensor_env=None, **kwargs
    ):
        """Create state encoder based on type.

        Args:
            encoder_type: Type of encoder ('cnn' or 'transformer')
            device: Device to create encoder on
            tensor_env: Tensor environment (required for transformer)
            **kwargs: Additional arguments for encoder creation

        Returns:
            State encoder instance
        """
        if encoder_type == "cnn":
            # For CNN, we need the card and action encoders
            cards_encoder = kwargs.get("cards_encoder")
            actions_encoder = kwargs.get("actions_encoder")
            if cards_encoder is None or actions_encoder is None:
                raise ValueError(
                    "CNN state encoder requires cards_encoder and actions_encoder"
                )
            return CNNStateEncoder(cards_encoder, actions_encoder, device)
        elif encoder_type == "transformer":
            # For transformer, we need tensor_env and device
            if tensor_env is None:
                raise ValueError("Transformer state encoder requires tensor_env")
            return TransformerStateEncoder(tensor_env, device)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    @staticmethod
    def get_available_model_types() -> list[str]:
        """Get list of available model types."""
        return ["cnn", "transformer"]  # Add more as implemented
