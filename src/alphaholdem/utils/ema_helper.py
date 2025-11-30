import copy
import torch.nn as nn


class EMAHelper:
    """Helper class for maintaining Exponential Moving Average of model weights."""

    def __init__(self, mu: float = 0.999):
        """Initialize EMA helper.

        Args:
            mu: EMA decay rate. Higher values (closer to 1.0) mean slower updates.
        """
        self.mu = mu
        self.shadow = {}

    def register(self, module: nn.Module) -> None:
        """Register a module's parameters for EMA tracking.

        Args:
            module: PyTorch module whose parameters should be tracked.
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module) -> None:
        """Update EMA shadow weights from current module parameters.

        Args:
            module: PyTorch module to update EMA from.
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def apply_to_module(self, module: nn.Module) -> None:
        """Apply EMA shadow weights to a module in-place.

        Args:
            module: PyTorch module to apply EMA weights to.
        """
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def create_ema_copy(self, module: nn.Module) -> nn.Module:
        """Create a deep copy of the module with EMA weights applied.

        Args:
            module: PyTorch module to copy.

        Returns:
            Deep copy of the module with EMA weights applied.
        """
        module_copy = copy.deepcopy(module)
        self.apply_to_module(module_copy)
        return module_copy

    def state_dict(self) -> dict:
        """Get the state dict of shadow weights.

        Returns:
            Dictionary mapping parameter names to shadow weight tensors.
        """
        return self.shadow

    def load_state_dict(self, state_dict: dict) -> None:
        """Load shadow weights from a state dict.

        Args:
            state_dict: Dictionary mapping parameter names to shadow weight tensors.
        """
        self.shadow = state_dict
