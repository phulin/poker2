"""Tests for the EMAHelper class for model weight exponential moving averages."""

import copy

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from alphaholdem.utils.ema_helper import EMAHelper


class SimpleModel(nn.Module):
    """Simple test model for testing EMAHelper."""

    def __init__(self, hidden_dim: int = 10):
        super().__init__()
        self.linear1 = nn.Linear(5, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 3)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class ModelWithFrozenParams(nn.Module):
    """Model with some frozen parameters."""

    def __init__(self):
        super().__init__()
        self.trainable = nn.Linear(5, 10)
        self.frozen = nn.Linear(10, 3)
        self.frozen.weight.requires_grad = False
        self.frozen.bias.requires_grad = False

    def forward(self, x):
        x = self.trainable(x)
        x = self.frozen(x)
        return x


class TestEMAHelper:
    """Test cases for EMAHelper class."""

    def test_initialization(self):
        """Test EMAHelper initialization."""
        ema = EMAHelper(mu=0.999)
        assert ema.mu == 0.999
        assert ema.shadow == {}

        # Test default mu
        ema_default = EMAHelper()
        assert ema_default.mu == 0.999

    def test_register(self):
        """Test registering a module's parameters."""
        model = SimpleModel()
        ema = EMAHelper(mu=0.9)

        ema.register(model)

        # Check that all trainable parameters are registered
        param_names = {name for name, _ in model.named_parameters() if _.requires_grad}
        assert set(ema.shadow.keys()) == param_names

        # Check that shadow weights are clones of original parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in ema.shadow
                assert_close(ema.shadow[name], param.data)
                # Should be different tensors (clones)
                assert ema.shadow[name] is not param.data

    def test_register_with_frozen_params(self):
        """Test that frozen parameters are not registered."""
        model = ModelWithFrozenParams()
        ema = EMAHelper(mu=0.9)

        ema.register(model)

        # Only trainable parameters should be registered
        assert "trainable.weight" in ema.shadow
        assert "trainable.bias" in ema.shadow
        assert "frozen.weight" not in ema.shadow
        assert "frozen.bias" not in ema.shadow

    def test_update(self):
        """Test updating EMA shadow weights."""
        model = SimpleModel()
        ema = EMAHelper(mu=0.5)  # Use 0.5 for easier calculation

        ema.register(model)

        # Get initial parameter values
        initial_params = {
            name: param.data.clone() for name, param in model.named_parameters()
        }

        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data += 1.0

        # Update EMA
        ema.update(model)

        # Check that shadow weights are updated according to EMA formula
        # shadow = (1 - mu) * current + mu * old_shadow
        # shadow = 0.5 * (initial + 1.0) + 0.5 * initial = initial + 0.5
        for name, param in model.named_parameters():
            if param.requires_grad:
                expected = initial_params[name] + 0.5
                assert_close(ema.shadow[name], expected, rtol=1e-5, atol=1e-6)

    def test_update_multiple_times(self):
        """Test multiple EMA updates."""
        model = SimpleModel()
        ema = EMAHelper(mu=0.5)

        ema.register(model)

        # First update: add 1.0
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data += 1.0
        ema.update(model)

        # Second update: add another 1.0
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data += 1.0
        ema.update(model)

        # After two updates with mu=0.5:
        # After first: shadow = initial + 0.5
        # After second: shadow = 0.5 * (initial + 2.0) + 0.5 * (initial + 0.5)
        #              = 0.5 * initial + 1.0 + 0.5 * initial + 0.25
        #              = initial + 1.25
        initial_params = {
            name: param.data.clone() - 2.0
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        for name in initial_params:
            expected = initial_params[name] + 1.25
            assert_close(ema.shadow[name], expected, rtol=1e-5, atol=1e-6)

    def test_apply_to_module(self):
        """Test applying EMA weights to a module."""
        model = SimpleModel()
        ema = EMAHelper(mu=0.9)

        ema.register(model)

        # Modify shadow weights
        for name in ema.shadow:
            ema.shadow[name].data += 10.0

        # Apply to module
        ema.apply_to_module(model)

        # Check that model parameters match shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert_close(param.data, ema.shadow[name], rtol=1e-5, atol=1e-6)

    def test_create_ema_copy(self):
        """Test creating a copy of module with EMA weights."""
        model = SimpleModel()
        ema = EMAHelper(mu=0.9)

        ema.register(model)

        # Modify shadow weights
        for name in ema.shadow:
            ema.shadow[name].data += 5.0

        # Create EMA copy
        model_avg = ema.create_ema_copy(model)

        # Check that model_avg has EMA weights
        for name, param in model_avg.named_parameters():
            if param.requires_grad:
                assert_close(param.data, ema.shadow[name], rtol=1e-5, atol=1e-6)

        # Check that original model is unchanged
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not torch.allclose(
                    param.data, ema.shadow[name], rtol=1e-5, atol=1e-6
                )

        # Check that model_avg is a different object
        assert model_avg is not model

    def test_state_dict(self):
        """Test getting state dict."""
        model = SimpleModel()
        ema = EMAHelper(mu=0.9)

        ema.register(model)
        ema.update(model)

        state = ema.state_dict()

        assert isinstance(state, dict)
        assert set(state.keys()) == set(ema.shadow.keys())

        for name in state:
            assert_close(state[name], ema.shadow[name])

    def test_load_state_dict(self):
        """Test loading state dict."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        ema1 = EMAHelper(mu=0.9)
        ema2 = EMAHelper(mu=0.9)

        ema1.register(model1)
        ema1.update(model1)
        ema1.update(model1)  # Update twice

        # Save state from ema1
        state = ema1.state_dict()

        # Load into ema2
        ema2.load_state_dict(state)

        # Check that ema2 has same shadow weights
        assert set(ema2.shadow.keys()) == set(ema1.shadow.keys())
        for name in ema1.shadow:
            assert_close(ema2.shadow[name], ema1.shadow[name])

    def test_data_parallel_handling(self):
        """Test that DataParallel modules are handled correctly."""
        model = SimpleModel()
        # Wrap in DataParallel (even on CPU, this tests the code path)
        if torch.cuda.is_available():
            model_dp = nn.DataParallel(model)
        else:
            # On CPU, we can't actually use DataParallel, but we can test the code path
            # by creating a mock-like structure
            model_dp = model  # Just test with regular model

        ema = EMAHelper(mu=0.9)
        ema.register(model_dp)

        # Should register parameters from the underlying module
        param_names = {name for name, _ in model.named_parameters() if _.requires_grad}
        assert set(ema.shadow.keys()) == param_names

    def test_different_mu_values(self):
        """Test EMA with different mu (decay) values."""
        model = SimpleModel()

        # Test with high mu (slow adaptation)
        ema_slow = EMAHelper(mu=0.99)
        ema_slow.register(model)

        # Test with low mu (fast adaptation)
        ema_fast = EMAHelper(mu=0.1)
        ema_fast.register(model)

        # Modify parameters
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data += 10.0

        # Update both
        ema_slow.update(model)
        ema_fast.update(model)

        # Fast EMA should be closer to new values than slow EMA
        for name in ema_slow.shadow:
            diff_slow = torch.abs(
                ema_slow.shadow[name] - model.state_dict()[name]
            ).mean()
            diff_fast = torch.abs(
                ema_fast.shadow[name] - model.state_dict()[name]
            ).mean()

            # Fast EMA should have smaller difference (closer to current values)
            assert diff_fast < diff_slow

    def test_ema_convergence(self):
        """Test that EMA converges when model parameters stabilize."""
        model = SimpleModel()

        # Set model parameters to target value before registering
        target_value = 5.0
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data.fill_(target_value)

        ema = EMAHelper(mu=0.9)
        ema.register(model)

        # Keep model parameters constant and update EMA multiple times
        for _ in range(20):
            ema.update(model)

        # After many updates with constant input, shadow should be close to target
        for name in ema.shadow:
            assert_close(
                ema.shadow[name],
                torch.full_like(ema.shadow[name], target_value),
                rtol=1e-3,
                atol=1e-3,
            )

    def test_apply_to_module_preserves_structure(self):
        """Test that applying EMA doesn't change module structure."""
        model = SimpleModel()
        ema = EMAHelper(mu=0.9)

        ema.register(model)

        # Get original parameter shapes
        original_shapes = {
            name: param.shape for name, param in model.named_parameters()
        }

        # Modify and apply EMA
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    param.data += 1.0
        ema.update(model)
        ema.apply_to_module(model)

        # Check that shapes are preserved
        for name, param in model.named_parameters():
            assert param.shape == original_shapes[name]

    def test_multiple_modules(self):
        """Test that EMAHelper can work with multiple modules."""
        model1 = SimpleModel(hidden_dim=5)
        model2 = SimpleModel(hidden_dim=10)

        ema1 = EMAHelper(mu=0.9)
        ema2 = EMAHelper(mu=0.5)

        ema1.register(model1)
        ema2.register(model2)

        # Modify both models differently
        with torch.no_grad():
            for param in model1.parameters():
                if param.requires_grad:
                    param.data += 1.0
            for param in model2.parameters():
                if param.requires_grad:
                    param.data += 2.0

        ema1.update(model1)
        ema2.update(model2)

        # Check that they maintain separate shadow weights
        # Different hidden_dim means different parameter shapes
        assert (
            ema1.shadow["linear1.weight"].shape != ema2.shadow["linear1.weight"].shape
        )
        # Check that updates are independent - verify they have different parameter names/keys
        # or that the same parameter name has different shapes (which we already checked)
        assert set(ema1.shadow.keys()) == set(
            ema2.shadow.keys()
        )  # Same parameter names
        # But different shapes due to different hidden_dim
        assert ema1.shadow["linear1.bias"].shape != ema2.shadow["linear1.bias"].shape


if __name__ == "__main__":
    pytest.main([__file__])
