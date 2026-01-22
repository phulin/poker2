"""
Tests for model context managers.

This module tests the context managers in model_context.py that temporarily
change model modes (train/eval) and restore the original state.
"""

import torch
import torch.nn as nn

from p2.utils.model_context import model_eval, model_train


class SimpleModel(nn.Module):
    """Simple test model for testing context managers."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


class TestModelContext:
    """Test cases for model context managers."""

    def test_model_train_context_manager(self):
        """Test that model_train context manager works correctly."""
        model = SimpleModel()

        # Start in eval mode
        model.eval()
        assert not model.training

        # Use context manager to temporarily set to train mode
        with model_train(model):
            assert model.training

        # Should restore to eval mode
        assert not model.training

    def test_model_train_context_manager_already_training(self):
        """Test model_train context manager when model is already in training mode."""
        model = SimpleModel()

        # Start in train mode
        model.train()
        assert model.training

        # Use context manager to temporarily set to train mode
        with model_train(model):
            assert model.training

        # Should restore to train mode
        assert model.training

    def test_model_eval_context_manager(self):
        """Test that model_eval context manager works correctly."""
        model = SimpleModel()

        # Start in train mode
        model.train()
        assert model.training

        # Use context manager to temporarily set to eval mode
        with model_eval(model):
            assert not model.training

        # Should restore to train mode
        assert model.training

    def test_model_eval_context_manager_already_eval(self):
        """Test model_eval context manager when model is already in eval mode."""
        model = SimpleModel()

        # Start in eval mode
        model.eval()
        assert not model.training

        # Use context manager to temporarily set to eval mode
        with model_eval(model):
            assert not model.training

        # Should restore to eval mode
        assert not model.training

    def test_nested_context_managers(self):
        """Test nested context managers work correctly."""
        model = SimpleModel()

        # Start in eval mode
        model.eval()
        assert not model.training

        # Nested context managers
        with model_train(model):
            assert model.training

            with model_eval(model):
                assert not model.training

            # Should be back to train mode
            assert model.training

        # Should restore to original eval mode
        assert not model.training

    def test_context_manager_with_exception(self):
        """Test that context managers restore state even when exceptions occur."""
        model = SimpleModel()

        # Start in eval mode
        model.eval()
        assert not model.training

        # Context manager with exception
        try:
            with model_train(model):
                assert model.training
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should restore to eval mode even after exception
        assert not model.training

    def test_multiple_models(self):
        """Test context managers work with multiple models."""
        model1 = SimpleModel()
        model2 = SimpleModel()

        # Set different initial states
        model1.train()
        model2.eval()

        assert model1.training
        assert not model2.training

        # Use context managers on both models
        with model_eval(model1), model_train(model2):
            assert not model1.training
            assert model2.training

        # Should restore original states
        assert model1.training
        assert not model2.training

    def test_context_manager_with_torch_no_grad(self):
        """Test context managers work correctly with torch.no_grad."""
        model = SimpleModel()

        # Start in train mode
        model.train()
        assert model.training

        # Combined with torch.no_grad
        with torch.no_grad(), model_eval(model):
            assert not model.training

        # Should restore to train mode
        assert model.training

    def test_context_manager_preserves_grad_state(self):
        """Test that context managers don't affect gradient computation state."""
        model = SimpleModel()
        x = torch.randn(5, 10, requires_grad=True)

        # Start in train mode
        model.train()
        assert model.training

        # Use context manager
        with model_eval(model):
            assert not model.training
            y = model(x)
            # Should still be able to compute gradients
            y.sum().backward()
            assert x.grad is not None

        # Should restore to train mode
        assert model.training
