"""
Tests for model utility functions.

This module tests the utility functions in model_utils.py for getting
predictions from models, handling legal masks, and converting between
different data formats.
"""

import torch
import torch.nn as nn

from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.utils.model_utils import (
    compute_masked_logits,
    get_best_action,
    get_log_probs,
    get_logits_log_probs_values,
    get_probs,
    get_probs_and_values,
)


class MockModelOutput:
    """Mock model output for testing."""

    def __init__(self, policy_logits, value, value_quantiles=None):
        self.policy_logits = policy_logits
        self.value = value
        self.value_quantiles = value_quantiles


class SimpleModel(nn.Module):
    """Simple test model for testing utility functions."""

    def __init__(self, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        self.linear = nn.Linear(10, num_actions)
        self.value_head = nn.Linear(10, 1)

    def forward(self, data):
        # Extract features from data
        if isinstance(data, StructuredEmbeddingData):
            # Use token_ids as features (simplified)
            # Need to ensure we have the right feature dimension
            batch_size = data.token_ids.shape[0]
            # Use deterministic features based on token_ids
            features = (
                data.token_ids.float().mean(dim=1, keepdim=True).expand(batch_size, 10)
            )
        else:
            features = data

        policy_logits = self.linear(features)
        value = self.value_head(features).squeeze(-1)

        return MockModelOutput(policy_logits, value)


class TestModelUtils:
    """Test cases for model utility functions."""

    def test_compute_masked_logits(self):
        """Test compute_masked_logits function."""
        batch_size, num_actions = 3, 4

        # Create test data
        logits = torch.randn(batch_size, num_actions)
        legal_masks = torch.tensor(
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, True, True, True],
            ]
        )

        # Compute masked logits
        masked_logits = compute_masked_logits(logits, legal_masks)

        # Check that illegal actions have very negative logits
        assert torch.all(masked_logits[~legal_masks] < -1e8)

        # Check that legal actions have original logits
        assert torch.allclose(masked_logits[legal_masks], logits[legal_masks])

    def test_get_log_probs_and_values(self):
        """Test get_log_probs_and_values function."""
        batch_size, num_actions = 2, 4

        # Create test data
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10)
        legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Get logits, log probs and values
        logits, log_probs, values, value_quantiles = get_logits_log_probs_values(
            model, data, legal_masks
        )

        # Check shapes
        assert logits.shape == (batch_size, num_actions)
        assert log_probs.shape == (batch_size, num_actions)
        assert values.shape == (batch_size,)

        # Check that logits and log probs are finite
        assert torch.all(torch.isfinite(logits))
        assert torch.all(torch.isfinite(log_probs))
        assert value_quantiles is None

    def test_get_log_probs(self):
        """Test get_log_probs function."""
        batch_size, num_actions = 2, 4

        # Create test data
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10)
        legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Get log probs
        log_probs = get_log_probs(model, data, legal_masks)

        # Check shape
        assert log_probs.shape == (batch_size, num_actions)

        # Check that log probs are finite
        assert torch.all(torch.isfinite(log_probs))

    def test_get_probs_and_values(self):
        """Test get_probs_and_values function."""
        batch_size, num_actions = 2, 4

        # Create test data
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10)
        legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Get probs and values (actually returns log_probs)
        log_probs, values = get_probs_and_values(model, data, legal_masks)

        # Check shapes
        assert log_probs.shape == (batch_size, num_actions)
        assert values.shape == (batch_size,)

        # Check that log probs are finite
        assert torch.all(torch.isfinite(log_probs))

    def test_get_probs(self):
        """Test get_probs function."""
        batch_size, num_actions = 2, 4

        # Create test data
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10)
        legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Get probs (actually returns log_probs)
        log_probs = get_probs(model, data, legal_masks)

        # Check shape
        assert log_probs.shape == (batch_size, num_actions)

        # Check that log probs are finite
        assert torch.all(torch.isfinite(log_probs))

    def test_get_best_action(self):
        """Test get_best_action function."""
        batch_size, num_actions = 3, 4

        # Create test data
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10)
        legal_masks = torch.tensor(
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, True, True, True],
            ]
        )

        # Get best actions
        best_actions = get_best_action(model, data, legal_masks)

        # Check shape
        assert best_actions.shape == (batch_size,)

        # Check that all actions are legal
        for i in range(batch_size):
            assert legal_masks[i, best_actions[i]]

    def test_with_structured_embedding_data(self):
        """Test utility functions with StructuredEmbeddingData."""
        batch_size, seq_len, num_actions = 2, 10, 4

        # Create StructuredEmbeddingData
        data = StructuredEmbeddingData.empty(
            batch_size=batch_size,
            seq_len=seq_len,
            num_bet_bins=10,
            device=torch.device("cpu"),
        )

        # Fill with some data
        data.token_ids = torch.randint(0, 100, (batch_size, seq_len))
        data.lengths = torch.tensor([seq_len, seq_len])

        # Create model and legal masks
        model = SimpleModel(num_actions)
        legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Test all utility functions
        log_probs = get_log_probs(model, data, legal_masks)
        log_probs2 = get_probs(model, data, legal_masks)
        logits, log_probs_vals, values, value_quantiles = get_logits_log_probs_values(
            model, data, legal_masks
        )
        log_probs_vals2, values2 = get_probs_and_values(model, data, legal_masks)
        best_actions = get_best_action(model, data, legal_masks)

        # Check shapes
        assert log_probs.shape == (batch_size, num_actions)
        assert log_probs2.shape == (batch_size, num_actions)
        assert logits.shape == (batch_size, num_actions)
        assert log_probs_vals.shape == (batch_size, num_actions)
        assert log_probs_vals2.shape == (batch_size, num_actions)
        assert values.shape == (batch_size,)
        assert values2.shape == (batch_size,)
        assert value_quantiles is None
        assert best_actions.shape == (batch_size,)

        # Check that values are consistent
        assert torch.allclose(values, values2)

    def test_legal_mask_handling(self):
        """Test that legal masks are properly handled."""
        batch_size, num_actions = 2, 4

        # Create test data
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10)

        # Test with different legal mask patterns
        legal_masks = torch.tensor(
            [
                [True, False, False, False],  # Only first action legal
                [False, False, False, True],  # Only last action legal
            ]
        )

        # Get predictions
        log_probs = get_probs(model, data, legal_masks)
        best_actions = get_best_action(model, data, legal_masks)

        # Check that illegal actions have very negative log probabilities
        assert torch.all(log_probs[0, 1:] < -1e8)
        assert torch.all(log_probs[1, :3] < -1e8)

        # Check that best actions are legal
        assert best_actions[0] == 0
        assert best_actions[1] == 3

    def test_gradient_flow(self):
        """Test that gradients flow correctly through utility functions."""
        batch_size, num_actions = 2, 4

        # Create test data with gradients
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10, requires_grad=True)
        legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Get predictions
        log_probs = get_probs(model, data, legal_masks)

        # Compute loss and backward
        loss = log_probs.sum()
        loss.backward()

        # Check that gradients exist
        assert data.grad is not None
        assert not torch.allclose(data.grad, torch.zeros_like(data.grad))

    def test_deterministic_behavior(self):
        """Test that utility functions produce deterministic results."""
        batch_size, num_actions = 2, 4

        # Create test data
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10)
        legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Get predictions twice
        log_probs1 = get_probs(model, data, legal_masks)
        log_probs2 = get_probs(model, data, legal_masks)

        # Should be identical
        assert torch.allclose(log_probs1, log_probs2)

    def test_edge_cases(self):
        """Test edge cases for utility functions."""
        batch_size, num_actions = 1, 1

        # Create test data
        model = SimpleModel(num_actions)
        data = torch.randn(batch_size, 10)
        legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

        # Test with single action
        log_probs = get_probs(model, data, legal_masks)
        assert log_probs.shape == (1, 1)
        # With single action, log prob should be 0 (log(1) = 0)
        assert torch.allclose(log_probs, torch.zeros(1, 1), atol=1e-6)

        # Test with all illegal actions
        legal_masks = torch.zeros(batch_size, num_actions, dtype=torch.bool)
        best_actions = get_best_action(model, data, legal_masks)
        assert best_actions.shape == (1,)
        # Should still return an action (even if illegal)
        assert best_actions[0] >= 0 and best_actions[0] < num_actions
