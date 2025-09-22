"""Tests for the EMA (Exponential Moving Average) class."""

import pytest
import torch
from alphaholdem.utils.ema import EMA


class TestEMA:
    """Test cases for the EMA class."""

    def test_ema_initialization(self):
        """Test EMA initialization with default and custom parameters."""
        # Test default initialization
        ema = EMA()
        assert ema.decay == 0.99
        assert ema.value == 0.0
        assert not ema.initialized

        # Test custom initialization
        ema_custom = EMA(decay=0.9, initial_value=5.0)
        assert ema_custom.decay == 0.9
        assert ema_custom.value == 5.0
        assert not ema_custom.initialized

    def test_ema_first_update(self):
        """Test that the first update sets the value directly."""
        ema = EMA(decay=0.9)

        result = ema.update(10.0)

        assert result == 10.0
        assert ema.value == 10.0
        assert ema.initialized

    def test_ema_subsequent_updates(self):
        """Test that subsequent updates use the EMA formula."""
        ema = EMA(decay=0.5, initial_value=0.0)

        # First update
        result1 = ema.update(10.0)
        assert result1 == 10.0

        # Second update: 0.5 * 10.0 + 0.5 * 20.0 = 15.0
        result2 = ema.update(20.0)
        assert result2 == 15.0

        # Third update: 0.5 * 15.0 + 0.5 * 30.0 = 22.5
        result3 = ema.update(30.0)
        assert result3 == 22.5

    def test_ema_decay_effects(self):
        """Test that different decay values produce expected behavior."""
        # High decay (slow adaptation)
        ema_slow = EMA(decay=0.95)
        ema_slow.update(10.0)
        result_slow = ema_slow.update(20.0)
        # 0.95 * 10.0 + 0.05 * 20.0 = 9.5 + 1.0 = 10.5

        # Low decay (fast adaptation)
        ema_fast = EMA(decay=0.1)
        ema_fast.update(10.0)
        result_fast = ema_fast.update(20.0)
        # 0.1 * 10.0 + 0.9 * 20.0 = 1.0 + 18.0 = 19.0

        assert result_slow < result_fast
        assert result_slow == 10.5
        assert result_fast == 19.0

    def test_ema_reset(self):
        """Test that reset returns EMA to initial state."""
        ema = EMA(decay=0.9, initial_value=0.0)

        # Update a few times
        ema.update(10.0)
        ema.update(20.0)
        assert ema.initialized
        assert ema.value != 0.0

        # Reset
        ema.reset(5.0)
        assert ema.value == 5.0
        assert not ema.initialized

        # Reset with default value
        ema.reset()
        assert ema.value == 0.0
        assert not ema.initialized

    def test_ema_sequence_convergence(self):
        """Test that EMA converges to a stable value with constant input."""
        ema = EMA(decay=0.9)

        # Feed constant value multiple times
        constant_value = 100.0
        values = []

        for i in range(10):
            result = ema.update(constant_value)
            values.append(result)

        # First value should be the constant value
        assert values[0] == constant_value
        # All subsequent values should also be the constant value
        assert all(v == constant_value for v in values)

    def test_ema_with_torch_tensors(self):
        """Test EMA with PyTorch tensor values."""
        ema = EMA(decay=0.8)

        # Test with tensor values
        tensor_val1 = torch.tensor(5.0)
        tensor_val2 = torch.tensor(15.0)

        result1 = ema.update(tensor_val1.item())
        assert result1 == 5.0

        result2 = ema.update(tensor_val2.item())
        # 0.8 * 5.0 + 0.2 * 15.0 = 4.0 + 3.0 = 7.0
        assert abs(result2 - 7.0) < 1e-10

    def test_ema_edge_cases(self):
        """Test EMA with edge case values."""
        ema = EMA(decay=0.5)

        # Test with zero
        result1 = ema.update(0.0)
        assert result1 == 0.0

        result2 = ema.update(0.0)
        assert result2 == 0.0

        # Test with negative values
        result3 = ema.update(-10.0)
        # 0.5 * 0.0 + 0.5 * (-10.0) = -5.0
        assert result3 == -5.0

        # Test with very large values
        result4 = ema.update(1e6)
        # 0.5 * (-5.0) + 0.5 * 1e6 = -2.5 + 5e5 = 499997.5
        assert abs(result4 - 499997.5) < 1e-6

    def test_ema_decay_validation(self):
        """Test that decay values are handled correctly."""
        # Test decay = 0 (should work but not smooth)
        ema_zero = EMA(decay=0.0)
        ema_zero.update(10.0)
        result = ema_zero.update(20.0)
        # 0.0 * 10.0 + 1.0 * 20.0 = 20.0
        assert result == 20.0

        # Test decay = 1 (should maintain first value)
        ema_one = EMA(decay=1.0)
        ema_one.update(10.0)
        result = ema_one.update(20.0)
        # 1.0 * 10.0 + 0.0 * 20.0 = 10.0
        assert result == 10.0

    def test_ema_multiple_instances(self):
        """Test that multiple EMA instances work independently."""
        ema1 = EMA(decay=0.5)
        ema2 = EMA(decay=0.8)

        # Update both with different values
        result1 = ema1.update(10.0)
        result2 = ema2.update(20.0)

        assert result1 == 10.0
        assert result2 == 20.0
        assert ema1.value != ema2.value

        # Update both with same value
        result1 = ema1.update(30.0)
        result2 = ema2.update(30.0)

        # Should be different due to different decay rates
        assert result1 != result2
        # ema1: 0.5 * 10.0 + 0.5 * 30.0 = 20.0
        # ema2: 0.8 * 20.0 + 0.2 * 30.0 = 16.0 + 6.0 = 22.0
        assert result1 == 20.0
        assert result2 == 22.0

    def test_ema_value_or_none(self):
        """Test the value_or_none method."""
        ema = EMA(decay=0.9)

        # Before any updates, should return None
        assert ema.value_or_none() is None

        # After first update, should return the value
        ema.update(10.0)
        assert ema.value_or_none() == 10.0

        # After subsequent updates, should return updated value
        ema.update(20.0)
        assert ema.value_or_none() == 11.0

        # After reset, should return None again
        ema.reset()
        assert ema.value_or_none() is None

        # After reset with custom value, should still return None (not initialized)
        ema.reset(5.0)
        assert ema.value_or_none() is None

        # After update following reset, should return the value
        ema.update(15.0)
        assert ema.value_or_none() == 15.0


if __name__ == "__main__":
    pytest.main([__file__])
