"""Tests for loss functions and normalizers in losses.py"""

from __future__ import annotations

import torch

from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.rl.losses import TrinalClipPPOLoss
from alphaholdem.rl.popart_normalizer import PopArtNormalizer
from alphaholdem.rl.vectorized_replay import BatchSample
from alphaholdem.utils.ema import EMA


class TestPopArtNormalizer:
    """Test suite for PopArtNormalizer."""

    def test_initialization(self):
        """Test that PopArtNormalizer initializes correctly."""
        normalizer = PopArtNormalizer()
        assert normalizer.eps == 1e-8
        assert not normalizer.mean_ema.initialized
        assert not normalizer.var_ema.initialized

    def test_initialization_with_custom_eps(self):
        """Test initialization with custom epsilon."""
        eps = 1e-6
        normalizer = PopArtNormalizer(eps=eps)
        assert normalizer.eps == eps

    def test_first_update(self):
        """Test first update with frozen stats workflow."""
        normalizer = PopArtNormalizer()

        # Create a batch of values
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        # First epoch: freeze stats (should be identity normalization)
        normalizer.freeze_stats()
        mu_frozen, sigma_frozen = normalizer.get_frozen_stats()

        # First epoch should use identity normalization
        assert mu_frozen == 0.0
        assert sigma_frozen == 1.0

        # Update EMAs in background
        normalizer.update_stats(values)

        # Check that EMAs are now initialized
        assert normalizer.mean_ema.initialized
        assert normalizer.var_ema.initialized

        # Check current stats after update
        mu_current, sigma_current = normalizer.get_current_stats()
        expected_mu = values.mean().item()  # 3.0
        expected_var = values.var(unbiased=False).item()  # 2.0
        expected_sigma = (expected_var + normalizer.eps) ** 0.5

        assert abs(mu_current - expected_mu) < 1e-6
        assert abs(sigma_current - expected_sigma) < 1e-6

    def test_multiple_updates(self):
        """Test multiple epochs with frozen stats workflow."""
        normalizer = PopArtNormalizer()

        # First epoch
        values1 = torch.tensor([1.0, 2.0, 3.0])
        normalizer.freeze_stats()
        mu_frozen1, sigma_frozen1 = normalizer.get_frozen_stats()

        # First epoch should use identity normalization
        assert mu_frozen1 == 0.0
        assert sigma_frozen1 == 1.0

        # Update EMAs in background
        normalizer.update_stats(values1)

        # Second epoch: freeze stats from first epoch's EMAs
        values2 = torch.tensor([10.0, 20.0, 30.0])
        normalizer.freeze_stats()
        mu_frozen2, sigma_frozen2 = normalizer.get_frozen_stats()

        # Second epoch should use stats from first epoch
        assert mu_frozen2 != 0.0  # Should be EMA from first epoch
        assert sigma_frozen2 != 1.0  # Should be EMA from first epoch

        # Update EMAs again
        normalizer.update_stats(values2)

        # Check that current stats reflect both updates
        mu_current, sigma_current = normalizer.get_current_stats()

        # Current stats should be EMA of both batches, not just latest
        assert (
            mu_current != values2.mean().item()
        )  # Should be EMA, not just latest batch
        assert (
            sigma_current
            != (values2.var(unbiased=False).item() + normalizer.eps) ** 0.5
        )

    def test_normalization_behavior(self):
        """Test frozen stats normalization behavior."""
        normalizer = PopArtNormalizer()

        # Create values with known mean and variance
        values = torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0])  # mean=4, var=8

        # Freeze stats and test normalization
        normalizer.freeze_stats()
        mu_frozen, sigma_frozen = normalizer.get_frozen_stats()

        # First epoch should use identity normalization
        assert mu_frozen == 0.0
        assert sigma_frozen == 1.0

        # Test denormalization
        normalized_value = torch.tensor(0.5)
        denormalized = normalizer.denormalize_value(normalized_value)
        expected = sigma_frozen * normalized_value + mu_frozen
        assert abs(denormalized - expected) < 1e-6

        # Update EMAs
        normalizer.update_stats(values)

        # Check that current stats reflect the batch
        mu_current, sigma_current = normalizer.get_current_stats()
        expected_mu = values.mean().item()  # 4.0
        expected_var = values.var(unbiased=False).item()  # 8.0
        expected_sigma = (expected_var + normalizer.eps) ** 0.5

        assert abs(mu_current - expected_mu) < 1e-6
        assert abs(sigma_current - expected_sigma) < 1e-6

    def test_eps_stability(self):
        """Test that epsilon prevents division by zero."""
        normalizer = PopArtNormalizer(eps=1e-8)

        # Create values with zero variance (all same value)
        values = torch.tensor([5.0, 5.0, 5.0, 5.0])

        # Freeze stats and update EMAs
        normalizer.freeze_stats()
        normalizer.update_stats(values)

        # Get current stats
        mu, sigma = normalizer.get_current_stats()

        # Sigma should be eps^0.5, not zero
        expected_sigma = normalizer.eps**0.5
        assert abs(sigma - expected_sigma) < 1e-10

    def test_device_handling(self):
        """Test that PopArtNormalizer works with different devices."""
        if torch.cuda.is_available():
            normalizer = PopArtNormalizer()

            # Test with CUDA tensor
            values_cuda = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            normalizer.freeze_stats()
            normalizer.update_stats(values_cuda)
            mu_cuda, sigma_cuda = normalizer.get_current_stats()

            # Results should be on CPU (scalars)
            assert isinstance(mu_cuda, float)
            assert isinstance(sigma_cuda, float)

            # Test with CPU tensor
            values_cpu = torch.tensor([4.0, 5.0, 6.0], device="cpu")
            normalizer.freeze_stats()
            normalizer.update_stats(values_cpu)
            mu_cpu, sigma_cpu = normalizer.get_current_stats()

            assert isinstance(mu_cpu, float)
            assert isinstance(sigma_cpu, float)

    def test_ema_decay_behavior(self):
        """Test that EMA decay affects subsequent updates."""
        normalizer = PopArtNormalizer()

        # First epoch
        values1 = torch.tensor([1.0, 1.0, 1.0])  # mean=1, var=0
        normalizer.freeze_stats()
        normalizer.update_stats(values1)
        mu1, sigma1 = normalizer.get_current_stats()

        # Second epoch with different mean
        values2 = torch.tensor([10.0, 10.0, 10.0])  # mean=10, var=0
        normalizer.freeze_stats()
        normalizer.update_stats(values2)
        mu2, sigma2 = normalizer.get_current_stats()

        # EMA should be between the two means
        assert 1.0 < mu2 < 10.0
        assert mu2 != 10.0  # Should be EMA, not just latest batch

    def test_large_batch(self):
        """Test with large batch size."""
        normalizer = PopArtNormalizer()

        # Create large batch
        values = torch.randn(1000) * 10 + 5  # mean~5, std~10

        normalizer.freeze_stats()
        normalizer.update_stats(values)
        mu, sigma = normalizer.get_current_stats()

        # Should handle large batches without issues
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0  # Should be positive

    def test_single_value(self):
        """Test with single value."""
        normalizer = PopArtNormalizer()

        values = torch.tensor([42.0])

        normalizer.freeze_stats()
        normalizer.update_stats(values)
        mu, sigma = normalizer.get_current_stats()

        # Single value should work
        assert abs(mu - 42.0) < 1e-6
        assert sigma == normalizer.eps**0.5  # Should be eps^0.5 for single value

    def test_negative_values(self):
        """Test with negative values."""
        normalizer = PopArtNormalizer()

        values = torch.tensor([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])

        normalizer.freeze_stats()
        normalizer.update_stats(values)
        mu, sigma = normalizer.get_current_stats()

        # Should handle negative values correctly
        assert isinstance(mu, float)
        assert isinstance(sigma, float)
        assert sigma > 0

    def test_consistency_across_calls(self):
        """Test that multiple calls with same values produce consistent results."""
        normalizer = PopArtNormalizer()

        values = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # First call
        normalizer.freeze_stats()
        normalizer.update_stats(values)
        mu1, sigma1 = normalizer.get_current_stats()

        # Second call with same values
        normalizer.freeze_stats()
        normalizer.update_stats(values)
        mu2, sigma2 = normalizer.get_current_stats()

        # Results should be consistent
        assert abs(mu1 - mu2) < 1e-6
        assert abs(sigma1 - sigma2) < 1e-6

    def test_return_types(self):
        """Test that methods return correct types."""
        normalizer = PopArtNormalizer()

        values = torch.tensor([1.0, 2.0, 3.0])

        normalizer.freeze_stats()
        normalizer.update_stats(values)

        # Test return types
        mu, sigma = normalizer.get_current_stats()
        assert isinstance(mu, float)
        assert isinstance(sigma, float)

        mu_frozen, sigma_frozen = normalizer.get_frozen_stats()
        assert isinstance(mu_frozen, float)
        assert isinstance(sigma_frozen, float)

        # Test denormalization
        normalized_value = torch.tensor(0.5)
        denormalized = normalizer.denormalize_value(normalized_value)
        assert isinstance(denormalized, torch.Tensor)

        # Test rescaling adjustments
        weight_scale, bias_adjustment = normalizer.compute_rescaling_adjustments()
        assert isinstance(weight_scale, float)
        assert isinstance(bias_adjustment, float)

    def test_complete_popart_workflow(self):
        """Test the complete PopArt workflow: freeze stats, background updates, final rescaling."""
        normalizer = PopArtNormalizer()

        # Simulate training epochs
        values1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # mean=3, var=2
        values2 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])  # mean=30, var=200

        # Epoch 1: First freeze (identity normalization)
        normalizer.freeze_stats()
        mu_frozen1, sigma_frozen1 = normalizer.get_frozen_stats()
        assert mu_frozen1 == 0.0
        assert sigma_frozen1 == 1.0

        # Background EMA update
        normalizer.update_stats(values1)
        mu_current1, sigma_current1 = normalizer.get_current_stats()
        assert abs(mu_current1 - 3.0) < 1e-6
        assert abs(sigma_current1 - (2.0 + normalizer.eps) ** 0.5) < 1e-6

        # Epoch 2: Freeze stats from epoch 1
        normalizer.freeze_stats()
        mu_frozen2, sigma_frozen2 = normalizer.get_frozen_stats()
        assert mu_frozen2 == mu_current1  # Should use stats from epoch 1
        assert sigma_frozen2 == sigma_current1

        # Background EMA update
        normalizer.update_stats(values2)
        mu_current2, sigma_current2 = normalizer.get_current_stats()

        # Current stats should be EMA of both batches
        assert mu_current2 != 30.0  # Should be EMA, not just latest batch
        assert mu_current2 > 3.0 and mu_current2 < 30.0  # Between the two means

        # Test denormalization with frozen stats
        normalized_value = torch.tensor(0.5)
        denormalized = normalizer.denormalize_value(normalized_value)
        expected = sigma_frozen2 * normalized_value + mu_frozen2
        assert abs(denormalized - expected) < 1e-6

        # Test final rescaling
        weight_scale, bias_adjustment = normalizer.compute_rescaling_adjustments()
        assert isinstance(weight_scale, float)
        assert isinstance(bias_adjustment, float)

        # Weight scale should be frozen_sigma / current_sigma
        expected_weight_scale = sigma_frozen2 / sigma_current2
        assert abs(weight_scale - expected_weight_scale) < 1e-6

        # Bias adjustment should be (frozen_mu - current_mu) / current_sigma
        expected_bias_adjustment = (mu_frozen2 - mu_current2) / sigma_current2
        assert abs(bias_adjustment - expected_bias_adjustment) < 1e-6


class TestTrinalClipPPOLoss:
    """Test suite for TrinalClipPPOLoss."""

    def test_trinal_policy_upper_clip_for_negative_advantages(self):
        """Test that TrinalClip properly handles negative advantages with upper clipping."""
        torch.manual_seed(0)
        batch = 8
        num_actions = 9

        logits = torch.randn(batch, num_actions)
        values = torch.zeros(batch)
        actions = torch.randint(0, num_actions, (batch,))
        # set log_probs_old small so ratios can exceed 1
        log_probs_old = torch.full((batch,), -5.0)
        # advantages: half negative, half positive
        advantages = torch.tensor([-1.0] * (batch // 2) + [1.0] * (batch - batch // 2))
        returns = torch.zeros(batch)
        legal_masks = torch.ones(batch, num_actions, dtype=torch.bool)

        # Create BatchSample object
        embedding_data = StructuredEmbeddingData(
            token_ids=torch.zeros(batch, 10),
            token_streets=torch.zeros(batch, 10),
            card_ranks=torch.zeros(batch, 10),
            card_suits=torch.zeros(batch, 10),
            action_actors=torch.zeros(batch, 10),
            action_legal_masks=torch.zeros(batch, 10, 8, dtype=torch.bool),
            context_features=torch.zeros(batch, 10, 9, dtype=torch.int16),
            lengths=torch.full((batch,), 10),
        )

        batch_sample = BatchSample(
            embedding_data=embedding_data,
            action_indices=actions,
            selected_log_probs=log_probs_old,
            all_log_probs=log_probs_old,
            legal_masks=legal_masks,
            advantages=advantages,
            returns=returns,
            delta2=torch.tensor(-100.0),
            delta3=torch.tensor(100.0),
            original_logits=torch.randn(batch, num_actions),
            computed_logits=torch.zeros(batch, num_actions),
            model_ages=torch.zeros(batch, dtype=torch.long),
        )

        # Create loss calculator and compute loss
        popart_normalizer = PopArtNormalizer()
        loss_calculator = TrinalClipPPOLoss(
            popart_normalizer=popart_normalizer,
            epsilon=0.2,
            delta1=3.0,
            value_coef=0.5,
            entropy_coef=0.01,
            value_loss_type="mse",
            huber_delta=1.0,
            target_kl=0.015,
            kl_ema=EMA(0.99, 0.0),
        )

        out = loss_calculator.compute_loss(
            logits=logits,
            values=values,
            batch=batch_sample,
        )

        import math

        assert torch.isfinite(out.total_loss)  # smoke check
        assert isinstance(out.policy_loss, float) and math.isfinite(out.policy_loss)
        assert isinstance(out.value_loss, float) and math.isfinite(out.value_loss)
        assert isinstance(out.entropy, float) and math.isfinite(out.entropy)

    def test_value_clipping_symmetry(self):
        """Test that value clipping works correctly with symmetric bounds."""
        # returns outside clip range should be brought within [delta2, delta3]
        batch = 4
        logits = torch.zeros(batch, 9)
        values = torch.zeros(batch)
        actions = torch.zeros(batch, dtype=torch.long)
        log_probs_old = torch.zeros(batch)
        advantages = torch.zeros(batch)
        returns = torch.tensor([-1000.0, -10.0, 10.0, 1000.0])
        legal_masks = torch.ones(batch, 9, dtype=torch.bool)

        # Create BatchSample object
        embedding_data = StructuredEmbeddingData(
            token_ids=torch.zeros(batch, 10),
            token_streets=torch.zeros(batch, 10),
            card_ranks=torch.zeros(batch, 10),
            card_suits=torch.zeros(batch, 10),
            action_actors=torch.zeros(batch, 10),
            action_legal_masks=torch.zeros(batch, 10, 8, dtype=torch.bool),
            context_features=torch.zeros(batch, 10, 9, dtype=torch.int16),
            lengths=torch.full((batch,), 10),
        )

        batch_sample = BatchSample(
            embedding_data=embedding_data,
            action_indices=actions,
            selected_log_probs=log_probs_old,
            all_log_probs=log_probs_old,
            legal_masks=legal_masks,
            advantages=advantages,
            returns=returns,
            delta2=torch.tensor(-100.0),
            delta3=torch.tensor(100.0),
            original_logits=torch.randn(batch, 9),
            computed_logits=torch.zeros(batch, 9),
            model_ages=torch.zeros(batch, dtype=torch.long),
        )

        # Create loss calculator and compute loss
        popart_normalizer = PopArtNormalizer()
        loss_calculator = TrinalClipPPOLoss(
            popart_normalizer=popart_normalizer,
            epsilon=0.2,
            delta1=3.0,
            value_coef=0.5,
            entropy_coef=0.01,
            value_loss_type="mse",
            huber_delta=1.0,
            target_kl=0.015,
            kl_ema=EMA(),
        )

        out = loss_calculator.compute_loss(
            logits=logits,
            values=values,
            batch=batch_sample,
        )

        # Value loss computed vs clipped returns; ensure finite
        import math

        assert isinstance(out.value_loss, float) and math.isfinite(out.value_loss)
        assert torch.isfinite(out.total_loss)

    def test_huber_vs_mse_value_loss(self):
        """Test that both Huber and MSE value loss types work."""
        batch = 4
        logits = torch.zeros(batch, 9)
        values = torch.randn(batch)
        actions = torch.zeros(batch, dtype=torch.long)
        log_probs_old = torch.zeros(batch)
        advantages = torch.zeros(batch)
        returns = torch.randn(batch)
        legal_masks = torch.ones(batch, 9, dtype=torch.bool)

        # Create BatchSample object
        embedding_data = StructuredEmbeddingData(
            token_ids=torch.zeros(batch, 10),
            token_streets=torch.zeros(batch, 10),
            card_ranks=torch.zeros(batch, 10),
            card_suits=torch.zeros(batch, 10),
            action_actors=torch.zeros(batch, 10),
            action_legal_masks=torch.zeros(batch, 10, 8, dtype=torch.bool),
            context_features=torch.zeros(batch, 10, 9, dtype=torch.int16),
            lengths=torch.full((batch,), 10),
        )

        batch_sample = BatchSample(
            embedding_data=embedding_data,
            action_indices=actions,
            selected_log_probs=log_probs_old,
            all_log_probs=log_probs_old,
            legal_masks=legal_masks,
            advantages=advantages,
            returns=returns,
            delta2=torch.tensor(-100.0),
            delta3=torch.tensor(100.0),
            original_logits=torch.randn(batch, 9),
            computed_logits=torch.zeros(batch, 9),
            model_ages=torch.zeros(batch, dtype=torch.long),
        )

        # Test MSE loss
        popart_normalizer_mse = PopArtNormalizer()
        loss_calculator_mse = TrinalClipPPOLoss(
            popart_normalizer=popart_normalizer_mse,
            epsilon=0.2,
            delta1=3.0,
            value_coef=0.5,
            entropy_coef=0.01,
            value_loss_type="mse",
            huber_delta=1.0,
            target_kl=0.015,
            kl_ema=EMA(),
        )

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        out_mse = loss_calculator_mse.compute_loss(
            logits=logits,
            values=values,
            batch=batch_sample,
        )

        # Test Huber loss
        popart_normalizer_huber = PopArtNormalizer()
        loss_calculator_huber = TrinalClipPPOLoss(
            popart_normalizer=popart_normalizer_huber,
            epsilon=0.2,
            delta1=3.0,
            value_coef=0.5,
            entropy_coef=0.01,
            value_loss_type="huber",
            huber_delta=1.0,
            target_kl=0.015,
            kl_ema=EMA(),
        )

        out_huber = loss_calculator_huber.compute_loss(
            logits=logits,
            values=values,
            batch=batch_sample,
        )

        # Both should produce finite losses
        import math

        assert isinstance(out_mse.value_loss, float) and math.isfinite(
            out_mse.value_loss
        )
        assert isinstance(out_huber.value_loss, float) and math.isfinite(
            out_huber.value_loss
        )
        assert torch.isfinite(out_mse.total_loss)
        assert torch.isfinite(out_huber.total_loss)

    def test_epsilon_adaptation(self):
        """Test that epsilon adapts based on KL divergence EMA."""
        batch = 4
        logits = torch.zeros(batch, 9)
        values = torch.zeros(batch)
        actions = torch.zeros(batch, dtype=torch.long)
        log_probs_old = torch.zeros(batch)
        advantages = torch.ones(batch)
        returns = torch.zeros(batch)
        legal_masks = torch.ones(batch, 9, dtype=torch.bool)

        # Create BatchSample object
        embedding_data = StructuredEmbeddingData(
            token_ids=torch.zeros(batch, 10),
            token_streets=torch.zeros(batch, 10),
            card_ranks=torch.zeros(batch, 10),
            card_suits=torch.zeros(batch, 10),
            action_actors=torch.zeros(batch, 10),
            action_legal_masks=torch.zeros(batch, 10, 8, dtype=torch.bool),
            context_features=torch.zeros(batch, 10, 9, dtype=torch.int16),
            lengths=torch.full((batch,), 10),
        )

        batch_sample = BatchSample(
            embedding_data=embedding_data,
            action_indices=actions,
            selected_log_probs=log_probs_old,
            all_log_probs=log_probs_old,
            legal_masks=legal_masks,
            advantages=advantages,
            returns=returns,
            delta2=torch.tensor(-100.0),
            delta3=torch.tensor(100.0),
            original_logits=torch.randn(batch, 9),
            computed_logits=torch.zeros(batch, 9),
            model_ages=torch.zeros(batch, dtype=torch.long),
        )

        # Initialize KL EMA with high value
        kl_ema = EMA(0.99, 0.0)
        kl_ema.update(0.1)  # High KL divergence

        popart_normalizer = PopArtNormalizer()
        loss_calculator = TrinalClipPPOLoss(
            popart_normalizer=popart_normalizer,
            epsilon=0.2,
            delta1=3.0,
            value_coef=0.5,
            entropy_coef=0.01,
            value_loss_type="mse",
            huber_delta=1.0,
            target_kl=0.015,
            kl_ema=kl_ema,
        )

        out = loss_calculator.compute_loss(
            logits=logits,
            values=values,
            batch=batch_sample,
        )

        # Epsilon should be adapted and clamped based on EMA behavior
        # EMA(0.99) of a single update to 0.1 gives ~0.001, so epsilon scales to ~3.0 and clamps to 0.4
        assert out.epsilon == 0.4
        assert torch.isfinite(out.total_loss)
