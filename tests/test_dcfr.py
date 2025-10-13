import torch

from alphaholdem.search.dcfr import (
    collapse_policy_full_to_4,
    collapse_legal_full_to_4,
    run_dcfr,
)
from alphaholdem.search.cfr_manager import CFRManager
from alphaholdem.rl.losses import CFRDistillationLoss, ValueLossType


def test_collapse_policy_full_to_4_sums_raise_bins():
    N, B = 3, 8
    prob = torch.zeros(N, B)
    prob[:, 0] = 0.1  # fold
    prob[:, 1] = 0.2  # call
    prob[:, 2:7] = 0.1  # five preset raise bins total 0.5
    prob[:, 7] = 0.2  # all-in
    out = collapse_policy_full_to_4(prob)
    assert out.shape == (N, 4)
    torch.testing.assert_close(out[:, 0], torch.full((N,), 0.1))
    torch.testing.assert_close(out[:, 1], torch.full((N,), 0.2))
    torch.testing.assert_close(out[:, 2], torch.full((N,), 0.5))
    torch.testing.assert_close(out[:, 3], torch.full((N,), 0.2))


def test_collapse_legal_full_to_4_any_raise_bin_legal():
    N, B = 2, 6
    mask = torch.zeros(N, B, dtype=torch.bool)
    mask[:, 0] = True
    mask[:, -1] = True  # all-in
    mask[0, 2] = True  # one preset legal for first row
    out = collapse_legal_full_to_4(mask)
    assert out.shape == (N, 4)
    assert out[0, 2]  # bet-pot is legal if any preset is legal
    assert not out[1, 2]


def test_run_dcfr_shapes_and_masking():
    # Build a tiny tree with depth=1: M = B * (1 + 4)
    BATCH = 2
    depth = 1
    BINS = 7
    M = BATCH * (1 + 4)
    logits_full = torch.zeros(M, BINS)
    legal_full = torch.zeros(M, BINS, dtype=torch.bool)
    # make fold/call/all-in legal everywhere and one preset
    legal_full[:, 0] = True
    legal_full[:, 1] = True
    legal_full[:, -1] = True
    legal_full[:, 2] = True
    # values for leaves only
    depth_offsets = [0, BATCH, BATCH + 4 * BATCH]
    values = torch.zeros(M)
    # leaves are last slice
    leaf_sl = slice(depth_offsets[1], depth_offsets[2])
    values[leaf_sl] = torch.tensor([1.0] * (4 * BATCH))

    # to_act: all parents act by p0 in this simple test
    to_act = torch.zeros(M, dtype=torch.long)
    res = run_dcfr(
        logits_full=logits_full,
        legal_mask_full=legal_full,
        values=values,
        to_act=to_act,
        depth_offsets=depth_offsets,
        depth=depth,
        iterations=10,
    )
    assert res.root_policy_collapsed.shape == (BATCH, 4)
    # probabilities are valid
    s = res.root_policy_collapsed.sum(dim=1)
    torch.testing.assert_close(s, torch.ones(BATCH))


def test_cfr_distillation_loss_computation():
    """Test that CFR distillation loss can be computed correctly."""
    BATCH = 2
    BINS = 7
    depth = 1

    # Create test inputs
    logits_full = torch.zeros(BATCH * (1 + 4), BINS)
    legal_full = torch.zeros(BATCH * (1 + 4), BINS, dtype=torch.bool)
    legal_full[:, 0] = True  # fold legal
    legal_full[:, 1] = True  # call legal
    legal_full[:, -1] = True  # all-in legal
    legal_full[:, 2] = True  # one preset legal

    values = torch.ones(BATCH * (1 + 4))
    depth_offsets = [0, BATCH, BATCH + 4 * BATCH]

    # Run DCFR
    to_act = torch.zeros(BATCH * (1 + 4), dtype=torch.long)
    res = run_dcfr(
        logits_full=logits_full,
        legal_mask_full=legal_full,
        values=values,
        to_act=to_act,
        depth_offsets=depth_offsets,
        depth=depth,
        iterations=5,
    )

    # Create dummy model policy
    model_logits = torch.randn(BATCH, BINS)
    model_legal = torch.zeros(BATCH, BINS, dtype=torch.bool)
    model_legal[:, 0] = True
    model_legal[:, 1] = True
    model_legal[:, -1] = True
    model_legal[:, 2] = True

    # Compute distillation loss as would be done in training
    masked_logits = torch.where(model_legal, model_logits, -1e9)
    model_probs_full = torch.softmax(masked_logits, dim=-1)
    model_probs_4 = CFRManager.collapse_policy_full_to_4(model_probs_full)

    # Test KL computation
    cfr_target = res.root_policy_collapsed
    cfr_target_stable = cfr_target + 1e-8
    model_probs_4_stable = model_probs_4 + 1e-8

    cfr_target_norm = cfr_target_stable / cfr_target_stable.sum(dim=-1, keepdim=True)
    model_probs_4_norm = model_probs_4_stable / model_probs_4_stable.sum(
        dim=-1, keepdim=True
    )

    kl_div = (cfr_target_norm * torch.log(cfr_target_norm / model_probs_4_norm)).sum(
        dim=-1
    )
    assert kl_div.shape == (BATCH,)
    assert (kl_div >= 0).all()  # KL divergence should be non-negative


def test_cfr_distillation_loss():
    """Test CFRDistillationLoss with dummy data."""
    from alphaholdem.rl.popart_normalizer import PopArtNormalizer

    BATCH = 2
    BINS = 7

    # Create CFR target
    cfr_target = torch.randn(BATCH, 4)
    cfr_target = torch.softmax(
        cfr_target, dim=-1
    )  # Make it a valid probability distribution

    # Create dummy inputs
    logits = torch.randn(BATCH, BINS)
    values = torch.randn(BATCH)
    advantages = torch.randn(BATCH)
    returns = torch.randn(BATCH)

    # Create a mock batch object
    class MockBatch:
        def __init__(self):
            self.action_indices = torch.randint(0, BINS, (BATCH,))
            self.advantages = advantages
            self.returns = returns
            self.delta2 = torch.full((BATCH,), -1.0)
            self.delta3 = torch.full((BATCH,), 1.0)
            self.legal_masks = torch.randint(0, 2, (BATCH, BINS)).bool()
            # Ensure at least one legal action per row
            for i in range(BATCH):
                if not self.legal_masks[i].any():
                    self.legal_masks[i, 0] = True

    batch = MockBatch()

    # Create PopArt normalizer
    popart = PopArtNormalizer()

    # Create CFR distillation loss
    loss_fn = CFRDistillationLoss(
        popart_normalizer=popart,
        value_coef=1.0,
        entropy_coef=0.01,
        value_loss_type=ValueLossType.mse,
        huber_delta=1.0,
    )

    # Test loss computation
    result = loss_fn.compute_loss(logits, values, batch, cfr_target=cfr_target)

    # Verify results
    assert result.total_loss > 0
    assert result.policy_loss > 0  # KL divergence should be positive
    assert result.value_loss_tensor > 0  # MSE loss should be positive
    assert result.entropy >= 0  # Entropy should be non-negative
    assert result.cfr_kl >= 0  # KL divergence should be non-negative
