from __future__ import annotations

import torch

from alphaholdem.rl.losses import trinal_clip_ppo_loss


def test_trinal_policy_upper_clip_for_negative_advantages():
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

    out = trinal_clip_ppo_loss(
        logits=logits,
        values=values,
        actions=actions,
        log_probs_old=log_probs_old,
        advantages=advantages,
        returns=returns,
        legal_masks=legal_masks,
        epsilon=0.2,
        delta1=3.0,
        delta2=torch.tensor(-100.0),
        delta3=torch.tensor(100.0),
        value_coef=0.5,
        entropy_coef=0.01,
    )

    assert torch.isfinite(out["total_loss"])  # smoke check


def test_value_clipping_symmetry():
    # returns outside clip range should be brought within [delta2, delta3]
    batch = 4
    logits = torch.zeros(batch, 9)
    values = torch.zeros(batch)
    actions = torch.zeros(batch, dtype=torch.long)
    log_probs_old = torch.zeros(batch)
    advantages = torch.zeros(batch)
    returns = torch.tensor([-1000.0, -10.0, 10.0, 1000.0])
    legal_masks = torch.ones(batch, 9, dtype=torch.bool)

    out = trinal_clip_ppo_loss(
        logits,
        values,
        actions,
        log_probs_old,
        advantages,
        returns,
        legal_masks,
        epsilon=0.2,
        delta1=3.0,
        delta2=torch.tensor(-100.0),
        delta3=torch.tensor(100.0),
        value_coef=0.5,
        entropy_coef=0.01,
    )
    # Value loss computed vs clipped returns; ensure finite
    assert torch.isfinite(out["value_loss"])
