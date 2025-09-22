from __future__ import annotations

import torch

from alphaholdem.rl.losses import TrinalClipPPOLoss
from alphaholdem.rl.vectorized_replay import BatchSample
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)


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
    )

    # Create loss calculator and compute loss
    loss_calculator = TrinalClipPPOLoss(
        epsilon=0.2,
        delta1=3.0,
        value_coef=0.5,
        entropy_coef=0.01,
        value_loss_type="mse",
        huber_delta=1.0,
        target_kl=0.015,
    )

    out = loss_calculator.compute_loss(
        logits=logits,
        values=values,
        batch=batch_sample,
    )

    assert torch.isfinite(out.total_loss)  # smoke check


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
    )

    # Create loss calculator and compute loss
    loss_calculator = TrinalClipPPOLoss(
        epsilon=0.2,
        delta1=3.0,
        value_coef=0.5,
        entropy_coef=0.01,
        value_loss_type="mse",
        huber_delta=1.0,
        target_kl=0.015,
    )

    out = loss_calculator.compute_loss(
        logits=logits,
        values=values,
        batch=batch_sample,
    )

    # Value loss computed vs clipped returns; ensure finite
    assert torch.isfinite(out.value_loss)
