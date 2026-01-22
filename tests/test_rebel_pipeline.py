import math

import torch

from p2.core.structured_config import Config, StratifyConfig, ValueHeadType
from p2.env.card_utils import (
    NUM_HANDS,
    combo_suit_permutation_inverse_tensor,
    suit_permutations_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from p2.models.model_output import ModelOutput
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.losses import RebelSupervisedLoss
from p2.rl.rebel_batch import RebelBatch


def make_env(num_envs: int = 4) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
        flop_showdown=False,
    )
    env.reset()
    return env


def test_rebel_feature_encoder_shapes():
    env = make_env(2)
    encoder = RebelFeatureEncoder(env, device=env.device, dtype=torch.float32)
    idxs = torch.tensor([0, 1], device=env.device)
    beliefs = torch.full(
        (2, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32, device=env.device
    )
    mlp_features = encoder.encode(beliefs)
    features = mlp_features[idxs]

    # Verify features structure
    assert features.context.shape == (
        2,
        4,
    )  # to_act, position, pot_fraction, has_bet_flag
    assert features.street.shape == (2,)  # street indices
    assert features.board.shape == (2, 5)  # board indices
    assert features.beliefs.shape == (2, 2 * NUM_HANDS)  # beliefs

    # Verify beliefs sum to 1.0
    hero = features.beliefs[:, :NUM_HANDS]
    opp = features.beliefs[:, NUM_HANDS:]
    torch.testing.assert_close(hero.sum(dim=1), torch.ones(2, device=env.device))
    torch.testing.assert_close(opp.sum(dim=1), torch.ones(2, device=env.device))


def test_rebel_supervised_loss_finite():
    loss_fn = RebelSupervisedLoss()
    batch_size, num_actions = 3, 5
    logits = torch.randn(batch_size, NUM_HANDS, num_actions, requires_grad=True)
    policy_targets = torch.softmax(
        torch.randn(batch_size, NUM_HANDS, num_actions), dim=-1
    )
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
    values = torch.randn(batch_size, 2, NUM_HANDS)
    # Beliefs must be normalized probabilities
    beliefs_raw = torch.rand(batch_size, 2 * NUM_HANDS)
    beliefs = beliefs_raw / beliefs_raw.sum(dim=1, keepdim=True)
    mlp_features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.zeros(batch_size, 5),
        beliefs=beliefs,
    )
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=values,
        legal_masks=legal_masks,
    )
    output = ModelOutput(
        policy_logits=logits, value=torch.zeros(batch_size), hand_values=values
    )
    loss_dict = loss_fn(output, batch)
    assert torch.isfinite(loss_dict["total_loss"]).all()
    loss_dict["total_loss"].backward()


def test_rebel_cfr_trainer_single_step_cpu():
    cfg = Config()
    cfg.num_steps = 1
    cfg.num_envs = 2  # Reduced from default for faster execution
    cfg.train.batch_size = 2  # Reduced from 4 for faster execution
    cfg.train.episodes_per_step = 1
    cfg.train.replay_buffer_batches = 1
    cfg.train.value_reuse_goal = 2
    cfg.train.learning_rate = 3e-4
    cfg.train.value_coef = 1.0
    cfg.train.entropy_coef = 0.0
    cfg.train.grad_clip = 1.0
    cfg.train.max_sequence_length = 16
    cfg.env.bet_bins = [0.5, 1.0]
    cfg.env.stack = 1000
    cfg.env.sb = 5
    cfg.env.bb = 10
    cfg.env.flop_showdown = False
    cfg.model.name = "rebel_ffn"
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    cfg.model.input_dim = 2661
    cfg.model.hidden_dim = 32  # Reduced from 64 for faster execution
    cfg.model.num_hidden_layers = 1  # Reduced from 2 for faster execution
    cfg.model.value_head_type = ValueHeadType.scalar
    cfg.model.detach_value_head = True
    cfg.search.enabled = True
    cfg.search.depth = 0
    cfg.search.iterations = 1
    cfg.search.warm_start_iterations = 0
    cfg.search.dcfr_plus_delay = 0
    cfg.search.branching = 2  # Reduced from 4 for faster execution
    cfg.train.stratify_streets = [
        StratifyConfig(threshold=0, probabilities=[0.25, 0.25, 0.25, 0.25])
    ]

    trainer = RebelCFRTrainer(cfg, torch.device("cpu"))
    metrics = trainer.train(num_steps=1)
    assert len(metrics) == 1
    assert metrics[0]["policy_loss"] is not None
    assert metrics[0]["loss"] is not None
    assert not math.isnan(metrics[0]["policy_loss"])
    # New metrics: mean sample counts for both buffers
    assert "value_buffer_mean_sample_count" in metrics[0]
    assert "policy_buffer_mean_sample_count" in metrics[0]
    assert isinstance(metrics[0]["value_buffer_mean_sample_count"], float)
    assert isinstance(metrics[0]["policy_buffer_mean_sample_count"], float)


def test_permutation_loss_echo_model():
    """
    Test that permutation loss is 0 for an "echo" model that outputs
    hand values = beliefs * torch.arange(1326).

    This verifies that the permutation loss logic correctly handles
    the case where the model commutes with suit permutations.

    The echo model outputs: hand_values[b, p, i] = beliefs[b, p, i] * i
    For the model to commute with permutations, when we permute the input beliefs,
    the permuted output should be: hand_values_permuted[b, p, j] = beliefs_permuted[b, p, j] * original_combo_index_for_j
    where original_combo_index_for_j is obtained via combo_suit_permutation_inverse_tensor.
    """
    device = torch.device("cpu")
    batch_size = 4
    num_actions = 5

    # Create an "echo" model: hand_values = beliefs * torch.arange(1326)
    combo_indices = torch.arange(NUM_HANDS, device=device, dtype=torch.float32)

    # Create features with beliefs = normalized torch.arange(1326)
    beliefs_flat = combo_indices.unsqueeze(0).expand(batch_size, -1).repeat(1, 2)
    beliefs_p0 = beliefs_flat[:, :NUM_HANDS]
    beliefs_p1 = beliefs_flat[:, NUM_HANDS:]
    beliefs_p0 = beliefs_p0 / beliefs_p0.sum(dim=1, keepdim=True)
    beliefs_p1 = beliefs_p1 / beliefs_p1.sum(dim=1, keepdim=True)
    beliefs = torch.cat([beliefs_p0, beliefs_p1], dim=1)

    features = MLPFeatures(
        context=torch.randn(batch_size, 4),
        street=torch.zeros(batch_size, dtype=torch.long),
        to_act=torch.zeros(batch_size, dtype=torch.long),
        board=torch.zeros(batch_size, 5, dtype=torch.long),
        beliefs=beliefs,
    )

    # Create "echo" model output: hand_values = beliefs * combo_indices
    beliefs_reshaped = beliefs.view(batch_size, 2, NUM_HANDS)
    hand_values = beliefs_reshaped * combo_indices.unsqueeze(0).unsqueeze(0)
    policy_logits = torch.randn(batch_size, NUM_HANDS, num_actions)
    value = hand_values.mean(dim=-1).mean(dim=-1)

    output = ModelOutput(
        policy_logits=policy_logits,
        value=value,
        hand_values=hand_values,
    )

    # Permute the features
    suit_permutations_idxs = torch.randint(0, 24, (batch_size,), device=device)
    suit_permutations = suit_permutations_tensor(device=device)[suit_permutations_idxs]
    features_permuted = features.clone()
    features_permuted.permute_suits(suit_permutations)

    # For the echo model to commute, permuted output should use original combo indices
    beliefs_permuted_reshaped = features_permuted.beliefs.view(batch_size, 2, NUM_HANDS)
    combo_permutations_inverse = combo_suit_permutation_inverse_tensor(device=device)[
        suit_permutations_idxs
    ]  # (batch_size, 1326)
    # combo_permutations_inverse[b, j] = original combo index that maps to permuted combo j

    # Echo model output for permuted features: use original combo indices
    hand_values_permuted = (
        beliefs_permuted_reshaped
        * combo_permutations_inverse.unsqueeze(1).expand(-1, 2, -1)
    )

    output_permuted = ModelOutput(
        policy_logits=policy_logits,
        value=hand_values_permuted.mean(dim=-1).mean(dim=-1),
        hand_values=hand_values_permuted,
    )

    # Create batch and compute loss
    # Need to provide dummy targets (not used in permutation loss computation)
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
    dummy_value_targets = torch.zeros(batch_size, 2, NUM_HANDS)
    batch = RebelBatch(
        features=features,
        policy_targets=None,
        value_targets=dummy_value_targets,
        legal_masks=legal_masks,
    )

    loss_fn = RebelSupervisedLoss(permutation_weight=1.0)
    loss_dict = loss_fn(
        output,
        batch,
        output_permuted=output_permuted,
        suit_permutation_idxs=suit_permutations_idxs,
    )

    # The permutation loss should be 0 (or very close to 0)
    permutation_loss = loss_dict["permutation_loss"]
    assert (
        permutation_loss < 1e-6
    ), f"Permutation loss should be ~0, got {permutation_loss}"
