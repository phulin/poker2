import math
import os
import tempfile

import torch

from alphaholdem.core.structured_config import Config, ValueHeadType
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.rl.losses import RebelSupervisedLoss
from alphaholdem.rl.rebel_replay import RebelBatch, RebelReplayBuffer
from alphaholdem.search.cfr_manager import CFRManager, SearchConfig
from alphaholdem.search.rebel_data_generator import NUM_HANDS


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
    for _ in (0, 1):
        beliefs = torch.full(
            (2, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32, device=env.device
        )
        mlp_features = encoder.encode(beliefs)
        features = mlp_features[idxs]
        # Combine all features for verification
        board_features = torch.where(features.board > 0, features.board / 51.0, -1.0)
        features_tensor = torch.cat(
            [features.context, board_features, features.beliefs], dim=-1
        )
    assert features_tensor.shape == (2, encoder.feature_dim)
    hero = features_tensor[:, 9 : 9 + encoder.belief_dim]
    opp = features_tensor[:, 9 + encoder.belief_dim :]
    torch.testing.assert_close(hero.sum(dim=1), torch.ones(2, device=env.device))
    torch.testing.assert_close(opp.sum(dim=1), torch.ones(2, device=env.device))


def test_rebel_replay_buffer_roundtrip():
    buffer = RebelReplayBuffer(
        capacity=16,
        feature_dim=10,
        num_actions=5,
        num_players=2,
        device=torch.device("cpu"),
    )
    features = torch.randn(4, 10)
    policy_targets = torch.softmax(torch.randn(4, NUM_HANDS, 5), dim=-1)
    value_targets = torch.randn(4, 2, NUM_HANDS)
    legal_masks = torch.ones(4, 5, dtype=torch.bool)
    batch = RebelBatch(
        features=features,
        policy_targets=policy_targets,
        value_targets=value_targets,
        legal_masks=legal_masks,
    )
    buffer.add_batch(batch)
    assert len(buffer) == 4
    sample = buffer.sample(2)
    assert sample.features.shape == (2, 10)
    assert sample.policy_targets.shape == (2, NUM_HANDS, 5)
    assert sample.value_targets.shape == (2, 2, NUM_HANDS)
    assert sample.legal_masks.shape == (2, 5)


def test_rebel_supervised_loss_finite():
    loss_fn = RebelSupervisedLoss()
    batch_size, num_actions = 3, 5
    logits = torch.randn(batch_size, NUM_HANDS, num_actions, requires_grad=True)
    hand_values = torch.randn(batch_size, 2, NUM_HANDS, requires_grad=True)
    policy_targets = torch.softmax(
        torch.randn(batch_size, NUM_HANDS, num_actions), dim=-1
    )
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)
    values = torch.randn(batch_size, 2, NUM_HANDS)
    batch = RebelBatch(
        features=torch.randn(batch_size, RebelFeatureEncoder.feature_dim),
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
    cfg.train.batch_size = 4
    cfg.train.replay_buffer_batches = 1
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
    cfg.model.input_dim = RebelFeatureEncoder.feature_dim
    cfg.model.hidden_dim = 64
    cfg.model.num_hidden_layers = 2
    cfg.model.value_head_type = ValueHeadType.scalar
    cfg.model.detach_value_head = True
    cfg.search.enabled = True
    cfg.search.depth = 0
    cfg.search.iterations = 1
    cfg.search.warm_start_iterations = 0
    cfg.search.dcfr_delay = 0
    cfg.search.branching = 4

    trainer = RebelCFRTrainer(cfg, torch.device("cpu"))
    metrics = trainer.train(num_steps=1)
    assert len(metrics) == 1
    assert metrics[0]["policy_loss"] is not None
    assert metrics[0]["loss"] is not None
    assert not math.isnan(metrics[0]["policy_loss"])
