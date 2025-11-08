import math

import torch

from alphaholdem.core.structured_config import Config, StratifyConfig, ValueHeadType
from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.rl.losses import RebelSupervisedLoss
from alphaholdem.rl.rebel_replay import RebelBatch, RebelReplayBuffer


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


def test_rebel_replay_buffer_roundtrip():
    buffer = RebelReplayBuffer(
        capacity=16,
        num_actions=5,
        num_players=2,
        num_context_features=4,
        device=torch.device("cpu"),
    )
    # Create MLPFeatures for the test
    mlp_features = MLPFeatures(
        context=torch.randn(4, 4),
        street=torch.zeros(4, dtype=torch.long),
        to_act=torch.zeros(4, dtype=torch.long),
        board=torch.zeros(4, 5, dtype=torch.long),
        beliefs=torch.randn(4, 2 * NUM_HANDS),
    )
    policy_targets = torch.softmax(torch.randn(4, NUM_HANDS, 5), dim=-1)
    value_targets = torch.randn(4, 2, NUM_HANDS)
    legal_masks = torch.ones(4, 5, dtype=torch.bool)
    batch = RebelBatch(
        features=mlp_features,
        policy_targets=policy_targets,
        value_targets=value_targets,
        legal_masks=legal_masks,
    )
    buffer.add_batch(batch)
    assert len(buffer) == 4
    sample = buffer.sample(2)
    assert sample.features.context.shape == (2, 4)
    assert sample.policy_targets.shape == (2, NUM_HANDS, 5)
    assert sample.value_targets.shape == (2, 2, NUM_HANDS)
    assert sample.legal_masks.shape == (2, 5)


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
