import math
import os
import tempfile

import torch

from alphaholdem.core.structured_config import Config, ValueHeadType
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
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
    for player in (0, 1):
        agents = torch.full((2,), player, dtype=torch.long, device=env.device)
        beliefs = torch.full(
            (2, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32, device=env.device
        )
        features = encoder.encode(agents, beliefs)[idxs]
    assert features.shape == (2, encoder.feature_dim)
    hero = features[:, 9 : 9 + encoder.belief_dim]
    opp = features[:, 9 + encoder.belief_dim :]
    torch.testing.assert_close(hero.sum(dim=1), torch.ones(2, device=env.device))
    torch.testing.assert_close(opp.sum(dim=1), torch.ones(2, device=env.device))


def test_cfr_manager_rebel_mode_runs():
    env = make_env(2)
    bet_bins = env.default_bet_bins
    manager = CFRManager(
        batch_size=2,
        env_proto=env,
        bet_bins=bet_bins,
        sequence_length=4,
        device=env.device,
        float_dtype=torch.float32,
        cfg=SearchConfig(depth=0, iterations=1, branching=4),
        use_rebel_features=True,
    )
    roots = manager.seed_roots(env, torch.tensor([0, 1]), src_tokens=None)
    model = RebelFFN(
        input_dim=RebelFeatureEncoder.feature_dim,
        num_actions=len(bet_bins) + 3,
        hidden_dim=32,
        num_hidden_layers=2,
    )
    model.to(env.device)
    res = manager.run_search(model)
    assert res.root_policy_collapsed.shape == (2, 4)
    torch.testing.assert_close(
        res.root_policy_collapsed.sum(dim=1), torch.ones(2, device=env.device)
    )
    assert res.root_hand_values is not None
    assert res.root_hand_value_weights is not None
    belief_dim = manager.rebel_encoder.belief_dim if manager.rebel_encoder else 1326
    assert res.root_hand_values.shape == (2, 2, belief_dim)


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
    acting = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    batch = RebelBatch(
        features=features,
        policy_targets=policy_targets,
        value_targets=value_targets,
        legal_masks=legal_masks,
        acting_players=acting,
    )
    buffer.add_batch(batch)
    assert len(buffer) == 4
    sample = buffer.sample(2)
    assert sample.features.shape == (2, 10)
    assert sample.policy_targets.shape == (2, NUM_HANDS, 5)
    assert sample.value_targets.shape == (2, 2, NUM_HANDS)
    assert sample.legal_masks.shape == (2, 5)
    assert sample.acting_players.shape == (2,)


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
    acting = torch.zeros(batch_size, dtype=torch.long)
    batch = RebelBatch(
        features=torch.randn(batch_size, RebelFeatureEncoder.feature_dim),
        policy_targets=policy_targets,
        value_targets=values,
        legal_masks=legal_masks,
        acting_players=acting,
    )
    loss_dict = loss_fn(logits, hand_values, batch)
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
    cfg.search.branching = 4

    trainer = RebelCFRTrainer(cfg, torch.device("cpu"))
    metrics = trainer.train(num_steps=1)
    assert len(metrics) == 1
    assert metrics[0].policy_loss is not None
    assert metrics[0].loss is not None
    assert not math.isnan(metrics[0].policy_loss)


def test_rebel_cfr_trainer_checkpoint_wandb_resumption():
    """Test that RebelCFRTrainer saves and loads wandb_run_id correctly."""

    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a config with wandb enabled
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
        cfg.search.branching = 4
        cfg.use_wandb = True
        cfg.wandb_project = "test_project"
        cfg.wandb_name = "test_run"

        device = torch.device("cpu")
        trainer = RebelCFRTrainer(cfg, device)

        # Run a training step
        trainer.train_step(1)

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path, step=1, wandb_run_id=None)

        # Verify file exists
        assert os.path.exists(checkpoint_path), "Checkpoint file not created"

        # Load checkpoint and verify wandb_run_id is None (since wandb is not actually initialized)
        loaded_step = trainer.load_checkpoint(checkpoint_path)

        # Verify loaded state
        assert trainer.cfg.wandb_run_id is None
        assert loaded_step == 1, f"Expected step 1, got {loaded_step}"

        # Test with a mock wandb run ID
        mock_wandb_run_id = "test_run_id_123"

        # Create a checkpoint with a mock wandb run ID
        checkpoint_data = torch.load(checkpoint_path, map_location=device)
        checkpoint_data["wandb_run_id"] = mock_wandb_run_id
        torch.save(checkpoint_data, checkpoint_path)

        # Load checkpoint and verify wandb_run_id is extracted correctly
        loaded_step = trainer.load_checkpoint(checkpoint_path)

        assert trainer.cfg.wandb_run_id == mock_wandb_run_id
        assert loaded_step == 1, f"Expected step 1, got {loaded_step}"

        print("✅ RebelCFRTrainer wandb resumption test passed!")
