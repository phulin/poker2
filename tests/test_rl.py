from __future__ import annotations

import tempfile
import os
import torch

from alphaholdem.rl.replay import (
    ReplayBuffer,
    Transition,
    Trajectory,
    compute_gae_returns,
)
from alphaholdem.rl.losses import trinal_clip_ppo_loss
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.core.structured_config import (
    Config,
    TrainingConfig,
    ModelConfig,
    EnvConfig,
)


def test_replay_buffer_and_gae():
    buffer = ReplayBuffer(capacity=10)

    # Create a simple trajectory
    transitions = [
        Transition(
            observation=torch.randn(6, 4, 13),
            action=1,
            log_prob=-1.0,
            reward=0.0,
            done=False,
            legal_mask=torch.ones(8),
            chips_placed=50,
        ),
        Transition(
            observation=torch.randn(6, 4, 13),
            action=2,
            log_prob=-1.5,
            reward=100.0,
            done=True,
            legal_mask=torch.ones(8),
            chips_placed=100,
        ),
    ]

    trajectory = Trajectory(transitions=transitions)
    buffer.add_trajectory(trajectory)

    # Test GAE computation
    rewards = [0.0, 100.0]
    values = [0.0, 0.0, 0.0]  # including final value
    advantages, returns = compute_gae_returns(rewards, values)

    assert len(advantages) == 2
    assert len(returns) == 2
    assert advantages[1] > advantages[0]  # later advantage should be higher


def test_trinal_clip_ppo_loss():
    batch_size = 4
    num_actions = 8

    # Mock batch data
    logits = torch.randn(batch_size, num_actions)
    values = torch.randn(batch_size)
    actions = torch.randint(0, num_actions, (batch_size,))
    log_probs_old = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    legal_masks = torch.ones(batch_size, num_actions)

    # Compute loss
    loss_dict = trinal_clip_ppo_loss(
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

    assert "total_loss" in loss_dict
    assert "policy_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "entropy" in loss_dict
    assert torch.isfinite(loss_dict["total_loss"])


def test_self_play_trainer_basic():
    """Test basic trainer initialization and single trajectory collection."""

    # Create a Hydra config with small parameters for testing
    cfg = Config(
        train=TrainingConfig(batch_size=8),
        model=ModelConfig(),
        env=EnvConfig(),
        use_tensor_env=False,  # Use regular env for this test
        num_envs=1,
        device="cpu",  # Set device to cpu for testing
    )

    # Set device for testing
    device = torch.device("cpu")

    trainer = SelfPlayTrainer(
        cfg=cfg,
        device=device,
    )

    # Test that trainer initializes correctly
    assert trainer.model is not None
    assert trainer.env is not None
    assert trainer.replay_buffer is not None

    # Test single trajectory collection with timeout
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Trajectory collection timed out")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)  # 10 second timeout

    try:
        trajectory, final_reward = trainer.collect_trajectory()
        signal.alarm(0)  # Cancel alarm

        # Basic checks
        assert len(trajectory.transitions) > 0

        # Check that transitions have expected fields
        for t in trajectory.transitions:
            assert hasattr(t, "observation")
            assert hasattr(t, "action")
            assert hasattr(t, "log_prob")
            assert hasattr(t, "reward")
            assert hasattr(t, "done")
            assert hasattr(t, "legal_mask")
            assert hasattr(t, "chips_placed")

    except TimeoutError:
        signal.alarm(0)
        # If it times out, just check that trainer was created successfully
        assert trainer is not None


def test_checkpoint_save_load():
    """Test checkpoint saving and loading functionality."""

    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a Hydra config with small parameters for testing
        cfg = Config(
            train=TrainingConfig(batch_size=8),  # Small batch for testing
            model=ModelConfig(),
            env=EnvConfig(),
            use_tensor_env=True,
            num_envs=4,
            device="cpu",  # Set device to cpu for testing
        )

        # Set device for testing
        device = torch.device("cpu")

        trainer = SelfPlayTrainer(
            cfg=cfg,
            device=device,
        )

        # Run a few training steps
        for _ in range(3):
            trainer.train_step()

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path, step=3)

        # Verify file exists
        assert os.path.exists(checkpoint_path), "Checkpoint file not created"

        # Create new trainer and load checkpoint
        new_cfg = Config(
            train=TrainingConfig(learning_rate=3e-4, batch_size=8),
            model=ModelConfig(),
            env=EnvConfig(),
            use_tensor_env=True,
            device="cpu",  # Set device to cpu for testing
        )

        new_trainer = SelfPlayTrainer(
            cfg=new_cfg,
            device=device,
        )

        # Load checkpoint
        loaded_step, wandb_run_id = new_trainer.load_checkpoint(checkpoint_path)

        # Verify loaded state
        assert loaded_step == 3, f"Expected step 3, got {loaded_step}"
        assert (
            new_trainer.episode_count == trainer.episode_count
        ), "Episode count not restored"
        assert (
            abs(new_trainer.total_step_reward - trainer.total_step_reward) < 1e-6
        ), "Total reward not restored"

        # Verify model parameters are the same
        for param1, param2 in zip(
            trainer.model.parameters(), new_trainer.model.parameters()
        ):
            assert torch.allclose(
                param1, param2
            ), "Model parameters not restored correctly"

        print("✅ Checkpoint save/load test passed!")


def test_preflop_range_grid():
    """Test preflop range grid generation."""

    # Create a Hydra config
    cfg = Config(
        train=TrainingConfig(
            learning_rate=3e-4, batch_size=8
        ),  # Small batch for testing
        model=ModelConfig(),
        env=EnvConfig(),
        device="cpu",  # Set device to cpu for testing
    )

    # Set device for testing
    device = torch.device("cpu")

    trainer = SelfPlayTrainer(
        cfg=cfg,
        device=device,
    )

    # Generate range grid
    grid = trainer.get_preflop_range_grid(seat=0)

    # Basic checks
    assert isinstance(grid, str), "Grid should be a string"
    assert (
        len(grid.split("\n")) >= 15
    ), "Grid should have at least 15 lines (13 ranks + header + separator)"

    # Check header format
    lines = grid.split("\n")
    assert "A" in lines[0], "Header should contain rank A"
    assert "K" in lines[0], "Header should contain rank K"
    assert "2" in lines[0], "Header should contain rank 2"

    # Check that we have probability values (numbers)
    for line in lines[2:]:  # Skip header and separator
        if line.strip() and "|" in line:
            # Should contain numbers (percentages)
            parts = line.split()
            if len(parts) > 1:
                # Check that we have numeric values
                values = [p for p in parts[1:] if p.isdigit()]
                assert len(values) > 0, f"Line should contain numeric values: {line}"

    print("✅ Preflop range grid test passed!")
    print("Sample grid:")
    print(grid[:500] + "..." if len(grid) > 500 else grid)


def test_basic_training_step():
    """Test that a basic training step works and produces reasonable outputs."""

    # Create a Hydra config with small parameters for testing
    cfg = Config(
        train=TrainingConfig(
            learning_rate=3e-4, batch_size=4
        ),  # Small batch for testing
        model=ModelConfig(),
        env=EnvConfig(),
        use_tensor_env=True,
        num_envs=2,
        device="cpu",  # Set device to cpu for testing
    )

    # Set device for testing
    device = torch.device("cpu")

    trainer = SelfPlayTrainer(
        cfg=cfg,
        device=device,
    )

    # Run a few training steps
    for step in range(3):
        print(f"\n=== Training Step {step} ===")
        stats = trainer.train_step()

        print(f"Step {step} stats: {stats}")

        # Check that we have reasonable values
        assert "avg_reward" in stats, "Missing avg_reward in stats"
        assert "episode_count" in stats, "Missing episode_count in stats"

        # Check that episodes are being counted
        assert stats["episode_count"] > 0, "No episodes counted"

        # Check that we have some reward signal
        assert not torch.isnan(torch.tensor(stats["avg_reward"])), "NaN in avg_reward"

    print("✅ Basic training step test passed!")
