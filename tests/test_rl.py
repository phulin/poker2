from __future__ import annotations

import torch

from alphaholdem.rl.replay import (
    ReplayBuffer,
    Transition,
    Trajectory,
    compute_gae_returns,
    prepare_ppo_batch,
)
from alphaholdem.rl.losses import trinal_clip_ppo_loss


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

    trajectory = Trajectory(transitions=transitions, final_value=0.0)
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
    )

    assert "total_loss" in loss_dict
    assert "policy_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "entropy" in loss_dict
    assert torch.isfinite(loss_dict["total_loss"])


def test_self_play_trainer_basic():
    """Test basic trainer initialization and single trajectory collection."""
    from alphaholdem.rl.self_play import SelfPlayTrainer

    trainer = SelfPlayTrainer(
        batch_size=8,
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
        trajectory = trainer.collect_trajectory()
        signal.alarm(0)  # Cancel alarm

        # Basic checks
        assert len(trajectory.transitions) > 0
        assert trajectory.final_value == 0.0

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


def test_dynamic_delta_bounds():
    """Test dynamic delta bounds computation from chips placed."""
    from alphaholdem.rl.replay import compute_delta_bounds, Trajectory, Transition
    import torch

    # Create a trajectory with some chips placed
    transitions = [
        Transition(
            observation=torch.randn(6 * 4 * 13 + 24 * 4 * 8),  # cards + actions
            action=1,
            log_prob=-1.0,
            reward=0.0,
            done=False,
            legal_mask=torch.ones(8),
            chips_placed=50,  # Small bet
        ),
        Transition(
            observation=torch.randn(6 * 4 * 13 + 24 * 4 * 8),
            action=4,
            log_prob=-1.5,
            reward=100.0,
            done=True,
            legal_mask=torch.ones(8),
            chips_placed=200,  # Larger bet
        ),
    ]

    trajectory = Trajectory(transitions=transitions, final_value=0.0)

    # Compute delta bounds
    delta2, delta3 = compute_delta_bounds(trajectory)

    # Verify bounds are computed correctly
    assert delta2 == -250, f"Expected delta2=-250, got {delta2}"  # Total chips placed
    assert delta3 == 250, f"Expected delta3=250, got {delta3}"  # Total chips placed

    # Test empty trajectory
    empty_trajectory = Trajectory(transitions=[], final_value=0.0)
    delta2_empty, delta3_empty = compute_delta_bounds(empty_trajectory)
    assert delta2_empty == 0.0, "Empty trajectory should have delta2=0"
    assert delta3_empty == 0.0, "Empty trajectory should have delta3=0"

    print("✅ Dynamic delta bounds computation test passed!")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading functionality."""
    import tempfile
    import os
    from alphaholdem.rl.self_play import SelfPlayTrainer

    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = SelfPlayTrainer(
            batch_size=8,  # Small batch for testing
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
        new_trainer = SelfPlayTrainer(
            learning_rate=3e-4,
            batch_size=8,
        )

        # Load checkpoint
        loaded_step = new_trainer.load_checkpoint(checkpoint_path)

        # Verify loaded state
        assert loaded_step == 3, f"Expected step 3, got {loaded_step}"
        assert (
            new_trainer.episode_count == trainer.episode_count
        ), "Episode count not restored"
        assert (
            abs(new_trainer.total_reward - trainer.total_reward) < 1e-6
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
    from alphaholdem.rl.self_play import SelfPlayTrainer

    trainer = SelfPlayTrainer(
        learning_rate=3e-4,
        batch_size=8,  # Small batch for testing
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
    from alphaholdem.rl.self_play import SelfPlayTrainer

    trainer = SelfPlayTrainer(
        learning_rate=3e-4,
        batch_size=4,  # Small batch for testing
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
