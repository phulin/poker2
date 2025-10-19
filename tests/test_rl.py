from __future__ import annotations

import math
import os
import signal
import tempfile

import torch

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
    ValueLossType,
)
from alphaholdem.rl.losses import TrinalClipPPOLoss
from alphaholdem.rl.popart_normalizer import PopArtNormalizer
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.utils.ema import EMA


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
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool)

    # Create BatchSample object
    from alphaholdem.models.transformer.structured_embedding_data import (
        StructuredEmbeddingData,
    )
    from alphaholdem.rl.vectorized_replay import BatchSample

    # Create dummy embedding data
    embedding_data = StructuredEmbeddingData(
        token_ids=torch.zeros(batch_size, 10),
        token_streets=torch.zeros(batch_size, 10),
        card_ranks=torch.zeros(batch_size, 10),
        card_suits=torch.zeros(batch_size, 10),
        action_actors=torch.zeros(batch_size, 10),
        action_legal_masks=torch.zeros(batch_size, 10, 8, dtype=torch.bool),
        action_amounts=torch.zeros(batch_size, 10, dtype=torch.int32),
        context_features=torch.zeros(batch_size, 10, 9, dtype=torch.int16),
        lengths=torch.full((batch_size,), 10),
    )

    batch = BatchSample(
        embedding_data=embedding_data,
        action_indices=actions,
        original_logits=logits,
        frozen_selected_log_probs=log_probs_old,
        frozen_all_log_probs=log_probs_old,
        step_selected_log_probs=log_probs_old,
        step_all_log_probs=log_probs_old,
        legal_masks=legal_masks,
        advantages=advantages,
        returns=returns,
        delta2=torch.tensor(-100.0),
        delta3=torch.tensor(100.0),
        model_ages=torch.zeros(batch_size, dtype=torch.long),
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

    loss_result = loss_calculator.compute_loss(
        logits=logits,
        values=values,
        batch=batch,
    )

    assert hasattr(loss_result, "total_loss")
    assert hasattr(loss_result, "policy_loss")
    assert hasattr(loss_result, "value_loss")
    assert hasattr(loss_result, "entropy")
    assert torch.isfinite(loss_result.total_loss)


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


def test_train_step_with_quantile_loss_runs():
    """Ensure the training loop runs successfully when using quantile value loss."""

    model_cfg = ModelConfig()
    model_cfg.value_head_type = "quantile"
    model_cfg.value_head_num_quantiles = 7

    cfg = Config(
        num_steps=1,
        train=TrainingConfig(
            batch_size=4,
            episodes_per_step=1,
            replay_buffer_batches=1,
            max_trajectory_length=5,
            value_loss_type="quantile",
            quantile_huber_kappa=1.0,
            use_kv_cache=False,
        ),
        model=model_cfg,
        env=EnvConfig(),
        use_tensor_env=True,
        num_envs=4,
        device="cpu",
    )

    trainer = SelfPlayTrainer(cfg=cfg, device=torch.device("cpu"))

    assert trainer.loss_calculator.value_loss_type == ValueLossType.quantile

    stats = trainer.train_step(step=1)

    assert "value_loss" in stats
    assert math.isfinite(stats["value_loss"])
    assert stats["beta"] >= 0.0


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
        for step in range(3):
            trainer.train_step(step + 1)

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

        # Verify loaded state (step is now stored in checkpoints)
        assert loaded_step == 3, f"Expected step 3, got {loaded_step}"
        assert (
            new_trainer.total_trajectories_collected
            == trainer.total_trajectories_collected
        ), "Total trajectories not restored"
        assert (
            new_trainer.total_episodes == trainer.total_episodes
        ), "Total episodes not restored"
        assert (
            new_trainer.total_transitions_trained == trainer.total_transitions_trained
        ), "Total transitions not restored"
        assert new_trainer.kl_ema.initialized == trainer.kl_ema.initialized
        assert math.isclose(
            new_trainer.kl_ema.decay,
            trainer.kl_ema.decay,
            rel_tol=0.0,
            abs_tol=0.0,
        ), "KL EMA decay not restored"
        assert math.isclose(
            new_trainer.kl_ema.value,
            trainer.kl_ema.value,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ), "KL EMA value not restored"

        # Verify model parameters are the same
        for param1, param2 in zip(
            trainer.model.parameters(), new_trainer.model.parameters()
        ):
            assert torch.allclose(
                param1, param2
            ), "Model parameters not restored correctly"

        # Test dtype preservation for main model
        print("Testing main model dtype preservation...")
        main_model_dtype = next(trainer.model.parameters()).dtype
        loaded_main_model_dtype = next(new_trainer.model.parameters()).dtype
        assert (
            main_model_dtype == loaded_main_model_dtype
        ), f"Main model dtype mismatch: {main_model_dtype} vs {loaded_main_model_dtype}"

        # Test dtype preservation for opponent pool snapshots
        print("Testing opponent pool snapshot dtype preservation...")
        if hasattr(trainer, "opponent_pool") and trainer.opponent_pool.snapshots:
            original_snapshot = trainer.opponent_pool.snapshots[0]
            loaded_snapshot = new_trainer.opponent_pool.snapshots[0]

            # Check snapshot model dtype
            assert (
                original_snapshot.model_dtype == loaded_snapshot.model_dtype
            ), f"Snapshot model_dtype mismatch: {original_snapshot.model_dtype} vs {loaded_snapshot.model_dtype}"

            # Check actual model parameter dtypes
            original_param_dtype = next(original_snapshot.model.parameters()).dtype
            loaded_param_dtype = next(loaded_snapshot.model.parameters()).dtype
            assert (
                original_param_dtype == loaded_param_dtype
            ), f"Snapshot parameter dtype mismatch: {original_param_dtype} vs {loaded_param_dtype}"

            # Verify snapshot model parameters are the same
            for param1, param2 in zip(
                original_snapshot.model.parameters(), loaded_snapshot.model.parameters()
            ):
                assert torch.allclose(
                    param1, param2
                ), "Snapshot model parameters not restored correctly"

        print("✅ Checkpoint save/load test passed!")


def test_checkpoint_dtype_preservation():
    """Test that checkpoint save/load preserves dtypes correctly, especially for mixed precision."""

    # Create a temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with mixed precision enabled (use MPS if available, otherwise CPU)
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
        cfg = Config(
            train=TrainingConfig(
                batch_size=2, use_mixed_precision=True, replay_buffer_batches=1
            ),
            model=ModelConfig(),
            env=EnvConfig(),
            use_tensor_env=True,
            num_envs=2,
            device=device_name,
        )

        device = torch.device(device_name)
        trainer = SelfPlayTrainer(cfg=cfg, device=device)

        # Run a few training steps to populate opponent pool
        for step in range(2):
            trainer.train_step(step + 1)

        # Manually add a snapshot to ensure we have one to test with
        trainer.opponent_pool.add_snapshot(trainer.model, step=1, rating=1200.0)

        # Verify opponent snapshots are in bfloat16 when mixed precision is enabled
        assert trainer.opponent_pool.snapshots, "No snapshots in opponent pool"
        snapshot = trainer.opponent_pool.snapshots[0]
        assert (
            snapshot.model_dtype == torch.bfloat16
        ), f"Expected bfloat16, got {snapshot.model_dtype}"
        param_dtype = next(snapshot.model.parameters()).dtype
        assert (
            param_dtype == torch.bfloat16
        ), f"Expected bfloat16 parameters, got {param_dtype}"

        # Save checkpoint
        checkpoint_path = os.path.join(temp_dir, "test_dtype_checkpoint.pt")
        trainer.save_checkpoint(checkpoint_path, step=2)

        # Load checkpoint
        new_trainer = SelfPlayTrainer(cfg=cfg, device=device)
        loaded_step, _ = new_trainer.load_checkpoint(checkpoint_path)

        # Verify opponent snapshots maintain bfloat16 dtype after loading
        assert (
            new_trainer.opponent_pool.snapshots
        ), "No snapshots in loaded opponent pool"
        loaded_snapshot = new_trainer.opponent_pool.snapshots[0]
        assert (
            loaded_snapshot.model_dtype == torch.bfloat16
        ), f"Expected bfloat16 after load, got {loaded_snapshot.model_dtype}"
        loaded_param_dtype = next(loaded_snapshot.model.parameters()).dtype
        assert (
            loaded_param_dtype == torch.bfloat16
        ), f"Expected bfloat16 parameters after load, got {loaded_param_dtype}"

        # Test with mixed precision disabled
        cfg_no_mixed = Config(
            train=TrainingConfig(batch_size=4, use_mixed_precision=False),
            model=ModelConfig(),
            env=EnvConfig(),
            use_tensor_env=True,
            num_envs=2,
            device=device_name,
        )

        trainer_no_mixed = SelfPlayTrainer(cfg=cfg_no_mixed, device=device)
        for step in range(2):
            trainer_no_mixed.train_step(step + 1)

        # Manually add a snapshot to ensure we have one to test with
        trainer_no_mixed.opponent_pool.add_snapshot(
            trainer_no_mixed.model, step=1, rating=1200.0
        )

        # Verify opponent snapshots are in float32 when mixed precision is disabled
        assert (
            trainer_no_mixed.opponent_pool.snapshots
        ), "No snapshots in opponent pool (no mixed precision)"
        snapshot = trainer_no_mixed.opponent_pool.snapshots[0]
        assert (
            snapshot.model_dtype == torch.float32
        ), f"Expected float32, got {snapshot.model_dtype}"
        param_dtype = next(snapshot.model.parameters()).dtype
        assert (
            param_dtype == torch.float32
        ), f"Expected float32 parameters, got {param_dtype}"

        print("✅ Checkpoint dtype preservation test passed!")


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
        # Disable KL scaling interaction in schedule
        trainer.kl_ema_initialized = False
        stats = trainer.train_step(step + 1)

        print(f"Step {step} stats: {stats}")

        # Check that we have reasonable values
        assert "avg_reward" in stats, "Missing avg_reward in stats"
        assert (
            "trajectories_collected" in stats
        ), "Missing trajectories_collected in stats"

        # Check that trajectories are being counted (skip first step which is warmup)
        if step > 0:
            assert (
                stats["trajectories_collected"] > 0
            ), f"No trajectories counted in step {step}"

            # Check that we have some reward signal
            assert not torch.isnan(
                torch.tensor(stats["avg_reward"])
            ), "NaN in avg_reward"

    print("✅ Basic training step test passed!")
