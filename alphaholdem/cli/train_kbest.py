#!/usr/bin/env python3
"""
Training script for AlphaHoldem with K-Best self-play.

This script demonstrates the K-Best self-play mechanism as described in the AlphaHoldem paper,
where the agent maintains a pool of K best historical versions and trains against them.
"""

import os
import time

import hydra
import torch

from ..core.structured_config import Config

# Import encoders and models to register them
from ..rl.self_play import SelfPlayTrainer
from ..rl.fixed_opponent_pool import FixedOpponentPool
from ..utils.config_loader import load_config_from_checkpoint
from ..utils.training_utils import (
    print_checkpoint_info,
    print_evaluation_results,
    print_preflop_range_grid,
    print_training_stats,
)


def train_exploiter(trainer, exploiter_trainer, step, cfg):
    """
    Train an exploiter against the current model and add it to the opponent pool.

    Args:
        trainer: Main SelfPlayTrainer instance
        exploiter_trainer: Exploiter SelfPlayTrainer instance with FixedOpponentPool
        step: Current training step
        cfg: Configuration object

    Returns:
        Dictionary with exploiter training statistics
    """
    print(f"\n🤖 Training exploiter at step {step + 1}...")
    exploiter_start_time = time.time()

    # Copy current model to exploiter
    exploiter_trainer.model.load_state_dict(trainer.model.state_dict())

    # Set the current model as the fixed opponent for the exploiter
    exploiter_trainer.opponent_pool.set_fixed_opponent(
        trainer.model, step + 1, trainer.opponent_pool.current_elo
    )

    # Train exploiter for specified number of steps
    exploiter_stats = []
    for exploiter_step in range(cfg.exploiter.training_steps):
        step_stats = exploiter_trainer.train_step(exploiter_step + 1)
        exploiter_stats.append(step_stats)

    # Calculate average exploiter performance
    avg_exploiter_reward = sum(s.get("avg_reward", 0) for s in exploiter_stats) / len(
        exploiter_stats
    )
    avg_exploiter_loss = sum(s.get("avg_loss", 0) for s in exploiter_stats) / len(
        exploiter_stats
    )

    # Add trained exploiter to main pool
    trainer.opponent_pool.add_snapshot(
        exploiter_trainer.model, step + 1, is_exploiter=True
    )

    exploiter_time = time.time() - exploiter_start_time
    print(f"✅ Exploiter training completed in {exploiter_time:.2f}s")
    print(f"   Average reward: {avg_exploiter_reward:.4f}")
    print(f"   Average loss: {avg_exploiter_loss:.4f}")
    print(f"   Pool size: {len(trainer.opponent_pool.snapshots)}")

    return {
        "exploiter_training_time": exploiter_time,
        "avg_exploiter_reward": avg_exploiter_reward,
        "avg_exploiter_loss": avg_exploiter_loss,
        "pool_size_after": len(trainer.opponent_pool.snapshots),
    }


def train_kbest(cfg: Config) -> SelfPlayTrainer:
    """
    Train AlphaHoldem agent using K-Best self-play with optional exploiter training.

    This function implements the main training loop with the following features:
    - K-Best opponent pool management
    - Optional exploiter training every N steps
    - Exploiters train against the current model using FixedOpponentPool
    - Comprehensive logging and checkpointing

    Args:
        cfg: Hydra configuration object containing all training parameters

    Returns:
        SelfPlayTrainer: The trained trainer instance
    """

    # Create checkpoint directory
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Set up device
    device = torch.device(
        "cuda"
        if cfg.device == "cuda" and torch.cuda.is_available()
        else (
            "mps"
            if cfg.device == "mps" and torch.backends.mps.is_available()
            else "cpu"
        )
    )
    print(f"Using device: {device}")

    if device.type == "cuda":
        print("✅ Using NVIDIA GPU (CUDA)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    elif device.type == "mps":
        print("✅ Using Apple M3 GPU (MPS)")
    elif cfg.device == "cpu":
        print("✅ Using CPU (selected)")
    else:
        print("⚠️ Using CPU (GPU not available)")

    # Load checkpoint if specified to get wandb run ID and merge config
    start_step = 0
    wandb_run_id_from_checkpoint = None
    merged_config = cfg  # Default to CLI config

    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(
            f"Loading checkpoint to extract config and wandb run ID: {cfg.resume_from}"
        )

        # Load config from checkpoint and merge with CLI overrides
        merged_config = load_config_from_checkpoint(cfg.resume_from, cfg)

        # Extract wandb run ID from checkpoint
        checkpoint = torch.load(
            cfg.resume_from, weights_only=False, map_location=device
        )
        wandb_run_id_from_checkpoint = checkpoint.get("wandb_run_id")
        if wandb_run_id_from_checkpoint:
            print(f"Found wandb run ID in checkpoint: {wandb_run_id_from_checkpoint}")
            merged_config.wandb_run_id = wandb_run_id_from_checkpoint
        else:
            print("No wandb run ID found in checkpoint")

    # Initialize trainer with merged config
    trainer = SelfPlayTrainer(cfg=merged_config, device=device)

    # Initialize exploiter trainer if enabled
    exploiter_trainer = None
    if merged_config.exploiter.enabled:
        print("🤖 Exploiter training enabled")

        # Create exploiter trainer with same config but different hyperparameters
        exploiter_cfg = merged_config
        exploiter_cfg.train.learning_rate = merged_config.exploiter.learning_rate
        exploiter_cfg.train.batch_size = merged_config.exploiter.batch_size
        exploiter_cfg.train.num_epochs = merged_config.exploiter.num_epochs
        exploiter_cfg.train.entropy_coef = merged_config.exploiter.entropy_coef

        exploiter_trainer = SelfPlayTrainer(cfg=exploiter_cfg, device=device)

        # Replace exploiter's opponent pool with a fixed opponent pool
        exploiter_trainer.opponent_pool = FixedOpponentPool(
            k_factor=merged_config.k_factor,
            use_mixed_precision=merged_config.train.use_mixed_precision,
        )

        print("✅ Exploiter trainer initialized")
    else:
        print("❌ Exploiter training disabled")

    # Load checkpoint if specified (after trainer initialization for wandb resumption)
    if cfg.resume_from and os.path.exists(cfg.resume_from):
        print(f"Loading checkpoint: {cfg.resume_from}")
        start_step, _ = trainer.load_checkpoint(cfg.resume_from)

    # Record training start time
    training_start_time = time.time()

    print(f"Starting K-Best training from step {start_step}")
    if cfg.use_tensor_env:
        print(f"Using tensorized environment with {cfg.num_envs} parallel environments")
    else:
        print("Using scalar environment")
    if cfg.env.flop_showdown:
        print("Using FLOP SHOWDOWN environment")
    if cfg.use_wandb:
        print(f"📊 Wandb logging enabled - check https://wandb.ai for real-time plots!")
        print(f"   Project: {cfg.wandb_project}")
        if cfg.wandb_name:
            print(f"   Run name: {cfg.wandb_name}")
        print(f"   Tags: {cfg.wandb_tags}")
    # Collection runs until batch_size steps; no fixed trajectories per step
    print(
        f"Training start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}"
    )
    print()

    # Training loop
    for step in range(start_step, cfg.num_steps):
        step_start_time = time.time()

        # Training step: collects until batch_size steps
        stats = trainer.train_step(step + 1)  # Pass 1-indexed step for consistency

        # Calculate times
        step_elapsed_time = time.time() - step_start_time
        total_elapsed_time = time.time() - training_start_time

        # Format times
        step_time_str = f"{step_elapsed_time:.2f}s"
        total_time_str = f"{total_elapsed_time:.1f}s"

        # Logging
        print_training_stats(stats, step, cfg.num_steps, step_time_str, total_time_str)
        # Optional clipping debug of first sample in batch
        if "first_ret" in stats:
            print(
                "  clip_debug "
                f"ret0 {stats['first_ret']:.4f} "
                f"d2 {stats['first_d2']:.4f} d3 {stats['first_d3']:.4f} "
                f"min {stats['first_min_b']:.4f} max {stats['first_max_b']:.4f} "
                f"retc {stats['first_ret_clipped']:.4f} "
                f"out_of_bounds {bool(stats['first_ret_out_of_bounds'])}"
            )

        # Evaluation against pool
        if (step + 1) % cfg.eval_interval == 0:
            eval_results = trainer.evaluate_against_pool(min_games=20)
            print_evaluation_results(eval_results)

        # Exploiter training
        if (
            merged_config.exploiter.enabled
            and exploiter_trainer is not None
            and (step + 1) % merged_config.exploiter.training_interval == 0
        ):
            exploiter_stats = train_exploiter(
                trainer, exploiter_trainer, step, merged_config
            )

        # Checkpointing
        if (step + 1) % cfg.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                cfg.checkpoint_dir, f"checkpoint_step_{step + 1}.pt"
            )
            trainer.save_checkpoint(checkpoint_path, step + 1)
            checkpoint_path = os.path.join(cfg.checkpoint_dir, f"latest_model.pt")
            trainer.save_checkpoint(checkpoint_path, step + 1)

            # Also save the best model if it has the highest ELO
            best_checkpoint_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            if (
                not os.path.exists(best_checkpoint_path)
                or stats["current_elo"] > trainer.opponent_pool.get_best_snapshot().elo
            ):
                trainer.save_checkpoint(best_checkpoint_path, step + 1)
                print_checkpoint_info(
                    best_checkpoint_path, stats["current_elo"], is_best=True
                )

            print_preflop_range_grid(trainer, step + 1, seat=0)

    # Final evaluation
    final_total_time = time.time() - training_start_time
    print(f"\nFinal evaluation against opponent pool...")
    final_eval = trainer.evaluate_against_pool(min_games=100)
    print_evaluation_results(final_eval)
    print(
        f"Total training time: {final_total_time:.1f}s ({final_total_time/3600:.2f} hours)"
    )

    # Save final checkpoint
    final_checkpoint_path = os.path.join(cfg.checkpoint_dir, "final_checkpoint.pt")
    trainer.save_checkpoint(final_checkpoint_path, cfg.num_steps)

    # Print preflop range grid
    print_preflop_range_grid(
        trainer, cfg.num_steps, seat=0, title="Final Preflop Range Grid"
    )

    return trainer


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg) -> None:
    """
    Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Convert Hydra config to Config dataclass
    from omegaconf import OmegaConf

    from alphaholdem.core.structured_config import (
        Config,
        EnvConfig,
        ExploiterConfig,
        ModelConfig,
        StateEncoderConfig,
        TrainingConfig,
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create nested config objects
    train_config = TrainingConfig(**cfg_dict.get("train", {}))
    model_config = ModelConfig(**cfg_dict.get("model", {}))
    env_config = EnvConfig(**cfg_dict.get("env", {}))
    state_encoder_config = StateEncoderConfig(**cfg_dict.get("state_encoder", {}))
    exploiter_config = ExploiterConfig(**cfg_dict.get("exploiter", {}))

    # Create Config dataclass
    config = Config(
        train=train_config,
        model=model_config,
        env=env_config,
        state_encoder=state_encoder_config,
        exploiter=exploiter_config,
        **{
            k: v
            for k, v in cfg_dict.items()
            if k not in ["train", "model", "env", "state_encoder", "exploiter"]
        },
    )

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)

    # Train the agent - pass config dataclass
    train_kbest(config)

    print("Training completed!")


if __name__ == "__main__":
    main()
