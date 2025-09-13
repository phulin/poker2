#!/usr/bin/env python3
"""
Training script for AlphaHoldem with K-Best self-play.

This script demonstrates the K-Best self-play mechanism as described in the AlphaHoldem paper,
where the agent maintains a pool of K best historical versions and trains against them.
"""

import os
import time
import torch
import hydra
from ..core.structured_config import Config
from ..rl.self_play import SelfPlayTrainer
from ..utils.config_loader import load_config_from_checkpoint

# Import encoders and models to register them
from ..models import cnn, heads
from ..utils.training_utils import (
    print_preflop_range_grid,
    print_training_stats,
    print_evaluation_results,
    print_checkpoint_info,
)


def train_kbest(cfg: Config) -> SelfPlayTrainer:
    """
    Train AlphaHoldem agent using K-Best self-play.

    Args:
        cfg: Hydra configuration object containing all training parameters
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
        TrainingConfig,
        ModelConfig,
        EnvConfig,
        StateEncoderConfig,
    )

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create nested config objects
    train_config = TrainingConfig(**cfg_dict.get("train", {}))
    model_config = ModelConfig(**cfg_dict.get("model", {}))
    env_config = EnvConfig(**cfg_dict.get("env", {}))
    state_encoder_config = StateEncoderConfig(**cfg_dict.get("state_encoder", {}))

    # Create Config dataclass
    config = Config(
        train=train_config,
        model=model_config,
        env=env_config,
        state_encoder=state_encoder_config,
        **{
            k: v
            for k, v in cfg_dict.items()
            if k not in ["train", "model", "env", "state_encoder"]
        },
    )

    # Set random seeds for reproducibility
    torch.manual_seed(config.seed)

    # Train the agent - pass config dataclass
    train_kbest(config)

    print("Training completed!")


if __name__ == "__main__":
    main()
