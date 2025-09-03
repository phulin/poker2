#!/usr/bin/env python3
"""Simple CLI to run AlphaHoldem self-play training."""

import argparse
import time
import os
from alphaholdem.rl.self_play import SelfPlayTrainer
from alphaholdem.utils.training_utils import print_preflop_range_grid, print_training_stats, print_checkpoint_info
from alphaholdem.core.config_loader import get_config


def main():
    parser = argparse.ArgumentParser(description="AlphaHoldem self-play training")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--trajectories-per-step", type=int, default=4, help="Trajectories per training step")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-bet-bins", type=int, default=9, help="Number of betting bins")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-interval", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, help="Resume training from checkpoint file")
    parser.add_argument("--config", type=str, help="Path to YAML config for components/hparams")
    args = parser.parse_args()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Optionally load YAML config (centralized defaults)
    cfg = get_config(args.config) if args.config else None
    if cfg:
        print(f"Using config: {args.config}")

    print(f"Starting AlphaHoldem training for {args.steps} steps...")
    print(f"Config: {args.trajectories_per_step} trajectories/step, batch_size={args.batch_size}, lr={args.lr}, nb={(cfg.nb if cfg else args.num_bet_bins)}")
    print(f"Checkpoints: {args.checkpoint_dir} (every {args.save_interval} steps)")
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        num_bet_bins=(cfg.nb if cfg else args.num_bet_bins),
        learning_rate=(cfg.learning_rate if cfg else args.lr),
        batch_size=(cfg.batch_size if cfg else args.batch_size),
        num_epochs=(cfg.num_epochs if cfg else 4),
        epsilon=(cfg.ppo_eps if cfg else 0.2),
        delta1=(cfg.ppo_delta1 if cfg else 3.0),
        gamma=(cfg.gamma if cfg else 0.999),
        gae_lambda=(cfg.gae_lambda if cfg else 0.95),
        value_coef=(cfg.value_coef if cfg else 0.5),
        entropy_coef=(cfg.entropy_coef if cfg else 0.01),
        grad_clip=(cfg.grad_clip if cfg else 1.0),
        config=args.config,
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_step = trainer.load_checkpoint(args.resume)
            print(f"Resuming from step {start_step}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")
    
    # Training loop
    start_time = time.time()
    for step in range(start_step, args.steps):
        stats = trainer.train_step(num_trajectories=(cfg.trajectories_per_step if cfg else args.trajectories_per_step))
        
        # Save checkpoint periodically
        if (step + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{step+1}.pt")
            trainer.save_checkpoint(checkpoint_path, step + 1)
            
            # Output preflop range grid to stdout
            print_preflop_range_grid(trainer, step + 1, seat=0)
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print_training_stats(stats, step, args.steps, f"{elapsed:.1f}s", f"{elapsed:.1f}s")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_checkpoint.pt")
    trainer.save_checkpoint(final_checkpoint_path, args.steps)
    
    print(f"Training completed! Total episodes: {trainer.episode_count}")
    print(f"Final checkpoint saved to: {final_checkpoint_path}")


if __name__ == "__main__":
    main()
