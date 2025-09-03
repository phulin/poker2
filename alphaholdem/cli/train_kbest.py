#!/usr/bin/env python3
"""
Training script for AlphaHoldem with K-Best self-play.

This script demonstrates the K-Best self-play mechanism as described in the AlphaHoldem paper,
where the agent maintains a pool of K best historical versions and trains against them.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
from alphaholdem.rl.self_play import SelfPlayTrainer


def train_kbest(
    num_steps: int = 1000,
    trajectories_per_step: int = 4,
    k_best_pool_size: int = 5,
    min_elo_diff: float = 50.0,
    checkpoint_interval: int = 100,
    eval_interval: int = 50,
    checkpoint_dir: str = "checkpoints",
    resume_from: str = None,
):
    """
    Train AlphaHoldem agent using K-Best self-play.
    
    Args:
        num_steps: Number of training steps
        trajectories_per_step: Number of trajectories to collect per step
        k_best_pool_size: Size of K-Best opponent pool
        min_elo_diff: Minimum ELO difference for pool updates
        checkpoint_interval: How often to save checkpoints
        eval_interval: How often to evaluate against pool
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize trainer with K-Best pool
    trainer = SelfPlayTrainer(
        num_bet_bins=9,
        learning_rate=1e-3,
        batch_size=256,
        num_epochs=4,
        gamma=0.999,
        gae_lambda=0.95,
        epsilon=0.2,
        delta1=3.0,
        value_coef=0.1,
        entropy_coef=0.01,
        grad_clip=1.0,
        k_best_pool_size=k_best_pool_size,
        min_elo_diff=min_elo_diff,
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        start_step = trainer.load_checkpoint(resume_from)
    
    # Record training start time
    training_start_time = time.time()
    
    print(f"Starting K-Best training from step {start_step}")
    print(f"K-Best pool size: {k_best_pool_size}")
    print(f"Min ELO difference: {min_elo_diff}")
    print(f"Trajectories per step: {trajectories_per_step}")
    print(f"Training start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
    print()
    
    # Training loop
    for step in range(start_step, num_steps):
        step_start_time = time.time()
        
        # Training step
        stats = trainer.train_step(num_trajectories=trajectories_per_step)
        
        # Calculate times
        step_elapsed_time = time.time() - step_start_time
        total_elapsed_time = time.time() - training_start_time
        
        # Format times
        step_time_str = f"{step_elapsed_time:.2f}s"
        total_time_str = f"{total_elapsed_time:.1f}s"
        
        # Logging
        print(f"Step {step + 1}/{num_steps} | "
              f"Avg Reward: {stats['avg_reward']:.2f} | "
              f"ELO: {stats['current_elo']:.1f} | "
              f"Step Time: {step_time_str} | "
              f"Total Time: {total_time_str}")
        
        # Evaluation against pool
        if (step + 1) % eval_interval == 0:
            print("Evaluating against opponent pool...")
            eval_results = trainer.evaluate_against_pool(num_games=50)
            print(f"Overall win rate: {eval_results['overall_win_rate']:.3f}")
            
            # Print individual opponent results
            for opponent_key, result in eval_results['opponent_results'].items():
                print(f"  {opponent_key}: {result['win_rate']:.3f} "
                      f"(ELO: {result['opponent_elo']:.1f})")
        
        # Checkpointing
        if (step + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step + 1}.pt")
            trainer.save_checkpoint(checkpoint_path, step + 1)
            
            # Also save the best model if it has the highest ELO
            best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
            if not os.path.exists(best_checkpoint_path) or stats['current_elo'] > trainer.opponent_pool.get_best_snapshot().elo:
                trainer.save_checkpoint(best_checkpoint_path, step + 1)
                print(f"New best model saved with ELO: {stats['current_elo']:.1f}")
    
    # Final evaluation
    final_total_time = time.time() - training_start_time
    print(f"\nFinal evaluation against opponent pool...")
    final_eval = trainer.evaluate_against_pool(num_games=100)
    print(f"Final overall win rate: {final_eval['overall_win_rate']:.3f}")
    print(f"Total training time: {final_total_time:.1f}s ({final_total_time/3600:.2f} hours)")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    trainer.save_checkpoint(final_checkpoint_path, num_steps)
    
    # Print preflop range grid
    print("\nPreflop range grid (button play):")
    print(trainer.get_preflop_range_grid(seat=0))
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train AlphaHoldem with K-Best self-play")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--trajectories-per-step", type=int, default=4, help="Trajectories per training step")
    parser.add_argument("--k-best-pool-size", type=int, default=5, help="Size of K-Best opponent pool")
    parser.add_argument("--min-elo-diff", type=float, default=50.0, help="Minimum ELO difference for pool updates")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Checkpoint save interval")
    parser.add_argument("--eval-interval", type=int, default=50, help="Evaluation interval")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume-from", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Train the agent
    trainer = train_kbest(
        num_steps=args.num_steps,
        trajectories_per_step=args.trajectories_per_step,
        k_best_pool_size=args.k_best_pool_size,
        min_elo_diff=args.min_elo_diff,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
