#!/usr/bin/env python3
"""Simple CLI to run AlphaHoldem self-play training."""

import argparse
import time
from alphaholdem.rl.self_play import SelfPlayTrainer


def main():
    parser = argparse.ArgumentParser(description="AlphaHoldem self-play training")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--trajectories-per-step", type=int, default=4, help="Trajectories per training step")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for PPO updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-bet-bins", type=int, default=9, help="Number of betting bins")
    args = parser.parse_args()
    
    print(f"Starting AlphaHoldem training for {args.steps} steps...")
    print(f"Config: {args.trajectories_per_step} trajectories/step, batch_size={args.batch_size}, lr={args.lr}")
    
    # Initialize trainer
    trainer = SelfPlayTrainer(
        num_bet_bins=args.num_bet_bins,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    
    # Training loop
    start_time = time.time()
    for step in range(args.steps):
        stats = trainer.train_step(num_trajectories=args.trajectories_per_step)
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{args.steps} | "
                  f"Episodes: {stats['episode_count']} | "
                  f"Avg Reward: {stats['avg_reward']:.2f} | "
                  f"Loss: {stats.get('avg_loss', 0):.4f} | "
                  f"Time: {elapsed:.1f}s")
    
    print(f"Training completed! Total episodes: {trainer.episode_count}")


if __name__ == "__main__":
    main()
