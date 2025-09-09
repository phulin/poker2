#!/usr/bin/env python3
"""
Debug script to test gradient computation during evaluation.
Runs: training step -> evaluation -> training step to detect gradient spikes.
"""

import torch
import os
from alphaholdem.rl.self_play import SelfPlayTrainer


def main():
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    print("🔍 Anomaly detection enabled - will show detailed gradient info")

    # Check if checkpoint exists
    checkpoint_path = "checkpoints/latest_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please run training first to create a checkpoint")
        return

    print(f"📁 Loading checkpoint: {checkpoint_path}")

    # Create trainer
    trainer = SelfPlayTrainer(
        use_tensor_env=True,
        num_envs=4,  # Small for debugging
        batch_size=8,
        k_best_pool_size=2,
        min_elo_diff=20.0,
        device=torch.device("cpu"),  # Use CPU for easier debugging
        use_wandb=False,
    )

    # Load checkpoint
    step, wandb_run_id = trainer.load_checkpoint(checkpoint_path)
    print(f"✅ Loaded checkpoint from step {step}")

    # Add a snapshot to opponent pool for evaluation
    if not trainer.opponent_pool.snapshots:
        trainer.opponent_pool.add_snapshot(trainer.model, step)
        print("📊 Added snapshot to opponent pool for evaluation")

    print("\n" + "=" * 60)
    print("STEP 1: TRAINING STEP")
    print("=" * 60)

    # First training step
    print("🚀 Running first training step...")
    stats1 = trainer.train_step(1)
    print(f"✅ Training step 1 completed")
    print(f"   Loss: {stats1.get('avg_loss', 'N/A'):.6f}")
    print(f"   Value loss: {stats1.get('value_loss', 'N/A'):.6f}")
    print(f"   Policy loss: {stats1.get('policy_loss', 'N/A'):.6f}")

    print("\n" + "=" * 60)
    print("STEP 2: EVALUATION")
    print("=" * 60)

    # Evaluation
    print("🔍 Running evaluation...")
    eval_results = trainer.evaluate_against_pool(num_games=5)
    print(f"✅ Evaluation completed")
    print(f"   Overall win rate: {eval_results.get('overall_win_rate', 'N/A'):.3f}")
    print(f"   Total games: {eval_results.get('total_games', 'N/A')}")

    print("\n" + "=" * 60)
    print("STEP 3: TRAINING STEP (AFTER EVALUATION)")
    print("=" * 60)

    # Second training step (this is where the spike should occur)
    print("🚀 Running second training step (after evaluation)...")
    stats2 = trainer.train_step(2)
    print(f"✅ Training step 2 completed")
    print(f"   Loss: {stats2.get('avg_loss', 'N/A'):.6f}")
    print(f"   Value loss: {stats2.get('value_loss', 'N/A'):.6f}")
    print(f"   Policy loss: {stats2.get('policy_loss', 'N/A'):.6f}")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Compare losses
    loss1 = stats1.get("avg_loss", 0)
    loss2 = stats2.get("avg_loss", 0)
    value_loss1 = stats1.get("value_loss", 0)
    value_loss2 = stats2.get("value_loss", 0)

    print(f"Training step 1 loss: {loss1:.6f}")
    print(f"Training step 2 loss: {loss2:.6f}")
    print(f"Loss change: {loss2 - loss1:.6f}")
    print()
    print(f"Training step 1 value loss: {value_loss1:.6f}")
    print(f"Training step 2 value loss: {value_loss2:.6f}")
    print(f"Value loss change: {value_loss2 - value_loss1:.6f}")

    if abs(loss2 - loss1) > 0.01:  # Significant change
        print("⚠️  SIGNIFICANT LOSS SPIKE DETECTED!")
        print("   This confirms the evaluation is affecting training")
    else:
        print("✅ No significant loss spike detected")

    print(
        "\n🔍 Check the anomaly detection output above for gradient computation details"
    )


if __name__ == "__main__":
    main()
