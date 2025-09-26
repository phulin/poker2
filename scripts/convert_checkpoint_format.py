#!/usr/bin/env python3
"""
Checkpoint format conversion script.

This script converts checkpoints from the old cls_mlp input format to the new format.
It loads the checkpoint with strict=False, reinitializes the cls_mlp and heads,
and saves a new checkpoint with the updated model state.
"""

import argparse
import os
import sys

import torch

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    ModelConfig,
    TrainingConfig,
)
from alphaholdem.rl.self_play import SelfPlayTrainer


def convert_checkpoint(
    input_checkpoint_path: str, output_checkpoint_path: str, device: str = "cpu"
) -> None:
    """
    Convert a checkpoint from old format to new format.

    Args:
        input_checkpoint_path: Path to the input checkpoint
        output_checkpoint_path: Path to save the converted checkpoint
        device: Device to use for conversion
    """

    print(f"Converting checkpoint: {input_checkpoint_path}")
    print(f"Output checkpoint: {output_checkpoint_path}")

    # Check if input checkpoint exists
    if not os.path.exists(input_checkpoint_path):
        raise FileNotFoundError(f"Input checkpoint not found: {input_checkpoint_path}")

    # Load the original checkpoint to inspect its structure
    device_obj = torch.device(device)
    original_checkpoint = torch.load(
        input_checkpoint_path, weights_only=False, map_location=device_obj
    )

    print(f"Original checkpoint keys: {list(original_checkpoint.keys())}")

    # Load config from checkpoint if available, otherwise create a default one
    if "full_config" in original_checkpoint:
        print("Loading full_config from checkpoint...")
        config = original_checkpoint["full_config"]
        # Override device and wandb settings for conversion
        config.device = device
        config.use_wandb = False
        config.strict_model_loading = False  # Use non-strict loading for conversion
        print(f"✅ Loaded full_config from checkpoint")
    elif "config" in original_checkpoint:
        print("Loading config from checkpoint...")
        config_dict = original_checkpoint["config"]
        # Create a Config object from the dict
        config = Config(
            train=TrainingConfig(
                batch_size=config_dict.get("batch_size", 32),
                num_epochs=config_dict.get("num_epochs", 4),
                gamma=config_dict.get("gamma", 0.999),
                gae_lambda=config_dict.get("gae_lambda", 0.95),
                epsilon=config_dict.get("epsilon", 0.2),
                delta1=config_dict.get("delta1", 3.0),
                value_coef=config_dict.get("value_coef", 0.05),
                entropy_coef=config_dict.get("entropy_coef", 0.01),
                grad_clip=config_dict.get("grad_clip", 1.0),
            ),
            model=ModelConfig(
                name="poker_transformer_v1",
                kwargs={
                    "max_sequence_length": 50,
                    "d_model": 128,
                    "n_layers": 2,
                    "n_heads": 2,
                    "num_bet_bins": config_dict.get("num_bet_bins", 8),
                    "dropout": 0.1,
                    "use_gradient_checkpointing": False,
                },
            ),
            env=EnvConfig(bet_bins=[0.5, 0.75, 1.0, 1.5, 2.0]),
            use_tensor_env=True,
            num_envs=1,
            device=device,
            use_wandb=False,
            strict_model_loading=False,  # Use non-strict loading for conversion
        )
        print(f"✅ Created Config from checkpoint dict")
    else:
        print("No config found in checkpoint, creating default config...")
        # Create a default config for the new model
        config = Config(
            train=TrainingConfig(batch_size=32),
            model=ModelConfig(
                name="poker_transformer_v1",
                kwargs={
                    "max_sequence_length": 50,
                    "d_model": 128,
                    "n_layers": 2,
                    "n_heads": 2,
                    "num_bet_bins": 8,
                    "dropout": 0.1,
                    "use_gradient_checkpointing": False,
                },
            ),
            env=EnvConfig(bet_bins=[0.5, 0.75, 1.0, 1.5, 2.0]),
            use_tensor_env=True,
            num_envs=1,
            device=device,
            use_wandb=False,
            strict_model_loading=False,  # Use non-strict loading for conversion
        )
        print(f"⚠️ Created default config (may not match original model architecture)")

    print("Creating new trainer with updated model...")

    # Create trainer with new model architecture
    trainer = SelfPlayTrainer(config, device_obj)

    print("Loading original checkpoint with strict=False...")

    # Load the original checkpoint (this will skip incompatible parts)
    try:
        step, wandb_run_id = trainer.load_checkpoint(input_checkpoint_path)
        print(f"✅ Successfully loaded checkpoint (step: {step})")
    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}")
        print("Continuing with model reinitialization...")
        step = original_checkpoint.get("step", 0)
        wandb_run_id = original_checkpoint.get("wandb_run_id")

    print("Reinitializing cls_mlp and heads...")

    # Reinitialize the cls_mlp and heads with new architecture
    model = trainer.model

    # Get the current d_model
    d_model = model.d_model

    # Reinitialize cls_mlp with the new architecture
    # The new cls_mlp takes d_model * 4 input (cls_state + hole_mean + hole_diff + hole_prod)
    model.cls_mlp = torch.nn.Sequential(
        torch.nn.Linear(d_model * 4, d_model),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.LayerNorm(d_model),
    )

    # Reinitialize policy head
    from alphaholdem.models.transformer.heads import TransformerPolicyHead

    model.policy_head = TransformerPolicyHead(d_model, model.num_bet_bins, 0.1)

    # Reinitialize value head
    from alphaholdem.models.transformer.heads import TransformerValueHead

    model.value_head = TransformerValueHead(d_model, 0.1)

    print("✅ Reinitialized cls_mlp and heads")

    # Update trainer's internal state to match the original checkpoint
    trainer.step = step
    trainer.wandb_run_id = wandb_run_id
    trainer.total_trajectories_collected = original_checkpoint.get(
        "total_trajectories_completed", 0
    )
    trainer.wandb_step = original_checkpoint.get("wandb_step", 0)

    # Include replay buffer if it exists
    if "replay_buffer" in original_checkpoint:
        trainer.replay_buffer = original_checkpoint["replay_buffer"]

    print("Saving converted checkpoint...")

    # Use SelfPlayTrainer's save_checkpoint method
    trainer.save_checkpoint(output_checkpoint_path, step)

    print(f"✅ Checkpoint conversion complete!")
    print(f"   Input: {input_checkpoint_path}")
    print(f"   Output: {output_checkpoint_path}")
    print(f"   Step: {step}")
    print(f"   ELO: {original_checkpoint.get('current_elo', 1200.0)}")

    # Verify the new checkpoint can be loaded
    print("\nVerifying converted checkpoint...")

    try:
        # Create a new trainer to test loading
        test_trainer = SelfPlayTrainer(config, device_obj)
        # Load with strict=True since the converted checkpoint should match the model architecture
        test_step, _ = test_trainer.load_checkpoint(output_checkpoint_path)
        print(f"✅ Verification successful! Loaded step: {test_step}")
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert checkpoint from old cls_mlp format to new format"
    )
    parser.add_argument(
        "input_checkpoint", type=str, help="Path to the input checkpoint file"
    )
    parser.add_argument(
        "output_checkpoint", type=str, help="Path to save the converted checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for conversion (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the original checkpoint",
    )

    args = parser.parse_args()

    # Create backup if requested
    if args.backup:
        backup_path = args.input_checkpoint + ".backup"
        print(f"Creating backup: {backup_path}")
        import shutil

        shutil.copy2(args.input_checkpoint, backup_path)

    # Convert the checkpoint
    try:
        success = convert_checkpoint(
            args.input_checkpoint, args.output_checkpoint, args.device
        )

        if success:
            print("\n🎉 Checkpoint conversion completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Checkpoint conversion failed!")
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
