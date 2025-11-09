#!/usr/bin/env python3
"""
Debug script to run data generation loop and show statistics about value targets by street.
"""

from __future__ import annotations

import os
import sys

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alphaholdem.core.structured_config import Config, ModelType
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer


def load_trainer_from_checkpoint(
    checkpoint_path: str, cfg: Config, device: torch.device
) -> RebelCFRTrainer:
    """Load trainer from checkpoint, inferring model architecture."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model"]

    # Detect BetterFFN
    is_better_ffn = "street_embedding.weight" in model_state

    # Infer bet bins from model
    policy_w = model_state.get("policy_head.weight")
    if policy_w is not None:
        out_dim = int(policy_w.shape[0])
        num_actions = max(3, out_dim // 1326)
        k = max(0, num_actions - 3)
        if k > 0:
            cfg.env.bet_bins = [0.5 * (i + 1) for i in range(k)]

    if is_better_ffn:
        cfg.model.name = ModelType.better_ffn
        if "post_norm.weight" in model_state:
            cfg.model.hidden_dim = int(model_state["post_norm.weight"].shape[0])
            cfg.model.range_hidden_dim = 128
            cfg.model.ffn_dim = int(cfg.model.hidden_dim * 2)
        trunk_layers: set[int] = set()
        for kname in model_state.keys():
            if kname.startswith("trunk."):
                parts = kname.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    trunk_layers.add(int(parts[1]))
        if trunk_layers:
            cfg.model.num_hidden_layers = max(trunk_layers) + 1
    else:
        cfg.model.name = ModelType.rebel_ffn

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    trainer.load_checkpoint(checkpoint_path)
    return trainer


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    checkpoint_path = "checkpoints-rebel/rebel_120_150.pt"
    num_samples = 50

    print(f"Loading checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load config from hydra
    cfg = Config.from_dict_config(dict_config)
    cfg.num_envs = 8
    cfg.search.iterations = 100
    cfg.search.dcfr_plus_delay = 30

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load trainer from checkpoint (infers model architecture)
    print("Loading trainer from checkpoint...")
    trainer = load_trainer_from_checkpoint(checkpoint_path, cfg, device)
    trainer.model.eval()

    # Clear buffers to start fresh
    trainer.value_buffer.clear()
    trainer.policy_buffer.clear()

    # Generate data
    print(f"Generating {num_samples} value samples...")
    trainer.data_generator.generate_data(num_samples)

    # Extract value targets and street information
    value_buffer = trainer.value_buffer
    num_samples_actual = len(value_buffer)
    print(f"Generated {num_samples_actual} samples in value buffer")

    # Get value targets: shape (num_samples, num_players, NUM_HANDS)
    value_targets = value_buffer.value_targets[:num_samples_actual]
    streets = value_buffer.features.street[:num_samples_actual]

    # Compute statistics by street
    street_names = ["preflop", "flop", "turn", "river"]
    print("\n" + "=" * 80)
    print("Value Target Statistics by Street")
    print("=" * 80)

    for street_idx, street_name in enumerate(street_names):
        mask = streets == street_idx
        num_street_samples = mask.sum().item()

        if num_street_samples == 0:
            print(f"\n{street_name.upper()} (street {street_idx}): No samples")
            continue

        street_values = value_targets[
            mask
        ]  # (num_street_samples, num_players, NUM_HANDS)

        # Compute std across all dimensions (samples, players, hands)
        std_value = street_values.std().item()

        # Compute mean absolute value
        mean_abs_value = street_values.abs().mean().item()

        # Also compute per-player statistics
        std_by_player = street_values.std(dim=(0, 2)).tolist()  # (num_players,)
        mean_abs_by_player = (
            street_values.abs().mean(dim=(0, 2)).tolist()
        )  # (num_players,)

        print(
            f"\n{street_name.upper()} (street {street_idx}): {num_street_samples} samples"
        )
        print(f"  Overall std: {std_value:.6f}")
        print(f"  Overall mean |value|: {mean_abs_value:.6f}")
        print(f"  Std by player: {[f'{s:.6f}' for s in std_by_player]}")
        print(f"  Mean |value| by player: {[f'{m:.6f}' for m in mean_abs_by_player]}")

        # Min/max for reference
        min_value = street_values.min().item()
        max_value = street_values.max().item()
        print(f"  Min value: {min_value:.6f}, Max value: {max_value:.6f}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("Overall Statistics (all streets)")
    print("=" * 80)
    overall_std = value_targets.std().item()
    overall_mean_abs = value_targets.abs().mean().item()
    print(f"Overall std: {overall_std:.6f}")
    print(f"Overall mean |value|: {overall_mean_abs:.6f}")
    print(f"Min value: {value_targets.min().item():.6f}")
    print(f"Max value: {value_targets.max().item():.6f}")


if __name__ == "__main__":
    main()
