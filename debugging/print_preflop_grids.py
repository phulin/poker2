#!/usr/bin/env python3
"""
Quick script: print preflop 13x13 grids (169 buckets) for a given BetterFFN/RebelFFN checkpoint,
using the standard training utility that runs proper ReBeL CFR search.

Usage:
  source venv/bin/activate
  python debugging/print_preflop_grids.py \
    checkpoint=/path/to/checkpoint.pt \
    device=cpu \
    search.iterations=150 search.depth=2 env.bet_bins=[0.5,1.0]

Any field in conf/config_rebel_cfr.yaml can be overridden via Hydra CLI (e.g., search.iterations=...).
"""

import os
from dataclasses import dataclass
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
import hydra
from omegaconf import DictConfig, OmegaConf

from alphaholdem.core.structured_config import Config
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.utils.training_utils import print_preflop_range_grid


@dataclass
class TopLevel:
    checkpoint: Optional[str] = None


cs = ConfigStore.instance()
cs.store(group="", name="toplevel_schema", node=TopLevel)


def load_trainer_from_checkpoint(
    checkpoint_path: str, cfg: Config, device: torch.device
) -> tuple[RebelCFRTrainer, int]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Inspect checkpoint for model type/shape hints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model"]

    # Detect BetterFFN via presence of embedding keys; otherwise assume RebelFFN
    is_better_ffn = "street_embedding.weight" in model_state

    # Infer action space (num_actions = out_dim / 1326) to align bet bins
    policy_w = model_state.get("policy_head.weight")
    if policy_w is not None:
        out_dim = int(policy_w.shape[0])
        num_actions = max(3, out_dim // 1326)
        k = max(0, num_actions - 3)
        if k > 0:
            cfg.env.bet_bins = [0.5 * (i + 1) for i in range(k)]

    # If BetterFFN, align hidden sizes and layers to avoid strict load errors
    if is_better_ffn:
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

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    loaded_step = trainer.load_checkpoint(checkpoint_path)
    return trainer, int(loaded_step)


@hydra.main(version_base=None, config_path="../conf", config_name="config_rebel_cfr")
def main(dict_config: DictConfig) -> None:
    checkpoint = dict_config.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            "Please provide checkpoint=/path/to/checkpoint.pt as a Hydra override."
        )
    checkpoint = to_absolute_path(checkpoint)
    container = OmegaConf.to_container(dict_config, resolve=True)
    container.pop("checkpoint")
    cfg = Config.from_dict(container)

    # Device selection
    device = torch.device(cfg.device)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    trainer, loaded_step = load_trainer_from_checkpoint(checkpoint, cfg, device)
    print_preflop_range_grid(
        trainer=trainer,
        step=loaded_step if loaded_step is not None else cfg.num_steps,
        title=f"Preflop Range Grid (Checkpoint: {os.path.basename(checkpoint)})",
        rebel=True,
    )


if __name__ == "__main__":
    main()
