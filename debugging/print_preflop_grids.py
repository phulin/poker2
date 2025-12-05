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

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from alphaholdem.core.structured_config import Config, ModelType, NonlinearityType
from alphaholdem.rl.cfr_trainer import RebelCFRTrainer
from alphaholdem.utils.training_utils import print_preflop_range_grid


@dataclass
class TopLevel:
    checkpoint: Optional[str] = None


cs = ConfigStore.instance()
cs.store(group="", name="toplevel_schema", node=TopLevel)


def _apply_checkpoint_model_config(cfg: Config, checkpoint: dict) -> Config | None:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        return None

    # Replace cfr_action_epsilon with sample_epsilon if present in checkpoint config
    # Handle both train and search sections (old checkpoints may have it in either)
    if isinstance(checkpoint_config, dict):
        # Handle search section
        if "search" in checkpoint_config and isinstance(
            checkpoint_config["search"], dict
        ):
            if "cfr_action_epsilon" in checkpoint_config["search"]:
                if "sample_epsilon" not in checkpoint_config["search"]:
                    checkpoint_config["search"]["sample_epsilon"] = checkpoint_config[
                        "search"
                    ].pop("cfr_action_epsilon")
                else:
                    checkpoint_config["search"].pop("cfr_action_epsilon")

        # Handle train section (in case it was mistakenly stored there)
        if "train" in checkpoint_config and isinstance(
            checkpoint_config["train"], dict
        ):
            if "cfr_action_epsilon" in checkpoint_config["train"]:
                # Move it to search section if not already there
                if "search" not in checkpoint_config:
                    checkpoint_config["search"] = {}
                if "sample_epsilon" not in checkpoint_config["search"]:
                    checkpoint_config["search"]["sample_epsilon"] = checkpoint_config[
                        "train"
                    ].pop("cfr_action_epsilon")
                else:
                    checkpoint_config["train"].pop("cfr_action_epsilon")

    checkpoint_cfg = Config.from_dict(checkpoint_config)
    cfg.model = checkpoint_cfg.model
    cfg.env.bet_bins = checkpoint_cfg.env.bet_bins
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    return checkpoint_cfg


def _infer_better_dimensions(
    cfg: Config, model_state: dict[str, torch.Tensor], model_type: ModelType
) -> None:
    cfg.model.hidden_dim = model_state["street_embedding.weight"].shape[1]
    if "belief_encoder.linear_in.weight" in model_state:
        belief_hidden = model_state["belief_encoder.linear_in.weight"].shape[0]
        cfg.model.range_hidden_dim = belief_hidden // 2
    # Fallback when using SwiGLU encoders: keep existing range_hidden_dim.

    if "trunk.0.inner.linear_in.weight" in model_state:
        cfg.model.ffn_dim = model_state["trunk.0.inner.linear_in.weight"].shape[0]

    trunk_layers = {
        int(k.split(".")[1])
        for k in model_state
        if k.startswith("trunk.") and ".inner." in k and k.split(".")[1].isdigit()
    }
    if trunk_layers:
        cfg.model.num_hidden_layers = max(trunk_layers) + 1

    policy_layers = {
        int(k.split(".")[1])
        for k in model_state
        if k.startswith("policy_head.") and k.split(".")[1].isdigit()
    }
    if policy_layers:
        cfg.model.num_policy_layers = max(policy_layers) + 1

    value_layers = {
        int(k.split(".")[1])
        for k in model_state
        if k.startswith("hand_value_head.") and k.split(".")[1].isdigit()
    }
    if value_layers:
        cfg.model.num_value_layers = max(value_layers) + 1

    if model_type == ModelType.better_ffn:
        trunk_layers = {
            int(k.split(".")[1])
            for k in model_state
            if k.startswith("trunk.") and ".inner." in k and k.split(".")[1].isdigit()
        }
        if trunk_layers:
            cfg.model.num_hidden_layers = max(trunk_layers) + 1

    if any("swiglu" in k for k in model_state):
        cfg.model.nonlinearity = NonlinearityType.swiglu


def load_trainer_from_checkpoint(
    checkpoint_path: str, cfg: Config, device: torch.device
) -> tuple[RebelCFRTrainer, int]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Inspect checkpoint to determine model type and architecture
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model"]

    checkpoint_cfg = _apply_checkpoint_model_config(cfg, checkpoint)
    if checkpoint_cfg is not None:
        print(f"Using model config from checkpoint: {cfg.model.name}")

    # Re-detect model type and core params from the actual state_dict to avoid
    # mismatches when user overrides conflict with the checkpoint config.
    detected_type: ModelType
    if "y_init" in model_state or "z_init" in model_state:
        detected_type = ModelType.better_trm
    elif "street_embedding.weight" in model_state:
        detected_type = ModelType.better_ffn
    else:
        detected_type = ModelType.rebel_ffn
    if cfg.model.name != detected_type:
        print(f"Overriding model type to {detected_type} based on checkpoint state")
        cfg.model.name = detected_type

    has_swiglu = any("swiglu" in k for k in model_state)
    cfg.model.nonlinearity = (
        NonlinearityType.swiglu if has_swiglu else NonlinearityType.gelu
    )

    if detected_type in (ModelType.better_ffn, ModelType.better_trm):
        _infer_better_dimensions(cfg, model_state, detected_type)
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3

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

    # Replace cfr_action_epsilon with sample_epsilon if present
    if "search" in container and "cfr_action_epsilon" in container.get("search", {}):
        if "sample_epsilon" not in container["search"]:
            container["search"]["sample_epsilon"] = container["search"].pop(
                "cfr_action_epsilon"
            )
        else:
            container["search"].pop("cfr_action_epsilon")

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
