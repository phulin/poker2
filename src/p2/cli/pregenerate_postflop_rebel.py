#!/usr/bin/env python3
"""Write bounded postflop ReBeL solved-example datasets for HP sweeps."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from p2.cli.train_rebel import _device_from_config
from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.search.postflop_spot_sampler import postflop_spot_sampler_metadata
from p2.search.rebel_solved_dataset import write_rebel_solved_dataset
from p2.utils.profiling import install_triton_compile_logger_from_env


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_model_metadata(cfg: Config) -> dict:
    checkpoint = cfg.search.closing_leaf_checkpoint
    if checkpoint is None:
        return {"role": "none", "checkpoint": None, "sha256": None}
    return {
        "role": "closing_leaf",
        "checkpoint": checkpoint,
        "sha256": _sha256_file(checkpoint),
    }


def _trim_batch(batch: RebelBatch, target_remaining: int) -> RebelBatch:
    if len(batch) <= target_remaining:
        return batch
    return batch[:target_remaining]


def pregenerate_postflop_rebel(cfg: Config) -> dict:
    """Generate a bounded solved dataset from live CFR roots."""

    if install_triton_compile_logger_from_env():
        print("Triton compile logging enabled via P2_TRITON_COMPILE_LOG=1")
    if cfg.data.mode != "live":
        raise ValueError("postflop pregeneration expects data.mode=live")

    pregenerate_cfg = cfg.rebel_pregenerate
    if pregenerate_cfg.root_source is not None:
        cfg.data.live_root_source = pregenerate_cfg.root_source

    device = _device_from_config(cfg)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    if trainer.data_generator is None:
        raise RuntimeError("postflop pregeneration requires a live data generator")

    value_batches: list[RebelBatch] = []
    policy_batches: list[RebelBatch] = []
    value_examples = 0
    policy_examples = 0
    generation_batches = 0

    while (
        value_examples < pregenerate_cfg.value_target_min
        or policy_examples < pregenerate_cfg.policy_target_min
    ):
        if (
            pregenerate_cfg.max_generation_batches is not None
            and generation_batches >= pregenerate_cfg.max_generation_batches
        ):
            break
        value_batch, policy_batch = trainer.data_generator.generate_data(
            pregenerate_cfg.generation_batch_size,
            return_value_batch=value_examples < pregenerate_cfg.value_target_min,
            return_policy_batch=policy_examples < pregenerate_cfg.policy_target_min,
            max_return_policy_samples=max(
                1, pregenerate_cfg.policy_target_min - policy_examples
            ),
        )
        generation_batches += 1
        if value_batch is not None and value_examples < pregenerate_cfg.value_target_min:
            batch = _trim_batch(
                value_batch,
                pregenerate_cfg.value_target_min - value_examples,
            )
            value_batches.append(batch.to(torch.device("cpu")))
            value_examples += len(batch)
        if (
            policy_batch is not None
            and policy_examples < pregenerate_cfg.policy_target_min
        ):
            batch = _trim_batch(
                policy_batch,
                pregenerate_cfg.policy_target_min - policy_examples,
            )
            policy_batches.append(batch.to(torch.device("cpu")))
            policy_examples += len(batch)
        print(
            "Generated "
            f"value={value_examples}/{pregenerate_cfg.value_target_min} "
            f"policy={policy_examples}/{pregenerate_cfg.policy_target_min}"
        )

    if value_examples < pregenerate_cfg.value_target_min:
        raise RuntimeError(
            "value target minimum was not reached before max_generation_batches"
        )
    if policy_examples < pregenerate_cfg.policy_target_min:
        raise RuntimeError(
            "policy target minimum was not reached before max_generation_batches"
        )

    manifest = write_rebel_solved_dataset(
        pregenerate_cfg.output_dir,
        value_batches=value_batches,
        policy_batches=policy_batches,
        metadata={
            "stage": pregenerate_cfg.stage,
            "root_streets": [cfg.data.live_root_source],
            "model_family": (
                cfg.model.name.value
                if hasattr(cfg.model.name, "value")
                else str(cfg.model.name)
            ),
            "action_schedule": {
                "bet_bins": list(cfg.env.bet_bins),
                "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
                "allin_by_depth": cfg.search.allin_by_depth,
            },
            "generator": {
                "seed": cfg.seed,
                "device": str(device),
                "generation_batches": generation_batches,
            },
            "target_model": _target_model_metadata(cfg),
            "model_config": asdict(cfg.model),
            "env_config": asdict(cfg.env),
            "search_config": asdict(cfg.search),
            "spot_sampler_config": {
                "live_root_source": cfg.data.live_root_source,
                **postflop_spot_sampler_metadata(),
            },
        },
    )
    print(
        f"wrote {manifest['value_examples']} value and "
        f"{manifest['policy_examples']} policy examples to "
        f"{pregenerate_cfg.output_dir}"
    )
    return manifest


@hydra.main(
    version_base=None,
    config_path="../../../conf",
    config_name="config_rebel_pregenerate_postflop",
)
def main(dict_config: DictConfig) -> None:
    config = Config.from_dict_config(dict_config)
    pregenerate_postflop_rebel(config)


if __name__ == "__main__":
    main()
