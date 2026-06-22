#!/usr/bin/env python3
"""Write bounded postflop ReBeL solved-example datasets for HP sweeps."""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from p2.config.rebel_load import load_rebel_config
from p2.core.structured_config import Config
from p2.models.mlp.better_ffn import BetterSplitFFN
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.runtime.training_run import device_from_config
from p2.search.cfr_evaluator import CFREvaluator
from p2.search.postflop_spot_sampler import postflop_spot_sampler_metadata
from p2.search.rebel_solved_dataset import RebelSolvedDatasetWriter
from p2.utils.profiling import install_triton_compile_logger_from_env


_ROOT_SOURCE_TO_STREET = {
    "random_flop": "flop",
    "random_flop_prefix": "flop",
    "random_turn": "turn",
    "random_turn_prefix": "turn",
    "random_river": "river",
    "random_river_prefix": "river",
}
_ROOT_SOURCE_CODES = {
    "random_flop": 0,
    "random_turn": 1,
    "random_river": 2,
    "random_flop_prefix": 3,
    "random_turn_prefix": 4,
    "random_river_prefix": 5,
    "multiway_preflop_handoff_natural_hu": 10,
    "multiway_preflop_handoff_forced_fold": 11,
}
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class _ExploitabilityAccumulator:
    count: int = 0
    total: float = 0.0
    log_total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, mbbg: torch.Tensor) -> None:
        values = mbbg.detach().float()
        if values.numel() == 0:
            return
        self.count += int(values.numel())
        self.total += float(values.sum().item())
        self.log_total += float(values.clamp_min(1e-6).log().sum().item())
        self.minimum = min(self.minimum, float(values.min().item()))
        self.maximum = max(self.maximum, float(values.max().item()))

    def summary(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "geomean": None,
                "min": None,
                "max": None,
            }
        return {
            "count": self.count,
            "mean": self.total / self.count,
            "geomean": math.exp(self.log_total / self.count),
            "min": self.minimum,
            "max": self.maximum,
        }


@torch.no_grad()
def _final_average_policy_exploitability_mbbg(evaluator: CFREvaluator) -> torch.Tensor:
    """Measure final average-policy local exploitability without changing targets."""

    latest_values = evaluator.latest_values.clone()
    values_avg = evaluator.values_avg.clone()
    last_model_values = (
        evaluator.last_model_values.clone()
        if evaluator.last_model_values is not None
        else None
    )
    evaluator._exploitability_cache_key = None
    evaluator._exploitability_cache = None

    try:
        evaluator.update_average_values_final()
        evaluator._exploitability_cache_key = None
        stats = evaluator._compute_exploitability()
        mbbg = evaluator._local_exploitability_mbbg(stats.local_exploitability)
        return mbbg.detach().float().cpu()
    finally:
        evaluator.latest_values = latest_values
        evaluator.values_avg = values_avg
        evaluator.last_model_values = last_model_values
        evaluator._exploitability_cache_key = None
        evaluator._exploitability_cache = None


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_metadata(path: str | Path) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return dict(checkpoint["metadata"])


def _maybe_sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    path_obj = Path(path)
    if not path_obj.exists():
        return None
    return _sha256_file(path_obj)


def _target_model_metadata(cfg: Config) -> dict:
    checkpoint = cfg.search.closing_leaf_checkpoint
    if checkpoint is None:
        return {"role": "none", "checkpoint": None, "sha256": None}
    checkpoint_metadata = _checkpoint_metadata(checkpoint)
    metadata = {
        "role": "closing_leaf",
        "checkpoint": checkpoint,
        "sha256": _sha256_file(checkpoint),
    }
    source_checkpoint = checkpoint_metadata.get("curriculum_source_checkpoint")
    if source_checkpoint:
        metadata["distilled_from_checkpoint"] = source_checkpoint
        metadata["distilled_from_sha256"] = _maybe_sha256_file(source_checkpoint)
    source_net = checkpoint_metadata.get("curriculum_from_net")
    if source_net:
        metadata["distilled_from_net"] = source_net
    target_net = checkpoint_metadata.get("curriculum_net")
    if target_net:
        metadata["net"] = target_net
    return metadata


def _root_streets(root_source: str) -> list[str]:
    return [_ROOT_SOURCE_TO_STREET.get(root_source, root_source)]


def _root_source_code(root_source: str) -> int:
    if root_source not in _ROOT_SOURCE_CODES:
        raise ValueError(f"Unsupported root_source for row tagging: {root_source!r}")
    return _ROOT_SOURCE_CODES[root_source]


def _quality_metadata(cfg: Config) -> dict[str, object]:
    return {
        "cfr_iterations": int(cfg.search.iterations),
        "cfr_type": cfg.search.cfr_type.value,
        "cfr_plus": bool(cfg.search.cfr_plus),
        "sparse": bool(cfg.search.sparse),
        "sparse_fused": bool(cfg.search.sparse_fused),
        "holdout_value_loss": None,
        "target_model_kl": None,
    }


def _git_text(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            check=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _code_version_metadata() -> dict[str, object]:
    commit = _git_text(["rev-parse", "HEAD"])
    status = _git_text(["status", "--porcelain", "--untracked-files=no"])
    return {
        "code_version": commit,
        "code_dirty": None if status is None else bool(status),
    }


def _component_feature_encoder_metadata(
    trainer: RebelCFRTrainer, component: torch.nn.Module
) -> dict[str, str]:
    metadata: dict[str, str] = {"model": type(component).__name__}
    encoder = component.create_feature_encoder(
        trainer.env,
        device=trainer.device,
        dtype=trainer.float_dtype,
    )
    metadata["encoder"] = type(encoder).__name__
    return metadata


def _feature_encoder_metadata(
    trainer: RebelCFRTrainer,
) -> dict[str, dict[str, str]]:
    model = trainer.model
    if type(model) is BetterSplitFFN:
        policy_model = model.policy_model
        value_model = model.value_model
    else:
        policy_model = model
        value_model = model
    return {
        "policy": _component_feature_encoder_metadata(trainer, policy_model),
        "value": _component_feature_encoder_metadata(trainer, value_model),
    }


def _load_model_weights_for_pregeneration(
    trainer: RebelCFRTrainer, checkpoint_path: str
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = checkpoint["model"]
    save_dtype_str = checkpoint.get("save_dtype")
    if save_dtype_str is not None and save_dtype_str != str(trainer.float_dtype):
        model_state = {
            key: (
                value.to(trainer.float_dtype)
                if value.dtype.is_floating_point
                else value
            )
            for key, value in model_state.items()
        }

    if checkpoint.get("model_component") == "value_model":
        if type(trainer.model) is not BetterSplitFFN:
            raise TypeError("value-only checkpoints require a BetterSplitFFN model")
        trainer.model.value_model.load_state_dict(
            model_state, strict=trainer.cfg.strict_model_loading
        )
    else:
        trainer.model.load_state_dict(
            model_state, strict=trainer.cfg.strict_model_loading
        )
    trainer._sync_inference_model()
    trainer.model.eval()


def _trim_batch(batch: RebelBatch, target_remaining: int) -> RebelBatch:
    if len(batch) <= target_remaining:
        return batch
    return batch[:target_remaining]


def _tag_root_source(batch: RebelBatch, root_source_code: int) -> RebelBatch:
    statistics = dict(batch.statistics)
    statistics["root_source"] = torch.full(
        (len(batch),),
        int(root_source_code),
        dtype=torch.long,
        device=batch.legal_masks.device,
    )
    return RebelBatch(
        features=batch.features,
        legal_masks=batch.legal_masks,
        policy_targets=batch.policy_targets,
        value_targets=batch.value_targets,
        statistics=statistics,
    )


def _max_policy_samples_for_generation(
    *,
    value_examples: int,
    policy_examples: int,
    value_target_min: int,
    policy_target_min: int,
    generation_batch_size: int,
) -> int:
    policy_remaining = max(0, policy_target_min - policy_examples)
    if policy_remaining == 0:
        return 0
    value_remaining = max(0, value_target_min - value_examples)
    if value_target_min <= 0 or value_remaining == 0:
        return policy_remaining
    value_this_batch = min(generation_batch_size, value_remaining)
    target_ratio = policy_target_min / value_target_min
    batch_cap = max(1, int(round(value_this_batch * target_ratio)))
    return min(policy_remaining, batch_cap)


def pregenerate_postflop_rebel(cfg: Config) -> dict:
    """Generate a bounded solved dataset from live CFR roots."""

    if install_triton_compile_logger_from_env():
        print("Triton compile logging enabled via P2_TRITON_COMPILE_LOG=1")
    if cfg.data.mode != "live":
        raise ValueError("postflop pregeneration expects data.mode=live")

    pregenerate_cfg = cfg.rebel_pregenerate
    if pregenerate_cfg.root_source is not None:
        cfg.data.live_root_source = pregenerate_cfg.root_source
    root_source_code = _root_source_code(cfg.data.live_root_source)

    device = device_from_config(cfg)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(cfg.seed)

    trainer = RebelCFRTrainer(cfg=cfg, device=device, pregeneration_only=True)
    if trainer.data_generator is None:
        raise RuntimeError("postflop pregeneration requires a live data generator")
    if cfg.resume_from:
        print(f"Loading model weights for pregeneration: {cfg.resume_from}")
        _load_model_weights_for_pregeneration(trainer, cfg.resume_from)

    writer = RebelSolvedDatasetWriter(
        pregenerate_cfg.output_dir,
        storage_float_dtype=pregenerate_cfg.storage_dtype,
    )
    value_examples = 0
    policy_examples = 0
    generation_batches = 0
    exploitability_accumulator = (
        _ExploitabilityAccumulator()
        if pregenerate_cfg.print_final_average_policy_exploitability
        else None
    )

    while (
        value_examples < pregenerate_cfg.value_target_min
        or policy_examples < pregenerate_cfg.policy_target_min
    ):
        if (
            pregenerate_cfg.max_generation_batches is not None
            and generation_batches >= pregenerate_cfg.max_generation_batches
        ):
            break
        max_policy_samples = _max_policy_samples_for_generation(
            value_examples=value_examples,
            policy_examples=policy_examples,
            value_target_min=pregenerate_cfg.value_target_min,
            policy_target_min=pregenerate_cfg.policy_target_min,
            generation_batch_size=pregenerate_cfg.generation_batch_size,
        )
        value_batch, policy_batch = trainer.data_generator.generate_data(
            pregenerate_cfg.generation_batch_size,
            return_value_batch=value_examples < pregenerate_cfg.value_target_min,
            return_policy_batch=policy_examples < pregenerate_cfg.policy_target_min,
            max_return_policy_samples=max_policy_samples,
        )
        generation_batches += 1
        if exploitability_accumulator is not None:
            exploitability_accumulator.update(
                _final_average_policy_exploitability_mbbg(
                    trainer.data_generator.evaluator
                )
            )
        if value_batch is not None and value_examples < pregenerate_cfg.value_target_min:
            batch = _trim_batch(
                value_batch,
                pregenerate_cfg.value_target_min - value_examples,
            )
            batch = _tag_root_source(batch, root_source_code)
            writer.append("value", batch)
            value_examples += len(batch)
        if (
            policy_batch is not None
            and policy_examples < pregenerate_cfg.policy_target_min
        ):
            batch = _trim_batch(
                policy_batch,
                pregenerate_cfg.policy_target_min - policy_examples,
            )
            batch = _tag_root_source(batch, root_source_code)
            writer.append("policy", batch)
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

    quality = _quality_metadata(cfg)
    if exploitability_accumulator is not None:
        exploitability_summary = exploitability_accumulator.summary()
        quality["final_average_policy_exploitability_mbbg"] = exploitability_summary
        print(
            "Final average-policy exploitability "
            f"(mbb/g over {exploitability_summary['count']} generated roots): "
            f"mean={exploitability_summary['mean']:.6g} "
            f"geomean={exploitability_summary['geomean']:.6g} "
            f"min={exploitability_summary['min']:.6g} "
            f"max={exploitability_summary['max']:.6g}"
        )

    manifest = writer.finalize(
        metadata={
            "stage": pregenerate_cfg.stage,
            "root_source": cfg.data.live_root_source,
            "root_source_codes": {
                str(root_source_code): cfg.data.live_root_source,
            },
            "root_streets": _root_streets(cfg.data.live_root_source),
            "model_family": cfg.model.name.value,
            "feature_encoder": _feature_encoder_metadata(trainer),
            "action_schedule": {
                "bet_bins": list(cfg.env.bet_bins),
                "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
                "allin_by_depth": cfg.search.allin_by_depth,
            },
            "generator": {
                "seed": cfg.seed,
                "device": str(device),
                "generation_batches": generation_batches,
                **_code_version_metadata(),
            },
            "target_model": _target_model_metadata(cfg),
            "quality": quality,
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
    config = load_rebel_config(dict_config)
    pregenerate_postflop_rebel(config)


if __name__ == "__main__":
    main()
