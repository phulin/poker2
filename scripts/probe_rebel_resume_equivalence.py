#!/usr/bin/env python3
"""Probe whether a ReBeL checkpoint resumes to an equivalent next step."""

from __future__ import annotations

import argparse
from dataclasses import MISSING, fields, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.rebel_batch import RebelBatch
from p2.runtime.training_run import device_from_config, setup_torch_runtime


def _filter_dataclass_fields(
    dataclass_type: type, container: dict[str, Any]
) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for field_info in fields(dataclass_type):
        if field_info.name not in container:
            continue
        value = container[field_info.name]
        default_factory = getattr(field_info, "default_factory", MISSING)
        if isinstance(value, dict) and default_factory is not MISSING:
            try:
                default_value = default_factory()
            except TypeError:
                default_value = None
            if default_value is not None and is_dataclass(default_value):
                value = _filter_dataclass_fields(type(default_value), value)
        clean[field_info.name] = value
    return clean


def _load_config(checkpoint_path: Path) -> Config:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    container = checkpoint.get("config")
    if not isinstance(container, dict):
        raise ValueError(f"checkpoint has no embedded config: {checkpoint_path}")
    cfg = Config.from_dict(_filter_dataclass_fields(Config, container))
    cfg.use_wandb = False
    return cfg


def _update_hash(digest: Any, label: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().contiguous().cpu()
    digest.update(label.encode())
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())


def _tensor_mapping_hash(mapping: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(mapping):
        value = mapping[key]
        if isinstance(value, torch.Tensor):
            _update_hash(digest, key, value)
    return digest.hexdigest()


def _batch_hash(batch: RebelBatch | None) -> str | None:
    if batch is None:
        return None
    digest = hashlib.sha256()
    tensors = {
        "context": batch.features.context,
        "street": batch.features.street,
        "to_act": batch.features.to_act,
        "board": batch.features.board,
        "beliefs": batch.features.beliefs,
        "legal_masks": batch.legal_masks,
        "policy_targets": batch.policy_targets,
        "value_targets": batch.value_targets,
        **{f"statistics/{key}": value for key, value in batch.statistics.items()},
    }
    for key in sorted(tensors):
        value = tensors[key]
        if isinstance(value, torch.Tensor):
            _update_hash(digest, key, value)
    return digest.hexdigest()


def _batch_summary(batch: RebelBatch | None) -> dict[str, Any] | None:
    if batch is None:
        return None
    street = batch.features.street.long()
    result: dict[str, Any] = {
        "count": len(batch),
        "hash": _batch_hash(batch),
        "street_counts": torch.bincount(street, minlength=4).cpu().tolist(),
    }
    if batch.value_targets is not None:
        result["target_abs_mean"] = float(batch.value_targets.float().abs().mean())
        result["target_rms"] = float(batch.value_targets.float().square().mean().sqrt())
    for key in ("pot", "scale"):
        value = batch.statistics.get(key)
        if value is not None:
            value = value.float()
            result[f"{key}_mean"] = float(value.mean())
            result[f"{key}_rms"] = float(value.square().mean().sqrt())
    pot = batch.statistics.get("pot")
    scale = batch.statistics.get("scale")
    if pot is not None and scale is not None:
        ratio = pot.float() / scale.float().clamp_min(1.0)
        result["pot_over_scale_mean"] = float(ratio.mean())
        result["pot_over_scale_rms"] = float(ratio.square().mean().sqrt())
    return result


def _replay_summary(buffer: Any) -> dict[str, Any]:
    valid = buffer._valid_physical_indices()
    sample_count = min(64, int(valid.numel()))
    logical = torch.linspace(
        0,
        max(0, int(valid.numel()) - 1),
        steps=sample_count,
        device=buffer.device,
    ).long()
    indices = valid[logical]
    digest = hashlib.sha256()
    tensors = {
        "context": buffer.features.context[indices],
        "street": buffer.features.street[indices],
        "to_act": buffer.features.to_act[indices],
        "board": buffer.features.board[indices],
        "beliefs": buffer.features.beliefs[indices],
        "legal_masks": buffer.legal_masks[indices],
        "sample_count": buffer.sample_count[indices],
        "policy_targets": (
            None if buffer.policy_targets is None else buffer.policy_targets[indices]
        ),
        "value_targets": (
            None if buffer.value_targets is None else buffer.value_targets[indices]
        ),
    }
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            _update_hash(digest, key, value)
    return {
        "size": len(buffer),
        "position": buffer.position,
        "start": buffer.start,
        "sample_hash": digest.hexdigest(),
    }


def _snapshot(trainer: RebelCFRTrainer, checkpoint_step: int) -> dict[str, Any]:
    generator_env = trainer.data_generator.env_proto.rng
    pbs_state = trainer.data_generator.state_dict()["current_pbs"]
    return {
        "checkpoint_step": checkpoint_step,
        "model_hash": _tensor_mapping_hash(trainer.model.state_dict()),
        "trainer_rng_hash": hashlib.sha256(
            trainer.rng.get_state().cpu().numpy().tobytes()
        ).hexdigest(),
        "buffer_rng_hash": hashlib.sha256(
            trainer.buffer_rng.get_state().cpu().numpy().tobytes()
        ).hexdigest(),
        "env_rng_hash": hashlib.sha256(
            generator_env.get_state().cpu().numpy().tobytes()
        ).hexdigest(),
        "global_cpu_rng_hash": hashlib.sha256(
            torch.get_rng_state().cpu().numpy().tobytes()
        ).hexdigest(),
        "global_cuda_rng_hashes": [
            hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()
            for state in torch.cuda.get_rng_state_all()
        ],
        "pbs_beliefs_hash": (
            None
            if pbs_state is None
            else _tensor_mapping_hash({"beliefs": pbs_state["beliefs"]})
        ),
        "pbs_env_hash": (
            None if pbs_state is None else _tensor_mapping_hash(pbs_state["env"])
        ),
        "last_extra": trainer.data_generator.last_extra,
        "value_replay": _replay_summary(trainer.value_buffer),
        "policy_replay": _replay_summary(trainer.policy_buffer),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("snapshot", "construct", "stages", "generate", "train-step"),
        default="snapshot",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-fill-uninitialized", action="store_true")
    # Footprint overrides so the probe can co-reside with a live training run.
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override cfg.num_envs (shrinks the CFR generation batch). "
        "When set, the restored current_pbs is dropped so roots are re-sampled "
        "at the smaller size.",
    )
    parser.add_argument(
        "--replay-buffer-device",
        default=None,
        help="Override cfg.train.replay_buffer_device (e.g. cpu) to keep the "
        "restored replay buffers off the GPU.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.checkpoint)
    if args.num_envs is not None:
        cfg.num_envs = int(args.num_envs)
    if args.replay_buffer_device is not None:
        cfg.train.replay_buffer_device = args.replay_buffer_device
    device = device_from_config(cfg)
    setup_torch_runtime(cfg, device)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if args.no_fill_uninitialized:
            torch.utils.deterministic.fill_uninitialized_memory = False
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    checkpoint_step = trainer.load_checkpoint(str(args.checkpoint))
    if args.num_envs is not None and trainer.data_generator is not None:
        # The restored current_pbs is sized to the original num_envs; drop it so
        # generate mode re-samples fresh roots at the reduced batch size.
        trainer.data_generator.current_pbs = None
    result: dict[str, Any] = {
        "mode": args.mode,
        "before": _snapshot(trainer, checkpoint_step),
    }

    next_step = checkpoint_step + 1
    if args.mode in ("construct", "stages"):
        trainer._apply_schedules(next_step)
        pbs = trainer.data_generator.current_pbs
        if pbs is None:
            # current_pbs was dropped by --num-envs; sample fresh roots so the
            # child writer still runs at the reduced batch size.
            pbs = trainer.data_generator._sample_roots(
                trainer.data_generator.target_batch_size
            )
            trainer.data_generator.current_pbs = pbs
        root_count = int(pbs.env.N)
        trainer.cfr_evaluator.initialize_subgame(
            pbs.env,
            torch.arange(root_count, device=trainer.device),
            pbs.beliefs[:root_count],
        )
        evaluator = trainer.cfr_evaluator
        tensor_names = (
            "parent_index",
            "action_from_parent",
            "valid_mask",
            "leaf_mask",
            "new_street_mask",
            "legal_mask",
            "child_mask",
            "prev_actor",
            "child_count",
            "child_offsets",
            "beliefs",
            "self_reach",
            "latest_values",
            "uniform_policy",
            "cumulative_regrets",
        )
        result["constructed"] = {
            name: _tensor_mapping_hash({name: getattr(evaluator, name)})
            for name in tensor_names
        }
        env_tensors = {
            name: value
            for name, value in vars(evaluator.env).items()
            if isinstance(value, torch.Tensor)
        }
        result["constructed"]["env"] = _tensor_mapping_hash(env_tensors)
        result["constructed"]["env_tensors"] = {
            name: _tensor_mapping_hash({name: value})
            for name, value in env_tensors.items()
        }
        # Per-field finite summary. Under torch deterministic uninitialized-memory
        # fill (NaN for float, sentinel for int), an unwritten field lights up.
        root_n = int(evaluator.root_nodes)

        def _field_finite_summary(value: torch.Tensor) -> dict[str, Any]:
            info: dict[str, Any] = {
                "dtype": str(value.dtype),
                "shape": tuple(value.shape),
            }
            if value.is_floating_point():
                nonfinite = (~torch.isfinite(value)).sum().item()
                info["nonfinite_total"] = int(nonfinite)
                # Split root rows [:root_n] vs child rows [root_n:].
                if value.shape[0] > root_n:
                    info["nonfinite_children"] = int(
                        (~torch.isfinite(value[root_n:])).sum().item()
                    )
                    info["nonfinite_roots"] = int(
                        (~torch.isfinite(value[:root_n])).sum().item()
                    )
            else:
                info["min"] = int(value.min().item()) if value.numel() else None
                info["max"] = int(value.max().item()) if value.numel() else None
                if value.shape[0] > root_n and value.numel():
                    child = value[root_n:]
                    info["child_min"] = int(child.min().item())
                    info["child_max"] = int(child.max().item())
            return info

        result["constructed"]["env_field_finite"] = {
            name: _field_finite_summary(value) for name, value in env_tensors.items()
        }
        result["constructed"]["total_nodes"] = evaluator.total_nodes
        result["constructed"]["root_nodes"] = evaluator.root_nodes
        if args.mode == "stages":

            def stage_hash() -> dict[str, Any]:
                summaries = {}
                for name in (
                    "policy_probs",
                    "policy_probs_avg",
                    "cumulative_regrets",
                    "beliefs",
                    "beliefs_avg",
                    "self_reach",
                    "self_reach_avg",
                    "latest_values",
                    "values_avg",
                ):
                    value = getattr(evaluator, name)
                    finite = value[torch.isfinite(value)].float()
                    summaries[name] = {
                        "hash": _tensor_mapping_hash({name: value}),
                        "mean": float(finite.mean()) if finite.numel() else None,
                        "rms": (
                            float(finite.square().mean().sqrt())
                            if finite.numel()
                            else None
                        ),
                        "nonfinite": int(value.numel() - finite.numel()),
                    }
                return summaries

            evaluator.initialize_policy_and_beliefs()
            result["policy_initialized"] = stage_hash()
            if evaluator.warm_start_iterations > 0:
                evaluator.warm_start()
            result["warm_started"] = stage_hash()
            evaluator.set_leaf_values(0)
            result["leaf_values_set"] = stage_hash()
            evaluator.compute_expected_values()
            result["expected_values_computed"] = stage_hash()
    elif args.mode == "generate":
        trainer._apply_schedules(next_step)
        value_batch, policy_batch = trainer.data_source.prepare_step(next_step)
        result["fresh_value"] = _batch_summary(value_batch)
        result["fresh_policy"] = _batch_summary(policy_batch)
        result["evaluator_stats"] = trainer.cfr_evaluator.stats
        result["after"] = _snapshot(trainer, checkpoint_step)
    elif args.mode == "train-step":
        result["metrics"] = trainer.train_step(next_step)
        result["after"] = _snapshot(trainer, checkpoint_step)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=float))
    print(json.dumps({"output": str(args.output), "mode": args.mode}))


if __name__ == "__main__":
    main()
