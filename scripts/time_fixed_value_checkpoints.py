#!/usr/bin/env python3
"""Benchmark fixed value-proposal checkpoints with GPU-resident pregen batches."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from p2.rl.cfr_trainer import RebelCFRTrainer
from run_value_arch_proposal import (
    DEFAULT_DATASET_MANIFEST,
    PROPOSALS,
    _benchmark_no_grad_value_inference,
    _build_config,
    _dataset_dir,
    _load_gpu_value_epoch,
    _load_manifest,
    _resolve_value_batch_size,
)


DEFAULT_FIXED_ROOT = Path("outputs/value_arch_proposals_fixed_100step_20260630")
DEFAULT_JSON_OUT = DEFAULT_FIXED_ROOT / "checkpoint_inference_timing.json"
DEFAULT_MD_OUT = DEFAULT_FIXED_ROOT / "checkpoint_inference_timing.md"


def _proposal_dirs(root: Path, requested: list[str]) -> list[Path]:
    if requested and requested != ["all"]:
        return [root / proposal for proposal in requested]
    dirs = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "checkpoints" / "rebel_final.pt").exists()
    ]
    return sorted(dirs, key=lambda path: (path.name != "baseline", path.name))


def _runner_args(
    *,
    proposal: str,
    dataset: Path,
    output_root: Path,
    steps: int,
    seed: int,
    shuffle: bool,
    compile_mode: str,
    timing_warmup_batches: int,
    timing_batches: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        proposals=[proposal],
        dataset=dataset,
        output_root=output_root,
        steps=steps,
        value_batch_size=None,
        validation_interval=50,
        seed=seed,
        shuffle=shuffle,
        compile_mode=compile_mode,
        timing_warmup_batches=timing_warmup_batches,
        timing_batches=timing_batches,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _effective_settings(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "metadata.json").get("effective_model_settings", {})


def _validation_loss(run_dir: Path) -> float | None:
    validation = _read_json(run_dir / "final_validation.json")
    value = validation.get("validation_value_loss")
    return float(value) if value is not None else None


def _format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _write_markdown(
    path: Path,
    results: list[dict[str, Any]],
    *,
    warmup_batches: int,
    timed_batches: int,
    compile_mode: str,
    value_head: str,
) -> None:
    lines = [
        "# Fixed Checkpoint Inference Timing",
        "",
        f"Timing uses {warmup_batches} 4096-example no-grad warmup/compile "
        f"batch(es), then {timed_batches} timed 4096-example value-forward "
        f"batches averaged with CUDA events. Compile mode: `{compile_mode}`. "
        f"Value head: `{value_head}`.",
        "",
        "| Proposal | Validation loss | Mean 4096 forward (s) | Runs (s) | Settings |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for result in results:
        timing = result["timing"]
        settings = result.get("effective_model_settings", {})
        setting_text = ", ".join(
            f"{key}={value}"
            for key, value in settings.items()
            if key
            in {
                "hidden_dim",
                "ffn_dim",
                "range_hidden_dim",
                "num_hidden_layers",
                "num_value_layers",
                "value_head_rank",
                "value_hand_basis_rank",
                "belief_low_rank_dim",
                "belief_low_rank_board_conditioned",
            }
        )
        runs = ", ".join(f"{value:.6f}" for value in timing["runtime_s"])
        lines.append(
            "| "
            f"`{result['proposal']}` | "
            f"{_format_float(result.get('validation_value_loss'), 10)} | "
            f"{timing['mean_s']:.6f} | "
            f"{runs} | "
            f"{setting_text} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--checkpoint-name", default="rebel_final.pt")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--timing-warmup-batches", type=int, default=3)
    parser.add_argument("--timing-batches", type=int, default=20)
    parser.add_argument("--value-head", default="post")
    parser.add_argument(
        "proposals",
        nargs="*",
        default=["all"],
        help="Proposal directory names to benchmark, or all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_root = args.fixed_root
    proposal_dirs = _proposal_dirs(fixed_root, args.proposals)
    if not proposal_dirs:
        raise ValueError(f"no checkpoint directories found under {fixed_root}")

    dataset_dir = _dataset_dir(args.dataset)
    manifest = _load_manifest(dataset_dir)
    bootstrap_args = _runner_args(
        proposal="baseline",
        dataset=args.dataset,
        output_root=fixed_root,
        steps=int(args.steps),
        seed=int(args.seed),
        shuffle=not args.no_shuffle,
        compile_mode=str(args.compile_mode),
        timing_warmup_batches=int(args.timing_warmup_batches),
        timing_batches=int(args.timing_batches),
    )
    value_batch_size = _resolve_value_batch_size(bootstrap_args, manifest)
    baseline_cfg, _, _, _ = _build_config(
        bootstrap_args,
        proposal="baseline",
        manifest=manifest,
        value_batch_size=value_batch_size,
    )
    device = torch.device(baseline_cfg.device)
    loaded_at = time.time()
    gpu_epoch = _load_gpu_value_epoch(
        dataset_dir=dataset_dir,
        manifest=manifest,
        device=device,
        batch_size=value_batch_size,
        steps=int(args.steps),
        shuffle_seed=int(args.seed),
        shuffle=not args.no_shuffle,
    )

    results: list[dict[str, Any]] = []
    for run_dir in proposal_dirs:
        proposal = run_dir.name
        if proposal not in PROPOSALS:
            raise ValueError(
                f"{proposal!r} is not a known run_value_arch_proposal preset"
            )
        checkpoint = run_dir / "checkpoints" / args.checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        cfg_args = _runner_args(
            proposal=proposal,
            dataset=args.dataset,
            output_root=fixed_root,
            steps=int(args.steps),
            seed=int(args.seed),
            shuffle=not args.no_shuffle,
            compile_mode=str(args.compile_mode),
            timing_warmup_batches=int(args.timing_warmup_batches),
            timing_batches=int(args.timing_batches),
        )
        cfg, _, _, _ = _build_config(
            cfg_args,
            proposal=proposal,
            manifest=manifest,
            value_batch_size=value_batch_size,
        )
        trainer = RebelCFRTrainer(cfg, torch.device(cfg.device))
        step = trainer.load_checkpoint(str(checkpoint), load_optimizer=False)
        timing = _benchmark_no_grad_value_inference(
            trainer=trainer,
            gpu_epoch=gpu_epoch,
            batch_size=4096,
            warmup_batches=int(args.timing_warmup_batches),
            timed_batches=int(args.timing_batches),
            value_head=str(args.value_head),
        )
        result = {
            "proposal": proposal,
            "checkpoint": str(checkpoint),
            "checkpoint_step": int(step),
            "validation_value_loss": _validation_loss(run_dir),
            "effective_model_settings": _effective_settings(run_dir),
            "timing": timing,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "fixed_root": str(fixed_root),
        "dataset": str(dataset_dir),
        "checkpoint_name": args.checkpoint_name,
        "shuffle": not args.no_shuffle,
        "shuffle_seed": int(args.seed),
        "compile_mode": str(args.compile_mode),
        "value_head": str(args.value_head),
        "timing_warmup_batches": int(args.timing_warmup_batches),
        "timing_batches": int(args.timing_batches),
        "gpu_epoch": {
            "examples": gpu_epoch.examples,
            "batch_size": gpu_epoch.batch_size,
            "tensor_bytes": gpu_epoch.tensor_bytes,
            "tensor_gib": gpu_epoch.tensor_bytes / (1024**3),
            "load_time_s": gpu_epoch.load_time_s,
        },
        "elapsed_s": time.time() - loaded_at,
        "results": results,
    }
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_markdown(
        args.md_out,
        results,
        warmup_batches=int(args.timing_warmup_batches),
        timed_batches=int(args.timing_batches),
        compile_mode=str(args.compile_mode),
        value_head=str(args.value_head),
    )
    print(json.dumps({"json_out": str(args.json_out), "md_out": str(args.md_out)}))


if __name__ == "__main__":
    main()
