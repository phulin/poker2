#!/usr/bin/env python3
"""Generate S_turn targets for identical saved roots at multiple CFR budgets."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from p2.config.rebel_load import load_rebel_config_file
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.rebel_solved_dataset import RebelSolvedDatasetWriter


DEFAULT_CLOSING_CHECKPOINT = Path(
    "checkpoints-eturn-distill-lr-current-teb-300k-lr0p01-wandb-20260708/"
    "t001_lr0p01_300000st_b1024/promoted/E_turn.pt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "outputs/rebel_postflop/paired_sturn_4096_300_1000_5000it_eturn300k_20260711"
)


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(item) for item in value.split(",") if item.strip()]
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--closing-checkpoint", type=Path, default=DEFAULT_CLOSING_CHECKPOINT)
    parser.add_argument("--examples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--iterations", type=_parse_int_list, default=[300, 1000, 5000])
    parser.add_argument("--solve-seeds", type=_parse_int_list, default=[9001])
    parser.add_argument("--repeat-300-seeds", type=_parse_int_list, default=[9002])
    parser.add_argument("--root-seed", type=int, default=88000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _build_config(args: argparse.Namespace):
    cfg = load_rebel_config_file("conf/config_rebel_curriculum_turn.yaml")
    cfg.device = str(args.device)
    cfg.seed = int(args.root_seed)
    cfg.use_wandb = False
    cfg.trueskill.enabled = False
    cfg.data.mode = "live"
    cfg.data.live_root_source = "random_turn"
    cfg.data.belief_mode = "mixed"
    cfg.data.belief_profile = "actions_12_end"
    cfg.train.batch_size = 2048
    cfg.train.episodes_per_step = 5
    cfg.train.replay_buffer_batches = 1
    cfg.train.replay_buffer_device = "cuda"
    cfg.train.value_reuse_goal = 2
    cfg.train.save_replay_buffers = False
    cfg.search.iterations = max(args.iterations)
    cfg.search.iterations_final = max(args.iterations)
    cfg.search.allin_call_terminal_abstraction = True
    cfg.search.closing_leaf_checkpoint = str(args.closing_checkpoint)
    cfg.checkpoint_dir = str(args.output_root / "scratch_checkpoints")
    return cfg


def _writer_key(iterations: int, solve_seed: int) -> str:
    return f"{iterations}it_seed{solve_seed}"


def _tag_pairs(
    batch,
    *,
    root_start: int,
    root_batch: int,
    iterations: int,
    solve_seed: int,
) -> None:
    device = batch.features.context.device
    count = len(batch)
    batch.statistics["paired_root_id"] = torch.arange(
        root_start,
        root_start + count,
        device=device,
        dtype=torch.int64,
    )
    batch.statistics["paired_root_batch"] = torch.full(
        (count,), root_batch, device=device, dtype=torch.int32
    )
    batch.statistics["paired_cfr_iterations"] = torch.full(
        (count,), iterations, device=device, dtype=torch.int32
    )
    batch.statistics["paired_solve_seed"] = torch.full(
        (count,), solve_seed, device=device, dtype=torch.int64
    )


@torch.no_grad()
def generate(args: argparse.Namespace) -> dict:
    if args.examples <= 0 or args.batch_size <= 0:
        raise ValueError("examples and batch-size must be positive")
    if args.examples % args.batch_size != 0:
        raise ValueError("examples must be divisible by batch-size")
    if not args.closing_checkpoint.exists():
        raise FileNotFoundError(args.closing_checkpoint)
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    cfg = _build_config(args)
    device = torch.device(cfg.device)
    torch.manual_seed(int(cfg.seed))
    trainer = RebelCFRTrainer(cfg=cfg, device=device, pregeneration_only=True)
    generator = trainer.data_generator
    if generator is None:
        raise RuntimeError("paired generation requires a live data generator")
    evaluator = generator.evaluator

    solve_specs = [
        (iterations, solve_seed)
        for iterations in args.iterations
        for solve_seed in args.solve_seeds
    ]
    solve_specs.extend((300, solve_seed) for solve_seed in args.repeat_300_seeds)
    if len(set(solve_specs)) != len(solve_specs):
        raise ValueError("duplicate iteration/solve-seed output requested")

    roots_dir = args.output_root / "roots"
    roots_dir.mkdir(parents=True)
    writers = {
        spec: RebelSolvedDatasetWriter(
            args.output_root / _writer_key(*spec),
            storage_float_dtype="float32",
        )
        for spec in solve_specs
    }
    timings: dict[str, list[float]] = {
        _writer_key(*spec): [] for spec in solve_specs
    }
    batches = args.examples // args.batch_size

    for batch_idx in range(batches):
        trainer.rng.manual_seed(int(args.root_seed) + batch_idx)
        roots = generator._sample_roots(args.batch_size)
        generator.current_pbs = roots
        root_state = generator._pbs_state_dict(roots)
        if root_state is None:
            raise RuntimeError("sampled roots unexpectedly serialized to None")
        torch.save(root_state, roots_dir / f"root_batch_{batch_idx:06d}.pt")

        for iterations, solve_seed in solve_specs:
            evaluator.cfr_iterations = int(iterations)
            trainer.rng.manual_seed(int(solve_seed) + batch_idx)
            generator.load_state_dict(
                {
                    "last_extra": 0,
                    "target_batch_size": args.batch_size,
                    "current_pbs": root_state,
                }
            )
            started = time.perf_counter()
            value_batch, _ = generator.generate_data(
                args.batch_size,
                return_value_batch=True,
                return_policy_batch=False,
                max_return_policy_samples=0,
            )
            if value_batch is None or len(value_batch) != args.batch_size:
                raise RuntimeError(
                    f"expected {args.batch_size} value rows, got "
                    f"{None if value_batch is None else len(value_batch)}"
                )
            _tag_pairs(
                value_batch,
                root_start=batch_idx * args.batch_size,
                root_batch=batch_idx,
                iterations=iterations,
                solve_seed=solve_seed,
            )
            writers[(iterations, solve_seed)].append("value", value_batch)
            elapsed = time.perf_counter() - started
            timings[_writer_key(iterations, solve_seed)].append(elapsed)
            print(
                f"batch={batch_idx + 1}/{batches} iterations={iterations} "
                f"solve_seed={solve_seed} elapsed={elapsed:.1f}s",
                flush=True,
            )

    manifests = {}
    for (iterations, solve_seed), writer in writers.items():
        key = _writer_key(iterations, solve_seed)
        manifests[key] = writer.finalize(
            metadata={
                "stage": "turn",
                "root_source": "random_turn",
                "root_streets": ["turn"],
                "model_family": cfg.model.name.value,
                "action_schedule": {
                    "bet_bins": list(cfg.env.bet_bins),
                    "bet_bins_by_depth": cfg.search.bet_bins_by_depth,
                    "allin_by_depth": cfg.search.allin_by_depth,
                },
                "generator": {
                    "kind": "paired_saved_roots",
                    "root_seed": int(args.root_seed),
                    "solve_seed": int(solve_seed),
                    "root_batches": int(batches),
                    "root_snapshot_dir": str(roots_dir),
                },
                "target_model": {
                    "checkpoint": str(args.closing_checkpoint.resolve()),
                    "net": "E_turn",
                    "role": "closing_leaf",
                },
                "quality": {
                    "cfr_iterations": int(iterations),
                    "cfr_type": cfg.search.cfr_type.value,
                    "sparse": bool(cfg.search.sparse),
                    "sparse_fused": bool(cfg.search.sparse_fused),
                },
                "model_config": asdict(cfg.model),
                "env_config": asdict(cfg.env),
                "search_config": {
                    **asdict(cfg.search),
                    "iterations": int(iterations),
                    "iterations_final": int(iterations),
                },
                "paired_group": str(args.output_root),
            }
        )

    summary = {
        "output_root": str(args.output_root),
        "examples": int(args.examples),
        "batch_size": int(args.batch_size),
        "root_seed": int(args.root_seed),
        "solve_specs": [
            {"iterations": iterations, "solve_seed": solve_seed}
            for iterations, solve_seed in solve_specs
        ],
        "timings_s": timings,
        "datasets": {
            key: str(args.output_root / key) for key in manifests
        },
    }
    (args.output_root / "paired_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    print(json.dumps(generate(_parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
