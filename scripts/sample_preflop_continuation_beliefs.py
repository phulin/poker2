#!/usr/bin/env python3
"""Sample bucket-to-bucket preflop continuation beliefs from current specialists."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from p2.config.rebel_load import validate_rebel_config
from p2.core.structured_config import Config
from p2.env.card_utils import PREFLOP_HANDS
from p2.env.pbs_env import PBSEnv
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.checkpoint_io import CheckpointIO
from p2.search.cfr_evaluator import PublicBeliefState
from p2.stages.preflop_backward_induction import (
    PublicStateBucketReader,
    _checkpoint_model_config,
    _copy_public_states_to_env,
    _make_env_from_manifest,
    _random_beliefs,
)
from p2.stages.preflop_buckets import PreflopBucketExecutionConfig
from p2.stages.preflop_buckets import build_run_config


DEFAULT_RUN_DIR = Path(
    "outputs/preflop_backward_induction/"
    "gated_chain_6p_epreflop_12end10ep_d7_rest_d4_lr00105_wsd0p6_300cfr_20260627_v5"
)
DEFAULT_STATE_DATASET = Path(
    "outputs/preflop_policy_states/"
    "eroymcd2_unique_buckets_20m_n5_cap5m_packed_20260622"
)
DEFAULT_OUTPUT = Path(
    "outputs/preflop_continuation_beliefs/"
    "cascade_1024_actions0_3_to_12end.pt"
)
ENV_SNAPSHOT_FIELDS = (
    "deck",
    "deck_pos",
    "button",
    "street",
    "to_act",
    "last_to_act",
    "pot",
    "min_raise",
    "actions_this_round",
    "actions_last_round",
    "acted_since_reset",
    "stacks",
    "committed",
    "has_folded",
    "is_allin",
    "starting_stacks",
    "scale",
    "board_onehot",
    "hole_onehot",
    "board_indices",
    "last_board_indices",
    "hole_indices",
    "chips_placed",
    "done",
    "winner",
)


def _load_config_from_checkpoint(path: Path) -> Config:
    checkpoint = CheckpointIO.load(str(path), map_location=torch.device("cpu"))
    raw_cfg = checkpoint.get("config")
    if raw_cfg is None:
        raise ValueError(f"{path} does not contain embedded config")
    if isinstance(raw_cfg, Config):
        return copy.deepcopy(raw_cfg)
    if isinstance(raw_cfg, dict):
        return Config.from_dict(raw_cfg)
    raise TypeError(f"unsupported checkpoint config type {type(raw_cfg)!r}")


def _checkpoint_step(path: Path) -> Any:
    try:
        return CheckpointIO.metadata(str(path), map_location=torch.device("cpu")).get(
            "bucket_train_step",
            CheckpointIO.metadata(str(path), map_location=torch.device("cpu")).get(
                "step", "unknown"
            ),
        )
    except Exception:
        return "unknown"


def _execution_from_config(
    cfg: Config,
    *,
    state_dataset: Path,
    device: str,
    seed: int,
    roots: int,
    iterations: int,
    run_dir: Path,
) -> PreflopBucketExecutionConfig:
    pb = cfg.preflop_buckets
    return PreflopBucketExecutionConfig(
        command="sample_continuation_beliefs",
        state_dataset=str(state_dataset),
        base_checkpoint=str(run_dir / "actions_8_11/checkpoints/specialist_final.pt"),
        resume_from=None,
        output_dir=str(run_dir),
        presolve_bucket=str(pb.presolve_bucket),
        train_bucket=None,
        device=device,
        seed=int(seed),
        depth=int(pb.depth),
        actions_12_15_depth=pb.actions_12_15_depth,
        actions_8_11_depth=pb.actions_8_11_depth,
        actions_4_7_depth=pb.actions_4_7_depth,
        actions_0_3_depth=pb.actions_0_3_depth,
        cfr_iterations=int(iterations),
        warm_start_iterations=int(pb.warm_start_iterations),
        sparse_fused=bool(pb.sparse_fused),
        compile="off",
        belief_mode=str(pb.belief_mode),
        belief_profile=str(pb.belief_profile),
        belief_hand_dim=int(pb.belief_hand_dim),
        states_per_bucket=int(roots),
        train_batch_size=int(pb.train_batch_size),
        policy_train_batch_size=pb.policy_train_batch_size,
        cfr_batch_size=int(roots),
        cfr_model_batch_size=(
            0
            if cfg.search.cfr_model_batch_size is None
            else int(cfg.search.cfr_model_batch_size)
        ),
        actions_12_15_cfr_batch_size=pb.actions_12_15_cfr_batch_size,
        actions_8_11_cfr_batch_size=pb.actions_8_11_cfr_batch_size,
        actions_12_15_epochs=int(pb.actions_12_15_epochs),
        validation_items=int(pb.validation_items),
        validation_cfr_iterations=int(pb.validation_cfr_iterations),
        validation_interval_steps=int(pb.validation_interval_steps),
        validation_eval_batch_size=int(pb.validation_eval_batch_size),
        replay_buffer_batches=int(pb.replay_buffer_batches),
        storage_dtype=str(pb.storage_dtype),
        write_solved_shards=False,
        allow_partial=bool(pb.allow_partial),
        overwrite=False,
        progress_roots=int(pb.progress_roots),
        snapshot_interval_steps=int(pb.snapshot_interval_steps),
        use_wandb=False,
        wandb_project=str(cfg.wandb_project),
        wandb_name=None,
        wandb_group=None,
        wandb_tags=tuple(str(tag) for tag in cfg.wandb_tags),
        student_init=pb.student_init,
        student_init_from_base=bool(pb.student_init_from_base),
        bootstrap_distill_checkpoint=pb.bootstrap_distill_checkpoint,
        bootstrap_distill_epochs=int(pb.bootstrap_distill_epochs),
        bootstrap_distill_rows=pb.bootstrap_distill_rows,
        bootstrap_distill_batch_size=pb.bootstrap_distill_batch_size,
        bootstrap_distill_train_value=bool(pb.bootstrap_distill_train_value),
        distill_batch_size=int(pb.distill_batch_size),
        distill_buckets=None if pb.distill_buckets is None else tuple(pb.distill_buckets),
        distill_train_value=bool(pb.distill_train_value),
        checkpoint_12_15=pb.distill_checkpoints.checkpoint_12_15,
        checkpoint_8_11=pb.distill_checkpoints.checkpoint_8_11,
        checkpoint_4_7=pb.distill_checkpoints.checkpoint_4_7,
        checkpoint_0_3=pb.distill_checkpoints.checkpoint_0_3,
    )


def _snapshot_env(env: PBSEnv) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in ENV_SNAPSHOT_FIELDS:
        if hasattr(env, name):
            value = getattr(env, name)
            if isinstance(value, torch.Tensor):
                out[name] = value.detach().cpu().clone()
    return out


def _pbs_payload(pbs: tuple[PBSEnv, torch.Tensor]) -> dict[str, Any]:
    env, beliefs = pbs
    return {
        "env": _snapshot_env(env),
        "beliefs": beliefs.detach().cpu().clone(),
    }


def _belief_summary(beliefs: torch.Tensor, env: PBSEnv) -> dict[str, Any]:
    with torch.no_grad():
        b = beliefs.float()
        live = (~env.has_folded) & (~env.is_allin)
        live_rows = b[live]
        if live_rows.numel() == 0:
            return {"rows": int(b.shape[0]), "live_player_rows": 0}
        entropy = -(live_rows.clamp_min(1.0e-12) * live_rows.clamp_min(1.0e-12).log()).sum(
            dim=-1
        )
        max_class = live_rows.max(dim=-1).values
        aa_mass = live_rows[:, PREFLOP_HANDS - 1]
        quantiles = torch.tensor(
            [0.5, 0.9, 0.95, 0.99, 1.0], device=beliefs.device
        )
        return {
            "rows": int(b.shape[0]),
            "live_player_rows": int(live_rows.shape[0]),
            "actions_min": int(env.actions_this_round[: b.shape[0]].min().item()),
            "actions_max": int(env.actions_this_round[: b.shape[0]].max().item()),
            "max_class_mean": float(max_class.mean().item()),
            "max_class_q": [
                float(x)
                for x in torch.quantile(max_class, quantiles).detach().cpu().tolist()
            ],
            "aa_mass_mean": float(aa_mass.mean().item()),
            "aa_mass_q": [
                float(x)
                for x in torch.quantile(aa_mass, quantiles).detach().cpu().tolist()
            ],
            "entropy_mean": float(entropy.mean().item()),
            "entropy_q": [
                float(x)
                for x in torch.quantile(entropy, quantiles).detach().cpu().tolist()
            ],
        }


def _copy_pbs_to_env(pbs_env: PBSEnv, dest: PBSEnv, count: int) -> None:
    dest.reset(torch.arange(count, device=dest.device))
    dest.copy_state_from(
        pbs_env,
        torch.arange(count, device=dest.device),
        torch.arange(count, device=dest.device),
    )
    if count < dest.N:
        dest.done[count:] = True


def _install_absolute_action_sampler(
    evaluator: Any,
    *,
    min_actions: int,
    max_actions: int | None,
) -> None:
    def sample_preflop_cutoff_roots() -> PublicBeliefState | None:
        roots = int(evaluator.root_nodes)
        total = int(evaluator.total_nodes)
        if total <= roots:
            return None
        rows = torch.arange(total, device=evaluator.device)
        candidate_mask = (
            (rows >= roots)
            & evaluator.valid_mask
            & (evaluator.env.street == 0)
            & (evaluator.env.actions_this_round >= int(min_actions))
            & (~evaluator.env.done)
            & (~evaluator.allin_call_mask)
            & (evaluator.env.to_act >= 0)
            & (evaluator.env.to_act < evaluator.num_players)
        )
        if max_actions is not None:
            candidate_mask &= evaluator.env.actions_this_round <= int(max_actions)
        candidates = torch.where(candidate_mask)[0]
        if candidates.numel() == 0:
            return None

        root_owner = evaluator._get_root_index()[candidates].clamp(min=0, max=roots - 1)
        scores = torch.rand(
            candidates.numel(),
            generator=evaluator.generator,
            device=evaluator.device,
            dtype=evaluator.float_dtype,
        )
        best_scores = torch.full(
            (roots,),
            -1.0,
            dtype=evaluator.float_dtype,
            device=evaluator.device,
        )
        best_scores.scatter_reduce_(
            0,
            root_owner,
            scores,
            reduce="amax",
            include_self=True,
        )
        chosen = candidates[scores >= best_scores[root_owner]]
        if chosen.numel() > roots:
            chosen = chosen[:roots]

        pbs = PublicBeliefState.from_proto(
            env_proto=evaluator.env,
            beliefs=evaluator.beliefs_sample[chosen].clone(),
            num_envs=chosen.numel(),
        )
        pbs.env.copy_state_from(
            evaluator.env,
            chosen,
            torch.arange(chosen.numel(), device=evaluator.device),
        )
        return pbs

    evaluator._sample_preflop_cutoff_roots = sample_preflop_cutoff_roots


def _make_solver(
    *,
    checkpoint: Path,
    base_template: Config,
    execution: PreflopBucketExecutionConfig,
    bucket_label: str,
    device: torch.device,
    roots: int,
    next_action_range: tuple[int, int | None] | None,
) -> RebelCFRTrainer:
    cfg = _checkpoint_model_config(base_template, str(checkpoint))
    cfg = build_run_config(
        cfg,
        execution,
        checkpoint_dir=Path("/tmp/preflop_continuation_beliefs") / bucket_label,
        num_steps=1,
        num_envs=roots,
        bucket_label=bucket_label,
    )
    cfg.device = str(device)
    cfg.use_wandb = False
    cfg.resume_from = None
    cfg.strict_model_loading = False
    cfg.model.compile = "off"
    cfg.search.iterations = int(execution.cfr_iterations)
    cfg.search.continuation_value_target_sampling = next_action_range is not None
    cfg.search.continuation_value_target_streets = [0]
    cfg.search.continuation_value_target_min_depth = 1
    cfg.search.continuation_value_target_max_depth = None
    validate_rebel_config(cfg)

    trainer = RebelCFRTrainer(cfg=cfg, device=device, pregeneration_only=True)
    CheckpointIO.load_model_weights(
        trainer,
        str(checkpoint),
        strict=False,
        sync_models=True,
    )
    trainer.model.eval()
    trainer.inference_model.eval()
    evaluator = trainer.cfr_evaluator
    evaluator.cfr_iterations = int(cfg.search.iterations)
    if hasattr(evaluator, "_graph_capture_regime"):
        evaluator._graph_capture_regime = lambda _t: None
    if next_action_range is not None:
        _install_absolute_action_sampler(
            evaluator,
            min_actions=next_action_range[0],
            max_actions=next_action_range[1],
        )
    return trainer


def _solve_stage(
    *,
    trainer: RebelCFRTrainer,
    env: PBSEnv,
    beliefs: torch.Tensor,
    sample_continuation: bool,
) -> tuple[PBSEnv | None, torch.Tensor | None, dict[str, Any]]:
    roots = int(beliefs.shape[0])
    root_indices = torch.arange(roots, device=trainer.device)
    evaluator = trainer.cfr_evaluator
    evaluator.initialize_subgame(env, root_indices, beliefs)
    next_pbs = evaluator.evaluate_cfr(
        training_mode=True,
        sample_continuation=sample_continuation,
    )
    stats = {
        "root_nodes": int(evaluator.root_nodes),
        "total_nodes": int(evaluator.total_nodes),
        "tree_depth": int(evaluator.tree_depth),
    }
    if next_pbs is None:
        return None, None, stats
    return next_pbs.env, next_pbs.beliefs, stats


def _resolve_defaults(run_dir: Path) -> dict[str, Path]:
    return {
        "0_3_solver": run_dir / "actions_4_7/checkpoints/specialist_inprogress.pt",
        "4_7_solver": run_dir / "actions_8_11/checkpoints/specialist_final.pt",
        "8_11_solver": run_dir / "actions_12_15/checkpoints/specialist_final.pt",
        "12_end_solver": run_dir / "actions_12_15/checkpoints/specialist_final.pt",
    }


def run(args: argparse.Namespace) -> Path:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    run_dir = args.run_dir.expanduser().resolve()
    state_dataset = args.state_dataset.expanduser().resolve()
    checkpoints = _resolve_defaults(run_dir)
    for key in list(checkpoints):
        override = getattr(args, f"checkpoint_{key}")
        if override is not None:
            checkpoints[key] = override.expanduser().resolve()
        else:
            checkpoints[key] = checkpoints[key].resolve()
        if not checkpoints[key].exists():
            raise FileNotFoundError(f"{key} checkpoint not found: {checkpoints[key]}")

    base_template = _load_config_from_checkpoint(checkpoints["0_3_solver"])
    execution = _execution_from_config(
        base_template,
        state_dataset=state_dataset,
        device=str(device),
        seed=int(args.seed),
        roots=int(args.roots),
        iterations=int(args.iterations),
        run_dir=run_dir,
    )
    reader = PublicStateBucketReader(
        state_dataset,
        "actions_0_3",
        allow_partial=False,
        seed=int(args.seed),
    )
    initial_states = next(
        reader.iter_state_batches(
            batch_size=int(args.roots),
            max_rows=int(args.roots),
            seed=int(args.seed),
        )
    )
    env0 = _make_env_from_manifest(
        reader.manifest,
        num_envs=int(args.roots),
        device=device,
        seed=int(args.seed) + 100,
    )
    rows = _copy_public_states_to_env(env0, initial_states)
    if rows != int(args.roots):
        raise RuntimeError(f"expected {args.roots} initial roots, got {rows}")
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed) + 900_000)
    beliefs0 = _random_beliefs(
        rows,
        base_template.env.num_players,
        device=device,
        rng=rng,
        mode=str(args.belief_mode),
        profile=str(base_template.preflop_buckets.belief_profile),
        hand_dim=int(base_template.preflop_buckets.belief_hand_dim),
    )

    stages = [
        ("actions_0_3", "0_3_solver", env0, beliefs0, (4, 7)),
        ("actions_4_7", "4_7_solver", None, None, (8, 11)),
        ("actions_8_11", "8_11_solver", None, None, (12, 15)),
        ("actions_12_end", "12_end_solver", None, None, None),
    ]
    payload: dict[str, Any] = {
        "metadata": {
            "kind": "preflop_bucket_continuation_beliefs",
            "roots_requested": int(args.roots),
            "seed": int(args.seed),
            "iterations": int(args.iterations),
            "belief_mode": str(args.belief_mode),
            "belief_profile": str(base_template.preflop_buckets.belief_profile),
            "belief_hand_dim": int(base_template.preflop_buckets.belief_hand_dim),
            "state_dataset": str(state_dataset),
            "run_dir": str(run_dir),
            "checkpoints": {key: str(value) for key, value in checkpoints.items()},
            "checkpoint_steps": {
                key: _checkpoint_step(value) for key, value in checkpoints.items()
            },
            "execution": asdict(execution),
        },
        "stages": {},
        "summaries": {},
    }

    current_env: PBSEnv | None = env0
    current_beliefs: torch.Tensor | None = beliefs0
    for label, checkpoint_key, stage_env, stage_beliefs, next_action_range in stages:
        if stage_env is not None:
            current_env = stage_env
        if stage_beliefs is not None:
            current_beliefs = stage_beliefs
        if current_env is None or current_beliefs is None:
            raise RuntimeError(f"missing roots for {label}")

        roots = int(current_beliefs.shape[0])
        trainer = _make_solver(
            checkpoint=checkpoints[checkpoint_key],
            base_template=base_template,
            execution=execution,
            bucket_label="actions_12_15" if label == "actions_12_end" else label,
            device=device,
            roots=roots,
            next_action_range=next_action_range,
        )
        payload["stages"][label] = {
            "checkpoint_key": checkpoint_key,
            "checkpoint": str(checkpoints[checkpoint_key]),
            "root": _pbs_payload((current_env, current_beliefs)),
        }
        payload["summaries"][f"{label}_root"] = _belief_summary(
            current_beliefs, current_env
        )

        next_env, next_beliefs, solve_stats = _solve_stage(
            trainer=trainer,
            env=current_env,
            beliefs=current_beliefs,
            sample_continuation=next_action_range is not None,
        )
        payload["stages"][label]["solve_stats"] = solve_stats
        if next_env is not None and next_beliefs is not None:
            payload["stages"][label]["sampled_continuation"] = _pbs_payload(
                (next_env, next_beliefs)
            )
            payload["summaries"][f"{label}_sampled_continuation"] = _belief_summary(
                next_beliefs,
                next_env,
            )
            current_env = next_env
            current_beliefs = next_beliefs
        else:
            current_env = None
            current_beliefs = None
        print(
            f"{label}: roots={roots} nodes={solve_stats['total_nodes']} "
            f"next_roots={0 if current_beliefs is None else int(current_beliefs.shape[0])}",
            flush=True,
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "metadata": payload["metadata"],
                "summaries": payload["summaries"],
                "stage_solve_stats": {
                    key: value["solve_stats"] for key, value in payload["stages"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"saved: {output}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    return output


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--state-dataset", type=Path, default=DEFAULT_STATE_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--roots", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--belief-mode",
        default="random",
        choices=("random", "uniform", "histogram", "mixed", "coverage", "topk"),
    )
    parser.add_argument("--checkpoint-0-3-solver", type=Path, default=None)
    parser.add_argument("--checkpoint-4-7-solver", type=Path, default=None)
    parser.add_argument("--checkpoint-8-11-solver", type=Path, default=None)
    parser.add_argument("--checkpoint-12-end-solver", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
