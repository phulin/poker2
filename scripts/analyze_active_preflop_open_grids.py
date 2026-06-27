#!/usr/bin/env python3
"""Analyze 6-player preflop open ranges from the active specialist checkpoint."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import TextIO

import torch

from p2.config.rebel_load import validate_rebel_config
from p2.core.structured_config import Config
from p2.env.analyze_tensor_env import _create_169_grid
from p2.env.card_utils import PREFLOP_HANDS, preflop_class_multiplicity_tensor
from p2.env.pbs_env import PBSEnv
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.checkpoint_io import CheckpointIO
from p2.utils.training_utils import print_combined_tables


DEFAULT_OUTPUT_ROOT = Path("outputs/preflop_backward_induction")
DEFAULT_RUN_NAME = "gated_chain_4_7_from8_11_gradfix_lr00105_300cfr_20260626"
CHECKPOINT_NAMES = (
    "specialist_inprogress.pt",
    "rebel_latest.pt",
    "specialist_final.pt",
)
POSITION_LABELS = ("LJ", "HJ", "CO", "BTN", "SB", "BB")
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
EVALUATOR_TREE_TENSOR_FIELDS = (
    "beliefs",
    "beliefs_avg",
    "beliefs_sample",
    "self_reach",
    "self_reach_avg",
    "policy_probs",
    "policy_probs_avg",
    "policy_probs_sample",
    "latest_values",
    "values_avg",
    "last_model_values",
    "cumulative_regrets",
    "positive_regrets",
    "prev_actor",
    "parent_index",
    "action_from_parent",
    "child_offsets",
    "child_count",
    "child_mask",
    "valid_mask",
    "leaf_mask",
    "new_street_mask",
    "showdown_mask",
    "allin_call_mask",
    "allowed_hands",
    "model_indices",
    "cutoff_indices",
    "cutoff_model_positions",
    "new_street_indices",
    "new_street_model_positions",
    "target_source",
)


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _active_run_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for pid_file in root.glob("*/trainer.pid"):
        try:
            pid = int(pid_file.read_text().strip())
        except (OSError, ValueError):
            continue
        if _is_process_alive(pid):
            out.append(pid_file.parent)
    return sorted(out, key=lambda path: path.stat().st_mtime, reverse=True)


def _latest_checkpoint_in_run(run_dir: Path) -> Path:
    candidates = [
        path
        for name in CHECKPOINT_NAMES
        for path in run_dir.glob(f"*/checkpoints/{name}")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(f"no specialist checkpoints found under {run_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


def _default_tree_output_dir(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    checkpoint_step: object,
) -> Path:
    if args.tree_output_dir is not None:
        return args.tree_output_dir.expanduser().resolve()
    if args.output is not None:
        output = args.output.expanduser().resolve()
        return output.with_suffix("")
    checkpoint_tag = _safe_name(checkpoint_path.parent.parent.name)
    return (
        Path("outputs")
        / "preflop_open_tree_snapshots"
        / f"{checkpoint_tag}_step{checkpoint_step}_iter{args.iterations}_seed{args.seed}"
    ).resolve()


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint.expanduser().resolve()
    if args.run_dir is not None:
        return _latest_checkpoint_in_run(args.run_dir.expanduser().resolve())

    active = _active_run_dirs(args.output_root)
    if active:
        return _latest_checkpoint_in_run(active[0])

    default_run = args.output_root / DEFAULT_RUN_NAME
    if default_run.exists():
        return _latest_checkpoint_in_run(default_run)

    run_dirs = [path for path in args.output_root.glob("*") if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"no runs found under {args.output_root}")
    latest_run = max(run_dirs, key=lambda path: path.stat().st_mtime)
    return _latest_checkpoint_in_run(latest_run)


def _load_checkpoint_config(checkpoint_path: Path) -> Config:
    checkpoint = CheckpointIO.load(str(checkpoint_path), map_location=torch.device("cpu"))
    raw_cfg = checkpoint.get("config")
    if raw_cfg is None:
        raise ValueError(f"{checkpoint_path} does not contain embedded config")
    if isinstance(raw_cfg, Config):
        cfg = copy.deepcopy(raw_cfg)
    elif isinstance(raw_cfg, dict):
        cfg = Config.from_dict(raw_cfg)
    else:
        raise TypeError(f"unsupported checkpoint config type: {type(raw_cfg)!r}")
    return cfg


def _analysis_config(
    checkpoint_path: Path,
    *,
    device: torch.device,
    iterations: int,
    seed: int,
    closing_leaf_checkpoint: str | Path | None,
) -> Config:
    cfg = _load_checkpoint_config(checkpoint_path)
    cfg.device = str(device)
    cfg.seed = int(seed)
    cfg.num_envs = 1
    cfg.use_wandb = False
    cfg.resume_from = None
    cfg.model.compile = "off"
    cfg.env.num_players = 6
    cfg.data.mode = "live"
    cfg.data.live_root_source = "self_play"
    cfg.data.warmup_self_play_roots = False
    cfg.search.sparse = True
    cfg.search.sparse_fused = True
    cfg.search.iterations = int(iterations)
    cfg.search.iterations_final = None
    cfg.search.closing_leaf_checkpoint = (
        None if closing_leaf_checkpoint is None else str(closing_leaf_checkpoint)
    )
    cfg.search.continuation_value_target_sampling = False
    cfg.strict_model_loading = False
    validate_rebel_config(cfg)
    return cfg


def _checkpoint_config_closing_leaf_checkpoint(checkpoint_path: Path) -> Path | None:
    cfg = _load_checkpoint_config(checkpoint_path)
    value = cfg.search.closing_leaf_checkpoint
    if value is None:
        return None
    return Path(str(value)).expanduser().resolve()


def _make_trainer(cfg: Config, checkpoint_path: Path, device: torch.device) -> RebelCFRTrainer:
    trainer = RebelCFRTrainer(cfg, device, pregeneration_only=True)
    CheckpointIO.load_model_weights(
        trainer,
        str(checkpoint_path),
        strict=False,
        sync_models=True,
    )
    trainer.model.eval()
    trainer.inference_model.eval()
    trainer.cfr_evaluator.cfr_iterations = int(cfg.search.iterations)
    # The analyzer constructs fresh one-root subgames and calls the feature
    # encoder during leaf evaluation. CUDA graph capture rejects the encoder's
    # scalar host-to-device tensor creation, so keep the fused evaluator but run
    # the CFR iteration loop eagerly.
    if hasattr(trainer.cfr_evaluator, "_graph_capture_regime"):
        trainer.cfr_evaluator._graph_capture_regime = lambda _t: None
    return trainer


def _set_closing_leaf_model(
    trainer: RebelCFRTrainer,
    checkpoint_path: Path,
) -> None:
    model = trainer.load_closing_leaf_model(str(checkpoint_path))
    value_model = getattr(model, "value_model", model)
    value_model.eval()
    evaluator = trainer.cfr_evaluator
    evaluator.closing_leaf_value_model = value_model
    evaluator.closing_leaf_value_encoder = None


def _make_open_env(cfg: Config, device: torch.device, rng: torch.Generator) -> PBSEnv:
    env = PBSEnv(
        num_envs=1,
        num_players=6,
        mean_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        rng=rng,
        float_dtype=torch.float32,
        stack_mode=cfg.env.stack_mode,
        min_stack_bb=cfg.env.min_stack_bb,
        mid_stack_bb=cfg.env.mid_stack_bb,
        max_stack_bb=cfg.env.max_stack_bb,
        high_stack_mass_ratio=cfg.env.high_stack_mass_ratio,
        force_heads_up_preflop_flop=getattr(
            cfg.env,
            "force_heads_up_preflop_flop",
            True,
        ),
    )
    # With the repo's seat convention, button=3 makes the first preflop actor
    # seat 0. We label that sequence LJ, HJ, CO, BTN, SB, BB below.
    env.reset(force_button=torch.tensor([3], dtype=torch.long, device=device))
    return env


def _uniform_preflop_beliefs(cfg: Config, device: torch.device) -> torch.Tensor:
    prior = preflop_class_multiplicity_tensor(device=device).to(dtype=torch.float32)
    prior = prior / prior.sum().clamp_min(1.0)
    return prior.expand(1, cfg.env.num_players, PREFLOP_HANDS).clone()


def _root_policy(
    evaluator,
    env: PBSEnv,
    beliefs: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    evaluator.initialize_subgame(
        env,
        torch.tensor([0], dtype=torch.long, device=env.device),
        beliefs,
    )
    evaluator.evaluate_cfr(training_mode=False)

    start = int(evaluator.depth_offsets[1])
    end = int(evaluator.depth_offsets[2])
    child_indices = torch.arange(start, end, dtype=torch.long, device=env.device)
    action_ids = evaluator.action_from_parent[child_indices]
    hand_dim = int(getattr(evaluator, "hand_dim", PREFLOP_HANDS))
    num_actions = int(evaluator.num_actions)
    root_policy = torch.zeros(
        num_actions,
        hand_dim,
        dtype=evaluator.policy_probs_avg.dtype,
        device=env.device,
    )
    legal = torch.zeros(num_actions, dtype=torch.bool, device=env.device)
    for child_idx, action_id in zip(child_indices, action_ids, strict=True):
        action = int(action_id.item())
        if action < 0 or action >= num_actions:
            continue
        root_policy[action] = evaluator.policy_probs_avg[child_idx]
        if hasattr(evaluator, "valid_mask") and evaluator.valid_mask.numel() > child_idx:
            legal[action] = bool(evaluator.valid_mask[child_idx].item())
        else:
            legal[action] = True
    root_policy = root_policy / root_policy.sum(dim=0, keepdim=True).clamp_min(1e-12)
    actor = int(env.to_act[0].item())
    values = evaluator.values_avg[0, actor].detach().clone()
    return actor, root_policy.T.detach().clone(), values, legal


def _to_cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {str(key): _to_cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_cpu_snapshot(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _snapshot_env(env: PBSEnv) -> dict[str, torch.Tensor]:
    return {
        name: getattr(env, name).detach().cpu().clone()
        for name in ENV_SNAPSHOT_FIELDS
        if hasattr(env, name)
    }


def _snapshot_evaluator_tensors(evaluator) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in EVALUATOR_TREE_TENSOR_FIELDS:
        value = getattr(evaluator, name, None)
        if isinstance(value, torch.Tensor):
            out[name] = value.detach().cpu().clone()
    depth_offsets = getattr(evaluator, "depth_offsets", None)
    if isinstance(depth_offsets, (list, tuple)):
        out["depth_offsets"] = torch.tensor(depth_offsets, dtype=torch.long)
    elif isinstance(depth_offsets, torch.Tensor):
        out["depth_offsets"] = depth_offsets.detach().cpu().clone()
    return out


def _save_tree_snapshot(
    *,
    output_dir: Path,
    position_index: int,
    position: str,
    checkpoint_path: Path,
    checkpoint_step: object,
    cutoff_checkpoint: Path | None,
    cfg: Config,
    evaluator,
    source_env: PBSEnv,
    source_beliefs: torch.Tensor,
    actor: int,
    probs: torch.Tensor,
    values: torch.Tensor,
    legal: torch.Tensor,
    fold_beliefs: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    action_means = _weighted_action_means(probs)
    config = asdict(cfg)
    config["search"]["closing_leaf_checkpoint"] = (
        None if cutoff_checkpoint is None else str(cutoff_checkpoint)
    )
    snapshot = {
        "metadata": {
            "position_index": int(position_index),
            "position": position,
            "actor": int(actor),
            "checkpoint": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "closing_leaf_checkpoint": (
                None if cutoff_checkpoint is None else str(cutoff_checkpoint)
            ),
            "iterations": int(cfg.search.iterations),
            "cfr_type": str(cfg.search.cfr_type),
            "predictive_cfr_dcfr_hybrid": bool(
                cfg.search.predictive_cfr_dcfr_hybrid
            ),
            "fold_beliefs": bool(fold_beliefs),
            "num_players": int(evaluator.num_players),
            "num_actions": int(evaluator.num_actions),
            "hand_dim": int(getattr(evaluator, "hand_dim", PREFLOP_HANDS)),
            "root_nodes": int(evaluator.root_nodes),
            "total_nodes": int(evaluator.total_nodes),
            "tree_depth": int(evaluator.tree_depth),
            "bet_bins": list(getattr(evaluator, "bet_bins", [])),
            "action_means": {
                "fold": float(action_means[0].item()),
                "call_check": float(action_means[1].item()),
                "bet_total": float(action_means[2:-1].sum().item()),
                "allin": float(action_means[-1].item()),
            },
        },
        "config": config,
        "source_env": _snapshot_env(source_env),
        "source_beliefs": source_beliefs.detach().cpu().clone(),
        "root_policy_probs": probs.detach().cpu().clone(),
        "root_actor_values_avg": values.detach().cpu().clone(),
        "root_legal": legal.detach().cpu().clone(),
        "tree_env": _snapshot_env(evaluator.env),
        "tree_tensors": _snapshot_evaluator_tensors(evaluator),
        "stats": _to_cpu_snapshot(getattr(evaluator, "stats", {})),
    }
    path = output_dir / f"{position_index:02d}_{position}_tree.pt"
    torch.save(snapshot, path)
    return path


def _weighted_action_means(probs: torch.Tensor) -> torch.Tensor:
    weights = preflop_class_multiplicity_tensor(device=probs.device).to(probs.dtype)
    weights = weights / weights.sum().clamp_min(1.0)
    return (probs * weights[:, None]).sum(dim=0)


def _belief_summary(belief: torch.Tensor) -> tuple[float, float, float]:
    weights = preflop_class_multiplicity_tensor(device=belief.device).to(belief.dtype)
    weights = weights / weights.sum().clamp_min(1.0)
    entropy = -(belief * belief.clamp_min(1e-12).log()).sum()
    uniform = weights
    kl_uniform = (belief * (belief.clamp_min(1e-12) / uniform.clamp_min(1e-12)).log()).sum()
    top = belief.max()
    return float(entropy.item()), float(kl_uniform.item()), float(top.item())


def _print_diagnostics(
    out: TextIO,
    *,
    position: str,
    env: PBSEnv,
    beliefs: torch.Tensor,
    actor: int,
    legal: torch.Tensor,
    probs: torch.Tensor,
    values: torch.Tensor,
) -> None:
    amounts, amount_legal = env.legal_bins_amounts_and_mask()
    means = _weighted_action_means(probs)
    value_weights = preflop_class_multiplicity_tensor(device=values.device).to(
        values.dtype
    )
    value_weights = value_weights / value_weights.sum().clamp_min(1.0)
    bet_total = means[2:-1].sum()
    entropy, kl_uniform, top = _belief_summary(beliefs[0, actor])
    print(f"\n--- {position} diagnostics ---", file=out)
    print(
        "state: "
        f"actor={actor} button={int(env.button[0].item())} "
        f"to_act={int(env.to_act[0].item())} "
        f"pot={int(env.pot[0].item())} "
        f"committed={env.committed[0].detach().cpu().tolist()} "
        f"stacks={env.stacks[0].detach().cpu().tolist()} "
        f"has_folded={env.has_folded[0].detach().cpu().tolist()}",
        file=out,
    )
    print(
        "legal_bins: "
        f"mask={legal.detach().cpu().tolist()} "
        f"amount_mask={amount_legal[0].detach().cpu().tolist()} "
        f"amounts={amounts[0].detach().cpu().tolist()}",
        file=out,
    )
    print(
        "weighted_action_mean: "
        f"fold={float(means[0].item()):.6f} "
        f"call={float(means[1].item()):.6f} "
        f"bet_total={float(bet_total.item()):.6f} "
        f"allin={float(means[-1].item()):.6f}",
        file=out,
    )
    print(
        "actor_belief: "
        f"entropy={entropy:.6f} kl_vs_combo_prior={kl_uniform:.6f} top_class={top:.6f}",
        file=out,
    )
    print(
        "actor_values: "
        f"mean={float(values.mean().item()):.6f} "
        f"weighted_mean={float((values * value_weights).sum().item()):.6f} "
        f"min={float(values.min().item()):.6f} "
        f"max={float(values.max().item()):.6f}",
        file=out,
    )


def _grid(probabilities: torch.Tensor) -> str:
    return _create_169_grid(
        probabilities.to(torch.float32).reshape(13, 13),
        "probability",
    )


def _print_position(
    out: TextIO,
    *,
    position: str,
    seat: int,
    legal: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    del legal
    fold_grid = _grid(probs[:, 0]).splitlines()
    call_grid = _grid(probs[:, 1]).splitlines()
    bet_grid = _grid(probs[:, 2:-1].sum(dim=1)).splitlines()
    allin_grid = _grid(probs[:, -1]).splitlines()
    print(f"\n=== {position} open node (seat {seat}) ===", file=out)
    with _redirect_stdout(out):
        print_combined_tables(
            [
                (call_grid, f"{position} call/check (%)"),
                (fold_grid, f"{position} fold (%)"),
            ],
        )
        print_combined_tables(
            [
                (bet_grid, f"{position} bet total (%)"),
                (allin_grid, f"{position} all-in (%)"),
            ],
        )


class _redirect_stdout:
    def __init__(self, target: TextIO) -> None:
        self.target = target
        self.original: TextIO | None = None

    def __enter__(self) -> None:
        self.original = sys.stdout
        sys.stdout = self.target

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.original is not None
        sys.stdout = self.original


def _step_fold_with_belief_update(
    env: PBSEnv,
    beliefs: torch.Tensor,
    *,
    actor: int,
    fold_probs: torch.Tensor,
) -> None:
    actor_belief = beliefs[0, actor] * fold_probs.to(beliefs.dtype)
    denom = actor_belief.sum().clamp_min(1e-12)
    beliefs[0, actor] = actor_belief / denom

    legal = env.legal_bins_mask()
    if not bool(legal[0, 0].item()):
        raise RuntimeError(f"fold is not legal for actor seat {actor}")
    env.step_bins(torch.zeros(1, dtype=torch.long, device=env.device), legal_masks=legal)


def run_analysis(args: argparse.Namespace) -> Path | None:
    checkpoint_path = _resolve_checkpoint(args)
    device = torch.device(args.device)
    original_cutoff_checkpoint = (
        args.sb_cutoff_checkpoint.expanduser().resolve()
        if args.sb_cutoff_checkpoint is not None
        else _checkpoint_config_closing_leaf_checkpoint(checkpoint_path)
    )
    cfg = _analysis_config(
        checkpoint_path,
        device=device,
        iterations=args.iterations,
        seed=args.seed,
        closing_leaf_checkpoint=checkpoint_path,
    )
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed))
    trainer = _make_trainer(cfg, checkpoint_path, device)
    evaluator = trainer.cfr_evaluator
    env = _make_open_env(cfg, device, rng)
    beliefs = _uniform_preflop_beliefs(cfg, device)
    active_cutoff_checkpoint: Path | None = checkpoint_path
    installed_cutoff_checkpoint: Path | None = checkpoint_path
    checkpoint_step = CheckpointIO.metadata(
        str(checkpoint_path),
        map_location=torch.device("cpu"),
    ).get("bucket_train_step", "unknown")
    tree_output_dir: Path | None = None
    if args.save_trees:
        tree_output_dir = _default_tree_output_dir(
            args=args,
            checkpoint_path=checkpoint_path,
            checkpoint_step=checkpoint_step,
        )

    output_handle: TextIO | None = None
    out: TextIO = sys.stdout
    output_path: Path | None = None
    if args.output is not None:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w")
        out = output_handle

    try:
        print(f"checkpoint: {checkpoint_path}", file=out)
        print(f"checkpoint_step: {checkpoint_step}", file=out)
        print(f"iterations: {cfg.search.iterations}", file=out)
        print(f"cfr_type: {cfg.search.cfr_type}", file=out)
        print(f"predictive_cfr_dcfr_hybrid: {cfg.search.predictive_cfr_dcfr_hybrid}", file=out)
        print(f"allin_by_depth: {cfg.search.allin_by_depth}", file=out)
        print(f"preflop_allin_169_model_checkpoint: {cfg.search.preflop_allin_169_model_checkpoint}", file=out)
        print(f"active_closing_leaf_checkpoint: {active_cutoff_checkpoint}", file=out)
        print(f"after_4_actions_closing_leaf_checkpoint: {original_cutoff_checkpoint}", file=out)
        if tree_output_dir is not None:
            print(f"tree_output_dir: {tree_output_dir}", file=out)
        print(f"stacks: {env.starting_stacks[0].detach().cpu().tolist()}", file=out)

        for index, position in enumerate(POSITION_LABELS):
            if bool(env.done[0].item()):
                print(
                    f"\n=== {position} open node ===\n"
                    "Terminal fold-through state: previous folds award the pot to "
                    "the remaining player, so there is no legal BB decision node.",
                    file=out,
                )
                break
            action_count = int(env.actions_this_round[0].item())
            cutoff_checkpoint = active_cutoff_checkpoint
            if action_count >= int(args.sb_cutoff_after_actions):
                cutoff_checkpoint = original_cutoff_checkpoint
            if cutoff_checkpoint != installed_cutoff_checkpoint and cutoff_checkpoint is not None:
                print(
                    f"\nswitching closing leaf checkpoint at {position} "
                    f"(actions_this_round={action_count}): {cutoff_checkpoint}",
                    file=out,
                )
                _set_closing_leaf_model(trainer, cutoff_checkpoint)
                installed_cutoff_checkpoint = cutoff_checkpoint
            elif cutoff_checkpoint is None and installed_cutoff_checkpoint is not None:
                evaluator.closing_leaf_value_model = None
                evaluator.closing_leaf_value_encoder = None
                installed_cutoff_checkpoint = None
            actor, probs, _values, legal = _root_policy(evaluator, env, beliefs)
            if tree_output_dir is not None:
                snapshot_path = _save_tree_snapshot(
                    output_dir=tree_output_dir,
                    position_index=index,
                    position=position,
                    checkpoint_path=checkpoint_path,
                    checkpoint_step=checkpoint_step,
                    cutoff_checkpoint=cutoff_checkpoint,
                    cfg=cfg,
                    evaluator=evaluator,
                    source_env=env,
                    source_beliefs=beliefs,
                    actor=actor,
                    probs=probs,
                    values=_values,
                    legal=legal,
                    fold_beliefs=args.fold_beliefs,
                )
                print(f"saved_tree: {snapshot_path}", file=out)
            _print_position(out, position=position, seat=actor, legal=legal, probs=probs)
            if args.diagnostics:
                _print_diagnostics(
                    out,
                    position=position,
                    env=env,
                    beliefs=beliefs,
                    actor=actor,
                    legal=legal,
                    probs=probs,
                    values=_values,
                )
            if index < len(POSITION_LABELS) - 1:
                if args.fold_beliefs:
                    _step_fold_with_belief_update(
                        env,
                        beliefs,
                        actor=actor,
                        fold_probs=probs[:, 0],
                    )
                else:
                    legal = env.legal_bins_mask()
                    env.step_bins(
                        torch.zeros(1, dtype=torch.long, device=env.device),
                        legal_masks=legal,
                    )
    finally:
        if output_handle is not None:
            output_handle.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 6-player preflop CFR analyzer from the active "
            "backward-induction checkpoint and print open/fold-through grids."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tree-output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sb-cutoff-checkpoint",
        type=Path,
        default=None,
        help=(
            "Closing-leaf checkpoint to use once actions_this_round reaches "
            "--sb-cutoff-after-actions. Defaults to the checkpoint config's "
            "original closing_leaf_checkpoint."
        ),
    )
    parser.add_argument("--sb-cutoff-after-actions", type=int, default=4)
    parser.add_argument(
        "--no-save-trees",
        dest="save_trees",
        action="store_false",
        help="Do not write per-position evaluator tree snapshots.",
    )
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--no-fold-beliefs",
        dest="fold_beliefs",
        action="store_false",
        help="Do not condition folded players' ranges on their fold probabilities.",
    )
    parser.set_defaults(fold_beliefs=True, save_trees=True)
    return parser.parse_args()


def main() -> None:
    output_path = run_analysis(parse_args())
    if output_path is not None:
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
