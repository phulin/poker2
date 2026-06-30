#!/usr/bin/env python3
"""Compare 6p EOS student and 2p teacher values on fixed leaves with sampled beliefs."""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config
from p2.core.structured_config import ModelType
from p2.core.structured_config import PreflopModelType
from p2.env.card_utils import IDX_TO_RANK
from p2.env.card_utils import PREFLOP_HANDS
from p2.env.card_utils import preflop_class_multiplicity_tensor
from p2.env.pbs_env import PBSEnv
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.rl.checkpoint_io import CheckpointIO
from p2.search.preflop_live_pair_distillation import _project_live_pair_env
from p2.stages.preflop_backward_induction import _random_beliefs

from distill_epreflop_6p_live_pair import DEFAULT_TARGET_CHECKPOINT
from distill_epreflop_6p_live_pair import _value_model


DEFAULT_SNAPSHOT = (
    "outputs/preflop_open_tree_snapshots/"
    "btn_only_even20k_step1900_iter5000_seed42_uniform_eos/03_BTN_tree.pt"
)
DEFAULT_STUDENT = (
    "outputs/epreflop_6p_live_pair/"
    "full_b16384_s5260_muon2e-2_adamw2e-3_warmup10_linear_shuffled_per_player_val/"
    "checkpoints/distilled_final.pt"
)
DEFAULT_OUTPUT = (
    "outputs/preflop_open_tree_snapshots/"
    "btn_only_even20k_step1900_iter5000_seed42_uniform_eos/"
    "eos_indomain_belief_probe.json"
)
NODE_SPECS = {
    "limp_fold_check": {
        "node": 44,
        "path": "call/check -> fold -> call/check",
    },
    "r325_fold_call": {
        "node": 100,
        "path": "r325 -> fold -> call/check",
    },
}
SELECTED_HANDS = (
    "AA",
    "KK",
    "QQ",
    "JJ",
    "TT",
    "99",
    "AKs",
    "AQs",
    "AKo",
    "KQs",
    "T9s",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--student-checkpoint", default=DEFAULT_STUDENT)
    parser.add_argument("--target-checkpoint", default=DEFAULT_TARGET_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--random-samples", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026062901)
    return parser.parse_args()


def _hand_name_to_idx(name: str) -> int:
    ranks = {rank: idx for idx, rank in enumerate(IDX_TO_RANK)}
    if len(name) == 2:
        rank = ranks[name[0]]
        return rank * 13 + rank
    hi = ranks[name[0]]
    lo = ranks[name[1]]
    if name[2] == "s":
        return max(hi, lo) * 13 + min(hi, lo)
    return min(hi, lo) * 13 + max(hi, lo)


def _load_student(path: Path, *, device: torch.device, batch_size: int):
    checkpoint = CheckpointIO.load(str(path), map_location=torch.device("cpu"))
    raw_cfg = checkpoint.get("config")
    if isinstance(raw_cfg, Config):
        cfg = copy.deepcopy(raw_cfg)
    elif isinstance(raw_cfg, dict):
        cfg = Config.from_dict(copy.deepcopy(raw_cfg))
    else:
        raise ValueError(f"{path} does not contain a trainer config")
    cfg.device = str(device)
    cfg.num_envs = int(batch_size)
    cfg.train.batch_size = int(batch_size)
    cfg.use_wandb = False
    cfg.model.compile = "off"
    cfg.data.warmup_self_play_roots = False
    cfg.train.save_replay_buffers = False
    cfg.train.replay_buffer_batches = 1
    cfg.train.replay_buffer_device = "cpu"
    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    CheckpointIO.load_model_weights(
        trainer,
        str(path),
        strict=cfg.strict_model_loading,
        sync_models=False,
    )
    trainer.model.eval()
    return trainer, _value_model(trainer.model)


def _load_teacher(path: Path, *, device: torch.device, batch_size: int) -> torch.nn.Module:
    cfg = Config()
    cfg.device = str(device)
    cfg.num_envs = int(batch_size)
    cfg.use_wandb = False
    cfg.model.name = ModelType.better_ffn
    cfg.model.preflop_hand_dim = PREFLOP_HANDS
    cfg.model.preflop_model_type = PreflopModelType.gated_token_mixer
    cfg.model.enforce_zero_sum = False
    cfg.env.num_players = 6
    cfg.data.live_root_source = "self_play"
    cfg.data.warmup_self_play_roots = False
    cfg.train.replay_buffer_batches = 1
    cfg.train.replay_buffer_device = "cpu"
    trainer = RebelCFRTrainer(
        cfg=cfg,
        device=device,
        pregeneration_only=True,
    )
    return trainer.load_closing_leaf_model(str(path))


def _env_from_snapshot(
    snapshot: dict[str, Any],
    *,
    node_ids: torch.Tensor,
    device: torch.device,
) -> PBSEnv:
    cfg = Config.from_dict(copy.deepcopy(snapshot["config"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    env = PBSEnv(
        num_envs=int(node_ids.numel()),
        num_players=6,
        mean_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        rng=generator,
        float_dtype=torch.float32,
        stack_mode=cfg.env.stack_mode,
        min_stack_bb=cfg.env.min_stack_bb,
        mid_stack_bb=cfg.env.mid_stack_bb,
        max_stack_bb=cfg.env.max_stack_bb,
        high_stack_mass_ratio=cfg.env.high_stack_mass_ratio,
        force_heads_up_preflop_flop=False,
    )
    tree_env = snapshot["tree_env"]
    node_ids_cpu = node_ids.detach().cpu()
    for name, source in tree_env.items():
        if not hasattr(env, name):
            continue
        target = getattr(env, name)
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            continue
        target[: node_ids.numel()].copy_(source[node_ids_cpu].to(device=device))
    if hasattr(env, "last_aggressive_amount"):
        env.last_aggressive_amount[: node_ids.numel()].zero_()
    return env


@torch.no_grad()
def _teacher_raw_values(
    target_model: torch.nn.Module,
    env: PBSEnv,
    beliefs: torch.Tensor,
) -> torch.Tensor:
    rows = int(beliefs.shape[0])
    device = env.device
    baseline = (
        env.stacks[:rows].to(torch.float32)
        - env.starting_stacks[:rows].to(torch.float32)
    ) / env.scale[:rows, None].to(torch.float32).clamp_min(1.0)
    out = baseline[:, :, None].expand(-1, -1, PREFLOP_HANDS).clone()
    row_indices = torch.arange(rows, dtype=torch.long, device=device)
    pair_players = torch.empty(rows, 2, dtype=torch.long, device=device)
    pair_players[:, 0] = 3
    pair_players[:, 1] = 5
    projected_env = _project_live_pair_env(env, row_indices, pair_players)
    pair_beliefs = beliefs.gather(
        1,
        pair_players[:, :, None].expand(-1, 2, PREFLOP_HANDS),
    )
    encoder = target_model.create_feature_encoder(
        env=projected_env,
        device=device,
        dtype=torch.float32,
    )
    features = encoder.encode(pair_beliefs, pre_chance_node=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with autocast:
        target_output = target_model(
            features,
            include_policy=False,
            apply_zero_sum=False,
        )
    pair_values = target_output.hand_values.to(torch.float32)
    out[:, 3, :] = pair_values[:, 0, :]
    out[:, 5, :] = pair_values[:, 1, :]
    return out


@torch.no_grad()
def _student_raw_values(
    trainer: RebelCFRTrainer,
    value_model: torch.nn.Module,
    env: PBSEnv,
    beliefs: torch.Tensor,
) -> torch.Tensor:
    encoder = value_model.create_feature_encoder(
        env=env,
        device=env.device,
        dtype=trainer.float_dtype,
    )
    features = encoder.encode(beliefs, pre_chance_node=True)
    with trainer.model_autocast():
        output = trainer.model(
            features,
            include_policy=False,
            apply_zero_sum=False,
        )
    return output.hand_values.to(torch.float32)


def _postprocess(
    raw_values: torch.Tensor,
    env: PBSEnv,
    beliefs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(raw_values.shape[0])
    live = (~env.has_folded[:rows]).to(torch.bool)
    scale = env.scale[:rows].to(torch.float32).clamp_min(1.0)
    stack_value = (
        env.stacks[:rows].to(torch.float32)
        - env.starting_stacks[:rows].to(torch.float32)
    ) / scale[:, None]
    values = torch.where(live[:, :, None], raw_values, stack_value[:, :, None])
    live_beliefs = torch.where(live[:, :, None], beliefs, torch.zeros_like(beliefs))
    denom = live_beliefs.sum(dim=(1, 2)).clamp_min(1.0e-12)
    correction = (values * live_beliefs).sum(dim=(1, 2)) / denom
    processed = torch.where(live[:, :, None], values - correction[:, None, None], values)
    return processed, correction


def _belief_stats(beliefs: torch.Tensor, *, player: int) -> dict[str, float]:
    bb = beliefs[:, player, :]
    aa_idx = _hand_name_to_idx("AA")
    max_class = bb.max(dim=-1).values
    entropy = -(bb * bb.clamp_min(1.0e-12).log()).sum(dim=-1)
    return {
        "aa_mass_mean": float(bb[:, aa_idx].mean().item()),
        "aa_mass_max": float(bb[:, aa_idx].max().item()),
        "max_class_mean": float(max_class.mean().item()),
        "max_class_p95": float(max_class.quantile(0.95).item()),
        "max_class_max": float(max_class.max().item()),
        "entropy_mean": float(entropy.mean().item()),
    }


def _summarize_node(
    *,
    node_slice: slice,
    beliefs: torch.Tensor,
    target_raw: torch.Tensor,
    student_raw: torch.Tensor,
    target_post: torch.Tensor,
    student_post: torch.Tensor,
    correction_target: torch.Tensor,
    correction_student: torch.Tensor,
    prior: torch.Tensor,
) -> dict[str, Any]:
    actor = 3
    raw_diff = student_raw[node_slice, actor] - target_raw[node_slice, actor]
    post_diff = student_post[node_slice, actor] - target_post[node_slice, actor]
    raw_abs = raw_diff.abs()
    post_abs = post_diff.abs()
    raw_weighted = (raw_abs * prior).sum(dim=-1)
    post_weighted = (post_abs * prior).sum(dim=-1)
    raw_bias = (raw_diff * prior).sum(dim=-1)
    post_bias = (post_diff * prior).sum(dim=-1)
    selected: dict[str, Any] = {}
    for hand in SELECTED_HANDS:
        idx = _hand_name_to_idx(hand)
        t_raw = target_raw[node_slice, actor, idx]
        s_raw = student_raw[node_slice, actor, idx]
        t_post = target_post[node_slice, actor, idx]
        s_post = student_post[node_slice, actor, idx]
        selected[hand] = {
            "target_raw_mean": float(t_raw.mean().item()),
            "student_raw_mean": float(s_raw.mean().item()),
            "raw_diff_mean": float((s_raw - t_raw).mean().item()),
            "target_post_mean": float(t_post.mean().item()),
            "student_post_mean": float(s_post.mean().item()),
            "post_diff_mean": float((s_post - t_post).mean().item()),
            "post_abs_diff_mean": float((s_post - t_post).abs().mean().item()),
        }
    return {
        "samples": node_slice.stop - node_slice.start,
        "belief_stats_bb": _belief_stats(beliefs[node_slice], player=5),
        "target_correction_mean": float(correction_target[node_slice].mean().item()),
        "student_correction_mean": float(correction_student[node_slice].mean().item()),
        "raw_weighted_mae_mean": float(raw_weighted.mean().item()),
        "raw_weighted_mae_p95": float(raw_weighted.quantile(0.95).item()),
        "raw_weighted_bias_mean": float(raw_bias.mean().item()),
        "post_weighted_mae_mean": float(post_weighted.mean().item()),
        "post_weighted_mae_p95": float(post_weighted.quantile(0.95).item()),
        "post_weighted_bias_mean": float(post_bias.mean().item()),
        "selected_hands": selected,
    }


def _run_belief_set(
    *,
    name: str,
    mode: str,
    samples_per_node: int,
    snapshot: dict[str, Any],
    student_trainer: RebelCFRTrainer,
    student_value_model: torch.nn.Module,
    target_model: torch.nn.Module,
    device: torch.device,
    rng: torch.Generator,
) -> dict[str, Any]:
    node_ids = torch.tensor(
        [spec["node"] for spec in NODE_SPECS.values() for _ in range(samples_per_node)],
        dtype=torch.long,
        device=device,
    )
    env = _env_from_snapshot(snapshot, node_ids=node_ids, device=device)
    beliefs = _random_beliefs(
        int(node_ids.numel()),
        6,
        device=device,
        rng=rng,
        mode=mode,
    )
    target_raw = _teacher_raw_values(target_model, env, beliefs)
    student_raw = _student_raw_values(student_trainer, student_value_model, env, beliefs)
    target_post, target_correction = _postprocess(target_raw, env, beliefs)
    student_post, student_correction = _postprocess(student_raw, env, beliefs)
    prior = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    prior = prior / prior.sum().clamp_min(1.0)

    nodes: dict[str, Any] = {}
    start = 0
    for node_name, spec in NODE_SPECS.items():
        node_slice = slice(start, start + samples_per_node)
        nodes[node_name] = {
            "node": int(spec["node"]),
            "path": str(spec["path"]),
            **_summarize_node(
                node_slice=node_slice,
                beliefs=beliefs,
                target_raw=target_raw,
                student_raw=student_raw,
                target_post=target_post,
                student_post=student_post,
                correction_target=target_correction,
                correction_student=student_correction,
                prior=prior,
            ),
        }
        start += samples_per_node
    return {
        "belief_set": name,
        "belief_mode": mode,
        "samples_per_node": samples_per_node,
        "nodes": nodes,
    }


def _print_summary(results: dict[str, Any]) -> None:
    for belief_name, belief_result in results["belief_sets"].items():
        print(f"\n=== {belief_name} ===")
        for node_name, node in belief_result["nodes"].items():
            print(
                f"{node_name}: post_mae={node['post_weighted_mae_mean']:.6f} "
                f"raw_mae={node['raw_weighted_mae_mean']:.6f} "
                f"BB max={node['belief_stats_bb']['max_class_mean']:.6f} "
                f"BB AA={node['belief_stats_bb']['aa_mass_mean']:.6f}"
            )
            for hand in ("AA", "KK", "QQ", "AKs", "KQs"):
                item = node["selected_hands"][hand]
                print(
                    f"  {hand}: target_post={item['target_post_mean']:.6f} "
                    f"student_post={item['student_post_mean']:.6f} "
                    f"diff={item['post_diff_mean']:+.6f}"
                )


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    snapshot_path = Path(args.snapshot)
    student_path = Path(args.student_checkpoint)
    target_path = Path(args.target_checkpoint)
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    batch_size = max(2, int(args.random_samples) * len(NODE_SPECS))
    student_trainer, student_value_model = _load_student(
        student_path,
        device=device,
        batch_size=batch_size,
    )
    target_model = _load_teacher(target_path, device=device, batch_size=batch_size)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(args.seed))
    results = {
        "snapshot": str(snapshot_path.resolve()),
        "student_checkpoint": str(student_path.resolve()),
        "target_checkpoint": str(target_path.resolve()),
        "seed": int(args.seed),
        "belief_sets": {
            "uniform": _run_belief_set(
                name="uniform",
                mode="uniform",
                samples_per_node=1,
                snapshot=snapshot,
                student_trainer=student_trainer,
                student_value_model=student_value_model,
                target_model=target_model,
                device=device,
                rng=rng,
            ),
            "random": _run_belief_set(
                name="random",
                mode="random",
                samples_per_node=int(args.random_samples),
                snapshot=snapshot,
                student_trainer=student_trainer,
                student_value_model=student_value_model,
                target_model=target_model,
                device=device,
                rng=rng,
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    _print_summary(results)
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
