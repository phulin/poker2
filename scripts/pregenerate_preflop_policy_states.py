#!/usr/bin/env python3
"""Sample compact multiplayer preflop public states from a saved policy head.

This is intentionally not a solved ReBeL target generator. It rolls PBSEnv
forward with a checkpointed policy head, records nonterminal preflop public
betting states before each sampled action, and writes compact sharded tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import NonlinearityType
from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    preflop_class_multiplicity_tensor,
)
from p2.env.pbs_env import PBSEnv
from p2.models.mlp.better_ffn import (
    BetterPreflopPolicyFFN,
    BetterPreflopTransformerPolicyFFN,
)


DEFAULT_CHECKPOINT = (
    "/home/user/poker2/checkpoints-rebel-curriculum-preflop_2000_p6_lr0p01_"
    "backupcons_actor_lam01_rb32_from2p_norb/preflop/rebel_latest.pt"
)

STATE_FIELDS = (
    "button",
    "street",
    "to_act",
    "last_to_act",
    "pot",
    "min_raise",
    "last_aggressive_amount",
    "actions_this_round",
    "actions_last_round",
    "stacks",
    "starting_stacks",
    "scale",
    "committed",
    "chips_placed",
    "has_folded",
    "is_allin",
    "acted_this_round",
    "done",
    "winner",
    "winners",
    "board_indices",
    "last_board_indices",
    "deck_pos",
)

INT16_FIELDS = {
    "button",
    "street",
    "to_act",
    "last_to_act",
    "actions_this_round",
    "actions_last_round",
    "winner",
    "board_indices",
    "last_board_indices",
    "deck_pos",
}
INT32_FIELDS = {
    "pot",
    "min_raise",
    "last_aggressive_amount",
    "stacks",
    "starting_stacks",
    "committed",
    "chips_placed",
}


@dataclass(frozen=True)
class BucketSpec:
    label: str
    low: int
    high: int


BUCKET_SPECS = (
    BucketSpec("actions_0_3", 0, 3),
    BucketSpec("actions_4_7", 4, 7),
    BucketSpec("actions_8_11", 8, 11),
    BucketSpec("actions_12_15", 12, 15),
)
FRONTIER_ACTION_COUNTS = (0, 4, 8, 12)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return _jsonable(value.value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    return torch.device(name)


def _model_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported model dtype: {name}")


def _enum_nonlinearity(value: Any) -> NonlinearityType:
    if isinstance(value, NonlinearityType):
        return value
    return NonlinearityType(str(value))


def _load_policy_model(
    checkpoint_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    compile_model: bool,
    strict: bool,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = dict(checkpoint.get("config", {}).get("model", {}))
    env_cfg = dict(checkpoint.get("config", {}).get("env", {}))
    num_players = int(env_cfg.get("num_players", 2))
    bet_bins = list(env_cfg.get("bet_bins", [0.5, 0.75, 1.0, 1.5, 2.0]))
    num_actions = int(model_cfg.get("num_actions", len(bet_bins) + 3))
    if num_actions <= 0:
        num_actions = len(bet_bins) + 3

    if int(model_cfg.get("preflop_hand_dim", NUM_HANDS)) != PREFLOP_HANDS:
        raise ValueError(
            "this generator requires a compact 169-hand preflop policy checkpoint"
        )

    common = dict(
        num_actions=num_actions,
        hidden_dim=int(model_cfg.get("hidden_dim", 1024)),
        range_hidden_dim=int(model_cfg.get("range_hidden_dim", 256)),
        ffn_dim=int(model_cfg.get("ffn_dim", 1024)),
        num_hidden_layers=int(model_cfg.get("num_hidden_layers", 3)),
        num_policy_layers=int(model_cfg.get("num_policy_layers", 3)),
        num_value_layers=int(model_cfg.get("num_value_layers", 3)),
        num_players=num_players,
        shared_trunk=bool(model_cfg.get("shared_trunk", True)),
        enforce_zero_sum=bool(model_cfg.get("enforce_zero_sum", True)),
        board_interaction_dim=int(model_cfg.get("board_interaction_dim", 0)),
        policy_rank=int(model_cfg.get("policy_rank", 64)),
        policy_hand_bias_rank=int(model_cfg.get("policy_hand_bias_rank", 32)),
        nonlinearity=_enum_nonlinearity(
            model_cfg.get("nonlinearity", NonlinearityType.leaky_relu)
        ),
        legacy_context_features=bool(model_cfg.get("legacy_context_features", False)),
    )
    model_type = str(model_cfg.get("preflop_model_type", "ffn")).lower()
    if model_type in {"transformer", "tokens", "player_transformer"}:
        model = BetterPreflopTransformerPolicyFFN(
            transformer_heads=int(model_cfg.get("preflop_transformer_heads", 8)),
            **common,
        )
    elif model_type in {"ffn", "mlp", "compact"}:
        model = BetterPreflopPolicyFFN(**common)
    else:
        raise ValueError(f"unsupported preflop_model_type: {model_type!r}")

    full_state = checkpoint["model"]
    policy_state = {
        key.removeprefix("policy_model."): value
        for key, value in full_state.items()
        if key.startswith("policy_model.")
    }
    if not policy_state:
        policy_state = full_state
    model.load_state_dict(policy_state, strict=strict)
    model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    if compile_model:
        model.compile_forward_modes(mode="reduce-overhead", fullgraph=True)

    metadata = {
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_wandb_run_id": checkpoint.get("wandb_run_id"),
        "checkpoint_config": checkpoint.get("config", {}),
        "model_config": model_cfg,
        "env_config": env_cfg,
        "model_type": model_type,
        "num_actions": num_actions,
        "num_players": num_players,
        "bet_bins": bet_bins,
    }
    return model, metadata


def _make_env(
    metadata: dict[str, Any],
    *,
    num_envs: int,
    device: torch.device,
    seed: int,
) -> PBSEnv:
    env_cfg = metadata["env_config"]
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    env = PBSEnv(
        num_envs=num_envs,
        num_players=int(metadata["num_players"]),
        mean_stack=int(env_cfg.get("stack", 10000)),
        sb=int(env_cfg.get("sb", 50)),
        bb=int(env_cfg.get("bb", 100)),
        default_bet_bins=list(metadata["bet_bins"]),
        device=device,
        rng=rng,
        float_dtype=torch.float32,
        stack_mode=str(env_cfg.get("stack_mode", "fixed")),
        min_stack_bb=int(env_cfg.get("min_stack_bb", 10)),
        mid_stack_bb=int(env_cfg.get("mid_stack_bb", 200)),
        max_stack_bb=int(env_cfg.get("max_stack_bb", 400)),
        high_stack_mass_ratio=float(env_cfg.get("high_stack_mass_ratio", 1.0 / 3.0)),
        force_heads_up_preflop_flop=True,
    )
    env.reset()
    return env


def _initial_beliefs(
    num_envs: int,
    num_players: int,
    *,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    if mode == "combo_uniform":
        base = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
        base = base / base.sum().clamp_min(1.0)
    elif mode == "class_uniform":
        base = torch.full(
            (PREFLOP_HANDS,),
            1.0 / float(PREFLOP_HANDS),
            dtype=torch.float32,
            device=device,
        )
    else:
        raise ValueError(f"unsupported belief init mode: {mode}")
    return base.view(1, 1, PREFLOP_HANDS).expand(
        num_envs, num_players, PREFLOP_HANDS
    ).clone()


def _compact_state_tensor(name: str, tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bool:
        return tensor.contiguous()
    if name in INT16_FIELDS:
        return tensor.to(torch.int16).contiguous()
    if name in INT32_FIELDS:
        return tensor.to(torch.int32).contiguous()
    if tensor.dtype.is_floating_point:
        return tensor.to(torch.float32).contiguous()
    return tensor.contiguous()


class ShardWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        shard_size: int,
        overwrite: bool,
    ) -> None:
        self.output_dir = output_dir
        self.shard_size = int(shard_size)
        self.overwrite = bool(overwrite)
        self.pending: dict[str, list[torch.Tensor]] = {name: [] for name in STATE_FIELDS}
        self.pending_count = 0
        self.total_rows = 0
        self.shard_index = 0
        self.shards: list[dict[str, Any]] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def append(self, env: PBSEnv, indices: torch.Tensor) -> None:
        rows = int(indices.numel())
        if rows <= 0:
            return
        for name in STATE_FIELDS:
            self.pending[name].append(
                _compact_state_tensor(name, getattr(env, name)[indices])
            )
        self.pending_count += rows

    @property
    def rows(self) -> int:
        return self.total_rows + self.pending_count

    def flush(self) -> dict[str, Any] | None:
        if self.pending_count <= 0:
            return None
        states = {
            name: torch.cat(chunks, dim=0)
            for name, chunks in self.pending.items()
            if chunks
        }
        rows = int(next(iter(states.values())).shape[0])
        shard_name = f"states_{self.shard_index:06d}.pt"
        shard_path = self.output_dir / shard_name
        if shard_path.exists() and not self.overwrite:
            raise FileExistsError(
                f"{shard_path} exists; choose a new output dir or pass --overwrite"
            )
        payload = {
            "format_version": 1,
            "kind": "preflop_policy_public_states_shard",
            "shard_index": self.shard_index,
            "num_rows": rows,
            "states": states,
        }
        torch.save(payload, shard_path)
        info = {"path": shard_name, "num_rows": rows}
        self.shards.append(info)
        self.total_rows += rows
        self.shard_index += 1
        self.pending = {name: [] for name in STATE_FIELDS}
        self.pending_count = 0
        return info


class FrontierPool:
    """Fixed-capacity GPU pool of public states plus internal rollout beliefs."""

    def __init__(
        self,
        *,
        env: PBSEnv,
        beliefs: torch.Tensor,
        capacity: int,
    ) -> None:
        self.capacity = int(capacity)
        self.count = 0
        self.dropped = 0
        self.device = env.device
        self.data = {
            name: torch.empty(
                (self.capacity, *getattr(env, name).shape[1:]),
                dtype=getattr(env, name).dtype,
                device=self.device,
            )
            for name in STATE_FIELDS
        }
        self.beliefs = torch.empty(
            (self.capacity, *beliefs.shape[1:]),
            dtype=beliefs.dtype,
            device=beliefs.device,
        )

    def append(self, env: PBSEnv, beliefs: torch.Tensor, indices: torch.Tensor) -> int:
        rows = int(indices.numel())
        if rows <= 0:
            return 0
        remaining = self.capacity - self.count
        take = min(rows, remaining)
        if take <= 0:
            self.dropped += rows
            return 0
        src = indices[:take]
        dst = slice(self.count, self.count + take)
        for name in STATE_FIELDS:
            self.data[name][dst] = getattr(env, name)[src]
        self.beliefs[dst] = beliefs[src]
        self.count += take
        self.dropped += rows - take
        return take

    def sample_into(
        self,
        env: PBSEnv,
        beliefs: torch.Tensor,
        dest: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> None:
        rows = int(dest.numel())
        if rows <= 0:
            return
        if self.count <= 0:
            raise RuntimeError("cannot sample from an empty frontier")
        src = torch.randint(
            0,
            self.count,
            (rows,),
            generator=generator,
            device=self.device,
        )
        for name in STATE_FIELDS:
            getattr(env, name)[dest] = self.data[name][src]
        beliefs[dest] = self.beliefs[src]
        env.board_onehot[dest] = False
        env.deck[dest] = 0

    def copy_indices_into(
        self,
        env: PBSEnv,
        beliefs: torch.Tensor,
        src: torch.Tensor,
        dest: torch.Tensor,
    ) -> None:
        rows = int(dest.numel())
        if rows <= 0:
            return
        if src.shape != dest.shape:
            raise ValueError("src and dest must have matching shapes")
        for name in STATE_FIELDS:
            getattr(env, name)[dest] = self.data[name][src]
        beliefs[dest] = self.beliefs[src]
        env.board_onehot[dest] = False
        env.deck[dest] = 0


def _policy_step(
    *,
    policy_model: torch.nn.Module,
    encoder: Any,
    env: PBSEnv,
    beliefs: torch.Tensor,
    rng: torch.Generator,
    args: argparse.Namespace,
    device: torch.device,
) -> bool:
    bin_amounts, legal = env.legal_bins_amounts_and_mask()
    active = torch.where((env.street == 0) & ~env.done)[0]
    if active.numel() == 0:
        return False

    legal_active = legal[active]
    features = encoder.encode(beliefs, indices=active)
    logits = policy_model._call_forward_policy(features).float()
    if args.temperature != 1.0:
        logits = logits / max(float(args.temperature), 1.0e-6)
    logits = logits.masked_fill(~legal_active[:, None, :], -1.0e9)
    hand_action_probs = torch.softmax(logits, dim=-1)

    actor = env.to_act[active].long()
    actor_beliefs = beliefs[active, actor]
    marginal = (hand_action_probs * actor_beliefs[:, :, None]).sum(dim=1)
    marginal = marginal * legal_active.to(marginal.dtype)
    legal_count = legal_active.sum(dim=1, keepdim=True).clamp_min(1)
    fallback = legal_active.to(marginal.dtype) / legal_count.to(marginal.dtype)
    marginal_sum = marginal.sum(dim=1, keepdim=True)
    marginal = torch.where(
        marginal_sum > 1.0e-12,
        marginal / marginal_sum.clamp_min(1.0e-12),
        fallback,
    )
    if args.epsilon > 0.0:
        eps = float(args.epsilon)
        marginal = (1.0 - eps) * marginal + eps * fallback
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    sampled_bins = torch.multinomial(
        marginal.float(),
        num_samples=1,
        replacement=True,
        generator=rng,
    ).squeeze(1)

    selected_hand_probs = hand_action_probs.gather(
        2,
        sampled_bins[:, None, None].expand(-1, PREFLOP_HANDS, 1),
    ).squeeze(2)
    updated_actor_beliefs = actor_beliefs * selected_hand_probs
    updated_sum = updated_actor_beliefs.sum(dim=1, keepdim=True)
    updated_actor_beliefs = torch.where(
        updated_sum > 1.0e-12,
        updated_actor_beliefs / updated_sum.clamp_min(1.0e-12),
        actor_beliefs,
    )
    beliefs[active, actor] = updated_actor_beliefs

    all_bins = torch.full((env.N,), -1, dtype=torch.long, device=device)
    all_bins[active] = sampled_bins
    env.step_bins(all_bins, bin_amounts=bin_amounts, legal_masks=legal)
    return True


def _write_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    writer: ShardWriter,
    complete: bool,
    elapsed_s: float,
) -> None:
    manifest = {
        "format_version": 1,
        "kind": "preflop_policy_public_states",
        "complete": complete,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed_s,
        "source_checkpoint": os.path.realpath(args.checkpoint),
        "source_wandb_run_id": metadata.get("checkpoint_wandb_run_id"),
        "source_step": metadata.get("checkpoint_step"),
        "num_rows": writer.total_rows + writer.pending_count,
        "state_fields": list(STATE_FIELDS),
        "state_dtype_policy": {
            "int16": sorted(INT16_FIELDS),
            "int32": sorted(INT32_FIELDS),
            "bool": [
                "has_folded",
                "is_allin",
                "acted_this_round",
                "done",
                "winners",
            ],
            "float32": ["scale"],
        },
        "shards": list(writer.shards),
        "rollout": {
            "num_envs": args.num_envs,
            "target_states": args.num_states,
            "shard_size": args.shard_size,
            "seed": args.seed,
            "belief_init": args.belief_init,
            "belief_update": "actor_bayes_update_from_sampled_legal_action",
            "action_sampling": "actor_belief_weighted_policy_probs",
            "temperature": args.temperature,
            "epsilon": args.epsilon,
            "max_actions_this_round": args.max_actions_this_round,
            "min_actions_this_round": args.min_actions_this_round,
        },
        "model": {
            "model_type": metadata.get("model_type"),
            "num_actions": metadata.get("num_actions"),
            "num_players": metadata.get("num_players"),
            "model_dtype": args.model_dtype,
            "compile": args.compile,
        },
        "env_config": metadata.get("env_config"),
        "model_config": metadata.get("model_config"),
    }
    path = output_dir / ("manifest.json" if complete else "manifest.partial.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)


def _write_stratified_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    writers: list[ShardWriter],
    frontiers: dict[int, FrontierPool],
    complete: bool,
    elapsed_s: float,
    stages_completed: int,
) -> None:
    buckets = []
    for spec, writer in zip(BUCKET_SPECS, writers, strict=True):
        buckets.append(
            {
                "label": spec.label,
                "low": spec.low,
                "high": spec.high,
                "target_rows": args.bucket_target,
                "num_rows": writer.rows,
                "shards": list(writer.shards),
            }
        )
    manifest = {
        "format_version": 1,
        "kind": "preflop_policy_public_states_stratified",
        "complete": complete,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed_s,
        "source_checkpoint": os.path.realpath(args.checkpoint),
        "source_wandb_run_id": metadata.get("checkpoint_wandb_run_id"),
        "source_step": metadata.get("checkpoint_step"),
        "num_rows": sum(writer.rows for writer in writers),
        "state_fields": list(STATE_FIELDS),
        "state_dtype_policy": {
            "int16": sorted(INT16_FIELDS),
            "int32": sorted(INT32_FIELDS),
            "bool": [
                "has_folded",
                "is_allin",
                "acted_this_round",
                "done",
                "winners",
            ],
            "float32": ["scale"],
        },
        "rollout": {
            "mode": "stratified_frontier",
            "num_envs": args.num_envs,
            "bucket_target": args.bucket_target,
            "total_target_states": args.bucket_target * len(BUCKET_SPECS),
            "shard_size": args.shard_size,
            "seed": args.seed,
            "belief_init": args.belief_init,
            "belief_update": "actor_bayes_update_from_sampled_legal_action",
            "action_sampling": "actor_belief_weighted_policy_probs",
            "temperature": args.temperature,
            "epsilon": args.epsilon,
            "frontier_capacity": args.frontier_capacity,
            "frontier_min_states": args.frontier_min_states,
        },
        "buckets": buckets,
        "frontiers": {
            str(action_count): {
                "count": pool.count,
                "capacity": pool.capacity,
                "dropped": pool.dropped,
            }
            for action_count, pool in sorted(frontiers.items())
        },
        "stages_completed": stages_completed,
        "model": {
            "model_type": metadata.get("model_type"),
            "num_actions": metadata.get("num_actions"),
            "num_players": metadata.get("num_players"),
            "model_dtype": args.model_dtype,
            "compile": args.compile,
        },
        "env_config": metadata.get("env_config"),
        "model_config": metadata.get("model_config"),
    }
    path = output_dir / ("manifest.json" if complete else "manifest.partial.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)


def _write_unique_frontier_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    writers: dict[int, ShardWriter],
    complete: bool,
    elapsed_s: float,
    root_states_sampled: int,
) -> None:
    frontiers = []
    for action_count in FRONTIER_ACTION_COUNTS:
        writer = writers[action_count]
        frontiers.append(
            {
                "action_count": action_count,
                "target_rows": (
                    args.frontier_result_cap
                    if args.frontier_result_cap > 0
                    else args.root_states
                ),
                "num_rows": writer.rows,
                "shards": list(writer.shards),
            }
        )
    manifest = {
        "format_version": 1,
        "kind": "preflop_policy_unique_frontier_starts",
        "complete": complete,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed_s,
        "source_checkpoint": os.path.realpath(args.checkpoint),
        "source_wandb_run_id": metadata.get("checkpoint_wandb_run_id"),
        "source_step": metadata.get("checkpoint_step"),
        "num_rows": sum(writer.rows for writer in writers.values()),
        "root_states_sampled": root_states_sampled,
        "root_states": args.root_states,
        "state_fields": list(STATE_FIELDS),
        "state_dtype_policy": {
            "int16": sorted(INT16_FIELDS),
            "int32": sorted(INT32_FIELDS),
            "bool": [
                "has_folded",
                "is_allin",
                "acted_this_round",
                "done",
                "winners",
            ],
            "float32": ["scale"],
        },
        "rollout": {
            "mode": (
                "unique_frontier_streaming"
                if args.unique_frontier_streaming
                else "unique_frontier_starts"
            ),
            "num_envs": args.num_envs,
            "frontier_action_counts": list(FRONTIER_ACTION_COUNTS),
            "frontier_start_target": args.frontier_start_target,
            "frontier_result_cap": args.frontier_result_cap,
            "frontier_attempts": args.frontier_attempts,
            "seed": args.seed,
            "belief_init": args.belief_init,
            "belief_update": "actor_bayes_update_from_sampled_legal_action",
            "action_sampling": "actor_belief_weighted_policy_probs",
            "temperature": args.temperature,
            "epsilon": args.epsilon,
            "uniqueness": (
                "single policy trajectory per sampled root; each trajectory can "
                "contribute at most one state to each frontier action count"
            ),
        },
        "frontiers": frontiers,
        "model": {
            "model_type": metadata.get("model_type"),
            "num_actions": metadata.get("num_actions"),
            "num_players": metadata.get("num_players"),
            "model_dtype": args.model_dtype,
            "compile": args.compile,
        },
        "env_config": metadata.get("env_config"),
        "model_config": metadata.get("model_config"),
    }
    path = output_dir / ("manifest.json" if complete else "manifest.partial.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)


def generate(args: argparse.Namespace) -> None:
    if args.unique_frontier:
        if args.unique_frontier_buckets:
            generate_unique_frontier_buckets_streaming(args)
            return
        if args.unique_frontier_streaming:
            generate_unique_frontier_streaming(args)
            return
        generate_unique_frontier(args)
        return
    if args.stratified:
        generate_stratified(args)
        return

    device = _device(args.device)
    dtype = _model_dtype(args.model_dtype, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; choose a new output dir or pass --overwrite"
        )

    policy_model, metadata = _load_policy_model(
        args.checkpoint,
        device=device,
        dtype=dtype,
        compile_model=args.compile,
        strict=not args.non_strict,
    )
    env = _make_env(metadata, num_envs=args.num_envs, device=device, seed=args.seed)
    encoder = policy_model.create_feature_encoder(env, device=device, dtype=dtype)
    beliefs = _initial_beliefs(
        args.num_envs,
        int(metadata["num_players"]),
        device=device,
        mode=args.belief_init,
    )
    uniform_belief = beliefs[0].clone()
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 1)
    writer = ShardWriter(
        output_dir,
        shard_size=args.shard_size,
        overwrite=args.overwrite,
    )

    started = time.time()
    last_progress = started
    steps = 0
    print(
        "Starting policy state generation: "
        f"target={args.num_states:,} num_envs={args.num_envs:,} "
        f"device={device} dtype={dtype} checkpoint_step={metadata['checkpoint_step']}"
    )
    print(f"Output: {output_dir}")

    with torch.inference_mode():
        while writer.total_rows + writer.pending_count < args.num_states:
            active = torch.where((env.street == 0) & ~env.done)[0]
            if active.numel() == 0:
                env.reset()
                beliefs[:] = uniform_belief
                continue

            record = active
            if args.min_actions_this_round > 0:
                record = record[
                    env.actions_this_round[record] >= args.min_actions_this_round
                ]
            remaining_goal = args.num_states - writer.total_rows - writer.pending_count
            remaining_shard = args.shard_size - writer.pending_count
            take = min(int(record.numel()), remaining_goal, remaining_shard)
            if take > 0:
                writer.append(env, record[:take])
                if writer.pending_count >= args.shard_size:
                    info = writer.flush()
                    elapsed = time.time() - started
                    _write_manifest(
                        output_dir,
                        args=args,
                        metadata=metadata,
                        writer=writer,
                        complete=False,
                        elapsed_s=elapsed,
                    )
                    assert info is not None
                    print(
                        f"wrote {info['path']} rows={info['num_rows']:,} "
                        f"total={writer.total_rows:,} elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            stepped = _policy_step(
                policy_model=policy_model,
                encoder=encoder,
                env=env,
                beliefs=beliefs,
                rng=rng,
                args=args,
                device=device,
            )
            if not stepped:
                continue
            steps += 1

            reset = (env.street != 0) | env.done
            if args.max_actions_this_round > 0:
                reset |= env.actions_this_round >= args.max_actions_this_round
            reset_idx = torch.where(reset)[0]
            if reset_idx.numel() > 0:
                env.reset(reset_idx)
                beliefs[reset_idx] = uniform_belief

            now = time.time()
            if now - last_progress >= args.progress_interval_s:
                elapsed = now - started
                total = writer.total_rows + writer.pending_count
                rate = total / max(elapsed, 1.0e-9)
                print(
                    f"progress rows={total:,}/{args.num_states:,} "
                    f"rollout_steps={steps:,} rate={rate:,.0f}/s "
                    f"pending={writer.pending_count:,}",
                    flush=True,
                )
                last_progress = now

    final_info = writer.flush()
    elapsed = time.time() - started
    _write_manifest(
        output_dir,
        args=args,
        metadata=metadata,
        writer=writer,
        complete=True,
        elapsed_s=elapsed,
    )
    if final_info is not None:
        print(
            f"wrote {final_info['path']} rows={final_info['num_rows']:,} "
            f"total={writer.total_rows:,}"
        )
    print(
        f"Done: rows={writer.total_rows:,} shards={len(writer.shards)} "
        f"elapsed={elapsed:.1f}s output={output_dir}"
    )


def _refill_from_stage_source(
    *,
    stage: int,
    env: PBSEnv,
    beliefs: torch.Tensor,
    rows: torch.Tensor,
    uniform_belief: torch.Tensor,
    frontiers: dict[int, FrontierPool],
    rng: torch.Generator,
) -> None:
    if rows.numel() == 0:
        return
    if stage == 0:
        env.reset(rows)
        beliefs[rows] = uniform_belief
        return
    source_count = BUCKET_SPECS[stage].low
    frontiers[source_count].sample_into(env, beliefs, rows, generator=rng)


def generate_stratified(args: argparse.Namespace) -> None:
    device = _device(args.device)
    dtype = _model_dtype(args.model_dtype, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; choose a new output dir or pass --overwrite"
        )

    policy_model, metadata = _load_policy_model(
        args.checkpoint,
        device=device,
        dtype=dtype,
        compile_model=args.compile,
        strict=not args.non_strict,
    )
    env = _make_env(metadata, num_envs=args.num_envs, device=device, seed=args.seed)
    encoder = policy_model.create_feature_encoder(env, device=device, dtype=dtype)
    beliefs = _initial_beliefs(
        args.num_envs,
        int(metadata["num_players"]),
        device=device,
        mode=args.belief_init,
    )
    uniform_belief = beliefs[0].clone()
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 1)
    writers = [
        ShardWriter(
            output_dir / spec.label,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
        )
        for spec in BUCKET_SPECS
    ]
    frontiers = {
        spec.low: FrontierPool(
            env=env,
            beliefs=beliefs,
            capacity=args.frontier_capacity,
        )
        for spec in BUCKET_SPECS[1:]
    }

    started = time.time()
    last_progress = started
    steps = 0
    stages_completed = 0
    print(
        "Starting stratified policy state generation: "
        f"target_per_bucket={args.bucket_target:,} num_envs={args.num_envs:,} "
        f"device={device} dtype={dtype} checkpoint_step={metadata['checkpoint_step']}"
    )
    print(f"Output: {output_dir}")

    with torch.inference_mode():
        for stage, spec in enumerate(BUCKET_SPECS):
            writer = writers[stage]
            next_frontier_count = spec.high + 1
            next_frontier = frontiers.get(next_frontier_count)
            required_next_frontier = (
                0
                if next_frontier is None
                else min(args.frontier_min_states, next_frontier.capacity)
            )
            _refill_from_stage_source(
                stage=stage,
                env=env,
                beliefs=beliefs,
                rows=env.arange_n,
                uniform_belief=uniform_belief,
                frontiers=frontiers,
                rng=rng,
            )
            print(
                f"stage {stage + 1}/4 {spec.label}: target={args.bucket_target:,} "
                f"next_frontier_min={required_next_frontier:,}",
                flush=True,
            )

            while writer.rows < args.bucket_target or (
                next_frontier is not None
                and next_frontier.count < required_next_frontier
            ):
                active = torch.where((env.street == 0) & ~env.done)[0]
                if active.numel() == 0:
                    _refill_from_stage_source(
                        stage=stage,
                        env=env,
                        beliefs=beliefs,
                        rows=env.arange_n,
                        uniform_belief=uniform_belief,
                        frontiers=frontiers,
                        rng=rng,
                    )
                    continue

                action_counts = env.actions_this_round[active]
                record = active[
                    (action_counts >= spec.low) & (action_counts <= spec.high)
                ]
                if record.numel() > 0 and writer.rows < args.bucket_target:
                    remaining_goal = args.bucket_target - writer.rows
                    remaining_shard = args.shard_size - writer.pending_count
                    take = min(int(record.numel()), remaining_goal, remaining_shard)
                    if take > 0:
                        writer.append(env, record[:take])
                        if writer.pending_count >= args.shard_size:
                            info = writer.flush()
                            elapsed = time.time() - started
                            _write_stratified_manifest(
                                output_dir,
                                args=args,
                                metadata=metadata,
                                writers=writers,
                                frontiers=frontiers,
                                complete=False,
                                elapsed_s=elapsed,
                                stages_completed=stages_completed,
                            )
                            assert info is not None
                            print(
                                f"wrote {spec.label}/{info['path']} "
                                f"rows={info['num_rows']:,} "
                                f"bucket_total={writer.total_rows:,} "
                                f"elapsed={elapsed:.1f}s",
                                flush=True,
                            )

                stepped = _policy_step(
                    policy_model=policy_model,
                    encoder=encoder,
                    env=env,
                    beliefs=beliefs,
                    rng=rng,
                    args=args,
                    device=device,
                )
                if not stepped:
                    continue
                steps += 1

                reset = (env.street != 0) | env.done
                if next_frontier is not None:
                    reached_next = (
                        (env.street == 0)
                        & ~env.done
                        & (env.actions_this_round == next_frontier_count)
                    )
                    next_frontier.append(env, beliefs, torch.where(reached_next)[0])
                    reset |= env.actions_this_round >= next_frontier_count
                else:
                    reset |= env.actions_this_round > spec.high
                reset_idx = torch.where(reset)[0]
                if reset_idx.numel() > 0:
                    _refill_from_stage_source(
                        stage=stage,
                        env=env,
                        beliefs=beliefs,
                        rows=reset_idx,
                        uniform_belief=uniform_belief,
                        frontiers=frontiers,
                        rng=rng,
                    )

                now = time.time()
                if now - last_progress >= args.progress_interval_s:
                    elapsed = now - started
                    bucket_rows = [writer_i.rows for writer_i in writers]
                    frontier_counts = {
                        k: pool.count for k, pool in sorted(frontiers.items())
                    }
                    rate = sum(bucket_rows) / max(elapsed, 1.0e-9)
                    print(
                        f"progress stage={spec.label} rows={bucket_rows} "
                        f"frontiers={frontier_counts} rollout_steps={steps:,} "
                        f"rate={rate:,.0f}/s",
                        flush=True,
                    )
                    last_progress = now

            final_info = writer.flush()
            stages_completed = stage + 1
            elapsed = time.time() - started
            _write_stratified_manifest(
                output_dir,
                args=args,
                metadata=metadata,
                writers=writers,
                frontiers=frontiers,
                complete=False,
                elapsed_s=elapsed,
                stages_completed=stages_completed,
            )
            if final_info is not None:
                print(
                    f"wrote {spec.label}/{final_info['path']} "
                    f"rows={final_info['num_rows']:,} "
                    f"bucket_total={writer.total_rows:,}",
                    flush=True,
                )
            print(
                f"completed {spec.label}: rows={writer.rows:,} "
                f"frontiers={{"
                + ", ".join(
                    f"{k}:{pool.count}" for k, pool in sorted(frontiers.items())
                )
                + "}",
                flush=True,
            )

    elapsed = time.time() - started
    _write_stratified_manifest(
        output_dir,
        args=args,
        metadata=metadata,
        writers=writers,
        frontiers=frontiers,
        complete=True,
        elapsed_s=elapsed,
        stages_completed=stages_completed,
    )
    print(
        f"Done: rows={sum(writer.rows for writer in writers):,} "
        f"buckets={[writer.rows for writer in writers]} "
        f"elapsed={elapsed:.1f}s output={output_dir}"
    )


def _writer_append_bounded(
    writer: ShardWriter,
    env: PBSEnv,
    indices: torch.Tensor,
    *,
    max_rows: int = 0,
) -> int:
    offset = 0
    total = int(indices.numel())
    if max_rows > 0:
        remaining_total = max_rows - writer.rows
        if remaining_total <= 0:
            return 0
        total = min(total, remaining_total)
    while offset < total:
        remaining_shard = writer.shard_size - writer.pending_count
        take = min(total - offset, remaining_shard)
        writer.append(env, indices[offset : offset + take])
        offset += take
        if writer.pending_count >= writer.shard_size:
            writer.flush()
    return offset


def _build_root_frontier(
    *,
    args: argparse.Namespace,
    env: PBSEnv,
    beliefs: torch.Tensor,
    uniform_belief: torch.Tensor,
    writer: ShardWriter | None,
    root_states: int | None = None,
    writer_cap: int = 0,
    flush_writer: bool = True,
) -> FrontierPool:
    root_states = args.root_states if root_states is None else int(root_states)
    pool = FrontierPool(env=env, beliefs=beliefs, capacity=root_states)
    created = 0
    while created < root_states:
        take = min(env.N, root_states - created)
        rows = torch.arange(take, device=env.device)
        env.reset(rows)
        beliefs[rows] = uniform_belief
        if take < env.N:
            env.done[take:] = True
        pool.append(env, beliefs, rows)
        if writer is not None:
            _writer_append_bounded(writer, env, rows, max_rows=writer_cap)
        created += take
    if writer is not None and flush_writer:
        writer.flush()
    return pool


def _advance_unique_frontier(
    *,
    source: FrontierPool,
    target_action_count: int,
    target: FrontierPool,
    writer: ShardWriter,
    policy_model: torch.nn.Module,
    encoder: Any,
    env: PBSEnv,
    beliefs: torch.Tensor,
    rng: torch.Generator,
    args: argparse.Namespace,
    device: torch.device,
    writer_cap: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    start_count = target_action_count - 4
    pending = torch.arange(source.count, dtype=torch.long, device=device)
    attempts_summary: list[dict[str, int]] = []
    rows_all = torch.arange(env.N, dtype=torch.long, device=device)

    for attempt in range(1, args.frontier_attempts + 1):
        if pending.numel() == 0:
            break
        next_pending: list[torch.Tensor] = []
        successes_this_attempt = 0
        attempted_this_attempt = int(pending.numel())

        for offset in range(0, int(pending.numel()), env.N):
            src = pending[offset : offset + env.N]
            rows = rows_all[: src.numel()]
            source.copy_indices_into(env, beliefs, src, rows)
            if src.numel() < env.N:
                env.done[src.numel() :] = True

            succeeded = torch.zeros(src.numel(), dtype=torch.bool, device=device)
            for _ in range(target_action_count - start_count):
                active = (
                    (env.street[rows] == 0)
                    & ~env.done[rows]
                    & ~succeeded
                    & (env.actions_this_round[rows] < target_action_count)
                )
                if not bool(active.any().item()):
                    break
                # Keep already-successful rows out of the shared policy step.
                if bool(succeeded.any().item()):
                    env.done[rows[succeeded]] = True
                stepped = _policy_step(
                    policy_model=policy_model,
                    encoder=encoder,
                    env=env,
                    beliefs=beliefs,
                    rng=rng,
                    args=args,
                    device=device,
                )
                if not stepped:
                    break
                reached = (
                    (env.street[rows] == 0)
                    & ~env.done[rows]
                    & (env.actions_this_round[rows] == target_action_count)
                    & ~succeeded
                )
                if bool(reached.any().item()):
                    reached_rows = rows[reached]
                    target.append(env, beliefs, reached_rows)
                    _writer_append_bounded(
                        writer,
                        env,
                        reached_rows,
                        max_rows=writer_cap,
                    )
                    succeeded |= reached
                    env.done[reached_rows] = True

            failed = ~succeeded
            if bool(failed.any().item()):
                next_pending.append(src[failed])
            successes_this_attempt += int(succeeded.sum().item())

        pending = (
            torch.cat(next_pending, dim=0)
            if next_pending
            else torch.empty(0, dtype=torch.long, device=device)
        )
        attempts_summary.append(
            {
                "attempt": attempt,
                "attempted": attempted_this_attempt,
                "successes": successes_this_attempt,
                "remaining": int(pending.numel()),
            }
        )
        if verbose:
            print(
                f"frontier {start_count}->{target_action_count} attempt {attempt}: "
                f"attempted={attempted_this_attempt:,} "
                f"successes={successes_this_attempt:,} "
                f"remaining={int(pending.numel()):,}",
                flush=True,
            )

    writer.flush()
    return {
        "source_count": source.count,
        "target_count": target.count,
        "target_action_count": target_action_count,
        "attempts": attempts_summary,
        "failed": int(pending.numel()),
    }


def generate_unique_frontier(args: argparse.Namespace) -> None:
    device = _device(args.device)
    dtype = _model_dtype(args.model_dtype, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; choose a new output dir or pass --overwrite"
        )

    policy_model, metadata = _load_policy_model(
        args.checkpoint,
        device=device,
        dtype=dtype,
        compile_model=args.compile,
        strict=not args.non_strict,
    )
    env = _make_env(metadata, num_envs=args.num_envs, device=device, seed=args.seed)
    encoder = policy_model.create_feature_encoder(env, device=device, dtype=dtype)
    beliefs = _initial_beliefs(
        args.num_envs,
        int(metadata["num_players"]),
        device=device,
        mode=args.belief_init,
    )
    uniform_belief = beliefs[0].clone()
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 1)
    writers = {
        action_count: ShardWriter(
            output_dir / f"frontier_{action_count}",
            shard_size=args.shard_size,
            overwrite=args.overwrite,
        )
        for action_count in FRONTIER_ACTION_COUNTS
    }

    started = time.time()
    print(
        "Starting unique-frontier generation: "
        f"roots={args.root_states:,} attempts={args.frontier_attempts} "
        f"num_envs={args.num_envs:,} device={device} dtype={dtype} "
        f"checkpoint_step={metadata['checkpoint_step']}"
    )
    print(f"Output: {output_dir}")
    pools: dict[int, FrontierPool] = {}
    stage_summaries: list[dict[str, Any]] = []
    with torch.inference_mode():
        pools[0] = _build_root_frontier(
            args=args,
            env=env,
            beliefs=beliefs,
            uniform_belief=uniform_belief,
            writer=writers[0],
            writer_cap=args.frontier_result_cap,
        )
        print(f"frontier 0 roots saved: {pools[0].count:,}", flush=True)
        for target_action_count in FRONTIER_ACTION_COUNTS[1:]:
            source = pools[target_action_count - 4]
            target = FrontierPool(
                env=env,
                beliefs=beliefs,
                capacity=source.count,
            )
            summary = _advance_unique_frontier(
                source=source,
                target_action_count=target_action_count,
                target=target,
                writer=writers[target_action_count],
                policy_model=policy_model,
                encoder=encoder,
                env=env,
                beliefs=beliefs,
                rng=rng,
                args=args,
                device=device,
                writer_cap=args.frontier_result_cap,
            )
            pools[target_action_count] = target
            stage_summaries.append(summary)
            elapsed = time.time() - started
            _write_unique_frontier_manifest(
                output_dir,
                args=args,
                metadata=metadata,
                writers=writers,
                complete=False,
                elapsed_s=elapsed,
                root_states_sampled=args.root_states,
            )
            print(
                f"completed frontier {target_action_count}: "
                f"{target.count:,}/{source.count:,} elapsed={elapsed:.1f}s",
                flush=True,
            )

    elapsed = time.time() - started
    _write_unique_frontier_manifest(
        output_dir,
        args=args,
        metadata=metadata,
        writers=writers,
        complete=True,
        elapsed_s=elapsed,
        root_states_sampled=args.root_states,
    )
    summary_path = output_dir / "attempt_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(stage_summaries), f, indent=2, sort_keys=True)
    print(
        "Done: frontiers={"
        + ", ".join(f"{k}:{pools[k].count}" for k in FRONTIER_ACTION_COUNTS)
        + f"}} elapsed={elapsed:.1f}s output={output_dir}"
    )


def _merge_unique_frontier_summary(
    aggregate: dict[int, dict[str, Any]], summary: dict[str, Any]
) -> None:
    target_action_count = int(summary["target_action_count"])
    entry = aggregate.setdefault(
        target_action_count,
        {
            "source_count": 0,
            "target_count": 0,
            "target_action_count": target_action_count,
            "attempts": [],
            "failed": 0,
        },
    )
    entry["source_count"] += int(summary["source_count"])
    entry["target_count"] += int(summary["target_count"])
    entry["failed"] += int(summary["failed"])
    attempts = entry["attempts"]
    for attempt_summary in summary["attempts"]:
        attempt_index = int(attempt_summary["attempt"]) - 1
        while len(attempts) <= attempt_index:
            attempts.append(
                {
                    "attempt": len(attempts) + 1,
                    "attempted": 0,
                    "successes": 0,
                    "remaining": 0,
                }
            )
        merged = attempts[attempt_index]
        merged["attempted"] += int(attempt_summary["attempted"])
        merged["successes"] += int(attempt_summary["successes"])
        merged["remaining"] += int(attempt_summary["remaining"])


def _unique_bucket_action_cap(args: argparse.Namespace) -> int:
    if args.frontier_bucket_action_cap > 0:
        return int(args.frontier_bucket_action_cap)
    if args.frontier_result_cap > 0:
        return max(1, (int(args.frontier_result_cap) + 3) // 4)
    return 0


def _append_unique_bucket_rows(
    *,
    writer: ShardWriter,
    per_action_written: dict[int, int],
    env: PBSEnv,
    rows: torch.Tensor,
    action_count: int,
    bucket_cap: int,
    action_cap: int,
) -> int:
    total = int(rows.numel())
    if total <= 0:
        return 0
    remaining_bucket = bucket_cap - writer.rows if bucket_cap > 0 else total
    remaining_action = (
        action_cap - per_action_written.get(action_count, 0)
        if action_cap > 0
        else total
    )
    take = min(total, remaining_bucket, remaining_action)
    if take <= 0:
        return 0
    written = _writer_append_bounded(
        writer,
        env,
        rows[:take],
        max_rows=bucket_cap,
    )
    per_action_written[action_count] = (
        per_action_written.get(action_count, 0) + written
    )
    return written


def _advance_unique_bucket(
    *,
    source: FrontierPool,
    bucket: BucketSpec,
    target_action_count: int,
    target: FrontierPool | None,
    writer: ShardWriter,
    per_action_written: dict[int, int],
    policy_model: torch.nn.Module,
    encoder: Any,
    env: PBSEnv,
    beliefs: torch.Tensor,
    rng: torch.Generator,
    args: argparse.Namespace,
    device: torch.device,
    bucket_cap: int,
    action_cap: int,
) -> dict[str, Any]:
    start_count = bucket.low
    width = target_action_count - start_count
    if width <= 0:
        raise ValueError("target_action_count must be greater than bucket.low")
    pending = torch.arange(source.count, dtype=torch.long, device=device)
    saved_by_source = torch.zeros(source.count, width, dtype=torch.bool, device=device)
    attempts_summary: list[dict[str, int]] = []
    rows_all = torch.arange(env.N, dtype=torch.long, device=device)

    for attempt in range(1, args.frontier_attempts + 1):
        if pending.numel() == 0:
            break
        next_pending: list[torch.Tensor] = []
        successes_this_attempt = 0
        attempted_this_attempt = int(pending.numel())

        for offset in range(0, int(pending.numel()), env.N):
            src = pending[offset : offset + env.N]
            rows = rows_all[: src.numel()]
            source.copy_indices_into(env, beliefs, src, rows)
            if src.numel() < env.N:
                env.done[src.numel() :] = True

            succeeded = torch.zeros(src.numel(), dtype=torch.bool, device=device)
            for _ in range(width):
                active = (
                    (env.street[rows] == 0)
                    & ~env.done[rows]
                    & ~succeeded
                    & (env.actions_this_round[rows] < target_action_count)
                )
                for action_count in range(bucket.low, bucket.high + 1):
                    local = action_count - start_count
                    if local < 0 or local >= width:
                        continue
                    save_pos = torch.where(
                        active
                        & (env.actions_this_round[rows] == action_count)
                        & ~saved_by_source[src, local]
                    )[0]
                    if save_pos.numel() == 0:
                        continue
                    written = _append_unique_bucket_rows(
                        writer=writer,
                        per_action_written=per_action_written,
                        env=env,
                        rows=rows[save_pos],
                        action_count=action_count,
                        bucket_cap=bucket_cap,
                        action_cap=action_cap,
                    )
                    if written > 0:
                        saved_by_source[src[save_pos[:written]], local] = True

                if not bool(active.any().item()):
                    break
                if bool(succeeded.any().item()):
                    env.done[rows[succeeded]] = True
                stepped = _policy_step(
                    policy_model=policy_model,
                    encoder=encoder,
                    env=env,
                    beliefs=beliefs,
                    rng=rng,
                    args=args,
                    device=device,
                )
                if not stepped:
                    break
                reached = (
                    (env.street[rows] == 0)
                    & ~env.done[rows]
                    & (env.actions_this_round[rows] == target_action_count)
                    & ~succeeded
                )
                if bool(reached.any().item()):
                    reached_rows = rows[reached]
                    if target is not None:
                        target.append(env, beliefs, reached_rows)
                    succeeded |= reached
                    env.done[reached_rows] = True

            failed = ~succeeded
            if bool(failed.any().item()):
                next_pending.append(src[failed])
            successes_this_attempt += int(succeeded.sum().item())

        pending = (
            torch.cat(next_pending, dim=0)
            if next_pending
            else torch.empty(0, dtype=torch.long, device=device)
        )
        attempts_summary.append(
            {
                "attempt": attempt,
                "attempted": attempted_this_attempt,
                "successes": successes_this_attempt,
                "remaining": int(pending.numel()),
            }
        )

    writer.flush()
    return {
        "source_count": source.count,
        "target_count": 0 if target is None else target.count,
        "target_action_count": target_action_count,
        "attempts": attempts_summary,
        "failed": int(pending.numel()),
    }


def _write_unique_bucket_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    writers: list[ShardWriter],
    per_action_written: dict[int, int],
    complete: bool,
    elapsed_s: float,
    root_states_sampled: int,
) -> None:
    buckets = []
    for spec, writer in zip(BUCKET_SPECS, writers, strict=True):
        buckets.append(
            {
                "label": spec.label,
                "low": spec.low,
                "high": spec.high,
                "target_rows": args.frontier_result_cap,
                "num_rows": writer.rows,
                "action_count_rows": {
                    str(action_count): per_action_written.get(action_count, 0)
                    for action_count in range(spec.low, spec.high + 1)
                },
                "shards": list(writer.shards),
            }
        )
    manifest = {
        "format_version": 1,
        "kind": "preflop_policy_unique_frontier_buckets",
        "complete": complete,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": elapsed_s,
        "source_checkpoint": os.path.realpath(args.checkpoint),
        "source_wandb_run_id": metadata.get("checkpoint_wandb_run_id"),
        "source_step": metadata.get("checkpoint_step"),
        "num_rows": sum(writer.rows for writer in writers),
        "root_states_sampled": root_states_sampled,
        "root_states": args.root_states,
        "state_fields": list(STATE_FIELDS),
        "state_dtype_policy": {
            "int16": sorted(INT16_FIELDS),
            "int32": sorted(INT32_FIELDS),
            "bool": [
                "has_folded",
                "is_allin",
                "acted_this_round",
                "done",
                "winners",
            ],
            "float32": ["scale"],
        },
        "rollout": {
            "mode": "unique_frontier_bucket_streaming",
            "num_envs": args.num_envs,
            "buckets": [
                {"label": spec.label, "low": spec.low, "high": spec.high}
                for spec in BUCKET_SPECS
            ],
            "frontier_attempts": args.frontier_attempts,
            "frontier_result_cap": args.frontier_result_cap,
            "frontier_bucket_action_cap": _unique_bucket_action_cap(args),
            "seed": args.seed,
            "belief_init": args.belief_init,
            "belief_update": "actor_bayes_update_from_sampled_legal_action",
            "action_sampling": "actor_belief_weighted_policy_probs",
            "temperature": args.temperature,
            "epsilon": args.epsilon,
            "uniqueness": (
                "for each root and four-action bucket, retries at most "
                "frontier_attempts continuations; saves at most one state per "
                "action count and carries at most one successor frontier forward"
            ),
        },
        "buckets": buckets,
        "model": {
            "model_type": metadata.get("model_type"),
            "num_actions": metadata.get("num_actions"),
            "num_players": metadata.get("num_players"),
            "model_dtype": args.model_dtype,
            "compile": args.compile,
        },
        "env_config": metadata.get("env_config"),
        "model_config": metadata.get("model_config"),
    }
    path = output_dir / ("manifest.json" if complete else "manifest.partial.json")
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(manifest), f, indent=2, sort_keys=True)


def generate_unique_frontier_buckets_streaming(args: argparse.Namespace) -> None:
    if not args.unique_frontier_streaming:
        raise ValueError("--unique-frontier-buckets requires --unique-frontier-streaming")
    device = _device(args.device)
    dtype = _model_dtype(args.model_dtype, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; choose a new output dir or pass --overwrite"
        )

    policy_model, metadata = _load_policy_model(
        args.checkpoint,
        device=device,
        dtype=dtype,
        compile_model=args.compile,
        strict=not args.non_strict,
    )
    env = _make_env(metadata, num_envs=args.num_envs, device=device, seed=args.seed)
    encoder = policy_model.create_feature_encoder(env, device=device, dtype=dtype)
    beliefs = _initial_beliefs(
        args.num_envs,
        int(metadata["num_players"]),
        device=device,
        mode=args.belief_init,
    )
    uniform_belief = beliefs[0].clone()
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 1)
    writers = [
        ShardWriter(
            output_dir / spec.label,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
        )
        for spec in BUCKET_SPECS
    ]
    per_action_written: dict[int, int] = {
        action_count: 0 for action_count in range(BUCKET_SPECS[-1].high + 1)
    }
    bucket_cap = int(args.frontier_result_cap)
    action_cap = _unique_bucket_action_cap(args)

    started = time.time()
    last_progress = started
    sampled_roots = 0
    chunk_index = 0
    stage_summaries: dict[int, dict[str, Any]] = {}
    print(
        "Starting streaming unique-frontier bucket generation: "
        f"roots={args.root_states:,} attempts={args.frontier_attempts} "
        f"cap_per_bucket={bucket_cap:,} cap_per_action={action_cap:,} "
        f"chunk_roots={args.num_envs:,} device={device} dtype={dtype} "
        f"checkpoint_step={metadata['checkpoint_step']}"
    )
    print(f"Output: {output_dir}")

    with torch.inference_mode():
        while sampled_roots < args.root_states:
            chunk_roots = min(args.num_envs, args.root_states - sampled_roots)
            pools: dict[int, FrontierPool] = {}
            pools[0] = _build_root_frontier(
                args=args,
                env=env,
                beliefs=beliefs,
                uniform_belief=uniform_belief,
                writer=None,
                root_states=chunk_roots,
                flush_writer=False,
            )
            for stage, spec in enumerate(BUCKET_SPECS):
                target_action_count = spec.high + 1
                source = pools[spec.low]
                target = (
                    None
                    if stage == len(BUCKET_SPECS) - 1
                    else FrontierPool(
                        env=env,
                        beliefs=beliefs,
                        capacity=source.count,
                    )
                )
                summary = _advance_unique_bucket(
                    source=source,
                    bucket=spec,
                    target_action_count=target_action_count,
                    target=target,
                    writer=writers[stage],
                    per_action_written=per_action_written,
                    policy_model=policy_model,
                    encoder=encoder,
                    env=env,
                    beliefs=beliefs,
                    rng=rng,
                    args=args,
                    device=device,
                    bucket_cap=bucket_cap,
                    action_cap=action_cap,
                )
                _merge_unique_frontier_summary(stage_summaries, summary)
                if target is not None:
                    pools[target_action_count] = target

            sampled_roots += chunk_roots
            chunk_index += 1
            now = time.time()
            if (
                now - last_progress >= args.progress_interval_s
                or sampled_roots >= args.root_states
            ):
                elapsed = now - started
                bucket_rows = {
                    spec.label: writer.rows
                    for spec, writer in zip(BUCKET_SPECS, writers, strict=True)
                }
                stage_counts = {
                    target: summary["target_count"]
                    for target, summary in sorted(stage_summaries.items())
                }
                rate = sampled_roots / max(elapsed, 1.0e-9)
                print(
                    f"progress chunks={chunk_index:,} "
                    f"roots={sampled_roots:,}/{args.root_states:,} "
                    f"saved={bucket_rows} reached={stage_counts} "
                    f"rate={rate:,.0f} roots/s elapsed={elapsed:.1f}s",
                    flush=True,
                )
                _write_unique_bucket_manifest(
                    output_dir,
                    args=args,
                    metadata=metadata,
                    writers=writers,
                    per_action_written=per_action_written,
                    complete=False,
                    elapsed_s=elapsed,
                    root_states_sampled=sampled_roots,
                )
                last_progress = now

    for writer in writers:
        writer.flush()

    elapsed = time.time() - started
    _write_unique_bucket_manifest(
        output_dir,
        args=args,
        metadata=metadata,
        writers=writers,
        per_action_written=per_action_written,
        complete=True,
        elapsed_s=elapsed,
        root_states_sampled=sampled_roots,
    )
    stage_summary_list = [
        stage_summaries[target]
        for target in sorted(stage_summaries)
    ]
    summary_path = output_dir / "attempt_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(stage_summary_list), f, indent=2, sort_keys=True)
    print(
        "Done: saved={"
        + ", ".join(
            f"{spec.label}:{writer.rows}"
            for spec, writer in zip(BUCKET_SPECS, writers, strict=True)
        )
        + "} reached={"
        + ", ".join(
            f"{target}:{stage_summaries[target]['target_count']}"
            for target in sorted(stage_summaries)
        )
        + f"}} sampled_roots={sampled_roots:,} elapsed={elapsed:.1f}s "
        f"output={output_dir}"
    )


def generate_unique_frontier_streaming(args: argparse.Namespace) -> None:
    device = _device(args.device)
    dtype = _model_dtype(args.model_dtype, device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; choose a new output dir or pass --overwrite"
        )

    policy_model, metadata = _load_policy_model(
        args.checkpoint,
        device=device,
        dtype=dtype,
        compile_model=args.compile,
        strict=not args.non_strict,
    )
    env = _make_env(metadata, num_envs=args.num_envs, device=device, seed=args.seed)
    encoder = policy_model.create_feature_encoder(env, device=device, dtype=dtype)
    beliefs = _initial_beliefs(
        args.num_envs,
        int(metadata["num_players"]),
        device=device,
        mode=args.belief_init,
    )
    uniform_belief = beliefs[0].clone()
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed + 1)
    writers = {
        action_count: ShardWriter(
            output_dir / f"frontier_{action_count}",
            shard_size=args.shard_size,
            overwrite=args.overwrite,
        )
        for action_count in FRONTIER_ACTION_COUNTS
    }

    started = time.time()
    last_progress = started
    sampled_roots = 0
    chunk_index = 0
    stage_summaries: dict[int, dict[str, Any]] = {}
    print(
        "Starting streaming unique-frontier generation: "
        f"roots={args.root_states:,} attempts={args.frontier_attempts} "
        f"cap_per_frontier={args.frontier_result_cap:,} "
        f"chunk_roots={args.num_envs:,} device={device} dtype={dtype} "
        f"checkpoint_step={metadata['checkpoint_step']}"
    )
    print(f"Output: {output_dir}")

    with torch.inference_mode():
        while sampled_roots < args.root_states:
            chunk_roots = min(args.num_envs, args.root_states - sampled_roots)
            pools: dict[int, FrontierPool] = {}
            pools[0] = _build_root_frontier(
                args=args,
                env=env,
                beliefs=beliefs,
                uniform_belief=uniform_belief,
                writer=writers[0],
                root_states=chunk_roots,
                writer_cap=args.frontier_result_cap,
                flush_writer=False,
            )
            for target_action_count in FRONTIER_ACTION_COUNTS[1:]:
                source = pools[target_action_count - 4]
                target = FrontierPool(
                    env=env,
                    beliefs=beliefs,
                    capacity=source.count,
                )
                summary = _advance_unique_frontier(
                    source=source,
                    target_action_count=target_action_count,
                    target=target,
                    writer=writers[target_action_count],
                    policy_model=policy_model,
                    encoder=encoder,
                    env=env,
                    beliefs=beliefs,
                    rng=rng,
                    args=args,
                    device=device,
                    writer_cap=args.frontier_result_cap,
                    verbose=False,
                )
                pools[target_action_count] = target
                _merge_unique_frontier_summary(stage_summaries, summary)

            sampled_roots += chunk_roots
            chunk_index += 1
            now = time.time()
            if (
                now - last_progress >= args.progress_interval_s
                or sampled_roots >= args.root_states
            ):
                elapsed = now - started
                writer_rows = {
                    action_count: writer.rows
                    for action_count, writer in sorted(writers.items())
                }
                stage_counts = {
                    target: summary["target_count"]
                    for target, summary in sorted(stage_summaries.items())
                }
                rate = sampled_roots / max(elapsed, 1.0e-9)
                print(
                    f"progress chunks={chunk_index:,} "
                    f"roots={sampled_roots:,}/{args.root_states:,} "
                    f"saved={writer_rows} reached={stage_counts} "
                    f"rate={rate:,.0f} roots/s elapsed={elapsed:.1f}s",
                    flush=True,
                )
                _write_unique_frontier_manifest(
                    output_dir,
                    args=args,
                    metadata=metadata,
                    writers=writers,
                    complete=False,
                    elapsed_s=elapsed,
                    root_states_sampled=sampled_roots,
                )
                last_progress = now

    for writer in writers.values():
        writer.flush()

    elapsed = time.time() - started
    _write_unique_frontier_manifest(
        output_dir,
        args=args,
        metadata=metadata,
        writers=writers,
        complete=True,
        elapsed_s=elapsed,
        root_states_sampled=sampled_roots,
    )
    stage_summary_list = [
        stage_summaries[target]
        for target in sorted(stage_summaries)
    ]
    summary_path = output_dir / "attempt_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(stage_summary_list), f, indent=2, sort_keys=True)
    print(
        "Done: saved={"
        + ", ".join(
            f"{action_count}:{writers[action_count].rows}"
            for action_count in FRONTIER_ACTION_COUNTS
        )
        + "} reached={"
        + ", ".join(
            f"{target}:{stage_summaries[target]['target_count']}"
            for target in sorted(stage_summaries)
        )
        + f"}} sampled_roots={sampled_roots:,} elapsed={elapsed:.1f}s "
        f"output={output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output-dir",
        default="outputs/preflop_policy_states/eroymcd2_3m",
    )
    parser.add_argument("--num-states", type=int, default=3_000_000)
    parser.add_argument("--stratified", action="store_true")
    parser.add_argument("--unique-frontier", action="store_true")
    parser.add_argument(
        "--unique-frontier-streaming",
        action="store_true",
        help="Process unique-frontier roots in num-env chunks instead of one GPU pool.",
    )
    parser.add_argument(
        "--unique-frontier-buckets",
        action="store_true",
        help=(
            "With --unique-frontier-streaming, write range buckets 0-3/4-7/8-11/12-15 "
            "along unique retry trajectories instead of exact frontier rows."
        ),
    )
    parser.add_argument("--root-states", type=int, default=100_000)
    parser.add_argument("--frontier-attempts", type=int, default=5)
    parser.add_argument("--frontier-start-target", type=int, default=500_000)
    parser.add_argument(
        "--frontier-result-cap",
        type=int,
        default=0,
        help="Maximum saved rows per frontier; 0 means unlimited.",
    )
    parser.add_argument(
        "--frontier-bucket-action-cap",
        type=int,
        default=0,
        help=(
            "Maximum saved rows per individual action count in unique bucket mode; "
            "0 derives ceil(frontier-result-cap / 4)."
        ),
    )
    parser.add_argument("--bucket-target", type=int, default=1_000_000)
    parser.add_argument("--frontier-capacity", type=int, default=262_144)
    parser.add_argument("--frontier-min-states", type=int, default=16_384)
    parser.add_argument("--num-envs", type=int, default=65_536)
    parser.add_argument("--shard-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--model-dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16"],
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument(
        "--belief-init",
        default="combo_uniform",
        choices=["combo_uniform", "class_uniform"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--min-actions-this-round", type=int, default=0)
    parser.add_argument("--max-actions-this-round", type=int, default=24)
    parser.add_argument("--progress-interval-s", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
