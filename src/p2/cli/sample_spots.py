#!/usr/bin/env python3
"""
Sample N spots from a BetterFFN checkpoint's self-play with varied streets.

Spots are taken from `RebelDataGenerator.current_pbs` after each CFR
evaluation pass. Each spot is the (env state + beliefs) tuple needed to
reconstruct a `PublicBeliefState` and re-run CFR with different parameters.

Output: torch.save dict with keys
  - env_config: dict to rebuild HUNLTensorEnv prototype
  - spots: dict of stacked tensors (one row per spot) covering every
    field copied by HUNLTensorEnv.copy_state_from, plus beliefs.
  - source_checkpoint: path
  - trainer_config: full Config (asdict)
  - per_street_counts: dict[str, int]

Use `load_spots` / `build_pbs_from_spots` (defined below) to reconstruct.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

import hydra
import torch
from omegaconf import DictConfig

from p2.core.structured_config import Config
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.rl.cfr_trainer import RebelCFRTrainer
from p2.search.cfr_evaluator import PublicBeliefState

STREETS = ["preflop", "flop", "turn", "river", "showdown"]

# Tensor fields on HUNLTensorEnv that fully describe a public/private state.
ENV_STATE_FIELDS = [
    "button", "street", "to_act", "last_to_act", "pot", "min_raise",
    "actions_this_round", "actions_last_round", "acted_since_reset",
    "stacks", "committed", "has_folded", "is_allin", "chips_placed",
    "starting_stacks", "scale",
    "board_onehot", "hole_onehot",
    "board_indices", "last_board_indices", "hole_indices",
    "done", "winner",
    "deck", "deck_pos",
]


def _device_from_config(cfg: Config) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg.device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _snapshot_pbs(pbs: PublicBeliefState) -> dict[str, torch.Tensor]:
    """Return a CPU dict of every state tensor for every env in `pbs`."""
    out: dict[str, torch.Tensor] = {}
    for name in ENV_STATE_FIELDS:
        out[name] = getattr(pbs.env, name).detach().to("cpu").clone()
    out["beliefs"] = pbs.beliefs.detach().to("cpu").clone()
    return out


def _select_rows(snapshot: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[idx].clone() for k, v in snapshot.items()}


def _concat_snapshots(snaps: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = snaps[0].keys()
    return {k: torch.cat([s[k] for s in snaps], dim=0) for k in keys}


def sample_spots(
    cfg: Config,
    output_path: str,
    num_spots: int,
    street_weights: list[float] | None = None,
    max_passes: int = 200,
) -> None:
    """Generate `num_spots` spots from self-play and save to `output_path`.

    Args:
        cfg: trainer config (must include resume_from pointing at a checkpoint).
        output_path: .pt destination.
        num_spots: total spots to collect.
        street_weights: optional per-street target proportions
            (preflop, flop, turn, river, showdown). Defaults to a uniform
            split across non-showdown streets ([0.25, 0.25, 0.25, 0.25, 0]).
        max_passes: hard cap on data-generation passes.
    """
    device = _device_from_config(cfg)
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    if not cfg.resume_from or not os.path.exists(cfg.resume_from):
        raise ValueError(
            f"resume_from must point at a checkpoint; got {cfg.resume_from!r}"
        )

    trainer = RebelCFRTrainer(cfg=cfg, device=device)
    print(f"Loading checkpoint: {cfg.resume_from}")
    # Sampling only runs inference/CFR; skip optimizer state, which can drift
    # in format across checkpoints and is irrelevant here.
    trainer.load_checkpoint(cfg.resume_from, load_optimizer=False)
    trainer.model.eval()

    if street_weights is None:
        street_weights = [0.25, 0.25, 0.25, 0.25, 0.0]
    assert len(street_weights) == 5
    targets = [int(round(num_spots * w)) for w in street_weights]
    # Round-off correction so totals == num_spots
    diff = num_spots - sum(targets)
    if diff != 0:
        # Add/subtract on the largest-weight street
        i = max(range(5), key=lambda j: street_weights[j])
        targets[i] += diff
    target_by_street = {STREETS[i]: targets[i] for i in range(5)}
    print(f"Per-street targets: {target_by_street}")

    collected_per_street: dict[int, list[dict[str, torch.Tensor]]] = {i: [] for i in range(5)}
    counts: dict[int, int] = {i: 0 for i in range(5)}

    def _absorb(snap: dict[str, torch.Tensor]) -> None:
        streets = snap["street"]  # CPU long [N]
        for s in range(5):
            if counts[s] >= targets[s]:
                continue
            mask = streets == s
            if not bool(mask.any()):
                continue
            idx_all = torch.where(mask)[0]
            need = targets[s] - counts[s]
            take = idx_all[: need] if idx_all.numel() > need else idx_all
            collected_per_street[s].append(_select_rows(snap, take))
            counts[s] += take.numel()

    dg = trainer.data_generator
    target_batch_size = dg.target_batch_size
    passes = 0
    # The evaluator runs on trainer.inference_model, which load_checkpoint
    # already synced from the (EMA) checkpoint weights, so no weight swap is
    # needed here.
    with torch.no_grad():
        while any(counts[i] < targets[i] for i in range(5)):
            if passes >= max_passes:
                print(
                    f"[warn] hit max_passes={max_passes} with counts={counts}; "
                    f"saving what we have."
                )
                break

            # Mirror RebelDataGenerator.generate_data, but snapshot the
            # root PBS (what the trainer hands to CFR) *and* the leaves
            # (current_pbs after evaluate_cfr).
            if dg.current_pbs is None:
                dg.current_pbs = dg._new_pbs(target_batch_size)
            elif dg.current_pbs.env.N < target_batch_size:
                dg.current_pbs = dg._extend_pbs(dg.current_pbs, target_batch_size)

            # Snapshot the root PBS — the actual input to CFR. We skip
            # the post-evaluate leaves: those are pre-chance states and
            # only become valid (post-chance) roots on the next pass.
            _absorb(_snapshot_pbs(dg.current_pbs))

            root_count = int(dg.current_pbs.env.N)
            root_indices = torch.arange(root_count, device=device)
            dg.evaluator.initialize_subgame(
                dg.current_pbs.env,
                root_indices,
                dg.current_pbs.beliefs[:root_count],
            )
            dg.current_pbs = dg.evaluator.evaluate_cfr()
            passes += 1
            if passes % 5 == 0:
                print(f"pass {passes}: counts={counts}")

    print(f"Done after {passes} passes. Final counts: {counts}")

    # Concatenate, preserving street order.
    all_snaps: list[dict[str, torch.Tensor]] = []
    for s in range(5):
        if collected_per_street[s]:
            all_snaps.append(_concat_snapshots(collected_per_street[s]))
    if not all_snaps:
        raise RuntimeError("No spots collected.")
    spots = _concat_snapshots(all_snaps)
    n = spots["street"].shape[0]
    print(f"Saving {n} spots to {output_path}")

    env = trainer.env
    env_config = {
        "mean_stack": env.mean_stack,
        "stack_mode": env.stack_mode,
        "min_stack_bb": env.min_stack_bb,
        "mid_stack_bb": env.mid_stack_bb,
        "max_stack_bb": env.max_stack_bb,
        "high_stack_mass_ratio": env.high_stack_mass_ratio,
        "sb": env.sb,
        "bb": env.bb,
        "default_bet_bins": list(env.default_bet_bins),
        "flop_showdown": env.flop_showdown,
        "randomize_stacks": env.randomize_stacks,
        "float_dtype": str(env.float_dtype),
    }

    payload: dict[str, Any] = {
        "env_config": env_config,
        "spots": spots,
        "source_checkpoint": os.path.realpath(cfg.resume_from),
        "trainer_config": asdict(cfg),
        "per_street_counts": {STREETS[s]: counts[s] for s in range(5)},
        "format_version": 1,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved {n} spots ({payload['per_street_counts']}) to {output_path}")


# --------------------------------------------------------------------------- #
# Reconstruction helpers (also useful from other scripts / notebooks).
# --------------------------------------------------------------------------- #

def load_spots(path: str, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, weights_only=False, map_location=map_location)


def build_pbs_from_spots(
    payload: dict[str, Any],
    device: torch.device,
    indices: torch.Tensor | slice | None = None,
) -> PublicBeliefState:
    """Reconstruct a PublicBeliefState from a saved spots payload.

    `indices` selects which spots to load (default: all).
    """
    spots = payload["spots"]
    n_total = spots["street"].shape[0]
    if indices is None:
        indices = slice(0, n_total)
    if isinstance(indices, slice):
        idx_t = torch.arange(n_total)[indices]
    else:
        idx_t = indices

    n = idx_t.numel() if isinstance(idx_t, torch.Tensor) else (idx_t.stop - idx_t.start)
    ec = payload["env_config"]
    float_dtype = (
        torch.bfloat16 if ec["float_dtype"] == "torch.bfloat16"
        else torch.float16 if ec["float_dtype"] == "torch.float16"
        else torch.float32
    )
    env = HUNLTensorEnv(
        num_envs=int(n),
        mean_stack=ec["mean_stack"],
        sb=ec["sb"],
        bb=ec["bb"],
        default_bet_bins=ec["default_bet_bins"],
        device=device,
        float_dtype=float_dtype,
        flop_showdown=ec.get("flop_showdown", False),
        randomize_stacks=ec.get("randomize_stacks", False),
        stack_mode=ec.get("stack_mode", "fixed"),
        min_stack_bb=ec.get("min_stack_bb", 10),
        mid_stack_bb=ec.get("mid_stack_bb", 200),
        max_stack_bb=ec.get("max_stack_bb", 400),
        high_stack_mass_ratio=ec.get("high_stack_mass_ratio", 1.0 / 3.0),
    )
    for name in ENV_STATE_FIELDS:
        getattr(env, name).copy_(spots[name][idx_t].to(device))

    beliefs = spots["beliefs"][idx_t].to(device)
    return PublicBeliefState(env=env, beliefs=beliefs)


# --------------------------------------------------------------------------- #
# Hydra entry point.
# --------------------------------------------------------------------------- #

@hydra.main(
    version_base=None, config_path="../../../conf", config_name="config_rebel_cfr"
)
def main(dict_config: DictConfig) -> None:
    cfg = Config.from_dict_config(dict_config)
    output_path = os.environ.get("SPOTS_OUTPUT", "spots.pt")
    num_spots = int(os.environ.get("SPOTS_N", "1024"))
    weights_env = os.environ.get("SPOTS_STREET_WEIGHTS")
    street_weights: list[float] | None = None
    if weights_env:
        street_weights = [float(x) for x in weights_env.split(",")]
    sample_spots(
        cfg=cfg,
        output_path=output_path,
        num_spots=num_spots,
        street_weights=street_weights,
    )


if __name__ == "__main__":
    main()
