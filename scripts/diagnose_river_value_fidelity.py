#!/usr/bin/env python3
"""Decompose river value-target variance for canonical hand-independent heads."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from p2.env.card_utils import NUM_HANDS, hand_combos_tensor
from p2.env.rules import rank_hands as rank_hands_torch
from p2.models.mlp.better_features import ValueScalarContext
from p2.rl.rebel_batch import RebelBatch
from p2.search.rebel_solved_dataset import RebelSolvedDataset


DEFAULT_DATASET = Path(
    "outputs/rebel_postflop/river_val_8192_10k_sapdcfr_nowarm_ctx41_20260630"
)
DEFAULT_BINS = (16, 32, 64, 128)
DEFAULT_SHOWDOWN_RANK_BINS = 96
POSNEG_POS_SCALE = 0.8543022528460094
POSNEG_NEG_SCALE = 0.4753640305061305
POSNEG_INTERCEPT = -0.010797645393242563


@dataclass
class RegressionAccumulator:
    names: tuple[str, ...]
    xtx: torch.Tensor
    xty: torch.Tensor
    yty: torch.Tensor
    weight_sum: torch.Tensor

    @classmethod
    def create(cls, names: tuple[str, ...], device: torch.device) -> RegressionAccumulator:
        dim = len(names)
        return cls(
            names=names,
            xtx=torch.zeros(dim, dim, dtype=torch.float64, device=device),
            xty=torch.zeros(dim, dtype=torch.float64, device=device),
            yty=torch.zeros((), dtype=torch.float64, device=device),
            weight_sum=torch.zeros((), dtype=torch.float64, device=device),
        )

    def update(self, x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> None:
        x64 = x.reshape(-1, x.shape[-1]).to(torch.float64)
        y64 = y.reshape(-1).to(torch.float64)
        w64 = weight.reshape(-1).to(torch.float64).clamp_min(0.0)
        valid = w64 > 0.0
        if not bool(valid.any().item()):
            return
        x64 = x64[valid]
        y64 = y64[valid]
        w64 = w64[valid]
        wx = x64 * w64[:, None]
        self.xtx += x64.T @ wx
        self.xty += x64.T @ (y64 * w64)
        self.yty += (y64.square() * w64).sum()
        self.weight_sum += w64.sum()

    def solve(self, ridge: float) -> dict[str, Any]:
        eye = torch.eye(self.xtx.shape[0], dtype=self.xtx.dtype, device=self.xtx.device)
        beta = torch.linalg.solve(self.xtx + float(ridge) * eye, self.xty)
        sse = (self.yty - beta.dot(self.xty)).clamp_min(0.0)
        mse = sse / self.weight_sum.clamp_min(1e-12)
        return {
            "features": list(self.names),
            "mse": float(mse.cpu().item()),
            "sse": float(sse.cpu().item()),
            "weight_sum": float(self.weight_sum.cpu().item()),
            "coefficients": {
                name: float(value)
                for name, value in zip(
                    self.names,
                    beta.detach().cpu().tolist(),
                    strict=True,
                )
            },
        }


def _dataset_dir(path: Path) -> Path:
    return path.parent if path.name == "manifest.json" else path


def _river_rank_groups(board: torch.Tensor) -> torch.Tensor:
    hand_ranks, sorted_indices = rank_hands_torch(board.int())
    sorted_ranks = hand_ranks.gather(1, sorted_indices.long())
    group_start = sorted_ranks[:, 1:] != sorted_ranks[:, :-1]
    group_start = torch.cat(
        (
            torch.ones(
                sorted_ranks.shape[0],
                1,
                device=sorted_ranks.device,
                dtype=torch.bool,
            ),
            group_start,
        ),
        dim=1,
    )
    sorted_groups = group_start.to(dtype=torch.long).cumsum(dim=1) - 1
    rank_groups = torch.empty_like(sorted_groups)
    rank_groups.scatter_(1, sorted_indices.long(), sorted_groups)
    return rank_groups


def _canonical_bins(
    beliefs: torch.Tensor,
    rank_groups: torch.Tensor,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = beliefs.sum(dim=1)
    group_mass = beliefs.new_zeros(beliefs.shape[0], NUM_HANDS)
    group_mass.scatter_add_(1, rank_groups, combined)
    cumulative_group = group_mass.cumsum(dim=1)
    total_ref = group_mass.sum(dim=1, keepdim=True).clamp_min(1e-8)
    u_group = (cumulative_group - 0.5 * group_mass) / total_ref
    u = u_group.gather(1, rank_groups).clamp(0.0, 1.0)
    k = (u * int(bins)).floor().clamp(max=int(bins) - 1).long()
    return u, k


def _board_allowed_hands(board: torch.Tensor, card_a: torch.Tensor, card_b: torch.Tensor) -> torch.Tensor:
    valid = board >= 0
    safe = torch.where(valid, board.long(), torch.full_like(board.long(), 52))
    board_onehot = torch.zeros(board.shape[0], 53, dtype=torch.bool, device=board.device)
    board_onehot.scatter_(1, safe, valid)
    board_onehot = board_onehot[:, :52]
    return ~(board_onehot[:, card_a] | board_onehot[:, card_b])


def _calculate_unblocked_mass(
    beliefs: torch.Tensor,
    card_a: torch.Tensor,
    card_b: torch.Tensor,
) -> torch.Tensor:
    flat = beliefs.reshape(-1, NUM_HANDS).float()
    total = flat.sum(dim=-1, keepdim=True)
    cardsum = torch.zeros(flat.shape[0], 52, dtype=flat.dtype, device=flat.device)
    card_a_idx = card_a[None, :].expand(flat.shape[0], -1)
    card_b_idx = card_b[None, :].expand(flat.shape[0], -1)
    cardsum.scatter_add_(1, card_a_idx, flat)
    cardsum.scatter_add_(1, card_b_idx, flat)
    unblocked = total - cardsum[:, card_a] - cardsum[:, card_b] + flat
    return unblocked.reshape_as(beliefs).clamp_min(0.0)


def _value_weights(
    batch: RebelBatch,
    beliefs: torch.Tensor,
    card_a: torch.Tensor,
    card_b: torch.Tensor,
) -> torch.Tensor:
    allowed = _board_allowed_hands(batch.features.board, card_a, card_b).to(
        dtype=beliefs.dtype
    )
    unblocked = _calculate_unblocked_mass(beliefs, card_a, card_b)
    players = beliefs.shape[1]
    player_ids = torch.arange(players, device=beliefs.device)
    non_focal = player_ids[None, :, None, None] != player_ids[None, None, :, None]
    weights = torch.where(
        non_focal,
        unblocked[:, None],
        torch.ones_like(unblocked[:, None]),
    ).prod(dim=2)
    weights = weights * allowed[:, None]
    folded = batch.statistics.get("has_folded")
    if folded is not None:
        live = ~folded.to(device=beliefs.device, dtype=torch.bool)
        weights = weights * live[:, :, None].to(dtype=weights.dtype)
    return weights


def _showdown_scalars(
    batch: RebelBatch,
    beliefs: torch.Tensor,
    rank_groups: torch.Tensor,
    *,
    rank_bins: int,
    blockers: bool,
    card_a: torch.Tensor,
    card_b: torch.Tensor,
    pos_scale: float,
    neg_scale: float,
    intercept: float,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    players = beliefs.shape[1]
    rank_groups = rank_groups.clamp(min=0, max=int(rank_bins) - 1)
    opponent_beliefs = beliefs.sum(dim=1, keepdim=True) - beliefs
    rank_idx = rank_groups[:, None, :].expand(-1, players, -1)
    rank_mass = beliefs.new_zeros(beliefs.shape[0], players, int(rank_bins))
    rank_mass.scatter_add_(2, rank_idx, opponent_beliefs)
    cumulative = rank_mass.cumsum(dim=2)
    tie_mass = rank_mass.gather(2, rank_idx)
    lower_mass = cumulative.gather(2, rank_idx) - tie_mass
    total_mass = rank_mass.sum(dim=2, keepdim=True).clamp_min(1e-8)

    if blockers:
        card_a_idx = card_a.view(1, 1, NUM_HANDS).expand_as(rank_idx)
        card_b_idx = card_b.view(1, 1, NUM_HANDS).expand_as(rank_idx)
        card_rank_bins = 52 * int(rank_bins)
        card_rank_mass = beliefs.new_zeros(beliefs.shape[0], players, card_rank_bins)
        flat_idx_a = card_a_idx * int(rank_bins) + rank_idx
        flat_idx_b = card_b_idx * int(rank_bins) + rank_idx
        card_rank_mass.scatter_add_(2, flat_idx_a, opponent_beliefs)
        card_rank_mass.scatter_add_(2, flat_idx_b, opponent_beliefs)
        card_rank_view = card_rank_mass.view(beliefs.shape[0], players, 52, int(rank_bins))
        card_mass = card_rank_view.sum(dim=3)
        card_rank_cumulative = card_rank_view.cumsum(dim=3).reshape(
            beliefs.shape[0],
            players,
            card_rank_bins,
        )
        card_tie_a = card_rank_mass.gather(2, flat_idx_a)
        card_tie_b = card_rank_mass.gather(2, flat_idx_b)
        card_lower_a = card_rank_cumulative.gather(2, flat_idx_a) - card_tie_a
        card_lower_b = card_rank_cumulative.gather(2, flat_idx_b) - card_tie_b
        same_combo_mass = opponent_beliefs
        blocked_tie = card_tie_a + card_tie_b - same_combo_mass
        blocked_lower = card_lower_a + card_lower_b
        blocked_total = (
            card_mass.gather(2, card_a_idx)
            + card_mass.gather(2, card_b_idx)
            - same_combo_mass
        )
        tie_mass = (tie_mass - blocked_tie).clamp_min(0.0)
        lower_mass = (lower_mass - blocked_lower).clamp_min(0.0)
        total_mass = (total_mass - blocked_total).clamp_min(1e-8)

    total = total_mass.expand(-1, players, NUM_HANDS)
    win_frac = lower_mass / total
    tie_frac = tie_mass / total
    loss_frac = (total - lower_mass - tie_mass).clamp_min(0.0) / total
    equity_score = 2.0 * win_frac + tie_frac - 1.0
    pot = batch.features.context[:, ValueScalarContext.POT.value].float()
    sdv = equity_score * pot[:, None, None]
    baseline = (
        sdv.clamp_min(0.0) * float(pos_scale)
        + sdv.clamp_max(0.0) * float(neg_scale)
        + float(intercept)
    )
    scalars = torch.stack(
        (
            beliefs,
            win_frac,
            tie_frac,
            loss_frac,
            equity_score,
            baseline,
        ),
        dim=-1,
    )
    return scalars, (
        "belief",
        "win_frac",
        "tie_frac",
        "loss_frac",
        "equity_score",
        "posneg_baseline",
    )


def _group_mean_prediction(
    y: torch.Tensor,
    weight: torch.Tensor,
    group_idx: torch.Tensor,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_idx = group_idx[:, None, :].expand(-1, y.shape[1], -1)
    group_weight = weight.new_zeros(y.shape[0], y.shape[1], bins)
    group_sum = weight.new_zeros(y.shape[0], y.shape[1], bins)
    group_weight.scatter_add_(2, group_idx, weight)
    group_sum.scatter_add_(2, group_idx, y * weight)
    group_mean = group_sum / group_weight.clamp_min(1e-12)
    pred = group_mean.gather(2, group_idx)
    return pred, group_weight


def _within_group_center(
    value: torch.Tensor,
    weight: torch.Tensor,
    group_idx: torch.Tensor,
    bins: int,
) -> torch.Tensor:
    group_idx = group_idx[:, None, :, None].expand(
        -1,
        value.shape[1],
        -1,
        value.shape[-1],
    )
    group_weight = weight.new_zeros(value.shape[0], value.shape[1], bins, 1)
    group_sum = weight.new_zeros(value.shape[0], value.shape[1], bins, value.shape[-1])
    group_weight.scatter_add_(2, group_idx[..., :1], weight[..., None])
    group_sum.scatter_add_(2, group_idx, value * weight[..., None])
    mean = group_sum / group_weight.clamp_min(1e-12)
    return value - mean.gather(2, group_idx)


def _scalar_sets(feature_names: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    index = {name: idx for idx, name in enumerate(feature_names)}
    return {
        "belief": (index["belief"],),
        "equity_score": (index["equity_score"],),
        "win_tie_loss": (
            index["win_frac"],
            index["tie_frac"],
            index["loss_frac"],
        ),
        "showdown_plus_belief": (
            index["belief"],
            index["win_frac"],
            index["tie_frac"],
            index["loss_frac"],
        ),
        "posneg_baseline": (index["posneg_baseline"],),
        "all_scalars": tuple(range(len(feature_names))),
    }


def _feature_subset(
    centered_features: torch.Tensor,
    feature_names: tuple[str, ...],
    indices: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    idx = torch.tensor(indices, dtype=torch.long, device=centered_features.device)
    names = tuple(feature_names[i] for i in indices)
    return centered_features.index_select(-1, idx), names


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    dataset_dir = _dataset_dir(Path(args.dataset))
    dataset = RebelSolvedDataset(dataset_dir)
    total_examples = dataset.stream_len("value")
    if args.max_examples is not None:
        total_examples = min(total_examples, int(args.max_examples))
    if total_examples <= 0:
        raise ValueError(f"dataset has no value examples: {dataset_dir}")

    bins_list = tuple(int(x) for x in args.bins)
    combos = hand_combos_tensor(device=device)
    card_a = combos[:, 0]
    card_b = combos[:, 1]
    global_weight_sum = torch.zeros((), dtype=torch.float64, device=device)
    global_target_sum = torch.zeros((), dtype=torch.float64, device=device)
    global_target_sq_sum = torch.zeros((), dtype=torch.float64, device=device)
    zero_sse = torch.zeros((), dtype=torch.float64, device=device)
    bin_sse = {
        bins: torch.zeros((), dtype=torch.float64, device=device)
        for bins in bins_list
    }
    bin_weight_sum = {
        bins: torch.zeros((), dtype=torch.float64, device=device)
        for bins in bins_list
    }
    scalar_accumulators: dict[int, dict[str, RegressionAccumulator]] = {}
    examples_seen = 0
    feature_names: tuple[str, ...] | None = None

    for start in range(0, total_examples, int(args.batch_size)):
        count = min(int(args.batch_size), total_examples - start)
        batch = dataset.get_batch(
            "value",
            start,
            count,
            device=device,
            float_dtype=torch.float32,
        )
        if batch.value_targets is None:
            raise ValueError("value batch unexpectedly lacks value_targets")
        if int(batch.features.hand_dim) != NUM_HANDS:
            raise ValueError(
                f"expected combo hand_dim={NUM_HANDS}, got {batch.features.hand_dim}"
            )
        if not bool((batch.features.street == 3).all().item()):
            raise ValueError("diagnostic expects river-only value data")

        beliefs = batch.features.beliefs.view(count, -1, NUM_HANDS).float()
        targets = batch.value_targets.float()
        weights = _value_weights(batch, beliefs, card_a, card_b)
        rank_groups = _river_rank_groups(batch.features.board)
        scalars, names = _showdown_scalars(
            batch,
            beliefs,
            rank_groups,
            rank_bins=int(args.showdown_rank_bins),
            blockers=bool(args.blockers),
            card_a=card_a,
            card_b=card_b,
            pos_scale=float(args.pos_scale),
            neg_scale=float(args.neg_scale),
            intercept=float(args.intercept),
        )
        feature_names = names

        weights64 = weights.to(torch.float64)
        targets64 = targets.to(torch.float64)
        global_weight_sum += weights64.sum()
        global_target_sum += (targets64 * weights64).sum()
        global_target_sq_sum += (targets64.square() * weights64).sum()
        zero_sse += (targets64.square() * weights64).sum()

        for bins in bins_list:
            u, group_idx = _canonical_bins(beliefs, rank_groups, bins)
            pred, group_weight = _group_mean_prediction(targets, weights, group_idx, bins)
            residual = targets - pred
            bin_sse[bins] += (residual.to(torch.float64).square() * weights64).sum()
            bin_weight_sum[bins] += group_weight.to(torch.float64).sum()

            y_centered = residual
            u_feature = u[:, None, :, None].expand(-1, targets.shape[1], -1, -1)
            all_features = torch.cat((u_feature, scalars), dim=-1)
            all_names = ("u", *names)
            centered_features = _within_group_center(
                all_features,
                weights,
                group_idx,
                bins,
            )
            sets = {
                "u": (0,),
                **{
                    key: tuple(i + 1 for i in value)
                    for key, value in _scalar_sets(names).items()
                },
            }
            if bins not in scalar_accumulators:
                scalar_accumulators[bins] = {}
                for set_name, indices in sets.items():
                    _, subset_names = _feature_subset(
                        centered_features,
                        all_names,
                        indices,
                    )
                    scalar_accumulators[bins][set_name] = (
                        RegressionAccumulator.create(subset_names, device)
                    )
            for set_name, indices in sets.items():
                x_subset, _ = _feature_subset(centered_features, all_names, indices)
                scalar_accumulators[bins][set_name].update(
                    x_subset,
                    y_centered,
                    weights,
                )

        examples_seen += count

    if feature_names is None:
        raise RuntimeError("no batches were processed")

    target_mean = global_target_sum / global_weight_sum.clamp_min(1e-12)
    variance_sse = global_target_sq_sum - target_mean.square() * global_weight_sum
    variance_mse = variance_sse / global_weight_sum.clamp_min(1e-12)
    zero_mse = zero_sse / global_weight_sum.clamp_min(1e-12)

    bin_results: dict[str, Any] = {}
    for bins in bins_list:
        strength_mse = bin_sse[bins] / bin_weight_sum[bins].clamp_min(1e-12)
        regressions = {
            name: acc.solve(float(args.ridge))
            for name, acc in sorted(scalar_accumulators[bins].items())
        }
        for reg in regressions.values():
            reg["explained_within_bin_fraction"] = (
                1.0 - reg["mse"] / max(float(strength_mse.cpu().item()), 1e-12)
            )
        bin_results[str(bins)] = {
            "strength_bin_oracle_mse": float(strength_mse.cpu().item()),
            "strength_bin_oracle_rmse": float(strength_mse.sqrt().cpu().item()),
            "explained_vs_global_variance_fraction": (
                1.0
                - float(strength_mse.cpu().item())
                / max(float(variance_mse.cpu().item()), 1e-12)
            ),
            "regressions": regressions,
        }

    return {
        "dataset": str(dataset_dir),
        "examples": examples_seen,
        "batch_size": int(args.batch_size),
        "device": str(device),
        "bins": list(bins_list),
        "showdown_rank_bins": int(args.showdown_rank_bins),
        "blockers": bool(args.blockers),
        "target_weight_sum": float(global_weight_sum.cpu().item()),
        "target_weighted_mean": float(target_mean.cpu().item()),
        "target_zero_mse": float(zero_mse.cpu().item()),
        "target_global_mean_variance_mse": float(variance_mse.cpu().item()),
        "target_global_mean_variance_rmse": float(variance_mse.sqrt().cpu().item()),
        "feature_names": list(feature_names),
        "bin_results": bin_results,
    }


def _parse_bins(value: str) -> tuple[int, ...]:
    bins = tuple(int(part) for part in value.split(",") if part)
    if not bins:
        raise argparse.ArgumentTypeError("at least one bin count is required")
    if any(bin_count <= 1 for bin_count in bins):
        raise argparse.ArgumentTypeError("all bin counts must be > 1")
    return bins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--bins", type=_parse_bins, default=DEFAULT_BINS)
    parser.add_argument("--showdown-rank-bins", type=int, default=DEFAULT_SHOWDOWN_RANK_BINS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument("--no-blockers", dest="blockers", action="store_false")
    parser.set_defaults(blockers=True)
    parser.add_argument("--pos-scale", type=float, default=POSNEG_POS_SCALE)
    parser.add_argument("--neg-scale", type=float, default=POSNEG_NEG_SCALE)
    parser.add_argument("--intercept", type=float, default=POSNEG_INTERCEPT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = diagnose(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
