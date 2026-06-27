from __future__ import annotations

from contextlib import nullcontext
from typing import Protocol

import torch

from p2.env.card_utils import PREFLOP_HANDS
from p2.env.pbs_env import PBSEnv
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch
from p2.rl.target_provenance import TARGET_SOURCE_CHANCE_EXPECTATION


class ValueFeatureEncoder(Protocol):
    belief_dim: int

    def encode(
        self,
        beliefs: torch.Tensor,
        pre_chance_node: torch.Tensor | bool | None = None,
        indices: torch.Tensor | None = None,
    ) -> MLPFeatures: ...


def _stack_value_baseline(env: PBSEnv, hand_dim: int) -> torch.Tensor:
    scale = env.scale.to(torch.float32).clamp_min(1.0)
    stack_value = (
        env.stacks.to(torch.float32) - env.starting_stacks.to(torch.float32)
    ) / scale[:, None]
    return stack_value[:, :, None].expand(-1, -1, hand_dim).clone()


def _first_pair_actor_after_button(
    *,
    button: torch.Tensor,
    pair_players: torch.Tensor,
    num_players: int,
) -> torch.Tensor:
    first = pair_players[:, 0]
    second = pair_players[:, 1]
    first_dist = (first - button) % num_players
    second_dist = (second - button) % num_players
    first_dist = torch.where(first_dist == 0, first_dist + num_players, first_dist)
    second_dist = torch.where(second_dist == 0, second_dist + num_players, second_dist)
    choose_first = first_dist < second_dist
    return torch.where(choose_first, first, second)


def _project_live_pair_env(
    env: PBSEnv,
    row_indices: torch.Tensor,
    pair_players: torch.Tensor,
) -> PBSEnv:
    projected = PBSEnv(
        num_envs=int(row_indices.numel()),
        num_players=2,
        mean_stack=env.mean_stack,
        sb=env.sb,
        bb=env.bb,
        default_bet_bins=env.default_bet_bins,
        device=env.device,
        rng=env.rng,
        float_dtype=env.float_dtype,
        stack_mode=env.stack_mode,
        min_stack_bb=env.min_stack_bb,
        mid_stack_bb=env.mid_stack_bb,
        max_stack_bb=env.max_stack_bb,
        high_stack_mass_ratio=env.high_stack_mass_ratio,
        force_heads_up_preflop_flop=False,
    )
    for field in (
        "street",
        "pot",
        "min_raise",
        "last_aggressive_amount",
        "actions_this_round",
        "actions_last_round",
        "scale",
        "done",
        "winner",
        "board_indices",
        "last_board_indices",
        "board_onehot",
        "deck",
        "deck_pos",
    ):
        getattr(projected, field)[:] = getattr(env, field)[row_indices]

    for field in (
        "stacks",
        "starting_stacks",
        "committed",
        "chips_placed",
        "has_folded",
        "is_allin",
        "acted_this_round",
        "winners",
    ):
        src = getattr(env, field)[row_indices]
        getattr(projected, field)[:] = src.gather(1, pair_players)

    source_button = env.button[row_indices]
    source_first = _first_pair_actor_after_button(
        button=source_button,
        pair_players=pair_players,
        num_players=env.num_players,
    )
    to_act_proj = (pair_players == source_first[:, None]).to(torch.long).argmax(dim=1)
    projected.to_act[:] = to_act_proj
    projected.button[:] = 1 - to_act_proj
    last_match = pair_players == env.last_to_act[row_indices, None]
    last_proj = last_match.to(torch.long).argmax(dim=1)
    projected.last_to_act[:] = torch.where(last_match.any(dim=1), last_proj, to_act_proj)
    return projected


@torch.no_grad()
def uniform_live_pair_value_targets(
    env: PBSEnv,
    beliefs: torch.Tensor,
    *,
    target_model: torch.nn.Module,
    target_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build `[B, 6, 169]` targets by uniformly enumerating live HU projections.

    Each unordered pair of currently live players is evaluated by the frozen
    2-player preflop boundary value model. Returned values are accumulated only
    for the participating players and averaged per player, so a player's target
    is not diluted by projections where that player is absent.
    """

    if env.num_players != 6:
        raise ValueError(f"expected a 6-player PBSEnv, got {env.num_players}")
    if beliefs.dim() != 3 or beliefs.shape[1:] != (6, PREFLOP_HANDS):
        raise ValueError(
            "beliefs must have shape [batch, 6, 169], got "
            f"{tuple(beliefs.shape)}"
        )
    if int(getattr(target_model, "num_players", 0)) != 2:
        raise ValueError("uniform live-pair targets require a 2-player target model")

    device = env.device
    rows = int(beliefs.shape[0])
    hand_dim = int(getattr(target_model, "hand_dim", PREFLOP_HANDS))
    if hand_dim != PREFLOP_HANDS:
        raise ValueError(f"target model must use 169 preflop hands, got {hand_dim}")

    baseline = _stack_value_baseline(env, PREFLOP_HANDS)[:rows].to(device=device)
    target_sum = torch.zeros_like(baseline)
    target_count = torch.zeros(
        rows,
        env.num_players,
        1,
        dtype=baseline.dtype,
        device=device,
    )
    flat_target_sum = target_sum.view(rows * env.num_players, PREFLOP_HANDS)
    flat_target_count = target_count.view(rows * env.num_players, 1)
    live = (~env.has_folded[:rows]) & (~env.done[:rows, None])

    target_model.eval()
    dtype = target_dtype or beliefs.dtype
    for first in range(env.num_players - 1):
        for second in range(first + 1, env.num_players):
            pair_mask = live[:, first] & live[:, second]
            row_indices = torch.where(pair_mask)[0]
            pair_players = torch.stack(
                (
                    torch.full_like(row_indices, first),
                    torch.full_like(row_indices, second),
                ),
                dim=1,
            )
            projected_env = _project_live_pair_env(env, row_indices, pair_players)
            pair_beliefs = beliefs[row_indices].gather(
                1,
                pair_players[:, :, None].expand(-1, 2, PREFLOP_HANDS),
            )
            encoder = target_model.create_feature_encoder(
                env=projected_env,
                device=device,
                dtype=dtype,
            )
            features = encoder.encode(pair_beliefs, pre_chance_node=True)
            with (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            ):
                output = target_model(
                    features,
                    include_policy=False,
                    apply_zero_sum=False,
                )
            pair_values = output.hand_values.to(dtype=baseline.dtype)
            first_flat = row_indices * env.num_players + first
            second_flat = row_indices * env.num_players + second
            flat_target_sum.index_add_(0, first_flat, pair_values[:, 0, :])
            flat_target_sum.index_add_(0, second_flat, pair_values[:, 1, :])
            ones = torch.ones(
                row_indices.numel(),
                1,
                dtype=baseline.dtype,
                device=device,
            )
            flat_target_count.index_add_(0, first_flat, ones)
            flat_target_count.index_add_(0, second_flat, ones)

    return torch.where(
        target_count > 0.0,
        target_sum / target_count.clamp_min(1.0),
        baseline,
    )


@torch.no_grad()
def build_preflop_live_pair_value_batch(
    env: PBSEnv,
    beliefs: torch.Tensor,
    *,
    value_encoder: ValueFeatureEncoder,
    target_model: torch.nn.Module,
    float_dtype: torch.dtype | None = None,
) -> RebelBatch:
    """Build a value-only 6p E_preflop distillation batch.

    `env` should contain preflop street-closed states represented after the
    closing action and before any deterministic force-HU fold mutation. In
    practice these states may have `street == 1` with `last_board_indices`
    carrying the pre-chance context; features are encoded with
    `pre_chance_node=True`.
    """

    rows = int(beliefs.shape[0])
    pre_features = value_encoder.encode(beliefs, pre_chance_node=True)
    value_targets = uniform_live_pair_value_targets(
        env,
        beliefs,
        target_model=target_model,
        target_dtype=float_dtype,
    )
    statistics = {
        "street": torch.zeros(rows, dtype=torch.long, device=env.device),
        "stage": torch.ones(rows, dtype=torch.long, device=env.device),
        "board": env.last_board_indices[:rows].clone(),
        "has_folded": env.has_folded[:rows].clone(),
        "is_allin": env.is_allin[:rows].clone(),
        "pot": env.pot[:rows].clone(),
        "scale": env.scale[:rows].clone(),
        "actions_this_round": env.actions_this_round[:rows].clone(),
        "target_source": torch.full(
            (rows,),
            TARGET_SOURCE_CHANCE_EXPECTATION,
            dtype=torch.long,
            device=env.device,
        ),
    }
    return RebelBatch(
        features=pre_features,
        legal_masks=env.legal_bins_mask()[:rows],
        value_targets=value_targets,
        statistics=statistics,
    )
