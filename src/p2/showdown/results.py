from __future__ import annotations

from dataclasses import dataclass

import torch

from p2.env.card_utils import NUM_HANDS


@dataclass
class PerHandEquityResult:
    equity_by_hand: torch.Tensor
    aggregate_equity: torch.Tensor
    denominator_by_hand: torch.Tensor
    seconds: float
    numerator_by_hand: torch.Tensor | None = None


def safe_divide_by_hand(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0e-30),
        torch.zeros_like(numerator),
    )


def aggregate_from_num_denom(
    hero_belief: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    return (hero_belief * numerator).sum() / (hero_belief * denominator).sum().clamp_min(
        1.0e-30,
    )


def aggregate_all_players_from_num_denom(
    beliefs: torch.Tensor,
    numerator_by_hand: torch.Tensor,
    denominator_by_hand: torch.Tensor,
) -> torch.Tensor:
    if beliefs.dim() == 3:
        return (
            (beliefs * numerator_by_hand).sum(dim=-1)
            / (beliefs * denominator_by_hand).sum(dim=-1).clamp_min(1.0e-30)
        ).to(torch.float32)
    values = [
        aggregate_from_num_denom(
            beliefs[player],
            numerator_by_hand[player],
            denominator_by_hand[player],
        )
        for player in range(beliefs.shape[0])
    ]
    return torch.stack(values).to(torch.float32)[None, :]


def scatter_active_to_full(
    active_values: torch.Tensor,
    active_ids: torch.Tensor,
    *,
    hand_count: int = NUM_HANDS,
) -> torch.Tensor:
    full_shape = (*active_values.shape[:-1], hand_count)
    full = torch.zeros(
        full_shape,
        dtype=active_values.dtype,
        device=active_values.device,
    )
    full[..., active_ids] = active_values
    return full
