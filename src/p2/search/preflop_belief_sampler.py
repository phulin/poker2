"""Compact preflop belief samplers for solver-query coverage."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch

from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    combo_to_preflop_class_tensor,
    preflop_class_multiplicity_tensor,
)

AA_CLASS_INDEX = PREFLOP_HANDS - 1
DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99, 1.0)


@dataclass(frozen=True)
class BeliefShapeProfile:
    """Observed or desired one-player belief-row shape."""

    name: str
    max_class_quantiles: tuple[tuple[float, float], ...]
    aa_mass_quantiles: tuple[tuple[float, float], ...]
    entropy_quantiles: tuple[tuple[float, float], ...]
    aa_top_probability: float


OBSERVED_CASCADE_PROFILES: dict[str, BeliefShapeProfile] = {
    "actions_0_3": BeliefShapeProfile(
        name="actions_0_3",
        max_class_quantiles=(
            (0.0, 1.0 / PREFLOP_HANDS),
            (0.5, 0.04280425235629082),
            (0.9, 0.05800998955965042),
            (0.95, 0.06398236751556396),
            (0.99, 0.07746865600347519),
            (1.0, 0.12924601137638092),
        ),
        aa_mass_quantiles=(
            (0.0, 0.0),
            (0.5, 0.003153153695166111),
            (0.9, 0.010264914482831955),
            (0.95, 0.013399523682892323),
            (0.99, 0.021110067144036293),
            (1.0, 0.042222507297992706),
        ),
        entropy_quantiles=(
            (0.0, 0.0),
            (0.5, 4.587871551513672),
            (0.9, 4.655216217041016),
            (0.95, 4.673076629638672),
            (0.99, 4.703066349029541),
            (1.0, 4.758608818054199),
        ),
        aa_top_probability=0.08,
    ),
    "actions_4_7": BeliefShapeProfile(
        name="actions_4_7",
        max_class_quantiles=(
            (0.0, 1.0 / PREFLOP_HANDS),
            (0.5, 0.057508599013090134),
            (0.9, 0.19391009211540222),
            (0.95, 0.26687464118003845),
            (0.99, 0.6680695414543152),
            (1.0, 1.0),
        ),
        aa_mass_quantiles=(
            (0.0, 0.0),
            (0.5, 0.005917159840464592),
            (0.9, 0.04425138235092163),
            (0.95, 0.08955816924571991),
            (0.99, 0.34380075335502625),
            (1.0, 1.0000001192092896),
        ),
        entropy_quantiles=(
            (0.0, 0.0),
            (0.5, 4.478400230407715),
            (0.9, 4.644408226013184),
            (0.95, 4.68853759765625),
            (0.99, 5.1298980712890625),
            (1.0, 5.1298980712890625),
        ),
        aa_top_probability=0.12,
    ),
    "actions_8_11": BeliefShapeProfile(
        name="actions_8_11",
        max_class_quantiles=(
            (0.0, 1.0 / PREFLOP_HANDS),
            (0.5, 0.09369343519210815),
            (0.9, 0.2776058614253998),
            (0.95, 0.4370025396347046),
            (0.99, 1.0),
            (1.0, 1.0),
        ),
        aa_mass_quantiles=(
            (0.0, 0.0),
            (0.5, 0.005917159840464592),
            (0.9, 0.07162988185882568),
            (0.95, 0.12399786710739136),
            (0.99, 0.5101838707923889),
            (1.0, 1.0),
        ),
        entropy_quantiles=(
            (0.0, 0.0),
            (0.5, 4.004179000854492),
            (0.9, 5.122889518737793),
            (0.95, 5.1298980712890625),
            (0.99, 5.1298980712890625),
            (1.0, 5.129899024963379),
        ),
        aa_top_probability=0.12,
    ),
    "actions_12_end": BeliefShapeProfile(
        name="actions_12_end",
        max_class_quantiles=(
            (0.0, 1.0 / PREFLOP_HANDS),
            (0.5, 0.10509036481380463),
            (0.9, 0.42445486783981323),
            (0.95, 0.6468064188957214),
            (0.99, 1.0),
            (1.0, 1.0),
        ),
        aa_mass_quantiles=(
            (0.0, 0.0),
            (0.5, 0.005917159840464592),
            (0.9, 0.06678014993667603),
            (0.95, 0.161063551902771),
            (0.99, 1.0),
            (1.0, 1.0),
        ),
        entropy_quantiles=(
            (0.0, 0.0),
            (0.5, 3.8623714447021484),
            (0.9, 5.1298980712890625),
            (0.95, 5.1298980712890625),
            (0.99, 5.129898548126221),
            (1.0, 5.129899024963379),
        ),
        aa_top_probability=0.10,
    ),
}


def _validate_hand_dim(hand_dim: int) -> int:
    if hand_dim not in (PREFLOP_HANDS, NUM_HANDS):
        raise ValueError(
            f"hand_dim must be one of {PREFLOP_HANDS} or {NUM_HANDS}, got {hand_dim}"
        )
    return int(hand_dim)


def _class_prior(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    prior = preflop_class_multiplicity_tensor(device=device).to(torch.float32)
    prior = prior / prior.sum().clamp_min(1.0e-12)
    return prior.to(dtype=dtype)


def _hand_prior(
    hand_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    hand_dim = _validate_hand_dim(hand_dim)
    if hand_dim == PREFLOP_HANDS:
        return _class_prior(device, dtype)
    return torch.full((NUM_HANDS,), 1.0 / NUM_HANDS, device=device, dtype=dtype)


def _aa_mask(hand_dim: int, device: torch.device) -> torch.Tensor:
    hand_dim = _validate_hand_dim(hand_dim)
    if hand_dim == PREFLOP_HANDS:
        mask = torch.zeros(PREFLOP_HANDS, device=device, dtype=torch.bool)
        mask[AA_CLASS_INDEX] = True
        return mask
    class_ids = combo_to_preflop_class_tensor(device=device)
    return class_ids == AA_CLASS_INDEX


def _sample_aa_indices(
    rows: int,
    *,
    hand_dim: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    aa_indices = torch.where(_aa_mask(hand_dim, device))[0]
    choice = torch.randint(
        0,
        aa_indices.numel(),
        (rows,),
        device=device,
        generator=generator,
    )
    return aa_indices.index_select(0, choice)


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return values / values.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def _profile(profile: str | BeliefShapeProfile) -> BeliefShapeProfile:
    if isinstance(profile, BeliefShapeProfile):
        return profile
    try:
        return OBSERVED_CASCADE_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(OBSERVED_CASCADE_PROFILES))
        raise KeyError(f"unknown belief profile {profile!r}; expected one of {choices}") from exc


def _sample_from_quantile_curve(
    points: tuple[tuple[float, float], ...],
    rows: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if rows <= 0:
        return torch.empty(0, device=device, dtype=dtype)
    q = torch.tensor([point[0] for point in points], device=device, dtype=torch.float32)
    values = torch.tensor(
        [point[1] for point in points], device=device, dtype=torch.float32
    )
    u = torch.rand(rows, device=device, generator=generator)
    return _quantile_curve_values(q, values, u).to(dtype=dtype)


def _quantile_curve_values(
    quantiles: torch.Tensor,
    values: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    hi = torch.bucketize(u, quantiles[1:], right=False) + 1
    hi = hi.clamp(max=quantiles.numel() - 1)
    lo = (hi - 1).clamp(min=0)
    q_lo = quantiles[lo]
    q_hi = quantiles[hi]
    v_lo = values[lo]
    v_hi = values[hi]
    frac = (u - q_lo) / (q_hi - q_lo).clamp_min(1.0e-12)
    return v_lo + frac * (v_hi - v_lo)


def _curve_tensors(
    points: tuple[tuple[float, float], ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([point[0] for point in points], device=device, dtype=torch.float32),
        torch.tensor([point[1] for point in points], device=device, dtype=torch.float32),
    )


def _entropy_curve_tensors(
    points: tuple[tuple[float, float], ...],
    *,
    hand_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    quantiles, values = _curve_tensors(points, device=device)
    if hand_dim != PREFLOP_HANDS:
        values = values * (math.log(hand_dim) / math.log(PREFLOP_HANDS))
    return quantiles, values


def _row_entropy(beliefs: torch.Tensor) -> torch.Tensor:
    probs = beliefs.clamp_min(1.0e-12)
    return -(probs * probs.log()).sum(dim=-1)


def sample_random_preflop_belief_rows(
    rows: int,
    *,
    hand_dim: int = PREFLOP_HANDS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Sample smooth random preflop belief rows around the natural prior."""

    device = torch.device(device)
    hand_dim = _validate_hand_dim(hand_dim)
    prior = _hand_prior(hand_dim, device, dtype)
    weights = torch.empty(
        rows,
        hand_dim,
        device=device,
        dtype=dtype,
    ).exponential_(1.0, generator=generator)
    if alpha != 1.0:
        # This is not an exact Dirichlet draw. It is a cheap deterministic
        # generator-compatible way to widen or sharpen the exponential prior.
        weights = weights.pow(1.0 / max(float(alpha), 1.0e-4))
    return _normalize_rows(weights * prior.view(1, hand_dim))


def sample_histogram_matched_preflop_belief_rows(
    rows: int,
    *,
    profile: str | BeliefShapeProfile,
    hand_dim: int = PREFLOP_HANDS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample rows whose max-hand tail follows an observed cascade profile."""

    device = torch.device(device)
    hand_dim = _validate_hand_dim(hand_dim)
    shape = _profile(profile)
    prior = _hand_prior(hand_dim, device, dtype)
    if rows <= 0:
        return torch.empty(rows, hand_dim, device=device, dtype=dtype)

    u = torch.rand(rows, device=device, generator=generator)
    q_max, v_max = _curve_tensors(shape.max_class_quantiles, device=device)
    q_aa, v_aa = _curve_tensors(shape.aa_mass_quantiles, device=device)
    q_entropy, v_entropy = _entropy_curve_tensors(
        shape.entropy_quantiles,
        hand_dim=hand_dim,
        device=device,
    )
    target_max = _quantile_curve_values(q_max, v_max, u).to(dtype=dtype)
    target_aa = _quantile_curve_values(q_aa, v_aa, u).to(dtype=dtype)
    target_entropy = _quantile_curve_values(q_entropy, v_entropy, 1.0 - u).to(
        dtype=dtype
    )
    target_max = torch.maximum(target_max, prior.max()).clamp_max(1.0)
    target_aa = torch.minimum(target_aa.clamp_min(0.0), target_max)
    uniform_tail = target_entropy >= math.log(hand_dim) - 1.0e-5

    aa_mask = _aa_mask(hand_dim, device)
    aa_count = aa_mask.sum().clamp_min(1).to(dtype=dtype)
    aa_distribution = aa_mask.to(dtype=dtype) / aa_count
    non_aa_prior = prior.clone()
    non_aa_prior = non_aa_prior.masked_fill(aa_mask, 0.0)
    non_aa_prior = non_aa_prior / non_aa_prior.sum().clamp_min(1.0e-12)
    top_idx = torch.multinomial(
        non_aa_prior.to(torch.float32),
        rows,
        replacement=True,
        generator=generator,
    )
    aa_top = target_max + target_aa > 1.0
    aa_top_idx = _sample_aa_indices(
        rows,
        hand_dim=hand_dim,
        device=device,
        generator=generator,
    )
    top_idx = torch.where(
        aa_top,
        aa_top_idx,
        top_idx,
    )
    target_aa = torch.where(aa_top, target_max, target_aa)

    rest_mask = torch.ones(rows, hand_dim, device=device, dtype=torch.bool)
    rest_mask.scatter_(1, top_idx[:, None], False)
    rest_mask = rest_mask & ~aa_mask.view(1, hand_dim)
    rest_mass = torch.where(
        aa_top,
        1.0 - target_max,
        1.0 - target_max - target_aa,
    ).clamp_min(0.0)
    rest_available = rest_mask.sum(dim=-1).clamp_min(1)
    rest_uniform = rest_mask.to(dtype=dtype) / rest_available[:, None].to(dtype=dtype)

    support_k = torch.ceil(rest_mass / target_max.clamp_min(1.0e-12)).to(torch.long)
    support_k = torch.minimum(support_k.clamp_min(1), rest_available)
    random_scores = torch.rand(
        rows,
        hand_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    ).masked_fill(~rest_mask, -1.0)
    order = random_scores.argsort(dim=-1, descending=True)
    rank = torch.arange(hand_dim, device=device).view(1, hand_dim)
    support_in_order = rank < support_k[:, None]
    low_support_ordered = support_in_order.to(dtype=dtype) / support_k[:, None].to(
        dtype=dtype
    )
    low_support = torch.zeros(rows, hand_dim, device=device, dtype=dtype)
    low_support.scatter_(1, order, low_support_ordered)

    fixed = torch.zeros(rows, hand_dim, device=device, dtype=dtype)
    fixed.scatter_(1, top_idx[:, None], target_max[:, None])
    fixed = fixed + (~aa_top).to(dtype=dtype)[:, None] * target_aa[
        :, None
    ] * aa_distribution.view(1, hand_dim)

    low = torch.zeros(rows, device=device, dtype=dtype)
    high = torch.ones(rows, device=device, dtype=dtype)
    for _ in range(10):
        mid = (low + high) * 0.5
        rest = (1.0 - mid[:, None]) * low_support + mid[:, None] * rest_uniform
        trial = rest * rest_mass[:, None] + fixed
        entropy = _row_entropy(_normalize_rows(trial))
        low = torch.where(entropy < target_entropy, mid, low)
        high = torch.where(entropy >= target_entropy, mid, high)

    mix = (low + high) * 0.5
    rest = (1.0 - mix[:, None]) * low_support + mix[:, None] * rest_uniform
    beliefs = rest * rest_mass[:, None] + fixed
    beliefs = _normalize_rows(beliefs)
    uniform = torch.full((1, hand_dim), 1.0 / hand_dim, device=device, dtype=dtype)
    beliefs = torch.where(uniform_tail[:, None], uniform, beliefs)
    return beliefs.to(dtype=dtype)


def sample_coverage_preflop_belief_rows(
    rows: int,
    *,
    hand_dim: int = PREFLOP_HANDS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample broad-to-near-delta beliefs across the simplex tail bins."""

    device = torch.device(device)
    hand_dim = _validate_hand_dim(hand_dim)
    prior = _hand_prior(hand_dim, device, dtype)
    if rows <= 0:
        return torch.empty(rows, hand_dim, device=device, dtype=dtype)

    edges = torch.cat(
        (
            prior.max().view(1),
            torch.tensor([0.05, 0.15, 0.4, 0.8, 1.0], device=device, dtype=dtype),
        )
    )
    bucket = torch.randint(0, edges.numel() - 1, (rows,), device=device, generator=generator)
    lo = edges[bucket]
    hi = edges[bucket + 1]
    target_max = lo + torch.rand(rows, device=device, dtype=dtype, generator=generator) * (
        hi - lo
    )

    top_idx = torch.randint(
        0,
        hand_dim,
        (rows,),
        device=device,
        generator=generator,
    )
    premium = torch.rand(rows, device=device, generator=generator) < 0.25
    aa_idx = _sample_aa_indices(
        rows,
        hand_dim=hand_dim,
        device=device,
        generator=generator,
    )
    top_idx = torch.where(
        premium,
        aa_idx,
        top_idx,
    )
    rest = prior.view(1, hand_dim).expand(rows, -1).clone()
    rest.scatter_(1, top_idx[:, None], 0.0)
    rest = _normalize_rows(rest)
    beliefs = rest * (1.0 - target_max[:, None])
    beliefs.scatter_(1, top_idx[:, None], target_max[:, None])
    return _normalize_rows(beliefs).to(dtype=dtype)


def sample_topk_preflop_belief_rows(
    rows: int,
    *,
    hand_dim: int = PREFLOP_HANDS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample top-k range-shaped beliefs with variable support size."""

    device = torch.device(device)
    hand_dim = _validate_hand_dim(hand_dim)
    prior = _hand_prior(hand_dim, device, dtype)
    k_values = [1, 2, 4, 8, 16, 32, 64]
    if hand_dim > PREFLOP_HANDS:
        k_values.extend([128, 256, 512])
    k_values.append(hand_dim)
    k_choices = torch.tensor(k_values, device=device)
    k_idx = torch.randint(
        0,
        k_choices.numel(),
        (rows,),
        device=device,
        generator=generator,
    )
    k_per_row = k_choices[k_idx]
    top_mass = 0.75 + 0.249 * torch.rand(
        rows, device=device, dtype=dtype, generator=generator
    )
    top_mass = torch.where(k_per_row == hand_dim, torch.ones_like(top_mass), top_mass)

    random_scores = torch.rand(
        rows,
        hand_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    order = random_scores.argsort(dim=-1, descending=True)
    rank = torch.arange(hand_dim, device=device).view(1, hand_dim)
    selected_ordered = rank < k_per_row[:, None]
    selected = torch.zeros(rows, hand_dim, device=device, dtype=torch.bool)
    selected.scatter_(1, order, selected_ordered)

    selected_weights = torch.zeros(rows, hand_dim, device=device, dtype=dtype)
    ordered_weights = prior.index_select(0, order.reshape(-1)).view(rows, hand_dim)
    ordered_weights = ordered_weights * selected_ordered.to(dtype=dtype)
    ordered_weights = _normalize_rows(ordered_weights)
    selected_weights.scatter_(1, order, ordered_weights)

    rest = prior.view(1, hand_dim).expand(rows, -1).masked_fill(selected, 0.0)
    rest = _normalize_rows(rest) * (1.0 - top_mass[:, None])
    beliefs = rest + selected_weights * top_mass[:, None]
    return _normalize_rows(beliefs).to(dtype=dtype)


def augment_empirical_preflop_belief_rows(
    empirical: torch.Tensor,
    rows: int,
    *,
    hand_dim: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Resample empirical beliefs and perturb by temperature plus prior mixing."""

    device = torch.device(device)
    flat = empirical.reshape(-1, empirical.shape[-1]).to(device=device, dtype=dtype)
    if hand_dim is None:
        hand_dim = int(flat.shape[-1])
    hand_dim = _validate_hand_dim(hand_dim)
    if flat.shape[-1] != hand_dim:
        raise ValueError(f"expected {hand_dim} hands, got {flat.shape[-1]}")
    if flat.shape[0] == 0:
        raise ValueError("empirical beliefs must contain at least one row")
    idx = torch.randint(0, flat.shape[0], (rows,), device=device, generator=generator)
    sampled = flat.index_select(0, idx).clamp_min(1.0e-12)
    log_tau = torch.empty(rows, device=device, dtype=dtype).uniform_(
        -0.7,
        0.7,
        generator=generator,
    )
    tau = log_tau.exp()
    perturbed = sampled.pow(tau[:, None])
    perturbed = _normalize_rows(perturbed)
    prior = _hand_prior(hand_dim, device, dtype)
    mix = 0.15 * torch.rand(rows, device=device, dtype=dtype, generator=generator)
    return _normalize_rows((1.0 - mix[:, None]) * perturbed + mix[:, None] * prior)


def sample_mixed_preflop_belief_rows(
    rows: int,
    *,
    profile: str | BeliefShapeProfile,
    hand_dim: int = PREFLOP_HANDS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    empirical_beliefs: torch.Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Sample rows from the proposed empirical/coverage belief mixture."""

    device = torch.device(device)
    hand_dim = _validate_hand_dim(hand_dim)
    default_weights = {
        "histogram": 0.45,
        "empirical_augmented": 0.20 if empirical_beliefs is not None else 0.0,
        "random": 0.15,
        "coverage": 0.10,
        "topk": 0.10,
    }
    if weights is not None:
        default_weights.update({key: float(value) for key, value in weights.items()})
    if empirical_beliefs is None:
        default_weights["histogram"] += default_weights.pop("empirical_augmented", 0.0)
    names = tuple(default_weights)
    probs = torch.tensor(
        [max(0.0, default_weights[name]) for name in names],
        device=device,
        dtype=torch.float32,
    )
    probs = probs / probs.sum().clamp_min(1.0e-12)
    component_ids = torch.multinomial(probs, rows, replacement=True, generator=generator)

    out = torch.zeros(rows, hand_dim, device=device, dtype=dtype)
    for component_index, name in enumerate(names):
        mask = component_ids == component_index
        if name == "histogram":
            values = sample_histogram_matched_preflop_belief_rows(
                rows,
                profile=profile,
                hand_dim=hand_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        elif name == "empirical_augmented":
            if empirical_beliefs is None:
                values = sample_histogram_matched_preflop_belief_rows(
                    rows,
                    profile=profile,
                    hand_dim=hand_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            else:
                values = augment_empirical_preflop_belief_rows(
                    empirical_beliefs,
                    rows,
                    hand_dim=hand_dim,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
        elif name == "random":
            values = sample_random_preflop_belief_rows(
                rows,
                hand_dim=hand_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        elif name == "coverage":
            values = sample_coverage_preflop_belief_rows(
                rows,
                hand_dim=hand_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        elif name == "topk":
            values = sample_topk_preflop_belief_rows(
                rows,
                hand_dim=hand_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        else:
            raise ValueError(f"unknown belief sampler component {name!r}")
        out = torch.where(mask[:, None], values, out)
    return _normalize_rows(out)


def sample_preflop_beliefs(
    rows: int,
    num_players: int = 2,
    *,
    profile: str | BeliefShapeProfile = "actions_4_7",
    mode: str = "mixed",
    hand_dim: int = PREFLOP_HANDS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
    empirical_beliefs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample preflop beliefs for any player count.

    The profile describes one player-row. The returned tensor has shape
    ``[rows, num_players, hand_dim]`` and is normalized along the hand dimension.
    """

    if rows < 0:
        raise ValueError(f"rows must be non-negative, got {rows}")
    if num_players < 1:
        raise ValueError(f"num_players must be positive, got {num_players}")
    hand_dim = _validate_hand_dim(hand_dim)
    total = int(rows) * int(num_players)
    if mode == "histogram":
        flat = sample_histogram_matched_preflop_belief_rows(
            total,
            profile=profile,
            hand_dim=hand_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    elif mode == "mixed":
        flat = sample_mixed_preflop_belief_rows(
            total,
            profile=profile,
            hand_dim=hand_dim,
            device=device,
            dtype=dtype,
            generator=generator,
            empirical_beliefs=empirical_beliefs,
        )
    elif mode == "coverage":
        flat = sample_coverage_preflop_belief_rows(
            total,
            hand_dim=hand_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    elif mode == "topk":
        flat = sample_topk_preflop_belief_rows(
            total,
            hand_dim=hand_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    elif mode == "random":
        flat = sample_random_preflop_belief_rows(
            total,
            hand_dim=hand_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        raise ValueError(
            "mode must be one of 'mixed', 'histogram', 'coverage', 'topk', or 'random'"
        )
    return flat.reshape(int(rows), int(num_players), hand_dim)


def belief_row_statistics(
    beliefs: torch.Tensor,
    *,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> dict[str, torch.Tensor]:
    """Return per-row belief distribution statistics for preflop beliefs."""

    hand_dim = _validate_hand_dim(int(beliefs.shape[-1]))
    flat = beliefs.reshape(-1, hand_dim).to(torch.float32)
    entropy = -(flat.clamp_min(1.0e-12) * flat.clamp_min(1.0e-12).log()).sum(dim=-1)
    max_class = flat.max(dim=-1).values
    if hand_dim == PREFLOP_HANDS:
        aa_mass = flat[:, AA_CLASS_INDEX]
    else:
        aa_mask = _aa_mask(hand_dim, flat.device)
        aa_mass = flat[:, aa_mask].sum(dim=-1)
    q = torch.tensor(quantiles, device=flat.device, dtype=torch.float32)
    return {
        "max_class": max_class,
        "aa_mass": aa_mass,
        "entropy": entropy,
        "max_class_q": torch.quantile(max_class, q),
        "aa_mass_q": torch.quantile(aa_mass, q),
        "entropy_q": torch.quantile(entropy, q),
    }
