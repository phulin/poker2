"""Real-hand, real-chip duplicate (mirrored) match player.

This replaces public-belief-space scoring. Hole cards are really dealt, the
env resolves showdowns, and the only payoff is the env's chip reward. No
range-EV, no belief-dependent scoring, so results are comparable across models
with different belief calibration.

Variance reduction, two independent mechanisms:

1. Duplicate pairing. Every pair of games shares one pre-dealt deck. Game
   ``2i`` deals hole cards ``(H0, H1)`` to seats ``(0, 1)`` with button ``b``;
   game ``2i+1`` deals ``(H1, H0)`` with button ``1 - b`` and the seat stacks
   swapped. Agent A sits in seat 0 in both, so in the second half it holds
   exactly what agent B held, from exactly the position agent B had. Agent A's
   two rewards are summed (we report half the sum, i.e. a per-game figure):
   if both agents played the same strategy the pair scores exactly zero
   regardless of how the cards ran out.

2. Common random numbers. Action sampling is inverse-CDF against a uniform
   tensor of shape ``[num_pairs, MAX_DECISIONS]`` indexed by a per-game
   decision counter, shared between the two halves of a pair. ``multinomial``
   cannot consume a supplied uniform, so it is not used. Coupling is only
   approximate: the CFR solve itself is stochastic and GPU-nondeterministic,
   and the halves decouple as soon as their action sequences diverge.

All ``2 * num_pairs`` games run in one batched env; there is no Python loop
over pairs or games.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

from p2.env.card_utils import combo_lookup_tensor
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.eval.agents import FOLD_BIN, MatchAgent
from p2.eval.records import (
    TERMINAL_FOLD,
    TERMINAL_SHOWDOWN,
    TERMINAL_TIMEOUT,
    GameBatchTensors,
    GameRecord,
    RecordWriter,
)

# Cards forced per game: 2 + 2 hole cards, flop, turn, river.
DECK_CARDS = 9
# Hard cap on decision rounds per game; matches pbs_games' game-iteration cap.
MAX_DECISIONS = 40


@dataclass
class MatchResult:
    """Duplicate-scored outcome of a match between agent A and agent B."""

    num_pairs: int
    num_games: int
    # Per-pair duplicate score for agent A: 0.5 * (reward_half0 + reward_half1).
    pair_diff_bb: torch.Tensor
    pair_diff_stacks: torch.Tensor
    # Raw per-game agent-A rewards, [2 * num_pairs].
    reward_bb: torch.Tensor
    reward_stacks: torch.Tensor
    mean_bb_per_100: float
    se_bb_per_100: float
    records: list[GameRecord]

    def summary(self) -> str:
        return (
            f"{self.mean_bb_per_100:+.2f} +/- {self.se_bb_per_100:.2f} bb/100 "
            f"over {self.num_pairs} duplicate pairs ({self.num_games} games)"
        )


def build_mirrored_decks(
    num_pairs: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Return ``[2 * num_pairs, 9]`` forced decks for duplicate pairs.

    Deck layout matches ``HUNLTensorEnv.reset``: cards 0-1 are seat 0's hole
    cards, 2-3 are seat 1's, and 4-8 are the flop/turn/river. Odd rows swap the
    two hole-card pairs and keep the board identical.
    """
    scores = torch.rand(num_pairs, 52, generator=generator, device=device)
    pair_decks = torch.topk(scores, DECK_CARDS, dim=1).indices  # distinct cards
    swap = torch.tensor([2, 3, 0, 1, 4, 5, 6, 7, 8], device=device, dtype=torch.long)
    mirrored = pair_decks.index_select(1, swap)
    return torch.stack((pair_decks, mirrored), dim=1).reshape(2 * num_pairs, DECK_CARDS)


def _mirrored_buttons(
    num_pairs: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    base = torch.randint(
        0, 2, (num_pairs,), generator=generator, device=device, dtype=torch.long
    )
    return torch.stack((base, 1 - base), dim=1).reshape(2 * num_pairs)


def _mirrored_starting_stacks(
    env: HUNLTensorEnv, num_pairs: int
) -> Optional[torch.Tensor]:
    """Per-pair starting stacks, swapped between seats in the mirrored half.

    ``None`` for the fixed-stack mode, where the env's own sampling is already
    identical for every game.
    """
    if env.stack_mode == "fixed":
        return None
    pair_stacks = env._sample_starting_stacks(num_pairs)  # [P, 2]
    mirrored = pair_stacks.flip(1)
    return torch.stack((pair_stacks, mirrored), dim=1).reshape(2 * num_pairs, 2)


def _sample_actions_inverse_cdf(
    probs: torch.Tensor, uniforms: torch.Tensor
) -> torch.Tensor:
    """Inverse-CDF sample of ``[A, B]`` weights against supplied ``[A]`` uniforms."""
    cdf = probs.cumsum(dim=1)
    totals = cdf[:, -1:].clamp(min=torch.finfo(cdf.dtype).tiny)
    cdf = cdf / totals
    # Guard the final bucket against fp round-off leaving cdf[-1] < u.
    cdf[:, -1] = 1.0
    picks = torch.searchsorted(cdf.contiguous(), uniforms[:, None].contiguous())
    return picks.squeeze(1).clamp(max=probs.shape[1] - 1)


def play_duplicate_match(
    agent_a: MatchAgent,
    agent_b: MatchAgent,
    env_proto: HUNLTensorEnv,
    num_pairs: int,
    seed: int,
    device: torch.device,
    recorder: RecordWriter | None = None,
    eval_id: str = "eval",
    extra_manifest: dict[str, Any] | None = None,
) -> MatchResult:
    """Play ``num_pairs`` duplicate pairs (``2 * num_pairs`` games) in one batch.

    Agent A occupies seat 0 in every game; the mirroring is done in the cards
    and the button, which is equivalent to swapping seats and keeps the env's
    p0-perspective reward directly usable as agent A's reward.
    """
    device = torch.device(device)
    if env_proto.device != device:
        raise ValueError(
            f"env_proto is on {env_proto.device} but the match device is {device}"
        )
    num_games = 2 * num_pairs
    if num_pairs <= 0:
        empty = torch.zeros(0, device=device)
        return MatchResult(
            num_pairs=0,
            num_games=0,
            pair_diff_bb=empty,
            pair_diff_stacks=empty,
            reward_bb=empty,
            reward_stacks=empty,
            mean_bb_per_100=0.0,
            se_bb_per_100=float("nan"),
            records=[],
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    env = HUNLTensorEnv.from_proto(env_proto, num_envs=num_games)
    # Isolate the match from the prototype's shared RNG stream so a match is
    # reproducible from `seed` alone. Only stack sampling consumes it; the deck
    # and the button are forced below.
    env.rng = generator
    env.reset(
        force_button=_mirrored_buttons(num_pairs, generator, device),
        force_deck=build_mirrored_decks(num_pairs, generator, device),
        force_starting_stacks=_mirrored_starting_stacks(env, num_pairs),
    )

    # Real dealt hands as 0..1325 combo indices, per seat.
    lookup = combo_lookup_tensor(device=device)
    hole_combo = lookup[env.hole_indices[:, :, 0], env.hole_indices[:, :, 1]]  # [G, 2]

    # Common random numbers: one uniform per (pair, decision index), shared by
    # both halves of the pair.
    uniforms = torch.rand(num_pairs, MAX_DECISIONS, generator=generator, device=device)

    agent_a.begin_match(env, seat=0)
    agent_b.begin_match(env, seat=1)

    rewards = torch.zeros(num_games, dtype=env.float_dtype, device=device)
    num_decisions = torch.zeros(num_games, dtype=torch.long, device=device)
    decision_counter = torch.zeros(num_games, dtype=torch.long, device=device)
    pair_of_game = torch.arange(num_games, device=device, dtype=torch.long) // 2

    active_indices = torch.arange(num_games, device=device, dtype=torch.long)
    folded = torch.zeros(num_games, dtype=torch.bool, device=device)

    for _ in range(MAX_DECISIONS):
        if active_indices.numel() == 0:
            break

        agent_a.observe(env, active_indices)
        agent_b.observe(env, active_indices)

        bin_amounts, legal_full = env.legal_bins_amounts_and_mask()
        legal = legal_full[active_indices]
        to_act = env.to_act[active_indices]

        probs_a = agent_a.action_probs(
            env, active_indices, hole_combo[active_indices, 0], legal
        )
        probs_b = agent_b.action_probs(
            env, active_indices, hole_combo[active_indices, 1], legal
        )
        probs = torch.where((to_act == 0)[:, None], probs_a, probs_b)
        probs = probs.clamp(min=0.0) * legal.to(probs.dtype)
        # If an agent put all its mass on illegal bins (or on bins its search
        # tree omitted), fall back to uniform over the legal bins.
        legal_uniform = legal.to(probs.dtype)
        probs = torch.where(probs.sum(dim=1, keepdim=True) > 0, probs, legal_uniform)

        counters = decision_counter[active_indices].clamp(max=MAX_DECISIONS - 1)
        shared_uniforms = uniforms[pair_of_game[active_indices], counters]
        actions = _sample_actions_inverse_cdf(probs, shared_uniforms)

        full_actions = torch.full((num_games,), -1, dtype=torch.long, device=device)
        full_actions[active_indices] = actions

        # `env.has_folded` is never written by HUNLTensorEnv.step, so track the
        # fold terminal ourselves from the bins actually submitted.
        folded |= full_actions == FOLD_BIN
        was_done = env.done.clone()
        step_rewards, _, _ = env.step_bins(
            full_actions, bin_amounts=bin_amounts, legal_masks=legal_full
        )
        rewards = torch.where(env.done & ~was_done, step_rewards, rewards)

        num_decisions[active_indices] += 1
        decision_counter[active_indices] += 1

        keep_mask = ~env.done[active_indices]
        agent_a.commit(actions, keep_mask)
        agent_b.commit(actions, keep_mask)
        active_indices = active_indices[keep_mask]

    # ---------------------------------------------------------------- scoring
    terminal_type = torch.where(
        ~env.done,
        torch.full_like(env.winner, TERMINAL_TIMEOUT),
        torch.where(
            folded,
            torch.full_like(env.winner, TERMINAL_FOLD),
            torch.full_like(env.winner, TERMINAL_SHOWDOWN),
        ),
    )

    effective_stack = env.scale.round().long()
    reward_bb = rewards * env.scale / float(env.bb)
    pair_reward_bb = reward_bb.view(num_pairs, 2)
    pair_diff_bb = 0.5 * (pair_reward_bb[:, 0] + pair_reward_bb[:, 1])
    pair_reward_stacks = rewards.view(num_pairs, 2)
    pair_diff_stacks = 0.5 * (pair_reward_stacks[:, 0] + pair_reward_stacks[:, 1])

    mean_bb = float(pair_diff_bb.mean().item()) if num_pairs > 0 else 0.0
    if num_pairs > 1:
        std_bb = float(pair_diff_bb.std(unbiased=True).item())
        se_bb = std_bb / math.sqrt(num_pairs)
    else:
        se_bb = float("nan")

    half = torch.arange(num_games, device=device, dtype=torch.long) % 2
    batch = GameBatchTensors(
        pair_id=pair_of_game,
        pair_half=half,
        seat_a=torch.zeros(num_games, dtype=torch.long, device=device),
        button=env.button,
        terminal_type=terminal_type,
        street=env.street,
        final_pot=env.pot,
        num_decisions=num_decisions,
        winner=env.winner,
        effective_stack=effective_stack,
        hole_a_combo=hole_combo[:, 0],
        hole_b_combo=hole_combo[:, 1],
        reward_a_stacks=rewards,
        board=env.board_indices,
        hole_a=env.hole_indices[:, 0, :],
        hole_b=env.hole_indices[:, 1, :],
    )
    records = batch.to_records(
        eval_id=eval_id,
        agent_a_id=agent_a.identity.id,
        agent_b_id=agent_b.identity.id,
        big_blind=env.bb,
        seed=int(seed),
        agent_a_fidelity=agent_a.search_fidelity(),
        agent_b_fidelity=agent_b.search_fidelity(),
    )

    if recorder is not None:
        recorder.write_manifest(
            match_manifest(
                agent_a=agent_a,
                agent_b=agent_b,
                env=env,
                num_pairs=num_pairs,
                seed=seed,
                eval_id=eval_id,
                extra=extra_manifest,
            )
        )
        recorder.append(records)

    return MatchResult(
        num_pairs=num_pairs,
        num_games=num_games,
        pair_diff_bb=pair_diff_bb,
        pair_diff_stacks=pair_diff_stacks,
        reward_bb=reward_bb,
        reward_stacks=rewards,
        mean_bb_per_100=100.0 * mean_bb,
        se_bb_per_100=100.0 * se_bb,
        records=records,
    )


def match_manifest(
    *,
    agent_a: MatchAgent,
    agent_b: MatchAgent,
    env: HUNLTensorEnv,
    num_pairs: int,
    seed: int,
    eval_id: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run configuration sidecar for a duplicate match."""
    manifest: dict[str, Any] = {
        "eval_id": eval_id,
        "scoring": "duplicate_real_hand_chips",
        "num_pairs": num_pairs,
        "num_games": 2 * num_pairs,
        "seed": seed,
        "max_decisions": MAX_DECISIONS,
        "agent_a": agent_a.identity.to_dict(),
        "agent_b": agent_b.identity.to_dict(),
        "agent_a_fidelity": agent_a.search_fidelity(),
        "agent_b_fidelity": agent_b.search_fidelity(),
        "env": {
            "sb": env.sb,
            "bb": env.bb,
            "mean_stack": env.mean_stack,
            "stack_mode": env.stack_mode,
            "bet_bins": list(env.default_bet_bins),
            "flop_showdown": env.flop_showdown,
        },
    }
    if extra:
        manifest["extra"] = extra
    return manifest
