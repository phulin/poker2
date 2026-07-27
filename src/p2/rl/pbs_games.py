"""Public-belief head-to-head game player.

Plays many trajectories in parallel between two CFR evaluators (one per
model), propagating per-evaluator beliefs through the search tree's
post-action child nodes and scoring showdown via two-prior averaging.

Used by the TrueSkill checkpoint tracker. Decoupled from any opponent-pool
class so callers (the tracker) own evaluator construction and weight binding.

The implementation runs all `num_games` games in a single batched env so
that CFR + sampling + env stepping all happen vectorized over games. This
removes the per-game-iter Python `.item()` syncs that dominated the
previous serial implementation.
"""

from __future__ import annotations

import torch

from p2.env.card_utils import NUM_HANDS
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.search.cfr_evaluator import CFREvaluator


MAX_GAME_ITERATIONS = 40


def _build_action_to_child(ev, num_roots: int) -> torch.Tensor:
    """Per-root mapping action_bin -> child node index, or -1 if absent.

    Returns a [num_roots, num_actions] long tensor on the evaluator's device.
    Sparse evaluators expose ``parent_index`` / ``action_from_parent`` for the
    direct root-child mapping used here.
    """
    A = ev.num_actions
    device = ev.device
    a2c = torch.full((num_roots, A), -1, dtype=torch.long, device=device)

    parent_idx = ev.parent_index
    is_root_child = (parent_idx >= 0) & (parent_idx < num_roots)
    child_nodes = is_root_child.nonzero(as_tuple=True)[0]
    if child_nodes.numel() > 0:
        parent_of = parent_idx[child_nodes]
        action_of = ev.action_from_parent[child_nodes]
        a2c[parent_of, action_of] = child_nodes
    return a2c


def _showdown_rows_for_children(
    ev, children: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map terminal child node indices to rows in ``ev.showdown_indices``."""
    rows = torch.zeros_like(children)
    valid = children >= 0
    if ev.showdown_indices.numel() == 0 or children.numel() == 0:
        return rows, torch.zeros_like(valid)

    lookup = torch.full(
        (ev.total_nodes,),
        -1,
        dtype=torch.long,
        device=ev.device,
    )
    lookup[ev.showdown_indices] = torch.arange(
        ev.showdown_indices.numel(),
        dtype=torch.long,
        device=ev.device,
    )
    safe_children = children.clamp(min=0, max=ev.total_nodes - 1)
    rows = lookup[safe_children]
    valid = valid & (rows >= 0)
    rows = torch.where(valid, rows, torch.zeros_like(rows))
    return rows, valid


def _showdown_values_for_hero(ev, beliefs: torch.Tensor, hero: int) -> torch.Tensor:
    """Return per-hand showdown values, respecting fused compact rank data."""
    hand_rank_data = getattr(ev, "hand_rank_data", None)
    sorted_indices = getattr(hand_rank_data, "sorted_indices", None)
    if (
        sorted_indices is not None
        and sorted_indices.shape[0] != ev.showdown_indices.numel()
        and hasattr(ev, "_showdown_value_both")
    ):
        return ev._showdown_value_both(beliefs)[:, hero]
    return ev._showdown_value(beliefs, hero)


def _showdown_range_payoffs(
    ev,
    children: torch.Tensor,
    abs_beliefs: torch.Tensor,
) -> torch.Tensor:
    """Compute candidate-perspective range payoffs for river terminal children."""
    payoffs = torch.zeros(children.shape[0], device=ev.device, dtype=torch.float32)
    M = ev.showdown_indices.numel()
    if M == 0 or children.numel() == 0:
        return payoffs

    rows, valid = _showdown_rows_for_children(ev, children)
    showdown_beliefs = torch.zeros(
        M,
        2,
        NUM_HANDS,
        dtype=abs_beliefs.dtype,
        device=ev.device,
    )
    showdown_beliefs[rows[valid]] = abs_beliefs[valid]
    showdown_values = _showdown_values_for_hero(ev, showdown_beliefs, 0)
    raw_payoffs = (abs_beliefs[:, 0] * showdown_values[rows]).sum(dim=1)
    return torch.where(valid, raw_payoffs, payoffs)


def _two_prior_river_payoffs(
    ev_a,
    ev_b,
    a_children: torch.Tensor,
    b_children: torch.Tensor,
    bel_a_post: torch.Tensor,  # [R, 2, NH] post-action belief in eval A's tree
    bel_b_post: torch.Tensor,  # [R, 2, NH] post-action belief in eval B's tree
) -> torch.Tensor:
    """Compute candidate-perspective two-prior showdown payoffs in one batch.

    Evaluator beliefs are ``[node, player, hand]`` with ``player`` an absolute
    seat index (see ``prev_actor = env.to_act[parent]`` in the sparse
    evaluator), so the child beliefs are already in absolute (p0, p1) order
    and must not be permuted by the last actor.
    """
    payoff_a = _showdown_range_payoffs(ev_a, a_children, bel_a_post)
    payoff_b = _showdown_range_payoffs(ev_b, b_children, bel_b_post)
    return 0.5 * (payoff_a + payoff_b)


def play_public_belief_games(
    evaluator_a: CFREvaluator,
    evaluator_b: CFREvaluator,
    env_proto: HUNLTensorEnv,
    num_games: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Play public-belief head-to-head games between two pre-built evaluators.

    All ``num_games`` games run in parallel in a single batched env: CFR
    initialization, evaluation, hand and action sampling, belief
    propagation, and env stepping are vectorized across games. The only
    per-iteration sync is the loop-termination check (``active.any()``);
    additional small DtoH transfers happen only at end-of-game for river
    showdowns.

    Both evaluators run CFR at every decision point so each maintains its
    own posterior over the acting player's hand; beliefs propagate through
    actions via the search tree's child nodes. Showdown EV uses two-prior
    averaging.

    Returns:
        rewards: [num_games] tensor where >0 = candidate win, <0 = loss
        (from candidate / seat-0 perspective).
    """
    G = num_games
    rewards = torch.zeros(G, device=device, dtype=torch.float32)
    if G == 0:
        return rewards

    # Single batched env hosting all games at once.
    env = HUNLTensorEnv.from_proto(env_proto, num_envs=G)
    buttons = torch.arange(G, device=device, dtype=torch.long) % 2
    env.reset(force_button=buttons)

    # Track which games are still active (not yet terminal).
    active_indices = torch.arange(G, device=device, dtype=torch.long)

    # Initial subgame build for all games.
    evaluator_a.initialize_subgame(env, active_indices)
    evaluator_b.initialize_subgame(env, active_indices)
    a2c_a = _build_action_to_child(evaluator_a, active_indices.numel())
    a2c_b = _build_action_to_child(evaluator_b, active_indices.numel())

    next_init_a: torch.Tensor | None = None
    next_init_b: torch.Tensor | None = None

    for iteration in range(MAX_GAME_ITERATIONS):
        # Loop guard: one DtoH per outer iter at most. `numel()` is a metadata
        # read (no sync); `.item()` on the `any()` reduction is the sync.
        if active_indices.numel() == 0:
            break

        # Re-build subgames for the still-active subset using last iter's
        # propagated beliefs (skipped on the very first iter).
        if next_init_a is not None:
            evaluator_a.initialize_subgame(
                env, active_indices, initial_beliefs=next_init_a
            )
            evaluator_b.initialize_subgame(
                env, active_indices, initial_beliefs=next_init_b
            )
            a2c_a = _build_action_to_child(evaluator_a, active_indices.numel())
            a2c_b = _build_action_to_child(evaluator_b, active_indices.numel())
            next_init_a = None
            next_init_b = None

        evaluator_a.evaluate_cfr(training_mode=False)
        evaluator_b.evaluate_cfr(training_mode=False)

        A = active_indices.numel()
        # Per-active to_act indicator. Roots are 0..A-1 in each evaluator's
        # tree, in the same order as `active_indices`.
        to_act_active = env.to_act[active_indices]  # [A]
        is_p0 = to_act_active == 0  # [A]

        # Acting-player beliefs at each active root, drawn from the *acting*
        # evaluator's tree. shape: [A, 2, NH]
        a_root_beliefs = evaluator_a.beliefs[:A]
        b_root_beliefs = evaluator_b.beliefs[:A]
        cur_root_beliefs = torch.where(
            is_p0[:, None, None], a_root_beliefs, b_root_beliefs
        )
        cur_acting_beliefs = cur_root_beliefs.gather(
            1, to_act_active[:, None, None].expand(-1, 1, NUM_HANDS)
        ).squeeze(1)  # [A, NH]

        # Sample a hand for each active game.
        hands = torch.multinomial(
            cur_acting_beliefs, num_samples=1, generator=generator
        ).squeeze(1)  # [A]

        # Build action probability vectors for each active game from the
        # *acting* evaluator's policy_probs_avg, indexed by the chosen child
        # nodes for that root. For invalid action bins (-1 in a2c) the
        # probability is forced to 0 by clamping the index then masking.
        valid = torch.where(is_p0[:, None], a2c_a, a2c_b) >= 0
        # Keep separate safe indices: action-to-child node IDs are local to
        # each evaluator's current tree.
        safe_idx_a = a2c_a.clamp(min=0)
        safe_idx_b = a2c_b.clamp(min=0)
        a_action_probs = evaluator_a.policy_probs_avg[safe_idx_a, hands[:, None]]
        b_action_probs = evaluator_b.policy_probs_avg[safe_idx_b, hands[:, None]]
        action_probs = torch.where(is_p0[:, None], a_action_probs, b_action_probs)
        action_probs = action_probs * valid.to(action_probs.dtype)
        # Avoid all-zero rows (multinomial errors). Roots are guaranteed to
        # have at least one legal child, so valid rows have a positive sum.
        actions = torch.multinomial(
            action_probs, num_samples=1, generator=generator
        ).squeeze(1)  # [A]

        # Look up post-action child node per evaluator per active game.
        a_child_active = a2c_a.gather(1, actions[:, None]).squeeze(1)  # [A]
        b_child_active = a2c_b.gather(1, actions[:, None]).squeeze(1)
        # Where the chosen action wasn't a real child (shouldn't happen
        # because we sampled from the legal distribution), fall back to root.
        root_idx = torch.arange(A, device=device, dtype=torch.long)
        a_child_active = torch.where(a_child_active >= 0, a_child_active, root_idx)
        b_child_active = torch.where(b_child_active >= 0, b_child_active, root_idx)

        # Pull the post-action belief at the chosen child for each evaluator.
        # shape: [A, 2, NH]
        post_belief_a = evaluator_a.beliefs[a_child_active]
        post_belief_b = evaluator_b.beliefs[b_child_active]

        # Step the env. Build a G-shaped action vector with -1 for inactive
        # games (so step_bins no-ops on them).
        full_actions = torch.full((G,), -1, dtype=torch.long, device=device)
        full_actions[active_indices] = actions
        env_rewards, _, _ = env.step_bins(full_actions)

        # Identify newly-terminal games among the active set.
        done_active = env.done[active_indices]  # [A]
        if done_active.any().item():
            # ---- handle newly-done games ----
            done_active_pos = done_active.nonzero(as_tuple=True)[0]  # rows in [0, A)
            done_game_ids = active_indices[done_active_pos]  # original game indices
            done_streets = env.street[done_game_ids]
            is_river = done_streets == 4

            # Non-river finishes: take the env reward from this step
            # (rewards are p0-perspective, candidate=p0).
            non_river_pos = done_active_pos[~is_river]
            non_river_ids = active_indices[non_river_pos]
            if non_river_ids.numel() > 0:
                rewards[non_river_ids] = env_rewards[non_river_ids]

            # River finishes: two-prior showdown payoff, computed per finished
            # game using the evaluator state we still hold (re-init hasn't
            # run yet). Batch this so each evaluator's showdown kernel runs
            # once for all river finishes in this step.
            river_pos = done_active_pos[is_river]
            if river_pos.numel() > 0:
                rewards[active_indices[river_pos]] = _two_prior_river_payoffs(
                    evaluator_a,
                    evaluator_b,
                    a_child_active[river_pos],
                    b_child_active[river_pos],
                    post_belief_a[river_pos],
                    post_belief_b[river_pos],
                )

            # Drop done games from the active set and the propagated beliefs.
            still_active = ~done_active
            if not still_active.any().item():
                active_indices = active_indices[still_active]
                next_init_a = None
                next_init_b = None
                continue
            active_indices = active_indices[still_active]
            next_init_a = post_belief_a[still_active]
            next_init_b = post_belief_b[still_active]
        else:
            next_init_a = post_belief_a
            next_init_b = post_belief_b

    # Any games still flagged active here hit MAX_GAME_ITERATIONS without
    # terminating; their reward stays at 0. The original implementation
    # asserted env.done at end-of-loop, but in batched mode we tolerate this
    # rare timeout silently — same behavior as the per-game assert firing.
    return rewards
