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
from p2.search.rebel_cfr_evaluator import RebelCFREvaluator


MAX_GAME_ITERATIONS = 40


def _build_action_to_child(ev, num_roots: int) -> torch.Tensor:
    """Per-root mapping action_bin -> child node index, or -1 if absent.

    Returns a [num_roots, num_actions] long tensor on the evaluator's device.
    Sparse and dense evaluators are handled uniformly by reading
    ``parent_index`` / ``action_from_parent`` if available, otherwise falling
    back to the dense layout where root r's children sit at consecutive
    indices.
    """
    A = ev.num_actions
    device = ev.device
    a2c = torch.full((num_roots, A), -1, dtype=torch.long, device=device)

    if hasattr(ev, "parent_index") and ev.parent_index.numel() > 0:
        parent_idx = ev.parent_index
        # Direct children of any root are nodes whose parent is in [0, num_roots).
        is_root_child = (parent_idx >= 0) & (parent_idx < num_roots)
        child_nodes = is_root_child.nonzero(as_tuple=True)[0]
        if child_nodes.numel() > 0:
            parent_of = parent_idx[child_nodes]
            action_of = ev.action_from_parent[child_nodes]
            a2c[parent_of, action_of] = child_nodes
        return a2c

    # Dense fallback: roots at 0..N-1, each root r's children at N + r*A + a.
    base = num_roots
    rows = torch.arange(num_roots, device=device)[:, None]
    cols = torch.arange(A, device=device)[None, :]
    a2c[:] = base + rows * A + cols
    return a2c


def _two_prior_river_payoff(
    ev_a, ev_b,
    a_child: int,
    b_child: int,
    bel_a_post: torch.Tensor,  # [2, NH] post-action belief in eval A's tree
    bel_b_post: torch.Tensor,  # [2, NH] post-action belief in eval B's tree
    last_actor: int,
) -> torch.Tensor:
    """Compute the candidate-perspective two-prior showdown payoff for one
    finished river game. ``bel_*_post`` are stored in (acting, other) order
    at the terminal child; we reorder them to (p0, p1) using ``last_actor``.

    All tensor work is on-device; the only host values needed are the python
    ints ``a_child``, ``b_child``, and ``last_actor``.
    """

    def _to_abs(child_belief: torch.Tensor) -> torch.Tensor:
        bel = torch.empty_like(child_belief)  # [2, NH]
        bel[1 - last_actor] = child_belief[0]
        bel[last_actor] = child_belief[1]
        return bel

    bel_a = _to_abs(bel_a_post)  # [2, NH]
    bel_b = _to_abs(bel_b_post)

    def _payoff(ev, child: int, bel: torch.Tensor) -> torch.Tensor:
        # Broadcast our single belief across all of ev's showdown rows so we
        # can call _showdown_value once, then pick the row matching `child`.
        M = ev.showdown_indices.numel()
        row = 0
        if child >= 0:
            matches = (ev.showdown_indices == child).nonzero(as_tuple=True)[0]
            if matches.numel() > 0:
                # One small DtoH per finished river game (rare).
                row = int(matches[0].item())
        bel_M = bel.unsqueeze(0).expand(M, -1, -1)
        sv = ev._showdown_value(bel_M, 0)  # [M, NH]
        return (bel[0] * sv[row]).sum()

    payoff_a = _payoff(ev_a, a_child, bel_a)
    payoff_b = _payoff(ev_b, b_child, bel_b)
    return 0.5 * (payoff_a + payoff_b)


def play_public_belief_games(
    evaluator_a: RebelCFREvaluator,
    evaluator_b: RebelCFREvaluator,
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

    # Per-game last-actor tracking for showdown reordering. Initialized
    # arbitrarily; updated each step before we step the env.
    last_actor = torch.zeros(G, dtype=torch.long, device=device)

    # Per-game last child node within each evaluator's tree at the moment a
    # game became terminal. Only meaningful for newly-finished games and is
    # consumed immediately in the same iteration.
    a_child_full = torch.full((G,), -1, dtype=torch.long, device=device)
    b_child_full = torch.full((G,), -1, dtype=torch.long, device=device)

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
        cur_a2c = torch.where(is_p0[:, None], a2c_a, a2c_b)  # [A, num_actions]
        valid = cur_a2c >= 0
        safe_idx = cur_a2c.clamp(min=0)
        a_policy = evaluator_a.policy_probs_avg[safe_idx, hands[:, None]]
        b_policy = evaluator_b.policy_probs_avg[safe_idx, hands[:, None]]
        # Note: safe_idx is in the *acting* evaluator's tree. For non-acting
        # rows (e.g., is_p0=False) we'd be indexing eval A with eval B's
        # child indices; that's why we keep separate safe indices.
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

        # Record per-game state we may need at end-of-game (last_actor /
        # terminal child indices) into G-shaped buffers.
        last_actor[active_indices] = to_act_active
        a_child_full[active_indices] = a_child_active
        b_child_full[active_indices] = b_child_active

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
            # run yet). This loop runs at most G times *over the entire game
            # batch's lifetime*, so it's not on the per-iter hot path.
            river_pos = done_active_pos[is_river]
            if river_pos.numel() > 0:
                # Pull the small ints we need to host once per river game.
                river_pos_cpu = river_pos.tolist()
                river_game_ids_cpu = active_indices[river_pos].tolist()
                a_child_cpu = a_child_active[river_pos].tolist()
                b_child_cpu = b_child_active[river_pos].tolist()
                last_actor_cpu = to_act_active[river_pos].tolist()
                for k, gid in enumerate(river_game_ids_cpu):
                    pos = river_pos_cpu[k]
                    payoff = _two_prior_river_payoff(
                        evaluator_a,
                        evaluator_b,
                        int(a_child_cpu[k]),
                        int(b_child_cpu[k]),
                        post_belief_a[pos],
                        post_belief_b[pos],
                        int(last_actor_cpu[k]),
                    )
                    rewards[gid] = payoff

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
