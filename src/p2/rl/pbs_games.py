"""Public-belief head-to-head game player.

Plays one trajectory at a time between two CFR evaluators (one per model),
propagating per-evaluator beliefs through the search tree's post-action child
nodes and scoring showdown via two-prior averaging.

Used by the TrueSkill checkpoint tracker. Decoupled from any opponent-pool
class so callers (the tracker) own evaluator construction and weight binding.
"""

from __future__ import annotations

import torch

from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.search.rebel_cfr_evaluator import RebelCFREvaluator


def _root_action_to_child(ev) -> dict[int, int]:
    """Map action-bin -> child node index among the immediate children of
    root (root_idx=0) for either dense or sparse CFR evaluators.

    Dense: children sit at consecutive indices 1..num_actions, indexed by bin.
    Sparse: children of root are nodes whose parent_index == 0; the bin they
    were reached by is in action_from_parent.
    """
    if hasattr(ev, "parent_index") and ev.parent_index.numel() > 0:
        children = (ev.parent_index == 0).nonzero(as_tuple=True)[0]
        return {
            int(ev.action_from_parent[c].item()): int(c.item()) for c in children
        }
    return {a: 1 + a for a in range(ev.num_actions)}


def play_public_belief_games(
    evaluator_a: RebelCFREvaluator,
    evaluator_b: RebelCFREvaluator,
    env_proto: HUNLTensorEnv,
    num_games: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Play public-belief head-to-head games between two pre-built evaluators.

    Both evaluators run CFR at every decision point so each maintains its own
    posterior over the acting player's hand; beliefs propagate through actions
    via the search tree's child nodes. Showdown EV uses two-prior averaging.

    Args:
        evaluator_a: Evaluator wrapping the candidate model
        evaluator_b: Evaluator wrapping the opponent model
        env_proto: Single-env prototype with the right stack/blind/bet-bin
            config; new game envs are spawned via from_proto each game
        num_games: Number of games to play
        generator: RNG for hand and action sampling
        device: Device for computation

    Returns:
        rewards: [num_games] tensor where >0 = candidate win, <0 = loss
        (from candidate / seat-0 perspective)
    """
    rewards = torch.zeros(num_games, device=device, dtype=torch.float32)

    for game_idx in range(num_games):
        env = HUNLTensorEnv.from_proto(env_proto)
        env.reset()

        # Alternate button assignment so candidate plays both seats equally.
        button = game_idx % 2
        env.button[0] = button

        roots = torch.tensor([0], device=device)
        evaluator_a.initialize_subgame(env, roots)
        evaluator_b.initialize_subgame(env, roots)

        max_iterations = 40
        iteration = 0
        next_init_a: torch.Tensor | None = None
        next_init_b: torch.Tensor | None = None
        while not env.done[0].item() and iteration < max_iterations:
            iteration += 1

            if next_init_a is not None:
                evaluator_a.initialize_subgame(env, roots, initial_beliefs=next_init_a)
                evaluator_b.initialize_subgame(env, roots, initial_beliefs=next_init_b)
                next_init_a = None
                next_init_b = None

            evaluator_a.evaluate_cfr(training_mode=False)
            evaluator_b.evaluate_cfr(training_mode=False)

            to_act = env.to_act[0].item()
            current_eval = evaluator_a if to_act == 0 else evaluator_b

            ce_action_to_child = _root_action_to_child(current_eval)
            a_action_to_child = _root_action_to_child(evaluator_a)
            b_action_to_child = _root_action_to_child(evaluator_b)

            hand = torch.multinomial(
                current_eval.beliefs[0, 0], num_samples=1, generator=generator
            ).item()
            num_actions = current_eval.num_actions
            action_probs = torch.zeros(
                num_actions, device=device, dtype=current_eval.policy_probs_avg.dtype
            )
            for a, child in ce_action_to_child.items():
                action_probs[a] = current_eval.policy_probs_avg[child, hand]
            action = torch.multinomial(
                action_probs, num_samples=1, generator=generator
            ).item()

            a_child = a_action_to_child.get(action)
            b_child = b_action_to_child.get(action)
            next_init_a = (
                evaluator_a.beliefs[a_child : a_child + 1].clone()
                if a_child is not None
                else evaluator_a.beliefs[0:1].clone()
            )
            next_init_b = (
                evaluator_b.beliefs[b_child : b_child + 1].clone()
                if b_child is not None
                else evaluator_b.beliefs[0:1].clone()
            )

            env_rewards, _, _ = env.step_bins(
                torch.full((1,), action, device=device, dtype=torch.long)
            )

        assert env.done[0].item(), "Environment should be done"

        if env.street[0].item() == 4:
            # Two-prior averaged showdown EV from p0's perspective using each
            # evaluator's post-action child beliefs (already carrying the
            # propagated posteriors). Reorder (acting, other) at the terminal
            # child back to absolute (p0, p1) using the last actor.
            last_actor = to_act

            def _to_abs(child_belief: torch.Tensor) -> torch.Tensor:
                bel = torch.empty_like(child_belief[0])  # [2, 1326]
                bel[1 - last_actor] = child_belief[0, 0]
                bel[last_actor] = child_belief[0, 1]
                return bel.unsqueeze(0)

            bel_a = _to_abs(next_init_a)
            bel_b = _to_abs(next_init_b)
            sv_a = evaluator_a._showdown_value(bel_a, 0)
            sv_b = evaluator_b._showdown_value(bel_b, 0)
            payoff_a = (bel_a[0, 0] * sv_a[0]).sum()
            payoff_b = (bel_b[0, 0] * sv_b[0]).sum()
            rewards[game_idx] = 0.5 * (
                float(payoff_a.item()) + float(payoff_b.item())
            )
        else:
            rewards[game_idx] = env_rewards[0].item()

    return rewards
