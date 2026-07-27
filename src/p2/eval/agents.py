"""Agents that can play real-hand, real-chip heads-up matches.

Every agent implements :class:`MatchAgent`, which is deliberately narrow so
that scripted bots need no model, no CFR evaluator, and no beliefs at all.

The search agent (:class:`SearchAgent`) keeps the public-belief plumbing from
``p2.rl.pbs_games`` -- it re-initializes its subgame every decision round and
propagates its beliefs through the *chosen public action*'s child node -- but
the beliefs are now only ever a model input. Scoring uses real chips, and the
agent picks its action from ``policy_probs_avg[child_node, actual_hand]``,
i.e. the hand it was actually dealt, never a hand sampled from its belief.

Round protocol, driven by ``p2.eval.duplicate_match``:

    begin_match(env, seat)          once, after env.reset()
    loop:
        observe(env, active)        rebuild subgame + run CFR for active games
        action_probs(...)           -> [A, num_bins] action weights
        commit(actions, keep)       propagate beliefs through the taken action
                                    and keep the rows that are still live
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from p2.env.card_utils import (
    NUM_HANDS,
    PREFLOP_HANDS,
    combo_to_preflop_class_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv

FOLD_BIN = 0
CHECK_CALL_BIN = 1


@dataclass(frozen=True)
class AgentIdentity:
    """Stable identity written into every game record.

    ``checkpoint``/``step`` identify a learned agent; scripted bots use only
    ``name``.
    """

    name: str
    kind: str = "scripted"
    checkpoint: Optional[str] = None
    step: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        if self.checkpoint is None and self.step is None:
            return self.name
        parts = [self.name]
        if self.checkpoint is not None:
            parts.append(str(self.checkpoint))
        if self.step is not None:
            parts.append(f"step={self.step}")
        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "checkpoint": self.checkpoint,
            "step": self.step,
            **({"extra": self.extra} if self.extra else {}),
        }


class MatchAgent:
    """Interface for anything that can sit in a seat of a real-hand match."""

    identity: AgentIdentity

    # -- identity / provenance ------------------------------------------------

    def search_fidelity(self) -> dict[str, Any]:
        """Search settings recorded per game so evals stay comparable later."""
        return {}

    # -- match lifecycle ------------------------------------------------------

    def begin_match(self, env: HUNLTensorEnv, seat: int) -> None:
        """Called once after ``env.reset()``. ``seat`` is this agent's seat id."""
        self.seat = int(seat)

    def observe(self, env: HUNLTensorEnv, active_indices: torch.Tensor) -> None:
        """Called at the start of every decision round for the live games.

        Called for *all* live games, not just the ones where this agent acts,
        because a search agent must keep its beliefs in sync with the public
        state regardless of who is to act.
        """

    def action_probs(
        self,
        env: HUNLTensorEnv,
        active_indices: torch.Tensor,
        hands: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[A, num_bins]`` non-negative action weights.

        Args:
            active_indices: ``[A]`` env rows still live this round.
            hands: ``[A]`` 0..1325 combo index of *this agent's* real hole cards.
            legal_mask: ``[A, num_bins]`` bool mask of legal bins.

        Rows where this agent is not to act are ignored by the caller, so
        implementations may compute all rows unconditionally.
        """
        raise NotImplementedError

    def commit(self, actions: torch.Tensor, keep_mask: torch.Tensor) -> None:
        """Observe the public action taken in each live game.

        Args:
            actions: ``[A]`` bin taken in each game from the last ``observe``.
            keep_mask: ``[A]`` bool, games that remain live for the next round.
        """


# ------------------------------------------------------------------ scripted


class _ScriptedAgent(MatchAgent):
    """Base for bots that need no model, no evaluator, and no beliefs."""

    def __init__(self, name: str):
        self.identity = AgentIdentity(name=name, kind="scripted")

    def _onehot(self, legal_mask: torch.Tensor, bin_index: int) -> torch.Tensor:
        probs = torch.zeros_like(legal_mask, dtype=torch.float32)
        probs[:, bin_index] = 1.0
        return probs


class FoldAgent(_ScriptedAgent):
    """Always folds; checks when folding is not legal (nothing to call)."""

    def __init__(self, name: str = "fold_bot"):
        super().__init__(name)

    def action_probs(self, env, active_indices, hands, legal_mask):
        fold = self._onehot(legal_mask, FOLD_BIN)
        check = self._onehot(legal_mask, CHECK_CALL_BIN)
        return torch.where(legal_mask[:, FOLD_BIN : FOLD_BIN + 1], fold, check)


class CallAgent(_ScriptedAgent):
    """Always checks or calls. Check/call is always legal."""

    def __init__(self, name: str = "call_bot"):
        super().__init__(name)

    def action_probs(self, env, active_indices, hands, legal_mask):
        return self._onehot(legal_mask, CHECK_CALL_BIN)


class RandomAgent(_ScriptedAgent):
    """Uniform over the legal bins."""

    def __init__(self, name: str = "random_bot"):
        super().__init__(name)

    def action_probs(self, env, active_indices, hands, legal_mask):
        return legal_mask.to(torch.float32)


# -------------------------------------------------------------------- search


def _build_action_to_child(evaluator, num_roots: int) -> torch.Tensor:
    """Per-root ``action_bin -> child node index`` map (-1 when absent).

    Same construction as ``p2.rl.pbs_games._build_action_to_child``; duplicated
    rather than imported because that module is owned elsewhere and this one
    must not depend on its PBS scoring path.
    """
    device = evaluator.device
    a2c = torch.full(
        (num_roots, evaluator.num_actions), -1, dtype=torch.long, device=device
    )
    parent_idx = evaluator.parent_index
    is_root_child = (parent_idx >= 0) & (parent_idx < num_roots)
    child_nodes = is_root_child.nonzero(as_tuple=True)[0]
    if child_nodes.numel() > 0:
        a2c[parent_idx[child_nodes], evaluator.action_from_parent[child_nodes]] = (
            child_nodes
        )
    return a2c


class SearchAgent(MatchAgent):
    """CFR-search agent playing its real dealt hand.

    Owns a pre-built CFR evaluator (build it exactly like
    ``TrueSkillTracker._build_eval``). Beliefs are maintained purely as search
    input: they are seeded uniform, propagated through the chosen public
    action's child node each round, and never used for scoring or for choosing
    an action.
    """

    def __init__(self, evaluator, identity: AgentIdentity | None = None):
        self.evaluator = evaluator
        self.identity = identity or AgentIdentity(name="search", kind="search")
        self.seat = 0
        self._pending_beliefs: torch.Tensor | None = None
        self._a2c: torch.Tensor | None = None
        self._num_roots = 0

    # -- provenance -----------------------------------------------------------

    def search_fidelity(self) -> dict[str, Any]:
        ev = self.evaluator
        fidelity: dict[str, Any] = {"evaluator": type(ev).__name__}
        for attr in (
            "cfr_iterations",
            "warm_start_iterations",
            "dcfr_delay",
            "max_depth",
            "depth",
            "num_actions",
            "hand_dim",
        ):
            value = getattr(ev, attr, None)
            if isinstance(value, (int, float)):
                fidelity[attr] = value
        return fidelity

    # -- lifecycle ------------------------------------------------------------

    def begin_match(self, env: HUNLTensorEnv, seat: int) -> None:
        self.seat = int(seat)
        self._pending_beliefs = None
        self._a2c = None
        self._num_roots = 0

    def observe(self, env: HUNLTensorEnv, active_indices: torch.Tensor) -> None:
        self.evaluator.initialize_subgame(
            env, active_indices, initial_beliefs=self._pending_beliefs
        )
        self.evaluator.evaluate_cfr(training_mode=False)
        self._num_roots = int(active_indices.numel())
        self._a2c = _build_action_to_child(self.evaluator, self._num_roots)
        self._pending_beliefs = None

    def _hand_rows(self, hands: torch.Tensor) -> torch.Tensor:
        """Map 1326-combo indices to the evaluator's hand dimension."""
        hand_dim = int(getattr(self.evaluator, "hand_dim", NUM_HANDS))
        if hand_dim == PREFLOP_HANDS:
            classes = combo_to_preflop_class_tensor(device=hands.device)
            return classes[hands]
        return hands

    def action_probs(
        self,
        env: HUNLTensorEnv,
        active_indices: torch.Tensor,
        hands: torch.Tensor,
        legal_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self._a2c is not None, "observe() must run before action_probs()"
        rows = self._hand_rows(hands)
        valid = self._a2c >= 0
        probs = self.evaluator.policy_probs_avg[self._a2c.clamp(min=0), rows[:, None]]
        return probs.to(torch.float32) * valid.to(torch.float32)

    def commit(self, actions: torch.Tensor, keep_mask: torch.Tensor) -> None:
        assert self._a2c is not None, "observe() must run before commit()"
        children = self._a2c.gather(1, actions[:, None]).squeeze(1)
        # Actions outside this evaluator's tree (only possible if the acting
        # agent is a scripted bot picking a bin the search tree omitted) fall
        # back to the root, i.e. the belief is left unchanged for that game.
        root_idx = torch.arange(
            self._num_roots, device=children.device, dtype=torch.long
        )
        children = torch.where(children >= 0, children, root_idx)
        post_beliefs = self.evaluator.beliefs[children]
        self._pending_beliefs = post_beliefs[keep_mask]
        self._a2c = None
