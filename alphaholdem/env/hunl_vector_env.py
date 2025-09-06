from __future__ import annotations

from typing import List, Tuple, Optional, Sequence

from .hunl_env import HUNLEnv
from .types import Action, GameState
from alphaholdem.encoding.action_mapping import bin_to_action


class HUNLVectorEnv:
    """Vectorized wrapper for multiple parallel HUNL environments.

    API (batched):
      - reset(seed: Optional[int]) -> List[GameState]
      - step(actions: Sequence[Action]) -> (List[GameState], List[float], List[bool], List[dict])
      - legal_action_bins(num_bet_bins: int) -> List[List[int]]

    Notes:
      - The wrapper preserves the single-env semantics but operates over a fixed
        number of parallel envs for better throughput.
      - Each sub-env maintains its own RNG state seeded from the base seed.
    """

    def __init__(
        self,
        num_envs: int,
        starting_stack,
        sb,
        bb,
        seed: Optional[int] = None,
    ) -> None:
        assert num_envs > 0, "num_envs must be positive"
        self.num_envs = num_envs
        self.envs: List[HUNLEnv] = [
            HUNLEnv(starting_stack=starting_stack, sb=sb, bb=bb)
            for _ in range(num_envs)
        ]
        # Optional seeding fan-out
        if seed is not None:
            for i, e in enumerate(self.envs):
                e.reset(seed=seed + i * 9973)
                # Immediately discard this initial reset; we'll reset again on demand
        self._last_states: List[Optional[GameState]] = [None for _ in range(num_envs)]

    def reset(self, seed: Optional[int] = None) -> List[GameState]:
        states: List[GameState] = []
        for i, e in enumerate(self.envs):
            s = e.reset(seed=(None if seed is None else seed + i * 9973))
            states.append(s)
            self._last_states[i] = s
        return states

    def step(
        self, actions: Sequence[Action]
    ) -> Tuple[List[GameState], List[float], List[bool], List[dict]]:
        assert len(actions) == self.num_envs, "actions length must equal num_envs"
        next_states: List[GameState] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[dict] = []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            s, r, d, info = e.step(a)
            next_states.append(s)
            rewards.append(float(r))
            dones.append(bool(d))
            infos.append(info)
            self._last_states[i] = s
        return next_states, rewards, dones, infos

    def states(self) -> List[Optional[GameState]]:
        """Return the most recently observed states (may be None before first reset)."""
        return list(self._last_states)

    def legal_action_bins(self, num_bet_bins: int) -> List[List[int]]:
        """Batched legal discrete action bins for each env (in current states)."""
        out: List[List[int]] = []
        for e in self.envs:
            out.append(e.legal_action_bins(num_bet_bins))
        return out

    def is_done(self) -> List[bool]:
        return [bool(e._require_state().terminal) for e in self.envs]

    def active_indices(self) -> List[int]:
        return [i for i, e in enumerate(self.envs) if not e._require_state().terminal]

    # --- Vectorized helpers over discrete bins ---------------------------------

    def step_bins(
        self, bin_indices: Sequence[int], num_bet_bins: int
    ) -> Tuple[List[GameState], List[float], List[bool], List[dict]]:
        """Step all envs with discrete bin indices.

        Converts each bin to a concrete Action using current env state, then steps.
        """
        assert len(bin_indices) == self.num_envs
        actions: List[Action] = []
        for i, e in enumerate(self.envs):
            s = e._require_state()
            a = bin_to_action(int(bin_indices[i]), s, num_bet_bins)
            actions.append(a)
        return self.step(actions)

    def map_bins_to_actions(
        self, bin_indices: Sequence[int], num_bet_bins: int
    ) -> List[Action]:
        """Map bins to concrete Actions for each env without stepping."""
        assert len(bin_indices) == self.num_envs
        out: List[Action] = []
        for i, e in enumerate(self.envs):
            s = e._require_state()
            out.append(bin_to_action(int(bin_indices[i]), s, num_bet_bins))
        return out
