"""Adapters translating `HUNLTensorEnv` states into batched CFR game nodes."""

from __future__ import annotations

from collections.abc import Protocol
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GameNodeBatch:
    """Lightweight container describing a batched decision node."""

    player_to_act: torch.Tensor
    legal_actions_mask: torch.Tensor
    legal_actions_amounts: torch.Tensor
    rewards: torch.Tensor
    is_terminal: torch.Tensor
    chance_mask: torch.Tensor

    def to_device(self, device: torch.device) -> "GameNodeBatch":
        return GameNodeBatch(
            player_to_act=self.player_to_act.to(device),
            legal_actions_mask=self.legal_actions_mask.to(device),
            legal_actions_amounts=self.legal_actions_amounts.to(device),
            rewards=self.rewards.to(device),
            is_terminal=self.is_terminal.to(device),
            chance_mask=self.chance_mask.to(device),
        )


class EnvAdapter(Protocol):
    """Protocol exposing the minimal hooks the solver expects."""

    device: torch.device
    batch_size: int

    def reset_batch(
        self, batch_size: Optional[int] = None
    ) -> None:  # pragma: no cover - protocol
        ...

    def current_nodes(self) -> GameNodeBatch:  # pragma: no cover - protocol
        ...

    def step_batch(self, actions: torch.Tensor) -> None:  # pragma: no cover - protocol
        ...

    def snapshot(self) -> dict[str, torch.Tensor]:  # pragma: no cover - protocol
        ...

    def restore(
        self, state: dict[str, torch.Tensor]
    ) -> None:  # pragma: no cover - protocol
        ...


class HUNLGameTreeAdapter:
    """Batched adapter around `HUNLTensorEnv` using its public tensor API."""

    def __init__(self, env: "HUNLTensorEnv") -> None:
        from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv

        if not isinstance(env, HUNLTensorEnv):  # type: ignore[name-defined]
            raise TypeError("env must be HUNLTensorEnv")
        self.env = env
        self.device = env.device
        self.batch_size = env.N
        self._state_tensors = [
            "deck",
            "deck_pos",
            "button",
            "street",
            "to_act",
            "pot",
            "min_raise",
            "actions_this_round",
            "acted_since_reset",
            "stacks",
            "committed",
            "has_folded",
            "is_allin",
            "board_onehot",
            "hole_onehot",
            "board_indices",
            "hole_indices",
            "chips_placed",
            "done",
            "winner",
            "action_history",
        ]
        self._last_rewards = torch.zeros(
            self.batch_size, dtype=env.float_dtype, device=self.device
        )
        self._cached_amounts: Optional[torch.Tensor] = None
        self._cached_mask: Optional[torch.Tensor] = None

    def reset_batch(self, batch_size: Optional[int] = None) -> None:
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError(
                "HUNLGameTreeAdapter expects batch_size to equal env.N; construct a new adapter for different sizes"
            )
        self.env.reset()
        self._last_rewards.zero_()
        self._cached_amounts = None
        self._cached_mask = None

    def current_nodes(self) -> GameNodeBatch:
        amounts, mask = self.env.legal_bins_amounts_and_mask()
        self._cached_amounts = amounts
        self._cached_mask = mask
        return GameNodeBatch(
            player_to_act=self.env.to_act.clone(),
            legal_actions_mask=mask.clone(),
            legal_actions_amounts=amounts.clone(),
            rewards=self._last_rewards.clone(),
            is_terminal=self.env.done.clone(),
            chance_mask=self._chance_mask(),
        )

    def step_batch(self, actions: torch.Tensor) -> None:
        if actions.shape[0] != self.batch_size:
            raise ValueError("actions tensor must match env batch size")
        if self._cached_amounts is None or self._cached_mask is None:
            amounts, mask = self.env.legal_bins_amounts_and_mask()
        else:
            amounts, mask = self._cached_amounts, self._cached_mask
        rewards, _, _, _, _ = self.env.step_bins(actions.long(), amounts, mask)
        self._last_rewards = rewards
        self._cached_amounts = None
        self._cached_mask = None

    def _chance_mask(self) -> torch.Tensor:
        return torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

    def snapshot(self) -> dict[str, torch.Tensor]:
        state = {name: getattr(self.env, name).clone() for name in self._state_tensors}
        state["rng_state"] = self.env.rng.get_state()
        state["last_rewards"] = self._last_rewards.clone()
        return state

    def restore(self, state: dict[str, torch.Tensor]) -> None:
        for name in self._state_tensors:
            tensor = getattr(self.env, name)
            tensor.copy_(state[name])
        if "rng_state" in state:
            self.env.rng.set_state(state["rng_state"])
        if "last_rewards" in state:
            self._last_rewards.copy_(state["last_rewards"])
        self._cached_amounts = None
        self._cached_mask = None


__all__ = ["EnvAdapter", "GameNodeBatch", "HUNLGameTreeAdapter"]
