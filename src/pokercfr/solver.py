"""Core solver loop scaffolding for vectorized CFR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from pokercfr.game_adapter import EnvAdapter, GameNodeBatch
from pokercfr.information_set import InformationSetEncoder, TensorHasher
from pokercfr.regret_store import RegretStoreConfig, TensorRegretStore


@dataclass
class CFRConfig:
    num_actions: int
    device: torch.device
    encoder: Optional[InformationSetEncoder] = None
    regret_dtype: torch.dtype = torch.float32


class CFRSolver:
    """High-level driver coordinating CFR iterations."""

    def __init__(self, adapter: EnvAdapter, config: CFRConfig) -> None:
        self.adapter = adapter
        self.config = config
        self.encoder = config.encoder or TensorHasher(dtype=torch.int64)
        store_config = RegretStoreConfig(
            num_actions=config.num_actions,
            device=config.device,
            dtype=config.regret_dtype,
        )
        self.store = TensorRegretStore(store_config)

    def solve(
        self, num_iterations: int, callback: Optional[Callable[[int], None]] = None
    ) -> None:
        for iteration in range(num_iterations):
            self._run_single_iteration()
            if callback is not None:
                callback(iteration)

    def _run_single_iteration(self) -> None:
        self.adapter.reset_batch()
        root = self.adapter.current_nodes()
        reach_p0 = torch.ones(root.player_to_act.shape[0], device=self.config.device)
        reach_p1 = torch.ones_like(reach_p0)
        active = torch.ones_like(root.is_terminal, dtype=torch.bool)
        self._cfr_traverse(root, reach_p0, reach_p1, active)

    def _cfr_traverse(
        self,
        node: GameNodeBatch,
        reach_prob_p0: torch.Tensor,
        reach_prob_p1: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not active_mask.any():
            return torch.zeros_like(node.rewards)

        terminal_mask = active_mask & node.is_terminal
        values = torch.zeros_like(node.rewards)
        if terminal_mask.any():
            values[terminal_mask] = node.rewards[terminal_mask]
            active_mask = active_mask & ~terminal_mask
            if not active_mask.any():
                return values

        base_snapshot = self.adapter.snapshot()

        for player_idx in (0, 1):
            mask = active_mask & (node.player_to_act == player_idx)
            if not mask.any():
                continue
            player_values = self._process_player_group(
                node=node,
                mask=mask,
                player_idx=player_idx,
                reach_p0=reach_prob_p0,
                reach_p1=reach_prob_p1,
                base_snapshot=base_snapshot,
            )
            values[mask] = player_values
            self.adapter.restore(base_snapshot)

        return values

    def _process_player_group(
        self,
        node: GameNodeBatch,
        mask: torch.Tensor,
        player_idx: int,
        reach_p0: torch.Tensor,
        reach_p1: torch.Tensor,
        base_snapshot: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        indices = torch.where(mask)[0]
        legal_mask = node.legal_actions_mask[indices]
        if legal_mask.sum(dim=-1).eq(0).any():
            raise ValueError("Encountered information set with no legal actions.")

        observations = legal_mask.to(self.config.regret_dtype)
        players = node.player_to_act[indices]
        infoset = self.encoder.encode(observations, players)

        regrets = self.store.get_regrets(infoset)
        strategy = self._regret_matching_positive(regrets, legal_mask)

        if player_idx == 0:
            reach_player = reach_p0[indices]
        else:
            reach_player = reach_p1[indices]
        self.store.update_strategy_sum(infoset, strategy, reach_player)

        num_envs = indices.shape[0]
        num_actions = strategy.shape[1]
        action_utilities = torch.zeros(
            (num_envs, num_actions),
            device=self.config.device,
            dtype=self.config.regret_dtype,
        )

        for action_idx in range(num_actions):
            local_mask = legal_mask[:, action_idx].bool()
            if not local_mask.any():
                continue

            global_indices = indices[local_mask]
            self.adapter.restore(base_snapshot)
            actions_tensor = torch.full(
                (self.adapter.batch_size,),
                -1,
                dtype=torch.long,
                device=self.config.device,
            )
            actions_tensor[global_indices] = action_idx
            self.adapter.step_batch(actions_tensor)
            child_node = self.adapter.current_nodes()

            new_reach_p0 = reach_p0.clone()
            new_reach_p1 = reach_p1.clone()
            probs = strategy[local_mask, action_idx]
            if player_idx == 0:
                new_reach_p0[global_indices] = reach_p0[global_indices] * probs
            else:
                new_reach_p1[global_indices] = reach_p1[global_indices] * probs

            new_active = torch.zeros_like(mask)
            new_active[global_indices] = True
            child_values = self._cfr_traverse(
                child_node,
                new_reach_p0,
                new_reach_p1,
                new_active,
            )
            child_values = child_values[global_indices]

            if player_idx == 0:
                util = child_values
            else:
                util = -child_values

            action_utilities[local_mask, action_idx] = util

        node_value = (strategy * action_utilities).sum(dim=-1)

        if player_idx == 0:
            values_for_p0 = node_value
            reach_opp = reach_p1[indices]
        else:
            values_for_p0 = -node_value
            reach_opp = reach_p0[indices]

        regret_delta = (
            action_utilities - node_value.unsqueeze(-1)
        ) * reach_opp.unsqueeze(-1)
        self.store.update_regrets(infoset, regret_delta)

        return values_for_p0


__all__ = ["CFRConfig", "CFRSolver"]
