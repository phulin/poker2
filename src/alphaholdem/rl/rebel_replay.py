from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RebelBatch:
    features: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    legal_masks: torch.Tensor
    acting_players: torch.Tensor
    value_weights: Optional[torch.Tensor] = None
    reach_weights: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> RebelBatch:
        return RebelBatch(
            features=self.features.to(device),
            policy_targets=self.policy_targets.to(device),
            value_targets=self.value_targets.to(device),
            legal_masks=self.legal_masks.to(device),
            acting_players=self.acting_players.to(device),
            value_weights=(
                None if self.value_weights is None else self.value_weights.to(device)
            ),
            reach_weights=(
                None if self.reach_weights is None else self.reach_weights.to(device)
            ),
        )


class RebelReplayBuffer:
    """Ring buffer storing ReBeL-style training examples."""

    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        num_actions: int,
        belief_dim: int,
        num_players: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.num_actions = int(num_actions)
        self.belief_dim = int(belief_dim)
        self.num_players = int(num_players)
        self.device = device
        self.dtype = dtype

        self.features = torch.zeros(
            self.capacity, self.feature_dim, dtype=dtype, device=device
        )
        self.policy_targets = torch.zeros(
            self.capacity, self.num_actions, dtype=dtype, device=device
        )
        self.value_targets = torch.zeros(
            self.capacity,
            self.num_players,
            self.belief_dim,
            dtype=dtype,
            device=device,
        )
        self.value_weights = torch.zeros(
            self.capacity,
            self.num_players,
            self.belief_dim,
            dtype=dtype,
            device=device,
        )
        self.legal_masks = torch.zeros(
            self.capacity, self.num_actions, dtype=torch.bool, device=device
        )
        self.acting_players = torch.zeros(
            self.capacity, dtype=torch.long, device=device
        )
        self.reach_weights = torch.zeros(self.capacity, dtype=dtype, device=device)

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        features: torch.Tensor,
        policy_targets: torch.Tensor,
        value_targets: torch.Tensor,
        legal_masks: torch.Tensor,
        acting_players: torch.Tensor,
        reach_weights: Optional[torch.Tensor] = None,
        value_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Append a batch of samples to the replay buffer."""
        batch_size = features.shape[0]
        if batch_size == 0:
            return
        if reach_weights is None:
            reach_weights = torch.ones(batch_size, dtype=self.dtype, device=self.device)
        if value_weights is None:
            value_weights = torch.ones(
                batch_size,
                self.num_players,
                self.belief_dim,
                dtype=self.dtype,
                device=self.device,
            )

        insert_start = self.position
        insert_end = self.position + batch_size

        if insert_end <= self.capacity:
            sl = slice(insert_start, insert_end)
            self.features[sl] = features.to(self.device, dtype=self.dtype)
            self.policy_targets[sl] = policy_targets.to(self.device, dtype=self.dtype)
            self.value_targets[sl] = value_targets.to(self.device, dtype=self.dtype)
            self.value_weights[sl] = value_weights.to(self.device, dtype=self.dtype)
            self.legal_masks[sl] = legal_masks.to(self.device)
            self.acting_players[sl] = acting_players.to(self.device)
            self.reach_weights[sl] = reach_weights.to(self.device, dtype=self.dtype)
        else:
            first = self.capacity - insert_start
            sl1 = slice(insert_start, self.capacity)
            sl2 = slice(0, insert_end % self.capacity)
            self.features[sl1] = features[:first].to(self.device, dtype=self.dtype)
            self.features[sl2] = features[first:].to(self.device, dtype=self.dtype)
            self.policy_targets[sl1] = policy_targets[:first].to(
                self.device, dtype=self.dtype
            )
            self.policy_targets[sl2] = policy_targets[first:].to(
                self.device, dtype=self.dtype
            )
            self.value_targets[sl1] = value_targets[:first].to(
                self.device, dtype=self.dtype
            )
            self.value_targets[sl2] = value_targets[first:].to(
                self.device, dtype=self.dtype
            )
            self.value_weights[sl1] = value_weights[:first].to(
                self.device, dtype=self.dtype
            )
            self.value_weights[sl2] = value_weights[first:].to(
                self.device, dtype=self.dtype
            )
            self.legal_masks[sl1] = legal_masks[:first].to(self.device)
            self.legal_masks[sl2] = legal_masks[first:].to(self.device)
            self.acting_players[sl1] = acting_players[:first].to(self.device)
            self.acting_players[sl2] = acting_players[first:].to(self.device)
            self.reach_weights[sl1] = reach_weights[:first].to(
                self.device, dtype=self.dtype
            )
            self.reach_weights[sl2] = reach_weights[first:].to(
                self.device, dtype=self.dtype
            )

        self.position = insert_end % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int, generator: Optional[torch.Generator] = None):
        """Uniformly sample a minibatch."""
        if self.size == 0:
            raise ValueError("RebelReplayBuffer is empty")
        if batch_size > self.size:
            batch_size = self.size
        idxs = torch.randint(
            0, self.size, (batch_size,), generator=generator, device=self.device
        )
        return RebelBatch(
            features=self.features[idxs],
            policy_targets=self.policy_targets[idxs],
            value_targets=self.value_targets[idxs],
            legal_masks=self.legal_masks[idxs],
            acting_players=self.acting_players[idxs],
            value_weights=self.value_weights[idxs],
            reach_weights=self.reach_weights[idxs],
        )

    def clear(self) -> None:
        self.position = 0
        self.size = 0
