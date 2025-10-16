from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS


@dataclass
class RebelBatch:
    features: torch.Tensor
    policy_targets: torch.Tensor
    value_targets: torch.Tensor
    legal_masks: torch.Tensor
    acting_players: torch.Tensor

    def __post_init__(self):
        assert self.features.shape[0] == self.policy_targets.shape[0]
        assert self.features.shape[0] == self.value_targets.shape[0]
        assert self.features.shape[0] == self.legal_masks.shape[0]
        assert self.features.shape[0] == self.acting_players.shape[0]

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: torch.Tensor | slice | int) -> RebelBatch:
        return RebelBatch(
            features=self.features[idx],
            policy_targets=self.policy_targets[idx],
            value_targets=self.value_targets[idx],
            legal_masks=self.legal_masks[idx],
            acting_players=self.acting_players[idx],
        )

    def to(self, device: torch.device) -> RebelBatch:
        return RebelBatch(
            features=self.features.to(device),
            policy_targets=self.policy_targets.to(device),
            value_targets=self.value_targets.to(device),
            legal_masks=self.legal_masks.to(device),
            acting_players=self.acting_players.to(device),
        )


class RebelReplayBuffer:
    """Ring buffer storing ReBeL-style training examples."""

    def __init__(
        self,
        capacity: int,
        feature_dim: int,
        num_actions: int,
        num_players: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.dtype = dtype

        self.features = torch.zeros(
            self.capacity, self.feature_dim, dtype=dtype, device=device
        )
        self.policy_targets = torch.zeros(
            self.capacity, NUM_HANDS, self.num_actions, dtype=dtype, device=device
        )
        self.value_targets = torch.zeros(
            self.capacity,
            self.num_players,
            NUM_HANDS,
            dtype=dtype,
            device=device,
        )
        self.legal_masks = torch.zeros(
            self.capacity, self.num_actions, dtype=torch.bool, device=device
        )
        self.acting_players = torch.zeros(
            self.capacity, dtype=torch.long, device=device
        )

        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(self, batch: RebelBatch) -> None:
        """Append a batch of RebelBatch samples to the replay buffer."""
        batch_size = len(batch)
        if batch_size == 0:
            return

        if batch_size >= self.capacity:
            batch = batch[-self.capacity :]
            batch_size = self.capacity

        features = batch.features.to(self.device, dtype=self.dtype)
        policy_targets = batch.policy_targets.to(self.device, dtype=self.dtype)
        value_targets = batch.value_targets.to(self.device, dtype=self.dtype)
        legal_masks = batch.legal_masks.to(self.device)
        acting_players = batch.acting_players.to(self.device)

        insert_start = self.position
        dest_indices = (
            torch.arange(batch_size, device=self.device) + insert_start
        ) % self.capacity

        self.features[dest_indices] = features
        self.policy_targets[dest_indices] = policy_targets
        self.value_targets[dest_indices] = value_targets
        self.legal_masks[dest_indices] = legal_masks
        self.acting_players[dest_indices] = acting_players

        self.position = (insert_start + batch_size) % self.capacity
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
        )

    def clear(self) -> None:
        self.position = 0
        self.size = 0
