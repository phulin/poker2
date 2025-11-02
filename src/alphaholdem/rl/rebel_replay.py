from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.models.mlp.mlp_features import MLPFeatures


@dataclass
class RebelBatch:
    features: MLPFeatures
    legal_masks: torch.Tensor
    policy_targets: Optional[torch.Tensor] = None
    value_targets: Optional[torch.Tensor] = None
    statistics: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        assert self.value_targets is not None or self.policy_targets is not None
        # Get shape from MLPFeatures
        batch_size = len(self.features)
        assert batch_size == self.legal_masks.shape[0]
        if self.policy_targets is not None:
            assert batch_size == self.policy_targets.shape[0]
        if self.value_targets is not None:
            assert batch_size == self.value_targets.shape[0]
        for key in self.statistics:
            assert batch_size == self.statistics[key].shape[0]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: torch.Tensor | slice | int) -> RebelBatch:
        return RebelBatch(
            features=self.features[idx],
            policy_targets=(
                self.policy_targets[idx] if self.policy_targets is not None else None
            ),
            value_targets=(
                self.value_targets[idx] if self.value_targets is not None else None
            ),
            legal_masks=self.legal_masks[idx],
            statistics={key: self.statistics[key][idx] for key in self.statistics},
        )

    def to(self, device: torch.device) -> RebelBatch:
        return RebelBatch(
            features=self.features.to(device),
            policy_targets=(
                self.policy_targets.to(device)
                if self.policy_targets is not None
                else None
            ),
            value_targets=(
                self.value_targets.to(device)
                if self.value_targets is not None
                else None
            ),
            legal_masks=self.legal_masks.to(device),
            statistics={
                key: self.statistics[key].to(device) for key in self.statistics
            },
        )


class RebelReplayBuffer:
    """Ring buffer storing ReBeL-style training examples."""

    def __init__(
        self,
        capacity: int,
        num_actions: int,
        num_players: int,
        num_context_features: int,
        device: torch.device,
        policy_targets: bool = True,
        value_targets: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.capacity = capacity
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.dtype = dtype

        self.features = MLPFeatures(
            context=torch.zeros(
                self.capacity, num_context_features, dtype=dtype, device=device
            ),
            street=torch.zeros(self.capacity, dtype=torch.long, device=device),
            board=torch.zeros(self.capacity, 5, dtype=torch.long, device=device),
            beliefs=torch.zeros(
                self.capacity, 2 * NUM_HANDS, dtype=dtype, device=device
            ),
        )

        if policy_targets:
            self.policy_targets = torch.zeros(
                self.capacity, NUM_HANDS, self.num_actions, dtype=dtype, device=device
            )
        else:
            self.policy_targets = None

        if value_targets:
            self.value_targets = torch.zeros(
                self.capacity,
                self.num_players,
                NUM_HANDS,
                dtype=dtype,
                device=device,
            )
        else:
            self.value_targets = None

        self.legal_masks = torch.zeros(
            self.capacity, self.num_actions, dtype=torch.bool, device=device
        )
        self.statistics = {}

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

        assert (self.policy_targets is None) == (batch.policy_targets is None)
        assert (self.value_targets is None) == (batch.value_targets is None)
        # We should never train on showdown value.
        assert (batch.features.street <= 3).all()

        batch = batch.to(self.device)

        insert_start = self.position
        dest_indices = (
            torch.arange(batch_size, device=self.device) + insert_start
        ) % self.capacity

        self.features[dest_indices] = batch.features
        if self.policy_targets is not None:
            self.policy_targets[dest_indices] = batch.policy_targets
        if self.value_targets is not None:
            self.value_targets[dest_indices] = batch.value_targets
        self.legal_masks[dest_indices] = batch.legal_masks
        for key in batch.statistics:
            if key not in self.statistics:
                self.statistics[key] = torch.zeros(
                    self.capacity,
                    *batch.statistics[key].shape[1:],
                    dtype=batch.statistics[key].dtype,
                    device=self.device,
                )
            self.statistics[key][dest_indices] = batch.statistics[key]

        self.position = (insert_start + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(
        self,
        batch_size: int,
        stratify_streets: list[float] | None = None,
        generator: Optional[torch.Generator] = None,
    ):
        """Uniformly sample a minibatch."""
        if self.size == 0:
            raise ValueError("RebelReplayBuffer is empty")
        if batch_size > self.size:
            batch_size = self.size

        if stratify_streets is None:
            idxs = torch.randint(
                0, self.size, (batch_size,), generator=generator, device=self.device
            )
        else:
            streets = self.features.street[: self.size]
            stratify_tensor = torch.tensor(stratify_streets, device=self.device)
            bins = torch.bincount(streets, minlength=4)
            stratify_tensor[bins == 0] = 0.0
            stratify_tensor /= stratify_tensor.sum()

            probs = stratify_tensor / bins
            all_probs = probs[streets]
            # sample without replacement
            idxs = torch.multinomial(all_probs, batch_size, generator=generator)

        return RebelBatch(
            features=self.features[idxs],
            policy_targets=(
                self.policy_targets[idxs] if self.policy_targets is not None else None
            ),
            value_targets=(
                self.value_targets[idxs] if self.value_targets is not None else None
            ),
            legal_masks=self.legal_masks[idxs],
            statistics={key: self.statistics[key][idxs] for key in self.statistics},
        )

    def clear(self) -> None:
        self.position = 0
        self.size = 0
