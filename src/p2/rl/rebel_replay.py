from __future__ import annotations

from typing import Optional

import torch

from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch


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
        decimate: float | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.capacity = capacity
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.dtype = dtype
        self.decimate = decimate
        self.generator = generator

        self.features = MLPFeatures(
            context=torch.zeros(
                self.capacity, num_context_features, dtype=dtype, device=device
            ),
            street=torch.zeros(self.capacity, dtype=torch.long, device=device),
            to_act=torch.zeros(self.capacity, dtype=torch.long, device=device),
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

        # Number of times each buffer element has been sampled
        self.sample_count = torch.zeros(self.capacity, dtype=torch.long, device=device)

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

        # Decimate if buffer is full and decimate is set
        if self.decimate is not None and self.size == self.capacity and batch_size > 0:
            keep_count = max(1, int(self.decimate * batch_size))
            if keep_count < batch_size:
                # Randomly sample indices to keep
                indices = torch.randperm(
                    batch_size, device=self.device, generator=self.generator
                )[:keep_count]
                batch = batch[indices]
                batch_size = keep_count

        insert_start = self.position
        dest_indices = (
            torch.arange(batch_size, device=self.device) + insert_start
        ) % self.capacity

        self.features[dest_indices] = batch.features
        if self.policy_targets is not None and batch.policy_targets is not None:
            self.policy_targets[dest_indices] = batch.policy_targets
        if self.value_targets is not None and batch.value_targets is not None:
            self.value_targets[dest_indices] = batch.value_targets
        self.legal_masks[dest_indices] = batch.legal_masks
        # Reset sample count when overwriting entries
        self.sample_count[dest_indices] = 0
        for key in batch.statistics:
            if key not in self.statistics:
                self.statistics[key] = torch.zeros(
                    self.capacity,
                    *batch.statistics[key].shape[1:],
                    dtype=batch.statistics[key].dtype,
                    device=self.device,
                )
            self.statistics[key][dest_indices] = batch.statistics[key]
        assert set(self.statistics.keys()) == set(batch.statistics.keys())

        self.position = (insert_start + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(
        self,
        batch_size: int,
        stratify_streets: list[float] | None = None,
    ):
        """Uniformly sample a minibatch."""
        if self.size == 0:
            raise ValueError("RebelReplayBuffer is empty")
        assert batch_size <= self.size, "Can't take more samples than we have."

        if stratify_streets is None:
            idxs = torch.randint(
                0,
                self.size,
                (batch_size,),
                generator=self.generator,
                device=self.device,
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
            idxs = torch.multinomial(all_probs, batch_size, generator=self.generator)

        # Increment sample counters for sampled indices
        self.sample_count[idxs] += 1

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


class RebelPolicyBuffer(RebelReplayBuffer):
    value_targets: None
    policy_targets: torch.Tensor

    def __init__(
        self,
        capacity: int,
        num_actions: int,
        num_players: int,
        num_context_features: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        decimate: float | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(
            capacity=capacity,
            num_actions=num_actions,
            num_players=num_players,
            num_context_features=num_context_features,
            device=device,
            policy_targets=True,
            value_targets=False,
            dtype=dtype,
            decimate=decimate,
            generator=generator,
        )


class RebelValueBuffer(RebelReplayBuffer):
    value_targets: torch.Tensor
    policy_targets: None

    def __init__(
        self,
        capacity: int,
        num_actions: int,
        num_players: int,
        num_context_features: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        decimate: float | None = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__(
            capacity=capacity,
            num_actions=num_actions,
            num_players=num_players,
            num_context_features=num_context_features,
            device=device,
            policy_targets=False,
            value_targets=True,
            dtype=dtype,
            decimate=decimate,
            generator=generator,
        )
