from __future__ import annotations

from typing import Any, Optional

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
        depth_stratify_decimate: bool = False,
        depth_stratify_sample: bool = False,
        depth_stratify_probs: list[float] | None = None,
        depth_stratify_buckets: int = 0,
    ) -> None:
        self.capacity = capacity
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.dtype = dtype
        self.decimate = decimate
        self.generator = generator
        self.depth_stratify_decimate = depth_stratify_decimate
        self.depth_stratify_sample = depth_stratify_sample
        self.depth_stratify_probs = depth_stratify_probs
        self.depth_stratify_buckets = depth_stratify_buckets

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

    def state_dict(self) -> dict[str, Any]:
        """Return replay-buffer contents for checkpoint sidecar storage."""
        cpu = torch.device("cpu")
        return {
            "capacity": self.capacity,
            "num_actions": self.num_actions,
            "num_players": self.num_players,
            "dtype": str(self.dtype),
            "policy_targets_enabled": self.policy_targets is not None,
            "value_targets_enabled": self.value_targets is not None,
            "features": {
                "context": self.features.context.detach().to(cpu).clone(),
                "street": self.features.street.detach().to(cpu).clone(),
                "to_act": self.features.to_act.detach().to(cpu).clone(),
                "board": self.features.board.detach().to(cpu).clone(),
                "beliefs": self.features.beliefs.detach().to(cpu).clone(),
            },
            "policy_targets": (
                self.policy_targets.detach().to(cpu).clone()
                if self.policy_targets is not None
                else None
            ),
            "value_targets": (
                self.value_targets.detach().to(cpu).clone()
                if self.value_targets is not None
                else None
            ),
            "legal_masks": self.legal_masks.detach().to(cpu).clone(),
            "statistics": {
                key: value.detach().to(cpu).clone()
                for key, value in self.statistics.items()
            },
            "sample_count": self.sample_count.detach().to(cpu).clone(),
            "position": self.position,
            "size": self.size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay-buffer contents saved by :meth:`state_dict`."""
        expected = {
            "capacity": self.capacity,
            "num_actions": self.num_actions,
            "num_players": self.num_players,
        }
        for key, value in expected.items():
            if int(state[key]) != value:
                raise ValueError(
                    f"Replay buffer {key} mismatch: checkpoint has "
                    f"{state[key]}, current buffer has {value}"
                )
        if bool(state["policy_targets_enabled"]) != (self.policy_targets is not None):
            raise ValueError("Replay buffer policy target layout mismatch")
        if bool(state["value_targets_enabled"]) != (self.value_targets is not None):
            raise ValueError("Replay buffer value target layout mismatch")

        features = state["features"]
        self.features.context.copy_(
            features["context"].to(device=self.device, dtype=self.features.context.dtype)
        )
        self.features.street.copy_(
            features["street"].to(device=self.device, dtype=self.features.street.dtype)
        )
        self.features.to_act.copy_(
            features["to_act"].to(device=self.device, dtype=self.features.to_act.dtype)
        )
        self.features.board.copy_(
            features["board"].to(device=self.device, dtype=self.features.board.dtype)
        )
        self.features.beliefs.copy_(
            features["beliefs"].to(device=self.device, dtype=self.features.beliefs.dtype)
        )

        if self.policy_targets is not None:
            self.policy_targets.copy_(
                state["policy_targets"].to(
                    device=self.device, dtype=self.policy_targets.dtype
                )
            )
        if self.value_targets is not None:
            self.value_targets.copy_(
                state["value_targets"].to(
                    device=self.device, dtype=self.value_targets.dtype
                )
            )
        self.legal_masks.copy_(
            state["legal_masks"].to(device=self.device, dtype=self.legal_masks.dtype)
        )
        self.sample_count.copy_(
            state["sample_count"].to(device=self.device, dtype=self.sample_count.dtype)
        )
        self.statistics = {
            key: value.to(device=self.device).clone()
            for key, value in state["statistics"].items()
        }
        self.position = int(state["position"])
        self.size = int(state["size"])

    def _depth_stratified_probs(self, depths: torch.Tensor) -> torch.Tensor:
        num_buckets = self.depth_stratify_buckets
        if num_buckets <= 0:
            raise ValueError("depth_stratify_buckets must be positive")

        depths = depths.long()
        safe_depths = depths.clamp(min=0, max=num_buckets - 1)
        valid = (depths >= 0) & (depths < num_buckets)
        counts = torch.bincount(safe_depths[valid], minlength=num_buckets).to(
            torch.float32
        )

        if self.depth_stratify_probs is None:
            bucket_probs = torch.ones(num_buckets, device=depths.device)
        else:
            bucket_probs = torch.tensor(
                self.depth_stratify_probs,
                dtype=torch.float32,
                device=depths.device,
            )
            if bucket_probs.numel() != num_buckets:
                raise ValueError(
                    "depth_stratify_probs length must match depth_stratify_buckets"
                )

        bucket_probs = bucket_probs.clamp(min=0.0)
        bucket_probs = torch.where(
            counts > 0, bucket_probs, torch.zeros_like(bucket_probs)
        )
        bucket_probs = bucket_probs / bucket_probs.sum().clamp(min=1e-12)
        item_probs = bucket_probs[safe_depths] / counts[safe_depths].clamp(min=1.0)
        return torch.where(valid, item_probs, torch.zeros_like(item_probs))

    def _depth_stratified_indices(
        self,
        depths: torch.Tensor,
        sample_count: int,
        replacement: bool,
    ) -> torch.Tensor:
        probs = self._depth_stratified_probs(depths)
        return torch.multinomial(
            probs,
            sample_count,
            replacement=replacement,
            generator=self.generator,
        )

    def _depth_stratified_subset_indices(
        self,
        depths: torch.Tensor,
        sample_count: int,
    ) -> torch.Tensor:
        num_buckets = self.depth_stratify_buckets
        if num_buckets <= 0:
            raise ValueError("depth_stratify_buckets must be positive")

        depths = depths.long()
        safe_depths = depths.clamp(min=0, max=num_buckets - 1)
        valid = (depths >= 0) & (depths < num_buckets)
        counts = torch.bincount(safe_depths[valid], minlength=num_buckets)[
            :num_buckets
        ].cpu()

        if self.depth_stratify_probs is None:
            bucket_probs = torch.ones(num_buckets)
        else:
            bucket_probs = torch.tensor(self.depth_stratify_probs, dtype=torch.float32)
            if bucket_probs.numel() != num_buckets:
                raise ValueError(
                    "depth_stratify_probs length must match depth_stratify_buckets"
                )

        bucket_probs = bucket_probs.clamp(min=0.0)
        bucket_probs = torch.where(
            counts > 0, bucket_probs, torch.zeros_like(bucket_probs)
        )
        bucket_probs = bucket_probs / bucket_probs.sum().clamp(min=1e-12)
        raw_quotas = bucket_probs * sample_count
        quotas = torch.floor(raw_quotas).to(torch.long).clamp(max=counts)
        remaining = sample_count - int(quotas.sum().item())
        while remaining > 0:
            capacity = counts - quotas
            if not bool((capacity > 0).any().item()):
                break
            scores = torch.where(
                capacity > 0,
                raw_quotas - quotas.to(raw_quotas.dtype),
                torch.full_like(raw_quotas, float("-inf")),
            )
            bucket = int(torch.argmax(scores).item())
            quotas[bucket] += 1
            remaining -= 1

        selected = []
        for bucket, quota in enumerate(quotas.tolist()):
            if quota <= 0:
                continue
            bucket_indices = torch.where(depths == bucket)[0]
            perm = torch.randperm(
                bucket_indices.numel(), device=depths.device, generator=self.generator
            )[:quota]
            selected.append(bucket_indices[perm])

        if not selected:
            return torch.empty(0, dtype=torch.long, device=depths.device)

        indices = torch.cat(selected)
        shuffle = torch.randperm(
            indices.numel(), device=depths.device, generator=self.generator
        )
        return indices[shuffle]

    def add_batch(self, batch: RebelBatch) -> None:
        """Append a batch of RebelBatch samples to the replay buffer."""
        batch_size = len(batch)
        if batch_size == 0:
            return

        assert (self.policy_targets is None) == (batch.policy_targets is None)
        assert (self.value_targets is None) == (batch.value_targets is None)

        decimating = (
            self.decimate is not None and self.size == self.capacity and batch_size > 0
        )
        if batch_size >= self.capacity and not decimating:
            if (
                self.depth_stratify_decimate
                and "node_depth" in batch.statistics
                and self.depth_stratify_buckets > 0
            ):
                depths = batch.statistics["node_depth"]
                if depths.device != self.device:
                    depths = depths.to(self.device, non_blocking=True)
                indices = self._depth_stratified_subset_indices(depths, self.capacity)
                if batch.features.context.device != indices.device:
                    indices = indices.to(
                        batch.features.context.device, non_blocking=True
                    )
                batch = batch[indices]
            else:
                batch = batch[-self.capacity :]
            batch_size = self.capacity

        # Decimate if buffer is full and decimate is set
        if decimating:
            keep_count = min(self.capacity, max(1, int(self.decimate * batch_size)))
            if keep_count < batch_size:
                if (
                    self.depth_stratify_decimate
                    and "node_depth" in batch.statistics
                    and self.depth_stratify_buckets > 0
                ):
                    depths = batch.statistics["node_depth"]
                    if depths.device != self.device:
                        depths = depths.to(self.device, non_blocking=True)
                    indices = self._depth_stratified_subset_indices(depths, keep_count)
                else:
                    # Randomly sample indices to keep
                    indices = torch.randperm(
                        batch_size, device=self.device, generator=self.generator
                    )[:keep_count]
                if batch.features.context.device != indices.device:
                    indices = indices.to(
                        batch.features.context.device, non_blocking=True
                    )
                batch = batch[indices]
                batch_size = keep_count

        # We should never train on showdown value. Avoid a GPU sync in the hot
        # data-generation path; CUDA batches are validated by downstream tests.
        if batch.features.street.device.type == "cpu":
            assert (batch.features.street <= 3).all()

        batch = batch.to(self.device)

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
            if (
                self.depth_stratify_sample
                and "node_depth" in self.statistics
                and self.depth_stratify_buckets > 0
            ):
                idxs = self._depth_stratified_indices(
                    self.statistics["node_depth"][: self.size],
                    batch_size,
                    replacement=True,
                )
            else:
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
        self.sample_count.scatter_add_(0, idxs, torch.ones_like(idxs))

        return RebelBatch(
            features=MLPFeatures(
                context=torch.index_select(self.features.context, 0, idxs),
                street=torch.index_select(self.features.street, 0, idxs),
                to_act=torch.index_select(self.features.to_act, 0, idxs),
                board=torch.index_select(self.features.board, 0, idxs),
                beliefs=torch.index_select(self.features.beliefs, 0, idxs),
            ),
            policy_targets=(
                torch.index_select(self.policy_targets, 0, idxs)
                if self.policy_targets is not None
                else None
            ),
            value_targets=(
                torch.index_select(self.value_targets, 0, idxs)
                if self.value_targets is not None
                else None
            ),
            legal_masks=torch.index_select(self.legal_masks, 0, idxs),
            statistics={
                key: torch.index_select(self.statistics[key], 0, idxs)
                for key in self.statistics
            },
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
        depth_stratify_decimate: bool = False,
        depth_stratify_sample: bool = False,
        depth_stratify_probs: list[float] | None = None,
        depth_stratify_buckets: int = 0,
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
            depth_stratify_decimate=depth_stratify_decimate,
            depth_stratify_sample=depth_stratify_sample,
            depth_stratify_probs=depth_stratify_probs,
            depth_stratify_buckets=depth_stratify_buckets,
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
