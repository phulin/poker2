from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from alphaholdem.env.card_utils import (
    combo_suit_permutation_inverse_tensor,
    suit_permutations_tensor,
)
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

    def with_permuted_targets(
        self,
        suit_permutations: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_players: int = 2,
    ) -> tuple[RebelBatch, torch.Tensor]:
        """
        Return a batch with features and targets permuted by suit.

        Args:
            suit_permutations: Optional explicit suit permutations [B, 4].
            generator: Optional RNG to match MLPFeatures.permute_suits API.
            num_players: Number of players used to reshape value targets.
        """
        permuted_features = self.features.clone()
        suit_permutations = permuted_features.permute_suits(
            suit_permutations, generator
        )

        # Map permutations back to their enumeration index.
        all_perms = suit_permutations_tensor(device=suit_permutations.device)
        matches = (suit_permutations[:, None, :] == all_perms[None, :, :]).all(dim=2)
        assert matches.any(
            dim=1
        ).all(), "Invalid suit permutation encountered in with_permuted_targets"
        suit_permutation_idxs = matches.float().argmax(dim=1)

        combo_permutations_inverse = combo_suit_permutation_inverse_tensor(
            device=suit_permutation_idxs.device
        )[suit_permutation_idxs]

        value_targets = None
        if self.value_targets is not None:
            value_targets = torch.gather(
                self.value_targets,
                2,
                combo_permutations_inverse[:, None, :].expand(-1, num_players, -1),
            )

        policy_targets = None
        if self.policy_targets is not None:
            policy_targets = torch.gather(
                self.policy_targets,
                1,
                combo_permutations_inverse[:, :, None].expand(
                    -1, -1, self.policy_targets.shape[2]
                ),
            )

        return (
            RebelBatch(
                features=permuted_features,
                policy_targets=policy_targets,
                value_targets=value_targets,
                legal_masks=self.legal_masks,
                statistics=self.statistics,
            ),
            suit_permutation_idxs,
        )

    @classmethod
    def cat(cls, batches: list[RebelBatch]) -> RebelBatch | None:
        """Concatenate a list of RebelBatch objects."""
        if not batches:
            return None

        # Filter out None batches if any
        batches = [b for b in batches if b is not None and len(b) > 0]
        if not batches:
            return None

        features = MLPFeatures.cat([b.features for b in batches])
        legal_masks = torch.cat([b.legal_masks for b in batches], dim=0)

        policy_targets = None
        if batches[0].policy_targets is not None:
            policy_targets = torch.cat(
                [b.policy_targets for b in batches if b.policy_targets is not None],
                dim=0,
            )

        value_targets = None
        if batches[0].value_targets is not None:
            value_targets = torch.cat(
                [b.value_targets for b in batches if b.value_targets is not None],
                dim=0,
            )

        statistics = {}
        if batches[0].statistics:
            for key in batches[0].statistics:
                statistics[key] = torch.cat([b.statistics[key] for b in batches], dim=0)

        return cls(
            features=features,
            legal_masks=legal_masks,
            policy_targets=policy_targets,
            value_targets=value_targets,
            statistics=statistics,
        )
