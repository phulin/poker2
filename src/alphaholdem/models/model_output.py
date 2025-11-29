"""Output dataclass for poker models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TRMLatent:
    """Latent variables for TRM."""

    y: torch.Tensor
    z: torch.Tensor

    def detach(self) -> TRMLatent:
        """Detach the latent variables."""
        return TRMLatent(
            y=self.y.detach(),
            z=self.z.detach(),
        )


@dataclass
class ModelOutput:
    """Unified output from poker models (both CNN and transformer)."""

    value: torch.Tensor
    """Value estimates of shape (batch_size,)"""

    policy_logits: torch.Tensor | None = None
    """Policy logits of shape (batch_size, num_actions)"""

    value_quantiles: torch.Tensor | None = None
    """Optional quantile value estimates of shape (batch_size, num_quantiles)"""

    hand_values: torch.Tensor | None = None
    """Optional per-hand value estimates of shape (batch_size, num_players, num_combos)"""

    kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    """KV cache dictionary keyed by layer ID for incremental generation (transformer only)"""

    encoded_with_permutation: torch.Tensor | None = None
    """Encoded belief features with permutation applied (PBS-style only)"""

    latent: TRMLatent | None = None
    """Latent y tensor of shape (batch_size, hidden_dim)"""

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary format for backward compatibility."""
        result = {
            "policy_logits": self.policy_logits,
            "value": self.value,
        }
        if self.value_quantiles is not None:
            result["value_quantiles"] = self.value_quantiles
        if self.hand_values is not None:
            result["hand_values"] = self.hand_values
        if self.kv_cache is not None:
            result["kv_cache"] = self.kv_cache
        if self.encoded_with_permutation is not None:
            result["encoded_with_permutation"] = self.encoded_with_permutation
        return result

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor]) -> ModelOutput:
        """Create from dictionary format."""
        return cls(
            policy_logits=data["policy_logits"],
            value=data["value"],
            value_quantiles=data.get("value_quantiles"),
            hand_values=data.get("hand_values"),
            kv_cache=data.get("kv_cache"),
            encoded_with_permutation=data.get("encoded_with_permutation"),
        )

    def __getitem__(self, index: torch.Tensor | slice | int) -> ModelOutput:
        """Get item by index."""
        return ModelOutput(
            value=self.value[index],
            policy_logits=(
                self.policy_logits[index] if self.policy_logits is not None else None
            ),
            value_quantiles=(
                self.value_quantiles[index]
                if self.value_quantiles is not None
                else None
            ),
            hand_values=(
                self.hand_values[index] if self.hand_values is not None else None
            ),
            kv_cache=self.kv_cache[index] if self.kv_cache is not None else None,
            encoded_with_permutation=(
                self.encoded_with_permutation[index]
                if self.encoded_with_permutation is not None
                else None
            ),
            latent=self.latent[index] if self.latent is not None else None,
        )

    @classmethod
    def cat(cls, outputs: list[ModelOutput]) -> ModelOutput | None:
        """Concatenate a list of ModelOutput objects."""
        if not outputs:
            return None

        # Concatenate required value tensor
        value = torch.cat([o.value for o in outputs], dim=0)

        # Concatenate optional policy_logits
        policy_logits = None
        if outputs[0].policy_logits is not None:
            policy_logits = torch.cat(
                [o.policy_logits for o in outputs if o.policy_logits is not None],
                dim=0,
            )

        # Concatenate optional value_quantiles
        value_quantiles = None
        if outputs[0].value_quantiles is not None:
            value_quantiles = torch.cat(
                [o.value_quantiles for o in outputs if o.value_quantiles is not None],
                dim=0,
            )

        # Concatenate optional hand_values
        hand_values = None
        if outputs[0].hand_values is not None:
            hand_values = torch.cat(
                [o.hand_values for o in outputs if o.hand_values is not None],
                dim=0,
            )

        # Concatenate optional encoded_with_permutation
        encoded_with_permutation = None
        if outputs[0].encoded_with_permutation is not None:
            encoded_with_permutation = torch.cat(
                [
                    o.encoded_with_permutation
                    for o in outputs
                    if o.encoded_with_permutation is not None
                ],
                dim=0,
            )

        # Concatenate optional latent (TRMLatent)
        latent = None
        if outputs[0].latent is not None:
            y_tensors = [o.latent.y for o in outputs if o.latent is not None]
            z_tensors = [o.latent.z for o in outputs if o.latent is not None]
            if y_tensors and z_tensors:
                latent = TRMLatent(
                    y=torch.cat(y_tensors, dim=0),
                    z=torch.cat(z_tensors, dim=0),
                )

        # kv_cache is not concatenated as it's a complex nested structure
        # that doesn't have a clear concatenation semantics
        kv_cache = None

        return cls(
            value=value,
            policy_logits=policy_logits,
            value_quantiles=value_quantiles,
            hand_values=hand_values,
            kv_cache=kv_cache,
            encoded_with_permutation=encoded_with_permutation,
            latent=latent,
        )
