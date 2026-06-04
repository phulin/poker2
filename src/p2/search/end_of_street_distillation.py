from __future__ import annotations

from typing import Literal, Protocol

import torch

from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch
from p2.rl.target_provenance import TARGET_SOURCE_CHANCE_EXPECTATION
from p2.search.chance_node_helper import ChanceNodeHelper
from p2.search.postflop_spot_sampler import ChanceRootSample


ChanceMode = Literal["auto", "single_card", "sample_flops"]


class ValueFeatureEncoder(Protocol):
    def encode(
        self,
        beliefs: torch.Tensor,
        pre_chance_node: torch.Tensor | bool | None = None,
        indices: torch.Tensor | None = None,
    ) -> MLPFeatures: ...


def _chance_mode_for_closed_street(closed_street: int) -> ChanceMode:
    if closed_street == 0:
        return "sample_flops"
    if closed_street in (1, 2):
        return "single_card"
    raise ValueError("closed_street must be one of [0, 1, 2]")


@torch.no_grad()
def build_end_of_street_value_batch(
    sample: ChanceRootSample,
    *,
    value_encoder: ValueFeatureEncoder,
    target_model: torch.nn.Module,
    chance_helper: ChanceNodeHelper | None = None,
    chance: ChanceMode = "auto",
    float_dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> RebelBatch:
    """Build a value-only `E_X` distillation batch from a sampled chance root.

    The target is the chance expectation of frozen `S_{X+1}`. The returned
    features are pre-chance `E_X` inputs, while target computation evaluates the
    model on synthetic post-chance `S_{X+1}` features.
    """

    closed_street = int(sample.closed_street)
    inferred_chance = _chance_mode_for_closed_street(closed_street)
    if chance == "auto":
        chance = inferred_chance
    if chance != inferred_chance:
        raise ValueError(
            f"closed_street={closed_street} requires chance={inferred_chance!r}, "
            f"got {chance!r}"
        )

    pbs = sample.pbs
    env = pbs.env
    device = env.device
    batch_size = int(env.N)
    pre_chance_beliefs = sample.pre_chance_beliefs.contiguous()
    num_players = int(pre_chance_beliefs.shape[1])
    dtype = float_dtype or pre_chance_beliefs.dtype

    target_model.eval()
    helper = chance_helper
    if helper is None:
        helper = ChanceNodeHelper(
            device=device,
            float_dtype=dtype,
            num_players=num_players,
            model=target_model,
            generator=generator,
        )

    root_indices = torch.arange(batch_size, device=device, dtype=torch.long)
    post_features = value_encoder.encode(pbs.beliefs, pre_chance_node=False)
    pre_features = value_encoder.encode(
        pre_chance_beliefs, pre_chance_node=True
    )

    if chance == "sample_flops":
        value_targets = helper.flop_chance_values(
            root_indices, post_features, pre_chance_beliefs
        )
    else:
        value_targets = helper.single_card_chance_values(
            root_indices,
            post_features,
            pre_chance_beliefs,
            env.last_board_indices,
        )

    closed_street_tensor = torch.full(
        (batch_size,), closed_street, dtype=torch.long, device=device
    )
    statistics = {
        "street": closed_street_tensor,
        "stage": 2 * closed_street_tensor + 1,
        "board": env.last_board_indices.clone(),
        "target_source": torch.full(
            (batch_size,),
            TARGET_SOURCE_CHANCE_EXPECTATION,
            dtype=torch.long,
            device=device,
        ),
    }

    return RebelBatch(
        features=pre_features,
        legal_masks=env.legal_bins_mask(),
        value_targets=value_targets,
        statistics=statistics,
    )
