from __future__ import annotations

from typing import Literal, Protocol

import torch

from p2.env.card_utils import (
    PREFLOP_HANDS,
    collapse_1326_to_169,
    expand_169_to_1326,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch
from p2.rl.target_provenance import TARGET_SOURCE_CHANCE_EXPECTATION
from p2.search.chance_node_helper import ChanceNodeHelper
from p2.search.postflop_spot_sampler import ChanceRootSample


ChanceMode = Literal["auto", "single_card", "sample_flops", "canonical_flops"]


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


def _allowed_chance_modes_for_closed_street(closed_street: int) -> set[str]:
    if closed_street == 0:
        return {"sample_flops", "canonical_flops"}
    if closed_street in (1, 2):
        return {"single_card"}
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
    allowed_chance_modes = _allowed_chance_modes_for_closed_street(closed_street)
    if chance not in allowed_chance_modes:
        raise ValueError(
            f"closed_street={closed_street} requires chance in "
            f"{sorted(allowed_chance_modes)!r}, "
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
    target_encoder = value_encoder
    if closed_street == 0 and hasattr(target_model, "create_feature_encoder"):
        target_encoder = target_model.create_feature_encoder(
            env=pbs.env,
            device=device,
            dtype=dtype,
        )
    post_features = target_encoder.encode(pbs.beliefs, pre_chance_node=False)

    compact_preflop = (
        closed_street == 0
        and int(getattr(value_encoder, "belief_dim", 0)) == PREFLOP_HANDS
    )
    target_pre_chance_beliefs = pre_chance_beliefs
    if compact_preflop:
        if pre_chance_beliefs.shape[-1] == PREFLOP_HANDS:
            pre_chance_beliefs_169 = pre_chance_beliefs
        else:
            pre_chance_beliefs_169 = collapse_1326_to_169(
                pre_chance_beliefs,
                reduction="sum",
            )
        pre_chance_beliefs_169 = pre_chance_beliefs_169.contiguous()
        pre_features = value_encoder.encode(
            pre_chance_beliefs_169, pre_chance_node=True
        )
        target_pre_chance_beliefs = expand_169_to_1326(
            pre_chance_beliefs_169,
            divide_by_multiplicity=True,
        ).contiguous()
    else:
        pre_features = value_encoder.encode(pre_chance_beliefs, pre_chance_node=True)

    if chance == "sample_flops":
        value_targets = helper.flop_chance_values(
            root_indices, post_features, target_pre_chance_beliefs
        )
    elif chance == "canonical_flops":
        value_targets = helper.canonical_flop_chance_values(
            root_indices, post_features, target_pre_chance_beliefs
        )
    else:
        value_targets = helper.single_card_chance_values(
            root_indices,
            post_features,
            target_pre_chance_beliefs,
            env.last_board_indices,
        )
    if compact_preflop:
        value_targets = collapse_1326_to_169(value_targets, reduction="mean")

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
