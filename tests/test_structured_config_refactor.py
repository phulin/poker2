from __future__ import annotations

import pytest

from p2.core.structured_config import (
    CFRType,
    Config,
    LrSchedule,
    ModelScope,
    ModelType,
    NonlinearityType,
    KLType,
    PolicyLossType,
    PolicyNodeWeighting,
    PPOClipping,
    StreetValueHeads,
    ValueHeadType,
    ValueLossType,
    WarmStartType,
)


def test_curriculum_requires_explicit_substeps_key() -> None:
    with pytest.raises(TypeError, match="river"):
        Config.from_dict(
            {
                "curriculum": {
                    "stages": ["river"],
                    "river": {
                        "kind": "train",
                        "net": "S_river",
                        "num_steps": 3,
                    },
                }
            }
        )


def test_curriculum_explicit_substeps_parse_from_net() -> None:
    cfg = Config.from_dict(
        {
            "curriculum": {
                "stages": ["turn"],
                "substeps": {
                    "turn": {
                        "kind": "train",
                        "net": "S_turn",
                        "from_net": "S_river",
                        "num_steps": 3,
                    }
                },
            }
        }
    )

    assert cfg.curriculum.stages == ["turn"]
    assert cfg.curriculum.substeps["turn"].from_net == "S_river"
    assert cfg.curriculum.substeps["turn"].num_steps == 3


def test_config_from_dict_coerces_yaml_strings_to_enums() -> None:
    cfg = Config.from_dict(
        {
            "train": {
                "lr_schedule": "linear",
                "policy_node_weighting": "reach",
                "policy_loss_type": "mse",
                "ppo_clipping": "single",
                "kl_type": "forward",
                "value_loss_type": "mse",
            },
            "model": {
                "name": "BetterFFN",
                "value_head_type": "scalar",
                "nonlinearity": "gelu",
                "street_value_heads": "pre",
            },
            "search": {
                "warm_start_type": "model_br",
                "cfr_type": "discounted",
                "model_scope": "single_street",
            },
        }
    )

    assert type(cfg.train.lr_schedule) is LrSchedule
    assert cfg.train.lr_schedule is LrSchedule.linear
    assert cfg.train.policy_node_weighting is PolicyNodeWeighting.reach
    assert cfg.train.policy_loss_type is PolicyLossType.mse
    assert cfg.train.ppo_clipping is PPOClipping.single
    assert cfg.train.kl_type is KLType.forward
    assert cfg.train.value_loss_type is ValueLossType.mse
    assert cfg.model.name is ModelType.better_ffn
    assert cfg.model.value_head_type is ValueHeadType.scalar
    assert cfg.model.nonlinearity is NonlinearityType.gelu
    assert cfg.model.street_value_heads is StreetValueHeads.pre
    assert cfg.search.warm_start_type is WarmStartType.model_br
    assert cfg.search.cfr_type is CFRType.discounted
    assert cfg.search.model_scope is ModelScope.single_street
