from __future__ import annotations

import pytest

from p2.core.structured_config import Config


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


def test_curriculum_explicit_substeps_parse_from_alias() -> None:
    cfg = Config.from_dict(
        {
            "curriculum": {
                "stages": ["turn"],
                "substeps": {
                    "turn": {
                        "kind": "train",
                        "net": "S_turn",
                        "from": "S_river",
                        "num_steps": 3,
                    }
                },
            }
        }
    )

    assert cfg.curriculum.stages == ["turn"]
    assert cfg.curriculum.substeps["turn"].from_net == "S_river"
    assert cfg.curriculum.substeps["turn"].num_steps == 3
