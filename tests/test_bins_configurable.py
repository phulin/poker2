from __future__ import annotations

from alphaholdem.core.structured_config import (
    Config,
    EnvConfig,
    TrainingConfig,
    ModelConfig,
)
from alphaholdem.encoding.action_mapping import (
    _bin_to_target_action,
    _action_to_bin_idx,
)
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.types import Action


def test_variable_bins_mapping_roundtrip(tmp_path):
    # Create a custom Hydra config with custom bins
    cfg = Config(
        train=TrainingConfig(),
        model=ModelConfig(),
        env=EnvConfig(
            stack=1000,
            sb=5,
            bb=10,
            bet_bins=[0.4, 0.9, 1.3, 2.1],
            card_encoder={"name": "cards_planes_v1", "kwargs": {}},
            action_encoder={
                "name": "actions_hu_v1",
                "kwargs": {"history_actions_per_round": 6},
            },
        ),
        device="cpu",  # Set device to cpu for testing
    )

    # Env state
    env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=42)
    s = env.reset()

    num_bet_bins = len(cfg.env.bet_bins) + 3

    # Map each bin (excluding fold/call and all-in) to target action and back
    for bin_idx in range(2, num_bet_bins - 1):
        target = _bin_to_target_action(bin_idx, s, num_bet_bins)
        assert target.kind in ("bet", "raise")
        back = _action_to_bin_idx(target, s, num_bet_bins)
        # Allow nearest-bin behavior; should map back to the same or adjacent when rounding tiny diffs
        assert abs(back - bin_idx) <= 1

    # All-in should still be the last index
    allin_idx = num_bet_bins - 1
    target = _bin_to_target_action(allin_idx, s, num_bet_bins)
    assert target.kind == "allin"
    back = _action_to_bin_idx(target, s, num_bet_bins)
    assert back == allin_idx
