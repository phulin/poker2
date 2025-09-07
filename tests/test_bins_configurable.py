from __future__ import annotations

from alphaholdem.core.config import load_config
from alphaholdem.encoding.action_mapping import (
    _bin_to_target_action,
    _action_to_bin_idx,
)
from alphaholdem.env.hunl_env import HUNLEnv
from alphaholdem.env.types import Action


def test_variable_bins_mapping_roundtrip(tmp_path):
    # Create a temp config with custom bins
    cfg_text = """
stack: 1000
sb: 5
bb: 10
seed: 0
ppo_eps: 0.2
ppo_delta1: 3.0
gamma: 0.999
gae_lambda: 0.95
entropy_coef: 0.01
value_coef: 0.5
grad_clip: 1.0
learning_rate: 1e-4
batch_size: 4096
num_epochs: 4
replay_buffer_batches: 4
value_loss_type: huber
huber_delta: 1.0
bet_bins: [0.4, 0.9, 1.3, 2.1]
card_encoder:
  name: cards_planes_v1
action_encoder:
  name: actions_hu_v1
model:
  name: siamese_convnet_v1
policy:
  name: categorical_v1
"""
    cfg_file = tmp_path / "custom.yaml"
    cfg_file.write_text(cfg_text)
    cfg = load_config(path=str(cfg_file))

    # Env state
    env = HUNLEnv(starting_stack=1000, sb=10, bb=20, seed=42)
    s = env.reset()

    num_bet_bins = len(cfg.bet_bins) + 3

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
