from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config, ModelType
from p2.env.card_utils import NUM_HANDS, combo_lookup_tensor
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.better_ffn import (
    BetterPolicyFFN,
    BetterSplitFFN,
    BetterStreetValueFFN,
)


DEFAULT_FORCE_DECK = [12, 25, 38, 51, 0, 13, 26, 1, 14]


def _load_split_model(
    snapshot: Path, device: torch.device
) -> tuple[BetterSplitFFN, Config]:
    checkpoint = torch.load(snapshot, map_location=device, weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])
    cfg.model.name = ModelType.better_ffn
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3

    common: dict[str, Any] = {
        "hidden_dim": cfg.model.hidden_dim,
        "range_hidden_dim": cfg.model.range_hidden_dim,
        "ffn_dim": cfg.model.ffn_dim,
        "num_hidden_layers": cfg.model.num_hidden_layers,
        "num_policy_layers": cfg.model.num_policy_layers,
        "num_value_layers": cfg.model.num_value_layers,
        "num_players": 2,
        "shared_trunk": cfg.model.shared_trunk,
        "enforce_zero_sum": cfg.model.enforce_zero_sum,
        "board_interaction_dim": cfg.model.board_interaction_dim,
        "policy_rank": cfg.model.policy_rank,
        "policy_hand_bias_rank": cfg.model.policy_hand_bias_rank,
        "nonlinearity": cfg.model.nonlinearity,
    }
    model = BetterSplitFFN(
        policy_model=BetterPolicyFFN(num_actions=cfg.model.num_actions, **common),
        value_model=BetterStreetValueFFN(num_actions=1, **common),
    )
    state = {
        key.removeprefix("_orig_mod."): (
            value.to(torch.float32) if value.dtype.is_floating_point else value
        )
        for key, value in checkpoint["model"].items()
        if not key.removeprefix("_orig_mod.").endswith("policy_factor_scale")
    }
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def _make_env(
    cfg: Config,
    device: torch.device,
    force_button: int,
    stack: int | None = None,
    sb: int | None = None,
    bb: int | None = None,
    force_deck: list[int] | None = None,
) -> HUNLTensorEnv:
    stack = cfg.env.stack if stack is None else stack
    sb = cfg.env.sb if sb is None else sb
    bb = cfg.env.bb if bb is None else bb
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=stack,
        sb=sb,
        bb=bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )
    env.reset(
        force_button=torch.tensor([force_button], dtype=torch.long, device=device),
        force_deck=torch.tensor(
            [force_deck or DEFAULT_FORCE_DECK],
            dtype=torch.long,
            device=device,
        ),
    )
    return env


def _combo_index(card_a: int, card_b: int) -> int:
    return int(combo_lookup_tensor()[card_a, card_b].item())


def _parse_int_list(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part) for part in value.split(",") if part]


def build_fixture(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cpu")
    model, cfg = _load_split_model(args.snapshot, device)
    env = _make_env(
        cfg,
        device,
        args.force_button,
        stack=args.stack,
        sb=args.sb,
        bb=args.bb,
        force_deck=_parse_int_list(args.force_deck) or None,
    )
    for action in _parse_int_list(args.actions):
        env.step_bins(torch.tensor([action], dtype=torch.long, device=device))
    beliefs = torch.full((1, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32)

    policy_encoder = model.policy_model.create_feature_encoder(
        env=env, device=device, dtype=torch.float32
    )
    value_encoder = model.value_model.create_feature_encoder(
        env=env, device=device, dtype=torch.float32
    )
    with torch.no_grad():
        policy = model.policy_model(
            policy_encoder.encode(beliefs), include_policy=True, include_value=False
        ).policy_logits[0]
        value_features = value_encoder.encode(
            beliefs,
            pre_chance_node=args.value_pre_chance,
        )
        values = model.value_model.forward_pre(value_features)[0]

    hands = {
        "AsKd": _combo_index(51, 24),
        "AsAh": _combo_index(51, 38),
        "2c7d": _combo_index(0, 18),
        "QsJh": _combo_index(49, 35),
    }
    return {
        "schemaVersion": 1,
        "source": "p2.webgpu_cfr.split_reference",
        "snapshot": str(args.snapshot),
        "numHands": NUM_HANDS,
        "numActions": cfg.model.num_actions,
        "forceButton": args.force_button,
        "valueStreet": int(value_features.street[0].item()),
        "valueBoard": value_features.board[0].long().tolist(),
        "hands": {
            label: {
                "index": index,
                "policyLogits": policy[index].float().tolist(),
                "handValues": values[:, index].float().tolist(),
            }
            for label, index in hands.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--force-button", type=int, default=1)
    parser.add_argument("--force-deck", type=str)
    parser.add_argument("--actions", type=str)
    parser.add_argument("--stack", type=int)
    parser.add_argument("--sb", type=int)
    parser.add_argument("--bb", type=int)
    parser.add_argument("--value-pre-chance", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    fixture = build_fixture(args)
    text = json.dumps(fixture, separators=(",", ":"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
