from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from p2.core.structured_config import Config, ModelType
from p2.env.card_utils import NUM_HANDS
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.better_ffn import BetterFFN


DEFAULT_FORCE_DECK = [12, 25, 38, 51, 0, 13, 26, 1, 14]


def _load_better_ffn(snapshot: Path, device: torch.device) -> tuple[BetterFFN, Config]:
    checkpoint = torch.load(snapshot, map_location=device, weights_only=False)
    cfg = Config.from_dict(checkpoint["config"])
    state = checkpoint["model"]
    if (
        cfg.model.name != ModelType.better_ffn
        and "street_embedding.weight" not in state
    ):
        raise ValueError(f"{snapshot} is not a BetterFFN checkpoint")

    cfg.model.name = ModelType.better_ffn
    cfg.model.num_actions = len(cfg.env.bet_bins) + 3
    model = BetterFFN(
        num_actions=cfg.model.num_actions,
        hidden_dim=cfg.model.hidden_dim,
        range_hidden_dim=cfg.model.range_hidden_dim,
        ffn_dim=cfg.model.ffn_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_policy_layers=cfg.model.num_policy_layers,
        num_value_layers=cfg.model.num_value_layers,
        num_players=2,
        shared_trunk=cfg.model.shared_trunk,
        enforce_zero_sum=cfg.model.enforce_zero_sum,
        board_interaction_dim=cfg.model.board_interaction_dim,
        nonlinearity=cfg.model.nonlinearity,
    )
    state = {
        key: value.to(torch.float32) if value.dtype.is_floating_point else value
        for key, value in state.items()
    }
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg


def _make_env(cfg: Config, device: torch.device, force_button: int) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )
    env.reset(
        force_button=torch.tensor([force_button], dtype=torch.long, device=device),
        force_deck=torch.tensor([DEFAULT_FORCE_DECK], dtype=torch.long, device=device),
    )
    return env


def _clone_env(env: HUNLTensorEnv) -> HUNLTensorEnv:
    out = HUNLTensorEnv.from_proto(env, num_envs=1)
    out.copy_state_from(
        env,
        torch.tensor([0], dtype=torch.long, device=env.device),
        torch.tensor([0], dtype=torch.long, device=env.device),
    )
    return out


def _model_values(
    model: BetterFFN, env: HUNLTensorEnv, beliefs: torch.Tensor
) -> torch.Tensor:
    encoder = model.create_feature_encoder(
        env=env, device=env.device, dtype=torch.float32
    )
    features = encoder.encode(beliefs)
    with torch.no_grad():
        out = model(features, include_policy=False, include_value=True)
    return out.hand_values[0].float()


def _build_problem(
    model: BetterFFN, env: HUNLTensorEnv, beliefs: torch.Tensor
) -> dict[str, Any]:
    actor = int(env.to_act[0].item())
    amounts, legal_mask_t = env.legal_bins_amounts_and_mask()
    legal_mask = legal_mask_t[0].to(torch.bool)
    num_actions = legal_mask.numel()
    child_values = torch.zeros(num_actions, 2, NUM_HANDS, dtype=torch.float32)

    for action in range(num_actions):
        if not bool(legal_mask[action].item()):
            continue
        child_env = _clone_env(env)
        rewards, _, _ = child_env.step_bins(
            torch.tensor([action], dtype=torch.long, device=env.device),
            bin_amounts=amounts[:, :],
            legal_masks=legal_mask_t[:, :],
        )
        if bool(child_env.done[0].item()):
            reward = float(rewards[0].item())
            child_values[action, 0].fill_(reward)
            child_values[action, 1].fill_(-reward)
        else:
            child_values[action] = _model_values(model, child_env, beliefs).cpu()

    return {
        "actor": actor,
        "legalMask": [1 if bool(v) else 0 for v in legal_mask.cpu().tolist()],
        "childValues": child_values.reshape(-1).tolist(),
    }


def _solve_local(
    problem: dict[str, Any],
    beliefs: torch.Tensor,
    num_actions: int,
    iterations: int,
    action: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    actor = int(problem["actor"])
    legal = torch.tensor(problem["legalMask"], dtype=torch.bool)
    child_values = torch.tensor(problem["childValues"], dtype=torch.float32).view(
        num_actions, 2, NUM_HANDS
    )
    regrets = torch.zeros(NUM_HANDS, num_actions, dtype=torch.float32)
    policy = torch.zeros_like(regrets)
    avg_policy = torch.zeros_like(regrets)

    for _ in range(iterations):
        positive = regrets.clamp(min=0.0)
        positive[:, ~legal] = 0.0
        denom = positive.sum(dim=1, keepdim=True)
        uniform = legal.float() / legal.float().sum().clamp(min=1.0)
        policy = torch.where(
            denom > 1.0e-8, positive / denom.clamp(min=1.0e-8), uniform
        )
        values = child_values[:, actor, :].T
        expected = (policy * values).sum(dim=1, keepdim=True)
        regrets[:, legal] += values[:, legal] - expected
        avg_policy += policy

    final_policy = avg_policy / float(max(iterations, 1))
    action_probs = (beliefs[0, actor, :, None].cpu() * final_policy).sum(dim=0)

    next_beliefs = None
    if action is not None:
        next_beliefs = beliefs.clone().cpu()
        updated = next_beliefs[0, actor] * final_policy[:, action]
        denom = updated.sum()
        if denom > 1.0e-12:
            next_beliefs[0, actor] = updated / denom
        next_beliefs = next_beliefs.to(beliefs.device)

    return final_policy, action_probs, next_beliefs


def _action_labels(bet_bins: list[float]) -> list[str]:
    labels = ["fold", "check_call"]
    labels.extend(f"bet_{v:g}x_pot" for v in bet_bins)
    labels.append("all_in")
    return labels


def build_fixture(
    snapshot: Path,
    spot: list[int],
    iterations: int,
    force_button: int,
) -> dict[str, Any]:
    device = torch.device("cpu")
    model, cfg = _load_better_ffn(snapshot, device)
    env = _make_env(cfg, device, force_button)
    beliefs = torch.full((1, 2, NUM_HANDS), 1.0 / NUM_HANDS, dtype=torch.float32)

    problems: list[dict[str, Any]] = []
    final_policy = None
    final_action_probs = None

    for action in spot:
        problem = _build_problem(model, env, beliefs)
        problem["action"] = int(action)
        _, _, next_beliefs = _solve_local(
            problem, beliefs, cfg.model.num_actions, iterations, action=action
        )
        if next_beliefs is None:
            raise RuntimeError("expected next beliefs")
        problems.append(problem)
        env.step_bins(torch.tensor([action], dtype=torch.long, device=device))
        beliefs = next_beliefs

    final_problem = _build_problem(model, env, beliefs)
    final_policy, final_action_probs, _ = _solve_local(
        final_problem, beliefs, cfg.model.num_actions, iterations
    )
    problems.append(final_problem)

    return {
        "schemaVersion": 1,
        "source": "p2.webgpu_cfr.reference",
        "snapshot": str(snapshot),
        "spot": spot,
        "iterations": iterations,
        "numHands": NUM_HANDS,
        "numActions": cfg.model.num_actions,
        "actionLabels": _action_labels(cfg.env.bet_bins),
        "initialBeliefs": torch.full((2, NUM_HANDS), 1.0 / NUM_HANDS)
        .reshape(-1)
        .tolist(),
        "problems": problems,
        "expected": {
            "beliefsAtSpot": beliefs.cpu().reshape(-1).tolist(),
            "actionProbs": final_action_probs.tolist(),
            "policy": final_policy.reshape(-1).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument(
        "--spot",
        type=str,
        default="",
        help="Comma-separated action-bin sequence, for example '1,2'.",
    )
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--force-button", type=int, default=1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    spot = [int(x) for x in args.spot.split(",") if x.strip()]
    fixture = build_fixture(args.snapshot, spot, args.iterations, args.force_button)
    text = json.dumps(fixture, separators=(",", ":"))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
