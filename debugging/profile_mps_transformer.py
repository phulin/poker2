"""Line-profile transformer encode/forward on a small MPS batch."""

from __future__ import annotations

import argparse

import torch
from line_profiler import LineProfiler

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder


def _sample_actions(env: HUNLTensorEnv) -> torch.Tensor:
    _, legal = env.legal_bins_amounts_and_mask()
    device = legal.device
    num_envs, num_bins = legal.shape
    probs = legal.float()
    # Fall back to no-op (-1) if an env is terminal
    actions = torch.full((num_envs,), -1, dtype=torch.long, device=device)
    for idx in range(num_envs):
        legal_bins = torch.where(probs[idx] > 0)[0]
        if legal_bins.numel() == 0:
            continue
        choice = legal_bins[torch.randint(0, legal_bins.numel(), (1,), device=device)]
        actions[idx] = choice
    return actions


def run_profile(device: torch.device, batch_size: int, num_steps: int) -> None:
    env = HUNLTensorEnv(
        num_envs=batch_size,
        starting_stack=20000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )
    env.reset()

    encoder = TransformerStateEncoder(env, device)
    model = PokerTransformerV1(
        d_model=128,
        n_layers=4,
        n_heads=4,
        num_bet_bins=env.num_bet_bins,
        dropout=0.1,
    ).to(device)
    model.eval()

    idxs = torch.arange(batch_size, device=device)
    kv_cache = None

    for _ in range(num_steps):
        structured = encoder.encode_tensor_states(player=0, idxs=idxs)
        structured = structured.to_device(device)
        with torch.no_grad():
            outputs = model(structured, kv_cache=kv_cache)
        kv_cache = outputs["kv_cache"]
        actions = _sample_actions(env)
        env.step_bins(actions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Line profile transformer encoding/forward on a small batch"
    )
    parser.add_argument("--batch", type=int, default=4, help="batch size / num envs")
    parser.add_argument("--steps", type=int, default=3, help="num sequential steps")
    args = parser.parse_args()

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    lp = LineProfiler()
    lp.add_function(TransformerStateEncoder.encode_tensor_states)
    lp.add_function(TransformerStateEncoder._gather_env_context)
    lp.add_function(TransformerStateEncoder._to_hero_context)
    lp.add_function(PokerTransformerV1.forward)

    profiled = lp(run_profile)
    profiled(device=device, batch_size=args.batch, num_steps=args.steps)
    lp.print_stats(stripzeros=True)


if __name__ == "__main__":
    main()
