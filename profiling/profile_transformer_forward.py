"""Profile PokerTransformer forward pass using line_profiler."""

from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.poker_transformer import PokerTransformerV1
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder


def build_env(device: torch.device) -> HUNLTensorEnv:
    return HUNLTensorEnv(
        num_envs=2,
        starting_stack=20000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
    )


def main() -> None:
    target_device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    env = build_env(target_device)
    env.reset()

    encoder = TransformerStateEncoder(env, target_device)
    idxs = torch.arange(env.N, device=target_device)
    structured = encoder.encode_tensor_states(player=0, idxs=idxs)
    structured = structured.to_device(target_device)

    model = PokerTransformerV1(
        d_model=256,
        n_layers=4,
        n_heads=4,
        num_bet_bins=env.num_bet_bins,
        dropout=0.1,
    ).to(target_device)

    model.eval()
    with torch.no_grad():
        for _ in range(2):
            model(structured)

        model(structured)


if __name__ == "__main__":
    main()
