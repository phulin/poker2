"""Workload runner for profiling transformer embeddings."""

import argparse

import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.transformer.embeddings import PokerFusedEmbedding
from alphaholdem.models.transformer.state_encoder import TransformerStateEncoder


def build_batch(batch_size: int, device: torch.device, dtype: torch.dtype):
    env = HUNLTensorEnv(
        num_envs=batch_size,
        starting_stack=20000,
        sb=50,
        bb=100,
        bet_bins=[0.5, 1.0, 1.5, 2.0],
        device=device,
        float_dtype=dtype,
    )
    env.reset()

    encoder = TransformerStateEncoder(env, device)
    idxs = torch.arange(env.N, device=device)
    data = encoder.encode_tensor_states(player=0, idxs=idxs)

    # Move structured tensors to the target device/dtype
    data = data.to_device(device).to(dtype)
    return data, env.num_bet_bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data, num_bet_bins = build_batch(args.batch_size, device, dtype)

    embedding = PokerFusedEmbedding(num_bet_bins=num_bet_bins, d_model=args.d_model).to(
        device
    )

    for _ in range(args.iterations):
        output = embedding(data)

        # Force some computation to keep the kernels from being optimized away.
        _ = output.sum().item()

        if device.type == "mps":
            torch.mps.synchronize()


if __name__ == "__main__":
    main()
