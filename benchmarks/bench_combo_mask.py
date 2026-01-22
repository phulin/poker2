import argparse
import os
import sys
import time
from typing import Tuple

import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from p2.env.card_utils import combo_to_onehot_tensor


def synchronize_device_if_needed(device: torch.device) -> None:
    if device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.synchronize()


def prepare_tensors(
    num_samples: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare combo_onehot and board_onehot tensors for benchmarking.

    Returns:
        combo_onehot_bool: [1326, 52] bool tensor
        combo_onehot_float: [1326, 52] float tensor
        board_onehot: [num_samples, 52] bool tensor with random cards
    """
    # Get combo_onehot tensors
    combo_onehot_bool = combo_to_onehot_tensor(device=device)
    combo_onehot_float = combo_onehot_bool.float()

    # Generate random board cards for each sample
    generator = torch.Generator(device=device).manual_seed(42)
    # Each sample has 0-5 cards on the board
    num_board_cards = torch.randint(
        0, 6, (num_samples,), generator=generator, device=device
    )

    board_onehot = torch.zeros(num_samples, 52, dtype=torch.bool, device=device)
    board_onehot_float = torch.zeros(
        num_samples, 52, dtype=torch.float32, device=device
    )

    for i in range(num_samples):
        n_cards = num_board_cards[i].item()
        if n_cards > 0:
            # Sample n_cards distinct cards
            cards = torch.randperm(52, generator=generator, device=device)[:n_cards]
            board_onehot[i, cards] = True
            board_onehot_float[i, cards] = 1.0

    return combo_onehot_bool, combo_onehot_float, board_onehot, board_onehot_float


@torch.no_grad()
def bench_boolean_approach(
    combo_onehot_bool: torch.Tensor,
    board_onehot: torch.Tensor,
    num_samples: int,
    repeats: int,
) -> float:
    """Benchmark boolean approach using expand, &, and any."""
    device = combo_onehot_bool.device

    # Warmup
    for _ in range(5):
        combo_onehot = combo_onehot_bool.unsqueeze(0).expand(
            num_samples, -1, -1
        )  # [num_samples, 1326, 52]
        allowed_mask = ~(combo_onehot & board_onehot.unsqueeze(1)).any(
            dim=2
        )  # [num_samples, 1326]
        _ = float(allowed_mask.sum().item())

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        combo_onehot = combo_onehot_bool.unsqueeze(0).expand(
            num_samples, -1, -1
        )  # [num_samples, 1326, 52]
        allowed_mask = ~(combo_onehot & board_onehot.unsqueeze(1)).any(
            dim=2
        )  # [num_samples, 1326]
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(allowed_mask.sum().item())

    return end - start


@torch.no_grad()
def bench_matrix_multiply_approach(
    combo_onehot_float: torch.Tensor,
    board_onehot_float: torch.Tensor,
    num_samples: int,
    repeats: int,
) -> float:
    """Benchmark matrix multiply approach using bmm."""
    device = combo_onehot_float.device

    # Warmup
    for _ in range(5):
        combo_onehot = combo_onehot_float.unsqueeze(0).expand(
            num_samples, -1, -1
        )  # [num_samples, 1326, 52]
        # Batch matrix multiply: [num_samples, 1326, 52] @ [num_samples, 52, 1] -> [num_samples, 1326, 1]
        overlaps = torch.bmm(combo_onehot, board_onehot_float.unsqueeze(-1)).squeeze(
            -1
        )  # [num_samples, 1326]
        allowed_mask = ~(overlaps > 0.0)  # [num_samples, 1326]
        _ = float(allowed_mask.sum().item())

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        combo_onehot = combo_onehot_float.unsqueeze(0).expand(
            num_samples, -1, -1
        )  # [num_samples, 1326, 52]
        # Batch matrix multiply: [num_samples, 1326, 52] @ [num_samples, 52, 1] -> [num_samples, 1326, 1]
        overlaps = torch.bmm(combo_onehot, board_onehot_float.unsqueeze(-1)).squeeze(
            -1
        )  # [num_samples, 1326]
        allowed_mask = ~(overlaps > 0.0)  # [num_samples, 1326]
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(allowed_mask.sum().item())

    return end - start


@torch.no_grad()
def bench_matrix_multiply_at_approach(
    combo_onehot_float: torch.Tensor,
    board_onehot_float: torch.Tensor,
    num_samples: int,
    repeats: int,
) -> float:
    """Benchmark matrix multiply approach using @ operator."""
    device = combo_onehot_float.device

    # Warmup
    for _ in range(5):
        combo_onehot = combo_onehot_float.unsqueeze(0).expand(
            num_samples, -1, -1
        )  # [num_samples, 1326, 52]
        # Matrix multiply: [num_samples, 1326, 52] @ [num_samples, 52, 1] -> [num_samples, 1326, 1]
        overlaps = (combo_onehot @ board_onehot_float.unsqueeze(-1)).squeeze(
            -1
        )  # [num_samples, 1326]
        allowed_mask = ~(overlaps > 0.0)  # [num_samples, 1326]
        _ = float(allowed_mask.sum().item())

    synchronize_device_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        combo_onehot = combo_onehot_float.unsqueeze(0).expand(
            num_samples, -1, -1
        )  # [num_samples, 1326, 52]
        # Matrix multiply: [num_samples, 1326, 52] @ [num_samples, 52, 1] -> [num_samples, 1326, 1]
        overlaps = (combo_onehot @ board_onehot_float.unsqueeze(-1)).squeeze(
            -1
        )  # [num_samples, 1326]
        allowed_mask = ~(overlaps > 0.0)  # [num_samples, 1326]
    synchronize_device_if_needed(device)
    end = time.perf_counter()

    # Prevent dead-code elimination
    _ = float(allowed_mask.sum().item())

    return end - start


def verify_correctness(
    combo_onehot_bool: torch.Tensor,
    combo_onehot_float: torch.Tensor,
    board_onehot: torch.Tensor,
    board_onehot_float: torch.Tensor,
    num_samples: int,
) -> None:
    """Verify that both approaches produce the same results."""
    # Boolean approach
    combo_onehot_b = combo_onehot_bool.unsqueeze(0).expand(num_samples, -1, -1)
    allowed_mask_bool = ~(combo_onehot_b & board_onehot.unsqueeze(1)).any(dim=2)

    # Matrix multiply with bmm
    combo_onehot_f = combo_onehot_float.unsqueeze(0).expand(num_samples, -1, -1)
    overlaps_bmm = torch.bmm(combo_onehot_f, board_onehot_float.unsqueeze(-1)).squeeze(
        -1
    )
    allowed_mask_bmm = ~(overlaps_bmm > 0.0)

    # Matrix multiply with @
    overlaps_at = (combo_onehot_f @ board_onehot_float.unsqueeze(-1)).squeeze(-1)
    allowed_mask_at = ~(overlaps_at > 0.0)

    # Compare results
    max_diff_bmm = (
        (allowed_mask_bool.float() - allowed_mask_bmm.float()).abs().max().item()
    )
    max_diff_at = (
        (allowed_mask_bool.float() - allowed_mask_at.float()).abs().max().item()
    )

    print(f"Max |diff| (bool vs bmm):  {max_diff_bmm:.3e}")
    print(f"Max |diff| (bool vs @):    {max_diff_at:.3e}")

    if max_diff_bmm > 1e-6 or max_diff_at > 1e-6:
        print("WARNING: Results differ between approaches!")


def run_for_device(
    device: torch.device,
    num_samples: int,
    repeats: int,
) -> None:
    print(f"\nDevice: {device}")
    print(f"num_samples: {num_samples:,}, repeats: {repeats}")

    combo_onehot_bool, combo_onehot_float, board_onehot, board_onehot_float = (
        prepare_tensors(num_samples, device)
    )

    # Verify correctness
    verify_correctness(
        combo_onehot_bool,
        combo_onehot_float,
        board_onehot,
        board_onehot_float,
        num_samples,
    )

    # Run benchmarks
    t_bool = bench_boolean_approach(
        combo_onehot_bool, board_onehot, num_samples, repeats
    )
    t_bmm = bench_matrix_multiply_approach(
        combo_onehot_float, board_onehot_float, num_samples, repeats
    )
    t_at = bench_matrix_multiply_at_approach(
        combo_onehot_float, board_onehot_float, num_samples, repeats
    )

    print(
        f"\nBoolean approach:       {t_bool:.6f} s total, {t_bool / repeats:.6e} s/iter"
    )
    print(f"Matrix multiply (bmm):  {t_bmm:.6f} s total, {t_bmm / repeats:.6e} s/iter")
    print(f"Matrix multiply (@):    {t_at:.6f} s total, {t_at / repeats:.6e} s/iter")
    print(f"Speedup (bool vs bmm):  {t_bool / t_bmm:.2f}x")
    print(f"Speedup (bool vs @):    {t_bool / t_at:.2f}x")
    print(f"Speedup (@ vs bmm):     {t_bmm / t_at:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark boolean vs matrix multiply approaches for combo masking"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples (num_samples)",
    )
    parser.add_argument(
        "--repeats", type=int, default=100, help="Number of timed iterations"
    )
    args = parser.parse_args()

    # MPS if available
    if torch.backends.mps.is_available():
        run_for_device(torch.device("mps"), args.num_samples, args.repeats)
    else:
        print("\nMPS not available; skipping MPS benchmarks.")

    # CUDA if available
    if torch.cuda.is_available():
        run_for_device(torch.device("cuda"), args.num_samples, args.repeats)
    else:
        print("\nCUDA not available; skipping CUDA benchmarks.")

    # If neither MPS nor CUDA, run on CPU
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("\nNo GPU available; running on CPU.")
        run_for_device(torch.device("cpu"), args.num_samples, args.repeats)


if __name__ == "__main__":
    main()
