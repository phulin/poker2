import argparse
import time
from typing import List, Tuple

import torch


def setup_tensors(
    num_parents: int, avg_children: int, feat_dim: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create parent features, child_counts per parent, and a parent_index per child.
    The structure mimics a tree where each parent has child_counts[p] children,
    and each child stores its parent index.
    """
    # Make child counts close to avg_children but with minor variance
    # Ensure at least 1 child per parent to avoid degenerate cases
    base = torch.full((num_parents,), avg_children, dtype=torch.int64, device=device)
    noise = torch.randint(
        low=-avg_children // 10,
        high=avg_children // 10 + 1,
        size=(num_parents,),
        dtype=torch.int64,
        device=device,
    )
    child_counts = torch.clamp(base + noise, min=1)  # on device

    parent_features = torch.randn(num_parents, feat_dim, device=device)

    # Construct parent_index for each child consistent with child_counts
    # Shape: [num_children]
    parent_index = torch.repeat_interleave(
        torch.arange(num_parents, device=device), child_counts
    )

    return parent_features, child_counts, parent_index


def device_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        # torch.mps.synchronize() has no device argument; it syncs current stream/device
        torch.mps.synchronize()
    else:
        # CPU: nothing to do
        pass


@torch.inference_mode()
def bench_repeat_interleave(
    parent_features: torch.Tensor, child_counts: torch.Tensor, iters: int
) -> float:
    """
    Measure time for expanding parent_features to children using repeat_interleave on child_counts.
    """
    device = parent_features.device
    device_synchronize(device)
    # Warmup
    for _ in range(3):
        _ = parent_features.repeat_interleave(child_counts, dim=0)
    device_synchronize(device)

    times: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = parent_features.repeat_interleave(child_counts, dim=0)
        device_synchronize(device)
        end = time.perf_counter()
        times.append(end - start)
    return min(times)


@torch.inference_mode()
def bench_gather(
    parent_features: torch.Tensor, parent_index: torch.Tensor, iters: int
) -> float:
    """
    Measure time for expanding parent_features to children using gather on parent_index.
    parent_index is of shape [num_children].
    """
    device = parent_features.device
    feat_dim = parent_features.shape[1]
    idx = parent_index.unsqueeze(1).expand(-1, feat_dim)

    device_synchronize(device)
    # Warmup
    for _ in range(3):
        _ = parent_features.gather(dim=0, index=idx)
    device_synchronize(device)

    times: List[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = parent_features.gather(dim=0, index=idx)
        device_synchronize(device)
        end = time.perf_counter()
        times.append(end - start)
    return min(times)


def humanize(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def run_once(
    num_parents: int, avg_children: int, feat_dim: int, iters: int, device: torch.device
) -> None:
    parent_features, child_counts, parent_index = setup_tensors(
        num_parents, avg_children, feat_dim, device
    )
    num_children = int(child_counts.sum().item())

    t_repeat = bench_repeat_interleave(parent_features, child_counts, iters)
    t_gather = bench_gather(parent_features, parent_index, iters)

    which = device.type.upper()
    print(
        f"[{which}] parents={humanize(num_parents)} children≈{humanize(num_children)} feat_dim={feat_dim}"
    )
    print(f"  repeat_interleave: {t_repeat * 1e3:.3f} ms")
    print(f"  gather           : {t_gather * 1e3:.3f} ms")
    speedup = (t_repeat / t_gather) if t_gather > 0 else float("inf")
    print(f"  speedup (repeat/gather): {speedup:.2f}x")
    print()


def detect_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available")
    if device_arg == "mps":
        if (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        raise RuntimeError(
            "MPS requested but torch.backends.mps.is_available() is False"
        )

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbenchmark: repeat_interleave(child_counts) vs gather(parent_index)"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Iterations per measurement; min time reported",
    )
    parser.add_argument(
        "--feat-dim", type=int, default=64, help="Feature dimension per parent/child"
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device selection strategy",
    )
    args = parser.parse_args()

    device = detect_device(args.device)
    torch.manual_seed(0)

    sizes = [
        # (num_parents, avg_children)
        (10_000, 2),
        (10_000, 4),
        (50_000, 2),
        (50_000, 4),
        (100_000, 2),
        (100_000, 4),
    ]

    for num_parents, avg_children in sizes:
        run_once(
            num_parents=num_parents,
            avg_children=avg_children,
            feat_dim=args.feat_dim,
            iters=args.iters,
            device=device,
        )


if __name__ == "__main__":
    main()
