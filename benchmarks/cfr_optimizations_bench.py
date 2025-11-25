"""Microbenchmarks for CFR evaluator optimizations.

This benchmark suite tests 5 optimization ideas:
1. Fused belief blocking + normalization
2. Log-space reach weight calculation (replacing scatter_reduce prod)
3. Optimized repeat_interleave in _fan_out
4. Cached combo_to_onehot calculations
5. Vectorized showdown value computation

Each optimization has:
- Baseline implementation (current code)
- Optimized implementation
- Correctness test
- Performance benchmark
"""

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    baseline_time: float
    optimized_time: float
    speedup: float
    passes_correctness: bool
    max_diff: float

    def __str__(self) -> str:
        status = "✓" if self.passes_correctness else "✗"
        return (
            f"{status} {self.name:50s} | "
            f"Baseline: {self.baseline_time*1000:8.3f}ms | "
            f"Optimized: {self.optimized_time*1000:8.3f}ms | "
            f"Speedup: {self.speedup:5.2f}x | "
            f"Max diff: {self.max_diff:.2e}"
        )


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def benchmark_function(
    func: Callable, *args, warmup: int = 10, iterations: int = 100, **kwargs
) -> float:
    """Benchmark a function with warmup and multiple iterations."""
    device = args[0].device if torch.is_tensor(args[0]) else get_device()

    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = func(*args, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    end = time.perf_counter()

    return (end - start) / iterations


# ============================================================================
# Optimization 1: Fused belief blocking + normalization
# ============================================================================


def baseline_block_and_normalize(
    beliefs: torch.Tensor, allowed_hands: torch.Tensor, allowed_hands_prob: torch.Tensor
) -> torch.Tensor:
    """Baseline: separate blocking and normalization (current implementation)."""
    beliefs = beliefs.clone()
    # Block beliefs
    beliefs.masked_fill_((~allowed_hands)[:, None, :], 0.0)

    # Normalize beliefs
    denom = beliefs.sum(dim=-1, keepdim=True)
    beliefs = torch.where(
        denom > 1e-8,
        beliefs / denom.clamp(min=1e-8),
        allowed_hands_prob[:, None, :],
    )
    return beliefs


def optimized_block_and_normalize(
    beliefs: torch.Tensor, allowed_hands: torch.Tensor, allowed_hands_prob: torch.Tensor
) -> torch.Tensor:
    """Optimized: fused blocking and normalization in single pass."""
    # Mask and compute sum in one pass
    masked_beliefs = torch.where(
        allowed_hands[:, None, :], beliefs, torch.zeros_like(beliefs)
    )
    denom = masked_beliefs.sum(dim=-1, keepdim=True)
    return torch.where(
        denom > 1e-8,
        masked_beliefs / denom.clamp(min=1e-8),
        allowed_hands_prob[:, None, :],
    )


def test_block_and_normalize(
    device: torch.device, batch_size: int = 128, num_hands: int = 1326
):
    """Test correctness and performance of belief blocking/normalization."""
    beliefs = torch.rand(batch_size, 2, num_hands, device=device)
    beliefs /= beliefs.sum(dim=-1, keepdim=True)
    allowed_hands = torch.rand(batch_size, num_hands, device=device) > 0.3
    allowed_hands_prob = allowed_hands.float()
    allowed_hands_prob /= allowed_hands_prob.sum(dim=-1, keepdim=True).clamp(min=1.0)

    # Correctness test
    baseline_result = baseline_block_and_normalize(
        beliefs, allowed_hands, allowed_hands_prob
    )
    optimized_result = optimized_block_and_normalize(
        beliefs, allowed_hands, allowed_hands_prob
    )
    max_diff = (baseline_result - optimized_result).abs().max().item()
    passes = max_diff < 1e-5

    # Performance test
    baseline_time = benchmark_function(
        baseline_block_and_normalize, beliefs, allowed_hands, allowed_hands_prob
    )
    optimized_time = benchmark_function(
        optimized_block_and_normalize, beliefs, allowed_hands, allowed_hands_prob
    )

    return BenchmarkResult(
        name="1. Fused belief blocking + normalization",
        baseline_time=baseline_time,
        optimized_time=optimized_time,
        speedup=baseline_time / optimized_time,
        passes_correctness=passes,
        max_diff=max_diff,
    )


# ============================================================================
# Optimization 2: Log-space reach weight calculation
# ============================================================================


def baseline_reach_weights(
    target: torch.Tensor,
    policy: torch.Tensor,
    prev_actor: torch.Tensor,
    num_hands: int = 1326,
) -> torch.Tensor:
    """Baseline: scatter_reduce with prod (current implementation)."""
    target = target.clone()
    prev_actor_indices = prev_actor[:, None, None].expand(-1, -1, num_hands)
    target.scatter_reduce_(
        dim=1,
        index=prev_actor_indices,
        src=policy[:, None],
        reduce="prod",
        include_self=True,
    )
    return target


def optimized_reach_weights(
    target: torch.Tensor,
    policy: torch.Tensor,
    prev_actor: torch.Tensor,
    num_hands: int = 1326,
) -> torch.Tensor:
    """Optimized: use log-space for numerical stability and potentially better performance."""
    target = target.clone()

    # Work in log space
    log_target = torch.log(target.clamp(min=1e-45))
    log_policy = torch.log(policy.clamp(min=1e-45))

    prev_actor_indices = prev_actor[:, None, None].expand(-1, -1, num_hands)

    # Sum in log space (equivalent to product in linear space)
    log_target.scatter_reduce_(
        dim=1,
        index=prev_actor_indices,
        src=log_policy[:, None],
        reduce="sum",
        include_self=True,
    )

    # Convert back
    return torch.exp(log_target)


def test_reach_weights(
    device: torch.device, batch_size: int = 256, num_hands: int = 1326
):
    """Test correctness and performance of reach weight calculation."""
    target = torch.rand(batch_size, 2, num_hands, device=device).clamp(min=0.01)
    policy = torch.rand(batch_size, num_hands, device=device)
    prev_actor = torch.randint(0, 2, (batch_size,), device=device)

    # Correctness test
    baseline_result = baseline_reach_weights(target, policy, prev_actor, num_hands)
    optimized_result = optimized_reach_weights(target, policy, prev_actor, num_hands)

    # Use relative difference due to numerical precision
    rel_diff = (
        ((baseline_result - optimized_result).abs() / (baseline_result.abs() + 1e-8))
        .max()
        .item()
    )
    max_diff = (baseline_result - optimized_result).abs().max().item()
    passes = rel_diff < 1e-4  # More lenient for numerical reasons

    # Performance test
    baseline_time = benchmark_function(
        baseline_reach_weights, target, policy, prev_actor, num_hands
    )
    optimized_time = benchmark_function(
        optimized_reach_weights, target, policy, prev_actor, num_hands
    )

    return BenchmarkResult(
        name="2. Log-space reach weight calculation",
        baseline_time=baseline_time,
        optimized_time=optimized_time,
        speedup=baseline_time / optimized_time,
        passes_correctness=passes,
        max_diff=max_diff,
    )


# ============================================================================
# Optimization 3: Optimized repeat_interleave in _fan_out
# ============================================================================


def baseline_fan_out(
    tensor: torch.Tensor, child_count: torch.Tensor, output_size: int
) -> torch.Tensor:
    """Baseline: use repeat_interleave (current implementation)."""
    return tensor.repeat_interleave(child_count, dim=0, output_size=output_size)


def optimized_fan_out(
    tensor: torch.Tensor, child_count: torch.Tensor, output_size: int
) -> torch.Tensor:
    """Optimized: use index-based approach for potentially better performance."""
    # Build index array
    indices = torch.repeat_interleave(
        torch.arange(tensor.shape[0], device=tensor.device),
        child_count,
        output_size=output_size,
    )
    return tensor[indices]


def test_fan_out(device: torch.device, num_nodes: int = 512):
    """Test correctness and performance of fan_out operation."""
    # Create realistic child count distribution
    child_count = torch.randint(1, 8, (num_nodes,), device=device)
    output_size = child_count.sum().item()

    # Create multi-dimensional tensor
    tensor = torch.randn(num_nodes, 2, 1326, device=device)

    # Correctness test
    baseline_result = baseline_fan_out(tensor, child_count, output_size)
    optimized_result = optimized_fan_out(tensor, child_count, output_size)
    max_diff = (baseline_result - optimized_result).abs().max().item()
    passes = max_diff < 1e-6

    # Performance test
    baseline_time = benchmark_function(
        baseline_fan_out, tensor, child_count, output_size
    )
    optimized_time = benchmark_function(
        optimized_fan_out, tensor, child_count, output_size
    )

    return BenchmarkResult(
        name="3. Optimized repeat_interleave in _fan_out",
        baseline_time=baseline_time,
        optimized_time=optimized_time,
        speedup=baseline_time / optimized_time,
        passes_correctness=passes,
        max_diff=max_diff,
    )


# ============================================================================
# Optimization 4: Cached combo_to_onehot calculations
# ============================================================================


def baseline_hand_blocking(
    combo_onehot: torch.Tensor, board_mask: torch.Tensor
) -> torch.Tensor:
    """Baseline: compute blocking on the fly (current pattern)."""
    # Simulate multiple calls as would happen in practice
    results = []
    for _ in range(10):
        result = (combo_onehot @ board_mask.T).T < 0.5
        results.append(result)
    return torch.stack(results)


def optimized_hand_blocking(
    combo_onehot: torch.Tensor, board_mask: torch.Tensor, cache: dict
) -> torch.Tensor:
    """Optimized: cache intermediate matrix multiplication result."""
    # Use board mask hash as cache key
    cache_key = id(board_mask)

    if cache_key not in cache:
        cache[cache_key] = combo_onehot @ board_mask.T

    results = []
    for _ in range(10):
        result = cache[cache_key].T < 0.5
        results.append(result)
    return torch.stack(results)


def test_hand_blocking(device: torch.device, batch_size: int = 128):
    """Test correctness and performance of hand blocking with caching."""
    num_hands = 1326
    num_cards = 52

    # Create combo_onehot tensor (simulated - real one would be loaded)
    combo_onehot = torch.rand(num_hands, num_cards, device=device) > 0.5
    combo_onehot = combo_onehot.float()

    # Create board masks
    board_mask = torch.zeros(batch_size, num_cards, device=device)
    for i in range(batch_size):
        board_indices = torch.randperm(num_cards, device=device)[:5]
        board_mask[i, board_indices] = 1.0

    cache = {}

    # Correctness test
    baseline_result = baseline_hand_blocking(combo_onehot, board_mask)
    optimized_result = optimized_hand_blocking(combo_onehot, board_mask, cache)
    max_diff = (baseline_result.float() - optimized_result.float()).abs().max().item()
    passes = max_diff < 1e-6

    # Performance test
    baseline_time = benchmark_function(
        baseline_hand_blocking, combo_onehot, board_mask, warmup=5, iterations=50
    )
    cache.clear()
    optimized_time = benchmark_function(
        optimized_hand_blocking,
        combo_onehot,
        board_mask,
        cache,
        warmup=5,
        iterations=50,
    )

    return BenchmarkResult(
        name="4. Cached combo_to_onehot calculations",
        baseline_time=baseline_time,
        optimized_time=optimized_time,
        speedup=baseline_time / optimized_time,
        passes_correctness=passes,
        max_diff=max_diff,
    )


# ============================================================================
# Optimization 5: Vectorized showdown value computation
# ============================================================================


def baseline_showdown_prefix_sums(
    b_opp_sorted: torch.Tensor, H_sorted: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: compute prefix sums with separate operations."""
    M, num_hands = b_opp_sorted.shape
    device = b_opp_sorted.device
    dtype = b_opp_sorted.dtype

    # Global prefix sum
    P = torch.cumsum(b_opp_sorted, dim=1)
    P = torch.cat([torch.zeros(M, 1, device=device, dtype=dtype), P], dim=1)

    # Per-card prefix sums
    per_card_mass = H_sorted.to(dtype) * b_opp_sorted.unsqueeze(-1)
    Pcards = torch.cumsum(per_card_mass, dim=1)
    Pcards = torch.cat(
        [torch.zeros(M, 1, 52, device=device, dtype=dtype), Pcards], dim=1
    )

    return P, Pcards


def optimized_showdown_prefix_sums(
    b_opp_sorted: torch.Tensor, H_sorted: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized: compute both prefix sums in more efficient way."""
    M, num_hands = b_opp_sorted.shape
    device = b_opp_sorted.device
    dtype = b_opp_sorted.dtype

    # Pad first, then cumsum (avoids cat overhead)
    P = F.pad(b_opp_sorted, (1, 0), value=0.0)
    P[:, 1:] = torch.cumsum(b_opp_sorted, dim=1)

    # Compute per-card mass with better memory pattern
    per_card_mass = b_opp_sorted.unsqueeze(-1) * H_sorted.to(dtype)
    Pcards = F.pad(per_card_mass, (0, 0, 1, 0), value=0.0)
    Pcards[:, 1:, :] = torch.cumsum(per_card_mass, dim=1)

    return P, Pcards


def test_showdown_prefix_sums(device: torch.device, batch_size: int = 64):
    """Test correctness and performance of showdown prefix sum computation."""
    num_hands = 1326
    num_cards = 52

    # Create test data
    b_opp_sorted = torch.rand(batch_size, num_hands, device=device)
    b_opp_sorted /= b_opp_sorted.sum(dim=1, keepdim=True)

    H_sorted = torch.rand(batch_size, num_hands, num_cards, device=device) > 0.5

    # Correctness test
    P_base, Pcards_base = baseline_showdown_prefix_sums(b_opp_sorted, H_sorted)
    P_opt, Pcards_opt = optimized_showdown_prefix_sums(b_opp_sorted, H_sorted)

    max_diff_P = (P_base - P_opt).abs().max().item()
    max_diff_Pcards = (Pcards_base - Pcards_opt).abs().max().item()
    max_diff = max(max_diff_P, max_diff_Pcards)
    passes = max_diff < 1e-5

    # Performance test
    baseline_time = benchmark_function(
        baseline_showdown_prefix_sums, b_opp_sorted, H_sorted
    )
    optimized_time = benchmark_function(
        optimized_showdown_prefix_sums, b_opp_sorted, H_sorted
    )

    return BenchmarkResult(
        name="5. Vectorized showdown value computation",
        baseline_time=baseline_time,
        optimized_time=optimized_time,
        speedup=baseline_time / optimized_time,
        passes_correctness=passes,
        max_diff=max_diff,
    )


# ============================================================================
# Optimization 6: Safe division patterns
# ============================================================================


def baseline_safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Baseline: torch.where with condition (current pattern)."""
    return torch.where(
        denominator > eps,
        numerator / denominator.clamp(min=eps),
        fallback,
    )


def opt1_safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Option 1: Use clamp + where without redundant clamp."""
    denom_clamped = denominator.clamp(min=eps)
    return torch.where(
        denominator > eps,
        numerator / denom_clamped,
        fallback,
    )


def opt2_safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Option 2: Divide by clamped, then fix up small values."""
    result = numerator / denominator.clamp(min=eps)
    return torch.where(denominator > eps, result, fallback)


def opt3_safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Option 3: Masked operations (avoids division on invalid elements)."""
    result = torch.empty_like(numerator)
    mask = denominator > eps
    result[mask] = numerator[mask] / denominator[mask]
    result[~mask] = fallback[~mask]
    return result


def opt4_safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    fallback: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Option 4: Use reciprocal (sometimes faster than division)."""
    safe_denom = torch.where(
        denominator > eps, denominator, torch.ones_like(denominator)
    )
    reciprocal = 1.0 / safe_denom
    result = numerator * reciprocal
    return torch.where(denominator > eps, result, fallback)


def test_safe_divide(
    device: torch.device, batch_size: int = 512, num_hands: int = 1326
):
    """Test different safe division patterns."""
    # Create test data with some zero/near-zero denominators
    numerator = torch.rand(batch_size, 2, num_hands, device=device)
    denominator = torch.rand(batch_size, 2, num_hands, device=device)
    # Make ~20% of denominators very small
    small_mask = torch.rand(batch_size, 2, num_hands, device=device) < 0.2
    denominator[small_mask] = torch.rand(small_mask.sum().item(), device=device) * 1e-10

    fallback = torch.ones(batch_size, 2, num_hands, device=device) * 0.5
    eps = 1e-8

    # Correctness tests
    baseline_result = baseline_safe_divide(numerator, denominator, fallback, eps)

    results = {}
    functions = {
        "baseline (where + clamp)": baseline_safe_divide,
        "opt1 (clamp once)": opt1_safe_divide,
        "opt2 (divide first)": opt2_safe_divide,
        "opt3 (masked ops)": opt3_safe_divide,
        "opt4 (reciprocal)": opt4_safe_divide,
    }

    print("\n6. Safe Division Pattern Comparison:")
    print("=" * 100)

    for name, func in functions.items():
        result = func(numerator, denominator, fallback, eps)
        max_diff = (baseline_result - result).abs().max().item()
        passes = max_diff < 1e-6

        bench_time = benchmark_function(func, numerator, denominator, fallback, eps)

        status = "✓" if passes else "✗"
        print(
            f"  {status} {name:30s} | Time: {bench_time*1000:8.3f}ms | Max diff: {max_diff:.2e}"
        )

        results[name] = {"time": bench_time, "passes": passes, "max_diff": max_diff}

    # Find best
    best_name = min(
        (k for k in results.keys() if k != "baseline (where + clamp)"),
        key=lambda k: results[k]["time"],
    )
    speedup = results["baseline (where + clamp)"]["time"] / results[best_name]["time"]

    print(f"\n  Best: {best_name} with {speedup:.2f}x speedup")
    print()

    return BenchmarkResult(
        name="6. Safe division patterns",
        baseline_time=results["baseline (where + clamp)"]["time"],
        optimized_time=results[best_name]["time"],
        speedup=speedup,
        passes_correctness=results[best_name]["passes"],
        max_diff=results[best_name]["max_diff"],
    )


# ============================================================================
# Main benchmark runner
# ============================================================================


def run_all_benchmarks(device: torch.device | None = None):
    """Run all optimization benchmarks."""
    if device is None:
        device = get_device()

    print(f"\nRunning CFR Evaluator Optimization Benchmarks on {device}")
    print("=" * 100)
    print()

    # Run all tests
    tests = [
        test_block_and_normalize,
        test_reach_weights,
        test_fan_out,
        test_hand_blocking,
        test_showdown_prefix_sums,
        test_safe_divide,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func(device)
            results.append(result)
            print(result)
        except Exception as e:
            print(f"✗ {test_func.__name__}: FAILED with error: {e}")

    print()
    print("=" * 100)
    print("\nSummary:")
    print(
        f"  Tests passed: {sum(r.passes_correctness for r in results)}/{len(results)}"
    )
    print(f"  Average speedup: {sum(r.speedup for r in results) / len(results):.2f}x")
    print(
        f"  Best speedup: {max(r.speedup for r in results):.2f}x ({max(results, key=lambda r: r.speedup).name})"
    )
    print()

    return results


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Run benchmarks
    results = run_all_benchmarks()
