import torch
import time


def current_impl(hand_values, player_beliefs, ignore_mask=None):
    # (B, P, H) * (B, P, H) -> (B, P, H) -> sum(2) -> (B, P, 1) -> mean(1) -> (B, 1, 1)
    hand_value_sums = (
        (hand_values * player_beliefs)
        .sum(dim=2, keepdim=True)
        .mean(dim=1, keepdim=True)
    )
    if ignore_mask is not None:
        hand_value_sums.masked_fill_(ignore_mask[:, None, None], 0.0)
    return hand_values - hand_value_sums


@torch.compile
def matmul_impl(hand_values, player_beliefs, ignore_mask=None):
    # hand_values: (B, P, H)
    # player_beliefs: (B, P, H)
    B, P, H = hand_values.shape

    # Reshape for bmm: (B*P, 1, H) @ (B*P, H, 1) -> (B*P, 1, 1)
    # We treat hand_values as row vectors and player_beliefs as column vectors
    hv_flat = hand_values.view(B * P, 1, H)
    pb_flat = player_beliefs.view(B * P, H, 1)

    # (B*P, 1, 1)
    sums = torch.bmm(hv_flat, pb_flat)

    # Reshape back to (B, P, 1)
    sums = sums.view(B, P, 1)

    # Mean over players
    means = sums.mean(dim=1, keepdim=True)  # (B, 1, 1)

    if ignore_mask is not None:
        means.masked_fill_(ignore_mask[:, None, None], 0.0)

    return hand_values - means


def benchmark():
    if not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = torch.device("cpu")
    else:
        print("Using MPS")
        device = torch.device("mps")

    B = 1024  # Batch size
    P = 2  # Players
    H = 1326  # Hands

    print(f"Benchmarking with B={B}, P={P}, H={H}")

    # Create random tensors
    hand_values = torch.randn(B, P, H, device=device, dtype=torch.float32)
    player_beliefs = torch.rand(B, P, H, device=device, dtype=torch.float32)
    # Normalize beliefs to sum to 1 roughly, though not strictly needed for perf
    player_beliefs = player_beliefs / player_beliefs.sum(dim=2, keepdim=True)

    ignore_mask = torch.rand(B, device=device) < 0.1

    # Warmup
    for _ in range(10):
        _ = current_impl(hand_values, player_beliefs, ignore_mask)
        _ = matmul_impl(hand_values, player_beliefs, ignore_mask)

    if device.type == "mps":
        torch.mps.synchronize()

    iterations = 1000

    # Benchmark Current
    start = time.time()
    for _ in range(iterations):
        _ = current_impl(hand_values, player_beliefs, ignore_mask)
    if device.type == "mps":
        torch.mps.synchronize()
    end = time.time()
    current_time = (end - start) / iterations
    print(f"Current Implementation: {current_time*1000:.4f} ms per iter")

    # Benchmark Matmul
    start = time.time()
    for _ in range(iterations):
        _ = matmul_impl(hand_values, player_beliefs, ignore_mask)
    if device.type == "mps":
        torch.mps.synchronize()
    end = time.time()
    matmul_time = (end - start) / iterations
    print(f"Matmul Implementation:  {matmul_time*1000:.4f} ms per iter")

    # Verify correctness
    res_current = current_impl(hand_values, player_beliefs, ignore_mask)
    res_matmul = matmul_impl(hand_values, player_beliefs, ignore_mask)

    # Check max difference
    diff = (res_current - res_matmul).abs().max().item()
    print(f"Max difference: {diff:.2e}")
    if diff < 1e-5:
        print("Implementations match!")
    else:
        print("Implementations DO NOT match!")


if __name__ == "__main__":
    benchmark()
