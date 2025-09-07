import torch
import pytest

from alphaholdem.rl.vectorized_replay import VectorizedReplayBuffer


def compute_gae_reference(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Reference GAE using straightforward per-trajectory reverse recursion.

    Args:
        rewards: [T]
        values: [T]
        dones: [T] bool
    Returns:
        advantages: [T]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)

    # Identify trajectory starts: t==0 or previous is done
    is_start = torch.zeros_like(dones, dtype=torch.bool)
    is_start[0] = True
    is_start[1:] = dones[:-1]
    starts = torch.where(is_start)[0]
    ends = torch.cat([starts[1:], torch.tensor([T], device=rewards.device)])

    for s, e in zip(starts.tolist(), ends.tolist()):
        # deltas for this traj
        deltas = torch.zeros(e - s, device=rewards.device, dtype=rewards.dtype)
        # non-terminal: r_t + gamma * V_{t+1} - V_t
        if e - s > 1:
            deltas[:-1] = (
                rewards[s : e - 1] + gamma * values[s + 1 : e] - values[s : e - 1]
            )
        # terminal: r_t - V_t wherever done==True inside [s, e)
        dseg = dones[s:e]
        rseg = rewards[s:e]
        vseg = values[s:e]
        deltas = torch.where(dseg, rseg - vseg, deltas)

        # reverse recursion
        adv = torch.zeros_like(deltas)
        for t in range(e - s - 1, -1, -1):
            if t == e - s - 1:
                adv[t] = deltas[t]
            else:
                adv[t] = deltas[t] + gamma * lam * adv[t + 1] * (
                    1.0 - dseg[t].to(deltas.dtype)
                )
        advantages[s:e] = adv

    return advantages


@pytest.mark.parametrize(
    "rewards,values,dones",
    [
        # Single terminal in the middle; rest non-terminal
        (
            torch.tensor([0.0, 0.0, 0.0, 0.995, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([-0.5, -0.2, -0.1, -0.2, -0.3, -0.1], dtype=torch.float32),
            torch.tensor([False, False, False, True, False, False]),
        ),
        # Multiple short trajectories in one flat buffer
        (
            torch.tensor([0.0, 0.995, 0.0, 0.0, -0.349, 0.0], dtype=torch.float32),
            torch.tensor([-0.1, -0.2, -0.1, -0.2, 0.04, -0.05], dtype=torch.float32),
            torch.tensor([False, True, False, False, True, False]),
        ),
        # Edge: last step non-terminal
        (
            torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([-0.2, -0.1, -0.15, -0.3], dtype=torch.float32),
            torch.tensor([False, False, False, False]),
        ),
    ],
)
def test_gae_vectorized_matches_reference(rewards, values, dones):
    device = torch.device("cpu")
    rewards = rewards.to(device)
    values = values.to(device)
    dones = dones.to(device)
    gamma = 0.999
    lam = 0.95

    # Reference
    ref_adv = compute_gae_reference(rewards, values, dones, gamma, lam)

    # DUT: call buffer's batch vectorized GAE directly to isolate rectification
    buf = VectorizedReplayBuffer(
        capacity=128, observation_dim=1, legal_mask_dim=1, device=device
    )
    adv_vec = buf._compute_gae_batch_vectorized(rewards, values, dones, gamma, lam)

    # Find first mismatch if any
    close = torch.isclose(ref_adv, adv_vec, rtol=1e-5, atol=1e-6)
    if not bool(close.all().item()):
        idx = torch.where(~close)[0][0].item()
        msg = (
            f"GAE mismatch at index {idx}: ref={ref_adv[idx].item():.8f}, "
            f"vec={adv_vec[idx].item():.8f}\n"
            f"rewards={rewards.tolist()}\nvalues={values.tolist()}\n"
            f"dones={dones.tolist()}\nref_adv={ref_adv.tolist()}\nvec_adv={adv_vec.tolist()}"
        )
        pytest.fail(msg)

    # Also compare returns
    ref_ret = ref_adv + values
    vec_ret = adv_vec + values
    close_r = torch.isclose(ref_ret, vec_ret, rtol=1e-5, atol=1e-6)
    if not bool(close_r.all().item()):
        idx = torch.where(~close_r)[0][0].item()
        msg = (
            f"Return mismatch at index {idx}: ref={ref_ret[idx].item():.8f}, "
            f"vec={vec_ret[idx].item():.8f}"
        )
        pytest.fail(msg)
