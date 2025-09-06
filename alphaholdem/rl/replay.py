from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import torch


@dataclass
class Transition:
    """Single transition in a trajectory."""

    observation: torch.Tensor  # encoded state
    action: int  # discrete action index
    log_prob: float  # log probability of action
    reward: float
    done: bool
    legal_mask: torch.Tensor  # legal action mask
    chips_placed: int  # for δ2/δ3 computation
    # Per-sample clipping bounds
    delta2: float = 0.0
    delta3: float = 0.0
    value: float = 0.0  # V(s_t) computed at action time
    advantage: float = 0.0  # GAE advantage (computed later)
    return_: float = 0.0  # GAE return (computed later)


@dataclass
class Trajectory:
    """Complete trajectory from reset to terminal."""

    transitions: List[Transition]


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.trajectories: List[Trajectory] = []
        self.position = 0

    def add_trajectory(self, trajectory: Trajectory) -> None:
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(trajectory)
        else:
            self.trajectories[self.position] = trajectory
            self.position = (self.position + 1) % self.capacity

    def sample_trajectories(self, num_trajectories: int) -> List[Trajectory]:
        """Sample trajectories for PPO updates."""
        if len(self.trajectories) == 0:
            return []
        indices = torch.randint(0, len(self.trajectories), (num_trajectories,))
        return [self.trajectories[i] for i in indices]

    def clear(self) -> None:
        self.trajectories.clear()
        self.position = 0

    def num_steps(self) -> int:
        """Total number of transitions (steps) stored across all trajectories."""
        return sum(len(t.transitions) for t in self.trajectories)

    def trim_to_steps(self, max_steps: int) -> None:
        """Trim oldest trajectories until total steps <= max_steps."""
        while self.num_steps() > max_steps and self.trajectories:
            # Remove the oldest trajectory (FIFO semantics)
            self.trajectories.pop(0)


def compute_gae_returns(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.999,
    lambda_: float = 0.95,
) -> tuple[List[float], List[float]]:
    """Compute GAE advantages and returns."""
    advantages = []
    returns = []

    # Compute advantages using GAE
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # Terminal state
            delta = rewards[t] - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] - values[t]

        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)

    # Compute returns from advantages
    for t in range(len(rewards)):
        returns.append(advantages[t] + values[t])

    return advantages, returns


def prepare_ppo_batch(trajectories: List[Trajectory]) -> dict:
    """Prepare batch for PPO updates."""
    observations = []
    actions = []
    log_probs_old = []
    advantages = []
    returns = []
    legal_masks = []
    delta2_list: List[float] = []
    delta3_list: List[float] = []

    for trajectory in trajectories:
        for transition in trajectory.transitions:
            observations.append(transition.observation)
            actions.append(transition.action)
            log_probs_old.append(transition.log_prob)
            advantages.append(transition.advantage)
            returns.append(transition.return_)
            legal_masks.append(transition.legal_mask)
            delta2_list.append(transition.delta2)
            delta3_list.append(transition.delta3)

    return {
        "observations": torch.stack(observations),
        "actions": torch.tensor(actions, dtype=torch.long),
        "log_probs_old": torch.tensor(log_probs_old, dtype=torch.float32),
        "advantages": torch.tensor(advantages, dtype=torch.float32),
        "returns": torch.tensor(returns, dtype=torch.float32),
        "legal_masks": torch.stack(legal_masks),
        "delta2": torch.tensor(delta2_list, dtype=torch.float32),
        "delta3": torch.tensor(delta3_list, dtype=torch.float32),
    }
