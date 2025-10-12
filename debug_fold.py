import torch
import sys

sys.path.append("src")
sys.path.append("tests")

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv


# Create environment exactly like the test
def _make_env(
    N=1, starting_stack=1000, sb=5, bb=10, default_bet_bins=None, device=None, seed=123
):
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    env = HUNLTensorEnv(
        num_envs=N,
        starting_stack=starting_stack,
        sb=sb,
        bb=bb,
        default_bet_bins=default_bet_bins,
        device=device,
        rng=rng,
    )
    env.reset()
    return env


# Monkey patch to add debug info
original_finish_and_assign_rewards = HUNLTensorEnv.finish_and_assign_rewards


def debug_finish_and_assign_rewards(self, env_indices, winners):
    print(
        f"DEBUG finish_and_assign_rewards: env_indices={env_indices}, winners={winners}"
    )
    print(f"DEBUG: pot before = {self.pot[env_indices]}")
    print(f"DEBUG: stacks before = {self.stacks[env_indices]}")
    print(f"DEBUG: my_stack (player 0) = {self.stacks[env_indices, 0]}")

    result = original_finish_and_assign_rewards(self, env_indices, winners)

    print(f"DEBUG: result = {result}")
    print(f"DEBUG: pot after = {self.pot[env_indices]}")
    print(f"DEBUG: stacks after = {self.stacks[env_indices]}")
    return result


HUNLTensorEnv.finish_and_assign_rewards = debug_finish_and_assign_rewards

# Run the test scenario
env = _make_env(N=2, starting_stack=1000, sb=25, bb=50, default_bet_bins=[0.5, 1.0])

print("Initial state:")
print(f"  to_act: {env.to_act}")
print(f"  pot: {env.pot}")
print(f"  stacks: {env.stacks}")
print(f"  committed: {env.committed}")

# This should cause player 0 to fold in env 0
bin_indices = torch.tensor([0, -1], device=env.device)
print(f"\\nCalling step_bins with {bin_indices}")

r, d, ta, ns, dc = env.step_bins(bin_indices)

print(f"\\nResults:")
print(f"  rewards: {r}")
print(f"  dones: {d}")
print(f"  winners: {env.winner}")
print(f"  final stacks: {env.stacks}")
