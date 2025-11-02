#!/usr/bin/env python3
"""
Minimal reproducer showing that CFR iterations accumulate zero regrets.

The script mirrors the structure of `RebelCFREvaluator.self_play_iteration`.
For every iteration it prints:
    - the max absolute regret produced by the current implementation
      (`compute_instantaneous_regrets(self.values_avg)`), and
    - the max absolute regret we would get if we supplied the achieved
      counterfactual values (`self.latest_values`) instead.

With the current code, the first quantity stays at 0 while the second is
strictly positive once the policy starts changing, demonstrating the bug.
"""

import torch

from alphaholdem.core.structured_config import CFRType
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.search.rebel_cfr_evaluator import RebelCFREvaluator


def make_env(device: torch.device) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=1,
        starting_stack=1000,
        sb=5,
        bb=10,
        default_bet_bins=[0.5, 1.5],
        device=device,
        float_dtype=torch.float32,
        flop_showdown=False,
    )
    env.reset()
    return env


def make_model(num_actions: int, device: torch.device) -> RebelFFN:
    model = RebelFFN(
        input_dim=2661,
        num_actions=num_actions,
        hidden_dim=512,
        num_hidden_layers=2,
        detach_value_head=True,
        num_players=2,
    )
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(0)
    model.init_weights(cpu_rng)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    device = torch.device("cpu")
    env = make_env(device)
    bet_bins = [0.5, 1.5]
    num_actions = len(bet_bins) + 3
    model = make_model(num_actions, device)

    evaluator = RebelCFREvaluator(
        search_batch_size=1,
        env_proto=env,
        model=model,
        bet_bins=bet_bins,
        max_depth=1,
        cfr_iterations=25,
        device=device,
        float_dtype=torch.float32,
        warm_start_iterations=15,
        cfr_type=CFRType.linear,
        cfr_avg=True,
    )

    roots = torch.tensor([0], device=device)
    evaluator.initialize_search(env, roots)
    evaluator.construct_subgame()
    evaluator.initialize_policy_and_beliefs()

    if evaluator.warm_start_iterations > 0:
        evaluator.set_leaf_values(0)
        evaluator.compute_expected_values()
        evaluator.warm_start()

    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values

    print("t | bug_max | correct_max | max_diff | equal_latest_avg")
    print("--+---------+-------------+----------+-----------------")

    for t in range(evaluator.warm_start_iterations, evaluator.cfr_iterations):
        zero_regrets = evaluator.compute_instantaneous_regrets(evaluator.values_avg)
        correct_regrets = evaluator.compute_instantaneous_regrets(
            evaluator.latest_values, evaluator.values_avg
        )

        max_zero = float(zero_regrets.abs().max().item())
        max_correct = float(correct_regrets.abs().max().item())
        max_diff = float((zero_regrets - correct_regrets).abs().max().item())
        equal_values = torch.allclose(
            evaluator.latest_values, evaluator.values_avg, atol=1e-6
        )

        print(
            f"{t:2d}| {max_zero:7.4f} | {max_correct:11.4f} | {max_diff:8.5f} | {equal_values}"
        )

        if evaluator.cfr_type == CFRType.linear:
            zero_regrets.masked_fill_(
                evaluator.env.to_act[:, None] == t % evaluator.num_players, 0.0
            )
        elif evaluator.cfr_type == CFRType.discounted:
            factor = torch.where(
                zero_regrets > 0,
                (t - 1) ** evaluator.dcfr_alpha,
                (t - 1) ** evaluator.dcfr_beta,
            )
            evaluator.cumulative_regrets *= factor / (factor + 1)
            evaluator.regret_weight_sums *= factor / (factor + 1)

        evaluator.regret_weight_sums += 1
        evaluator.cumulative_regrets += zero_regrets

        old_policy_probs = evaluator.policy_probs.clone()
        evaluator.update_policy(t)
        evaluator._record_stats(t, old_policy_probs)

        evaluator.set_leaf_values(t)
        evaluator.compute_expected_values()

        old, new = evaluator._get_mixing_weights(t)
        evaluator.values_avg *= old
        evaluator.values_avg += new * evaluator.latest_values
        evaluator.values_avg /= old + new


if __name__ == "__main__":
    main()
