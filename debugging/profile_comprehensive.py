#!/usr/bin/env python3
"""
Comprehensive line-by-line profiling of all key methods in RebelCFREvaluator.
"""

import torch

from p2.env.card_utils import NUM_HANDS
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.search.rebel_cfr_evaluator import RebelCFREvaluator


# MockModel for testing
class MockModel:
    def __init__(
        self,
        logits=None,
        hand_values=None,
        num_actions=None,
        num_players=2,
        device=None,
        dtype=None,
    ):
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.dtype = dtype
        self.logits = logits
        self.hand_values = hand_values

    def __call__(self, features):
        batch_size = len(features)
        if self.logits is not None:
            logits = (
                self.logits.unsqueeze(0)
                .expand(batch_size, NUM_HANDS, -1)
                .to(self.device)
            )
        else:
            logits = torch.zeros(
                batch_size, NUM_HANDS, 6, device=self.device, dtype=self.dtype
            )

        if self.hand_values is not None:
            values = self.hand_values.expand(batch_size, -1, -1).to(self.device)
        else:
            values = torch.zeros(
                batch_size, 2, NUM_HANDS, device=self.device, dtype=self.dtype
            )

        # Return a mock object with the expected attributes
        class MockOutput:
            def __init__(self, policy_logits, hand_values):
                self.policy_logits = policy_logits
                self.hand_values = hand_values

        return MockOutput(logits, values)


def comprehensive_profiling():
    """Run comprehensive profiling of all key CFR methods."""
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
    float_dtype = torch.float32

    print(f"Using device: {device}")
    print("Running comprehensive CFR profiling...")

    # Create environment
    env_proto = HUNLTensorEnv(
        num_envs=5,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=device,
        float_dtype=float_dtype,
        flop_showdown=False,
    )

    # Create model
    bet_bins = env_proto.default_bet_bins
    model = MockModel(
        logits=torch.zeros(len(bet_bins) + 3, dtype=float_dtype),
        hand_values=torch.zeros(5, 2, NUM_HANDS, dtype=float_dtype),
        device=device,
        dtype=float_dtype,
    )

    # Create evaluator
    evaluator = RebelCFREvaluator(
        search_batch_size=5,
        env_proto=env_proto,
        model=model,
        bet_bins=bet_bins,
        max_depth=2,
        cfr_iterations=150,
        device=device,
        float_dtype=float_dtype,
    )

    # Initialize search
    roots = torch.arange(evaluator.root_nodes)
    evaluator.initialize_subgame(env_proto, roots)

    print("Profiling realistic CFR workflow...")

    # Profile the realistic workflow: initialize_search + self_play_iteration
    print("\n=== Profiling initialize_search + self_play_iteration ===")

    # Initialize search (this will be profiled)
    evaluator.initialize_subgame(env_proto, roots)

    # Run the realistic workflow multiple times
    for i in range(10):
        print(f"Running iteration {i + 1}/10")

        # Run a complete self-play iteration (this will be profiled)
        pbs = evaluator.evaluate_cfr(training_mode=True)
        if pbs is not None:
            print(f"  Generated {pbs.beliefs.shape[0]} samples")

    print("\n=== Profiling complete! ===")
    print("Run with kernprof to see line-by-line results:")
    print("kernprof -l -v profile_comprehensive.py")


if __name__ == "__main__":
    comprehensive_profiling()
