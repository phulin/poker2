from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.card_utils import combo_blocking_tensor
from alphaholdem.env.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.utils.model_utils import compute_masked_logits

T_WARM = 15
NUM_HANDS = 1326


@dataclass
class CFRResult:
    """Result from CFR search containing policies, values, and belief states."""

    root_policy: torch.Tensor  # [batch_size, num_actions]
    root_policy_avg: torch.Tensor  # [batch_size, num_actions]
    root_policy_sampled: torch.Tensor  # [batch_size, num_actions] for safe search
    root_values: torch.Tensor  # [batch_size]
    belief_states: Dict[int, torch.Tensor]  # depth -> belief states
    training_targets: Dict[str, torch.Tensor]  # for value network training


@dataclass
class TrainingBatch:
    """Training data extracted from CFR search."""

    features: torch.Tensor  # [batch_size, feature_dim]
    policy_targets: torch.Tensor  # [batch_size, num_actions]
    value_targets: torch.Tensor  # [batch_size, num_players, belief_dim]
    belief_states: torch.Tensor  # [batch_size, num_players, belief_dim]
    weights: torch.Tensor  # [batch_size]


@dataclass
class PublicBeliefState:
    """Public belief state for both players (? TODO)."""

    env: HUNLTensorEnv
    beliefs: torch.Tensor  # [batch_size, num_players, belief_dim]

    @classmethod
    def from_proto(
        cls, proto: HUNLTensorEnv, beliefs: torch.Tensor
    ) -> PublicBeliefState:
        return PublicBeliefState(
            env=HUNLTensorEnv(
                num_envs=proto.num_envs,
                starting_stack=proto.starting_stack,
                sb=proto.sb,
                bb=proto.bb,
                default_bet_bins=proto.default_bet_bins,
                device=proto.device,
                float_dtype=proto.float_dtype,
                flop_showdown=proto.flop_showdown,
            ),
            beliefs=beliefs,
        )


class RebelCFREvaluator:
    """ReBeL CFR Evaluator implementing the precise SELFPLAY algorithm."""

    def __init__(
        self,
        search_batch_size: int,
        env_proto: HUNLTensorEnv,
        model: RebelFFN,
        bet_bins: list[float],
        max_depth: int,
        cfr_iterations: int,
        sample_count: int,
        device: torch.device,
        float_dtype: torch.dtype,
        belief_samples: int = 1,
    ):
        assert cfr_iterations > T_WARM

        self.search_batch_size = search_batch_size
        self.model = model
        self.max_depth = max_depth
        self.bet_bins = bet_bins
        self.cfr_iterations = cfr_iterations
        self.sample_count = sample_count
        self.device = device
        self.float_dtype = float_dtype
        self.belief_samples = belief_samples

        self.num_players = 2
        self.num_actions = len(bet_bins) + 3

        self.depth_offsets = (
            (
                self.search_batch_size
                * torch.pow(self.num_actions, torch.arange(max_depth + 1))
            )
            .cumsum(0)
            .tolist()
        )
        self.depth_offsets.insert(0, 0)
        self.total_nodes = self.depth_offsets[-1]
        self.all_depths = torch.zeros(
            self.total_nodes, dtype=torch.long, device=self.device
        )
        for i in range(self.max_depth):
            self.all_depths[self.depth_offsets[i] : self.depth_offsets[i + 1]] = i

        # Subgame environment
        self.env = HUNLTensorEnv(
            num_envs=self.total_nodes,
            starting_stack=env_proto.starting_stack,
            sb=env_proto.sb,
            bb=env_proto.bb,
            default_bet_bins=self.bet_bins,
            device=device,
            float_dtype=float_dtype,
            flop_showdown=env_proto.flop_showdown,
        )
        self.valid_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )
        # If finished, don't continue searching below this node.
        # Different from env.done, which is set when the node is terminal.
        self.leaf_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )

        self.policy_probs = torch.zeros(
            self.total_nodes,
            NUM_HANDS,
            self.num_actions,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.policy_probs_avg = torch.zeros(
            self.total_nodes,
            NUM_HANDS,
            self.num_actions,
            device=self.device,
            dtype=self.float_dtype,
        )
        # One value per node per player per hand
        self.values = torch.zeros(
            self.total_nodes,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        # One belief per node per player per hand
        # NB: beliefs[k, 0] is the belief ABOUT the acting player.
        # Not the belief OF the acting player.
        self.beliefs = torch.zeros(
            self.total_nodes,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )

        # Feature encoder for belief computation
        self.feature_encoder = RebelFeatureEncoder(
            env=self.env,
            device=device,
            dtype=float_dtype,
        )

    def initialize_search(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor,
    ) -> None:
        """Initialize search tree structure for given root states.

        Args:
            src_env: Source environment containing root states
            src_indices: Indices of root states in source environment
        """
        assert src_indices.shape[0] == self.search_batch_size
        assert src_indices.min().item() == 0
        assert src_indices.max().item() == self.total_nodes - 1
        assert initial_beliefs.shape[0] == src_indices.shape[0]

        self.env.copy_state_from(src_env, src_indices, src_indices, copy_deck=True)
        self.valid_mask.zero_()
        self.valid_mask[src_indices] = True
        self.leaf_mask.zero_()
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
        self.values.zero_()
        self.beliefs.zero_()
        self.beliefs[src_indices] = initial_beliefs

    def construct_subgame(self) -> None:
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

        for depth in range(self.max_depth - 1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            action_bins = torch.full((M,), -1, dtype=torch.long, device=self.device)
            legal_masks = self.env.legal_bins_mask()
            for action in range(self.num_actions):
                current_legal_mask = (
                    legal_masks[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.leaf_mask[offset:offset_next]
                )
                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                new_legal_indices = current_legal_indices + (action + 1) * B**depth * N
                assert new_legal_indices.max().item() < M

                self.env.copy_state_from(
                    self.env, new_legal_indices, current_legal_indices
                )
                self.valid_mask[new_legal_indices] = True

                action_bins[new_legal_indices] = action

            # TODO: To stop on street, capture new_streets here.
            self.env.step_bins(action_bins, legal_masks=legal_masks)
            self.leaf_mask[action_bins >= 0] |= self.env.done[action_bins >= 0]

        self.leaf_mask[offset:offset_next] |= self.valid_mask[offset:offset_next]

    def evaluate_model_on_valid(self) -> ModelOutput:
        features = self._encode_current_states()
        return self.model(features)

    def initialize_policy(self, model_output: ModelOutput) -> None:
        """Initialize policy for all nodes."""

        valid_indices = torch.where(self.valid_mask & ~self.leaf_mask)[0]
        valid_legal_masks = self.env.legal_bins_mask()[valid_indices]
        self.policy_probs[valid_indices] = F.softmax(
            compute_masked_logits(model_output.policy_logits, valid_legal_masks), dim=-1
        )

        # TODO: Warm-start CFR sim per appendix J with best-response.

    def initialize_beliefs(self, model_output: ModelOutput) -> None:
        """Initialize beliefs for all nodes."""
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

        legal_masks = self.env.legal_bins_mask()
        self.beliefs[self.search_batch_size :].zero_()

        for depth in range(self.max_depth - 1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            for action in range(B):
                current_legal_mask = (
                    legal_masks[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.leaf_mask[offset:offset_next]
                )
                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                actor = self.env.to_act[current_legal_indices]

                next_legal_indices = current_legal_indices + (action + 1) * B**depth * N
                assert next_legal_indices.max().item() < M
                assert self.valid_mask[next_legal_indices].all().item()

                probs = self.policy_probs[current_legal_indices]

                # Bayesian update assuming both players are playing the same policy.
                self.beliefs[next_legal_indices, actor] = (
                    self.beliefs[current_legal_indices, actor] * probs[:, None, :]
                )
                self.beliefs[next_legal_indices, 1 - actor] = self.beliefs[
                    current_legal_indices, 1 - actor
                ]

        # Normalize all beliefs.
        valid_beliefs = self.beliefs[self.valid_mask]
        self.beliefs = valid_beliefs / valid_beliefs.sum(dim=-1, keepdim=True)

    def set_leaf_values(self, model_output: ModelOutput) -> None:
        """Set leaf node values using model output."""
        leaf_indices = torch.where(self.leaf_mask)[0]
        self.values[leaf_indices] = model_output.hand_values[leaf_indices]

    def compute_expected_values(self) -> torch.Tensor:
        """Compute expected value of the current subgame using the given policy."""
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

        legal_masks = self.env.legal_bins_mask()
        new_values = torch.zeros_like(self.values)

        # Leaf values are already set, so start at max - 2
        for depth in range(self.max_depth - 2, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            # Compute EV by taking the weighted average of the next-depth values
            for action in range(B):
                current_legal_mask = (
                    legal_masks[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.leaf_mask[offset:offset_next]
                )
                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                next_legal_indices = current_legal_indices + (action + 1) * B**depth * N
                assert next_legal_indices.max().item() < M
                assert self.valid_mask[next_legal_indices].all().item()

                # [n, num_actions]
                probs = self.policy_probs[next_legal_indices]
                next_values = self.values[next_legal_indices]
                new_values[current_legal_indices] += (
                    probs[:, None, None, :] * next_values[:, :, None, None]
                ).sum(dim=-1)

        return new_values

    def update_policy(self) -> None:
        """Update the policy using CFR."""

        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

        blocked = combo_blocking_tensor(device=self.device)
        regret_indices = torch.where(self.valid_mask & ~self.leaf_mask)[0]

        legal_masks = self.env.legal_bins_mask()
        regrets = torch.zeros_like(self.policy_probs)

        for depth in range(self.max_depth - 1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            for action in range(B):
                current_legal_mask = (
                    legal_masks[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.leaf_mask[offset:offset_next]
                )
                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                actor = self.env.to_act[current_legal_indices]

                weights = self.beliefs[current_legal_indices, 1 - actor] @ ~blocked

                next_legal_indices = current_legal_indices + (action + 1) * B**depth * N
                assert next_legal_indices.max().item() < M
                assert self.valid_mask[next_legal_indices].all().item()

                regrets[current_legal_indices] += (
                    weights
                    * (
                        self.values[next_legal_indices]
                        - self.values[current_legal_indices]
                    )
                ).sum(dim=-1)

        regrets = regrets.clamp(min=0)
        regret_sum = regrets[regret_indices].sum(dim=-1, keepdim=True)
        self.policy_probs[regret_indices] = torch.where(
            regret_sum > 1e-8,
            regrets[regret_indices] / regret_sum,
            1.0 / self.num_actions,
        )

    def self_play_iteration(self) -> PublicBeliefState:
        N = self.search_batch_size, self.total_nodes

        self.construct_subgame()
        model_output = self.evaluate_model_on_valid()
        self.initialize_policy(model_output)
        self.initialize_beliefs(model_output)
        self.set_leaf_values(model_output)
        self.compute_expected_values()

        next_pbs = PublicBeliefState.from_proto(
            env=self.env,
            beliefs=torch.zeros(self.sample_count, self.num_players, NUM_HANDS),
        )
        value_training_data = []

        leaf_indices = self._leaf_node_indices()
        sample_envs = leaf_indices[
            torch.randint(0, leaf_indices.numel(), (self.sample_count,))
        ]
        t_sample = torch.randint(T_WARM, self.cfr_iterations, (self.sample_count,))
        for t in range(T_WARM, self.cfr_iterations):
            # If t == t_sample, sample leaf PBS
            sample_now = torch.where(t_sample == t)[0]
            if sample_now.numel() > 0:
                src_indices = sample_envs[sample_now]
                next_pbs.env.copy_state_from(self.env, src_indices, sample_now)
                next_pbs.beliefs[sample_now] = self.beliefs[src_indices]

            self.update_policy()

            # Update average policy.
            policy_probs = F.softmax(self.policy_probs[self.valid_mask], dim=-1)
            self.policy_probs_avg[self.valid_mask] = (
                t * self.policy_probs_avg[self.valid_mask] + policy_probs
            ) / (t + 1)

            model_output = self.evaluate_model_on_valid()
            self.set_leaf_values(model_output)

            new_expected_values = self.compute_expected_values()
            self.values = (t * self.values + new_expected_values) / (t + 1)

        return next_pbs

    def _encode_current_states(self) -> torch.Tensor:
        """Encode current environment states for policy network input."""
        # FIXME: Indexing order...
        # Get current acting players
        to_act = self.env.to_act[: self.search_batch_size]

        # Encode features for each player
        features = []
        for player in range(self.num_players):
            player_mask = to_act == player
            if player_mask.any():
                player_indices = torch.where(player_mask)[0]
                agents = torch.full(
                    (player_indices.numel(),),
                    player,
                    device=self.device,
                    dtype=torch.long,
                )

                # Get belief states for encoding
                beliefs = self._compute_current_beliefs()
                hero_beliefs = (
                    beliefs[player_indices, player] if beliefs.numel() > 0 else None
                )
                opp_beliefs = (
                    beliefs[player_indices, 1 - player] if beliefs.numel() > 0 else None
                )

                encoded = self.feature_encoder.encode(
                    player_indices, agents, hero_beliefs, opp_beliefs
                )
                features.append(encoded)

        assert features
        return torch.cat(features, dim=0)
