from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, Optional

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
class PublicBeliefState:
    """Public belief state for both players (? TODO)."""

    env: HUNLTensorEnv
    beliefs: torch.Tensor  # [batch_size, num_players, NUM_HANDS]

    @classmethod
    def from_proto(
        cls,
        env_proto: HUNLTensorEnv,
        beliefs: torch.Tensor,
        num_envs: Optional[int] = None,
    ) -> PublicBeliefState:
        return PublicBeliefState(
            env=HUNLTensorEnv.from_proto(env_proto, num_envs=num_envs),
            beliefs=beliefs,
        )

    def __post_init__(self):
        assert self.beliefs.shape[0] == self.env.N


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
        device: torch.device,
        float_dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
        warm_start_iterations: int = T_WARM,
    ):
        assert cfr_iterations > warm_start_iterations

        self.search_batch_size = search_batch_size
        self.model = model
        self.max_depth = max_depth
        self.bet_bins = bet_bins
        self.warm_start_iterations = warm_start_iterations
        self.cfr_iterations = cfr_iterations
        self.device = device
        self.float_dtype = float_dtype
        self.generator = generator

        self.num_players = 2
        self.num_actions = len(bet_bins) + 3
        self.all_hands = torch.arange(NUM_HANDS, device=self.device)

        # Compute depth offsets: slice i holds nodes at depth i
        self.depth_offsets: list[int] = [0]
        nodes_at_depth = self.search_batch_size
        for _ in range(self.max_depth):
            self.depth_offsets.append(self.depth_offsets[-1] + nodes_at_depth)
            nodes_at_depth *= self.num_actions
        self.total_nodes = self.depth_offsets[-1]
        self.all_depths = torch.zeros(
            self.total_nodes, dtype=torch.long, device=self.device
        )
        for i in range(self.max_depth):
            self.all_depths[self.depth_offsets[i] : self.depth_offsets[i + 1]] = i

        # Subgame environment
        self.env = HUNLTensorEnv.from_proto(
            env_proto,
            num_envs=self.total_nodes,
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
        # Compatibility matrix: compatibility[i, j] = 1 if combos i and j do not overlap
        blocking = combo_blocking_tensor(device=self.device)
        self.combo_compat = (~blocking).to(dtype=self.float_dtype)

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
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        """Initialize search tree structure for given root states.

        Args:
            src_env: Source environment containing root states
            src_indices: Indices of root states in source environment
        """
        assert src_indices.shape[0] == self.search_batch_size
        assert src_indices.min().item() >= 0
        if initial_beliefs is None:
            initial_beliefs = torch.full(
                (self.search_batch_size, self.num_players, NUM_HANDS),
                1.0 / NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
        else:
            assert initial_beliefs.shape[0] == src_indices.shape[0]
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        dest_indices = torch.arange(self.search_batch_size, device=self.device)
        self.env.copy_state_from(src_env, src_indices, dest_indices, copy_deck=True)
        self.valid_mask.zero_()
        self.valid_mask[dest_indices] = True
        self.leaf_mask.zero_()
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
        self.values.zero_()
        self.beliefs.zero_()
        self.beliefs[dest_indices] = initial_beliefs

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

        leaf_start = self.depth_offsets[self.max_depth - 1]
        leaf_end = self.depth_offsets[self.max_depth]
        self.leaf_mask[leaf_start:leaf_end] |= self.valid_mask[leaf_start:leaf_end]

    def evaluate_model_on_valid(self) -> ModelOutput:
        eval_indices = torch.where(self.valid_mask)[0]
        if eval_indices.numel() == 0:
            empty = torch.empty(
                0,
                self.feature_encoder.feature_dim,
                device=self.device,
                dtype=self.float_dtype,
            )
            return self.model(empty)

        features = self.encode_current_states(eval_indices)
        return self.model(features)

    def initialize_policy(self, model_output: ModelOutput) -> None:
        """Initialize policy for all nodes."""

        non_leaf_indices = torch.where(self.valid_mask & ~self.leaf_mask)[0]
        logits = model_output.policy_logits[non_leaf_indices]
        valid_legal_masks = self.env.legal_bins_mask()[non_leaf_indices]
        masked_logits = compute_masked_logits(logits, valid_legal_masks)
        self.policy_probs[non_leaf_indices] = F.softmax(masked_logits, dim=-1)

        # TODO: Warm-start CFR sim per appendix J with best-response.

    def initialize_beliefs(self, model_output: ModelOutput) -> None:
        """Initialize beliefs for all nodes."""

        legal_masks = self.env.legal_bins_mask()
        self.beliefs[self.search_batch_size :].zero_()

        for _, action, current_indices, next_indices in self._valid_actions(
            legal_masks
        ):
            actor = self.env.to_act[current_indices].to(torch.long)

            probs = self.policy_probs[current_indices, :, action]

            # Bayesian update assuming both players follow the same policy.
            updated_actor = self.beliefs[current_indices, actor] * probs
            self.beliefs[next_indices, actor] = updated_actor
            self.beliefs[next_indices, 1 - actor] = self.beliefs[
                current_indices, 1 - actor
            ]

        # Normalize beliefs in-place for valid nodes.
        valid_indices = torch.where(self.valid_mask)[0]
        if valid_indices.numel() > 0:
            beliefs = self.beliefs[valid_indices]
            denom = beliefs.sum(dim=-1, keepdim=True).clamp_min_(1e-12)
            self.beliefs[valid_indices] = beliefs / denom

    def set_leaf_values(self, model_output: ModelOutput) -> None:
        """Set leaf node values using model output."""
        if self._current_eval_indices is None:
            raise RuntimeError("Model must be evaluated before setting leaf values.")
        if model_output.hand_values is None:
            raise ValueError("Model must provide hand_values for ReBeL search.")

        eval_indices = self._current_eval_indices
        leaf_mask = self.leaf_mask[eval_indices]
        if not leaf_mask.any():
            return
        leaf_indices = eval_indices[leaf_mask]
        hand_values = model_output.hand_values[leaf_mask].to(self.float_dtype)
        self.values[leaf_indices] = hand_values

    def compute_expected_values(self) -> torch.Tensor:
        """Compute expected value of the current subgame using the given policy."""

        legal_masks = self.env.legal_bins_mask()
        new_values = self.values.clone()
        # First iteration: leaf values already populated; back propagate expectations
        for _, action, current_indices, next_indices in self._valid_actions(
            legal_masks
        ):
            probs = self.policy_probs[current_indices, :, action]
            child_values = new_values[next_indices]
            new_values[current_indices] += child_values * probs[:, None, :]

        return new_values

    def update_policy(self) -> None:
        """Update the policy using CFR."""

        regret_indices = torch.where(self.valid_mask & ~self.leaf_mask)[0]

        legal_masks = self.env.legal_bins_mask()
        regrets = torch.zeros_like(self.policy_probs)

        for _, action, current_indices, next_indices in self._valid_actions(
            legal_masks
        ):
            actor = self.env.to_act[current_indices]
            row_ids = torch.arange(
                current_indices.numel(), device=self.device, dtype=torch.long
            )

            opp_beliefs = self.beliefs[current_indices][row_ids, 1 - actor]
            weights = opp_beliefs @ self.combo_compat

            next_vals = self.values[next_indices, actor]
            current_vals = self.values[current_indices, actor]
            advantages = next_vals - current_vals
            regrets[current_indices, :, action] = weights * advantages

        # Zero out invalid actions explicitly
        regrets *= legal_masks.unsqueeze(1).to(self.float_dtype)

        positive_regrets = regrets.clamp(min=0)
        regret_sum = positive_regrets[regret_indices].sum(dim=-1, keepdim=True)
        legal = legal_masks[regret_indices]
        legal_float = legal.to(dtype=self.float_dtype)
        uniform_policy = legal_float / legal_float.sum(dim=-1, keepdim=True).clamp_min(
            1.0
        )
        uniform_policy = uniform_policy.unsqueeze(1).expand(-1, NUM_HANDS, -1)
        updated = torch.where(
            regret_sum > 1e-8,
            positive_regrets[regret_indices] / regret_sum.clamp_min(1e-8),
            uniform_policy,
        )
        # Ensure illegal actions remain zero
        updated *= legal.unsqueeze(1)
        self.policy_probs[regret_indices] = updated

    def self_play_iteration(self) -> PublicBeliefState:
        self.construct_subgame()
        model_output = self.evaluate_model_on_valid()
        self.initialize_policy(model_output)
        self.initialize_beliefs(model_output)
        self.set_leaf_values(model_output)
        self.values = self.compute_expected_values()

        leaf_indices = torch.where(self.valid_mask & self.leaf_mask & ~self.env.done)[0]
        sample_count = min(leaf_indices.numel(), self.search_batch_size)
        next_pbs = PublicBeliefState.from_proto(
            env_proto=self.env,
            beliefs=torch.zeros(sample_count, self.num_players, NUM_HANDS),
            num_envs=sample_count,
        )

        sample_envs = leaf_indices[
            torch.randperm(leaf_indices.numel(), generator=self.generator)[
                :sample_count
            ]
        ]
        t_sample = torch.randint(
            self.warm_start_iterations, self.cfr_iterations, (sample_count,)
        )

        for t in range(self.warm_start_iterations, self.cfr_iterations):
            # If t == t_sample, sample leaf PBS
            sample_now = torch.where(t_sample == t)[0]
            if sample_now.numel() > 0 and sample_envs.numel() > 0:
                src_indices = sample_envs[sample_now]
                next_pbs.env.copy_state_from(self.env, src_indices, sample_now)
                next_pbs.beliefs[sample_now] = self.beliefs[src_indices]

            self.update_policy()

            # Update average policy.
            self.policy_probs_avg[self.valid_mask] = (
                t * self.policy_probs_avg[self.valid_mask]
                + self.policy_probs[self.valid_mask]
            ) / (t + 1)

            model_output = self.evaluate_model_on_valid()
            self.set_leaf_values(model_output)

            new_expected_values = self.compute_expected_values()
            self.values = (t * self.values + new_expected_values) / (t + 1)

        return next_pbs

    def _valid_nodes(self) -> Generator[tuple[int, torch.Tensor]]:
        for depth in range(self.max_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            mask = self.valid_mask[offset:offset_next]
            yield depth, offset + torch.where(mask)[0]

    def _valid_actions(
        self, legal_masks: Optional[torch.Tensor] = None
    ) -> Generator[tuple[int, int, torch.Tensor, torch.Tensor]]:
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions
        if legal_masks is None:
            legal_masks = self.env.legal_bins_mask()

        for depth in range(self.max_depth - 1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            for action in range(B):
                current_legal_mask = (
                    legal_masks[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.leaf_mask[offset:offset_next]
                )
                if not current_legal_mask.any():
                    continue

                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                next_legal_indices = current_legal_indices + (action + 1) * B**depth * N
                assert next_legal_indices.max().item() < M
                assert self.valid_mask[next_legal_indices].all().item()

                yield depth, action, current_legal_indices, next_legal_indices

    def _split_beliefs(
        self, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (hero_beliefs, opp_beliefs) aligned with `indices`.

        We interpret self.beliefs[idx, player] as the belief over `player`'s private hands.
        """
        if indices.numel() == 0:
            empty = torch.empty(
                0, NUM_HANDS, device=self.device, dtype=self.float_dtype
            )
            return empty, empty

        beliefs = self.beliefs[indices]
        hero = torch.zeros(
            indices.numel(), NUM_HANDS, device=self.device, dtype=self.float_dtype
        )
        opp = torch.zeros_like(hero)

        to_act = self.env.to_act[indices]
        idxs_p0 = torch.where(to_act == 0)[0]
        idxs_p1 = torch.where(to_act == 1)[0]
        if idxs_p0.any():
            hero[idxs_p0] = beliefs[idxs_p0, 0]
            opp[idxs_p0] = beliefs[idxs_p0, 1]
        if idxs_p1.any():
            hero[idxs_p1] = beliefs[idxs_p1, 1]
            opp[idxs_p1] = beliefs[idxs_p1, 0]

        return hero, opp

    def encode_current_states(self, indices: torch.Tensor) -> torch.Tensor:
        """Encode environment states for policy network input."""
        if indices.numel() == 0:
            return torch.empty(
                0,
                self.feature_encoder.feature_dim,
                device=self.device,
                dtype=self.float_dtype,
            )

        hero_beliefs, opp_beliefs = self._split_beliefs(indices)
        return self.feature_encoder.encode(
            indices,
            self.env.to_act[indices],
            hero_beliefs=hero_beliefs,
            opp_beliefs=opp_beliefs,
        )

    def _leaf_node_indices(self) -> torch.Tensor:
        """Return flattened indices for valid nodes marked as leaves."""
        return torch.where(self.valid_mask & self.leaf_mask)[0]
