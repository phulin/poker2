from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, Optional

import torch

from alphaholdem.core.structured_config import CFRType
from alphaholdem.env.card_utils import (
    NUM_HANDS,
    calculate_unblocked_mass,
    combo_to_onehot_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.rl.rebel_batch import RebelBatch
from alphaholdem.search.cfr_evaluator import CFREvaluator, ExploitabilityStats
from alphaholdem.search.chance_node_helper import ChanceNodeHelper
from alphaholdem.utils.profiling import profile

T_WARM = 15


@dataclass
class PublicBeliefState:
    """Public belief state for both players.

    Attributes:
        env: Vectorised poker environment standing at a public state.
        beliefs: Beliefs representing the range at this node (post-chance for
            regular states, pre-chance for street-end nodes).
    """

    env: HUNLTensorEnv
    beliefs: torch.Tensor  # [batch_size, num_players, NUM_HANDS]

    @classmethod
    def from_proto(
        cls,
        env_proto: HUNLTensorEnv,
        beliefs: torch.Tensor,
        num_envs: Optional[int] = None,
    ) -> PublicBeliefState:
        """Create a new belief state with an environment cloned from `env_proto`.

        Args:
            env_proto: Template environment whose configuration should be reused.
            beliefs: Belief tensor shaped `[batch, players, NUM_HANDS]`.
            num_envs: Optional override for the number of vectorised environments.
        """
        return PublicBeliefState(
            env=HUNLTensorEnv.from_proto(env_proto, num_envs=num_envs),
            beliefs=beliefs,
        )

    def __post_init__(self) -> None:
        assert self.beliefs.shape[0] == self.env.N


@dataclass
class HandRankData:
    sorted_indices: torch.Tensor
    inv_sorted: torch.Tensor
    H: torch.Tensor
    card_ok: torch.Tensor
    hand_ok_mask: torch.Tensor
    hand_ok_mask_sorted: torch.Tensor
    hands_c1c2_sorted: torch.Tensor
    L_idx: torch.Tensor
    R_idx: torch.Tensor


class RebelCFREvaluator(CFREvaluator):
    """ReBeL CFR Evaluator implementing the precise SELFPLAY algorithm."""

    root_nodes: int
    model: RebelFFN | BetterFFN
    max_depth: int
    bet_bins: list[float]
    cfr_iterations: int
    warm_start_iterations: int
    sample_epsilon: float
    device: torch.device
    float_dtype: torch.dtype
    generator: Optional[torch.Generator]
    num_players: int
    num_actions: int
    all_hands: torch.Tensor
    depth_offsets: list[int]
    total_nodes: int
    env: HUNLTensorEnv
    valid_mask: torch.Tensor
    leaf_mask: torch.Tensor
    showdown_indices: torch.Tensor
    showdown_actors: torch.Tensor
    prev_actor: torch.Tensor
    folded_mask: torch.Tensor
    folded_rewards: torch.Tensor
    new_street_mask: torch.Tensor
    allowed_hands: torch.Tensor
    allowed_hands_prob: torch.Tensor
    policy_probs: torch.Tensor
    policy_probs_avg: torch.Tensor
    policy_probs_sample: torch.Tensor
    self_reach: torch.Tensor
    self_reach_avg: torch.Tensor
    cumulative_regrets: torch.Tensor
    regret_weight_sums: torch.Tensor
    last_model_values: torch.Tensor | None
    # NOTE: Latest values and values_avg are EVs, NOT CFVs.
    latest_values: torch.Tensor
    values_avg: torch.Tensor
    beliefs: torch.Tensor
    beliefs_avg: torch.Tensor
    legal_mask: Optional[torch.Tensor]
    feature_encoder: RebelFeatureEncoder | BetterFeatureEncoder
    hand_rank_data: Optional[HandRankData]
    stats: dict[str, float]

    def __init__(
        self,
        search_batch_size: int,
        env_proto: HUNLTensorEnv,
        model: RebelFFN | BetterFFN,
        bet_bins: list[float],
        max_depth: int,
        cfr_iterations: int,
        device: torch.device,
        float_dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
        warm_start_iterations: int = T_WARM,
        cfr_type: CFRType = CFRType.linear,
        cfr_avg: bool = True,
        dcfr_alpha: float = 1.5,
        dcfr_beta: float = 0.0,
        dcfr_gamma: float = 2.0,
        dcfr_delay: int = 0,
        sample_epsilon: float = 0.25,
    ):
        assert warm_start_iterations < cfr_iterations

        self.root_nodes = search_batch_size
        self.model = model
        self.max_depth = max_depth
        self.bet_bins = bet_bins
        self.cfr_iterations = cfr_iterations
        self.warm_start_iterations = max(0, warm_start_iterations)
        self.cfr_type = cfr_type
        self.cfr_avg = cfr_avg
        self.dcfr_alpha = dcfr_alpha
        self.dcfr_beta = dcfr_beta
        self.dcfr_gamma = dcfr_gamma
        self.dcfr_delay = dcfr_delay
        self.sample_epsilon = sample_epsilon
        self.device = device
        self.float_dtype = float_dtype
        self.generator = generator

        self.num_players = 2
        self.num_actions = len(bet_bins) + 3
        self.all_hands = torch.arange(NUM_HANDS, device=self.device)

        # Compute depth offsets: slice i holds nodes at depth i
        self.depth_offsets: list[int] = [0]
        nodes_at_depth = self.root_nodes
        for _ in range(self.max_depth + 1):
            self.depth_offsets.append(self.depth_offsets[-1] + nodes_at_depth)
            nodes_at_depth *= self.num_actions
        self.total_nodes = self.depth_offsets[-1]

        N, M = self.root_nodes, self.total_nodes

        # Subgame environment
        self.env = HUNLTensorEnv.from_proto(
            env_proto,
            num_envs=M,
        )
        self.valid_mask = torch.zeros(M, dtype=torch.bool, device=self.device)
        # If finished, don't continue searching below this node.
        # Different from env.done, which is set when the node is terminal.
        # Should be a subset of valid_mask.
        self.leaf_mask = torch.zeros(M, dtype=torch.bool, device=self.device)
        self.showdown_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.showdown_actors = torch.empty(0, dtype=torch.long, device=self.device)

        # Set in construct_subgame and not updated.
        self.folded_mask = torch.zeros(M, dtype=torch.bool, device=self.device)
        self.folded_rewards = torch.zeros(M, dtype=self.float_dtype, device=self.device)
        self.new_street_mask = torch.zeros(M, dtype=torch.bool, device=self.device)

        # Notionally, at its parent, what is the probability of acting to get to this node?
        self.policy_probs = torch.zeros(
            M,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
        self.policy_probs_sample = torch.zeros_like(self.policy_probs)
        # Cumulative regret of taking this node vs the best at the parent node.
        self.cumulative_regrets = torch.zeros_like(self.policy_probs)
        # Running per-infoset sums of positive regret mass over hands
        self.regret_weight_sums = torch.zeros_like(self.policy_probs)

        self.last_model_values = None

        # One value per node per player per hand
        self.latest_values = torch.zeros(
            M,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.values_avg = torch.zeros_like(self.latest_values)
        # One belief per node per player per hand
        # NB: beliefs[k, 0] is the belief ABOUT the acting player.
        # Not the belief OF the acting player.
        self.beliefs = torch.zeros(
            M,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.beliefs_avg = torch.zeros_like(self.beliefs)
        self.self_reach = torch.zeros_like(self.beliefs)
        self.self_reach_avg = torch.zeros_like(self.beliefs)
        self.root_pre_chance_beliefs = torch.zeros(
            self.root_nodes,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )

        self.legal_mask = None

        # Invalid for [0, N) because we don't know the previous actor.
        self.prev_actor = torch.zeros(M, dtype=torch.long, device=self.device)

        self.combo_onehot_float = combo_to_onehot_tensor(device=self.device).float()
        # if self.device.type == "cuda":
        #     self.combo_onehot_float = self.combo_onehot_float.to_sparse_csr()
        self.chance_helper = ChanceNodeHelper(
            device=self.device,
            float_dtype=self.float_dtype,
            num_players=self.num_players,
            model=self.model,
        )

        self.allowed_hands = torch.zeros(
            M, NUM_HANDS, device=self.device, dtype=torch.bool
        )
        self.allowed_hands_prob = torch.zeros(
            M, NUM_HANDS, device=self.device, dtype=self.float_dtype
        )

        # Feature encoder for belief computation
        if isinstance(self.model, BetterFFN):
            self.feature_encoder = BetterFeatureEncoder(
                env=self.env,
                device=self.device,
                dtype=self.float_dtype,
            )
        else:
            self.feature_encoder = RebelFeatureEncoder(
                env=self.env,
                device=self.device,
                dtype=self.float_dtype,
            )

        self.hand_rank_data = None
        self.stats = {}

        # PyTorch profiler setup
        self.profiler_enabled = False
        self.profiler = None

    def enable_profiler(self, output_dir: str = "profiler_logs") -> None:
        """Enable PyTorch profiler with stack traces."""
        self.profiler_enabled = True
        self.profiler_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create profiler with stack traces and TensorBoard support
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=10, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,  # Enable stack traces
            with_flops=True,
            with_modules=True,
        )
        self.profiler.start()

    def disable_profiler(self) -> None:
        """Disable PyTorch profiler."""
        self.profiler_enabled = False
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None

    def profiler_step(self) -> None:
        """Step the profiler if enabled."""
        if self.profiler_enabled and self.profiler is not None:
            self.profiler.step()

    @profile
    def initialize_search(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        """Copy root states into the search tree and reset per-node buffers.

        Args:
            src_env: Batched environment that holds the source root public states.
            src_indices: Row indices inside `src_env` to copy into the tree roots.
            initial_beliefs: Optional belief tensor aligned with `src_indices`.
        """
        assert src_indices.shape[0] == self.root_nodes
        assert src_indices.min().item() >= 0
        if initial_beliefs is None:
            initial_beliefs = torch.full(
                (self.root_nodes, self.num_players, NUM_HANDS),
                1.0 / NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
        else:
            assert initial_beliefs.shape[0] == src_indices.shape[0]
            assert initial_beliefs.shape[1] == self.num_players
            assert initial_beliefs.shape[2] == NUM_HANDS
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        N = self.root_nodes
        dest_indices = torch.arange(N, device=self.device)
        self.env.reset()
        self.env.copy_state_from(src_env, src_indices, dest_indices, copy_deck=True)
        self.valid_mask.zero_()
        self.valid_mask[:N] = True
        self.leaf_mask.zero_()
        self.leaf_mask[:N] = self.env.done[:N]
        self.showdown_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.showdown_actors = torch.empty(0, dtype=torch.long, device=self.device)
        self.folded_mask.zero_()
        self.folded_rewards.zero_()
        self.new_street_mask.zero_()
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
        self.policy_probs_sample.zero_()
        self.cumulative_regrets.zero_()
        self.regret_weight_sums.zero_()
        self.allowed_hands.zero_()
        self.allowed_hands_prob.zero_()
        self.prev_actor.zero_()
        self.last_model_values = None
        self.latest_values.zero_()
        self.values_avg.zero_()
        self.beliefs.zero_()
        self.beliefs[:N] = initial_beliefs
        self.root_pre_chance_beliefs[:] = initial_beliefs
        self.self_reach.zero_()
        self.self_reach[:N] = 1.0
        self.self_reach_avg.zero_()
        self.self_reach_avg[:N] = 1.0
        self.legal_mask = None
        self.hand_rank_data = None
        self.stats.clear()

    @profile
    def construct_subgame(self) -> None:
        """Expand the tree by cloning legal successor states at each depth."""
        N, M, B = self.root_nodes, self.total_nodes, self.num_actions

        for depth in range(self.max_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            action_bins = torch.full((M,), -1, dtype=torch.long, device=self.device)
            legal_masks_1 = self.env.legal_bins_mask()
            for action in range(self.num_actions):
                # don't currently have a way to get a subset of the masks
                current_legal_mask = (
                    legal_masks_1[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.env.done[offset:offset_next]
                )
                if depth > 0:
                    # On new street, stop at the first node.
                    no_actions = self.env.actions_this_round[offset:offset_next] == 0
                    current_legal_mask &= ~no_actions
                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                if current_legal_indices.numel() == 0:
                    continue
                new_legal_indices = (
                    offset_next + (current_legal_indices - offset) * B + action
                )
                assert new_legal_indices.max().item() < M

                self.env.copy_state_from(
                    self.env, current_legal_indices, new_legal_indices
                )
                self.valid_mask[new_legal_indices] = True

                action_bins[new_legal_indices] = action

            # now that we've copied the states, update the legal masks and amounts
            bin_amounts_2, legal_masks_2 = self.env.legal_bins_amounts_and_mask()
            rewards, _, _ = self.env.step_bins(
                action_bins, bin_amounts=bin_amounts_2, legal_masks=legal_masks_2
            )

            # New street -> mark as done. Don't want to deal w/ chance nodes.
            # actions_this_round is 0 after acting iff street advanced.
            valid = self.valid_mask[offset_next:offset_next_next]
            done = self.env.done[offset_next:offset_next_next]
            no_actions = self.env.actions_this_round[offset_next:offset_next_next] == 0
            self.new_street_mask[offset_next:offset_next_next] = no_actions
            self.leaf_mask[offset_next:offset_next_next] = valid & (done | no_actions)

            # Showdown values get set in set_leaf_values. Fold values get set here.
            self.folded_mask[offset_next:offset_next_next] = (
                self.valid_mask[offset_next:offset_next_next]
                & (action_bins[offset_next:offset_next_next] == 0)
                & self.env.done[offset_next:offset_next_next]
            )
            self.folded_rewards[offset_next:offset_next_next] = torch.where(
                self.folded_mask[offset_next:offset_next_next],
                rewards[offset_next:offset_next_next],
                0.0,
            )

        torch.where(
            self.folded_mask[:, None, None],
            torch.stack([self.folded_rewards, -self.folded_rewards], dim=1)[:, :, None],
            self.latest_values,
            out=self.latest_values,
        )
        self.values_avg[:] = self.latest_values

        leaf_start = self.depth_offsets[self.max_depth]
        leaf_end = self.depth_offsets[self.max_depth + 1]
        self.leaf_mask[leaf_start:leaf_end] = self.valid_mask[leaf_start:leaf_end]
        self.showdown_indices = torch.where(self.env.street == 4)[0]
        self.showdown_actors = self.env.to_act[self.showdown_indices]

        root_board_mask = self.env.board_onehot[:N].any(dim=1).reshape(N, -1).float()
        root_allowed = (self.combo_onehot_float @ root_board_mask.T).T < 0.5
        root_allowed_prob = root_allowed.float()
        root_allowed_prob /= root_allowed_prob.sum(dim=-1, keepdim=True).clamp(min=1)

        self.allowed_hands[:] = self._fan_out_deep(root_allowed)
        self.allowed_hands &= self.valid_mask[:, None]

        self.allowed_hands_prob[:] = self._fan_out_deep(root_allowed_prob)
        self.allowed_hands_prob.masked_fill_((~self.valid_mask)[:, None], 0.0)
        self.prev_actor[N:] = self._fan_out(self.env.to_act)

        self.legal_mask = self.env.legal_bins_mask()
        valid_legal_masks = self.legal_mask[self.valid_mask & ~self.leaf_mask]
        has_legal = valid_legal_masks.any(dim=-1)
        assert has_legal.all(), "Every valid node must have at least one legal action."

        self.stats["evaluator_street"] = (
            self.env.street[self.valid_mask].float().mean().item()
        )

    @profile
    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> None:
        """Calculate self reach weights for each node.

        Note we don't need to consider the chance nodes because we stop on street.

        Returns:
            reach_weights: [M, 2] tensor of reach weights for each node.
        """

        for depth in range(self.max_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            weights_src = target[offset:offset_next]
            target_dest = target[offset_next:offset_next_next]
            target_dest[:] = self._fan_out(weights_src, sliced=True)

            prev_actor_dest = self.prev_actor[offset_next:offset_next_next]
            prev_actor_indices = prev_actor_dest[:, None, None].expand(
                -1, -1, NUM_HANDS
            )
            policy_dest = policy[offset_next:offset_next_next]
            target_dest.scatter_reduce_(
                dim=1,
                index=prev_actor_indices,
                src=policy_dest[:, None],
                reduce="prod",
                include_self=True,
            )

        target.masked_fill_((~self.valid_mask)[:, None, None], 0.0)
        self._block_beliefs(target)
        return target

    @profile
    def _propagate_all_beliefs(
        self,
        target: torch.Tensor | None = None,
        reach_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propagate beliefs from all valid nodes to all valid nodes."""
        N = self.root_nodes

        if target is None:
            target = self.beliefs
        if reach_weights is None:
            reach_weights = self.self_reach

        target[:] = self._fan_out_deep(target[:N]) * reach_weights

        # Precondition: reach_weights should be board-blocked, so the multiplication
        # will block target as well. All that's left is normalizing.
        self._normalize_beliefs(target)

        # assert torch.allclose(target.sum(dim=2), self.valid_mask[:, None].float())

    def _propagate_level_beliefs(self, depth: int):
        """Propagate beliefs from all nodes at a given level to all nodes at the next level."""

        offset = self.depth_offsets[depth]
        offset_next = self.depth_offsets[depth + 1]
        offset_next_next = self.depth_offsets[depth + 2]

        probs = self.policy_probs[offset_next:offset_next_next]
        self.beliefs[offset_next:offset_next_next] = self._fan_out(
            self.beliefs[offset:offset_next], sliced=True
        )

        indices = torch.arange(offset_next, offset_next_next, device=self.device)
        self.beliefs[indices, self.prev_actor[offset_next:offset_next_next]] *= probs

    @profile
    def _block_beliefs(self, target: torch.Tensor | None = None) -> torch.Tensor:
        """Block beliefs based on the board."""
        if target is None:
            target = self.beliefs
        target.masked_fill_((~self.allowed_hands)[:, None, :], 0.0)

    @profile
    def _normalize_beliefs(self, target: torch.Tensor | None = None) -> torch.Tensor:
        """Normalize beliefs across hands in-place for valid nodes."""
        if target is None:
            target = self.beliefs

        denom = target.sum(dim=-1, keepdim=True)
        # If the action probability of getting to a node is 0, our
        # bayesian update will make the beliefs in that state all 0.
        # So we set them to uniform.
        torch.where(
            denom > 1e-10,
            target / denom.clamp(min=1e-12),
            self.allowed_hands_prob[:, None, :],
            out=target,
        )

        # That step will have made invalid nodes' beliefs uniform.
        # Which is not what we want.
        target.masked_fill_((~self.valid_mask)[:, None, None], 0.0)

    @torch.no_grad()
    @profile
    def initialize_policy_and_beliefs(self) -> None:
        """Push public beliefs down the tree using the freshly initialised policy."""
        self.policy_probs.zero_()

        self.model.eval()
        for depth in range(self.max_depth):
            # Beliefs and policy probs already filled in on current level.
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]
            indices = torch.arange(offset, offset_next, device=self.device)
            probs = (
                self._get_model_policy_probs(indices)
                .permute(0, 2, 1)
                .reshape(-1, NUM_HANDS)
            )
            # Fill in policy probs for the next level.
            self.policy_probs[offset_next:offset_next_next] = probs
            # Propagate beliefs from the current level to the next level.
            self._propagate_level_beliefs(depth)

            self._block_beliefs()
            self._normalize_beliefs()

        if self.max_depth == 0:
            # skipped the loop entirely
            self._block_beliefs()
            self._normalize_beliefs()

        self.policy_probs.masked_fill_((~self.valid_mask)[:, None], 0.0)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)

        self.policy_probs_avg[:] = self.policy_probs
        self.self_reach_avg[:] = self.self_reach
        self.beliefs_avg[:] = self.beliefs

    def warm_start(self) -> None:
        # Simple warm start: use model values and do a best-response pass
        self.set_leaf_values(0)
        self.compute_expected_values(self.policy_probs, self.latest_values)

        # [M, ]
        values_br_p0 = self._best_response_values(
            self.policy_probs, self.latest_values, torch.zeros_like(self.env.to_act)
        )
        values_br_p1 = self._best_response_values(
            self.policy_probs, self.latest_values, torch.ones_like(self.env.to_act)
        )
        values_br = torch.where(
            self.prev_actor[:, None, None] == 0, values_br_p0, values_br_p1
        )

        assert values_br.isfinite().all()

        # heuristic: scale regrets by the number of warm start iterations
        regrets = self.compute_instantaneous_regrets(
            values_achieved=values_br, values_expected=self.latest_values
        )
        self.cumulative_regrets += self.warm_start_iterations * regrets
        self.regret_weight_sums += self.warm_start_iterations
        self.update_policy(self.warm_start_iterations)

    @torch.no_grad()
    @profile
    def set_leaf_values(self, t: int) -> None:
        """Set leaf values from model or terminal states."""

        # Set model values for non-terminal leaves
        model_mask = self.leaf_mask & ~self.env.done
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        model_indices = torch.where(model_mask)[0]
        features = self.feature_encoder.encode(
            beliefs, pre_chance_node=self.new_street_mask
        )
        model_output = self.model(features[model_indices])

        if not self.cfr_avg or t <= 1 or self.last_model_values is None:
            self.latest_values[model_indices] = model_output.hand_values
        else:
            # Mix with previous values (CFR-AVG style)
            old, new = self._get_mixing_weights(t)
            self.latest_values[model_indices] = (
                (old + new) * model_output.hand_values - old * self.last_model_values
            ) / new
            # TODO: fix zero-sum drift
        self.last_model_values = model_output.hand_values

        # Set showdown values
        showdown_values_p0 = self._showdown_value(0, self.showdown_indices)
        showdown_values_p1 = self._showdown_value(1, self.showdown_indices)
        self.latest_values[self.showdown_indices, 0] = showdown_values_p0
        self.latest_values[self.showdown_indices, 1] = showdown_values_p1

    @profile
    def compute_expected_values(
        self,
        policy: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> None:
        """Back up leaf hand values to their ancestors under the provided policy.

        Args:
            policy: [M, B, 1326] tensor of policy probabilities.
            values: [M, 2, 1326] tensor of values, with leaf values already populated.
        """

        if policy is None:
            policy = self.policy_probs
        if values is None:
            values = self.latest_values
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        # First iteration: leaf values already populated; back propagate expectations
        for depth in range(self.max_depth - 1, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            # Pull back values to the source nodes (NB we are storing EVs, not CFVs).
            # First, we have to marginalize over all private hands.
            # Pull back policy: [K, B, 1326]
            policy_level = self._pull_back(
                policy[offset_next:offset_next_next], sliced=True
            )
            actor_indices = self.env.to_act[offset:offset_next]
            actor_indices_expanded = actor_indices[:, None, None].expand(
                -1, -1, NUM_HANDS
            )
            level_beliefs = beliefs[offset:offset_next]
            actor_beliefs = level_beliefs.gather(
                1, actor_indices_expanded
            )  # [K, 1, 1326]
            marginal_policy = policy_level * actor_beliefs  # [K, B, 1326]
            policy_blocked = calculate_unblocked_mass(marginal_policy)
            matchup_values = calculate_unblocked_mass(actor_beliefs)
            opponent_conditioned_policy = torch.where(
                matchup_values > 1e-12, policy_blocked / matchup_values, 0.0
            )

            indices = torch.arange(offset_next - offset, device=self.device)
            child_values_src = self._pull_back(
                values[offset_next:offset_next_next], sliced=True
            ).clone()  # [K, B, 2, 1326]
            child_values_src[indices, :, actor_indices] *= policy_level
            child_values_src[
                indices, :, 1 - actor_indices
            ] *= opponent_conditioned_policy

            torch.where(
                self.leaf_mask[offset:offset_next, None, None],
                values[offset:offset_next],
                child_values_src.sum(dim=1),
                out=values[offset:offset_next],
            )
            values[offset:offset_next].masked_fill_(
                (~self.valid_mask)[offset:offset_next, None, None], 0.0
            )

    @profile
    def compute_instantaneous_regrets(
        self, values_achieved: torch.Tensor, values_expected: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute regrets for every valid non-leaf information set.

        Args:
            values: [M, 2, 1326] tensor of values for each node.
            values_expected: [M, 2, 1326] tensor of expected values for each node, or none to use values.

        Returns:
            regrets: [M, 1326] tensor of regrets for taking the action to get to the node.
        """

        if values_expected is None:
            values_expected = values_achieved

        bottom = self.depth_offsets[1]
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        regrets = torch.zeros_like(self.policy_probs)

        src_actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)
        prev_actor_indices = self.prev_actor[bottom:, None, None].expand(
            -1, -1, NUM_HANDS
        )

        # This represents the opponent's reach prob at the src node.
        # Then actor acts at the transition src -> dest node.
        # Unblocked mass translates opponent-hand space to hero-hand space.
        opponent_global_reach = calculate_unblocked_mass(beliefs.flip(dims=[1]))
        src_weights = opponent_global_reach.gather(1, src_actor_indices).squeeze(1)

        # Weight advantages by our mass unblocked by the opponent hands.
        weights = self._fan_out(src_weights)

        # The value at a node is already the EV over all actions.
        actor_values = values_expected.gather(1, src_actor_indices).squeeze(1)  # bottom
        actor_values_expected = self._fan_out(actor_values)
        actor_values_achieved = (
            values_achieved[bottom:].gather(1, prev_actor_indices).squeeze(1)
        )

        advantages = actor_values_achieved - actor_values_expected

        regrets[bottom:] = weights * advantages

        regrets.masked_fill_(~self.valid_mask[:, None], 0.0)
        return regrets

    @profile
    def update_policy(self, t: int) -> None:
        """Apply a regret-matching update to every valid non-leaf information set."""

        bottom = self.depth_offsets[1]

        positive_regrets = self.cumulative_regrets.clamp(min=0)
        positive_src = self._pull_back(positive_regrets)
        regret_sum = positive_src.sum(dim=1)
        regret_sum_dest = self._fan_out(regret_sum)

        updated = torch.where(
            regret_sum_dest > 1e-8,
            positive_regrets[bottom:],
            1.0,
        )
        updated.masked_fill_(~self.valid_mask[bottom:, None], 0.0)

        updated_src = self._pull_back(updated, sliced=True)
        updated_src /= updated_src.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.policy_probs[bottom:] = self._push_down(updated_src)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)
        self._propagate_all_beliefs(self.beliefs, self.self_reach)

        self.update_average_policy(t)
        self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    @profile
    def update_average_policy(self, t: int) -> None:
        """Update the average policy by mixing it with the current policy."""

        if (
            self.cfr_type in [CFRType.discounted, CFRType.discounted_plus]
            and t <= self.dcfr_delay
        ):
            self.policy_probs_avg[:] = self.policy_probs
            return
        elif t == 0:
            self.policy_probs_avg[:] = self.policy_probs
            return

        N = self.root_nodes
        top = self.depth_offsets[-2]

        policy_probs_src = self._pull_back(self.policy_probs)
        policy_probs_avg_src = self._pull_back(self.policy_probs_avg)

        actor_indices = self.env.to_act[:top, None, None].expand(-1, -1, NUM_HANDS)
        reach_actor = self.self_reach[:top].gather(1, actor_indices)
        reach_avg_actor = self.self_reach_avg[:top].gather(1, actor_indices)

        old, new = self._get_mixing_weights(t)
        reach_avg_actor *= old
        reach_actor *= new
        unweighted = (old * policy_probs_avg_src + new * policy_probs_src) / (old + new)
        policy_probs_avg_src *= reach_avg_actor
        policy_probs_avg_src += policy_probs_src * reach_actor
        denom = reach_avg_actor + reach_actor
        policy_probs_avg_src /= denom.clamp(min=1e-12)
        torch.where(
            denom < 1e-12, unweighted, policy_probs_avg_src, out=policy_probs_avg_src
        )
        self.policy_probs_avg[N:] = self._push_down(policy_probs_avg_src)

    def sample_leaves(self, training_mode: bool) -> None:
        """Sample leaves from `self.policy_probs_sample`."""

        N, B = self.root_nodes, self.num_actions
        top = self.depth_offsets[-2]

        players = torch.randint(
            0, 2, (N,), generator=self.generator, device=self.device
        )
        sample_epsilon = self.sample_epsilon if training_mode else 0.0

        # Don't sample nodes that are done. No point in continuing search from there.
        done_src = self._pull_back(self.env.done)

        # Calculate uniform sampling probabilities.
        legal_masks = self.legal_mask.clone()
        legal_masks[:top] &= ~done_src
        legal_counts = legal_masks.float().sum(dim=-1, keepdim=True)
        uniform = torch.where(legal_masks, 1 / legal_counts, 0)

        # Calculate policy sampling probabilities, excluding done nodes.
        policy_probs_by_src = self._pull_back(self.policy_probs_sample).clone()
        policy_probs_by_src.masked_fill_(done_src[:, :, None], 0.0)
        denom = policy_probs_by_src.sum(dim=1, keepdim=True)
        policy_probs_by_src = torch.where(
            denom >= 1e-12, policy_probs_by_src / denom, uniform[:top, :, None]
        )

        # If a node has no legal actions after filtering done nodes, it's a leaf.
        effective_leaf_mask = self.leaf_mask | (legal_counts.squeeze(1) == 0)

        # select a hand for every root node.
        actor_beliefs = (
            self.beliefs[:N]
            .gather(1, self.env.to_act[:N, None, None].expand(-1, -1, NUM_HANDS))
            .squeeze(1)
        )
        selected_hands = torch.multinomial(
            actor_beliefs, 1, generator=self.generator
        ).squeeze(1)

        # start with root nodes and descend to leaves
        sampled_nodes = torch.arange(N, device=self.device)

        depth = 0
        active_mask = ~effective_leaf_mask[sampled_nodes]
        while active_mask.any():
            assert depth <= self.max_depth
            active_nodes = sampled_nodes[active_mask]
            active_count = active_nodes.numel()
            assert self.valid_mask[active_nodes].all()

            to_act = self.env.to_act[active_nodes]
            player_mask = players[active_mask]
            sample_uniformly = (
                torch.rand(active_count, generator=self.generator, device=self.device)
                < sample_epsilon
            )
            sample_uniformly &= to_act == player_mask

            policy_probs_active = policy_probs_by_src[
                active_nodes, :, selected_hands[active_mask]
            ]
            actions = torch.multinomial(
                torch.where(
                    sample_uniformly[:, None],
                    uniform[active_nodes],
                    policy_probs_active,
                ),
                num_samples=1,
                generator=self.generator,
            ).squeeze(1)

            offset_next = self.depth_offsets[depth + 1]
            offset = self.depth_offsets[depth]
            sampled_nodes[active_mask] = (
                offset_next + (active_nodes - offset) * B + actions
            )
            # remove node from the active mask once it's a leaf
            active_mask = ~effective_leaf_mask[sampled_nodes]
            depth += 1

        assert (~self.env.done[sampled_nodes]).all()

        # Don't sample root nodes.
        sampled_continue = sampled_nodes[sampled_nodes >= N]
        pbs = PublicBeliefState.from_proto(
            env_proto=self.env,
            beliefs=self.beliefs[sampled_continue],
            num_envs=sampled_continue.numel(),
        )
        pbs.env.copy_state_from(
            self.env,
            sampled_continue,
            torch.arange(sampled_continue.numel(), device=self.device),
        )
        return pbs

    @profile
    def evaluate_cfr(self, training_mode: bool = True) -> Optional[PublicBeliefState]:
        """Run one instance of the CFR loop and produce leaf samples for replay."""

        self.construct_subgame()
        self.initialize_policy_and_beliefs()

        if self.warm_start_iterations > 0:
            self.warm_start()

        # use t=0 here so set_leaf_values doesn't do the CFR-AVG de-averaging.
        self.set_leaf_values(0)
        self.compute_expected_values()
        self.values_avg[:] = self.latest_values

        self.t_sample = self._get_sampling_schedule()
        for t in range(self.warm_start_iterations, self.cfr_iterations):
            self.profiler_step()  # Profile start of CFR iteration
            self.cfr_iteration(t)

        self._record_action_mix()
        self._record_cfr_entropy()
        self._record_cumulative_regret()

        return self.sample_leaves(training_mode)

    def cfr_iteration(self, t: int) -> None:
        """Run one CFR iteration."""

        torch.where(
            (self.t_sample == t)[:, None],
            self.policy_probs,
            self.policy_probs_sample,
            out=self.policy_probs_sample,
        )

        regrets = self.compute_instantaneous_regrets(self.latest_values)

        if self.cfr_type == CFRType.linear:  # Alternate updates.
            regrets.masked_fill_(self.prev_actor[:, None] == t % self.num_players, 0.0)
        elif (
            self.cfr_type == CFRType.discounted
            or self.cfr_type == CFRType.discounted_plus
        ):
            numerator = torch.where(
                self.cumulative_regrets > 0, t**self.dcfr_alpha, t**self.dcfr_beta
            )
            denominator = torch.where(
                self.cumulative_regrets > 0,
                (t + 1) ** self.dcfr_alpha,
                (t + 1) ** self.dcfr_beta,
            )
            self.cumulative_regrets *= numerator
            self.cumulative_regrets /= denominator
            self.regret_weight_sums *= numerator
            self.regret_weight_sums /= denominator
        self.regret_weight_sums += 1
        self.cumulative_regrets += regrets
        # This is the CFR+ trick.
        self.cumulative_regrets.clamp_(min=0)

        old_policy_probs = self.policy_probs.clone()
        self.update_policy(t)
        self._record_stats(t, old_policy_probs)

        self.set_leaf_values(t)
        self.compute_expected_values()

        old, new = self._get_mixing_weights(t)
        self.values_avg *= old
        self.values_avg += new * self.latest_values
        self.values_avg /= old + new

    @profile
    def training_data(
        self, exclude_start: bool = True
    ) -> tuple[RebelBatch, RebelBatch, RebelBatch]:
        """Aggregate model targets for supervised learning.

        Returns:
            Tuple of (start_of_street_value_batch, end_of_previous_street_value_batch,
            policy_batch).
        """
        N = self.root_nodes
        top = self.depth_offsets[-2]
        policy_targets = self._pull_back(self.policy_probs_avg)
        policy_targets = policy_targets.permute(0, 2, 1)

        # Nominally we'd need to divide by reach weights here, but since we're only
        # taking the first level of the tree, those weights would all be 1.
        # Valid values will always be between -1.0 and 1.0, so we can clamp targets to that range.
        value_targets = self.values_avg[:N].clamp(-1.0, 1.0)
        high = value_targets.abs().max(dim=-1).values > 10
        if high.any():
            print(f"WARNING: Value targets are too large ({high.sum().item()} hands)")

        features = self.feature_encoder.encode(self.beliefs_avg, pre_chance_node=False)[
            :top
        ]
        bin_amounts, legal_masks = self.env.legal_bins_amounts_and_mask()
        statistics = {
            "to_act": self.env.to_act,
            "street": self.env.street,
            "stage": 2 * self.env.street,
            "board": self.env.board_indices,
            "pot": self.env.pot,
            "bet_amounts": bin_amounts,
        }

        exploit_stats = self._compute_exploitability()

        # Value batch gets root states only. These should all be valid.
        value_statistics = {key: statistics[key][:N].clone() for key in statistics}
        value_statistics["local_exploitability"] = exploit_stats.local_exploitability
        value_statistics["local_br_values"] = exploit_stats.local_best_response_values

        # Policy batch gets all valid, non-leaf states.
        valid_top = self.valid_mask[:top] & ~self.leaf_mask[:top]
        policy_statistics = {
            key: statistics[key][:top][valid_top] for key in statistics
        }
        value_batch = RebelBatch(
            features=features[:N],
            value_targets=value_targets,
            legal_masks=legal_masks[:N],
            statistics=value_statistics,
        )

        street_root = self.env.street[:N]
        actions_root = self.env.actions_this_round[:N]
        root_nodes = (street_root == 0) & (actions_root == 0)
        if exclude_start:
            value_batch = value_batch[~root_nodes]
        policy_batch = RebelBatch(
            features=features[valid_top],
            policy_targets=policy_targets[valid_top],
            legal_masks=legal_masks[:top][valid_top],
            statistics=policy_statistics,
        )

        # Prepare end-of-previous-street value batch using pre-chance beliefs.
        pre_features_all = self.feature_encoder.encode(
            self.beliefs, pre_chance_node=True
        )
        # Clone root slice and override belief encoding with pre-chance beliefs.
        pre_features_root = pre_features_all[:N].clone()
        pre_beliefs = self.root_pre_chance_beliefs[:N].reshape(N, -1)
        pre_features_root.beliefs = pre_beliefs

        value_targets_pre = value_targets.clone()
        value_statistics_pre = {
            key: value_statistics[key].clone() for key in value_statistics
        }
        value_statistics_pre["board"] = self.env.last_board_indices[:N].clone()
        prev_street = torch.where(
            (street_root > 0) & (street_root < 4) & (actions_root == 0),
            street_root - 1,
            street_root,
        )
        value_statistics_pre["street"] = prev_street
        value_statistics_pre["stage"] = 2 * prev_street + 1

        start_mask = actions_root == 0

        turn_river_mask = start_mask & ((street_root == 2) | (street_root == 3))
        if turn_river_mask.any():
            expected_turn_river = self.chance_helper.single_card_chance_values(
                torch.where(turn_river_mask)[0],
                features[:N],
                self.root_pre_chance_beliefs,
                self.env.last_board_indices,
            )
            value_targets_pre[turn_river_mask] = expected_turn_river

        flop_mask = start_mask & (street_root == 1)
        if flop_mask.any():
            # The helper assumes suit symmetry, which is enforced by the encoder’s
            # suit-symmetry loss (alphaholdem/rl/losses.py).
            expected_flop = self.chance_helper.flop_chance_values(
                torch.where(flop_mask)[0],
                features[:N],
                self.root_pre_chance_beliefs,
            )
            value_targets_pre[flop_mask] = expected_flop

        transition_mask = turn_river_mask | flop_mask
        pre_value_batch = RebelBatch(
            features=pre_features_root,
            value_targets=value_targets_pre,
            legal_masks=legal_masks[:N],
            statistics=value_statistics_pre,
        )[transition_mask]

        return value_batch, pre_value_batch, policy_batch

    def _record_stats(self, t: int, old_policy_probs: torch.Tensor) -> None:
        """Record statistics about the policy update."""

        if (
            t
            in torch.linspace(self.warm_start_iterations, self.cfr_iterations - 1, 5)
            .round()
            .int()
            .tolist()
        ):
            diff = self._pull_back(self.policy_probs) - self._pull_back(
                old_policy_probs
            )
            # Can either player get to this node with a given hand?
            reachable = (self.self_reach > 0).any(dim=1)[: self.depth_offsets[-2]]
            diff.masked_fill_(~reachable[:, None, :], 0.0)
            reachable_hand_count = reachable.sum(dim=-1)
            reachable_nodes = reachable_hand_count > 0

            # Sum over action probabilities and hands - will divide by reachable hand count.
            diff_sum_nodes = diff.abs().sum(dim=1).sum(dim=1)
            node_delta = torch.where(
                reachable_nodes, diff_sum_nodes / reachable_hand_count, 0.0
            )
            node_delta_mean = node_delta.sum() / reachable_nodes.sum()
            self.stats[f"cfr_delta.{t + 1}"] = node_delta_mean.item()

    def _record_cfr_entropy(self) -> None:
        """Record the entropy of the policy."""
        if self.max_depth == 0:
            return
        N = self.root_nodes
        actions = self._pull_back(self.policy_probs_avg)[:N]
        mask = self.valid_mask[:N] & ~self.leaf_mask[:N]
        probs = actions[mask]
        entropy = torch.where(probs > 1e-12, -(probs * probs.log()), 0.0)
        self.stats["cfr_entropy"] = entropy.sum(dim=1).mean().item()

    def _record_cumulative_regret(self) -> None:
        self.stats["mean_positive_regret"] = (
            self.cumulative_regrets.clamp(min=0).mean().item()
        )

        N = self.root_nodes

        # Compute something like the theoretical exploitability bound.
        regret_quotient = self.cumulative_regrets.clamp(min=0) / self.regret_weight_sums
        regret_quotient_src = self._pull_back(regret_quotient)[:N]
        # take max over actions.
        regret_quotient_max = regret_quotient_src.max(dim=1).values
        # sum over hands.
        regret_quotient_sum = regret_quotient_max.sum(dim=1)
        # take mean over parallelized envs.
        regret_quotient_mean = regret_quotient_sum.sum() / self.valid_mask[:N].sum()
        self.stats["mean_regret_bound"] = regret_quotient_mean.item()

    def _best_response_values(
        self,
        policy: torch.Tensor,
        base_values: torch.Tensor,
        deviating_player: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, B = self.root_nodes, self.num_actions
        top = self.depth_offsets[-2]
        if deviating_player is None:
            deviating_player = self._fan_out_deep(self.env.to_act[:N])

        values_br = torch.where(self.leaf_mask[:, None, None], base_values, 0.0)

        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
        min_value = torch.finfo(base_values.dtype).min

        policy_src_all = self._pull_back(policy)

        actor_indices = self.env.to_act[:top, None, None].expand(-1, -1, NUM_HANDS)
        actor_beliefs = beliefs.gather(1, actor_indices).squeeze(1)

        marginal_policy = policy_src_all * actor_beliefs[:, None, :]
        policy_blocked = calculate_unblocked_mass(marginal_policy)
        matchup_mass = calculate_unblocked_mass(actor_beliefs)
        opponent_conditioned_policy = torch.where(
            matchup_mass[:, None, :] > 1e-12,
            policy_blocked / matchup_mass[:, None, :],
            0.0,
        )

        for depth in range(self.max_depth - 1, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            indices = torch.arange(offset_next - offset, device=self.device)
            actor = self.env.to_act[offset:offset_next]
            deviator = deviating_player[offset:offset_next]
            illegal = ~self.legal_mask[offset:offset_next]

            values_src = self._pull_back(
                values_br[offset_next:offset_next_next], sliced=True
            )  # [K, B, 2, 1326]
            policy_src = policy_src_all[offset:offset_next]
            opponent_policy = opponent_conditioned_policy[offset:offset_next]

            actor_indices = actor[:, None, None, None].expand(-1, B, 1, NUM_HANDS)
            opp_indices = (1 - actor)[:, None, None, None].expand(-1, B, 1, NUM_HANDS)
            # Both [K, B, 1326]
            actor_values_src = values_src.gather(2, actor_indices).squeeze(2)
            opp_values_src = values_src.gather(2, opp_indices).squeeze(2)

            actor_values_for_best = actor_values_src.masked_fill(
                illegal[:, :, None], min_value
            )
            best_action = actor_values_for_best.argmax(dim=1)  # [K, 1326]
            # [K, 1326]
            best_actor_values = actor_values_src.gather(
                1, best_action[:, None, :]
            ).squeeze(1)

            # Public belief over deviator hands at s (not action-dependent)
            deviator_beliefs = actor_beliefs[offset:offset_next]

            # 1) Histogram the deviator belief by the BR-chosen action a*(h_i)
            #    mass_by_action[a, h_i] = b_i(h_i|s) if a*(h_i)==a else 0
            mass_by_action = torch.zeros(
                deviator_beliefs.size(0),
                B,
                deviator_beliefs.size(1),
                dtype=deviator_beliefs.dtype,
                device=self.device,
            )  # [n_dev, A, H_dev]
            # Partition belief by best action.
            mass_by_action.scatter_add_(
                1, best_action[:, None, :], deviator_beliefs[:, None, :]
            )

            # 2) Blocker-project that mass to opponent hands and normalize per h_-i
            mass_blocked = calculate_unblocked_mass(mass_by_action)  # [M, B, 1326]
            dev_match = calculate_unblocked_mass(deviator_beliefs)[
                :, None, :
            ]  # [M, 1, 1326]
            P_dev = torch.where(
                dev_match > 1e-12,
                mass_blocked / dev_match,  # P_dev(a | s, h_-i)
                0.0,
            )  # [M, B, 1326]

            # 3) Expectation of opponent continuation values under P_dev
            v_opp_exp = (P_dev * opp_values_src).sum(dim=1)  # [M, 1326]

            # Actor: deviating player gets best value, otherwise average value.
            actor_values = torch.where(
                (deviator == actor)[:, None],
                best_actor_values,  # case 1
                (actor_values_src * policy_src).sum(dim=1),  # case 3
            )
            # Non-actor: deviating player gets average value.
            # Non-deviating player gets value assuming deviating player plays best action.
            opp_values = torch.where(
                (deviator == actor)[:, None],
                v_opp_exp,  # case 2
                (opp_values_src * opponent_policy).sum(dim=1),  # case 4
            )

            values_br[indices + offset, actor] = actor_values
            values_br[indices + offset, 1 - actor] = opp_values

            # Re-add leaf values (which were just overwritten).
            torch.where(
                self.leaf_mask[offset:offset_next, None, None],
                base_values[offset:offset_next],
                values_br[offset:offset_next],
                out=values_br[offset:offset_next],
            )

        return values_br

    def _compute_exploitability(self) -> ExploitabilityStats:
        N = self.root_nodes
        if N == 0:
            empty = torch.empty(0, device=self.device, dtype=self.float_dtype)
            empty2 = torch.empty(0, 2, device=self.device, dtype=self.float_dtype)
            return ExploitabilityStats(
                local_exploitability=empty,
                local_best_response_values=empty2,
            )

        policy = self.policy_probs_avg if self.cfr_avg else self.policy_probs
        leaf_values = self.values_avg if self.cfr_avg else self.latest_values

        base_values = torch.where(self.leaf_mask[:, None, None], leaf_values, 0.0)
        self.compute_expected_values(policy=policy, values=base_values)
        br_values = self._best_response_values(policy, leaf_values)

        root_indices = torch.arange(N, device=self.device)
        root_actor = self.env.to_act[:N]

        base_root = base_values[root_indices, root_actor]  # (N, NUM_HANDS)
        br_root = br_values[root_indices, root_actor]  # (N, NUM_HANDS)

        # Aggregate over hands using beliefs
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
        root_beliefs_actor = beliefs[root_indices, root_actor]  # (N, NUM_HANDS)

        # Weight improvements by beliefs
        improvements_per_hand = br_root - base_root  # (N, NUM_HANDS)
        improvements = (improvements_per_hand * root_beliefs_actor).sum(dim=-1)  # (N,)

        # Aggregate best response values for both players
        # For the acting player: use best response value
        # For the opponent: use base value
        root_opponent = 1 - root_actor
        br_values_actor = (br_root * root_beliefs_actor).sum(dim=-1)  # (N,)
        base_root_opponent = base_values[root_indices, root_opponent]  # (N, NUM_HANDS)
        root_beliefs_opponent = beliefs[root_indices, root_opponent]  # (N, NUM_HANDS)
        base_values_opponent = (base_root_opponent * root_beliefs_opponent).sum(
            dim=-1
        )  # (N,)

        br_values_agg = torch.stack(
            [br_values_actor, base_values_opponent], dim=-1
        )  # (N, 2)

        return ExploitabilityStats(
            local_exploitability=improvements, local_best_response_values=br_values_agg
        )

    def _record_action_mix(self) -> None:
        """Record the action mix of the policy."""
        actions = self._pull_back(self.policy_probs_avg)
        mask = self.valid_mask & ~self.leaf_mask
        mask = mask[: actions.shape[0]]
        allowed_hands = self.allowed_hands[: actions.shape[0]][mask]
        # self.policy_probs_avg is already masked by allowed hands.
        action_mix_by_node = actions[mask].sum(dim=2) / allowed_hands.sum(
            dim=1, keepdim=True
        )
        self.stats["action_mix"] = {
            "fold": action_mix_by_node[:, 0].mean().item(),
            "call": action_mix_by_node[:, 1].mean().item(),
            "bet": action_mix_by_node[:, 2:-1].mean(dim=1).mean().item(),
            "allin": action_mix_by_node[:, -1].mean().item(),
        }

    def _valid_nodes(
        self, bottom_up: bool = False
    ) -> Generator[tuple[int, torch.Tensor]]:
        """Yield `(depth, indices)` pairs for nodes marked as valid.

        torch.where is slow, so don't use this function in the inner CFR loop.
        """
        for depth in (
            range(self.max_depth + 1)
            if not bottom_up
            else range(self.max_depth, -1, -1)
        ):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            mask = self.valid_mask[offset:offset_next]
            yield depth, offset + torch.where(mask)[0]

    def _pull_back(self, data: torch.Tensor, sliced=False) -> torch.Tensor:
        """Pull back data to all parent nodes.
        Args:
            data: Data to pull back, organized by destination node, shape [M, ...].
            sliced: Whether this is already sliced to the bottom of the tree ([M - N, ...]).
        Returns:
            Data by source node/action, shape [M - N * B ** D, B, *data.shape[1:]].
        """
        bottom = self.depth_offsets[1] if not sliced else 0
        return data[bottom:].view(-1, self.num_actions, *data.shape[1:])

    def _push_down(self, data: torch.Tensor) -> torch.Tensor:
        """Push down data to all child nodes.
        Args:
            data: Data to push down, shape [M, B, ...].
        Returns:
            Data by child node, shape [M - N, ...].
        """
        assert data.shape[1] == self.num_actions
        top = self.depth_offsets[-2]
        return data[:top, None].reshape(-1, *data.shape[2:])

    def _fan_out(self, data: torch.Tensor, sliced=False) -> torch.Tensor:
        """Fanout data to all children nodes.

        Args:
            data: Data to fanout.
            sliced: Whether this is already sliced to the top of the tree ([M - N * B ** D, ...]).
        Returns:
            Fanout data, shape [M - N, *data.shape[1:]].
        """
        top = self.depth_offsets[-2]
        data_sliced = data[:top, None] if not sliced else data[:, None]
        return (
            data_sliced.expand(-1, self.num_actions, *data.shape[1:])
            .clone()
            .reshape(-1, *data.shape[1:])
        )

    def _fan_out_deep(self, data: torch.Tensor) -> torch.Tensor:
        """Broadcast root-aligned tensors across every node in the tree.

        Args:
            data: Tensor shaped [N, ...] aligned with the root batch.
        Returns:
            Tensor shaped [M, ...] with each root slice repeated for every node
            that descends from that root.
        """
        output = torch.zeros(
            self.total_nodes, *data.shape[1:], device=self.device, dtype=data.dtype
        )
        output[: self.root_nodes] = data
        for depth in range(self.max_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]
            output[offset_next:offset_next_next] = self._fan_out(
                output[offset:offset_next], sliced=True
            )
        return output
