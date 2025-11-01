from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, Optional

import torch
import torch.nn.functional as F

from alphaholdem.core.structured_config import CFRType
from alphaholdem.env.card_utils import (
    NUM_HANDS,
    combo_to_onehot_tensor,
    hand_combos_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rules import rank_hands
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.rl.rebel_replay import RebelBatch
from alphaholdem.search.chance_node_helper import ChanceNodeHelper
from alphaholdem.utils.model_utils import compute_masked_logits
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


@dataclass
class ExploitabilityStats:
    local_exploitability: torch.Tensor
    local_br_policy: torch.Tensor
    local_br_values: torch.Tensor
    local_br_improvement: torch.Tensor


class RebelCFREvaluator:
    """ReBeL CFR Evaluator implementing the precise SELFPLAY algorithm."""

    search_batch_size: int
    model: BetterFFN
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
    prev_actor: torch.Tensor
    allowed_hands: torch.Tensor
    allowed_hands_prob: torch.Tensor
    policy_probs: torch.Tensor
    policy_probs_avg: torch.Tensor
    reach_weights: torch.Tensor
    reach_weights_avg: torch.Tensor
    cumulative_regrets: torch.Tensor
    regret_weight_sums: torch.Tensor
    last_model_values: torch.Tensor | None
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

        self.search_batch_size = search_batch_size
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
        nodes_at_depth = self.search_batch_size
        for _ in range(self.max_depth + 1):
            self.depth_offsets.append(self.depth_offsets[-1] + nodes_at_depth)
            nodes_at_depth *= self.num_actions
        self.total_nodes = self.depth_offsets[-1]

        N, M = self.search_batch_size, self.total_nodes

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

        # Set in construct_subgame and not updated.
        self.folded_mask = torch.zeros(M, dtype=torch.bool, device=self.device)
        self.folded_rewards = torch.zeros(M, dtype=self.float_dtype, device=self.device)

        # Notionally, at its parent, what is the probability of acting to get to this node?
        self.policy_probs = torch.zeros(
            M,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
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
        self.reach_weights = torch.zeros_like(self.beliefs)
        self.reach_weights_avg = torch.zeros_like(self.beliefs)
        self.root_pre_chance_beliefs = torch.zeros(
            self.search_batch_size,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )

        self.legal_mask = None

        # Invalid for [0, N) because we don't know the previous actor.
        self.prev_actor = torch.zeros(M, dtype=torch.long, device=self.device)
        self.root_index = torch.zeros(M, dtype=torch.long, device=self.device)

        self.combo_onehot_float = combo_to_onehot_tensor(device=self.device).float()
        # if self.device.type == "cuda":
        #     self.combo_onehot_float = self.combo_onehot_float.to_sparse_csr()
        self.chance_helper = ChanceNodeHelper(
            device=self.device,
            float_dtype=self.float_dtype,
            num_players=self.num_players,
            model=self.model,
        )
        self.combo_onehot_bool = self.chance_helper.combo_onehot_bool

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
            assert initial_beliefs.shape[1] == self.num_players
            assert initial_beliefs.shape[2] == NUM_HANDS
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        N = self.search_batch_size
        dest_indices = torch.arange(N, device=self.device)
        self.env.reset()
        self.env.copy_state_from(src_env, src_indices, dest_indices, copy_deck=True)
        self.valid_mask.zero_()
        self.valid_mask[:N] = True
        self.leaf_mask.zero_()
        self.leaf_mask[:N] = self.env.done[:N]
        self.showdown_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.folded_mask.zero_()
        self.folded_rewards.zero_()
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
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
        self.reach_weights.zero_()
        self.reach_weights_avg.zero_()
        self.legal_mask = None
        self.hand_rank_data = None
        self.root_index.zero_()
        self.root_index[:N] = torch.arange(N, device=self.device, dtype=torch.long)

    @profile
    def construct_subgame(self) -> None:
        """Expand the tree by cloning legal successor states at each depth."""
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

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
                self.root_index[new_legal_indices] = self.root_index[
                    current_legal_indices
                ]
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
            self.leaf_mask[offset_next:offset_next_next] = valid & (done | no_actions)

            # Showdown and fold values get set in set_leaf_values.
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

        self.values_avg[:] = self.latest_values

        leaf_start = self.depth_offsets[self.max_depth]
        leaf_end = self.depth_offsets[self.max_depth + 1]
        self.leaf_mask[leaf_start:leaf_end] = self.valid_mask[leaf_start:leaf_end]
        self.showdown_indices = torch.where(self.env.street == 4)[0]

        root_board_mask = self.env.board_onehot[:N].any(dim=1).reshape(N, -1).float()
        root_allowed = (self.combo_onehot_float @ root_board_mask.T).T < 0.5
        root_allowed_prob = root_allowed.float()
        root_allowed_prob /= root_allowed_prob.sum(dim=-1, keepdim=True).clamp(min=1)

        self.allowed_hands[:] = self._fan_out_deep(root_allowed)
        self.allowed_hands &= self.valid_mask[:, None]

        self.allowed_hands_prob[:] = self._fan_out_deep(root_allowed_prob)
        self.allowed_hands_prob.masked_fill_(~self.valid_mask[:, None], 0.0)
        self.prev_actor[N:] = self._fan_out(self.env.to_act)

        self.legal_mask = self.env.legal_bins_mask()
        valid_legal_masks = self.legal_mask[self.valid_mask & ~self.leaf_mask]
        has_legal = valid_legal_masks.any(dim=-1)
        assert has_legal.all(), "Every valid node must have at least one legal action."

    @torch.no_grad()
    @profile
    def _get_model_policy_probs(self, indices: torch.Tensor) -> torch.Tensor:
        features = self.feature_encoder.encode(self.beliefs)
        model_output = self.model(features[indices])
        logits = model_output.policy_logits
        legal_masks = self.legal_mask[indices]
        masked_logits = compute_masked_logits(logits, legal_masks[:, None, :])
        return F.softmax(masked_logits, dim=-1)

    @profile
    def _calculate_reach_weights(self, policy: torch.Tensor) -> torch.Tensor:
        """Calculate self reach weights for each node.

        Note we don't need to consider the chance nodes because we stop on street.

        Returns:
            reach_weights: [M, 2] tensor of reach weights for each node.
        """

        N, M = self.search_batch_size, self.total_nodes

        reach_weights = torch.zeros(M, 2, NUM_HANDS, device=self.device)
        reach_weights[:N] = 1.0

        for depth in range(self.max_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            weights_src = reach_weights[offset:offset_next]
            weights_dest = self._fan_out(weights_src, sliced=True)

            indices = torch.arange(offset_next_next - offset_next, device=self.device)
            prev_actor_dest = self.prev_actor[offset_next:offset_next_next]
            weights_dest[indices, prev_actor_dest] *= policy[
                offset_next:offset_next_next
            ]
            reach_weights[offset_next:offset_next_next] = weights_dest

        reach_weights.masked_fill_(~self.valid_mask[:, None, None], 0.0)
        return reach_weights

    def _initialize_with_copy(self, target: torch.Tensor | None = None) -> torch.Tensor:
        """Initialize the non-root nodes of the tree with a copy of the root nodes."""
        N, M = self.search_batch_size, self.total_nodes
        factor = M // N - 1
        target[N:] = (
            target[:N, None]
            .expand(-1, factor, -1, -1)
            .reshape(-1, self.num_players, NUM_HANDS)
        )

    @profile
    def _propagate_all_beliefs(
        self,
        target: torch.Tensor | None = None,
        source: torch.Tensor | None = None,
        reach_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propagate beliefs from all valid nodes to all valid nodes."""

        if target is None:
            target = self.beliefs
        if source is None:
            source = self.policy_probs
        if reach_weights is None:
            reach_weights = self.reach_weights

        self._initialize_with_copy(target)
        target *= reach_weights

        self._block_beliefs(target)
        self._normalize_beliefs(target)

        target.masked_fill_(~self.valid_mask[:, None, None], 0.0)

        assert torch.allclose(target.sum(dim=2), self.valid_mask[:, None].float())

    def _propagate_level_beliefs(self, depth: int):
        """Propagate beliefs from all nodes at a given level to all nodes at the next level."""
        N, M = self.search_batch_size, self.total_nodes

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
        target.masked_fill_(~self.allowed_hands[:, None, :], 0.0)

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

    @torch.no_grad()
    @profile
    def initialize_policy_and_beliefs(self) -> None:
        """Push public beliefs down the tree using the freshly initialised policy."""
        self.policy_probs.zero_()

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

        self.policy_probs.masked_fill_(~self.valid_mask[:, None], 0.0)

        self.reach_weights = self._calculate_reach_weights(self.policy_probs)
        self._block_beliefs(self.reach_weights)

        self.policy_probs_avg[:] = self.policy_probs
        self.reach_weights_avg[:] = self.reach_weights
        self.beliefs_avg[:] = self.beliefs

    @profile
    def warm_start(self) -> None:
        N, B = self.search_batch_size, self.num_actions
        min_value = torch.finfo(self.float_dtype).min

        # [M, ]
        values_br = torch.where(self.leaf_mask[:, None, None], self.latest_values, 0.0)

        all_actions = torch.arange(B, device=self.device)[None, :]
        assert (self.latest_values[~self.valid_mask] == 0.0).all()
        for depth, current_indices in self._valid_nodes(bottom_up=True):
            if depth == self.max_depth:
                continue

            offset_next = self.depth_offsets[depth + 1]
            offset = self.depth_offsets[depth]
            next_indices = (
                offset_next + (current_indices[:, None] - offset) * B + all_actions
            )
            # [n]
            actor = self.env.to_act[current_indices]
            opp = 1 - actor
            legal_mask = self.legal_mask[current_indices]
            # [n, B, 1326]
            next_actor_values = values_br[next_indices, actor[:, None]]
            next_actor_values.masked_fill_(~legal_mask[:, :, None], min_value)
            next_opp_values = values_br[next_indices, opp[:, None]]
            next_opp_values.masked_fill_(~legal_mask[:, :, None], 0.0)

            # Best-response per hand (mask illegal actions to -inf)
            best_actions = next_actor_values.argmax(dim=1)  # [n, 1326]
            best_values = next_actor_values.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            # take the player-to-act's best action, and the other player's average action
            # (NB see set_leaf_values note for why we use the sum)
            values_br[current_indices, actor] = best_values
            values_br[current_indices, opp] = next_opp_values.sum(dim=1)

        assert values_br[self.valid_mask].isfinite().all()

        # heuristic: scale regrets by the number of warm start iterations
        regrets = self.compute_instantaneous_regrets(
            values_achieved=values_br, values_expected=self.latest_values
        )
        self.cumulative_regrets += self.warm_start_iterations * regrets
        self.regret_weight_sums += self.warm_start_iterations
        self.update_policy(self.warm_start_iterations + 1)

    def _get_mixing_weights(self, t: int) -> torch.Tensor:
        """Get the mixing weights for the current iteration."""

        if self.cfr_type == CFRType.standard:
            return t - 1, 1
        elif self.cfr_type == CFRType.linear:
            return t - 1, 2
        elif self.cfr_type == CFRType.discounted:
            if t > self.dcfr_delay:
                return max(0, t - 1 - self.dcfr_delay), 2
            else:
                return 0, 1

    @torch.no_grad()
    @profile
    def set_leaf_values(self, t: int) -> None:
        """Populate per-hand payoffs for nodes marked as leaves."""

        # Set estimated leaf value from model for non-terminal nodes.
        model_mask = self.leaf_mask & ~self.env.done

        # CFR-AVG: subgame rooted at PBS from average policy
        features = self.feature_encoder.encode(
            self.beliefs_avg if self.cfr_avg else self.beliefs,
            pre_chance_node=True,
        )
        model_output = self.model(features[model_mask])
        if t <= 1 or not self.cfr_avg or self.last_model_values is None:
            self.latest_values[model_mask] = model_output.hand_values
        else:
            old, new = self._get_mixing_weights(t)
            self.latest_values[model_mask] = (
                old + new
            ) * model_output.hand_values - old * self.last_model_values
        self.last_model_values = model_output.hand_values

        # Translate EVs from model to opponent-reach-weighted CFVs for CFR.
        self.latest_values[model_mask] *= self.reach_weights[model_mask].flip(dims=[1])

        # Translate stored folded rewards to CFR-weighted CFVs.
        folded_reward = torch.stack(
            [
                self.folded_rewards[:, None] * self.reach_weights[:, 1],
                -self.folded_rewards[:, None] * self.reach_weights[:, 0],
            ],
            dim=1,
        )
        torch.where(
            self.folded_mask[:, None, None],
            folded_reward,
            self.latest_values,
            out=self.latest_values,
        )

        # Showdown values need to be updated based on beliefs.
        # The env has hands it uses for showdown, but those are fake.
        showdown_values = self._showdown_value(self.showdown_indices)
        self.latest_values[self.showdown_indices, 0] = (
            showdown_values * self.reach_weights[self.showdown_indices, 1]
        )
        self.latest_values[self.showdown_indices, 1] = (
            -showdown_values * self.reach_weights[self.showdown_indices, 0]
        )

    @profile
    def compute_expected_values(self) -> torch.Tensor:
        """Back up leaf hand values to their ancestors under the current policy."""

        self.latest_values.masked_fill_(~self.leaf_mask[:, None, None], 0.0)
        # First iteration: leaf values already populated; back propagate expectations
        for depth in range(self.max_depth - 1, -1, -1):
            self.profiler_step()  # Profile each depth iteration

            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            former_actor = self.prev_actor[offset_next:offset_next_next]
            dest_indices = torch.arange(
                offset_next_next - offset_next, device=self.device
            )

            # Pull back values to the source nodes.
            # NB: we are strategically ignoring the effect of blockers here.
            # It's a minor change, but it makes the computation much simpler:
            # all we have to do is add the opponent values as they already include
            # opponent reach (so effectively includes policy probability).
            # The original ReBeL source code does this too.
            values_weighted = self.latest_values[offset_next:offset_next_next].clone()
            values_weighted[dest_indices, former_actor] *= self.policy_probs[
                offset_next:offset_next_next
            ]
            values_src = self._pull_back(values_weighted, sliced=True).sum(dim=1)
            torch.where(
                self.leaf_mask[offset:offset_next, None, None],
                self.latest_values[offset:offset_next],
                values_src,
                out=self.latest_values[offset:offset_next],
            )
            self.latest_values[offset:offset_next].masked_fill_(
                ~self.valid_mask[offset:offset_next, None, None], 0.0
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

        M = self.total_nodes
        bottom = self.depth_offsets[1]

        regrets = torch.zeros_like(self.policy_probs)

        src_actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)
        src_actor_indices_fanout = self._fan_out(self.env.to_act)[:, None, None].expand(
            -1, -1, NUM_HANDS
        )
        src_opp_indices = (1 - self.env.to_act)[:, None, None].expand(-1, -1, NUM_HANDS)

        # This represents the opponent's reach prob at the src node.
        # Then actor acts at the transition src -> dest node.
        src_opp_beliefs = self.beliefs.gather(1, src_opp_indices).squeeze(1)
        src_opp_beliefs_fanout = self._fan_out(src_opp_beliefs)

        # The value at a node is already the EV over all actions.
        actor_values = values_expected.gather(1, src_actor_indices).squeeze(1)  # bottom
        actor_values_expected = self._fan_out(actor_values)
        actor_values_achieved = (
            values_achieved[bottom:].gather(1, src_actor_indices_fanout).squeeze(1)
        )

        advantages = actor_values_achieved - actor_values_expected

        # The goal here is to compute opp_range @ compatible (= opp_beliefs)
        # blocking = combo_onehot_float @ combo_onehot_float.T - torch.eye(1326)
        # => compatible = ~blocking = (1 - blocking)
        # => the below weight computation is equivalent to opp_beliefs @ compatible
        combo_onehot = self.combo_onehot_float
        multiply = combo_onehot @ (combo_onehot.T @ src_opp_beliefs_fanout.T)
        weights = (
            src_opp_beliefs_fanout.sum(dim=-1, keepdim=True)
            - multiply.T
            + src_opp_beliefs_fanout
        )
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
        updated_dest = self._push_down(updated_src)
        torch.where(
            self.valid_mask[bottom:, None],
            updated_dest,
            self.policy_probs[bottom:],
            out=self.policy_probs[bottom:],
        )

        self.reach_weights = self._calculate_reach_weights(self.policy_probs)
        self._block_beliefs(self.reach_weights)
        self._propagate_all_beliefs(self.beliefs, self.policy_probs, self.reach_weights)

        self.update_average_policy(t)
        self.reach_weights_avg = self._calculate_reach_weights(self.policy_probs_avg)
        self._block_beliefs(self.reach_weights_avg)
        self._propagate_all_beliefs(
            self.beliefs_avg, self.policy_probs_avg, self.reach_weights_avg
        )

    @profile
    def sample_leaf(
        self,
        root_indices: torch.Tensor,
        pbs: PublicBeliefState,
        pbs_start_idx: int,
        training_mode: bool,
    ) -> None:
        """Roll out from `root_indices` and copy the reached leaves into `pbs`.

        Args:
            root_indices: Indices of root nodes to sample trajectories from.
            pbs: Destination public belief state that will hold sampled leaves.
            pbs_start_idx: Offset inside `pbs` to start writing sampled rows.
            training_mode: Whether epsilon-greedy exploration is enabled.
        """
        M, B = self.total_nodes, self.num_actions
        valid_leaf_mask = self.leaf_mask & ~self.env.done
        count = root_indices.numel()
        assert count > 0
        assert count <= valid_leaf_mask.sum().item()

        players = torch.randint(
            0, 2, (count,), generator=self.generator, device=self.device
        )
        sample_epsilon = self.sample_epsilon if training_mode else 0.0
        legal_masks = self.env.legal_bins_mask()
        legal_counts = legal_masks.float().sum(dim=-1, keepdim=True)
        uniform = torch.where(legal_masks, 1 / legal_counts, 0)

        # select a hand for every valid node
        selected_hands = torch.zeros(M, dtype=torch.long, device=self.device)
        valid_indices = torch.where(self.valid_mask)[0]
        selected_hands[valid_indices] = torch.multinomial(
            self.beliefs[valid_indices, self.env.to_act[valid_indices]],
            1,
            generator=self.generator,
        ).squeeze(1)

        # start with root nodes and descend to leaves
        sampled_nodes = root_indices.clone()

        depth = 0
        active_mask = ~self.leaf_mask[sampled_nodes]
        policy_probs_by_src = self._pull_back(self.policy_probs)
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
                active_nodes, :, selected_hands[active_nodes]
            ]
            actions = torch.multinomial(
                torch.where(
                    sample_uniformly.view(-1, 1),
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
            active_mask = ~self.leaf_mask[sampled_nodes]
            depth += 1

        dest_indices = torch.arange(
            pbs_start_idx, pbs_start_idx + count, device=self.device
        )
        pbs.env.copy_state_from(self.env, sampled_nodes, dest_indices)
        pbs.beliefs[pbs_start_idx : pbs_start_idx + count] = self.beliefs[sampled_nodes]

    @profile
    def update_average_policy(self, t: int) -> None:
        """Update the average policy by mixing it with the current policy."""

        if self.cfr_type == CFRType.discounted and t <= self.dcfr_delay:
            self.policy_probs_avg[:] = self.policy_probs
            return
        elif t == 0:
            self.policy_probs_avg[:] = self.policy_probs
            return

        M, N = self.total_nodes, self.search_batch_size

        prev_actor = torch.zeros(M, dtype=torch.long, device=self.device)
        prev_actor[N:] = self._fan_out(self.env.to_act)

        reach_weights_prev = torch.ones_like(self.reach_weights)
        reach_weights_prev[N:] = self._fan_out(self.reach_weights)
        reach_weights_avg_prev = torch.ones_like(self.reach_weights_avg)
        reach_weights_avg_prev[N:] = self._fan_out(self.reach_weights_avg)

        # In the root nodes, prev_actor is invalid, but that's OK because
        # reach_weights is the same (1.0) for all players there.
        # Note we have to use the previous node's reach weights, since the policy probs
        # really live on that node (and otherwise we're double-multiplying)
        prev_actor_indices = prev_actor[:, None, None].expand(-1, -1, NUM_HANDS)
        reach_actor = torch.gather(reach_weights_prev, 1, prev_actor_indices).squeeze(1)
        reach_avg_actor = torch.gather(
            reach_weights_avg_prev, 1, prev_actor_indices
        ).squeeze(1)

        old, new = self._get_mixing_weights(t)
        reach_avg_actor *= old
        reach_actor *= new
        self.policy_probs_avg *= reach_avg_actor
        self.policy_probs_avg += self.policy_probs * reach_actor
        denom = reach_avg_actor + reach_actor
        self.policy_probs_avg /= denom.clamp(min=1e-8)
        self.policy_probs_avg.masked_fill_(denom < 1e-8, 0.0)

    def _get_sampling_schedule(self) -> torch.Tensor:
        leaf_indices = torch.where(self.leaf_mask & ~self.env.done)[0]
        leaf_indices = leaf_indices[
            torch.randperm(
                leaf_indices.numel(), generator=self.generator, device=self.device
            )
        ]
        sample_count = min(leaf_indices.numel(), self.search_batch_size)
        sampled_leaf_indices = leaf_indices[:sample_count]
        if sample_count > 0:
            if self.cfr_type == CFRType.discounted:
                sample_low = max(self.warm_start_iterations + 1, self.dcfr_delay + 1)
                sample_low = min(sample_low, self.cfr_iterations - 1)
            else:
                sample_low = min(
                    self.warm_start_iterations + 1, self.cfr_iterations - 1
                )
            sample_high = max(self.cfr_iterations, sample_low)
            distribution = (
                torch.arange(
                    sample_low, sample_high, dtype=torch.float32, device=self.device
                )
                + 1
                if self.cfr_type != CFRType.standard
                else torch.ones(sample_high - sample_low, device=self.device)
            )
            distribution /= distribution.sum()
            t_sample = torch.multinomial(
                distribution, sample_count, replacement=True, generator=self.generator
            )
            t_sample += sample_low

            return t_sample, sampled_leaf_indices

        return torch.empty(0, dtype=torch.long, device=self.device), torch.empty(
            0, dtype=torch.long, device=self.device
        )

    @profile
    def self_play_iteration(
        self, training_mode: bool = True
    ) -> Optional[PublicBeliefState]:
        """Run one iteration through the CFR loop and produce leaf samples for replay."""

        self.stats.clear()

        self.construct_subgame()
        self.initialize_policy_and_beliefs()

        if self.warm_start_iterations > 0:
            self.set_leaf_values(0)
            self.compute_expected_values()
            self.warm_start()

        # use t=0 here so set_leaf_values doesn't do the CFR-AVG de-averaging.
        self.set_leaf_values(0)
        self.compute_expected_values()
        self.values_avg[:] = self.latest_values

        t_sample, sampled_leaf_indices = self._get_sampling_schedule()
        sample_count = t_sample.numel()
        next_pbs = None
        if sample_count > 0:
            beliefs_template = torch.zeros(
                sample_count, self.num_players, NUM_HANDS, device=self.device
            )
            next_pbs = PublicBeliefState.from_proto(
                env_proto=self.env,
                beliefs=beliefs_template,
                num_envs=sample_count,
            )
            next_pbs_idx = 0

        for t in range(self.warm_start_iterations, self.cfr_iterations):
            self.profiler_step()  # Profile start of CFR iteration

            if sample_count > 0:
                # If t == t_sample, sample leaf PBS
                sample_now = torch.where(t_sample == t)[0]
                if sample_now.numel() > 0:
                    self.sample_leaf(
                        sampled_leaf_indices[sample_now],
                        next_pbs,
                        next_pbs_idx,
                        training_mode=training_mode,
                    )
                    next_pbs_idx += sample_now.numel()

            regrets = self.compute_instantaneous_regrets(self.values_avg)
            if self.cfr_type == CFRType.linear:
                # Alternate updates.
                regrets.masked_fill_(
                    self.env.to_act[:, None] == t % self.num_players, 0.0
                )
            elif self.cfr_type == CFRType.discounted:
                factor = torch.where(
                    regrets > 0, (t - 1) ** self.dcfr_alpha, (t - 1) ** self.dcfr_beta
                )
                self.cumulative_regrets *= factor / (factor + 1)
                self.regret_weight_sums *= factor / (factor + 1)
            self.regret_weight_sums += 1
            self.cumulative_regrets += regrets

            old_policy_probs = self.policy_probs.clone()
            self.update_policy(t)
            self._record_stats(t, old_policy_probs)

            self.set_leaf_values(t)
            self.compute_expected_values()

            old, new = self._get_mixing_weights(t)
            self.values_avg *= old
            self.values_avg += new * self.latest_values
            self.values_avg /= old + new

        self._record_action_mix()
        self._record_cfr_entropy()
        self._record_cumulative_regret()

        return next_pbs

    @profile
    def training_data(
        self, exclude_start: bool = True
    ) -> tuple[RebelBatch, RebelBatch, RebelBatch]:
        """Aggregate model targets for supervised learning.

        Returns:
            Tuple of (start_of_street_value_batch, end_of_previous_street_value_batch,
            policy_batch).
        """
        N = self.search_batch_size
        top = self.depth_offsets[-2]
        policy_targets = self._pull_back(self.policy_probs_avg)
        policy_targets = policy_targets.permute(0, 2, 1)

        # Nominally we'd need to divide by reach weights here, but since we're only
        # taking the first level of the tree, those weights would all be 1.
        value_targets = self.values_avg
        if value_targets.abs().max() > 100:
            print("WARNING: Value targets are too large")

        features = self.feature_encoder.encode(self.beliefs_avg, pre_chance_node=False)[
            :top
        ]
        bin_amounts, legal_masks = self.env.legal_bins_amounts_and_mask()
        statistics = {
            "to_act": self.env.to_act,
            "street": self.env.street,
            "board": self.env.board_indices,
            "pot": self.env.pot,
            "bet_amounts": bin_amounts,
        }

        exploit_stats = self._compute_exploitability()

        # Value batch gets root states only. These should all be valid.
        value_statistics = {key: statistics[key][:N] for key in statistics}
        value_statistics["local_exploitability"] = exploit_stats.local_exploitability
        value_statistics["local_br_policy"] = exploit_stats.local_br_policy
        value_statistics["local_br_values"] = exploit_stats.local_br_values
        value_statistics["local_br_improvement"] = exploit_stats.local_br_improvement

        # Policy batch gets all valid, non-leaf states.
        valid_top = self.valid_mask[:top] & ~self.leaf_mask[:top]
        policy_statistics = {
            key: statistics[key][:top][valid_top] for key in statistics
        }
        value_batch = RebelBatch(
            features=features[:N],
            value_targets=value_targets[:N],
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
        # Prepare end-of-street value batch using pre-chance beliefs.
        pre_features_all = self.feature_encoder.encode(
            self.beliefs, pre_chance_node=True
        )
        # Clone root slice and override belief encoding with pre-chance beliefs.
        pre_features_root = pre_features_all[:N].clone()
        pre_beliefs = self.root_pre_chance_beliefs[:N].reshape(N, -1)
        pre_features_root.beliefs = pre_beliefs * 2 - 1

        value_targets_pre = value_targets[:N].clone()
        value_statistics_pre = {key: statistics[key][:N].clone() for key in statistics}
        value_statistics_pre["board"] = self.env.last_board_indices[:N].clone()
        prev_street = torch.where(
            (street_root > 0) & (street_root < 4) & (actions_root == 0),
            street_root - 1,
            street_root,
        )
        value_statistics_pre["street"] = prev_street

        start_mask = actions_root == 0

        turn_river_mask = start_mask & ((street_root == 2) | (street_root == 3))
        if turn_river_mask.any():
            expected_turn_river = self.chance_helper.single_card_chance_values(
                torch.where(turn_river_mask)[0],
                features[:N],
                self.root_pre_chance_beliefs,
                self.reach_weights,
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
                self.reach_weights,
            )
            value_targets_pre[flop_mask] = expected_flop

        pre_value_batch = RebelBatch(
            features=pre_features_root,
            value_targets=value_targets_pre,
            legal_masks=legal_masks[:N],
            statistics=value_statistics_pre,
        )
        if exclude_start:
            pre_value_batch = pre_value_batch[~root_nodes]

        return value_batch, pre_value_batch, policy_batch

    @profile
    def _showdown_value(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Exact river showdown EV using rank-CDF + blocker correction.
        Returns per-hand EV [N, 1326] (unsorted/original hand order) per env.
        """

        M = indices.numel()
        device = self.device
        dtype = torch.float32  # or match belief dtype

        if M == 0:
            return torch.zeros(0, NUM_HANDS, device=device, dtype=dtype)

        # --- Beliefs & boards ---
        beliefs = self.beliefs[indices]  # (M,2,1326)
        b_self = beliefs[:, 0, :].to(dtype)  # (M,1326)
        b_opp = beliefs[:, 1, :].to(dtype)  # (M,1326)
        board = self.env.board_indices[indices].int()  # (M,5)

        # Sorted position k (0..1325) replicated across batch
        k = torch.arange(NUM_HANDS, device=device).expand(M, -1)  # (M,1326)

        if self.hand_rank_data is None:
            # --- Ranks & sorted order per env (river deterministic strength) ---
            # hand_ranks: (M,1326) any integer/monotone rank key s.t. equal => tie
            # sorted_indices: argsort by (rank, tiebreak) ascending (weaker -> stronger)
            hand_ranks, sorted_indices = rank_hands(board)  # both (M,1326)

            # Ranks in sorted order
            ranks_sorted = torch.gather(hand_ranks, 1, sorted_indices)  # (M,1326)
            assert torch.all(
                ranks_sorted[:, 1:] >= ranks_sorted[:, :-1]
            ), "rank_hands order is descending; flip or fix rank_hands"

            # --- Tie groups: start flags, group ids, [L,R] spans per sorted position ---
            is_start = torch.ones_like(ranks_sorted, dtype=torch.bool)  # (M,1326)
            is_start[:, 1:] = ranks_sorted[:, 1:] != ranks_sorted[:, :-1]
            group_id = is_start.cumsum(dim=1, dtype=torch.int) - 1  # (M,1326), 0..G-1

            # For each group id, store first/last index in sorted order
            starts = torch.full(
                (M, NUM_HANDS), NUM_HANDS, dtype=torch.int, device=device
            )
            ends = torch.full((M, NUM_HANDS), -1, dtype=torch.int, device=device)
            starts.scatter_reduce_(
                1, group_id, k.int(), reduce="amin", include_self=True
            )
            ends.scatter_reduce_(1, group_id, k.int(), reduce="amax", include_self=True)

            # L,R per sorted position
            L = torch.gather(starts, 1, group_id)  # (M,1326)
            R = torch.gather(ends, 1, group_id)  # (M,1326)
            L_idx = L
            R_idx = (R + 1).clamp(max=NUM_HANDS)
            assert (L <= R).all(), "L must be <= R"
            assert torch.all(
                torch.gather(ranks_sorted, 1, L) == torch.gather(ranks_sorted, 1, R)
            ), "L/R must have same rank"

            # Inverse permutation (sorted->original) for mapping EV back
            inv_sorted = torch.argsort(sorted_indices, dim=1)  # (M,1326)

            # --- Hand/card incidence & board masking ---
            combo_to_onehot = combo_to_onehot_tensor(device=device)  # (1326,52)
            hands_c1c2 = hand_combos_tensor(device=device)  # (1326,2)

            # Per-env mask for cards not on the board: True = usable card
            card_ok = torch.ones((M, 52), dtype=torch.bool, device=device)
            card_ok.scatter_(1, board, False)  # False for board cards

            # Hand usable mask (unsorted): hand must use only ok cards
            H = combo_to_onehot.unsqueeze(0).expand(M, -1, -1)  # (M,1326,52)
            hand_ok_mask = self.allowed_hands[indices]
            hand_ok_mask_sorted = torch.gather(hand_ok_mask, 1, sorted_indices)

            # Cards (c1,c2) of each *sorted* hand per env
            hands_c1c2_sorted = torch.gather(
                hands_c1c2.unsqueeze(0).expand(M, -1, -1),  # (M,1326,2)
                1,
                sorted_indices.unsqueeze(-1).expand(-1, -1, 2),
            )  # (M,1326,2)

            self.hand_rank_data = HandRankData(
                sorted_indices=sorted_indices,
                inv_sorted=inv_sorted,
                H=H,
                card_ok=card_ok,
                hand_ok_mask=hand_ok_mask,
                hand_ok_mask_sorted=hand_ok_mask_sorted,
                hands_c1c2_sorted=hands_c1c2_sorted,
                L_idx=L_idx,
                R_idx=R_idx,
            )
        else:
            sorted_indices = self.hand_rank_data.sorted_indices
            inv_sorted = self.hand_rank_data.inv_sorted
            H = self.hand_rank_data.H
            card_ok = self.hand_rank_data.card_ok
            hand_ok_mask = self.hand_rank_data.hand_ok_mask
            hand_ok_mask_sorted = self.hand_rank_data.hand_ok_mask_sorted
            hands_c1c2_sorted = self.hand_rank_data.hands_c1c2_sorted
            L_idx = self.hand_rank_data.L_idx
            R_idx = self.hand_rank_data.R_idx

        assert not (b_self * ~hand_ok_mask).any()
        assert not (b_opp * ~hand_ok_mask).any()
        assert torch.allclose(b_self.sum(dim=1), self.valid_mask[indices].float())
        assert torch.allclose(b_opp.sum(dim=1), self.valid_mask[indices].float())

        c1 = hands_c1c2_sorted[..., 0]  # (M,1326)
        c2 = hands_c1c2_sorted[..., 1]  # (M,1326)

        # Sort opponent marginal by strength order
        b_opp_sorted = torch.gather(b_opp, 1, sorted_indices)  # (M,1326)

        # Hand->card incidence in sorted order with board columns zeroed
        H_sorted = torch.gather(H, 1, sorted_indices.unsqueeze(-1).expand(-1, -1, 52))
        H_sorted = H_sorted & card_ok.unsqueeze(1)  # (M,1326,52)

        # --- Prefix sums over opponent mass (global and per-card), left-padded ---
        P = torch.cumsum(b_opp_sorted, dim=1)  # (M,1326)
        P = torch.cat(
            [torch.zeros(M, 1, device=device, dtype=dtype), P], dim=1
        )  # (M,1327)

        per_card_mass = H_sorted.to(dtype) * b_opp_sorted.unsqueeze(-1)  # (M,1326,52)
        Pcards = torch.cumsum(per_card_mass, dim=1)  # (M,1326,52)
        # -- Prefix sums over opponent mass, per card --
        Pcards = torch.cat(
            [torch.zeros(M, 1, 52, device=device, dtype=dtype), Pcards], dim=1
        )  # (M,1327,52)

        # --- Win/tie masses for each sorted position ---

        # Gather needed prefixes
        P_before = torch.gather(P, 1, L_idx)  # (M,1326)
        Pcards_before = torch.gather(
            Pcards, 1, L_idx.unsqueeze(-1).expand(-1, -1, 52)
        )  # (M,1326,52)

        # Win mass: all strictly weaker, excluding blockers
        Pcards_k_c1 = Pcards_before.gather(2, c1.unsqueeze(-1)).squeeze(-1)
        Pcards_k_c2 = Pcards_before.gather(2, c2.unsqueeze(-1)).squeeze(-1)
        win_mass = P_before - Pcards_k_c1 - Pcards_k_c2

        # Tie mass over [L,R] inclusive, excluding blockers
        P_R = torch.gather(P, 1, R_idx)
        P_L = torch.gather(P, 1, L_idx)
        seg_sum = P_R - P_L  # (M,1326)

        gL = L_idx.unsqueeze(-1).expand(-1, -1, 52)
        gR = R_idx.unsqueeze(-1).expand(-1, -1, 52)
        Pcards_R = torch.gather(Pcards, 1, gR)  # (M,1326,52)
        Pcards_L = torch.gather(Pcards, 1, gL)  # (M,1326,52)
        seg_c1 = (Pcards_R - Pcards_L).gather(2, c1.unsqueeze(-1)).squeeze(-1)
        seg_c2 = (Pcards_R - Pcards_L).gather(2, c2.unsqueeze(-1)).squeeze(-1)
        # Re-add hero combo mass (present in both seg_c1 and seg_c2)
        tie_mass = seg_sum - seg_c1 - seg_c2 + b_opp_sorted

        # --- Denominator: compatible opp mass for each hero hand (unsorted belief) ---
        Pc_last = Pcards[:, -1, :]  # (M, 52) totals per card
        denom = (
            1.0 - Pc_last.gather(1, c1) - Pc_last.gather(1, c2) + b_opp_sorted
        ).clamp(min=1e-12)
        valid_denom = denom > 1e-12
        assert ((valid_denom) | ((win_mass < 1e-5) & (tie_mass < 1e-5))).all()

        # Probabilities & EV (in sorted order)
        win_prob = torch.where(valid_denom, win_mass / denom, 0.0)
        tie_prob = torch.where(valid_denom, tie_mass / denom, 0.0)
        loss_prob = torch.where(valid_denom, 1.0 - win_prob - tie_prob, 0.0)

        EV_hand_sorted = win_prob - loss_prob

        # Map per-hand EV back to original hand order
        EV_hand = torch.gather(EV_hand_sorted, 1, inv_sorted)  # (M,1326)
        EV_hand = EV_hand * hand_ok_mask.to(dtype)  # zero impossible hands

        # Range EV for the player
        potential = (
            self.env.stacks[indices, 0]
            + self.env.pot[indices]
            - self.env.starting_stack
        )

        return EV_hand * potential[:, None] / self.env.scale

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
            reachable = (self.reach_weights > 0).any(dim=1)[: self.depth_offsets[-2]]
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
        N = self.search_batch_size
        actions = self._pull_back(self.policy_probs_avg)[:N]
        mask = self.valid_mask[:N] & ~self.leaf_mask[:N]
        probs = actions[mask]
        entropy = torch.where(probs > 1e-12, -(probs * probs.log()), 0.0)
        self.stats["cfr_entropy"] = entropy.sum(dim=1).mean().item()

    def _record_cumulative_regret(self) -> None:
        self.stats["mean_positive_regret"] = (
            self.cumulative_regrets.clamp(min=0).mean().item()
        )

        N = self.search_batch_size

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

    def _policy_values(
        self,
        policy: torch.Tensor,
        leaf_values: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate `policy` without any deviations.

        Returns per-node counterfactual values constructed from `leaf_values` via
        bottom-up expectations using `policy`.
        """

        assert self.legal_mask is not None

        values = torch.zeros_like(leaf_values)
        values[self.leaf_mask] = leaf_values[self.leaf_mask]

        B = self.num_actions
        all_actions = torch.arange(B, device=self.device)[None, :]

        for depth, current_indices in self._valid_nodes(bottom_up=True):
            if depth == self.max_depth:
                continue

            non_leaf = ~self.leaf_mask[current_indices]
            current = current_indices[non_leaf]
            if current.numel() == 0:
                continue

            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            next_indices = offset_next + (current[:, None] - offset) * B + all_actions

            legal = self.legal_mask[current]

            flat_next = next_indices.reshape(-1)
            next_values = values[flat_next].view(-1, B, self.num_players, NUM_HANDS)
            next_policy = policy[flat_next].view(-1, B, NUM_HANDS)

            next_values = next_values.clone()
            next_policy = next_policy.clone()
            next_values.masked_fill_(~legal[:, :, None, None], 0.0)
            next_policy.masked_fill_(~legal[:, :, None], 0.0)

            policy_sum = next_policy.sum(dim=1, keepdim=True).clamp_(min=1e-12)
            norm_policy = next_policy / policy_sum
            expected = (next_values * norm_policy[:, :, None, :]).sum(dim=1)

            values[current] = expected

        values.masked_fill_(~self.valid_mask[:, None, None], 0.0)
        return values

    def _best_response_values(
        self,
        policy: torch.Tensor,
        base_values: torch.Tensor,
        target_player: int,
    ) -> torch.Tensor:
        """Compute counterfactual values when `target_player` deviates optimally.

        Mirrors the warm_start structure while allowing only `target_player`
        to pick their best action; the opponent follows `policy`.
        """

        assert target_player in (0, 1)
        values_br = torch.zeros_like(base_values)
        values_br[self.leaf_mask] = base_values[self.leaf_mask]

        B = self.num_actions
        all_actions = torch.arange(B, device=self.device)[None, :]
        min_value = torch.finfo(base_values.dtype).min

        for depth, current_indices in self._valid_nodes(bottom_up=True):
            if depth == self.max_depth:
                continue

            non_leaf = ~self.leaf_mask[current_indices]
            current = current_indices[non_leaf]
            if current.numel() == 0:
                continue

            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            next_indices = offset_next + (current[:, None] - offset) * B + all_actions

            legal = self.legal_mask[current]
            actor = self.env.to_act[current]
            target_mask = actor == target_player

            flat_next = next_indices.reshape(-1)
            next_values = values_br[flat_next].view(-1, B, self.num_players, NUM_HANDS)
            next_policy = policy[flat_next].view(-1, B, NUM_HANDS)

            next_values = next_values.clone()
            next_policy = next_policy.clone()
            next_values.masked_fill_(~legal[:, :, None, None], 0.0)
            next_policy.masked_fill_(~legal[:, :, None], 0.0)

            player_values = next_values[:, :, target_player, :].clone()
            player_values.masked_fill_(~legal[:, :, None], min_value)
            best_action = player_values.argmax(dim=1)
            gather_idx = best_action.unsqueeze(1)

            best_player = torch.gather(player_values, 1, gather_idx).squeeze(1)
            opp_slice = next_values[:, :, 1 - target_player, :]
            best_opp = torch.gather(opp_slice, 1, gather_idx).squeeze(1)

            policy_sum = next_policy.sum(dim=1, keepdim=True).clamp_(min=1e-12)
            norm_policy = next_policy / policy_sum
            expected = (next_values * norm_policy[:, :, None, :]).sum(dim=1)

            values_br[current] = expected
            target_indices = current[target_mask]
            values_br[target_indices, target_player] = best_player[target_mask]
            values_br[target_indices, 1 - target_player] = best_opp[target_mask]

        values_br.masked_fill_(~self.valid_mask[:, None, None], 0.0)
        return values_br

    def _compute_exploitability(self) -> ExploitabilityStats:
        """Record depth-limited best-response exploitability estimate."""

        policy = self.policy_probs_avg
        leaf_values = self.values_avg
        reach_weights = self.reach_weights_avg if self.cfr_avg else self.reach_weights
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        base_values = self._policy_values(policy, leaf_values)
        br_values_p0 = self._best_response_values(policy, base_values, target_player=0)
        br_values_p1 = self._best_response_values(policy, base_values, target_player=1)

        N = self.search_batch_size
        if N == 0:
            empty = torch.empty(0, device=self.device, dtype=self.float_dtype)
            empty2 = torch.empty(0, 2, device=self.device, dtype=self.float_dtype)
            return ExploitabilityStats(
                local_exploitability=empty,
                local_br_policy=empty2,
                local_br_values=empty2.clone(),
                local_br_improvement=empty2.clone(),
            )

        def _cf_to_ev(values: torch.Tensor, opp_reach: torch.Tensor) -> torch.Tensor:
            denom = opp_reach.clamp(min=1e-12)
            return torch.where(opp_reach > 0, values / denom, 0.0)

        base_root = base_values[:N]
        br_root_p0 = br_values_p0[:N]
        br_root_p1 = br_values_p1[:N]
        reach_root = reach_weights[:N]

        base_ev_p0 = _cf_to_ev(base_root[:, 0], reach_root[:, 1])
        base_ev_p1 = _cf_to_ev(base_root[:, 1], reach_root[:, 0])
        br_ev_p0 = _cf_to_ev(br_root_p0[:, 0], reach_root[:, 1])
        br_ev_p1 = _cf_to_ev(br_root_p1[:, 1], reach_root[:, 0])

        beliefs_root = beliefs[:N]
        # Weight each player's EV by their own root belief distribution. Mixing based on
        # the acting seat caused best-response improvements to become negative.
        beliefs_p0 = beliefs_root[:, 0, :]
        beliefs_p1 = beliefs_root[:, 1, :]

        base_player0 = (base_ev_p0 * beliefs_p0).sum(dim=1)
        base_player1 = (base_ev_p1 * beliefs_p1).sum(dim=1)
        br_player0 = (br_ev_p0 * beliefs_p0).sum(dim=1)
        br_player1 = (br_ev_p1 * beliefs_p1).sum(dim=1)

        improvement_p0 = br_player0 - base_player0
        improvement_p1 = br_player1 - base_player1
        total_exploitability = improvement_p0 + improvement_p1

        return ExploitabilityStats(
            local_exploitability=total_exploitability,
            local_br_policy=torch.stack([base_player0, base_player1], dim=1),
            local_br_values=torch.stack([br_player0, br_player1], dim=1),
            local_br_improvement=torch.stack([improvement_p0, improvement_p1], dim=1),
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
            .view(-1, *data.shape[1:])
        )

    def _fan_out_deep(self, data: torch.Tensor) -> torch.Tensor:
        """Broadcast root-aligned tensors across every node in the tree.

        Args:
            data: Tensor shaped [N, ...] aligned with the root batch.
        Returns:
            Tensor shaped [M, ...] with each root slice repeated for every node
            that descends from that root.
        """
        assert data.shape[0] == self.search_batch_size
        assert self.total_nodes % self.search_batch_size == 0
        copies = self.total_nodes // self.search_batch_size
        expanded = data[:, None].expand(-1, copies, *data.shape[1:])
        return expanded.reshape(-1, *data.shape[1:]).clone()

    def _leaf_node_indices(self) -> torch.Tensor:
        """Return flattened indices for valid nodes marked as leaves."""
        return torch.where(self.leaf_mask)[0]
