from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Generator, Optional

import torch
import torch.nn.functional as F

from alphaholdem.core.structured_config import CFRType, ModelType
from alphaholdem.env.card_utils import (
    combo_to_onehot_tensor,
    hand_combos_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.env.rules import rank_hands
from alphaholdem.models.mlp.rebel_ffn import NUM_HANDS, RebelFFN
from alphaholdem.rl.rebel_replay import RebelBatch
from alphaholdem.utils.model_utils import compute_masked_logits
from alphaholdem.utils.profiling import profile

T_WARM = 15


@dataclass
class PublicBeliefState:
    """Public belief state for both players."""

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
            beliefs: Initial belief tensor shaped `[batch, players, NUM_HANDS]`.
            num_envs: Optional override for the number of vectorised environments.
        """
        return PublicBeliefState(
            env=HUNLTensorEnv.from_proto(env_proto, num_envs=num_envs),
            beliefs=beliefs,
        )

    def __post_init__(self):
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
    allowed_hands: torch.Tensor
    allowed_hands_prob: torch.Tensor
    policy_probs: torch.Tensor
    policy_probs_avg: torch.Tensor
    reach_weights: torch.Tensor
    reach_weights_avg: torch.Tensor
    cumulative_regrets: torch.Tensor
    values: torch.Tensor
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
        model_type: ModelType,
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
        assert cfr_iterations > warm_start_iterations

        self.search_batch_size = search_batch_size
        self.model = model
        self.max_depth = max_depth
        self.bet_bins = bet_bins
        self.cfr_iterations = cfr_iterations
        self.warm_start_iterations = warm_start_iterations
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
        # Should be a subset of valid_mask.
        self.leaf_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )

        # Set in construct_subgame and not updated.
        self.folded_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )
        self.folded_rewards = torch.zeros(
            self.total_nodes, dtype=self.float_dtype, device=self.device
        )

        # Notionally, at its parent, what is the probability of acting to get to this node?
        self.policy_probs = torch.zeros(
            self.total_nodes,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
        # Cumulative regret of taking this node vs the best at the parent node.
        self.cumulative_regrets = torch.zeros_like(self.policy_probs)

        # One value per node per player per hand
        self.values = torch.zeros(
            self.total_nodes,
            self.num_players,
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.values_avg = torch.zeros_like(self.values)
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
        self.beliefs_avg = torch.zeros_like(self.beliefs)
        self.reach_weights = torch.zeros_like(self.beliefs)
        self.reach_weights_avg = torch.zeros_like(self.beliefs)

        self.legal_mask = None

        self.combo_onehot_float = combo_to_onehot_tensor(device=self.device).float()
        self.allowed_hands = torch.zeros(
            self.total_nodes, NUM_HANDS, device=self.device, dtype=torch.bool
        )
        self.allowed_hands_prob = torch.zeros(
            self.total_nodes, NUM_HANDS, device=self.device, dtype=self.float_dtype
        )

        # Feature encoder for belief computation
        if model_type == ModelType.better_ffn:
            self.feature_encoder = BetterFeatureEncoder(
                env=self.env,
                device=self.device,
                dtype=self.float_dtype,
            )
        elif model_type == ModelType.rebel_ffn:
            self.feature_encoder = RebelFeatureEncoder(
                env=self.env,
                device=self.device,
                dtype=self.float_dtype,
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")

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
        self.folded_mask.zero_()
        self.folded_rewards.zero_()
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
        self.cumulative_regrets.zero_()
        self.values.zero_()
        self.values_avg.zero_()
        self.beliefs.zero_()
        self.beliefs[:N] = initial_beliefs
        self.reach_weights.zero_()
        self.reach_weights_avg.zero_()
        self.legal_mask = None
        self.hand_rank_data = None

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

        self.values_avg[:] = self.values

        leaf_start = self.depth_offsets[self.max_depth]
        leaf_end = self.depth_offsets[self.max_depth + 1]
        self.leaf_mask[leaf_start:leaf_end] = self.valid_mask[leaf_start:leaf_end]

        self.allowed_hands[:] = (
            self.env.board_onehot.any(dim=1).view(-1, 52).float()
            @ self.combo_onehot_float.T
        ) < 0.5
        self.allowed_hands_prob[
            :
        ] = self.allowed_hands.float() / self.allowed_hands.sum(
            dim=-1, keepdim=True
        ).clamp(
            min=1
        )

        self.legal_mask = self.env.legal_bins_mask()
        valid_legal_masks = self.legal_mask[self.valid_mask & ~self.leaf_mask]
        has_legal = valid_legal_masks.any(dim=-1)
        assert has_legal.all(), "Every valid node must have at least one legal action."

        assert self.values[self.valid_mask].abs().max() <= 5

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

        # Invalid for [0, N) because we don't know the previous actor.
        prev_actor = torch.zeros(M, dtype=torch.long, device=self.device)
        prev_actor[N:] = self._fan_out(self.env.to_act)

        for depth in range(self.max_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            weights_src = reach_weights[offset:offset_next]
            weights_dest = self._fan_out(weights_src, sliced=True)

            indices = torch.arange(offset_next_next - offset_next, device=self.device)
            prev_actor_dest = prev_actor[offset_next:offset_next_next]
            weights_dest[indices, prev_actor_dest] *= policy[
                offset_next:offset_next_next
            ]
            reach_weights[offset_next:offset_next_next] = weights_dest

        reach_weights.masked_fill_(~self.valid_mask[:, None, None], 0.0)
        return reach_weights

    def _initialize_with_copy(self, target: torch.Tensor | None = None) -> torch.Tensor:
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

        N, M = self.search_batch_size, self.total_nodes

        if target is None:
            target = self.beliefs
        if source is None:
            source = self.policy_probs
        if reach_weights is None:
            reach_weights = self.reach_weights

        prev_actor = torch.zeros(M, dtype=torch.long, device=self.device)
        prev_actor[N:] = self._fan_out(self.env.to_act)

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

        prev_actor = torch.zeros(M, dtype=torch.long, device=self.device)
        prev_actor[N:] = self._fan_out(self.env.to_act)

        probs = self.policy_probs[offset_next:offset_next_next]
        self.beliefs[offset_next:offset_next_next] = self._fan_out(
            self.beliefs[offset:offset_next], sliced=True
        )

        indices = torch.arange(offset_next, offset_next_next, device=self.device)
        self.beliefs[indices, prev_actor[offset_next:offset_next_next]] *= probs

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
        values_br = torch.where(self.leaf_mask[:, None, None], self.values, 0.0)

        all_actions = torch.arange(B, device=self.device)[None, :]
        assert (self.values[~self.valid_mask] == 0.0).all()
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
            values_achieved=values_br, values_expected=self.values
        )
        self.cumulative_regrets += self.warm_start_iterations * regrets
        self.update_policy(self.warm_start_iterations + 1)

    @torch.no_grad()
    @profile
    def set_leaf_values(self) -> None:
        """Populate per-hand payoffs for nodes marked as leaves."""

        # Set estimated leaf value from model for non-terminal nodes.
        model_mask = self.leaf_mask & ~self.env.done

        features = self.feature_encoder.encode(
            self.beliefs_avg if self.cfr_avg else self.beliefs,
        )
        model_output = self.model(features[model_mask])
        self.values[model_mask] = model_output.hand_values

        self.values[model_mask] *= self.reach_weights[model_mask].flip(dims=[1])

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
            self.values,
            out=self.values,
        )

        # Fold values were set in construct_subgame and don't need updating.
        # Showdown values need to be updated based on beliefs.
        # The env has hands it uses for showdown, but those are fake.
        showdown = torch.where(self.env.street == 4)[0]
        # assert self.env.done[showdown].all()
        # assert self.leaf_mask[showdown].all()
        showdown_values = self._showdown_value(showdown)
        self.values[showdown, 0] = showdown_values * self.reach_weights[showdown, 1]
        self.values[showdown, 1] = -showdown_values * self.reach_weights[showdown, 0]

    @profile
    def compute_expected_values(self) -> torch.Tensor:
        """Back up leaf hand values to their ancestors under the current policy."""

        self.values.masked_fill_(~self.leaf_mask[:, None, None], 0.0)
        # First iteration: leaf values already populated; back propagate expectations
        for depth in range(self.max_depth - 1, -1, -1):
            self.profiler_step()  # Profile each depth iteration

            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            former_actor = self._fan_out(self.env.to_act[offset:offset_next])
            dest_indices = torch.arange(
                offset_next_next - offset_next, device=self.device
            )

            # Pull back values to the source nodes.
            # NB: we are strategically ignoring the effect of blockers here.
            # It's a minor change, but it makes the computation much simpler:
            # all we have to do is add the opponent values as they already include
            # opponent reach (so effectively includes policy probability).
            # The original ReBeL source code does this too.
            values_weighted = self.values[offset_next:offset_next_next].clone()
            values_weighted[dest_indices, former_actor] *= self.policy_probs[
                offset_next:offset_next_next
            ]
            values_src = self._pull_back(values_weighted, sliced=True).sum(dim=1)
            torch.where(
                self.leaf_mask[offset:offset_next, None, None],
                self.values[offset:offset_next],
                values_src,
                out=self.values[offset:offset_next],
            )
            self.values[offset:offset_next].masked_fill_(
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

        actor_dest = torch.zeros(M, dtype=torch.long, device=self.device)
        actor_dest[bottom:] = self._fan_out(self.env.to_act)

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
        weights = (
            src_opp_beliefs_fanout.sum(dim=-1, keepdim=True)
            - src_opp_beliefs_fanout @ combo_onehot @ combo_onehot.T
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

        if self.cfr_type == CFRType.discounted:
            reach_avg_actor *= max(0, t - 1 - self.dcfr_delay)
        else:
            reach_avg_actor *= t - 1
        weight = 2 if self.cfr_type != CFRType.standard else 1
        reach_actor *= weight
        self.policy_probs_avg *= reach_avg_actor
        self.policy_probs_avg += self.policy_probs * reach_actor
        denom = reach_avg_actor + reach_actor
        torch.where(
            denom > 1e-8,
            self.policy_probs_avg / denom.clamp(min=1e-8),
            torch.zeros_like(self.policy_probs_avg),
            out=self.policy_probs_avg,
        )

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
                sample_low = min(sample_low, self.cfr_iterations)
            else:
                sample_low = min(self.warm_start_iterations + 1, self.cfr_iterations)
            sample_high = max(self.cfr_iterations, sample_low)
            distribution = (
                torch.arange(
                    sample_low, sample_high, dtype=torch.float32, device=self.device
                )
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
            self.set_leaf_values()
            self.compute_expected_values()
            self.warm_start()

        self.set_leaf_values()
        self.compute_expected_values()
        self.values_avg[:] = self.values

        t_sample, sampled_leaf_indices = self._get_sampling_schedule()
        sample_count = t_sample.numel()
        next_pbs = None
        if sample_count > 0:
            next_pbs = PublicBeliefState.from_proto(
                env_proto=self.env,
                beliefs=torch.zeros(
                    sample_count, self.num_players, NUM_HANDS, device=self.device
                ),
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

            regrets = self.compute_instantaneous_regrets(self.values)
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
            self.cumulative_regrets += regrets

            old_policy_probs = self.policy_probs.clone()
            self.update_policy(t)
            self._record_stats(t, old_policy_probs)

            self.set_leaf_values()
            self.compute_expected_values()

            if self.cfr_type == CFRType.discounted:
                if t > self.dcfr_delay:
                    linear_weight = max(0, t - 1 - self.dcfr_delay)
                    self.values_avg *= linear_weight
                    self.values_avg += 2 * self.values
                    self.values_avg /= linear_weight + 2
                else:
                    self.values_avg[:] = self.values
            else:
                weight = 2 if self.cfr_type != CFRType.standard else 1
                self.values_avg *= t - 1
                self.values_avg += weight * self.values
                self.values_avg /= t - 1 + weight

        self.stats["mean_positive_regret"] = (
            self.cumulative_regrets.clamp(min=0).mean().item()
        )
        self._record_action_mix()
        self._record_cfr_entropy()

        return next_pbs

    def sample_data(self) -> RebelBatch:
        """Aggregate model targets from the current root batch for supervised learning."""
        N = self.search_batch_size
        top = self.depth_offsets[-2]
        policy_targets = self._pull_back(self.policy_probs_avg)
        policy_targets = policy_targets.permute(0, 2, 1)

        # Nominally we'd need to divide by reach weights here, but since we're only
        # taking the first level of the tree, those weights would all be 1.
        value_targets = self.values_avg
        if value_targets.abs().max() > 100:
            print("WARNING: Value targets are too large")

        features = self.feature_encoder.encode(self.beliefs_avg)[:top]
        bin_amounts, legal_masks = self.env.legal_bins_amounts_and_mask()
        statistics = {
            "to_act": self.env.to_act,
            "street": self.env.street,
            "board": self.env.board_indices,
            "pot": self.env.pot,
            "bet_amounts": bin_amounts,
        }

        # Value batch gets root states only.
        value_batch = RebelBatch(
            features=features[:N],
            value_targets=value_targets[:N],
            legal_masks=legal_masks[:N],
            statistics={key: statistics[key][:N] for key in statistics},
        )
        # Policy batch gets all states.
        policy_batch = RebelBatch(
            features=features,
            policy_targets=policy_targets,
            legal_masks=legal_masks[:top],
            statistics={key: statistics[key][:top] for key in statistics},
        )
        return value_batch, policy_batch

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
            conflicts = H & (~card_ok.unsqueeze(1))  # mark hands using any blocked card
            hand_ok_mask = ~conflicts.any(
                dim=2
            )  # True only if both cards are available
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
        N = self.search_batch_size
        actions = self._pull_back(self.policy_probs_avg)[:N]
        mask = self.valid_mask[:N] & ~self.leaf_mask[:N]
        probs = actions[mask]
        entropy = torch.where(probs > 1e-12, -(probs * probs.log()), 0.0)
        self.stats["cfr_entropy"] = entropy.sum(dim=1).mean().item()

    def _record_action_mix(self) -> None:
        """Record the action mix of the policy."""
        actions = self._pull_back(self.policy_probs_avg)
        mask = self.valid_mask & ~self.leaf_mask
        mask = mask[: actions.shape[0]]
        self.stats["action_mix"] = {
            "fold": actions[mask, 0].mean().item(),
            "call": actions[mask, 1].mean().item(),
            "bet": actions[mask, 2:-1].sum(dim=1).mean().item(),
            "allin": actions[mask, -1].mean().item(),
        }

    def _valid_nodes(
        self, bottom_up: bool = False
    ) -> Generator[tuple[int, torch.Tensor]]:
        """Yield `(depth, indices)` pairs for nodes marked as valid."""
        for depth in (
            range(self.max_depth + 1)
            if not bottom_up
            else range(self.max_depth, -1, -1)
        ):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            mask = self.valid_mask[offset:offset_next]
            yield depth, offset + torch.where(mask)[0]

    def _valid_actions(
        self, bottom_up: bool = False
    ) -> Generator[tuple[int, int, torch.Tensor, torch.Tensor]]:
        """Iterate over legal transitions, optionally in bottom-up order."""
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

        for depth in (
            range(self.max_depth)
            if not bottom_up
            else range(self.max_depth - 1, -1, -1)
        ):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            for action in range(B):
                current_legal_mask = (
                    self.legal_mask[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.leaf_mask[offset:offset_next]
                )
                if not current_legal_mask.any():
                    continue

                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                next_legal_indices = (
                    offset_next + (current_legal_indices - offset) * B + action
                )
                assert next_legal_indices.max().item() < M
                assert self.valid_mask[next_legal_indices].all()

                yield depth, action, current_legal_indices, next_legal_indices

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

    def _leaf_node_indices(self) -> torch.Tensor:
        """Return flattened indices for valid nodes marked as leaves."""
        return torch.where(self.leaf_mask)[0]
