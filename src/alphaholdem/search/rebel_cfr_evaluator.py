from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import torch

from alphaholdem.core.structured_config import CFRType
from alphaholdem.env.card_utils import (
    NUM_HANDS,
    combo_to_onehot_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.better_trm import BetterTRM
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.search.cfr_evaluator import (
    CFREvaluator,
    HandRankData,
    PublicBeliefState,
    padded_indices,
)
from alphaholdem.search.chance_node_helper import ChanceNodeHelper
from alphaholdem.utils.profiling import profile

T_WARM = 15


class RebelCFREvaluator(CFREvaluator):
    """ReBeL CFR Evaluator implementing the precise SELFPLAY algorithm."""

    root_nodes: int
    model: RebelFFN | BetterFFN | BetterTRM
    max_depth: int
    bet_bins: list[float]
    cfr_iterations: int
    warm_start_iterations: int
    sample_epsilon: float
    device: torch.device
    float_dtype: torch.dtype
    generator: torch.Generator | None
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
    child_mask: torch.Tensor
    feature_encoder: RebelFeatureEncoder | BetterFeatureEncoder
    hand_rank_data: HandRankData | None
    stats: dict[str, float]

    def __init__(
        self,
        search_batch_size: int,
        env_proto: HUNLTensorEnv,
        model: RebelFFN | BetterFFN | BetterTRM,
        bet_bins: list[float],
        max_depth: int,
        cfr_iterations: int,
        device: torch.device,
        float_dtype: torch.dtype,
        generator: torch.Generator | None = None,
        warm_start_iterations: int = T_WARM,
        num_supervisions: int = 1,
        cfr_type: CFRType = CFRType.linear,
        cfr_avg: bool = True,
        dcfr_alpha: float = 1.5,
        dcfr_beta: float = 0.0,
        dcfr_gamma: float = 2.0,
        dcfr_delay: int = 0,
        sample_epsilon: float = 0.25,
        value_targets_from_final_policy: bool = False,
    ):
        assert warm_start_iterations < cfr_iterations

        self.root_nodes = search_batch_size
        self.model = model
        self.max_depth = max_depth
        self.tree_depth = max_depth
        self.bet_bins = bet_bins
        self.cfr_iterations = cfr_iterations
        self.warm_start_iterations = max(0, warm_start_iterations)
        self.num_supervisions = num_supervisions
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
        self.use_final_policy_values = value_targets_from_final_policy

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
        self.legal_mask = torch.zeros(
            M, self.num_actions, dtype=torch.bool, device=self.device
        )
        self.child_mask = torch.zeros_like(self.legal_mask)
        self.child_count = torch.zeros(M, dtype=torch.long, device=self.device)
        self.showdown_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.showdown_actors = torch.empty(0, dtype=torch.long, device=self.device)
        self.showdown_potential = torch.empty(
            0, 2, dtype=self.float_dtype, device=self.device
        )

        # Set during initialize_subgame and not updated.
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
        self.uniform_policy = torch.zeros_like(self.policy_probs)
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
        if isinstance(self.model, (BetterFFN, BetterTRM)):
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
        self.profiler_output_dir = None

    @profile
    def initialize_subgame(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        """Copy root states into the search tree, reset per-node buffers, and expand.

        Args:
            src_env: Batched environment that holds the source root public states.
            src_indices: Row indices inside `src_env` to copy into the tree roots.
            initial_beliefs: Optional belief tensor aligned with `src_indices`.
        """
        assert src_indices.shape[0] == self.root_nodes
        assert src_indices.min().item() >= 0
        if initial_beliefs is not None:
            assert initial_beliefs.shape[0] == src_indices.shape[0]
            assert initial_beliefs.shape[1] == self.num_players
            assert initial_beliefs.shape[2] == NUM_HANDS

        N = self.root_nodes
        # Reset and zero tensors before calling base class
        dest_indices = torch.arange(N, device=self.device)
        self.env.reset()
        self.env.copy_state_from(src_env, src_indices, dest_indices, copy_deck=True)
        self.valid_mask.zero_()
        self.valid_mask[:N] = True
        self.leaf_mask.zero_()
        self.leaf_mask[:N] = self.env.done[:N]
        self.legal_mask.zero_()
        self.child_mask.zero_()
        self.child_count.zero_()
        self.showdown_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.showdown_actors = torch.empty(0, dtype=torch.long, device=self.device)
        self.showdown_potential = torch.empty(
            0, 2, dtype=self.float_dtype, device=self.device
        )
        self.folded_mask.zero_()
        self.folded_rewards.zero_()
        self.new_street_mask.zero_()
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
        self.policy_probs_sample.zero_()
        self.uniform_policy.zero_()
        self.cumulative_regrets.zero_()
        self.regret_weight_sums.zero_()
        self.allowed_hands.zero_()
        self.allowed_hands_prob.zero_()
        self.prev_actor.zero_()
        self.last_model_values = None
        self.latest_values.zero_()
        self.values_avg.zero_()
        self.beliefs.zero_()
        self.beliefs_avg.zero_()
        self.self_reach.zero_()
        self.self_reach_avg.zero_()
        self.hand_rank_data = None
        self.stats.clear()

        # Call base class to handle beliefs and common initialization
        super().initialize_subgame(src_env, src_indices, initial_beliefs)

        # Rebel-specific: mask allowed_hands with valid_mask
        self.allowed_hands &= self.valid_mask[:, None]
        self.allowed_hands_prob.masked_fill_((~self.valid_mask)[:, None], 0.0)

    @profile
    def _construct_subgame(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
    ) -> None:
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

        top = self.depth_offsets[-2]
        self.leaf_mask[top:] = self.valid_mask[top:]
        self.legal_mask = self.env.legal_bins_mask()
        self.child_mask[:top] = self._pull_back(self.valid_mask)
        self.child_count = self.child_mask.sum(dim=-1)
        valid_child_masks = self.child_mask[self.valid_mask & ~self.leaf_mask]
        has_legal = valid_child_masks.any(dim=-1)
        assert has_legal.all(), "Every valid node must have at least one legal action."
        legal_valid_nonleaf = (
            self.legal_mask & self.valid_mask[:, None] & ~self.leaf_mask[:, None]
        )
        assert (self.child_mask == legal_valid_nonleaf).all()
        assert (self.child_count[self.leaf_mask | ~self.valid_mask] == 0).all()
        assert (self.child_count[~self.leaf_mask & self.valid_mask] > 0).all()

        showdown_padding = max(1, self.root_nodes // 2)
        self.showdown_indices = padded_indices(self.env.street == 4, showdown_padding)
        self.showdown_actors = self.env.to_act[self.showdown_indices]
        self.showdown_potential = (
            self.env.stacks[self.showdown_indices]
            + self.env.pot[self.showdown_indices, None]
            - self.env.starting_stack
        )

        self.prev_actor[N:] = self._fan_out(self.env.to_act)

        # Compute uniform_policy based on legal_counts (number of siblings)
        child_count = self.child_mask.float().sum(dim=-1, keepdim=True)
        child_count_dest = self._fan_out(child_count)
        self.uniform_policy[:N].zero_()
        self.uniform_policy[N:] = torch.where(
            child_count_dest > 0, 1.0 / child_count_dest, 0.0
        )

    @profile
    def _mask_invalid(self, tensor: torch.Tensor) -> None:
        """Mask invalid nodes in the tensor."""
        if tensor.dim() == 2:
            # For 2D tensors like [M, 1326] (regrets, policy_probs)
            tensor.masked_fill_((~self.valid_mask)[:, None], 0.0)
        else:
            # For 3D+ tensors like [M, 2, 1326] (values, beliefs)
            tensor.masked_fill_((~self.valid_mask)[:, None, None], 0.0)

    def _propagate_level_beliefs(self, depth: int) -> None:
        """Propagate beliefs from all nodes at a given level to all nodes at the next level."""
        offset_next = self.depth_offsets[depth + 1]
        offset_next_next = self.depth_offsets[depth + 2]

        probs = self.policy_probs[offset_next:offset_next_next]
        self.beliefs[offset_next:offset_next_next] = self._fan_out(
            self.beliefs, level=depth
        )

        indices = torch.arange(offset_next, offset_next_next, device=self.device)
        self.beliefs[indices, self.prev_actor[offset_next:offset_next_next]] *= probs

    @profile
    def sample_leaves(self, training_mode: bool) -> None:
        """Sample leaves from `self.policy_probs_sample`."""

        N, B = self.root_nodes, self.num_actions
        top = self.depth_offsets[-2]

        players = torch.randint(
            0, 2, (N,), generator=self.generator, device=self.device
        )
        sample_epsilon = self.sample_epsilon if training_mode else 0.0

        # Don't sample nodes that are done. No point in continuing search from there.
        # This means we need a separate valid child mask for sampling.
        done_src = self._pull_back(self.env.done)
        sampling_masks = self.child_mask.clone()
        sampling_masks[:top] &= ~done_src
        sampling_counts = sampling_masks.float().sum(dim=-1, keepdim=True)

        # Calculate uniform sampling probabilities as backup.
        uniform = torch.where(sampling_masks, 1 / sampling_counts, 0)

        # Calculate policy sampling probabilities, excluding done nodes.
        policy_probs_by_src = self._pull_back(self.policy_probs_sample).clone()
        policy_probs_by_src.masked_fill_(done_src[:, :, None], 0.0)
        denom = policy_probs_by_src.sum(dim=1, keepdim=True)
        policy_probs_by_src = torch.where(
            denom >= 1e-12,
            policy_probs_by_src / denom,
            uniform[:top, :, None],
        )

        # If a node has no legal actions after filtering done nodes, it's a leaf.
        effective_leaf_mask = self.leaf_mask | (sampling_counts.squeeze(1) == 0)

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

        assert effective_leaf_mask[sampled_nodes].all()
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

    def _pull_back(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Pull back data to all parent nodes.
        Args:
            data: Data to pull back, organized by destination node, shape [M, ...].
            level: Depth level to pull back from, or None for all levels.
        Returns:
            Data by source node/action, shape [M - N * B ** D, B, *data.shape[1:]].
        """
        if level is None:
            bottom = self.depth_offsets[1]
            return data[bottom:].view(-1, self.num_actions, *data.shape[1:])
        else:
            # Slice internally to that level's children
            offset_next = self.depth_offsets[level + 1]
            offset_next_next = self.depth_offsets[level + 2]
            return data[offset_next:offset_next_next].view(
                -1, self.num_actions, *data.shape[1:]
            )

    def _pull_back_sum(
        self, tensor: torch.Tensor, out: torch.Tensor, level: int | None = None
    ) -> None:
        """Pull back tensor and sum into output tensor.

        For dense CFR: do pull-back then sum over actions.
        """

        if level is None:
            start = self.depth_offsets[1]
            end = self.total_nodes
        else:
            start = self.depth_offsets[level + 1]
            end = self.depth_offsets[level + 2]
        expected = end - start

        if tensor.shape[0] == self.total_nodes:
            sliced_tensor = tensor[start:end]
        elif tensor.shape[0] == expected:
            sliced_tensor = tensor
        else:
            raise ValueError(
                f"Tensor length {tensor.shape[0]} does not match expected slice {expected}"
            )

        pulled = sliced_tensor.view(-1, self.num_actions, *tensor.shape[1:])

        # Sum over actions (dim=1) and write to output
        if level is None:
            top = self.depth_offsets[-2]
            out[:top] += pulled.sum(dim=1)
        else:
            offset = self.depth_offsets[level]
            offset_next = self.depth_offsets[level + 1]
            out[offset:offset_next] += pulled.sum(dim=1)

    def _push_down(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Push down data to all child nodes.
        Args:
            data: Data to push down, shape [M, B, ...].
            level: Depth level to push down from, or None for all levels.
        Returns:
            Data by child node, shape [M - N, ...].
        """
        assert data.shape[1] == self.num_actions
        if level is None:
            top = self.depth_offsets[-2]
            return data[:top, None].reshape(-1, *data.shape[2:])
        else:
            # Slice internally to that level's parents
            offset = self.depth_offsets[level]
            offset_next = self.depth_offsets[level + 1]
            return data[offset:offset_next, None].reshape(-1, *data.shape[2:])

    def _fan_out(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Fanout data to all children nodes.

        Args:
            data: Data to fanout.
            level: Depth level to fan out from, or None for all levels.
        Returns:
            Fanout data, shape [M - N, *data.shape[1:]].
        """
        if level is None:
            top = self.depth_offsets[-2]
            data_sliced = data[:top, None]
        else:
            # Slice internally to that level's parents
            offset = self.depth_offsets[level]
            offset_next = self.depth_offsets[level + 1]
            data_sliced = data[offset:offset_next, None]
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
            output[offset_next:offset_next_next] = self._fan_out(output, level=depth)
        return output
