from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from alphaholdem.core.structured_config import Config
from alphaholdem.env.card_utils import NUM_HANDS, combo_to_onehot_tensor
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.search.cfr_evaluator import CFREvaluator, PublicBeliefState
from alphaholdem.search.chance_node_helper import ChanceNodeHelper
from alphaholdem.utils.profiling import profile


class SparseCFREvaluator(CFREvaluator):
    def __init__(
        self, model: RebelFFN | BetterFFN, device: torch.device, cfg: Config
    ) -> None:
        self.model = model
        self.device = device
        self.cfg = cfg

        self.float_dtype = torch.float32
        self.num_players = 2
        self.bet_bins = cfg.env.bet_bins
        self.num_actions = len(self.bet_bins) + 3

        search_cfg = cfg.search
        train_cfg = cfg.train

        self.max_depth = search_cfg.depth
        self.cfr_iterations = search_cfg.iterations
        self.warm_start_iterations = max(
            0, min(search_cfg.warm_start_iterations, max(1, self.cfr_iterations - 1))
        )
        self.cfr_type = search_cfg.cfr_type
        self.cfr_avg = search_cfg.cfr_avg
        self.dcfr_alpha = search_cfg.dcfr_alpha
        self.dcfr_beta = search_cfg.dcfr_beta
        self.dcfr_gamma = search_cfg.dcfr_gamma
        self.dcfr_delay = getattr(search_cfg, "dcfr_plus_delay", 0)
        self.sample_epsilon = getattr(train_cfg, "cfr_action_epsilon", 0.0)

        self.generator = torch.Generator(device=self.device)
        self.combo_onehot_float = combo_to_onehot_tensor(device=self.device).float()
        self.chance_helper = ChanceNodeHelper(
            device=self.device,
            float_dtype=self.float_dtype,
            num_players=self.num_players,
            model=self.model,
        )
        self.stats: dict[str, float] = {}

        # PyTorch profiler setup
        self.profiler_enabled = False
        self.profiler = None
        self.profiler_output_dir = None

        self.total_nodes = 0
        self.root_nodes = 0
        self.depth_offsets = [0]
        self.env: Optional[HUNLTensorEnv] = None

        self.leaf_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.new_street_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.valid_mask = torch.empty(0, dtype=torch.bool, device=self.device)

        self.legal_mask = torch.empty(
            0, self.num_actions, dtype=torch.bool, device=self.device
        )
        self.allowed_hands = torch.empty(
            0, NUM_HANDS, dtype=torch.bool, device=self.device
        )
        self.allowed_hands_prob = torch.empty(
            0, NUM_HANDS, dtype=self.float_dtype, device=self.device
        )

        self.parent_index = torch.empty(0, dtype=torch.long, device=self.device)
        self.action_from_parent = torch.empty(0, dtype=torch.long, device=self.device)
        self.child_count = torch.empty(0, dtype=torch.long, device=self.device)
        self.child_offsets = torch.empty(0, dtype=torch.long, device=self.device)
        self.prev_actor = torch.empty(0, dtype=torch.long, device=self.device)

        self.beliefs = torch.empty(
            0, self.num_players, NUM_HANDS, dtype=self.float_dtype, device=self.device
        )
        self.beliefs_avg = torch.empty_like(self.beliefs)
        self.root_pre_chance_beliefs = torch.empty(
            0, self.num_players, NUM_HANDS, dtype=self.float_dtype, device=self.device
        )
        self.latest_values = torch.empty(
            0, self.num_players, NUM_HANDS, dtype=self.float_dtype, device=self.device
        )
        self.values_avg = torch.empty_like(self.latest_values)
        self.self_reach = torch.empty_like(self.beliefs)
        self.self_reach_avg = torch.empty_like(self.beliefs)
        self.last_model_values: Optional[torch.Tensor] = None

        self.policy_probs = torch.empty(
            0, NUM_HANDS, dtype=self.float_dtype, device=self.device
        )
        self.policy_probs_avg = torch.empty_like(self.policy_probs)
        self.policy_probs_sample = torch.empty_like(self.policy_probs)
        self.cumulative_regrets = torch.empty_like(self.policy_probs)
        self.regret_weight_sums = torch.empty_like(self.policy_probs)

        self.feature_encoder: Optional[RebelFeatureEncoder | BetterFeatureEncoder] = (
            None
        )

        self.sampled_leaf_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.t_sample = torch.empty(0, dtype=torch.long, device=self.device)
        self.sample_count = 0
        self.next_pbs = None
        self.next_pbs_idx = 0

    @profile
    def initialize_subgame(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        assert src_indices.dim() == 1, "src_indices must be 1-D"
        num_roots = src_indices.shape[0]
        assert num_roots > 0, "must supply at least one root state"

        root_env = HUNLTensorEnv.from_proto(src_env, num_envs=num_roots)
        root_dest = torch.arange(num_roots, device=self.device)
        root_env.copy_state_from(src_env, src_indices, root_dest)

        env_levels: list[HUNLTensorEnv] = [root_env]
        parent_index_levels = [
            torch.full((num_roots,), -1, dtype=torch.long, device=self.device)
        ]
        action_levels = [
            torch.full((num_roots,), -1, dtype=torch.long, device=self.device)
        ]
        reward_levels = [
            torch.zeros(num_roots, dtype=self.float_dtype, device=self.device)
        ]

        self.depth_offsets = [0, num_roots]
        depth = 0
        while depth < self.max_depth:
            parent_start, parent_end = self.depth_offsets[-2], self.depth_offsets[-1]
            parent_env = env_levels[-1]
            legal_mask = parent_env.legal_bins_mask()
            done_mask = parent_env.done
            legal_mask = legal_mask & (~done_mask)[:, None]
            stop_mask = parent_env.actions_this_round == 0
            if depth > 0:
                legal_mask &= (~stop_mask)[:, None]
            child_counts = legal_mask.sum(dim=-1)
            action_bins = torch.where(legal_mask)[1]
            child_count = action_bins.numel()
            if child_count == 0:
                break

            env_next = parent_env.repeat_interleave(
                child_counts, output_size=child_count
            )

            parent_indices_level = torch.arange(
                parent_start, parent_end, device=self.device
            ).repeat_interleave(child_counts, output_size=child_count)

            rewards, _, _ = env_next.step_bins(action_bins)
            env_levels.append(env_next)
            parent_index_levels.append(parent_indices_level)
            action_levels.append(action_bins)
            reward_levels.append(rewards.to(self.float_dtype))

            depth += 1
            self.depth_offsets.append(self.depth_offsets[-1] + env_next.N)

        self.total_nodes = self.depth_offsets[-1]
        self.root_nodes = num_roots
        self.top_nodes = (
            self.depth_offsets[-2] if len(self.depth_offsets) > 1 else num_roots
        )

        # Initialize valid_mask as all ones (all nodes are valid in sparse structure)
        self.valid_mask = torch.ones(
            self.total_nodes, dtype=torch.bool, device=self.device
        )

        self.env = HUNLTensorEnv.from_proto(env_levels[-1], num_envs=self.total_nodes)
        cursor = 0
        for level_env in env_levels:
            count = level_env.N
            src_idx = torch.arange(count, device=self.device)
            dst_idx = torch.arange(cursor, cursor + count, device=self.device)
            self.env.copy_state_from(level_env, src_idx, dst_idx)
            cursor += count

        self.parent_index = torch.cat(parent_index_levels)
        self.action_from_parent = torch.cat(action_levels)
        rewards_tensor = torch.cat(reward_levels)

        self.legal_mask = self.env.legal_bins_mask()

        root_mask = torch.zeros(self.total_nodes, dtype=torch.bool, device=self.device)
        root_mask[:num_roots] = True
        self.new_street_mask = (self.env.actions_this_round == 0) & ~root_mask
        self.leaf_mask = self.env.done | self.new_street_mask
        self.child_mask = self.legal_mask & (~self.leaf_mask)[:, None]

        self.child_count = torch.zeros(
            self.total_nodes, dtype=torch.long, device=self.device
        )

        bottom = self.depth_offsets[1]
        parents = self.parent_index[bottom:]
        self.child_count.scatter_add_(
            0, parents, torch.ones_like(parents, dtype=torch.long)
        )
        self.child_offsets = bottom + torch.cumsum(self.child_count, dim=0)

        self.showdown_indices = torch.where(self.env.street == 4)[0]
        self.showdown_actors = self.env.to_act[self.showdown_indices]

        self.prev_actor = torch.full(
            (self.total_nodes,), -1, dtype=torch.long, device=self.device
        )
        self.prev_actor[self.root_nodes :] = self.env.to_act[
            self.parent_index[self.root_nodes :]
        ]

        self.policy_probs = torch.zeros(
            self.total_nodes, NUM_HANDS, dtype=self.float_dtype, device=self.device
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
        self.policy_probs_sample = torch.zeros_like(self.policy_probs)
        self.cumulative_regrets = torch.zeros_like(self.policy_probs)
        self.regret_weight_sums = torch.zeros_like(self.policy_probs)

        child_count_dest = self._fan_out(self.child_count)
        self.uniform_policy = torch.zeros_like(self.policy_probs)
        self.uniform_policy[self.root_nodes :] = (1.0 / child_count_dest)[
            :, None
        ].expand(-1, NUM_HANDS)

        self.beliefs = torch.zeros(
            self.total_nodes,
            self.num_players,
            NUM_HANDS,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.beliefs_avg = torch.zeros_like(self.beliefs)
        self.root_pre_chance_beliefs = torch.zeros(
            num_roots,
            self.num_players,
            NUM_HANDS,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.self_reach = torch.zeros_like(self.beliefs)
        self.self_reach[: self.root_nodes] = 1.0
        self.self_reach_avg = self.self_reach.clone()

        # Showdown and leaf values are set in set_leaf_values, as they vary by beliefs.
        self.latest_values = torch.zeros_like(self.beliefs)
        folded_mask = (self.action_from_parent == 0) & self.env.done
        self.latest_values[folded_mask, 0] = rewards_tensor[folded_mask][:, None]
        self.latest_values[folded_mask, 1] = -rewards_tensor[folded_mask][:, None]
        self.values_avg = self.latest_values.clone()
        self.last_model_values = None

        if initial_beliefs is None:
            initial_beliefs = torch.full(
                (num_roots, self.num_players, NUM_HANDS),
                1.0 / NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
        else:
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        self.beliefs[:num_roots] = initial_beliefs
        self.beliefs_avg[:num_roots] = initial_beliefs
        self.root_pre_chance_beliefs[:] = initial_beliefs
        self.self_reach[:num_roots] = 1.0
        self.self_reach_avg[:num_roots] = 1.0

        board_mask_root = (
            self.env.board_onehot[:num_roots].any(dim=1).reshape(num_roots, -1).float()
        )
        root_allowed = (self.combo_onehot_float @ board_mask_root.T).T < 0.5
        root_allowed_prob = root_allowed.float()
        root_allowed_prob /= root_allowed_prob.sum(dim=-1, keepdim=True).clamp(min=1.0)

        self.allowed_hands = self._fan_out_deep(root_allowed)
        self.allowed_hands_prob = self._fan_out_deep(root_allowed_prob)

        if isinstance(self.model, BetterFFN):
            self.feature_encoder = BetterFeatureEncoder(
                env=self.env, device=self.device, dtype=self.float_dtype
            )
        else:
            self.feature_encoder = RebelFeatureEncoder(
                env=self.env, device=self.device, dtype=self.float_dtype
            )

    def _mask_invalid(self, tensor: torch.Tensor) -> None:
        """Mask invalid nodes in the tensor. Noop for sparse evaluator (all nodes are valid)."""
        pass

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

    def sample_leaves(self, training_mode: bool) -> PublicBeliefState:
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

            child_offsets = self.child_offsets[active_nodes]
            sampled_nodes[active_mask] = child_offsets + actions

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

    def _fan_out(self, tensor: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Fan out tensor to all children nodes."""
        if level is None:
            start = 0
            end = self.depth_offsets[-2]
            size = self.total_nodes - self.depth_offsets[1]
        else:
            start = self.depth_offsets[level]
            end = self.depth_offsets[level + 1]
            next = self.depth_offsets[level + 2]
            size = next - end

        repeats = self.child_count[start:end]
        # Have to clone to work around PyTorch bug on MPS.
        if self.device.type == "mps":
            repeats = repeats.clone()

        return tensor[start:end].repeat_interleave(repeats, dim=0, output_size=size)

    def _push_down(
        self, tensor: torch.Tensor, level: int | None = None
    ) -> torch.Tensor:
        """Push down tensor of shape [K, B, ...] to all child nodes."""
        if level is None:
            start = self.depth_offsets[1]
            end = self.total_nodes
        else:
            start = self.depth_offsets[level + 1]
            end = self.depth_offsets[level + 2]

        parent_indices = self.parent_index[start:end]  # [num_children]
        actions = self.action_from_parent[start:end]  # [num_children]

        # Gather values for the children from the appropriate slice of the input tensor.
        # policy_probs[parent_idx, action, ...] => prob[child_idx, ...]
        return tensor[parent_indices, actions]

    def _pull_back(
        self, tensor: torch.Tensor, level: int | None = None
    ) -> torch.Tensor:
        """Pull back child-aligned tensor to per-parent action slices."""
        if level is None:
            parent_start = self.depth_offsets[0]
            parent_end = self.depth_offsets[-2]
            child_start = self.depth_offsets[1]
            child_end = self.total_nodes
        else:
            parent_start = self.depth_offsets[level]
            parent_end = self.depth_offsets[level + 1]
            child_start = self.depth_offsets[level + 1]
            child_end = self.depth_offsets[level + 2]

        out = torch.zeros(
            (parent_end - parent_start, self.num_actions, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=self.device,
        )

        parent_indices = self.parent_index[child_start:child_end]
        actions = self.action_from_parent[child_start:child_end]
        out[parent_indices - parent_start, actions] = tensor[child_start:child_end]
        return out

    def _pull_back_sum(
        self, tensor: torch.Tensor, out: torch.Tensor, level: int | None = None
    ) -> None:
        """Pull back tensor of shape [K, B, ...] to all parent nodes."""
        if level is None:
            start = self.depth_offsets[1]
            end = self.total_nodes
        else:
            start = self.depth_offsets[level + 1]
            end = self.depth_offsets[level + 2]

        parent_indices = self.parent_index[start:end]
        expected = end - start
        if tensor.shape[0] == self.total_nodes:
            sliced_tensor = tensor[start:end]
        elif tensor.shape[0] == expected:
            sliced_tensor = tensor
        else:
            raise ValueError(
                f"Tensor length {tensor.shape[0]} does not match expected slice {expected}"
            )

        # Sum values for the children from the appropriate slice of the input tensor.
        # policy_probs[parent_idx, action, ...] => prob[child_idx, ...]
        out.scatter_reduce_(
            dim=0,
            index=parent_indices[(...,) + (None,) * (tensor.dim() - 1)].expand(
                -1, *sliced_tensor.shape[1:]
            ),
            src=sliced_tensor,
            reduce="sum",
            include_self=True,
        )

    def _fan_out_deep(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fan out a root-aligned tensor across every node in the tree."""
        output = torch.zeros(
            (self.total_nodes, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=self.device,
        )
        output[: self.root_nodes] = tensor
        for depth in range(self.max_depth):
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]
            output[offset_next:offset_next_next] = self._fan_out(output, depth)

        return output
