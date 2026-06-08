import torch

from p2.core.action_schedule import (
    ActionSchedule,
    legal_action_mask_for_depth,
    legal_action_masks_by_depth,
    make_action_schedule,
)
from p2.core.structured_config import CFRType, Config
from p2.env.card_utils import NUM_HANDS, combo_to_onehot_tensor
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.models.base_mlp_model import BaseMLPModel
from p2.models.mlp.better_feature_encoder import BetterFeatureEncoder
from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from p2.search.cfr_evaluator import (
    CFREvaluator,
    HandRankData,
    PublicBeliefState,
    STREETS,
    padded_indices,
)
from p2.search.chance_node_helper import ChanceNodeHelper
from p2.utils.profiling import profile


class SparseCFREvaluator(CFREvaluator):
    def __init__(
        self,
        model: BaseMLPModel,
        device: torch.device,
        cfg: Config,
        generator: torch.Generator | None = None,
        closing_leaf_model: BaseMLPModel | None = None,
    ) -> None:
        self.model = model
        self.policy_model = getattr(model, "policy_model", model)
        self.value_model = getattr(model, "value_model", model)
        self.closing_leaf_value_model = (
            getattr(closing_leaf_model, "value_model", closing_leaf_model)
            if closing_leaf_model is not None
            else None
        )
        self.device = device
        self.cfg = cfg

        self.float_dtype = torch.float32
        self.num_players = int(cfg.env.num_players)
        self.action_schedule: ActionSchedule = make_action_schedule(
            cfg.env.bet_bins,
            cfg.search.bet_bins_by_depth,
            cfg.search.allin_by_depth,
        )
        cfg.env.bet_bins = list(self.action_schedule.bet_bins)
        cfg.model.num_actions = self.action_schedule.num_actions
        if self.action_schedule.num_depths is not None:
            cfg.search.depth = self.action_schedule.num_depths
        self.bet_bins = list(self.action_schedule.bet_bins)
        self.num_actions = self.action_schedule.num_actions
        self.action_masks_by_depth = legal_action_masks_by_depth(
            self.action_schedule, device=self.device
        )

        self.num_supervisions = cfg.model.num_supervisions

        self.max_depth = cfg.search.depth
        self.cfr_iterations = cfg.search.iterations
        self.warm_start_iterations = max(
            0, min(cfg.search.warm_start_iterations, max(1, self.cfr_iterations - 1))
        )
        self.warm_start_type = cfg.search.warm_start_type
        self.warm_start_multiplier = cfg.search.warm_start_multiplier
        self.warm_start_regret_multiplier = (
            cfg.search.warm_start_regret_multiplier
            if cfg.search.warm_start_regret_multiplier is not None
            else cfg.search.warm_start_multiplier
        )
        exact_sparse = type(self) is SparseCFREvaluator
        self._warm_start_policy_prior = None
        self._warm_start_prior_tau = None
        self._warm_start_prior_start_t = 0
        self._warm_start_prior_horizon = 0
        self._warm_start_regrets = None
        self._warm_start_regret_decay = (
            cfg.search.warm_start_regret_decay if exact_sparse else "none"
        )
        self._warm_start_regret_decay_horizon = cfg.search.warm_start_regret_decay_horizon
        self._warm_start_regret_decay_floor = cfg.search.warm_start_regret_decay_floor
        self._warm_start_regret_start_t = 0
        self._warm_start_ftrl_enabled = exact_sparse
        self._warm_start_ftrl_mode = cfg.search.warm_start_ftrl_mode
        self._warm_start_ftrl_tau_scale = cfg.search.warm_start_ftrl_tau_scale
        self._warm_start_ftrl_horizon = cfg.search.warm_start_ftrl_horizon
        self._warm_start_ftrl_floor = cfg.search.warm_start_ftrl_floor
        self.cfr_type = cfg.search.cfr_type
        self.sapcfr_alpha = cfg.search.sapcfr_alpha
        self.predictive_cfr_delay = cfg.search.predictive_cfr_delay
        self._predictive_cfr_dcfr_hybrid = cfg.search.predictive_cfr_dcfr_hybrid
        self._predictive_cfr_enabled = exact_sparse and self.cfr_type in (
            CFRType.pcfr,
            CFRType.sapcfr,
        )
        self.cfr_avg = cfg.search.cfr_avg
        self.cfr_plus = cfg.search.cfr_plus
        self.dcfr_alpha = cfg.search.dcfr_alpha
        self.dcfr_beta = cfg.search.dcfr_beta
        self.dcfr_gamma = cfg.search.dcfr_gamma
        self.dcfr_alpha_initial = cfg.search.dcfr_alpha
        self.dcfr_beta_initial = cfg.search.dcfr_beta
        self.dcfr_gamma_initial = cfg.search.dcfr_gamma
        self.dcfr_alpha_final = cfg.search.dcfr_alpha_final
        self.dcfr_beta_final = cfg.search.dcfr_beta_final
        self.dcfr_gamma_final = cfg.search.dcfr_gamma_final
        self.dcfr_delay = cfg.search.dcfr_plus_delay
        self.sample_epsilon = cfg.search.sample_epsilon
        self.use_final_policy_values = cfg.search.value_targets_from_final_policy

        self.generator = generator or torch.Generator(device=self.device).manual_seed(
            cfg.seed
        )
        self.combo_onehot_float = combo_to_onehot_tensor(device=self.device).float()
        self.chance_helper = ChanceNodeHelper(
            device=self.device,
            float_dtype=self.float_dtype,
            num_players=self.num_players,
            model=self.value_model,
            generator=self.generator,
        )
        self.stats: dict[str, float] = {}
        self.hand_rank_data: HandRankData | None = None

        # PyTorch profiler setup
        self.profiler_enabled = False
        self.profiler = None
        self.profiler_output_dir = None

        self.total_nodes = 0
        self.tree_depth = 0
        self.root_nodes = cfg.num_envs
        self.depth_offsets = [0]
        self.env: HUNLTensorEnv | PBSEnv | None = None
        policy_model = getattr(model, "policy_model", model)
        self.hand_dim = int(getattr(policy_model, "hand_dim", NUM_HANDS))

        self.leaf_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.new_street_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.valid_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.model_indices = torch.empty(0, dtype=torch.long, device=self.device)

        self.legal_mask = torch.empty(
            0, self.num_actions, dtype=torch.bool, device=self.device
        )
        self.allowed_hands = torch.empty(
            0, self.hand_dim, dtype=torch.bool, device=self.device
        )
        self.allowed_hands_prob = torch.empty(
            0, self.hand_dim, dtype=self.float_dtype, device=self.device
        )

        self.parent_index = torch.empty(0, dtype=torch.long, device=self.device)
        self.action_from_parent = torch.empty(0, dtype=torch.long, device=self.device)
        self.child_count = torch.empty(0, dtype=torch.long, device=self.device)
        self.child_offsets = torch.empty(0, dtype=torch.long, device=self.device)
        self.prev_actor = torch.empty(0, dtype=torch.long, device=self.device)

        self.beliefs = torch.empty(
            0,
            self.num_players,
            self.hand_dim,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.beliefs_avg = torch.empty_like(self.beliefs)
        self.beliefs_sample = torch.empty_like(self.beliefs)
        self.root_pre_chance_beliefs = torch.empty(
            0,
            self.num_players,
            self.hand_dim,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.latest_values = torch.empty(
            0,
            self.num_players,
            self.hand_dim,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.values_avg = torch.empty_like(self.latest_values)
        self.self_reach = torch.empty_like(self.beliefs)
        self.self_reach_avg = torch.empty_like(self.beliefs)
        self.last_model_values: torch.Tensor | None = None

        self.latent_y = torch.empty(0, self.model.hidden_dim, device=self.device)
        self.latent_z = torch.empty(0, self.model.hidden_dim, device=self.device)

        self.policy_probs = torch.empty(
            0, self.hand_dim, dtype=self.float_dtype, device=self.device
        )
        self.policy_probs_avg = torch.empty_like(self.policy_probs)
        self.average_policy_numerator = torch.empty_like(self.policy_probs)
        self.average_policy_denominator = torch.empty_like(self.policy_probs)
        self.average_policy_initialized = False
        self.policy_probs_sample = torch.empty_like(self.policy_probs)
        self.cumulative_regrets = torch.empty_like(self.policy_probs)

        self.feature_encoder: RebelFeatureEncoder | BetterFeatureEncoder | None = None
        self.policy_feature_encoder: (
            RebelFeatureEncoder | BetterFeatureEncoder | None
        ) = None
        self.value_feature_encoder: (
            RebelFeatureEncoder | BetterFeatureEncoder | None
        ) = None

        self.sampled_leaf_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.continuation_value_target_indices = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        self.t_sample = torch.empty(0, dtype=torch.long, device=self.device)
        self.sample_count = 0
        self.next_pbs: PublicBeliefState | None = None
        self.next_pbs_idx = 0
        self.allin_payoff_resolver = None
        self.allin_call_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.allin_call_parent_indices = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        self.allin_call_mask = torch.empty(0, dtype=torch.bool, device=self.device)

    @profile
    def _construct_subgame(
        self,
        src_env: HUNLTensorEnv | PBSEnv,
        src_indices: torch.Tensor,
    ) -> None:
        """Construct the sparse subgame tree structure."""
        assert src_indices.dim() == 1, "src_indices must be 1-D"
        num_roots = src_indices.shape[0]
        assert num_roots > 0, "must supply at least one root state"

        env_cls = type(src_env)
        root_env = env_cls.from_proto(src_env, num_envs=num_roots)
        root_dest = torch.arange(num_roots, device=self.device)
        root_env.copy_state_from(src_env, src_indices, root_dest)

        env_levels: list[HUNLTensorEnv | PBSEnv] = [root_env]
        parent_index_levels = [
            torch.full((num_roots,), -1, dtype=torch.long, device=self.device)
        ]
        action_levels = [
            torch.full((num_roots,), -1, dtype=torch.long, device=self.device)
        ]
        vector_rewards = isinstance(src_env, PBSEnv)
        reward_shape = (num_roots, self.num_players) if vector_rewards else (num_roots,)
        reward_levels = [
            torch.zeros(reward_shape, dtype=self.float_dtype, device=self.device)
        ]
        allin_leaf_levels = [
            torch.zeros(num_roots, dtype=torch.bool, device=self.device)
        ]

        self.depth_offsets = [0, num_roots]
        depth = 0
        while depth < self.max_depth:
            parent_start = self.depth_offsets[-2]
            parent_env = env_levels[-1]
            legal_mask = parent_env.legal_bins_mask()
            legal_mask &= self._action_mask_for_depth(depth)[None, :]
            legal_mask &= (~allin_leaf_levels[-1])[:, None]
            stop_mask = parent_env.actions_this_round == 0
            if depth > 0:
                legal_mask &= (~stop_mask)[:, None]
            child_counts = legal_mask.sum(dim=-1)
            action_bins = torch.where(legal_mask)[1]
            child_count = action_bins.numel()
            if child_count == 0:
                break

            parent_local_indices = torch.arange(
                parent_env.N, device=self.device
            ).repeat_interleave(child_counts, output_size=child_count)
            env_next = parent_env.gather_rows(parent_local_indices)

            parent_indices_level = parent_local_indices + parent_start

            allin_leaf_level = self._allin_call_child_mask(
                parent_env,
                parent_local_indices,
                action_bins,
            )
            rewards, _, _ = env_next.step_bins(action_bins)
            env_levels.append(env_next)
            parent_index_levels.append(parent_indices_level)
            action_levels.append(action_bins)
            reward_levels.append(rewards.to(self.float_dtype))
            allin_leaf_levels.append(allin_leaf_level)

            depth += 1
            self.depth_offsets.append(self.depth_offsets[-1] + env_next.N)

        self.tree_depth = max(0, len(self.depth_offsets) - 2)
        self.total_nodes = self.depth_offsets[-1]
        self.root_nodes = num_roots
        self.top_nodes = (
            self.depth_offsets[-2] if len(self.depth_offsets) > 1 else num_roots
        )

        # Initialize valid_mask as all ones (all nodes are valid in sparse structure)
        # This should not be used anywhere outside of tests.
        self.valid_mask = torch.ones(
            self.total_nodes, dtype=torch.bool, device=self.device
        )

        self.env = env_cls.from_proto(env_levels[-1], num_envs=self.total_nodes)
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
        allin_leaf_tensor = torch.cat(allin_leaf_levels)

        self.legal_mask = self.env.legal_bins_mask()
        self._apply_depth_action_masks(self.legal_mask)

        root_mask = torch.zeros(self.total_nodes, dtype=torch.bool, device=self.device)
        root_mask[:num_roots] = True
        self.new_street_mask = (self.env.actions_this_round == 0) & ~root_mask
        self.leaf_mask = self.env.done | self.new_street_mask
        self.leaf_mask[self.depth_offsets[-2] :] = True
        self.leaf_mask |= allin_leaf_tensor
        self.new_street_mask.masked_fill_(allin_leaf_tensor, False)
        self._mark_allin_call_leaves()
        self.model_indices = self._compute_model_indices()
        self._validate_model_leaf_phases()
        self.child_mask = (
            self.legal_mask & self.valid_mask[:, None] & (~self.leaf_mask)[:, None]
        )
        self.child_count = self.child_mask.sum(dim=-1)
        assert self.valid_mask.sum() == self.root_nodes + self.child_count.sum()
        assert (self.child_count[self.leaf_mask] == 0).all()
        assert (self.child_count[~self.leaf_mask] > 0).all()
        assert (~self.leaf_mask[: self.root_nodes]).all()

        bottom = self.depth_offsets[1]
        self.child_offsets = (
            bottom + torch.cumsum(self.child_count, dim=0) - self.child_count
        )

        showdown_padding = max(1, self.root_nodes // 2)
        self.showdown_indices = padded_indices(self.env.street == 4, showdown_padding)
        self.showdown_actors = self.env.to_act[self.showdown_indices]
        # Compute potential for both players: [M, 2]
        self.showdown_potential = (
            self.env.stacks[self.showdown_indices]
            + self.env.pot[self.showdown_indices, None]
            - self.env.starting_stacks[self.showdown_indices]
        )

        self.prev_actor = torch.full(
            (self.total_nodes,), -1, dtype=torch.long, device=self.device
        )
        self.prev_actor[self.root_nodes :] = self.env.to_act[
            self.parent_index[self.root_nodes :]
        ]

        self.policy_probs = torch.zeros(
            self.total_nodes, self.hand_dim, dtype=self.float_dtype, device=self.device
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
        self.average_policy_numerator = torch.zeros_like(self.policy_probs)
        self.average_policy_denominator = torch.zeros_like(self.policy_probs)
        self.average_policy_initialized = False
        self.policy_probs_sample = torch.zeros_like(self.policy_probs)
        self.cumulative_regrets = torch.zeros_like(self.policy_probs)
        self._last_instantaneous_regrets = torch.zeros_like(self.policy_probs)

        # `parent_index[root_nodes:]` is already the child-to-parent expansion
        # needed here; using it avoids another repeat_interleave during subgame
        # construction.
        child_count_dest = self.child_count[self.parent_index[self.root_nodes :]]
        self.uniform_policy = torch.zeros_like(self.policy_probs)
        self.uniform_policy[self.root_nodes :] = (1.0 / child_count_dest)[
            :, None
        ].expand(-1, self.hand_dim)

        self.beliefs = torch.zeros(
            self.total_nodes,
            self.num_players,
            self.hand_dim,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.beliefs_avg = torch.zeros_like(self.beliefs)
        self.beliefs_sample = torch.zeros_like(self.beliefs)
        self.root_pre_chance_beliefs = torch.zeros(
            num_roots,
            self.num_players,
            self.hand_dim,
            dtype=self.float_dtype,
            device=self.device,
        )
        self.self_reach = torch.zeros_like(self.beliefs)
        self.self_reach[: self.root_nodes] = 1.0
        self.self_reach_avg = self.self_reach.clone()

        # Showdown and leaf values are set in set_leaf_values, as they vary by beliefs.
        self.latest_values = torch.zeros_like(self.beliefs)
        folded_mask = (self.action_from_parent == 0) & self.env.done
        if rewards_tensor.dim() == 2:
            self.latest_values[folded_mask] = rewards_tensor[folded_mask, :, None]
        else:
            self.latest_values[folded_mask, 0] = rewards_tensor[folded_mask][:, None]
            self.latest_values[folded_mask, 1] = -rewards_tensor[folded_mask][:, None]
        self.values_avg = self.latest_values.clone()
        self.last_model_values = None

        # Set up feature encoder (sparse-specific, done after tree construction)
        self.policy_feature_encoder = self.policy_model.create_feature_encoder(
            env=self.env, device=self.device, dtype=self.float_dtype
        )
        self.value_feature_encoder = self.value_model.create_feature_encoder(
            env=self.env, device=self.device, dtype=self.float_dtype
        )
        self.feature_encoder = self.policy_feature_encoder

    def _action_mask_for_depth(self, depth: int) -> torch.Tensor:
        if self.action_masks_by_depth.shape[0] == 1:
            return self.action_masks_by_depth[0]
        return legal_action_mask_for_depth(
            self.action_schedule,
            depth,
            device=self.device,
        )

    def _apply_depth_action_masks(self, legal_mask: torch.Tensor) -> None:
        if self.action_schedule.bet_bins_by_depth is None:
            return
        for depth in range(min(self.max_depth, len(self.depth_offsets) - 1)):
            start = self.depth_offsets[depth]
            end = self.depth_offsets[depth + 1]
            legal_mask[start:end] &= self._action_mask_for_depth(depth)[None, :]

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

        N = self.root_nodes
        top = self.depth_offsets[-2]
        self.continuation_value_target_indices = torch.empty(
            0, dtype=torch.long, device=self.device
        )

        players = torch.randint(
            0,
            self.num_players,
            (N,),
            generator=self.generator,
            device=self.device,
        )
        sample_epsilon = self.sample_epsilon if training_mode else 0.0

        # Don't sample nodes that are done or all-in-call leaves: there's no
        # decision to continue from, and re-rooting at an all-in-call child
        # would replace the resolver's exact runout EV with a model-based
        # estimate at a state the model isn't trained on.
        done_src = self._pull_back(self.env.done)
        allin_src = self._pull_back(self.allin_call_mask)
        skip_src = done_src | allin_src
        sampling_masks = self.child_mask.clone()
        sampling_masks[:top] &= ~skip_src
        sampling_counts = sampling_masks.float().sum(dim=-1, keepdim=True)

        # Calculate uniform sampling probabilities as backup.
        uniform = torch.where(sampling_masks, 1 / sampling_counts, 0)

        # Calculate policy sampling probabilities, excluding skipped children.
        policy_probs_by_src = self._pull_back(self.policy_probs_sample).clone()
        policy_probs_by_src.masked_fill_(skip_src[:, :, None], 0.0)
        denom = policy_probs_by_src.sum(dim=1, keepdim=True)
        policy_probs_by_src = torch.where(
            denom >= 1e-12,
            policy_probs_by_src / denom,
            uniform[:top, :, None],
        )

        # For deriving action index, we still need to use the base child mask.
        child_action_index = self.child_mask.int().cumsum(dim=1) - self.child_mask.int()

        # If a node has no legal actions after filtering done nodes, it's a leaf.
        effective_leaf_mask = self.leaf_mask | (sampling_counts.squeeze(1) == 0)

        # Select a private hand for each player at every root. The sampled path
        # uses the acting player's hand at each node.
        selected_hands_by_player = self._sample_root_hands_by_player()

        # start with root nodes and descend to leaves
        sampled_nodes = torch.arange(N, device=self.device)
        target_sampling = (
            training_mode and self._continuation_value_target_sampling_enabled()
        )
        target_enabled = torch.zeros(N, dtype=torch.bool, device=self.device)
        target_depths = torch.zeros(N, dtype=torch.long, device=self.device)
        abort_nodes = torch.full((N,), -1, dtype=torch.long, device=self.device)
        if target_sampling:
            bounds = self._continuation_value_target_depth_bounds()
            target_streets = self._continuation_value_target_streets()
            if bounds is not None and target_streets:
                min_depth, max_depth = bounds
                root_street = self.env.street[:N]
                for street in target_streets:
                    target_enabled |= root_street == street
                if target_enabled.any():
                    depth_draws = torch.randint(
                        min_depth,
                        max_depth + 1,
                        (N,),
                        generator=self.generator,
                        device=self.device,
                    )
                    target_depths = torch.where(
                        target_enabled, depth_draws, target_depths
                    )
                    hit_root = (
                        target_enabled
                        & (target_depths == 0)
                        & (~self.env.done[sampled_nodes])
                        & (~self.allin_call_mask[sampled_nodes])
                    )
                    abort_nodes[hit_root] = sampled_nodes[hit_root]

        depth = 0
        active_mask = ~effective_leaf_mask[sampled_nodes] & (abort_nodes < 0)
        while active_mask.any():
            assert depth <= self.max_depth
            active_nodes = sampled_nodes[active_mask]
            active_count = active_nodes.numel()
            active_root_indices = torch.where(active_mask)[0]

            to_act = self.env.to_act[active_nodes]
            player_mask = players[active_mask]
            sample_uniformly = (
                torch.rand(active_count, generator=self.generator, device=self.device)
                < sample_epsilon
            )
            sample_uniformly &= to_act == player_mask

            selected_hands = selected_hands_by_player[active_mask].gather(
                1, to_act[:, None]
            )
            policy_probs_active = policy_probs_by_src[
                active_nodes, :, selected_hands.squeeze(1)
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
            child_actions = child_action_index[active_nodes, actions]
            sampled_nodes[active_mask] = child_offsets + child_actions

            depth += 1
            if target_sampling:
                reached_nodes = sampled_nodes[active_root_indices]
                target_hit = (
                    target_enabled[active_root_indices]
                    & (target_depths[active_root_indices] == depth)
                    & (~self.env.done[reached_nodes])
                    & (~self.allin_call_mask[reached_nodes])
                )
                if target_hit.any():
                    hit_roots = active_root_indices[target_hit]
                    abort_nodes[hit_roots] = reached_nodes[target_hit]

            # remove node from the active mask once it's a leaf
            active_mask = ~effective_leaf_mask[sampled_nodes] & (abort_nodes < 0)

        continue_roots = abort_nodes < 0
        continue_nodes = sampled_nodes[continue_roots]
        assert effective_leaf_mask[continue_nodes].all()
        assert (~self.env.done[continue_nodes]).all()
        assert (~self.allin_call_mask[continue_nodes]).all()

        if target_sampling:
            self.continuation_value_target_indices = torch.empty(
                0, dtype=torch.long, device=self.device
            )

        # Don't sample root nodes.
        if target_sampling:
            abort_continue = abort_nodes[abort_nodes >= N]
            leaf_continue = continue_nodes[continue_nodes >= N]
            sampled_continue = torch.cat((abort_continue, leaf_continue), dim=0)
        else:
            sampled_continue = continue_nodes[continue_nodes >= N]
        pbs = PublicBeliefState.from_proto(
            env_proto=self.env,
            beliefs=self.beliefs_sample[sampled_continue],
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

    def _policy_targets_for_nodes(
        self, node_indices: torch.Tensor, top: int
    ) -> torch.Tensor:
        """Return policy targets for selected sparse parent nodes only."""
        out = torch.zeros(
            (node_indices.numel(), self.num_actions, self.hand_dim),
            dtype=self.policy_probs_avg.dtype,
            device=self.device,
        )
        if node_indices.numel() == 0 or self.total_nodes <= self.depth_offsets[1]:
            return out.permute(0, 2, 1)

        cols = torch.arange(self.num_actions, dtype=torch.long, device=self.device)
        counts = self.child_count[node_indices]
        valid = cols[None, :] < counts[:, None]
        child_indices = self.child_offsets[node_indices, None] + cols[None, :]
        safe_child_indices = torch.where(valid, child_indices, 0)
        actions = self.action_from_parent[safe_child_indices]
        valid = valid & (actions >= 0) & (actions < self.num_actions)
        rows = torch.arange(node_indices.numel(), dtype=torch.long, device=self.device)[
            :, None
        ].expand_as(actions)
        out[rows[valid], actions[valid]] = self.policy_probs_avg[
            safe_child_indices[valid]
        ]
        return out.permute(0, 2, 1)

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

    def _record_action_mix(self) -> None:
        """Record action mix without materializing [parent, action, hand]."""
        top = self.depth_offsets[-2]
        child_start = self.depth_offsets[1]
        child_end = self.total_nodes
        parent_index = self.parent_index[child_start:child_end]
        action_from_parent = self.action_from_parent[child_start:child_end]

        denom = self.allowed_hands[:top].sum(dim=1).clamp(min=1)
        child_mass = self.policy_probs_avg[child_start:child_end].sum(
            dim=1
        ) / denom.index_select(0, parent_index)

        action_mix_by_node = torch.zeros(
            top,
            self.num_actions,
            dtype=self.policy_probs_avg.dtype,
            device=self.device,
        )
        action_mix_by_node[parent_index, action_from_parent] = child_mass

        self.stats["action_mix"] = self._summarize_action_mix(
            action_mix_by_node[~self.leaf_mask[:top]]
        )
        self.stats["root_action_mix"] = self._summarize_action_mix(
            action_mix_by_node[: self.root_nodes][~self.leaf_mask[: self.root_nodes]]
        )

    def _record_cfr_entropy(self) -> None:
        """Record root policy entropy without full-tree action pullback."""
        if self.max_depth == 0:
            return
        N = self.root_nodes
        child_start = self.depth_offsets[1]
        child_end = self.depth_offsets[2]
        parent_index = self.parent_index[child_start:child_end]

        probs = self.policy_probs_avg[child_start:child_end]
        child_entropy = torch.where(probs > 1e-5, -(probs * probs.log()), 0.0)
        entropy_by_root = torch.zeros(
            N,
            probs.shape[1],
            dtype=probs.dtype,
            device=self.device,
        )
        entropy_by_root.scatter_reduce_(
            dim=0,
            index=parent_index[:, None].expand_as(child_entropy),
            src=child_entropy,
            reduce="sum",
            include_self=True,
        )
        self.stats["cfr_entropy"] = entropy_by_root[~self.leaf_mask[:N]].mean().item()

    def _record_cumulative_regret(self) -> None:
        self.stats["mean_positive_regret"] = (
            self.cumulative_regrets.clamp(min=0).mean().item()
        )

        exploit_stats = self._compute_exploitability()
        exploit_mbbg = self._local_exploitability_mbbg(
            exploit_stats.local_exploitability
        )
        self.stats["local_exploitability"] = (
            exploit_stats.local_exploitability.mean().item()
        )
        self.stats["local_exploitability_mbbg"] = exploit_mbbg.mean().item()

        N = self.root_nodes
        root_streets = self.env.street[:N]
        self.stats["local_exploitability_street"] = {
            street_name: exploit_stats.local_exploitability[root_streets == i]
            .mean()
            .item()
            for i, street_name in enumerate(STREETS)
            if (root_streets == i).any()
        }
        self.stats["local_exploitability_mbbg_street"] = {
            street_name: exploit_mbbg[root_streets == i].mean().item()
            for i, street_name in enumerate(STREETS)
            if (root_streets == i).any()
        }
        self.stats["local_exploitability_max"] = (
            exploit_stats.local_exploitability.max().item()
        )
        self.stats["local_exploitability_min"] = (
            exploit_stats.local_exploitability.min().item()
        )
        self._save_high_exploitability_roots(exploit_stats.local_exploitability)

    def _fan_out_deep(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fan out a root-aligned tensor across every node in the tree."""
        output = torch.zeros(
            (self.total_nodes, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=self.device,
        )
        output[: self.root_nodes] = tensor
        for depth in range(self.tree_depth):
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]
            output[offset_next:offset_next_next] = self._fan_out(output, depth)

        return output
