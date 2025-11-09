from dataclasses import dataclass
from typing import Optional

import torch

from alphaholdem.core.structured_config import CFRType, Config
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
from alphaholdem.rl.rebel_replay import RebelBatch
from alphaholdem.search.cfr_evaluator import CFREvaluator
from alphaholdem.search.chance_node_helper import ChanceNodeHelper
from alphaholdem.utils.profiling import profile


@dataclass
class ExploitabilityStats:
    local_exploitability: torch.Tensor
    local_br_policy: torch.Tensor
    local_br_values: torch.Tensor
    local_br_improvement: torch.Tensor


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

        self.total_nodes = 0
        self.root_nodes = 0
        self.depth_offsets = [0]
        self.env: Optional[HUNLTensorEnv] = None

        self.leaf_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.new_street_mask = torch.empty(0, dtype=torch.bool, device=self.device)

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
        self.self_reach_avg = torch.zeros_like(self.beliefs)

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

    def _initialize_with_copy(self, target: torch.Tensor) -> None:
        for depth in range(self.max_depth):
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]
            target[offset_next:offset_next_next] = self._fan_out(target, depth)

    def _block_beliefs(self, target: torch.Tensor) -> None:
        if target.numel() == 0:
            return
        target.masked_fill_(~self.allowed_hands[:, None, :], 0.0)

    def _normalize_beliefs(self, target: torch.Tensor) -> None:
        if target.numel() == 0:
            return
        denom = target.sum(dim=-1, keepdim=True)
        uniform = self.allowed_hands_prob[:, None, :]
        torch.where(
            denom > 1e-10,
            target / denom.clamp(min=1e-12),
            uniform,
            out=target,
        )

    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> None:
        target[: self.root_nodes] = 1.0

        for depth in range(self.max_depth):
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            target_dest = target[offset_next:offset_next_next]
            target_dest[:] = self._fan_out(target, depth)

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

        self._block_beliefs(target)

    def _propagate_all_beliefs(
        self, target: torch.Tensor, reach_weights: torch.Tensor
    ) -> None:
        self._initialize_with_copy(target)
        target *= reach_weights
        self._block_beliefs(target)
        self._normalize_beliefs(target)

    def _get_mixing_weights(self, t: int) -> tuple[float, float]:
        if self.cfr_type == CFRType.standard:
            return float(t - 1), 1.0
        if self.cfr_type == CFRType.linear:
            return float(t - 1), 2.0
        if self.cfr_type == CFRType.discounted:
            return (
                float((t - 1) ** self.dcfr_gamma),
                float(t**self.dcfr_gamma),
            )
        if self.cfr_type == CFRType.discounted_plus:
            if t > self.dcfr_delay:
                t_delay = t - self.dcfr_delay
                return float(t_delay - 1), 2.0
            return 0.0, 1.0
        return float(t - 1), 1.0

    def initialize_policy_and_beliefs(self) -> None:
        self.policy_probs.zero_()
        policy_probs_src = torch.empty(
            self.depth_offsets[-2], self.num_actions, NUM_HANDS, device=self.device
        )

        for depth in range(len(self.depth_offsets) - 2):
            # Initialize policy at each level based on beliefs, then propagate beliefs to the next level.
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            policy_probs_src[offset:offset_next] = self._get_model_policy_probs(
                torch.arange(offset, offset_next, device=self.device)
            ).permute(
                0, 2, 1
            )  # [P, B, NUM_HANDS]
            self.policy_probs[offset_next:offset_next_next] = self._push_down(
                policy_probs_src, depth
            )

            prev_actor_dest = self.prev_actor[offset_next:offset_next_next]
            prev_actor_indices = prev_actor_dest[:, None, None].expand(
                -1, -1, NUM_HANDS
            )
            policy_dest = self.policy_probs[offset_next:offset_next_next]
            self.beliefs.scatter_reduce_(
                dim=1,
                index=prev_actor_indices,
                src=policy_dest[:, None],
                reduce="prod",
                include_self=True,
            )

        self._block_beliefs(self.beliefs)
        self._normalize_beliefs(self.beliefs)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)

        self.policy_probs_avg[:] = self.policy_probs
        self.self_reach_avg[:] = self.self_reach
        self.beliefs_avg[:] = self.beliefs

    def warm_start(self) -> None:
        # Simple warm start: use model values and do a best-response pass
        self.set_leaf_values(0)
        self.compute_expected_values()

        # Compute regrets and accumulate
        regrets = self.compute_instantaneous_regrets(self.latest_values)
        self.cumulative_regrets += 15 * regrets  # Scale by warm start iterations
        self.regret_weight_sums += 15
        self.update_policy(16)

    @torch.no_grad()
    @profile
    def set_leaf_values(self, t: int) -> None:
        """Set leaf values from model or terminal states."""

        # Set model values for non-terminal leaves
        model_mask = self.leaf_mask & ~self.env.done
        model_indices = torch.where(model_mask)[0]
        features = self.feature_encoder.encode(self.beliefs, pre_chance_node=True)
        model_output = self.model(features[model_indices])

        if t <= 1 or self.last_model_values is None:
            self.latest_values[model_indices] = model_output.hand_values
        else:
            # Mix with previous values (CFR-AVG style)
            old, new = t - 1, t
            self.latest_values[model_indices] = (
                (old + new) * model_output.hand_values - old * self.last_model_values
            ) / new
        self.last_model_values = model_output.hand_values

        # Set showdown values
        showdown_values_p0 = self._showdown_value(0, self.showdown_indices)
        showdown_values_p1 = self._showdown_value(1, self.showdown_indices)
        self.latest_values[self.showdown_indices, 0] = showdown_values_p0
        self.latest_values[self.showdown_indices, 1] = showdown_values_p1

    def compute_expected_values(
        self,
        policy: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> None:
        """Back up values from leaves to root under the provided policy."""
        if policy is None:
            policy = self.policy_probs
        if values is None:
            values = self.latest_values
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        bottom, top = self.depth_offsets[1], self.depth_offsets[-2]
        actor_indices = self.env.to_act[:top]
        actor_indices_expanded = actor_indices[:top, None, None].expand(
            -1, -1, NUM_HANDS
        )
        actor_beliefs = beliefs[:top].gather(1, actor_indices_expanded).squeeze(1)
        beliefs_dest = self._fan_out(actor_beliefs)
        marginal_policy = beliefs_dest * policy[bottom:]

        policy_blocked = calculate_unblocked_mass(marginal_policy)
        matchup_values = calculate_unblocked_mass(beliefs_dest)
        opponent_conditioned_policy = torch.zeros_like(policy)
        torch.where(
            matchup_values > 1e-12,
            policy_blocked / matchup_values,
            torch.zeros_like(policy_blocked),
            out=opponent_conditioned_policy[bottom:],
        )

        for depth in range(len(self.depth_offsets) - 3, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            actor_indices = self.env.to_act[offset:offset_next]
            actor_indices_expanded = actor_indices[:, None, None].expand(
                -1, -1, NUM_HANDS
            )

            indices = torch.arange(offset_next_next - offset_next, device=self.device)
            prev_actor_indices = self.prev_actor[offset_next:offset_next_next]
            weighted_child_values = values[offset_next:offset_next_next].clone()
            weighted_child_values[indices, prev_actor_indices, :] *= policy[
                offset_next:offset_next_next
            ]
            weighted_child_values[
                indices, 1 - prev_actor_indices, :
            ] *= opponent_conditioned_policy[offset_next:offset_next_next]

            self._pull_back_sum(weighted_child_values, values, depth)

    def compute_instantaneous_regrets(
        self, values_achieved: torch.Tensor, values_expected: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute instantaneous regrets for each action at each node."""
        if values_expected is None:
            values_expected = values_achieved

        bottom = self.depth_offsets[1]

        regrets = torch.zeros_like(self.policy_probs)

        if self.total_nodes <= self.root_nodes:
            return regrets

        actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)
        parent_actor_values = values_expected.gather(1, actor_indices).squeeze(1)
        parent_actor_values = self._fan_out(parent_actor_values)

        prev_actor_indices = self.prev_actor[bottom:][:, None, None].expand(
            -1, -1, NUM_HANDS
        )
        child_actor_values = (
            values_achieved[bottom:].gather(1, prev_actor_indices).squeeze(1)
        )

        opponent_indices = (1 - self.env.to_act)[:, None, None].expand(
            -1, -1, NUM_HANDS
        )
        opponent_beliefs = self.beliefs.gather(1, opponent_indices).squeeze(1)
        weights = self._fan_out(calculate_unblocked_mass(opponent_beliefs))

        regrets[bottom:] = weights * (child_actor_values - parent_actor_values)

        return regrets

    @profile
    def update_policy(self, t: int) -> None:
        """Update policy using regret matching."""
        positive_regrets = self.cumulative_regrets.clamp(min=0.0)
        regret_sum = torch.zeros_like(self.policy_probs)

        for depth in range(len(self.depth_offsets) - 2):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            self._pull_back_sum(positive_regrets, regret_sum, depth)
            denom = self._fan_out(regret_sum, depth)

            torch.where(
                denom > 1e-8,
                positive_regrets[offset_next:offset_next_next] / denom.clamp(min=1e-8),
                self.uniform_policy[offset_next:offset_next_next],
                out=self.policy_probs[offset_next:offset_next_next],
            )
            regret_sum[offset:offset_next] += positive_regrets[offset:offset_next]

        self._calculate_reach_weights(self.self_reach, self.policy_probs)
        self._propagate_all_beliefs(self.beliefs, self.self_reach)

        self.update_average_policy(t)
        self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    def update_average_policy(self, t: int) -> None:
        if self.cfr_type == CFRType.discounted and t <= self.dcfr_delay:
            self.policy_probs_avg[:] = self.policy_probs
            return
        if t == 0:
            self.policy_probs_avg[:] = self.policy_probs
            return

        N = self.root_nodes

        old, new = self._get_mixing_weights(t)

        actor_indices = self._fan_out(self.env.to_act)[:, None, None].expand(
            -1, -1, NUM_HANDS
        )
        reach_actor = self.self_reach[N:].gather(1, actor_indices).squeeze(1)
        reach_avg_actor = self.self_reach_avg[N:].gather(1, actor_indices).squeeze(1)

        reach_avg_actor *= old
        reach_actor *= new

        reach_actor_dest = self._fan_out(reach_actor)
        reach_avg_actor_dest = self._fan_out(reach_avg_actor)

        numerator = (
            reach_avg_actor_dest * self.policy_probs_avg[N:]
            + reach_actor_dest * self.policy_probs[N:]
        )
        denom = reach_avg_actor_dest + reach_actor_dest
        unweighted = (old * self.policy_probs_avg[N:] + new * self.policy_probs[N:]) / (
            old + new
        )

        torch.where(
            denom > 1e-12,
            numerator / denom.clamp(min=1e-12),
            unweighted,
            out=self.policy_probs_avg[N:],
        )
        self.policy_probs_avg[:N] = 0.0

    def sample_leaf(self, t: int) -> None:
        """Sample a leaf node for training (placeholder)."""
        # This would sample a leaf node based on reach probabilities
        # For now, just a placeholder

    @profile
    def cfr_iteration(self, t: int, training_mode: bool = True) -> None:
        """Run one CFR iteration."""
        # Compute regrets
        regrets = self.compute_instantaneous_regrets(self.latest_values)

        # Update cumulative regrets
        self.regret_weight_sums += 1
        self.cumulative_regrets += regrets
        # CFR+ trick: clamp regrets to non-negative
        self.cumulative_regrets.clamp_(min=0)

        # Update policy
        self.update_policy(t)

        # Set leaf values and back up
        self.set_leaf_values(t)
        self.compute_expected_values()

        # Update average values
        if t == 0:
            self.values_avg[:] = self.latest_values
        else:
            old, new = t - 1, t
            self.values_avg = (old * self.values_avg + new * self.latest_values) / (
                old + new
            )

    def evaluate_cfr(self, num_iterations: int | None = None) -> None:
        """Run CFR iterations to evaluate the subgame."""
        total_iterations = num_iterations or self.cfr_iterations
        if total_iterations <= 0:
            return

        self.model.eval()

        self.initialize_policy_and_beliefs()
        self.set_leaf_values(0)
        self.compute_expected_values()
        self.values_avg[:] = self.latest_values

        warm_iters = min(self.warm_start_iterations, max(0, total_iterations - 1))
        if warm_iters > 0:
            self.warm_start()

        for t in range(warm_iters, total_iterations):
            self.cfr_iteration(t, training_mode=False)

    def _best_response_values(
        self,
        policy: torch.Tensor,
        base_values: torch.Tensor,
        target_player: int,
    ) -> torch.Tensor:
        assert target_player in (0, 1)
        values_br = torch.zeros_like(base_values)
        values_br[self.leaf_mask] = base_values[self.leaf_mask]

        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
        min_value = torch.finfo(base_values.dtype).min

        for depth in range(len(self.depth_offsets) - 2, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            parents = torch.arange(offset, offset_next, device=self.device)

            non_leaf = ~self.leaf_mask[parents]
            parents = parents[non_leaf]

            legal = self.child_mask[parents]

            actor = self.env.to_act[parents]
            target_mask = actor == target_player

            next_values = self._pull_back(values_br)[parents]  # [n, B, players, hands]
            next_policy = self._pull_back(policy)[parents]  # [n, B, hands]

            next_values = next_values.clone()
            next_policy = next_policy.clone()

            next_values.masked_fill_(~legal[:, :, None, None], 0.0)
            next_policy.masked_fill_(~legal[:, :, None], 0.0)

            parent_beliefs = beliefs[parents]
            actor_indices = actor[:, None, None]
            actor_beliefs = parent_beliefs.gather(1, actor_indices).squeeze(1)

            marginal_policy = next_policy * actor_beliefs[:, None, :]
            policy_blocked = calculate_unblocked_mass(marginal_policy)
            matchup_mass = calculate_unblocked_mass(actor_beliefs)
            opponent_policy = torch.where(
                matchup_mass[:, None, :] > 1e-12,
                policy_blocked / matchup_mass[:, None, :],
                0.0,
            )

            player_values = next_values[:, :, target_player, :].clone()
            player_values.masked_fill_(~legal[:, :, None], min_value)
            best_action = player_values.argmax(dim=1)
            gather_idx = best_action.unsqueeze(1)

            best_player = torch.gather(player_values, 1, gather_idx).squeeze(1)
            opp_slice = next_values[:, :, 1 - target_player, :]
            best_opp = torch.gather(opp_slice, 1, gather_idx).squeeze(1)

            actor_onehot = torch.nn.functional.one_hot(
                actor, num_classes=self.num_players
            ).to(next_policy.dtype)
            actor_onehot = actor_onehot[:, None, :, None]
            policy_matrix = (
                actor_onehot * next_policy[:, :, None, :]
                + (1 - actor_onehot) * opponent_policy[:, :, None, :]
            )
            expected = (next_values * policy_matrix).sum(dim=1)

            values_br[parents] = expected
            if target_mask.any():
                target_indices = parents[target_mask]
                values_br[target_indices, target_player] = best_player[target_mask]
                values_br[target_indices, 1 - target_player] = best_opp[target_mask]

        return values_br

    def _compute_exploitability(self) -> ExploitabilityStats:
        N = self.root_nodes
        if N == 0:
            empty = torch.empty(0, device=self.device, dtype=self.float_dtype)
            empty2 = torch.empty(0, 2, device=self.device, dtype=self.float_dtype)
            return ExploitabilityStats(
                local_exploitability=empty,
                local_br_policy=empty2,
                local_br_values=empty2,
                local_br_improvement=empty2,
            )

        policy = self.policy_probs_avg
        leaf_values = self.values_avg
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        base_values = torch.where(
            self.leaf_mask[:, None, None], leaf_values, 0.0
        ).clamp(-1.0, 1.0)
        self.compute_expected_values(policy=policy, values=base_values)
        br_values_p0 = self._best_response_values(policy, base_values, target_player=0)
        br_values_p1 = self._best_response_values(policy, base_values, target_player=1)

        base_root = base_values[:N]
        br_root_p0 = br_values_p0[:N]
        br_root_p1 = br_values_p1[:N]
        beliefs_root = beliefs[:N]

        beliefs_p0 = beliefs_root[:, 0, :]
        beliefs_p1 = beliefs_root[:, 1, :]

        base_player0 = (base_root[:, 0] * beliefs_p0).sum(dim=1)
        base_player1 = (base_root[:, 1] * beliefs_p1).sum(dim=1)
        base_players = torch.stack([base_player0, base_player1], dim=1)

        br_player0 = (br_root_p0[:, 0] * beliefs_p0).sum(dim=1)
        br_player1 = (br_root_p1[:, 1] * beliefs_p1).sum(dim=1)
        br_players = torch.stack([br_player0, br_player1], dim=1)

        improvements = br_players - base_players
        total_exploitability = improvements.sum(dim=1) / 2

        return ExploitabilityStats(
            local_exploitability=total_exploitability,
            local_br_policy=base_players,
            local_br_values=br_players,
            local_br_improvement=improvements,
        )

    def training_data(
        self, exclude_start: bool = True
    ) -> tuple[RebelBatch, RebelBatch, RebelBatch]:
        """Return training data from CFR evaluation."""
        N = self.root_nodes
        top = self.depth_offsets[-2] if len(self.depth_offsets) > 1 else N

        policy_targets = self._pull_back(self.policy_probs_avg)
        policy_targets = policy_targets[:top].permute(0, 2, 1)

        value_targets = self.values_avg[:N].clamp(-1.0, 1.0)
        if value_targets.numel() > 0:
            high = value_targets.abs().max(dim=-1).values > 10
            if high.any():
                print(
                    f"WARNING: Value targets are too large ({high.sum().item()} hands)"
                )

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

        value_statistics = {key: statistics[key][:N] for key in statistics}
        value_statistics["local_exploitability"] = exploit_stats.local_exploitability
        value_statistics["local_br_policy"] = exploit_stats.local_br_policy
        value_statistics["local_br_values"] = exploit_stats.local_br_values
        value_statistics["local_br_improvement"] = exploit_stats.local_br_improvement

        valid_top = ~self.leaf_mask[:top]
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

        pre_features_all = self.feature_encoder.encode(
            self.beliefs, pre_chance_node=True
        )
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

    def _pull_back(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pull back child-aligned tensor to per-parent action slices."""
        out = torch.zeros(
            (self.depth_offsets[-2], self.num_actions, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=self.device,
        )

        start = self.depth_offsets[1]
        end = self.total_nodes
        parent_indices = self.parent_index[start:end]
        actions = self.action_from_parent[start:end]
        out[parent_indices, actions] = tensor[start:end]
        return out

    def _pull_back_sum(
        self, tensor: torch.Tensor, out: torch.tensor, level: int | None = None
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
            slice_tensor = tensor[start:end]
        elif tensor.shape[0] == expected:
            slice_tensor = tensor
        else:
            raise ValueError(
                f"Tensor length {tensor.shape[0]} does not match expected slice {expected}"
            )

        # Sum values for the children from the appropriate slice of the input tensor.
        # policy_probs[parent_idx, action, ...] => prob[child_idx, ...]
        out.scatter_reduce_(
            dim=0,
            index=parent_indices[(...,) + (None,) * (tensor.dim() - 1)].expand(
                -1, *slice_tensor.shape[1:]
            ),
            src=slice_tensor,
            reduce="sum",
            include_self=False,
        )

    def _fan_out_deep(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fan out a root-aligned tensor across every node in the tree."""
        output = torch.zeros(
            (self.total_nodes, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=self.device,
        )
        output[: self.root_nodes] = tensor
        for depth in range(len(self.depth_offsets) - 2):
            child_start = self.depth_offsets[depth + 1]
            child_end = self.depth_offsets[depth + 2]
            if child_end <= child_start:
                continue
            parent_indices = self.parent_index[child_start:child_end]
            valid = parent_indices >= 0
            if valid.all():
                output[child_start:child_end] = output[parent_indices]
            else:
                output_slice = output[child_start:child_end]
                output_slice[valid] = output[parent_indices[valid]]
                output_slice[~valid] = 0
        return output
