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


@dataclass
class ExploitabilityStats:
    local_exploitability: torch.Tensor
    local_br_policy: torch.Tensor
    local_br_values: torch.Tensor
    local_br_improvement: torch.Tensor


from alphaholdem.search.cfr_evaluator import CFREvaluator
from alphaholdem.search.chance_node_helper import ChanceNodeHelper


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

        self.valid_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.leaf_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.new_street_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.folded_mask = torch.empty(0, dtype=torch.bool, device=self.device)
        self.folded_rewards = torch.empty(0, dtype=self.float_dtype, device=self.device)

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
            parent_env = env_levels[-1]
            legal_mask = parent_env.legal_bins_mask()
            done_mask = parent_env.done
            legal_mask = legal_mask & (~done_mask)[:, None]
            stop_mask = parent_env.actions_this_round == 0
            if depth > 0:
                legal_mask = legal_mask & (~stop_mask)[:, None]
            child_count = int(legal_mask.sum().item())
            if child_count == 0:
                break

            env_next = HUNLTensorEnv.from_proto(parent_env, num_envs=child_count)
            action_bins = torch.full(
                (child_count,), -1, dtype=torch.long, device=self.device
            )

            parent_offset = self.depth_offsets[-1] - parent_env.N
            parent_indices_level = torch.empty(
                child_count, dtype=torch.long, device=self.device
            )
            action_indices_level = torch.empty(
                child_count, dtype=torch.long, device=self.device
            )

            write_ptr = 0
            for i in range(parent_env.N):
                legal_actions = torch.where(legal_mask[i])[0]
                count = int(legal_actions.numel())
                if count == 0:
                    continue
                src_indices = torch.full(
                    (count,), i, dtype=torch.long, device=self.device
                )
                dst_indices = torch.arange(
                    write_ptr, write_ptr + count, device=self.device
                )
                env_next.copy_state_from(parent_env, src_indices, dst_indices)
                action_bins[write_ptr : write_ptr + count] = legal_actions
                parent_indices_level[write_ptr : write_ptr + count] = parent_offset + i
                action_indices_level[write_ptr : write_ptr + count] = legal_actions
                write_ptr += count

            rewards, _, _ = env_next.step_bins(action_bins)
            env_levels.append(env_next)
            parent_index_levels.append(parent_indices_level)
            action_levels.append(action_indices_level)
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

        self.valid_mask = torch.ones(
            self.total_nodes, dtype=torch.bool, device=self.device
        )
        self.legal_mask = self.env.legal_bins_mask()

        root_mask = torch.zeros(self.total_nodes, dtype=torch.bool, device=self.device)
        root_mask[:num_roots] = True
        self.new_street_mask = (self.env.actions_this_round == 0) & ~root_mask
        self.leaf_mask = self.env.done | self.new_street_mask
        self.child_mask = self.legal_mask & (~self.leaf_mask)[:, None]

        non_root = ~root_mask
        self.folded_mask = non_root & self.leaf_mask & (self.action_from_parent == 0)
        self.folded_rewards = torch.zeros(
            self.total_nodes, dtype=self.float_dtype, device=self.device
        )
        self.folded_rewards[self.folded_mask] = rewards_tensor[self.folded_mask]

        self.child_count = torch.zeros(
            self.total_nodes, dtype=torch.long, device=self.device
        )
        if self.total_nodes > self.root_nodes:
            children = torch.arange(
                self.root_nodes, self.total_nodes, device=self.device
            )
            parents = self.parent_index[children]
            valid_children = parents >= 0
            if valid_children.any():
                ones = torch.ones_like(parents[valid_children])
                self.child_count.scatter_add_(
                    0, parents[valid_children], ones.to(torch.long)
                )
        self.child_offsets = torch.cumsum(self.child_count, dim=0)

        self.prev_actor = torch.full(
            (self.total_nodes,), -1, dtype=torch.long, device=self.device
        )
        if self.total_nodes > self.root_nodes:
            child_indices = torch.arange(
                self.root_nodes, self.total_nodes, device=self.device
            )
            parent_indices = self.parent_index[child_indices]
            valid = parent_indices >= 0
            if valid.any():
                self.prev_actor[child_indices[valid]] = self.env.to_act[
                    parent_indices[valid]
                ]

        self.policy_probs = torch.zeros(
            self.total_nodes, NUM_HANDS, dtype=self.float_dtype, device=self.device
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
        self.cumulative_regrets = torch.zeros_like(self.policy_probs)
        self.regret_weight_sums = torch.zeros_like(self.policy_probs)

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
        self.latest_values = torch.zeros_like(self.beliefs)
        self.values_avg = torch.zeros_like(self.beliefs)
        self.self_reach = torch.zeros_like(self.beliefs)
        self.self_reach_avg = torch.zeros_like(self.beliefs)
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
        self.allowed_hands &= self.valid_mask[:, None]

        self.allowed_hands_prob = self._fan_out_deep(root_allowed_prob)
        self.allowed_hands_prob.masked_fill_(~self.valid_mask[:, None], 0.0)

        if isinstance(self.model, BetterFFN):
            self.feature_encoder = BetterFeatureEncoder(
                env=self.env, device=self.device, dtype=self.float_dtype
            )
        else:
            self.feature_encoder = RebelFeatureEncoder(
                env=self.env, device=self.device, dtype=self.float_dtype
            )

    def _level_range(self, depth: int) -> tuple[int, int]:
        return self.depth_offsets[depth], self.depth_offsets[depth + 1]

    def _child_lookup(
        self, depth: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if depth + 2 >= len(self.depth_offsets):
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        parent_start, parent_end = self._level_range(depth)
        child_start = self.depth_offsets[depth + 1]
        child_end = self.depth_offsets[depth + 2]
        child_indices = torch.arange(child_start, child_end, device=self.device)
        if child_indices.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        parent_indices = self.parent_index[child_indices]
        mask = (parent_indices >= parent_start) & (parent_indices < parent_end)
        if not mask.any():
            return (
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
            )
        child_indices = child_indices[mask]
        parent_indices = parent_indices[mask]
        actions = self.action_from_parent[child_indices]
        return child_indices, parent_indices, actions

    def _initialize_with_copy(self, target: torch.Tensor) -> None:
        for depth in range(len(self.depth_offsets) - 2):
            child_indices, parent_indices, _ = self._child_lookup(depth)
            if child_indices.numel() == 0:
                continue
            target[child_indices] = target[parent_indices]

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
        target.masked_fill_(~self.valid_mask[:, None, None], 0.0)

    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> torch.Tensor:
        target.zero_()
        if self.root_nodes > 0:
            target[: self.root_nodes] = 1.0
        for depth in range(len(self.depth_offsets) - 2):
            child_indices, parent_indices, _ = self._child_lookup(depth)
            if child_indices.numel() == 0:
                continue
            parent_weights = target[parent_indices]
            child_weights = parent_weights.clone()
            actor = self.prev_actor[child_indices]
            row_idx = torch.arange(child_indices.numel(), device=self.device)
            child_weights[row_idx, actor, :] *= policy[child_indices]
            target[child_indices] = child_weights
        target.masked_fill_(~self.valid_mask[:, None, None], 0.0)
        self._block_beliefs(target)
        return target

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
        self.model.eval()
        for depth in range(len(self.depth_offsets) - 2):
            parent_start, parent_end = self._level_range(depth)
            parent_indices = torch.arange(parent_start, parent_end, device=self.device)
            if parent_indices.numel() == 0:
                continue

            logits = self._get_model_policy_probs(parent_indices)  # [P, NUM_HANDS, A]
            child_indices, parent_abs, actions = self._child_lookup(depth)
            if child_indices.numel() == 0:
                continue

            parent_rel = parent_abs - parent_start
            parent_probs = logits[parent_rel, :, actions]
            self.policy_probs[child_indices] = parent_probs

            parent_beliefs = self.beliefs[parent_abs].clone()
            actor = self.prev_actor[child_indices]
            row_idx = torch.arange(child_indices.numel(), device=self.device)
            parent_beliefs[row_idx, actor, :] *= parent_probs
            self.beliefs[child_indices] = parent_beliefs

        self._block_beliefs(self.beliefs)
        self._normalize_beliefs(self.beliefs)
        self.policy_probs.masked_fill_(~self.valid_mask[:, None], 0.0)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)

        self.policy_probs_avg.copy_(self.policy_probs)
        self.self_reach_avg.copy_(self.self_reach)
        self.beliefs_avg.copy_(self.beliefs)

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
    def set_leaf_values(self, t: int) -> None:
        """Set leaf values from model or terminal states."""
        # Set terminal values (fold/showdown)
        terminal_mask = self.leaf_mask & self.env.done
        if terminal_mask.any():
            # For terminal nodes, compute rewards
            # Note: This is simplified - in practice you'd compute actual terminal rewards
            # Here we assume latest_values already has terminal values set
            pass

        # Set model values for non-terminal leaves
        model_mask = self.leaf_mask & ~self.env.done
        features = self.feature_encoder.encode(self.beliefs, pre_chance_node=True)
        self.model.eval()
        model_output = self.model(features[model_mask])

        if t <= 1 or self.last_model_values is None:
            self.latest_values[model_mask] = model_output.hand_values
        else:
            # Mix with previous values (CFR-AVG style)
            old, new = t - 1, t
            self.latest_values[model_mask] = (
                (old + new) * model_output.hand_values - old * self.last_model_values
            ) / new
        self.last_model_values = model_output.hand_values

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

        for depth in range(len(self.depth_offsets) - 3, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            actor_indices = self.env.to_act[offset:offset_next]
            actor_indices_expanded = actor_indices[:, None, None].expand(
                -1, -1, NUM_HANDS
            )

            level_beliefs = beliefs[offset:offset_next]
            actor_beliefs = level_beliefs.gather(1, actor_indices_expanded).squeeze(1)

            parent_indices = self.parent_index[offset_next:offset_next_next]
            total_children = parent_indices.numel()
            if total_children == 0:
                continue
            relative = parent_indices - offset
            valid = (relative >= 0) & (relative < actor_beliefs.shape[0])
            beliefs_dest = torch.zeros(
                (total_children, NUM_HANDS),
                dtype=self.float_dtype,
                device=self.device,
            )
            beliefs_dest[valid] = actor_beliefs[relative[valid]]

            # All computation on the destination node.
            policy_dest = policy[offset_next:offset_next_next]
            marginal_policy = beliefs_dest * policy_dest
            policy_blocked = calculate_unblocked_mass(marginal_policy)
            matchup_values = calculate_unblocked_mass(beliefs_dest)
            opponent_conditioned_policy = torch.where(
                matchup_values > 1e-12, policy_blocked / matchup_values, 0.0
            )

            indices = torch.arange(offset_next_next - offset_next, device=self.device)
            prev_actor_indices = self.env.to_act[parent_indices]
            weighted_child_values = values[offset_next:offset_next_next].clone()
            weighted_child_values[indices, prev_actor_indices, :] *= policy_dest
            weighted_child_values[
                indices, 1 - prev_actor_indices, :
            ] *= opponent_conditioned_policy

            self._pull_back_sum(weighted_child_values, values, depth)

    def compute_instantaneous_regrets(
        self, values_achieved: torch.Tensor, values_expected: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute instantaneous regrets for each action at each node."""
        if values_expected is None:
            values_expected = values_achieved

        regrets = torch.zeros_like(self.policy_probs)

        if self.total_nodes <= self.root_nodes:
            return regrets

        child_indices = torch.arange(
            self.root_nodes, self.total_nodes, device=self.device
        )
        parent_indices = self.parent_index[child_indices]
        valid = parent_indices >= 0
        if not valid.any():
            return regrets

        child_indices = child_indices[valid]
        parent_indices = parent_indices[valid]

        actor = self.env.to_act[parent_indices]
        opp = 1 - actor

        parent_values_expected = values_expected[parent_indices, actor, :]
        child_values_achieved = values_achieved[child_indices, actor, :]
        advantages = child_values_achieved - parent_values_expected

        opponent_beliefs = self.beliefs[parent_indices, opp, :]
        weights = calculate_unblocked_mass(opponent_beliefs)

        regrets[child_indices] = weights * advantages
        regrets.masked_fill_(~self.valid_mask[:, None], 0.0)

        return regrets

    def update_policy(self, t: int) -> None:
        """Update policy using regret matching."""
        positive_regrets = self.cumulative_regrets.clamp(min=0.0)
        new_policy = torch.zeros_like(self.policy_probs)

        for depth in range(len(self.depth_offsets) - 2):
            parent_start, parent_end = self._level_range(depth)
            child_indices, parent_indices, _ = self._child_lookup(depth)
            if child_indices.numel() == 0:
                continue

            parent_rel = parent_indices - parent_start
            parent_count = parent_end - parent_start

            regret_sum = torch.zeros(
                (parent_count, NUM_HANDS),
                dtype=self.float_dtype,
                device=self.device,
            )
            regret_sum.index_add_(0, parent_rel, positive_regrets[child_indices])

            denom = regret_sum[parent_rel]
            legal_count = (
                self.child_count[parent_indices].clamp(min=1).to(self.float_dtype)
            )
            uniform = (1.0 / legal_count).unsqueeze(-1).expand(-1, NUM_HANDS)

            new_policy[child_indices] = torch.where(
                denom > 1e-8,
                positive_regrets[child_indices] / denom.clamp(min=1e-8),
                uniform,
            )

        self.policy_probs.zero_()
        self.policy_probs.copy_(new_policy)
        self.policy_probs.masked_fill_(~self.valid_mask[:, None], 0.0)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)
        self._propagate_all_beliefs(self.beliefs, self.self_reach)

        self.update_average_policy(t)
        self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    def _update_beliefs_from_policy(self) -> None:
        self._calculate_reach_weights(self.self_reach, self.policy_probs)
        self._propagate_all_beliefs(self.beliefs, self.self_reach)

    def update_average_policy(self, t: int) -> None:
        if self.cfr_type == CFRType.discounted and t <= self.dcfr_delay:
            self.policy_probs_avg.copy_(self.policy_probs)
            return
        if t == 0:
            self.policy_probs_avg.copy_(self.policy_probs)
            return

        if self.total_nodes <= self.root_nodes:
            return

        old, new = self._get_mixing_weights(t)

        child_indices = torch.arange(
            self.root_nodes, self.total_nodes, device=self.device
        )
        parent_indices = self.parent_index[child_indices]
        valid = parent_indices >= 0
        if not valid.any():
            return

        child_indices = child_indices[valid]
        parent_indices = parent_indices[valid]
        actor = self.env.to_act[parent_indices]

        reach_avg = self.self_reach_avg[parent_indices, actor, :]
        reach_cur = self.self_reach[parent_indices, actor, :]

        old_weights = old * reach_avg
        new_weights = new * reach_cur

        numerator = (
            old_weights * self.policy_probs_avg[child_indices]
            + new_weights * self.policy_probs[child_indices]
        )
        denom = old_weights + new_weights
        unweighted = (
            old * self.policy_probs_avg[child_indices]
            + new * self.policy_probs[child_indices]
        ) / (old + new)

        self.policy_probs_avg[child_indices] = torch.where(
            denom > 1e-12,
            numerator / denom.clamp(min=1e-12),
            unweighted,
        )
        self.policy_probs_avg[: self.root_nodes] = 0.0
        self.policy_probs_avg.masked_fill_(~self.valid_mask[:, None], 0.0)

    def sample_leaf(self, t: int) -> None:
        """Sample a leaf node for training (placeholder)."""
        # This would sample a leaf node based on reach probabilities
        # For now, just a placeholder

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
            parent_start, parent_end = self._level_range(depth)
            parents = torch.arange(parent_start, parent_end, device=self.device)
            if parents.numel() == 0:
                continue

            non_leaf = ~self.leaf_mask[parents]
            parents = parents[non_leaf]
            if parents.numel() == 0:
                continue

            legal = self.child_mask[parents]
            if not legal.any():
                continue

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

        values_br.masked_fill_(~self.valid_mask[:, None, None], 0.0)
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
        return tensor[start:end].repeat_interleave(
            self.child_count[start:end], dim=0, output_size=size
        )

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
