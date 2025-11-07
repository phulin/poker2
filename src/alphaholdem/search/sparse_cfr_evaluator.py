from alphaholdem.core.structured_config import Config
from alphaholdem.env.card_utils import NUM_HANDS, calculate_unblocked_mass
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.rl.rebel_replay import RebelBatch
from alphaholdem.search.cfr_evaluator import CFREvaluator
from alphaholdem.utils.model_utils import compute_masked_logits

import torch


class SparseCFREvaluator(CFREvaluator):
    def __init__(
        self, model: RebelFFN | BetterFFN, device: torch.device, cfg: Config
    ) -> None:
        self.model = model
        self.device = device
        self.cfg = cfg

        self.num_players = 2
        self.num_actions = len(cfg.env.bet_bins) + 3
        self.bet_bins = cfg.env.bet_bins

        self.total_nodes = 0
        self.depth_offsets = [0]
        self.leaf_mask = torch.empty(0, dtype=torch.bool, device=self.device)

        self.beliefs = torch.empty(
            0, self.num_players, NUM_HANDS, dtype=torch.float32, device=self.device
        )
        self.latest_values = torch.empty(
            0, self.num_players, NUM_HANDS, dtype=torch.float32, device=self.device
        )
        self.values_avg = torch.empty(
            0, self.num_players, NUM_HANDS, dtype=torch.float32, device=self.device
        )
        self.cumulative_regrets = torch.empty(
            0, NUM_HANDS, self.num_actions, dtype=torch.float32, device=self.device
        )
        self.policy_probs = torch.empty(
            0, NUM_HANDS, self.num_actions, dtype=torch.float32, device=self.device
        )
        self.policy_probs_avg = torch.empty(
            0, NUM_HANDS, self.num_actions, dtype=torch.float32, device=self.device
        )
        self.reach_weights = torch.empty(
            0, self.num_players, NUM_HANDS, dtype=torch.float32, device=self.device
        )
        self.reach_weights_avg = torch.empty(
            0, self.num_players, NUM_HANDS, dtype=torch.float32, device=self.device
        )
        self.regret_weight_sums = torch.empty(
            0, NUM_HANDS, self.num_actions, dtype=torch.float32, device=self.device
        )

        self.feature_encoder = None  # Will be initialized in initialize_subgame
        self.last_model_values = None
        self.parent_index = torch.empty(0, dtype=torch.long, device=self.device)
        self.action_from_parent = torch.empty(0, dtype=torch.long, device=self.device)

    def initialize_subgame(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        initial_envs = HUNLTensorEnv.from_proto(src_env, num_envs=src_indices.shape[0])
        initial_envs.copy_state_from(
            src_env, src_indices, torch.arange(initial_envs.N, device=self.device)
        )
        env_levels = [initial_envs]
        self.depth_offsets = [0, initial_envs.N]
        all_rewards = [torch.zeros(initial_envs.N, device=self.device)]
        self.root_nodes = initial_envs.N

        legal_mask = initial_envs.legal_bins_mask().cpu()
        next_count = legal_mask.sum().item()
        count = initial_envs.N + next_count

        depth = 0
        while next_count > 0 and depth < 10:
            depth += 1
            env_level = env_levels[-1]
            env_level_next = HUNLTensorEnv.from_proto(env_level, num_envs=next_count)
            taken = 0
            action_bins = torch.full(
                (next_count,), -1, dtype=torch.long, device=self.device
            )
            for i in range(env_level.N):
                legal = legal_mask[i].sum().item()
                if legal == 0:
                    continue
                legal_actions = torch.where(legal_mask[i])[0]
                for j, action in enumerate(legal_actions.tolist()):
                    env_level_next.copy_state_from(
                        env_level, slice(i, i + 1), slice(taken + j, taken + j + 1)
                    )
                    action_bins[taken + j] = action
                taken += legal

            rewards, _, _ = env_level_next.step_bins(action_bins)
            all_rewards.append(rewards)

            env_levels.append(env_level_next)
            self.depth_offsets.append(count)

            legal_mask = env_level_next.legal_bins_mask().cpu()
            # Don't go any further on envs that hit a new street.
            legal_mask[env_level_next.actions_this_round == 0] = False
            next_count = legal_mask.sum().item()
            count += next_count

        self.total_nodes = self.depth_offsets[-1]
        self.top_nodes = self.depth_offsets[-2]

        self.env = HUNLTensorEnv.from_proto(env_levels[-1], num_envs=self.total_nodes)
        copied = 0
        for env in env_levels:
            self.env.copy_state_from(
                env,
                torch.arange(env.N, device=self.device),
                torch.arange(copied, copied + env.N, device=self.device),
            )
            copied += env.N

        root_mask = torch.zeros(self.total_nodes, dtype=torch.bool, device=self.device)
        root_mask[: self.root_nodes] = True
        new_street = (self.env.actions_this_round == 0) & ~root_mask
        self.leaf_mask = self.env.done | new_street

        self.legal_mask = self.env.legal_bins_mask()
        self.child_mask = self.legal_mask & (~new_street)[:, None]
        self.child_count = self.child_mask.sum(dim=1)
        self.prev_actor = self.env.to_act.repeat_interleave(
            self.child_count, dim=0, output_size=self.total_nodes
        )

        # Build parent_index: maps each node to its parent (-1 for root nodes)
        self.parent_index = torch.full(
            (self.total_nodes,), -1, dtype=torch.long, device=self.device
        )
        # Build action_from_parent: maps each node to the action taken from parent
        self.action_from_parent = torch.full(
            (self.total_nodes,), -1, dtype=torch.long, device=self.device
        )

        # Track which nodes are children (not roots)
        cumsum_child_count = torch.cumsum(self.child_count, dim=0)

        # For each depth, assign parent indices and actions
        for depth in range(len(self.depth_offsets) - 1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            if offset_next >= self.total_nodes:
                break

            # Get parent nodes at this depth
            parent_indices = torch.arange(offset, offset_next, device=self.device)

            for parent_idx in parent_indices:
                if self.leaf_mask[parent_idx]:
                    continue  # Skip leaves

                # Compute child start index
                child_start_idx = (
                    offset_next
                    + (
                        cumsum_child_count[parent_idx] - self.child_count[parent_idx]
                    ).item()
                )
                child_end_idx = child_start_idx + self.child_count[parent_idx].item()

                if child_end_idx > self.total_nodes:
                    continue

                child_indices = torch.arange(
                    child_start_idx, child_end_idx, device=self.device, dtype=torch.long
                )
                if child_indices.numel() == 0:
                    continue

                legal_actions = torch.where(self.child_mask[parent_idx])[0]
                if legal_actions.numel() != child_indices.numel():
                    continue

                # Assign parent index and action for each child
                self.parent_index[child_indices] = parent_idx
                self.action_from_parent[child_indices] = legal_actions

        self.beliefs = torch.zeros(
            self.total_nodes,
            self.num_players,
            NUM_HANDS,
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_probs = torch.zeros(
            self.total_nodes,
            NUM_HANDS,
            self.num_actions,
            dtype=torch.float32,
            device=self.device,
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
        self.cumulative_regrets = torch.zeros_like(self.policy_probs)
        self.regret_weight_sums = torch.zeros_like(self.policy_probs)
        self.latest_values = torch.zeros(
            self.total_nodes,
            self.num_players,
            NUM_HANDS,
            dtype=torch.float32,
            device=self.device,
        )
        self.values_avg = torch.zeros_like(self.latest_values)
        self.reach_weights = torch.zeros_like(self.beliefs)
        self.reach_weights_avg = torch.zeros_like(self.beliefs)

        if initial_beliefs is not None:
            self.beliefs[: self.root_nodes] = initial_beliefs
            self.reach_weights[: self.root_nodes] = 1.0
            self.reach_weights_avg[: self.root_nodes] = 1.0

        # Initialize feature encoder after env is created
        if isinstance(self.model, BetterFFN):
            self.feature_encoder = BetterFeatureEncoder(
                env=self.env,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            self.feature_encoder = RebelFeatureEncoder(
                env=self.env,
                device=self.device,
                dtype=torch.float32,
            )

    def initialize_policy_and_beliefs(self) -> None:
        # For each level, get the policy probs from the model and use those to propagate beliefs down the tree.
        for depth in range(len(self.depth_offsets) - 1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            if depth + 2 >= len(self.depth_offsets):
                # No children at this depth
                break
            offset_next_next = self.depth_offsets[depth + 2]
            indices = torch.arange(offset, offset_next, device=self.device)
            probs = self._get_model_policy_probs(indices)
            self.policy_probs[offset:offset_next] = probs

            # Extract child probabilities: probs is [num_nodes, NUM_HANDS, num_actions]
            # We need to get probabilities for each child action
            # Get parent indices and actions for children
            child_indices = torch.arange(
                offset_next, offset_next_next, device=self.device
            )
            parent_indices_abs = self.parent_index[child_indices]
            # Filter to only valid children with valid parents
            valid_mask = (parent_indices_abs >= offset) & (
                parent_indices_abs < offset_next
            )
            if not valid_mask.any():
                continue
            child_indices_valid = child_indices[valid_mask]
            parent_indices = (
                parent_indices_abs[valid_mask] - offset
            )  # Relative to current depth
            child_actions = self.action_from_parent[child_indices_valid]

            # Gather probabilities: [num_children, NUM_HANDS]
            child_probs = (
                probs[parent_indices]
                .gather(
                    dim=2, index=child_actions[:, None, None].expand(-1, NUM_HANDS, 1)
                )
                .squeeze(2)
            )

            child_beliefs = torch.repeat_interleave(
                self.beliefs[offset:offset_next],
                self.child_count[offset:offset_next],
                dim=0,
                output_size=offset_next_next - offset_next,
            )
            # Only update beliefs for valid children
            child_beliefs[
                valid_mask, self.prev_actor[child_indices_valid]
            ] *= child_probs
            self.beliefs[child_indices_valid] = child_beliefs[valid_mask]

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

    def compute_expected_values(self) -> None:
        """Back up values from leaves to root using current policy (vectorized)."""
        # Start with current values (leaf values are already correct)
        parent_values = self.latest_values.clone()

        # Process depth by depth, bottom-up
        for depth in range(len(self.depth_offsets) - 2, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            if offset_next >= self.total_nodes:
                continue

            # Get all children at depth+1 that have parents at depth
            child_mask = (self.parent_index >= offset) & (
                self.parent_index < offset_next
            )
            child_indices = torch.where(child_mask)[0]

            if child_indices.numel() == 0:
                continue

            # Get parent indices for these children
            parent_indices = self.parent_index[child_indices]  # [num_children]
            actions = self.action_from_parent[child_indices]  # [num_children]

            # Get child values (already computed from deeper levels)
            child_values = parent_values[child_indices]  # [num_children, 2, NUM_HANDS]

            # Get policy probabilities for the actions taken
            # parent_indices are the parents, actions are the actions
            parent_policies = self.policy_probs[
                parent_indices
            ]  # [num_children, NUM_HANDS, num_actions]
            action_probs = parent_policies.gather(
                dim=2, index=actions[:, None, None].expand(-1, NUM_HANDS, 1)
            ).squeeze(
                2
            )  # [num_children, NUM_HANDS]

            # Weight child values by action probabilities
            # Expand action_probs to match child_values shape: [num_children, NUM_HANDS] -> [num_children, 2, NUM_HANDS]
            weighted_child_values = child_values * action_probs[:, None, :]

            # Aggregate weighted child values to parents using scatter_reduce
            # Prepare index tensor for scatter_reduce: [num_children, 2, NUM_HANDS]
            parent_indices_expanded = parent_indices[:, None, None].expand(
                -1, self.num_players, NUM_HANDS
            )

            # Use scatter_reduce to sum weighted child values to parents
            # This accumulates into parent_values[parent_indices]
            parent_values.scatter_reduce_(
                dim=0,
                index=parent_indices_expanded,
                src=weighted_child_values,
                reduce="sum",
                include_self=False,
            )

        # Update latest_values with computed parent values (only for non-leaves)
        non_leaf_mask = ~self.leaf_mask
        self.latest_values[non_leaf_mask] = parent_values[non_leaf_mask]

    def compute_instantaneous_regrets(
        self, values_achieved: torch.Tensor, values_expected: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute instantaneous regrets for each action at each node."""
        if values_expected is None:
            values_expected = values_achieved

        regrets = torch.zeros_like(
            self.policy_probs
        )  # [total_nodes, NUM_HANDS, num_actions]

        child_indices = torch.arange(
            self.root_nodes, self.total_nodes, device=self.device
        )

        # Get parent indices and actions for these children
        parent_indices = self.parent_index[child_indices]  # [num_children]
        actions = self.action_from_parent[child_indices]  # [num_children]

        # Get actor and opponent for each parent
        parent_actors = self.env.to_act[parent_indices]  # [num_children]
        parent_opps = 1 - parent_actors  # [num_children]

        # Expected values at parents (for the actor)
        parent_values_expected = values_expected[
            parent_indices, parent_actors
        ]  # [num_children, NUM_HANDS]

        # Values achieved at children (for the actor)
        child_values_achieved = values_achieved[
            child_indices, parent_actors
        ]  # [num_children, NUM_HANDS]

        # Compute advantages: child_value - expected_value
        advantages = (
            child_values_achieved - parent_values_expected
        )  # [num_children, NUM_HANDS]

        # Weight by opponent unblocked mass (matchup probability)
        parent_opp_beliefs = self.beliefs[
            parent_indices, parent_opps
        ]  # [num_children, NUM_HANDS]
        # Calculate unblocked mass: probability opponent's hand is compatible/unblocked
        parent_opp_unblocked = calculate_unblocked_mass(
            parent_opp_beliefs
        )  # [num_children, NUM_HANDS]
        weighted_regrets = (
            advantages * parent_opp_unblocked
        )  # [num_children, NUM_HANDS]

        # Use advanced indexing to set regrets vectorized
        num_children = child_indices.numel()
        hand_indices = torch.arange(NUM_HANDS, device=self.device)  # [NUM_HANDS]

        # Expand parent_indices and actions to match weighted_regrets shape
        # parent_indices: [num_children] -> [num_children, NUM_HANDS]
        parent_indices_expanded = parent_indices[:, None].expand(-1, NUM_HANDS)
        # actions: [num_children] -> [num_children, NUM_HANDS]
        actions_expanded = actions[:, None].expand(-1, NUM_HANDS)
        # hand_indices: [NUM_HANDS] -> [num_children, NUM_HANDS]
        hand_indices_expanded = hand_indices[None, :].expand(num_children, -1)

        # Use advanced indexing: regrets[parent_idx, hand_idx, action_idx] = weighted_regret
        regrets[parent_indices_expanded, hand_indices_expanded, actions_expanded] = (
            weighted_regrets
        )

        return regrets

    def update_policy(self, t: int) -> None:
        """Update policy using regret matching."""
        # Regret matching: policy = positive_regrets / sum(positive_regrets)
        positive_regrets = self.cumulative_regrets.clamp(min=0.0)

        # Sum positive regrets per hand
        regret_sum = positive_regrets.sum(
            dim=-1, keepdim=True
        )  # [total_nodes, NUM_HANDS, 1]

        # Normalize to get policy
        new_policy = torch.where(
            regret_sum > 1e-8,
            positive_regrets / regret_sum.clamp(min=1e-8),
            torch.ones_like(positive_regrets)
            / self.num_actions,  # Uniform if no regrets
        )

        # Mask illegal actions
        legal_mask_expanded = self.child_mask.unsqueeze(
            1
        )  # [total_nodes, 1, num_actions]
        new_policy = torch.where(
            legal_mask_expanded, new_policy, torch.zeros_like(new_policy)
        )

        # Renormalize after masking
        new_policy_sum = new_policy.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        new_policy = new_policy / new_policy_sum

        self.policy_probs = new_policy

        # Update beliefs based on new policy
        self._update_beliefs_from_policy()

        # Update average policy
        self.update_average_policy(t)

    def _update_beliefs_from_policy(self) -> None:
        """Propagate beliefs down the tree using current policy (vectorized)."""
        # Process depth by depth
        for depth in range(len(self.depth_offsets) - 1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = (
                self.depth_offsets[depth + 2]
                if depth + 2 < len(self.depth_offsets)
                else self.total_nodes
            )

            if offset_next >= self.total_nodes:
                break

            # Copy beliefs down using repeat_interleave (same pattern as initialize_policy_and_beliefs)
            child_beliefs = torch.repeat_interleave(
                self.beliefs[offset:offset_next],
                self.child_count[offset:offset_next],
                dim=0,
                output_size=offset_next_next - offset_next,
            )  # [num_children, 2, NUM_HANDS]

            # Get policy probabilities for children (where child_mask is True)
            child_probs = self.policy_probs[offset:offset_next][
                self.child_mask[offset:offset_next]
            ]  # [num_children, NUM_HANDS]

            # Multiply actor's beliefs by action probabilities
            child_indices = torch.arange(
                offset_next_next - offset_next, device=self.device
            )
            child_beliefs[
                child_indices, self.prev_actor[offset_next:offset_next_next]
            ] *= child_probs

            # Normalize beliefs
            belief_sums = child_beliefs.sum(
                dim=-1, keepdim=True
            )  # [num_children, 2, 1]
            child_beliefs = child_beliefs / belief_sums.clamp(min=1e-12)

            # Assign back to beliefs tensor
            self.beliefs[offset_next:offset_next_next] = child_beliefs

    def update_average_policy(self, t: int) -> None:
        """Update average policy using linear CFR averaging."""
        if t == 0:
            self.policy_probs_avg[:] = self.policy_probs
            return

        # Linear CFR: weight by t
        old, new = t - 1, t
        self.policy_probs_avg = (
            old * self.policy_probs_avg + new * self.policy_probs
        ) / (old + new)

    def sample_leaf(self, t: int) -> None:
        """Sample a leaf node for training (placeholder)."""
        # This would sample a leaf node based on reach probabilities
        # For now, just a placeholder
        pass

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

    def evaluate_cfr(self, num_iterations: int = 100) -> None:
        """Run CFR iterations to evaluate the subgame."""
        # Initialize policy and beliefs
        self.initialize_policy_and_beliefs()

        # Warm start (optional)
        if num_iterations > 15:
            self.warm_start()

        # Run CFR iterations
        for t in range(num_iterations):
            self.cfr_iteration(t, training_mode=False)

    def training_data(
        self, exclude_start: bool = True
    ) -> tuple[RebelBatch, RebelBatch, RebelBatch]:
        """Return training data from CFR evaluation."""
        # This is a placeholder - would need to implement proper batch creation
        # similar to RebelCFREvaluator.training_data
        raise NotImplementedError(
            "training_data not yet implemented for SparseCFREvaluator"
        )
