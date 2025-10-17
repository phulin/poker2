from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import torch
import torch.nn.functional as F

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.card_utils import (
    combo_blocking_tensor,
    combo_to_onehot_tensor,
    hand_combos_tensor,
)
from alphaholdem.env.rebel_feature_encoder import RebelFeatureEncoder
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
    model: RebelFFN
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
    policy_probs: torch.Tensor
    policy_probs_avg: torch.Tensor
    cumulative_regrets: torch.Tensor
    values: torch.Tensor
    beliefs: torch.Tensor
    legal_mask: Optional[torch.Tensor]
    combo_compat: torch.Tensor
    feature_encoder: RebelFeatureEncoder
    hand_rank_data: Optional[HandRankData]

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
        sample_epsilon: float = 0.25,
    ):
        assert cfr_iterations > warm_start_iterations

        self.search_batch_size = search_batch_size
        self.model = model
        self.max_depth = max_depth
        self.bet_bins = bet_bins
        self.cfr_iterations = cfr_iterations
        self.warm_start_iterations = warm_start_iterations
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

        self.policy_probs = torch.zeros(
            self.total_nodes,
            NUM_HANDS,
            self.num_actions,
            device=self.device,
            dtype=self.float_dtype,
        )
        self.policy_probs_avg = torch.zeros_like(self.policy_probs)
        self.cumulative_regrets = torch.zeros_like(self.policy_probs)

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

        self.legal_mask = None
        self.illegal_mask = None

        # Compatibility matrix: compatibility[i, j] = 1 if combos i and j do not overlap
        blocking = combo_blocking_tensor(device=self.device)
        self.combo_compat = (~blocking).to(dtype=self.float_dtype)

        # Feature encoder for belief computation
        self.feature_encoder = RebelFeatureEncoder(
            env=self.env,
            device=device,
            dtype=float_dtype,
        )

        self.hand_rank_data = None

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
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        dest_indices = torch.arange(self.search_batch_size, device=self.device)
        self.env.reset()
        self.env.copy_state_from(src_env, src_indices, dest_indices, copy_deck=True)
        self.valid_mask.zero_()
        self.valid_mask[dest_indices] = True
        self.leaf_mask.zero_()
        self.leaf_mask[dest_indices] = self.env.done[dest_indices]
        self.policy_probs.zero_()
        self.policy_probs_avg.zero_()
        self.cumulative_regrets.zero_()
        self.values.zero_()
        self.beliefs.zero_()
        self.beliefs[dest_indices] = initial_beliefs
        self.legal_mask = None
        self.illegal_mask = None
        self.hand_rank_data = None

    def construct_subgame(self) -> None:
        """Expand the tree by cloning legal successor states at each depth."""
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions

        for depth in range(self.max_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            action_bins = torch.full((M,), -1, dtype=torch.long, device=self.device)
            # don't currently have a way to get a subset of the masks
            legal_masks = self.env.legal_bins_mask()
            for action in range(self.num_actions):
                current_legal_mask = (
                    legal_masks[offset:offset_next, action]
                    & self.valid_mask[offset:offset_next]
                    & ~self.env.done[offset:offset_next]
                )
                current_legal_indices = torch.where(current_legal_mask)[0] + offset
                if current_legal_indices.numel() == 0:
                    continue

                new_legal_indices = current_legal_indices + (action + 1) * B**depth * N
                assert new_legal_indices.max().item() < M

                self.env.copy_state_from(
                    self.env, current_legal_indices, new_legal_indices
                )
                self.valid_mask[new_legal_indices] = True

                action_bins[new_legal_indices] = action

            # TODO: To stop on street, capture new_streets here.
            rewards, _, _ = self.env.step_bins(action_bins, legal_masks=legal_masks)

            # Showdown values get set in set_leaf_values.
            finished_folded = (action_bins == 0) & self.env.done
            self.values[finished_folded, 0] = rewards[finished_folded].view(-1, 1)
            self.values[finished_folded, 1] = -rewards[finished_folded].view(-1, 1)

        leaf_start = self.depth_offsets[self.max_depth]
        leaf_end = self.depth_offsets[self.max_depth + 1]
        self.leaf_mask[leaf_start:leaf_end] = self.valid_mask[leaf_start:leaf_end]
        self.leaf_mask[self.valid_mask & self.env.done] = True

        self.legal_mask = self.env.legal_bins_mask()
        self.illegal_mask = ~self.legal_mask
        valid_legal_masks = self.legal_mask[self.valid_mask & ~self.leaf_mask]
        has_legal = valid_legal_masks.any(dim=-1)
        assert has_legal.all(), "Every valid node must have at least one legal action."

        assert self.values[self.valid_mask].abs().max() <= 5

    @torch.no_grad()
    def _get_model_policy_probs(self, indices: torch.Tensor) -> torch.Tensor:
        features = self.encode_current_states(indices)
        model_output = self.model(features)
        logits = model_output.policy_logits
        legal_masks = self.legal_mask[indices]
        masked_logits = compute_masked_logits(logits, legal_masks[:, None, :])
        return F.softmax(masked_logits, dim=-1)

    def _propagate_beliefs(
        self,
        action: int,
        current_legal_indices: torch.Tensor,
        next_legal_indices: torch.Tensor,
    ) -> None:
        """Propagate beliefs from current legal indices to next legal indices."""
        legal_actor = self.env.to_act[current_legal_indices]
        self.beliefs[next_legal_indices, legal_actor] = (
            self.beliefs[current_legal_indices, legal_actor]
            * self.policy_probs[current_legal_indices, :, action]
        )
        self.beliefs[next_legal_indices, 1 - legal_actor] = self.beliefs[
            current_legal_indices, 1 - legal_actor
        ]

    def _block_beliefs(self) -> None:
        """Block beliefs based on the board."""
        combo_onehot = combo_to_onehot_tensor(device=self.device).float()
        board_onehot = (
            self.env.board_onehot[self.valid_mask].any(dim=1).view(-1, 52).float()
        )
        # [N, 52] @ [52, 1326]
        blocked = (board_onehot @ combo_onehot.T).clamp(0, 1)
        self.beliefs[self.valid_mask] *= 1 - blocked[:, None, :]

    def _normalize_beliefs(self) -> None:
        """Normalize beliefs across hands in-place for valid nodes."""
        valid_indices = torch.where(self.valid_mask)[0]
        if valid_indices.numel() > 0:
            beliefs = self.beliefs[valid_indices]
            denom = beliefs.sum(dim=-1, keepdim=True)
            # If the action probability of getting to a node is 0, our
            # bayesian update will make the beliefs in that state all 0.
            # So we set them to uniform.
            self.beliefs[valid_indices] = torch.where(
                denom > 1e-9, beliefs / denom, 1.0 / NUM_HANDS
            )

    def _block_and_normalize_beliefs(self) -> None:
        # A little inefficient, but normalize twice to handle the case where
        # the action probability of getting to a node is 0 (restore uniform beliefs
        # in the first normalize and then block/normalize again).
        self._normalize_beliefs()
        self._block_beliefs()
        self._normalize_beliefs()

    @torch.no_grad()
    def initialize_policy_and_beliefs(self) -> None:
        """Push public beliefs down the tree using the freshly initialised policy."""
        N, B = self.search_batch_size, self.num_actions

        self.policy_probs.zero_()
        self.beliefs[self.search_batch_size :].zero_()

        for depth, current_indices in self._valid_nodes():
            probs = self._get_model_policy_probs(current_indices)

            self.policy_probs[current_indices] = probs
            self.policy_probs_avg[current_indices] = probs

            if depth >= self.max_depth:
                continue

            legal_masks = self.legal_mask[current_indices]

            # Bayesian update assuming both players follow the same policy.
            for action in range(self.num_actions):
                legal = legal_masks[:, action]
                next_indices = current_indices + (action + 1) * B**depth * N
                current_legal_indices = current_indices[legal]
                if current_legal_indices.numel() == 0:
                    continue
                next_legal_indices = next_indices[legal]
                self._propagate_beliefs(
                    action, current_legal_indices, next_legal_indices
                )

        self._block_and_normalize_beliefs()

    def warm_start(self) -> None:
        N, B = self.search_batch_size, self.num_actions
        min_value = torch.finfo(self.float_dtype).min

        leaf_indices = torch.where(self.leaf_mask)[0]
        # [M, ]
        temp_values = torch.zeros_like(self.values)
        temp_values[leaf_indices] = self.values[leaf_indices]

        all_actions = torch.arange(B, device=self.device)[None, :]
        assert (self.values[~self.valid_mask] == 0.0).all()
        for depth, current_indices in self._valid_nodes(bottom_up=True):
            if depth == self.max_depth:
                continue

            next_indices = current_indices[:, None] + (all_actions + 1) * B**depth * N
            # [n]
            actor = self.env.to_act[current_indices]
            opp = 1 - actor
            # [n, B, 1326]
            next_actor_values = temp_values[next_indices, actor[:, None]]
            next_opp_values = temp_values[next_indices, opp[:, None]]

            # [n, B] - dot product over ranges of hand values
            actor_action_values = (
                self.policy_probs[current_indices].permute(0, 2, 1) * next_actor_values
            ).sum(dim=-1)
            opp_action_values_all = (
                self.policy_probs[current_indices].permute(0, 2, 1) * next_opp_values
            )
            opp_action_values = opp_action_values_all.sum(dim=-1)

            # take the player-to-act's best action, and the other player's average action
            actor_action_values.masked_fill_(
                ~self.legal_mask[current_indices], min_value
            )
            opp_action_values.masked_fill_(~self.legal_mask[current_indices], 0)
            all_indices = torch.arange(current_indices.numel(), device=self.device)
            temp_values[current_indices, actor] = next_actor_values[
                all_indices, actor_action_values.argmax(dim=-1)
            ]
            temp_values[current_indices, opp] = opp_action_values_all.sum(dim=1)

        assert temp_values[self.valid_mask].isfinite().all()

        # heuristic: scale regrets by the number of warm start iterations
        regrets = self.compute_instantaneous_regrets(temp_values)
        self.cumulative_regrets += self.warm_start_iterations * regrets
        self.update_policy()
        self.policy_probs_avg[:] = self.policy_probs

    @torch.no_grad()
    @profile
    def set_leaf_values(self) -> None:
        """Populate cached per-hand payoffs for nodes marked as leaves."""

        # Set estimated leaf value from model for non-terminal nodes.
        leaf_indices = torch.where(self.leaf_mask & ~self.env.done)[0]
        if leaf_indices.numel() == 0:
            return

        features = self.encode_current_states(leaf_indices)
        model_output = self.model(features)
        self.values[leaf_indices] = model_output.hand_values

        # Fold values were set in construct_subgame and don't need updating.
        # Showdown values need to be updated based on beliefs.
        # The env has hands it uses for showdown, but those are fake.
        showdown = torch.where(self.env.street == 4)[0]
        assert self.env.done[showdown].all()
        assert self.leaf_mask[showdown].all()
        showdown_values = self._showdown_value(showdown)
        self.values[showdown, 0] = showdown_values
        self.values[showdown, 1] = -showdown_values

    @profile
    def compute_expected_values(self) -> torch.Tensor:
        """Back up leaf hand values to their ancestors under the current policy."""

        new_values = self.values.clone()
        non_leaf_mask = self.valid_mask & ~self.leaf_mask
        if non_leaf_mask.any():
            new_values[non_leaf_mask] = 0.0
        # First iteration: leaf values already populated; back propagate expectations
        for _, action, current_indices, next_indices in self._valid_actions(
            bottom_up=True
        ):
            probs = self.policy_probs[current_indices, :, action]
            child_values = new_values[next_indices]
            new_values[current_indices] += child_values * probs[:, None, :]

        return new_values

    @profile
    def compute_instantaneous_regrets(self, values: torch.Tensor) -> torch.Tensor:
        """Compute regrets for every valid non-leaf information set."""

        regrets = torch.zeros_like(self.policy_probs)

        for _, action, current_indices, next_indices in self._valid_actions():
            actor = self.env.to_act[current_indices]
            row_ids = torch.arange(
                current_indices.numel(), device=self.device, dtype=torch.long
            )

            opp_beliefs = self.beliefs[current_indices][row_ids, 1 - actor]
            weights = opp_beliefs @ self.combo_compat

            next_vals = values[next_indices, actor]
            current_vals = values[current_indices, actor]
            advantages = next_vals - current_vals
            regrets[current_indices, :, action] = weights * advantages

        # Zero out invalid actions explicitly
        regrets *= self.legal_mask.unsqueeze(1).to(self.float_dtype)
        return regrets

    @profile
    def update_policy(self) -> None:
        """Apply a regret-matching update to every valid non-leaf information set."""

        regret_mask = self.valid_mask & ~self.leaf_mask

        positive_regrets = self.cumulative_regrets.clamp(min=0)
        regret_sum = positive_regrets.sum(dim=-1, keepdim=True)
        updated = torch.where(
            regret_sum > 1e-8,
            positive_regrets,
            1.0,
        )
        # Ensure illegal actions remain zero
        updated.masked_fill_(self.illegal_mask[:, None, :], 0.0)
        updated /= updated.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        torch.where(
            regret_mask[:, None, None],
            updated,
            self.policy_probs,
            out=self.policy_probs,
        )

        for _, action, current_indices, next_indices in self._valid_actions():
            self._propagate_beliefs(action, current_indices, next_indices)

        self._block_and_normalize_beliefs()

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
        N, M, B = self.search_batch_size, self.total_nodes, self.num_actions
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
            self.beliefs[valid_indices, self.env.to_act[valid_indices]], 1
        ).squeeze(1)

        # start with root nodes and descend to leaves
        sampled_nodes = root_indices.clone()

        depth = 0
        active_mask = ~self.leaf_mask[sampled_nodes]
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

            policy_probs_active = self.policy_probs[
                active_nodes, selected_hands[active_nodes]
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

            sampled_nodes[active_mask] = active_nodes + (actions + 1) * B**depth * N
            active_mask = ~self.leaf_mask[sampled_nodes]
            depth += 1

        dest_indices = torch.arange(
            pbs_start_idx, pbs_start_idx + count, device=self.device
        )
        pbs.env.copy_state_from(self.env, sampled_nodes, dest_indices)
        pbs.beliefs[pbs_start_idx : pbs_start_idx + count] = self.beliefs[sampled_nodes]

    @profile
    def self_play_iteration(
        self, training_mode: bool = True
    ) -> Optional[PublicBeliefState]:
        """Run one iteration through the CFR loop and produce leaf samples for replay."""
        self.construct_subgame()
        self.initialize_policy_and_beliefs()
        if self.warm_start_iterations > 0:
            self.set_leaf_values()
            self.warm_start()
        self.set_leaf_values()
        self.values = self.compute_expected_values()

        leaf_indices = torch.where(self.leaf_mask & ~self.env.done)[0]
        sample_count = min(leaf_indices.numel(), self.search_batch_size)
        next_pbs = None
        next_pbs_idx = 0
        if sample_count > 0:
            next_pbs = PublicBeliefState.from_proto(
                env_proto=self.env,
                beliefs=torch.zeros(
                    sample_count, self.num_players, NUM_HANDS, device=self.device
                ),
                num_envs=sample_count,
            )
            t_sample = torch.randint(
                self.warm_start_iterations,
                self.cfr_iterations,
                (sample_count,),
                device=self.device,
            )

        for t in range(self.warm_start_iterations, self.cfr_iterations):
            if sample_count > 0:
                # If t == t_sample, sample leaf PBS
                sample_now = torch.where(t_sample == t)[0]
                if sample_now.numel() > 0:
                    self.sample_leaf(
                        sample_now,
                        next_pbs,
                        next_pbs_idx,
                        training_mode=training_mode,
                    )
                    next_pbs_idx += sample_now.numel()

            regrets = self.compute_instantaneous_regrets(self.values)
            self.cumulative_regrets += regrets
            self.update_policy()

            # Update average policy.
            self.policy_probs_avg[self.valid_mask] = (
                t * self.policy_probs_avg[self.valid_mask]
                + self.policy_probs[self.valid_mask]
            ) / (t + 1)

            self.set_leaf_values()

            new_expected_values = self.compute_expected_values()

            # Clip accumulated values to prevent extreme outliers
            # Typical poker hand values should be bounded by stack size
            max_hand_value = (
                self.env.starting_stack / self.env.scale * 4
            )  # 2x starting stack in scaled units

            # Debug logging for extreme values
            extreme_values = (
                torch.abs(new_expected_values[self.valid_mask]) > max_hand_value
            )
            if torch.any(extreme_values):  # Warn at 80% of max
                extreme_count = extreme_values.sum().item()
                max_val = new_expected_values[self.valid_mask].max().item()
                min_val = new_expected_values[self.valid_mask].min().item()
                print(f"WARNING: Large hand values detected at iteration {t}")
                print(f"  Extreme values count: {extreme_count}")
                print(f"  Value range: [{min_val:.2f}, {max_val:.2f}]")
                print(f"  Max allowed: {max_hand_value:.2f}")

            self.values = (t * self.values + new_expected_values) / (t + 1)

        return next_pbs

    def sample_data(self) -> RebelBatch:
        """Aggregate model targets from the current root batch for supervised learning."""
        indices = torch.where(self.valid_mask[: self.search_batch_size])[0]
        return RebelBatch(
            features=self.encode_current_states(indices),
            policy_targets=self.policy_probs_avg[indices],
            value_targets=self.values[indices],
            legal_masks=self.env.legal_bins_mask()[indices],
            acting_players=self.env.to_act[indices],
        )

    def _showdown_value(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Exact river showdown EV using rank-CDF + blocker correction.
        Returns EV and per-hand EV (unsorted/original hand order) per env.
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
        assert torch.allclose(b_self.sum(dim=1), torch.ones_like(b_self.sum(dim=1)))
        assert torch.allclose(b_opp.sum(dim=1), torch.ones_like(b_opp.sum(dim=1)))

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
        assert ((denom > 1e-12) | ((win_mass < 1e-5) & (tie_mass < 1e-5))).all()

        # Probabilities & EV (in sorted order)
        win_prob = torch.where(denom > 1e-12, win_mass / denom, 0.0)
        tie_prob = torch.where(denom > 1e-12, tie_mass / denom, 0.0)

        EV_hand_sorted = win_prob + 0.5 * tie_prob

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
                next_legal_indices = current_legal_indices + (action + 1) * B**depth * N
                assert next_legal_indices.max().item() < M
                assert self.valid_mask[next_legal_indices].all()

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
        return torch.where(self.leaf_mask)[0]
