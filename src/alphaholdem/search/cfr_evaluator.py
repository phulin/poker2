"""Base CFR evaluator class with shared methods."""

from __future__ import annotations

import os
from abc import ABC
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from alphaholdem.core.structured_config import CFRType
from alphaholdem.env.card_utils import (
    NUM_HANDS,
    calculate_unblocked_mass,
    combo_to_onehot_tensor,
    hand_combos_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.env.rules import rank_hands
from alphaholdem.models.mlp.better_feature_encoder import BetterFeatureEncoder
from alphaholdem.models.mlp.better_ffn import BetterFFN
from alphaholdem.models.mlp.better_trm import BetterTRM
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder
from alphaholdem.models.mlp.rebel_ffn import RebelFFN
from alphaholdem.models.model_output import TRMLatent
from alphaholdem.rl.rebel_batch import RebelBatch
from alphaholdem.utils.model_utils import compute_masked_logits
from alphaholdem.utils.profiling import profile

STREETS = ["preflop", "flop", "turn", "river"]


@dataclass
class ExploitabilityStats:
    local_exploitability: torch.Tensor
    local_best_response_values: torch.Tensor


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
        num_envs: int | None = None,
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


def padded_indices(mask: torch.Tensor, alignment: int) -> torch.Tensor:
    """Compute indices from mask, padded to a multiple of alignment by repeating the last item."""
    indices = torch.where(mask)[0]
    current_len = indices.numel()
    if current_len > 0:
        remainder = current_len % alignment
        if remainder != 0:
            padding_size = alignment - remainder
            last_item = indices[-1:]
            padding = last_item.repeat(padding_size)
            indices = torch.cat([indices, padding])
    return indices


class CFREvaluator(ABC):
    """Base class for CFR evaluators with shared methods."""

    model: RebelFFN | BetterFFN | BetterTRM
    device: torch.device
    env: HUNLTensorEnv
    feature_encoder: RebelFeatureEncoder | BetterFeatureEncoder
    cfr_type: CFRType
    num_supervisions: int
    root_nodes: int
    total_nodes: int
    beliefs: torch.Tensor
    beliefs_avg: torch.Tensor
    legal_mask: torch.Tensor
    # Common fields shared by both evaluators
    float_dtype: torch.dtype
    num_players: int
    num_actions: int
    max_depth: int
    tree_depth: int
    cfr_iterations: int
    warm_start_iterations: int
    cfr_avg: bool
    dcfr_alpha: float
    dcfr_beta: float
    dcfr_gamma: float
    dcfr_delay: int
    sample_epsilon: float
    use_final_policy_values: bool
    generator: torch.Generator | None
    valid_mask: torch.Tensor
    leaf_mask: torch.Tensor
    child_mask: torch.Tensor
    child_count: torch.Tensor
    new_street_mask: torch.Tensor
    model_indices: torch.Tensor
    allowed_hands: torch.Tensor
    allowed_hands_prob: torch.Tensor
    policy_probs: torch.Tensor
    policy_probs_avg: torch.Tensor
    policy_probs_sample: torch.Tensor
    uniform_policy: torch.Tensor
    cumulative_regrets: torch.Tensor
    regret_weight_sums: torch.Tensor
    latest_values: torch.Tensor
    values_avg: torch.Tensor
    self_reach: torch.Tensor
    self_reach_avg: torch.Tensor
    root_pre_chance_beliefs: torch.Tensor
    latent: TRMLatent | None
    last_model_values: torch.Tensor | None
    showdown_indices: torch.Tensor
    showdown_actors: torch.Tensor
    showdown_potential: torch.Tensor
    prev_actor: torch.Tensor
    combo_onehot_float: torch.Tensor
    chance_helper: object  # ChanceNodeHelper - avoiding circular import
    stats: dict[str, float]
    depth_offsets: list[int]
    # Profiler fields (optional, initialized by subclasses if needed)
    profiler_enabled: bool
    profiler: any
    profiler_output_dir: str | None

    # ============================================================================
    # Abstract Methods (must be implemented by subclasses)
    # ============================================================================

    def _fan_out_deep(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement _fan_out_deep.")

    def _construct_subgame(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
    ) -> None:
        """Construct the subgame tree structure (subclass-specific implementation).

        This method should:
        - Copy root states from src_env to self.env
        - Expand the tree by creating child nodes
        - Set up depth_offsets, valid_mask, leaf_mask, etc.
        - Initialize environment states for all nodes

        Args:
            src_env: Batched environment that holds the source root public states.
            src_indices: Row indices inside `src_env` to copy into the tree roots.
        """
        raise NotImplementedError("Subclasses must implement _construct_subgame.")

    def sample_leaves(self, training_mode: bool) -> any:
        """Sample leaves from `self.policy_probs_sample`.

        Returns:
            PublicBeliefState or None depending on subclass implementation.
        """
        raise NotImplementedError("Subclasses must implement sample_leaves.")

    def _fan_out(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Fanout data to all children nodes."""
        raise NotImplementedError("Subclasses must implement _fan_out.")

    def _pull_back(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Pull back data to all parent nodes."""
        raise NotImplementedError("Subclasses must implement _pull_back.")

    def _pull_back_sum(
        self, tensor: torch.Tensor, out: torch.Tensor, level: int | None = None
    ) -> None:
        """Pull back tensor and sum into output tensor."""
        raise NotImplementedError("Subclasses must implement _pull_back_sum.")

    def _push_down(self, data: torch.Tensor, level: int | None = None) -> torch.Tensor:
        """Push down data to all child nodes.

        Args:
            data: Data to push down, shape [M, B, ...].
            level: Depth level to push down from, or None for all levels.
        Returns:
            Data by child node, shape [M - N, ...].
        """
        raise NotImplementedError("Subclasses must implement _push_down.")

    def _mask_invalid(self, tensor: torch.Tensor) -> None:
        """Mask invalid nodes in the tensor. Noop for sparse evaluator."""
        raise NotImplementedError("Subclasses must implement _mask_invalid.")

    def _propagate_level_beliefs(self, depth: int) -> None:
        """Propagate beliefs from all nodes at a given level to all nodes at the next level."""
        raise NotImplementedError("Subclasses must implement _propagate_level_beliefs.")

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _block_beliefs(self, target: torch.Tensor | None = None) -> None:
        """Block beliefs based on the board."""
        if target is None:
            target = self.beliefs
        target.masked_fill_((~self.allowed_hands)[:, None, :], 0.0)

    def _normalize_beliefs(self, target: torch.Tensor | None = None) -> None:
        """Normalize beliefs across hands in-place for valid nodes.

        Note: allowed_hands_prob should be 0 on invalid nodes, so invalid nodes
        will automatically get 0 beliefs when denom is 0.
        """
        if target is None:
            target = self.beliefs

        denom = target.sum(dim=-1, keepdim=True)
        # If the action probability of getting to a node is 0, our
        # bayesian update will make the beliefs in that state all 0.
        # So we set them to uniform (allowed_hands_prob).
        # For invalid nodes, allowed_hands_prob is 0, so they get 0 beliefs.
        torch.where(
            denom > 1e-8,
            target / denom.clamp(min=1e-8),
            self.allowed_hands_prob[:, None, :],
            out=target,
        )

    def _compute_model_indices(self) -> torch.Tensor:
        """Compute model indices from leaf mask, padded to a multiple of num_envs.

        Returns:
            Tensor of indices where model evaluation is needed, padded to a multiple
            of num_envs by repeating the last item.
        """
        model_mask = self.leaf_mask & ~self.env.done
        return padded_indices(model_mask, self.root_nodes)

    def _get_mixing_weights(self, t: int) -> tuple[float, float]:
        """Get the mixing weights for the current iteration (0-indexed).

        For iteration t (0-indexed), returns (old, new) where:
        - old: weight for the previous average policy
        - new: weight for the current iteration's policy
        """
        if self.cfr_type == CFRType.standard:
            return t, 1
        elif self.cfr_type == CFRType.linear:
            return t, 2
        elif self.cfr_type == CFRType.discounted:
            return t**self.dcfr_gamma, (t + 1) ** self.dcfr_gamma
        elif self.cfr_type == CFRType.discounted_plus:
            if t > self.dcfr_delay:
                t_delay = t - self.dcfr_delay
                return t_delay, 2
            else:
                return 0, 1

    @torch.no_grad()
    def _get_model_policy_probs(self, indices: torch.Tensor) -> torch.Tensor:
        """Get policy probabilities from model for given indices."""
        features = self.feature_encoder.encode(self.beliefs, indices=indices)
        if isinstance(self.model, BetterTRM):
            latent = None
            for supervision in range(self.num_supervisions):
                model_output = self.model(
                    features,
                    include_policy=supervision == self.num_supervisions - 1,
                    include_value=False,
                    latent=latent,
                )
                latent = model_output.latent
        else:
            model_output = self.model(features, include_policy=True)

        logits = model_output.policy_logits
        legal_masks = self.legal_mask[indices]
        masked_logits = compute_masked_logits(logits, legal_masks[:, None, :])
        probs = F.softmax(masked_logits, dim=-1)
        probs.masked_fill_(
            (self.child_count[indices] == 0)[:, None, None],
            0.0,
        )
        return probs

    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> None:
        """Calculate self reach weights for each node.

        Note: Root nodes should already be initialized to 1.0 in initialize_subgame
        and are never updated by this method.
        """
        for depth in range(self.tree_depth):
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            target_dest = target[offset_next:offset_next_next]
            target_dest[:] = self._fan_out(target, level=depth)

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

        self._mask_invalid(target)
        self._block_beliefs(target)

    def _propagate_all_beliefs(
        self,
        target: torch.Tensor | None = None,
        reach_weights: torch.Tensor | None = None,
    ) -> None:
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

    def _get_sampling_schedule(self) -> torch.Tensor:
        N = self.root_nodes
        if self.cfr_type == CFRType.discounted_plus:
            sample_low = max(self.warm_start_iterations, self.dcfr_delay)
        else:
            sample_low = self.warm_start_iterations
        sample_low = min(sample_low, self.cfr_iterations)
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
            distribution, N, replacement=True, generator=self.generator
        )
        t_sample += sample_low

        return self._fan_out_deep(t_sample)

    def _init_hand_rank_data(self) -> None:
        device = self.device
        indices = self.showdown_indices
        M = indices.numel()
        board = self.env.board_indices[indices].int()  # (M,5)

        # Sorted position k (0..1325) replicated across batch
        k = torch.arange(NUM_HANDS, device=device).expand(
            indices.numel(), -1
        )  # (M,1326)

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
        starts = torch.full((M, NUM_HANDS), NUM_HANDS, dtype=torch.int, device=device)
        ends = torch.full((M, NUM_HANDS), -1, dtype=torch.int, device=device)
        starts.scatter_reduce_(1, group_id, k.int(), reduce="amin", include_self=True)
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

    @torch.compile(dynamic=True)
    def _showdown_value_both(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Compute showdown values for both players."""
        result = torch.empty_like(beliefs)
        result[:, 0] = self._showdown_value(beliefs, 0)
        result[:, 1] = self._showdown_value(beliefs, 1)
        return result

    def _showdown_value(self, beliefs: torch.Tensor, hero: int) -> torch.Tensor:
        """
        Exact river showdown EV using rank-CDF + blocker correction.
        Returns per-hand EV [N, 1326] (unsorted/original hand order) per env.
        Result is from hero perspective.

        Args:
            hero: Index of hero player (0 or 1).
            indices: Indices of nodes to compute showdown values for.

        Returns:
            Per-hand EV [N, 1326] (unsorted/original hand order) per env.
        """
        indices = self.showdown_indices
        M = indices.numel()
        device = self.device
        dtype = torch.float32  # or match belief dtype
        villain = 1 - hero

        if M == 0:
            return torch.zeros(0, NUM_HANDS, device=device, dtype=dtype)

        # --- Beliefs & boards ---
        # Showdown value always uses the normal beliefs, not the average beliefs.
        # We store it in latest_values which always corresponds to non-average beliefs.
        b_opp = beliefs[:, villain, :].to(dtype)  # (M,1326)

        sorted_indices = self.hand_rank_data.sorted_indices
        inv_sorted = self.hand_rank_data.inv_sorted
        H = self.hand_rank_data.H
        card_ok = self.hand_rank_data.card_ok
        hand_ok_mask = self.hand_rank_data.hand_ok_mask
        hands_c1c2_sorted = self.hand_rank_data.hands_c1c2_sorted
        L_idx = self.hand_rank_data.L_idx
        R_idx = self.hand_rank_data.R_idx

        c1 = hands_c1c2_sorted[..., 0]  # (M,1326)
        c2 = hands_c1c2_sorted[..., 1]  # (M,1326)

        # Sort opponent marginal by strength order
        b_opp_sorted = b_opp.gather(1, sorted_indices)  # (M,1326)

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
        ).clamp(min=1e-8)
        valid_denom = denom > 1e-8
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
        potential = self.showdown_potential[:, hero]

        return EV_hand * potential[:, None] / self.env.scale

    def _best_response_values(
        self,
        policy: torch.Tensor,
        beliefs: torch.Tensor,
        base_values: torch.Tensor,
        deviating_player: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute best response values."""
        N, B = self.root_nodes, self.num_actions
        top = self.depth_offsets[-2]
        if deviating_player is None:
            deviating_player = self._fan_out_deep(self.env.to_act[:N])

        values_br = torch.where(self.leaf_mask[:, None, None], base_values, 0.0)

        min_value = torch.finfo(base_values.dtype).min

        policy_src_all = self._pull_back(policy)

        actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)
        actor_beliefs = beliefs.gather(1, actor_indices).squeeze(1)[:top]

        marginal_policy = policy_src_all * actor_beliefs[:, None, :]
        policy_blocked = calculate_unblocked_mass(marginal_policy)
        matchup_mass = calculate_unblocked_mass(actor_beliefs)
        opponent_conditioned_policy = torch.where(
            matchup_mass[:, None, :] > 1e-8,
            policy_blocked / matchup_mass[:, None, :],
            0.0,
        )

        for depth in range(self.tree_depth - 1, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]

            indices = torch.arange(offset_next - offset, device=self.device)
            actor = self.env.to_act[offset:offset_next]
            deviator = deviating_player[offset:offset_next]
            invalid_children = ~self.child_mask[offset:offset_next]

            values_src = self._pull_back(values_br, level=depth)  # [K, B, 2, 1326]
            policy_src = policy_src_all[offset:offset_next]
            opponent_policy = opponent_conditioned_policy[offset:offset_next]

            actor_indices = actor[:, None, None, None].expand(-1, B, 1, NUM_HANDS)
            opp_indices = (1 - actor)[:, None, None, None].expand(-1, B, 1, NUM_HANDS)
            # Both [K, B, 1326]
            actor_values_src = values_src.gather(2, actor_indices).squeeze(2)
            opp_values_src = values_src.gather(2, opp_indices).squeeze(2)

            actor_values_for_best = actor_values_src.masked_fill(
                invalid_children[:, :, None], min_value
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
            dev_match = matchup_mass[offset:offset_next][:, None, :]  # [M, 1, 1326]
            P_dev = torch.where(
                dev_match > 1e-8,
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

        policy = self.policy_probs_avg
        beliefs = self.beliefs_avg
        leaf_values = self.values_avg.clamp(-1.0, 1.0)

        base_values = torch.zeros_like(leaf_values)
        self.compute_expected_values(
            policy=policy, beliefs=beliefs, leaf_values=leaf_values, values=base_values
        )
        br_values = self._best_response_values(policy, beliefs, leaf_values)

        root_indices = torch.arange(N, device=self.device)
        root_actor = self.env.to_act[:N]

        base_root = base_values[root_indices, root_actor]  # (N, NUM_HANDS)
        br_root = br_values[root_indices, root_actor]  # (N, NUM_HANDS)

        # Aggregate over hands using beliefs
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

    # ============================================================================
    # Core Logic Methods (in order called by cfr_iteration and evaluate_cfr)
    # ============================================================================

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
        N = self.root_nodes

        # Construct the subgame tree first (subclass-specific, allocates tensors)
        self._construct_subgame(src_env, src_indices)

        # Handle initial beliefs
        if initial_beliefs is None:
            initial_beliefs = torch.full(
                (N, self.num_players, NUM_HANDS),
                1.0 / NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
        else:
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        # Set initial beliefs
        self.beliefs[:N] = initial_beliefs
        self.beliefs_avg[:N] = initial_beliefs
        self.root_pre_chance_beliefs[:] = initial_beliefs
        self.self_reach[:N] = 1.0
        self.self_reach_avg[:N] = 1.0

        # latent always have shape [model_indices.numel(), model.hidden_dim]
        self.model_indices = self._compute_model_indices()
        self.latent = None

        # Compute allowed hands from root board
        board_mask_root = self.env.board_onehot[:N].any(dim=1).reshape(N, -1).float()
        root_allowed = (self.combo_onehot_float @ board_mask_root.T).T < 0.5
        root_allowed_prob = root_allowed.float()
        root_allowed_prob /= root_allowed_prob.sum(dim=-1, keepdim=True).clamp(min=1.0)

        # Fan out allowed hands to all nodes
        self.allowed_hands = self._fan_out_deep(root_allowed)
        self.allowed_hands_prob = self._fan_out_deep(root_allowed_prob)

        # Initialize hand rank data
        self._init_hand_rank_data()

        # Record statistics
        self.stats["evaluator_street"] = self.env.street[:N].float().mean().item()
        self.stats["evaluator_total_nodes"] = float(self.total_nodes)
        self.stats["evaluator_root_nodes"] = float(self.root_nodes)
        self.stats["evaluator_tree_depth"] = float(self.tree_depth)

    @torch.no_grad()
    @profile
    def initialize_policy_and_beliefs(self) -> None:
        """Push public beliefs down the tree using the freshly initialised policy."""
        self.policy_probs.zero_()
        self.model.eval()

        # Use defensive loop bounds: len(depth_offsets) - 2 ensures we don't go out of bounds
        if self.tree_depth == 0:
            # No depth to process, just block and normalize beliefs
            self._block_beliefs()
            self._normalize_beliefs()
            self._mask_invalid(self.policy_probs)
            self._calculate_reach_weights(self.self_reach, self.policy_probs)
            self.policy_probs_avg[:] = self.policy_probs
            self.self_reach_avg[:] = self.self_reach
            self.beliefs_avg[:] = self.beliefs
            return

        # Pre-allocate policy_probs_src for efficiency (used by sparse, but harmless for dense)
        top = self.depth_offsets[-2] if len(self.depth_offsets) > 1 else self.root_nodes
        policy_probs_src = torch.empty(
            top, self.num_actions, NUM_HANDS, device=self.device, dtype=self.float_dtype
        )

        for depth in range(self.tree_depth):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            # Get policy probabilities from model for nodes at current depth
            indices = torch.arange(offset, offset_next, device=self.device)
            model_policy = self._get_model_policy_probs(indices)  # [K, B, NUM_HANDS]
            policy_probs_src[offset:offset_next] = model_policy.permute(0, 2, 1)

            # Push down policy to children using _push_down (works for both dense and sparse)
            self.policy_probs[offset_next:offset_next_next] = self._push_down(
                policy_probs_src, level=depth
            )

            # Propagate beliefs from current level to next level
            self._propagate_level_beliefs(depth)

            # Block and normalize beliefs after each level
            self._block_beliefs()
            self._normalize_beliefs()

        # Mask invalid policy probs (noop for sparse, masks for dense)
        self._mask_invalid(self.policy_probs)

        # Calculate reach weights
        self._calculate_reach_weights(self.self_reach, self.policy_probs)

        # Initialize averages
        self.policy_probs_avg[:] = self.policy_probs
        self.self_reach_avg[:] = self.self_reach
        self.beliefs_avg[:] = self.beliefs

    def warm_start(self) -> None:
        """Simple warm start: use model values and do a best-response pass."""
        self.set_leaf_values(0)
        if not self.latest_values.isfinite().all():
            num_nonfinite = (~self.latest_values.isfinite()).sum().item()
            raise ValueError(
                f"Non-finite values in latest_values after set_leaf_values: "
                f"{num_nonfinite} non-finite elements out of {self.latest_values.numel()}"
            )

        self.compute_expected_values()
        if not self.latest_values.isfinite().all():
            num_nonfinite = (~self.latest_values.isfinite()).sum().item()
            raise ValueError(
                f"Non-finite values in latest_values after compute_expected_values: "
                f"{num_nonfinite} non-finite elements out of {self.latest_values.numel()}"
            )

        self._record_initial_exploitability()

        # [M, ]
        values_br_p0 = self._best_response_values(
            self.policy_probs,
            self.beliefs,
            self.latest_values,
            torch.zeros_like(self.env.to_act),
        )
        values_br_p1 = self._best_response_values(
            self.policy_probs,
            self.beliefs,
            self.latest_values,
            torch.ones_like(self.env.to_act),
        )
        # NB: Invalid on root nodes, but we don't use them for regret/policy calculation.
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

    def _maybe_enforce_zero_sum(
        self,
        hand_values: torch.Tensor,
        player_beliefs: torch.Tensor,
        ignore_mask: torch.Tensor | None = None,
    ) -> None:
        """
        Enforce zero-sum constraint on hand values by subtracting the weighted average.

        Args:
            hand_values: Tensor of shape (batch, num_players, NUM_HANDS)
            player_beliefs: Tensor of shape (batch, num_players, NUM_HANDS)
        """
        if self.model.enforce_zero_sum:
            hand_value_sums = (
                (hand_values * player_beliefs)
                .sum(dim=2, keepdim=True)
                .mean(dim=1, keepdim=True)
            )
            if ignore_mask is not None:
                hand_value_sums.masked_fill_(ignore_mask[:, None, None], 0.0)
            return hand_values - hand_value_sums
        else:
            return hand_values

    @torch.compile(dynamic=True)
    def _set_model_values(
        self, t: int, beliefs: torch.Tensor, features: MLPFeatures
    ) -> None:
        # Set model values for non-terminal leaves
        if isinstance(self.model, BetterTRM):
            # Note self.latent gets reinitialized for each subgame.
            model_output = self.model(
                features,
                include_policy=False,
                latent=self.latent,
            )
            self.latent = model_output.latent
        else:
            model_output = self.model(features, include_policy=False)

        if not self.cfr_avg or t <= 1 or self.last_model_values is None:
            new_values = torch.index_copy(
                self.latest_values,
                0,
                self.model_indices,
                model_output.hand_values,
            )
        else:
            # Mix with previous values (CFR-AVG style)
            old, new = self._get_mixing_weights(t)
            unmixed = (
                old + new
            ) * model_output.hand_values - old * self.last_model_values
            unmixed /= new
            unmixed = self._maybe_enforce_zero_sum(unmixed, beliefs)
            new_values = torch.index_copy(
                self.latest_values,
                0,
                self.model_indices,
                unmixed,
            )
        return new_values, model_output.hand_values

    @torch.no_grad()
    def set_leaf_values(self, t: int, beliefs: torch.Tensor | None = None) -> None:
        """Set leaf values from model or terminal states."""
        if beliefs is None:
            beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        features = self.feature_encoder.encode(
            beliefs, pre_chance_node=self.new_street_mask
        )

        # Pass the same beliefs used for feature encoding to _set_model_values
        # so that zero-sum enforcement is consistent with the model input
        new_values, last_model_values = self._set_model_values(
            t, beliefs[self.model_indices], features[self.model_indices]
        )
        # this is necessary because of torch.compile.
        self.latest_values = new_values.clone()
        self.last_model_values = last_model_values.clone()

        # Set showdown values
        showdown_beliefs = beliefs[self.showdown_indices]
        showdown_values = self._showdown_value_both(showdown_beliefs)
        self.latest_values[self.showdown_indices] = showdown_values

    def compute_expected_values(
        self,
        policy: torch.Tensor | None = None,
        beliefs: torch.Tensor | None = None,
        leaf_values: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> None:
        """Back up values from leaves to root under the provided policy."""
        if policy is None:
            policy = self.policy_probs
        if beliefs is None:
            beliefs = self.beliefs
        if leaf_values is None:
            leaf_values = self.latest_values
        if values is None:
            values = leaf_values

        if leaf_values is values:
            values.masked_fill_((~self.leaf_mask)[:, None, None], 0.0)
        else:
            torch.where(
                self.leaf_mask[:, None, None],
                leaf_values,
                torch.zeros_like(values),
                out=values,
            )

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
            matchup_values > 1e-8,
            policy_blocked / matchup_values,
            torch.zeros_like(policy_blocked),
            out=opponent_conditioned_policy[bottom:],
        )

        for depth in range(self.tree_depth - 1, -1, -1):
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

            self._pull_back_sum(weighted_child_values, values, level=depth)

    def compute_instantaneous_regrets(
        self, values_achieved: torch.Tensor, values_expected: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute instantaneous regrets for each action at each node.

        Args:
            values_achieved: [M, 2, 1326] tensor of values for each node.
            values_expected: [M, 2, 1326] tensor of expected values for each node, or none to use values_achieved.

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

        # Mask invalid nodes (noop for sparse, masks invalid nodes for dense)
        self._mask_invalid(regrets)

        return regrets

    @profile
    def update_policy(self, t: int) -> None:
        """Update policy using regret matching."""
        bottom = self.depth_offsets[1]
        positive_regrets = self.cumulative_regrets.clamp(min=0.0)
        regret_sum = torch.zeros_like(self.policy_probs)

        self._pull_back_sum(positive_regrets, regret_sum)
        denom = self._fan_out(regret_sum)

        # Get uniform policy fallback (1.0 / num_actions per node)
        uniform_fallback = self.uniform_policy[bottom:]

        torch.where(
            denom > 1e-8,
            positive_regrets[bottom:] / denom.clamp(min=1e-8),
            uniform_fallback,
            out=self.policy_probs[bottom:],
        )
        self._mask_invalid(self.policy_probs)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)
        self._propagate_all_beliefs(self.beliefs, self.self_reach)

        self.update_average_policy(t)
        self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    def update_average_policy(self, t: int) -> None:
        """Update the average policy by mixing it with the current policy."""
        if (
            self.cfr_type in [CFRType.discounted, CFRType.discounted_plus]
            and t <= self.dcfr_delay
        ):
            self.policy_probs_avg[:] = self.policy_probs
            return
        if t == 0:
            self.policy_probs_avg[:] = self.policy_probs
            return

        N = self.root_nodes

        old, new = self._get_mixing_weights(t)

        # Get actor indices at source nodes (nodes that have children)
        # _fan_out expects tensors aligned with source nodes (0 to depth_offsets[-2])
        top = self.depth_offsets[-2]
        actor_indices = self.env.to_act[:top, None, None].expand(-1, -1, NUM_HANDS)
        reach_actor = self.self_reach[:top].gather(1, actor_indices).squeeze(1)
        reach_avg_actor = self.self_reach_avg[:top].gather(1, actor_indices).squeeze(1)

        reach_avg_actor *= old
        reach_actor *= new

        # Fan out reach weights to get per-action reach weights
        reach_actor_dest = self._fan_out(reach_actor)
        reach_avg_actor_dest = self._fan_out(reach_avg_actor)

        # Compute weighted average of policies
        numerator = (
            reach_avg_actor_dest * self.policy_probs_avg[N:]
            + reach_actor_dest * self.policy_probs[N:]
        )
        denom = reach_avg_actor_dest + reach_actor_dest
        unweighted = (old * self.policy_probs_avg[N:] + new * self.policy_probs[N:]) / (
            old + new
        )

        torch.where(
            denom > 1e-8,
            numerator / denom.clamp(min=1e-8),
            unweighted,
            out=self.policy_probs_avg[N:],
        )

        policy_sum = torch.zeros(
            self.depth_offsets[-2],
            NUM_HANDS,
            device=self.device,
            dtype=self.float_dtype,
        )
        self._pull_back_sum(self.policy_probs_avg, policy_sum)
        policy_denom = self._fan_out(policy_sum)
        torch.where(
            policy_denom > 1e-8,
            self.policy_probs_avg[N:] / policy_denom.clamp(min=1e-8),
            self.policy_probs_avg[N:],
            out=self.policy_probs_avg[N:],
        )

        # Root nodes don't have policies (they're decision nodes, not action nodes)
        self.policy_probs_avg[:N] = 0.0

    def update_average_values(self, t: int) -> None:
        """
        Update average values with weighted average and enforce zero-sum constraint.

        Args:
            t: Current iteration number
        """

        old, new = self._get_mixing_weights(t)
        self.values_avg *= old
        self.values_avg += new * self.latest_values
        self.values_avg /= old + new
        self.values_avg[:] = self._maybe_enforce_zero_sum(
            self.values_avg, self.beliefs_avg, ignore_mask=self.env.done
        )

    def update_average_values_final(self) -> None:
        """
        Update average values with final policy values.
        """
        # Seed latest_values with the leaf values under beliefs_avg
        self.set_leaf_values(0, beliefs=self.beliefs_avg)
        # Using latest_values as leaf values, compute EVs into values_avg
        self.compute_expected_values(
            self.policy_probs_avg,
            self.beliefs_avg,
            self.latest_values,
            self.values_avg,
        )
        # Possibly redundant: enforce zero-sum on values_avg
        self.values_avg[:] = self._maybe_enforce_zero_sum(
            self.values_avg, self.beliefs_avg, ignore_mask=self.env.done
        )

    @profile
    def cfr_iteration(self, t: int) -> None:
        """Run one CFR iteration."""
        torch.where(
            (self.t_sample == t)[:, None],
            self.policy_probs,
            self.policy_probs_sample,
            out=self.policy_probs_sample,
        )

        # Compute regrets
        regrets = self.compute_instantaneous_regrets(self.latest_values)

        if self.cfr_type == CFRType.linear:  # Alternate updates.
            regrets.masked_fill_(self.prev_actor[:, None] == t % self.num_players, 0.0)
        elif self.cfr_type in [CFRType.discounted, CFRType.discounted_plus]:
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

        # Update cumulative regrets
        self.regret_weight_sums += 1
        self.cumulative_regrets += regrets

        # CFR+ trick: clamp regrets to non-negative
        if self.cfr_type == CFRType.discounted_plus:
            self.cumulative_regrets.clamp_(min=0)

        # Update policy
        old_policy_probs = self.policy_probs.clone()
        self.update_policy(t)
        self._record_stats(t, old_policy_probs)

        # Set leaf values and back up
        self.set_leaf_values(t)
        self.compute_expected_values()

        # Update average values
        if not self.use_final_policy_values:
            self.update_average_values(t)

    @profile
    def training_data(
        self, exclude_start: bool = True
    ) -> tuple[RebelBatch, RebelBatch, RebelBatch]:
        """Return training data from CFR evaluation."""
        N = self.root_nodes
        top = self.depth_offsets[-2] if len(self.depth_offsets) > 1 else N

        policy_targets = self._pull_back(self.policy_probs_avg)
        policy_targets = policy_targets[:top].permute(0, 2, 1)

        value_targets = self.values_avg[:N].clamp(-1.0, 1.0)

        value_targets = self.values_avg[:N].clamp(-1.0, 1.0)

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
        value_statistics["local_best_response_values"] = (
            exploit_stats.local_best_response_values
        )

        # Policy batch gets all valid, non-leaf states.
        # Use valid_mask directly (works for both: sparse has all-ones, dense has computed mask)
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

    def evaluate_cfr(self, training_mode: bool = True) -> PublicBeliefState:
        """Run CFR iterations to evaluate the subgame.

        Returns:
            Result of sample_leaves (PublicBeliefState for sparse, Optional[PublicBeliefState] for rebel).
        """
        self.model.eval()

        self.initialize_policy_and_beliefs()

        if self.warm_start_iterations > 0:
            self.warm_start()

        # Use t=0 here so set_leaf_values doesn't do the CFR-AVG de-averaging.
        self.set_leaf_values(0)
        self.compute_expected_values()
        self.values_avg[:] = self.latest_values

        self.t_sample = self._get_sampling_schedule()
        for t in range(self.warm_start_iterations, self.cfr_iterations):
            self.profiler_step()  # Profile start of CFR iteration
            self.cfr_iteration(t)

        if self.use_final_policy_values:
            self.update_average_values_final()

        # Record statistics
        self._record_action_mix()
        self._record_cfr_entropy()
        self._record_cumulative_regret()

        return self.sample_leaves(training_mode)

    # ============================================================================
    # Profiler Methods
    # ============================================================================

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

    # ============================================================================
    # Statistics Methods
    # ============================================================================

    def _record_stats(self, t: int, old_policy_probs: torch.Tensor) -> None:
        """Record statistics about the policy update."""

        # Compute the 5 percentile points (0, 25, 50, 75, 100)
        percentile_ts = (
            torch.linspace(self.warm_start_iterations, self.cfr_iterations - 1, 5)
            .round()
            .int()
            .tolist()
        )
        percentiles = [0, 25, 50, 75, 100]

        if t in percentile_ts:
            # Find which percentile this t corresponds to
            percentile_idx = percentile_ts.index(t)
            percentile = percentiles[percentile_idx]

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
            self.stats[f"cfr_delta.{percentile}"] = node_delta_mean.item()

    def _record_cfr_entropy(self) -> None:
        """Record the entropy of the policy."""
        if self.max_depth == 0:
            return
        N = self.root_nodes
        actions = self._pull_back(self.policy_probs_avg)[:N]
        mask = self.valid_mask[:N] & ~self.leaf_mask[:N]
        probs = actions[mask]
        entropy = torch.where(probs > 1e-8, -(probs * probs.log()), 0.0)
        self.stats["cfr_entropy"] = entropy.sum(dim=1).mean().item()

    def _record_initial_exploitability(self) -> None:
        """Record the initial exploitability."""
        N = self.root_nodes
        root_streets = self.env.street[:N]
        exploit_stats = self._compute_exploitability()
        self.stats["local_exploitability_init"] = (
            exploit_stats.local_exploitability.mean().item()
        )
        self.stats["local_exploitability_init_street"] = {
            street_name: (
                exploit_stats.local_exploitability[root_streets == i].mean().item()
            )
            for i, street_name in enumerate(STREETS)
            if (root_streets == i).any()
        }

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

        # Compute and record exploitability as a generation-time statistic
        exploit_stats = self._compute_exploitability()
        self.stats["local_exploitability"] = (
            exploit_stats.local_exploitability.mean().item()
        )

        # Record exploitability by street
        root_streets = self.env.street[:N]  # (N,)
        self.stats["local_exploitability_street"] = {
            street_name: exploit_stats.local_exploitability[root_streets == i]
            .mean()
            .item()
            for i, street_name in enumerate(STREETS)
            if (root_streets == i).any()
        }
        self.stats["local_exploitability_max"] = (
            exploit_stats.local_exploitability.max().item()
        )
        self.stats["local_exploitability_min"] = (
            exploit_stats.local_exploitability.min().item()
        )

        # Check for high exploitability and save game tree if needed
        max_exploitability = exploit_stats.local_exploitability.max()
        if max_exploitability > 10.0:
            import time

            timestamp = int(time.time())
            high_exploit_roots = torch.where(exploit_stats.local_exploitability > 10.0)[
                0
            ]

            for root_idx in high_exploit_roots.tolist():
                # 1. Identify all nodes belonging to this root's tree
                # Create a mask for just this root
                root_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
                root_mask[root_idx] = True

                # Fan out to find all children
                tree_mask = self._fan_out_deep(root_mask)
                # Include the root itself
                tree_mask[:N] = root_mask

                # Get indices of all nodes in this tree
                tree_indices = torch.where(tree_mask)[0]

                # 2. Create a sub-environment for this tree
                # We need to map the global indices to local indices (0 to len(tree_indices)-1)
                # However, HUNLTensorEnv expects a contiguous range or specific indices.
                # Since we want to save the *state*, we can create a new env of the right size
                # and copy the states.
                sub_env = HUNLTensorEnv.from_proto(self.env, num_envs=len(tree_indices))
                # We can use copy_state_from to copy from self.env[tree_indices] to sub_env[0..M]
                sub_env.copy_state_from(
                    self.env,
                    tree_indices,
                    torch.arange(len(tree_indices), device=self.device),
                )
                sub_env.generator = None

                # 3. Collect relevant tensors sliced by tree_indices
                parent_index = getattr(self, "parent_index", None)
                action_from_parent = getattr(self, "action_from_parent", None)
                saved_data = {
                    "env_state": sub_env,  # This might be heavy, but it's the most robust way
                    "tree_indices": tree_indices,
                    "root_idx": root_idx,
                    "exploitability": exploit_stats.local_exploitability[
                        root_idx
                    ].item(),
                    "policy_probs": self.policy_probs[tree_indices].cpu(),
                    "policy_probs_avg": self.policy_probs_avg[tree_indices].cpu(),
                    "beliefs": self.beliefs[tree_indices].cpu(),
                    "beliefs_avg": self.beliefs_avg[tree_indices].cpu(),
                    "latest_values": self.latest_values[tree_indices].cpu(),
                    "values_avg": self.values_avg[tree_indices].cpu(),
                    "cumulative_regrets": self.cumulative_regrets[tree_indices].cpu(),
                    "regret_weight_sums": self.regret_weight_sums[tree_indices].cpu(),
                    "model_state_dict": self.model.state_dict(),  # No optimizer state as requested
                    "leaf_mask": self.leaf_mask[tree_indices].cpu(),
                    "parent_index": (
                        parent_index[tree_indices].cpu()
                        if parent_index is not None
                        else None
                    ),
                    "action_from_parent": (
                        action_from_parent[tree_indices].cpu()
                        if action_from_parent is not None
                        else None
                    ),
                    "new_street_mask": self.new_street_mask[tree_indices].cpu(),
                    "depth_offsets": self.depth_offsets,
                    "self_reach": self.self_reach[tree_indices].cpu(),
                    "self_reach_avg": self.self_reach_avg[tree_indices].cpu(),
                    "allowed_hands": self.allowed_hands[tree_indices].cpu(),
                }

                filename = f"high_exploitability_root_{root_idx}_{timestamp}.pt"
                print(f"Saving high exploitability game tree to {filename}")
                torch.save(saved_data, filename)

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
