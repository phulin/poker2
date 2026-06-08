from __future__ import annotations

import torch

from p2.core.structured_config import CFRType
from p2.env.card_utils import PREFLOP_HANDS
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.pbs_env import PBSEnv
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.target_provenance import (
    TARGET_SOURCE_CFR_BACKUP,
    TARGET_SOURCE_CLOSING_NET,
    TARGET_SOURCE_EXACT_TERMINAL,
)
from p2.search.cfr_evaluator import CFREvaluator, ExploitabilityStats
from p2.search.fused_cfr_triton import (
    fused_average_policy_mix_multiway_with_tensors_,
    fused_average_policy_reach_beliefs_depth_preflop_multiway_,
    fused_avg_values_multiway_,
    fused_compact_regret_dcfr_update_multiway_with_tensors_,
    fused_model_values_writeback_multiway_,
    fused_parent_sum_divide_,
    fused_preflop169_parent_sum_opp_,
    fused_policy_reach_beliefs_depth_preflop_multiway_,
    fused_policy_renorm_reach_depth_multiway_,
    fused_preflop_multiway_beliefs_from_reach_,
    fused_regret_tail_multiway_,
    select_actor_beliefs_and_marginal_policy_multiway_triton_out_,
)
from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator
from p2.search.preflop_sparse_cfr_evaluator import PreflopSparseCFREvaluator
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


class FusedPreflopSparseCFREvaluator(FusedSparseCFREvaluator):
    """Compact-only fused sparse boundary for multiway S_preflop training.

    Tree construction, policy initialization, beliefs, values, regrets, and
    policy tensors are native 169-hand preflop classes. Card-removal math uses
    the exact class compatibility projection from ``PreflopSparseCFREvaluator``;
    generic fused average-policy/value/writeback kernels are retained where
    they are already parameterized by the final hand dimension.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if int(getattr(self, "hand_dim", PREFLOP_HANDS)) != PREFLOP_HANDS:
            raise ValueError(
                "FusedPreflopSparseCFREvaluator is compact-only; attach a "
                f"{PREFLOP_HANDS}-hand preflop policy/value model"
            )
        self._ensure_fused_attrs()
        self.warm_start_iterations = 0
        self._preflop_ev_actor_beliefs_buf: torch.Tensor | None = None
        self._preflop_ev_marginal_policy_buf: torch.Tensor | None = None
        self._preflop_ev_numer_unblocked_buf: torch.Tensor | None = None
        self._preflop_ev_denom_unblocked_buf: torch.Tensor | None = None

    @property
    def _compact_preflop(self) -> bool:
        return True

    def _construct_subgame(
        self,
        src_env: HUNLTensorEnv | PBSEnv,
        src_indices: torch.Tensor,
    ) -> None:
        SparseCFREvaluator._construct_subgame(self, src_env, src_indices)

    def _continuation_value_target_sampling_enabled(self) -> bool:
        return True

    def _continuation_value_target_streets(self) -> tuple[int, ...]:
        return (0,)

    def warm_start(self) -> None:
        return None

    def _preflop_unblocked_mass(self, class_mass: torch.Tensor) -> torch.Tensor:
        return PreflopSparseCFREvaluator._preflop_unblocked_mass(self, class_mass)

    def _preflop_ev_buffers(
        self,
        rows: int,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return PreflopSparseCFREvaluator._preflop_ev_buffers(
            self,
            rows,
            dtype=dtype,
        )

    def _preflop_unblocked_projection_for(
        self,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        return PreflopSparseCFREvaluator._preflop_unblocked_projection_for(self, ref)

    def _ensure_preflop_fused_ev_buffers(
        self,
        top: int,
        num_children: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_shape = (top, PREFLOP_HANDS)
        child_shape = (num_children, PREFLOP_HANDS)
        actor_buf = getattr(self, "_preflop_ev_actor_beliefs_buf", None)
        denom_buf = getattr(self, "_preflop_ev_denom_unblocked_buf", None)
        marginal_buf = getattr(self, "_preflop_ev_marginal_policy_buf", None)
        numer_buf = getattr(self, "_preflop_ev_numer_unblocked_buf", None)
        if (
            actor_buf is None
            or actor_buf.shape != actor_shape
            or actor_buf.device != self.device
        ):
            actor_buf = self.beliefs.new_empty(actor_shape)
            self._preflop_ev_actor_beliefs_buf = actor_buf
        if (
            denom_buf is None
            or denom_buf.shape != actor_shape
            or denom_buf.device != self.device
        ):
            denom_buf = self.beliefs.new_empty(actor_shape)
            self._preflop_ev_denom_unblocked_buf = denom_buf
        if (
            marginal_buf is None
            or marginal_buf.shape != child_shape
            or marginal_buf.device != self.device
        ):
            marginal_buf = self.policy_probs.new_empty(child_shape)
            self._preflop_ev_marginal_policy_buf = marginal_buf
        if (
            numer_buf is None
            or numer_buf.shape != child_shape
            or numer_buf.device != self.device
        ):
            numer_buf = self.policy_probs.new_empty(child_shape)
            self._preflop_ev_numer_unblocked_buf = numer_buf
        return (
            actor_buf,
            marginal_buf,
            denom_buf,
            numer_buf,
        )

    def initialize_subgame(
        self,
        src_env: HUNLTensorEnv | PBSEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        if not isinstance(src_env, PBSEnv):
            raise TypeError("FusedPreflopSparseCFREvaluator requires PBSEnv roots")
        if src_indices.dim() != 1:
            raise AssertionError("src_indices must be 1-D")
        if src_indices.numel() == 0:
            raise AssertionError("must supply at least one root state")
        root_streets = src_env.street[src_indices]
        if not (root_streets == 0).all():
            raise ValueError(
                "FusedPreflopSparseCFREvaluator only accepts street-0 roots"
            )
        root_boards = src_env.board_indices[src_indices]
        if (root_boards >= 0).any():
            raise ValueError(
                "FusedPreflopSparseCFREvaluator roots must have no public board"
            )
        self._ensure_fused_attrs()
        self._invalidate_subgame_caches()
        initial_beliefs = PreflopSparseCFREvaluator._compact_initial_beliefs(
            self,
            src_indices.numel(),
            initial_beliefs,
        )
        CFREvaluator.initialize_subgame(self, src_env, src_indices, initial_beliefs)
        PreflopSparseCFREvaluator._validate_compact_shapes(self)
        self._prepare_tree_slices()
        self._reset_average_policy_accumulators()

    def _prepare_compact_leaf_sampling(self, training_mode: bool) -> None:
        if self._continuation_value_target_sampling_enabled():
            self._sample_leaf_enabled = False
            return
        super()._prepare_compact_leaf_sampling(training_mode)
        assert self._sample_leaf_players is not None
        self._sample_leaf_players.random_(
            0, self.num_players, generator=self.generator
        )

    def _model_features_for_beliefs(
        self, beliefs_at_model: torch.Tensor
    ) -> MLPFeatures:
        key = (
            int(self._subgame_generation),
            int(self.model_indices.data_ptr()),
            int(self.new_street_mask.data_ptr()),
            int(self.model_indices.numel()),
        )
        if (
            self._static_model_feature_key != key
            or self._static_model_feature_fields is None
        ):
            value_encoder = getattr(self, "value_feature_encoder", self.feature_encoder)
            static_features = value_encoder.encode(
                self.beliefs,
                pre_chance_node=self.new_street_mask,
                indices=self.model_indices,
            )
            self._static_model_feature_fields = (
                static_features.context,
                static_features.street,
                static_features.to_act,
                static_features.board,
            )
            self._static_model_feature_key = key

        ctx, street, to_act, board = self._static_model_feature_fields
        return MLPFeatures(
            context=ctx,
            street=street,
            to_act=to_act,
            board=board,
            beliefs=beliefs_at_model.reshape(-1, self.num_players * PREFLOP_HANDS),
            hand_dim=PREFLOP_HANDS,
        )

    def _init_hand_rank_data(self) -> None:
        PreflopSparseCFREvaluator._init_hand_rank_data(self)

    def _empty_allin_call_partitions(self) -> None:
        PreflopSparseCFREvaluator._empty_allin_call_partitions(self)

    def _cache_preflop_allin_live_partitions(self) -> None:
        PreflopSparseCFREvaluator._cache_preflop_allin_live_partitions(self)

    def _cache_allin_call_street_partitions(
        self, parent_streets: torch.Tensor
    ) -> None:
        PreflopSparseCFREvaluator._cache_allin_call_street_partitions(
            self,
            parent_streets,
        )

    def _allin_call_child_mask(
        self,
        parent_env: HUNLTensorEnv | PBSEnv,
        parent_local_indices: torch.Tensor,
        action_bins: torch.Tensor,
    ) -> torch.Tensor:
        return PreflopSparseCFREvaluator._allin_call_child_mask(
            self,
            parent_env,
            parent_local_indices,
            action_bins,
        )

    def _mark_allin_call_leaves(self) -> None:
        PreflopSparseCFREvaluator._mark_allin_call_leaves(self)

    def _ensure_preflop_allin_169_oracle(self):
        return PreflopSparseCFREvaluator._ensure_preflop_allin_169_oracle(self)

    def _set_allin_call_values(self, beliefs: torch.Tensor) -> None:
        PreflopSparseCFREvaluator._set_allin_call_values(self, beliefs)

    def _compute_policy_node_reach(self, top: int) -> torch.Tensor:
        return PreflopSparseCFREvaluator._compute_policy_node_reach(self, top)

    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> None:
        CFREvaluator._calculate_reach_weights(self, target, policy)

    def _propagate_all_beliefs(
        self,
        target: torch.Tensor | None = None,
        reach_weights: torch.Tensor | None = None,
    ) -> None:
        if target is None:
            target = self.beliefs
        if reach_weights is None:
            reach_weights = self.self_reach
        if target.device.type != "cuda":
            CFREvaluator._propagate_all_beliefs(self, target, reach_weights)
            return

        self._prepare_tree_slices()
        root_index = self._get_root_index()
        fused_preflop_multiway_beliefs_from_reach_(
            beliefs=target,
            reach=reach_weights,
            allowed_prob=self.allowed_hands_prob,
            root_index=root_index,
            start=self.root_nodes,
        )

    def _refresh_fused_t_scalars(self, t: int) -> None:
        self._ensure_fused_attrs()
        mix_old, mix_new = self._get_mixing_weights(t)
        self._t_scalars.update(
            t=t,
            dcfr_alpha=self.dcfr_alpha,
            dcfr_beta=self.dcfr_beta,
            mix_old=float(mix_old),
            mix_new=float(mix_new),
            predictive_scale=self._predictive_policy_scale_for_t(t),
            current_player=t % self.num_players,
        )

    def compute_expected_values(
        self,
        policy: torch.Tensor | None = None,
        beliefs: torch.Tensor | None = None,
        leaf_values: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
    ) -> None:
        if policy is None:
            policy = self.policy_probs
        if beliefs is None:
            beliefs = self.beliefs
        if leaf_values is None:
            leaf_values = self.latest_values
        if values is None:
            values = leaf_values
        if values.device.type != "cuda":
            PreflopSparseCFREvaluator.compute_expected_values(
                self,
                policy=policy,
                beliefs=beliefs,
                leaf_values=leaf_values,
                values=values,
            )
            return

        use_leaf_source = leaf_values is not values
        if not use_leaf_source:
            pass
        elif self.tree_depth == 0:
            torch.where(
                self.leaf_mask[:, None, None],
                leaf_values,
                torch.zeros_like(values),
                out=values,
            )

        self._prepare_tree_slices()
        bottom, top = self._bottom, self._top
        parent_index_bottom = self._parent_index_bottom
        child_offsets_top = self._child_offsets_top
        child_count_top = self._child_count_top
        to_act_top = self._to_act_top
        assert bottom is not None
        assert top is not None
        assert parent_index_bottom is not None
        assert child_offsets_top is not None
        assert child_count_top is not None
        assert to_act_top is not None

        actor_beliefs, marginal_policy, denom_unblocked, numer_unblocked = (
            self._ensure_preflop_fused_ev_buffers(top, parent_index_bottom.numel())
        )
        beliefs_c = beliefs.contiguous()
        policy_c = policy.contiguous()
        prev_actor_c = self.prev_actor.contiguous()
        select_actor_beliefs_and_marginal_policy_multiway_triton_out_(
            beliefs_c,
            to_act_top,
            policy_c,
            child_offsets_top,
            child_count_top,
            bottom,
            actor_beliefs,
            marginal_policy,
            max_children=self.num_actions,
            block_h=256,
        )
        projection = self._preflop_unblocked_projection_for(beliefs_c).contiguous()
        torch.mm(actor_beliefs, projection, out=denom_unblocked)
        torch.mm(marginal_policy, projection, out=numer_unblocked)
        marginal_action_policy = marginal_policy.sum(dim=-1).contiguous()

        for depth in range(self.tree_depth - 1, -1, -1):
            fused_preflop169_parent_sum_opp_(
                values=values,
                prev_actor=prev_actor_c,
                policy=policy_c,
                marginal_action_policy=marginal_action_policy,
                numer_unblocked=numer_unblocked,
                denom_unblocked=denom_unblocked,
                child_offsets=self._child_offsets_by_depth[depth],
                child_count=self._child_count_by_depth[depth],
                parent_base=self.depth_offsets[depth],
                child_base=bottom,
                max_children=self.num_actions,
                max_children_pow2=self._child_count_pow2_by_depth[depth],
                leaf_values=leaf_values if use_leaf_source else None,
                leaf_mask=self.leaf_mask.contiguous() if use_leaf_source else None,
                has_folded=(
                    self.env.has_folded.contiguous()
                    if hasattr(self.env, "has_folded")
                    else None
                ),
            )

    def compute_instantaneous_regrets(
        self,
        values_achieved: torch.Tensor,
        values_expected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values_expected is None:
            values_expected = values_achieved
        if values_achieved.device.type != "cuda":
            return PreflopSparseCFREvaluator.compute_instantaneous_regrets(
                self,
                values_achieved,
                values_expected=values_expected,
            )

        self._prepare_tree_slices()
        bottom = self._bottom
        top = self._top
        parent_index_all = self._parent_index_all
        to_act_top = self._to_act_top
        assert bottom is not None
        assert top is not None
        assert parent_index_all is not None
        assert to_act_top is not None

        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
        unblocked_reach = self._preflop_unblocked_mass(beliefs[:top])
        player_ids = torch.arange(self.num_players, device=self.device)
        other_live = player_ids[None, :, None] != to_act_top[:, None, None]
        if hasattr(self.env, "has_folded"):
            other_live &= ~self.env.has_folded[:top, :, None]
        src_weights = torch.where(
            other_live,
            unblocked_reach.clamp_min(1e-12),
            torch.ones_like(unblocked_reach),
        ).prod(dim=1)
        src_weights *= self.allowed_hands[:top].to(dtype=self.float_dtype)

        regrets = torch.zeros_like(self.policy_probs)
        fused_regret_tail_multiway_(
            regrets=regrets,
            values_achieved=values_achieved.contiguous(),
            values_expected=values_expected[:top].contiguous(),
            to_act=to_act_top,
            src_weights=src_weights.contiguous(),
            parent_index=parent_index_all,
            prev_actor=self.prev_actor.contiguous(),
            bottom=bottom,
            block_h=256,
        )
        self._mask_invalid(regrets)
        return regrets

    def _renormalize_policy_reach(
        self,
        policy: torch.Tensor,
        reach: torch.Tensor,
    ) -> None:
        self._prepare_tree_slices()
        policy[: self.root_nodes] = 0.0
        prev_actor = self.prev_actor.contiguous()
        for depth in range(self.tree_depth):
            fused_policy_renorm_reach_depth_multiway_(
                policy=policy,
                reach=reach,
                allowed_mask=self.allowed_hands,
                child_offsets=self._child_offsets_by_depth[depth],
                child_count=self._child_count_by_depth[depth],
                prev_actor=prev_actor,
                parent_base=self.depth_offsets[depth],
                max_children=self.num_actions,
                update_reach=True,
            )

    def _renormalize_policy_reach_beliefs(
        self,
        policy: torch.Tensor,
        reach: torch.Tensor,
        beliefs: torch.Tensor,
    ) -> None:
        self._prepare_tree_slices()
        policy[: self.root_nodes] = 0.0
        root_index = self._get_root_index()
        prev_actor = self.prev_actor.contiguous()
        for depth in range(self.tree_depth):
            fused_policy_reach_beliefs_depth_preflop_multiway_(
                policy=policy,
                reach=reach,
                beliefs=beliefs,
                allowed_mask=self.allowed_hands,
                allowed_prob=self.allowed_hands_prob,
                root_index=root_index,
                child_offsets=self._child_offsets_by_depth[depth],
                child_count=self._child_count_by_depth[depth],
                prev_actor=prev_actor,
                parent_base=self.depth_offsets[depth],
                max_children=self.num_actions,
            )

    def _regret_match_current_policy(self, t: int | None = None) -> None:
        if self._try_apply_warm_start_ftrl_policy(t):
            self._renormalize_policy_reach_beliefs(
                self.policy_probs,
                self.self_reach,
                self.beliefs,
            )
            return

        self._prepare_tree_slices()
        bottom = self._bottom
        child_offsets_top = self._child_offsets_top
        child_count_top = self._child_count_top
        assert child_offsets_top is not None
        assert child_count_top is not None

        positive_regrets = self._ensure_positive_regrets_buf()
        if not self._fused_positive_regrets_valid:
            scale = self._predictive_policy_scale_for_t(t)
            last_regrets = getattr(self, "_last_instantaneous_regrets", None)
            if scale > 0.0 and last_regrets is not None:
                torch.add(
                    self.cumulative_regrets,
                    last_regrets,
                    alpha=scale,
                    out=positive_regrets,
                )
                torch.clamp(positive_regrets, min=0.0, out=positive_regrets)
            else:
                torch.clamp(self.cumulative_regrets, min=0.0, out=positive_regrets)
        self._fused_positive_regrets_valid = False

        assert bottom is not None
        fused_parent_sum_divide_(
            values=positive_regrets,
            fallback=self.uniform_policy[bottom:],
            child_offsets=child_offsets_top,
            child_count=child_count_top,
            out=self.policy_probs[bottom:],
            out_offset=bottom,
            max_children=self.num_actions,
            uniform_count_fallback=True,
            block_h=256,
        )
        self._mask_invalid(self.policy_probs)
        self._renormalize_policy_reach_beliefs(
            self.policy_probs,
            self.self_reach,
            self.beliefs,
        )

    def update_policy(self, t: int) -> None:
        self._refresh_fused_t_scalars(t)
        self._regret_match_current_policy(t)
        avg_beliefs_updated = self.update_average_policy(
            t,
            update_reach=True,
            update_beliefs=True,
        )
        if not avg_beliefs_updated:
            self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    def update_average_policy(
        self,
        t: int,
        update_reach: bool = False,
        update_beliefs: bool = False,
        weight_override: float | None = None,
    ) -> bool:
        defer_avg_policy = not self.cfr_avg and self.use_final_policy_values
        if self._uses_dcfr_backbone() and self._average_accumulation_delayed(t):
            if defer_avg_policy:
                self.average_policy_initialized = False
                return False
            self.policy_probs_avg[:] = self.policy_probs
            self.average_policy_initialized = False
            if update_reach:
                if update_beliefs:
                    self._renormalize_policy_reach_beliefs(
                        self.policy_probs_avg,
                        self.self_reach_avg,
                        self.beliefs_avg,
                    )
                    return True
                self._renormalize_policy_reach(
                    self.policy_probs_avg,
                    self.self_reach_avg,
                )
            return False

        self._prepare_tree_slices()
        root_count = self.root_nodes
        numerator, denominator = self._ensure_average_policy_buffers()
        parent_index_all = self._parent_index_all
        assert parent_index_all is not None
        new = self._t_scalars.mix_new
        if weight_override is not None:
            new = torch.tensor(
                float(weight_override), dtype=self.float_dtype, device=self.device
            )
        if (
            update_reach
            and update_beliefs
            and not defer_avg_policy
            and self.num_actions <= 8
        ):
            self.policy_probs_avg[:root_count] = 0.0
            root_index = self._get_root_index()
            prev_actor = self.prev_actor.contiguous()
            to_act = self.env.to_act.contiguous()
            for depth in range(self.tree_depth):
                fused_average_policy_reach_beliefs_depth_preflop_multiway_(
                    policy_probs_avg=self.policy_probs_avg,
                    average_policy_numerator=numerator,
                    average_policy_denominator=denominator,
                    policy_probs=self.policy_probs,
                    self_reach=self.self_reach,
                    self_reach_avg=self.self_reach_avg,
                    beliefs_avg=self.beliefs_avg,
                    allowed_mask=self.allowed_hands,
                    allowed_prob=self.allowed_hands_prob,
                    root_index=root_index,
                    child_offsets=self._child_offsets_by_depth[depth],
                    child_count=self._child_count_by_depth[depth],
                    prev_actor=prev_actor,
                    to_act=to_act,
                    new=new,
                    parent_base=self.depth_offsets[depth],
                    max_children=self.num_actions,
                )
            self.average_policy_initialized = True
            return True
        fused_average_policy_mix_multiway_with_tensors_(
            policy_probs_avg=self.policy_probs_avg,
            average_policy_numerator=numerator,
            average_policy_denominator=denominator,
            policy_probs=self.policy_probs,
            self_reach=self.self_reach,
            to_act=self.env.to_act.contiguous(),
            parent_index=parent_index_all,
            new=new,
            bottom=root_count,
            block_h=1024 if defer_avg_policy else 512,
            write_policy=not defer_avg_policy,
        )
        self.average_policy_initialized = True
        if not defer_avg_policy:
            self._renormalize_average_policy(
                update_reach=update_reach,
                update_beliefs=update_beliefs,
            )
            return bool(update_reach and update_beliefs)
        return False

    def _renormalize_average_policy(
        self,
        update_reach: bool,
        update_beliefs: bool = False,
    ) -> None:
        if update_reach:
            if update_beliefs:
                self._renormalize_policy_reach_beliefs(
                    self.policy_probs_avg,
                    self.self_reach_avg,
                    self.beliefs_avg,
                )
                return
            self._renormalize_policy_reach(
                self.policy_probs_avg, self.self_reach_avg
            )
            return
        self.policy_probs_avg[: self.root_nodes] = 0.0
        self._prepare_tree_slices()
        prev_actor = self.prev_actor.contiguous()
        for depth in range(self.tree_depth):
            fused_policy_renorm_reach_depth_multiway_(
                policy=self.policy_probs_avg,
                reach=self.self_reach_avg,
                allowed_mask=self.allowed_hands,
                child_offsets=self._child_offsets_by_depth[depth],
                child_count=self._child_count_by_depth[depth],
                prev_actor=prev_actor,
                parent_base=self.depth_offsets[depth],
                max_children=self.num_actions,
                update_reach=False,
            )

    def update_average_values(self, t: int) -> None:
        self._refresh_fused_t_scalars(t)
        old, new = self._get_mixing_weights(t)
        if old + new == 0:
            return
        fused_avg_values_multiway_(
            values_avg=self.values_avg,
            latest_values=self.latest_values,
            beliefs=self.beliefs_avg,
            old=self._t_scalars.mix_old,
            new=self._t_scalars.mix_new,
            inv_total=self._t_scalars.mix_inv_total,
            enforce_zero_sum=bool(self.model.enforce_zero_sum)
            and self.num_players == 2,
            ignore_mask=self.env.done,
        )

    def _set_model_values_impl(
        self,
        t: int,
        beliefs: torch.Tensor,
        features: MLPFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_fused_attrs()
        hand_values, model_applied_zero_sum = (
            self._model_leaf_values_for_fused_writeback(features)
        )
        do_mix = (
            self.cfr_avg
            and t > 1
            and self.last_model_values is not None
            and not self._average_accumulation_delayed(t)
        )
        store_last = bool(self.cfr_avg)
        hand_dim = int(getattr(self, "hand_dim", PREFLOP_HANDS))
        if store_last:
            last_shape = (hand_values.shape[0], self.num_players, hand_dim)
            if self._last_model_values_buf is None or (
                self._last_model_values_buf.shape != last_shape
            ):
                self._last_model_values_buf = self.latest_values.new_empty(last_shape)
            last_out = self._last_model_values_buf
            last_model_values = (
                self.last_model_values.contiguous() if do_mix else hand_values
            )
        else:
            last_out = hand_values
            last_model_values = hand_values

        enforce_writeback_zero_sum = (
            bool(self.model.enforce_zero_sum)
            and self.num_players == 2
            and (do_mix or not model_applied_zero_sum)
        )
        fused_model_values_writeback_multiway_(
            hand_values=hand_values,
            last_model_values=last_model_values,
            beliefs=beliefs.contiguous(),
            model_indices=self.model_indices.contiguous(),
            latest_values=self.latest_values,
            last_out=last_out,
            old_plus_new_over_new=self._t_scalars.mix_onon,
            old_over_new=self._t_scalars.mix_oon,
            do_mix=do_mix,
            enforce_zero_sum=enforce_writeback_zero_sum,
            store_last=store_last,
        )
        self.last_model_values = last_out if store_last else None
        return self.latest_values, last_out

    @torch.no_grad()
    def set_leaf_values(self, t: int, beliefs: torch.Tensor | None = None) -> None:
        self._refresh_fused_t_scalars(t)
        if beliefs is None:
            beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        if self.model_indices.numel() > 0:
            beliefs_at_model = beliefs[self.model_indices]
            features_at_model = self._model_features_for_beliefs(beliefs_at_model)
            self._set_model_values(t, beliefs_at_model, features_at_model)
        else:
            empty_shape = (0, self.num_players, PREFLOP_HANDS)
            if self._last_model_values_buf is None or (
                self._last_model_values_buf.shape != empty_shape
            ):
                self._last_model_values_buf = self.latest_values.new_empty(empty_shape)
            self.last_model_values = self._last_model_values_buf

        compact_public_preflop = (
            beliefs.shape[-1] == PREFLOP_HANDS and isinstance(self.env, PBSEnv)
        )
        if self.showdown_indices.numel() > 0 and not compact_public_preflop:
            showdown_beliefs = beliefs[self.showdown_indices]
            if self.num_players == 2:
                showdown_values = self._showdown_value_both(showdown_beliefs)
                self.latest_values[self.showdown_indices] = showdown_values
            elif isinstance(self.env, PBSEnv):
                showdown_rewards = self.env.expected_showdown_rewards(
                    showdown_beliefs,
                    env_indices=self.showdown_indices,
                )
                self.latest_values[self.showdown_indices] = showdown_rewards[:, :, None]
            else:
                raise NotImplementedError(
                    "Multiway showdown values require a PBSEnv-backed public state."
                )
        set_allin = getattr(self, "_set_allin_call_values", None)
        if set_allin is not None:
            set_allin(beliefs)

    def cfr_iteration(self, t: int) -> None:
        if self.cfr_type == CFRType.linear or (
            self.cfr_type in (CFRType.pcfr, CFRType.sapcfr)
            and not self._predictive_cfr_uses_dcfr()
        ):
            CFREvaluator.cfr_iteration(self, t)
            return

        self.apply_schedules(t)
        self._refresh_fused_t_scalars(t)

        sample_mask = self.t_sample == t
        torch.where(
            sample_mask[:, None],
            self.policy_probs,
            self.policy_probs_sample,
            out=self.policy_probs_sample,
        )
        torch.where(
            sample_mask[:, None, None],
            self.beliefs,
            self.beliefs_sample,
            out=self.beliefs_sample,
        )

        self._prepare_tree_slices()
        top = self._top
        child_offsets_top = self._child_offsets_top
        child_count_top = self._child_count_top
        to_act_top = self._to_act_top
        assert top is not None
        assert child_offsets_top is not None
        assert child_count_top is not None
        assert to_act_top is not None

        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
        unblocked_reach = self._preflop_unblocked_mass(beliefs[:top])
        player_ids = torch.arange(self.num_players, device=self.device)
        other_live = player_ids[None, :, None] != to_act_top[:, None, None]
        if hasattr(self.env, "has_folded"):
            other_live &= ~self.env.has_folded[:top, :, None]
        src_weights = torch.where(
            other_live,
            unblocked_reach.clamp_min(1e-12),
            torch.ones_like(unblocked_reach),
        ).prod(dim=1)
        src_weights *= self.allowed_hands[:top].to(dtype=self.float_dtype)

        positive_regrets_out = self._ensure_positive_regrets_buf()
        last_regrets = (
            self._ensure_last_instantaneous_regrets_buf()
            if getattr(self, "_predictive_cfr_enabled", False)
            else None
        )
        fused_compact_regret_dcfr_update_multiway_with_tensors_(
            src_weights=src_weights.contiguous(),
            values_achieved=self.latest_values.contiguous(),
            values_expected=self.latest_values[:top].contiguous(),
            to_act=to_act_top,
            child_offsets=child_offsets_top,
            child_count=child_count_top,
            prev_actor=self.prev_actor.contiguous(),
            cumulative_regrets=self.cumulative_regrets,
            t_alpha_num=self._t_scalars.t_alpha_num,
            t_beta_num=self._t_scalars.t_beta_num,
            t_alpha_den=self._t_scalars.t_alpha_den,
            t_beta_den=self._t_scalars.t_beta_den,
            apply_dcfr=self._uses_dcfr_backbone(),
            cfr_plus=self.cfr_plus,
            max_children=self.num_actions,
            positive_regrets_out=positive_regrets_out,
            last_instantaneous_regrets=last_regrets,
            prediction_scale=self._t_scalars.predictive_scale,
            current_player=self._t_scalars.current_player,
            block_h=256,
        )
        self._fused_positive_regrets_valid = True

        if t in self._record_stats_percentile_ts():
            old_policy_probs = self.policy_probs.clone()
            self.update_policy(t)
            self._record_stats(t, old_policy_probs)
        else:
            self.update_policy(t)

        self.set_leaf_values(t)
        self.compute_expected_values()

    def _record_stats(self, t: int, old_policy_probs: torch.Tensor) -> None:
        return None

    def sample_leaves(self, training_mode: bool):
        pbs = SparseCFREvaluator.sample_leaves(self, training_mode)
        keep = pbs.env.street == 0
        if bool(keep.all().item()):
            return pbs
        keep_indices = torch.where(keep)[0]
        filtered = pbs.beliefs[keep_indices].clone()
        out = type(pbs).from_proto(
            env_proto=pbs.env,
            beliefs=filtered,
            num_envs=keep_indices.numel(),
        )
        out.env.copy_state_from(
            pbs.env,
            keep_indices,
            torch.arange(keep_indices.numel(), device=self.device),
        )
        return out

    def evaluate_cfr(
        self, training_mode: bool = True, sample_continuation: bool = True
    ):
        return CFREvaluator.evaluate_cfr(self, training_mode, sample_continuation)

    def _compute_exploitability(self) -> ExploitabilityStats:
        local = torch.zeros(
            self.root_nodes, dtype=self.float_dtype, device=self.device
        )
        br_values = torch.zeros(
            self.root_nodes,
            self.num_players,
            dtype=self.float_dtype,
            device=self.device,
        )
        return ExploitabilityStats(
            local_exploitability=local,
            local_best_response_values=br_values,
        )

    def _root_leaf_target_source_counts(self, num_roots: int) -> dict[str, torch.Tensor]:
        device = self.device
        return {
            "leaf_total_count": torch.zeros(num_roots, dtype=torch.long, device=device),
            f"leaf_target_source_{TARGET_SOURCE_CFR_BACKUP}_count": torch.zeros(
                num_roots, dtype=torch.long, device=device
            ),
            f"leaf_target_source_{TARGET_SOURCE_EXACT_TERMINAL}_count": torch.zeros(
                num_roots, dtype=torch.long, device=device
            ),
            f"leaf_target_source_{TARGET_SOURCE_CLOSING_NET}_count": torch.zeros(
                num_roots, dtype=torch.long, device=device
            ),
        }
