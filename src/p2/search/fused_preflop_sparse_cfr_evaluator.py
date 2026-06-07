from __future__ import annotations

import torch

from p2.env.card_utils import NUM_HANDS
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
    _preprocess_unblocked_stats,
    fused_average_policy_mix_multiway_with_tensors_,
    fused_avg_values_multiway_,
    fused_model_values_writeback_multiway_,
    fused_policy_renorm_reach_depth_multiway_,
    fused_regret_tail_multiway_,
    fused_weighted_parent_sum_inline_opp_multiway,
    multiway_regret_src_weights_at_parents_triton,
    select_actor_beliefs_and_marginal_policy_multiway_triton_out_,
)
from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


class FusedPreflopSparseCFREvaluator(FusedSparseCFREvaluator):
    """Fused sparse evaluator boundary for multiway S_preflop training.

    This subclass uses preflop-specific multiway EV/regret kernels and routes
    PBSEnv construction through the reference sparse implementation. Policy
    averaging and model writeback stay on the reference implementation until
    the remaining heads-up fused helpers are generalized.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.warm_start_iterations = 0

    def _construct_subgame(
        self,
        src_env: HUNLTensorEnv | PBSEnv,
        src_indices: torch.Tensor,
    ) -> None:
        SparseCFREvaluator._construct_subgame(self, src_env, src_indices)

    def _continuation_value_target_sampling_enabled(self) -> bool:
        return True

    def _continuation_value_target_replace_roots(self) -> bool:
        return True

    def _continuation_value_target_streets(self) -> tuple[int, ...]:
        return (0,)

    def warm_start(self) -> None:
        return None

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
        super().initialize_subgame(src_env, src_indices, initial_beliefs)

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
            beliefs=beliefs_at_model.reshape(-1, self.num_players * NUM_HANDS),
        )

    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> None:
        CFREvaluator._calculate_reach_weights(self, target, policy)

    def _propagate_all_beliefs(
        self,
        target: torch.Tensor | None = None,
        reach_weights: torch.Tensor | None = None,
    ) -> None:
        CFREvaluator._propagate_all_beliefs(self, target, reach_weights)

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

        assert policy.is_contiguous()
        assert beliefs.is_contiguous()
        assert leaf_values.is_contiguous()
        assert values.is_contiguous()
        use_leaf_source = leaf_values is not values
        if use_leaf_source and self.tree_depth == 0:
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
        assert parent_index_bottom is not None
        assert child_offsets_top is not None
        assert child_count_top is not None
        assert to_act_top is not None

        actor_beliefs, marginal_policy = self._ensure_ev_policy_buffers(
            top,
            parent_index_bottom.numel(),
        )
        select_actor_beliefs_and_marginal_policy_multiway_triton_out_(
            beliefs,
            to_act_top,
            policy,
            child_offsets_top,
            child_count_top,
            bottom,
            actor_beliefs,
            marginal_policy,
            max_children=self.num_actions,
        )

        numer_s, numer_cardsum = _preprocess_unblocked_stats(marginal_policy)
        denom_s, denom_cardsum = _preprocess_unblocked_stats(actor_beliefs)
        leaf_mask = self.leaf_mask.contiguous()
        for depth in range(self.tree_depth - 1, -1, -1):
            fused_weighted_parent_sum_inline_opp_multiway(
                values=values,
                prev_actor=self.prev_actor.contiguous(),
                policy_hero=policy,
                actor_beliefs=actor_beliefs,
                numer_s=numer_s,
                numer_cardsum=numer_cardsum,
                denom_s=denom_s,
                denom_cardsum=denom_cardsum,
                child_offsets=self._child_offsets_by_depth[depth],
                child_count=self._child_count_by_depth[depth],
                parent_base=self.depth_offsets[depth],
                child_base=bottom,
                max_children=self.num_actions,
                max_children_pow2=self.num_actions,
                leaf_values=leaf_values if use_leaf_source else None,
                leaf_mask=leaf_mask if use_leaf_source else None,
            )

    def compute_instantaneous_regrets(
        self,
        values_achieved: torch.Tensor,
        values_expected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values_expected is None:
            values_expected = values_achieved

        self._prepare_tree_slices()
        bottom = self._bottom
        top = self._top
        parent_index_all = self._parent_index_all
        to_act_top = self._to_act_top
        assert parent_index_all is not None
        assert to_act_top is not None

        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
        regrets = torch.zeros_like(self.policy_probs)
        has_folded = getattr(self.env, "has_folded", None)
        src_weights = multiway_regret_src_weights_at_parents_triton(
            beliefs.contiguous(),
            to_act_top,
            top,
            has_folded=has_folded.contiguous() if has_folded is not None else None,
            allowed_mask=self.allowed_hands[:top].contiguous(),
        )
        fused_regret_tail_multiway_(
            regrets=regrets,
            values_achieved=values_achieved.contiguous(),
            values_expected=values_expected[:top].contiguous(),
            to_act=to_act_top,
            src_weights=src_weights,
            parent_index=parent_index_all,
            prev_actor=self.prev_actor.contiguous(),
            bottom=bottom,
        )
        self._mask_invalid(regrets)
        return regrets

    def update_policy(self, t: int) -> None:
        self._refresh_fused_t_scalars(t)
        self._regret_match_current_policy(t)
        self.update_average_policy(t, update_reach=True)
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    def update_average_policy(
        self,
        t: int,
        update_reach: bool = False,
        weight_override: float | None = None,
    ) -> None:
        defer_avg_policy = not self.cfr_avg and self.use_final_policy_values
        if self._uses_dcfr_backbone() and self._average_accumulation_delayed(t):
            if defer_avg_policy:
                self.average_policy_initialized = False
                return
            self.policy_probs_avg[:] = self.policy_probs
            self.average_policy_initialized = False
            if update_reach:
                self._calculate_reach_weights(
                    self.self_reach_avg, self.policy_probs_avg
                )
            return

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
            self._renormalize_average_policy(update_reach=update_reach)

    def _renormalize_average_policy(self, update_reach: bool) -> None:
        root_count = self.root_nodes
        self.policy_probs_avg[:root_count] = 0.0
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
                update_reach=update_reach,
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
        if store_last:
            last_shape = (hand_values.shape[0], self.num_players, NUM_HANDS)
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
            empty_shape = (0, self.num_players, NUM_HANDS)
            if self._last_model_values_buf is None or (
                self._last_model_values_buf.shape != empty_shape
            ):
                self._last_model_values_buf = self.latest_values.new_empty(empty_shape)
            self.last_model_values = self._last_model_values_buf

        if self.showdown_indices.numel() > 0:
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
        CFREvaluator.cfr_iteration(self, t)

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
