from __future__ import annotations

import os

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
from p2.search.cfr_evaluator import CFREvaluator, ExploitabilityStats, PublicBeliefState
from p2.search.fused_cfr_triton import (
    GraphedCFRIteration,
    fused_average_policy_mix_multiway_with_tensors_,
    fused_average_policy_reach_beliefs_depth_preflop_multiway_,
    fused_avg_values_multiway_,
    fused_compact_regret_dcfr_update_multiway_with_tensors_,
    fused_hu_closing_postprocess_writeback_multiway_,
    fused_hu_closing_selected_beliefs_writeback_multiway_,
    fused_model_values_postprocess_writeback_multiway_,
    fused_model_values_writeback_multiway_,
    fused_parent_sum_divide_,
    fused_preflop169_project_rows_,
    fused_preflop169_parent_sum_opp_rank_stats_,
    fused_preflop169_parent_sum_opp_stats_,
    fused_preflop169_parent_sum_opp_,
    fused_preflop169_src_weights_rank_stats_multiway_,
    fused_preflop169_src_weights_stats_multiway_,
    fused_preflop169_src_weights_from_unblocked_multiway_,
    fused_preflop169_src_weights_multiway_,
    fused_policy_reach_beliefs_depth_preflop_multiway_,
    fused_policy_renorm_reach_depth_multiway_,
    fused_preflop_multiway_beliefs_from_reach_,
    fused_preflop_sample_snapshot_multiway_,
    fused_regret_tail_multiway_,
    preflop169_unblocked_rank_stats_out_,
    preflop169_unblocked_rank_mass_triton_out_,
    preflop169_unblocked_mass_triton_out_,
    preflop169_unblocked_stats_out_,
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
        if self.hand_dim != PREFLOP_HANDS:
            raise ValueError(
                "FusedPreflopSparseCFREvaluator is compact-only; attach a "
                f"{PREFLOP_HANDS}-hand preflop policy/value model"
            )
        self._ensure_fused_attrs()
        self.warm_start_iterations = 0
        self._preflop_ev_actor_beliefs_buf: torch.Tensor | None = None
        self._preflop_ev_marginal_policy_buf: torch.Tensor | None = None
        self._preflop_ev_marginal_action_policy_buf: torch.Tensor | None = None
        self._preflop_ev_numer_unblocked_buf: torch.Tensor | None = None
        self._preflop_ev_denom_unblocked_buf: torch.Tensor | None = None
        self._preflop_ev_actor_stats_buf: torch.Tensor | None = None
        self._preflop_ev_marginal_stats_buf: torch.Tensor | None = None
        self._preflop_ev_actor_rank_stats_buf: torch.Tensor | None = None
        self._preflop_ev_marginal_rank_stats_buf: torch.Tensor | None = None
        self._preflop_regret_src_weights_buf: torch.Tensor | None = None
        self._preflop_regret_src_stats_buf: torch.Tensor | None = None
        self._preflop_regret_src_rank_stats_buf: torch.Tensor | None = None
        self._preflop_allowed_float_buf: torch.Tensor | None = None
        self._preflop_allowed_float_key: tuple[int, int, int, torch.dtype] | None = None
        self._preflop_player_ids_buf: torch.Tensor | None = None
        self._preflop_model_beliefs_buf: torch.Tensor | None = None
        self._preflop_model_beliefs_key: tuple[int, int, int, int, torch.dtype] | None = None
        self._model_leaf_duplicate_src: torch.Tensor | None = None
        self._model_leaf_duplicate_dst: torch.Tensor | None = None
        self._model_leaf_duplicate_copy_buf: torch.Tensor | None = None
        self._preflop_cutoff_last_values_buf: torch.Tensor | None = None
        self._preflop_new_street_baseline_last_values_buf: torch.Tensor | None = None
        self._preflop_new_street_last_values_buf: torch.Tensor | None = None
        self._preflop_partition_last_values_valid = False
        self._preflop_partition_last_values_marker: torch.Tensor | None = None
        self._preflop_partition_feature_cache: dict[
            tuple[int, int, int, int, int, int],
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ] = {}
        self._preflop_partition_node_cache: dict[
            tuple[int, int, int, int], torch.Tensor
        ] = {}
        self._preflop_partition_beliefs_cache: dict[
            tuple[int, int, int, int, torch.dtype], torch.Tensor
        ] = {}

    @property
    def _compact_preflop(self) -> bool:
        return True

    def _ensure_fused_attrs(self) -> None:
        super()._ensure_fused_attrs()
        if not hasattr(self, "_preflop_partition_feature_cache"):
            self._preflop_partition_feature_cache = {}
        if not hasattr(self, "_preflop_partition_node_cache"):
            self._preflop_partition_node_cache = {}
        if not hasattr(self, "_preflop_partition_beliefs_cache"):
            self._preflop_partition_beliefs_cache = {}
        if not hasattr(self, "_model_leaf_duplicate_src"):
            self._model_leaf_duplicate_src = None
        if not hasattr(self, "_model_leaf_duplicate_dst"):
            self._model_leaf_duplicate_dst = None
        if not hasattr(self, "_model_leaf_duplicate_copy_buf"):
            self._model_leaf_duplicate_copy_buf = None
        if not hasattr(self, "_preflop_partition_last_values_valid"):
            self._preflop_partition_last_values_valid = False
        if not hasattr(self, "_preflop_partition_last_values_marker"):
            self._preflop_partition_last_values_marker = None

    def _invalidate_subgame_caches(self) -> None:
        super()._invalidate_subgame_caches()
        self._ensure_fused_attrs()
        self._preflop_partition_feature_cache.clear()
        self._preflop_partition_node_cache.clear()
        self._preflop_partition_beliefs_cache.clear()

    def _construct_subgame(
        self,
        src_env: HUNLTensorEnv | PBSEnv,
        src_indices: torch.Tensor,
    ) -> None:
        PreflopSparseCFREvaluator._construct_subgame(self, src_env, src_indices)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_shape = (top, PREFLOP_HANDS)
        child_shape = (num_children, PREFLOP_HANDS)
        actor_buf = getattr(self, "_preflop_ev_actor_beliefs_buf", None)
        denom_buf = getattr(self, "_preflop_ev_denom_unblocked_buf", None)
        marginal_buf = getattr(self, "_preflop_ev_marginal_policy_buf", None)
        marginal_action_buf = getattr(
            self,
            "_preflop_ev_marginal_action_policy_buf",
            None,
        )
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
        if (
            marginal_action_buf is None
            or marginal_action_buf.shape != (num_children,)
            or marginal_action_buf.device != self.device
        ):
            marginal_action_buf = self.policy_probs.new_empty((num_children,))
            self._preflop_ev_marginal_action_policy_buf = marginal_action_buf
        return (
            actor_buf,
            marginal_buf,
            denom_buf,
            numer_buf,
            marginal_action_buf,
        )

    def _preflop_use_fused_projection(self) -> bool:
        flag = os.environ.get("P2_PREFLOP_FUSED_PROJECTION", "0")
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _preflop_use_fused_src_weights(self, top: int) -> bool:
        flag = os.environ.get("P2_PREFLOP_FUSED_SRC_WEIGHTS")
        if flag is None or flag.strip().lower() == "auto":
            threshold = int(os.environ.get("P2_PREFLOP_FUSED_SRC_WEIGHTS_MAX_TOP", "0"))
            return top <= threshold
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _preflop_use_fused_src_weight_tail(self) -> bool:
        flag = os.environ.get("P2_PREFLOP_FUSED_SRC_WEIGHT_TAIL", "1")
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _preflop_use_rank_stats_src_weights(self) -> bool:
        flag = os.environ.get("P2_PREFLOP_RANK_STATS_SRC_WEIGHTS", "1")
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _preflop_use_rank_stats_ev(self) -> bool:
        flag = os.environ.get("P2_PREFLOP_RANK_STATS_EV", "1")
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _preflop_use_rank_stats_parent_ev(self) -> bool:
        flag = os.environ.get("P2_PREFLOP_RANK_STATS_PARENT_EV", "1")
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _preflop_use_compact_stats_src_weights(self) -> bool:
        flag = os.environ.get("P2_PREFLOP_COMPACT_STATS_SRC_WEIGHTS", "0")
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _preflop_use_compact_stats_unblocked(self) -> bool:
        flag = os.environ.get("P2_PREFLOP_COMPACT_STATS_UNBLOCKED", "0")
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _preflop_use_compact_stats_ev(self) -> bool:
        flag = os.environ.get("P2_PREFLOP_COMPACT_STATS_EV", "0")
        return flag.strip().lower() not in {"0", "false", "off", "no"}

    def _ensure_preflop_regret_src_weights_buf(self, top: int) -> torch.Tensor:
        shape = (top, PREFLOP_HANDS)
        buf = getattr(self, "_preflop_regret_src_weights_buf", None)
        if buf is None or buf.shape != shape or buf.device != self.device:
            buf = self.policy_probs.new_empty(shape)
            self._preflop_regret_src_weights_buf = buf
        return buf

    def _ensure_preflop_stats_buf(
        self,
        attr: str,
        rows: int,
        width: int = 53,
    ) -> torch.Tensor:
        shape = (rows, width)
        buf = getattr(self, attr, None)
        if (
            buf is None
            or buf.shape != shape
            or buf.device != self.device
            or buf.dtype != self.float_dtype
        ):
            buf = self.policy_probs.new_empty(shape)
            setattr(self, attr, buf)
        return buf

    def _ensure_preflop_regret_src_stats_buf(self, rows: int) -> torch.Tensor:
        return self._ensure_preflop_stats_buf("_preflop_regret_src_stats_buf", rows)

    def _ensure_preflop_regret_src_rank_stats_buf(self, rows: int) -> torch.Tensor:
        return self._ensure_preflop_stats_buf(
            "_preflop_regret_src_rank_stats_buf",
            rows,
            14,
        )

    def _ensure_preflop_ev_stats_bufs(
        self,
        top: int,
        num_children: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_stats = self._ensure_preflop_stats_buf(
            "_preflop_ev_actor_stats_buf",
            top,
        )
        marginal_stats = self._ensure_preflop_stats_buf(
            "_preflop_ev_marginal_stats_buf",
            num_children,
        )
        return actor_stats, marginal_stats

    def _ensure_preflop_ev_rank_stats_bufs(
        self,
        top: int,
        num_children: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_stats = self._ensure_preflop_stats_buf(
            "_preflop_ev_actor_rank_stats_buf",
            top,
            14,
        )
        marginal_stats = self._ensure_preflop_stats_buf(
            "_preflop_ev_marginal_rank_stats_buf",
            num_children,
            14,
        )
        return actor_stats, marginal_stats

    def _preflop_allowed_float(self, top: int) -> torch.Tensor:
        allowed = self.allowed_hands[:top]
        key = (
            int(getattr(self, "_subgame_generation", 0)),
            int(allowed.data_ptr()),
            int(top),
            self.float_dtype,
        )
        buf = getattr(self, "_preflop_allowed_float_buf", None)
        if (
            buf is None
            or getattr(self, "_preflop_allowed_float_key", None) != key
            or buf.shape != (top, PREFLOP_HANDS)
            or buf.device != self.device
        ):
            buf = allowed.to(dtype=self.float_dtype).contiguous()
            self._preflop_allowed_float_buf = buf
            self._preflop_allowed_float_key = key
        return buf

    def _preflop_player_ids(self) -> torch.Tensor:
        player_ids = getattr(self, "_preflop_player_ids_buf", None)
        if (
            player_ids is None
            or player_ids.numel() != self.num_players
            or player_ids.device != self.device
        ):
            player_ids = torch.arange(self.num_players, device=self.device)
            self._preflop_player_ids_buf = player_ids
        return player_ids

    def _preflop_regret_src_weights(
        self,
        beliefs: torch.Tensor,
        top: int,
        to_act_top: torch.Tensor,
    ) -> torch.Tensor:
        if beliefs.device.type == "cuda" and self._preflop_use_rank_stats_src_weights():
            out = self._ensure_preflop_regret_src_weights_buf(top)
            fused_preflop169_src_weights_rank_stats_multiway_(
                class_mass=beliefs[:top].contiguous(),
                to_act=to_act_top,
                allowed_weight=self._preflop_allowed_float(top),
                out=out,
                stats_out=self._ensure_preflop_regret_src_rank_stats_buf(
                    top * self.num_players
                ),
            )
            return out

        if (
            beliefs.device.type == "cuda"
            and self._preflop_use_compact_stats_src_weights()
        ):
            out = self._ensure_preflop_regret_src_weights_buf(top)
            fused_preflop169_src_weights_stats_multiway_(
                class_mass=beliefs[:top].contiguous(),
                to_act=to_act_top,
                allowed_weight=self._preflop_allowed_float(top),
                out=out,
                stats_out=self._ensure_preflop_regret_src_stats_buf(
                    top * self.num_players
                ),
            )
            return out

        if beliefs.device.type == "cuda" and self._preflop_use_fused_src_weights(top):
            out = self._ensure_preflop_regret_src_weights_buf(top)
            projection = self._preflop_unblocked_projection_for(beliefs).contiguous()
            fused_preflop169_src_weights_multiway_(
                class_mass=beliefs[:top].contiguous(),
                projection=projection,
                to_act=to_act_top,
                allowed_mask=self.allowed_hands[:top].contiguous(),
                out=out,
            )
            return out

        if (
            beliefs.device.type == "cuda"
            and self._preflop_use_compact_stats_unblocked()
        ):
            unblocked_reach = torch.empty_like(beliefs[:top])
            flat_in = beliefs[:top].contiguous().view(top * self.num_players, -1)
            flat_out = unblocked_reach.view(top * self.num_players, -1)
            preflop169_unblocked_mass_triton_out_(
                flat_in,
                flat_out,
                stats_out=self._ensure_preflop_regret_src_stats_buf(
                    top * self.num_players
                ),
            )
        else:
            unblocked_reach = self._preflop_unblocked_mass(beliefs[:top]).contiguous()
        if beliefs.device.type == "cuda" and self._preflop_use_fused_src_weight_tail():
            out = self._ensure_preflop_regret_src_weights_buf(top)
            fused_preflop169_src_weights_from_unblocked_multiway_(
                unblocked=unblocked_reach,
                to_act=to_act_top,
                allowed_weight=self._preflop_allowed_float(top),
                out=out,
            )
            return out

        unblocked_reach.clamp_min_(1e-12)
        player_ids = self._preflop_player_ids()
        other_live = player_ids[None, :, None] != to_act_top[:, None, None]
        src_weights = torch.where(
            other_live,
            unblocked_reach,
            1.0,
        ).prod(dim=1)
        src_weights *= self._preflop_allowed_float(top)
        return src_weights.contiguous()

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
        self._init_fused_feature_encoders()
        PreflopSparseCFREvaluator._validate_compact_shapes(self)
        self._prepare_tree_slices()
        self._reset_average_policy_accumulators()

    def _prepare_compact_leaf_sampling(self, training_mode: bool) -> None:
        super()._prepare_compact_leaf_sampling(training_mode)
        assert self._sample_leaf_players is not None
        self._sample_leaf_players.random_(0, self.num_players, generator=self.generator)

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
            value_encoder = self.value_feature_encoder
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

    def _features_for_model_positions(
        self,
        features: MLPFeatures,
        positions: torch.Tensor,
        encoder=None,
    ) -> MLPFeatures:
        if (
            encoder is not None
            or not positions.is_contiguous()
            or not features.beliefs.is_contiguous()
            or features.hand_dim != PREFLOP_HANDS
        ):
            return super()._features_for_model_positions(
                features,
                positions,
                encoder,
            )
        self._ensure_fused_attrs()
        rows = int(positions.numel())
        feature_key = (
            int(self._subgame_generation),
            int(features.context.data_ptr()),
            int(features.beliefs.data_ptr()),
            int(positions.data_ptr()),
            rows,
            int(features.hand_dim),
        )
        cached = self._preflop_partition_feature_cache.get(feature_key)
        if cached is None:
            belief_shape = (rows, features.beliefs.shape[1])
            belief_buf = features.beliefs.new_empty(belief_shape)
            cached = (
                torch.index_select(features.context, 0, positions),
                torch.index_select(features.street, 0, positions),
                torch.index_select(features.to_act, 0, positions),
                torch.index_select(features.board, 0, positions),
                belief_buf,
            )
            self._preflop_partition_feature_cache[feature_key] = cached
        ctx, street, to_act, board, belief_buf = cached
        torch.index_select(features.beliefs, 0, positions, out=belief_buf)
        return MLPFeatures(
            context=ctx,
            street=street,
            to_act=to_act,
            board=board,
            beliefs=belief_buf,
            hand_dim=features.hand_dim,
        )

    def _model_beliefs_for_values(self, beliefs: torch.Tensor) -> torch.Tensor:
        if (
            beliefs is self.beliefs
            and self._model_leaf_scatter_enabled
            and self._model_leaf_beliefs_valid
            and self._model_leaf_beliefs_buf is not None
            and self._model_leaf_beliefs_buf.shape
            == (int(self.model_indices.numel()), self.num_players, PREFLOP_HANDS)
            and self._model_leaf_beliefs_buf.device == beliefs.device
            and self._model_leaf_beliefs_buf.dtype == beliefs.dtype
        ):
            return self._model_leaf_beliefs_buf
        if (
            os.environ.get("P2_PREFLOP_MODEL_BELIEF_GATHER_OUT", "1")
            .strip()
            .lower()
            in {"0", "false", "off", "no"}
        ):
            return beliefs[self.model_indices]
        m = int(self.model_indices.numel())
        if beliefs.dim() != 3 or beliefs.shape[1:] != (
            self.num_players,
            PREFLOP_HANDS,
        ):
            return beliefs[self.model_indices]
        shape = (m, self.num_players, PREFLOP_HANDS)
        key = (
            int(self._subgame_generation),
            int(beliefs.data_ptr()),
            int(self.model_indices.data_ptr()),
            m,
            beliefs.dtype,
        )
        buf = getattr(self, "_preflop_model_beliefs_buf", None)
        if (
            buf is None
            or getattr(self, "_preflop_model_beliefs_key", None) != key
            or buf.shape != shape
            or buf.device != beliefs.device
            or buf.dtype != beliefs.dtype
        ):
            self._preflop_model_beliefs_buf = beliefs.new_empty(shape)
            self._preflop_model_beliefs_key = key
            buf = self._preflop_model_beliefs_buf
        assert buf is not None
        return torch.index_select(
            beliefs,
            0,
            self.model_indices.contiguous(),
            out=buf,
        )

    def _ensure_model_leaf_belief_buffers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[bool, ...]]:
        m = int(self.model_indices.numel())
        shape = (m, self.num_players, PREFLOP_HANDS)
        if (
            self._model_leaf_beliefs_buf is None
            or self._model_leaf_beliefs_buf.shape != shape
            or self._model_leaf_beliefs_buf.dtype != self.beliefs.dtype
            or self._model_leaf_beliefs_buf.device != self.beliefs.device
        ):
            self._model_leaf_beliefs_buf = self.beliefs.new_empty(shape)
            self._model_leaf_beliefs_valid = False

        total = int(self.beliefs.shape[0])
        key = (
            int(self._subgame_generation),
            int(self.model_indices.data_ptr()),
            m,
            total,
        )
        if self._model_leaf_slot_key != key or self._model_leaf_slot is None:
            slot = torch.full(
                (total,),
                -1,
                device=self.device,
                dtype=torch.int64,
            )
            if m > 0:
                slot[self.model_indices] = torch.arange(
                    m,
                    device=self.device,
                    dtype=torch.int64,
                )
                canonical_slots = slot[self.model_indices]
                model_slots = torch.arange(
                    m,
                    device=self.device,
                    dtype=torch.int64,
                )
                duplicate_mask = canonical_slots != model_slots
                self._model_leaf_duplicate_src = canonical_slots[
                    duplicate_mask
                ].contiguous()
                self._model_leaf_duplicate_dst = model_slots[
                    duplicate_mask
                ].contiguous()
                self._model_leaf_beliefs_buf.copy_(
                    self._model_beliefs_for_values(self.beliefs)
                )
                model_indices_cpu = self.model_indices.detach().cpu()
                depth_has_slot = []
                for depth in range(self.tree_depth):
                    start = int(self.depth_offsets[depth + 1])
                    end = int(self.depth_offsets[depth + 2])
                    in_depth = (model_indices_cpu >= start) & (model_indices_cpu < end)
                    depth_has_slot.append(bool(in_depth.any().item()))
                self._model_leaf_depth_has_slot = tuple(depth_has_slot)
            else:
                self._model_leaf_depth_has_slot = (False,) * int(self.tree_depth)
                self._model_leaf_duplicate_src = None
                self._model_leaf_duplicate_dst = None
            self._model_leaf_slot = slot.contiguous()
            self._model_leaf_slot_key = key
            self._model_leaf_beliefs_valid = False

        return (
            self._model_leaf_beliefs_buf,
            self._model_leaf_slot,
            self._model_leaf_depth_has_slot,
        )

    def _copy_duplicate_model_leaf_belief_slots(self, leaf_out: torch.Tensor) -> None:
        src = self._model_leaf_duplicate_src
        dst = self._model_leaf_duplicate_dst
        if src is None or dst is None or src.numel() == 0:
            return
        shape = (int(src.numel()), leaf_out.shape[1], leaf_out.shape[2])
        buf = self._model_leaf_duplicate_copy_buf
        if (
            buf is None
            or buf.shape != shape
            or buf.dtype != leaf_out.dtype
            or buf.device != leaf_out.device
        ):
            buf = leaf_out.new_empty(shape)
            self._model_leaf_duplicate_copy_buf = buf
        torch.index_select(leaf_out, 0, src, out=buf)
        leaf_out.index_copy_(0, dst, buf)

    def _init_hand_rank_data(self) -> None:
        PreflopSparseCFREvaluator._init_hand_rank_data(self)

    def _empty_allin_call_partitions(self) -> None:
        PreflopSparseCFREvaluator._empty_allin_call_partitions(self)

    def _cache_preflop_allin_live_partitions(self) -> None:
        PreflopSparseCFREvaluator._cache_preflop_allin_live_partitions(self)

    def _cache_allin_call_street_partitions(self, parent_streets: torch.Tensor) -> None:
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
        if self._skip_t_scalars_update:
            return
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

        (
            actor_beliefs,
            marginal_policy,
            denom_unblocked,
            numer_unblocked,
            marginal_action_policy,
        ) = self._ensure_preflop_fused_ev_buffers(top, parent_index_bottom.numel())
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
        if beliefs_c.device.type == "cuda" and self._preflop_use_rank_stats_ev():
            actor_rank_stats, marginal_rank_stats = (
                self._ensure_preflop_ev_rank_stats_bufs(
                    top,
                    parent_index_bottom.numel(),
                )
            )
            if self._preflop_use_rank_stats_parent_ev():
                preflop169_unblocked_rank_stats_out_(actor_beliefs, actor_rank_stats)
                preflop169_unblocked_rank_stats_out_(
                    marginal_policy,
                    marginal_rank_stats,
                )
                for depth in range(self.tree_depth - 1, -1, -1):
                    fused_preflop169_parent_sum_opp_rank_stats_(
                        values=values,
                        prev_actor=prev_actor_c,
                        policy=policy_c,
                        actor_beliefs=actor_beliefs,
                        marginal_policy=marginal_policy,
                        actor_stats=actor_rank_stats,
                        marginal_stats=marginal_rank_stats,
                        child_offsets=self._child_offsets_by_depth[depth],
                        child_count=self._child_count_by_depth[depth],
                        parent_base=self.depth_offsets[depth],
                        child_base=bottom,
                        max_children=self.num_actions,
                        max_children_pow2=self._child_count_pow2_by_depth[depth],
                        leaf_values=leaf_values if use_leaf_source else None,
                        leaf_mask=(
                            self.leaf_mask.contiguous() if use_leaf_source else None
                        ),
                        has_folded=(
                            self.env.has_folded.contiguous()
                            if hasattr(self.env, "has_folded")
                            else None
                        ),
                    )
            else:
                preflop169_unblocked_rank_mass_triton_out_(
                    actor_beliefs,
                    denom_unblocked,
                    stats_out=actor_rank_stats,
                )
                preflop169_unblocked_rank_mass_triton_out_(
                    marginal_policy,
                    numer_unblocked,
                    stats_out=marginal_rank_stats,
                    row_sum=marginal_action_policy,
                )
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
                        leaf_mask=(
                            self.leaf_mask.contiguous() if use_leaf_source else None
                        ),
                        has_folded=(
                            self.env.has_folded.contiguous()
                            if hasattr(self.env, "has_folded")
                            else None
                        ),
                    )
        elif self._preflop_use_compact_stats_ev():
            actor_stats, marginal_stats = self._ensure_preflop_ev_stats_bufs(
                top,
                parent_index_bottom.numel(),
            )
            preflop169_unblocked_stats_out_(actor_beliefs, actor_stats)
            preflop169_unblocked_stats_out_(marginal_policy, marginal_stats)
            for depth in range(self.tree_depth - 1, -1, -1):
                fused_preflop169_parent_sum_opp_stats_(
                    values=values,
                    prev_actor=prev_actor_c,
                    policy=policy_c,
                    actor_beliefs=actor_beliefs,
                    marginal_policy=marginal_policy,
                    actor_stats=actor_stats,
                    marginal_stats=marginal_stats,
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
        else:
            projection = self._preflop_unblocked_projection_for(beliefs_c).contiguous()
            if self._preflop_use_fused_projection():
                fused_preflop169_project_rows_(
                    actor_beliefs,
                    projection,
                    denom_unblocked,
                )
                fused_preflop169_project_rows_(
                    marginal_policy,
                    projection,
                    numer_unblocked,
                    row_sum=marginal_action_policy,
                )
            else:
                torch.mm(actor_beliefs, projection, out=denom_unblocked)
                torch.mm(marginal_policy, projection, out=numer_unblocked)
                torch.sum(marginal_policy, dim=-1, out=marginal_action_policy)

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
        src_weights = self._preflop_regret_src_weights(
            beliefs,
            top,
            to_act_top,
        )

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
        write_model_leaf_beliefs = (
            beliefs is self.beliefs
            and self._model_leaf_scatter_enabled
            and int(self.model_indices.numel()) > 0
        )
        if beliefs is self.beliefs:
            self._model_leaf_beliefs_valid = False
        leaf_out = None
        leaf_slot = None
        leaf_depth_has_slot: tuple[bool, ...] = ()
        if write_model_leaf_beliefs:
            leaf_out, leaf_slot, leaf_depth_has_slot = (
                self._ensure_model_leaf_belief_buffers()
            )
        for depth in range(self.tree_depth):
            scatter_depth = (
                depth < len(leaf_depth_has_slot) and leaf_depth_has_slot[depth]
            )
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
                leaf_slot=leaf_slot if scatter_depth else None,
                leaf_out=leaf_out if scatter_depth else None,
            )
        if beliefs is self.beliefs:
            if leaf_out is not None:
                self._copy_duplicate_model_leaf_belief_slots(leaf_out)
            self._model_leaf_beliefs_valid = leaf_out is not None

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
            self._renormalize_policy_reach(self.policy_probs_avg, self.self_reach_avg)
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

    def update_average_values(self, t: int, *, refresh_t_scalars: bool = True) -> None:
        if refresh_t_scalars:
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

    def _use_partitioned_model_writeback(self) -> bool:
        return (
            os.environ.get("P2_PREFLOP_PARTITIONED_MODEL_WRITEBACK", "1")
            .strip()
            .lower()
            not in {"0", "false", "off", "no"}
        )

    def _reuse_cutoff_feature_beliefs_for_writeback(self) -> bool:
        return (
            os.environ.get("P2_PREFLOP_REUSE_CUTOFF_FEATURE_BELIEFS", "1")
            .strip()
            .lower()
            not in {"0", "false", "off", "no"}
        )

    def _use_selected_hu_closing_beliefs_for_writeback(self) -> bool:
        return (
            os.environ.get("P2_PREFLOP_SELECTED_HU_CLOSING_BELIEFS", "1")
            .strip()
            .lower()
            not in {"0", "false", "off", "no"}
        )

    def _use_fused_sample_snapshot(self) -> bool:
        return (
            os.environ.get("P2_PREFLOP_FUSED_SAMPLE_SNAPSHOT", "1")
            .strip()
            .lower()
            not in {"0", "false", "off", "no"}
        )

    def _partition_last_values_buffer(
        self,
        attr: str,
        shape: tuple[int, int, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = getattr(self, attr, None)
        if (
            buf is None
            or buf.shape != shape
            or buf.dtype != dtype
            or buf.device != self.latest_values.device
        ):
            buf = self.latest_values.new_empty(shape, dtype=dtype)
            setattr(self, attr, buf)
            self._preflop_partition_last_values_valid = False
        return buf

    def _partition_node_indices_for_positions(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_fused_attrs()
        key = (
            int(self._subgame_generation),
            int(self.model_indices.data_ptr()),
            int(positions.data_ptr()),
            int(positions.numel()),
        )
        out = self._preflop_partition_node_cache.get(key)
        if out is None:
            out = torch.index_select(self.model_indices, 0, positions).contiguous()
            self._preflop_partition_node_cache[key] = out
        return out

    def _partition_beliefs_for_positions(
        self,
        beliefs_at_model: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_fused_attrs()
        rows = int(positions.numel())
        shape = (rows, self.num_players, self.hand_dim)
        key = (
            int(beliefs_at_model.data_ptr()),
            int(positions.data_ptr()),
            rows,
            int(self._subgame_generation),
            beliefs_at_model.dtype,
        )
        out = self._preflop_partition_beliefs_cache.get(key)
        if (
            out is None
            or out.shape != shape
            or out.device != beliefs_at_model.device
            or out.dtype != beliefs_at_model.dtype
        ):
            out = beliefs_at_model.new_empty(shape)
            self._preflop_partition_beliefs_cache[key] = out
        torch.index_select(beliefs_at_model, 0, positions, out=out)
        return out

    def _writeback_model_values_partition(
        self,
        *,
        hand_values: torch.Tensor,
        beliefs_at_model: torch.Tensor,
        positions: torch.Tensor,
        position_beliefs: torch.Tensor | None = None,
        last_attr: str,
        do_mix: bool,
        store_last: bool,
    ) -> None:
        if positions.numel() == 0:
            return
        if position_beliefs is None:
            position_beliefs = self._partition_beliefs_for_positions(
                beliefs_at_model,
                positions,
            )
        else:
            position_beliefs = position_beliefs.view(
                int(positions.numel()), self.num_players, self.hand_dim
            )
        node_indices = self._partition_node_indices_for_positions(positions)
        hand_values = hand_values.contiguous()
        if store_last:
            last_out = self._partition_last_values_buffer(
                last_attr,
                tuple(hand_values.shape),
                hand_values.dtype,
            )
            last_model_values = last_out if do_mix else hand_values
        else:
            last_out = hand_values
            last_model_values = hand_values
        fused_model_values_postprocess_writeback_multiway_(
            hand_values=hand_values,
            last_model_values=last_model_values,
            beliefs=position_beliefs,
            node_indices=node_indices,
            latest_values=self.latest_values,
            last_out=last_out,
            has_folded=self.env.has_folded.contiguous(),
            stacks=self.env.stacks.contiguous(),
            starting_stacks=self.env.starting_stacks.contiguous(),
            scale=self.env.scale.contiguous(),
            old_plus_new_over_new=self._t_scalars.mix_onon,
            old_over_new=self._t_scalars.mix_oon,
            do_mix=do_mix,
            store_last=store_last,
        )

    def _writeback_heads_up_closing_values_partition(
        self,
        *,
        hand_values: torch.Tensor,
        live_players: torch.Tensor,
        beliefs_at_model: torch.Tensor,
        positions: torch.Tensor,
        selected_beliefs: torch.Tensor | None = None,
        last_attr: str,
        do_mix: bool,
        store_last: bool,
    ) -> None:
        if positions.numel() == 0:
            return
        node_indices = self._partition_node_indices_for_positions(positions)
        hand_values = hand_values.contiguous()
        if store_last:
            last_shape = (hand_values.shape[0], self.num_players, self.hand_dim)
            last_out = self._partition_last_values_buffer(
                last_attr,
                last_shape,
                hand_values.dtype,
            )
            last_model_values = last_out if do_mix else last_out
        else:
            last_shape = (hand_values.shape[0], self.num_players, self.hand_dim)
            last_out = self.latest_values.new_empty(last_shape)
            last_model_values = last_out
        if (
            selected_beliefs is not None
            and self._use_selected_hu_closing_beliefs_for_writeback()
        ):
            fused_hu_closing_selected_beliefs_writeback_multiway_(
                hand_values=hand_values,
                last_model_values=last_model_values,
                selected_beliefs=selected_beliefs.view(
                    hand_values.shape[0], 2, self.hand_dim
                ).contiguous(),
                node_indices=node_indices,
                live_players=live_players.contiguous(),
                latest_values=self.latest_values,
                last_out=last_out,
                has_folded=self.env.has_folded.contiguous(),
                stacks=self.env.stacks.contiguous(),
                starting_stacks=self.env.starting_stacks.contiguous(),
                scale=self.env.scale.contiguous(),
                old_plus_new_over_new=self._t_scalars.mix_onon,
                old_over_new=self._t_scalars.mix_oon,
                do_mix=do_mix,
                store_last=store_last,
            )
        else:
            position_beliefs = self._partition_beliefs_for_positions(
                beliefs_at_model,
                positions,
            )
            fused_hu_closing_postprocess_writeback_multiway_(
                hand_values=hand_values,
                last_model_values=last_model_values,
                beliefs=position_beliefs,
                node_indices=node_indices,
                live_players=live_players.contiguous(),
                latest_values=self.latest_values,
                last_out=last_out,
                has_folded=self.env.has_folded.contiguous(),
                stacks=self.env.stacks.contiguous(),
                starting_stacks=self.env.starting_stacks.contiguous(),
                scale=self.env.scale.contiguous(),
                old_plus_new_over_new=self._t_scalars.mix_onon,
                old_over_new=self._t_scalars.mix_oon,
                do_mix=do_mix,
                store_last=store_last,
            )

    def _try_set_model_values_partitioned(
        self,
        t: int,
        beliefs: torch.Tensor,
        features: MLPFeatures,
    ) -> bool:
        if not self._use_partitioned_model_writeback():
            return False
        if self._model_scope() != "mixed_street":
            return False
        if self.closing_leaf_value_model is None:
            return False
        if not self._can_project_heads_up_closing_model():
            return False

        self._ensure_model_index_partitions()
        store_last = bool(self.cfr_avg)
        do_mix = (
            store_last
            and t > 1
            and getattr(self, "_preflop_partition_last_values_valid", False)
            and not self._average_accumulation_delayed(t)
        )
        wrote_any = False

        cutoff_positions = self.cutoff_model_positions
        if cutoff_positions.numel() > 0:
            cutoff_features = self._features_for_model_positions(
                features, cutoff_positions
            )
            cutoff_values, _ = self._eval_model_for_fused_writeback(
                self.value_model,
                cutoff_features,
                use_pre_head=False,
            )
            self._writeback_model_values_partition(
                hand_values=cutoff_values,
                beliefs_at_model=beliefs,
                positions=cutoff_positions,
                position_beliefs=(
                    cutoff_features.beliefs
                    if self._reuse_cutoff_feature_beliefs_for_writeback()
                    else None
                ),
                last_attr="_preflop_cutoff_last_values_buf",
                do_mix=do_mix,
                store_last=store_last,
            )
            wrote_any = True

        baseline_positions = self.new_street_baseline_model_positions
        if baseline_positions.numel() > 0:
            baseline_values = self._cached_stack_value_baseline_for_model_positions(
                baseline_positions,
                self.hand_dim,
            )
            self._writeback_model_values_partition(
                hand_values=baseline_values,
                beliefs_at_model=beliefs,
                positions=baseline_positions,
                last_attr="_preflop_new_street_baseline_last_values_buf",
                do_mix=do_mix,
                store_last=store_last,
            )
            wrote_any = True

        hu_positions = self.new_street_hu_model_positions
        if hu_positions.numel() > 0:
            closing_features, live_players = self._heads_up_projected_closing_features(
                features,
                hu_positions,
                self.closing_leaf_value_encoder,
                validate_live=False,
            )
            closing_values, _ = self._eval_model_for_fused_writeback(
                self.closing_leaf_value_model,
                closing_features,
                use_pre_head=False,
            )
            self._writeback_heads_up_closing_values_partition(
                hand_values=closing_values,
                live_players=live_players,
                beliefs_at_model=beliefs,
                positions=hu_positions,
                selected_beliefs=closing_features.beliefs,
                last_attr="_preflop_new_street_last_values_buf",
                do_mix=do_mix,
                store_last=store_last,
            )
            wrote_any = True

        self._preflop_partition_last_values_valid = bool(store_last and wrote_any)
        if store_last and wrote_any:
            marker = getattr(self, "_preflop_partition_last_values_marker", None)
            if marker is None:
                marker = self.latest_values.new_empty(
                    (0, self.num_players, self.hand_dim)
                )
                self._preflop_partition_last_values_marker = marker
            self.last_model_values = marker
        else:
            self.last_model_values = None
        return wrote_any

    def _set_model_values_impl(
        self,
        t: int,
        beliefs: torch.Tensor,
        features: MLPFeatures,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_fused_attrs()
        if self._try_set_model_values_partitioned(t, beliefs, features):
            last = self.last_model_values
            if last is None:
                last = self.latest_values.new_empty(
                    (0, self.num_players, self.hand_dim)
                )
            return self.latest_values, last

        hand_values, model_applied_zero_sum = (
            self._model_leaf_values_for_fused_writeback(features)
        )
        hand_values = self._postprocess_model_leaf_values(
            hand_values,
            beliefs,
            self.model_indices,
        ).contiguous()
        model_applied_zero_sum = True
        do_mix = (
            self.cfr_avg
            and t > 1
            and self.last_model_values is not None
            and not self._average_accumulation_delayed(t)
        )
        store_last = bool(self.cfr_avg)
        hand_dim = self.hand_dim
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
            and not model_applied_zero_sum
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
            beliefs_at_model = self._model_beliefs_for_values(beliefs)
            features_at_model = self._model_features_for_beliefs(beliefs_at_model)
            self._set_model_values(t, beliefs_at_model, features_at_model)
        else:
            empty_shape = (0, self.num_players, PREFLOP_HANDS)
            if self._last_model_values_buf is None or (
                self._last_model_values_buf.shape != empty_shape
            ):
                self._last_model_values_buf = self.latest_values.new_empty(empty_shape)
            self.last_model_values = self._last_model_values_buf

        compact_public_preflop = beliefs.shape[-1] == PREFLOP_HANDS and isinstance(
            self.env, PBSEnv
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
        self._ensure_fused_attrs()
        if self.cfr_type == CFRType.linear or (
            self.cfr_type in (CFRType.pcfr, CFRType.sapcfr)
            and not self._predictive_cfr_uses_dcfr()
        ):
            CFREvaluator.cfr_iteration(self, t)
            return

        if not self._skip_t_scalars_update:
            self.apply_schedules(t)
        self._refresh_fused_t_scalars(t)

        if self._use_fused_sample_snapshot():
            fused_preflop_sample_snapshot_multiway_(
                self.policy_probs,
                self.policy_probs_sample,
                self.beliefs,
                self.beliefs_sample,
                self.t_sample.contiguous(),
                self._t_scalars.t_tensor,
                block_m=8,
                block_h=256,
            )
        else:
            sample_mask = self.t_sample == self._t_scalars.t_tensor
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
        src_weights = self._preflop_regret_src_weights(
            beliefs,
            top,
            to_act_top,
        )

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
        if not self.use_final_policy_values:
            self.update_average_values(t, refresh_t_scalars=False)

    def _record_stats(self, t: int, old_policy_probs: torch.Tensor) -> None:
        return None

    def _sample_preflop_cutoff_roots(self) -> PublicBeliefState | None:
        bounds = self._continuation_value_target_depth_bounds()
        min_actions = int(bounds[0]) if bounds is not None else int(self.max_depth)
        N = int(self.root_nodes)
        total = int(self.total_nodes)
        if total <= N:
            return None

        rows = torch.arange(total, device=self.device)
        candidate_mask = (
            (rows >= N)
            & self.valid_mask
            & (self.env.street == 0)
            & (self.env.actions_this_round >= min_actions)
            & (~self.env.done)
            & (~self.allin_call_mask)
            & (self.env.to_act >= 0)
            & (self.env.to_act < self.num_players)
        )
        candidates = torch.where(candidate_mask)[0]
        if candidates.numel() == 0:
            return None

        root_owner = self._get_root_index()[candidates].clamp(min=0, max=N - 1)
        scores = torch.rand(
            candidates.numel(),
            generator=self.generator,
            device=self.device,
            dtype=self.float_dtype,
        )
        best_scores = torch.full((N,), -1.0, dtype=self.float_dtype, device=self.device)
        best_scores.scatter_reduce_(
            0, root_owner, scores, reduce="amax", include_self=True
        )
        chosen = candidates[scores >= best_scores[root_owner]]
        if chosen.numel() > N:
            chosen = chosen[:N]

        pbs = PublicBeliefState.from_proto(
            env_proto=self.env,
            beliefs=self.beliefs_sample[chosen].clone(),
            num_envs=chosen.numel(),
        )
        pbs.env.copy_state_from(
            self.env,
            chosen,
            torch.arange(chosen.numel(), device=self.device),
        )
        return pbs

    def sample_leaves(self, training_mode: bool):
        if training_mode and self._continuation_value_target_sampling_enabled():
            pbs = self._sample_preflop_cutoff_roots()
            if pbs is not None:
                return pbs
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
        if not self._preflop_cuda_graph_evaluate_enabled():
            return CFREvaluator.evaluate_cfr(self, training_mode, sample_continuation)

        self._ensure_fused_attrs()
        self.model.eval()

        self.initialize_policy_and_beliefs()
        if self.warm_start_iterations > 0:
            self.warm_start()

        # Use t=0 here so set_leaf_values doesn't do CFR-AVG de-averaging.
        self.set_leaf_values(0)
        self.compute_expected_values()
        self.values_avg[:] = self.latest_values

        self.t_sample = self._get_sampling_schedule()
        start = self.warm_start_iterations
        end = self.cfr_iterations
        stat_iters = self._record_stats_percentile_ts()

        runners: dict[str, GraphedCFRIteration] = {}
        t = start
        while t < end:
            self.profiler_step()
            regime = self._graph_capture_regime(t)
            can_capture = (
                regime is not None
                and regime not in runners
                and t + 1 < end
                and self._graph_capture_regime(t + 1) == regime
                and t not in stat_iters
                and (t + 1) not in stat_iters
            )
            if can_capture:
                runner = GraphedCFRIteration(self)
                runner.capture(t_warmup=t, t_capture=t + 1)
                runners[regime] = runner
                t += 1
                continue

            runner = runners.get(regime) if regime is not None else None
            if runner is not None and t not in stat_iters:
                runner.replay(t=t)
            else:
                self.cfr_iteration(t)
            t += 1

        if self.use_final_policy_values:
            self.update_average_values_final()

        self._record_action_mix()
        self._record_cfr_entropy()
        self._record_cumulative_regret()

        if not sample_continuation:
            return None
        return self.sample_leaves(training_mode)

    def _preflop_cuda_graph_evaluate_enabled(self) -> bool:
        graph_flag = os.environ.get("P2_PREFLOP_CUDA_GRAPH_EVALUATE")
        if graph_flag is not None and graph_flag.lower() in {"0", "false", "no", "off"}:
            return False
        if self.device.type != "cuda":
            return False
        return not (
            self.cfr_type == CFRType.linear
            or (
                self.cfr_type in (CFRType.pcfr, CFRType.sapcfr)
                and not self._predictive_cfr_uses_dcfr()
            )
        )

    def _compute_exploitability(self) -> ExploitabilityStats:
        local = torch.zeros(self.root_nodes, dtype=self.float_dtype, device=self.device)
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

    def _root_leaf_target_source_counts(
        self, num_roots: int
    ) -> dict[str, torch.Tensor]:
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
