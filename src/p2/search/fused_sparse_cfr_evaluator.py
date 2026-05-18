"""FusedSparseCFREvaluator — drop-in subclass using Triton-fused kernels.

Overrides just the methods where fusion applies; every other code path
inherits unchanged from ``SparseCFREvaluator``. Semantics must match the
parent class bit-close (float rounding may differ by ~1 ULP due to fused
multiply-add in Triton).

Fusion points
-------------
* ``cfr_iteration`` — DCFR rescale + accumulate + clamp into ``fused_dcfr_update_with_tensors_``.
* ``_normalize_beliefs`` — block + normalize via ``fused_block_and_normalize_beliefs_``.
* ``_calculate_reach_weights`` — per-depth fan-out × policy fused into
  ``fused_reach_weights_depth_``.
* ``_propagate_all_beliefs`` — gather root beliefs, multiply by reach,
  block, and normalize in one ``fused_deep_beliefs_`` kernel.
* ``update_policy`` / ``update_average_policy`` — parent-aligned positive-regret
  sum + in-kernel divide via ``fused_parent_sum`` + ``fused_divide_by_parent_sum_``.
* ``update_average_policy`` — average-policy renorm + average-reach propagation
  via ``fused_policy_renorm_reach_depth_`` in the hot ``update_policy`` path.
* ``compute_expected_values`` — per-depth weight + parent-sum reduce via
  ``fused_weighted_parent_sum``.
* ``compute_instantaneous_regrets`` — fan-out + gather + sub + mul into
  ``fused_regret_tail_``.
* ``update_average_policy`` mixing — ``fused_average_policy_mix_with_tensors_``.
* ``update_average_values`` mixing — ``fused_update_average_values_with_tensors_``.
* ``_set_model_values_impl`` mixing — ``fused_model_values_mix_with_tensors``.
"""

from __future__ import annotations

import os

import torch

from p2.core.structured_config import CFRType
from p2.env.card_utils import NUM_HANDS
from p2.env.rules_triton import (
    rank_hands_triton,
    triton_is_available as _rules_triton_ok,
)
from p2.search.fused_cfr_triton import (
    fused_average_policy_mix_with_tensors_,
    fused_avg_values_zero_sum_,
    fused_br_best_action_mass,
    fused_br_finalize_depth_,
    fused_block_and_normalize_beliefs_,
    fused_deep_beliefs_,
    fused_divide_by_parent_sum_,
    fused_model_values_writeback_,
    fused_parent_sum,
    fused_parent_sum_divide_,
    fused_policy_sample_update_,
    fused_policy_renorm_reach_depth_,
    fused_regret_dcfr_update_with_tensors_,
    fused_reach_weights_depth_,
    fused_regret_tail_,
    fused_weighted_parent_sum,
    fused_weighted_parent_sum_child_opp,
    GraphedCFRIteration,
    precompute_showdown_extras,
    showdown_ev_v15,
    ShowdownGraphRunner,
    triton_is_available,
    TScalars,
    unblocked_mass_opp_at_parents_triton,
    unblocked_mass_ratio_indirect_triton,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


# Inductor folds the 7 separate aten::index ops in `set_leaf_values` (one per
# MLPFeatures field at model_indices, plus beliefs at model_indices, plus
# beliefs at showdown_indices) into a single graph. The structural win is
# that model_indices and showdown_indices are read once and reused across
# tensors instead of producing 6 redundant index gathers. dynamic=True keeps
# a single artifact across runs with different model_indices /
# showdown_indices sizes — both vary per CFR root configuration.
@torch.compile(dynamic=True)
def _set_leaf_gather(
    beliefs: torch.Tensor,
    feat_context: torch.Tensor,
    feat_street: torch.Tensor,
    feat_to_act: torch.Tensor,
    feat_board: torch.Tensor,
    feat_beliefs: torch.Tensor,
    model_indices: torch.Tensor,
    showdown_indices: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    return (
        beliefs[model_indices],
        feat_context[model_indices],
        feat_street[model_indices],
        feat_to_act[model_indices],
        feat_board[model_indices],
        feat_beliefs[model_indices],
        beliefs[showdown_indices],
    )


class FusedSparseCFREvaluator(SparseCFREvaluator):
    """SparseCFREvaluator with Triton-fused pointwise/reduction kernels.

    Requires CUDA + Triton. Falls back to parent implementation for anything
    not listed in the module docstring.
    """

    def __init__(self, *args, compile_model: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.device.type != "cuda":
            raise ValueError("FusedSparseCFREvaluator requires a CUDA device.")
        if not triton_is_available():
            raise RuntimeError(
                "Triton is not installed; FusedSparseCFREvaluator is unavailable."
            )

        # Swap in the Triton hand ranker for subgame setup. _init_hand_rank_data
        # uses the module-level binding; rebinding it module-wide is the least
        # invasive way to retarget the call. Only valid-hand relative order is
        # used downstream; blocked combos are zeroed by allowed_hands before any
        # rank-dependent cumsum.
        if _rules_triton_ok():
            import p2.search.cfr_evaluator as _ce

            if _ce.rank_hands is not rank_hands_triton:
                _ce.rank_hands = rank_hands_triton

        # Inductor-fused GEMM epilogues for the MLP forward pass. dynamic=True
        # keeps a single compiled graph as model_indices count varies. TF32 is
        # safe here: the NN only produces leaf value estimates; the precision-
        # sensitive DCFR regret accumulation stays in fp32.
        if compile_model and self.model is not None:
            torch.set_float32_matmul_precision("high")
            try:
                self.model = torch.compile(self.model, dynamic=True)
            except Exception:
                pass

        # Reused across update_policy calls to avoid reallocating the fan-out denom.
        self._fused_positive_regrets_buf: torch.Tensor | None = None
        # Lazy cache of root_index[i] = root ancestor row for node i.
        self._root_index: torch.Tensor | None = None
        self._root_index_total: int = -1
        # Device-side t-derived scalars (filled per-iteration; read by kernels
        # via pointer → full CFR iteration is CUDA-graph capturable).
        self._t_scalars = TScalars(self.device, dtype=self.float_dtype)
        # When True, cfr_iteration assumes TScalars was filled externally.
        # Set by GraphedCFRIteration during capture / before replay.
        self._skip_t_scalars_update: bool = False
        # Pre-allocated buffer used by set_leaf_values to keep
        # self.last_model_values pinned across calls (no rebinding → graph-safe).
        self._last_model_values_buf: torch.Tensor | None = None
        # When True, cfr_iteration skips the full policy_probs.clone() kept for
        # _record_stats. Set by GraphedCFRIteration when stats are stubbed out.
        self._skip_record_stats: bool = False
        self._opt_reuse_positive_regrets = _env_flag("P2_FUSED_OPT_REUSE_POSITIVE")
        self._opt_parent_sum_divide = _env_flag("P2_FUSED_OPT_PARENT_SUM_DIVIDE")
        self._opt_sparse_sample = _env_flag("P2_FUSED_OPT_SPARSE_SAMPLE")
        self._opt_leaf_feature_cache = _env_flag("P2_FUSED_OPT_LEAF_FEATURE_CACHE")
        self._opt_child_opp_policy = _env_flag("P2_FUSED_OPT_CHILD_OPP_POLICY")
        self._fused_positive_regrets_valid: bool = False
        self._sample_update_rows: torch.Tensor | None = None
        self._sample_update_counts: torch.Tensor | None = None
        self._sample_update_key: tuple[int, int, int] | None = None
        self._static_model_feature_key: tuple[int, int, int] | None = None
        self._static_model_feature_fields: tuple[torch.Tensor, ...] | None = None
        self._br_action_parent_index_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._tree_slice_key: tuple[int, ...] | None = None
        self._bottom: int = 0
        self._top: int = 0
        self._parent_index_all: torch.Tensor | None = None
        self._parent_index_bottom: torch.Tensor | None = None
        self._parent_index_nonroot: torch.Tensor | None = None
        self._child_offsets_top: torch.Tensor | None = None
        self._child_count_top: torch.Tensor | None = None
        self._to_act_top: torch.Tensor | None = None
        self._action_from_parent_all: torch.Tensor | None = None
        self._child_offsets_by_depth: tuple[torch.Tensor, ...] = ()
        self._child_count_by_depth: tuple[torch.Tensor, ...] = ()
        self._exploitability_cache_key: (
            tuple[tuple[int, int, tuple[int, ...]], ...] | None
        ) = None
        self._exploitability_cache = None

    def _init_hand_rank_data(self) -> None:
        """Build hand-rank data, then precompute the constant-per-subgame
        showdown EV inputs and capture a CUDA graph for the EV pipeline.
        The graph is keyed on (M=showdown_indices.numel(), NUM_HANDS) and
        replays via persistent buffers."""
        super()._init_hand_rank_data()
        if self.hand_rank_data is not None and self.showdown_indices.numel() > 0:
            self._showdown_extras = precompute_showdown_extras(
                self.hand_rank_data,
                self.env,
                self.showdown_indices,
            )
            self._showdown_graph_runner = ShowdownGraphRunner(
                extras=self._showdown_extras,
                M=self.showdown_indices.numel(),
                NUM_HANDS=self.beliefs.shape[-1],
                device=self.device,
            )
        else:
            self._showdown_extras = None
            self._showdown_graph_runner = None

    def _showdown_value_both(self, beliefs: torch.Tensor) -> torch.Tensor:
        """Triton + CUDA-graph fast path. Returns the runner's persistent
        output buffer — callers must consume / copy before the next call.
        Falls back to the eager-Triton path or the compiled-PyTorch
        baseline when no graph or extras are available (e.g. empty
        showdown set)."""
        runner = getattr(self, "_showdown_graph_runner", None)
        if runner is not None:
            return runner(beliefs)
        extras = getattr(self, "_showdown_extras", None)
        if extras is None:
            return super()._showdown_value_both(beliefs)
        return showdown_ev_v15(beliefs, extras)

    def _ensure_fused_attrs(self) -> None:
        """Populate optional fused-only attributes if the object was constructed
        via ``__class__``-swap (which bypasses ``__init__``). No-op otherwise.
        """
        if not hasattr(self, "_t_scalars"):
            self._t_scalars = TScalars(self.device, dtype=self.float_dtype)
        if not hasattr(self, "_skip_t_scalars_update"):
            self._skip_t_scalars_update = False
        if not hasattr(self, "_last_model_values_buf"):
            self._last_model_values_buf = None
        if not hasattr(self, "_fused_positive_regrets_buf"):
            self._fused_positive_regrets_buf = None
        if not hasattr(self, "_skip_record_stats"):
            self._skip_record_stats = False
        if not hasattr(self, "_opt_reuse_positive_regrets"):
            self._opt_reuse_positive_regrets = _env_flag("P2_FUSED_OPT_REUSE_POSITIVE")
        if not hasattr(self, "_opt_parent_sum_divide"):
            self._opt_parent_sum_divide = _env_flag("P2_FUSED_OPT_PARENT_SUM_DIVIDE")
        if not hasattr(self, "_opt_sparse_sample"):
            self._opt_sparse_sample = _env_flag("P2_FUSED_OPT_SPARSE_SAMPLE")
        if not hasattr(self, "_opt_leaf_feature_cache"):
            self._opt_leaf_feature_cache = _env_flag("P2_FUSED_OPT_LEAF_FEATURE_CACHE")
        if not hasattr(self, "_opt_child_opp_policy"):
            self._opt_child_opp_policy = _env_flag("P2_FUSED_OPT_CHILD_OPP_POLICY")
        if not hasattr(self, "_fused_positive_regrets_valid"):
            self._fused_positive_regrets_valid = False
        if not hasattr(self, "_sample_update_rows"):
            self._sample_update_rows = None
        if not hasattr(self, "_sample_update_counts"):
            self._sample_update_counts = None
        if not hasattr(self, "_sample_update_key"):
            self._sample_update_key = None
        if not hasattr(self, "_static_model_feature_key"):
            self._static_model_feature_key = None
        if not hasattr(self, "_static_model_feature_fields"):
            self._static_model_feature_fields = None
        if not hasattr(self, "_br_action_parent_index_cache"):
            self._br_action_parent_index_cache = {}
        if not hasattr(self, "_tree_slice_key"):
            self._tree_slice_key = None
        if not hasattr(self, "_child_offsets_by_depth"):
            self._child_offsets_by_depth = ()
        if not hasattr(self, "_child_count_by_depth"):
            self._child_count_by_depth = ()
        if not hasattr(self, "_action_from_parent_all"):
            self._action_from_parent_all = None
        if not hasattr(self, "_exploitability_cache_key"):
            self._exploitability_cache_key = None
        if not hasattr(self, "_exploitability_cache"):
            self._exploitability_cache = None

    def initialize_subgame(self, *args, **kwargs) -> None:
        super().initialize_subgame(*args, **kwargs)
        self._prepare_tree_slices()
        self._reset_average_policy_accumulators()

    def _prepare_tree_slices(self) -> None:
        """Cache static contiguous tree slices for the current sparse subgame."""
        bottom = (
            self.depth_offsets[1] if len(self.depth_offsets) > 1 else self.root_nodes
        )
        top = self.depth_offsets[-2] if len(self.depth_offsets) > 1 else self.root_nodes
        key = (
            int(self.total_nodes),
            int(self.root_nodes),
            int(bottom),
            int(top),
            int(self.parent_index.data_ptr()),
            int(self.child_offsets.data_ptr()),
            int(self.child_count.data_ptr()),
            int(self.action_from_parent.data_ptr()),
            int(self.env.to_act.data_ptr()),
        )
        if self._tree_slice_key == key:
            return

        self._tree_slice_key = key
        self._bottom = bottom
        self._top = top
        self._parent_index_all = self.parent_index.contiguous()
        self._parent_index_bottom = self.parent_index[bottom:].contiguous()
        self._parent_index_nonroot = self.parent_index[self.root_nodes :].contiguous()
        self._child_offsets_top = self.child_offsets[:top].contiguous()
        self._child_count_top = self.child_count[:top].contiguous()
        self._to_act_top = self.env.to_act[:top].contiguous()
        self._action_from_parent_all = self.action_from_parent.contiguous()
        self._child_offsets_by_depth = tuple(
            self.child_offsets[
                self.depth_offsets[d] : self.depth_offsets[d + 1]
            ].contiguous()
            for d in range(self.tree_depth)
        )
        self._child_count_by_depth = tuple(
            self.child_count[
                self.depth_offsets[d] : self.depth_offsets[d + 1]
            ].contiguous()
            for d in range(self.tree_depth)
        )

    def _tree_slices(self) -> None:
        if self._tree_slice_key is None:
            self._prepare_tree_slices()

    def _action_parent_index(self, rows: int, actions: int) -> torch.Tensor:
        """Return [0,0,...,1,1,...] mapping flattened [rows, actions] to rows."""
        key = (rows, actions)
        cached = self._br_action_parent_index_cache.get(key)
        if cached is not None:
            return cached
        out = torch.arange(
            rows, device=self.device, dtype=torch.long
        ).repeat_interleave(actions)
        self._br_action_parent_index_cache[key] = out.contiguous()
        return self._br_action_parent_index_cache[key]

    def _conditioned_action_ratio(
        self,
        numer_by_action: torch.Tensor,
        denom_by_node: torch.Tensor,
    ) -> torch.Tensor:
        """Compute unblocked(numer[action]) / unblocked(denom[node]).

        ``numer_by_action`` is [N, A, H] and ``denom_by_node`` is [N, H].
        The Triton ratio kernel computes the blocker projections and the
        guarded divide in one pass, gathering denominator stats by row index
        instead of expanding them across actions.
        """
        rows, actions, hands = numer_by_action.shape
        parent_index = self._action_parent_index(rows, actions)
        flat = numer_by_action.reshape(rows * actions, hands).contiguous()
        ratio = unblocked_mass_ratio_indirect_triton(
            numer_target=flat,
            denom_target=denom_by_node.contiguous(),
            parent_index=parent_index,
        )
        return ratio.view(rows, actions, hands)

    def _get_root_index(self) -> torch.Tensor:
        cached = getattr(self, "_root_index", None)
        cached_total = getattr(self, "_root_index_total", -1)
        if cached is not None and cached_total == self.total_nodes:
            return cached
        ri = torch.empty(self.total_nodes, dtype=torch.long, device=self.device)
        N = self.root_nodes
        ri[:N] = torch.arange(N, device=self.device)
        for d in range(self.tree_depth):
            start = self.depth_offsets[d + 1]
            end = self.depth_offsets[d + 2]
            ri[start:end] = ri[self.parent_index[start:end]]
        self._root_index = ri
        self._root_index_total = self.total_nodes
        return ri

    def _ensure_positive_regrets_buf(self) -> torch.Tensor:
        if (
            self._fused_positive_regrets_buf is None
            or self._fused_positive_regrets_buf.shape != self.cumulative_regrets.shape
        ):
            self._fused_positive_regrets_buf = torch.empty_like(self.cumulative_regrets)
            self._fused_positive_regrets_valid = False
        return self._fused_positive_regrets_buf

    def _prepare_sample_update_table(self) -> None:
        if not self._opt_sparse_sample:
            return
        key = (
            int(self.t_sample.data_ptr()),
            int(self.total_nodes),
            int(self.cfr_iterations),
        )
        if self._sample_update_key == key:
            return

        t_sample = self.t_sample.to(device=self.device, dtype=torch.long)
        valid_sample = t_sample < self.cfr_iterations
        valid_rows = torch.nonzero(valid_sample, as_tuple=False).flatten()
        t_sample_valid = t_sample.index_select(0, valid_rows)
        counts = torch.bincount(
            t_sample_valid,
            minlength=self.cfr_iterations,
        )[: self.cfr_iterations].contiguous()
        max_updates = int(counts.max().item()) if counts.numel() else 0
        if max_updates == 0:
            rows = torch.empty(
                self.cfr_iterations,
                0,
                dtype=torch.long,
                device=self.device,
            )
        else:
            order = torch.argsort(t_sample_valid, stable=True)
            sorted_t = t_sample_valid.index_select(0, order)
            sorted_rows = valid_rows.index_select(0, order)
            starts = torch.cumsum(counts, dim=0) - counts
            position = torch.arange(
                order.numel(),
                device=self.device,
                dtype=torch.long,
            ) - starts.index_select(0, sorted_t)
            rows = torch.empty(
                self.cfr_iterations,
                max_updates,
                dtype=torch.long,
                device=self.device,
            )
            rows[sorted_t, position] = sorted_rows

        self._sample_update_rows = rows.contiguous()
        self._sample_update_counts = counts
        self._sample_update_key = key

    def _model_features_for_beliefs(
        self, beliefs_at_model: torch.Tensor
    ) -> MLPFeatures:
        key = (
            int(self.model_indices.data_ptr()),
            int(self.new_street_mask.data_ptr()),
            int(self.model_indices.numel()),
        )
        if (
            self._static_model_feature_key != key
            or self._static_model_feature_fields is None
        ):
            static_features = self.feature_encoder.encode(
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
            beliefs=beliefs_at_model.reshape(-1, 2 * NUM_HANDS),
        )

    # ------------------------------------------------------------------
    # Beliefs: fused block + normalize.
    # ------------------------------------------------------------------

    def _normalize_beliefs(self, target: torch.Tensor | None = None) -> None:
        # Re-applies the board mask before normalizing; idempotent on already-
        # masked input, so safe for callers that pre-blocked.
        if target is None:
            target = self.beliefs
        fused_block_and_normalize_beliefs_(
            target, self.allowed_hands, self.allowed_hands_prob
        )

    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> None:
        # Fused per-depth propagation: reach[c, p, h] = reach[parent, p, h] *
        # (policy[c, h] if p == prev_actor[c] else 1.0), zeroed where the
        # child's allowed_hands mask is False (board changes across chance
        # nodes). The block step is folded into the kernel; no post-hoc
        # _block_beliefs call needed.
        for depth in range(self.tree_depth):
            start = self.depth_offsets[depth + 1]
            end = self.depth_offsets[depth + 2]
            fused_reach_weights_depth_(
                reach=target,
                policy=policy,
                allowed_mask=self.allowed_hands,
                parent_index=self.parent_index,
                prev_actor=self.prev_actor,
                start=start,
                end=end,
            )

    def _propagate_all_beliefs(
        self,
        target: torch.Tensor | None = None,
        reach_weights: torch.Tensor | None = None,
    ) -> None:
        if target is None:
            target = self.beliefs
        if reach_weights is None:
            reach_weights = self.self_reach

        # The kernel skips root rows (idempotent: out[:N] == root_beliefs by
        # construction since reach[:N] == 1 and root beliefs are pre-normalized),
        # so non-root programs can read root_beliefs directly from target[:N]
        # without needing a separate clone.
        fused_deep_beliefs_(
            out=target,
            root_beliefs=target,
            reach_weights=reach_weights,
            allowed_prob=self.allowed_hands_prob,
            root_index=self._get_root_index(),
            num_roots=self.root_nodes,
        )

    # ------------------------------------------------------------------
    # Regret matching: parent-aligned sum + in-kernel divide.
    # ------------------------------------------------------------------

    def update_policy(self, t: int) -> None:
        self._prepare_tree_slices()
        bottom = self._bottom
        child_offsets_top = self._child_offsets_top
        child_count_top = self._child_count_top
        parent_index_bottom = self._parent_index_bottom
        assert child_offsets_top is not None
        assert child_count_top is not None
        assert parent_index_bottom is not None
        positive_regrets = self._ensure_positive_regrets_buf()
        if not (
            self._opt_reuse_positive_regrets and self._fused_positive_regrets_valid
        ):
            torch.clamp(self.cumulative_regrets, min=0.0, out=positive_regrets)
        self._fused_positive_regrets_valid = False

        # Parent-aligned sum (no child broadcast), then a divide kernel that
        # gathers from parent_sum via parent_index on the fly. Skips
        # materializing the [num_children, H] denom intermediate.
        uniform_fallback = self.uniform_policy[bottom:].contiguous()
        if self._opt_parent_sum_divide:
            fused_parent_sum_divide_(
                values=positive_regrets.contiguous(),
                fallback=uniform_fallback,
                child_offsets=child_offsets_top,
                child_count=child_count_top,
                out=self.policy_probs[bottom:],
                out_offset=bottom,
                max_children=self.num_actions,
            )
        else:
            parent_sum = fused_parent_sum(
                values=positive_regrets.contiguous(),
                child_offsets=child_offsets_top,
                child_count=child_count_top,
                max_children=self.num_actions,
            )
            fused_divide_by_parent_sum_(
                pos=positive_regrets[bottom:].contiguous(),
                fallback=uniform_fallback,
                parent_sum=parent_sum,
                parent_index=parent_index_bottom,
                out=self.policy_probs[bottom:],
            )
        self._mask_invalid(self.policy_probs)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)
        self._propagate_all_beliefs(self.beliefs, self.self_reach)

        self.update_average_policy(t, update_reach=True)
        if self.cfr_avg or not self.use_final_policy_values:
            self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    def _refresh_average_beliefs(self) -> None:
        self._propagate_all_beliefs(self.beliefs_avg, self.self_reach_avg)

    # ------------------------------------------------------------------
    # Expected values: fused weight + parent-sum reduce.
    # ------------------------------------------------------------------

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

        use_leaf_source = leaf_values is not values
        if not use_leaf_source:
            # Skip the (~leaf_mask) zero: every non-leaf row is overwritten by
            # the parent_sum sweep below (count == 0 iff leaf, so non-leaf
            # parents always have children to reduce). Leaf rows are preserved
            # by parent_sum's `if count == 0: return` early-out.
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
        to_act_top = self._to_act_top
        assert parent_index_bottom is not None
        assert to_act_top is not None
        actor_indices = to_act_top
        actor_indices_expanded = actor_indices[:top, None, None].expand(
            -1, -1, NUM_HANDS
        )
        actor_beliefs = beliefs[:top].gather(1, actor_indices_expanded).squeeze(1)
        # Skip materializing beliefs_dest as a separate tensor — fan-out is done
        # inline via index_select, and the denom side of the ratio kernel
        # gathers from actor_beliefs via parent_index instead.
        marginal_policy = (
            actor_beliefs.index_select(0, parent_index_bottom) * policy[bottom:]
        )

        opponent_conditioned_policy_child = unblocked_mass_ratio_indirect_triton(
            numer_target=marginal_policy.contiguous(),
            denom_target=actor_beliefs.contiguous(),
            parent_index=parent_index_bottom,
        )
        if self._opt_child_opp_policy:
            opponent_conditioned_policy = opponent_conditioned_policy_child
        else:
            opponent_conditioned_policy = torch.zeros_like(policy)
            opponent_conditioned_policy[bottom:] = opponent_conditioned_policy_child

        for depth in range(self.tree_depth - 1, -1, -1):
            parent_base = self.depth_offsets[depth]
            # Fused weight + parent-sum: replaces the per-child clone +
            # scatter_reduce pair with one parent-aligned reduce.
            if self._opt_child_opp_policy:
                fused_weighted_parent_sum_child_opp(
                    values=values,
                    prev_actor=self.prev_actor,
                    policy_hero=policy,
                    policy_opp_child=opponent_conditioned_policy,
                    child_offsets=self._child_offsets_by_depth[depth],
                    child_count=self._child_count_by_depth[depth],
                    parent_base=parent_base,
                    child_base=bottom,
                    max_children=self.num_actions,
                    leaf_values=leaf_values if use_leaf_source else None,
                    leaf_mask=self.leaf_mask.contiguous() if use_leaf_source else None,
                )
            else:
                fused_weighted_parent_sum(
                    values=values,
                    prev_actor=self.prev_actor,
                    policy_hero=policy,
                    policy_opp=opponent_conditioned_policy,
                    child_offsets=self._child_offsets_by_depth[depth],
                    child_count=self._child_count_by_depth[depth],
                    parent_base=parent_base,
                    max_children=self.num_actions,
                    leaf_values=leaf_values if use_leaf_source else None,
                    leaf_mask=self.leaf_mask.contiguous() if use_leaf_source else None,
                )

    # ------------------------------------------------------------------
    # Instantaneous regrets: fused fan-out + gather + sub + mul.
    # ------------------------------------------------------------------

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

        # Compute opponent-reach unblocked mass only at parent rows [0, top)
        # rather than all [total, 2] rows — saves ~13× memory traffic.
        src_weights = self._regret_src_weights(beliefs, top)

        # actor_values is now picked inside fused_regret_tail_ via to_act —
        # no caller-side aten::index, no [top, H] intermediate buffer.
        fused_regret_tail_(
            regrets=regrets,
            values_achieved=values_achieved.contiguous(),
            values_expected=values_expected[:top].contiguous(),
            to_act=to_act_top,
            src_weights=src_weights.contiguous(),
            parent_index=parent_index_all,
            prev_actor=self.prev_actor.contiguous(),
            bottom=bottom,
        )

        self._mask_invalid(regrets)
        return regrets

    def _regret_src_weights(self, beliefs: torch.Tensor, top: int) -> torch.Tensor:
        return unblocked_mass_opp_at_parents_triton(
            beliefs,
            self.env.to_act,
            top,
            allowed_mask=self.allowed_hands[:top].contiguous(),
        )

    def _current_exploitability_cache_key(
        self,
    ) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        tensors = (self.policy_probs_avg, self.beliefs_avg, self.values_avg)
        return tuple(
            (int(t.data_ptr()), int(t._version), tuple(t.shape)) for t in tensors
        )

    def _compute_exploitability(self):
        key = self._current_exploitability_cache_key()
        if (
            self._exploitability_cache_key == key
            and self._exploitability_cache is not None
        ):
            return self._exploitability_cache
        out = super()._compute_exploitability()
        self._exploitability_cache_key = key
        self._exploitability_cache = out
        return out

    def _best_response_values(
        self,
        policy: torch.Tensor,
        beliefs: torch.Tensor,
        base_values: torch.Tensor,
        deviating_player: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fused blocker-projection fast path for local best response stats.

        This mirrors ``CFREvaluator._best_response_values`` but replaces the
        expensive ``calculate_unblocked_mass(...); calculate_unblocked_mass(...);
        where(denom > eps, numer / denom, 0)`` sequences with
        ``unblocked_mass_ratio_indirect_triton``. Exploitability is monitoring
        only, so keeping the reference tree recursion while accelerating the
        blocker math is the highest-leverage low-risk change.
        """
        self._ensure_fused_attrs()

        N, B = self.root_nodes, self.num_actions
        top = self.depth_offsets[-2]
        if deviating_player is None:
            deviating_player = self._fan_out_deep(self.env.to_act[:N])

        values_br = torch.where(self.leaf_mask[:, None, None], base_values, 0.0)

        policy_src_all = self._pull_back(policy)

        actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)
        actor_beliefs = beliefs.gather(1, actor_indices).squeeze(1)[:top]

        marginal_policy = policy_src_all * actor_beliefs[:, None, :]
        opponent_conditioned_policy = self._conditioned_action_ratio(
            marginal_policy, actor_beliefs
        )

        for depth in range(self.tree_depth - 1, -1, -1):
            offset = self.depth_offsets[depth]
            offset_next = self.depth_offsets[depth + 1]
            action_from_parent = self._action_from_parent_all
            assert action_from_parent is not None
            mass_by_action, best_actor_values = fused_br_best_action_mass(
                values=values_br,
                actor_beliefs=actor_beliefs,
                to_act=self.env.to_act.contiguous(),
                deviator=deviating_player.contiguous(),
                child_offsets=self._child_offsets_by_depth[depth],
                child_count=self._child_count_by_depth[depth],
                action_from_parent=action_from_parent,
                parent_base=offset,
                num_actions=B,
                max_children=self.num_actions,
            )
            p_dev = self._conditioned_action_ratio(
                mass_by_action,
                actor_beliefs[offset:offset_next].contiguous(),
            )
            fused_br_finalize_depth_(
                values=values_br,
                policy=policy,
                opponent_policy=opponent_conditioned_policy,
                p_dev=p_dev,
                best_values=best_actor_values,
                to_act=self.env.to_act.contiguous(),
                deviator=deviating_player.contiguous(),
                child_offsets=self._child_offsets_by_depth[depth],
                child_count=self._child_count_by_depth[depth],
                action_from_parent=action_from_parent,
                parent_base=offset,
                num_actions=B,
                max_children=self.num_actions,
            )

        return values_br

    # ------------------------------------------------------------------
    # Update average policy: fused mixing + parent-aligned renorm.
    # ------------------------------------------------------------------

    def update_average_policy(self, t: int, update_reach: bool = False) -> None:
        self._update_average_policy_true(t, update_reach=update_reach)

    def _ensure_average_policy_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._ensure_average_policy_accumulators()

    def _renormalize_average_policy(self, update_reach: bool) -> None:
        N = self.root_nodes
        self.policy_probs_avg[:N] = 0.0
        if update_reach:
            prev_actor = self.prev_actor.contiguous()
            for depth in range(self.tree_depth):
                fused_policy_renorm_reach_depth_(
                    policy=self.policy_probs_avg,
                    reach=self.self_reach_avg,
                    allowed_mask=self.allowed_hands,
                    child_offsets=self._child_offsets_by_depth[depth],
                    child_count=self._child_count_by_depth[depth],
                    prev_actor=prev_actor,
                    parent_base=self.depth_offsets[depth],
                    max_children=self.num_actions,
                )
            return

        self._prepare_tree_slices()
        child_offsets_top = self._child_offsets_top
        child_count_top = self._child_count_top
        parent_index_nonroot = self._parent_index_nonroot
        assert child_offsets_top is not None
        assert child_count_top is not None
        assert parent_index_nonroot is not None

        child_slice = self.policy_probs_avg[N:].contiguous()
        if self._opt_parent_sum_divide:
            fused_parent_sum_divide_(
                values=self.policy_probs_avg.contiguous(),
                fallback=child_slice,
                child_offsets=child_offsets_top,
                child_count=child_count_top,
                out=self.policy_probs_avg[N:],
                out_offset=N,
                max_children=self.num_actions,
                eps=1e-5,
            )
        else:
            parent_sum = fused_parent_sum(
                values=self.policy_probs_avg.contiguous(),
                child_offsets=child_offsets_top,
                child_count=child_count_top,
                max_children=self.num_actions,
            )
            fused_divide_by_parent_sum_(
                pos=child_slice,
                fallback=child_slice,
                parent_sum=parent_sum,
                parent_index=parent_index_nonroot,
                out=self.policy_probs_avg[N:],
                eps=1e-5,
            )

    def _update_average_policy_true(
        self, t: int, update_reach: bool = False
    ) -> None:
        if (
            self.cfr_type in [CFRType.discounted, CFRType.discounted_plus]
            and t <= self.dcfr_delay
        ):
            self.policy_probs_avg[:] = self.policy_probs
            if update_reach:
                self._calculate_reach_weights(
                    self.self_reach_avg, self.policy_probs_avg
                )
            self.average_policy_initialized = False
            return

        self._prepare_tree_slices()
        N = self.root_nodes
        num, den = self._ensure_average_policy_buffers()
        parent_index_all = self._parent_index_all
        assert parent_index_all is not None
        fused_average_policy_mix_with_tensors_(
            policy_probs_avg=self.policy_probs_avg,
            average_policy_numerator=num,
            average_policy_denominator=den,
            policy_probs=self.policy_probs,
            self_reach=self.self_reach,
            to_act=self.env.to_act.contiguous(),
            parent_index=parent_index_all,
            new=self._t_scalars.mix_new,
            bottom=N,
        )
        self.average_policy_initialized = True

        self._renormalize_average_policy(update_reach=update_reach)

    # ------------------------------------------------------------------
    # Update average values: fused mixing.
    # ------------------------------------------------------------------

    def update_average_values(self, t: int) -> None:
        fused_avg_values_zero_sum_(
            values_avg=self.values_avg,
            latest_values=self.latest_values,
            beliefs=self.beliefs_avg,
            old=self._t_scalars.mix_old,
            new=self._t_scalars.mix_new,
            inv_total=self._t_scalars.mix_inv_total,
            enforce_zero_sum=bool(self.model.enforce_zero_sum),
            ignore_mask=self.env.done,
        )

    # ------------------------------------------------------------------
    # Model value mixing: fused (old+new)*h - old*l / new.
    # ------------------------------------------------------------------

    def _set_model_values_impl(self, t, beliefs, features):
        from p2.models.mlp.better_trm import BetterTRM

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if isinstance(self.model, BetterTRM):
                model_output = self.model(
                    features, include_policy=False, latent=self.latent
                )
                self.latent = model_output.latent
            else:
                model_output = self.model(features, include_policy=False)
        hand_values = model_output.hand_values.contiguous()

        do_mix = self.cfr_avg and t > 1 and self.last_model_values is not None
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
        fused_model_values_writeback_(
            hand_values=hand_values,
            last_model_values=last_model_values,
            beliefs=beliefs.contiguous(),
            model_indices=self.model_indices.contiguous(),
            latest_values=self.latest_values,
            last_out=last_out,
            old_plus_new_over_new=self._t_scalars.mix_onon,
            old_over_new=self._t_scalars.mix_oon,
            do_mix=do_mix,
            enforce_zero_sum=bool(self.model.enforce_zero_sum) and do_mix,
            store_last=store_last,
        )
        self.last_model_values = last_out if store_last else None
        return self.latest_values, last_out

    @torch.no_grad()
    def set_leaf_values(self, t: int, beliefs: torch.Tensor | None = None) -> None:
        """Graph-safe override: ``_set_model_values_impl`` writes into
        ``self.latest_values`` in-place, so we skip the ``.copy_()`` round-trip
        the parent class needs. ``self.last_model_values`` is pinned to a
        persistent buffer for the same reason.
        """
        if beliefs is None:
            beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        if self.model_indices.numel() > 0:
            if self._opt_leaf_feature_cache:
                beliefs_at_model = beliefs[self.model_indices]
                features_at_model = self._model_features_for_beliefs(beliefs_at_model)
                showdown_beliefs = beliefs[self.showdown_indices]
            else:
                features = self.feature_encoder.encode(
                    beliefs, pre_chance_node=self.new_street_mask
                )
                # Fused gather: 7 aten::index calls collapse into one Inductor graph
                # so model_indices / showdown_indices are loaded once each and reused
                # across the indexed tensors, instead of being re-issued per field.
                (
                    beliefs_at_model,
                    ctx,
                    street,
                    to_act,
                    board,
                    feat_beliefs,
                    showdown_beliefs,
                ) = _set_leaf_gather(
                    beliefs,
                    features.context,
                    features.street,
                    features.to_act,
                    features.board,
                    features.beliefs,
                    self.model_indices,
                    self.showdown_indices,
                )
                features_at_model = MLPFeatures(
                    context=ctx,
                    street=street,
                    to_act=to_act,
                    board=board,
                    beliefs=feat_beliefs,
                )
            self._set_model_values(t, beliefs_at_model, features_at_model)
        else:
            empty_shape = (0, self.num_players, NUM_HANDS)
            if self._last_model_values_buf is None or (
                self._last_model_values_buf.shape != empty_shape
            ):
                self._last_model_values_buf = self.latest_values.new_empty(empty_shape)
            self.last_model_values = self._last_model_values_buf
            showdown_beliefs = beliefs[self.showdown_indices]

        showdown_values = self._showdown_value_both(showdown_beliefs)
        self.latest_values[self.showdown_indices] = showdown_values

    # ------------------------------------------------------------------
    # CFR iteration: fused DCFR update.
    # ------------------------------------------------------------------

    def cfr_iteration(self, t: int) -> None:
        self._ensure_fused_attrs()
        # Fill device-side scalars for DCFR rescale + mixing weights.
        # Skipped when GraphedCFRIteration has already pre-filled them for
        # this replay, so the graph doesn't re-capture host→device fills.
        if not self._skip_t_scalars_update:
            self.apply_schedules(t)
            mix_old, mix_new = self._get_mixing_weights(t)
            self._t_scalars.update(
                t=t,
                dcfr_alpha=self.dcfr_alpha,
                dcfr_beta=self.dcfr_beta,
                mix_old=float(mix_old),
                mix_new=float(mix_new),
            )

        if self._opt_sparse_sample:
            self._prepare_sample_update_table()
            fused_policy_sample_update_(
                self.policy_probs,
                self.policy_probs_sample,
                self._sample_update_rows,
                self._sample_update_counts,
                self._t_scalars.t_tensor,
            )
        else:
            torch.where(
                (self.t_sample == self._t_scalars.t_tensor)[:, None],
                self.policy_probs,
                self.policy_probs_sample,
                out=self.policy_probs_sample,
            )

        if self.cfr_type == CFRType.linear:
            # Linear CFR not supported by the fused kernel; use parent path.
            regrets = self.compute_instantaneous_regrets(self.latest_values)
            regrets.masked_fill_(self.prev_actor[:, None] == t % self.num_players, 0.0)
            self.regret_weight_sums += 1
            self.cumulative_regrets += regrets
        else:
            apply_dcfr = self.cfr_type in (CFRType.discounted, CFRType.discounted_plus)
            positive_regrets_out = (
                self._ensure_positive_regrets_buf()
                if self._opt_reuse_positive_regrets
                else None
            )
            self._prepare_tree_slices()
            bottom = self._bottom
            top = self._top
            parent_index_all = self._parent_index_all
            to_act_top = self._to_act_top
            assert parent_index_all is not None
            assert to_act_top is not None
            beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
            src_weights = self._regret_src_weights(beliefs, top)
            fused_regret_dcfr_update_with_tensors_(
                values_achieved=self.latest_values.contiguous(),
                values_expected=self.latest_values[:top].contiguous(),
                to_act=to_act_top,
                src_weights=src_weights.contiguous(),
                parent_index=parent_index_all,
                prev_actor=self.prev_actor.contiguous(),
                cumulative_regrets=self.cumulative_regrets,
                regret_weight_sums=self.regret_weight_sums,
                t_alpha_num=self._t_scalars.t_alpha_num,
                t_beta_num=self._t_scalars.t_beta_num,
                t_alpha_den=self._t_scalars.t_alpha_den,
                t_beta_den=self._t_scalars.t_beta_den,
                bottom=bottom,
                apply_dcfr=apply_dcfr,
                cfr_plus=self.cfr_plus,
                positive_regrets_out=positive_regrets_out,
            )
            self._fused_positive_regrets_valid = positive_regrets_out is not None

        if self._skip_record_stats:
            self.update_policy(t)
        else:
            # _record_stats only uses old_policy_probs at 5 percentile
            # iterations (see CFREvaluator._record_stats). Skipping the
            # full-tensor clone on the other ~395 of 400 iterations cuts a
            # large per-CFR-iter DtoD copy.
            if t in self._record_stats_percentile_ts():
                old_policy_probs = self.policy_probs.clone()
                self.update_policy(t)
                self._record_stats(t, old_policy_probs)
            else:
                self.update_policy(t)

        self.set_leaf_values(t)
        self.compute_expected_values()

        if not self.use_final_policy_values:
            self.update_average_values(t)

    def prepare_replay(self, t: int) -> None:
        """Host-side prep for a CUDA-graph replay at iteration ``t``.

        Updates Python schedules + TScalars device tensors OUTSIDE any captured
        region. Call this immediately before ``graph.replay()`` to run the
        captured iteration with a different ``t``.
        """
        self.apply_schedules(t)
        mix_old, mix_new = self._get_mixing_weights(t)
        self._t_scalars.update(
            t=t,
            dcfr_alpha=self.dcfr_alpha,
            dcfr_beta=self.dcfr_beta,
            mix_old=float(mix_old),
            mix_new=float(mix_new),
        )

    def _graph_capture_regime(self, t: int) -> str | None:
        """Return the Python-branch regime that is safe to CUDA-graph replay."""
        if t < 2:
            return None
        if (
            self.cfr_type in (CFRType.discounted, CFRType.discounted_plus)
            and t <= self.dcfr_delay
        ):
            return "pre_dcfr_delay"
        return "post_dcfr_delay"

    @torch.no_grad()
    def evaluate_cfr(self, training_mode: bool = True):
        """CFR-loop with per-call CUDA graphs for each Python branch regime.

        Runs ``t < 2`` uncaptured, then captures/replays one graph for the
        pre-DCFR-delay average-policy branch and another graph for the
        post-delay branch. The capture step's side-stream warmup is itself a
        real CFR iteration at ``t``, so no snapshot/restore is needed (CUDA
        graph capture is record-only — the body recorded for ``t+1`` doesn't
        execute until the first replay). Iterations whose ``t`` is in
        ``_record_stats_percentile_ts()`` run uncaptured so stats hooks still
        fire, and capture is deferred past any such iter. Per-call capture is
        required because ``initialize_subgame`` reallocates the per-evaluator
        tensors.
        """
        self._ensure_fused_attrs()
        self.model.eval()

        self.initialize_policy_and_beliefs()
        if self.warm_start_iterations > 0:
            self.warm_start()

        self.set_leaf_values(0)
        self.compute_expected_values()
        self.values_avg[:] = self.latest_values

        self.t_sample = self._get_sampling_schedule()
        self._prepare_sample_update_table()

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
                # The warmup iter is a real CFR step at t; the captured body
                # for t+1 doesn't execute until the first replay below.
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

        if not self.cfr_avg and self.use_final_policy_values:
            self._refresh_average_beliefs()

        if self.use_final_policy_values:
            self.update_average_values_final()

        self._record_action_mix()
        self._record_cfr_entropy()
        self._record_cumulative_regret()

        return self.sample_leaves(training_mode)
