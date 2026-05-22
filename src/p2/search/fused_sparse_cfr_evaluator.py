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
* ``update_policy`` — optional per-depth fusion of reach propagation, belief
  propagation, and deferred average-policy accumulation.
* ``update_policy`` / ``update_average_policy`` — parent-aligned positive-regret
  sum + in-kernel divide via ``fused_parent_sum_divide_``.
* ``update_average_policy`` — average-policy renorm + average-reach propagation
  via ``fused_policy_renorm_reach_depth_`` in the hot ``update_policy`` path.
* ``compute_expected_values`` — per-depth weight + parent-sum reduce with
  inline opponent blocker projections.
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
from p2.models.mlp.better_ffn import BetterFFN
from p2.models.mlp.mlp_features import MLPFeatures
from p2.search.allin_payoff import (
    FLOP_I8_SCALE,
    I16_SCALE,
    write_allin_belief_card_stats_split_triton_,
    write_allin_table_values_card_denom_dot_values_triton_,
    write_allin_table_values_triton_,
)
from p2.search.fused_cfr_triton import (
    fused_average_policy_mix_with_tensors_,
    fused_avg_values_zero_sum_,
    fused_br_best_action_mass,
    fused_br_finalize_depth_,
    fused_block_and_normalize_beliefs_,
    fused_deep_beliefs_,
    fused_model_values_writeback_,
    fused_parent_sum_divide_,
    fused_policy_sample_update_,
    fused_policy_renorm_reach_depth_,
    fused_reach_beliefs_avg_depth_,
    fused_reach_beliefs_avg_scratch_depth_,
    fused_reach_weights_depth_,
    fused_regret_tail_,
    fused_unblocked_regret_dcfr_update_with_tensors_,
    fused_weighted_parent_sum_inline_opp_both,
    fused_weighted_parent_sum_inline_opp_both_noleaf,
    GraphedCFRIteration,
    precompute_showdown_extras,
    showdown_ev_v15,
    ShowdownGraphRunner,
    triton_is_available,
    TScalars,
    _preprocess_unblocked_stats,
    _preprocess_unblocked_stats_out,
    unblocked_mass_opp_at_parents_triton,
    unblocked_mass_ratio_indirect_triton,
    marginal_policy_triton_out_,
    select_actor_beliefs_triton_out_,
    select_opponent_beliefs_triton_out_,
)
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


def _compile_setting_from_env(cfg=None) -> str:
    mode = os.environ.get("P2_FUSED_COMPILE_MODE")
    if mode is None and cfg is not None:
        mode = getattr(cfg.model, "compile", "default")
    value = str(mode if mode is not None else "default").strip().lower()
    if value in {"0", "false", "no", "none"}:
        return "off"
    if value in {"", "true", "yes", "1"}:
        return "default"
    if value not in {"off", "default", "max-autotune"}:
        raise ValueError(
            "compile mode must be one of: off, default, max-autotune; "
            f"got {mode!r}"
        )
    return value


def _compile_kwargs_from_env(cfg=None) -> dict[str, object]:
    kwargs: dict[str, object] = {"dynamic": True}
    mode = _compile_setting_from_env(cfg)
    if mode == "max-autotune":
        kwargs["mode"] = mode
    return kwargs


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
        self._compile_kwargs = _compile_kwargs_from_env(self.cfg)
        compile_setting = _compile_setting_from_env(self.cfg)
        if compile_model and compile_setting != "off" and self.model is not None:
            torch.set_float32_matmul_precision("high")
            try:
                if hasattr(self.model, "compile_forward_modes"):
                    if getattr(self.model, "_compiled_forward_value", None) is None:
                        self.model.compile_forward_modes(**self._compile_kwargs)
                else:
                    self.model = torch.compile(self.model, **self._compile_kwargs)
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
        self._fused_positive_regrets_valid: bool = False
        self._reach_scratch_a: torch.Tensor | None = None
        self._reach_scratch_b: torch.Tensor | None = None
        self._reach_scratch_width: int = 0
        self._ev_actor_beliefs_buf: torch.Tensor | None = None
        self._ev_marginal_policy_buf: torch.Tensor | None = None
        self._regret_src_target_buf: torch.Tensor | None = None
        self._regret_src_stats_buf: torch.Tensor | None = None
        self._sample_update_rows: torch.Tensor | None = None
        self._sample_update_counts: torch.Tensor | None = None
        self._sample_update_key: tuple[int, int, int] | None = None
        self._static_model_base_key: tuple[int, int, int] | None = None
        self._static_model_base_features: torch.Tensor | None = None
        self._static_model_base_fn = None
        self._static_model_base_fn_key: int | None = None
        self._static_model_feature_key: tuple[int, int, int] | None = None
        self._static_model_feature_fields: tuple[torch.Tensor, ...] | None = None
        self._leaf_belief_gather_indices: torch.Tensor | None = None
        self._leaf_belief_gather_key: tuple[int, int, int, int] | None = None
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
        self._child_count_pow2_by_depth: tuple[int, ...] = ()
        self._exploitability_cache_key: (
            tuple[tuple[int, int, tuple[int, ...]], ...] | None
        ) = None
        self._exploitability_cache = None

    def _init_hand_rank_data(self) -> None:
        """Build hand-rank data, then precompute the constant-per-subgame
        showdown EV inputs and capture a CUDA graph for the EV pipeline.
        The graph is keyed on (M=showdown_indices.numel(), NUM_HANDS) and
        replays via persistent buffers."""
        self.showdown_indices = torch.where(self.env.street == 4)[0].contiguous()
        self.showdown_actors = self.env.to_act[self.showdown_indices]
        self.showdown_potential = (
            self.env.stacks[self.showdown_indices]
            + self.env.pot[self.showdown_indices, None]
            - self.env.starting_stacks[self.showdown_indices]
        )
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
        if not hasattr(self, "_compile_kwargs"):
            self._compile_kwargs = _compile_kwargs_from_env()
        if not hasattr(self, "_fused_positive_regrets_valid"):
            self._fused_positive_regrets_valid = False
        if not hasattr(self, "_reach_scratch_a"):
            self._reach_scratch_a = None
        if not hasattr(self, "_reach_scratch_b"):
            self._reach_scratch_b = None
        if not hasattr(self, "_reach_scratch_width"):
            self._reach_scratch_width = 0
        if not hasattr(self, "_ev_actor_beliefs_buf"):
            self._ev_actor_beliefs_buf = None
        if not hasattr(self, "_ev_marginal_policy_buf"):
            self._ev_marginal_policy_buf = None
        if not hasattr(self, "_regret_src_target_buf"):
            self._regret_src_target_buf = None
        if not hasattr(self, "_regret_src_stats_buf"):
            self._regret_src_stats_buf = None
        if not hasattr(self, "_sample_update_rows"):
            self._sample_update_rows = None
        if not hasattr(self, "_sample_update_counts"):
            self._sample_update_counts = None
        if not hasattr(self, "_sample_update_key"):
            self._sample_update_key = None
        if not hasattr(self, "_static_model_base_key"):
            self._static_model_base_key = None
        if not hasattr(self, "_static_model_base_features"):
            self._static_model_base_features = None
        if not hasattr(self, "_static_model_base_fn"):
            self._static_model_base_fn = None
        if not hasattr(self, "_static_model_base_fn_key"):
            self._static_model_base_fn_key = None
        if not hasattr(self, "_static_model_feature_key"):
            self._static_model_feature_key = None
        if not hasattr(self, "_static_model_feature_fields"):
            self._static_model_feature_fields = None
        if not hasattr(self, "_leaf_belief_gather_indices"):
            self._leaf_belief_gather_indices = None
        if not hasattr(self, "_leaf_belief_gather_key"):
            self._leaf_belief_gather_key = None
        if not hasattr(self, "_br_action_parent_index_cache"):
            self._br_action_parent_index_cache = {}
        if not hasattr(self, "_tree_slice_key"):
            self._tree_slice_key = None
        if not hasattr(self, "_child_offsets_by_depth"):
            self._child_offsets_by_depth = ()
        if not hasattr(self, "_child_count_by_depth"):
            self._child_count_by_depth = ()
        if not hasattr(self, "_child_count_pow2_by_depth"):
            self._child_count_pow2_by_depth = ()
        if not hasattr(self, "_action_from_parent_all"):
            self._action_from_parent_all = None
        if not hasattr(self, "_exploitability_cache_key"):
            self._exploitability_cache_key = None
        if not hasattr(self, "_exploitability_cache"):
            self._exploitability_cache = None
        if not hasattr(self, "beliefs_sample"):
            self.beliefs_sample = torch.zeros_like(self.beliefs)

    def initialize_subgame(self, *args, **kwargs) -> None:
        super().initialize_subgame(*args, **kwargs)
        self._static_model_base_key = None
        self._static_model_base_features = None
        self._leaf_belief_gather_indices = None
        self._leaf_belief_gather_key = None
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
        self._child_count_pow2_by_depth = tuple(
            1 << max(0, int(counts.max().item()) - 1).bit_length()
            if counts.numel() > 0
            else 1
            for counts in self._child_count_by_depth
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

    def _ensure_reach_scratch_buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        max_width = 1
        for d in range(self.tree_depth + 1):
            width = self.depth_offsets[d + 1] - self.depth_offsets[d]
            max_width = max(max_width, int(width))
        shape = (max_width, self.num_players, NUM_HANDS)
        if self._reach_scratch_a is None or self._reach_scratch_a.shape != shape:
            self._reach_scratch_a = self.beliefs.new_empty(shape)
            self._reach_scratch_b = self.beliefs.new_empty(shape)
            self._reach_scratch_width = max_width
        assert self._reach_scratch_b is not None
        return self._reach_scratch_a, self._reach_scratch_b

    def _ensure_ev_policy_buffers(
        self, top: int, num_children: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor_shape = (top, NUM_HANDS)
        marginal_shape = (num_children, NUM_HANDS)
        if (
            self._ev_actor_beliefs_buf is None
            or self._ev_actor_beliefs_buf.shape != actor_shape
        ):
            self._ev_actor_beliefs_buf = self.beliefs.new_empty(actor_shape)
        if (
            self._ev_marginal_policy_buf is None
            or self._ev_marginal_policy_buf.shape != marginal_shape
        ):
            self._ev_marginal_policy_buf = self.policy_probs.new_empty(
                marginal_shape
            )
        return self._ev_actor_beliefs_buf, self._ev_marginal_policy_buf

    def _ensure_regret_src_buffers(
        self, top: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_shape = (top, NUM_HANDS)
        stats_shape = (top, 53)
        if (
            self._regret_src_target_buf is None
            or self._regret_src_target_buf.shape != target_shape
        ):
            self._regret_src_target_buf = self.beliefs.new_empty(target_shape)
        if (
            self._regret_src_stats_buf is None
            or self._regret_src_stats_buf.shape != stats_shape
        ):
            self._regret_src_stats_buf = self.beliefs.new_empty(stats_shape)
        return self._regret_src_target_buf, self._regret_src_stats_buf

    def _prepare_sample_update_table(self) -> None:
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

    def _leaf_beliefs_for_model_and_showdown(
        self, beliefs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = int(self.model_indices.numel())
        s = int(self.showdown_indices.numel())
        key = (
            int(self.model_indices.data_ptr()),
            int(self.showdown_indices.data_ptr()),
            m,
            s,
        )
        if self._leaf_belief_gather_key != key or self._leaf_belief_gather_indices is None:
            self._leaf_belief_gather_indices = torch.cat(
                (self.model_indices, self.showdown_indices),
                dim=0,
            ).contiguous()
            self._leaf_belief_gather_key = key

        gathered = beliefs[self._leaf_belief_gather_indices]
        return gathered[:m], gathered[m:]

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
        if not self._fused_positive_regrets_valid:
            torch.clamp(self.cumulative_regrets, min=0.0, out=positive_regrets)
        self._fused_positive_regrets_valid = False

        # Parent-aligned sum (no child broadcast), then a divide kernel that
        # gathers from parent_sum via parent_index on the fly. Skips
        # materializing the [num_children, H] denom intermediate.
        uniform_fallback = self.uniform_policy[bottom:].contiguous()
        fused_parent_sum_divide_(
            values=positive_regrets.contiguous(),
            fallback=uniform_fallback,
            child_offsets=child_offsets_top,
            child_count=child_count_top,
            out=self.policy_probs[bottom:],
            out_offset=bottom,
            max_children=self.num_actions,
        )
        self._mask_invalid(self.policy_probs)

        defer_avg_reach = not self.cfr_avg and self.use_final_policy_values
        defer_avg_policy = not self.cfr_avg and self.use_final_policy_values
        if defer_avg_reach and defer_avg_policy:
            skip_avg_update = (
                self.cfr_type in [CFRType.discounted, CFRType.discounted_plus]
                and t <= self.dcfr_delay
            )
            write_average_policy = not skip_avg_update
            avg_num, avg_den = self._ensure_average_policy_buffers()
            root_index = self._get_root_index()
            parent_index_all = self._parent_index_all
            assert parent_index_all is not None
            to_act = self.env.to_act.contiguous()
            prev_actor = self.prev_actor.contiguous()
            use_scratch_reach = self._skip_record_stats
            if use_scratch_reach:
                scratch_a, scratch_b = self._ensure_reach_scratch_buffers()
                parent_reach = scratch_a
                child_reach = scratch_b
                for depth in range(self.tree_depth):
                    start = self.depth_offsets[depth + 1]
                    end = self.depth_offsets[depth + 2]
                    parent_base = self.depth_offsets[depth]
                    store_child = depth < self.tree_depth - 1
                    fused_reach_beliefs_avg_scratch_depth_(
                        parent_reach=parent_reach,
                        child_reach=child_reach,
                        beliefs=self.beliefs,
                        policy=self.policy_probs,
                        allowed_mask=self.allowed_hands,
                        allowed_prob=self.allowed_hands_prob,
                        root_index=root_index,
                        parent_index=parent_index_all,
                        prev_actor=prev_actor,
                        to_act=to_act,
                        average_policy_numerator=avg_num,
                        average_policy_denominator=avg_den,
                        new=self._t_scalars.mix_new,
                        parent_base=parent_base,
                        start=start,
                        end=end,
                        root_parent=depth == 0,
                        write_average_policy=write_average_policy,
                        store_child=store_child,
                    )
                    parent_reach, child_reach = child_reach, parent_reach
            else:
                for depth in range(self.tree_depth):
                    fused_reach_beliefs_avg_depth_(
                        reach=self.self_reach,
                        beliefs=self.beliefs,
                        policy=self.policy_probs,
                        allowed_mask=self.allowed_hands,
                        allowed_prob=self.allowed_hands_prob,
                        root_index=root_index,
                        parent_index=parent_index_all,
                        prev_actor=prev_actor,
                        to_act=to_act,
                        average_policy_numerator=avg_num,
                        average_policy_denominator=avg_den,
                        new=self._t_scalars.mix_new,
                        start=self.depth_offsets[depth + 1],
                        end=self.depth_offsets[depth + 2],
                        write_average_policy=write_average_policy,
                        # Final-depth reach has no descendants and stats only read
                        # non-leaf reach. When stats are stubbed, keep leaf reach
                        # register-local and avoid two full leaf-row stores.
                        store_reach=(
                            depth < self.tree_depth - 1 or not self._skip_record_stats
                        ),
                    )
            self.average_policy_initialized = write_average_policy
        else:
            self._calculate_reach_weights(self.self_reach, self.policy_probs)
            self._propagate_all_beliefs(self.beliefs, self.self_reach)
            self.update_average_policy(t, update_reach=not defer_avg_reach)
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
        actor_beliefs, marginal_policy = self._ensure_ev_policy_buffers(
            top,
            parent_index_bottom.numel(),
        )
        select_actor_beliefs_triton_out_(
            beliefs,
            actor_indices,
            top,
            actor_beliefs,
        )
        marginal_policy_triton_out_(
            actor_beliefs,
            policy,
            parent_index_bottom,
            bottom,
            marginal_policy,
        )

        numer_s, numer_cardsum = _preprocess_unblocked_stats(marginal_policy)
        denom_s, denom_cardsum = _preprocess_unblocked_stats(actor_beliefs)
        for depth in range(self.tree_depth - 1, -1, -1):
            if not use_leaf_source:
                fused_weighted_parent_sum_inline_opp_both_noleaf(
                    values=values,
                    prev_actor=self.prev_actor,
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
                )
            else:
                fused_weighted_parent_sum_inline_opp_both(
                    values=values,
                    prev_actor=self.prev_actor,
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
                    leaf_values=leaf_values,
                    leaf_mask=self.leaf_mask.contiguous(),
                    block_h=512,
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

    def _update_average_policy_true(
        self, t: int, update_reach: bool = False
    ) -> None:
        defer_avg_policy = not self.cfr_avg and self.use_final_policy_values
        if (
            self.cfr_type in [CFRType.discounted, CFRType.discounted_plus]
            and t <= self.dcfr_delay
        ):
            if defer_avg_policy:
                self.average_policy_initialized = False
                return
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
            block_h=1024 if defer_avg_policy else 512,
            write_policy=not defer_avg_policy,
        )
        self.average_policy_initialized = True

        if not defer_avg_policy:
            self._renormalize_average_policy(update_reach=update_reach)

    def _finalize_deferred_average_policy(self) -> None:
        """Materialize normalized average policy after deferred accumulation."""
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
            new=self._t_scalars.zero,
            bottom=N,
        )
        self._renormalize_average_policy(update_reach=False)

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

        model_applied_zero_sum = False
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_model = getattr(self.model, "_orig_mod", self.model)
            if isinstance(base_model, BetterTRM):
                model_output = self.model(
                    features, include_policy=False, latent=self.latent
                )
                self.latent = model_output.latent
                model_applied_zero_sum = bool(base_model.enforce_zero_sum)
            elif isinstance(base_model, BetterFFN):
                model_key = id(base_model)
                if (
                    self._static_model_base_fn is None
                    or self._static_model_base_fn_key != model_key
                ):
                    if (
                        getattr(self.model, "_orig_mod", None) is not None
                        or getattr(base_model, "_compiled_forward_value", None)
                        is not None
                    ):
                        self._static_model_base_fn = torch.compile(
                            base_model.static_feature_base,
                            **self._compile_kwargs,
                        )
                    else:
                        self._static_model_base_fn = base_model.static_feature_base
                    self._static_model_base_fn_key = model_key
                key = (
                    int(features.context.data_ptr()),
                    int(features.street.data_ptr()),
                    int(features.board.data_ptr()),
                )
                if (
                    self._static_model_base_key != key
                    or self._static_model_base_features is None
                ):
                    # Clone outside torch.compile so compile modes that enable
                    # cudagraphs can safely reuse this cached tensor across
                    # repeated model calls.
                    self._static_model_base_features = self._static_model_base_fn(
                        features
                    ).clone()
                    self._static_model_base_key = key
                model_output = self.model(
                    features,
                    include_policy=False,
                    apply_zero_sum=True,
                    static_base_features=self._static_model_base_features,
                )
                model_applied_zero_sum = bool(base_model.enforce_zero_sum)
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
        # ``apply_zero_sum`` controls when the projection is applied, not if.
        # BetterFFN/BetterTRM already return zero-sum values when configured to
        # do so. CFR-AVG mixing can break that projection, and models without
        # internal projection still need the fused writeback to apply it.
        enforce_writeback_zero_sum = bool(self.model.enforce_zero_sum) and (
            do_mix or not model_applied_zero_sum
        )
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
            enforce_zero_sum=enforce_writeback_zero_sum,
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
            beliefs_at_model, showdown_beliefs = (
                self._leaf_beliefs_for_model_and_showdown(beliefs)
            )
            features_at_model = self._model_features_for_beliefs(beliefs_at_model)
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
        self._set_allin_call_values(beliefs)

    def _set_allin_call_values(self, beliefs: torch.Tensor) -> None:
        indices = getattr(self, "allin_call_indices", None)
        if indices is None or indices.numel() == 0:
            return
        resolver = self._ensure_allin_payoff_resolver()
        indices_by_street = getattr(self, "allin_call_indices_by_street", None)
        boards_by_street = getattr(self, "allin_call_boards_by_street", None)
        if indices_by_street is None or boards_by_street is None:
            self._cache_allin_call_street_partitions(
                self.env.street[self.allin_call_parent_indices]
            )
            indices_by_street = self.allin_call_indices_by_street
            boards_by_street = self.allin_call_boards_by_street

        node_idx = indices_by_street[0]
        if node_idx.numel() > 0:
            table, scale = resolver.payoff_for_board(boards_by_street[0].new_empty(0), 0)
            write_allin_table_values_triton_(
                table=table,
                beliefs=beliefs,
                node_indices=node_idx,
                latest_values=self.latest_values,
                stacks=self.env.stacks,
                pot=self.env.pot,
                starting_stacks=self.env.starting_stacks,
                env_scale=self.env.scale,
                table_scale=scale,
            )

        flop_node_idx = indices_by_street[1]
        turn_node_idx = indices_by_street[2]
        flop_tables = getattr(self, "allin_flop_tables_i8", None)
        flop_ids = getattr(self, "allin_flop_table_ids", None)
        flop_stats = getattr(self, "allin_flop_stats_buffer", None)
        turn_tables = getattr(self, "allin_turn_tables_i16", None)
        turn_ids = getattr(self, "allin_turn_table_ids", None)
        turn_stats = getattr(self, "allin_turn_stats_buffer", None)
        if flop_node_idx.numel() > 0:
            if flop_tables is None or flop_ids is None or flop_stats is None or flop_tables.numel() == 0:
                raise RuntimeError("Fused all-in flop evaluation requires cached flop tables.")
        if turn_node_idx.numel() > 0:
            if turn_tables is None or turn_ids is None or turn_stats is None or turn_tables.numel() == 0:
                raise RuntimeError("Fused all-in turn evaluation requires cached turn tables.")
        if flop_node_idx.numel() > 0 or turn_node_idx.numel() > 0:
            if flop_stats is None or turn_stats is None:
                raise RuntimeError("Fused all-in evaluation requires cached stats buffers.")
            write_allin_belief_card_stats_split_triton_(
                beliefs=beliefs,
                node_indices0=flop_node_idx,
                stats_buffer0=flop_stats,
                node_indices1=turn_node_idx,
                stats_buffer1=turn_stats,
            )

        node_idx = flop_node_idx
        if node_idx.numel() > 0:
            flop_tables = getattr(self, "allin_flop_tables_i8", None)
            flop_ids = getattr(self, "allin_flop_table_ids", None)
            flop_stats = getattr(self, "allin_flop_stats_buffer", None)
            if flop_tables is None or flop_ids is None or flop_stats is None or flop_tables.numel() == 0:
                raise RuntimeError("Fused all-in flop evaluation requires cached flop tables.")
            write_allin_table_values_card_denom_dot_values_triton_(
                table=flop_tables,
                beliefs=beliefs,
                node_indices=node_idx,
                latest_values=self.latest_values,
                stacks=self.env.stacks,
                pot=self.env.pot,
                starting_stacks=self.env.starting_stacks,
                env_scale=self.env.scale,
                table_scale=FLOP_I8_SCALE,
                canon_ids=flop_ids,
                stats_buffer=flop_stats,
                block_h=64,
                block_k=64,
                block_p=8,
            )

        node_idx = turn_node_idx
        if node_idx.numel() > 0:
            turn_tables = getattr(self, "allin_turn_tables_i16", None)
            turn_ids = getattr(self, "allin_turn_table_ids", None)
            turn_stats = getattr(self, "allin_turn_stats_buffer", None)
            if turn_tables is None or turn_ids is None or turn_stats is None or turn_tables.numel() == 0:
                raise RuntimeError("Fused all-in turn evaluation requires cached turn tables.")
            write_allin_table_values_card_denom_dot_values_triton_(
                table=turn_tables,
                beliefs=beliefs,
                node_indices=node_idx,
                latest_values=self.latest_values,
                stacks=self.env.stacks,
                pot=self.env.pot,
                starting_stacks=self.env.starting_stacks,
                env_scale=self.env.scale,
                table_scale=I16_SCALE,
                canon_ids=turn_ids,
                stats_buffer=turn_stats,
                block_h=64,
                block_k=128,
                block_p=8,
            )

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

        self._prepare_sample_update_table()
        fused_policy_sample_update_(
            self.policy_probs,
            self.policy_probs_sample,
            self._sample_update_rows,
            self._sample_update_counts,
            self._t_scalars.t_tensor,
        )
        fused_policy_sample_update_(
            self.beliefs.view(self.total_nodes, -1),
            self.beliefs_sample.view(self.total_nodes, -1),
            self._sample_update_rows,
            self._sample_update_counts,
            self._t_scalars.t_tensor,
            block_h=1024,
        )

        if self.cfr_type == CFRType.linear:
            # Linear CFR not supported by the fused kernel; use parent path.
            regrets = self.compute_instantaneous_regrets(self.latest_values)
            regrets.masked_fill_(self.prev_actor[:, None] == t % self.num_players, 0.0)
            self.cumulative_regrets += regrets
        else:
            apply_dcfr = self.cfr_type in (CFRType.discounted, CFRType.discounted_plus)
            positive_regrets_out = self._ensure_positive_regrets_buf()
            self._prepare_tree_slices()
            top = self._top
            to_act_top = self._to_act_top
            assert to_act_top is not None
            beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
            src_target, src_stats = self._ensure_regret_src_buffers(top)
            select_opponent_beliefs_triton_out_(
                beliefs,
                self.env.to_act.contiguous(),
                top,
                src_target,
            )
            _preprocess_unblocked_stats_out(src_target, src_stats)
            child_offsets_top = self._child_offsets_top
            child_count_top = self._child_count_top
            assert child_offsets_top is not None
            assert child_count_top is not None
            fused_unblocked_regret_dcfr_update_with_tensors_(
                target=src_target,
                stats=src_stats,
                allowed_mask=self.allowed_hands[:top].contiguous(),
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
                apply_dcfr=apply_dcfr,
                cfr_plus=self.cfr_plus,
                max_children=self.num_actions,
                positive_regrets_out=positive_regrets_out,
            )
            self._fused_positive_regrets_valid = True

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
            self._finalize_deferred_average_policy()
            self.self_reach_avg[: self.root_nodes] = 1.0
            self._calculate_reach_weights(
                self.self_reach_avg, self.policy_probs_avg
            )
            self._refresh_average_beliefs()

        if self.use_final_policy_values:
            self.update_average_values_final()

        self._record_action_mix()
        self._record_cfr_entropy()
        self._record_cumulative_regret()

        return self.sample_leaves(training_mode)
