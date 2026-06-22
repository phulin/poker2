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

import copy
import inspect
import os
from contextlib import contextmanager

import torch

from p2.core.structured_config import CFRType
from p2.env import rules
from p2.env.card_utils import NUM_HANDS, combo_to_onehot_tensor, hand_combos_tensor
from p2.env.env_gather_triton import gather_env_rows_into_triton
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.env.rules_triton import (
    rank_hands_triton,
    triton_is_available as _rules_triton_ok,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.search.allin_payoff import (
    FLOP_I8_SCALE,
    I16_SCALE,
    combo_cards_i32_tensor,
    write_allin_belief_card_stats_split_triton_,
    write_allin_grouped_table_values_card_denom_dot_values_triton_,
    write_allin_table_values_card_denom_dot_triton_,
    write_allin_table_values_card_denom_dot_values_triton_,
)
from p2.search.fused_cfr_triton import (
    fused_average_policy_mix_with_tensors_,
    fused_avg_values_zero_sum_,
    fused_br_best_action_mass,
    fused_br_finalize_depth_,
    fused_block_and_normalize_beliefs_,
    fused_cfr_delta_stats,
    fused_deep_beliefs_,
    fused_model_values_writeback_,
    fused_parent_sum_divide_,
    fused_policy_renorm_reach_depth_,
    fused_reach_beliefs_avg_depth_,
    fused_reach_beliefs_avg_scratch_depth_,
    fused_reach_weights_depth_,
    fused_regret_tail_,
    fused_sample_leaf_compact_,
    fused_unblocked_regret_dcfr_update_with_tensors_,
    fused_weighted_parent_sum_inline_opp_both,
    fused_weighted_parent_sum_inline_opp_both_noleaf,
    GraphedCFRIteration,
    precompute_showdown_active_extras,
    precompute_showdown_extras,
    showdown_ev_v15,
    ShowdownActiveCardBlockDirectGraphRunner,
    triton_is_available,
    TScalars,
    _preprocess_unblocked_stats,
    _preprocess_unblocked_stats_out,
    unblocked_mass_opp_at_parents_triton,
    unblocked_mass_ratio_indirect_triton,
    select_actor_beliefs_and_marginal_policy_triton_out_,
    select_opponent_beliefs_triton_out_,
)
from p2.search.cfr_evaluator import HandRankData, PublicBeliefState, padded_indices
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator
from p2.search.subgame_constructor_triton import (
    copy_child_cards_triton_,
    finalize_tree_masks_triton_,
    init_belief_value_tensors_triton_,
    init_policy_tensors_triton_,
    legal_counts_triton_,
    root_allowed_from_board_indices_triton_,
    write_children_same_street_triton_,
)


_INIT_PROFILE_HOOK = None


@contextmanager
def _init_profile_region(tag: str):
    hook = _INIT_PROFILE_HOOK
    if hook is None:
        yield
    else:
        with hook(tag):
            yield


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
            f"compile mode must be one of: off, default, max-autotune; got {mode!r}"
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
        self._last_instantaneous_regrets: torch.Tensor | None = None
        self._predictive_cfr_enabled = self.cfr_type in (
            CFRType.pcfr,
            CFRType.sapcfr,
        )
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
        self._sample_root_rows: torch.Tensor | None = None
        self._sample_root_counts: torch.Tensor | None = None
        self._sample_root_key: tuple[int, int, int] | None = None
        self._sample_leaf_enabled: bool = False
        self._sample_leaf_training_mode: bool = True
        self._sample_leaf_epsilon: float = 0.0
        self._sample_leaf_players: torch.Tensor | None = None
        self._sample_leaf_hands: torch.Tensor | None = None
        self._sample_leaf_uniform_draws: torch.Tensor | None = None
        self._sample_leaf_action_draws: torch.Tensor | None = None
        self._sample_leaf_target_enabled: torch.Tensor | None = None
        self._sample_leaf_target_depths: torch.Tensor | None = None
        self._sample_leaf_indices_padded: torch.Tensor | None = None
        self._sample_leaf_beliefs_padded: torch.Tensor | None = None
        self._sample_leaf_ready_padded: torch.Tensor | None = None
        self._sample_leaf_effective_leaf_mask: torch.Tensor | None = None
        self._sample_leaf_sampling_masks: torch.Tensor | None = None
        self._sample_leaf_uniform_policy: torch.Tensor | None = None
        self._sample_leaf_child_nodes_by_action: torch.Tensor | None = None
        self._static_model_base_key: tuple[int, int, int, int] | None = None
        self._static_model_base_features: torch.Tensor | None = None
        self._static_model_base_fn = None
        self._static_model_base_fn_key: int | None = None
        self._static_model_feature_key: tuple[int, int, int, int] | None = None
        self._static_model_feature_fields: tuple[torch.Tensor, ...] | None = None
        self._leaf_belief_gather_indices: torch.Tensor | None = None
        self._leaf_belief_gather_key: tuple[int, int, int] | None = None
        self._model_leaf_scatter_enabled: bool = os.environ.get(
            "P2_FUSED_MODEL_LEAF_SCATTER", "1"
        ).strip().lower() not in {"0", "false", "off", "no"}
        self._model_leaf_beliefs_buf: torch.Tensor | None = None
        self._model_leaf_slot: torch.Tensor | None = None
        self._model_leaf_slot_key: tuple[int, int, int, int] | None = None
        self._model_leaf_depth_has_slot: tuple[bool, ...] = ()
        self._model_leaf_beliefs_valid: bool = False
        self._subgame_generation: int = 0
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
        self._subgame_capacity: int = 0
        self._subgame_env: HUNLTensorEnv | None = None
        self._subgame_arange: torch.Tensor | None = None
        self._subgame_parent_index: torch.Tensor | None = None
        self._subgame_action_from_parent: torch.Tensor | None = None
        self._subgame_rewards: torch.Tensor | None = None
        self._subgame_allin_leaf: torch.Tensor | None = None
        self._subgame_valid_mask: torch.Tensor | None = None
        self._subgame_root_mask: torch.Tensor | None = None
        self._subgame_new_street_mask: torch.Tensor | None = None
        self._subgame_leaf_mask: torch.Tensor | None = None
        self._subgame_legal_amounts: torch.Tensor | None = None
        self._subgame_legal_mask: torch.Tensor | None = None
        self._subgame_child_mask: torch.Tensor | None = None
        self._subgame_bet_bins_t: torch.Tensor | None = None
        self._subgame_prev_actor: torch.Tensor | None = None
        self._subgame_child_counts: torch.Tensor | None = None
        self._subgame_child_offsets: torch.Tensor | None = None

    def _build_hand_rank_data_for_indices(self, indices: torch.Tensor) -> HandRankData:
        device = self.device
        board = self.env.board_indices[indices].int()
        m = indices.numel()
        k = torch.arange(NUM_HANDS, device=device).expand(m, -1)

        import p2.search.cfr_evaluator as _ce

        hand_ranks, sorted_indices = _ce.rank_hands(board)
        ranks_sorted = torch.gather(hand_ranks, 1, sorted_indices)
        is_start = torch.ones_like(ranks_sorted, dtype=torch.bool)
        is_start[:, 1:] = ranks_sorted[:, 1:] != ranks_sorted[:, :-1]
        group_id = is_start.cumsum(dim=1, dtype=torch.int) - 1

        starts = torch.full((m, NUM_HANDS), NUM_HANDS, dtype=torch.int, device=device)
        ends = torch.full((m, NUM_HANDS), -1, dtype=torch.int, device=device)
        starts.scatter_reduce_(1, group_id, k.int(), reduce="amin", include_self=True)
        ends.scatter_reduce_(1, group_id, k.int(), reduce="amax", include_self=True)

        l_idx = torch.gather(starts, 1, group_id)
        r_idx = (torch.gather(ends, 1, group_id) + 1).clamp(max=NUM_HANDS)
        inv_sorted = torch.argsort(sorted_indices, dim=1)

        combo_to_onehot = combo_to_onehot_tensor(device=device)
        hands_c1c2 = hand_combos_tensor(device=device)
        card_ok = torch.ones((m, 52), dtype=torch.bool, device=device)
        card_ok.scatter_(1, board, False)
        h = combo_to_onehot.unsqueeze(0).expand(m, -1, -1)
        hand_ok_mask = self.allowed_hands[indices]
        hand_ok_mask_sorted = torch.gather(hand_ok_mask, 1, sorted_indices)
        hands_c1c2_sorted = torch.gather(
            hands_c1c2.unsqueeze(0).expand(m, -1, -1),
            1,
            sorted_indices.unsqueeze(-1).expand(-1, -1, 2),
        )
        return HandRankData(
            sorted_indices=sorted_indices,
            inv_sorted=inv_sorted,
            H=h,
            card_ok=card_ok,
            hand_ok_mask=hand_ok_mask,
            hand_ok_mask_sorted=hand_ok_mask_sorted,
            hands_c1c2_sorted=hands_c1c2_sorted,
            L_idx=l_idx,
            R_idx=r_idx,
        )

    def _init_hand_rank_data(self) -> None:
        """Build rank data by unique river-root board, then map each showdown
        leaf to that board's precompute row. Same-street construction never
        deals cards below a root, so river descendants share the root board."""
        self.showdown_indices = torch.where(self.env.street == 4)[0].contiguous()
        self.showdown_actors = self.env.to_act[self.showdown_indices]
        self.showdown_potential = (
            self.env.stacks[self.showdown_indices]
            + self.env.pot[self.showdown_indices, None]
            - self.env.starting_stacks[self.showdown_indices]
        )
        if self.showdown_indices.numel() > 0:
            root_index = self._get_root_index()
            showdown_roots = root_index[self.showdown_indices]
            rank_indices, extra_indices = torch.unique(
                showdown_roots, sorted=True, return_inverse=True
            )
            self.hand_rank_data = self._build_hand_rank_data_for_indices(rank_indices)
            self._showdown_extras = precompute_showdown_active_extras(
                self.hand_rank_data,
                self.env,
                rank_indices,
                scale_indices=self.showdown_indices,
                extra_indices=extra_indices,
            )
            source_beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs
            self._showdown_graph_runner = ShowdownActiveCardBlockDirectGraphRunner(
                extras=self._showdown_extras,
                M=self.showdown_indices.numel(),
                NUM_HANDS=self.beliefs.shape[-1],
                device=self.device,
                source_beliefs=source_beliefs,
                row_indices=self.showdown_indices,
                latest_values=self.latest_values,
            )
            self._showdown_fallback_extras = None
        else:
            self.hand_rank_data = None
            self._showdown_extras = None
            self._showdown_fallback_extras = None
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
        fallback_extras = getattr(self, "_showdown_fallback_extras", None)
        if fallback_extras is None:
            root_index = self._get_root_index()
            showdown_roots = root_index[self.showdown_indices]
            rank_indices, extra_indices = torch.unique(
                showdown_roots, sorted=True, return_inverse=True
            )
            fallback_extras = precompute_showdown_extras(
                self.hand_rank_data,
                self.env,
                rank_indices,
                scale_indices=self.showdown_indices,
                extra_indices=extra_indices,
            )
            self._showdown_fallback_extras = fallback_extras
        return showdown_ev_v15(beliefs, fallback_extras)

    def _record_stats(self, t: int, old_policy_probs: torch.Tensor) -> None:
        """Record policy-update stats with a fused CFR-delta reduction."""
        percentile_ts = (
            torch.linspace(self.warm_start_iterations, self.cfr_iterations - 1, 5)
            .round()
            .int()
            .tolist()
        )
        percentiles = [0, 25, 50, 75, 100]
        if t not in percentile_ts:
            return

        percentile = percentiles[percentile_ts.index(t)]
        top = self.depth_offsets[-2]
        if top == 0:
            return
        self._prepare_tree_slices()
        child_offsets_top = self._child_offsets_top
        child_count_top = self._child_count_top
        assert child_offsets_top is not None
        assert child_count_top is not None

        stats = fused_cfr_delta_stats(
            policy=self.policy_probs.contiguous(),
            old_policy=old_policy_probs.contiguous(),
            self_reach=self.self_reach.contiguous(),
            child_offsets=child_offsets_top,
            child_count=child_count_top,
            max_children=self.num_actions,
        )
        node_count = stats[1]
        node_delta_mean = torch.where(
            node_count > 0, stats[0] / node_count.clamp_min(1.0), 0.0
        )
        self.stats[f"cfr_delta.{percentile}"] = node_delta_mean.item()

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
        if not hasattr(self, "_last_instantaneous_regrets"):
            self._last_instantaneous_regrets = None
        if not hasattr(self, "_predictive_cfr_enabled"):
            self._predictive_cfr_enabled = self.cfr_type in (
                CFRType.pcfr,
                CFRType.sapcfr,
            )
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
        if not hasattr(self, "_sample_root_rows"):
            self._sample_root_rows = None
        if not hasattr(self, "_sample_root_counts"):
            self._sample_root_counts = None
        if not hasattr(self, "_sample_root_key"):
            self._sample_root_key = None
        if not hasattr(self, "_sample_leaf_enabled"):
            self._sample_leaf_enabled = False
        if not hasattr(self, "_sample_leaf_training_mode"):
            self._sample_leaf_training_mode = True
        if not hasattr(self, "_sample_leaf_epsilon"):
            self._sample_leaf_epsilon = 0.0
        if not hasattr(self, "_sample_leaf_players"):
            self._sample_leaf_players = None
        if not hasattr(self, "_sample_leaf_hands"):
            self._sample_leaf_hands = None
        if not hasattr(self, "_sample_leaf_uniform_draws"):
            self._sample_leaf_uniform_draws = None
        if not hasattr(self, "_sample_leaf_action_draws"):
            self._sample_leaf_action_draws = None
        if not hasattr(self, "_sample_leaf_indices_padded"):
            self._sample_leaf_indices_padded = None
        if not hasattr(self, "_sample_leaf_beliefs_padded"):
            self._sample_leaf_beliefs_padded = None
        if not hasattr(self, "_sample_leaf_ready_padded"):
            self._sample_leaf_ready_padded = None
        if not hasattr(self, "_sample_leaf_effective_leaf_mask"):
            self._sample_leaf_effective_leaf_mask = None
        if not hasattr(self, "_sample_leaf_sampling_masks"):
            self._sample_leaf_sampling_masks = None
        if not hasattr(self, "_sample_leaf_uniform_policy"):
            self._sample_leaf_uniform_policy = None
        if not hasattr(self, "_sample_leaf_child_nodes_by_action"):
            self._sample_leaf_child_nodes_by_action = None
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
        if not hasattr(self, "_model_leaf_scatter_enabled"):
            self._model_leaf_scatter_enabled = os.environ.get(
                "P2_FUSED_MODEL_LEAF_SCATTER", "1"
            ).strip().lower() not in {"0", "false", "off", "no"}
        if not hasattr(self, "_model_leaf_beliefs_buf"):
            self._model_leaf_beliefs_buf = None
        if not hasattr(self, "_model_leaf_slot"):
            self._model_leaf_slot = None
        if not hasattr(self, "_model_leaf_slot_key"):
            self._model_leaf_slot_key = None
        if not hasattr(self, "_model_leaf_depth_has_slot"):
            self._model_leaf_depth_has_slot = ()
        if not hasattr(self, "_model_leaf_beliefs_valid"):
            self._model_leaf_beliefs_valid = False
        if not hasattr(self, "_subgame_generation"):
            self._subgame_generation = 0
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
        if not hasattr(self, "_subgame_capacity"):
            self._subgame_capacity = 0
        if not hasattr(self, "_subgame_env"):
            self._subgame_env = None
        if not hasattr(self, "_subgame_arange"):
            self._subgame_arange = None
        if not hasattr(self, "_subgame_parent_index"):
            self._subgame_parent_index = None
        if not hasattr(self, "_subgame_action_from_parent"):
            self._subgame_action_from_parent = None
        if not hasattr(self, "_subgame_rewards"):
            self._subgame_rewards = None
        if not hasattr(self, "_subgame_allin_leaf"):
            self._subgame_allin_leaf = None
        if not hasattr(self, "_subgame_valid_mask"):
            self._subgame_valid_mask = None
        if not hasattr(self, "_subgame_root_mask"):
            self._subgame_root_mask = None
        if not hasattr(self, "_subgame_new_street_mask"):
            self._subgame_new_street_mask = None
        if not hasattr(self, "_subgame_leaf_mask"):
            self._subgame_leaf_mask = None
        if not hasattr(self, "_subgame_legal_amounts"):
            self._subgame_legal_amounts = None
        if not hasattr(self, "_subgame_legal_mask"):
            self._subgame_legal_mask = None
        if not hasattr(self, "_subgame_child_mask"):
            self._subgame_child_mask = None
        if not hasattr(self, "_subgame_bet_bins_t"):
            self._subgame_bet_bins_t = None
        if not hasattr(self, "_subgame_prev_actor"):
            self._subgame_prev_actor = None
        if not hasattr(self, "_subgame_child_counts"):
            self._subgame_child_counts = None
        if not hasattr(self, "_subgame_child_offsets"):
            self._subgame_child_offsets = None

    def _invalidate_subgame_caches(self) -> None:
        """Drop cached tensors whose contents depend on the current subgame.

        Persistent subgame buffers intentionally reuse storage across data-generator
        calls. Pointer-based cache keys alone are therefore not enough: a later
        subgame can have the same addresses and shapes but different states.
        """
        self._subgame_generation += 1
        self._root_index = None
        self._root_index_total = -1
        self._tree_slice_key = None
        self._static_model_base_key = None
        self._static_model_base_features = None
        self._static_model_feature_key = None
        self._static_model_feature_fields = None
        self._leaf_belief_gather_indices = None
        self._leaf_belief_gather_key = None
        self._model_leaf_slot = None
        self._model_leaf_slot_key = None
        self._model_leaf_depth_has_slot = ()
        self._model_leaf_beliefs_valid = False
        self._sample_root_rows = None
        self._sample_root_counts = None
        self._sample_root_key = None
        self._sample_leaf_enabled = False
        self._sample_leaf_players = None
        self._sample_leaf_hands = None
        self._sample_leaf_uniform_draws = None
        self._sample_leaf_action_draws = None
        self._sample_leaf_indices_padded = None
        self._sample_leaf_beliefs_padded = None
        self._sample_leaf_ready_padded = None
        self._sample_leaf_effective_leaf_mask = None
        self._sample_leaf_sampling_masks = None
        self._sample_leaf_uniform_policy = None
        self._sample_leaf_child_nodes_by_action = None
        self._fused_positive_regrets_valid = False
        self._exploitability_cache_key = None
        self._exploitability_cache = None

    def _ensure_subgame_capacity(self, capacity: int, proto: HUNLTensorEnv) -> None:
        if self._subgame_env is not None and self._subgame_capacity >= capacity:
            return

        capacity = max(1, int(capacity))
        self._subgame_capacity = capacity
        self._subgame_env = HUNLTensorEnv.from_proto(
            proto, num_envs=capacity, init_state=False
        )
        self._subgame_arange = torch.arange(capacity, device=self.device)
        self._subgame_parent_index = torch.empty(
            capacity, dtype=torch.long, device=self.device
        )
        self._subgame_action_from_parent = torch.empty(
            capacity, dtype=torch.long, device=self.device
        )
        self._subgame_rewards = torch.empty(
            capacity, dtype=self.float_dtype, device=self.device
        )
        self._subgame_allin_leaf = torch.empty(
            capacity, dtype=torch.bool, device=self.device
        )
        self._subgame_valid_mask = torch.empty(
            capacity, dtype=torch.bool, device=self.device
        )
        self._subgame_root_mask = torch.empty(
            capacity, dtype=torch.bool, device=self.device
        )
        self._subgame_new_street_mask = torch.empty(
            capacity, dtype=torch.bool, device=self.device
        )
        self._subgame_leaf_mask = torch.empty(
            capacity, dtype=torch.bool, device=self.device
        )
        self._subgame_legal_amounts = torch.empty(
            capacity, self.num_actions, dtype=torch.long, device=self.device
        )
        self._subgame_legal_mask = torch.empty(
            capacity, self.num_actions, dtype=torch.bool, device=self.device
        )
        self._subgame_child_mask = torch.empty(
            capacity, self.num_actions, dtype=torch.bool, device=self.device
        )
        self._subgame_prev_actor = torch.empty(
            capacity, dtype=torch.long, device=self.device
        )
        self._subgame_child_counts = torch.empty(
            capacity, dtype=torch.long, device=self.device
        )
        self._subgame_child_offsets = torch.empty(
            capacity, dtype=torch.long, device=self.device
        )
        self._subgame_bet_bins_t = torch.tensor(
            [0] * 2 + self.bet_bins + [0],
            dtype=torch.float32,
            device=self.device,
        )

    def _active_subgame_env_view(self, total_nodes: int) -> HUNLTensorEnv:
        assert self._subgame_env is not None
        assert self._subgame_arange is not None
        env = copy.copy(self._subgame_env)
        env.N = total_nodes
        env.arange_n = self._subgame_arange[:total_nodes]
        for name in (
            "deck",
            "deck_pos",
            "button",
            "street",
            "to_act",
            "last_to_act",
            "pot",
            "min_raise",
            "actions_this_round",
            "actions_last_round",
            "acted_since_reset",
            "stacks",
            "committed",
            "has_folded",
            "is_allin",
            "starting_stacks",
            "scale",
            "board_onehot",
            "hole_onehot",
            "board_indices",
            "last_board_indices",
            "hole_indices",
            "chips_placed",
            "done",
            "winner",
        ):
            setattr(env, name, getattr(self._subgame_env, name)[:total_nodes])
        return env

    def _legal_bins_amounts_and_mask_range(
        self, start: int, end: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._subgame_env is not None
        assert self._subgame_arange is not None
        assert self._subgame_legal_amounts is not None
        assert self._subgame_legal_mask is not None
        assert self._subgame_bet_bins_t is not None
        env = self._subgame_env
        rows = self._subgame_arange[start:end]
        n = end - start
        amounts = self._subgame_legal_amounts[:n]
        mask = self._subgame_legal_mask[:n]
        amounts.fill_(-1)
        mask.zero_()

        me = env.to_act[rows]
        opp = 1 - me
        me_stack = env.stacks[rows, me]
        opp_stack = env.stacks[rows, opp]
        me_committed = env.committed[rows, me]
        opp_committed = env.committed[rows, opp]
        to_call = opp_committed - me_committed

        mask[:, 0] = to_call > 0
        mask[:, 1] = True

        can_bet_raise = (me_stack > 0) & (opp_stack > 0)
        additional_amounts = (
            self._subgame_bet_bins_t[2:-1].view(1, self.num_actions - 3)
            * env.pot[rows].view(n, 1)
        ).long()
        bet_raise_amounts = to_call.view(n, 1) + additional_amounts
        bet_raise_legal = (
            can_bet_raise.view(n, 1)
            & (bet_raise_amounts <= me_stack.view(n, 1))
            & (additional_amounts >= env.min_raise[rows].view(n, 1))
        )
        amounts[:, 2:-1] = bet_raise_amounts
        mask[:, 2:-1] = bet_raise_legal

        amounts[:, self.num_actions - 1] = me_stack
        mask[:, self.num_actions - 1] = me_stack > 0

        me_allin = env.is_allin[rows, me]
        opp_allin = env.is_allin[rows, opp]
        mask[:, 0] = torch.where(opp_allin, True, mask[:, 0])
        mask[:, 2:-1] = torch.where(opp_allin[:, None], False, mask[:, 2:-1])
        mask[:, self.num_actions - 1] = torch.where(
            opp_allin, False, mask[:, self.num_actions - 1]
        )
        mask[:, 0] = torch.where(me_allin, False, mask[:, 0])
        mask[:, 1] = torch.where(me_allin, True, mask[:, 1])
        mask[:, 2:-1] = torch.where(me_allin[:, None], False, mask[:, 2:-1])
        mask[:, self.num_actions - 1] = torch.where(
            me_allin, False, mask[:, self.num_actions - 1]
        )
        mask &= (~env.done[rows])[:, None]
        return amounts, mask

    def _allin_call_child_mask_rows(
        self, parent_rows: torch.Tensor, action_bins: torch.Tensor
    ) -> torch.Tensor:
        assert self._subgame_env is not None
        if not self._allin_abstraction_enabled() or action_bins.numel() == 0:
            return torch.zeros_like(action_bins, dtype=torch.bool)
        env = self._subgame_env
        actor = env.to_act[parent_rows]
        opp = 1 - actor
        parent_to_call = (
            env.committed[parent_rows, opp] - env.committed[parent_rows, actor]
        )
        parent_street = env.street[parent_rows]
        mask = (
            (action_bins == 1)
            & env.is_allin[parent_rows, opp]
            & (parent_to_call > 0)
            & (parent_street > 0)
            & (parent_street < 3)
        )
        return mask

    def _bet_rows(
        self, rows: torch.Tensor, players: torch.Tensor, chips: torch.Tensor
    ) -> None:
        assert self._subgame_env is not None
        env = self._subgame_env
        env.stacks[rows, players] -= chips
        env.committed[rows, players] += chips
        env.pot[rows] += chips
        env.chips_placed[rows, players] += chips

    def _step_bins_range(
        self,
        start: int,
        count: int,
        action_bins: torch.Tensor,
        selected_amounts: torch.Tensor,
    ) -> torch.Tensor:
        assert self._subgame_env is not None
        assert self._subgame_arange is not None
        assert self._subgame_rewards is not None
        env = self._subgame_env
        rows = self._subgame_arange[start : start + count]
        rewards = self._subgame_rewards[start : start + count]
        rewards.zero_()

        all_in_index = self.num_actions - 1
        actor_idx = env.to_act[rows]
        other_idx = 1 - actor_idx
        actor_stack = env.stacks[rows, actor_idx]
        actor_committed = env.committed[rows, actor_idx]
        other_committed = env.committed[rows, other_idx]
        to_call = other_committed - actor_committed

        is_fold = action_bins == 0
        is_check_call = action_bins == 1
        is_bet_raise = (action_bins >= 2) & (action_bins < all_in_index)
        is_allin = action_bins == all_in_index

        fold_rows = rows[is_fold]
        if fold_rows.numel() > 0:
            rewards[is_fold] = env.finish_and_assign_rewards(
                fold_rows, other_idx[is_fold]
            )

        call_rows = rows[is_check_call]
        if call_rows.numel() > 0:
            call_amount = torch.minimum(
                to_call[is_check_call], actor_stack[is_check_call]
            )
            self._bet_rows(call_rows, actor_idx[is_check_call], call_amount)

        allin_rows = rows[is_allin]
        if allin_rows.numel() > 0:
            all_in_amount = actor_stack[is_allin]
            allin_actor = actor_idx[is_allin]
            self._bet_rows(allin_rows, allin_actor, all_in_amount)
            env.is_allin[allin_rows, allin_actor] = True

        bet_rows = rows[is_bet_raise]
        if bet_rows.numel() > 0:
            bet_raise_amount = selected_amounts[is_bet_raise]
            self._bet_rows(bet_rows, actor_idx[is_bet_raise], bet_raise_amount)
            env.min_raise[bet_rows] = torch.maximum(
                env.min_raise[bet_rows], bet_raise_amount - to_call[is_bet_raise]
            )

        env.last_to_act[rows] = actor_idx
        env.to_act[rows] = other_idx
        env.actions_this_round[rows] += 1
        env.acted_since_reset[rows] = True
        env.last_board_indices[rows, :] = env.board_indices[rows, :]

        equal_committed = env.committed[rows, 0] == env.committed[rows, 1]
        all_in_committed = (
            (env.is_allin[rows, 0] & env.is_allin[rows, 1])
            | (
                env.is_allin[rows, 0]
                & (env.committed[rows, 0] <= env.committed[rows, 1])
            )
            | (
                env.is_allin[rows, 1]
                & (env.committed[rows, 1] <= env.committed[rows, 0])
            )
        )
        round_closed = (
            ~env.done[rows]
            & (equal_committed | all_in_committed)
            & (env.actions_this_round[rows] >= 2)
        )
        round_rows = rows[round_closed]
        if round_rows.numel() == 0:
            return rewards

        env.committed[round_rows, :] = 0
        env.actions_last_round[round_rows] = env.actions_this_round[round_rows]
        env.actions_this_round[round_rows] = 0
        env.to_act[round_rows] = 1 - env.button[round_rows]
        env.min_raise[round_rows] = env.bb
        street = env.street[round_rows]

        showdown_mask = street == (0 if env.flop_showdown else 3)
        showdown_rows = round_rows[showdown_mask]
        if showdown_rows.numel() > 0:
            ab_plane = env.hole_onehot[showdown_rows].any(dim=2) | env.board_onehot[
                showdown_rows
            ].any(dim=1).unsqueeze(1)
            cmp = rules.compare_7_single_batch(ab_plane)
            env.winner[showdown_rows[cmp > 0]] = 0
            env.winner[showdown_rows[cmp < 0]] = 1
            env.winner[showdown_rows[cmp == 0]] = 2
            local = showdown_rows - start
            rewards[local] = env.finish_and_assign_rewards(
                showdown_rows, env.winner[showdown_rows]
            )

        flop_rows = round_rows[street == 0]
        if flop_rows.numel() > 0:
            pos = env.deck_pos[flop_rows]
            c0 = env.deck[flop_rows, pos]
            c1 = env.deck[flop_rows, pos + 1]
            c2 = env.deck[flop_rows, pos + 2]
            env.deck_pos[flop_rows] = pos + 3
            env.board_onehot[flop_rows, 0] = env.card_onehot_cache[c0]
            env.board_onehot[flop_rows, 1] = env.card_onehot_cache[c1]
            env.board_onehot[flop_rows, 2] = env.card_onehot_cache[c2]
            env.board_indices[flop_rows, 0] = c0
            env.board_indices[flop_rows, 1] = c1
            env.board_indices[flop_rows, 2] = c2

        turn_rows = round_rows[street == 1]
        if turn_rows.numel() > 0:
            pos = env.deck_pos[turn_rows]
            c = env.deck[turn_rows, pos]
            env.deck_pos[turn_rows] = pos + 1
            env.board_onehot[turn_rows, 3] = env.card_onehot_cache[c]
            env.board_indices[turn_rows, 3] = c

        river_rows = round_rows[street == 2]
        if river_rows.numel() > 0:
            pos = env.deck_pos[river_rows]
            c = env.deck[river_rows, pos]
            env.deck_pos[river_rows] = pos + 1
            env.board_onehot[river_rows, 4] = env.card_onehot_cache[c]
            env.board_indices[river_rows, 4] = c

        env.street[round_rows] += 1
        return rewards

    def _mark_constructed_allin_call_leaves(
        self, allin_leaf_tensor: torch.Tensor
    ) -> None:
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        self.allin_call_indices = empty
        self.allin_call_parent_indices = empty
        self.allin_call_indices_by_street = (empty, empty, empty)
        self.allin_call_parent_indices_by_street = (empty, empty, empty)
        empty_boards = torch.empty(0, 5, dtype=torch.long, device=self.device)
        self.allin_call_boards_by_street = (empty_boards, empty_boards, empty_boards)
        self.allin_call_mask = torch.zeros(
            self.total_nodes, dtype=torch.bool, device=self.device
        )
        if not self._allin_abstraction_enabled() or self.total_nodes <= self.root_nodes:
            self._cache_allin_call_street_partitions(empty)
            return

        indices = torch.nonzero(allin_leaf_tensor, as_tuple=False).flatten()
        if indices.numel() == 0:
            self._cache_allin_call_street_partitions(empty)
            return

        self.allin_call_indices = indices.contiguous()
        self.allin_call_parent_indices = self.parent_index[indices].contiguous()
        self.allin_call_mask[self.allin_call_indices] = True
        self.leaf_mask[self.allin_call_indices] = True
        self.new_street_mask[self.allin_call_indices] = False
        parent_street = self.env.street[self.allin_call_parent_indices].contiguous()
        self._cache_allin_call_street_partitions(parent_street)

    def _cache_allin_turn_tables(self) -> None:
        super()._cache_allin_turn_tables()
        turn_ids = getattr(self, "allin_turn_table_ids", None)
        if turn_ids is None or turn_ids.numel() == 0:
            empty = torch.empty(0, 0, dtype=torch.long, device=self.device)
            self.allin_turn_group_positions = empty
            return
        counts = torch.bincount(
            turn_ids, minlength=int(self.allin_turn_tables_i16.shape[0])
        )
        max_count = int(counts.max().item()) if counts.numel() > 0 else 0
        if max_count <= 0:
            self.allin_turn_group_positions = torch.empty(
                counts.numel(), 0, dtype=torch.long, device=self.device
            )
            return
        order = torch.argsort(turn_ids)
        sorted_ids = turn_ids[order]
        offsets = counts.cumsum(0)
        starts = offsets - counts
        local = torch.arange(order.numel(), device=self.device) - starts[sorted_ids]
        group_positions = torch.full(
            (counts.numel(), max_count),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        group_positions[sorted_ids, local] = order
        self.allin_turn_group_positions = group_positions.contiguous()

    def _construct_subgame(
        self, src_env: HUNLTensorEnv, src_indices: torch.Tensor
    ) -> None:
        self._ensure_fused_attrs()
        assert src_indices.dim() == 1, "src_indices must be 1-D"
        num_roots = src_indices.shape[0]
        assert num_roots > 0, "must supply at least one root state"

        capacity = max(
            self._subgame_capacity,
            int(num_roots)
            * max(2, self.num_actions)
            * max(1, self.max_depth * self.max_depth),
        )
        while True:
            self._ensure_subgame_capacity(capacity, src_env)
            if self._construct_subgame_attempt(src_env, src_indices):
                return
            capacity *= 2

    def _construct_subgame_attempt(
        self, src_env: HUNLTensorEnv, src_indices: torch.Tensor
    ) -> bool:
        assert self._subgame_env is not None
        assert self._subgame_arange is not None
        assert self._subgame_parent_index is not None
        assert self._subgame_action_from_parent is not None
        assert self._subgame_rewards is not None
        assert self._subgame_allin_leaf is not None
        assert self._subgame_valid_mask is not None
        assert self._subgame_root_mask is not None
        assert self._subgame_new_street_mask is not None
        assert self._subgame_leaf_mask is not None
        assert self._subgame_child_mask is not None
        assert self._subgame_prev_actor is not None
        assert self._subgame_child_counts is not None
        assert self._subgame_child_offsets is not None
        assert self._subgame_legal_mask is not None
        assert self._subgame_bet_bins_t is not None
        num_roots = int(src_indices.shape[0])
        capacity = self._subgame_capacity
        env = self._subgame_env

        with _init_profile_region("cfr_init_attempt_gather_env"):
            gather_env_rows_into_triton(src_env, env, src_indices, dst_start=0)
        with _init_profile_region("cfr_init_attempt_root_init"):
            self._subgame_parent_index[:num_roots].fill_(-1)
            self._subgame_action_from_parent[:num_roots].fill_(-1)
            self._subgame_rewards[:num_roots].zero_()
            self._subgame_allin_leaf[:num_roots].zero_()
        allin_abstraction = self._allin_abstraction_enabled()

        depth_offsets = [0, num_roots]
        cursor = num_roots
        depth = 0
        while depth < self.max_depth:
            parent_start = depth_offsets[-2]
            parent_end = depth_offsets[-1]
            parent_count = parent_end - parent_start
            child_counts = self._subgame_child_counts[:parent_count]
            child_offsets = self._subgame_child_offsets[:parent_count]
            with _init_profile_region("cfr_init_attempt_legal_counts"):
                legal_counts_triton_(
                    env,
                    self._subgame_legal_mask,
                    child_counts,
                    self._subgame_allin_leaf,
                    self._subgame_bet_bins_t,
                    parent_start=parent_start,
                    parent_count=parent_count,
                    stop_new_street=depth > 0,
                    num_actions=self.num_actions,
                )
                self._subgame_legal_mask[parent_start:parent_end] &= (
                    self._action_mask_for_depth(depth)[None, :]
                )
                child_counts.copy_(
                    self._subgame_legal_mask[parent_start:parent_end].sum(dim=-1)
                )
            with _init_profile_region("cfr_init_attempt_cumsum_offsets"):
                torch.cumsum(child_counts, dim=0, out=child_offsets)
                child_offsets -= child_counts
            with _init_profile_region("cfr_init_attempt_child_count_sync"):
                child_count = int(child_counts.sum().item())
            if child_count == 0:
                break
            child_end = cursor + child_count
            if child_end > capacity:
                return False

            with _init_profile_region("cfr_init_attempt_write_children"):
                write_children_same_street_triton_(
                    env,
                    self._subgame_legal_mask,
                    child_offsets,
                    self._subgame_parent_index,
                    self._subgame_action_from_parent,
                    self._subgame_rewards,
                    self._subgame_allin_leaf,
                    self._subgame_bet_bins_t,
                    parent_start=parent_start,
                    parent_count=parent_count,
                    dst_start=cursor,
                    allin_abstraction=allin_abstraction,
                )
            with _init_profile_region("cfr_init_attempt_copy_child_cards"):
                copy_child_cards_triton_(
                    env,
                    self._subgame_parent_index,
                    child_start=cursor,
                    child_count=child_count,
                )

            cursor = child_end
            depth += 1
            depth_offsets.append(cursor)

        self.depth_offsets = depth_offsets
        self.tree_depth = max(0, len(self.depth_offsets) - 2)
        self.total_nodes = cursor
        self.root_nodes = num_roots
        self.top_nodes = (
            self.depth_offsets[-2] if len(self.depth_offsets) > 1 else num_roots
        )

        self.env = self._active_subgame_env_view(cursor)
        self.parent_index = self._subgame_parent_index[:cursor]
        self.action_from_parent = self._subgame_action_from_parent[:cursor]
        self._root_index = None
        self._root_index_total = -1
        rewards_tensor = self._subgame_rewards[:cursor]
        allin_leaf_tensor = self._subgame_allin_leaf[:cursor]

        self.valid_mask = self._subgame_valid_mask[:cursor]
        with _init_profile_region("cfr_init_attempt_final_legal_counts"):
            legal_counts_triton_(
                env,
                self._subgame_legal_mask,
                self._subgame_child_counts[:cursor],
                self._subgame_allin_leaf,
                self._subgame_bet_bins_t,
                parent_start=0,
                parent_count=cursor,
                stop_new_street=False,
                num_actions=self.num_actions,
            )
        self.legal_mask = self._subgame_legal_mask[:cursor]
        self._apply_depth_action_masks(self.legal_mask)

        root_mask = self._subgame_root_mask[:cursor]
        self.new_street_mask = self._subgame_new_street_mask[:cursor]
        self.leaf_mask = self._subgame_leaf_mask[:cursor]
        self.child_mask = self._subgame_child_mask[:cursor]
        self.child_count = self._subgame_child_counts[:cursor]
        self.prev_actor = self._subgame_prev_actor[:cursor]
        with _init_profile_region("cfr_init_attempt_finalize_masks"):
            finalize_tree_masks_triton_(
                env,
                self.legal_mask,
                self.parent_index,
                allin_leaf_tensor,
                self.valid_mask,
                root_mask,
                self.new_street_mask,
                self.leaf_mask,
                self.child_mask,
                self.child_count,
                self.prev_actor,
                total_nodes=cursor,
                root_nodes=self.root_nodes,
                top_nodes=self.top_nodes,
                num_actions=self.num_actions,
            )
        self._mark_constructed_allin_call_leaves(allin_leaf_tensor)
        self._refresh_model_indices()
        self._validate_model_leaf_phases()
        bottom = self.depth_offsets[1]
        with _init_profile_region("cfr_init_attempt_child_offsets"):
            self.child_offsets = (
                bottom + torch.cumsum(self.child_count, dim=0) - self.child_count
            )

        with _init_profile_region("cfr_init_attempt_showdown_setup"):
            showdown_padding = max(1, self.root_nodes // 2)
            self.showdown_indices = padded_indices(
                self.env.street[:cursor] == 4, showdown_padding
            )
            self.showdown_actors = self.env.to_act[self.showdown_indices]
            self.showdown_potential = (
                self.env.stacks[self.showdown_indices]
                + self.env.pot[self.showdown_indices, None]
                - self.env.starting_stacks[self.showdown_indices]
            )

        with _init_profile_region("cfr_init_attempt_policy_alloc"):
            self.policy_probs = torch.empty(
                cursor, NUM_HANDS, dtype=self.float_dtype, device=self.device
            )
            self.policy_probs_avg = torch.empty_like(self.policy_probs)
            self.average_policy_numerator = torch.empty_like(self.policy_probs)
            self.average_policy_denominator = torch.empty_like(self.policy_probs)
            self.average_policy_initialized = False
            self.policy_probs_sample = torch.empty_like(self.policy_probs)
            self.cumulative_regrets = torch.empty_like(self.policy_probs)
            self._last_instantaneous_regrets = torch.zeros_like(self.policy_probs)
            self.uniform_policy = torch.empty_like(self.policy_probs)
        with _init_profile_region("cfr_init_attempt_policy_init"):
            init_policy_tensors_triton_(
                self.uniform_policy,
                self.cumulative_regrets,
                self.parent_index,
                self.child_count,
                total_nodes=cursor,
                root_nodes=self.root_nodes,
                num_hands=NUM_HANDS,
            )

        with _init_profile_region("cfr_init_attempt_belief_alloc"):
            self.beliefs = torch.empty(
                cursor,
                self.num_players,
                NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
            self.beliefs_avg = torch.empty_like(self.beliefs)
            self.beliefs_sample = torch.empty_like(self.beliefs)
            self.root_pre_chance_beliefs = torch.empty(
                num_roots,
                self.num_players,
                NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
            self.self_reach = torch.empty_like(self.beliefs)
            self.self_reach_avg = torch.empty_like(self.beliefs)

            self.latest_values = torch.empty_like(self.beliefs)
        with _init_profile_region("cfr_init_attempt_belief_init"):
            init_belief_value_tensors_triton_(
                self.beliefs,
                self.self_reach,
                self.latest_values,
                self.action_from_parent,
                self.env.done,
                rewards_tensor,
                total_nodes=cursor,
                root_nodes=self.root_nodes,
                num_hands=NUM_HANDS,
            )
        with _init_profile_region("cfr_init_attempt_values_avg_alloc"):
            self.values_avg = torch.empty_like(self.latest_values)
        self.last_model_values = None

        with _init_profile_region("cfr_init_attempt_feature_encoder"):
            self._init_fused_feature_encoders()
        return True

    def _init_fused_feature_encoders(self) -> None:
        self.policy_feature_encoder = self.policy_model.create_feature_encoder(
            env=self.env, device=self.device, dtype=self.float_dtype
        )
        self.value_feature_encoder = self.value_model.create_feature_encoder(
            env=self.env, device=self.device, dtype=self.float_dtype
        )
        self.closing_leaf_value_encoder = None
        if self.closing_leaf_value_model is not None:
            closing_players = int(
                getattr(
                    self.closing_leaf_value_model,
                    "num_players",
                    self.num_players,
                )
            )
            if closing_players != self.num_players:
                if not self._can_project_heads_up_closing_model():
                    raise ValueError(
                        "closing leaf value model num_players="
                        f"{closing_players} is incompatible with evaluator "
                        f"num_players={self.num_players}"
                    )
                self.closing_leaf_value_encoder = (
                    self.closing_leaf_value_model.create_feature_encoder(
                        env=self.env,
                        device=self.device,
                        dtype=self.float_dtype,
                    )
                )
            else:
                self.closing_leaf_value_encoder = (
                    self.closing_leaf_value_model.create_feature_encoder(
                        env=self.env,
                        device=self.device,
                        dtype=self.float_dtype,
                    )
                )
        self.feature_encoder = self.policy_feature_encoder

    def _root_allowed_from_board_indices(
        self, n: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_allowed = torch.empty(n, NUM_HANDS, dtype=torch.bool, device=self.device)
        root_allowed_prob = torch.empty(
            n, NUM_HANDS, dtype=self.float_dtype, device=self.device
        )
        combo_card_a, combo_card_b = combo_cards_i32_tensor(device=self.device)
        root_allowed_from_board_indices_triton_(
            self.env.board_indices,
            combo_card_a,
            combo_card_b,
            root_allowed,
            root_allowed_prob,
            n_roots=n,
            num_hands=NUM_HANDS,
        )
        return root_allowed, root_allowed_prob

    def initialize_subgame(
        self,
        src_env: HUNLTensorEnv,
        src_indices: torch.Tensor,
        initial_beliefs: torch.Tensor | None = None,
    ) -> None:
        self._construct_subgame(src_env, src_indices)
        self._invalidate_subgame_caches()
        n = self.root_nodes

        if initial_beliefs is None:
            initial_beliefs = torch.full(
                (n, self.num_players, NUM_HANDS),
                1.0 / NUM_HANDS,
                dtype=self.float_dtype,
                device=self.device,
            )
        else:
            initial_beliefs = initial_beliefs.to(
                device=self.device, dtype=self.float_dtype
            )

        self.root_pre_chance_beliefs[:n] = initial_beliefs
        self.self_reach[:n] = 1.0
        self.self_reach_avg[:n] = 1.0

        self._refresh_model_indices()
        self._validate_model_leaf_phases()
        self.latent = None

        root_allowed, root_allowed_prob = self._root_allowed_from_board_indices(n)
        self.allowed_hands = self._fan_out_deep(root_allowed)
        self.allowed_hands_prob = self._fan_out_deep(root_allowed_prob)

        root_beliefs = self.beliefs[:n]
        root_beliefs[:] = initial_beliefs
        root_beliefs.masked_fill_((~root_allowed)[:, None, :], 0.0)
        denom = root_beliefs.sum(dim=-1, keepdim=True)
        torch.where(
            denom > 1e-5,
            root_beliefs / denom,
            root_allowed_prob[:, None, :],
            out=root_beliefs,
        )
        self.beliefs_avg[:n] = root_beliefs

        self._init_hand_rank_data()
        self.stats["evaluator_street"] = self.env.street[:n].float().mean().item()
        self.stats["evaluator_total_nodes"] = float(self.total_nodes)
        self.stats["evaluator_root_nodes"] = float(self.root_nodes)
        self.stats["evaluator_tree_depth"] = float(self.tree_depth)

        self._prepare_tree_slices()
        self._reset_average_policy_accumulators()

    def _prepare_tree_slices(self) -> None:
        """Cache static contiguous tree slices for the current sparse subgame."""
        bottom = (
            self.depth_offsets[1] if len(self.depth_offsets) > 1 else self.root_nodes
        )
        top = self.depth_offsets[-2] if len(self.depth_offsets) > 1 else self.root_nodes
        key = (
            int(self._subgame_generation),
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

    def _ensure_last_instantaneous_regrets_buf(self) -> torch.Tensor:
        last = getattr(self, "_last_instantaneous_regrets", None)
        if last is None or last.shape != self.cumulative_regrets.shape:
            self._last_instantaneous_regrets = torch.zeros_like(self.cumulative_regrets)
        assert self._last_instantaneous_regrets is not None
        return self._last_instantaneous_regrets

    def _predictive_policy_scale_for_t(self, t: int) -> float:
        if not getattr(self, "_predictive_cfr_enabled", False):
            return 0.0
        if not self._predictive_cfr_active(t):
            return 0.0
        return self._predictive_cfr_prediction_scale()

    def _uses_dcfr_backbone(self) -> bool:
        return self.cfr_type == CFRType.discounted or self._predictive_cfr_uses_dcfr()

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
            self._ev_marginal_policy_buf = self.policy_probs.new_empty(marginal_shape)
        return self._ev_actor_beliefs_buf, self._ev_marginal_policy_buf

    def _ensure_regret_src_buffers(self, top: int) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _prepare_sample_root_table(self) -> None:
        N = self.root_nodes
        key = (
            int(self.t_sample.data_ptr()),
            int(N),
            int(self.cfr_iterations),
        )
        if self._sample_root_key == key:
            return

        t_sample_root = self.t_sample[:N].to(device=self.device, dtype=torch.long)
        valid_sample = t_sample_root < self.cfr_iterations
        valid_roots = torch.nonzero(valid_sample, as_tuple=False).flatten()
        t_sample_valid = t_sample_root.index_select(0, valid_roots)
        counts = torch.bincount(
            t_sample_valid,
            minlength=self.cfr_iterations,
        )[: self.cfr_iterations].contiguous()
        max_updates = int(counts.max().item()) if counts.numel() else 0
        rows = torch.full(
            (self.cfr_iterations, max_updates),
            N,
            dtype=torch.long,
            device=self.device,
        )
        if max_updates > 0:
            order = torch.argsort(t_sample_valid, stable=True)
            sorted_t = t_sample_valid.index_select(0, order)
            sorted_roots = valid_roots.index_select(0, order)
            starts = torch.cumsum(counts, dim=0) - counts
            position = torch.arange(
                order.numel(),
                device=self.device,
                dtype=torch.long,
            ) - starts.index_select(0, sorted_t)
            rows[sorted_t, position] = sorted_roots

        self._sample_root_rows = rows.contiguous()
        self._sample_root_counts = counts
        self._sample_root_key = key

    def _prepare_compact_leaf_sampling(self, training_mode: bool) -> None:
        self._prepare_sample_root_table()
        N = self.root_nodes
        top = self.depth_offsets[-2]
        depth = max(0, int(self.tree_depth))
        self._sample_leaf_enabled = True
        self._sample_leaf_training_mode = training_mode
        self._sample_leaf_epsilon = self.sample_epsilon if training_mode else 0.0
        self._sample_leaf_players = torch.randint(
            0, 2, (N,), generator=self.generator, device=self.device
        )
        self._sample_leaf_hands = self._sample_root_hands_by_player()
        self._sample_leaf_uniform_draws = torch.rand(
            depth, N, generator=self.generator, device=self.device
        )
        self._sample_leaf_action_draws = torch.rand(
            depth, N, generator=self.generator, device=self.device
        )
        target_enabled = torch.zeros(N, dtype=torch.bool, device=self.device)
        target_depths = torch.full((N,), depth + 1, dtype=torch.long, device=self.device)
        if training_mode and self._continuation_value_target_sampling_enabled():
            bounds = self._continuation_value_target_depth_bounds()
            target_streets = self._continuation_value_target_streets()
            if bounds is not None and target_streets:
                min_depth, max_depth = bounds
                root_street = self.env.street[:N]
                for street in target_streets:
                    target_enabled |= root_street == street
                if bool(target_enabled.any().item()):
                    draws = torch.randint(
                        min_depth,
                        max_depth + 1,
                        (N,),
                        generator=self.generator,
                        device=self.device,
                    )
                    target_depths = torch.where(target_enabled, draws, target_depths)
        self._sample_leaf_target_enabled = target_enabled
        self._sample_leaf_target_depths = target_depths
        self._sample_leaf_indices_padded = torch.full(
            (N + 1,), N, dtype=torch.long, device=self.device
        )
        self._sample_leaf_indices_padded[:N] = torch.arange(N, device=self.device)
        self._sample_leaf_beliefs_padded = torch.zeros(
            N + 1,
            self.num_players,
            NUM_HANDS,
            dtype=self.float_dtype,
            device=self.device,
        )
        self._sample_leaf_ready_padded = torch.zeros(
            N + 1, dtype=torch.bool, device=self.device
        )

        # Mirror the sparse/rebel filters: skip done children and all-in-call
        # leaves so the fused kernel never re-roots at one. See
        # sparse_cfr_evaluator.sample_leaves for rationale.
        done_src = self._pull_back(self.env.done)
        allin_src = self._pull_back(self.allin_call_mask)
        skip_src = done_src | allin_src
        sampling_masks = self.child_mask.clone()
        sampling_masks[:top] &= ~skip_src
        sampling_counts = sampling_masks.float().sum(dim=-1, keepdim=True)
        self._sample_leaf_sampling_masks = sampling_masks
        self._sample_leaf_uniform_policy = torch.where(
            sampling_masks,
            sampling_counts.clamp(min=1.0).reciprocal(),
            torch.zeros_like(sampling_counts),
        )
        self._sample_leaf_effective_leaf_mask = self.leaf_mask | (
            sampling_counts.squeeze(1) == 0
        )

        child_nodes_by_action = torch.full(
            (self.total_nodes, self.num_actions),
            0,
            dtype=torch.long,
            device=self.device,
        )
        nonroot = torch.arange(
            self.root_nodes, self.total_nodes, dtype=torch.long, device=self.device
        )
        child_nodes_by_action[
            self.parent_index[nonroot],
            self.action_from_parent[nonroot],
        ] = nonroot
        self._sample_leaf_child_nodes_by_action = child_nodes_by_action

    def _sample_compact_leaves_for_current_t(self) -> None:
        if not self._sample_leaf_enabled:
            return
        assert self._sample_root_rows is not None
        assert self._sample_root_counts is not None
        assert self._sample_leaf_players is not None
        assert self._sample_leaf_hands is not None
        assert self._sample_leaf_uniform_draws is not None
        assert self._sample_leaf_action_draws is not None
        assert self._sample_leaf_target_enabled is not None
        assert self._sample_leaf_target_depths is not None
        assert self._sample_leaf_indices_padded is not None
        assert self._sample_leaf_beliefs_padded is not None
        assert self._sample_leaf_ready_padded is not None
        assert self._sample_leaf_effective_leaf_mask is not None
        assert self._sample_leaf_sampling_masks is not None
        assert self._sample_leaf_uniform_policy is not None
        assert self._sample_leaf_child_nodes_by_action is not None

        fused_sample_leaf_compact_(
            policy_probs=self.policy_probs.contiguous(),
            beliefs=self.beliefs.contiguous(),
            sample_root_rows=self._sample_root_rows,
            sample_root_counts=self._sample_root_counts,
            t=self._t_scalars.t_tensor,
            players=self._sample_leaf_players.contiguous(),
            hands=self._sample_leaf_hands.contiguous(),
            uniform_draws=self._sample_leaf_uniform_draws.contiguous(),
            action_draws=self._sample_leaf_action_draws.contiguous(),
            target_enabled=self._sample_leaf_target_enabled.contiguous(),
            target_depths=self._sample_leaf_target_depths.contiguous(),
            effective_leaf_mask=self._sample_leaf_effective_leaf_mask.contiguous(),
            sampling_masks=self._sample_leaf_sampling_masks.contiguous(),
            uniform_policy=self._sample_leaf_uniform_policy.contiguous(),
            child_nodes_by_action=self._sample_leaf_child_nodes_by_action.contiguous(),
            to_act=self.env.to_act.contiguous(),
            done=self.env.done.contiguous(),
            allin_call_mask=self.allin_call_mask.contiguous(),
            new_street_mask=self.new_street_mask.contiguous(),
            out_nodes=self._sample_leaf_indices_padded,
            out_beliefs=self._sample_leaf_beliefs_padded,
            out_ready=self._sample_leaf_ready_padded,
            sample_epsilon=self._sample_leaf_epsilon,
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
            beliefs=beliefs_at_model.reshape(-1, 2 * NUM_HANDS),
        )

    def _leaf_beliefs_for_model(self, beliefs: torch.Tensor) -> torch.Tensor:
        m = int(self.model_indices.numel())
        key = (
            int(self._subgame_generation),
            int(self.model_indices.data_ptr()),
            m,
        )
        if (
            self._leaf_belief_gather_key != key
            or self._leaf_belief_gather_indices is None
        ):
            self._leaf_belief_gather_indices = self.model_indices.contiguous()
            self._leaf_belief_gather_key = key

        return beliefs[self._leaf_belief_gather_indices]

    def _ensure_model_leaf_belief_buffers(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[bool, ...]]:
        m = int(self.model_indices.numel())
        shape = (m, self.num_players, NUM_HANDS)
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
                self._model_leaf_beliefs_buf.copy_(
                    self._leaf_beliefs_for_model(self.beliefs)
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
            self._model_leaf_slot = slot.contiguous()
            self._model_leaf_slot_key = key
            self._model_leaf_beliefs_valid = False

        return (
            self._model_leaf_beliefs_buf,
            self._model_leaf_slot,
            self._model_leaf_depth_has_slot,
        )

    def _model_leaf_beliefs_for_values(self, beliefs: torch.Tensor) -> torch.Tensor:
        if (
            beliefs is self.beliefs
            and self._model_leaf_scatter_enabled
            and self._model_leaf_beliefs_valid
            and self._model_leaf_beliefs_buf is not None
            and self._model_leaf_beliefs_buf.shape[0] == int(self.model_indices.numel())
        ):
            return self._model_leaf_beliefs_buf
        return self._leaf_beliefs_for_model(beliefs)

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
        # (policy[c, h] if p == prev_actor[c] else 1.0). The evaluator does not
        # cross street boundaries; leaf nodes may represent pre-chance states,
        # but still use root/pre-chance board blockers. Apply the allowed-hand
        # mask at depth 0 only and let invalid-hand zeros propagate deeper.
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
                apply_allowed_mask=depth == 0,
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
        self._model_leaf_beliefs_valid = False
        bottom = self._bottom
        child_offsets_top = self._child_offsets_top
        child_count_top = self._child_count_top
        parent_index_bottom = self._parent_index_bottom
        assert child_offsets_top is not None
        assert child_count_top is not None
        assert parent_index_bottom is not None
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
            uniform_count_fallback=True,
        )
        self._mask_invalid(self.policy_probs)

        defer_avg_reach = not self.cfr_avg and self.use_final_policy_values
        defer_avg_policy = not self.cfr_avg and self.use_final_policy_values
        if defer_avg_reach and defer_avg_policy:
            skip_avg_update = (
                self._uses_dcfr_backbone() and self._average_accumulation_delayed(t)
            )
            write_average_policy = not skip_avg_update
            avg_num, avg_den = self._ensure_average_policy_buffers()
            root_index = self._get_root_index()
            parent_index_all = self._parent_index_all
            assert parent_index_all is not None
            to_act = self.env.to_act.contiguous()
            prev_actor = self.prev_actor.contiguous()
            leaf_out = None
            leaf_slot = None
            leaf_depth_has_slot: tuple[bool, ...] = ()
            if (
                self._model_leaf_scatter_enabled
                and not self.cfr_avg
                and int(self.model_indices.numel()) > 0
            ):
                leaf_out, leaf_slot, leaf_depth_has_slot = (
                    self._ensure_model_leaf_belief_buffers()
                )
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
                    scatter_depth = (
                        depth < len(leaf_depth_has_slot) and leaf_depth_has_slot[depth]
                    )
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
                        apply_allowed_mask=depth == 0,
                        leaf_slot=leaf_slot if scatter_depth else None,
                        leaf_out=leaf_out if scatter_depth else None,
                    )
                    parent_reach, child_reach = child_reach, parent_reach
            else:
                for depth in range(self.tree_depth):
                    scatter_depth = (
                        depth < len(leaf_depth_has_slot) and leaf_depth_has_slot[depth]
                    )
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
                        apply_allowed_mask=depth == 0,
                        # Final-depth reach has no descendants and stats only read
                        # non-leaf reach. When stats are stubbed, keep leaf reach
                        # register-local and avoid two full leaf-row stores.
                        store_reach=(
                            depth < self.tree_depth - 1 or not self._skip_record_stats
                        ),
                        leaf_slot=leaf_slot if scatter_depth else None,
                        leaf_out=leaf_out if scatter_depth else None,
                    )
            self._model_leaf_beliefs_valid = leaf_out is not None
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
        select_actor_beliefs_and_marginal_policy_triton_out_(
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
            marginal_policy = (
                policy_src_all[offset:offset_next]
                * actor_beliefs[offset:offset_next, None, :]
            ).contiguous()
            opponent_conditioned_policy = self._conditioned_action_ratio(
                marginal_policy,
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
                opponent_policy_base=offset,
            )

        return values_br

    # ------------------------------------------------------------------
    # Update average policy: fused mixing + parent-aligned renorm.
    # ------------------------------------------------------------------

    def update_average_policy(
        self,
        t: int,
        update_reach: bool = False,
        weight_override: float | None = None,
    ) -> None:
        self._update_average_policy_true(
            t, update_reach=update_reach, weight_override=weight_override
        )

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
        new = self._t_scalars.mix_new
        if weight_override is not None:
            new = torch.tensor(
                float(weight_override), dtype=self.float_dtype, device=self.device
            )
        fused_average_policy_mix_with_tensors_(
            policy_probs_avg=self.policy_probs_avg,
            average_policy_numerator=num,
            average_policy_denominator=den,
            policy_probs=self.policy_probs,
            self_reach=self.self_reach,
            to_act=self.env.to_act.contiguous(),
            parent_index=parent_index_all,
            new=new,
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
        old, new = self._get_mixing_weights(t)
        if old + new == 0:
            return
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

    def _eval_model_for_fused_writeback(
        self,
        value_model,
        features: MLPFeatures,
        *,
        use_pre_head: bool,
    ) -> tuple[torch.Tensor, bool]:
        from p2.models.mlp.better_trm import BetterTRM

        model_applied_zero_sum = False
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model = value_model
            base_model = getattr(model, "_orig_mod", model)
            if type(base_model) is BetterTRM:
                model_output = model(features, include_policy=False, latent=self.latent)
                self.latent = model_output.latent
                model_applied_zero_sum = bool(base_model.enforce_zero_sum)
            elif hasattr(base_model, "static_feature_base") and hasattr(
                base_model, "forward_value_static_base"
            ):
                model_key = id(base_model)
                if (
                    self._static_model_base_fn is None
                    or self._static_model_base_fn_key != model_key
                ):
                    if (
                        getattr(model, "_orig_mod", None) is not None
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
                    self._static_model_base_accepts_value_head = (
                        "value_head"
                        in inspect.signature(
                            base_model.forward_value_static_base
                        ).parameters
                    )
                key = (
                    int(self._subgame_generation),
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
                value_kwargs = {"apply_zero_sum": True}
                if getattr(self, "_static_model_base_accepts_value_head", False):
                    value_kwargs["value_head"] = "pre" if use_pre_head else "auto"
                call_static_value = getattr(
                    base_model,
                    "_call_forward_value_static_base",
                    base_model.forward_value_static_base,
                )
                model_output = call_static_value(
                    features,
                    self._static_model_base_features,
                    **value_kwargs,
                )
                model_applied_zero_sum = bool(base_model.enforce_zero_sum)
            else:
                if use_pre_head and hasattr(model, "forward_pre"):
                    hand_values = model.forward_pre(features).contiguous()
                    model_applied_zero_sum = bool(
                        getattr(base_model, "enforce_zero_sum", False)
                    )
                    model_output = None
                else:
                    model_output = model(features, include_policy=False)
        if model_output is not None:
            hand_values = model_output.hand_values.contiguous()
        return hand_values, model_applied_zero_sum

    def _model_leaf_values_for_fused_writeback(
        self, features: MLPFeatures
    ) -> tuple[torch.Tensor, bool]:
        value_model = self.value_model
        closing_value_model = self.closing_leaf_value_model
        scope = self._model_scope()
        if scope == "mixed_street" and closing_value_model is not None:
            self._ensure_model_index_partitions()
            hand_values = self.latest_values.new_empty(
                (
                    len(features),
                    self.num_players,
                    self.hand_dim,
                )
            )
            model_applied_zero_sum = True
            evaluated_any = False
            if self.cutoff_model_positions.numel() > 0:
                cutoff_values, cutoff_zero_sum = self._eval_model_for_fused_writeback(
                    value_model,
                    self._features_for_model_positions(
                        features, self.cutoff_model_positions
                    ),
                    use_pre_head=False,
                )
                hand_values.index_copy_(
                    0,
                    self.cutoff_model_positions,
                    cutoff_values.to(dtype=hand_values.dtype),
                )
                model_applied_zero_sum = model_applied_zero_sum and cutoff_zero_sum
                evaluated_any = True
            if self.new_street_model_positions.numel() > 0:
                closing_encoder = self.closing_leaf_value_encoder
                if self._can_project_heads_up_closing_model():
                    node_indices = self.model_indices[self.new_street_model_positions]
                    live_counts = self._live_counts_for_nodes(node_indices)
                    baseline_local = torch.where(live_counts < 2)[0]
                    if baseline_local.numel() > 0:
                        baseline_positions = self.new_street_model_positions[
                            baseline_local
                        ]
                        baseline_values = self._stack_value_baseline(
                            node_indices[baseline_local],
                            self.hand_dim,
                        )
                        hand_values.index_copy_(
                            0,
                            baseline_positions,
                            baseline_values.to(dtype=hand_values.dtype),
                        )
                        evaluated_any = True
                    hu_local = torch.where(live_counts >= 2)[0]
                    if hu_local.numel() == 0:
                        return hand_values, bool(
                            evaluated_any and model_applied_zero_sum
                        )
                    hu_positions = self.new_street_model_positions[hu_local]
                    closing_features, live_players = (
                        self._heads_up_projected_closing_features(
                            features,
                            hu_positions,
                            closing_encoder,
                        )
                    )
                    closing_values, closing_zero_sum = (
                        self._eval_model_for_fused_writeback(
                            closing_value_model,
                            closing_features,
                            use_pre_head=False,
                        )
                    )
                    closing_values = self._scatter_heads_up_closing_values(
                        closing_values,
                        live_players,
                        target_hand_dim=self.hand_dim,
                        node_indices=self.model_indices[hu_positions],
                    )
                    hand_values.index_copy_(
                        0,
                        hu_positions,
                        closing_values.to(dtype=hand_values.dtype),
                    )
                    model_applied_zero_sum = model_applied_zero_sum and closing_zero_sum
                    evaluated_any = True
                    return hand_values, bool(evaluated_any and model_applied_zero_sum)
                closing_values, closing_zero_sum = self._eval_model_for_fused_writeback(
                    closing_value_model,
                    self._features_for_model_positions(
                        features,
                        self.new_street_model_positions,
                        closing_encoder,
                    ),
                    use_pre_head=False,
                )
                hand_values.index_copy_(
                    0,
                    self.new_street_model_positions,
                    closing_values.to(dtype=hand_values.dtype),
                )
                model_applied_zero_sum = model_applied_zero_sum and closing_zero_sum
                evaluated_any = True
            return hand_values, bool(evaluated_any and model_applied_zero_sum)

        if scope in ("mixed_street", "single_street"):
            return self._eval_model_for_fused_writeback(
                value_model,
                features,
                use_pre_head=False,
            )

        if scope != "end_of_street":
            raise ValueError(f"Unknown search.model_scope: {scope!r}")
        if closing_value_model is not None:
            positions = torch.arange(
                len(features), dtype=torch.long, device=features.context.device
            )
            closing_encoder = self.closing_leaf_value_encoder
            if self._can_project_heads_up_closing_model():
                node_indices = self.model_indices[positions]
                live_counts = self._live_counts_for_nodes(node_indices)
                projected_values = self.latest_values.new_empty(
                    (
                        len(features),
                        self.num_players,
                        self.hand_dim,
                    )
                )
                baseline_local = torch.where(live_counts < 2)[0]
                if baseline_local.numel() > 0:
                    projected_values.index_copy_(
                        0,
                        baseline_local,
                        self._stack_value_baseline(
                            node_indices[baseline_local],
                            self.hand_dim,
                        ),
                    )
                hu_local = torch.where(live_counts >= 2)[0]
                if hu_local.numel() == 0:
                    return projected_values, False
                hu_positions = positions[hu_local]
                closing_features, live_players = (
                    self._heads_up_projected_closing_features(
                        features,
                        hu_positions,
                        closing_encoder,
                    )
                )
                closing_values, model_zero_sum = self._eval_model_for_fused_writeback(
                    closing_value_model,
                    closing_features,
                    use_pre_head=False,
                )
                closing_values = self._scatter_heads_up_closing_values(
                    closing_values,
                    live_players,
                    target_hand_dim=self.hand_dim,
                    node_indices=self.model_indices[hu_positions],
                )
                projected_values.index_copy_(0, hu_local, closing_values)
                return projected_values, model_zero_sum
            features = self._features_for_model_positions(
                features,
                positions,
                closing_encoder,
            )
        return self._eval_model_for_fused_writeback(
            closing_value_model or value_model,
            features,
            use_pre_head=False,
        )

    def _set_model_values_impl(self, t, beliefs, features):
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

        runner = getattr(self, "_showdown_graph_runner", None)

        if self.model_indices.numel() > 0:
            beliefs_at_model = self._model_leaf_beliefs_for_values(beliefs)
            features_at_model = self._model_features_for_beliefs(beliefs_at_model)
            self._set_model_values(t, beliefs_at_model, features_at_model)
            if runner is not None:
                runner.write_indexed(beliefs, self.showdown_indices, self.latest_values)
                self._set_allin_call_values(beliefs)
                return
            showdown_beliefs = beliefs[self.showdown_indices]
        else:
            empty_shape = (0, self.num_players, NUM_HANDS)
            if self._last_model_values_buf is None or (
                self._last_model_values_buf.shape != empty_shape
            ):
                self._last_model_values_buf = self.latest_values.new_empty(empty_shape)
            self.last_model_values = self._last_model_values_buf
            if runner is not None:
                runner.write_indexed(beliefs, self.showdown_indices, self.latest_values)
                self._set_allin_call_values(beliefs)
                return
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
            table, scale = resolver.payoff_for_board(
                boards_by_street[0].new_empty(0), 0
            )
            preflop_stats = getattr(self, "allin_preflop_stats_buffer", None)
            if (
                preflop_stats is None
                or preflop_stats.shape != (node_idx.numel(), 2, 53)
                or preflop_stats.dtype != torch.float32
                or preflop_stats.device != self.device
            ):
                preflop_stats = torch.empty(
                    node_idx.numel(),
                    2,
                    53,
                    dtype=torch.float32,
                    device=self.device,
                )
                self.allin_preflop_stats_buffer = preflop_stats
            write_allin_table_values_card_denom_dot_triton_(
                table=table,
                beliefs=beliefs,
                node_indices=node_idx,
                latest_values=self.latest_values,
                stacks=self.env.stacks,
                pot=self.env.pot,
                starting_stacks=self.env.starting_stacks,
                env_scale=self.env.scale,
                table_scale=scale,
                stats_buffer=preflop_stats,
                block_h=64,
                block_k=128,
                block_p=8,
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
            if (
                flop_tables is None
                or flop_ids is None
                or flop_stats is None
                or flop_tables.numel() == 0
            ):
                raise RuntimeError(
                    "Fused all-in flop evaluation requires cached flop tables."
                )
        if turn_node_idx.numel() > 0:
            if (
                turn_tables is None
                or turn_ids is None
                or turn_stats is None
                or turn_tables.numel() == 0
            ):
                raise RuntimeError(
                    "Fused all-in turn evaluation requires cached turn tables."
                )
        if flop_node_idx.numel() > 0 or turn_node_idx.numel() > 0:
            if flop_stats is None or turn_stats is None:
                raise RuntimeError(
                    "Fused all-in evaluation requires cached stats buffers."
                )
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
            if (
                flop_tables is None
                or flop_ids is None
                or flop_stats is None
                or flop_tables.numel() == 0
            ):
                raise RuntimeError(
                    "Fused all-in flop evaluation requires cached flop tables."
                )
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
            turn_group_positions = getattr(self, "allin_turn_group_positions", None)
            if (
                turn_tables is None
                or turn_ids is None
                or turn_stats is None
                or turn_group_positions is None
                or turn_tables.numel() == 0
            ):
                raise RuntimeError(
                    "Fused all-in turn evaluation requires cached turn tables."
                )
            if turn_group_positions.numel() > 0:
                write_allin_grouped_table_values_card_denom_dot_values_triton_(
                    table=turn_tables,
                    beliefs=beliefs,
                    node_indices=node_idx,
                    group_positions=turn_group_positions,
                    latest_values=self.latest_values,
                    stacks=self.env.stacks,
                    pot=self.env.pot,
                    starting_stacks=self.env.starting_stacks,
                    env_scale=self.env.scale,
                    table_scale=I16_SCALE,
                    stats_buffer=turn_stats,
                    block_h=64,
                    block_k=64,
                    block_nodes=32,
                    num_warps=8,
                )
            else:
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
                    block_k=64,
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
                predictive_scale=self._predictive_policy_scale_for_t(t),
                current_player=t % self.num_players,
            )

        self._sample_compact_leaves_for_current_t()

        if self.cfr_type == CFRType.linear or (
            self.cfr_type in (CFRType.pcfr, CFRType.sapcfr)
            and not self._predictive_cfr_uses_dcfr()
        ):
            # Linear CFR not supported by the fused kernel; use parent path.
            regrets = self.compute_instantaneous_regrets(self.latest_values)
            regrets.masked_fill_(self.prev_actor[:, None] == t % self.num_players, 0.0)
            self.cumulative_regrets += regrets
        else:
            apply_dcfr = self._uses_dcfr_backbone()
            positive_regrets_out = self._ensure_positive_regrets_buf()
            last_regrets = (
                self._ensure_last_instantaneous_regrets_buf()
                if getattr(self, "_predictive_cfr_enabled", False)
                else None
            )
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
                last_instantaneous_regrets=last_regrets,
                prediction_scale=self._t_scalars.predictive_scale,
                current_player=self._t_scalars.current_player,
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

    def sample_leaves(self, training_mode: bool) -> PublicBeliefState:
        if (
            not self._sample_leaf_enabled
            or self._sample_leaf_indices_padded is None
            or self._sample_leaf_beliefs_padded is None
            or self._sample_leaf_ready_padded is None
            or self._sample_leaf_training_mode != training_mode
        ):
            pbs = super().sample_leaves(training_mode)
            self._deal_sampled_chance_roots(pbs.env)
            self._normalize_sampled_public_state(pbs.env)
            return pbs

        N = self.root_nodes
        sampled_nodes = self._sample_leaf_indices_padded[:N]
        continue_mask = (
            self._sample_leaf_ready_padded[:N]
            & (sampled_nodes >= N)
            & (~self.env.done[sampled_nodes])
            & (~self.allin_call_mask[sampled_nodes])
        )
        root_indices = torch.where(continue_mask)[0]
        sampled_continue = sampled_nodes[root_indices]
        pbs = PublicBeliefState.from_proto(
            env_proto=self.env,
            beliefs=self._sample_leaf_beliefs_padded[root_indices],
            num_envs=sampled_continue.numel(),
        )
        pbs.env.copy_state_from(
            self.env,
            sampled_continue,
            torch.arange(sampled_continue.numel(), device=self.device),
        )
        self._deal_sampled_chance_roots(pbs.env)
        self._normalize_sampled_public_state(pbs.env)
        return pbs

    def _deal_sampled_chance_roots(self, env: HUNLTensorEnv) -> None:
        if env.N == 0:
            return
        start_of_street = env.actions_this_round == 0

        flop = start_of_street & (env.street == 1) & (env.board_indices[:, 0] < 0)
        pos = env.deck_pos[flop]
        c0 = env.deck[flop, pos]
        c1 = env.deck[flop, pos + 1]
        c2 = env.deck[flop, pos + 2]
        env.deck_pos[flop] = pos + 3
        env.board_indices[flop, 0] = c0
        env.board_indices[flop, 1] = c1
        env.board_indices[flop, 2] = c2

        turn = start_of_street & (env.street == 2) & (env.board_indices[:, 3] < 0)
        pos = env.deck_pos[turn]
        c = env.deck[turn, pos]
        env.deck_pos[turn] = pos + 1
        env.board_indices[turn, 3] = c

        river = start_of_street & (env.street == 3) & (env.board_indices[:, 4] < 0)
        pos = env.deck_pos[river]
        c = env.deck[river, pos]
        env.deck_pos[river] = pos + 1
        env.board_indices[river, 4] = c
        self._refresh_card_onehots_from_indices(env)

    def _refresh_card_onehots_from_indices(self, env: HUNLTensorEnv) -> None:
        env.board_onehot.zero_()
        env.hole_onehot.zero_()
        for slot in range(5):
            cards = env.board_indices[:, slot]
            valid = cards >= 0
            env.board_onehot[valid, slot] = env.card_onehot_cache[cards[valid]]
        for player in range(2):
            for slot in range(2):
                cards = env.hole_indices[:, player, slot]
                valid = cards >= 0
                env.hole_onehot[valid, player, slot] = env.card_onehot_cache[
                    cards[valid]
                ]

    def _normalize_sampled_public_state(self, env: HUNLTensorEnv) -> None:
        """Repair fields intentionally skipped by fused construction.

        Sampled leaves exclude folded terminal states, so every public state
        exported for the next resolver has no folded player. Keeping this fixup
        here avoids two bool stores per constructed child in the hot writer.
        """
        if env.N == 0:
            return
        env.has_folded.zero_()

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
            predictive_scale=self._predictive_policy_scale_for_t(t),
            current_player=t % self.num_players,
        )

    def _graph_capture_regime(self, t: int) -> str | None:
        """Return the Python-branch regime that is safe to CUDA-graph replay."""
        if t < 2:
            return None
        if self._uses_dcfr_backbone() and t <= self.dcfr_delay:
            return "pre_dcfr_delay"
        return "post_dcfr_delay"

    @torch.no_grad()
    def evaluate_cfr(
        self, training_mode: bool = True, sample_continuation: bool = True
    ):
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
        if sample_continuation:
            self._prepare_compact_leaf_sampling(training_mode)
        else:
            self._sample_leaf_enabled = False

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
            self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
            self._refresh_average_beliefs()

        if self.use_final_policy_values:
            self.update_average_values_final()

        self._record_action_mix()
        self._record_cfr_entropy()
        self._record_cumulative_regret()

        if not sample_continuation:
            return None
        return self.sample_leaves(training_mode)
