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
* ``compute_expected_values`` — per-depth weight + parent-sum reduce via
  ``fused_weighted_parent_sum``.
* ``compute_instantaneous_regrets`` — fan-out + gather + sub + mul into
  ``fused_regret_tail_``.
* ``update_average_policy`` mixing — ``fused_average_policy_mix_with_tensors_``.
* ``update_average_values`` mixing — ``fused_update_average_values_with_tensors_``.
* ``_set_model_values_impl`` mixing — ``fused_model_values_mix_with_tensors``.
"""

from __future__ import annotations

import torch

from p2.core.structured_config import CFRType
from p2.env.card_utils import NUM_HANDS
from p2.env.rules_triton import rank_hands_triton, triton_is_available as _rules_triton_ok
from p2.search.fused_cfr_triton import (
    fused_average_policy_mix_with_tensors_,
    fused_avg_values_zero_sum_,
    fused_block_and_normalize_beliefs_,
    fused_dcfr_update_with_tensors_,
    fused_deep_beliefs_,
    fused_divide_by_parent_sum_,
    fused_model_values_mix_zero_sum,
    fused_parent_sum,
    fused_reach_weights_depth_,
    fused_regret_tail_,
    fused_weighted_parent_sum,
    triton_is_available,
    TScalars,
    unblocked_mass_opp_at_parents_triton,
    unblocked_mass_ratio_indirect_triton,
)
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


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
        N = self.root_nodes
        if target is None:
            target = self.beliefs
        if reach_weights is None:
            reach_weights = self.self_reach

        # Single kernel: gather root beliefs via root_index, multiply by reach,
        # mask with allowed_hands, row-normalize (fallback to allowed_prob).
        # Clone: the kernel writes back to target[:N] too (mask + normalize),
        # but non-root programs read root_beliefs[root_index[i]] — must be a
        # separate buffer from `out` to avoid a cross-program read/write race.
        root_beliefs = target[:N].clone()
        fused_deep_beliefs_(
            out=target,
            root_beliefs=root_beliefs,
            reach_weights=reach_weights,
            allowed_mask=self.allowed_hands,
            allowed_prob=self.allowed_hands_prob,
            root_index=self._get_root_index(),
        )

    # ------------------------------------------------------------------
    # Regret matching: parent-aligned sum + in-kernel divide.
    # ------------------------------------------------------------------

    def update_policy(self, t: int) -> None:
        bottom = self.depth_offsets[1]
        top = self.depth_offsets[-2]
        if (
            self._fused_positive_regrets_buf is None
            or self._fused_positive_regrets_buf.shape != self.cumulative_regrets.shape
        ):
            self._fused_positive_regrets_buf = torch.empty_like(self.cumulative_regrets)
        positive_regrets = self._fused_positive_regrets_buf
        torch.clamp(self.cumulative_regrets, min=0.0, out=positive_regrets)

        # Parent-aligned sum (no child broadcast), then a divide kernel that
        # gathers from parent_sum via parent_index on the fly. Skips
        # materializing the [num_children, H] denom intermediate.
        parent_sum = fused_parent_sum(
            values=positive_regrets.contiguous(),
            child_offsets=self.child_offsets[:top].contiguous(),
            child_count=self.child_count[:top].contiguous(),
            max_children=self.num_actions,
        )
        uniform_fallback = self.uniform_policy[bottom:].contiguous()
        fused_divide_by_parent_sum_(
            pos=positive_regrets[bottom:].contiguous(),
            fallback=uniform_fallback,
            parent_sum=parent_sum,
            parent_index=self.parent_index[bottom:].contiguous(),
            out=self.policy_probs[bottom:],
        )
        self._mask_invalid(self.policy_probs)

        self._calculate_reach_weights(self.self_reach, self.policy_probs)
        self._propagate_all_beliefs(self.beliefs, self.self_reach)

        self.update_average_policy(t)
        self._calculate_reach_weights(self.self_reach_avg, self.policy_probs_avg)
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

        if leaf_values is values:
            # Skip the (~leaf_mask) zero: every non-leaf row is overwritten by
            # the parent_sum sweep below (count == 0 iff leaf, so non-leaf
            # parents always have children to reduce). Leaf rows are preserved
            # by parent_sum's `if count == 0: return` early-out.
            pass
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
        # Skip materializing beliefs_dest as a separate tensor — fan-out is done
        # inline via index_select, and the denom side of the ratio kernel
        # gathers from actor_beliefs via parent_index instead.
        marginal_policy = (
            actor_beliefs.index_select(0, self.parent_index[bottom:]) * policy[bottom:]
        )

        opponent_conditioned_policy = torch.zeros_like(policy)
        opponent_conditioned_policy[bottom:] = unblocked_mass_ratio_indirect_triton(
            numer_target=marginal_policy.contiguous(),
            denom_target=actor_beliefs.contiguous(),
            parent_index=self.parent_index[bottom:].contiguous(),
        )

        for depth in range(self.tree_depth - 1, -1, -1):
            parent_base = self.depth_offsets[depth]
            parent_end = self.depth_offsets[depth + 1]
            # Fused weight + parent-sum: replaces the per-child clone +
            # scatter_reduce pair with one parent-aligned reduce.
            fused_weighted_parent_sum(
                values=values,
                prev_actor=self.prev_actor,
                policy_hero=policy,
                policy_opp=opponent_conditioned_policy,
                child_offsets=self.child_offsets[parent_base:parent_end].contiguous(),
                child_count=self.child_count[parent_base:parent_end].contiguous(),
                parent_base=parent_base,
                max_children=self.num_actions,
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

        bottom = self.depth_offsets[1]
        top = self.depth_offsets[-2]
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        regrets = torch.zeros_like(self.policy_probs)

        # Compute opponent-reach unblocked mass only at parent rows [0, top)
        # rather than all [total, 2] rows — saves ~13× memory traffic.
        src_weights = unblocked_mass_opp_at_parents_triton(
            beliefs, self.env.to_act, top
        )  # [top, H]

        # actor_values is parent-aligned and gathered lazily per-child inside
        # fused_regret_tail_; only compute it at parent rows too.
        # actor_values[p, h] = values_expected[p, to_act[p], h]
        row_idx = torch.arange(top, device=self.device)
        actor_values = values_expected[:top][row_idx, self.env.to_act[:top], :]

        fused_regret_tail_(
            regrets=regrets,
            values_achieved=values_achieved.contiguous(),
            actor_values=actor_values.contiguous(),
            src_weights=src_weights.contiguous(),
            parent_index=self.parent_index.contiguous(),
            prev_actor=self.prev_actor.contiguous(),
            bottom=bottom,
        )

        self._mask_invalid(regrets)
        return regrets

    # ------------------------------------------------------------------
    # Update average policy: fused mixing + parent-aligned renorm.
    # ------------------------------------------------------------------

    def update_average_policy(self, t: int) -> None:
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

        fused_average_policy_mix_with_tensors_(
            policy_probs_avg=self.policy_probs_avg,
            policy_probs=self.policy_probs,
            self_reach=self.self_reach,
            self_reach_avg=self.self_reach_avg,
            to_act=self.env.to_act.contiguous(),
            parent_index=self.parent_index.contiguous(),
            old=self._t_scalars.mix_old,
            new=self._t_scalars.mix_new,
            total_weight=self._t_scalars.mix_total,
            bottom=N,
        )

        top = self.depth_offsets[-2]
        parent_sum = fused_parent_sum(
            values=self.policy_probs_avg.contiguous(),
            child_offsets=self.child_offsets[:top].contiguous(),
            child_count=self.child_count[:top].contiguous(),
            max_children=self.num_actions,
        )
        # For renorm, fallback is the un-normalized policy itself (i.e. identity).
        child_slice = self.policy_probs_avg[N:].contiguous()
        fused_divide_by_parent_sum_(
            pos=child_slice,
            fallback=child_slice,
            parent_sum=parent_sum,
            parent_index=self.parent_index[N:].contiguous(),
            out=self.policy_probs_avg[N:],
            eps=1e-5,
        )
        self.policy_probs_avg[:N] = 0.0

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

        if isinstance(self.model, BetterTRM):
            model_output = self.model(features, include_policy=False, latent=self.latent)
            self.latent = model_output.latent
        else:
            model_output = self.model(features, include_policy=False)

        if not self.cfr_avg or t <= 1 or self.last_model_values is None:
            self.latest_values.index_copy_(
                0, self.model_indices, model_output.hand_values
            )
        else:
            unmixed = torch.empty_like(model_output.hand_values)
            fused_model_values_mix_zero_sum(
                hand_values=model_output.hand_values.contiguous(),
                last_model_values=self.last_model_values.contiguous(),
                beliefs=beliefs.contiguous(),
                old_plus_new_over_new=self._t_scalars.mix_onon,
                old_over_new=self._t_scalars.mix_oon,
                out=unmixed,
                enforce_zero_sum=bool(self.model.enforce_zero_sum),
            )
            self.latest_values.index_copy_(0, self.model_indices, unmixed)
        return self.latest_values, model_output.hand_values

    @torch.no_grad()
    def set_leaf_values(self, t: int, beliefs: torch.Tensor | None = None) -> None:
        """Graph-safe override: ``_set_model_values_impl`` writes into
        ``self.latest_values`` in-place, so we skip the ``.copy_()`` round-trip
        the parent class needs. ``self.last_model_values`` is pinned to a
        persistent buffer for the same reason.
        """
        if beliefs is None:
            beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        features = self.feature_encoder.encode(
            beliefs, pre_chance_node=self.new_street_mask
        )
        _, last_model_values = self._set_model_values(
            t, beliefs[self.model_indices], features[self.model_indices]
        )
        if self._last_model_values_buf is None or (
            self._last_model_values_buf.shape != last_model_values.shape
        ):
            self._last_model_values_buf = torch.empty_like(last_model_values)
        self._last_model_values_buf.copy_(last_model_values)
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

        torch.where(
            (self.t_sample == self._t_scalars.t_tensor)[:, None],
            self.policy_probs,
            self.policy_probs_sample,
            out=self.policy_probs_sample,
        )

        regrets = self.compute_instantaneous_regrets(self.latest_values)

        if self.cfr_type == CFRType.linear:
            # Linear CFR not supported by the fused kernel; use parent path.
            regrets.masked_fill_(self.prev_actor[:, None] == t % self.num_players, 0.0)
            self.regret_weight_sums += 1
            self.cumulative_regrets += regrets
        else:
            apply_dcfr = self.cfr_type in (CFRType.discounted, CFRType.discounted_plus)
            fused_dcfr_update_with_tensors_(
                cumulative_regrets=self.cumulative_regrets,
                regret_weight_sums=self.regret_weight_sums,
                regrets=regrets,
                t_alpha_num=self._t_scalars.t_alpha_num,
                t_beta_num=self._t_scalars.t_beta_num,
                t_alpha_den=self._t_scalars.t_alpha_den,
                t_beta_den=self._t_scalars.t_beta_den,
                apply_dcfr=apply_dcfr,
                cfr_plus=self.cfr_plus,
            )

        if self._skip_record_stats:
            self.update_policy(t)
        else:
            old_policy_probs = self.policy_probs.clone()
            self.update_policy(t)
            self._record_stats(t, old_policy_probs)

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
