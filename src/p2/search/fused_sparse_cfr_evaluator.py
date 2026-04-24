"""FusedSparseCFREvaluator — drop-in subclass using Triton-fused kernels.

Overrides just the methods where fusion applies; every other code path
inherits unchanged from ``SparseCFREvaluator``. Semantics must match the
parent class bit-close (float rounding may differ by ~1 ULP due to fused
multiply-add in Triton).

Fusion points
-------------
* ``cfr_iteration`` — DCFR rescale + accumulate + clamp block replaced by one
  Triton kernel (``fused_dcfr_update_``).
* ``_block_beliefs`` + ``_normalize_beliefs`` — fused into one kernel
  (``fused_block_and_normalize_beliefs_``). The parent class does these as
  separate calls; we override both and add a combined helper that callers use.
* ``update_policy`` — the ``where(denom > eps, pos/denom, uniform)`` tail is
  one kernel (``fused_regret_matching_divide_``).
* ``compute_expected_values`` — the per-depth ``.clone() + fancy-index mul × 2``
  block is one kernel (``fused_weight_child_values``).
"""

from __future__ import annotations

import torch

from p2.core.structured_config import CFRType
from p2.env.card_utils import NUM_HANDS
from p2.search.fused_cfr_triton import (
    fused_average_policy_mix_,
    fused_block_and_normalize_beliefs_,
    fused_dcfr_update_,
    fused_divide_by_parent_sum_,
    fused_model_values_mix,
    fused_parent_sum,
    fused_regret_matching_divide_,
    fused_regret_tail_,
    fused_sibling_sum,
    fused_update_average_values_,
    fused_weight_child_values,
    triton_is_available,
    unblocked_mass_ratio_triton,
    unblocked_mass_triton,
)

# Local alias so method bodies below read the same as parent, but route to Triton.
calculate_unblocked_mass = unblocked_mass_triton
from p2.search.sparse_cfr_evaluator import SparseCFREvaluator


class FusedSparseCFREvaluator(SparseCFREvaluator):
    """SparseCFREvaluator with Triton-fused pointwise/reduction kernels.

    Requires CUDA + Triton. Falls back to parent implementation for anything
    not listed in the module docstring.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.device.type != "cuda":
            raise ValueError("FusedSparseCFREvaluator requires a CUDA device.")
        if not triton_is_available():
            raise RuntimeError(
                "Triton is not installed; FusedSparseCFREvaluator is unavailable."
            )
        # Cache a buffer reused across update_policy calls to avoid reallocating
        # the fan-out denom every iteration.
        self._fused_positive_regrets_buf: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Beliefs: fused block + normalize.
    # ------------------------------------------------------------------

    def _block_beliefs(self, target: torch.Tensor | None = None) -> None:
        # Defer to the fused combined op via the helper below. Callers that
        # only need blocking (without normalize) still get correct behavior
        # because this path only runs inside _calculate_reach_weights /
        # _propagate_all_beliefs, both of which also normalize next.
        # For safety, if someone calls block without normalize, fall back to
        # parent behavior (pure masked_fill).
        super()._block_beliefs(target)

    def _normalize_beliefs(self, target: torch.Tensor | None = None) -> None:
        # Parent path: masked_fill (via _block_beliefs) is already done by the
        # caller (_calculate_reach_weights) before this; then this runs
        # sum + where. Replicate fused (block+normalize) by re-applying the
        # mask then normalizing — idempotent: masking zeros stays zero.
        if target is None:
            target = self.beliefs
        fused_block_and_normalize_beliefs_(
            target, self.allowed_hands, self.allowed_hands_prob
        )

    def _calculate_reach_weights(
        self, target: torch.Tensor, policy: torch.Tensor
    ) -> None:
        # Replicates parent exactly except the final block+normalize is fused.
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

        # Parent does: _mask_invalid (noop for sparse) + _block_beliefs (mask).
        # Then the caller typically calls _propagate_all_beliefs which does
        # fan_out_deep * reach_weights + normalize. Reach weights themselves
        # get blocked here but NOT normalized; the normalization happens inside
        # _propagate_all_beliefs on the resulting beliefs tensor, not on the
        # reach weights. So we keep the parent's block step and skip normalize.
        self._mask_invalid(target)
        super()._block_beliefs(target)

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

        target[:] = self._fan_out_deep(target[:N]) * reach_weights
        # Parent calls _normalize_beliefs; our override routes to fused kernel.
        fused_block_and_normalize_beliefs_(
            target, self.allowed_hands, self.allowed_hands_prob
        )

    # ------------------------------------------------------------------
    # Regret matching: fused divide tail.
    # ------------------------------------------------------------------

    def update_policy(self, t: int) -> None:
        bottom = self.depth_offsets[1]
        top = self.depth_offsets[-2]
        # Reuse buffer across iterations to avoid realloc.
        if (
            self._fused_positive_regrets_buf is None
            or self._fused_positive_regrets_buf.shape != self.cumulative_regrets.shape
        ):
            self._fused_positive_regrets_buf = torch.empty_like(self.cumulative_regrets)
        positive_regrets = self._fused_positive_regrets_buf
        torch.clamp(self.cumulative_regrets, min=0.0, out=positive_regrets)

        # Approach C: parent-aligned sum (no child broadcast), then a divide
        # kernel that gathers from parent_sum via parent_index on the fly.
        # Skips materializing the [num_children, H] denom intermediate.
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
    # Expected values: fused child-value weighting.
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

        # One Triton kernel produces where(unblocked(beliefs_dest) > eps,
        # unblocked(marginal_policy) / unblocked(beliefs_dest), 0).
        opponent_conditioned_policy = torch.zeros_like(policy)
        opponent_conditioned_policy[bottom:] = unblocked_mass_ratio_triton(
            marginal_policy.contiguous(), beliefs_dest.contiguous()
        )

        for depth in range(self.tree_depth - 1, -1, -1):
            offset_next = self.depth_offsets[depth + 1]
            offset_next_next = self.depth_offsets[depth + 2]

            prev_actor_indices = self.prev_actor[offset_next:offset_next_next]
            src_values = values[offset_next:offset_next_next].contiguous()
            policy_hero = policy[offset_next:offset_next_next].contiguous()
            policy_opp = opponent_conditioned_policy[
                offset_next:offset_next_next
            ].contiguous()

            weighted_child_values = torch.empty_like(src_values)
            fused_weight_child_values(
                values_src=src_values,
                prev_actor=prev_actor_indices.contiguous(),
                policy_hero=policy_hero,
                policy_opp=policy_opp,
                out=weighted_child_values,
            )

            self._pull_back_sum(weighted_child_values, values, level=depth)

    # ------------------------------------------------------------------
    # Instantaneous regrets: fused fan_out + gather + sub + mul tail.
    # ------------------------------------------------------------------

    def compute_instantaneous_regrets(
        self,
        values_achieved: torch.Tensor,
        values_expected: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if values_expected is None:
            values_expected = values_achieved

        bottom = self.depth_offsets[1]
        beliefs = self.beliefs_avg if self.cfr_avg else self.beliefs

        regrets = torch.zeros_like(self.policy_probs)

        src_actor_indices = self.env.to_act[:, None, None].expand(-1, -1, NUM_HANDS)

        opponent_global_reach = calculate_unblocked_mass(beliefs.flip(dims=[1]))
        src_weights = opponent_global_reach.gather(1, src_actor_indices).squeeze(1)
        weights = self._fan_out(src_weights)
        actor_values = values_expected.gather(1, src_actor_indices).squeeze(1)

        fused_regret_tail_(
            regrets=regrets,
            values_achieved=values_achieved.contiguous(),
            actor_values=actor_values.contiguous(),
            weights=weights.contiguous(),
            parent_index=self.parent_index.contiguous(),
            prev_actor=self.prev_actor.contiguous(),
            bottom=bottom,
        )

        self._mask_invalid(regrets)
        return regrets

    # ------------------------------------------------------------------
    # Update average policy: fused mixing step.
    # ------------------------------------------------------------------

    def update_average_policy(self, t: int) -> None:
        from p2.core.structured_config import CFRType

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

        fused_average_policy_mix_(
            policy_probs_avg=self.policy_probs_avg,
            policy_probs=self.policy_probs,
            self_reach=self.self_reach,
            self_reach_avg=self.self_reach_avg,
            to_act=self.env.to_act.contiguous(),
            parent_index=self.parent_index.contiguous(),
            old=float(old),
            new=float(new),
            bottom=N,
        )

        # Approach C: parent-aligned sum + in-kernel divide with parent-index gather.
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
        old, new = self._get_mixing_weights(t)
        fused_update_average_values_(
            self.values_avg, self.latest_values, float(old), float(new)
        )
        self.values_avg[:] = self._maybe_enforce_zero_sum(
            self.values_avg, self.beliefs_avg, ignore_mask=self.env.done
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
            new_values = torch.index_copy(
                self.latest_values, 0, self.model_indices, model_output.hand_values
            )
        else:
            old, new = self._get_mixing_weights(t)
            unmixed = fused_model_values_mix(
                model_output.hand_values.contiguous(),
                self.last_model_values.contiguous(),
                float(old),
                float(new),
            )
            unmixed = self._maybe_enforce_zero_sum(unmixed, beliefs)
            new_values = torch.index_copy(
                self.latest_values, 0, self.model_indices, unmixed
            )
        return new_values, model_output.hand_values

    # ------------------------------------------------------------------
    # CFR iteration: fused DCFR update.
    # ------------------------------------------------------------------

    def cfr_iteration(self, t: int) -> None:
        self.apply_schedules(t)

        torch.where(
            (self.t_sample == t)[:, None],
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
            fused_dcfr_update_(
                cumulative_regrets=self.cumulative_regrets,
                regret_weight_sums=self.regret_weight_sums,
                regrets=regrets,
                t=t,
                cfr_type=self.cfr_type,
                dcfr_alpha=self.dcfr_alpha,
                dcfr_beta=self.dcfr_beta,
                cfr_plus=self.cfr_plus,
            )

        old_policy_probs = self.policy_probs.clone()
        self.update_policy(t)
        self._record_stats(t, old_policy_probs)

        self.set_leaf_values(t)
        self.compute_expected_values()

        if not self.use_final_policy_values:
            self.update_average_values(t)
