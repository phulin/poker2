"""Triton + CUDA-graph fusions for a single CFR iteration.

Triton kernels (pointwise + simple reductions):

1. ``fused_dcfr_update_`` — DCFR rescale + weight_sum bump + cumulative-regret
   accumulate + optional CFR+ clamp. Replaces ~7 PyTorch kernels with 1.
   Fuses the block between ``compute_instantaneous_regrets`` and
   ``update_policy`` in ``cfr_iteration``.

2. ``fused_block_and_normalize_beliefs_`` — ``_block_beliefs`` +
   ``_normalize_beliefs`` fused: board-mask, row-sum over hands, divide /
   fallback to uniform. 4 kernels → 1.

3. ``fused_regret_matching_divide_`` — the ``where(denom > eps, pos/denom,
   uniform)`` tail of ``update_policy``. 3 kernels → 1.

4. ``fused_weight_child_values_`` — the ``.clone()`` + two fancy-index in-place
   multiplies inside the per-depth loop of ``compute_expected_values``.
   3 kernels → 1, called ``max_depth`` times per iteration.

Plus:

5. ``GraphedCFRIteration`` — captures one ``evaluator.cfr_iteration(t)`` call
   into a CUDA graph and exposes ``.replay()`` for benchmarking launch
   overhead.

The fused variant of ``SparseCFREvaluator`` lives in
``fused_sparse_cfr_evaluator.py`` as ``FusedSparseCFREvaluator`` — a subclass
that overrides only the methods affected by fusion.

Scope / caveats
---------------
* The CUDA graph bakes in any ``t``-derived Python scalars at capture time
  (DCFR exponents, mixing weights inside ``update_average_policy`` /
  ``update_average_values`` / ``_set_model_values_impl``, comparison against
  ``t_sample``). ``replay()`` therefore repeats iteration-T math on each call,
  which makes this a launch-overhead benchmark and a per-iteration correctness
  check, not a drop-in replacement for a full CFR run. A drop-in replacement
  would need those scalars lifted to 0-D device tensors throughout the
  evaluator so a single graph can service all iterations.
* ``_record_stats`` is skipped during capture — it contains ``.item()`` which
  forces a CUDA sync incompatible with graph capture.
* ``apply_schedules`` is also skipped; it only mutates Python floats.
* Targets the default config path: ``cfr_type in {discounted, discounted_plus}``,
  ``cfr_plus=False``, no active DCFR parameter schedule (``dcfr_*_final`` all
  ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dep
    triton = None
    tl = None

from p2.core.structured_config import CFRType


def triton_is_available() -> bool:
    return triton is not None


if triton is not None:

    @triton.jit
    def _fused_dcfr_update_kernel(
        regrets_ptr,
        cumul_ptr,
        weight_ptr,
        pos_out_ptr,
        t_alpha_num,
        t_beta_num,
        t_alpha_den,
        t_beta_den,
        N,
        APPLY_DCFR: tl.constexpr,
        CFR_PLUS: tl.constexpr,
        WRITE_POS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        r = tl.load(regrets_ptr + offs, mask=mask, other=0.0)
        c = tl.load(cumul_ptr + offs, mask=mask, other=0.0)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0)

        if APPLY_DCFR:
            positive = c > 0.0
            num = tl.where(positive, t_alpha_num, t_beta_num)
            den = tl.where(positive, t_alpha_den, t_beta_den)
            # Match PyTorch: `c *= num; c /= den` (two rounding steps).
            c = c * num
            c = c / den
            w = w * num
            w = w / den

        w = w + 1.0
        c = c + r

        if CFR_PLUS:
            c = tl.maximum(c, 0.0)

        tl.store(cumul_ptr + offs, c, mask=mask)
        tl.store(weight_ptr + offs, w, mask=mask)

        if WRITE_POS:
            pos = tl.maximum(c, 0.0)
            tl.store(pos_out_ptr + offs, pos, mask=mask)


def fused_dcfr_update_(
    cumulative_regrets: torch.Tensor,
    regret_weight_sums: torch.Tensor,
    regrets: torch.Tensor,
    t: int,
    cfr_type: CFRType,
    dcfr_alpha: float,
    dcfr_beta: float,
    cfr_plus: bool,
    positive_regrets_out: torch.Tensor | None = None,
    block_size: int = 1024,
) -> None:
    """In-place fused DCFR update.

    Replicates this sequence from ``cfr_evaluator.cfr_iteration``::

        if cfr_type in {discounted, discounted_plus}:
            num = where(c > 0, t**a, t**b)
            den = where(c > 0, t**a + 1, t**b + 1)
            c *= num; c /= den
            w *= num; w /= den
        w += 1
        c += regrets
        if cfr_plus:
            c.clamp_(min=0)

    with identical PyTorch ordering (so two-step rescale rounding matches).
    Writes ``clamp(c, 0)`` to ``positive_regrets_out`` if provided.

    Does *not* support ``CFRType.linear`` (which needs per-node ``prev_actor``
    masking — not in the default config path).
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if cumulative_regrets.device.type != "cuda":
        raise ValueError("fused_dcfr_update_ requires CUDA tensors.")
    if cfr_type == CFRType.linear:
        raise NotImplementedError(
            "Linear CFR path not supported; default config uses discounted_plus."
        )

    assert cumulative_regrets.is_contiguous()
    assert regret_weight_sums.is_contiguous()
    assert regrets.is_contiguous()
    assert cumulative_regrets.shape == regret_weight_sums.shape == regrets.shape

    apply_dcfr = cfr_type in (CFRType.discounted, CFRType.discounted_plus)
    if apply_dcfr:
        t_alpha_num = float(t**dcfr_alpha)
        t_beta_num = float(t**dcfr_beta)
        t_alpha_den = t_alpha_num + 1.0
        t_beta_den = t_beta_num + 1.0
    else:
        t_alpha_num = t_beta_num = t_alpha_den = t_beta_den = 1.0

    n = cumulative_regrets.numel()
    write_pos = positive_regrets_out is not None
    if write_pos:
        assert positive_regrets_out.is_contiguous()
        assert positive_regrets_out.shape == cumulative_regrets.shape
        pos_ptr = positive_regrets_out
    else:
        # Kernel requires a valid pointer even if unused; reuse cumul (not read).
        pos_ptr = cumulative_regrets

    grid = (triton.cdiv(n, block_size),)
    _fused_dcfr_update_kernel[grid](
        regrets,
        cumulative_regrets,
        regret_weight_sums,
        pos_ptr,
        t_alpha_num,
        t_beta_num,
        t_alpha_den,
        t_beta_den,
        n,
        APPLY_DCFR=apply_dcfr,
        CFR_PLUS=cfr_plus,
        WRITE_POS=write_pos,
        BLOCK=block_size,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 2: block + normalize beliefs (row-wise over hand axis).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_block_normalize_kernel(
        target_ptr,             # [R, H] flat view of target (R = N*P or N)
        allowed_mask_ptr,       # [R_outer, H] bool (broadcast along P)
        allowed_prob_ptr,       # [R_outer, H] fallback
        row_to_outer_stride,    # stride from row index to outer index (P for [N,P,H], 1 for [N,H])
        H,
        EPS,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        outer = row // row_to_outer_stride
        target_row_ptr = target_ptr + row * H
        mask_row_ptr = allowed_mask_ptr + outer * H
        prob_row_ptr = allowed_prob_ptr + outer * H

        # Pass 1: load, apply block mask, accumulate sum.
        offs = tl.arange(0, BLOCK_H)
        total = tl.zeros((), dtype=tl.float32)
        for start in tl.range(0, H, BLOCK_H):
            off = start + offs
            m = off < H
            t = tl.load(target_row_ptr + off, mask=m, other=0.0)
            allowed = tl.load(mask_row_ptr + off, mask=m, other=0).to(tl.int1)
            t = tl.where(allowed, t, 0.0)
            tl.store(target_row_ptr + off, t, mask=m)
            total += tl.sum(tl.where(m, t, 0.0))

        # Pass 2: divide or fallback.
        use_div = total > EPS
        for start in tl.range(0, H, BLOCK_H):
            off = start + offs
            m = off < H
            if use_div:
                t = tl.load(target_row_ptr + off, mask=m, other=0.0)
                t = t / total
            else:
                t = tl.load(prob_row_ptr + off, mask=m, other=0.0)
            tl.store(target_row_ptr + off, t, mask=m)


def fused_block_and_normalize_beliefs_(
    target: torch.Tensor,
    allowed_hands: torch.Tensor,
    allowed_hands_prob: torch.Tensor,
    eps: float = 1e-5,
) -> None:
    """In-place: block `target` by `allowed_hands`, then normalize rows over the
    last axis; fall back to `allowed_hands_prob` where the row sum is <= eps.

    Replicates ``_block_beliefs`` followed by ``_normalize_beliefs``.

    Shapes:
      target:             [N, P, H] or [N, H]
      allowed_hands:      [N, H] bool
      allowed_hands_prob: [N, H]
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert target.is_contiguous()
    assert allowed_hands.is_contiguous()
    assert allowed_hands_prob.is_contiguous()
    assert target.device.type == "cuda"

    if target.dim() == 3:
        n, p, h = target.shape
        total_rows = n * p
        stride = p
    elif target.dim() == 2:
        n, h = target.shape
        total_rows = n
        stride = 1
    else:
        raise ValueError(f"target must be 2D or 3D, got {target.shape}")

    assert allowed_hands.shape == (n, h)
    assert allowed_hands_prob.shape == (n, h)

    # BLOCK_H must be a power of two and cover hand-axis chunks.
    block_h = 512
    grid = (total_rows,)
    _fused_block_normalize_kernel[grid](
        target,
        allowed_hands,
        allowed_hands_prob,
        stride,
        h,
        eps,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 3: regret-matching divide tail of update_policy.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_regret_matching_divide_kernel(
        positive_regrets_ptr,  # [N, H]
        denom_ptr,             # [N, H]
        uniform_ptr,           # [N, H]
        out_ptr,               # [N, H]
        N_elements,
        EPS,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_elements

        pos = tl.load(positive_regrets_ptr + offs, mask=mask, other=0.0)
        den = tl.load(denom_ptr + offs, mask=mask, other=0.0)
        use_div = den > EPS
        den_safe = tl.maximum(den, EPS)
        divided = pos / den_safe
        fallback = tl.load(uniform_ptr + offs, mask=mask, other=0.0)
        result = tl.where(use_div, divided, fallback)
        tl.store(out_ptr + offs, result, mask=mask)


def fused_regret_matching_divide_(
    positive_regrets: torch.Tensor,
    denom: torch.Tensor,
    uniform_fallback: torch.Tensor,
    out: torch.Tensor,
    eps: float = 1e-8,
    block_size: int = 1024,
) -> None:
    """Compute `out = where(denom > eps, pos/max(denom,eps), uniform)` in one kernel.

    All tensors must be contiguous CUDA tensors with the same shape.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert positive_regrets.is_contiguous()
    assert denom.is_contiguous()
    assert uniform_fallback.is_contiguous()
    assert out.is_contiguous()
    assert positive_regrets.shape == denom.shape == uniform_fallback.shape == out.shape
    n = positive_regrets.numel()
    grid = (triton.cdiv(n, block_size),)
    _fused_regret_matching_divide_kernel[grid](
        positive_regrets,
        denom,
        uniform_fallback,
        out,
        n,
        eps,
        BLOCK=block_size,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 4: policy-weight child values inside compute_expected_values loop.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_weight_child_values_kernel(
        values_src_ptr,        # [M, 2, H]
        prev_actor_ptr,        # [M] int64
        policy_hero_ptr,       # [M, H] — policy at child
        policy_opp_ptr,        # [M, H] — opponent_conditioned_policy at child
        out_ptr,               # [M, 2, H]
        M, H,
        BLOCK_H: tl.constexpr,
    ):
        m_idx = tl.program_id(0)
        p_idx = tl.program_id(1)
        prev_actor = tl.load(prev_actor_ptr + m_idx)
        is_hero = (p_idx == prev_actor)

        row_offset = (m_idx * 2 + p_idx) * H
        pol_row_offset = m_idx * H

        for start in tl.range(0, H, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            mask = offs < H
            v = tl.load(values_src_ptr + row_offset + offs, mask=mask, other=0.0)
            if is_hero:
                p = tl.load(policy_hero_ptr + pol_row_offset + offs, mask=mask, other=0.0)
            else:
                p = tl.load(policy_opp_ptr + pol_row_offset + offs, mask=mask, other=0.0)
            tl.store(out_ptr + row_offset + offs, v * p, mask=mask)


def fused_weight_child_values(
    values_src: torch.Tensor,        # [M, 2, H]
    prev_actor: torch.Tensor,        # [M]
    policy_hero: torch.Tensor,       # [M, H]
    policy_opp: torch.Tensor,        # [M, H]
    out: torch.Tensor,               # [M, 2, H]
    block_h: int = 512,
) -> None:
    """Fused weighted copy used inside ``compute_expected_values``.

    For each (m, p, h):
      out[m, p, h] = values_src[m, p, h] * (
          policy_hero[m, h] if p == prev_actor[m] else policy_opp[m, h]
      )

    Replaces ``values[offset_next:offset_next_next].clone()`` + two fancy-index
    in-place multiplies with a single kernel.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values_src.is_contiguous() and values_src.dim() == 3
    assert out.is_contiguous() and out.shape == values_src.shape
    assert prev_actor.is_contiguous() and prev_actor.dim() == 1
    assert policy_hero.is_contiguous() and policy_opp.is_contiguous()
    m, p, h = values_src.shape
    assert p == 2, "Only supports 2 players."
    assert prev_actor.shape[0] == m
    assert policy_hero.shape == (m, h) and policy_opp.shape == (m, h)

    grid = (m, 2)
    _fused_weight_child_values_kernel[grid](
        values_src,
        prev_actor,
        policy_hero,
        policy_opp,
        out,
        m, h,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 5: update_average_values mixing.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_update_average_values_kernel(
        values_avg_ptr,     # [N, 2, H] in/out
        latest_ptr,         # [N, 2, H]
        old_scalar,
        new_scalar,
        inv_total,          # 1 / (old + new)
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(values_avg_ptr + offs, mask=mask, other=0.0)
        v = tl.load(latest_ptr + offs, mask=mask, other=0.0)
        out = (a * old_scalar + v * new_scalar) * inv_total
        tl.store(values_avg_ptr + offs, out, mask=mask)


def fused_update_average_values_(
    values_avg: torch.Tensor,
    latest_values: torch.Tensor,
    old: float,
    new: float,
    block_size: int = 1024,
) -> None:
    """In-place: values_avg = (values_avg * old + latest_values * new) / (old + new).

    Replaces the 3-kernel PyTorch sequence ``values_avg *= old; values_avg +=
    new * latest_values; values_avg /= (old + new)`` with one kernel.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values_avg.is_contiguous()
    assert latest_values.is_contiguous()
    assert values_avg.shape == latest_values.shape
    total = float(old) + float(new)
    assert total != 0.0
    inv_total = 1.0 / total
    n = values_avg.numel()
    grid = (triton.cdiv(n, block_size),)
    _fused_update_average_values_kernel[grid](
        values_avg,
        latest_values,
        float(old),
        float(new),
        inv_total,
        n,
        BLOCK=block_size,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 6: compute_instantaneous_regrets tail (fan_out + gather + sub + mul).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_regret_tail_kernel(
        values_achieved_ptr,     # [total, 2, H]
        actor_values_ptr,        # [top, H]     (pre-fan_out source)
        weights_ptr,             # [total-bottom, H]
        parent_index_ptr,        # [total] — parent_index[c] gives parent row in [0, top)
        prev_actor_ptr,          # [total] — 0 or 1
        regrets_ptr,             # [total, H] output (only rows [bottom, total) written)
        bottom,
        total,
        H,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0) + bottom  # child row
        if c >= total:
            return
        parent = tl.load(parent_index_ptr + c)
        prev_actor = tl.load(prev_actor_ptr + c)

        # Row base pointers.
        exp_row = actor_values_ptr + parent * H
        ach_row = values_achieved_ptr + (c * 2 + prev_actor) * H
        w_row = weights_ptr + (c - bottom) * H
        out_row = regrets_ptr + c * H

        for start in tl.range(0, H, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            mask = offs < H
            expected = tl.load(exp_row + offs, mask=mask, other=0.0)
            achieved = tl.load(ach_row + offs, mask=mask, other=0.0)
            w = tl.load(w_row + offs, mask=mask, other=0.0)
            tl.store(out_row + offs, w * (achieved - expected), mask=mask)


def fused_regret_tail_(
    regrets: torch.Tensor,              # [total, H] — in/out (only [bottom:] written)
    values_achieved: torch.Tensor,      # [total, 2, H]
    actor_values: torch.Tensor,         # [top, H]
    weights: torch.Tensor,              # [total-bottom, H]
    parent_index: torch.Tensor,         # [total] int64
    prev_actor: torch.Tensor,           # [total] int64
    bottom: int,
    block_h: int = 512,
) -> None:
    """Fused tail of ``compute_instantaneous_regrets``.

    For each child row ``c in [bottom, total)`` and hand ``h``::

        regrets[c, h] = weights[c - bottom, h] * (
            values_achieved[c, prev_actor[c], h]
            - actor_values[parent_index[c], h]
        )

    Replaces the PyTorch sequence ``fan_out(actor_values) + gather +
    subtract + multiply + assign`` (5 kernels + 2 intermediates) with one
    kernel and zero intermediate allocations.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert regrets.is_contiguous() and regrets.dim() == 2
    assert values_achieved.is_contiguous() and values_achieved.dim() == 3
    assert values_achieved.shape[1] == 2
    assert actor_values.is_contiguous() and actor_values.dim() == 2
    assert weights.is_contiguous() and weights.dim() == 2
    assert parent_index.is_contiguous() and parent_index.dim() == 1
    assert prev_actor.is_contiguous() and prev_actor.dim() == 1

    total, h = regrets.shape
    assert values_achieved.shape == (total, 2, h)
    assert weights.shape == (total - bottom, h)
    assert parent_index.shape == (total,)
    assert prev_actor.shape == (total,)

    grid = (total - bottom,)
    _fused_regret_tail_kernel[grid](
        values_achieved,
        actor_values,
        weights,
        parent_index,
        prev_actor,
        regrets,
        bottom,
        total,
        h,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 7: update_average_policy mixing (pre-renormalization).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_average_policy_mix_kernel(
        self_reach_ptr,         # [total, 2, H]
        self_reach_avg_ptr,     # [total, 2, H]
        policy_probs_ptr,       # [total, H]
        policy_probs_avg_ptr,   # [total, H] in/out
        to_act_ptr,             # [total] int64
        parent_index_ptr,       # [total] int64
        old_scalar,
        new_scalar,
        total_weight,           # old + new
        bottom,                 # first child row to update
        total,
        H,
        EPS,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0) + bottom
        if c >= total:
            return
        parent = tl.load(parent_index_ptr + c)
        actor = tl.load(to_act_ptr + parent)

        reach_row = self_reach_ptr + (parent * 2 + actor) * H
        reach_avg_row = self_reach_avg_ptr + (parent * 2 + actor) * H
        policy_row = policy_probs_ptr + c * H
        avg_row = policy_probs_avg_ptr + c * H

        for start in tl.range(0, H, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            mask = offs < H
            reach_a = tl.load(reach_avg_row + offs, mask=mask, other=0.0) * old_scalar
            reach_n = tl.load(reach_row + offs, mask=mask, other=0.0) * new_scalar
            avg = tl.load(avg_row + offs, mask=mask, other=0.0)
            cur = tl.load(policy_row + offs, mask=mask, other=0.0)

            num = reach_a * avg + reach_n * cur
            den = reach_a + reach_n
            unweighted = (old_scalar * avg + new_scalar * cur) / total_weight
            out = tl.where(den > EPS, num / den, unweighted)
            tl.store(avg_row + offs, out, mask=mask)


def fused_average_policy_mix_(
    policy_probs_avg: torch.Tensor,   # [total, H] in/out
    policy_probs: torch.Tensor,       # [total, H]
    self_reach: torch.Tensor,         # [total, 2, H]
    self_reach_avg: torch.Tensor,     # [total, 2, H]
    to_act: torch.Tensor,             # [total]
    parent_index: torch.Tensor,       # [total]
    old: float,
    new: float,
    bottom: int,
    eps: float = 1e-5,
    block_h: int = 512,
) -> None:
    """Fused mixing step of ``update_average_policy`` (pre-renormalization).

    For each child row ``c in [bottom, total)`` and hand ``h``, replicates the
    PyTorch sequence::

        reach_avg_actor = self_reach_avg[parent, to_act[parent], h] * old
        reach_actor     = self_reach[parent, to_act[parent], h]     * new
        num = reach_avg_actor * policy_probs_avg[c, h] + reach_actor * policy_probs[c, h]
        den = reach_avg_actor + reach_actor
        unweighted = (old * policy_probs_avg[c, h] + new * policy_probs[c, h]) / (old + new)
        policy_probs_avg[c, h] = where(den > eps, num / den, unweighted)

    Collapses ~8 kernels + 5 intermediate buffers into one kernel. Caller is
    responsible for running the subsequent per-parent renormalization
    (``_pull_back_sum`` + ``_fan_out`` + divide) and zeroing the root slice.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy_probs_avg.is_contiguous() and policy_probs_avg.dim() == 2
    assert policy_probs.is_contiguous() and policy_probs.dim() == 2
    assert self_reach.is_contiguous() and self_reach.dim() == 3
    assert self_reach_avg.is_contiguous() and self_reach_avg.dim() == 3
    assert self_reach.shape[1] == 2 and self_reach_avg.shape[1] == 2
    total, h = policy_probs_avg.shape
    assert policy_probs.shape == (total, h)
    assert self_reach.shape == (total, 2, h)
    assert self_reach_avg.shape == (total, 2, h)
    assert to_act.shape == (total,)
    assert parent_index.shape == (total,)
    total_weight = float(old) + float(new)
    assert total_weight != 0.0

    grid = (total - bottom,)
    _fused_average_policy_mix_kernel[grid](
        self_reach,
        self_reach_avg,
        policy_probs,
        policy_probs_avg,
        to_act,
        parent_index,
        float(old),
        float(new),
        total_weight,
        bottom,
        total,
        h,
        eps,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 8: unblocked mass in O(N) (replaces fp64 [1326, 1326] GEMM).
# ---------------------------------------------------------------------------


_UNBLOCKED_NUM_HANDS = 1326
_UNBLOCKED_NUM_CARDS = 52


if triton is not None:

    @triton.jit
    def _unblocked_mass_finalize_kernel(
        target_ptr,    # [B, H]
        cardsum_ptr,   # [B, NUM_CARDS]
        S_ptr,         # [B]
        card_a_ptr,    # [H] int32
        card_b_ptr,    # [H] int32
        out_ptr,       # [B, H]
        H,
        NUM_CARDS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        S = tl.load(S_ptr + b)

        t_row = target_ptr + b * H
        cs_row = cardsum_ptr + b * NUM_CARDS
        out_row = out_ptr + b * H

        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        t = tl.load(t_row + offs, mask=mask, other=0.0)
        ca = tl.load(card_a_ptr + offs, mask=mask, other=0)
        cb = tl.load(card_b_ptr + offs, mask=mask, other=0)

        csa = tl.load(cs_row + ca, mask=mask, other=0.0)
        csb = tl.load(cs_row + cb, mask=mask, other=0.0)

        out = S - csa - csb + t
        out = tl.maximum(out, 0.0)
        tl.store(out_row + offs, out, mask=mask)


# Per-device cache of (card_a, card_b) int32 tensors (the [H,2] combo→card LUT).
_combo_cards_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_combo_cards(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = device
    cached = _combo_cards_cache.get(key)
    if cached is not None:
        return cached
    from p2.env.card_utils import hand_combos_tensor

    combos = hand_combos_tensor(device=device)  # [1326, 2] long
    card_a = combos[:, 0].to(torch.int32).contiguous()
    card_b = combos[:, 1].to(torch.int32).contiguous()
    _combo_cards_cache[key] = (card_a, card_b)
    return card_a, card_b


def unblocked_mass_triton(target: torch.Tensor) -> torch.Tensor:
    """O(N) replacement for ``calculate_unblocked_mass``.

    Implements the inclusion-exclusion reformulation::

        unblocked[(a,b)] = S - cardsum[a] - cardsum[b] + target[(a,b)]

    where ``S = sum_h target[h]`` and ``cardsum[c] = sum_{h : combo h contains c}
    target[h]``. Matches the existing op's output (including the ``clamp(min=0)``
    tail and reshape) to within fp32 rounding.

    Accepts any shape ending in ``H = 1326``. Returns a tensor of the same
    shape.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if target.device.type != "cuda":
        raise ValueError("unblocked_mass_triton requires CUDA tensors.")

    orig_shape = target.shape
    assert orig_shape[-1] == _UNBLOCKED_NUM_HANDS, (
        f"last dim must be {_UNBLOCKED_NUM_HANDS}, got {orig_shape}"
    )
    flat = target.reshape(-1, _UNBLOCKED_NUM_HANDS).contiguous()
    b, h = flat.shape

    # S and cardsum via native PyTorch (tiny work). scatter_add_ is memory-bound
    # and fast; keeping it in PyTorch avoids replicating the reduction in Triton.
    # Accumulate in fp32 — matches downstream consumers; fp64 path removed as
    # the O(N) formula has no catastrophic cancellation risk for realistic reach.
    card_a, card_b = _get_combo_cards(flat.device)
    s = flat.sum(dim=-1)  # [B]
    cardsum = torch.zeros(
        b, _UNBLOCKED_NUM_CARDS, device=flat.device, dtype=flat.dtype
    )
    # scatter_add with int64 indices expected.
    card_a_long = card_a.to(torch.int64)[None, :].expand(b, -1)
    card_b_long = card_b.to(torch.int64)[None, :].expand(b, -1)
    cardsum.scatter_add_(1, card_a_long, flat)
    cardsum.scatter_add_(1, card_b_long, flat)

    out = torch.empty_like(flat)
    # BLOCK_H must cover H; next power of two above 1326 is 2048.
    _unblocked_mass_finalize_kernel[(b,)](
        flat,
        cardsum.contiguous(),
        s.contiguous(),
        card_a,
        card_b,
        out,
        h,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        BLOCK_H=2048,
        num_warps=4,
    )
    return out.view(orig_shape)


# ---------------------------------------------------------------------------
# Kernel 9: sibling sum (replaces _pull_back_sum + _fan_out).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _sibling_sum_kernel(
        values_ptr,         # [total, H] (children contiguous per parent)
        child_offsets_ptr,  # [num_parents] — first child absolute index
        child_count_ptr,    # [num_parents]
        out_ptr,            # [num_children, H] (out_row = first + i - out_offset)
        out_offset,         # absolute row index of first child (typically bottom)
        H,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """2D-tile variant: load [MAX_CHILDREN, BLOCK_H] in one coalesced
        transaction, reduce on axis 0, broadcast to a 2D store. Depends on
        sibling rows being contiguous in memory (which is true — the sparse
        evaluator lays children out per-parent via child_offsets)."""
        p = tl.program_id(0)
        hb = tl.program_id(1)

        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)

        row_offs = tl.arange(0, MAX_CHILDREN)
        col_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        row_mask = row_offs < count
        col_mask = col_offs < H
        mask_2d = row_mask[:, None] & col_mask[None, :]

        ptrs = values_ptr + (first + row_offs)[:, None] * H + col_offs[None, :]
        tile = tl.load(ptrs, mask=mask_2d, other=0.0)   # [MC, BH]
        acc = tl.sum(tile, axis=0)                      # [BH]

        out_ptrs = (
            out_ptr
            + (first + row_offs - out_offset)[:, None] * H
            + col_offs[None, :]
        )
        bcast = tl.broadcast_to(acc[None, :], (MAX_CHILDREN, BLOCK_H))
        tl.store(out_ptrs, bcast, mask=mask_2d)


if triton is not None:

    @triton.jit
    def _parent_sum_kernel(
        values_ptr,         # [total, H]
        child_offsets_ptr,  # [num_parents]
        child_count_ptr,    # [num_parents]
        out_ptr,            # [num_parents, H]
        H,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """Same 2D-tile load + sum as sibling-sum, but writes one row per
        parent (skips the broadcast store). Output shape [num_parents, H].
        """
        p = tl.program_id(0)
        hb = tl.program_id(1)

        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)

        row_offs = tl.arange(0, MAX_CHILDREN)
        col_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        row_mask = row_offs < count
        col_mask = col_offs < H
        mask_2d = row_mask[:, None] & col_mask[None, :]

        ptrs = values_ptr + (first + row_offs)[:, None] * H + col_offs[None, :]
        tile = tl.load(ptrs, mask=mask_2d, other=0.0)
        acc = tl.sum(tile, axis=0)

        tl.store(out_ptr + p * H + col_offs, acc, mask=col_mask)

    @triton.jit
    def _fused_divide_by_parent_sum_kernel(
        pos_ptr,             # [num_children, H] numerator
        fallback_ptr,        # [num_children, H] value where denom <= eps
        parent_sum_ptr,      # [num_parents, H]
        parent_index_ptr,    # [num_children] — child c → parent row
        out_ptr,             # [num_children, H]
        num_children,
        H,
        EPS,
        BLOCK_C: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        cb = tl.program_id(0)
        hb = tl.program_id(1)

        c_offs = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        h_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        c_mask = c_offs < num_children
        h_mask = h_offs < H
        mask_2d = c_mask[:, None] & h_mask[None, :]

        parents = tl.load(parent_index_ptr + c_offs, mask=c_mask, other=0)  # [BC]
        d_ptrs = parent_sum_ptr + parents[:, None] * H + h_offs[None, :]
        d = tl.load(d_ptrs, mask=mask_2d, other=0.0)

        p_ptrs = pos_ptr + c_offs[:, None] * H + h_offs[None, :]
        f_ptrs = fallback_ptr + c_offs[:, None] * H + h_offs[None, :]
        p = tl.load(p_ptrs, mask=mask_2d, other=0.0)
        f = tl.load(f_ptrs, mask=mask_2d, other=0.0)

        use_div = d > EPS
        d_safe = tl.maximum(d, EPS)
        result = tl.where(use_div, p / d_safe, f)

        out_ptrs = out_ptr + c_offs[:, None] * H + h_offs[None, :]
        tl.store(out_ptrs, result, mask=mask_2d)


def fused_parent_sum(
    values: torch.Tensor,          # [total, H]
    child_offsets: torch.Tensor,   # [num_parents]
    child_count: torch.Tensor,     # [num_parents]
    max_children: int = 8,
    block_h: int = 512,
) -> torch.Tensor:
    """2D-tile per-parent sum over siblings. Output ``[num_parents, H]`` —
    does *not* broadcast back to children (use
    ``fused_divide_by_parent_sum_`` to consume directly)."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 2
    assert child_offsets.is_contiguous() and child_offsets.dim() == 1
    assert child_count.is_contiguous() and child_count.shape == child_offsets.shape

    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2

    num_parents = child_offsets.shape[0]
    h = values.shape[1]
    out = torch.empty(num_parents, h, device=values.device, dtype=values.dtype)
    grid = (num_parents, triton.cdiv(h, block_h))
    _parent_sum_kernel[grid](
        values,
        child_offsets,
        child_count,
        out,
        h,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


def fused_divide_by_parent_sum_(
    pos: torch.Tensor,           # [num_children, H] numerator
    fallback: torch.Tensor,      # [num_children, H] fallback for denom <= eps
    parent_sum: torch.Tensor,    # [num_parents, H]
    parent_index: torch.Tensor,  # [num_children] int64 — child idx → parent row
    out: torch.Tensor,           # [num_children, H] (may alias pos or fallback)
    eps: float = 1e-8,
    block_c: int = 32,
    block_h: int = 128,
) -> None:
    """Compute ``out[c] = where(parent_sum[parent_index[c]] > eps,
    pos[c] / parent_sum[parent_index[c]], fallback[c])`` in one kernel
    without materializing ``parent_sum[parent_index[c]]`` as a separate tensor.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert pos.is_contiguous() and fallback.is_contiguous() and out.is_contiguous()
    assert parent_sum.is_contiguous() and parent_index.is_contiguous()
    assert pos.dim() == 2 and fallback.shape == pos.shape == out.shape
    assert parent_sum.dim() == 2 and parent_sum.shape[1] == pos.shape[1]
    num_children, h = pos.shape
    assert parent_index.shape == (num_children,)

    grid = (triton.cdiv(num_children, block_c), triton.cdiv(h, block_h))
    _fused_divide_by_parent_sum_kernel[grid](
        pos,
        fallback,
        parent_sum,
        parent_index,
        out,
        num_children,
        h,
        eps,
        BLOCK_C=block_c,
        BLOCK_H=block_h,
        num_warps=4,
    )


def fused_sibling_sum(
    values: torch.Tensor,          # [total, H]
    child_offsets: torch.Tensor,   # [num_parents] — child_offsets[p] gives first child idx
    child_count: torch.Tensor,     # [num_parents]
    bottom: int,                   # first child absolute index
    num_children: int,
    max_children: int = 8,
    block_h: int = 512,
) -> torch.Tensor:
    """For each child c in [bottom, bottom+num_children), compute the sum over
    its siblings (including itself) in ``values[c_sibling, :]``. Writes a
    child-aligned tensor of shape ``[num_children, H]``.

    Replaces the ``_pull_back_sum → _fan_out`` pattern with one kernel + no
    per-parent intermediate buffer.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 2
    assert child_offsets.is_contiguous() and child_offsets.dim() == 1
    assert child_count.is_contiguous() and child_count.dim() == 1
    assert child_offsets.shape == child_count.shape
    h = values.shape[1]
    num_parents = child_offsets.shape[0]

    # Triton requires tl.arange length to be a power of 2.
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2

    out = torch.empty(num_children, h, device=values.device, dtype=values.dtype)
    grid = (num_parents, triton.cdiv(h, block_h))
    _sibling_sum_kernel[grid](
        values,
        child_offsets,
        child_count,
        out,
        bottom,
        h,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# Kernel 10: fused unblocked-mass ratio (both numer and denom + where/div).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _unblocked_mass_ratio_kernel(
        numer_target_ptr,     # [B, H] marginal_policy
        denom_target_ptr,     # [B, H] beliefs_dest
        numer_cardsum_ptr,    # [B, 52]
        denom_cardsum_ptr,    # [B, 52]
        numer_S_ptr,          # [B]
        denom_S_ptr,          # [B]
        card_a_ptr,
        card_b_ptr,
        out_ptr,              # [B, H]
        H,
        NUM_CARDS: tl.constexpr,
        EPS,
        BLOCK_H: tl.constexpr,
    ):
        b = tl.program_id(0)
        Sn = tl.load(numer_S_ptr + b)
        Sd = tl.load(denom_S_ptr + b)
        n_row = numer_target_ptr + b * H
        d_row = denom_target_ptr + b * H
        ncs_row = numer_cardsum_ptr + b * NUM_CARDS
        dcs_row = denom_cardsum_ptr + b * NUM_CARDS
        out_row = out_ptr + b * H

        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        nt = tl.load(n_row + offs, mask=mask, other=0.0)
        dt = tl.load(d_row + offs, mask=mask, other=0.0)
        ca = tl.load(card_a_ptr + offs, mask=mask, other=0)
        cb = tl.load(card_b_ptr + offs, mask=mask, other=0)

        ncsa = tl.load(ncs_row + ca, mask=mask, other=0.0)
        ncsb = tl.load(ncs_row + cb, mask=mask, other=0.0)
        dcsa = tl.load(dcs_row + ca, mask=mask, other=0.0)
        dcsb = tl.load(dcs_row + cb, mask=mask, other=0.0)

        numer = tl.maximum(Sn - ncsa - ncsb + nt, 0.0)
        denom = tl.maximum(Sd - dcsa - dcsb + dt, 0.0)
        ratio = tl.where(denom > EPS, numer / denom, 0.0)
        tl.store(out_row + offs, ratio, mask=mask)


def unblocked_mass_ratio_triton(
    numer_target: torch.Tensor,  # [B, H] or [..., H]
    denom_target: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute ``where(unblocked(denom) > eps, unblocked(numer) / unblocked(denom), 0)``
    in one fused kernel (plus pytorch-side S/cardsum preprocessing).

    Replaces the triplet ``unblocked(x); unblocked(y); where(y > eps, x/y, 0)``
    used inside ``compute_expected_values`` with a single Triton kernel output.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert numer_target.shape == denom_target.shape
    orig = numer_target.shape
    n = numer_target.reshape(-1, _UNBLOCKED_NUM_HANDS).contiguous()
    d = denom_target.reshape(-1, _UNBLOCKED_NUM_HANDS).contiguous()
    b, h = n.shape

    card_a, card_b = _get_combo_cards(n.device)
    card_a_long = card_a.to(torch.int64)[None, :].expand(b, -1)
    card_b_long = card_b.to(torch.int64)[None, :].expand(b, -1)

    Sn = n.sum(dim=-1)
    Sd = d.sum(dim=-1)
    ncs = torch.zeros(b, _UNBLOCKED_NUM_CARDS, device=n.device, dtype=n.dtype)
    dcs = torch.zeros(b, _UNBLOCKED_NUM_CARDS, device=d.device, dtype=d.dtype)
    ncs.scatter_add_(1, card_a_long, n)
    ncs.scatter_add_(1, card_b_long, n)
    dcs.scatter_add_(1, card_a_long, d)
    dcs.scatter_add_(1, card_b_long, d)

    out = torch.empty_like(n)
    _unblocked_mass_ratio_kernel[(b,)](
        n, d, ncs.contiguous(), dcs.contiguous(),
        Sn.contiguous(), Sd.contiguous(),
        card_a, card_b, out,
        h,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        EPS=eps,
        BLOCK_H=2048,
        num_warps=4,
    )
    return out.view(orig)


# ---------------------------------------------------------------------------
# Kernel 11: _set_model_values_impl CFR-AVG mixing math (pointwise).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_model_values_mix_kernel(
        hand_values_ptr,        # [M, ...]
        last_model_values_ptr,  # [M, ...]
        out_ptr,                # [M, ...]
        old_plus_new_over_new,  # (old + new) / new
        old_over_new,           # old / new
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        h = tl.load(hand_values_ptr + offs, mask=mask, other=0.0)
        l = tl.load(last_model_values_ptr + offs, mask=mask, other=0.0)
        out = old_plus_new_over_new * h - old_over_new * l
        tl.store(out_ptr + offs, out, mask=mask)


def fused_model_values_mix(
    hand_values: torch.Tensor,
    last_model_values: torch.Tensor,
    old: float,
    new: float,
    block_size: int = 1024,
) -> torch.Tensor:
    """Compute ``((old + new) * hand_values - old * last_model_values) / new``
    in one kernel. Replaces the 4-kernel PyTorch sequence ``(old+new)*hand -
    old*last; /= new`` used inside ``_set_model_values_impl``'s CFR-AVG branch.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert hand_values.is_contiguous() and last_model_values.is_contiguous()
    assert hand_values.shape == last_model_values.shape
    assert new != 0.0
    out = torch.empty_like(hand_values)
    n = hand_values.numel()
    grid = (triton.cdiv(n, block_size),)
    _fused_model_values_mix_kernel[grid](
        hand_values,
        last_model_values,
        out,
        float((old + new) / new),
        float(old / new),
        n,
        BLOCK=block_size,
        num_warps=4,
    )
    return out


# ---------------------------------------------------------------------------
# CUDA graph capture of a full cfr_iteration.
# ---------------------------------------------------------------------------


@dataclass
class _EvaluatorStateSnapshot:
    """Subset of evaluator tensors mutated by cfr_iteration."""

    names: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]

    @classmethod
    def from_evaluator(cls, evaluator) -> "_EvaluatorStateSnapshot":
        names = [
            "policy_probs",
            "policy_probs_avg",
            "policy_probs_sample",
            "cumulative_regrets",
            "regret_weight_sums",
            "self_reach",
            "self_reach_avg",
            "beliefs",
            "beliefs_avg",
            "latest_values",
            "values_avg",
        ]
        tensors = tuple(getattr(evaluator, n).detach().clone() for n in names)
        return cls(tuple(names), tensors)

    def restore_to(self, evaluator) -> None:
        for name, saved in zip(self.names, self.tensors):
            getattr(evaluator, name).copy_(saved)

    def max_abs_diff(self, other: "_EvaluatorStateSnapshot") -> dict[str, float]:
        assert self.names == other.names
        out = {}
        for name, a, b in zip(self.names, self.tensors, other.tensors):
            out[name] = (a - b).abs().max().item()
        return out


class GraphedCFRIteration:
    """Captures one ``evaluator.cfr_iteration(t_capture)`` into a CUDA graph.

    Usage::

        runner = GraphedCFRIteration(evaluator)
        runner.capture(t_capture=warm_start)   # records the graph
        runner.replay()                        # re-runs iteration t_capture

    On replay, all kernel launch parameters are baked in (including the Python
    ``t`` used for DCFR scalars and mixing weights). This is intended for
    launch-overhead measurement and single-iteration correctness comparison,
    not production iteration.
    """

    def __init__(self, evaluator) -> None:
        if evaluator.device.type != "cuda":
            raise ValueError("GraphedCFRIteration requires a CUDA evaluator.")
        self.evaluator = evaluator
        self._graph: torch.cuda.CUDAGraph | None = None
        self._captured_t: int | None = None
        # We disable _record_stats during capture (has .item()).
        self._orig_record_stats = evaluator._record_stats

    def _stub_record_stats(self, t, old_policy_probs):  # noqa: ARG002
        return

    def capture(self, t_capture: int, num_warmup: int = 2) -> None:
        """Warm-up a few real iterations, then capture one into a CUDA graph.

        Warm-up is required so that cuDNN / cuBLAS workspaces and any lazy
        allocations stabilize before capture.
        """
        ev = self.evaluator

        # Warm up with real iterations so allocations stabilize.
        for i in range(num_warmup):
            ev.cfr_iteration(t_capture + i)

        # Disable the stat-recording .item() sync during capture.
        ev._record_stats = self._stub_record_stats
        try:
            # Separate stream (required by torch.cuda.graph).
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                # One more warm-up on the capture stream.
                ev.cfr_iteration(t_capture)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                ev.cfr_iteration(t_capture)
        finally:
            ev._record_stats = self._orig_record_stats

        self._graph = graph
        self._captured_t = t_capture

    def replay(self) -> None:
        if self._graph is None:
            raise RuntimeError("capture() must be called before replay().")
        self._graph.replay()

    @property
    def captured_t(self) -> int | None:
        return self._captured_t
