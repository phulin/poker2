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
        t_alpha_num_ptr,
        t_beta_num_ptr,
        t_alpha_den_ptr,
        t_beta_den_ptr,
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
            t_alpha_num = tl.load(t_alpha_num_ptr)
            t_beta_num = tl.load(t_beta_num_ptr)
            t_alpha_den = tl.load(t_alpha_den_ptr)
            t_beta_den = tl.load(t_beta_den_ptr)
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
        t_alpha_num_v = float(t**dcfr_alpha)
        t_beta_num_v = float(t**dcfr_beta)
        t_alpha_den_v = t_alpha_num_v + 1.0
        t_beta_den_v = t_beta_num_v + 1.0
    else:
        t_alpha_num_v = t_beta_num_v = t_alpha_den_v = t_beta_den_v = 1.0

    dev = cumulative_regrets.device
    dt = cumulative_regrets.dtype
    t_alpha_num = torch.tensor(t_alpha_num_v, dtype=dt, device=dev)
    t_beta_num = torch.tensor(t_beta_num_v, dtype=dt, device=dev)
    t_alpha_den = torch.tensor(t_alpha_den_v, dtype=dt, device=dev)
    t_beta_den = torch.tensor(t_beta_den_v, dtype=dt, device=dev)

    fused_dcfr_update_with_tensors_(
        cumulative_regrets=cumulative_regrets,
        regret_weight_sums=regret_weight_sums,
        regrets=regrets,
        t_alpha_num=t_alpha_num,
        t_beta_num=t_beta_num,
        t_alpha_den=t_alpha_den,
        t_beta_den=t_beta_den,
        apply_dcfr=apply_dcfr,
        cfr_plus=cfr_plus,
        positive_regrets_out=positive_regrets_out,
        block_size=block_size,
    )


def fused_dcfr_update_with_tensors_(
    cumulative_regrets: torch.Tensor,
    regret_weight_sums: torch.Tensor,
    regrets: torch.Tensor,
    t_alpha_num: torch.Tensor,
    t_beta_num: torch.Tensor,
    t_alpha_den: torch.Tensor,
    t_beta_den: torch.Tensor,
    apply_dcfr: bool,
    cfr_plus: bool,
    positive_regrets_out: torch.Tensor | None = None,
    block_size: int = 1024,
) -> None:
    """Graph-capturable DCFR update: scalars come from pre-filled 0-D tensors."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    n = cumulative_regrets.numel()
    write_pos = positive_regrets_out is not None
    pos_ptr = positive_regrets_out if write_pos else cumulative_regrets
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
        old_scalar_ptr,
        new_scalar_ptr,
        inv_total_ptr,      # 1 / (old + new)
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        a = tl.load(values_avg_ptr + offs, mask=mask, other=0.0)
        v = tl.load(latest_ptr + offs, mask=mask, other=0.0)
        old_scalar = tl.load(old_scalar_ptr)
        new_scalar = tl.load(new_scalar_ptr)
        inv_total = tl.load(inv_total_ptr)
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
    dev = values_avg.device
    dt = values_avg.dtype
    old_t = torch.tensor(float(old), dtype=dt, device=dev)
    new_t = torch.tensor(float(new), dtype=dt, device=dev)
    inv_t = torch.tensor(1.0 / total, dtype=dt, device=dev)
    fused_update_average_values_with_tensors_(
        values_avg, latest_values, old_t, new_t, inv_t, block_size=block_size
    )


def fused_update_average_values_with_tensors_(
    values_avg: torch.Tensor,
    latest_values: torch.Tensor,
    old: torch.Tensor,
    new: torch.Tensor,
    inv_total: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Graph-capturable version: scalars come from pre-filled 0-D tensors."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    n = values_avg.numel()
    grid = (triton.cdiv(n, block_size),)
    _fused_update_average_values_kernel[grid](
        values_avg,
        latest_values,
        old,
        new,
        inv_total,
        n,
        BLOCK=block_size,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 5b: update_average_values mixing + zero-sum subtract fused.
#   Replaces fused_update_average_values_with_tensors_ + _maybe_enforce_zero_sum.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_avg_values_zs_kernel(
        values_avg_ptr,    # [N, 2, H] in/out
        latest_ptr,        # [N, 2, H]
        beliefs_ptr,       # [N, 2, H]
        ignore_mask_ptr,   # [N] bool (only read if HAS_IGNORE)
        old_ptr,           # 0-D
        new_ptr,           # 0-D
        inv_total_ptr,     # 0-D
        N, H,
        HAS_IGNORE: tl.constexpr,
        ENFORCE_ZS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        n = tl.program_id(0)
        if n >= N:
            return
        old = tl.load(old_ptr)
        new = tl.load(new_ptr)
        inv_total = tl.load(inv_total_ptr)

        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        a0_ptr = values_avg_ptr + (n * 2 + 0) * H + offs
        a1_ptr = values_avg_ptr + (n * 2 + 1) * H + offs
        l0_ptr = latest_ptr + (n * 2 + 0) * H + offs
        l1_ptr = latest_ptr + (n * 2 + 1) * H + offs

        a0 = tl.load(a0_ptr, mask=mask, other=0.0)
        a1 = tl.load(a1_ptr, mask=mask, other=0.0)
        l0 = tl.load(l0_ptr, mask=mask, other=0.0)
        l1 = tl.load(l1_ptr, mask=mask, other=0.0)
        v0 = (a0 * old + l0 * new) * inv_total
        v1 = (a1 * old + l1 * new) * inv_total

        s = tl.zeros((), dtype=tl.float32)
        if ENFORCE_ZS:
            b0 = tl.load(beliefs_ptr + (n * 2 + 0) * H + offs, mask=mask, other=0.0)
            b1 = tl.load(beliefs_ptr + (n * 2 + 1) * H + offs, mask=mask, other=0.0)
            s = 0.5 * (
                tl.sum(tl.where(mask, v0 * b0, 0.0))
                + tl.sum(tl.where(mask, v1 * b1, 0.0))
            )
            if HAS_IGNORE:
                ig = tl.load(ignore_mask_ptr + n).to(tl.int1)
                s = tl.where(ig, 0.0, s)

        tl.store(a0_ptr, v0 - s, mask=mask)
        tl.store(a1_ptr, v1 - s, mask=mask)


def fused_avg_values_zero_sum_(
    values_avg: torch.Tensor,        # [N, 2, H] in/out
    latest_values: torch.Tensor,     # [N, 2, H]
    beliefs: torch.Tensor,           # [N, 2, H]
    old: torch.Tensor,               # 0-D
    new: torch.Tensor,               # 0-D
    inv_total: torch.Tensor,         # 0-D
    enforce_zero_sum: bool,
    ignore_mask: torch.Tensor | None = None,  # [N] bool
    block_h: int = 2048,
) -> None:
    """In-place: mix values_avg with latest_values, then (optionally) subtract
    per-row 0.5 * sum_p sum_h(v_p * b_p) to enforce zero-sum.

    Replaces ``fused_update_average_values_with_tensors_`` followed by
    ``_maybe_enforce_zero_sum`` in ``FusedSparseCFREvaluator.update_average_values``.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values_avg.is_contiguous() and latest_values.is_contiguous()
    assert beliefs.is_contiguous()
    assert values_avg.shape == latest_values.shape == beliefs.shape
    assert values_avg.dim() == 3 and values_avg.shape[1] == 2
    n_rows, _, h = values_avg.shape
    assert h <= block_h, f"BLOCK_H={block_h} must cover H={h}"
    if ignore_mask is not None:
        assert ignore_mask.is_contiguous() and ignore_mask.shape == (n_rows,)
        ignore_ptr = ignore_mask
    else:
        # Triton requires a real tensor pointer; never read when HAS_IGNORE=False.
        ignore_ptr = values_avg
    grid = (n_rows,)
    _fused_avg_values_zs_kernel[grid](
        values_avg,
        latest_values,
        beliefs,
        ignore_ptr,
        old,
        new,
        inv_total,
        n_rows, h,
        HAS_IGNORE=ignore_mask is not None,
        ENFORCE_ZS=enforce_zero_sum,
        BLOCK_H=block_h,
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Kernel 6: compute_instantaneous_regrets tail (fan_out + gather + sub + mul).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_regret_tail_kernel(
        values_achieved_ptr,     # [total, 2, H]
        actor_values_ptr,        # [top, H]     (parent-aligned)
        src_weights_ptr,         # [top, H]     (parent-aligned — was post-fan_out)
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

        # Row base pointers — weights now gathered from parent, same as expected.
        exp_row = actor_values_ptr + parent * H
        w_row = src_weights_ptr + parent * H
        ach_row = values_achieved_ptr + (c * 2 + prev_actor) * H
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
    actor_values: torch.Tensor,         # [top, H] — parent-aligned
    src_weights: torch.Tensor,          # [top, H] — parent-aligned (was post-fan_out)
    parent_index: torch.Tensor,         # [total] int64
    prev_actor: torch.Tensor,           # [total] int64
    bottom: int,
    block_h: int = 512,
) -> None:
    """Fused tail of ``compute_instantaneous_regrets``.

    For each child row ``c in [bottom, total)`` and hand ``h``::

        regrets[c, h] = src_weights[parent_index[c], h] * (
            values_achieved[c, prev_actor[c], h]
            - actor_values[parent_index[c], h]
        )

    Replaces the sequence ``fan_out(actor_values) + fan_out(src_weights) +
    gather + subtract + multiply + assign`` (6 kernels + 3 intermediates)
    with one kernel. ``src_weights`` is now parent-aligned — the fan-out is
    performed via the in-kernel ``parent_index`` gather, eliminating the
    ``[num_children, H]`` weights buffer.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert regrets.is_contiguous() and regrets.dim() == 2
    assert values_achieved.is_contiguous() and values_achieved.dim() == 3
    assert values_achieved.shape[1] == 2
    assert actor_values.is_contiguous() and actor_values.dim() == 2
    assert src_weights.is_contiguous() and src_weights.dim() == 2
    assert actor_values.shape == src_weights.shape
    assert parent_index.is_contiguous() and parent_index.dim() == 1
    assert prev_actor.is_contiguous() and prev_actor.dim() == 1

    total, h = regrets.shape
    assert values_achieved.shape == (total, 2, h)
    assert parent_index.shape == (total,)
    assert prev_actor.shape == (total,)

    grid = (total - bottom,)
    _fused_regret_tail_kernel[grid](
        values_achieved,
        actor_values,
        src_weights,
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
        old_scalar_ptr,
        new_scalar_ptr,
        total_weight_ptr,       # old + new
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

        old_scalar = tl.load(old_scalar_ptr)
        new_scalar = tl.load(new_scalar_ptr)
        total_weight = tl.load(total_weight_ptr)

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
    dev = policy_probs_avg.device
    dt = policy_probs_avg.dtype
    old_t = torch.tensor(float(old), dtype=dt, device=dev)
    new_t = torch.tensor(float(new), dtype=dt, device=dev)
    tot_t = torch.tensor(total_weight, dtype=dt, device=dev)
    fused_average_policy_mix_with_tensors_(
        policy_probs_avg, policy_probs, self_reach, self_reach_avg,
        to_act, parent_index, old_t, new_t, tot_t,
        bottom=bottom, eps=eps, block_h=block_h,
    )


def fused_average_policy_mix_with_tensors_(
    policy_probs_avg: torch.Tensor,
    policy_probs: torch.Tensor,
    self_reach: torch.Tensor,
    self_reach_avg: torch.Tensor,
    to_act: torch.Tensor,
    parent_index: torch.Tensor,
    old: torch.Tensor,
    new: torch.Tensor,
    total_weight: torch.Tensor,
    bottom: int,
    eps: float = 1e-5,
    block_h: int = 512,
) -> None:
    """Graph-capturable version: scalars come from pre-filled 0-D tensors."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    total, h = policy_probs_avg.shape
    grid = (total - bottom,)
    _fused_average_policy_mix_kernel[grid](
        self_reach,
        self_reach_avg,
        policy_probs,
        policy_probs_avg,
        to_act,
        parent_index,
        old,
        new,
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

# Per-device cache of the [1326, 53] card-projection tensor. Column 0 is all
# ones (yielding S = sum via matmul) and columns 1-52 are card-membership
# indicators (yielding cardsum[c] for c in [0, 52) via matmul).
_card_projection_cache: dict[torch.device, torch.Tensor] = {}


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


def _get_card_projection(device: torch.device) -> torch.Tensor:
    """Return the cached [1326, 53] projection tensor (fp32). Column 0 = ones,
    columns 1..52 = membership indicators for card (col - 1) across combos.

    ``(target @ P)[b, 0]`` = ``sum_h target[b, h]`` = S.
    ``(target @ P)[b, 1 + c]`` = ``sum_h target[b, h] * [combo h contains card c]``
    = cardsum[b, c].
    """
    cached = _card_projection_cache.get(device)
    if cached is not None:
        return cached
    from p2.env.card_utils import hand_combos_tensor

    combos = hand_combos_tensor(device=device)  # [1326, 2]
    P = torch.zeros(_UNBLOCKED_NUM_HANDS, 1 + _UNBLOCKED_NUM_CARDS, device=device)
    P[:, 0] = 1.0
    idx = torch.arange(_UNBLOCKED_NUM_HANDS, device=device)
    P[idx, 1 + combos[:, 0]] = 1.0
    P[idx, 1 + combos[:, 1]] = 1.0
    _card_projection_cache[device] = P.contiguous()
    return _card_projection_cache[device]


def _preprocess_unblocked_stats(
    target: torch.Tensor,  # [B, H] contiguous
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (S[B], cardsum[B, NUM_CARDS]) for use with the finalize kernel.

    Implemented as a single ``target @ P`` matmul (P: [H, 53]) that produces
    ``[B, 53]`` = ``(S, cardsum)`` stacked. ~3× faster than the previous
    ``sum + 2× scatter_add_`` sequence because (a) it's one kernel instead of
    three, and (b) native matmul uses hardware matrix multiply-accumulate
    (tensor cores at fp16, well-tuned fp32 path otherwise) instead of atomic
    scatter adds.
    """
    assert target.is_contiguous() and target.dim() == 2
    assert target.shape[1] == _UNBLOCKED_NUM_HANDS
    P = _get_card_projection(target.device).to(target.dtype)
    stacked = target @ P  # [B, 53]
    s = stacked[:, 0].contiguous()
    cardsum = stacked[:, 1:].contiguous()
    return s, cardsum


class ParentBeliefUnblockedStats:
    """Caches S + cardsum at parent shape for both player slices of beliefs.

    Construct once per CFR iteration from ``beliefs[:top]`` and reuse in both
    ``compute_instantaneous_regrets`` (opponent slice) and
    ``compute_expected_values`` (actor slice). Eliminates redundant
    ``sum + scatter_add`` preprocessing work when both call sites operate on
    the same beliefs tensor.
    """

    def __init__(self, beliefs_parents: torch.Tensor) -> None:
        # beliefs_parents: [top, 2, H]
        assert beliefs_parents.dim() == 3 and beliefs_parents.shape[1] == 2
        top, p, h = beliefs_parents.shape
        self.top = top
        self.beliefs_parents = beliefs_parents
        flat = beliefs_parents.reshape(top * 2, h).contiguous()
        s_flat, cs_flat = _preprocess_unblocked_stats(flat)
        # Reshape back to [top, 2, ...] for slicing by player.
        self._S = s_flat.view(top, 2)
        self._cardsum = cs_flat.view(top, 2, _UNBLOCKED_NUM_CARDS)

    def slice_for_player(
        self, player_per_node: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (target, S, cardsum) all aligned on ``player_per_node``.

        ``player_per_node[p] in {0, 1}`` selects which of the two slices to
        pick for each parent. Typically ``to_act[:top]`` (actor slice) or
        ``1 - to_act[:top]`` (opponent slice).
        """
        row_idx = torch.arange(self.top, device=self.beliefs_parents.device)
        target = self.beliefs_parents[row_idx, player_per_node, :].contiguous()
        s = self._S[row_idx, player_per_node].contiguous()
        cardsum = self._cardsum[row_idx, player_per_node, :].contiguous()
        return target, s, cardsum


def unblocked_mass_opp_at_parents_triton(
    beliefs: torch.Tensor,   # [total, 2, H]
    to_act: torch.Tensor,    # [total] int64
    top: int,
    cached_stats: ParentBeliefUnblockedStats | None = None,
) -> torch.Tensor:
    """Compute ``unblocked_mass(beliefs[:top, 1 - to_act[:top], :])``, returning
    a ``[top, H]`` tensor of opponent-reach unblocked mass at each parent node.

    Replaces the sequence::

        opponent_global_reach = calculate_unblocked_mass(beliefs.flip(dims=[1]))
        src_weights = opponent_global_reach.gather(1, to_act).squeeze(1)

    which processes the full ``[total, 2, H]`` tensor. Here we only touch
    ``[top, H]`` — 2 × total / top ≈ 13× less input at production scale, so
    ~5× less memory traffic for this unblocked-mass call.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    total, p, h = beliefs.shape
    assert p == 2 and h == _UNBLOCKED_NUM_HANDS

    opp_idx = (1 - to_act[:top]).to(torch.int64)
    if cached_stats is not None:
        target, s, cardsum = cached_stats.slice_for_player(opp_idx)
    else:
        row_idx = torch.arange(top, device=beliefs.device)
        target = beliefs[:top][row_idx, opp_idx, :].contiguous()
        s, cardsum = _preprocess_unblocked_stats(target)

    card_a, card_b = _get_combo_cards(target.device)
    out = torch.empty_like(target)
    _unblocked_mass_finalize_kernel[(top,)](
        target, cardsum, s, card_a, card_b, out,
        _UNBLOCKED_NUM_HANDS,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        BLOCK_H=2048,
        num_warps=4,
    )
    return out


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
    s, cardsum = _preprocess_unblocked_stats(flat)

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
    def _unblocked_mass_ratio_indirect_kernel(
        numer_target_ptr,     # [num_children, H] marginal_policy
        denom_target_ptr,     # [top, H] actor_beliefs (parent-aligned)
        numer_cardsum_ptr,    # [num_children, 52]
        denom_cardsum_ptr,    # [top, 52]
        numer_S_ptr,          # [num_children]
        denom_S_ptr,          # [top]
        parent_index_ptr,     # [num_children] — child c → parent in [0, top)
        card_a_ptr,
        card_b_ptr,
        out_ptr,              # [num_children, H]
        H,
        NUM_CARDS: tl.constexpr,
        EPS,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0)
        parent = tl.load(parent_index_ptr + c)

        Sn = tl.load(numer_S_ptr + c)
        Sd = tl.load(denom_S_ptr + parent)

        n_row = numer_target_ptr + c * H
        d_row = denom_target_ptr + parent * H
        ncs_row = numer_cardsum_ptr + c * NUM_CARDS
        dcs_row = denom_cardsum_ptr + parent * NUM_CARDS
        out_row = out_ptr + c * H

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


def unblocked_mass_ratio_indirect_triton(
    numer_target: torch.Tensor,   # [num_children, H] marginal_policy
    denom_target: torch.Tensor,   # [top, H] actor_beliefs (parent-aligned)
    parent_index: torch.Tensor,   # [num_children] int64 — child → parent idx
    eps: float = 1e-5,
    denom_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Like ``unblocked_mass_ratio_triton`` but the denominator's target lives
    at parent-aligned shape ``[top, H]``. Each child's denom is gathered inside
    the kernel via ``parent_index``.

    Savings vs the direct version:
      - denom-side scatter_add processes ``top`` rows instead of
        ``num_children`` (~5× less at production scale).
      - denom ``cardsum`` buffer is 5× smaller → better L2 cache behavior in
        the kernel.

    Returns ``[num_children, H]``.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert numer_target.is_contiguous() and numer_target.dim() == 2
    assert denom_target.is_contiguous() and denom_target.dim() == 2
    assert parent_index.is_contiguous() and parent_index.dim() == 1

    num_children, h = numer_target.shape
    top = denom_target.shape[0]
    assert denom_target.shape == (top, h)
    assert parent_index.shape == (num_children,)
    assert h == _UNBLOCKED_NUM_HANDS

    card_a, card_b = _get_combo_cards(numer_target.device)
    Sn, ncs = _preprocess_unblocked_stats(numer_target)
    if denom_stats is not None:
        Sd, dcs = denom_stats
        assert Sd.shape == (top,) and dcs.shape == (top, _UNBLOCKED_NUM_CARDS)
    else:
        Sd, dcs = _preprocess_unblocked_stats(denom_target)

    out = torch.empty_like(numer_target)
    _unblocked_mass_ratio_indirect_kernel[(num_children,)](
        numer_target,
        denom_target,
        ncs.contiguous(),
        dcs.contiguous(),
        Sn.contiguous(),
        Sd.contiguous(),
        parent_index,
        card_a,
        card_b,
        out,
        h,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        EPS=eps,
        BLOCK_H=2048,
        num_warps=4,
    )
    return out


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
        old_plus_new_over_new_ptr,  # (old + new) / new
        old_over_new_ptr,           # old / new
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        h = tl.load(hand_values_ptr + offs, mask=mask, other=0.0)
        l = tl.load(last_model_values_ptr + offs, mask=mask, other=0.0)
        old_plus_new_over_new = tl.load(old_plus_new_over_new_ptr)
        old_over_new = tl.load(old_over_new_ptr)
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
    dev = hand_values.device
    dt = hand_values.dtype
    onon_t = torch.tensor(float((old + new) / new), dtype=dt, device=dev)
    oon_t = torch.tensor(float(old / new), dtype=dt, device=dev)
    out = torch.empty_like(hand_values)
    fused_model_values_mix_with_tensors(hand_values, last_model_values, onon_t, oon_t, out, block_size=block_size)
    return out


def fused_model_values_mix_with_tensors(
    hand_values: torch.Tensor,
    last_model_values: torch.Tensor,
    old_plus_new_over_new: torch.Tensor,
    old_over_new: torch.Tensor,
    out: torch.Tensor,
    block_size: int = 1024,
) -> None:
    """Graph-capturable version: scalars come from pre-filled 0-D tensors."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    n = hand_values.numel()
    grid = (triton.cdiv(n, block_size),)
    _fused_model_values_mix_kernel[grid](
        hand_values,
        last_model_values,
        out,
        old_plus_new_over_new,
        old_over_new,
        n,
        BLOCK=block_size,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 11b: model-values CFR-AVG mixing + zero-sum subtract fused.
#   Replaces fused_model_values_mix_with_tensors + _maybe_enforce_zero_sum.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_model_values_mix_zs_kernel(
        h_ptr,        # [M, 2, H] hand_values
        l_ptr,        # [M, 2, H] last_model_values
        b_ptr,        # [M, 2, H] beliefs
        out_ptr,      # [M, 2, H]
        onon_ptr,     # 0-D (old + new) / new
        oon_ptr,      # 0-D old / new
        M, H,
        ENFORCE_ZS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        if m >= M:
            return
        onon = tl.load(onon_ptr)
        oon = tl.load(oon_ptr)

        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        h0_p = h_ptr + (m * 2 + 0) * H + offs
        h1_p = h_ptr + (m * 2 + 1) * H + offs
        l0_p = l_ptr + (m * 2 + 0) * H + offs
        l1_p = l_ptr + (m * 2 + 1) * H + offs
        o0_p = out_ptr + (m * 2 + 0) * H + offs
        o1_p = out_ptr + (m * 2 + 1) * H + offs

        h0 = tl.load(h0_p, mask=mask, other=0.0)
        h1 = tl.load(h1_p, mask=mask, other=0.0)
        l0 = tl.load(l0_p, mask=mask, other=0.0)
        l1 = tl.load(l1_p, mask=mask, other=0.0)
        u0 = h0 * onon - l0 * oon
        u1 = h1 * onon - l1 * oon

        s = tl.zeros((), dtype=tl.float32)
        if ENFORCE_ZS:
            b0 = tl.load(b_ptr + (m * 2 + 0) * H + offs, mask=mask, other=0.0)
            b1 = tl.load(b_ptr + (m * 2 + 1) * H + offs, mask=mask, other=0.0)
            s = 0.5 * (
                tl.sum(tl.where(mask, u0 * b0, 0.0))
                + tl.sum(tl.where(mask, u1 * b1, 0.0))
            )

        tl.store(o0_p, u0 - s, mask=mask)
        tl.store(o1_p, u1 - s, mask=mask)


def fused_model_values_mix_zero_sum(
    hand_values: torch.Tensor,             # [M, 2, H]
    last_model_values: torch.Tensor,       # [M, 2, H]
    beliefs: torch.Tensor,                 # [M, 2, H]
    old_plus_new_over_new: torch.Tensor,   # 0-D
    old_over_new: torch.Tensor,            # 0-D
    out: torch.Tensor,                     # [M, 2, H]
    enforce_zero_sum: bool,
    block_h: int = 2048,
) -> None:
    """Compute ``((old+new)*h - old*l) / new`` and (optionally) subtract the
    per-row zero-sum mean ``0.5 * sum_p sum_h(out_p * b_p)`` in one kernel.

    Replaces ``fused_model_values_mix_with_tensors`` followed by
    ``_maybe_enforce_zero_sum`` in ``FusedSparseCFREvaluator._set_model_values_impl``.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert hand_values.is_contiguous() and last_model_values.is_contiguous()
    assert beliefs.is_contiguous() and out.is_contiguous()
    assert hand_values.shape == last_model_values.shape == beliefs.shape == out.shape
    assert hand_values.dim() == 3 and hand_values.shape[1] == 2
    m, _, h = hand_values.shape
    assert h <= block_h, f"BLOCK_H={block_h} must cover H={h}"
    grid = (m,)
    _fused_model_values_mix_zs_kernel[grid](
        hand_values,
        last_model_values,
        beliefs,
        out,
        old_plus_new_over_new,
        old_over_new,
        m, h,
        ENFORCE_ZS=enforce_zero_sum,
        BLOCK_H=block_h,
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Kernel 12: fused weighted parent-sum for compute_expected_values depth loop.
#   Combines fused_weight_child_values + _pull_back_sum into a single
#   parent-aligned reduction.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_weighted_parent_sum_kernel(
        values_ptr,          # [total, 2, H] in/out
        prev_actor_ptr,      # [total]
        policy_hero_ptr,     # [total, H]
        policy_opp_ptr,      # [total, H]
        child_offsets_ptr,   # [num_parents] — absolute first-child row
        child_count_ptr,     # [num_parents]
        parent_base,         # absolute row of first parent in this depth slice
        H,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        player = tl.program_id(1)
        hb = tl.program_id(2)

        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        # Leaf "parents" (at intermediate depths in the sparse tree) have no
        # children; their values were set by set_leaf_values and must not be
        # overwritten. Original path used scatter_reduce(include_self=True),
        # which was a no-op for leaves.
        if count == 0:
            return

        col_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        col_mask = col_offs < H

        # Per-child accumulation. Loading the full [MAX_CHILDREN, BLOCK_H] tile
        # at once spills heavily to local memory (ncu: stack 3072 B, 96.9% of
        # local stores uncoalesced). Looping keeps live state at one [BLOCK_H]
        # accumulator, fits in registers, and skips children past `count`.
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                pa = tl.load(prev_actor_ptr + child)
                pol_ptr = tl.where(pa == player, policy_hero_ptr, policy_opp_ptr)
                pol = tl.load(pol_ptr + child * H + col_offs, mask=col_mask, other=0.0)
                v = tl.load(
                    values_ptr + (child * 2 + player) * H + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                acc += v * pol

        out_ptrs = values_ptr + ((parent_base + p) * 2 + player) * H + col_offs
        tl.store(out_ptrs, acc, mask=col_mask)


def fused_weighted_parent_sum(
    values: torch.Tensor,            # [total, 2, H] in/out
    prev_actor: torch.Tensor,        # [total]
    policy_hero: torch.Tensor,       # [total, H]
    policy_opp: torch.Tensor,        # [total, H]
    child_offsets: torch.Tensor,     # [num_parents] — absolute child row
    child_count: torch.Tensor,       # [num_parents]
    parent_base: int,
    max_children: int = 8,
    block_h: int = 2048,
) -> None:
    """Fuses ``fused_weight_child_values`` + ``_pull_back_sum`` for one depth.

    For each parent ``p`` in ``[parent_base, parent_base + num_parents)`` and
    ``player`` in ``{0, 1}``::

        values[parent_base + p, player, h] = sum_{i=0..count[p]-1}
            values[first[p] + i, player, h] *
            (policy_hero[first[p] + i, h] if player == prev_actor[first[p]+i]
             else policy_opp[first[p]+i, h])

    Parent rows must be pre-zeroed by the caller (they are — non-leaf rows are
    ``masked_fill(~leaf_mask, 0)`` at the top of ``compute_expected_values``).
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 3 and values.shape[1] == 2
    assert policy_hero.is_contiguous() and policy_opp.is_contiguous()
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    total, two, h = values.shape
    assert policy_hero.shape == (total, h) and policy_opp.shape == (total, h)
    assert prev_actor.shape == (total,)

    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    num_parents = child_offsets.shape[0]

    grid = (num_parents, 2, triton.cdiv(h, block_h))
    _fused_weighted_parent_sum_kernel[grid](
        values,
        prev_actor,
        policy_hero,
        policy_opp,
        child_offsets,
        child_count,
        parent_base,
        h,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 13: fused reach-weights per-depth propagation.
#   Replaces _fan_out + scatter_reduce(prod) from _calculate_reach_weights.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_reach_weights_kernel(
        reach_ptr,              # [total, 2, H] in/out
        policy_ptr,             # [total, H]
        allowed_mask_ptr,       # [total, H] bool
        parent_index_ptr,       # [total]
        prev_actor_ptr,         # [total]
        start,
        end,
        H,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0) + start
        if c >= end:
            return
        hb = tl.program_id(1)

        parent = tl.load(parent_index_ptr + c)
        prev_actor = tl.load(prev_actor_ptr + c)

        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H

        # Process both players in one program: shared parent_index/prev_actor
        # loads, plus a single policy load reused for whichever player is hero
        # (the other player just copies parent reach unchanged).
        pol = tl.load(policy_ptr + c * H + offs, mask=mask, other=0.0)
        # Fused block: zero out hands that are invalid at child c (board may
        # have changed across a chance node), eliminating the post-hoc
        # _block_beliefs masked_fill_.
        al = tl.load(allowed_mask_ptr + c * H + offs, mask=mask, other=0).to(tl.int1)

        v0 = tl.load(reach_ptr + (parent * 2 + 0) * H + offs, mask=mask, other=0.0)
        v1 = tl.load(reach_ptr + (parent * 2 + 1) * H + offs, mask=mask, other=0.0)
        if prev_actor == 0:
            v0 = v0 * pol
        else:
            v1 = v1 * pol
        v0 = tl.where(al, v0, 0.0)
        v1 = tl.where(al, v1, 0.0)
        tl.store(reach_ptr + (c * 2 + 0) * H + offs, v0, mask=mask)
        tl.store(reach_ptr + (c * 2 + 1) * H + offs, v1, mask=mask)


def fused_reach_weights_depth_(
    reach: torch.Tensor,             # [total, 2, H] in/out
    policy: torch.Tensor,            # [total, H]
    allowed_mask: torch.Tensor,      # [total, H] bool
    parent_index: torch.Tensor,      # [total]
    prev_actor: torch.Tensor,        # [total]
    start: int,
    end: int,
    block_h: int = 2048,
) -> None:
    """For each child row ``c in [start, end)`` and player ``p``::

        reach[c, p, h] = reach[parent_index[c], p, h] *
                        (policy[c, h] if p == prev_actor[c] else 1.0)

    Replaces the per-depth ``fan_out + scatter_reduce(prod)`` pair in
    ``_calculate_reach_weights``. Caller must invoke per depth in top-down
    order (children depend on freshly-computed parents).
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert reach.is_contiguous() and reach.dim() == 3 and reach.shape[1] == 2
    assert policy.is_contiguous() and policy.dim() == 2
    total, two, h = reach.shape
    assert policy.shape == (total, h)
    assert allowed_mask.shape == (total, h) and allowed_mask.is_contiguous()
    assert parent_index.shape == (total,) and prev_actor.shape == (total,)
    n = end - start
    if n <= 0:
        return
    grid = (n, triton.cdiv(h, block_h))
    _fused_reach_weights_kernel[grid](
        reach,
        policy,
        allowed_mask,
        parent_index,
        prev_actor,
        start,
        end,
        h,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 14: fan_out_deep * reach_weights + block + normalize in one kernel.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_deep_beliefs_kernel(
        root_beliefs_ptr,       # [N, 2, H]
        reach_ptr,              # [total, 2, H]
        allowed_mask_ptr,       # [total, H] bool
        allowed_prob_ptr,       # [total, H]
        root_index_ptr,         # [total]
        out_ptr,                # [total, 2, H]
        H,
        EPS,
        BLOCK_H: tl.constexpr,
    ):
        i = tl.program_id(0)
        p = tl.program_id(1)

        root_idx = tl.load(root_index_ptr + i)
        root_row = root_beliefs_ptr + (root_idx * 2 + p) * H
        reach_row = reach_ptr + (i * 2 + p) * H
        out_row = out_ptr + (i * 2 + p) * H
        mask_row = allowed_mask_ptr + i * H
        prob_row = allowed_prob_ptr + i * H

        # Single-tile path: H (=1326) fits in BLOCK_H (=2048). Keep `v`
        # register-resident across the sum + normalize so we don't spill the
        # intermediate to global memory and re-read it (was ~33% of this
        # kernel's DRAM traffic).
        off = tl.arange(0, BLOCK_H)
        m = off < H
        rb = tl.load(root_row + off, mask=m, other=0.0)
        rw = tl.load(reach_row + off, mask=m, other=0.0)
        al = tl.load(mask_row + off, mask=m, other=0).to(tl.int1)
        v = tl.where(al, rb * rw, 0.0)
        total = tl.sum(v)
        if total > EPS:
            out_v = v / total
        else:
            out_v = tl.load(prob_row + off, mask=m, other=0.0)
        tl.store(out_row + off, out_v, mask=m)


def fused_deep_beliefs_(
    out: torch.Tensor,               # [total, 2, H] in/out (root rows overwritten too)
    root_beliefs: torch.Tensor,      # [N, 2, H]
    reach_weights: torch.Tensor,     # [total, 2, H]
    allowed_mask: torch.Tensor,      # [total, H] bool
    allowed_prob: torch.Tensor,      # [total, H]
    root_index: torch.Tensor,        # [total] int64
    eps: float = 1e-5,
    block_h: int = 2048,
) -> None:
    """Fuses ``_fan_out_deep(root_beliefs) * reach_weights`` + block +
    normalize into one kernel. Replaces ``_propagate_all_beliefs``.

    For each node ``i`` and player ``p``::

        v = root_beliefs[root_index[i], p, :] * reach_weights[i, p, :]
        v = where(allowed_mask[i], v, 0)
        s = v.sum()
        out[i, p, :] = where(s > eps, v / s, allowed_prob[i])
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert out.is_contiguous() and out.dim() == 3 and out.shape[1] == 2
    assert root_beliefs.is_contiguous() and root_beliefs.dim() == 3
    assert reach_weights.is_contiguous() and reach_weights.shape == out.shape
    total, two, h = out.shape
    n = root_beliefs.shape[0]
    assert root_beliefs.shape == (n, 2, h)
    assert allowed_mask.shape == (total, h) and allowed_prob.shape == (total, h)
    assert allowed_mask.is_contiguous() and allowed_prob.is_contiguous()
    assert root_index.shape == (total,) and root_index.is_contiguous()
    assert h <= block_h, f"deep_beliefs assumes H ({h}) <= BLOCK_H ({block_h})"

    grid = (total, 2)
    _fused_deep_beliefs_kernel[grid](
        root_beliefs,
        reach_weights,
        allowed_mask,
        allowed_prob,
        root_index,
        out,
        h,
        eps,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# TScalars: device-side t-derived scalars for graph-capturable iteration.
# ---------------------------------------------------------------------------


class TScalars:
    """Container of pre-allocated 0-D device tensors for t-derived scalars.

    Populate via ``.update(t, ...)`` BEFORE entering a captured region. During
    graph capture, the kernels read these tensors via pointers, so replay
    picks up whatever value ``.update`` last wrote — no host→device copies
    baked into the graph.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        def _z():
            return torch.zeros((), dtype=dtype, device=device)

        self.device = device
        self.dtype = dtype
        # DCFR rescale scalars (all fp)
        self.t_alpha_num = _z()
        self.t_beta_num = _z()
        self.t_alpha_den = _z()
        self.t_beta_den = _z()
        # Policy/value averaging mix (old, new, old+new, 1/(old+new))
        self.mix_old = _z()
        self.mix_new = _z()
        self.mix_total = _z()
        self.mix_inv_total = _z()
        # Model-values mix (for _set_model_values_impl): (old+new)/new and old/new
        self.mix_onon = _z()  # (old + new) / new
        self.mix_oon = _z()   # old / new
        # t as int64 device scalar (for t_sample == t comparisons)
        self.t_tensor = torch.zeros((), dtype=torch.long, device=device)

    def update(
        self,
        t: int,
        dcfr_alpha: float,
        dcfr_beta: float,
        mix_old: float,
        mix_new: float,
    ) -> None:
        """Write t-derived scalars into the device tensors.

        Always a host→device copy via ``.fill_(python_float)`` — call OUTSIDE
        any captured region (before ``graph.replay()``).
        """
        t_alpha_num = float(t ** dcfr_alpha)
        t_beta_num = float(t ** dcfr_beta)
        self.t_alpha_num.fill_(t_alpha_num)
        self.t_beta_num.fill_(t_beta_num)
        self.t_alpha_den.fill_(t_alpha_num + 1.0)
        self.t_beta_den.fill_(t_beta_num + 1.0)
        total = float(mix_old) + float(mix_new)
        self.mix_old.fill_(float(mix_old))
        self.mix_new.fill_(float(mix_new))
        self.mix_total.fill_(total)
        self.mix_inv_total.fill_(1.0 / total if total != 0.0 else 1.0)
        if float(mix_new) != 0.0:
            self.mix_onon.fill_(total / float(mix_new))
            self.mix_oon.fill_(float(mix_old) / float(mix_new))
        self.t_tensor.fill_(int(t))


# ---------------------------------------------------------------------------
# Kernel 15: UNUSED showdown EV kernel (attempted — reverted).
#
# Tried a Triton rewrite of CFREvaluator._showdown_value to skip the
# [M, H, 52] per_card_mass cumsum. The approach computes 8 prefix-sum
# accumulators per (env m, sorted position k) by scanning all H sorted
# positions once, trading the [M, H, 52] memory pass for an O(M·H²) compute
# pattern.
#
# At the reference mixed-street shape (M=3264, H=1326, 52 cards), that
# reformulation is algorithmically unfavorable: H² ≈ 1.76M beats H·52 ≈ 69k
# per env by 25×, and in practice the tiled Triton version came in at
# 20–60× slower than the torch.compile'd PyTorch baseline (which already
# fuses the cumsum tightly with hardware-accelerated scan). Kept here for
# reference; disconnected from the evaluator.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _showdown_ev_kernel(
        b_opp_sorted_ptr,      # [M, H] fp32
        c1_sorted_ptr,         # [M, H] int32
        c2_sorted_ptr,         # [M, H] int32
        L_idx_ptr,             # [M, H] int32
        R_idx_ptr,             # [M, H] int32
        ev_sorted_out_ptr,     # [M, H] fp32 — OUT
        H,
        EPS,
        BLOCK_K: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        """Tiled [BLOCK_K × BLOCK_J] showdown-EV kernel.

        Grid: (M, ceil(H / BLOCK_K)). Each program owns BLOCK_K contiguous
        hero sorted-positions and scans all H villain positions in BLOCK_J
        tiles, accumulating eight per-hero-hand prefix sums that replace the
        [M, H, 52] ``per_card_mass``/``Pcards`` intermediate.

        card_ok masking is not needed: ``_block_beliefs`` already zeros
        b_opp on board-conflicting combos, so they contribute nothing.
        """
        m = tl.program_id(0)
        k_block = tl.program_id(1)

        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H

        row_b = b_opp_sorted_ptr + m * H
        row_c1 = c1_sorted_ptr + m * H
        row_c2 = c2_sorted_ptr + m * H
        row_L = L_idx_ptr + m * H
        row_R = R_idx_ptr + m * H

        c1_hero = tl.load(row_c1 + k_offs, mask=mask_k, other=-1).to(tl.int32)
        c2_hero = tl.load(row_c2 + k_offs, mask=mask_k, other=-1).to(tl.int32)
        L = tl.load(row_L + k_offs, mask=mask_k, other=0).to(tl.int32)
        R = tl.load(row_R + k_offs, mask=mask_k, other=0).to(tl.int32)
        b_k = tl.load(row_b + k_offs, mask=mask_k, other=0.0)

        P_L = tl.zeros([BLOCK_K], dtype=tl.float32)
        P_R = tl.zeros([BLOCK_K], dtype=tl.float32)
        Pc_L_c1 = tl.zeros([BLOCK_K], dtype=tl.float32)
        Pc_L_c2 = tl.zeros([BLOCK_K], dtype=tl.float32)
        Pc_R_c1 = tl.zeros([BLOCK_K], dtype=tl.float32)
        Pc_R_c2 = tl.zeros([BLOCK_K], dtype=tl.float32)
        Pc_last_c1 = tl.zeros([BLOCK_K], dtype=tl.float32)
        Pc_last_c2 = tl.zeros([BLOCK_K], dtype=tl.float32)

        for j_start in tl.range(0, H, BLOCK_J):
            j_offs = j_start + tl.arange(0, BLOCK_J)
            mask_j = j_offs < H
            b_j = tl.load(row_b + j_offs, mask=mask_j, other=0.0)      # [BJ]
            cj1 = tl.load(row_c1 + j_offs, mask=mask_j, other=-1).to(tl.int32)
            cj2 = tl.load(row_c2 + j_offs, mask=mask_j, other=-1).to(tl.int32)

            # [BLOCK_K, BLOCK_J] masks
            j_bc = j_offs[None, :]   # [1, BJ]
            in_before_L = j_bc < L[:, None]                              # [BK, BJ]
            in_before_R = j_bc < R[:, None]
            has_c1 = (cj1[None, :] == c1_hero[:, None]) | (cj2[None, :] == c1_hero[:, None])
            has_c2 = (cj1[None, :] == c2_hero[:, None]) | (cj2[None, :] == c2_hero[:, None])
            b_bc = tl.where(mask_j[None, :], b_j[None, :], 0.0).broadcast_to((BLOCK_K, BLOCK_J))

            P_L += tl.sum(tl.where(in_before_L, b_bc, 0.0), axis=1)
            P_R += tl.sum(tl.where(in_before_R, b_bc, 0.0), axis=1)
            Pc_L_c1 += tl.sum(tl.where(in_before_L & has_c1, b_bc, 0.0), axis=1)
            Pc_L_c2 += tl.sum(tl.where(in_before_L & has_c2, b_bc, 0.0), axis=1)
            Pc_R_c1 += tl.sum(tl.where(in_before_R & has_c1, b_bc, 0.0), axis=1)
            Pc_R_c2 += tl.sum(tl.where(in_before_R & has_c2, b_bc, 0.0), axis=1)
            Pc_last_c1 += tl.sum(tl.where(has_c1, b_bc, 0.0), axis=1)
            Pc_last_c2 += tl.sum(tl.where(has_c2, b_bc, 0.0), axis=1)

        win_mass = P_L - Pc_L_c1 - Pc_L_c2
        seg_sum = P_R - P_L
        seg_c1 = Pc_R_c1 - Pc_L_c1
        seg_c2 = Pc_R_c2 - Pc_L_c2
        tie_mass = seg_sum - seg_c1 - seg_c2 + b_k

        denom = 1.0 - Pc_last_c1 - Pc_last_c2 + b_k
        use_div = denom > EPS
        denom_safe = tl.maximum(denom, EPS)
        win_prob = tl.where(use_div, win_mass / denom_safe, 0.0)
        tie_prob = tl.where(use_div, tie_mass / denom_safe, 0.0)
        loss_prob = tl.where(use_div, 1.0 - win_prob - tie_prob, 0.0)
        ev = win_prob - loss_prob
        tl.store(ev_sorted_out_ptr + m * H + k_offs, ev, mask=mask_k)


def showdown_ev_triton(
    b_opp_sorted: torch.Tensor,       # [M, H] fp32
    c1_sorted: torch.Tensor,          # [M, H] int
    c2_sorted: torch.Tensor,          # [M, H] int
    L_idx: torch.Tensor,              # [M, H] int
    R_idx: torch.Tensor,              # [M, H] int
    eps: float = 1e-8,
    block_k: int = 32,
    block_j: int = 128,
) -> torch.Tensor:
    """Compute ``EV_hand_sorted`` [M, H] directly from sorted-order inputs.

    Replaces the PyTorch ``_showdown_value`` pipeline (per_card_mass cumsum +
    Pcards gathers + divide) with one Triton kernel. Caller remains
    responsible for: permuting EV back via ``inv_sorted``; multiplying by
    ``hand_ok_mask``; and scaling by ``potential / scale``.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert b_opp_sorted.is_contiguous() and b_opp_sorted.dim() == 2
    M, H = b_opp_sorted.shape
    assert c1_sorted.shape == (M, H) and c2_sorted.shape == (M, H)
    assert L_idx.shape == (M, H) and R_idx.shape == (M, H)
    c1_i32 = c1_sorted.to(torch.int32).contiguous()
    c2_i32 = c2_sorted.to(torch.int32).contiguous()
    L_i32 = L_idx.to(torch.int32).contiguous()
    R_i32 = R_idx.to(torch.int32).contiguous()
    out = torch.empty(M, H, device=b_opp_sorted.device, dtype=torch.float32)

    grid = (M, triton.cdiv(H, block_k))
    _showdown_ev_kernel[grid](
        b_opp_sorted.contiguous(),
        c1_i32,
        c2_i32,
        L_i32,
        R_i32,
        out,
        H,
        eps,
        BLOCK_K=block_k,
        BLOCK_J=block_j,
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
    """Captures ``evaluator.cfr_iteration`` into a CUDA graph, replayable for
    any ``t`` that falls in the same Python-branch regime as ``t_capture``.

    Usage::

        runner = GraphedCFRIteration(evaluator)
        runner.capture(t_capture=warm_start)   # records the graph
        runner.replay(t=warm_start + 1)        # runs iteration t=warm_start+1
        runner.replay(t=warm_start + 2)        # ...and so on

    On replay, host-side Python schedules are re-applied and the evaluator's
    ``TScalars`` device tensors are refreshed *outside* the captured region,
    so the kernels (which read scalars via pointers) pick up the new ``t``.

    Constraints on ``t`` values used for replay:
      - Must follow the same Python-level branches as ``t_capture`` (e.g.,
        both ``t > dcfr_delay`` / both ``t > 1`` so ``last_model_values`` is
        populated). Typical practice: warm up past early-t branches, then
        capture and reuse for the remaining iterations.
      - Tree structure (depth_offsets, child_offsets, ...) is baked in at
        capture time. Don't re-construct subgames after capture.
    """

    def __init__(self, evaluator) -> None:
        if evaluator.device.type != "cuda":
            raise ValueError("GraphedCFRIteration requires a CUDA evaluator.")
        self.evaluator = evaluator
        self._graph: torch.cuda.CUDAGraph | None = None
        self._captured_t: int | None = None
        self._orig_record_stats = evaluator._record_stats

    def _stub_record_stats(self, t, old_policy_probs):  # noqa: ARG002
        return

    def capture(self, t_capture: int, num_warmup: int = 2) -> None:
        """Warm-up a few real iterations, then capture one into a CUDA graph."""
        ev = self.evaluator
        if not hasattr(ev, "_t_scalars"):
            raise ValueError(
                "Evaluator must be a FusedSparseCFREvaluator (or have a "
                "._t_scalars TScalars holder) for graph capture."
            )

        for i in range(num_warmup):
            ev.cfr_iteration(t_capture + i)

        ev._record_stats = self._stub_record_stats
        prev_skip_stats = getattr(ev, "_skip_record_stats", False)
        ev._skip_record_stats = True
        try:
            # Fill scalars once before capture; inside capture we skip the
            # Python-side update so no host→device fills get baked in.
            ev.prepare_replay(t_capture)

            prev_skip = ev._skip_t_scalars_update
            ev._skip_t_scalars_update = True
            try:
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    # One warm-up on the capture stream (scalars pre-filled).
                    ev.cfr_iteration(t_capture)
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=s):
                    ev.cfr_iteration(t_capture)
            finally:
                ev._skip_t_scalars_update = prev_skip
        finally:
            ev._record_stats = self._orig_record_stats
            ev._skip_record_stats = prev_skip_stats

        self._graph = graph
        self._captured_t = t_capture

    def replay(self, t: int | None = None) -> None:
        """Replay the captured iteration. If ``t`` is given, refresh host-side
        schedules + TScalars (outside the graph) so the kernels compute with
        that ``t`` — otherwise replay reuses whatever scalars are already in
        the device tensors.
        """
        if self._graph is None:
            raise RuntimeError("capture() must be called before replay().")
        if t is not None:
            self.evaluator.prepare_replay(t)
        self._graph.replay()

    @property
    def captured_t(self) -> int | None:
        return self._captured_t
