"""Triton + CUDA-graph fusions for a single CFR iteration.

Triton kernels (pointwise + simple reductions):

1. ``fused_dcfr_update_`` — DCFR rescale + cumulative-regret
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

6. ``fused_cfr_delta_stats`` — direct parent/hand reduction for the CFR policy
   delta metric, avoiding dense child→action pullback temporaries.

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
* Targets the default config path: ``cfr_type == discounted``,
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

        c = c + r

        if CFR_PLUS:
            c = tl.maximum(c, 0.0)

        tl.store(cumul_ptr + offs, c, mask=mask)

        if WRITE_POS:
            pos = tl.maximum(c, 0.0)
            tl.store(pos_out_ptr + offs, pos, mask=mask)


def fused_dcfr_update_(
    cumulative_regrets: torch.Tensor,
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

        if cfr_type == discounted:
            num = where(c > 0, t**a, t**b)
            den = where(c > 0, t**a + 1, t**b + 1)
            c *= num; c /= den
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
            "Linear CFR path not supported; default config uses discounted."
        )

    assert cumulative_regrets.is_contiguous()
    assert regrets.is_contiguous()
    assert cumulative_regrets.shape == regrets.shape

    apply_dcfr = cfr_type == CFRType.discounted
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
        target_ptr,  # [R, H] flat view of target (R = N*P or N)
        allowed_mask_ptr,  # [R_outer, H] bool (broadcast along P)
        allowed_prob_ptr,  # [R_outer, H] fallback
        row_to_outer_stride,  # stride from row index to outer index (P for [N,P,H], 1 for [N,H])
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
        denom_ptr,  # [N, H]
        uniform_ptr,  # [N, H]
        out_ptr,  # [N, H]
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
        values_src_ptr,  # [M, 2, H]
        prev_actor_ptr,  # [M] int64
        policy_hero_ptr,  # [M, H] — policy at child
        policy_opp_ptr,  # [M, H] — opponent_conditioned_policy at child
        out_ptr,  # [M, 2, H]
        M,
        H,
        BLOCK_H: tl.constexpr,
    ):
        m_idx = tl.program_id(0)
        p_idx = tl.program_id(1)
        prev_actor = tl.load(prev_actor_ptr + m_idx)
        is_hero = p_idx == prev_actor

        row_offset = (m_idx * 2 + p_idx) * H
        pol_row_offset = m_idx * H

        for start in tl.range(0, H, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            mask = offs < H
            v = tl.load(values_src_ptr + row_offset + offs, mask=mask, other=0.0)
            if is_hero:
                p = tl.load(
                    policy_hero_ptr + pol_row_offset + offs, mask=mask, other=0.0
                )
            else:
                p = tl.load(
                    policy_opp_ptr + pol_row_offset + offs, mask=mask, other=0.0
                )
            tl.store(out_ptr + row_offset + offs, v * p, mask=mask)


def fused_weight_child_values(
    values_src: torch.Tensor,  # [M, 2, H]
    prev_actor: torch.Tensor,  # [M]
    policy_hero: torch.Tensor,  # [M, H]
    policy_opp: torch.Tensor,  # [M, H]
    out: torch.Tensor,  # [M, 2, H]
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
        m,
        h,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 5: update_average_values mixing.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_update_average_values_kernel(
        values_avg_ptr,  # [N, 2, H] in/out
        latest_ptr,  # [N, 2, H]
        old_scalar_ptr,
        new_scalar_ptr,
        inv_total_ptr,  # 1 / (old + new)
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
        values_avg_ptr,  # [N, 2, H] in/out
        latest_ptr,  # [N, 2, H]
        beliefs_ptr,  # [N, 2, H]
        ignore_mask_ptr,  # [N] bool (only read if HAS_IGNORE)
        old_ptr,  # 0-D
        new_ptr,  # 0-D
        inv_total_ptr,  # 0-D
        N,
        H,
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
    values_avg: torch.Tensor,  # [N, 2, H] in/out
    latest_values: torch.Tensor,  # [N, 2, H]
    beliefs: torch.Tensor,  # [N, 2, H]
    old: torch.Tensor,  # 0-D
    new: torch.Tensor,  # 0-D
    inv_total: torch.Tensor,  # 0-D
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
        n_rows,
        h,
        HAS_IGNORE=ignore_mask is not None,
        ENFORCE_ZS=enforce_zero_sum,
        BLOCK_H=block_h,
        num_warps=8,
    )


if triton is not None:

    @triton.jit
    def _fused_avg_values_multiway_kernel(
        values_avg_ptr,  # [N, P, H] in/out
        latest_ptr,  # [N, P, H]
        beliefs_ptr,  # [N, P, H]
        ignore_mask_ptr,  # [N] bool (only read if HAS_IGNORE)
        old_ptr,  # 0-D
        new_ptr,  # 0-D
        inv_total_ptr,  # 0-D
        N,
        H,
        NUM_PLAYERS: tl.constexpr,
        HAS_IGNORE: tl.constexpr,
        ENFORCE_ZS: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        n = tl.program_id(0)
        if n >= N:
            return
        old = tl.load(old_ptr)
        new = tl.load(new_ptr)
        inv_total = tl.load(inv_total_ptr)

        players = tl.arange(0, BLOCK_P)
        offs = tl.arange(0, BLOCK_H)
        player_mask = players < NUM_PLAYERS
        hand_mask = offs < H
        mask = player_mask[:, None] & hand_mask[None, :]
        ptrs = (n * NUM_PLAYERS + players[:, None]) * H + offs[None, :]

        avg = tl.load(values_avg_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        latest = tl.load(latest_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
        mixed = (avg * old + latest * new) * inv_total

        correction = tl.zeros((), dtype=tl.float32)
        if ENFORCE_ZS:
            beliefs = tl.load(beliefs_ptr + ptrs, mask=mask, other=0.0).to(tl.float32)
            total = tl.sum(tl.where(mask, mixed * beliefs, 0.0))
            correction = total / NUM_PLAYERS
            if HAS_IGNORE:
                ignore = tl.load(ignore_mask_ptr + n).to(tl.int1)
                correction = tl.where(ignore, 0.0, correction)

        tl.store(values_avg_ptr + ptrs, mixed - correction, mask=mask)


def fused_avg_values_multiway_(
    values_avg: torch.Tensor,  # [N, P, H] in/out
    latest_values: torch.Tensor,  # [N, P, H]
    beliefs: torch.Tensor,  # [N, P, H]
    old: torch.Tensor,  # 0-D
    new: torch.Tensor,  # 0-D
    inv_total: torch.Tensor,  # 0-D
    enforce_zero_sum: bool,
    ignore_mask: torch.Tensor | None = None,  # [N] bool
    block_h: int = 2048,
) -> None:
    """Multiway average-value mix with optional row zero-sum projection."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values_avg.is_contiguous() and latest_values.is_contiguous()
    assert beliefs.is_contiguous()
    assert values_avg.shape == latest_values.shape == beliefs.shape
    assert values_avg.dim() == 3 and values_avg.shape[1] >= 2
    n_rows, players, h = values_avg.shape
    assert h <= block_h, f"BLOCK_H={block_h} must cover H={h}"
    if ignore_mask is not None:
        assert ignore_mask.is_contiguous() and ignore_mask.shape == (n_rows,)
        ignore_ptr = ignore_mask
    else:
        ignore_ptr = values_avg
    block_p = 1
    while block_p < players:
        block_p *= 2
    _fused_avg_values_multiway_kernel[(n_rows,)](
        values_avg,
        latest_values,
        beliefs,
        ignore_ptr,
        old,
        new,
        inv_total,
        n_rows,
        h,
        NUM_PLAYERS=players,
        HAS_IGNORE=ignore_mask is not None,
        ENFORCE_ZS=enforce_zero_sum,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Kernel 6: compute_instantaneous_regrets tail (fan_out + gather + sub + mul).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_regret_tail_kernel(
        values_achieved_ptr,  # [total, 2, H]
        values_expected_ptr,  # [top, 2, H]  (was: actor_values [top, H])
        to_act_ptr,  # [top]        — selects the actor row inside values_expected
        src_weights_ptr,  # [top, H]     (parent-aligned — was post-fan_out)
        parent_index_ptr,  # [total] — parent_index[c] gives parent row in [0, top)
        prev_actor_ptr,  # [total] — 0 or 1
        regrets_ptr,  # [total, H] output (only rows [bottom, total) written)
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
        # In-kernel actor select: replaces a [top, H] aten::index + intermediate
        # buffer in the caller. One extra scalar load per program.
        actor_p = tl.load(to_act_ptr + parent)

        # Row base pointers — weights now gathered from parent, same as expected.
        exp_row = values_expected_ptr + (parent * 2 + actor_p) * H
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
    regrets: torch.Tensor,  # [total, H] — in/out (only [bottom:] written)
    values_achieved: torch.Tensor,  # [total, 2, H]
    values_expected: torch.Tensor,  # [top, 2, H] — actor row picked in-kernel
    to_act: torch.Tensor,  # [top]      — int64, picks the actor row
    src_weights: torch.Tensor,  # [top, H] — parent-aligned (was post-fan_out)
    parent_index: torch.Tensor,  # [total] int64
    prev_actor: torch.Tensor,  # [total] int64
    bottom: int,
    block_h: int = 512,
) -> None:
    """Fused tail of ``compute_instantaneous_regrets``.

    For each child row ``c in [bottom, total)`` and hand ``h``::

        regrets[c, h] = src_weights[parent_index[c], h] * (
            values_achieved[c, prev_actor[c], h]
            - values_expected[parent_index[c], to_act[parent_index[c]], h]
        )

    Replaces the sequence ``fan_out(actor_values) + fan_out(src_weights) +
    gather + subtract + multiply + assign`` (6 kernels + 3 intermediates)
    with one kernel. ``src_weights`` is parent-aligned and ``actor_values``
    is now picked from ``values_expected`` via an in-kernel ``to_act``
    gather, eliminating the caller-side ``aten::index`` and the ``[top, H]``
    intermediate buffer.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert regrets.is_contiguous() and regrets.dim() == 2
    assert values_achieved.is_contiguous() and values_achieved.dim() == 3
    assert values_achieved.shape[1] == 2
    assert values_expected.is_contiguous() and values_expected.dim() == 3
    assert values_expected.shape[1] == 2
    assert src_weights.is_contiguous() and src_weights.dim() == 2
    assert to_act.is_contiguous() and to_act.dim() == 1
    assert parent_index.is_contiguous() and parent_index.dim() == 1
    assert prev_actor.is_contiguous() and prev_actor.dim() == 1

    total, h = regrets.shape
    top = values_expected.shape[0]
    assert values_achieved.shape == (total, 2, h)
    assert src_weights.shape == (top, h)
    assert to_act.shape == (top,)
    assert parent_index.shape == (total,)
    assert prev_actor.shape == (total,)

    grid = (total - bottom,)
    _fused_regret_tail_kernel[grid](
        values_achieved,
        values_expected,
        to_act,
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


if triton is not None:

    @triton.jit
    def _fused_regret_tail_multiway_kernel(
        values_achieved_ptr,  # [total, P, H]
        values_expected_ptr,  # [top, P, H]
        to_act_ptr,  # [top]
        src_weights_ptr,  # [top, H]
        parent_index_ptr,  # [total]
        prev_actor_ptr,  # [total]
        regrets_ptr,  # [total, H] output (only rows [bottom, total) written)
        bottom,
        total,
        H,
        NUM_PLAYERS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0) + bottom
        if c >= total:
            return
        parent = tl.load(parent_index_ptr + c)
        prev_actor = tl.load(prev_actor_ptr + c)
        actor_p = tl.load(to_act_ptr + parent)

        exp_row = values_expected_ptr + (parent * NUM_PLAYERS + actor_p) * H
        ach_row = values_achieved_ptr + (c * NUM_PLAYERS + prev_actor) * H
        w_row = src_weights_ptr + parent * H
        out_row = regrets_ptr + c * H

        for start in tl.range(0, H, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            mask = offs < H
            expected = tl.load(exp_row + offs, mask=mask, other=0.0)
            achieved = tl.load(ach_row + offs, mask=mask, other=0.0)
            w = tl.load(w_row + offs, mask=mask, other=0.0)
            tl.store(out_row + offs, w * (achieved - expected), mask=mask)


def fused_regret_tail_multiway_(
    regrets: torch.Tensor,  # [total, H]
    values_achieved: torch.Tensor,  # [total, P, H]
    values_expected: torch.Tensor,  # [top, P, H]
    to_act: torch.Tensor,  # [top]
    src_weights: torch.Tensor,  # [top, H]
    parent_index: torch.Tensor,  # [total]
    prev_actor: torch.Tensor,  # [total]
    bottom: int,
    block_h: int = 512,
) -> None:
    """Multiway fused tail of ``compute_instantaneous_regrets``."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert regrets.is_contiguous() and regrets.dim() == 2
    assert values_achieved.is_contiguous() and values_achieved.dim() == 3
    assert values_expected.is_contiguous() and values_expected.dim() == 3
    assert src_weights.is_contiguous() and src_weights.dim() == 2
    assert to_act.is_contiguous() and to_act.dim() == 1
    assert parent_index.is_contiguous() and parent_index.dim() == 1
    assert prev_actor.is_contiguous() and prev_actor.dim() == 1

    total, h = regrets.shape
    top, p, h_expected = values_expected.shape
    assert h_expected == h
    assert values_achieved.shape == (total, p, h)
    assert src_weights.shape == (top, h)
    assert to_act.shape == (top,)
    assert parent_index.shape == (total,)
    assert prev_actor.shape == (total,)

    grid = (total - bottom,)
    _fused_regret_tail_multiway_kernel[grid](
        values_achieved,
        values_expected,
        to_act,
        src_weights,
        parent_index,
        prev_actor,
        regrets,
        bottom,
        total,
        h,
        NUM_PLAYERS=p,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 6b: unblocked source-weight finalize + regret/DCFR update.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_unblocked_regret_dcfr_update_kernel(
        target_ptr,  # [top, H] opponent beliefs at parent rows
        stats_ptr,  # [top, 53] stacked (S, cardsum)
        card_a_ptr,  # [H]
        card_b_ptr,  # [H]
        allowed_ptr,  # optional [top, H]
        values_achieved_ptr,  # [total, 2, H]
        values_expected_ptr,  # [top, 2, H]
        to_act_ptr,  # [top]
        child_offsets_ptr,  # [top]
        child_count_ptr,  # [top]
        prev_actor_ptr,  # [total]
        cumul_ptr,  # [total, H]
        pos_out_ptr,  # [total, H]
        last_regrets_ptr,  # optional [total, H]
        prediction_scale_ptr,
        current_player_ptr,
        t_alpha_num_ptr,
        t_beta_num_ptr,
        t_alpha_den_ptr,
        t_beta_den_ptr,
        H,
        NUM_CARDS: tl.constexpr,
        APPLY_DCFR: tl.constexpr,
        CFR_PLUS: tl.constexpr,
        WRITE_POS: tl.constexpr,
        HAS_PREDICTIVE: tl.constexpr,
        HAS_ALLOWED: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)

        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H

        ca = tl.load(card_a_ptr + offs, mask=mask, other=0)
        cb = tl.load(card_b_ptr + offs, mask=mask, other=0)
        S = tl.load(stats_ptr + p * (NUM_CARDS + 1))
        t = tl.load(target_ptr + p * H + offs, mask=mask, other=0.0)
        csa = tl.load(
            stats_ptr + p * (NUM_CARDS + 1) + 1 + ca,
            mask=mask,
            other=0.0,
        )
        csb = tl.load(
            stats_ptr + p * (NUM_CARDS + 1) + 1 + cb,
            mask=mask,
            other=0.0,
        )
        src_w = tl.maximum(S - csa - csb + t, 0.0)
        if HAS_ALLOWED:
            allowed = tl.load(allowed_ptr + p * H + offs, mask=mask, other=0).to(
                tl.int1
            )
            src_w = tl.where(allowed, src_w, 0.0)

        actor = tl.load(to_act_ptr + p)
        expected = tl.load(
            values_expected_ptr + (p * 2 + actor) * H + offs,
            mask=mask,
            other=0.0,
        )

        if APPLY_DCFR:
            t_alpha_num = tl.load(t_alpha_num_ptr)
            t_beta_num = tl.load(t_beta_num_ptr)
            t_alpha_den = tl.load(t_alpha_den_ptr)
            t_beta_den = tl.load(t_beta_den_ptr)
        if HAS_PREDICTIVE:
            prediction_scale = tl.load(prediction_scale_ptr)
            current_player = tl.load(current_player_ptr)

        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                prev_actor = tl.load(prev_actor_ptr + child)
                achieved = tl.load(
                    values_achieved_ptr + (child * 2 + prev_actor) * H + offs,
                    mask=mask,
                    other=0.0,
                )
                r = src_w * (achieved - expected)

                cumul_offs = child * H + offs
                c = tl.load(cumul_ptr + cumul_offs, mask=mask, other=0.0)
                if APPLY_DCFR:
                    positive = c > 0.0
                    num = tl.where(positive, t_alpha_num, t_beta_num)
                    den = tl.where(positive, t_alpha_den, t_beta_den)
                    c = c * num
                    c = c / den

                c = c + r
                if CFR_PLUS:
                    c = tl.maximum(c, 0.0)

                tl.store(cumul_ptr + cumul_offs, c, mask=mask)
                if WRITE_POS:
                    policy_c = c
                    if HAS_PREDICTIVE:
                        last = tl.load(
                            last_regrets_ptr + cumul_offs,
                            mask=mask,
                            other=0.0,
                        )
                        observed = prev_actor != current_player
                        last = tl.where(observed, r, last)
                        tl.store(last_regrets_ptr + cumul_offs, last, mask=mask)
                        policy_c = c + prediction_scale * last
                    tl.store(pos_out_ptr + cumul_offs, tl.maximum(policy_c, 0.0), mask=mask)


def fused_unblocked_regret_dcfr_update_with_tensors_(
    target: torch.Tensor,
    stats: torch.Tensor,
    allowed_mask: torch.Tensor | None,
    values_achieved: torch.Tensor,
    values_expected: torch.Tensor,
    to_act: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    prev_actor: torch.Tensor,
    cumulative_regrets: torch.Tensor,
    t_alpha_num: torch.Tensor,
    t_beta_num: torch.Tensor,
    t_alpha_den: torch.Tensor,
    t_beta_den: torch.Tensor,
    apply_dcfr: bool,
    cfr_plus: bool,
    max_children: int,
    positive_regrets_out: torch.Tensor | None = None,
    last_instantaneous_regrets: torch.Tensor | None = None,
    prediction_scale: torch.Tensor | None = None,
    current_player: torch.Tensor | None = None,
    block_h: int = 512,
) -> None:
    """Finalize parent opponent reach and update child cumulative regrets.

    Keeps each parent source-weight tile in registers while updating all child
    rows.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert target.is_contiguous() and target.dim() == 2
    assert target.shape[1] == _UNBLOCKED_NUM_HANDS
    assert stats.is_contiguous()
    assert stats.shape == (target.shape[0], 1 + _UNBLOCKED_NUM_CARDS)
    assert values_achieved.is_contiguous() and values_achieved.dim() == 3
    assert values_expected.is_contiguous() and values_expected.dim() == 3
    assert values_achieved.shape[1] == 2 and values_expected.shape[1] == 2
    assert to_act.is_contiguous() and to_act.shape == (target.shape[0],)
    assert child_offsets.is_contiguous() and child_offsets.shape == (target.shape[0],)
    assert child_count.is_contiguous() and child_count.shape == child_offsets.shape
    assert prev_actor.is_contiguous() and prev_actor.shape == (
        values_achieved.shape[0],
    )
    assert cumulative_regrets.is_contiguous()
    total, h = cumulative_regrets.shape
    top = target.shape[0]
    assert h == _UNBLOCKED_NUM_HANDS
    assert values_achieved.shape == (total, 2, h)
    assert values_expected.shape == (top, 2, h)
    if allowed_mask is not None:
        assert allowed_mask.is_contiguous() and allowed_mask.shape == target.shape
        allowed_ptr = allowed_mask
    else:
        allowed_ptr = target
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    write_pos = positive_regrets_out is not None
    pos_ptr = positive_regrets_out if write_pos else cumulative_regrets
    has_predictive = last_instantaneous_regrets is not None
    if has_predictive:
        assert last_instantaneous_regrets is not None
        assert last_instantaneous_regrets.is_contiguous()
        assert last_instantaneous_regrets.shape == cumulative_regrets.shape
        assert prediction_scale is not None and prediction_scale.dim() == 0
        assert current_player is not None and current_player.dim() == 0
        last_ptr = last_instantaneous_regrets
        pred_scale_ptr = prediction_scale
        current_player_ptr = current_player
    else:
        last_ptr = cumulative_regrets
        pred_scale_ptr = t_alpha_num
        current_player_ptr = t_alpha_num
    card_a, card_b = _get_combo_cards(target.device)
    grid = (top, triton.cdiv(h, block_h))
    _fused_unblocked_regret_dcfr_update_kernel[grid](
        target,
        stats,
        card_a,
        card_b,
        allowed_ptr,
        values_achieved,
        values_expected,
        to_act,
        child_offsets,
        child_count,
        prev_actor,
        cumulative_regrets,
        pos_ptr,
        last_ptr,
        pred_scale_ptr,
        current_player_ptr,
        t_alpha_num,
        t_beta_num,
        t_alpha_den,
        t_beta_den,
        h,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        APPLY_DCFR=apply_dcfr,
        CFR_PLUS=cfr_plus,
        WRITE_POS=write_pos,
        HAS_PREDICTIVE=has_predictive,
        HAS_ALLOWED=allowed_mask is not None,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 6c: compact multiway source-weight regret/DCFR update.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_compact_regret_dcfr_update_multiway_kernel(
        src_weights_ptr,  # [top, H]
        values_achieved_ptr,  # [total, P, H]
        values_expected_ptr,  # [top, P, H]
        to_act_ptr,  # [top]
        child_offsets_ptr,  # [top]
        child_count_ptr,  # [top]
        prev_actor_ptr,  # [total]
        cumul_ptr,  # [total, H]
        pos_out_ptr,  # [total, H]
        last_regrets_ptr,  # optional [total, H]
        prediction_scale_ptr,
        current_player_ptr,
        t_alpha_num_ptr,
        t_beta_num_ptr,
        t_alpha_den_ptr,
        t_beta_den_ptr,
        H,
        NUM_PLAYERS: tl.constexpr,
        APPLY_DCFR: tl.constexpr,
        CFR_PLUS: tl.constexpr,
        WRITE_POS: tl.constexpr,
        HAS_PREDICTIVE: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)

        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H

        actor = tl.load(to_act_ptr + p)
        expected = tl.load(
            values_expected_ptr + (p * NUM_PLAYERS + actor) * H + offs,
            mask=mask,
            other=0.0,
        )
        w = tl.load(src_weights_ptr + p * H + offs, mask=mask, other=0.0)

        if APPLY_DCFR:
            t_alpha_num = tl.load(t_alpha_num_ptr)
            t_beta_num = tl.load(t_beta_num_ptr)
            t_alpha_den = tl.load(t_alpha_den_ptr)
            t_beta_den = tl.load(t_beta_den_ptr)
        if HAS_PREDICTIVE:
            prediction_scale = tl.load(prediction_scale_ptr)
            current_player = tl.load(current_player_ptr)

        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                prev_actor = tl.load(prev_actor_ptr + child)
                achieved = tl.load(
                    values_achieved_ptr
                    + (child * NUM_PLAYERS + prev_actor) * H
                    + offs,
                    mask=mask,
                    other=0.0,
                )
                r = w * (achieved - expected)

                cumul_offs = child * H + offs
                c = tl.load(cumul_ptr + cumul_offs, mask=mask, other=0.0)
                if APPLY_DCFR:
                    positive = c > 0.0
                    num = tl.where(positive, t_alpha_num, t_beta_num)
                    den = tl.where(positive, t_alpha_den, t_beta_den)
                    c = c * num
                    c = c / den

                c = c + r
                if CFR_PLUS:
                    c = tl.maximum(c, 0.0)

                tl.store(cumul_ptr + cumul_offs, c, mask=mask)
                if WRITE_POS:
                    policy_c = c
                    if HAS_PREDICTIVE:
                        last = tl.load(
                            last_regrets_ptr + cumul_offs,
                            mask=mask,
                            other=0.0,
                        )
                        observed = prev_actor != current_player
                        last = tl.where(observed, r, last)
                        tl.store(last_regrets_ptr + cumul_offs, last, mask=mask)
                        policy_c = c + prediction_scale * last
                    tl.store(
                        pos_out_ptr + cumul_offs,
                        tl.maximum(policy_c, 0.0),
                        mask=mask,
                    )


def fused_compact_regret_dcfr_update_multiway_with_tensors_(
    src_weights: torch.Tensor,
    values_achieved: torch.Tensor,
    values_expected: torch.Tensor,
    to_act: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    prev_actor: torch.Tensor,
    cumulative_regrets: torch.Tensor,
    t_alpha_num: torch.Tensor,
    t_beta_num: torch.Tensor,
    t_alpha_den: torch.Tensor,
    t_beta_den: torch.Tensor,
    apply_dcfr: bool,
    cfr_plus: bool,
    max_children: int,
    positive_regrets_out: torch.Tensor | None = None,
    last_instantaneous_regrets: torch.Tensor | None = None,
    prediction_scale: torch.Tensor | None = None,
    current_player: torch.Tensor | None = None,
    block_h: int = 256,
) -> None:
    """Update compact multiway cumulative regrets from parent source weights."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert src_weights.is_contiguous() and src_weights.dim() == 2
    assert values_achieved.is_contiguous() and values_achieved.dim() == 3
    assert values_expected.is_contiguous() and values_expected.dim() == 3
    assert to_act.is_contiguous() and to_act.dim() == 1
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    assert prev_actor.is_contiguous() and prev_actor.dim() == 1
    assert cumulative_regrets.is_contiguous() and cumulative_regrets.dim() == 2
    total, h = cumulative_regrets.shape
    top, p, h_expected = values_expected.shape
    assert h_expected == h
    assert src_weights.shape == (top, h)
    assert values_achieved.shape == (total, p, h)
    assert to_act.shape == (top,)
    assert child_offsets.shape == (top,)
    assert child_count.shape == (top,)
    assert prev_actor.shape == (total,)
    write_pos = positive_regrets_out is not None
    pos_ptr = positive_regrets_out if write_pos else cumulative_regrets
    has_predictive = last_instantaneous_regrets is not None
    if has_predictive:
        assert last_instantaneous_regrets is not None
        assert last_instantaneous_regrets.is_contiguous()
        assert last_instantaneous_regrets.shape == cumulative_regrets.shape
        assert prediction_scale is not None and prediction_scale.dim() == 0
        assert current_player is not None and current_player.dim() == 0
        last_ptr = last_instantaneous_regrets
        pred_scale_ptr = prediction_scale
        current_player_ptr = current_player
    else:
        last_ptr = cumulative_regrets
        pred_scale_ptr = t_alpha_num
        current_player_ptr = t_alpha_num
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    grid = (top, triton.cdiv(h, block_h))
    _fused_compact_regret_dcfr_update_multiway_kernel[grid](
        src_weights,
        values_achieved,
        values_expected,
        to_act,
        child_offsets,
        child_count,
        prev_actor,
        cumulative_regrets,
        pos_ptr,
        last_ptr,
        pred_scale_ptr,
        current_player_ptr,
        t_alpha_num,
        t_beta_num,
        t_alpha_den,
        t_beta_den,
        h,
        NUM_PLAYERS=p,
        APPLY_DCFR=apply_dcfr,
        CFR_PLUS=cfr_plus,
        WRITE_POS=write_pos,
        HAS_PREDICTIVE=has_predictive,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 7: update_average_policy mixing (pre-renormalization).
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_average_policy_mix_kernel(
        self_reach_ptr,  # [total, 2, H]
        policy_probs_ptr,  # [total, H]
        policy_probs_avg_ptr,  # [total, H] in/out
        avg_num_ptr,  # [total, H] in/out
        avg_den_ptr,  # [total, H] in/out
        to_act_ptr,  # [total] int64
        parent_index_ptr,  # [total] int64
        new_scalar_ptr,
        bottom,  # first child row to update
        total,
        H,
        EPS,
        WRITE_AVG: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0)
        if c >= total:
            return
        if c < bottom:
            if WRITE_AVG:
                for start in tl.range(0, H, BLOCK_H):
                    offs = start + tl.arange(0, BLOCK_H)
                    mask = offs < H
                    tl.store(policy_probs_avg_ptr + c * H + offs, 0.0, mask=mask)
            return
        parent = tl.load(parent_index_ptr + c)
        actor = tl.load(to_act_ptr + parent)

        new_scalar = tl.load(new_scalar_ptr)

        reach_row = self_reach_ptr + (parent * 2 + actor) * H
        policy_row = policy_probs_ptr + c * H
        avg_row = policy_probs_avg_ptr + c * H
        num_row = avg_num_ptr + c * H
        den_row = avg_den_ptr + c * H

        for start in tl.range(0, H, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            mask = offs < H
            reach_n = tl.load(reach_row + offs, mask=mask, other=0.0) * new_scalar
            cur = tl.load(policy_row + offs, mask=mask, other=0.0)
            num_old = tl.load(num_row + offs, mask=mask, other=0.0)
            den_old = tl.load(den_row + offs, mask=mask, other=0.0)

            num = num_old + reach_n * cur
            den = den_old + reach_n
            tl.store(num_row + offs, num, mask=mask)
            tl.store(den_row + offs, den, mask=mask)
            if WRITE_AVG:
                out = tl.where(den > EPS, num / tl.maximum(den, EPS), cur)
                tl.store(avg_row + offs, out, mask=mask)


def fused_average_policy_mix_(
    policy_probs_avg: torch.Tensor,  # [total, H] in/out
    average_policy_numerator: torch.Tensor,  # [total, H] in/out
    average_policy_denominator: torch.Tensor,  # [total, H] in/out
    policy_probs: torch.Tensor,  # [total, H]
    self_reach: torch.Tensor,  # [total, 2, H]
    to_act: torch.Tensor,  # [total]
    parent_index: torch.Tensor,  # [total]
    new: float,
    bottom: int,
    eps: float = 1e-5,
    block_h: int = 512,
    write_policy: bool = True,
) -> None:
    """Fused true-CFR average-policy accumulation (pre-renormalization).

    For each child row ``c in [bottom, total)`` and hand ``h``, replicates the
    PyTorch sequence::

        reach_actor = self_reach[parent, to_act[parent], h] * new
        average_policy_numerator[c, h] += reach_actor * policy_probs[c, h]
        average_policy_denominator[c, h] += reach_actor
        policy_probs_avg[c, h] = where(
            average_policy_denominator[c, h] > eps,
            average_policy_numerator[c, h] / average_policy_denominator[c, h],
            policy_probs[c, h],
        )

    Collapses gather + multiply + two accumulator writes + divide into one
    kernel. Caller is
    responsible for running the subsequent per-parent renormalization
    (``_pull_back_sum`` + ``_fan_out`` + divide) and zeroing the root slice.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy_probs_avg.is_contiguous() and policy_probs_avg.dim() == 2
    assert (
        average_policy_numerator.is_contiguous() and average_policy_numerator.dim() == 2
    )
    assert (
        average_policy_denominator.is_contiguous()
        and average_policy_denominator.dim() == 2
    )
    assert policy_probs.is_contiguous() and policy_probs.dim() == 2
    assert self_reach.is_contiguous() and self_reach.dim() == 3
    assert self_reach.shape[1] == 2
    total, h = policy_probs_avg.shape
    assert average_policy_numerator.shape == (total, h)
    assert average_policy_denominator.shape == (total, h)
    assert policy_probs.shape == (total, h)
    assert self_reach.shape == (total, 2, h)
    assert to_act.shape == (total,)
    assert parent_index.shape == (total,)
    dev = policy_probs_avg.device
    dt = policy_probs_avg.dtype
    new_t = torch.tensor(float(new), dtype=dt, device=dev)
    fused_average_policy_mix_with_tensors_(
        policy_probs_avg,
        average_policy_numerator,
        average_policy_denominator,
        policy_probs,
        self_reach,
        to_act,
        parent_index,
        new_t,
        bottom=bottom,
        eps=eps,
        block_h=block_h,
        write_policy=write_policy,
    )


def fused_average_policy_mix_with_tensors_(
    policy_probs_avg: torch.Tensor,
    average_policy_numerator: torch.Tensor,
    average_policy_denominator: torch.Tensor,
    policy_probs: torch.Tensor,
    self_reach: torch.Tensor,
    to_act: torch.Tensor,
    parent_index: torch.Tensor,
    new: torch.Tensor,
    bottom: int,
    eps: float = 1e-5,
    block_h: int = 512,
    write_policy: bool = True,
) -> None:
    """Graph-capturable version: scalars come from pre-filled 0-D tensors."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    total, h = policy_probs_avg.shape
    grid = (total,)
    _fused_average_policy_mix_kernel[grid](
        self_reach,
        policy_probs,
        policy_probs_avg,
        average_policy_numerator,
        average_policy_denominator,
        to_act,
        parent_index,
        new,
        bottom,
        total,
        h,
        eps,
        WRITE_AVG=write_policy,
        BLOCK_H=block_h,
        num_warps=4,
    )


if triton is not None:

    @triton.jit
    def _fused_average_policy_mix_multiway_kernel(
        self_reach_ptr,  # [total, P, H]
        policy_probs_ptr,  # [total, H]
        policy_probs_avg_ptr,  # [total, H] in/out
        avg_num_ptr,  # [total, H] in/out
        avg_den_ptr,  # [total, H] in/out
        to_act_ptr,  # [total] int64
        parent_index_ptr,  # [total] int64
        new_scalar_ptr,
        bottom,
        total,
        H,
        NUM_PLAYERS: tl.constexpr,
        EPS,
        WRITE_AVG: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0)
        if c >= total:
            return
        if c < bottom:
            if WRITE_AVG:
                for start in tl.range(0, H, BLOCK_H):
                    offs = start + tl.arange(0, BLOCK_H)
                    mask = offs < H
                    tl.store(policy_probs_avg_ptr + c * H + offs, 0.0, mask=mask)
            return

        parent = tl.load(parent_index_ptr + c)
        actor = tl.load(to_act_ptr + parent)
        new_scalar = tl.load(new_scalar_ptr)
        reach_row = self_reach_ptr + (parent * NUM_PLAYERS + actor) * H
        policy_row = policy_probs_ptr + c * H
        avg_row = policy_probs_avg_ptr + c * H
        num_row = avg_num_ptr + c * H
        den_row = avg_den_ptr + c * H

        for start in tl.range(0, H, BLOCK_H):
            offs = start + tl.arange(0, BLOCK_H)
            mask = offs < H
            reach_n = tl.load(reach_row + offs, mask=mask, other=0.0) * new_scalar
            cur = tl.load(policy_row + offs, mask=mask, other=0.0)
            num_old = tl.load(num_row + offs, mask=mask, other=0.0)
            den_old = tl.load(den_row + offs, mask=mask, other=0.0)

            num = num_old + reach_n * cur
            den = den_old + reach_n
            tl.store(num_row + offs, num, mask=mask)
            tl.store(den_row + offs, den, mask=mask)
            if WRITE_AVG:
                out = tl.where(den > EPS, num / tl.maximum(den, EPS), cur)
                tl.store(avg_row + offs, out, mask=mask)


def fused_average_policy_mix_multiway_with_tensors_(
    policy_probs_avg: torch.Tensor,
    average_policy_numerator: torch.Tensor,
    average_policy_denominator: torch.Tensor,
    policy_probs: torch.Tensor,
    self_reach: torch.Tensor,
    to_act: torch.Tensor,
    parent_index: torch.Tensor,
    new: torch.Tensor,
    bottom: int,
    eps: float = 1e-5,
    block_h: int = 512,
    write_policy: bool = True,
) -> None:
    """Multiway true-CFR average-policy accumulation."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy_probs_avg.is_contiguous() and policy_probs_avg.dim() == 2
    assert average_policy_numerator.is_contiguous()
    assert average_policy_denominator.is_contiguous()
    assert policy_probs.is_contiguous() and policy_probs.dim() == 2
    assert self_reach.is_contiguous() and self_reach.dim() == 3
    assert to_act.is_contiguous() and parent_index.is_contiguous()
    total, h = policy_probs_avg.shape
    players = self_reach.shape[1]
    assert players >= 2
    assert average_policy_numerator.shape == (total, h)
    assert average_policy_denominator.shape == (total, h)
    assert policy_probs.shape == (total, h)
    assert self_reach.shape == (total, players, h)
    assert to_act.shape == (total,)
    assert parent_index.shape == (total,)
    _fused_average_policy_mix_multiway_kernel[(total,)](
        self_reach,
        policy_probs,
        policy_probs_avg,
        average_policy_numerator,
        average_policy_denominator,
        to_act,
        parent_index,
        new,
        bottom,
        total,
        h,
        NUM_PLAYERS=players,
        EPS=eps,
        WRITE_AVG=write_policy,
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
        target_ptr,  # [B, H]
        cardsum_ptr,  # [B, NUM_CARDS]
        S_ptr,  # [B]
        card_a_ptr,  # [H] int32
        card_b_ptr,  # [H] int32
        allowed_ptr,  # optional [B, H] bool mask
        out_ptr,  # [B, H]
        H,
        NUM_CARDS: tl.constexpr,
        HAS_ALLOWED: tl.constexpr,
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
        if HAS_ALLOWED:
            allowed = tl.load(allowed_ptr + b * H + offs, mask=mask, other=0).to(
                tl.int1
            )
            out = tl.where(allowed, out, 0.0)
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


def _preprocess_unblocked_stats_out(
    target: torch.Tensor,  # [B, H] contiguous
    stacked_out: torch.Tensor,  # [B, 53] contiguous
) -> None:
    """Write stacked ``(S, cardsum)`` stats into a caller-owned buffer."""
    assert target.is_contiguous() and target.dim() == 2
    assert target.shape[1] == _UNBLOCKED_NUM_HANDS
    assert stacked_out.is_contiguous()
    assert stacked_out.shape == (target.shape[0], 1 + _UNBLOCKED_NUM_CARDS)
    P = _get_card_projection(target.device).to(target.dtype)
    torch.mm(target, P, out=stacked_out)


if triton is not None:

    @triton.jit
    def _select_player_beliefs_kernel(
        beliefs_ptr,  # [total, 2, H]
        to_act_ptr,  # [top]
        out_ptr,  # [top, H]
        H,
        OPPONENT: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        hb = tl.program_id(1)
        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H
        actor = tl.load(to_act_ptr + row)
        player = 1 - actor if OPPONENT else actor
        vals = tl.load(
            beliefs_ptr + (row * 2 + player) * H + offs,
            mask=mask,
            other=0.0,
        )
        tl.store(out_ptr + row * H + offs, vals, mask=mask)


def select_actor_beliefs_triton(
    beliefs: torch.Tensor,
    to_act: torch.Tensor,
    top: int,
    block_h: int = 1024,
) -> torch.Tensor:
    """Materialize ``beliefs[row, to_act[row], hand]`` for parent rows."""
    out = torch.empty(
        (top, beliefs.shape[-1]), device=beliefs.device, dtype=beliefs.dtype
    )
    select_actor_beliefs_triton_out_(beliefs, to_act, top, out, block_h=block_h)
    return out


def select_actor_beliefs_triton_out_(
    beliefs: torch.Tensor,
    to_act: torch.Tensor,
    top: int,
    out: torch.Tensor,
    block_h: int = 1024,
) -> None:
    """Write ``beliefs[row, to_act[row], hand]`` for parent rows into ``out``."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert beliefs.is_contiguous() and beliefs.dim() == 3 and beliefs.shape[1] == 2
    assert to_act.is_contiguous()
    h = beliefs.shape[-1]
    assert out.is_contiguous() and out.shape == (top, h)
    _select_player_beliefs_kernel[(top, triton.cdiv(h, block_h))](
        beliefs,
        to_act,
        out,
        h,
        OPPONENT=False,
        BLOCK_H=block_h,
        num_warps=4,
    )


def select_opponent_beliefs_triton_out_(
    beliefs: torch.Tensor,
    to_act: torch.Tensor,
    top: int,
    out: torch.Tensor,
    block_h: int = 2048,
) -> None:
    """Write ``beliefs[row, 1 - to_act[row], hand]`` for parent rows into ``out``."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert beliefs.is_contiguous() and beliefs.dim() == 3 and beliefs.shape[1] == 2
    assert to_act.is_contiguous()
    h = beliefs.shape[-1]
    assert out.is_contiguous() and out.shape == (top, h)
    _select_player_beliefs_kernel[(top, triton.cdiv(h, block_h))](
        beliefs,
        to_act,
        out,
        h,
        OPPONENT=True,
        BLOCK_H=block_h,
        num_warps=4,
    )


if triton is not None:

    @triton.jit(do_not_specialize=["bottom", "num_children"])
    def _marginal_policy_kernel(
        actor_beliefs_ptr,  # [top, H]
        policy_ptr,  # [total, H]
        parent_index_ptr,  # [num_children]
        out_ptr,  # [num_children, H]
        bottom,
        num_children,
        H,
        BLOCK_H: tl.constexpr,
    ):
        child_rel = tl.program_id(0)
        hb = tl.program_id(1)
        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = (child_rel < num_children) & (offs < H)
        parent = tl.load(parent_index_ptr + child_rel, mask=child_rel < num_children)
        belief = tl.load(
            actor_beliefs_ptr + parent * H + offs,
            mask=mask,
            other=0.0,
        )
        pol = tl.load(
            policy_ptr + (bottom + child_rel) * H + offs,
            mask=mask,
            other=0.0,
        )
        tl.store(out_ptr + child_rel * H + offs, belief * pol, mask=mask)


def marginal_policy_triton(
    actor_beliefs: torch.Tensor,
    policy: torch.Tensor,
    parent_index_bottom: torch.Tensor,
    bottom: int,
    block_h: int = 512,
) -> torch.Tensor:
    """Materialize ``actor_beliefs[parent_index_bottom] * policy[bottom:]``."""
    out = torch.empty(
        (parent_index_bottom.numel(), actor_beliefs.shape[-1]),
        device=policy.device,
        dtype=policy.dtype,
    )
    marginal_policy_triton_out_(
        actor_beliefs,
        policy,
        parent_index_bottom,
        bottom,
        out,
        block_h=block_h,
    )
    return out


def marginal_policy_triton_out_(
    actor_beliefs: torch.Tensor,
    policy: torch.Tensor,
    parent_index_bottom: torch.Tensor,
    bottom: int,
    out: torch.Tensor,
    block_h: int = 512,
) -> None:
    """Write ``actor_beliefs[parent_index_bottom] * policy[bottom:]`` into ``out``."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert actor_beliefs.is_contiguous() and actor_beliefs.dim() == 2
    assert policy.is_contiguous() and policy.dim() == 2
    assert parent_index_bottom.is_contiguous()
    num_children = parent_index_bottom.numel()
    h = actor_beliefs.shape[-1]
    assert out.is_contiguous() and out.shape == (num_children, h)
    _marginal_policy_kernel[(num_children, triton.cdiv(h, block_h))](
        actor_beliefs,
        policy,
        parent_index_bottom,
        out,
        bottom,
        num_children,
        h,
        BLOCK_H=block_h,
        num_warps=4,
    )


if triton is not None:

    @triton.jit
    def _select_actor_beliefs_and_marginal_policy_kernel(
        beliefs_ptr,  # [total, 2, H]
        to_act_ptr,  # [top]
        policy_ptr,  # [total, H]
        child_offsets_ptr,  # [top], absolute first child
        child_count_ptr,  # [top]
        actor_out_ptr,  # [top, H]
        marginal_out_ptr,  # [total - bottom, H]
        bottom,
        H,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        parent = tl.program_id(0)
        hb = tl.program_id(1)
        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H

        actor = tl.load(to_act_ptr + parent)
        belief = tl.load(
            beliefs_ptr + (parent * 2 + actor) * H + offs,
            mask=mask,
            other=0.0,
        )
        tl.store(actor_out_ptr + parent * H + offs, belief, mask=mask)

        first = tl.load(child_offsets_ptr + parent)
        count = tl.load(child_count_ptr + parent)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                child_rel = child - bottom
                pol = tl.load(
                    policy_ptr + child * H + offs,
                    mask=mask,
                    other=0.0,
                )
                tl.store(
                    marginal_out_ptr + child_rel * H + offs,
                    belief * pol,
                    mask=mask,
                )


def select_actor_beliefs_and_marginal_policy_triton_out_(
    beliefs: torch.Tensor,
    to_act: torch.Tensor,
    policy: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    bottom: int,
    actor_out: torch.Tensor,
    marginal_out: torch.Tensor,
    max_children: int,
    block_h: int = 512,
) -> None:
    """Select actor beliefs and write child marginal policies in one pass."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert beliefs.is_contiguous() and beliefs.dim() == 3 and beliefs.shape[1] == 2
    assert to_act.is_contiguous() and to_act.dim() == 1
    assert policy.is_contiguous() and policy.dim() == 2
    assert child_offsets.is_contiguous() and child_offsets.shape == to_act.shape
    assert child_count.is_contiguous() and child_count.shape == to_act.shape
    top = to_act.numel()
    h = beliefs.shape[-1]
    assert policy.shape == (beliefs.shape[0], h)
    assert actor_out.is_contiguous() and actor_out.shape == (top, h)
    assert marginal_out.is_contiguous() and marginal_out.dim() == 2
    assert marginal_out.shape[1] == h
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    _select_actor_beliefs_and_marginal_policy_kernel[
        (top, triton.cdiv(h, block_h))
    ](
        beliefs,
        to_act,
        policy,
        child_offsets,
        child_count,
        actor_out,
        marginal_out,
        bottom,
        h,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )


if triton is not None:

    @triton.jit
    def _select_actor_beliefs_and_marginal_policy_multiway_kernel(
        beliefs_ptr,  # [total, P, H]
        to_act_ptr,  # [top]
        policy_ptr,  # [total, H]
        child_offsets_ptr,  # [top], absolute first child
        child_count_ptr,  # [top]
        actor_out_ptr,  # [top, H]
        marginal_out_ptr,  # [total - bottom, H]
        bottom,
        H,
        NUM_PLAYERS: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        parent = tl.program_id(0)
        hb = tl.program_id(1)
        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H

        actor = tl.load(to_act_ptr + parent)
        belief = tl.load(
            beliefs_ptr + (parent * NUM_PLAYERS + actor) * H + offs,
            mask=mask,
            other=0.0,
        )
        tl.store(actor_out_ptr + parent * H + offs, belief, mask=mask)

        first = tl.load(child_offsets_ptr + parent)
        count = tl.load(child_count_ptr + parent)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                child_rel = child - bottom
                pol = tl.load(
                    policy_ptr + child * H + offs,
                    mask=mask,
                    other=0.0,
                )
                tl.store(
                    marginal_out_ptr + child_rel * H + offs,
                    belief * pol,
                    mask=mask,
                )


def select_actor_beliefs_and_marginal_policy_multiway_triton_out_(
    beliefs: torch.Tensor,
    to_act: torch.Tensor,
    policy: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    bottom: int,
    actor_out: torch.Tensor,
    marginal_out: torch.Tensor,
    max_children: int,
    block_h: int = 512,
) -> None:
    """Multiway actor-belief select plus child marginal-policy materialization."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert beliefs.is_contiguous() and beliefs.dim() == 3
    assert beliefs.shape[1] >= 2
    assert to_act.is_contiguous() and to_act.dim() == 1
    assert policy.is_contiguous() and policy.dim() == 2
    assert child_offsets.is_contiguous() and child_offsets.shape == to_act.shape
    assert child_count.is_contiguous() and child_count.shape == to_act.shape
    top = to_act.numel()
    total, p, h = beliefs.shape
    assert policy.shape == (total, h)
    assert actor_out.is_contiguous() and actor_out.shape == (top, h)
    assert marginal_out.is_contiguous() and marginal_out.dim() == 2
    assert marginal_out.shape[1] == h
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    _select_actor_beliefs_and_marginal_policy_multiway_kernel[
        (top, triton.cdiv(h, block_h))
    ](
        beliefs,
        to_act,
        policy,
        child_offsets,
        child_count,
        actor_out,
        marginal_out,
        bottom,
        h,
        NUM_PLAYERS=p,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )


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
    beliefs: torch.Tensor,  # [total, 2, H]
    to_act: torch.Tensor,  # [total] int64
    top: int,
    cached_stats: ParentBeliefUnblockedStats | None = None,
    allowed_mask: torch.Tensor | None = None,
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
        target = torch.empty((top, h), device=beliefs.device, dtype=beliefs.dtype)
        select_opponent_beliefs_triton_out_(beliefs, to_act, top, target)
        s, cardsum = _preprocess_unblocked_stats(target)

    card_a, card_b = _get_combo_cards(target.device)
    out = torch.empty_like(target)
    has_allowed = allowed_mask is not None
    allowed_ptr = allowed_mask if has_allowed else out
    if has_allowed:
        assert allowed_mask.is_contiguous() and allowed_mask.shape == target.shape
    _unblocked_mass_finalize_kernel[(top,)](
        target,
        cardsum,
        s,
        card_a,
        card_b,
        allowed_ptr,
        out,
        _UNBLOCKED_NUM_HANDS,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        HAS_ALLOWED=has_allowed,
        BLOCK_H=2048,
        num_warps=8,
    )
    return out


if triton is not None:

    @triton.jit
    def _multiway_regret_src_weights_at_parents_kernel(
        beliefs_ptr,  # [total, P, H]
        stats_s_ptr,  # [top * P]
        stats_cardsum_ptr,  # [top * P, 52]
        to_act_ptr,  # [top]
        has_folded_ptr,  # optional [total, P]
        allowed_ptr,  # optional [top, H]
        card_a_ptr,  # [H]
        card_b_ptr,  # [H]
        out_ptr,  # [top, H]
        H,
        NUM_PLAYERS: tl.constexpr,
        NUM_CARDS: tl.constexpr,
        HAS_FOLDED: tl.constexpr,
        HAS_ALLOWED: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        parent = tl.program_id(0)
        hb = tl.program_id(1)
        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H
        actor = tl.load(to_act_ptr + parent)
        ca = tl.load(card_a_ptr + offs, mask=mask, other=0)
        cb = tl.load(card_b_ptr + offs, mask=mask, other=0)

        prod = tl.full([BLOCK_H], 1.0, dtype=tl.float32)
        for player in tl.static_range(0, BLOCK_P):
            if player < NUM_PLAYERS:
                include = player != actor
                if HAS_FOLDED:
                    folded = tl.load(has_folded_ptr + parent * NUM_PLAYERS + player).to(
                        tl.int1
                    )
                    include = include & (~folded)
                stats_row = parent * NUM_PLAYERS + player
                target = tl.load(
                    beliefs_ptr + stats_row * H + offs,
                    mask=mask,
                    other=0.0,
                )
                s = tl.load(stats_s_ptr + stats_row)
                csa = tl.load(
                    stats_cardsum_ptr + stats_row * NUM_CARDS + ca,
                    mask=mask,
                    other=0.0,
                )
                csb = tl.load(
                    stats_cardsum_ptr + stats_row * NUM_CARDS + cb,
                    mask=mask,
                    other=0.0,
                )
                mass = tl.maximum(s - csa - csb + target, 0.0)
                mass = tl.maximum(mass, 1.0e-12)
                prod *= tl.where(include, mass, 1.0)

        if HAS_ALLOWED:
            allowed = tl.load(allowed_ptr + parent * H + offs, mask=mask, other=0).to(
                tl.int1
            )
            prod = tl.where(allowed, prod, 0.0)
        tl.store(out_ptr + parent * H + offs, prod, mask=mask)


def multiway_regret_src_weights_at_parents_triton(
    beliefs: torch.Tensor,  # [total, P, H]
    to_act: torch.Tensor,  # [top]
    top: int,
    has_folded: torch.Tensor | None = None,  # [total, P]
    allowed_mask: torch.Tensor | None = None,  # [top, H]
    block_h: int = 512,
) -> torch.Tensor:
    """Compute multiway opponent reach products at parent rows.

    Matches the reference source-weight path in ``CFREvaluator``:
    blocker-project each non-acting live player's reach mass into the acting
    player's hand space, clamp each included player to ``1e-12``, multiply
    across players, and apply the parent allowed-hand mask.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert beliefs.is_contiguous() and beliefs.dim() == 3
    total, p, h = beliefs.shape
    assert p >= 2 and h == _UNBLOCKED_NUM_HANDS
    assert 0 <= top <= total
    assert to_act.is_contiguous() and to_act.dim() == 1 and to_act.numel() >= top

    flat = beliefs[:top].reshape(top * p, h).contiguous()
    stats_s, stats_cardsum = _preprocess_unblocked_stats(flat)
    out = torch.empty((top, h), device=beliefs.device, dtype=beliefs.dtype)
    has_folds = has_folded is not None
    if has_folds:
        assert has_folded is not None
        assert has_folded.is_contiguous() and has_folded.shape == (total, p)
        folded_ptr = has_folded
    else:
        folded_ptr = beliefs
    has_allowed = allowed_mask is not None
    if has_allowed:
        assert allowed_mask is not None
        assert allowed_mask.is_contiguous() and allowed_mask.shape == (top, h)
        allowed_ptr = allowed_mask
    else:
        allowed_ptr = out
    block_p = 1
    while block_p < p:
        block_p *= 2
    card_a, card_b = _get_combo_cards(beliefs.device)
    _multiway_regret_src_weights_at_parents_kernel[
        (top, triton.cdiv(h, block_h))
    ](
        beliefs,
        stats_s,
        stats_cardsum,
        to_act,
        folded_ptr,
        allowed_ptr,
        card_a,
        card_b,
        out,
        h,
        NUM_PLAYERS=p,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        HAS_FOLDED=has_folds,
        HAS_ALLOWED=has_allowed,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


def unblocked_mass_finalize_triton_out_(
    target: torch.Tensor,
    s: torch.Tensor,
    cardsum: torch.Tensor,
    out: torch.Tensor,
    allowed_mask: torch.Tensor | None = None,
) -> None:
    """Finalize ``unblocked_mass`` from precomputed row/card sums."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert target.is_contiguous() and target.dim() == 2
    assert target.shape[1] == _UNBLOCKED_NUM_HANDS
    assert s.is_contiguous() and s.shape == (target.shape[0],)
    assert cardsum.is_contiguous()
    assert cardsum.shape == (target.shape[0], _UNBLOCKED_NUM_CARDS)
    assert out.is_contiguous() and out.shape == target.shape
    card_a, card_b = _get_combo_cards(target.device)
    has_allowed = allowed_mask is not None
    allowed_ptr = allowed_mask if has_allowed else out
    if has_allowed:
        assert allowed_mask is not None
        assert allowed_mask.is_contiguous() and allowed_mask.shape == target.shape
    _unblocked_mass_finalize_kernel[(target.shape[0],)](
        target,
        cardsum,
        s,
        card_a,
        card_b,
        allowed_ptr,
        out,
        _UNBLOCKED_NUM_HANDS,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        HAS_ALLOWED=has_allowed,
        BLOCK_H=2048,
        num_warps=8,
    )


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
        out,
        h,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        HAS_ALLOWED=False,
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
        values_ptr,  # [total, H] (children contiguous per parent)
        child_offsets_ptr,  # [num_parents] — first child absolute index
        child_count_ptr,  # [num_parents]
        out_ptr,  # [num_children, H] (out_row = first + i - out_offset)
        out_offset,  # absolute row index of first child (typically bottom)
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
        tile = tl.load(ptrs, mask=mask_2d, other=0.0)  # [MC, BH]
        acc = tl.sum(tile, axis=0)  # [BH]

        out_ptrs = (
            out_ptr + (first + row_offs - out_offset)[:, None] * H + col_offs[None, :]
        )
        bcast = tl.broadcast_to(acc[None, :], (MAX_CHILDREN, BLOCK_H))
        tl.store(out_ptrs, bcast, mask=mask_2d)


if triton is not None:

    @triton.jit
    def _fused_parent_sum_divide_kernel(
        values_ptr,  # [total, H] absolute-row values to sum over siblings
        fallback_ptr,  # [num_children, H] child-aligned fallback
        child_offsets_ptr,  # [num_parents] absolute first-child row
        child_count_ptr,  # [num_parents]
        out_ptr,  # [num_children, H] child-aligned output
        out_offset,  # absolute row index corresponding to out row 0
        H,
        EPS,
        UNIFORM_COUNT_FALLBACK: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)

        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        row_offs = tl.arange(0, MAX_CHILDREN)
        col_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        row_mask = row_offs < count
        col_mask = col_offs < H
        mask_2d = row_mask[:, None] & col_mask[None, :]

        ptrs = values_ptr + (first + row_offs)[:, None] * H + col_offs[None, :]
        vals = tl.load(ptrs, mask=mask_2d, other=0.0)
        denom = tl.sum(vals, axis=0)
        use_div = denom > EPS
        denom_safe = tl.maximum(denom, EPS)
        divided = vals / denom_safe[None, :]

        child_rel = first + row_offs - out_offset
        if UNIFORM_COUNT_FALLBACK:
            fallback = 1.0 / count.to(tl.float32)
        else:
            f_ptrs = fallback_ptr + child_rel[:, None] * H + col_offs[None, :]
            fallback = tl.load(f_ptrs, mask=mask_2d, other=0.0)
        result = tl.where(use_div[None, :], divided, fallback)

        out_ptrs = out_ptr + child_rel[:, None] * H + col_offs[None, :]
        tl.store(out_ptrs, result, mask=mask_2d)


def fused_parent_sum_divide_(
    values: torch.Tensor,  # [total, H] absolute-row numerator
    fallback: torch.Tensor,  # [num_children, H] fallback for denom <= eps
    child_offsets: torch.Tensor,  # [num_parents] absolute child row
    child_count: torch.Tensor,  # [num_parents]
    out: torch.Tensor,  # [num_children, H]
    out_offset: int,
    max_children: int = 8,
    eps: float = 1e-8,
    block_h: int = 512,
    uniform_count_fallback: bool = False,
) -> None:
    """For each parent, sum its child rows in ``values`` and immediately write
    normalized child rows to ``out``.

    ``fallback`` and ``out`` are child-aligned starting at ``out_offset``;
    ``values`` is indexed by absolute tree row.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 2
    assert fallback.is_contiguous() and out.is_contiguous()
    assert fallback.shape == out.shape and fallback.dim() == 2
    assert child_offsets.is_contiguous() and child_offsets.dim() == 1
    assert child_count.is_contiguous() and child_count.shape == child_offsets.shape
    assert values.shape[1] == fallback.shape[1]

    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2

    num_parents = child_offsets.shape[0]
    h = values.shape[1]
    grid = (num_parents, triton.cdiv(h, block_h))
    _fused_parent_sum_divide_kernel[grid](
        values,
        fallback,
        child_offsets,
        child_count,
        out,
        out_offset,
        h,
        eps,
        UNIFORM_COUNT_FALLBACK=uniform_count_fallback,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 9b: CFR delta metric without dense pullback/scatter temporaries.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_cfr_delta_stats_kernel(
        policy_ptr,  # [total, H]
        old_policy_ptr,  # [total, H]
        self_reach_ptr,  # [total, 2, H]
        child_offsets_ptr,  # [num_parents]
        child_count_ptr,  # [num_parents]
        out_ptr,  # [2], numerator sum and reachable-node count
        H,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        parent = tl.program_id(0)

        first = tl.load(child_offsets_ptr + parent)
        count = tl.load(child_count_ptr + parent)

        child_offsets = tl.arange(0, MAX_CHILDREN)
        hand_offsets = tl.arange(0, BLOCK_H)
        hand_mask = hand_offsets < H
        child_mask = child_offsets < count

        child_rows = first + child_offsets
        ptrs = policy_ptr + child_rows[:, None] * H + hand_offsets[None, :]
        old_ptrs = old_policy_ptr + child_rows[:, None] * H + hand_offsets[None, :]
        mask = child_mask[:, None] & hand_mask[None, :]
        policy = tl.load(ptrs, mask=mask, other=0.0)
        old_policy = tl.load(old_ptrs, mask=mask, other=0.0)
        delta_by_hand = tl.sum(tl.abs(policy - old_policy), axis=0)

        reach0 = tl.load(
            self_reach_ptr + (parent * 2) * H + hand_offsets,
            mask=hand_mask,
            other=0.0,
        )
        reach1 = tl.load(
            self_reach_ptr + (parent * 2 + 1) * H + hand_offsets,
            mask=hand_mask,
            other=0.0,
        )
        reachable = (reach0 > 0.0) | (reach1 > 0.0)

        reachable_count = tl.sum(tl.where(reachable, 1.0, 0.0), axis=0)
        delta_sum = tl.sum(tl.where(reachable, delta_by_hand, 0.0), axis=0)
        has_reachable = reachable_count > 0.0
        node_delta = tl.where(has_reachable, delta_sum / reachable_count, 0.0)
        node_count = tl.where(has_reachable, 1.0, 0.0)

        tl.atomic_add(out_ptr, node_delta, sem="relaxed")
        tl.atomic_add(out_ptr + 1, node_count, sem="relaxed")


def fused_cfr_delta_stats(
    policy: torch.Tensor,
    old_policy: torch.Tensor,
    self_reach: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    max_children: int,
    block_h: int = 2048,
) -> torch.Tensor:
    """Return ``[sum_node_delta, reachable_node_count]`` for CFR delta stats.

    This matches ``CFREvaluator._record_stats`` for sparse child-contiguous
    trees, but computes directly from child rows. It avoids materializing
    ``_pull_back(policy)``, ``_pull_back(old_policy)``, child deltas, or a
    parent-by-hand scatter buffer.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    if policy.device.type != "cuda":
        raise ValueError("fused_cfr_delta_stats requires CUDA tensors.")
    assert policy.is_contiguous() and old_policy.is_contiguous()
    assert self_reach.is_contiguous() and self_reach.dim() == 3
    assert child_offsets.is_contiguous() and child_offsets.dim() == 1
    assert child_count.is_contiguous() and child_count.shape == child_offsets.shape
    assert policy.shape == old_policy.shape
    assert self_reach.shape[0] >= child_offsets.shape[0]
    assert self_reach.shape[1] == 2
    assert self_reach.shape[2] == policy.shape[1]

    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2

    out = torch.empty((2,), device=policy.device, dtype=torch.float32)
    out.zero_()
    h = policy.shape[1]
    if block_h < h:
        raise ValueError(f"block_h={block_h} must cover all {h} hands.")
    _fused_cfr_delta_stats_kernel[(child_offsets.shape[0],)](
        policy,
        old_policy,
        self_reach,
        child_offsets,
        child_count,
        out,
        h,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=8,
    )
    return out


# ---------------------------------------------------------------------------
# Kernel 10b: average-policy renormalization plus reach propagation for one depth.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _policy_renorm_reach_depth_kernel(
        policy_ptr,  # [total, H] in/out
        reach_ptr,  # [total, 2, H] in/out
        allowed_mask_ptr,  # [total, H] bool
        child_offsets_ptr,  # [num_parents]
        child_count_ptr,  # [num_parents]
        prev_actor_ptr,  # [total]
        parent_base,
        H,
        EPS,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)
        parent = parent_base + p
        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H

        denom = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                denom += tl.load(policy_ptr + child * H + offs, mask=mask, other=0.0)
        use_div = denom > EPS
        denom_safe = tl.maximum(denom, EPS)

        parent0 = tl.load(reach_ptr + (parent * 2 + 0) * H + offs, mask=mask, other=0.0)
        parent1 = tl.load(reach_ptr + (parent * 2 + 1) * H + offs, mask=mask, other=0.0)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                raw = tl.load(policy_ptr + child * H + offs, mask=mask, other=0.0)
                pol = tl.where(use_div, raw / denom_safe, raw)
                tl.store(policy_ptr + child * H + offs, pol, mask=mask)

                prev_actor = tl.load(prev_actor_ptr + child)
                allowed = tl.load(
                    allowed_mask_ptr + child * H + offs, mask=mask, other=0
                ).to(tl.int1)
                r0 = tl.where(prev_actor == 0, parent0 * pol, parent0)
                r1 = tl.where(prev_actor == 1, parent1 * pol, parent1)
                r0 = tl.where(allowed, r0, 0.0)
                r1 = tl.where(allowed, r1, 0.0)
                tl.store(reach_ptr + (child * 2 + 0) * H + offs, r0, mask=mask)
                tl.store(reach_ptr + (child * 2 + 1) * H + offs, r1, mask=mask)


def fused_policy_renorm_reach_depth_(
    policy: torch.Tensor,
    reach: torch.Tensor,
    allowed_mask: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    prev_actor: torch.Tensor,
    parent_base: int,
    max_children: int,
    eps: float = 1e-5,
    block_h: int = 1024,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy.is_contiguous() and policy.dim() == 2
    assert reach.is_contiguous() and reach.shape == (
        policy.shape[0],
        2,
        policy.shape[1],
    )
    assert allowed_mask.is_contiguous() and allowed_mask.shape == policy.shape
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    assert prev_actor.is_contiguous()
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    h = policy.shape[1]
    grid = (child_offsets.shape[0], triton.cdiv(h, block_h))
    _policy_renorm_reach_depth_kernel[grid](
        policy,
        reach,
        allowed_mask,
        child_offsets,
        child_count,
        prev_actor,
        parent_base,
        h,
        eps,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )


if triton is not None:

    @triton.jit
    def _policy_renorm_reach_depth_multiway_kernel(
        policy_ptr,  # [total, H] in/out
        reach_ptr,  # [total, P, H] in/out
        allowed_mask_ptr,  # [total, H] bool
        child_offsets_ptr,  # [num_parents]
        child_count_ptr,  # [num_parents]
        prev_actor_ptr,  # [total]
        parent_base,
        H,
        NUM_PLAYERS: tl.constexpr,
        EPS,
        UPDATE_REACH: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)
        parent = parent_base + p
        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs < H
        denom = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                denom += tl.load(policy_ptr + child * H + offs, mask=mask_h, other=0.0)
        use_div = denom > EPS
        denom_safe = tl.maximum(denom, EPS)

        players = tl.arange(0, BLOCK_P)
        player_mask = players < NUM_PLAYERS
        value_mask = player_mask[:, None] & mask_h[None, :]
        parent_reach = tl.load(
            reach_ptr + (parent * NUM_PLAYERS + players[:, None]) * H + offs[None, :],
            mask=value_mask,
            other=0.0,
        )

        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                raw = tl.load(policy_ptr + child * H + offs, mask=mask_h, other=0.0)
                pol = tl.where(use_div, raw / denom_safe, raw)
                tl.store(policy_ptr + child * H + offs, pol, mask=mask_h)

                if UPDATE_REACH:
                    prev_actor = tl.load(prev_actor_ptr + child)
                    allowed = tl.load(
                        allowed_mask_ptr + child * H + offs,
                        mask=mask_h,
                        other=0,
                    ).to(tl.int1)
                    child_reach = tl.where(
                        players[:, None] == prev_actor,
                        parent_reach * pol[None, :],
                        parent_reach,
                    )
                    child_reach = tl.where(allowed[None, :], child_reach, 0.0)
                    tl.store(
                        reach_ptr
                        + (child * NUM_PLAYERS + players[:, None]) * H
                        + offs[None, :],
                        child_reach,
                        mask=value_mask,
                    )


def fused_policy_renorm_reach_depth_multiway_(
    policy: torch.Tensor,
    reach: torch.Tensor,
    allowed_mask: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    prev_actor: torch.Tensor,
    parent_base: int,
    max_children: int,
    update_reach: bool,
    eps: float = 1e-5,
    block_h: int = 512,
) -> None:
    """Multiway average-policy sibling renorm with optional reach propagation."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy.is_contiguous() and policy.dim() == 2
    assert reach.is_contiguous() and reach.dim() == 3
    total, h = policy.shape
    players = reach.shape[1]
    assert players >= 2 and reach.shape == (total, players, h)
    assert allowed_mask.is_contiguous() and allowed_mask.shape == policy.shape
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    assert prev_actor.is_contiguous()
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    block_p = 1
    while block_p < players:
        block_p *= 2
    _policy_renorm_reach_depth_multiway_kernel[
        (child_offsets.shape[0], triton.cdiv(h, block_h))
    ](
        policy,
        reach,
        allowed_mask,
        child_offsets,
        child_count,
        prev_actor,
        parent_base,
        h,
        NUM_PLAYERS=players,
        EPS=eps,
        UPDATE_REACH=update_reach,
        MAX_CHILDREN=mc_pow2,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        num_warps=4,
    )


if triton is not None:

    @triton.jit
    def _policy_reach_beliefs_depth_preflop_multiway_kernel(
        policy_ptr,  # [total, H] in/out
        reach_ptr,  # [total, P, H] in/out
        beliefs_ptr,  # [total, P, H] in/out
        allowed_mask_ptr,  # [total, H] bool
        allowed_prob_ptr,  # [total, H]
        root_index_ptr,  # [total]
        child_offsets_ptr,  # [num_parents]
        child_count_ptr,  # [num_parents]
        prev_actor_ptr,  # [total]
        parent_base,
        H,
        NUM_PLAYERS: tl.constexpr,
        EPS,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        parent = parent_base + p
        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        hands = tl.arange(0, BLOCK_H)
        hand_mask = hands < H
        players = tl.arange(0, BLOCK_P)
        player_mask = players < NUM_PLAYERS
        value_mask = player_mask[:, None] & hand_mask[None, :]

        denom = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                denom += tl.load(
                    policy_ptr + child * H + hands,
                    mask=hand_mask,
                    other=0.0,
                )
        use_div = denom > EPS
        denom_safe = tl.maximum(denom, EPS)

        parent_reach = tl.load(
            reach_ptr + (parent * NUM_PLAYERS + players[:, None]) * H + hands[None, :],
            mask=value_mask,
            other=0.0,
        )

        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                raw = tl.load(
                    policy_ptr + child * H + hands,
                    mask=hand_mask,
                    other=0.0,
                )
                pol = tl.where(use_div, raw / denom_safe, raw)
                tl.store(policy_ptr + child * H + hands, pol, mask=hand_mask)

                prev_actor = tl.load(prev_actor_ptr + child)
                allowed = tl.load(
                    allowed_mask_ptr + child * H + hands,
                    mask=hand_mask,
                    other=0,
                ).to(tl.int1)
                child_reach = tl.where(
                    players[:, None] == prev_actor,
                    parent_reach * pol[None, :],
                    parent_reach,
                )
                child_reach = tl.where(allowed[None, :], child_reach, 0.0)
                tl.store(
                    reach_ptr
                    + (child * NUM_PLAYERS + players[:, None]) * H
                    + hands[None, :],
                    child_reach,
                    mask=value_mask,
                )

                root = tl.load(root_index_ptr + child)
                root_belief = tl.load(
                    beliefs_ptr
                    + (root * NUM_PLAYERS + players[:, None]) * H
                    + hands[None, :],
                    mask=value_mask,
                    other=0.0,
                )
                unnorm = root_belief * child_reach
                belief_denom = tl.sum(unnorm, axis=1)
                fallback = tl.load(
                    allowed_prob_ptr + child * H + hands,
                    mask=hand_mask,
                    other=0.0,
                )
                out = tl.where(
                    belief_denom[:, None] > EPS,
                    unnorm / tl.maximum(belief_denom[:, None], EPS),
                    fallback[None, :],
                )
                tl.store(
                    beliefs_ptr
                    + (child * NUM_PLAYERS + players[:, None]) * H
                    + hands[None, :],
                    out,
                    mask=value_mask,
                )


def fused_policy_reach_beliefs_depth_preflop_multiway_(
    policy: torch.Tensor,
    reach: torch.Tensor,
    beliefs: torch.Tensor,
    allowed_mask: torch.Tensor,
    allowed_prob: torch.Tensor,
    root_index: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    prev_actor: torch.Tensor,
    parent_base: int,
    max_children: int,
    eps: float = 1e-5,
    block_h: int = 256,
) -> None:
    """Sibling renorm + reach + compact belief propagation for one depth."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy.is_contiguous() and policy.dim() == 2
    assert reach.is_contiguous() and beliefs.is_contiguous()
    assert reach.shape == beliefs.shape
    total, h = policy.shape
    players = reach.shape[1]
    assert reach.shape == (total, players, h)
    assert allowed_mask.is_contiguous() and allowed_mask.shape == policy.shape
    assert allowed_prob.is_contiguous() and allowed_prob.shape == policy.shape
    assert root_index.is_contiguous() and root_index.shape == (total,)
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    assert prev_actor.is_contiguous()
    if h > block_h:
        raise ValueError(f"hand dim {h} exceeds block_h {block_h}")
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    block_p = 1
    while block_p < players:
        block_p *= 2
    _policy_reach_beliefs_depth_preflop_multiway_kernel[(child_offsets.shape[0],)](
        policy,
        reach,
        beliefs,
        allowed_mask,
        allowed_prob,
        root_index,
        child_offsets,
        child_count,
        prev_actor,
        parent_base,
        h,
        NUM_PLAYERS=players,
        EPS=eps,
        MAX_CHILDREN=mc_pow2,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        num_warps=8,
    )


if triton is not None:

    @triton.jit
    def _average_policy_reach_beliefs_depth_preflop_multiway_kernel(
        policy_avg_ptr,  # [total, H] out/current avg policy
        avg_num_ptr,  # [total, H] in/out
        avg_den_ptr,  # [total, H] in/out
        policy_ptr,  # [total, H] current policy
        current_reach_ptr,  # [total, P, H]
        avg_reach_ptr,  # [total, P, H] in/out
        beliefs_ptr,  # [total, P, H] in/out
        allowed_mask_ptr,  # [total, H] bool
        allowed_prob_ptr,  # [total, H]
        root_index_ptr,  # [total]
        child_offsets_ptr,  # [num_parents]
        child_count_ptr,  # [num_parents]
        prev_actor_ptr,  # [total]
        to_act_ptr,  # [total]
        new_scalar_ptr,
        parent_base,
        H,
        NUM_PLAYERS: tl.constexpr,
        EPS,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        parent = parent_base + p
        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        hands = tl.arange(0, BLOCK_H)
        hand_mask = hands < H
        players = tl.arange(0, BLOCK_P)
        player_mask = players < NUM_PLAYERS
        value_mask = player_mask[:, None] & hand_mask[None, :]

        actor = tl.load(to_act_ptr + parent)
        new_scalar = tl.load(new_scalar_ptr)
        reach_n = (
            tl.load(
                current_reach_ptr + (parent * NUM_PLAYERS + actor) * H + hands,
                mask=hand_mask,
                other=0.0,
            )
            * new_scalar
        )

        raw0 = tl.zeros([BLOCK_H], dtype=tl.float32)
        raw1 = tl.zeros([BLOCK_H], dtype=tl.float32)
        raw2 = tl.zeros([BLOCK_H], dtype=tl.float32)
        raw3 = tl.zeros([BLOCK_H], dtype=tl.float32)
        raw4 = tl.zeros([BLOCK_H], dtype=tl.float32)
        raw5 = tl.zeros([BLOCK_H], dtype=tl.float32)
        raw6 = tl.zeros([BLOCK_H], dtype=tl.float32)
        raw7 = tl.zeros([BLOCK_H], dtype=tl.float32)
        denom_policy = tl.zeros([BLOCK_H], dtype=tl.float32)

        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                cur = tl.load(policy_ptr + child * H + hands, mask=hand_mask, other=0.0)
                num_old = tl.load(avg_num_ptr + child * H + hands, mask=hand_mask, other=0.0)
                den_old = tl.load(avg_den_ptr + child * H + hands, mask=hand_mask, other=0.0)
                num = num_old + reach_n * cur
                den = den_old + reach_n
                tl.store(avg_num_ptr + child * H + hands, num, mask=hand_mask)
                tl.store(avg_den_ptr + child * H + hands, den, mask=hand_mask)
                raw = tl.where(den > EPS, num / tl.maximum(den, EPS), cur)
                denom_policy += raw
                if i == 0:
                    raw0 = raw
                elif i == 1:
                    raw1 = raw
                elif i == 2:
                    raw2 = raw
                elif i == 3:
                    raw3 = raw
                elif i == 4:
                    raw4 = raw
                elif i == 5:
                    raw5 = raw
                elif i == 6:
                    raw6 = raw
                else:
                    raw7 = raw

        use_div = denom_policy > EPS
        denom_safe = tl.maximum(denom_policy, EPS)
        parent_reach = tl.load(
            avg_reach_ptr
            + (parent * NUM_PLAYERS + players[:, None]) * H
            + hands[None, :],
            mask=value_mask,
            other=0.0,
        )

        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                raw = raw0
                if i == 1:
                    raw = raw1
                elif i == 2:
                    raw = raw2
                elif i == 3:
                    raw = raw3
                elif i == 4:
                    raw = raw4
                elif i == 5:
                    raw = raw5
                elif i == 6:
                    raw = raw6
                elif i >= 7:
                    raw = raw7
                pol = tl.where(use_div, raw / denom_safe, raw)
                tl.store(policy_avg_ptr + child * H + hands, pol, mask=hand_mask)

                prev_actor = tl.load(prev_actor_ptr + child)
                allowed = tl.load(
                    allowed_mask_ptr + child * H + hands,
                    mask=hand_mask,
                    other=0,
                ).to(tl.int1)
                child_reach = tl.where(
                    players[:, None] == prev_actor,
                    parent_reach * pol[None, :],
                    parent_reach,
                )
                child_reach = tl.where(allowed[None, :], child_reach, 0.0)
                tl.store(
                    avg_reach_ptr
                    + (child * NUM_PLAYERS + players[:, None]) * H
                    + hands[None, :],
                    child_reach,
                    mask=value_mask,
                )

                root = tl.load(root_index_ptr + child)
                root_belief = tl.load(
                    beliefs_ptr
                    + (root * NUM_PLAYERS + players[:, None]) * H
                    + hands[None, :],
                    mask=value_mask,
                    other=0.0,
                )
                unnorm = root_belief * child_reach
                belief_denom = tl.sum(unnorm, axis=1)
                fallback = tl.load(
                    allowed_prob_ptr + child * H + hands,
                    mask=hand_mask,
                    other=0.0,
                )
                belief = tl.where(
                    belief_denom[:, None] > EPS,
                    unnorm / tl.maximum(belief_denom[:, None], EPS),
                    fallback[None, :],
                )
                tl.store(
                    beliefs_ptr
                    + (child * NUM_PLAYERS + players[:, None]) * H
                    + hands[None, :],
                    belief,
                    mask=value_mask,
                )


def fused_average_policy_reach_beliefs_depth_preflop_multiway_(
    policy_probs_avg: torch.Tensor,
    average_policy_numerator: torch.Tensor,
    average_policy_denominator: torch.Tensor,
    policy_probs: torch.Tensor,
    self_reach: torch.Tensor,
    self_reach_avg: torch.Tensor,
    beliefs_avg: torch.Tensor,
    allowed_mask: torch.Tensor,
    allowed_prob: torch.Tensor,
    root_index: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    prev_actor: torch.Tensor,
    to_act: torch.Tensor,
    new: torch.Tensor,
    parent_base: int,
    max_children: int,
    eps: float = 1e-5,
    block_h: int = 256,
) -> None:
    """Average-policy mix + renorm + avg reach/beliefs for one preflop depth."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy_probs_avg.is_contiguous() and policy_probs_avg.dim() == 2
    assert average_policy_numerator.is_contiguous()
    assert average_policy_denominator.is_contiguous()
    assert policy_probs.is_contiguous() and policy_probs.shape == policy_probs_avg.shape
    assert self_reach.is_contiguous() and self_reach_avg.is_contiguous()
    assert beliefs_avg.is_contiguous() and beliefs_avg.shape == self_reach_avg.shape
    total, h = policy_probs_avg.shape
    players = self_reach.shape[1]
    assert self_reach.shape == (total, players, h)
    assert self_reach_avg.shape == (total, players, h)
    assert average_policy_numerator.shape == (total, h)
    assert average_policy_denominator.shape == (total, h)
    assert allowed_mask.is_contiguous() and allowed_mask.shape == (total, h)
    assert allowed_prob.is_contiguous() and allowed_prob.shape == (total, h)
    assert root_index.is_contiguous() and root_index.shape == (total,)
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    assert prev_actor.is_contiguous() and prev_actor.shape == (total,)
    assert to_act.is_contiguous() and to_act.shape == (total,)
    assert new.dim() == 0
    if h > block_h:
        raise ValueError(f"hand dim {h} exceeds block_h {block_h}")
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    if mc_pow2 > 8:
        raise ValueError(
            "fused_average_policy_reach_beliefs_depth_preflop_multiway_ "
            f"supports at most 8 children, got {max_children}"
        )
    block_p = 1
    while block_p < players:
        block_p *= 2
    _average_policy_reach_beliefs_depth_preflop_multiway_kernel[
        (child_offsets.shape[0],)
    ](
        policy_probs_avg,
        average_policy_numerator,
        average_policy_denominator,
        policy_probs,
        self_reach,
        self_reach_avg,
        beliefs_avg,
        allowed_mask,
        allowed_prob,
        root_index,
        child_offsets,
        child_count,
        prev_actor,
        to_act,
        new,
        parent_base,
        h,
        NUM_PLAYERS=players,
        EPS=eps,
        MAX_CHILDREN=mc_pow2,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        num_warps=8,
    )


if triton is not None:

    @triton.jit
    def _preflop_multiway_beliefs_from_reach_kernel(
        beliefs_ptr,  # [total, P, H] in/out; root rows are source beliefs
        reach_ptr,  # [total, P, H]
        allowed_prob_ptr,  # [total, H]
        root_index_ptr,  # [total]
        start,
        total,
        H,
        NUM_PLAYERS: tl.constexpr,
        EPS,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0) + start
        p = tl.program_id(1)
        if c >= total:
            return

        root = tl.load(root_index_ptr + c)
        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        root_belief = tl.load(
            beliefs_ptr + (root * NUM_PLAYERS + p) * H + offs,
            mask=mask,
            other=0.0,
        )
        reach = tl.load(
            reach_ptr + (c * NUM_PLAYERS + p) * H + offs,
            mask=mask,
            other=0.0,
        )
        unnorm = root_belief * reach
        denom = tl.sum(unnorm, axis=0)
        fallback = tl.load(
            allowed_prob_ptr + c * H + offs,
            mask=mask,
            other=0.0,
        )
        out = tl.where(denom > EPS, unnorm / tl.maximum(denom, EPS), fallback)
        tl.store(
            beliefs_ptr + (c * NUM_PLAYERS + p) * H + offs,
            out,
            mask=mask,
        )


def fused_preflop_multiway_beliefs_from_reach_(
    beliefs: torch.Tensor,
    reach: torch.Tensor,
    allowed_prob: torch.Tensor,
    root_index: torch.Tensor,
    *,
    start: int,
    eps: float = 1e-5,
    block_h: int = 256,
) -> None:
    """Compact preflop belief propagation from root beliefs and reach weights.

    Root rows are treated as immutable source distributions. Callers should pass
    ``start=root_nodes`` so the kernel only rewrites descendant rows.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert beliefs.is_contiguous() and beliefs.dim() == 3
    assert reach.is_contiguous() and reach.shape == beliefs.shape
    assert allowed_prob.is_contiguous() and allowed_prob.shape == beliefs.shape[:1] + (
        beliefs.shape[2],
    )
    assert root_index.is_contiguous() and root_index.shape == (beliefs.shape[0],)
    total, players, h = beliefs.shape
    if total <= start:
        return
    if h > block_h:
        raise ValueError(f"hand dim {h} exceeds block_h {block_h}")
    grid = (total - start, players)
    _preflop_multiway_beliefs_from_reach_kernel[grid](
        beliefs,
        reach,
        allowed_prob,
        root_index,
        start,
        total,
        h,
        NUM_PLAYERS=players,
        EPS=eps,
        BLOCK_H=block_h,
        num_warps=8,
    )


if triton is not None:

    @triton.jit
    def _fused_preflop169_parent_sum_opp_kernel(
        values_ptr,  # [total, P, H] in/out
        leaf_values_ptr,  # optional [total, P, H]
        leaf_mask_ptr,  # optional [total]
        prev_actor_ptr,  # [total]
        has_folded_ptr,  # optional [total, P]
        policy_ptr,  # [total, H]
        marginal_action_ptr,  # [num_children]
        numer_unblocked_ptr,  # [num_children, H]
        denom_unblocked_ptr,  # [top, H]
        child_offsets_ptr,  # [num_parents] absolute first child
        child_count_ptr,  # [num_parents]
        parent_base,
        child_base,
        H,
        NUM_PLAYERS: tl.constexpr,
        EPS,
        HAS_FOLDED: tl.constexpr,
        HAS_LEAF_SOURCE: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        parent_rel = tl.program_id(0)
        hb = tl.program_id(1)
        row = parent_base + parent_rel
        first = tl.load(child_offsets_ptr + parent_rel)
        count = tl.load(child_count_ptr + parent_rel)
        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offs < H
        players = tl.arange(0, BLOCK_P)
        player_mask = players < NUM_PLAYERS
        value_mask = player_mask[:, None] & mask_h[None, :]

        if count == 0:
            if HAS_LEAF_SOURCE:
                is_leaf = tl.load(leaf_mask_ptr + row).to(tl.int1)
                src = tl.load(
                    leaf_values_ptr
                    + (row * NUM_PLAYERS + players[:, None]) * H
                    + offs[None, :],
                    mask=value_mask & is_leaf,
                    other=0.0,
                )
                tl.store(
                    values_ptr
                    + (row * NUM_PLAYERS + players[:, None]) * H
                    + offs[None, :],
                    src,
                    mask=value_mask,
                )
            return

        denom = tl.load(
            denom_unblocked_ptr + row * H + offs,
            mask=mask_h,
            other=0.0,
        )
        denom_safe = tl.maximum(denom, EPS)
        acc = tl.zeros([BLOCK_P, BLOCK_H], dtype=tl.float32)

        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                child_rel = child - child_base
                prev_actor = tl.load(prev_actor_ptr + child)
                hero_pol = tl.load(
                    policy_ptr + child * H + offs,
                    mask=mask_h,
                    other=0.0,
                )
                numer = tl.load(
                    numer_unblocked_ptr + child_rel * H + offs,
                    mask=mask_h,
                    other=0.0,
                )
                marginal_action = tl.load(marginal_action_ptr + child_rel)
                opp_pol = tl.where(denom > EPS, numer / denom_safe, 0.0)
                folded = tl.zeros([BLOCK_P], dtype=tl.int1)
                if HAS_FOLDED:
                    folded = tl.load(
                        has_folded_ptr + row * NUM_PLAYERS + players,
                        mask=player_mask,
                        other=0,
                    ).to(tl.int1)
                live_non_actor = (players[:, None] != prev_actor) & (~folded[:, None])
                pol = tl.where(
                    players[:, None] == prev_actor,
                    hero_pol[None, :],
                    tl.where(live_non_actor, opp_pol[None, :], marginal_action),
                )
                vals = tl.load(
                    values_ptr
                    + (child * NUM_PLAYERS + players[:, None]) * H
                    + offs[None, :],
                    mask=value_mask,
                    other=0.0,
                )
                if HAS_LEAF_SOURCE:
                    leaf_vals = tl.load(
                        leaf_values_ptr
                        + (child * NUM_PLAYERS + players[:, None]) * H
                        + offs[None, :],
                        mask=value_mask,
                        other=0.0,
                    )
                    child_is_leaf = tl.load(leaf_mask_ptr + child).to(tl.int1)
                    vals = tl.where(child_is_leaf, leaf_vals, vals)
                    tl.store(
                        values_ptr
                        + (child * NUM_PLAYERS + players[:, None]) * H
                        + offs[None, :],
                        leaf_vals,
                        mask=value_mask & child_is_leaf,
                    )
                acc += vals * pol

        tl.store(
            values_ptr + (row * NUM_PLAYERS + players[:, None]) * H + offs[None, :],
            acc,
            mask=value_mask,
        )


def fused_preflop169_parent_sum_opp_(
    values: torch.Tensor,
    prev_actor: torch.Tensor,
    policy: torch.Tensor,
    marginal_action_policy: torch.Tensor,
    numer_unblocked: torch.Tensor,
    denom_unblocked: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    parent_base: int,
    child_base: int,
    max_children: int,
    max_children_pow2: int | None = None,
    eps: float = 1.0e-8,
    leaf_values: torch.Tensor | None = None,
    leaf_mask: torch.Tensor | None = None,
    has_folded: torch.Tensor | None = None,
    block_h: int = 256,
) -> None:
    """Preflop-169 analogue of fused sparse inline-opponent EV backup."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 3
    assert policy.is_contiguous() and policy.dim() == 2
    assert marginal_action_policy.is_contiguous() and marginal_action_policy.dim() == 1
    assert numer_unblocked.is_contiguous() and numer_unblocked.dim() == 2
    assert denom_unblocked.is_contiguous() and denom_unblocked.dim() == 2
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    assert prev_actor.is_contiguous()
    total, players, h = values.shape
    assert policy.shape == (total, h)
    assert marginal_action_policy.shape == (numer_unblocked.shape[0],)
    assert numer_unblocked.shape[1] == h
    assert denom_unblocked.shape[1] == h
    has_leaf_source = leaf_values is not None
    if has_leaf_source:
        assert leaf_values is not None and leaf_mask is not None
        assert leaf_values.is_contiguous() and leaf_values.shape == values.shape
        assert leaf_mask.is_contiguous() and leaf_mask.shape == (total,)
        leaf_values_ptr = leaf_values
        leaf_mask_ptr = leaf_mask
    else:
        leaf_values_ptr = values
        leaf_mask_ptr = child_count
    has_folds = has_folded is not None
    if has_folds:
        assert has_folded is not None
        assert has_folded.is_contiguous() and has_folded.shape == (total, players)
        has_folded_ptr = has_folded
    else:
        has_folded_ptr = child_count
    if max_children_pow2 is None:
        mc_pow2 = 1
        while mc_pow2 < max_children:
            mc_pow2 *= 2
    else:
        mc_pow2 = max_children_pow2
    block_p = 1
    while block_p < players:
        block_p *= 2
    if child_offsets.numel() == 0:
        return
    _fused_preflop169_parent_sum_opp_kernel[
        (child_offsets.shape[0], triton.cdiv(h, block_h))
    ](
        values,
        leaf_values_ptr,
        leaf_mask_ptr,
        prev_actor,
        has_folded_ptr,
        policy,
        marginal_action_policy,
        numer_unblocked,
        denom_unblocked,
        child_offsets,
        child_count,
        parent_base,
        child_base,
        h,
        NUM_PLAYERS=players,
        EPS=eps,
        HAS_FOLDED=has_folds,
        HAS_LEAF_SOURCE=has_leaf_source,
        MAX_CHILDREN=mc_pow2,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        num_warps=4,
    )


def fused_sibling_sum(
    values: torch.Tensor,  # [total, H]
    child_offsets: torch.Tensor,  # [num_parents] — child_offsets[p] gives first child idx
    child_count: torch.Tensor,  # [num_parents]
    bottom: int,  # first child absolute index
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
        numer_target_ptr,  # [num_children, H] marginal_policy
        denom_target_ptr,  # [top, H] actor_beliefs (parent-aligned)
        numer_cardsum_ptr,  # [num_children, 52]
        denom_cardsum_ptr,  # [top, 52]
        numer_S_ptr,  # [num_children]
        denom_S_ptr,  # [top]
        parent_index_ptr,  # [num_children] — child c → parent in [0, top)
        card_a_ptr,
        card_b_ptr,
        out_ptr,  # [num_children, H]
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
        numer_target_ptr,  # [B, H] marginal_policy
        denom_target_ptr,  # [B, H] beliefs_dest
        numer_cardsum_ptr,  # [B, 52]
        denom_cardsum_ptr,  # [B, 52]
        numer_S_ptr,  # [B]
        denom_S_ptr,  # [B]
        card_a_ptr,
        card_b_ptr,
        out_ptr,  # [B, H]
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
    numer_target: torch.Tensor,  # [num_children, H] marginal_policy
    denom_target: torch.Tensor,  # [top, H] actor_beliefs (parent-aligned)
    parent_index: torch.Tensor,  # [num_children] int64 — child → parent idx
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
        n,
        d,
        ncs.contiguous(),
        dcs.contiguous(),
        Sn.contiguous(),
        Sd.contiguous(),
        card_a,
        card_b,
        out,
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
        hand_values_ptr,  # [M, ...]
        last_model_values_ptr,  # [M, ...]
        out_ptr,  # [M, ...]
        old_plus_new_over_new_ptr,  # (old + new) / new
        old_over_new_ptr,  # old / new
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        h = tl.load(hand_values_ptr + offs, mask=mask, other=0.0)
        last = tl.load(last_model_values_ptr + offs, mask=mask, other=0.0)
        old_plus_new_over_new = tl.load(old_plus_new_over_new_ptr)
        old_over_new = tl.load(old_over_new_ptr)
        out = old_plus_new_over_new * h - old_over_new * last
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
    fused_model_values_mix_with_tensors(
        hand_values, last_model_values, onon_t, oon_t, out, block_size=block_size
    )
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
        h_ptr,  # [M, 2, H] hand_values
        l_ptr,  # [M, 2, H] last_model_values
        b_ptr,  # [M, 2, H] beliefs
        out_ptr,  # [M, 2, H]
        onon_ptr,  # 0-D (old + new) / new
        oon_ptr,  # 0-D old / new
        M,
        H,
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
    hand_values: torch.Tensor,  # [M, 2, H]
    last_model_values: torch.Tensor,  # [M, 2, H]
    beliefs: torch.Tensor,  # [M, 2, H]
    old_plus_new_over_new: torch.Tensor,  # 0-D
    old_over_new: torch.Tensor,  # 0-D
    out: torch.Tensor,  # [M, 2, H]
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
        m,
        h,
        ENFORCE_ZS=enforce_zero_sum,
        BLOCK_H=block_h,
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Kernel 11c: model-values mix + zero-sum + indexed latest/last writeback.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_model_values_writeback_kernel(
        h_ptr,  # [M, 2, H] hand_values
        l_ptr,  # [M, 2, H] previous last_model_values
        b_ptr,  # [M, 2, H] beliefs
        idx_ptr,  # [M] absolute latest_values row
        latest_ptr,  # [T, 2, H]
        last_out_ptr,  # [M, 2, H]
        onon_ptr,  # 0-D (old + new) / new
        oon_ptr,  # 0-D old / new
        M,
        H,
        DO_MIX: tl.constexpr,
        ENFORCE_ZS: tl.constexpr,
        STORE_LAST: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        if m >= M:
            return

        row = tl.load(idx_ptr + m)
        onon = tl.load(onon_ptr)
        oon = tl.load(oon_ptr)

        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        h0_p = h_ptr + (m * 2 + 0) * H + offs
        h1_p = h_ptr + (m * 2 + 1) * H + offs
        l0_p = l_ptr + (m * 2 + 0) * H + offs
        l1_p = l_ptr + (m * 2 + 1) * H + offs
        b0_p = b_ptr + (m * 2 + 0) * H + offs
        b1_p = b_ptr + (m * 2 + 1) * H + offs

        latest0_p = latest_ptr + (row * 2 + 0) * H + offs
        latest1_p = latest_ptr + (row * 2 + 1) * H + offs
        last0_p = last_out_ptr + (m * 2 + 0) * H + offs
        last1_p = last_out_ptr + (m * 2 + 1) * H + offs

        h0 = tl.load(h0_p, mask=mask, other=0.0).to(tl.float32)
        h1 = tl.load(h1_p, mask=mask, other=0.0).to(tl.float32)

        v0 = h0
        v1 = h1
        if DO_MIX:
            l0 = tl.load(l0_p, mask=mask, other=0.0).to(tl.float32)
            l1 = tl.load(l1_p, mask=mask, other=0.0).to(tl.float32)
            v0 = h0 * onon - l0 * oon
            v1 = h1 * onon - l1 * oon

        s = tl.zeros((), dtype=tl.float32)
        if ENFORCE_ZS:
            b0 = tl.load(b0_p, mask=mask, other=0.0).to(tl.float32)
            b1 = tl.load(b1_p, mask=mask, other=0.0).to(tl.float32)
            s = 0.5 * (
                tl.sum(tl.where(mask, v0 * b0, 0.0))
                + tl.sum(tl.where(mask, v1 * b1, 0.0))
            )

        tl.store(latest0_p, v0 - s, mask=mask)
        tl.store(latest1_p, v1 - s, mask=mask)
        if STORE_LAST:
            tl.store(last0_p, h0, mask=mask)
            tl.store(last1_p, h1, mask=mask)


def fused_model_values_writeback_(
    hand_values: torch.Tensor,  # [M, 2, H]
    last_model_values: torch.Tensor,  # [M, 2, H]
    beliefs: torch.Tensor,  # [M, 2, H]
    model_indices: torch.Tensor,  # [M]
    latest_values: torch.Tensor,  # [T, 2, H]
    last_out: torch.Tensor,  # [M, 2, H]
    old_plus_new_over_new: torch.Tensor,  # 0-D
    old_over_new: torch.Tensor,  # 0-D
    do_mix: bool,
    enforce_zero_sum: bool,
    store_last: bool = True,
    block_h: int = 2048,
) -> None:
    """Write model leaf values directly into evaluator buffers.

    Combines the previous sequence ``to(fp32) -> optional mix/zero-sum ->
    index_copy_(latest_values, model_indices, ...) -> last_out.copy_(hand_values)``.
    ``last_model_values`` may alias ``hand_values`` when ``do_mix`` is false.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert hand_values.is_contiguous()
    assert last_model_values.is_contiguous()
    assert beliefs.is_contiguous()
    assert model_indices.is_contiguous()
    assert latest_values.is_contiguous()
    assert last_out.is_contiguous()
    assert (
        hand_values.shape == last_model_values.shape == beliefs.shape == last_out.shape
    )
    assert hand_values.dim() == 3 and hand_values.shape[1] == 2
    m, _, h = hand_values.shape
    assert model_indices.shape == (m,)
    assert h <= block_h, f"BLOCK_H={block_h} must cover H={h}"
    _fused_model_values_writeback_kernel[(m,)](
        hand_values,
        last_model_values,
        beliefs,
        model_indices,
        latest_values,
        last_out,
        old_plus_new_over_new,
        old_over_new,
        m,
        h,
        DO_MIX=do_mix,
        ENFORCE_ZS=enforce_zero_sum,
        STORE_LAST=store_last,
        BLOCK_H=block_h,
        num_warps=8 if do_mix else 4,
    )


if triton is not None:

    @triton.jit
    def _fused_model_values_writeback_multiway_kernel(
        h_ptr,  # [M, P, H] hand_values
        l_ptr,  # [M, P, H] previous last_model_values
        b_ptr,  # [M, P, H] beliefs
        idx_ptr,  # [M] absolute latest_values row
        latest_ptr,  # [T, P, H]
        last_out_ptr,  # [M, P, H]
        onon_ptr,  # 0-D (old + new) / new
        oon_ptr,  # 0-D old / new
        M,
        H,
        NUM_PLAYERS: tl.constexpr,
        DO_MIX: tl.constexpr,
        ENFORCE_ZS: tl.constexpr,
        STORE_LAST: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        if m >= M:
            return
        row = tl.load(idx_ptr + m)
        onon = tl.load(onon_ptr)
        oon = tl.load(oon_ptr)

        players = tl.arange(0, BLOCK_P)
        offs = tl.arange(0, BLOCK_H)
        player_mask = players < NUM_PLAYERS
        hand_mask = offs < H
        mask = player_mask[:, None] & hand_mask[None, :]
        model_ptrs = (m * NUM_PLAYERS + players[:, None]) * H + offs[None, :]
        latest_ptrs = (row * NUM_PLAYERS + players[:, None]) * H + offs[None, :]

        h_vals = tl.load(h_ptr + model_ptrs, mask=mask, other=0.0).to(tl.float32)
        vals = h_vals
        if DO_MIX:
            last_vals = tl.load(l_ptr + model_ptrs, mask=mask, other=0.0).to(tl.float32)
            vals = h_vals * onon - last_vals * oon

        correction = tl.zeros((), dtype=tl.float32)
        if ENFORCE_ZS:
            beliefs = tl.load(b_ptr + model_ptrs, mask=mask, other=0.0).to(tl.float32)
            correction = tl.sum(tl.where(mask, vals * beliefs, 0.0)) / NUM_PLAYERS

        tl.store(latest_ptr + latest_ptrs, vals - correction, mask=mask)
        if STORE_LAST:
            tl.store(last_out_ptr + model_ptrs, h_vals, mask=mask)


def fused_model_values_writeback_multiway_(
    hand_values: torch.Tensor,  # [M, P, H]
    last_model_values: torch.Tensor,  # [M, P, H]
    beliefs: torch.Tensor,  # [M, P, H]
    model_indices: torch.Tensor,  # [M]
    latest_values: torch.Tensor,  # [T, P, H]
    last_out: torch.Tensor,  # [M, P, H]
    old_plus_new_over_new: torch.Tensor,  # 0-D
    old_over_new: torch.Tensor,  # 0-D
    do_mix: bool,
    enforce_zero_sum: bool,
    store_last: bool = True,
    block_h: int = 2048,
) -> None:
    """Multiway model leaf writeback with optional CFR-AVG value mixing."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert hand_values.is_contiguous()
    assert last_model_values.is_contiguous()
    assert beliefs.is_contiguous()
    assert model_indices.is_contiguous()
    assert latest_values.is_contiguous()
    assert last_out.is_contiguous()
    assert (
        hand_values.shape == last_model_values.shape == beliefs.shape == last_out.shape
    )
    assert hand_values.dim() == 3 and hand_values.shape[1] >= 2
    m, players, h = hand_values.shape
    assert model_indices.shape == (m,)
    assert latest_values.shape[1:] == (players, h)
    assert h <= block_h, f"BLOCK_H={block_h} must cover H={h}"
    block_p = 1
    while block_p < players:
        block_p *= 2
    _fused_model_values_writeback_multiway_kernel[(m,)](
        hand_values,
        last_model_values,
        beliefs,
        model_indices,
        latest_values,
        last_out,
        old_plus_new_over_new,
        old_over_new,
        m,
        h,
        NUM_PLAYERS=players,
        DO_MIX=do_mix,
        ENFORCE_ZS=enforce_zero_sum,
        STORE_LAST=store_last,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        num_warps=8 if do_mix else 4,
    )


if triton is not None:

    @triton.jit
    def _fused_weighted_parent_sum_inline_opp_both_kernel(
        values_ptr,  # [total, 2, H] in/out
        leaf_values_ptr,  # optional [total, 2, H]
        leaf_mask_ptr,  # optional [total] bool
        prev_actor_ptr,  # [total]
        policy_hero_ptr,  # [total, H]
        actor_beliefs_ptr,  # [top, H]
        numer_s_ptr,  # [num_children]
        numer_cardsum_ptr,  # [num_children, 52]
        denom_s_ptr,  # [top]
        denom_cardsum_ptr,  # [top, 52]
        card_a_ptr,  # [H]
        card_b_ptr,  # [H]
        child_offsets_ptr,  # [num_parents] absolute first-child row
        child_count_ptr,  # [num_parents]
        parent_base,  # absolute row of first parent in this depth slice
        child_base,  # absolute row corresponding to numer row 0
        H,
        NUM_CARDS: tl.constexpr,
        EPS,
        HAS_LEAF_SOURCE: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)

        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        row = parent_base + p
        col_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        col_mask = col_offs < H

        if count == 0:
            if HAS_LEAF_SOURCE:
                is_leaf = tl.load(leaf_mask_ptr + row).to(tl.int1)
                src0 = tl.load(
                    leaf_values_ptr + (row * 2) * H + col_offs,
                    mask=col_mask & is_leaf,
                    other=0.0,
                )
                src1 = tl.load(
                    leaf_values_ptr + (row * 2 + 1) * H + col_offs,
                    mask=col_mask & is_leaf,
                    other=0.0,
                )
                tl.store(values_ptr + (row * 2) * H + col_offs, src0, mask=col_mask)
                tl.store(
                    values_ptr + (row * 2 + 1) * H + col_offs,
                    src1,
                    mask=col_mask,
                )
            return

        ca = tl.load(card_a_ptr + col_offs, mask=col_mask, other=0)
        cb = tl.load(card_b_ptr + col_offs, mask=col_mask, other=0)
        dt = tl.load(
            actor_beliefs_ptr + row * H + col_offs,
            mask=col_mask,
            other=0.0,
        )
        sd = tl.load(denom_s_ptr + row)
        dcsa = tl.load(denom_cardsum_ptr + row * NUM_CARDS + ca, mask=col_mask)
        dcsb = tl.load(denom_cardsum_ptr + row * NUM_CARDS + cb, mask=col_mask)

        acc0 = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                child_rel = child - child_base
                pa = tl.load(prev_actor_ptr + child)
                hero_pol = tl.load(
                    policy_hero_ptr + child * H + col_offs,
                    mask=col_mask,
                    other=0.0,
                )

                nt = dt * hero_pol
                sn = tl.load(numer_s_ptr + child_rel)
                ncsa = tl.load(
                    numer_cardsum_ptr + child_rel * NUM_CARDS + ca,
                    mask=col_mask,
                    other=0.0,
                )
                ncsb = tl.load(
                    numer_cardsum_ptr + child_rel * NUM_CARDS + cb,
                    mask=col_mask,
                    other=0.0,
                )
                numer = tl.maximum(sn - ncsa - ncsb + nt, 0.0)
                denom = tl.maximum(sd - dcsa - dcsb + dt, 0.0)
                opp_pol = tl.where(denom > EPS, numer / denom, 0.0)
                pol0 = tl.where(pa == 0, hero_pol, opp_pol)
                pol1 = tl.where(pa == 1, hero_pol, opp_pol)

                v0 = tl.load(
                    values_ptr + (child * 2) * H + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                v1 = tl.load(
                    values_ptr + (child * 2 + 1) * H + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                if HAS_LEAF_SOURCE:
                    leaf_v0 = tl.load(
                        leaf_values_ptr + (child * 2) * H + col_offs,
                        mask=col_mask,
                        other=0.0,
                    )
                    leaf_v1 = tl.load(
                        leaf_values_ptr + (child * 2 + 1) * H + col_offs,
                        mask=col_mask,
                        other=0.0,
                    )
                    child_is_leaf = tl.load(leaf_mask_ptr + child).to(tl.int1)
                    v0 = tl.where(child_is_leaf, leaf_v0, v0)
                    v1 = tl.where(child_is_leaf, leaf_v1, v1)
                    tl.store(
                        values_ptr + (child * 2) * H + col_offs,
                        leaf_v0,
                        mask=col_mask & child_is_leaf,
                    )
                    tl.store(
                        values_ptr + (child * 2 + 1) * H + col_offs,
                        leaf_v1,
                        mask=col_mask & child_is_leaf,
                    )
                acc0 += v0 * pol0
                acc1 += v1 * pol1

        tl.store(values_ptr + (row * 2) * H + col_offs, acc0, mask=col_mask)
        tl.store(values_ptr + (row * 2 + 1) * H + col_offs, acc1, mask=col_mask)


def fused_weighted_parent_sum_inline_opp_both(
    values: torch.Tensor,
    prev_actor: torch.Tensor,
    policy_hero: torch.Tensor,
    actor_beliefs: torch.Tensor,
    numer_s: torch.Tensor,
    numer_cardsum: torch.Tensor,
    denom_s: torch.Tensor,
    denom_cardsum: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    parent_base: int,
    child_base: int,
    max_children: int = 8,
    max_children_pow2: int | None = None,
    eps: float = 1e-5,
    leaf_values: torch.Tensor | None = None,
    leaf_mask: torch.Tensor | None = None,
    block_h: int = 512,
) -> None:
    """Two-player variant of inline-opponent EV backup.

    Computes the opponent-conditioned ratio once per child/hand block and uses
    it for both player accumulators, reducing duplicate ratio work.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 3 and values.shape[1] == 2
    assert policy_hero.is_contiguous() and actor_beliefs.is_contiguous()
    assert numer_s.is_contiguous() and numer_cardsum.is_contiguous()
    assert denom_s.is_contiguous() and denom_cardsum.is_contiguous()
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    total, two, h = values.shape
    top = actor_beliefs.shape[0]
    num_children = numer_s.shape[0]
    assert two == 2 and policy_hero.shape == (total, h)
    assert actor_beliefs.shape == (top, h)
    assert numer_cardsum.shape == (num_children, _UNBLOCKED_NUM_CARDS)
    assert denom_cardsum.shape == (top, _UNBLOCKED_NUM_CARDS)
    assert denom_s.shape == (top,)
    assert prev_actor.shape == (total,)
    has_leaf_source = leaf_values is not None
    if has_leaf_source:
        assert leaf_values is not None and leaf_mask is not None
        assert leaf_values.is_contiguous() and leaf_values.shape == values.shape
        assert leaf_mask.is_contiguous() and leaf_mask.shape == (total,)
        leaf_values_ptr = leaf_values
        leaf_mask_ptr = leaf_mask
    else:
        leaf_values_ptr = values
        leaf_mask_ptr = child_count

    if max_children_pow2 is None:
        mc_pow2 = 1
        while mc_pow2 < max_children:
            mc_pow2 *= 2
    else:
        mc_pow2 = max_children_pow2
    num_parents = child_offsets.shape[0]
    card_a, card_b = _get_combo_cards(values.device)

    grid = (num_parents, triton.cdiv(h, block_h))
    _fused_weighted_parent_sum_inline_opp_both_kernel[grid](
        values,
        leaf_values_ptr,
        leaf_mask_ptr,
        prev_actor,
        policy_hero,
        actor_beliefs,
        numer_s,
        numer_cardsum,
        denom_s,
        denom_cardsum,
        card_a,
        card_b,
        child_offsets,
        child_count,
        parent_base,
        child_base,
        h,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        EPS=eps,
        HAS_LEAF_SOURCE=has_leaf_source,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )


if triton is not None:

    @triton.jit
    def _fused_weighted_parent_sum_inline_opp_both_noleaf_kernel(
        values_ptr,  # [total, 2, H] in/out
        prev_actor_ptr,  # [total]
        policy_hero_ptr,  # [total, H]
        actor_beliefs_ptr,  # [top, H]
        numer_s_ptr,  # [num_children]
        numer_cardsum_ptr,  # [num_children, 52]
        denom_s_ptr,  # [top]
        denom_cardsum_ptr,  # [top, 52]
        card_a_ptr,  # [H]
        card_b_ptr,  # [H]
        child_offsets_ptr,  # [num_parents] absolute first-child row
        child_count_ptr,  # [num_parents]
        parent_base,  # absolute row of first parent in this depth slice
        child_base,  # absolute row corresponding to numer row 0
        H,
        NUM_CARDS: tl.constexpr,
        EPS,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)

        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        row = parent_base + p
        col_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        col_mask = col_offs < H
        ca = tl.load(card_a_ptr + col_offs, mask=col_mask, other=0)
        cb = tl.load(card_b_ptr + col_offs, mask=col_mask, other=0)
        dt = tl.load(
            actor_beliefs_ptr + row * H + col_offs,
            mask=col_mask,
            other=0.0,
        )
        sd = tl.load(denom_s_ptr + row)
        dcsa = tl.load(denom_cardsum_ptr + row * NUM_CARDS + ca, mask=col_mask)
        dcsb = tl.load(denom_cardsum_ptr + row * NUM_CARDS + cb, mask=col_mask)
        pa = tl.load(prev_actor_ptr + first)

        acc0 = tl.zeros([BLOCK_H], dtype=tl.float32)
        acc1 = tl.zeros([BLOCK_H], dtype=tl.float32)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                child_rel = child - child_base
                hero_pol = tl.load(
                    policy_hero_ptr + child * H + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                nt = dt * hero_pol
                sn = tl.load(numer_s_ptr + child_rel)
                ncsa = tl.load(
                    numer_cardsum_ptr + child_rel * NUM_CARDS + ca,
                    mask=col_mask,
                    other=0.0,
                )
                ncsb = tl.load(
                    numer_cardsum_ptr + child_rel * NUM_CARDS + cb,
                    mask=col_mask,
                    other=0.0,
                )
                numer = tl.maximum(sn - ncsa - ncsb + nt, 0.0)
                denom = tl.maximum(sd - dcsa - dcsb + dt, 0.0)
                opp_pol = tl.where(denom > EPS, numer / denom, 0.0)
                pol0 = tl.where(pa == 0, hero_pol, opp_pol)
                pol1 = tl.where(pa == 1, hero_pol, opp_pol)
                v0 = tl.load(
                    values_ptr + (child * 2) * H + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                v1 = tl.load(
                    values_ptr + (child * 2 + 1) * H + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                acc0 += v0 * pol0
                acc1 += v1 * pol1

        tl.store(values_ptr + (row * 2) * H + col_offs, acc0, mask=col_mask)
        tl.store(values_ptr + (row * 2 + 1) * H + col_offs, acc1, mask=col_mask)


def fused_weighted_parent_sum_inline_opp_both_noleaf(
    values: torch.Tensor,
    prev_actor: torch.Tensor,
    policy_hero: torch.Tensor,
    actor_beliefs: torch.Tensor,
    numer_s: torch.Tensor,
    numer_cardsum: torch.Tensor,
    denom_s: torch.Tensor,
    denom_cardsum: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    parent_base: int,
    child_base: int,
    max_children: int,
    eps: float = 1e-5,
    block_h: int = 512,
    num_warps: int = 8,
) -> None:
    """No-leaf-source hot variant of inline-opponent EV backup."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 3 and values.shape[1] == 2
    assert policy_hero.is_contiguous() and actor_beliefs.is_contiguous()
    assert numer_s.is_contiguous() and numer_cardsum.is_contiguous()
    assert denom_s.is_contiguous() and denom_cardsum.is_contiguous()
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    total, two, h = values.shape
    top = actor_beliefs.shape[0]
    num_children = numer_s.shape[0]
    assert two == 2 and policy_hero.shape == (total, h)
    assert actor_beliefs.shape == (top, h)
    assert numer_cardsum.shape == (num_children, _UNBLOCKED_NUM_CARDS)
    assert denom_cardsum.shape == (top, _UNBLOCKED_NUM_CARDS)
    assert denom_s.shape == (top,)
    assert prev_actor.shape == (total,)

    num_parents = child_offsets.shape[0]
    card_a, card_b = _get_combo_cards(values.device)
    grid = (num_parents, triton.cdiv(h, block_h))
    _fused_weighted_parent_sum_inline_opp_both_noleaf_kernel[grid](
        values,
        prev_actor,
        policy_hero,
        actor_beliefs,
        numer_s,
        numer_cardsum,
        denom_s,
        denom_cardsum,
        card_a,
        card_b,
        child_offsets,
        child_count,
        parent_base,
        child_base,
        h,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        EPS=eps,
        MAX_CHILDREN=max_children,
        BLOCK_H=block_h,
        num_warps=num_warps,
    )


if triton is not None:

    @triton.jit
    def _fused_weighted_parent_sum_inline_opp_multiway_kernel(
        values_ptr,  # [total, P, H] in/out
        leaf_values_ptr,  # optional [total, P, H]
        leaf_mask_ptr,  # optional [total] bool
        prev_actor_ptr,  # [total]
        policy_hero_ptr,  # [total, H]
        actor_beliefs_ptr,  # [top, H]
        numer_s_ptr,  # [num_children]
        numer_cardsum_ptr,  # [num_children, 52]
        denom_s_ptr,  # [top]
        denom_cardsum_ptr,  # [top, 52]
        card_a_ptr,  # [H]
        card_b_ptr,  # [H]
        child_offsets_ptr,  # [num_parents] absolute first-child row
        child_count_ptr,  # [num_parents]
        parent_base,  # absolute row of first parent in this depth slice
        child_base,  # absolute row corresponding to numer row 0
        H,
        NUM_PLAYERS: tl.constexpr,
        NUM_CARDS: tl.constexpr,
        EPS,
        HAS_LEAF_SOURCE: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        parent_rel = tl.program_id(0)
        hb = tl.program_id(1)
        row = parent_base + parent_rel
        first = tl.load(child_offsets_ptr + parent_rel)
        count = tl.load(child_count_ptr + parent_rel)
        col_offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        col_mask = col_offs < H
        players = tl.arange(0, BLOCK_P)
        player_mask = players < NUM_PLAYERS
        value_mask = player_mask[:, None] & col_mask[None, :]

        if count == 0:
            if HAS_LEAF_SOURCE:
                is_leaf = tl.load(leaf_mask_ptr + row).to(tl.int1)
                src = tl.load(
                    leaf_values_ptr
                    + (row * NUM_PLAYERS + players[:, None]) * H
                    + col_offs[None, :],
                    mask=value_mask & is_leaf,
                    other=0.0,
                )
                tl.store(
                    values_ptr
                    + (row * NUM_PLAYERS + players[:, None]) * H
                    + col_offs[None, :],
                    src,
                    mask=value_mask,
                )
            return

        ca = tl.load(card_a_ptr + col_offs, mask=col_mask, other=0)
        cb = tl.load(card_b_ptr + col_offs, mask=col_mask, other=0)
        dt = tl.load(
            actor_beliefs_ptr + row * H + col_offs,
            mask=col_mask,
            other=0.0,
        )
        sd = tl.load(denom_s_ptr + row)
        dcsa = tl.load(denom_cardsum_ptr + row * NUM_CARDS + ca, mask=col_mask)
        dcsb = tl.load(denom_cardsum_ptr + row * NUM_CARDS + cb, mask=col_mask)

        acc = tl.zeros([BLOCK_P, BLOCK_H], dtype=tl.float32)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                child_rel = child - child_base
                prev_actor = tl.load(prev_actor_ptr + child)
                hero_pol = tl.load(
                    policy_hero_ptr + child * H + col_offs,
                    mask=col_mask,
                    other=0.0,
                )
                nt = dt * hero_pol
                sn = tl.load(numer_s_ptr + child_rel)
                ncsa = tl.load(
                    numer_cardsum_ptr + child_rel * NUM_CARDS + ca,
                    mask=col_mask,
                    other=0.0,
                )
                ncsb = tl.load(
                    numer_cardsum_ptr + child_rel * NUM_CARDS + cb,
                    mask=col_mask,
                    other=0.0,
                )
                numer = tl.maximum(sn - ncsa - ncsb + nt, 0.0)
                denom = tl.maximum(sd - dcsa - dcsb + dt, 0.0)
                opp_pol = tl.where(denom > EPS, numer / denom, 0.0)
                pol = tl.where(
                    players[:, None] == prev_actor,
                    hero_pol[None, :],
                    opp_pol[None, :],
                )
                vals = tl.load(
                    values_ptr
                    + (child * NUM_PLAYERS + players[:, None]) * H
                    + col_offs[None, :],
                    mask=value_mask,
                    other=0.0,
                )
                if HAS_LEAF_SOURCE:
                    leaf_vals = tl.load(
                        leaf_values_ptr
                        + (child * NUM_PLAYERS + players[:, None]) * H
                        + col_offs[None, :],
                        mask=value_mask,
                        other=0.0,
                    )
                    child_is_leaf = tl.load(leaf_mask_ptr + child).to(tl.int1)
                    vals = tl.where(child_is_leaf, leaf_vals, vals)
                    tl.store(
                        values_ptr
                        + (child * NUM_PLAYERS + players[:, None]) * H
                        + col_offs[None, :],
                        leaf_vals,
                        mask=value_mask & child_is_leaf,
                    )
                acc += vals * pol

        tl.store(
            values_ptr
            + (row * NUM_PLAYERS + players[:, None]) * H
            + col_offs[None, :],
            acc,
            mask=value_mask,
        )


def fused_weighted_parent_sum_inline_opp_multiway(
    values: torch.Tensor,
    prev_actor: torch.Tensor,
    policy_hero: torch.Tensor,
    actor_beliefs: torch.Tensor,
    numer_s: torch.Tensor,
    numer_cardsum: torch.Tensor,
    denom_s: torch.Tensor,
    denom_cardsum: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    parent_base: int,
    child_base: int,
    max_children: int,
    max_children_pow2: int | None = None,
    eps: float = 1e-5,
    leaf_values: torch.Tensor | None = None,
    leaf_mask: torch.Tensor | None = None,
    block_h: int = 256,
) -> None:
    """Multiway inline-opponent EV backup."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 3
    assert policy_hero.is_contiguous() and actor_beliefs.is_contiguous()
    assert numer_s.is_contiguous() and numer_cardsum.is_contiguous()
    assert denom_s.is_contiguous() and denom_cardsum.is_contiguous()
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    total, p, h = values.shape
    top = actor_beliefs.shape[0]
    num_children = numer_s.shape[0]
    assert p >= 2 and policy_hero.shape == (total, h)
    assert actor_beliefs.shape == (top, h)
    assert numer_cardsum.shape == (num_children, _UNBLOCKED_NUM_CARDS)
    assert denom_cardsum.shape == (top, _UNBLOCKED_NUM_CARDS)
    assert denom_s.shape == (top,)
    assert prev_actor.shape == (total,)
    has_leaf_source = leaf_values is not None
    if has_leaf_source:
        assert leaf_values is not None and leaf_mask is not None
        assert leaf_values.is_contiguous() and leaf_values.shape == values.shape
        assert leaf_mask.is_contiguous() and leaf_mask.shape == (total,)
        leaf_values_ptr = leaf_values
        leaf_mask_ptr = leaf_mask
    else:
        leaf_values_ptr = values
        leaf_mask_ptr = child_count
    if max_children_pow2 is None:
        mc_pow2 = 1
        while mc_pow2 < max_children:
            mc_pow2 *= 2
    else:
        mc_pow2 = max_children_pow2
    block_p = 1
    while block_p < p:
        block_p *= 2
    num_parents = child_offsets.shape[0]
    card_a, card_b = _get_combo_cards(values.device)
    _fused_weighted_parent_sum_inline_opp_multiway_kernel[
        (num_parents, triton.cdiv(h, block_h))
    ](
        values,
        leaf_values_ptr,
        leaf_mask_ptr,
        prev_actor,
        policy_hero,
        actor_beliefs,
        numer_s,
        numer_cardsum,
        denom_s,
        denom_cardsum,
        card_a,
        card_b,
        child_offsets,
        child_count,
        parent_base,
        child_base,
        h,
        NUM_PLAYERS=p,
        NUM_CARDS=_UNBLOCKED_NUM_CARDS,
        EPS=eps,
        HAS_LEAF_SOURCE=has_leaf_source,
        MAX_CHILDREN=mc_pow2,
        BLOCK_P=block_p,
        BLOCK_H=block_h,
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Kernel 12b: fused best-response depth backup helpers.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _br_best_action_mass_kernel(
        values_ptr,  # [total, 2, H] current child values / parent out
        actor_beliefs_ptr,  # [top, H]
        to_act_ptr,  # [total]
        deviator_ptr,  # [total]
        child_offsets_ptr,  # [num_parents] absolute first child
        child_count_ptr,  # [num_parents]
        action_from_parent_ptr,  # [total]
        mass_ptr,  # [num_parents, A, H]
        best_value_ptr,  # [num_parents, H]
        parent_base,
        H,
        NUM_ACTIONS: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)
        row = parent_base + p
        actor = tl.load(to_act_ptr + row)
        deviator = tl.load(deviator_ptr + row)
        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)

        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H

        best = tl.full([BLOCK_H], -3.4028234663852886e38, tl.float32)
        best_action = tl.zeros([BLOCK_H], tl.int64)
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                action = tl.load(action_from_parent_ptr + child)
                v = tl.load(
                    values_ptr + (child * 2 + actor) * H + offs,
                    mask=mask,
                    other=-3.4028234663852886e38,
                )
                take = v > best
                best = tl.where(take, v, best)
                best_action = tl.where(take, action, best_action)

        belief = tl.load(actor_beliefs_ptr + row * H + offs, mask=mask, other=0.0)
        use_mass = deviator == actor
        for a in tl.static_range(0, NUM_ACTIONS):
            out = tl.where((best_action == a) & use_mass, belief, 0.0)
            tl.store(mass_ptr + (p * NUM_ACTIONS + a) * H + offs, out, mask=mask)
        tl.store(best_value_ptr + p * H + offs, best, mask=mask)

    @triton.jit
    def _br_finalize_depth_kernel(
        values_ptr,  # [total, 2, H] in/out
        policy_ptr,  # [total, H] child policy
        opponent_policy_ptr,  # [top, A, H]
        p_dev_ptr,  # [num_parents, A, H]
        best_value_ptr,  # [num_parents, H]
        to_act_ptr,  # [total]
        deviator_ptr,  # [total]
        child_offsets_ptr,  # [num_parents]
        child_count_ptr,  # [num_parents]
        action_from_parent_ptr,  # [total]
        parent_base,
        opponent_policy_base,
        H,
        NUM_ACTIONS: tl.constexpr,
        MAX_CHILDREN: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        p = tl.program_id(0)
        hb = tl.program_id(1)
        row = parent_base + p
        actor = tl.load(to_act_ptr + row)
        deviator = tl.load(deviator_ptr + row)
        first = tl.load(child_offsets_ptr + p)
        count = tl.load(child_count_ptr + p)
        if count == 0:
            return

        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H

        actor_avg = tl.zeros([BLOCK_H], tl.float32)
        opp_avg = tl.zeros([BLOCK_H], tl.float32)
        opp_br = tl.zeros([BLOCK_H], tl.float32)
        opp = 1 - actor
        for i in tl.static_range(0, MAX_CHILDREN):
            if i < count:
                child = first + i
                action = tl.load(action_from_parent_ptr + child)
                actor_v = tl.load(
                    values_ptr + (child * 2 + actor) * H + offs,
                    mask=mask,
                    other=0.0,
                )
                opp_v = tl.load(
                    values_ptr + (child * 2 + opp) * H + offs,
                    mask=mask,
                    other=0.0,
                )
                pol = tl.load(policy_ptr + child * H + offs, mask=mask, other=0.0)
                opp_pol = tl.load(
                    opponent_policy_ptr
                    + ((row - opponent_policy_base) * NUM_ACTIONS + action) * H
                    + offs,
                    mask=mask,
                    other=0.0,
                )
                dev_pol = tl.load(
                    p_dev_ptr + (p * NUM_ACTIONS + action) * H + offs,
                    mask=mask,
                    other=0.0,
                )
                actor_avg += actor_v * pol
                opp_avg += opp_v * opp_pol
                opp_br += opp_v * dev_pol

        best = tl.load(best_value_ptr + p * H + offs, mask=mask, other=0.0)
        actor_out = tl.where(deviator == actor, best, actor_avg)
        opp_out = tl.where(deviator == actor, opp_br, opp_avg)
        tl.store(values_ptr + (row * 2 + actor) * H + offs, actor_out, mask=mask)
        tl.store(values_ptr + (row * 2 + opp) * H + offs, opp_out, mask=mask)


def fused_br_best_action_mass(
    values: torch.Tensor,
    actor_beliefs: torch.Tensor,
    to_act: torch.Tensor,
    deviator: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    action_from_parent: torch.Tensor,
    parent_base: int,
    num_actions: int,
    max_children: int,
    block_h: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 3 and values.shape[1] == 2
    assert actor_beliefs.is_contiguous() and actor_beliefs.dim() == 2
    assert to_act.is_contiguous() and deviator.is_contiguous()
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    assert action_from_parent.is_contiguous()
    num_parents = child_offsets.shape[0]
    h = values.shape[-1]
    assert actor_beliefs.shape[1] == h
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    mass = torch.empty(
        (num_parents, num_actions, h),
        device=values.device,
        dtype=values.dtype,
    )
    best = torch.empty((num_parents, h), device=values.device, dtype=values.dtype)
    grid = (num_parents, triton.cdiv(h, block_h))
    _br_best_action_mass_kernel[grid](
        values,
        actor_beliefs,
        to_act,
        deviator,
        child_offsets,
        child_count,
        action_from_parent,
        mass,
        best,
        parent_base,
        h,
        NUM_ACTIONS=num_actions,
        MAX_CHILDREN=mc_pow2,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return mass, best


def fused_br_finalize_depth_(
    values: torch.Tensor,
    policy: torch.Tensor,
    opponent_policy: torch.Tensor,
    p_dev: torch.Tensor,
    best_values: torch.Tensor,
    to_act: torch.Tensor,
    deviator: torch.Tensor,
    child_offsets: torch.Tensor,
    child_count: torch.Tensor,
    action_from_parent: torch.Tensor,
    parent_base: int,
    num_actions: int,
    max_children: int,
    opponent_policy_base: int = 0,
    block_h: int = 2048,
) -> None:
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert values.is_contiguous() and values.dim() == 3 and values.shape[1] == 2
    assert policy.is_contiguous() and policy.dim() == 2
    assert opponent_policy.is_contiguous() and opponent_policy.dim() == 3
    assert p_dev.is_contiguous() and p_dev.dim() == 3
    assert best_values.is_contiguous() and best_values.dim() == 2
    assert to_act.is_contiguous() and deviator.is_contiguous()
    assert child_offsets.is_contiguous() and child_count.is_contiguous()
    assert action_from_parent.is_contiguous()
    num_parents = child_offsets.shape[0]
    h = values.shape[-1]
    assert policy.shape == (values.shape[0], h)
    assert opponent_policy.shape[1:] == (num_actions, h)
    assert opponent_policy_base <= parent_base
    assert opponent_policy.shape[0] >= parent_base + num_parents - opponent_policy_base
    assert p_dev.shape == (num_parents, num_actions, h)
    assert best_values.shape == (num_parents, h)
    mc_pow2 = 1
    while mc_pow2 < max_children:
        mc_pow2 *= 2
    grid = (num_parents, triton.cdiv(h, block_h))
    _br_finalize_depth_kernel[grid](
        values,
        policy,
        opponent_policy,
        p_dev,
        best_values,
        to_act,
        deviator,
        child_offsets,
        child_count,
        action_from_parent,
        parent_base,
        opponent_policy_base,
        h,
        NUM_ACTIONS=num_actions,
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
        reach_ptr,  # [total, 2, H] in/out
        policy_ptr,  # [total, H]
        allowed_mask_ptr,  # [total, H] bool
        parent_index_ptr,  # [total]
        prev_actor_ptr,  # [total]
        start,
        end,
        H,
        APPLY_ALLOWED: tl.constexpr,
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
        v0 = tl.load(reach_ptr + (parent * 2 + 0) * H + offs, mask=mask, other=0.0)
        v1 = tl.load(reach_ptr + (parent * 2 + 1) * H + offs, mask=mask, other=0.0)
        if prev_actor == 0:
            v0 = v0 * pol
        else:
            v1 = v1 * pol
        if APPLY_ALLOWED:
            al = tl.load(allowed_mask_ptr + c * H + offs, mask=mask, other=0).to(
                tl.int1
            )
            v0 = tl.where(al, v0, 0.0)
            v1 = tl.where(al, v1, 0.0)
        tl.store(reach_ptr + (c * 2 + 0) * H + offs, v0, mask=mask)
        tl.store(reach_ptr + (c * 2 + 1) * H + offs, v1, mask=mask)


def fused_reach_weights_depth_(
    reach: torch.Tensor,  # [total, 2, H] in/out
    policy: torch.Tensor,  # [total, H]
    allowed_mask: torch.Tensor,  # [total, H] bool
    parent_index: torch.Tensor,  # [total]
    prev_actor: torch.Tensor,  # [total]
    start: int,
    end: int,
    apply_allowed_mask: bool = True,
    block_h: int = 2048,
) -> None:
    """For each child row ``c in [start, end)`` and player ``p``::

        reach[c, p, h] = reach[parent_index[c], p, h] *
                        (policy[c, h] if p == prev_actor[c] else 1.0)

    Replaces the per-depth ``fan_out + scatter_reduce(prod)`` pair in
    ``_calculate_reach_weights``. Caller must invoke per depth in top-down
    order (children depend on freshly-computed parents). For same-board trees,
    callers can apply the allowed-hand mask only at depth 0; invalid-hand zeros
    then propagate to every deeper node.
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
        APPLY_ALLOWED=apply_allowed_mask,
        BLOCK_H=block_h,
        num_warps=4,
    )


if triton is not None:

    @triton.jit
    def _fused_reach_beliefs_avg_depth_kernel(
        reach_ptr,  # [total, 2, H] in/out
        beliefs_ptr,  # [total, 2, H] in/out
        policy_ptr,  # [total, H]
        allowed_mask_ptr,  # [total, H] bool
        allowed_prob_ptr,  # [total, H]
        root_index_ptr,  # [total]
        parent_index_ptr,  # [total]
        prev_actor_ptr,  # [total]
        to_act_ptr,  # [total]
        avg_num_ptr,  # [total, H] in/out if WRITE_AVG
        avg_den_ptr,  # [total, H] in/out if WRITE_AVG
        leaf_slot_ptr,  # [total], -1 for rows that should not be scattered
        leaf_out_ptr,  # [M, 2, H] OUT if WRITE_LEAF
        new_scalar_ptr,
        start,
        end,
        H,
        EPS,
        APPLY_ALLOWED: tl.constexpr,
        WRITE_AVG: tl.constexpr,
        STORE_REACH: tl.constexpr,
        WRITE_LEAF: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        c = tl.program_id(0) + start
        if c >= end:
            return

        parent = tl.load(parent_index_ptr + c)
        prev_actor = tl.load(prev_actor_ptr + c)
        root = tl.load(root_index_ptr + c)

        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        pol = tl.load(policy_ptr + c * H + offs, mask=mask, other=0.0)

        parent0 = tl.load(
            reach_ptr + (parent * 2 + 0) * H + offs,
            mask=mask,
            other=0.0,
        )
        parent1 = tl.load(
            reach_ptr + (parent * 2 + 1) * H + offs,
            mask=mask,
            other=0.0,
        )
        child0 = tl.where(prev_actor == 0, parent0 * pol, parent0)
        child1 = tl.where(prev_actor == 1, parent1 * pol, parent1)
        if APPLY_ALLOWED:
            allowed = tl.load(allowed_mask_ptr + c * H + offs, mask=mask, other=0).to(
                tl.int1
            )
            child0 = tl.where(allowed, child0, 0.0)
            child1 = tl.where(allowed, child1, 0.0)

        if STORE_REACH:
            tl.store(reach_ptr + (c * 2 + 0) * H + offs, child0, mask=mask)
            tl.store(reach_ptr + (c * 2 + 1) * H + offs, child1, mask=mask)

        if WRITE_AVG:
            new_scalar = tl.load(new_scalar_ptr)
            parent_actor = tl.where(prev_actor == 0, parent0, parent1)
            reach_n = parent_actor * new_scalar
            num_old = tl.load(avg_num_ptr + c * H + offs, mask=mask, other=0.0)
            den_old = tl.load(avg_den_ptr + c * H + offs, mask=mask, other=0.0)
            tl.store(
                avg_num_ptr + c * H + offs,
                num_old + reach_n * pol,
                mask=mask,
            )
            tl.store(avg_den_ptr + c * H + offs, den_old + reach_n, mask=mask)

        root0 = tl.load(
            beliefs_ptr + (root * 2 + 0) * H + offs,
            mask=mask,
            other=0.0,
        )
        root1 = tl.load(
            beliefs_ptr + (root * 2 + 1) * H + offs,
            mask=mask,
            other=0.0,
        )
        b0 = root0 * child0
        b1 = root1 * child1
        sum0 = tl.sum(b0, axis=0)
        sum1 = tl.sum(b1, axis=0)
        fallback = tl.load(allowed_prob_ptr + c * H + offs, mask=mask, other=0.0)
        out0 = tl.where(sum0 > EPS, b0 / sum0, fallback)
        out1 = tl.where(sum1 > EPS, b1 / sum1, fallback)
        tl.store(beliefs_ptr + (c * 2 + 0) * H + offs, out0, mask=mask)
        tl.store(beliefs_ptr + (c * 2 + 1) * H + offs, out1, mask=mask)
        if WRITE_LEAF:
            slot_raw = tl.load(leaf_slot_ptr + c)
            slot = tl.maximum(slot_raw, 0)
            leaf_mask = mask & (slot_raw >= 0)
            tl.store(leaf_out_ptr + (slot * 2 + 0) * H + offs, out0, mask=leaf_mask)
            tl.store(leaf_out_ptr + (slot * 2 + 1) * H + offs, out1, mask=leaf_mask)


def fused_reach_beliefs_avg_depth_(
    reach: torch.Tensor,
    beliefs: torch.Tensor,
    policy: torch.Tensor,
    allowed_mask: torch.Tensor,
    allowed_prob: torch.Tensor,
    root_index: torch.Tensor,
    parent_index: torch.Tensor,
    prev_actor: torch.Tensor,
    to_act: torch.Tensor,
    average_policy_numerator: torch.Tensor,
    average_policy_denominator: torch.Tensor,
    new: torch.Tensor,
    start: int,
    end: int,
    write_average_policy: bool,
    store_reach: bool = True,
    apply_allowed_mask: bool = True,
    leaf_slot: torch.Tensor | None = None,
    leaf_out: torch.Tensor | None = None,
    eps: float = 1e-5,
    block_h: int = 2048,
) -> None:
    """Fuse reach propagation, belief normalization, and deferred avg-policy.

    The average-policy update is optional so pre-DCFR-delay iterations can use
    the same reach/belief fusion without changing deferred-average semantics.
    For same-board trees, callers can apply the allowed-hand mask only at depth
    0; invalid-hand zeros then propagate to every deeper node.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert reach.is_contiguous() and reach.dim() == 3 and reach.shape[1] == 2
    assert beliefs.is_contiguous() and beliefs.shape == reach.shape
    assert policy.is_contiguous() and policy.dim() == 2
    total, two, h = reach.shape
    assert two == 2 and policy.shape == (total, h)
    assert allowed_mask.is_contiguous() and allowed_mask.shape == (total, h)
    assert allowed_prob.is_contiguous() and allowed_prob.shape == (total, h)
    assert root_index.is_contiguous() and root_index.shape == (total,)
    assert parent_index.is_contiguous() and parent_index.shape == (total,)
    assert prev_actor.is_contiguous() and prev_actor.shape == (total,)
    assert to_act.is_contiguous() and to_act.shape == (total,)
    assert average_policy_numerator.is_contiguous()
    assert average_policy_numerator.shape == (total, h)
    assert average_policy_denominator.is_contiguous()
    assert average_policy_denominator.shape == (total, h)
    assert new.is_cuda and new.numel() == 1
    write_leaf = leaf_slot is not None and leaf_out is not None
    if write_leaf:
        assert leaf_slot is not None and leaf_out is not None
        assert leaf_slot.is_contiguous() and leaf_slot.shape == (total,)
        assert leaf_out.is_contiguous() and leaf_out.dim() == 3
        assert leaf_out.shape[1:] == (2, h)
    else:
        leaf_slot = parent_index
        leaf_out = beliefs
    assert h <= block_h, f"fused reach/beliefs assumes H ({h}) <= BLOCK_H ({block_h})"
    n = end - start
    if n <= 0:
        return
    grid = (n,)
    _fused_reach_beliefs_avg_depth_kernel[grid](
        reach,
        beliefs,
        policy,
        allowed_mask,
        allowed_prob,
        root_index,
        parent_index,
        prev_actor,
        to_act,
        average_policy_numerator,
        average_policy_denominator,
        leaf_slot,
        leaf_out,
        new,
        start,
        end,
        h,
        eps,
        APPLY_ALLOWED=apply_allowed_mask,
        WRITE_AVG=write_average_policy,
        STORE_REACH=store_reach,
        WRITE_LEAF=write_leaf,
        BLOCK_H=block_h,
        num_warps=8,
    )


if triton is not None:

    @triton.jit
    def _fused_reach_beliefs_avg_scratch_depth_kernel(
        parent_reach_ptr,  # [parent_count, 2, H], unused for ROOT_PARENT
        child_reach_ptr,  # [child_count, 2, H] OUT if STORE_CHILD
        beliefs_ptr,  # [total, 2, H] in/out
        policy_ptr,  # [total, H]
        allowed_mask_ptr,  # [total, H] bool
        allowed_prob_ptr,  # [total, H]
        root_index_ptr,  # [total]
        parent_index_ptr,  # [total]
        prev_actor_ptr,  # [total]
        to_act_ptr,  # [total]
        avg_num_ptr,  # [total, H] in/out if WRITE_AVG
        avg_den_ptr,  # [total, H] in/out if WRITE_AVG
        leaf_slot_ptr,  # [total], -1 for rows that should not be scattered
        leaf_out_ptr,  # [M, 2, H] OUT if WRITE_LEAF
        new_scalar_ptr,
        parent_base,
        start,
        end,
        H,
        EPS,
        ROOT_PARENT: tl.constexpr,
        APPLY_ALLOWED: tl.constexpr,
        WRITE_AVG: tl.constexpr,
        STORE_CHILD: tl.constexpr,
        WRITE_LEAF: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        c_rel = tl.program_id(0)
        c = c_rel + start
        if c >= end:
            return

        parent = tl.load(parent_index_ptr + c)
        prev_actor = tl.load(prev_actor_ptr + c)
        root = parent if ROOT_PARENT else tl.load(root_index_ptr + c)

        offs = tl.arange(0, BLOCK_H)
        mask = offs < H

        pol = tl.load(policy_ptr + c * H + offs, mask=mask, other=0.0)

        if ROOT_PARENT:
            parent0 = tl.full([BLOCK_H], 1.0, tl.float32)
            parent1 = tl.full([BLOCK_H], 1.0, tl.float32)
        else:
            parent_rel = parent - parent_base
            parent0 = tl.load(
                parent_reach_ptr + (parent_rel * 2 + 0) * H + offs,
                mask=mask,
                other=0.0,
            )
            parent1 = tl.load(
                parent_reach_ptr + (parent_rel * 2 + 1) * H + offs,
                mask=mask,
                other=0.0,
            )

        child0 = tl.where(prev_actor == 0, parent0 * pol, parent0)
        child1 = tl.where(prev_actor == 1, parent1 * pol, parent1)
        if APPLY_ALLOWED:
            allowed = tl.load(allowed_mask_ptr + c * H + offs, mask=mask, other=0).to(
                tl.int1
            )
            child0 = tl.where(allowed, child0, 0.0)
            child1 = tl.where(allowed, child1, 0.0)

        if STORE_CHILD:
            tl.store(child_reach_ptr + (c_rel * 2 + 0) * H + offs, child0, mask=mask)
            tl.store(child_reach_ptr + (c_rel * 2 + 1) * H + offs, child1, mask=mask)

        if WRITE_AVG:
            new_scalar = tl.load(new_scalar_ptr)
            parent_actor = tl.where(prev_actor == 0, parent0, parent1)
            reach_n = parent_actor * new_scalar
            num_old = tl.load(avg_num_ptr + c * H + offs, mask=mask, other=0.0)
            den_old = tl.load(avg_den_ptr + c * H + offs, mask=mask, other=0.0)
            tl.store(
                avg_num_ptr + c * H + offs,
                num_old + reach_n * pol,
                mask=mask,
            )
            tl.store(avg_den_ptr + c * H + offs, den_old + reach_n, mask=mask)

        root0 = tl.load(
            beliefs_ptr + (root * 2 + 0) * H + offs,
            mask=mask,
            other=0.0,
        )
        root1 = tl.load(
            beliefs_ptr + (root * 2 + 1) * H + offs,
            mask=mask,
            other=0.0,
        )
        b0 = root0 * child0
        b1 = root1 * child1
        sum0 = tl.sum(b0, axis=0)
        sum1 = tl.sum(b1, axis=0)
        fallback = tl.load(allowed_prob_ptr + c * H + offs, mask=mask, other=0.0)
        out0 = tl.where(sum0 > EPS, b0 / sum0, fallback)
        out1 = tl.where(sum1 > EPS, b1 / sum1, fallback)
        tl.store(beliefs_ptr + (c * 2 + 0) * H + offs, out0, mask=mask)
        tl.store(beliefs_ptr + (c * 2 + 1) * H + offs, out1, mask=mask)
        if WRITE_LEAF:
            slot_raw = tl.load(leaf_slot_ptr + c)
            slot = tl.maximum(slot_raw, 0)
            leaf_mask = mask & (slot_raw >= 0)
            tl.store(leaf_out_ptr + (slot * 2 + 0) * H + offs, out0, mask=leaf_mask)
            tl.store(leaf_out_ptr + (slot * 2 + 1) * H + offs, out1, mask=leaf_mask)


def fused_reach_beliefs_avg_scratch_depth_(
    parent_reach: torch.Tensor,
    child_reach: torch.Tensor,
    beliefs: torch.Tensor,
    policy: torch.Tensor,
    allowed_mask: torch.Tensor,
    allowed_prob: torch.Tensor,
    root_index: torch.Tensor,
    parent_index: torch.Tensor,
    prev_actor: torch.Tensor,
    to_act: torch.Tensor,
    average_policy_numerator: torch.Tensor,
    average_policy_denominator: torch.Tensor,
    new: torch.Tensor,
    parent_base: int,
    start: int,
    end: int,
    root_parent: bool,
    write_average_policy: bool,
    store_child: bool = True,
    apply_allowed_mask: bool = True,
    leaf_slot: torch.Tensor | None = None,
    leaf_out: torch.Tensor | None = None,
    eps: float = 1e-5,
    block_h: int = 2048,
) -> None:
    """Fused reach/belief/average update with depth-local reach storage."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert parent_reach.is_contiguous() and parent_reach.dim() == 3
    assert child_reach.is_contiguous() and child_reach.dim() == 3
    assert beliefs.is_contiguous() and beliefs.dim() == 3 and beliefs.shape[1] == 2
    assert policy.is_contiguous() and policy.dim() == 2
    total, two, h = beliefs.shape
    assert two == 2 and policy.shape == (total, h)
    assert parent_reach.shape[1:] == (2, h)
    assert child_reach.shape[1:] == (2, h)
    assert allowed_mask.is_contiguous() and allowed_mask.shape == (total, h)
    assert allowed_prob.is_contiguous() and allowed_prob.shape == (total, h)
    assert root_index.is_contiguous() and root_index.shape == (total,)
    assert parent_index.is_contiguous() and parent_index.shape == (total,)
    assert prev_actor.is_contiguous() and prev_actor.shape == (total,)
    assert to_act.is_contiguous() and to_act.shape == (total,)
    assert average_policy_numerator.is_contiguous()
    assert average_policy_numerator.shape == (total, h)
    assert average_policy_denominator.is_contiguous()
    assert average_policy_denominator.shape == (total, h)
    assert new.is_cuda and new.numel() == 1
    write_leaf = leaf_slot is not None and leaf_out is not None
    if write_leaf:
        assert leaf_slot is not None and leaf_out is not None
        assert leaf_slot.is_contiguous() and leaf_slot.shape == (total,)
        assert leaf_out.is_contiguous() and leaf_out.dim() == 3
        assert leaf_out.shape[1:] == (2, h)
    else:
        leaf_slot = parent_index
        leaf_out = beliefs
    assert h <= block_h, f"fused reach/beliefs assumes H ({h}) <= BLOCK_H ({block_h})"
    n = end - start
    if n <= 0:
        return
    assert child_reach.shape[0] >= n
    grid = (n,)
    _fused_reach_beliefs_avg_scratch_depth_kernel[grid](
        parent_reach,
        child_reach,
        beliefs,
        policy,
        allowed_mask,
        allowed_prob,
        root_index,
        parent_index,
        prev_actor,
        to_act,
        average_policy_numerator,
        average_policy_denominator,
        leaf_slot,
        leaf_out,
        new,
        parent_base,
        start,
        end,
        h,
        eps,
        ROOT_PARENT=root_parent,
        APPLY_ALLOWED=apply_allowed_mask,
        WRITE_AVG=write_average_policy,
        STORE_CHILD=store_child,
        WRITE_LEAF=write_leaf,
        BLOCK_H=block_h,
        num_warps=8,
    )


# ---------------------------------------------------------------------------
# Kernel 14: fan_out_deep * reach_weights + block + normalize in one kernel.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_deep_beliefs_kernel(
        root_beliefs_ptr,  # [N, 2, H] (read-only; same storage as out[:N])
        reach_ptr,  # [total, 2, H]; pre-blocked by reach kernel
        allowed_prob_ptr,  # [total, H]
        root_index_ptr,  # [total - N] (only non-root rows)
        out_ptr,  # [total, 2, H]
        N,  # number of root rows (skipped; idempotent)
        H,
        EPS,
        BLOCK_H: tl.constexpr,
    ):
        # i indexes non-root rows in [N, total). Roots are idempotent
        # (out[i] = root_beliefs[i] for i < N) so we skip them entirely; that
        # also removes the cross-program race that previously required cloning
        # root_beliefs from out[:N].
        i = tl.program_id(0) + N
        p = tl.program_id(1)

        root_idx = tl.load(root_index_ptr + (i - N))
        root_row = root_beliefs_ptr + (root_idx * 2 + p) * H
        reach_row = reach_ptr + (i * 2 + p) * H
        out_row = out_ptr + (i * 2 + p) * H
        prob_row = allowed_prob_ptr + i * H

        # Single-tile path: H (=1326) fits in BLOCK_H (=2048). Keep `v`
        # register-resident across the sum + normalize so we don't spill the
        # intermediate to global memory. For same-board trees, invalid-hand
        # reach is zeroed at depth 0 and that zero propagates to all deeper
        # nodes, so rb * rw is already correctly masked — no extra allowed_mask
        # load needed here.
        off = tl.arange(0, BLOCK_H)
        m = off < H
        rb = tl.load(root_row + off, mask=m, other=0.0)
        rw = tl.load(reach_row + off, mask=m, other=0.0)
        v = rb * rw
        total = tl.sum(v)
        if total > EPS:
            out_v = v / total
        else:
            out_v = tl.load(prob_row + off, mask=m, other=0.0)
        tl.store(out_row + off, out_v, mask=m)


def fused_deep_beliefs_(
    out: torch.Tensor,  # [total, 2, H] in/out (only [N:] is written)
    root_beliefs: torch.Tensor,  # [>=N, 2, H]; only rows [:N] are read
    reach_weights: torch.Tensor,  # [total, 2, H]; assumed pre-blocked
    allowed_prob: torch.Tensor,  # [total, H]
    root_index: torch.Tensor,  # [total] int64
    num_roots: int | None = None,
    eps: float = 1e-5,
    block_h: int = 2048,
) -> None:
    """Fuses ``_fan_out_deep(root_beliefs) * reach_weights`` + normalize into
    one kernel. Replaces ``_propagate_all_beliefs``. ``reach_weights`` must
    already be blocked (zero where allowed_mask is False). For same-board trees,
    blocking depth 0 is enough because invalid-hand zeros propagate down the
    tree, so no extra block step is needed here.

    For each node ``i`` and player ``p``::

        v = root_beliefs[root_index[i], p, :] * reach_weights[i, p, :]
        s = v.sum()
        out[i, p, :] = where(s > eps, v / s, allowed_prob[i])
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert out.is_contiguous() and out.dim() == 3 and out.shape[1] == 2
    assert root_beliefs.is_contiguous() and root_beliefs.dim() == 3
    assert reach_weights.is_contiguous() and reach_weights.shape == out.shape
    total, two, h = out.shape
    # root_beliefs may be the full out tensor (when roots are idempotent so we
    # can read directly without cloning) or a [N, 2, H] snapshot. Caller must
    # pass num_roots in the former case; in the latter we infer.
    n = num_roots if num_roots is not None else root_beliefs.shape[0]
    assert root_beliefs.shape[0] >= n and root_beliefs.shape[1:] == (2, h)
    assert allowed_prob.shape == (total, h) and allowed_prob.is_contiguous()
    assert root_index.shape == (total,) and root_index.is_contiguous()
    assert h <= block_h, f"deep_beliefs assumes H ({h}) <= BLOCK_H ({block_h})"

    # Roots are idempotent (out[:N] == root_beliefs by construction). Skip them
    # in-kernel so non-root programs can read root_beliefs straight from
    # out[:N] without needing a cloned snapshot.
    if total <= n:
        return
    grid = (total - n, 2)
    _fused_deep_beliefs_kernel[grid](
        root_beliefs,
        reach_weights,
        allowed_prob,
        root_index[n:].contiguous()
        if not root_index[n:].is_contiguous()
        else root_index[n:],
        out,
        n,
        h,
        eps,
        BLOCK_H=block_h,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Kernel 15: sparse policy sample snapshots and compact sampled leaves.
# ---------------------------------------------------------------------------


if triton is not None:

    @triton.jit
    def _fused_policy_sample_update_kernel(
        policy_ptr,  # [total, H]
        sample_ptr,  # [total, H] in/out
        rows_ptr,  # [num_iters, max_updates] int64
        counts_ptr,  # [num_iters] int64
        t_ptr,  # 0-D int64
        max_updates,
        H,
        BLOCK_H: tl.constexpr,
    ):
        j = tl.program_id(0)
        hb = tl.program_id(1)
        t = tl.load(t_ptr)
        count = tl.load(counts_ptr + t)
        if j >= count:
            return

        row = tl.load(rows_ptr + t * max_updates + j)
        offs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = offs < H
        vals = tl.load(policy_ptr + row * H + offs, mask=mask, other=0.0)
        tl.store(sample_ptr + row * H + offs, vals, mask=mask)

    @triton.jit
    def _fused_sample_leaf_compact_kernel(
        policy_ptr,  # [total, H]
        beliefs_ptr,  # [total, 2 * H]
        rows_ptr,  # [num_iters, max_updates]
        counts_ptr,  # [num_iters]
        t_ptr,  # scalar int64
        players_ptr,  # [N]
        hands_ptr,  # [N, 2]
        uniform_draws_ptr,  # [D, N]
        action_draws_ptr,  # [D, N]
        target_enabled_ptr,  # [N] bool
        target_depths_ptr,  # [N]
        effective_leaf_ptr,  # [total] bool
        sampling_masks_ptr,  # [total, B] bool
        uniform_policy_ptr,  # [total, B]
        child_nodes_ptr,  # [total, B]
        to_act_ptr,  # [total]
        done_ptr,  # [total] bool
        allin_call_ptr,  # [total] bool
        new_street_ptr,  # [total] bool
        out_nodes_ptr,  # [N + 1]
        out_beliefs_ptr,  # [N + 1, 2 * H]
        out_ready_ptr,  # [N + 1]
        max_updates,
        N,
        B: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        SAMPLE_EPS: tl.constexpr,
        BLOCK_BELIEF: tl.constexpr,
        BELIEF_BLOCKS: tl.constexpr,
    ):
        slot = tl.program_id(0)
        t = tl.load(t_ptr)
        count = tl.load(counts_ptr + t)
        valid = slot < count
        root = tl.load(rows_ptr + t * max_updates + slot, mask=valid, other=N)
        safe_root = tl.minimum(root, N - 1)
        node = safe_root
        active = valid & (~tl.load(effective_leaf_ptr + node, mask=valid, other=1))
        target_enabled = tl.load(target_enabled_ptr + safe_root, mask=valid, other=0)
        target_depth = tl.load(target_depths_ptr + safe_root, mask=valid, other=D + 1)
        stopped_for_target = False

        for depth in tl.static_range(0, D):
            to_act = tl.load(to_act_ptr + node, mask=valid, other=0)
            player = tl.load(players_ptr + safe_root, mask=valid, other=0)
            uniform_draw = tl.load(
                uniform_draws_ptr + depth * N + safe_root,
                mask=valid,
                other=1.0,
            )
            sample_uniform = (uniform_draw < SAMPLE_EPS) & (to_act == player) & active
            hand = tl.load(hands_ptr + safe_root * 2 + to_act, mask=valid, other=0)
            action_draw = tl.load(
                action_draws_ptr + depth * N + safe_root,
                mask=valid,
                other=1.0,
            )

            denom = 0.0
            for a in tl.static_range(0, B):
                child = tl.load(child_nodes_ptr + node * B + a, mask=valid, other=0)
                legal = tl.load(
                    sampling_masks_ptr + node * B + a,
                    mask=valid,
                    other=0,
                )
                p = tl.load(
                    policy_ptr + child * H + hand, mask=valid & legal, other=0.0
                )
                denom += p

            cdf = 0.0
            action = B - 1
            chosen = False
            for a in tl.static_range(0, B):
                child = tl.load(child_nodes_ptr + node * B + a, mask=valid, other=0)
                legal = tl.load(
                    sampling_masks_ptr + node * B + a,
                    mask=valid,
                    other=0,
                )
                p = tl.load(
                    policy_ptr + child * H + hand, mask=valid & legal, other=0.0
                )
                uniform_p = tl.load(
                    uniform_policy_ptr + node * B + a,
                    mask=valid,
                    other=0.0,
                )
                policy_p = tl.where(denom >= 1.0e-12, p / denom, uniform_p)
                prob = tl.where(sample_uniform, uniform_p, policy_p)
                cdf += prob
                take = (~chosen) & (action_draw <= cdf)
                action = tl.where(take, a, action)
                chosen = chosen | take

            next_node = tl.load(
                child_nodes_ptr + node * B + action, mask=valid, other=0
            )
            node = tl.where(active, next_node, node)
            done_now = tl.load(done_ptr + node, mask=valid, other=1)
            allin_now = tl.load(allin_call_ptr + node, mask=valid, other=1)
            hit_target = (
                active
                & target_enabled
                & (target_depth == depth + 1)
                & (node >= N)
                & (~done_now)
                & (~allin_now)
            )
            stopped_for_target = stopped_for_target | hit_target
            active = (
                active
                & (~hit_target)
                & (~tl.load(effective_leaf_ptr + node, mask=valid, other=1))
            )

        done = tl.load(done_ptr + node, mask=valid, other=1)
        allin_call = tl.load(allin_call_ptr + node, mask=valid, other=1)
        effective_leaf = tl.load(effective_leaf_ptr + node, mask=valid, other=0)
        new_street = tl.load(new_street_ptr + node, mask=valid, other=0)
        ready = (
            valid
            & (node >= N)
            & (~done)
            & (~allin_call)
            & (stopped_for_target | (effective_leaf & new_street))
        )
        tl.store(out_nodes_ptr + root, node, mask=valid)
        tl.store(out_ready_ptr + root, ready, mask=valid)

        belief_h = 2 * H
        for belief_block in tl.static_range(0, BELIEF_BLOCKS):
            offs = belief_block * BLOCK_BELIEF + tl.arange(0, BLOCK_BELIEF)
            belief_mask = ready & (offs < belief_h)
            vals = tl.load(
                beliefs_ptr + node * belief_h + offs,
                mask=belief_mask,
                other=0.0,
            )
            tl.store(out_beliefs_ptr + root * belief_h + offs, vals, mask=belief_mask)


def fused_policy_sample_update_(
    policy_probs: torch.Tensor,
    policy_probs_sample: torch.Tensor,
    sample_rows: torch.Tensor,
    sample_counts: torch.Tensor,
    t: torch.Tensor,
    block_h: int = 512,
) -> None:
    """Copy only rows whose sampling iteration equals device scalar ``t``.

    ``sample_rows`` is a padded ``[num_iters, max_updates]`` table of row ids,
    and ``sample_counts[t]`` gives the valid row count for iteration ``t``.
    """
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy_probs.is_contiguous() and policy_probs.dim() == 2
    assert (
        policy_probs_sample.is_contiguous()
        and policy_probs_sample.shape == policy_probs.shape
    )
    assert sample_rows.is_contiguous() and sample_rows.dim() == 2
    assert sample_counts.is_contiguous() and sample_counts.dim() == 1
    assert t.dim() == 0

    max_updates = sample_rows.shape[1]
    if max_updates == 0:
        return
    h = policy_probs.shape[1]
    grid = (max_updates, triton.cdiv(h, block_h))
    _fused_policy_sample_update_kernel[grid](
        policy_probs,
        policy_probs_sample,
        sample_rows,
        sample_counts,
        t,
        max_updates,
        h,
        BLOCK_H=block_h,
        num_warps=4,
    )


def fused_sample_leaf_compact_(
    policy_probs: torch.Tensor,
    beliefs: torch.Tensor,
    sample_root_rows: torch.Tensor,
    sample_root_counts: torch.Tensor,
    t: torch.Tensor,
    players: torch.Tensor,
    hands: torch.Tensor,
    uniform_draws: torch.Tensor,
    action_draws: torch.Tensor,
    target_enabled: torch.Tensor,
    target_depths: torch.Tensor,
    effective_leaf_mask: torch.Tensor,
    sampling_masks: torch.Tensor,
    uniform_policy: torch.Tensor,
    child_nodes_by_action: torch.Tensor,
    to_act: torch.Tensor,
    done: torch.Tensor,
    allin_call_mask: torch.Tensor,
    new_street_mask: torch.Tensor,
    out_nodes: torch.Tensor,
    out_beliefs: torch.Tensor,
    out_ready: torch.Tensor,
    *,
    sample_epsilon: float,
    block_belief: int = 1024,
) -> None:
    """Sample current-iteration leaves for roots assigned to ``t``."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    assert policy_probs.is_contiguous() and policy_probs.dim() == 2
    assert beliefs.is_contiguous() and beliefs.dim() == 3
    assert sample_root_rows.is_contiguous() and sample_root_rows.dim() == 2
    assert sample_root_counts.is_contiguous() and sample_root_counts.dim() == 1
    assert t.dim() == 0
    assert players.is_contiguous() and players.dim() == 1
    assert hands.is_contiguous() and hands.dim() == 2 and hands.shape[1] == 2
    assert uniform_draws.is_contiguous() and uniform_draws.dim() == 2
    assert action_draws.is_contiguous() and action_draws.shape == uniform_draws.shape
    assert target_enabled.is_contiguous() and target_enabled.dim() == 1
    assert target_depths.is_contiguous() and target_depths.shape == target_enabled.shape
    assert effective_leaf_mask.is_contiguous() and effective_leaf_mask.dim() == 1
    assert sampling_masks.is_contiguous() and sampling_masks.dim() == 2
    assert (
        uniform_policy.is_contiguous() and uniform_policy.shape == sampling_masks.shape
    )
    assert (
        child_nodes_by_action.is_contiguous()
        and child_nodes_by_action.shape == sampling_masks.shape
    )
    assert to_act.is_contiguous() and done.is_contiguous()
    assert allin_call_mask.is_contiguous() and allin_call_mask.shape == done.shape
    assert new_street_mask.is_contiguous() and new_street_mask.dim() == 1
    assert out_nodes.is_contiguous() and out_ready.is_contiguous()
    assert out_beliefs.is_contiguous() and out_beliefs.dim() == 3

    max_updates = sample_root_rows.shape[1]
    if max_updates == 0:
        return
    n = players.shape[0]
    h = policy_probs.shape[1]
    b = sampling_masks.shape[1]
    d = uniform_draws.shape[0]
    assert beliefs.shape == (policy_probs.shape[0], 2, h)
    assert target_enabled.shape[0] == n
    assert out_beliefs.shape == (n + 1, 2, h)
    assert out_nodes.shape[0] == n + 1 and out_ready.shape[0] == n + 1
    belief_blocks = triton.cdiv(2 * h, block_belief)
    grid = (max_updates,)
    _fused_sample_leaf_compact_kernel[grid](
        policy_probs,
        beliefs,
        sample_root_rows,
        sample_root_counts,
        t,
        players,
        hands,
        uniform_draws,
        action_draws,
        target_enabled,
        target_depths,
        effective_leaf_mask,
        sampling_masks,
        uniform_policy,
        child_nodes_by_action,
        to_act,
        done,
        allin_call_mask,
        new_street_mask,
        out_nodes,
        out_beliefs,
        out_ready,
        max_updates,
        n,
        B=b,
        H=h,
        D=d,
        SAMPLE_EPS=float(sample_epsilon),
        BLOCK_BELIEF=block_belief,
        BELIEF_BLOCKS=belief_blocks,
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

    def __init__(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> None:
        def _z():
            return torch.zeros((), dtype=dtype, device=device)

        self.device = device
        self.dtype = dtype
        self.zero = _z()
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
        # Predictive CFR policy extraction.
        self.predictive_scale = _z()
        self.current_player = torch.zeros((), dtype=torch.long, device=device)
        # Model-values mix (for _set_model_values_impl): (old+new)/new and old/new
        self.mix_onon = _z()  # (old + new) / new
        self.mix_oon = _z()  # old / new
        # t as int64 device scalar (for t_sample == t comparisons)
        self.t_tensor = torch.zeros((), dtype=torch.long, device=device)

    def update(
        self,
        t: int,
        dcfr_alpha: float,
        dcfr_beta: float,
        mix_old: float,
        mix_new: float,
        predictive_scale: float = 0.0,
        current_player: int = 0,
    ) -> None:
        """Write t-derived scalars into the device tensors.

        Always a host→device copy via ``.fill_(python_float)`` — call OUTSIDE
        any captured region (before ``graph.replay()``).
        """
        t_discount = max(1, int(t))
        t_alpha_num = float(t_discount**dcfr_alpha)
        t_beta_num = float(t_discount**dcfr_beta)
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
        self.predictive_scale.fill_(float(predictive_scale))
        self.current_player.fill_(int(current_player))
        self.t_tensor.fill_(int(t))


# ---------------------------------------------------------------------------
# CUDA graph capture of a full cfr_iteration.
# ---------------------------------------------------------------------------


@dataclass
class _EvaluatorStateSnapshot:
    """Subset of evaluator tensors mutated by cfr_iteration."""

    names: tuple[str, ...]
    tensors: tuple[torch.Tensor, ...]
    reach_top: int | None = None
    ignore_self_reach: bool = False

    @classmethod
    def from_evaluator(cls, evaluator) -> "_EvaluatorStateSnapshot":
        names = [
            "policy_probs",
            "policy_probs_avg",
            "average_policy_numerator",
            "average_policy_denominator",
            "policy_probs_sample",
            "beliefs_sample",
            "cumulative_regrets",
            "self_reach",
            "self_reach_avg",
            "beliefs",
            "beliefs_avg",
            "latest_values",
            "values_avg",
        ]
        tensors = tuple(getattr(evaluator, n).detach().clone() for n in names)
        reach_top = None
        if hasattr(evaluator, "depth_offsets") and len(evaluator.depth_offsets) >= 2:
            # Leaf self_reach can be skipped in graph/stat-free hot paths:
            # no downstream CFR math consumes it, and _record_stats only reads
            # non-leaf reach. Keep graph parity focused on observable state.
            reach_top = int(evaluator.depth_offsets[-2])
        return cls(tuple(names), tensors, reach_top, ignore_self_reach=True)

    def restore_to(self, evaluator) -> None:
        for name, saved in zip(self.names, self.tensors):
            getattr(evaluator, name).copy_(saved)

    def max_abs_diff(self, other: "_EvaluatorStateSnapshot") -> dict[str, float]:
        assert self.names == other.names
        out = {}
        for name, a, b in zip(self.names, self.tensors, other.tensors):
            if name == "self_reach" and self.reach_top is not None:
                if self.ignore_self_reach or other.ignore_self_reach:
                    out[name] = 0.0
                    continue
                top = self.reach_top
                if other.reach_top is not None:
                    top = min(top, other.reach_top)
                a = a[:top]
                b = b[:top]
            out[name] = (a - b).abs().max().item()
        return out


class GraphedCFRIteration:
    """Captures ``evaluator.cfr_iteration`` into a CUDA graph, replayable for
    any ``t`` that falls in the same Python-branch regime as ``t_capture``.

    Usage::

        runner = GraphedCFRIteration(evaluator)
        runner.capture(t_warmup=warm_start, t_capture=warm_start + 1)
        runner.replay(t=warm_start + 2)        # runs iteration t=warm_start+2
        runner.replay(t=warm_start + 3)        # ...and so on

    Capture executes two *real* CFR iterations against evaluator state: one
    on the capture stream (used to JIT-compile kernels and seed the graph
    allocator pool) and one inside the graph itself (recorded for replay).
    Callers are responsible for sequencing surrounding iterations so the
    state mutation from these two iters is the intended forward progress.

    On replay, host-side Python schedules are re-applied and the evaluator's
    ``TScalars`` device tensors are refreshed *outside* the captured region,
    so the kernels (which read scalars via pointers) pick up the new ``t``.

    Constraints on ``t`` values used for replay:
      - Must follow the same Python-level branches as ``t_capture`` (e.g.,
        both on the same side of ``dcfr_delay`` / both ``t > 1`` so
        ``last_model_values`` is populated). Same constraint applies to
        ``t_warmup`` vs ``t_capture``.
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

    def capture(self, t_warmup: int, t_capture: int) -> None:
        """Warm up the capture stream by running ``cfr_iteration(t_warmup)``
        on it, then capture ``cfr_iteration(t_capture)`` into a CUDA graph.

        Only the warmup iteration mutates evaluator state — CUDA graph
        capture is record-only, so the kernels issued under the capture
        context are recorded but not executed. The warmup iter therefore
        doubles as a real CFR step at ``t_warmup``; the captured graph
        encodes work for ``t_capture`` and runs it on each ``replay()``.

        ``_record_stats`` is stubbed for the warmup because it contains
        ``.item()`` calls that are incompatible with graph capture (and
        would also stub during capture); callers must choose
        ``t_warmup`` / ``t_capture`` so neither coincides with a
        stat-recording iteration.
        """
        ev = self.evaluator
        if not hasattr(ev, "_t_scalars"):
            raise ValueError(
                "Evaluator must be a FusedSparseCFREvaluator (or have a "
                "._t_scalars TScalars holder) for graph capture."
            )

        ev._record_stats = self._stub_record_stats
        prev_skip_stats = getattr(ev, "_skip_record_stats", False)
        ev._skip_record_stats = True
        prev_skip_scalars = ev._skip_t_scalars_update
        ev._skip_t_scalars_update = True
        try:
            s = torch.cuda.Stream()

            # Warmup phase: real iter at t_warmup on the capture stream. Fill
            # TScalars on the main stream first, then have s wait so its
            # kernels read the freshly written values.
            ev.prepare_replay(t_warmup)
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                ev.cfr_iteration(t_warmup)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            # Capture phase: real iter at t_capture recorded into the graph.
            # Replay later overwrites TScalars before each replay() call, so
            # the kernels (which load scalars via pointers) pick up the new t.
            ev.prepare_replay(t_capture)
            s.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=s):
                ev.cfr_iteration(t_capture)
        finally:
            ev._skip_t_scalars_update = prev_skip_scalars
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


# ---------------------------------------------------------------------------
# Showdown EV in Triton — replaces the torch.compile()'d
# CFREvaluator._showdown_value_both with a 3-kernel pipeline.
#
# The PyTorch baseline materializes a [M, H, 52] per_card_mass cumsum
# (~96% zeros, since each villain hand contains exactly 2 of 52 cards) and
# then does four large gathers on it. We exploit the sparsity:
#
#   * Subgame setup (precompute_showdown_extras, called once per
#     subgame): for each (env m, card c) record the sorted j-positions
#     where card c appears in `hands_c1c2_sorted` — `card_positions[M, 52,
#     MC=64]` padded with H. From those, derive six [M, H] int32
#     `slot_*` tensors that count how many of card c1[k]/c2[k]'s
#     positions are strictly less than L[k] / R[k] / H. All six slot
#     tensors are pure functions of the tree structure (no belief
#     dependence) and stay valid for every CFR iteration on the subgame.
#
#   * Per call (showdown_ev_v15, three Triton kernels):
#       1. _showdown_setup_b_P_kernel — gather `b_opp_sorted` along
#          `sorted_indices` for both heroes and build P_padded (cumsum
#          with leading 0).
#       2. _showdown_build_cum_kernel — per-card cumsum on the SPARSE
#          positions only, output shape [M, 2, 52, MC]. Bandwidth ~200×
#          smaller than the PyTorch [M, H, 52] tensor.
#       3. _showdown_ev_v15_kernel — per (m, k) program does six SCALAR
#          loads from `card_cumsum` at the precomputed slot offsets
#          (instead of loading [BLOCK_K, MC] blocks and reducing them),
#          computes win/tie/loss EV for both heroes via tl.static_range,
#          multiplies by the precomputed hand_ok_sorted × scale_factor,
#          and scatters into `ev_out[m, hero, sorted_indices[m, k]]`.
#
#   * ShowdownGraphRunner wraps the 3-kernel sequence in a CUDA graph
#     keyed on the subgame's (M, NUM_HANDS); replay does a single
#     `copy_()` into a persistent input buffer. Captured in
#     FusedSparseCFREvaluator._init_hand_rank_data.
# ---------------------------------------------------------------------------


SHOWDOWN_MAX_PER_CARD = 64  # power-of-2 padding; non-board cards have 51 each
SHOWDOWN_RIVER_ACTIVE_HANDS = 1081
SHOWDOWN_RIVER_ACTIVE_CARDS = 47


def precompute_showdown_card_positions(
    hands_c1c2_sorted: torch.Tensor,
    num_cards: int = 52,
    max_per_card: int = SHOWDOWN_MAX_PER_CARD,
) -> torch.Tensor:
    """For each (env m, card c), find sorted j-positions where c appears
    in hands_c1c2_sorted. Returns [M, 52, max_per_card] int64 padded with H.
    Computed once per subgame; reused across CFR iters."""
    M, H, _ = hands_c1c2_sorted.shape
    device = hands_c1c2_sorted.device
    cards = torch.arange(num_cards, device=device).view(1, 1, num_cards)
    c1 = hands_c1c2_sorted[..., 0]
    c2 = hands_c1c2_sorted[..., 1]
    incidence = (c1.unsqueeze(-1) == cards) | (c2.unsqueeze(-1) == cards)
    slots = incidence.cumsum(dim=1, dtype=torch.int32)
    slot1 = slots.gather(2, c1.unsqueeze(-1)).squeeze(-1) - 1
    slot2 = slots.gather(2, c2.unsqueeze(-1)).squeeze(-1) - 1

    out = torch.full(
        (M * num_cards, max_per_card),
        H,
        dtype=torch.long,
        device=device,
    )
    row_base = torch.arange(M, device=device, dtype=torch.long).view(M, 1) * num_cards
    rows1 = (row_base + c1.to(torch.long)).reshape(-1)
    rows2 = (row_base + c2.to(torch.long)).reshape(-1)
    positions = torch.arange(H, device=device, dtype=torch.long).view(1, H).expand(M, H)
    out[rows1, slot1.reshape(-1).to(torch.long)] = positions.reshape(-1)
    out[rows2, slot2.reshape(-1).to(torch.long)] = positions.reshape(-1)
    return out.view(M, num_cards, max_per_card)


def precompute_showdown_lookup_slots(
    card_positions: torch.Tensor,  # [M, 52, MC] padded with H
    L_idx: torch.Tensor,  # [M, H]
    R_idx: torch.Tensor,
    hands_c1c2_sorted: torch.Tensor,  # [M, H, 2]
) -> tuple[torch.Tensor, ...]:
    """Per (m, k), per lookup type, count #positions of c1[k]/c2[k] that are
    strictly less than L[k] / R[k] / H. Returns six [M, H] int32 tensors:
    slot_L_c1, slot_L_c2, slot_R_c1, slot_R_c2, slot_last_c1, slot_last_c2."""
    _, _, mc = card_positions.shape
    H = L_idx.shape[1]
    c1 = hands_c1c2_sorted[..., 0]
    c2 = hands_c1c2_sorted[..., 1]
    pos_for_c1 = card_positions.gather(1, c1.unsqueeze(-1).expand(-1, -1, mc))
    pos_for_c2 = card_positions.gather(1, c2.unsqueeze(-1).expand(-1, -1, mc))
    H_t = torch.full_like(L_idx, H)
    return (
        (pos_for_c1 < L_idx.unsqueeze(-1)).sum(dim=-1).to(torch.int32),
        (pos_for_c2 < L_idx.unsqueeze(-1)).sum(dim=-1).to(torch.int32),
        (pos_for_c1 < R_idx.unsqueeze(-1)).sum(dim=-1).to(torch.int32),
        (pos_for_c2 < R_idx.unsqueeze(-1)).sum(dim=-1).to(torch.int32),
        (pos_for_c1 < H_t.unsqueeze(-1)).sum(dim=-1).to(torch.int32),
        (pos_for_c2 < H_t.unsqueeze(-1)).sum(dim=-1).to(torch.int32),
    )


def precompute_showdown_card_positions_and_lookup_slots(
    hands_c1c2_sorted: torch.Tensor,
    L_idx: torch.Tensor,
    R_idx: torch.Tensor,
    num_cards: int = 52,
    max_per_card: int = SHOWDOWN_MAX_PER_CARD,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Build per-card sorted positions and lookup slots from one cumsum pass."""
    M, H, _ = hands_c1c2_sorted.shape
    device = hands_c1c2_sorted.device
    c1 = hands_c1c2_sorted[..., 0].to(torch.long)
    c2 = hands_c1c2_sorted[..., 1].to(torch.long)
    cards = torch.arange(num_cards, device=device).view(1, 1, num_cards)
    incidence = (c1.unsqueeze(-1) == cards) | (c2.unsqueeze(-1) == cards)
    slots = incidence.cumsum(dim=1, dtype=torch.int32)
    slot1 = slots.gather(2, c1.unsqueeze(-1)).squeeze(-1) - 1
    slot2 = slots.gather(2, c2.unsqueeze(-1)).squeeze(-1) - 1

    card_positions = torch.full(
        (M * num_cards, max_per_card),
        H,
        dtype=torch.long,
        device=device,
    )
    row_base = torch.arange(M, device=device, dtype=torch.long).view(M, 1) * num_cards
    rows1 = (row_base + c1).reshape(-1)
    rows2 = (row_base + c2).reshape(-1)
    positions = torch.arange(H, device=device, dtype=torch.long).view(1, H).expand(M, H)
    card_positions[rows1, slot1.reshape(-1).to(torch.long)] = positions.reshape(-1)
    card_positions[rows2, slot2.reshape(-1).to(torch.long)] = positions.reshape(-1)
    card_positions = card_positions.view(M, num_cards, max_per_card)

    m_idx = torch.arange(M, device=device).view(M, 1).expand(M, H)
    safe_l = (L_idx - 1).clamp(min=0).to(torch.long)
    safe_r = (R_idx - 1).clamp(min=0).to(torch.long)
    has_l = L_idx > 0
    has_r = R_idx > 0
    slot_L_c1 = torch.where(
        has_l, slots[m_idx, safe_l, c1], torch.zeros_like(L_idx, dtype=torch.int32)
    )
    slot_L_c2 = torch.where(
        has_l, slots[m_idx, safe_l, c2], torch.zeros_like(L_idx, dtype=torch.int32)
    )
    slot_R_c1 = torch.where(
        has_r, slots[m_idx, safe_r, c1], torch.zeros_like(R_idx, dtype=torch.int32)
    )
    slot_R_c2 = torch.where(
        has_r, slots[m_idx, safe_r, c2], torch.zeros_like(R_idx, dtype=torch.int32)
    )
    counts = slots[:, -1, :]
    slot_last_c1 = counts.gather(1, c1).to(torch.int32)
    slot_last_c2 = counts.gather(1, c2).to(torch.int32)
    return card_positions, (
        slot_L_c1,
        slot_L_c2,
        slot_R_c1,
        slot_R_c2,
        slot_last_c1,
        slot_last_c2,
    )


if triton is not None:

    @triton.jit
    def _showdown_setup_b_P_kernel(
        beliefs_ptr,  # [M, 2, NUM_HANDS] fp32 (orig hand order)
        extra_index_ptr,  # [M] int64, maps showdown row -> board-extra row
        sorted_indices_ptr,  # [E, H] int64
        b_opp_both_ptr,  # [M, 2, H] fp32 OUT
        P_padded_both_ptr,  # [M, 2, H+1] fp32 OUT (with leading 0)
        H,
        NUM_HANDS,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        villain = 1 - hero
        extra = tl.load(extra_index_ptr + m)
        h_offs = tl.arange(0, BLOCK_H)
        mask_h = h_offs < H
        sorted_idx = tl.load(
            sorted_indices_ptr + extra * H + h_offs,
            mask=mask_h,
            other=0,
        )
        safe_idx = tl.where(mask_h, sorted_idx, 0)
        b_at = tl.load(
            beliefs_ptr + m * 2 * NUM_HANDS + villain * NUM_HANDS + safe_idx,
            mask=mask_h,
            other=0.0,
        )
        tl.store(
            b_opp_both_ptr + m * 2 * H + hero * H + h_offs,
            b_at,
            mask=mask_h,
        )
        cum = tl.cumsum(b_at, axis=0)
        tl.store(
            P_padded_both_ptr + m * 2 * (H + 1) + hero * (H + 1) + h_offs + 1,
            cum,
            mask=mask_h,
        )
        tl.store(P_padded_both_ptr + m * 2 * (H + 1) + hero * (H + 1) + 0, 0.0)

    @triton.jit
    def _showdown_setup_b_P_compact_kernel(
        beliefs_ptr,  # [M, 2, H] fp32 in active sorted order
        b_opp_both_ptr,  # [M, 2, H] fp32 OUT
        P_padded_both_ptr,  # [M, 2, H+1] fp32 OUT (with leading 0)
        H,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        villain = 1 - hero
        h_offs = tl.arange(0, BLOCK_H)
        mask_h = h_offs < H
        b_at = tl.load(
            beliefs_ptr + m * 2 * H + villain * H + h_offs,
            mask=mask_h,
            other=0.0,
        )
        tl.store(
            b_opp_both_ptr + m * 2 * H + hero * H + h_offs,
            b_at,
            mask=mask_h,
        )
        cum = tl.cumsum(b_at, axis=0)
        tl.store(
            P_padded_both_ptr + m * 2 * (H + 1) + hero * (H + 1) + h_offs + 1,
            cum,
            mask=mask_h,
        )
        tl.store(P_padded_both_ptr + m * 2 * (H + 1) + hero * (H + 1) + 0, 0.0)

    @triton.jit
    def _showdown_setup_P_compact_kernel(
        beliefs_ptr,  # [M, 2, H] fp32 in active sorted order
        P_padded_both_ptr,  # [M, 2, H+1] fp32 OUT (with leading 0)
        H,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        villain = 1 - hero
        h_offs = tl.arange(0, BLOCK_H)
        mask_h = h_offs < H
        b_at = tl.load(
            beliefs_ptr + m * 2 * H + villain * H + h_offs,
            mask=mask_h,
            other=0.0,
        )
        cum = tl.cumsum(b_at, axis=0)
        tl.store(
            P_padded_both_ptr + m * 2 * (H + 1) + hero * (H + 1) + h_offs + 1,
            cum,
            mask=mask_h,
        )
        tl.store(P_padded_both_ptr + m * 2 * (H + 1) + hero * (H + 1) + 0, 0.0)

    @triton.jit
    def _showdown_build_cum_kernel(
        b_opp_sorted_ptr,  # [M, 2, H] fp32
        extra_index_ptr,  # [M] int64
        card_positions_ptr,  # [E, NUM_CARDS, MC] int32 (positions; pad=H)
        cum_out_ptr,  # [M, 2, NUM_CARDS, MC] fp32 OUT
        H,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        hc = tl.program_id(1)
        hero = hc // NUM_CARDS
        c = hc % NUM_CARDS
        extra = tl.load(extra_index_ptr + m)
        slot_off = tl.arange(0, MC_K)
        positions = tl.load(
            card_positions_ptr + extra * NUM_CARDS * MC_K + c * MC_K + slot_off,
        )
        in_range = positions < H
        safe_pos = tl.where(in_range, positions, 0)
        b_at = tl.load(
            b_opp_sorted_ptr + m * 2 * H + hero * H + safe_pos,
        )
        b_at = tl.where(in_range, b_at, 0.0)
        cum = tl.cumsum(b_at, axis=0)
        tl.store(
            cum_out_ptr
            + m * 2 * NUM_CARDS * MC_K
            + hero * NUM_CARDS * MC_K
            + c * MC_K
            + slot_off,
            cum,
        )

    @triton.jit
    def _showdown_build_cum_compact_kernel(
        beliefs_ptr,  # [M, 2, H] compact active sorted beliefs
        extra_index_ptr,  # [M] int64
        card_positions_ptr,  # [E, NUM_CARDS, MC] int32 (positions; pad=H)
        cum_out_ptr,  # [M, 2, NUM_CARDS, MC] fp32 OUT
        H,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        hc = tl.program_id(1)
        hero = hc // NUM_CARDS
        villain = 1 - hero
        c = hc % NUM_CARDS
        extra = tl.load(extra_index_ptr + m)
        slot_off = tl.arange(0, MC_K)
        positions = tl.load(
            card_positions_ptr + extra * NUM_CARDS * MC_K + c * MC_K + slot_off,
        )
        in_range = positions < H
        safe_pos = tl.where(in_range, positions, 0)
        b_at = tl.load(
            beliefs_ptr + m * 2 * H + villain * H + safe_pos,
            mask=in_range,
            other=0.0,
        )
        cum = tl.cumsum(b_at, axis=0)
        tl.store(
            cum_out_ptr
            + m * 2 * NUM_CARDS * MC_K
            + hero * NUM_CARDS * MC_K
            + c * MC_K
            + slot_off,
            cum,
        )

    @triton.jit
    def _showdown_build_cum_compact_both_heroes_kernel(
        beliefs_ptr,  # [M, 2, H] compact active sorted beliefs
        extra_index_ptr,  # [M] int64
        card_positions_ptr,  # [E, NUM_CARDS, MC] int32 (positions; pad=H)
        cum_out_ptr,  # [M, 2, NUM_CARDS, MC] fp32 OUT
        H,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        c = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        slot_off = tl.arange(0, MC_K)
        positions = tl.load(
            card_positions_ptr + extra * NUM_CARDS * MC_K + c * MC_K + slot_off,
        )
        in_range = positions < H
        safe_pos = tl.where(in_range, positions, 0)
        out_base = cum_out_ptr + m * 2 * NUM_CARDS * MC_K + c * MC_K
        for hero in tl.static_range(2):
            villain = 1 - hero
            b_at = tl.load(
                beliefs_ptr + m * 2 * H + villain * H + safe_pos,
                mask=in_range,
                other=0.0,
            )
            cum = tl.cumsum(b_at, axis=0)
            tl.store(out_base + hero * NUM_CARDS * MC_K + slot_off, cum)

    @triton.jit
    def _showdown_build_cum_compact_card_block_kernel(
        beliefs_ptr,  # [M, 2, H] compact active sorted beliefs
        extra_index_ptr,  # [M] int64
        card_positions_ptr,  # [E, NUM_CARDS, MC] int32 (positions; pad=H)
        cum_out_ptr,  # [M, 2, NUM_CARDS, MC] fp32 OUT
        H,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        CARD_BLOCK: tl.constexpr,
    ):
        m = tl.program_id(0)
        card_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        slot_off = tl.arange(0, MC_K)[:, None]
        card_off = tl.arange(0, CARD_BLOCK)[None, :]
        cards = card_block * CARD_BLOCK + card_off
        card_mask = cards < NUM_CARDS
        positions = tl.load(
            card_positions_ptr + extra * NUM_CARDS * MC_K + cards * MC_K + slot_off,
            mask=card_mask,
            other=H,
        )
        in_range = positions < H
        safe_pos = tl.where(in_range, positions, 0)
        out_base = cum_out_ptr + m * 2 * NUM_CARDS * MC_K + cards * MC_K + slot_off
        for hero in tl.static_range(2):
            villain = 1 - hero
            b_at = tl.load(
                beliefs_ptr + m * 2 * H + villain * H + safe_pos,
                mask=in_range & card_mask,
                other=0.0,
            )
            cum = tl.cumsum(b_at, axis=0)
            tl.store(out_base + hero * NUM_CARDS * MC_K, cum, mask=card_mask)

    @triton.jit
    def _showdown_setup_P_active_full_kernel(
        beliefs_ptr,  # [M, 2, NUM_HANDS]
        extra_index_ptr,  # [M]
        sorted_indices_ptr,  # [E, H_ACTIVE]
        P_padded_both_ptr,  # [M, 2, H_ACTIVE+1] OUT
        H_ACTIVE,
        NUM_HANDS,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        villain = 1 - hero
        extra = tl.load(extra_index_ptr + m)
        h_offs = tl.arange(0, BLOCK_H)
        mask_h = h_offs < H_ACTIVE
        hand_idx = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + h_offs, mask=mask_h, other=0
        )
        b_at = tl.load(
            beliefs_ptr + m * 2 * NUM_HANDS + villain * NUM_HANDS + hand_idx,
            mask=mask_h,
            other=0.0,
        )
        cum = tl.cumsum(b_at, axis=0)
        tl.store(
            P_padded_both_ptr
            + m * 2 * (H_ACTIVE + 1)
            + hero * (H_ACTIVE + 1)
            + h_offs
            + 1,
            cum,
            mask=mask_h,
        )
        tl.store(
            P_padded_both_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1),
            0.0,
        )

    @triton.jit
    def _showdown_build_cum_active_full_card_block_kernel(
        beliefs_ptr,  # [M, 2, NUM_HANDS]
        extra_index_ptr,  # [M]
        sorted_indices_ptr,  # [E, H_ACTIVE]
        card_positions_ptr,  # [E, NUM_CARDS, MC] active positions; pad=H_ACTIVE
        cum_out_ptr,  # [M, 2, NUM_CARDS, MC] OUT
        H_ACTIVE,
        NUM_HANDS,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        CARD_BLOCK: tl.constexpr,
    ):
        m = tl.program_id(0)
        card_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        slot_off = tl.arange(0, MC_K)[:, None]
        card_off = tl.arange(0, CARD_BLOCK)[None, :]
        cards = card_block * CARD_BLOCK + card_off
        card_mask = cards < NUM_CARDS
        positions = tl.load(
            card_positions_ptr + extra * NUM_CARDS * MC_K + cards * MC_K + slot_off,
            mask=card_mask,
            other=H_ACTIVE,
        )
        in_range = positions < H_ACTIVE
        safe_pos = tl.where(in_range, positions, 0)
        hand_idx = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + safe_pos,
            mask=in_range & card_mask,
            other=0,
        )
        out_base = cum_out_ptr + m * 2 * NUM_CARDS * MC_K + cards * MC_K + slot_off
        for hero in tl.static_range(2):
            villain = 1 - hero
            b_at = tl.load(
                beliefs_ptr + m * 2 * NUM_HANDS + villain * NUM_HANDS + hand_idx,
                mask=in_range & card_mask,
                other=0.0,
            )
            cum = tl.cumsum(b_at, axis=0)
            tl.store(out_base + hero * NUM_CARDS * MC_K, cum, mask=card_mask)

    @triton.jit
    def _showdown_setup_P_active_full_indexed_kernel(
        beliefs_ptr,  # [N, 2, NUM_HANDS]
        row_index_ptr,  # [M]
        extra_index_ptr,  # [M]
        sorted_indices_ptr,  # [E, H_ACTIVE]
        P_padded_both_ptr,  # [M, 2, H_ACTIVE+1] OUT
        H_ACTIVE,
        NUM_HANDS,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        villain = 1 - hero
        src_row = tl.load(row_index_ptr + m)
        extra = tl.load(extra_index_ptr + m)
        h_offs = tl.arange(0, BLOCK_H)
        mask_h = h_offs < H_ACTIVE
        hand_idx = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + h_offs, mask=mask_h, other=0
        )
        b_at = tl.load(
            beliefs_ptr + src_row * 2 * NUM_HANDS + villain * NUM_HANDS + hand_idx,
            mask=mask_h,
            other=0.0,
        )
        cum = tl.cumsum(b_at, axis=0)
        tl.store(
            P_padded_both_ptr
            + m * 2 * (H_ACTIVE + 1)
            + hero * (H_ACTIVE + 1)
            + h_offs
            + 1,
            cum,
            mask=mask_h,
        )
        tl.store(
            P_padded_both_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1),
            0.0,
        )

    @triton.jit
    def _showdown_build_cum_active_full_card_block_indexed_kernel(
        beliefs_ptr,  # [N, 2, NUM_HANDS]
        row_index_ptr,  # [M]
        extra_index_ptr,  # [M]
        sorted_indices_ptr,  # [E, H_ACTIVE]
        card_positions_ptr,  # [E, NUM_CARDS, MC] active positions; pad=H_ACTIVE
        cum_out_ptr,  # [M, 2, NUM_CARDS, MC] OUT
        H_ACTIVE,
        NUM_HANDS,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        CARD_BLOCK: tl.constexpr,
    ):
        m = tl.program_id(0)
        card_block = tl.program_id(1)
        src_row = tl.load(row_index_ptr + m)
        extra = tl.load(extra_index_ptr + m)
        slot_off = tl.arange(0, MC_K)[:, None]
        card_off = tl.arange(0, CARD_BLOCK)[None, :]
        cards = card_block * CARD_BLOCK + card_off
        card_mask = cards < NUM_CARDS
        positions = tl.load(
            card_positions_ptr + extra * NUM_CARDS * MC_K + cards * MC_K + slot_off,
            mask=card_mask,
            other=H_ACTIVE,
        )
        in_range = positions < H_ACTIVE
        safe_pos = tl.where(in_range, positions, 0)
        hand_idx = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + safe_pos,
            mask=in_range & card_mask,
            other=0,
        )
        out_base = cum_out_ptr + m * 2 * NUM_CARDS * MC_K + cards * MC_K + slot_off
        for hero in tl.static_range(2):
            villain = 1 - hero
            b_at = tl.load(
                beliefs_ptr + src_row * 2 * NUM_HANDS + villain * NUM_HANDS + hand_idx,
                mask=in_range & card_mask,
                other=0.0,
            )
            cum = tl.cumsum(b_at, axis=0)
            tl.store(out_base + hero * NUM_CARDS * MC_K, cum, mask=card_mask)

    @triton.jit
    def _showdown_gather_active_beliefs_kernel(
        beliefs_ptr,  # [M, 2, NUM_HANDS]
        extra_index_ptr,  # [M]
        sorted_indices_ptr,  # [E, H_ACTIVE]
        compact_ptr,  # [M, 2, H_ACTIVE] OUT
        H_ACTIVE,
        NUM_HANDS,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        k_block = tl.program_id(2)
        extra = tl.load(extra_index_ptr + m)
        k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < H_ACTIVE
        hand_idx = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + k, mask=mask, other=0
        )
        values = tl.load(
            beliefs_ptr + m * 2 * NUM_HANDS + hero * NUM_HANDS + hand_idx,
            mask=mask,
            other=0.0,
        )
        tl.store(
            compact_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k, values, mask=mask
        )

    @triton.jit
    def _showdown_scatter_active_values_kernel(
        compact_ptr,  # [M, 2, H_ACTIVE]
        extra_index_ptr,  # [M]
        sorted_indices_ptr,  # [E, H_ACTIVE]
        ev_out_ptr,  # [M, 2, NUM_HANDS] OUT
        H_ACTIVE,
        NUM_HANDS,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        k_block = tl.program_id(2)
        extra = tl.load(extra_index_ptr + m)
        k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < H_ACTIVE
        hand_idx = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + k, mask=mask, other=0
        )
        values = tl.load(
            compact_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k,
            mask=mask,
            other=0.0,
        )
        tl.store(
            ev_out_ptr + m * 2 * NUM_HANDS + hero * NUM_HANDS + hand_idx,
            values,
            mask=mask,
        )

    @triton.jit
    def _copy_indexed_belief_rows_kernel(
        src_ptr,  # [N, 2, NUM_HANDS]
        row_index_ptr,  # [M]
        dst_ptr,  # [M, 2, NUM_HANDS]
        NUM_HANDS,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        h_block = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = h < NUM_HANDS
        src_row = tl.load(row_index_ptr + m)
        values = tl.load(
            src_ptr + src_row * 2 * NUM_HANDS + hero * NUM_HANDS + h,
            mask=mask,
            other=0.0,
        )
        tl.store(
            dst_ptr + m * 2 * NUM_HANDS + hero * NUM_HANDS + h,
            values,
            mask=mask,
        )

    @triton.jit
    def _copy_rows_to_indexed_values_kernel(
        src_ptr,  # [M, 2, NUM_HANDS]
        row_index_ptr,  # [M]
        dst_ptr,  # [N, 2, NUM_HANDS]
        NUM_HANDS,
        BLOCK_H: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        h_block = tl.program_id(2)
        h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = h < NUM_HANDS
        dst_row = tl.load(row_index_ptr + m)
        values = tl.load(
            src_ptr + m * 2 * NUM_HANDS + hero * NUM_HANDS + h,
            mask=mask,
            other=0.0,
        )
        tl.store(
            dst_ptr + dst_row * 2 * NUM_HANDS + hero * NUM_HANDS + h,
            values,
            mask=mask,
        )

    @triton.jit
    def _showdown_ev_v15_kernel(
        b_opp_sorted_ptr,  # [M, 2, H]
        P_padded_ptr,  # [M, 2, H+1]
        card_cumsum_ptr,  # [M, 2, 52, MC]
        extra_index_ptr,  # [M] int64
        L_idx_ptr,
        R_idx_ptr,
        c1_sorted_ptr,
        c2_sorted_ptr,
        sorted_indices_ptr,  # [E, H] — write target index
        hand_ok_sorted_ptr,  # [E, H] uint8
        scale_factor_ptr,  # [M, 2] — potential / scale
        slot_L_c1_ptr,
        slot_L_c2_ptr,
        slot_R_c1_ptr,
        slot_R_c2_ptr,
        slot_last_c1_ptr,
        slot_last_c2_ptr,
        ev_out_ptr,  # [M, 2, H] in unsorted order
        H,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H

        L = tl.load(L_idx_ptr + extra * H + k_offs, mask=mask_k, other=0)
        R = tl.load(R_idx_ptr + extra * H + k_offs, mask=mask_k, other=0)
        c1 = tl.load(c1_sorted_ptr + extra * H + k_offs, mask=mask_k, other=0)
        c2 = tl.load(c2_sorted_ptr + extra * H + k_offs, mask=mask_k, other=0)
        out_k = tl.load(sorted_indices_ptr + extra * H + k_offs, mask=mask_k, other=0)
        ok_int = tl.load(hand_ok_sorted_ptr + extra * H + k_offs, mask=mask_k, other=0)
        sL1 = tl.load(slot_L_c1_ptr + extra * H + k_offs, mask=mask_k, other=0)
        sL2 = tl.load(slot_L_c2_ptr + extra * H + k_offs, mask=mask_k, other=0)
        sR1 = tl.load(slot_R_c1_ptr + extra * H + k_offs, mask=mask_k, other=0)
        sR2 = tl.load(slot_R_c2_ptr + extra * H + k_offs, mask=mask_k, other=0)
        sLast1 = tl.load(slot_last_c1_ptr + extra * H + k_offs, mask=mask_k, other=0)
        sLast2 = tl.load(slot_last_c2_ptr + extra * H + k_offs, mask=mask_k, other=0)

        has_L1 = sL1 > 0
        has_L2 = sL2 > 0
        has_R1 = sR1 > 0
        has_R2 = sR2 > 0
        has_Last1 = sLast1 > 0
        has_Last2 = sLast2 > 0
        iL1 = tl.maximum(sL1 - 1, 0)
        iL2 = tl.maximum(sL2 - 1, 0)
        iR1 = tl.maximum(sR1 - 1, 0)
        iR2 = tl.maximum(sR2 - 1, 0)
        iLast1 = tl.maximum(sLast1 - 1, 0)
        iLast2 = tl.maximum(sLast2 - 1, 0)
        ok_factor = ok_int.to(tl.float32)

        for hero in tl.static_range(2):
            b_at_k = tl.load(
                b_opp_sorted_ptr + m * 2 * H + hero * H + k_offs, mask=mask_k, other=0.0
            )
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L,
                mask=mask_k,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R,
                mask=mask_k,
                other=0.0,
            )
            cum_base = (
                card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            )
            Pcards_L_c1 = tl.where(
                has_L1, tl.load(cum_base + c1 * MC_K + iL1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_L_c2 = tl.where(
                has_L2, tl.load(cum_base + c2 * MC_K + iL2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c1 = tl.where(
                has_R1, tl.load(cum_base + c1 * MC_K + iR1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c2 = tl.where(
                has_R2, tl.load(cum_base + c2 * MC_K + iR2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_last_c1 = tl.where(
                has_Last1,
                tl.load(cum_base + c1 * MC_K + iLast1, mask=mask_k, other=0.0),
                0.0,
            )
            Pcards_last_c2 = tl.where(
                has_Last2,
                tl.load(cum_base + c2 * MC_K + iLast2, mask=mask_k, other=0.0),
                0.0,
            )

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )

            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                ev_out_ptr + m * 2 * H + hero * H + out_k,
                EV * ok_factor * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_ev_active_kernel(
        b_opp_sorted_ptr,  # [M, 2, H_ACTIVE]
        P_padded_ptr,  # [M, 2, H_ACTIVE+1]
        card_cumsum_ptr,  # [M, 2, NUM_CARDS, MC]
        extra_index_ptr,  # [M] int64
        L_idx_ptr,
        R_idx_ptr,
        c1_sorted_ptr,
        c2_sorted_ptr,
        sorted_indices_ptr,  # [E, H_ACTIVE] — write target full-hand index
        scale_factor_ptr,  # [M, 2] — potential / scale
        slot_L_c1_ptr,
        slot_L_c2_ptr,
        slot_R_c1_ptr,
        slot_R_c2_ptr,
        slot_last_c1_ptr,
        slot_last_c2_ptr,
        ev_out_ptr,  # [M, 2, NUM_HANDS] in unsorted full-hand order
        H_ACTIVE,
        NUM_HANDS,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H_ACTIVE

        L = tl.load(L_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        R = tl.load(R_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        c1 = tl.load(c1_sorted_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        c2 = tl.load(c2_sorted_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        out_k = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        )
        sL1 = tl.load(
            slot_L_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sL2 = tl.load(
            slot_L_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sR1 = tl.load(
            slot_R_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sR2 = tl.load(
            slot_R_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sLast1 = tl.load(
            slot_last_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sLast2 = tl.load(
            slot_last_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)

        has_L1 = sL1 > 0
        has_L2 = sL2 > 0
        has_R1 = sR1 > 0
        has_R2 = sR2 > 0
        has_Last1 = sLast1 > 0
        has_Last2 = sLast2 > 0
        iL1 = tl.maximum(sL1 - 1, 0)
        iL2 = tl.maximum(sL2 - 1, 0)
        iR1 = tl.maximum(sR1 - 1, 0)
        iR2 = tl.maximum(sR2 - 1, 0)
        iLast1 = tl.maximum(sLast1 - 1, 0)
        iLast2 = tl.maximum(sLast2 - 1, 0)

        for hero in tl.static_range(2):
            b_at_k = tl.load(
                b_opp_sorted_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k_offs,
                mask=mask_k,
                other=0.0,
            )
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + L,
                mask=mask_k,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + R,
                mask=mask_k,
                other=0.0,
            )
            cum_base = (
                card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            )
            Pcards_L_c1 = tl.where(
                has_L1, tl.load(cum_base + c1 * MC_K + iL1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_L_c2 = tl.where(
                has_L2, tl.load(cum_base + c2 * MC_K + iL2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c1 = tl.where(
                has_R1, tl.load(cum_base + c1 * MC_K + iR1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c2 = tl.where(
                has_R2, tl.load(cum_base + c2 * MC_K + iR2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_last_c1 = tl.where(
                has_Last1,
                tl.load(cum_base + c1 * MC_K + iLast1, mask=mask_k, other=0.0),
                0.0,
            )
            Pcards_last_c2 = tl.where(
                has_Last2,
                tl.load(cum_base + c2 * MC_K + iLast2, mask=mask_k, other=0.0),
                0.0,
            )

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )

            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                ev_out_ptr + m * 2 * NUM_HANDS + hero * NUM_HANDS + out_k,
                EV * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_ev_active_compact_kernel(
        b_opp_sorted_ptr,  # [M, 2, H_ACTIVE]
        P_padded_ptr,  # [M, 2, H_ACTIVE+1]
        card_cumsum_ptr,  # [M, 2, NUM_CARDS, MC]
        extra_index_ptr,  # [M] int64
        L_idx_ptr,
        R_idx_ptr,
        c1_sorted_ptr,
        c2_sorted_ptr,
        scale_factor_ptr,  # [M, 2] — potential / scale
        slot_L_c1_ptr,
        slot_L_c2_ptr,
        slot_R_c1_ptr,
        slot_R_c2_ptr,
        slot_last_c1_ptr,
        slot_last_c2_ptr,
        ev_out_ptr,  # [M, 2, H_ACTIVE] in active sorted order
        H_ACTIVE,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H_ACTIVE

        L = tl.load(L_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        R = tl.load(R_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        c1 = tl.load(c1_sorted_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        c2 = tl.load(c2_sorted_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        sL1 = tl.load(
            slot_L_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sL2 = tl.load(
            slot_L_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sR1 = tl.load(
            slot_R_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sR2 = tl.load(
            slot_R_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sLast1 = tl.load(
            slot_last_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        sLast2 = tl.load(
            slot_last_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)

        has_L1 = sL1 > 0
        has_L2 = sL2 > 0
        has_R1 = sR1 > 0
        has_R2 = sR2 > 0
        has_Last1 = sLast1 > 0
        has_Last2 = sLast2 > 0
        iL1 = tl.maximum(sL1 - 1, 0)
        iL2 = tl.maximum(sL2 - 1, 0)
        iR1 = tl.maximum(sR1 - 1, 0)
        iR2 = tl.maximum(sR2 - 1, 0)
        iLast1 = tl.maximum(sLast1 - 1, 0)
        iLast2 = tl.maximum(sLast2 - 1, 0)

        for hero in tl.static_range(2):
            b_at_k = tl.load(
                b_opp_sorted_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k_offs,
                mask=mask_k,
                other=0.0,
            )
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + L,
                mask=mask_k,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + R,
                mask=mask_k,
                other=0.0,
            )
            cum_base = (
                card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            )
            Pcards_L_c1 = tl.where(
                has_L1, tl.load(cum_base + c1 * MC_K + iL1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_L_c2 = tl.where(
                has_L2, tl.load(cum_base + c2 * MC_K + iL2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c1 = tl.where(
                has_R1, tl.load(cum_base + c1 * MC_K + iR1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c2 = tl.where(
                has_R2, tl.load(cum_base + c2 * MC_K + iR2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_last_c1 = tl.where(
                has_Last1,
                tl.load(cum_base + c1 * MC_K + iLast1, mask=mask_k, other=0.0),
                0.0,
            )
            Pcards_last_c2 = tl.where(
                has_Last2,
                tl.load(cum_base + c2 * MC_K + iLast2, mask=mask_k, other=0.0),
                0.0,
            )

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )

            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                ev_out_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k_offs,
                EV * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_ev_active_offset_compact_kernel(
        b_opp_sorted_ptr,  # [M, 2, H_ACTIVE]
        P_padded_ptr,  # [M, 2, H_ACTIVE+1]
        card_cumsum_ptr,  # [M, 2, NUM_CARDS, MC]
        extra_index_ptr,  # [M] int64
        L_idx_ptr,
        R_idx_ptr,
        scale_factor_ptr,  # [M, 2] — potential / scale
        off_L_c1_ptr,
        off_L_c2_ptr,
        off_R_c1_ptr,
        off_R_c2_ptr,
        off_last_c1_ptr,
        off_last_c2_ptr,
        ev_out_ptr,  # [M, 2, H_ACTIVE] in active sorted order
        H_ACTIVE,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H_ACTIVE

        L = tl.load(L_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        R = tl.load(R_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        off_L_c1 = tl.load(
            off_L_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_L_c2 = tl.load(
            off_L_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_R_c1 = tl.load(
            off_R_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_R_c2 = tl.load(
            off_R_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_last_c1 = tl.load(
            off_last_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_last_c2 = tl.load(
            off_last_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)

        has_L1 = off_L_c1 >= 0
        has_L2 = off_L_c2 >= 0
        has_R1 = off_R_c1 >= 0
        has_R2 = off_R_c2 >= 0
        has_last1 = off_last_c1 >= 0
        has_last2 = off_last_c2 >= 0
        idx_L_c1 = tl.maximum(off_L_c1, 0)
        idx_L_c2 = tl.maximum(off_L_c2, 0)
        idx_R_c1 = tl.maximum(off_R_c1, 0)
        idx_R_c2 = tl.maximum(off_R_c2, 0)
        idx_last_c1 = tl.maximum(off_last_c1, 0)
        idx_last_c2 = tl.maximum(off_last_c2, 0)

        for hero in tl.static_range(2):
            b_at_k = tl.load(
                b_opp_sorted_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k_offs,
                mask=mask_k,
                other=0.0,
            )
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + L,
                mask=mask_k,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + R,
                mask=mask_k,
                other=0.0,
            )
            cum_base = (
                card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            )
            Pcards_L_c1 = tl.where(
                has_L1, tl.load(cum_base + idx_L_c1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_L_c2 = tl.where(
                has_L2, tl.load(cum_base + idx_L_c2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c1 = tl.where(
                has_R1, tl.load(cum_base + idx_R_c1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c2 = tl.where(
                has_R2, tl.load(cum_base + idx_R_c2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_last_c1 = tl.where(
                has_last1,
                tl.load(cum_base + idx_last_c1, mask=mask_k, other=0.0),
                0.0,
            )
            Pcards_last_c2 = tl.where(
                has_last2,
                tl.load(cum_base + idx_last_c2, mask=mask_k, other=0.0),
                0.0,
            )

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )

            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                ev_out_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k_offs,
                EV * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_ev_active_offset_compact_nobopp_kernel(
        beliefs_ptr,  # [M, 2, H_ACTIVE] compact active sorted beliefs
        P_padded_ptr,  # [M, 2, H_ACTIVE+1]
        card_cumsum_ptr,  # [M, 2, NUM_CARDS, MC]
        extra_index_ptr,  # [M] int64
        L_idx_ptr,
        R_idx_ptr,
        scale_factor_ptr,  # [M, 2] — potential / scale
        off_L_c1_ptr,
        off_L_c2_ptr,
        off_R_c1_ptr,
        off_R_c2_ptr,
        off_last_c1_ptr,
        off_last_c2_ptr,
        ev_out_ptr,  # [M, 2, H_ACTIVE] in active sorted order
        H_ACTIVE,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H_ACTIVE

        L = tl.load(L_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        R = tl.load(R_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        off_L_c1 = tl.load(
            off_L_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_L_c2 = tl.load(
            off_L_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_R_c1 = tl.load(
            off_R_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_R_c2 = tl.load(
            off_R_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_last_c1 = tl.load(
            off_last_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_last_c2 = tl.load(
            off_last_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)

        has_L1 = off_L_c1 >= 0
        has_L2 = off_L_c2 >= 0
        has_R1 = off_R_c1 >= 0
        has_R2 = off_R_c2 >= 0
        has_last1 = off_last_c1 >= 0
        has_last2 = off_last_c2 >= 0
        idx_L_c1 = tl.maximum(off_L_c1, 0)
        idx_L_c2 = tl.maximum(off_L_c2, 0)
        idx_R_c1 = tl.maximum(off_R_c1, 0)
        idx_R_c2 = tl.maximum(off_R_c2, 0)
        idx_last_c1 = tl.maximum(off_last_c1, 0)
        idx_last_c2 = tl.maximum(off_last_c2, 0)

        for hero in tl.static_range(2):
            villain = 1 - hero
            b_at_k = tl.load(
                beliefs_ptr + m * 2 * H_ACTIVE + villain * H_ACTIVE + k_offs,
                mask=mask_k,
                other=0.0,
            )
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + L,
                mask=mask_k,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + R,
                mask=mask_k,
                other=0.0,
            )
            cum_base = (
                card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            )
            Pcards_L_c1 = tl.load(cum_base + idx_L_c1, mask=mask_k & has_L1, other=0.0)
            Pcards_L_c2 = tl.load(cum_base + idx_L_c2, mask=mask_k & has_L2, other=0.0)
            Pcards_R_c1 = tl.load(cum_base + idx_R_c1, mask=mask_k & has_R1, other=0.0)
            Pcards_R_c2 = tl.load(cum_base + idx_R_c2, mask=mask_k & has_R2, other=0.0)
            Pcards_last_c1 = tl.load(
                cum_base + idx_last_c1, mask=mask_k & has_last1, other=0.0
            )
            Pcards_last_c2 = tl.load(
                cum_base + idx_last_c2, mask=mask_k & has_last2, other=0.0
            )

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )

            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                ev_out_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k_offs,
                EV * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_ev_active_offset_full_nobopp_kernel(
        beliefs_ptr,  # [M, 2, NUM_HANDS]
        P_padded_ptr,  # [M, 2, H_ACTIVE+1]
        card_cumsum_ptr,  # [M, 2, NUM_CARDS, MC]
        extra_index_ptr,  # [M] int64
        sorted_indices_ptr,  # [E, H_ACTIVE]
        rank_bounds_packed_ptr,
        scale_factor_ptr,  # [M, 2] — potential / scale
        off_L_pair_ptr,
        off_R_pair_ptr,
        off_last_pair_ptr,
        ev_out_ptr,  # [M, 2, NUM_HANDS]
        H_ACTIVE,
        NUM_HANDS,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H_ACTIVE

        hand_idx = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        )
        rank_bounds = tl.load(
            rank_bounds_packed_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        L = rank_bounds & 0xFFFF
        R = (rank_bounds >> 16) & 0xFFFF

        off_L_pair = tl.load(
            off_L_pair_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        off_R_pair = tl.load(
            off_R_pair_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        off_last_pair = tl.load(
            off_last_pair_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        enc_L_c1 = off_L_pair & 0xFFFF
        enc_L_c2 = (off_L_pair >> 16) & 0xFFFF
        enc_R_c1 = off_R_pair & 0xFFFF
        enc_R_c2 = (off_R_pair >> 16) & 0xFFFF
        enc_last_c1 = off_last_pair & 0xFFFF
        enc_last_c2 = (off_last_pair >> 16) & 0xFFFF

        has_L1 = enc_L_c1 > 0
        has_L2 = enc_L_c2 > 0
        has_R1 = enc_R_c1 > 0
        has_R2 = enc_R_c2 > 0
        has_last1 = enc_last_c1 > 0
        has_last2 = enc_last_c2 > 0
        idx_L_c1 = tl.maximum(enc_L_c1 - 1, 0)
        idx_L_c2 = tl.maximum(enc_L_c2 - 1, 0)
        idx_R_c1 = tl.maximum(enc_R_c1 - 1, 0)
        idx_R_c2 = tl.maximum(enc_R_c2 - 1, 0)
        idx_last_c1 = tl.maximum(enc_last_c1 - 1, 0)
        idx_last_c2 = tl.maximum(enc_last_c2 - 1, 0)

        for hero in tl.static_range(2):
            p_base = P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1)
            P_k = tl.load(p_base + k_offs, mask=mask_k, other=0.0)
            P_k_next = tl.load(p_base + k_offs + 1, mask=mask_k, other=0.0)
            b_at_k = P_k_next - P_k
            P_L = tl.load(
                p_base + L,
                mask=mask_k,
                other=0.0,
            )
            P_R = tl.load(
                p_base + R,
                mask=mask_k,
                other=0.0,
            )
            cum_base = (
                card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            )
            Pcards_L_c1 = tl.load(cum_base + idx_L_c1, mask=mask_k & has_L1, other=0.0)
            Pcards_L_c2 = tl.load(cum_base + idx_L_c2, mask=mask_k & has_L2, other=0.0)
            Pcards_R_c1 = tl.load(cum_base + idx_R_c1, mask=mask_k & has_R1, other=0.0)
            Pcards_R_c2 = tl.load(cum_base + idx_R_c2, mask=mask_k & has_R2, other=0.0)
            Pcards_last_c1 = tl.load(
                cum_base + idx_last_c1, mask=mask_k & has_last1, other=0.0
            )
            Pcards_last_c2 = tl.load(
                cum_base + idx_last_c2, mask=mask_k & has_last2, other=0.0
            )

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )

            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                ev_out_ptr + m * 2 * NUM_HANDS + hero * NUM_HANDS + hand_idx,
                EV * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_ev_active_offset_full_nobopp_indexed_kernel(
        P_padded_ptr,  # [M, 2, H_ACTIVE+1]
        card_cumsum_ptr,  # [M, 2, NUM_CARDS, MC]
        row_index_ptr,  # [M]
        extra_index_ptr,  # [M] int64
        sorted_indices_ptr,  # [E, H_ACTIVE]
        rank_bounds_packed_ptr,
        scale_factor_ptr,  # [M, 2] — potential / scale
        off_L_pair_ptr,
        off_R_pair_ptr,
        off_last_pair_ptr,
        latest_values_ptr,  # [N, 2, NUM_HANDS]
        H_ACTIVE,
        NUM_HANDS,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        dst_row = tl.load(row_index_ptr + m)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H_ACTIVE

        hand_idx = tl.load(
            sorted_indices_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        )
        rank_bounds = tl.load(
            rank_bounds_packed_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        L = rank_bounds & 0xFFFF
        R = (rank_bounds >> 16) & 0xFFFF

        off_L_pair = tl.load(
            off_L_pair_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        off_R_pair = tl.load(
            off_R_pair_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        off_last_pair = tl.load(
            off_last_pair_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0
        ).to(tl.int32)
        enc_L_c1 = off_L_pair & 0xFFFF
        enc_L_c2 = (off_L_pair >> 16) & 0xFFFF
        enc_R_c1 = off_R_pair & 0xFFFF
        enc_R_c2 = (off_R_pair >> 16) & 0xFFFF
        enc_last_c1 = off_last_pair & 0xFFFF
        enc_last_c2 = (off_last_pair >> 16) & 0xFFFF

        has_L1 = enc_L_c1 > 0
        has_L2 = enc_L_c2 > 0
        has_R1 = enc_R_c1 > 0
        has_R2 = enc_R_c2 > 0
        has_last1 = enc_last_c1 > 0
        has_last2 = enc_last_c2 > 0
        idx_L_c1 = tl.maximum(enc_L_c1 - 1, 0)
        idx_L_c2 = tl.maximum(enc_L_c2 - 1, 0)
        idx_R_c1 = tl.maximum(enc_R_c1 - 1, 0)
        idx_R_c2 = tl.maximum(enc_R_c2 - 1, 0)
        idx_last_c1 = tl.maximum(enc_last_c1 - 1, 0)
        idx_last_c2 = tl.maximum(enc_last_c2 - 1, 0)

        for hero in tl.static_range(2):
            p_base = P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1)
            P_k = tl.load(p_base + k_offs, mask=mask_k, other=0.0)
            P_k_next = tl.load(p_base + k_offs + 1, mask=mask_k, other=0.0)
            b_at_k = P_k_next - P_k
            P_L = tl.load(p_base + L, mask=mask_k, other=0.0)
            P_R = tl.load(p_base + R, mask=mask_k, other=0.0)
            cum_base = (
                card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            )
            Pcards_L_c1 = tl.load(cum_base + idx_L_c1, mask=mask_k & has_L1, other=0.0)
            Pcards_L_c2 = tl.load(cum_base + idx_L_c2, mask=mask_k & has_L2, other=0.0)
            Pcards_R_c1 = tl.load(cum_base + idx_R_c1, mask=mask_k & has_R1, other=0.0)
            Pcards_R_c2 = tl.load(cum_base + idx_R_c2, mask=mask_k & has_R2, other=0.0)
            Pcards_last_c1 = tl.load(
                cum_base + idx_last_c1, mask=mask_k & has_last1, other=0.0
            )
            Pcards_last_c2 = tl.load(
                cum_base + idx_last_c2, mask=mask_k & has_last2, other=0.0
            )

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )

            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                latest_values_ptr
                + dst_row * 2 * NUM_HANDS
                + hero * NUM_HANDS
                + hand_idx,
                EV * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_ev_active_offset_compact_nobopp_fast_ev_kernel(
        beliefs_ptr,  # [M, 2, H_ACTIVE] compact active sorted beliefs
        P_padded_ptr,  # [M, 2, H_ACTIVE+1]
        card_cumsum_ptr,  # [M, 2, NUM_CARDS, MC]
        extra_index_ptr,  # [M] int64
        L_idx_ptr,
        R_idx_ptr,
        scale_factor_ptr,  # [M, 2] — potential / scale
        off_L_c1_ptr,
        off_L_c2_ptr,
        off_R_c1_ptr,
        off_R_c2_ptr,
        off_last_c1_ptr,
        off_last_c2_ptr,
        ev_out_ptr,  # [M, 2, H_ACTIVE] in active sorted order
        H_ACTIVE,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = k_offs < H_ACTIVE

        L = tl.load(L_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        R = tl.load(R_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        off_L_c1 = tl.load(
            off_L_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_L_c2 = tl.load(
            off_L_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_R_c1 = tl.load(
            off_R_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_R_c2 = tl.load(
            off_R_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_last_c1 = tl.load(
            off_last_c1_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)
        off_last_c2 = tl.load(
            off_last_c2_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=-1
        ).to(tl.int32)

        has_L1 = off_L_c1 >= 0
        has_L2 = off_L_c2 >= 0
        has_R1 = off_R_c1 >= 0
        has_R2 = off_R_c2 >= 0
        has_last1 = off_last_c1 >= 0
        has_last2 = off_last_c2 >= 0
        idx_L_c1 = tl.maximum(off_L_c1, 0)
        idx_L_c2 = tl.maximum(off_L_c2, 0)
        idx_R_c1 = tl.maximum(off_R_c1, 0)
        idx_R_c2 = tl.maximum(off_R_c2, 0)
        idx_last_c1 = tl.maximum(off_last_c1, 0)
        idx_last_c2 = tl.maximum(off_last_c2, 0)

        for hero in tl.static_range(2):
            villain = 1 - hero
            b_at_k = tl.load(
                beliefs_ptr + m * 2 * H_ACTIVE + villain * H_ACTIVE + k_offs,
                mask=mask_k,
                other=0.0,
            )
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + L,
                mask=mask_k,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + R,
                mask=mask_k,
                other=0.0,
            )
            cum_base = (
                card_cumsum_ptr + m * 2 * NUM_CARDS * MC_K + hero * NUM_CARDS * MC_K
            )
            Pcards_L_c1 = tl.where(
                has_L1, tl.load(cum_base + idx_L_c1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_L_c2 = tl.where(
                has_L2, tl.load(cum_base + idx_L_c2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c1 = tl.where(
                has_R1, tl.load(cum_base + idx_R_c1, mask=mask_k, other=0.0), 0.0
            )
            Pcards_R_c2 = tl.where(
                has_R2, tl.load(cum_base + idx_R_c2, mask=mask_k, other=0.0), 0.0
            )
            Pcards_last_c1 = tl.where(
                has_last1,
                tl.load(cum_base + idx_last_c1, mask=mask_k, other=0.0),
                0.0,
            )
            Pcards_last_c2 = tl.where(
                has_last2,
                tl.load(cum_base + idx_last_c2, mask=mask_k, other=0.0),
                0.0,
            )

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )
            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            EV = tl.where(valid, (2.0 * win_mass + tie_mass) / denom_safe - 1.0, 0.0)
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                ev_out_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k_offs,
                EV * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_ev_active_direct_compact_kernel(
        b_opp_sorted_ptr,  # [M, 2, H_ACTIVE]
        P_padded_ptr,  # [M, 2, H_ACTIVE+1]
        card_positions_ptr,  # [E, NUM_CARDS, MC]
        extra_index_ptr,  # [M] int64
        L_idx_ptr,
        R_idx_ptr,
        c1_sorted_ptr,
        c2_sorted_ptr,
        scale_factor_ptr,  # [M, 2] — potential / scale
        ev_out_ptr,  # [M, 2, H_ACTIVE] in active sorted order
        H_ACTIVE,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        slot_offs = tl.arange(0, MC_K)
        mask_k = k_offs < H_ACTIVE

        L = tl.load(L_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        R = tl.load(R_idx_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        c1 = tl.load(c1_sorted_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)
        c2 = tl.load(c2_sorted_ptr + extra * H_ACTIVE + k_offs, mask=mask_k, other=0)

        pos_base = card_positions_ptr + extra * NUM_CARDS * MC_K
        pos1 = tl.load(pos_base + c1[:, None] * MC_K + slot_offs[None, :])
        pos2 = tl.load(pos_base + c2[:, None] * MC_K + slot_offs[None, :])
        in1 = pos1 < H_ACTIVE
        in2 = pos2 < H_ACTIVE
        safe1 = tl.where(in1, pos1, 0)
        safe2 = tl.where(in2, pos2, 0)

        for hero in tl.static_range(2):
            b_base = b_opp_sorted_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE
            b_at_k = tl.load(b_base + k_offs, mask=mask_k, other=0.0)
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + L,
                mask=mask_k,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H_ACTIVE + 1) + hero * (H_ACTIVE + 1) + R,
                mask=mask_k,
                other=0.0,
            )

            b1 = tl.load(b_base + safe1, mask=mask_k[:, None] & in1, other=0.0)
            b2 = tl.load(b_base + safe2, mask=mask_k[:, None] & in2, other=0.0)
            Pcards_L_c1 = tl.sum(tl.where((pos1 < L[:, None]) & in1, b1, 0.0), axis=1)
            Pcards_L_c2 = tl.sum(tl.where((pos2 < L[:, None]) & in2, b2, 0.0), axis=1)
            Pcards_R_c1 = tl.sum(tl.where((pos1 < R[:, None]) & in1, b1, 0.0), axis=1)
            Pcards_R_c2 = tl.sum(tl.where((pos2 < R[:, None]) & in2, b2, 0.0), axis=1)
            Pcards_last_c1 = tl.sum(tl.where(in1, b1, 0.0), axis=1)
            Pcards_last_c2 = tl.sum(tl.where(in2, b2, 0.0), axis=1)

            win_mass = P_L - Pcards_L_c1 - Pcards_L_c2
            tie_mass = (
                (P_R - P_L)
                - (Pcards_R_c1 - Pcards_L_c1)
                - (Pcards_R_c2 - Pcards_L_c2)
                + b_at_k
            )

            denom = 1.0 - Pcards_last_c1 - Pcards_last_c2 + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(
                ev_out_ptr + m * 2 * H_ACTIVE + hero * H_ACTIVE + k_offs,
                EV * scale,
                mask=mask_k,
            )

    @triton.jit
    def _showdown_card_corr_scatter_kernel(
        b_opp_sorted_ptr,  # [M, 2, H]
        extra_index_ptr,  # [M]
        card_positions_ptr,  # [E, C, MC]
        occ_slot_L_ptr,  # [E, C, MC]
        occ_slot_R_ptr,  # [E, C, MC]
        card_slot_count_ptr,  # [E, C]
        corr_L_ptr,  # [M, 2, H]
        corr_R_ptr,  # [M, 2, H]
        corr_total_ptr,  # [M, 2, H]
        H,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        c = tl.program_id(2)
        extra = tl.load(extra_index_ptr + m)
        slot = tl.arange(0, MC_K)
        positions = tl.load(
            card_positions_ptr + extra * NUM_CARDS * MC_K + c * MC_K + slot
        )
        in_range = positions < H
        safe_pos = tl.where(in_range, positions, 0)
        b_base = b_opp_sorted_ptr + m * 2 * H + hero * H
        b_at = tl.load(b_base + safe_pos, mask=in_range, other=0.0)

        sL = tl.load(occ_slot_L_ptr + extra * NUM_CARDS * MC_K + c * MC_K + slot).to(
            tl.int32
        )
        sR = tl.load(occ_slot_R_ptr + extra * NUM_CARDS * MC_K + c * MC_K + slot).to(
            tl.int32
        )
        count = tl.load(card_slot_count_ptr + extra * NUM_CARDS + c).to(tl.int32)
        prefix_slot = tl.arange(0, MC_K)
        vL = tl.sum(
            tl.where(
                (prefix_slot[None, :] < sL[:, None]) & in_range[None, :],
                b_at[None, :],
                0.0,
            ),
            axis=1,
        )
        vR = tl.sum(
            tl.where(
                (prefix_slot[None, :] < sR[:, None]) & in_range[None, :],
                b_at[None, :],
                0.0,
            ),
            axis=1,
        )
        vTotal = tl.sum(tl.where(in_range, b_at, 0.0), axis=0)
        vTotal = tl.where(count > 0, vTotal, 0.0)

        out_base = m * 2 * H + hero * H
        tl.atomic_add(
            corr_L_ptr + out_base + safe_pos, vL, sem="relaxed", mask=in_range
        )
        tl.atomic_add(
            corr_R_ptr + out_base + safe_pos, vR, sem="relaxed", mask=in_range
        )
        tl.atomic_add(
            corr_total_ptr + out_base + safe_pos, vTotal, sem="relaxed", mask=in_range
        )

    @triton.jit
    def _showdown_corr_finish_compact_kernel(
        b_opp_sorted_ptr,  # [M, 2, H]
        P_padded_ptr,  # [M, 2, H+1]
        corr_L_ptr,  # [M, 2, H]
        corr_R_ptr,  # [M, 2, H]
        corr_total_ptr,  # [M, 2, H]
        extra_index_ptr,  # [M]
        L_idx_ptr,
        R_idx_ptr,
        scale_factor_ptr,
        ev_out_ptr,  # [M, 2, H]
        H,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < H
        L = tl.load(L_idx_ptr + extra * H + k, mask=mask, other=0)
        R = tl.load(R_idx_ptr + extra * H + k, mask=mask, other=0)

        for hero in tl.static_range(2):
            row_base = m * 2 * H + hero * H
            b_at_k = tl.load(b_opp_sorted_ptr + row_base + k, mask=mask, other=0.0)
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L,
                mask=mask,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R,
                mask=mask,
                other=0.0,
            )
            corr_L = tl.load(corr_L_ptr + row_base + k, mask=mask, other=0.0)
            corr_R = tl.load(corr_R_ptr + row_base + k, mask=mask, other=0.0)
            corr_total = tl.load(corr_total_ptr + row_base + k, mask=mask, other=0.0)
            win_mass = P_L - corr_L
            tie_mass = (P_R - P_L) - (corr_R - corr_L) + b_at_k
            denom = 1.0 - corr_total + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(ev_out_ptr + row_base + k, EV * scale, mask=mask)

    @triton.jit
    def _showdown_card_corr_dualslot_kernel(
        beliefs_ptr,  # [M, 2, H]
        extra_index_ptr,  # [M]
        card_positions_ptr,  # [E, C, MC]
        occ_slot_L_ptr,  # [E, C, MC]
        occ_slot_R_ptr,  # [E, C, MC]
        card_slot_count_ptr,  # [E, C]
        occ_is_c2_ptr,  # [E, C, MC]
        corr_L1_ptr,  # [M, 2, H]
        corr_L2_ptr,  # [M, 2, H]
        corr_R1_ptr,  # [M, 2, H]
        corr_R2_ptr,  # [M, 2, H]
        corr_T1_ptr,  # [M, 2, H]
        corr_T2_ptr,  # [M, 2, H]
        H,
        NUM_CARDS: tl.constexpr,
        MC_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        hero = tl.program_id(1)
        c = tl.program_id(2)
        villain = 1 - hero
        extra = tl.load(extra_index_ptr + m)
        slot = tl.arange(0, MC_K)
        base = extra * NUM_CARDS * MC_K + c * MC_K + slot
        positions = tl.load(card_positions_ptr + base)
        in_range = positions < H
        safe_pos = tl.where(in_range, positions, 0)
        b_base = beliefs_ptr + m * 2 * H + villain * H
        b_at = tl.load(b_base + safe_pos, mask=in_range, other=0.0)

        sL = tl.load(occ_slot_L_ptr + base).to(tl.int32)
        sR = tl.load(occ_slot_R_ptr + base).to(tl.int32)
        count = tl.load(card_slot_count_ptr + extra * NUM_CARDS + c).to(tl.int32)
        prefix_slot = tl.arange(0, MC_K)
        vL = tl.sum(
            tl.where(
                (prefix_slot[None, :] < sL[:, None]) & in_range[None, :],
                b_at[None, :],
                0.0,
            ),
            axis=1,
        )
        vR = tl.sum(
            tl.where(
                (prefix_slot[None, :] < sR[:, None]) & in_range[None, :],
                b_at[None, :],
                0.0,
            ),
            axis=1,
        )
        vT = tl.sum(tl.where(in_range, b_at, 0.0), axis=0)
        vT = tl.where(count > 0, vT, 0.0)
        is_c2 = tl.load(occ_is_c2_ptr + base, mask=in_range, other=0).to(tl.int1)
        out_base = m * 2 * H + hero * H
        mask_1 = in_range & ~is_c2
        mask_2 = in_range & is_c2
        tl.store(corr_L1_ptr + out_base + safe_pos, vL, mask=mask_1)
        tl.store(corr_L2_ptr + out_base + safe_pos, vL, mask=mask_2)
        tl.store(corr_R1_ptr + out_base + safe_pos, vR, mask=mask_1)
        tl.store(corr_R2_ptr + out_base + safe_pos, vR, mask=mask_2)
        tl.store(corr_T1_ptr + out_base + safe_pos, vT, mask=mask_1)
        tl.store(corr_T2_ptr + out_base + safe_pos, vT, mask=mask_2)

    @triton.jit
    def _showdown_dualslot_finish_compact_kernel(
        beliefs_ptr,  # [M, 2, H]
        P_padded_ptr,  # [M, 2, H+1]
        corr_L1_ptr,
        corr_L2_ptr,
        corr_R1_ptr,
        corr_R2_ptr,
        corr_T1_ptr,
        corr_T2_ptr,
        extra_index_ptr,  # [M]
        L_idx_ptr,
        R_idx_ptr,
        scale_factor_ptr,
        ev_out_ptr,  # [M, 2, H]
        H,
        BLOCK_K: tl.constexpr,
    ):
        m = tl.program_id(0)
        k_block = tl.program_id(1)
        extra = tl.load(extra_index_ptr + m)
        k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = k < H
        L = tl.load(L_idx_ptr + extra * H + k, mask=mask, other=0)
        R = tl.load(R_idx_ptr + extra * H + k, mask=mask, other=0)

        for hero in tl.static_range(2):
            villain = 1 - hero
            row_base = m * 2 * H + hero * H
            belief_base = m * 2 * H + villain * H
            b_at_k = tl.load(beliefs_ptr + belief_base + k, mask=mask, other=0.0)
            P_L = tl.load(
                P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + L,
                mask=mask,
                other=0.0,
            )
            P_R = tl.load(
                P_padded_ptr + m * 2 * (H + 1) + hero * (H + 1) + R,
                mask=mask,
                other=0.0,
            )
            corr_L = tl.load(corr_L1_ptr + row_base + k, mask=mask, other=0.0)
            corr_L += tl.load(corr_L2_ptr + row_base + k, mask=mask, other=0.0)
            corr_R = tl.load(corr_R1_ptr + row_base + k, mask=mask, other=0.0)
            corr_R += tl.load(corr_R2_ptr + row_base + k, mask=mask, other=0.0)
            corr_T = tl.load(corr_T1_ptr + row_base + k, mask=mask, other=0.0)
            corr_T += tl.load(corr_T2_ptr + row_base + k, mask=mask, other=0.0)
            win_mass = P_L - corr_L
            tie_mass = (P_R - P_L) - (corr_R - corr_L) + b_at_k
            denom = 1.0 - corr_T + b_at_k
            valid = denom > 1e-8
            denom_safe = tl.where(valid, denom, 1.0)
            win_prob = tl.where(valid, win_mass / denom_safe, 0.0)
            tie_prob = tl.where(valid, tie_mass / denom_safe, 0.0)
            loss_prob = tl.where(valid, 1.0 - win_prob - tie_prob, 0.0)
            EV = win_prob - loss_prob
            scale = tl.load(scale_factor_ptr + m * 2 + hero)
            tl.store(ev_out_ptr + row_base + k, EV * scale, mask=mask)


def precompute_showdown_extras(
    hrd,
    env,
    showdown_indices,
    *,
    scale_indices=None,
    extra_indices=None,
):
    """Compute everything v15 needs that's a function of the subgame
    structure (i.e., constant across CFR iters within a subgame).

    Returns a dict with:
      card_positions, slot_L_c1, slot_L_c2, slot_R_c1, slot_R_c2,
      slot_last_c1, slot_last_c2, hand_ok_sorted, scale_factor, c1, c2.
    """
    card_positions, slot_tensors = precompute_showdown_card_positions_and_lookup_slots(
        hrd.hands_c1c2_sorted,
        hrd.L_idx,
        hrd.R_idx,
    )
    hand_ok_sorted = (
        hrd.hand_ok_mask.gather(1, hrd.sorted_indices).to(torch.uint8).contiguous()
    )
    if scale_indices is None:
        scale_indices = showdown_indices
    if extra_indices is None:
        extra_indices = torch.arange(
            int(scale_indices.numel()), device=scale_indices.device, dtype=torch.long
        )
    showdown_potential = (
        env.stacks[scale_indices]
        + env.pot[scale_indices, None]
        - env.starting_stacks[scale_indices]
    )  # [M, 2]
    env_scale = env.scale[scale_indices]  # [M]
    scale_factor = (showdown_potential / env_scale[:, None]).contiguous()
    c1 = hrd.hands_c1c2_sorted[..., 0].contiguous()
    c2 = hrd.hands_c1c2_sorted[..., 1].contiguous()
    return {
        "card_positions": card_positions.to(torch.int32),
        "slot_L_c1": slot_tensors[0],
        "slot_L_c2": slot_tensors[1],
        "slot_R_c1": slot_tensors[2],
        "slot_R_c2": slot_tensors[3],
        "slot_last_c1": slot_tensors[4],
        "slot_last_c2": slot_tensors[5],
        "hand_ok_sorted": hand_ok_sorted,
        "scale_factor": scale_factor,
        "c1": c1,
        "c2": c2,
        "L_idx": hrd.L_idx.contiguous(),
        "R_idx": hrd.R_idx.contiguous(),
        "sorted_indices": hrd.sorted_indices.contiguous(),
        "extra_indices": extra_indices.to(torch.long).contiguous(),
    }


def precompute_showdown_active_extras(
    hrd,
    env,
    showdown_indices,
    *,
    scale_indices=None,
    extra_indices=None,
    include_experimental=False,
):
    """Compute exact HU showdown metadata over board-legal river hands only.

    This mirrors :func:`precompute_showdown_extras`, but each unique river board
    is compacted to its 1081 legal hands and 47 non-board cards. The EV runner
    scatters active-hand values back into the full 1326-hand output.
    """
    device = hrd.sorted_indices.device
    e_count, full_hands = hrd.sorted_indices.shape
    active_hands = SHOWDOWN_RIVER_ACTIVE_HANDS
    active_cards = SHOWDOWN_RIVER_ACTIVE_CARDS

    positions = torch.arange(full_hands, device=device).expand(e_count, -1)
    hand_ok_sorted = hrd.hand_ok_mask.gather(1, hrd.sorted_indices)
    active_pos = torch.where(hand_ok_sorted, positions, full_hands)
    active_pos = active_pos.sort(dim=1).values[:, :active_hands].contiguous()

    active_sorted_indices = hrd.sorted_indices.gather(1, active_pos).contiguous()
    gather_2 = active_pos.unsqueeze(-1).expand(-1, -1, 2)
    active_c1c2_global = hrd.hands_c1c2_sorted.gather(1, gather_2).contiguous()

    board = env.board_indices[showdown_indices].int()
    card_ok = torch.ones(e_count, 52, dtype=torch.bool, device=device)
    card_ok.scatter_(1, board, False)
    global_cards = torch.arange(52, device=device).expand(e_count, -1)
    active_global_cards = torch.where(card_ok, global_cards, 52)
    active_global_cards = (
        active_global_cards.sort(dim=1).values[:, :active_cards].to(torch.int64)
    )
    card_to_local = torch.full((e_count, 52), -1, dtype=torch.int64, device=device)
    local_ids = torch.arange(active_cards, device=device, dtype=torch.int64)
    card_to_local.scatter_(1, active_global_cards, local_ids.expand(e_count, -1))
    active_c1c2_local = card_to_local.gather(
        1, active_c1c2_global.reshape(e_count, -1)
    ).reshape(e_count, active_hands, 2)

    full_group_key = hrd.L_idx.gather(1, active_pos)
    is_start = torch.ones(e_count, active_hands, dtype=torch.bool, device=device)
    is_start[:, 1:] = full_group_key[:, 1:] != full_group_key[:, :-1]
    group_id = is_start.cumsum(dim=1, dtype=torch.int32) - 1
    k = torch.arange(active_hands, device=device, dtype=torch.int32).expand(e_count, -1)
    starts = torch.full(
        (e_count, active_hands), active_hands, dtype=torch.int32, device=device
    )
    ends = torch.full((e_count, active_hands), -1, dtype=torch.int32, device=device)
    starts.scatter_reduce_(1, group_id, k, reduce="amin", include_self=True)
    ends.scatter_reduce_(1, group_id, k, reduce="amax", include_self=True)
    l_idx = starts.gather(1, group_id).contiguous()
    r_idx = (ends.gather(1, group_id) + 1).clamp(max=active_hands).contiguous()

    card_positions, slot_tensors = precompute_showdown_card_positions_and_lookup_slots(
        active_c1c2_local,
        l_idx,
        r_idx,
        num_cards=active_cards,
    )
    c1_local = active_c1c2_local[..., 0].to(torch.int32).contiguous()
    c2_local = active_c1c2_local[..., 1].to(torch.int32).contiguous()
    slot_l_c1 = slot_tensors[0].to(torch.int32)
    slot_l_c2 = slot_tensors[1].to(torch.int32)
    slot_r_c1 = slot_tensors[2].to(torch.int32)
    slot_r_c2 = slot_tensors[3].to(torch.int32)
    slot_last_c1 = slot_tensors[4].to(torch.int32)
    slot_last_c2 = slot_tensors[5].to(torch.int32)

    def _flat_offsets(cards: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        offsets = cards * SHOWDOWN_MAX_PER_CARD + slots - 1
        return torch.where(slots > 0, offsets, torch.full_like(offsets, -1))

    off_l_c1 = _flat_offsets(c1_local, slot_l_c1).to(torch.int16).contiguous()
    off_l_c2 = _flat_offsets(c2_local, slot_l_c2).to(torch.int16).contiguous()
    off_r_c1 = _flat_offsets(c1_local, slot_r_c1).to(torch.int16).contiguous()
    off_r_c2 = _flat_offsets(c2_local, slot_r_c2).to(torch.int16).contiguous()
    off_last_c1 = _flat_offsets(c1_local, slot_last_c1).to(torch.int16).contiguous()
    off_last_c2 = _flat_offsets(c2_local, slot_last_c2).to(torch.int16).contiguous()

    rank_bounds_packed = (
        l_idx.to(torch.int32) | (r_idx.to(torch.int32) << 16)
    ).contiguous()

    def _pack_encoded_offsets(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_enc = a.to(torch.int32) + 1
        b_enc = b.to(torch.int32) + 1
        return (a_enc | (b_enc << 16)).contiguous()

    off_l_pair = _pack_encoded_offsets(off_l_c1, off_l_c2)
    off_r_pair = _pack_encoded_offsets(off_r_c1, off_r_c2)
    off_last_pair = _pack_encoded_offsets(off_last_c1, off_last_c2)

    if scale_indices is None:
        scale_indices = showdown_indices
    if extra_indices is None:
        extra_indices = torch.arange(
            int(scale_indices.numel()), device=scale_indices.device, dtype=torch.long
        )
    showdown_potential = (
        env.stacks[scale_indices]
        + env.pot[scale_indices, None]
        - env.starting_stacks[scale_indices]
    )
    env_scale = env.scale[scale_indices]
    scale_factor = (showdown_potential / env_scale[:, None]).contiguous()
    result = {
        "card_positions": card_positions.to(torch.int32),
        "slot_L_c1": slot_tensors[0].to(torch.uint8).contiguous(),
        "slot_L_c2": slot_tensors[1].to(torch.uint8).contiguous(),
        "slot_R_c1": slot_tensors[2].to(torch.uint8).contiguous(),
        "slot_R_c2": slot_tensors[3].to(torch.uint8).contiguous(),
        "slot_last_c1": slot_tensors[4].to(torch.uint8).contiguous(),
        "slot_last_c2": slot_tensors[5].to(torch.uint8).contiguous(),
        "scale_factor": scale_factor,
        "c1": c1_local,
        "c2": c2_local,
        "L_idx": l_idx,
        "R_idx": r_idx,
        "sorted_indices": active_sorted_indices.to(torch.int32),
        "extra_indices": extra_indices.to(torch.long).contiguous(),
        "active_cards": active_global_cards.to(torch.int32).contiguous(),
        "off_L_c1": off_l_c1,
        "off_L_c2": off_l_c2,
        "off_R_c1": off_r_c1,
        "off_R_c2": off_r_c2,
        "off_last_c1": off_last_c1,
        "off_last_c2": off_last_c2,
        "rank_bounds_packed": rank_bounds_packed,
        "off_L_pair": off_l_pair,
        "off_R_pair": off_r_pair,
        "off_last_pair": off_last_pair,
    }
    if include_experimental:
        occ_pos = card_positions.to(torch.long)
        valid_occ = occ_pos < active_hands
        safe_occ = occ_pos.clamp(max=active_hands - 1)
        occ_l = l_idx.gather(1, safe_occ.reshape(e_count, -1)).reshape(
            e_count, active_cards, SHOWDOWN_MAX_PER_CARD
        )
        occ_r = r_idx.gather(1, safe_occ.reshape(e_count, -1)).reshape(
            e_count, active_cards, SHOWDOWN_MAX_PER_CARD
        )
        occ_slot_l = ((occ_pos[:, :, None, :] < occ_l[:, :, :, None]).sum(dim=-1)).to(
            torch.uint8
        )
        occ_slot_r = ((occ_pos[:, :, None, :] < occ_r[:, :, :, None]).sum(dim=-1)).to(
            torch.uint8
        )
        occ_slot_l = torch.where(valid_occ, occ_slot_l, torch.zeros_like(occ_slot_l))
        occ_slot_r = torch.where(valid_occ, occ_slot_r, torch.zeros_like(occ_slot_r))
        card_slot_count = valid_occ.sum(dim=-1).to(torch.uint8).contiguous()
        occ_c2 = c2_local.gather(1, safe_occ.reshape(e_count, -1)).reshape(
            e_count, active_cards, SHOWDOWN_MAX_PER_CARD
        )
        occ_card = (
            torch.arange(active_cards, device=device, dtype=torch.int32)
            .view(1, active_cards, 1)
            .expand(e_count, -1, SHOWDOWN_MAX_PER_CARD)
        )
        result.update(
            {
                "occ_slot_L": occ_slot_l.contiguous(),
                "occ_slot_R": occ_slot_r.contiguous(),
                "card_slot_count": card_slot_count,
                "occ_is_c2": ((occ_c2 == occ_card) & valid_occ)
                .to(torch.uint8)
                .contiguous(),
            }
        )
    return result


def showdown_ev_v15(
    beliefs: torch.Tensor,  # [M, 2, NUM_HANDS]
    extras: dict,
    block_k: int = 64,
) -> torch.Tensor:
    """Drop-in replacement for `CFREvaluator._showdown_value_both` using the
    three-kernel Triton pipeline (setup_b_P → build_cum → main_ev).
    Returns `[M, 2, NUM_HANDS]` with `hand_ok` masking and
    `potential / scale` scaling already applied."""
    if not triton_is_available():
        raise RuntimeError("Triton is not installed.")
    M, _, NUM_HANDS = beliefs.shape
    H = extras["sorted_indices"].shape[1]
    device = beliefs.device
    dtype = torch.float32

    block_h = 1
    while block_h < H:
        block_h *= 2

    b_opp_both = torch.empty(M, 2, H, device=device, dtype=dtype)
    P_padded_both = torch.empty(M, 2, H + 1, device=device, dtype=dtype)
    cum_both = torch.empty(M, 2, 52, SHOWDOWN_MAX_PER_CARD, device=device, dtype=dtype)
    ev_unsorted = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
    extra_indices = extras["extra_indices"].contiguous()

    _showdown_setup_b_P_kernel[(M, 2)](
        beliefs.contiguous(),
        extra_indices,
        extras["sorted_indices"],
        b_opp_both,
        P_padded_both,
        H,
        NUM_HANDS,
        BLOCK_H=block_h,
    )
    _showdown_build_cum_kernel[(M, 2 * 52)](
        b_opp_both,
        extra_indices,
        extras["card_positions"],
        cum_both,
        H,
        NUM_CARDS=52,
        MC_K=SHOWDOWN_MAX_PER_CARD,
    )
    grid = (M, triton.cdiv(NUM_HANDS, block_k))
    _showdown_ev_v15_kernel[grid](
        b_opp_both,
        P_padded_both,
        cum_both,
        extra_indices,
        extras["L_idx"],
        extras["R_idx"],
        extras["c1"],
        extras["c2"],
        extras["sorted_indices"],
        extras["hand_ok_sorted"],
        extras["scale_factor"],
        extras["slot_L_c1"],
        extras["slot_L_c2"],
        extras["slot_R_c1"],
        extras["slot_R_c2"],
        extras["slot_last_c1"],
        extras["slot_last_c2"],
        ev_unsorted,
        NUM_HANDS,
        NUM_CARDS=52,
        MC_K=SHOWDOWN_MAX_PER_CARD,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return ev_unsorted


# ---------------------------------------------------------------------------
# CUDA-graph wrapper for the showdown EV pipeline. Captures the three
# kernels above into one graph keyed on a fixed (M, NUM_HANDS) shape;
# replay does just a copy_() into a persistent input buffer + replay().
# See FusedSparseCFREvaluator._init_hand_rank_data for the per-subgame
# capture site.
# ---------------------------------------------------------------------------


class ShowdownGraphRunner:
    """Captures the showdown EV pipeline as a CUDA graph for one fixed
    (M, NUM_HANDS) configuration. Subsequent calls only do a copy_() into
    the persistent input buffer, then replay the graph.

    All buffers live on the runner; the returned EV tensor is the same
    object across calls (callers must consume / copy before the next
    call if they need the value to outlive the next replay).
    """

    def __init__(
        self,
        extras: dict,
        M: int,
        NUM_HANDS: int,
        device: torch.device,
        block_k: int = 64,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.NUM_HANDS = NUM_HANDS
        self.device = device
        self.block_k = block_k

        H = extras["sorted_indices"].shape[1]
        self.H = H
        block_h = 1
        while block_h < H:
            block_h *= 2
        self.block_h = block_h

        # Persistent buffers.
        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
        self.b_opp_both = torch.empty(M, 2, H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, H + 1, device=device, dtype=dtype)
        self.cum_both = torch.empty(
            M,
            2,
            52,
            SHOWDOWN_MAX_PER_CARD,
            device=device,
            dtype=dtype,
        )
        self.ev_out = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)

        # Warm up the kernels on a side stream so JIT compile + autotune
        # happen outside the capture. Three replays match torch.cuda.graph
        # docs' recommendation.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        # Capture.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_b_P_kernel[(self.M, 2)](
            self.beliefs_in,
            extra_indices,
            e["sorted_indices"],
            self.b_opp_both,
            self.P_padded,
            self.H,
            self.NUM_HANDS,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_kernel[(self.M, 2 * 52)](
            self.b_opp_both,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=52,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )
        grid = (self.M, triton.cdiv(self.NUM_HANDS, self.block_k))
        _showdown_ev_v15_kernel[grid](
            self.b_opp_both,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["c1"],
            e["c2"],
            e["sorted_indices"],
            e["hand_ok_sorted"],
            e["scale_factor"],
            e["slot_L_c1"],
            e["slot_L_c2"],
            e["slot_R_c1"],
            e["slot_R_c2"],
            e["slot_last_c1"],
            e["slot_last_c2"],
            self.ev_out,
            self.NUM_HANDS,
            NUM_CARDS=52,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        # When called from inside an outer CUDA graph capture, replay() of
        # a pre-captured graph isn't allowed; emit the kernels directly so
        # they're recorded into the outer graph instead.
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownActiveGraphRunner:
    """CUDA-graph runner for exact river showdown over compact active hands."""

    def __init__(
        self,
        extras: dict,
        M: int,
        NUM_HANDS: int,
        device: torch.device,
        block_k: int = 64,
        zero_output: bool = True,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.NUM_HANDS = NUM_HANDS
        self.device = device
        self.block_k = block_k
        self.zero_output = zero_output

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
        self.b_opp_both = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.cum_both = torch.empty(
            M,
            2,
            self.num_cards,
            SHOWDOWN_MAX_PER_CARD,
            device=device,
            dtype=dtype,
        )
        self.ev_out = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        if self.zero_output:
            self.ev_out.zero_()
        _showdown_setup_b_P_kernel[(self.M, 2)](
            self.beliefs_in,
            extra_indices,
            e["sorted_indices"],
            self.b_opp_both,
            self.P_padded,
            self.H,
            self.NUM_HANDS,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_kernel[(self.M, 2 * self.num_cards)](
            self.b_opp_both,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_kernel[grid](
            self.b_opp_both,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["c1"],
            e["c2"],
            e["sorted_indices"],
            e["scale_factor"],
            e["slot_L_c1"],
            e["slot_L_c2"],
            e["slot_R_c1"],
            e["slot_R_c2"],
            e["slot_last_c1"],
            e["slot_last_c2"],
            self.ev_out,
            self.H,
            self.NUM_HANDS,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownActiveCompactGraphRunner:
    """CUDA-graph runner returning exact river CFVs in active sorted order."""

    def __init__(
        self,
        extras: dict,
        M: int,
        NUM_HANDS: int,
        device: torch.device,
        block_k: int = 64,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.NUM_HANDS = NUM_HANDS
        self.device = device
        self.block_k = block_k

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
        self.b_opp_both = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.cum_both = torch.empty(
            M,
            2,
            self.num_cards,
            SHOWDOWN_MAX_PER_CARD,
            device=device,
            dtype=dtype,
        )
        self.ev_out = torch.empty(M, 2, self.H, device=device, dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_b_P_kernel[(self.M, 2)](
            self.beliefs_in,
            extra_indices,
            e["sorted_indices"],
            self.b_opp_both,
            self.P_padded,
            self.H,
            self.NUM_HANDS,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_kernel[(self.M, 2 * self.num_cards)](
            self.b_opp_both,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_compact_kernel[grid](
            self.b_opp_both,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["c1"],
            e["c2"],
            e["scale_factor"],
            e["slot_L_c1"],
            e["slot_L_c2"],
            e["slot_R_c1"],
            e["slot_R_c2"],
            e["slot_last_c1"],
            e["slot_last_c2"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownFullyCompactGraphRunner:
    """CUDA-graph runner for compact input beliefs and compact exact CFVs."""

    def __init__(
        self,
        extras: dict,
        M: int,
        device: torch.device,
        block_k: int = 64,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.device = device
        self.block_k = block_k

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.b_opp_both = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.cum_both = torch.empty(
            M,
            2,
            self.num_cards,
            SHOWDOWN_MAX_PER_CARD,
            device=device,
            dtype=dtype,
        )
        self.ev_out = torch.empty(M, 2, self.H, device=device, dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_b_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.b_opp_both,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_kernel[(self.M, 2 * self.num_cards)](
            self.b_opp_both,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_compact_kernel[grid](
            self.b_opp_both,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["c1"],
            e["c2"],
            e["scale_factor"],
            e["slot_L_c1"],
            e["slot_L_c2"],
            e["slot_R_c1"],
            e["slot_R_c2"],
            e["slot_last_c1"],
            e["slot_last_c2"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownFullyCompactOffsetGraphRunner:
    """Compact exact showdown with precomputed flat card-cumsum offsets."""

    def __init__(
        self,
        extras: dict,
        M: int,
        device: torch.device,
        block_k: int = 64,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.device = device
        self.block_k = block_k

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.b_opp_both = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.cum_both = torch.empty(
            M,
            2,
            self.num_cards,
            SHOWDOWN_MAX_PER_CARD,
            device=device,
            dtype=dtype,
        )
        self.ev_out = torch.empty(M, 2, self.H, device=device, dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_b_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.b_opp_both,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_kernel[(self.M, 2 * self.num_cards)](
            self.b_opp_both,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_offset_compact_kernel[grid](
            self.b_opp_both,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["scale_factor"],
            e["off_L_c1"],
            e["off_L_c2"],
            e["off_R_c1"],
            e["off_R_c2"],
            e["off_last_c1"],
            e["off_last_c2"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownFullyCompactNoBOppGraphRunner:
    """Compact exact showdown without materializing opponent-belief rows."""

    def __init__(
        self,
        extras: dict,
        M: int,
        device: torch.device,
        block_k: int = 64,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.device = device
        self.block_k = block_k

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.cum_both = torch.empty(
            M,
            2,
            self.num_cards,
            SHOWDOWN_MAX_PER_CARD,
            device=device,
            dtype=dtype,
        )
        self.ev_out = torch.empty(M, 2, self.H, device=device, dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_compact_kernel[(self.M, 2 * self.num_cards)](
            self.beliefs_in,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_offset_compact_nobopp_kernel[grid](
            self.beliefs_in,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["scale_factor"],
            e["off_L_c1"],
            e["off_L_c2"],
            e["off_R_c1"],
            e["off_R_c2"],
            e["off_last_c1"],
            e["off_last_c2"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownFullyCompactNoBOppFastEVGraphRunner(
    ShowdownFullyCompactNoBOppGraphRunner
):
    """No-bopp compact runner with algebraically reduced EV arithmetic."""

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_compact_kernel[(self.M, 2 * self.num_cards)](
            self.beliefs_in,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_offset_compact_nobopp_fast_ev_kernel[grid](
            self.beliefs_in,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["scale_factor"],
            e["off_L_c1"],
            e["off_L_c2"],
            e["off_R_c1"],
            e["off_R_c2"],
            e["off_last_c1"],
            e["off_last_c2"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )


class ShowdownFullyCompactNoBOppBothHeroCumGraphRunner(
    ShowdownFullyCompactNoBOppGraphRunner
):
    """No-bopp compact runner building both hero cumsums per card program."""

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_compact_both_heroes_kernel[(self.M, self.num_cards)](
            self.beliefs_in,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_offset_compact_nobopp_kernel[grid](
            self.beliefs_in,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["scale_factor"],
            e["off_L_c1"],
            e["off_L_c2"],
            e["off_R_c1"],
            e["off_R_c2"],
            e["off_last_c1"],
            e["off_last_c2"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )


class ShowdownFullyCompactNoBOppCardBlockCumGraphRunner(
    ShowdownFullyCompactNoBOppGraphRunner
):
    """No-bopp compact runner building multiple cards per cumsum program."""

    def __init__(
        self,
        extras: dict,
        M: int,
        device: torch.device,
        block_k: int = 64,
        card_block: int = 2,
        finish_num_warps: int = 4,
    ) -> None:
        self.card_block = card_block
        self.finish_num_warps = finish_num_warps
        super().__init__(extras=extras, M=M, device=device, block_k=block_k)

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_compact_card_block_kernel[
            (self.M, triton.cdiv(self.num_cards, self.card_block))
        ](
            self.beliefs_in,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            CARD_BLOCK=self.card_block,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_offset_compact_nobopp_kernel[grid](
            self.beliefs_in,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["scale_factor"],
            e["off_L_c1"],
            e["off_L_c2"],
            e["off_R_c1"],
            e["off_R_c2"],
            e["off_last_c1"],
            e["off_last_c2"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=self.finish_num_warps,
        )


class ShowdownFullyCompactNoBOppCardBlockFastEVGraphRunner(
    ShowdownFullyCompactNoBOppCardBlockCumGraphRunner
):
    """Card-block cumsum runner using the reduced EV finish formula."""

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_compact_card_block_kernel[
            (self.M, triton.cdiv(self.num_cards, self.card_block))
        ](
            self.beliefs_in,
            extra_indices,
            e["card_positions"],
            self.cum_both,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            CARD_BLOCK=self.card_block,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_offset_compact_nobopp_fast_ev_kernel[grid](
            self.beliefs_in,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["scale_factor"],
            e["off_L_c1"],
            e["off_L_c2"],
            e["off_R_c1"],
            e["off_R_c2"],
            e["off_last_c1"],
            e["off_last_c2"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=self.finish_num_warps,
        )


class ShowdownActiveCardBlockGraphRunner:
    """Production runner for exact river showdown with full-shape I/O.

    Prefix math runs in active board-legal river-hand order, but the runner
    reads full `[M, 2, NUM_HANDS]` beliefs and writes full shaped EV output
    directly.
    """

    def __init__(
        self,
        extras: dict,
        M: int,
        NUM_HANDS: int,
        device: torch.device,
        block_k: int = 256,
        card_block: int = 64,
        finish_num_warps: int = 8,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.NUM_HANDS = NUM_HANDS
        self.device = device
        self.block_k = block_k
        self.card_block = card_block
        self.finish_num_warps = finish_num_warps

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.cum_both = torch.empty(
            M,
            2,
            self.num_cards,
            SHOWDOWN_MAX_PER_CARD,
            device=device,
            dtype=dtype,
        )
        self.ev_out = torch.empty(M, 2, NUM_HANDS, device=device, dtype=dtype)
        self.ev_out.zero_()

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_P_active_full_kernel[(self.M, 2)](
            self.beliefs_in,
            extra_indices,
            e["sorted_indices"],
            self.P_padded,
            self.H,
            self.NUM_HANDS,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_active_full_card_block_kernel[
            (self.M, triton.cdiv(self.num_cards, self.card_block))
        ](
            self.beliefs_in,
            extra_indices,
            e["sorted_indices"],
            e["card_positions"],
            self.cum_both,
            self.H,
            self.NUM_HANDS,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            CARD_BLOCK=self.card_block,
        )
        finish_grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_offset_full_nobopp_kernel[finish_grid](
            self.beliefs_in,
            self.P_padded,
            self.cum_both,
            extra_indices,
            e["sorted_indices"],
            e["rank_bounds_packed"],
            e["scale_factor"],
            e["off_L_pair"],
            e["off_R_pair"],
            e["off_last_pair"],
            self.ev_out,
            self.H,
            self.NUM_HANDS,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=self.finish_num_warps,
        )

    def _copy_indexed_beliefs(
        self, beliefs: torch.Tensor, indices: torch.Tensor
    ) -> None:
        grid = (self.M, 2, triton.cdiv(self.NUM_HANDS, 2048))
        _copy_indexed_belief_rows_kernel[grid](
            beliefs,
            indices,
            self.beliefs_in,
            self.NUM_HANDS,
            BLOCK_H=2048,
            num_warps=4,
        )

    def _scatter_indexed_values(
        self, latest_values: torch.Tensor, indices: torch.Tensor
    ) -> None:
        grid = (self.M, 2, triton.cdiv(self.NUM_HANDS, 2048))
        _copy_rows_to_indexed_values_kernel[grid](
            self.ev_out,
            indices,
            latest_values,
            self.NUM_HANDS,
            BLOCK_H=2048,
            num_warps=4,
        )

    def write_indexed(
        self,
        beliefs: torch.Tensor,
        indices: torch.Tensor,
        latest_values: torch.Tensor,
    ) -> torch.Tensor:
        self._copy_indexed_beliefs(beliefs, indices)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        self._scatter_indexed_values(latest_values, indices)
        return self.ev_out

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownActiveCardBlockDirectGraphRunner(ShowdownActiveCardBlockGraphRunner):
    """Indexed exact showdown runner that reads/writes evaluator rows directly."""

    def __init__(
        self,
        extras: dict,
        M: int,
        NUM_HANDS: int,
        device: torch.device,
        source_beliefs: torch.Tensor,
        row_indices: torch.Tensor,
        latest_values: torch.Tensor,
        block_k: int = 256,
        card_block: int = 64,
        finish_num_warps: int = 8,
    ) -> None:
        self.source_beliefs = source_beliefs
        self.row_indices = row_indices.contiguous()
        self.latest_values = latest_values
        super().__init__(
            extras=extras,
            M=M,
            NUM_HANDS=NUM_HANDS,
            device=device,
            block_k=block_k,
            card_block=card_block,
            finish_num_warps=finish_num_warps,
        )

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_P_active_full_indexed_kernel[(self.M, 2)](
            self.source_beliefs,
            self.row_indices,
            extra_indices,
            e["sorted_indices"],
            self.P_padded,
            self.H,
            self.NUM_HANDS,
            BLOCK_H=self.block_h,
        )
        _showdown_build_cum_active_full_card_block_indexed_kernel[
            (self.M, triton.cdiv(self.num_cards, self.card_block))
        ](
            self.source_beliefs,
            self.row_indices,
            extra_indices,
            e["sorted_indices"],
            e["card_positions"],
            self.cum_both,
            self.H,
            self.NUM_HANDS,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            CARD_BLOCK=self.card_block,
        )
        finish_grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_offset_full_nobopp_indexed_kernel[finish_grid](
            self.P_padded,
            self.cum_both,
            self.row_indices,
            extra_indices,
            e["sorted_indices"],
            e["rank_bounds_packed"],
            e["scale_factor"],
            e["off_L_pair"],
            e["off_R_pair"],
            e["off_last_pair"],
            self.latest_values,
            self.H,
            self.NUM_HANDS,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=self.finish_num_warps,
        )

    def write_indexed(
        self,
        beliefs: torch.Tensor,
        indices: torch.Tensor,
        latest_values: torch.Tensor,
    ) -> torch.Tensor:
        if (
            beliefs.data_ptr() == self.source_beliefs.data_ptr()
            and latest_values.data_ptr() == self.latest_values.data_ptr()
            and indices.data_ptr() == self.row_indices.data_ptr()
        ):
            if torch.cuda.is_current_stream_capturing():
                self._launch_pipeline()
            else:
                self.graph.replay()
            return self.ev_out

        # The captured graph for this subclass reads from source_beliefs and
        # writes directly into self.latest_values. The base-class fallback graph
        # would therefore still use the direct pointers and ignore beliefs_in.
        # For non-captured pointers, run the base indexed pipeline eagerly.
        self._copy_indexed_beliefs(beliefs, indices)
        ShowdownActiveCardBlockGraphRunner._launch_pipeline(self)
        self._scatter_indexed_values(latest_values, indices)
        return self.ev_out

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        # __call__ accepts compact [M, 2, H] beliefs. The direct graph is only
        # valid for its captured full source_beliefs/row_indices pair, so use the
        # base compact pipeline here.
        self.beliefs_in.copy_(beliefs)
        ShowdownActiveCardBlockGraphRunner._launch_pipeline(self)
        return self.ev_out


class ShowdownFullyCompactDirectGraphRunner:
    """Compact exact showdown runner with direct sparse-card finish.

    This keeps the rank-prefix setup but skips materializing the
    `[M, 2, cards, slots]` card-cumsum buffer. The finish kernel sums the
    sparse blocker masses from `card_positions` directly.
    """

    def __init__(
        self,
        extras: dict,
        M: int,
        device: torch.device,
        block_k: int = 64,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.device = device
        self.block_k = block_k

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.b_opp_both = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.ev_out = torch.empty(M, 2, self.H, device=device, dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_b_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.b_opp_both,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_ev_active_direct_compact_kernel[grid](
            self.b_opp_both,
            self.P_padded,
            e["card_positions"],
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["c1"],
            e["c2"],
            e["scale_factor"],
            self.ev_out,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownFullyCompactScatterDirectGraphRunner:
    """Compact exact showdown with card-prefix scatter correction buffers."""

    def __init__(
        self,
        extras: dict,
        M: int,
        device: torch.device,
        block_k: int = 64,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.device = device
        self.block_k = block_k

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.b_opp_both = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.corr_L = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.corr_R = torch.empty_like(self.corr_L)
        self.corr_total = torch.empty_like(self.corr_L)
        self.ev_out = torch.empty(M, 2, self.H, device=device, dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_b_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.b_opp_both,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        self.corr_L.zero_()
        self.corr_R.zero_()
        self.corr_total.zero_()
        _showdown_card_corr_scatter_kernel[(self.M, 2, self.num_cards)](
            self.b_opp_both,
            extra_indices,
            e["card_positions"],
            e["occ_slot_L"],
            e["occ_slot_R"],
            e["card_slot_count"],
            self.corr_L,
            self.corr_R,
            self.corr_total,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            num_warps=2,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_corr_finish_compact_kernel[grid](
            self.b_opp_both,
            self.P_padded,
            self.corr_L,
            self.corr_R,
            self.corr_total,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["scale_factor"],
            self.ev_out,
            self.H,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out


class ShowdownFullyCompactDualSlotGraphRunner:
    """Compact exact showdown with deterministic per-card contribution stores."""

    def __init__(
        self,
        extras: dict,
        M: int,
        device: torch.device,
        block_k: int = 64,
    ) -> None:
        if not triton_is_available():
            raise RuntimeError("Triton is not installed.")
        self.extras = extras
        self.M = M
        self.device = device
        self.block_k = block_k

        self.H = extras["sorted_indices"].shape[1]
        self.num_cards = extras["card_positions"].shape[1]
        block_h = 1
        while block_h < self.H:
            block_h *= 2
        self.block_h = block_h

        dtype = torch.float32
        self.beliefs_in = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.P_padded = torch.empty(M, 2, self.H + 1, device=device, dtype=dtype)
        self.corr_L1 = torch.empty(M, 2, self.H, device=device, dtype=dtype)
        self.corr_L2 = torch.empty_like(self.corr_L1)
        self.corr_R1 = torch.empty_like(self.corr_L1)
        self.corr_R2 = torch.empty_like(self.corr_L1)
        self.corr_T1 = torch.empty_like(self.corr_L1)
        self.corr_T2 = torch.empty_like(self.corr_L1)
        self.ev_out = torch.empty(M, 2, self.H, device=device, dtype=dtype)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._launch_pipeline()
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._launch_pipeline()

    def _launch_pipeline(self) -> None:
        e = self.extras
        extra_indices = e["extra_indices"]
        _showdown_setup_P_compact_kernel[(self.M, 2)](
            self.beliefs_in,
            self.P_padded,
            self.H,
            BLOCK_H=self.block_h,
        )
        _showdown_card_corr_dualslot_kernel[(self.M, 2, self.num_cards)](
            self.beliefs_in,
            extra_indices,
            e["card_positions"],
            e["occ_slot_L"],
            e["occ_slot_R"],
            e["card_slot_count"],
            e["occ_is_c2"],
            self.corr_L1,
            self.corr_L2,
            self.corr_R1,
            self.corr_R2,
            self.corr_T1,
            self.corr_T2,
            self.H,
            NUM_CARDS=self.num_cards,
            MC_K=SHOWDOWN_MAX_PER_CARD,
            num_warps=2,
        )
        grid = (self.M, triton.cdiv(self.H, self.block_k))
        _showdown_dualslot_finish_compact_kernel[grid](
            self.beliefs_in,
            self.P_padded,
            self.corr_L1,
            self.corr_L2,
            self.corr_R1,
            self.corr_R2,
            self.corr_T1,
            self.corr_T2,
            extra_indices,
            e["L_idx"],
            e["R_idx"],
            e["scale_factor"],
            self.ev_out,
            self.H,
            BLOCK_K=self.block_k,
            num_warps=4,
        )

    def __call__(self, beliefs: torch.Tensor) -> torch.Tensor:
        self.beliefs_in.copy_(beliefs)
        if torch.cuda.is_current_stream_capturing():
            self._launch_pipeline()
        else:
            self.graph.replay()
        return self.ev_out
