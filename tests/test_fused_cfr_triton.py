"""Correctness tests for the fused Triton CFR helpers."""

from __future__ import annotations

import pytest
import torch

from p2.core.structured_config import CFRType


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "cfr_type,cfr_plus",
    [
        (CFRType.discounted, False),
        (CFRType.discounted_plus, False),
        (CFRType.discounted_plus, True),
    ],
)
@pytest.mark.parametrize("t", [1, 7, 42])
def test_fused_dcfr_update_matches_pytorch(cfr_type: CFRType, cfr_plus: bool, t: int) -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_dcfr_update_

    device = torch.device("cuda")
    torch.manual_seed(0xC0FFEE + t + int(cfr_plus))

    shape = (137, 1326)  # realistic-ish [total_nodes, NUM_HANDS]
    cumul = torch.randn(shape, device=device, dtype=torch.float32)
    cumul[:20] = cumul[:20].abs()  # some strictly positive
    cumul[20:40] = -cumul[20:40].abs()  # some strictly negative
    weight = torch.rand(shape, device=device, dtype=torch.float32) * 5.0
    regrets = torch.randn(shape, device=device, dtype=torch.float32) * 0.1

    alpha, beta = 1.5, 0.5

    # Reference: replicate cfr_iteration block exactly.
    cumul_ref = cumul.clone()
    weight_ref = weight.clone()
    r_ref = regrets.clone()

    if cfr_type in (CFRType.discounted, CFRType.discounted_plus):
        numerator = torch.where(
            cumul_ref > 0, t**alpha, t**beta
        )
        denominator = torch.where(
            cumul_ref > 0, t**alpha + 1, t**beta + 1
        )
        cumul_ref *= numerator
        cumul_ref /= denominator
        weight_ref *= numerator
        weight_ref /= denominator

    weight_ref += 1
    cumul_ref += r_ref
    if cfr_plus:
        cumul_ref.clamp_(min=0)
    pos_ref = cumul_ref.clamp(min=0)

    # Fused kernel.
    cumul_out = cumul.clone()
    weight_out = weight.clone()
    pos_out = torch.empty_like(cumul_out)
    fused_dcfr_update_(
        cumulative_regrets=cumul_out,
        regret_weight_sums=weight_out,
        regrets=regrets,
        t=t,
        cfr_type=cfr_type,
        dcfr_alpha=alpha,
        dcfr_beta=beta,
        cfr_plus=cfr_plus,
        positive_regrets_out=pos_out,
    )

    # Tolerances accommodate Triton-side FMA fusion vs PyTorch's separate
    # mul/div kernels; absolute diffs observed at ~1 ULP (5e-7).
    torch.testing.assert_close(cumul_out, cumul_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(weight_out, weight_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pos_out, pos_ref, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# CUDA-graph capture correctness.
# ---------------------------------------------------------------------------


def _build_evaluator(num_envs: int = 4, depth: int = 3):
    """Construct a small SparseCFREvaluator for testing."""
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from p2.core.structured_config import Config
    from p2.env.hunl_tensor_env import HUNLTensorEnv
    from p2.models.mlp.rebel_ffn import RebelFFN
    from p2.search.sparse_cfr_evaluator import SparseCFREvaluator

    conf_dir = str((__import__("pathlib").Path(__file__).parent.parent / "conf").resolve())
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        dcfg = compose(
            config_name="config_rebel_cfr",
            overrides=[
                f"num_envs={num_envs}",
                f"search.depth={depth}",
                "search.iterations=10",
                "search.warm_start_iterations=0",
                "use_wandb=false",
            ],
        )
    cfg = Config.from_dict_config(dcfg)

    device = torch.device("cuda")
    torch.manual_seed(cfg.seed)

    env = HUNLTensorEnv(
        num_envs=cfg.num_envs,
        starting_stack=cfg.env.stack,
        sb=cfg.env.sb,
        bb=cfg.env.bb,
        default_bet_bins=cfg.env.bet_bins,
        device=device,
        float_dtype=torch.float32,
        flop_showdown=cfg.env.flop_showdown,
    )
    env.reset()
    root_indices = torch.arange(cfg.num_envs, dtype=torch.long, device=device)

    model = RebelFFN(
        input_dim=cfg.model.input_dim,
        num_actions=cfg.model.num_actions,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
        detach_value_head=cfg.model.detach_value_head,
        num_players=2,
    )
    cpu_rng = torch.Generator(device="cpu").manual_seed(42)
    model.init_weights(cpu_rng)
    model.to(device).eval()

    evaluator = SparseCFREvaluator(model=model, device=device, cfg=cfg)
    evaluator.initialize_subgame(env, root_indices)
    evaluator.initialize_policy_and_beliefs()
    evaluator.set_leaf_values(0)
    evaluator.compute_expected_values()
    evaluator.values_avg[:] = evaluator.latest_values
    evaluator.t_sample = evaluator._get_sampling_schedule()
    return evaluator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_block_and_normalize_beliefs_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_block_and_normalize_beliefs_

    device = torch.device("cuda")
    torch.manual_seed(7)
    n, p, h = 53, 2, 1326
    target = torch.rand(n, p, h, device=device)
    # Force some rows to be near-zero after masking to exercise the fallback.
    allowed = torch.rand(n, h, device=device) > 0.3
    allowed[10:15] = False  # fully blocked rows → fallback branch
    allowed_prob = torch.rand(n, h, device=device)
    allowed_prob *= allowed.float()
    allowed_prob /= allowed_prob.sum(-1, keepdim=True).clamp(min=1e-8)

    target_ref = target.clone()
    # Reference: _block_beliefs then _normalize_beliefs from cfr_evaluator.
    target_ref.masked_fill_((~allowed)[:, None, :], 0.0)
    denom = target_ref.sum(dim=-1, keepdim=True)
    target_ref = torch.where(denom > 1e-5, target_ref / denom, allowed_prob[:, None, :])

    target_fused = target.clone().contiguous()
    fused_block_and_normalize_beliefs_(
        target_fused, allowed.contiguous(), allowed_prob.contiguous()
    )
    torch.testing.assert_close(target_fused, target_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_regret_matching_divide_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_regret_matching_divide_

    device = torch.device("cuda")
    torch.manual_seed(11)
    n, h = 400, 1326
    pos = torch.rand(n, h, device=device)
    denom = torch.rand(n, h, device=device)
    # Force some rows near zero to exercise fallback.
    denom[:50] = 0.0
    uniform = torch.rand(n, h, device=device)

    ref = torch.where(denom > 1e-8, pos / denom.clamp(min=1e-8), uniform)

    out = torch.empty_like(pos)
    fused_regret_matching_divide_(pos, denom, uniform, out)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_weight_child_values_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_weight_child_values

    device = torch.device("cuda")
    torch.manual_seed(13)
    m, h = 97, 1326
    values = torch.randn(m, 2, h, device=device)
    prev_actor = torch.randint(0, 2, (m,), device=device, dtype=torch.long)
    policy_hero = torch.rand(m, h, device=device)
    policy_opp = torch.rand(m, h, device=device)

    # Reference: clone + two fancy-index in-place multiplies.
    ref = values.clone()
    idx = torch.arange(m, device=device)
    ref[idx, prev_actor, :] *= policy_hero
    ref[idx, 1 - prev_actor, :] *= policy_opp

    out = torch.empty_like(values)
    fused_weight_child_values(
        values_src=values.contiguous(),
        prev_actor=prev_actor.contiguous(),
        policy_hero=policy_hero.contiguous(),
        policy_opp=policy_opp.contiguous(),
        out=out,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_update_average_values_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_update_average_values_

    device = torch.device("cuda")
    torch.manual_seed(17)
    avg = torch.randn(40, 2, 1326, device=device)
    latest = torch.randn_like(avg)
    old, new = 3.0, 2.0

    ref = avg.clone()
    ref *= old
    ref += new * latest
    ref /= old + new

    out = avg.clone().contiguous()
    fused_update_average_values_(out, latest.contiguous(), old, new)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_regret_tail_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_regret_tail_

    device = torch.device("cuda")
    torch.manual_seed(19)

    total = 80
    top = 30     # parents live in [0, top)
    bottom = top  # children live in [bottom, total)
    h = 1326

    values_achieved = torch.randn(total, 2, h, device=device)
    actor_values = torch.randn(top, h, device=device)
    src_weights = torch.randn(top, h, device=device)
    # Children's parent_index points into [0, top).
    parent_index = torch.randint(0, top, (total,), device=device, dtype=torch.long)
    prev_actor = torch.randint(0, 2, (total,), device=device, dtype=torch.long)

    # Reference via explicit PyTorch sequence with parent_index gather for weights.
    ref = torch.zeros(total, h, device=device)
    expected = actor_values[parent_index[bottom:]]
    weights = src_weights[parent_index[bottom:]]
    idx = torch.arange(total - bottom, device=device)
    achieved = values_achieved[bottom:][idx, prev_actor[bottom:], :]
    ref[bottom:] = weights * (achieved - expected)

    out = torch.zeros(total, h, device=device).contiguous()
    fused_regret_tail_(
        regrets=out,
        values_achieved=values_achieved.contiguous(),
        actor_values=actor_values.contiguous(),
        src_weights=src_weights.contiguous(),
        parent_index=parent_index.contiguous(),
        prev_actor=prev_actor.contiguous(),
        bottom=bottom,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unblocked_mass_ratio_indirect_matches_baseline() -> None:
    pytest.importorskip("triton")
    from p2.env.card_utils import calculate_unblocked_mass
    from p2.search.fused_cfr_triton import unblocked_mass_ratio_indirect_triton

    device = torch.device("cuda")
    torch.manual_seed(53)
    top = 50
    num_children = 250
    h = 1326

    actor_beliefs = torch.rand(top, h, device=device)
    parent_index = torch.randint(0, top, (num_children,), device=device, dtype=torch.long)
    policy = torch.rand(num_children, h, device=device)
    marginal_policy = actor_beliefs[parent_index] * policy

    # Reference: full pipeline using the dense GEMM baseline.
    beliefs_dest = actor_beliefs[parent_index]  # [num_children, H]
    n = calculate_unblocked_mass(marginal_policy)
    d = calculate_unblocked_mass(beliefs_dest)
    ref = torch.where(d > 1e-5, n / d.clamp(min=1e-20), torch.zeros_like(d))

    out = unblocked_mass_ratio_indirect_triton(
        numer_target=marginal_policy.contiguous(),
        denom_target=actor_beliefs.contiguous(),
        parent_index=parent_index.contiguous(),
    )
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unblocked_mass_opp_at_parents_matches_full_pipeline() -> None:
    pytest.importorskip("triton")
    from p2.env.card_utils import calculate_unblocked_mass
    from p2.search.fused_cfr_triton import unblocked_mass_opp_at_parents_triton

    device = torch.device("cuda")
    torch.manual_seed(47)
    total, h = 100, 1326
    top = 40
    beliefs = torch.rand(total, 2, h, device=device)
    to_act = torch.randint(0, 2, (total,), device=device, dtype=torch.long)

    # Reference: the original pipeline.
    opp = calculate_unblocked_mass(beliefs.flip(dims=[1]))
    src_actor = to_act[:, None, None].expand(-1, -1, h)
    ref = opp.gather(1, src_actor).squeeze(1)[:top]

    out = unblocked_mass_opp_at_parents_triton(beliefs, to_act, top)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_average_policy_mix_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_average_policy_mix_

    device = torch.device("cuda")
    torch.manual_seed(23)

    total = 100
    bottom = 20  # children start here
    h = 1326

    self_reach = torch.rand(total, 2, h, device=device)
    self_reach_avg = torch.rand(total, 2, h, device=device)
    policy = torch.rand(total, h, device=device)
    policy /= policy.sum(-1, keepdim=True).clamp(min=1e-8)
    policy_avg = torch.rand(total, h, device=device)
    policy_avg /= policy_avg.sum(-1, keepdim=True).clamp(min=1e-8)
    # Force some (parent, actor) combinations to give den ≈ 0 to exercise fallback.
    self_reach[:5] = 0.0
    self_reach_avg[:5] = 0.0
    to_act = torch.randint(0, 2, (total,), device=device, dtype=torch.long)
    parent_index = torch.randint(0, bottom, (total,), device=device, dtype=torch.long)

    old, new = 4.0, 2.0

    # Reference: mirror fused kernel math exactly.
    ref = policy_avg.clone()
    for c in range(bottom, total):
        parent = parent_index[c].item()
        actor = to_act[parent].item()
        reach_a = self_reach_avg[parent, actor] * old
        reach_n = self_reach[parent, actor] * new
        avg_row = policy_avg[c]
        cur_row = policy[c]
        num = reach_a * avg_row + reach_n * cur_row
        den = reach_a + reach_n
        unw = (old * avg_row + new * cur_row) / (old + new)
        ref[c] = torch.where(den > 1e-5, num / den, unw)

    out = policy_avg.clone().contiguous()
    fused_average_policy_mix_(
        policy_probs_avg=out,
        policy_probs=policy.contiguous(),
        self_reach=self_reach.contiguous(),
        self_reach_avg=self_reach_avg.contiguous(),
        to_act=to_act.contiguous(),
        parent_index=parent_index.contiguous(),
        old=old,
        new=new,
        bottom=bottom,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("shape", [(1, 1326), (37, 1326), (13, 2, 1326), (200, 2, 1326)])
def test_unblocked_mass_triton_matches_pytorch(shape) -> None:
    pytest.importorskip("triton")
    from p2.env.card_utils import calculate_unblocked_mass
    from p2.search.fused_cfr_triton import unblocked_mass_triton

    device = torch.device("cuda")
    torch.manual_seed(29)
    target = torch.rand(shape, device=device, dtype=torch.float32)

    ref = calculate_unblocked_mass(target)
    out = unblocked_mass_triton(target)

    # O(N) formula is numerically easier than the fp64 GEMM → tight tolerance.
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unblocked_mass_triton_edge_cases() -> None:
    """Zero target, point-mass targets, and targets concentrated on one card."""
    pytest.importorskip("triton")
    from p2.env.card_utils import calculate_unblocked_mass, hand_combos_tensor
    from p2.search.fused_cfr_triton import unblocked_mass_triton

    device = torch.device("cuda")
    combos = hand_combos_tensor(device=device)

    # Zero target → zero output.
    zero = torch.zeros(4, 1326, device=device)
    torch.testing.assert_close(
        unblocked_mass_triton(zero), calculate_unblocked_mass(zero)
    )

    # Point mass on combo 0: output should equal compatibility row 0.
    point = torch.zeros(1, 1326, device=device)
    point[0, 0] = 1.0
    ref = calculate_unblocked_mass(point)
    out = unblocked_mass_triton(point)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    # Self-compatibility should be zero (a combo blocks itself).
    assert out[0, 0].item() == 0.0

    # Target mass on all combos containing card 0 → output should be zero on
    # all combos that share card 0.
    concentrated = torch.zeros(1, 1326, device=device)
    card0_mask = (combos[:, 0] == 0) | (combos[:, 1] == 0)
    concentrated[0, card0_mask] = 1.0
    ref = calculate_unblocked_mass(concentrated)
    out = unblocked_mass_triton(concentrated)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    # All combos containing card 0 should have unblocked mass 0 (blocked by all
    # of our support).
    assert torch.all(out[0, card0_mask] == 0).item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_parent_sum_and_divide_matches_baseline() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import (
        fused_divide_by_parent_sum_,
        fused_parent_sum,
    )

    device = torch.device("cuda")
    torch.manual_seed(43)
    child_counts = torch.tensor([3, 2, 5, 1, 4], device=device, dtype=torch.long)
    bottom = 7
    num_parents = child_counts.numel()
    num_children = int(child_counts.sum().item())
    total = bottom + num_children
    h = 1326
    child_offsets = bottom + torch.cumsum(child_counts, dim=0) - child_counts

    values = torch.randn(total, h, device=device)
    pos = torch.rand(num_children, h, device=device)
    fallback = torch.rand(num_children, h, device=device)
    parent_index_rel = torch.repeat_interleave(
        torch.arange(num_parents, device=device), child_counts
    )

    # --- parent_sum kernel ---
    ref_parent_sum = torch.zeros(num_parents, h, device=device)
    for p in range(num_parents):
        first = int(child_offsets[p].item())
        cnt = int(child_counts[p].item())
        ref_parent_sum[p] = values[first : first + cnt].sum(0)

    out_parent_sum = fused_parent_sum(
        values=values.contiguous(),
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        max_children=8,
    )
    torch.testing.assert_close(out_parent_sum, ref_parent_sum, rtol=1e-5, atol=1e-5)

    # --- divide-by-parent-sum kernel ---
    denom = ref_parent_sum[parent_index_rel]  # [num_children, H]
    eps = 1e-8
    ref_div = torch.where(
        denom > eps, pos / denom.clamp(min=eps), fallback
    )
    out_div = torch.empty_like(pos)
    fused_divide_by_parent_sum_(
        pos=pos.contiguous(),
        fallback=fallback.contiguous(),
        parent_sum=out_parent_sum.contiguous(),
        parent_index=parent_index_rel.contiguous(),
        out=out_div,
        eps=eps,
    )
    torch.testing.assert_close(out_div, ref_div, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_sibling_sum_matches_scatter_plus_gather() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_sibling_sum

    device = torch.device("cuda")
    torch.manual_seed(31)

    # Small synthetic tree: 5 parents, each with 2-4 children.
    child_counts = torch.tensor([3, 2, 4, 2, 3], device=device, dtype=torch.long)
    bottom = 10  # assume 10 rows of other stuff before children
    num_parents = child_counts.numel()
    num_children = int(child_counts.sum().item())
    total = bottom + num_children
    h = 1326

    child_offsets = bottom + torch.cumsum(child_counts, dim=0) - child_counts

    values = torch.randn(total, h, device=device)

    # Reference: for each child, sum siblings.
    ref = torch.empty(num_children, h, device=device)
    for p in range(num_parents):
        first = int(child_offsets[p].item())
        count = int(child_counts[p].item())
        sib_sum = values[first : first + count].sum(dim=0)
        for i in range(count):
            ref[first + i - bottom] = sib_sum

    out = fused_sibling_sum(
        values=values.contiguous(),
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        bottom=bottom,
        num_children=num_children,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_unblocked_mass_ratio_matches_sequential_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.env.card_utils import calculate_unblocked_mass
    from p2.search.fused_cfr_triton import unblocked_mass_ratio_triton

    device = torch.device("cuda")
    torch.manual_seed(37)
    b, h = 120, 1326
    numer = torch.rand(b, h, device=device)
    denom = torch.rand(b, h, device=device)
    denom[:10] = 0.0  # exercise the fallback branch

    n = calculate_unblocked_mass(numer)
    d = calculate_unblocked_mass(denom)
    ref = torch.where(d > 1e-5, n / d.clamp(min=1e-20), torch.zeros_like(d))

    out = unblocked_mass_ratio_triton(numer, denom)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_model_values_mix_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_model_values_mix

    device = torch.device("cuda")
    torch.manual_seed(41)
    h = torch.randn(80, 2, 1326, device=device)
    l = torch.randn_like(h)
    old, new = 7.0, 3.0
    ref = ((old + new) * h - old * l) / new
    out = fused_model_values_mix(h.contiguous(), l.contiguous(), old, new)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_sparse_evaluator_matches_baseline_across_iterations() -> None:
    """Run N iterations of each evaluator from identical init; compare state."""
    pytest.importorskip("triton")
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    # Build one baseline evaluator and clone its init state into a fused one.
    ev_base = _build_evaluator(num_envs=4, depth=3)
    ev_fused = _build_evaluator(num_envs=4, depth=3)
    # Swap the fused evaluator's class to inherit the same initial tensors.
    ev_fused.__class__ = FusedSparseCFREvaluator
    ev_fused._fused_positive_regrets_buf = None

    # Mirror all mutable state from base → fused so they start identical.
    for name in [
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
    ]:
        getattr(ev_fused, name).copy_(getattr(ev_base, name))

    for t in range(1, 6):
        ev_base.cfr_iteration(t)
        ev_fused.cfr_iteration(t)
    torch.cuda.synchronize()

    for name in ["policy_probs", "cumulative_regrets", "beliefs", "self_reach", "values_avg"]:
        a = getattr(ev_base, name)
        b = getattr(ev_fused, name)
        # Accumulated rounding over 5 iterations; generous but still tight.
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4, msg=f"mismatch on {name}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graphed_cfr_iteration_matches_uncaptured() -> None:
    """FusedSparseCFREvaluator.set_leaf_values pins latest_values storage
    across calls (no rebinding), so the graph's captured kernels continue to
    write into self.latest_values on replay.
    """
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import (
        GraphedCFRIteration,
        _EvaluatorStateSnapshot,
    )
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    ev = _build_evaluator(num_envs=4, depth=3)
    # Swap to fused (pinned storage; device-scalar TScalars).
    ev.__class__ = FusedSparseCFREvaluator
    ev._ensure_fused_attrs()

    # Prime with a couple of iterations so we're past any t<=1 branches.
    for t in range(1, 4):
        ev.cfr_iteration(t)
    torch.cuda.synchronize()

    # Snapshot pre-state. This is what both "baseline" and "graph" start from.
    pre = _EvaluatorStateSnapshot.from_evaluator(ev)

    # Baseline: run one uncaptured iteration at t=T.
    T = 5
    # Stub _record_stats the same way the graph path does, so any nondeterminism
    # from that path is excluded from the comparison.
    orig_stats = ev._record_stats
    ev._record_stats = lambda t, old: None
    try:
        ev.cfr_iteration(T)
        torch.cuda.synchronize()
        baseline = _EvaluatorStateSnapshot.from_evaluator(ev)

        # Restore pre-state, capture one iter at t=T, replay.
        pre.restore_to(ev)
        torch.cuda.synchronize()

        runner = GraphedCFRIteration(ev)
        # The capture() method does its own warmup iterations that mutate state,
        # so we re-restore right before the actual captured region runs.
        # Here we call the lower-level path manually: warmup then capture.
        # Easiest: let capture() run, then restore, then replay — the captured
        # graph operates on the same underlying tensors so replay starts from
        # whatever state is in them.
        runner.capture(t_capture=T, num_warmup=1)

        pre.restore_to(ev)
        torch.cuda.synchronize()
        runner.replay()
        torch.cuda.synchronize()
        replayed = _EvaluatorStateSnapshot.from_evaluator(ev)
    finally:
        ev._record_stats = orig_stats

    diffs = baseline.max_abs_diff(replayed)
    worst = max(diffs.values())
    assert worst < 1e-4, f"Graph replay diverged from baseline: {diffs}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graphed_cfr_iteration_replays_across_t() -> None:
    """Capture once at t_capture, then replay for several different t values
    and check each replay matches a fresh uncaptured run at that t.
    """
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import (
        GraphedCFRIteration,
        _EvaluatorStateSnapshot,
    )
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    # Two independent evaluators: one for uncaptured baseline, one for graph.
    ev_base = _build_evaluator(num_envs=4, depth=3)
    ev_base.__class__ = FusedSparseCFREvaluator
    ev_base._ensure_fused_attrs()
    ev_graph = _build_evaluator(num_envs=4, depth=3)
    ev_graph.__class__ = FusedSparseCFREvaluator
    ev_graph._ensure_fused_attrs()

    # Mirror init state.
    for name in [
        "policy_probs", "policy_probs_avg", "policy_probs_sample",
        "cumulative_regrets", "regret_weight_sums",
        "self_reach", "self_reach_avg", "beliefs", "beliefs_avg",
        "latest_values", "values_avg",
    ]:
        getattr(ev_graph, name).copy_(getattr(ev_base, name))

    # Prime both past early-t branches so subsequent iters share Python branch.
    for t in range(1, 4):
        ev_base.cfr_iteration(t)
        ev_graph.cfr_iteration(t)
    torch.cuda.synchronize()

    # Stub _record_stats on both so comparison isn't polluted by .item() paths.
    ev_base._record_stats = lambda t, old: None
    ev_graph._record_stats = lambda t, old: None

    # Capture one iteration at t=4; we'll replay for t=5, 6, 7.
    runner = GraphedCFRIteration(ev_graph)
    runner.capture(t_capture=4, num_warmup=0)
    # Capture consumed one iteration at t=4, so both are now post-t=4.
    # Step baseline forward once (uncaptured) to match.
    ev_base.cfr_iteration(4)
    torch.cuda.synchronize()

    for replay_t in [5, 6, 7]:
        ev_base.cfr_iteration(replay_t)
        runner.replay(t=replay_t)
        torch.cuda.synchronize()
        a = _EvaluatorStateSnapshot.from_evaluator(ev_base)
        b = _EvaluatorStateSnapshot.from_evaluator(ev_graph)
        diffs = a.max_abs_diff(b)
        worst = max(diffs.values())
        assert worst < 1e-2, (
            f"t={replay_t}: replay diverges from baseline: {diffs}"
        )


# ---------------------------------------------------------------------------
# Kernel 12-14 correctness tests: weighted parent-sum, reach weights, deep beliefs.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_weighted_parent_sum_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_weighted_parent_sum

    device = torch.device("cuda")
    torch.manual_seed(71)

    # Tiny tree with variable child counts.
    child_counts = torch.tensor([3, 1, 4, 2], device=device, dtype=torch.long)
    parent_base = 5
    num_parents = child_counts.numel()
    num_children = int(child_counts.sum().item())
    total = parent_base + num_parents + num_children
    h = 1326

    child_offsets = (
        parent_base + num_parents + torch.cumsum(child_counts, dim=0) - child_counts
    )
    values = torch.randn(total, 2, h, device=device)
    # Pre-zero parent rows — kernel overwrites them.
    values[parent_base : parent_base + num_parents] = 0.0
    prev_actor = torch.randint(0, 2, (total,), device=device, dtype=torch.long)
    policy_hero = torch.rand(total, h, device=device)
    policy_opp = torch.rand(total, h, device=device)

    # Reference: loop over parents, sum weighted children.
    ref = values.clone()
    for p_rel in range(num_parents):
        first = int(child_offsets[p_rel].item())
        count = int(child_counts[p_rel].item())
        for player in range(2):
            acc = torch.zeros(h, device=device)
            for i in range(count):
                c = first + i
                pa = int(prev_actor[c].item())
                pol = policy_hero[c] if player == pa else policy_opp[c]
                acc = acc + values[c, player] * pol
            ref[parent_base + p_rel, player] = acc

    out = values.clone().contiguous()
    fused_weighted_parent_sum(
        values=out,
        prev_actor=prev_actor.contiguous(),
        policy_hero=policy_hero.contiguous(),
        policy_opp=policy_opp.contiguous(),
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        parent_base=parent_base,
        max_children=8,
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_reach_weights_depth_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_reach_weights_depth_

    device = torch.device("cuda")
    torch.manual_seed(73)

    # Simulate two depths of a tree: root rows [0, 4), depth-1 rows [4, 10).
    total, h = 10, 1326
    reach = torch.rand(total, 2, h, device=device)
    reach[:4] = 1.0  # root reach = 1
    policy = torch.rand(total, h, device=device)
    parent_index = torch.full((total,), -1, dtype=torch.long, device=device)
    parent_index[4:10] = torch.randint(0, 4, (6,), device=device, dtype=torch.long)
    prev_actor = torch.randint(0, 2, (total,), device=device, dtype=torch.long)
    prev_actor[:4] = -1

    # Reference: for each child c in [4, 10) and player p:
    # reach[c, p, h] = reach[parent, p, h] * (policy[c] if p==prev_actor[c] else 1)
    ref = reach.clone()
    for c in range(4, 10):
        parent = int(parent_index[c].item())
        pa = int(prev_actor[c].item())
        for player in range(2):
            base = ref[parent, player]
            if player == pa:
                ref[c, player] = base * policy[c]
            else:
                ref[c, player] = base

    out = reach.clone().contiguous()
    fused_reach_weights_depth_(
        reach=out,
        policy=policy.contiguous(),
        parent_index=parent_index.contiguous(),
        prev_actor=prev_actor.contiguous(),
        start=4,
        end=10,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_deep_beliefs_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_deep_beliefs_

    device = torch.device("cuda")
    torch.manual_seed(79)

    n, total, h = 3, 20, 1326
    root_beliefs = torch.rand(n, 2, h, device=device)
    reach = torch.rand(total, 2, h, device=device)
    root_index = torch.cat(
        [
            torch.arange(n, device=device, dtype=torch.long),
            torch.randint(0, n, (total - n,), device=device, dtype=torch.long),
        ]
    )
    allowed_mask = torch.rand(total, h, device=device) > 0.3
    # Force some rows to be fully blocked to exercise fallback.
    allowed_mask[5:7] = False
    allowed_prob = torch.rand(total, h, device=device)
    allowed_prob *= allowed_mask.float()
    allowed_prob /= allowed_prob.sum(-1, keepdim=True).clamp(min=1e-8)

    # Reference: replicate fan_out_deep * reach + block + normalize.
    fanned = root_beliefs[root_index]  # [total, 2, h]
    ref = fanned * reach
    ref.masked_fill_((~allowed_mask)[:, None, :], 0.0)
    denom = ref.sum(dim=-1, keepdim=True)
    ref = torch.where(denom > 1e-5, ref / denom, allowed_prob[:, None, :])

    out = torch.zeros_like(ref).contiguous()
    fused_deep_beliefs_(
        out=out,
        root_beliefs=root_beliefs.contiguous(),
        reach_weights=reach.contiguous(),
        allowed_mask=allowed_mask.contiguous(),
        allowed_prob=allowed_prob.contiguous(),
        root_index=root_index.contiguous(),
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
