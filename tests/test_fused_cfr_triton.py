"""Correctness tests for the fused Triton CFR helpers."""

from __future__ import annotations

import copy

import pytest
import torch

from p2.core.structured_config import CFRType


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "cfr_type,cfr_plus",
    [
        (CFRType.discounted, False),
        (CFRType.discounted, True),
    ],
)
@pytest.mark.parametrize("t", [1, 7, 42])
def test_fused_dcfr_update_matches_pytorch(
    cfr_type: CFRType, cfr_plus: bool, t: int
) -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_dcfr_update_

    device = torch.device("cuda")
    torch.manual_seed(0xC0FFEE + t + int(cfr_plus))

    shape = (137, 1326)  # realistic-ish [total_nodes, NUM_HANDS]
    cumul = torch.randn(shape, device=device, dtype=torch.float32)
    cumul[:20] = cumul[:20].abs()  # some strictly positive
    cumul[20:40] = -cumul[20:40].abs()  # some strictly negative
    regrets = torch.randn(shape, device=device, dtype=torch.float32) * 0.1

    alpha, beta = 1.5, 0.5

    # Reference: replicate cfr_iteration block exactly.
    cumul_ref = cumul.clone()
    r_ref = regrets.clone()

    if cfr_type == CFRType.discounted:
        numerator = torch.where(cumul_ref > 0, t**alpha, t**beta)
        denominator = torch.where(cumul_ref > 0, t**alpha + 1, t**beta + 1)
        cumul_ref *= numerator
        cumul_ref /= denominator

    cumul_ref += r_ref
    if cfr_plus:
        cumul_ref.clamp_(min=0)
    pos_ref = cumul_ref.clamp(min=0)

    # Fused kernel.
    cumul_out = cumul.clone()
    pos_out = torch.empty_like(cumul_out)
    fused_dcfr_update_(
        cumulative_regrets=cumul_out,
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
    torch.testing.assert_close(pos_out, pos_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_cfr_delta_stats_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_cfr_delta_stats

    device = torch.device("cuda")
    torch.manual_seed(0xC0DE)

    h = 1326
    child_counts = torch.tensor([3, 0, 5, 2, 4], device=device, dtype=torch.long)
    parent_count = child_counts.numel()
    child_offsets = torch.empty_like(child_counts)
    child_offsets[0] = parent_count
    child_offsets[1:] = parent_count + child_counts[:-1].cumsum(dim=0)
    total = int(parent_count + child_counts.sum().item())

    policy = torch.rand(total, h, device=device)
    old_policy = torch.rand_like(policy)
    self_reach = torch.zeros(total, 2, h, device=device)
    reachable = torch.rand(parent_count, h, device=device) > 0.2
    reachable[1, :7] = True
    self_reach[:parent_count, 0] = reachable.to(self_reach.dtype)

    expected_num = torch.zeros((), device=device)
    expected_den = torch.zeros((), device=device)
    for parent in range(parent_count):
        first = int(child_offsets[parent].item())
        count = int(child_counts[parent].item())
        node_reachable = reachable[parent]
        reachable_count = node_reachable.sum()
        if reachable_count > 0:
            child_delta = (
                policy[first : first + count] - old_policy[first : first + count]
            ).abs()
            delta_sum = child_delta[:, node_reachable].sum()
            expected_num += delta_sum / reachable_count
            expected_den += 1

    actual = fused_cfr_delta_stats(
        policy=policy,
        old_policy=old_policy,
        self_reach=self_reach,
        child_offsets=child_offsets,
        child_count=child_counts,
        max_children=8,
    )

    torch.testing.assert_close(actual[0], expected_num, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[1], expected_den, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# CUDA-graph capture correctness.
# ---------------------------------------------------------------------------


def test_fused_sparse_graph_capture_regime_splits_dcfr_delay() -> None:
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    ev = FusedSparseCFREvaluator.__new__(FusedSparseCFREvaluator)
    ev.cfr_type = CFRType.discounted
    ev.dcfr_delay = 5

    assert ev._graph_capture_regime(0) is None
    assert ev._graph_capture_regime(1) is None
    assert ev._graph_capture_regime(2) == "pre_dcfr_delay"
    assert ev._graph_capture_regime(5) == "pre_dcfr_delay"
    assert ev._graph_capture_regime(6) == "post_dcfr_delay"

    ev.cfr_type = CFRType.standard
    assert ev._graph_capture_regime(2) == "post_dcfr_delay"
    assert ev._graph_capture_regime(5) == "post_dcfr_delay"

    ev.cfr_type = CFRType.sapcfr
    ev._predictive_cfr_dcfr_hybrid = True
    assert ev._graph_capture_regime(2) == "pre_dcfr_delay"
    assert ev._graph_capture_regime(5) == "pre_dcfr_delay"
    assert ev._graph_capture_regime(6) == "post_dcfr_delay"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kernel_name", ["optimized", "legacy"])
def test_same_street_constructor_marks_implicit_allin_river_call(
    kernel_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("triton")
    from p2.env.hunl_tensor_env import HUNLTensorEnv
    from p2.search.subgame_constructor_triton import write_children_same_street_triton_

    if kernel_name == "legacy":
        monkeypatch.setenv("P2_WRITE_CHILDREN_KERNEL", "legacy")
    else:
        monkeypatch.delenv("P2_WRITE_CHILDREN_KERNEL", raising=False)

    device = torch.device("cuda")
    env = HUNLTensorEnv(
        num_envs=2,
        mean_stack=10000,
        sb=25,
        bb=50,
        device=device,
    )
    env.reset()
    env.street[0] = 3
    env.to_act[0] = 1
    env.button[0] = 0
    env.stacks[0] = torch.tensor([0, 9590], device=device)
    env.starting_stacks[0] = torch.tensor([9592, 9590], device=device)
    env.scale[0] = 9590
    env.committed[0] = torch.tensor([9592, 0], device=device)
    env.chips_placed[0] = torch.tensor([9592, 0], device=device)
    env.pot[0] = 9592
    env.is_allin[0] = torch.tensor([True, False], device=device)
    env.has_folded[0] = False
    env.done[0] = False
    env.actions_this_round[0] = 4
    env.actions_last_round[0] = 0

    num_actions = len(env.default_bet_bins) + 3
    legal_mask = torch.zeros(2, num_actions, dtype=torch.bool, device=device)
    legal_mask[0, 1] = True
    child_offsets = torch.zeros(1, dtype=torch.long, device=device)
    parent_index = torch.full((2,), -1, dtype=torch.long, device=device)
    action_from_parent = torch.full((2,), -1, dtype=torch.long, device=device)
    rewards = torch.zeros(2, dtype=torch.float32, device=device)
    allin_leaf = torch.zeros(2, dtype=torch.bool, device=device)
    bet_bins = torch.tensor(
        [0.0, 0.0, *env.default_bet_bins, 0.0],
        dtype=torch.float32,
        device=device,
    )

    write_children_same_street_triton_(
        env,
        legal_mask,
        child_offsets,
        parent_index,
        action_from_parent,
        rewards,
        allin_leaf,
        bet_bins,
        parent_start=0,
        parent_count=1,
        dst_start=1,
        allin_abstraction=True,
    )
    torch.cuda.synchronize()

    assert parent_index[1].item() == 0
    assert action_from_parent[1].item() == 1
    assert env.is_allin[1].tolist() == [True, True]
    assert env.done[1].item() is True
    assert env.street[1].item() == 4
    assert allin_leaf[1].item() is False


def _build_evaluator(num_envs: int = 2, depth: int = 2):
    """Construct a small SparseCFREvaluator for testing."""
    from hydra import compose, initialize_config_dir

    from p2.core.structured_config import Config
    from p2.env.hunl_tensor_env import HUNLTensorEnv
    from p2.models.mlp.rebel_ffn import RebelFFN
    from p2.search.sparse_cfr_evaluator import SparseCFREvaluator

    conf_dir = str(
        (__import__("pathlib").Path(__file__).parent.parent / "conf").resolve()
    )
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        dcfg = compose(
            config_name="config_rebel_cfr",
            overrides=[
                f"num_envs={num_envs}",
                f"search.depth={depth}",
                "search.iterations=8",
                "search.warm_start_iterations=0",
                "model.hidden_dim=32",
                "model.ffn_dim=32",
                "model.num_hidden_layers=1",
                "model.num_value_layers=1",
                "model.num_policy_layers=1",
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
def test_fused_initialize_subgame_invalidates_static_feature_cache() -> None:
    pytest.importorskip("triton")
    from hydra import compose, initialize_config_dir

    from p2.core.structured_config import Config
    from p2.env.hunl_tensor_env import HUNLTensorEnv
    from p2.env.rules import rank_hands as torch_rank_hands
    from p2.models.mlp.rebel_ffn import RebelFFN
    from p2.search import cfr_evaluator as cfr_evaluator_module
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    conf_dir = str(
        (__import__("pathlib").Path(__file__).parent.parent / "conf").resolve()
    )
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        dcfg = compose(
            config_name="config_rebel_cfr",
            overrides=[
                "num_envs=4",
                "search.depth=2",
                "search.iterations=3",
                "search.warm_start_iterations=0",
                "model.hidden_dim=32",
                "model.ffn_dim=32",
                "model.num_hidden_layers=1",
                "model.num_value_layers=1",
                "model.num_policy_layers=1",
                "use_wandb=false",
            ],
        )
    cfg = Config.from_dict_config(dcfg)
    device = torch.device("cuda")

    def make_env(scale_multiplier: int) -> HUNLTensorEnv:
        torch.manual_seed(11)
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
        env.scale *= scale_multiplier
        return env

    model = RebelFFN(
        input_dim=cfg.model.input_dim,
        num_actions=cfg.model.num_actions,
        hidden_dim=cfg.model.hidden_dim,
        num_hidden_layers=cfg.model.num_hidden_layers,
        detach_value_head=cfg.model.detach_value_head,
        num_players=2,
    ).to(device)
    evaluator = FusedSparseCFREvaluator(
        model=model,
        device=device,
        cfg=cfg,
        compile_model=False,
    )
    assert cfr_evaluator_module.rank_hands is torch_rank_hands
    root_indices = torch.arange(cfg.num_envs, dtype=torch.long, device=device)

    evaluator.initialize_subgame(make_env(2), root_indices)
    evaluator._model_features_for_beliefs(evaluator.beliefs[evaluator.model_indices])
    assert evaluator._static_model_feature_fields is not None
    stale_context = evaluator._static_model_feature_fields[0].clone()

    evaluator.initialize_subgame(make_env(3), root_indices)
    cached = evaluator._model_features_for_beliefs(
        evaluator.beliefs[evaluator.model_indices]
    )
    direct = evaluator.feature_encoder.encode(
        evaluator.beliefs,
        pre_chance_node=evaluator.new_street_mask,
        indices=evaluator.model_indices,
    )

    torch.cuda.synchronize()
    assert not torch.equal(cached.context, stale_context)
    torch.testing.assert_close(cached.context, direct.context)
    torch.testing.assert_close(cached.street, direct.street)
    torch.testing.assert_close(cached.to_act, direct.to_act)
    torch.testing.assert_close(cached.board, direct.board)


def _mirror_evaluator_state(src, dst) -> None:
    """Make two independently built evaluators start from the same subgame.

    The test helper constructs fresh envs, whose private/public cards may differ.
    Copy all tree/static tensors as well as mutable CFR tensors before comparing
    baseline and fused iteration math.
    """
    assert src.total_nodes == dst.total_nodes
    idx = torch.arange(src.total_nodes, dtype=torch.long, device=src.device)
    dst.env.copy_state_from(src.env, idx, idx)
    dst.depth_offsets = list(src.depth_offsets)
    dst.tree_depth = src.tree_depth
    dst.root_nodes = src.root_nodes
    if hasattr(src, "top_nodes"):
        dst.top_nodes = src.top_nodes
    dst.hand_rank_data = src.hand_rank_data

    for name in [
        "leaf_mask",
        "new_street_mask",
        "valid_mask",
        "model_indices",
        "legal_mask",
        "allowed_hands",
        "allowed_hands_prob",
        "parent_index",
        "action_from_parent",
        "child_count",
        "child_offsets",
        "prev_actor",
        "child_mask",
        "showdown_indices",
        "showdown_actors",
        "showdown_potential",
        "uniform_policy",
        "beliefs",
        "beliefs_avg",
        "root_pre_chance_beliefs",
        "latest_values",
        "values_avg",
        "self_reach",
        "self_reach_avg",
        "policy_probs",
        "policy_probs_avg",
        "policy_probs_sample",
        "cumulative_regrets",
        "t_sample",
    ]:
        if hasattr(src, name):
            setattr(dst, name, getattr(src, name).clone())
    dst.last_model_values = (
        None if src.last_model_values is None else src.last_model_values.clone()
    )


def _clone_evaluator_state(src):
    """Shallow-copy an initialized evaluator and clone top-level tensor state."""
    dst = copy.copy(src)
    for name, value in vars(src).items():
        if torch.is_tensor(value):
            setattr(dst, name, value.clone())
    dst.depth_offsets = list(src.depth_offsets)
    dst.last_model_values = (
        None if src.last_model_values is None else src.last_model_values.clone()
    )
    return dst


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
    top = 30  # parents live in [0, top)
    bottom = top  # children live in [bottom, total)
    h = 1326

    values_achieved = torch.randn(total, 2, h, device=device)
    values_expected = torch.randn(top, 2, h, device=device)
    to_act = torch.randint(0, 2, (top,), device=device, dtype=torch.long)
    src_weights = torch.randn(top, h, device=device)
    # Children's parent_index points into [0, top).
    parent_index = torch.randint(0, top, (total,), device=device, dtype=torch.long)
    prev_actor = torch.randint(0, 2, (total,), device=device, dtype=torch.long)

    # Reference via explicit PyTorch sequence with parent_index gather for weights
    # and the in-kernel actor pick equivalent: actor_values = values_expected[p, to_act[p]].
    ref = torch.zeros(total, h, device=device)
    parent_idx_children = parent_index[bottom:]
    actor_values = values_expected[
        torch.arange(top, device=device), to_act, :
    ]  # [top, h]
    expected = actor_values[parent_idx_children]
    weights = src_weights[parent_idx_children]
    idx = torch.arange(total - bottom, device=device)
    achieved = values_achieved[bottom:][idx, prev_actor[bottom:], :]
    ref[bottom:] = weights * (achieved - expected)

    out = torch.zeros(total, h, device=device).contiguous()
    fused_regret_tail_(
        regrets=out,
        values_achieved=values_achieved.contiguous(),
        values_expected=values_expected.contiguous(),
        to_act=to_act.contiguous(),
        src_weights=src_weights.contiguous(),
        parent_index=parent_index.contiguous(),
        prev_actor=prev_actor.contiguous(),
        bottom=bottom,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multiway_regret_src_weights_and_tail_match_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.env.card_utils import calculate_unblocked_mass
    from p2.search.fused_cfr_triton import (
        fused_regret_tail_multiway_,
        multiway_regret_src_weights_at_parents_triton,
    )

    device = torch.device("cuda")
    torch.manual_seed(1901)
    total, top, players, h = 17, 9, 4, 1326
    bottom = top

    beliefs = torch.rand(total, players, h, device=device)
    beliefs /= beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    to_act = torch.randint(0, players, (top,), device=device, dtype=torch.long)
    has_folded = torch.rand(total, players, device=device) < 0.2
    has_folded[torch.arange(top, device=device), to_act] = False
    allowed = torch.rand(top, h, device=device) > 0.05

    unblocked = calculate_unblocked_mass(beliefs[:top])
    player_ids = torch.arange(players, device=device)
    other_live = player_ids[None, :, None] != to_act[:, None, None]
    other_live &= ~has_folded[:top, :, None]
    ref_weights = torch.where(
        other_live,
        unblocked.clamp_min(1e-12),
        torch.ones_like(unblocked),
    ).prod(dim=1)
    ref_weights *= allowed.to(ref_weights.dtype)

    weights = multiway_regret_src_weights_at_parents_triton(
        beliefs.contiguous(),
        to_act.contiguous(),
        top,
        has_folded=has_folded.contiguous(),
        allowed_mask=allowed.contiguous(),
    )
    torch.testing.assert_close(weights, ref_weights, rtol=1e-4, atol=1e-6)

    values_achieved = torch.randn(total, players, h, device=device) * 0.1
    values_expected = torch.randn(top, players, h, device=device) * 0.1
    parent_index = torch.randint(0, top, (total,), device=device, dtype=torch.long)
    prev_actor = torch.randint(0, players, (total,), device=device, dtype=torch.long)

    ref = torch.zeros(total, h, device=device)
    child_parents = parent_index[bottom:]
    expected = values_expected[
        child_parents,
        to_act[child_parents],
        :,
    ]
    idx = torch.arange(total - bottom, device=device)
    achieved = values_achieved[bottom:][idx, prev_actor[bottom:], :]
    ref[bottom:] = ref_weights[child_parents] * (achieved - expected)

    out = torch.zeros(total, h, device=device).contiguous()
    fused_regret_tail_multiway_(
        regrets=out,
        values_achieved=values_achieved.contiguous(),
        values_expected=values_expected.contiguous(),
        to_act=to_act.contiguous(),
        src_weights=weights.contiguous(),
        parent_index=parent_index.contiguous(),
        prev_actor=prev_actor.contiguous(),
        bottom=bottom,
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multiway_ev_backup_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.env.card_utils import calculate_unblocked_mass
    from p2.search.fused_cfr_triton import (
        _preprocess_unblocked_stats,
        fused_weighted_parent_sum_inline_opp_multiway,
        select_actor_beliefs_and_marginal_policy_multiway_triton_out_,
    )

    device = torch.device("cuda")
    torch.manual_seed(1902)
    players, h = 4, 1326
    child_counts = torch.tensor([2, 3, 1], device=device, dtype=torch.long)
    top = child_counts.numel()
    bottom = top
    child_offsets = torch.empty_like(child_counts)
    child_offsets[0] = bottom
    child_offsets[1:] = bottom + child_counts[:-1].cumsum(0)
    total = int((bottom + child_counts.sum()).item())

    beliefs = torch.rand(total, players, h, device=device)
    beliefs /= beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    to_act = torch.randint(0, players, (top,), device=device, dtype=torch.long)
    prev_actor = torch.zeros(total, device=device, dtype=torch.long)
    policy = torch.zeros(total, h, device=device)
    for parent in range(top):
        first = int(child_offsets[parent].item())
        count = int(child_counts[parent].item())
        prev_actor[first : first + count] = to_act[parent]
        raw = torch.rand(count, h, device=device)
        policy[first : first + count] = raw / raw.sum(dim=0, keepdim=True)
    values = torch.randn(total, players, h, device=device) * 0.1

    actor_beliefs_ref = beliefs[
        torch.arange(top, device=device),
        to_act,
        :,
    ]
    parent_index = torch.repeat_interleave(
        torch.arange(top, device=device),
        child_counts,
    )
    beliefs_dest = actor_beliefs_ref[parent_index]
    marginal_ref = beliefs_dest * policy[bottom:]
    policy_blocked = calculate_unblocked_mass(marginal_ref)
    matchup = calculate_unblocked_mass(beliefs_dest)
    opp_policy = torch.where(
        matchup > 1e-5,
        policy_blocked / matchup.clamp_min(1e-20),
        torch.zeros_like(matchup),
    )

    ref = values.clone()
    player_ids = torch.arange(players, device=device)
    for parent in range(top):
        first = int(child_offsets[parent].item())
        count = int(child_counts[parent].item())
        acc = torch.zeros(players, h, device=device)
        for child in range(first, first + count):
            rel = child - bottom
            weights = torch.where(
                player_ids[:, None] == prev_actor[child],
                policy[child][None, :],
                opp_policy[rel][None, :],
            )
            acc += ref[child] * weights
        ref[parent] = acc

    actor_beliefs = torch.empty(top, h, device=device)
    marginal = torch.empty(total - bottom, h, device=device)
    select_actor_beliefs_and_marginal_policy_multiway_triton_out_(
        beliefs.contiguous(),
        to_act.contiguous(),
        policy.contiguous(),
        child_offsets.contiguous(),
        child_counts.contiguous(),
        bottom,
        actor_beliefs,
        marginal,
        max_children=4,
    )
    torch.testing.assert_close(actor_beliefs, actor_beliefs_ref, rtol=0, atol=0)
    torch.testing.assert_close(marginal, marginal_ref, rtol=1e-6, atol=1e-7)

    numer_s, numer_cardsum = _preprocess_unblocked_stats(marginal.contiguous())
    denom_s, denom_cardsum = _preprocess_unblocked_stats(actor_beliefs.contiguous())
    out = values.clone().contiguous()
    fused_weighted_parent_sum_inline_opp_multiway(
        values=out,
        prev_actor=prev_actor.contiguous(),
        policy_hero=policy.contiguous(),
        actor_beliefs=actor_beliefs.contiguous(),
        numer_s=numer_s,
        numer_cardsum=numer_cardsum,
        denom_s=denom_s,
        denom_cardsum=denom_cardsum,
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        parent_base=0,
        child_base=bottom,
        max_children=4,
    )
    torch.testing.assert_close(out[:top], ref[:top], rtol=1e-4, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multiway_average_policy_update_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import (
        fused_average_policy_mix_multiway_with_tensors_,
        fused_policy_renorm_reach_depth_multiway_,
    )

    device = torch.device("cuda")
    torch.manual_seed(1903)
    parents, players, h = 4, 4, 1326
    child_counts = torch.tensor([2, 1, 3, 2], device=device, dtype=torch.long)
    bottom = parents
    child_offsets = torch.empty_like(child_counts)
    child_offsets[0] = bottom
    child_offsets[1:] = bottom + child_counts[:-1].cumsum(0)
    total = int((bottom + child_counts.sum()).item())
    parent_index = torch.zeros(total, device=device, dtype=torch.long)
    for parent in range(parents):
        first = int(child_offsets[parent].item())
        count = int(child_counts[parent].item())
        parent_index[first : first + count] = parent

    to_act = torch.randint(0, players, (total,), device=device, dtype=torch.long)
    prev_actor = torch.zeros(total, device=device, dtype=torch.long)
    policy = torch.zeros(total, h, device=device)
    for parent in range(parents):
        first = int(child_offsets[parent].item())
        count = int(child_counts[parent].item())
        prev_actor[first : first + count] = to_act[parent]
        raw = torch.rand(count, h, device=device)
        policy[first : first + count] = raw / raw.sum(dim=0, keepdim=True)

    self_reach = torch.rand(total, players, h, device=device)
    avg = torch.rand(total, h, device=device)
    num = torch.rand(total, h, device=device) * 0.1
    den = torch.rand(total, h, device=device) * 0.1
    allowed = torch.rand(total, h, device=device) > 0.05
    new = torch.tensor(7.0, device=device)

    ref_avg = avg.clone()
    ref_num = num.clone()
    ref_den = den.clone()
    row = torch.arange(total - bottom, device=device) + bottom
    parents_for_child = parent_index[bottom:]
    actor_reach = self_reach[parents_for_child, to_act[parents_for_child]]
    contribution = actor_reach * new
    ref_num[bottom:] += contribution * policy[bottom:]
    ref_den[bottom:] += contribution
    ref_avg[:bottom] = 0.0
    ref_avg[bottom:] = torch.where(
        ref_den[bottom:] > 1e-5,
        ref_num[bottom:] / ref_den[bottom:].clamp_min(1e-5),
        policy[bottom:],
    )

    ref_reach = self_reach.clone()
    player_ids = torch.arange(players, device=device)
    for parent in range(parents):
        first = int(child_offsets[parent].item())
        count = int(child_counts[parent].item())
        denom = ref_avg[first : first + count].sum(dim=0)
        for child in range(first, first + count):
            pol = torch.where(
                denom > 1e-5,
                ref_avg[child] / denom.clamp_min(1e-5),
                ref_avg[child],
            )
            ref_avg[child] = pol
            child_reach = torch.where(
                player_ids[:, None] == prev_actor[child],
                ref_reach[parent] * pol[None, :],
                ref_reach[parent],
            )
            ref_reach[child] = torch.where(allowed[child][None, :], child_reach, 0.0)

    out_avg = avg.clone().contiguous()
    out_num = num.clone().contiguous()
    out_den = den.clone().contiguous()
    out_reach = self_reach.clone().contiguous()
    fused_average_policy_mix_multiway_with_tensors_(
        policy_probs_avg=out_avg,
        average_policy_numerator=out_num,
        average_policy_denominator=out_den,
        policy_probs=policy.contiguous(),
        self_reach=self_reach.contiguous(),
        to_act=to_act.contiguous(),
        parent_index=parent_index.contiguous(),
        new=new,
        bottom=bottom,
    )
    fused_policy_renorm_reach_depth_multiway_(
        policy=out_avg,
        reach=out_reach,
        allowed_mask=allowed.contiguous(),
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        prev_actor=prev_actor.contiguous(),
        parent_base=0,
        max_children=4,
        update_reach=True,
    )

    torch.testing.assert_close(out_num[row], ref_num[row], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_den[row], ref_den[row], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_avg, ref_avg, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_reach[bottom:], ref_reach[bottom:], rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multiway_average_values_update_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_avg_values_multiway_

    device = torch.device("cuda")
    torch.manual_seed(1904)
    rows, players, h = 13, 4, 1326
    avg = torch.randn(rows, players, h, device=device) * 0.1
    latest = torch.randn_like(avg) * 0.1
    beliefs = torch.rand_like(avg)
    beliefs /= beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ignore = torch.zeros(rows, dtype=torch.bool, device=device)
    ignore[3] = True
    old = torch.tensor(5.0, device=device)
    new = torch.tensor(2.0, device=device)
    inv = torch.tensor(1.0 / 7.0, device=device)

    ref = (avg * old + latest * new) * inv
    correction = (ref * beliefs).sum(dim=2, keepdim=True).mean(dim=1, keepdim=True)
    correction.masked_fill_(ignore[:, None, None], 0.0)
    ref -= correction

    out = avg.clone().contiguous()
    fused_avg_values_multiway_(
        values_avg=out,
        latest_values=latest.contiguous(),
        beliefs=beliefs.contiguous(),
        old=old,
        new=new,
        inv_total=inv,
        enforce_zero_sum=True,
        ignore_mask=ignore.contiguous(),
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_accumulate_weighted_values_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_accumulate_weighted_values_

    device = torch.device("cuda")
    torch.manual_seed(1905)
    accum = torch.randn(17, 6, 169, device=device)
    latest = torch.randn_like(accum)
    weight = torch.tensor(0.375, device=device)
    ref = accum + latest * weight

    out = accum.clone().contiguous()
    fused_accumulate_weighted_values_(out, latest.contiguous(), weight)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_multiway_model_values_writeback_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_model_values_writeback_multiway_

    device = torch.device("cuda")
    torch.manual_seed(1905)
    model_rows, total, players, h = 5, 11, 4, 1326
    hand_values = torch.randn(model_rows, players, h, device=device) * 0.1
    last_model = torch.randn_like(hand_values) * 0.1
    beliefs = torch.rand_like(hand_values)
    beliefs /= beliefs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    model_indices = torch.tensor([0, 2, 5, 7, 10], device=device, dtype=torch.long)
    latest = torch.randn(total, players, h, device=device) * 0.1
    last_out = torch.empty_like(hand_values)
    onon = torch.tensor(1.75, device=device)
    oon = torch.tensor(0.75, device=device)

    mixed = hand_values * onon - last_model * oon
    correction = (mixed * beliefs).sum(dim=2, keepdim=True).mean(dim=1, keepdim=True)
    ref_latest = latest.clone()
    ref_latest[model_indices] = mixed - correction

    fused_model_values_writeback_multiway_(
        hand_values=hand_values.contiguous(),
        last_model_values=last_model.contiguous(),
        beliefs=beliefs.contiguous(),
        model_indices=model_indices.contiguous(),
        latest_values=latest,
        last_out=last_out,
        old_plus_new_over_new=onon,
        old_over_new=oon,
        do_mix=True,
        enforce_zero_sum=True,
        store_last=True,
    )
    torch.testing.assert_close(latest, ref_latest, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(last_out, hand_values, rtol=0, atol=0)


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
    parent_index = torch.randint(
        0, top, (num_children,), device=device, dtype=torch.long
    )
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
def test_select_opponent_beliefs_matches_advanced_indexing() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import select_opponent_beliefs_triton_out_

    device = torch.device("cuda")
    torch.manual_seed(53)
    top, total, h = 17, 23, 1326
    beliefs = torch.rand(total, 2, h, device=device)
    to_act = torch.randint(0, 2, (total,), device=device)
    out = torch.empty(top, h, device=device)

    select_opponent_beliefs_triton_out_(beliefs, to_act, top, out)

    row_idx = torch.arange(top, device=device)
    ref = beliefs[:top][row_idx, 1 - to_act[:top], :].contiguous()
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


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
    policy = torch.rand(total, h, device=device)
    policy /= policy.sum(-1, keepdim=True).clamp(min=1e-8)
    policy_avg = torch.rand(total, h, device=device)
    policy_avg /= policy_avg.sum(-1, keepdim=True).clamp(min=1e-8)
    avg_num = torch.rand(total, h, device=device)
    avg_den = torch.rand(total, h, device=device) * 4.0
    # Force some (parent, actor) combinations to give den ≈ 0 to exercise fallback.
    self_reach[:5] = 0.0
    to_act = torch.randint(0, 2, (total,), device=device, dtype=torch.long)
    parent_index = torch.randint(0, bottom, (total,), device=device, dtype=torch.long)
    fallback_rows = parent_index < 5
    avg_num[fallback_rows] = 0.0
    avg_den[fallback_rows] = 0.0

    new = 2.0

    # Reference: mirror fused kernel math exactly.
    ref = policy_avg.clone()
    ref_num = avg_num.clone()
    ref_den = avg_den.clone()
    ref[:bottom] = 0.0
    for c in range(bottom, total):
        parent = parent_index[c].item()
        actor = to_act[parent].item()
        reach_n = self_reach[parent, actor] * new
        cur_row = policy[c]
        ref_num[c] += reach_n * cur_row
        ref_den[c] += reach_n
        ref[c] = torch.where(ref_den[c] > 1e-5, ref_num[c] / ref_den[c], cur_row)

    out = policy_avg.clone().contiguous()
    out_num = avg_num.clone().contiguous()
    out_den = avg_den.clone().contiguous()
    fused_average_policy_mix_(
        policy_probs_avg=out,
        average_policy_numerator=out_num,
        average_policy_denominator=out_den,
        policy_probs=policy.contiguous(),
        self_reach=self_reach.contiguous(),
        to_act=to_act.contiguous(),
        parent_index=parent_index.contiguous(),
        new=new,
        bottom=bottom,
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_num, ref_num, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out_den, ref_den, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape", [(1, 1326), (37, 1326), (13, 2, 1326), (200, 2, 1326)]
)
def test_unblocked_mass_triton_matches_pytorch(shape) -> None:
    pytest.importorskip("triton")
    from p2.env.card_utils import calculate_unblocked_mass
    from p2.search.fused_cfr_triton import unblocked_mass_triton

    device = torch.device("cuda")
    torch.manual_seed(29)
    target = torch.rand(shape, device=device, dtype=torch.float32)

    ref = calculate_unblocked_mass(target)
    out = unblocked_mass_triton(target)

    # O(N) formula is numerically easier than the compatibility GEMM.
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
def test_fused_parent_sum_divide_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_parent_sum_divide_

    device = torch.device("cuda")
    torch.manual_seed(61)
    child_counts = torch.tensor([3, 0, 5, 1, 4], device=device, dtype=torch.long)
    bottom = 9
    num_parents = child_counts.numel()
    num_children = int(child_counts.sum().item())
    total = bottom + num_children
    h = 1326
    child_offsets = bottom + torch.cumsum(child_counts, dim=0) - child_counts

    values = torch.rand(total, h, device=device)
    fallback = torch.rand(num_children, h, device=device)

    ref = torch.empty(num_children, h, device=device)
    for p in range(num_parents):
        count = int(child_counts[p].item())
        if count == 0:
            continue
        first = int(child_offsets[p].item())
        rel = first - bottom
        vals = values[first : first + count]
        denom = vals.sum(0)
        ref[rel : rel + count] = torch.where(
            denom > 1e-8,
            vals / denom.clamp(min=1e-8),
            fallback[rel : rel + count],
        )

    out = torch.empty_like(ref)
    fused_parent_sum_divide_(
        values=values.contiguous(),
        fallback=fallback.contiguous(),
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        out=out,
        out_offset=bottom,
        max_children=8,
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_unblocked_regret_dcfr_update_predictive_matches_pytorch() -> None:
    pytest.importorskip("triton")
    from p2.env.card_utils import hand_combos_tensor
    from p2.search.fused_cfr_triton import (
        _preprocess_unblocked_stats_out,
        fused_unblocked_regret_dcfr_update_with_tensors_,
    )

    device = torch.device("cuda")
    torch.manual_seed(71)
    top = 5
    child_counts = torch.tensor([3, 0, 2, 4, 1], device=device, dtype=torch.long)
    bottom = top
    total = int(bottom + child_counts.sum().item())
    h = 1326
    child_offsets = bottom + torch.cumsum(child_counts, dim=0) - child_counts

    target = torch.rand(top, h, device=device)
    stats = torch.empty(top, 53, device=device)
    _preprocess_unblocked_stats_out(target.contiguous(), stats)
    allowed = torch.rand(top, h, device=device) > 0.1
    values_achieved = torch.randn(total, 2, h, device=device)
    values_expected = torch.randn(top, 2, h, device=device)
    to_act = torch.randint(0, 2, (top,), dtype=torch.long, device=device)
    prev_actor = torch.randint(0, 2, (total,), dtype=torch.long, device=device)
    cumulative = torch.randn(total, h, device=device)
    last = torch.randn_like(cumulative) * 0.1
    t_alpha_num = torch.tensor(7.0**1.5, device=device)
    t_beta_num = torch.tensor(7.0**0.5, device=device)
    t_alpha_den = t_alpha_num + 1.0
    t_beta_den = t_beta_num + 1.0
    prediction_scale = torch.tensor(1.0 / 3.0, device=device)
    current_player = torch.tensor(1, dtype=torch.long, device=device)

    cumulative_ref = cumulative.clone()
    last_ref = last.clone()
    pos_ref = torch.empty_like(cumulative)
    combos = hand_combos_tensor(device=device)
    ca = combos[:, 0]
    cb = combos[:, 1]
    for p in range(top):
        count = int(child_counts[p].item())
        if count == 0:
            continue
        first = int(child_offsets[p].item())
        total_mass = target[p].sum()
        card_mass = torch.zeros(52, device=device)
        card_mass.scatter_add_(0, ca, target[p])
        card_mass.scatter_add_(0, cb, target[p])
        src_w = (total_mass - card_mass[ca] - card_mass[cb] + target[p]).clamp_min(0.0)
        src_w = torch.where(allowed[p], src_w, torch.zeros_like(src_w))
        expected = values_expected[p, to_act[p]]
        for child in range(first, first + count):
            achieved = values_achieved[child, prev_actor[child]]
            regret = src_w * (achieved - expected)
            c = cumulative_ref[child]
            scale = torch.where(c > 0, t_alpha_num, t_beta_num)
            denom = torch.where(c > 0, t_alpha_den, t_beta_den)
            c = c * scale
            c = c / denom
            c = c + regret
            cumulative_ref[child] = c
            if prev_actor[child] != current_player:
                last_ref[child] = regret
            pos_ref[child] = (c + prediction_scale * last_ref[child]).clamp_min(0.0)

    cumulative_out = cumulative.clone()
    last_out = last.clone()
    pos_out = torch.empty_like(cumulative)
    fused_unblocked_regret_dcfr_update_with_tensors_(
        target=target.contiguous(),
        stats=stats.contiguous(),
        allowed_mask=allowed.contiguous(),
        values_achieved=values_achieved.contiguous(),
        values_expected=values_expected.contiguous(),
        to_act=to_act.contiguous(),
        child_offsets=child_offsets.contiguous(),
        child_count=child_counts.contiguous(),
        prev_actor=prev_actor.contiguous(),
        cumulative_regrets=cumulative_out,
        t_alpha_num=t_alpha_num,
        t_beta_num=t_beta_num,
        t_alpha_den=t_alpha_den,
        t_beta_den=t_beta_den,
        apply_dcfr=True,
        cfr_plus=False,
        max_children=4,
        positive_regrets_out=pos_out,
        last_instantaneous_regrets=last_out,
        prediction_scale=prediction_scale,
        current_player=current_player,
    )

    child_slice = slice(bottom, total)
    torch.testing.assert_close(
        cumulative_out[child_slice], cumulative_ref[child_slice], rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        last_out[child_slice], last_ref[child_slice], rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        pos_out[child_slice], pos_ref[child_slice], rtol=1e-4, atol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_policy_sample_update_matches_where() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import fused_policy_sample_update_

    device = torch.device("cuda")
    torch.manual_seed(67)
    total, h, num_iters = 113, 1326, 11
    policy = torch.rand(total, h, device=device)
    sample = torch.rand(total, h, device=device)
    t_sample = torch.randint(0, num_iters, (total,), device=device)

    counts_cpu = torch.bincount(t_sample.cpu(), minlength=num_iters)
    max_updates = int(counts_cpu.max().item())
    rows_cpu = torch.zeros(num_iters, max_updates, dtype=torch.long)
    for t_idx in range(num_iters):
        rows = torch.nonzero(t_sample.cpu() == t_idx, as_tuple=False).flatten()
        rows_cpu[t_idx, : rows.numel()] = rows
    rows = rows_cpu.to(device)
    counts = counts_cpu.to(device)

    for t_idx in [0, 3, 7]:
        ref = torch.where((t_sample == t_idx)[:, None], policy, sample)
        out = sample.clone().contiguous()
        t_tensor = torch.tensor(t_idx, device=device, dtype=torch.long)
        fused_policy_sample_update_(
            policy.contiguous(),
            out,
            rows.contiguous(),
            counts.contiguous(),
            t_tensor,
        )
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_preflop_sample_snapshot_rows_multiway_matches_where() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import (
        fused_preflop_sample_snapshot_rows_multiway_,
    )

    device = torch.device("cuda")
    torch.manual_seed(71)
    total, players, h, num_iters = 97, 6, 169, 13
    policy = torch.rand(total, h, device=device)
    policy_sample = torch.rand(total, h, device=device)
    beliefs = torch.rand(total, players, h, device=device)
    beliefs_sample = torch.rand(total, players, h, device=device)
    t_sample = torch.randint(0, num_iters, (total,), device=device)

    counts_cpu = torch.bincount(t_sample.cpu(), minlength=num_iters)
    max_updates = int(counts_cpu.max().item())
    rows_cpu = torch.zeros(num_iters, max_updates, dtype=torch.long)
    for t_idx in range(num_iters):
        rows = torch.nonzero(t_sample.cpu() == t_idx, as_tuple=False).flatten()
        rows_cpu[t_idx, : rows.numel()] = rows
    rows = rows_cpu.to(device)
    counts = counts_cpu.to(device)

    for t_idx in [0, 5, 11]:
        mask = t_sample == t_idx
        ref_policy = torch.where(mask[:, None], policy, policy_sample)
        ref_beliefs = torch.where(mask[:, None, None], beliefs, beliefs_sample)
        out_policy = policy_sample.clone().contiguous()
        out_beliefs = beliefs_sample.clone().contiguous()
        t_tensor = torch.tensor(t_idx, device=device, dtype=torch.long)
        fused_preflop_sample_snapshot_rows_multiway_(
            policy.contiguous(),
            out_policy,
            beliefs.contiguous(),
            out_beliefs,
            rows.contiguous(),
            counts.contiguous(),
            t_tensor,
        )
        torch.testing.assert_close(out_policy, ref_policy, rtol=0, atol=0)
        torch.testing.assert_close(out_beliefs, ref_beliefs, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_preflop_public_policy_reach_matches_generic() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import (
        fused_policy_reach_beliefs_depth_preflop_multiway_,
        fused_policy_reach_beliefs_depth_preflop_public_multiway_,
    )

    device = torch.device("cuda")
    torch.manual_seed(79)
    parents, max_children, players, h = 23, 4, 6, 169
    child_count = torch.randint(1, max_children + 1, (parents,), device=device)
    child_offsets = parents + torch.cumsum(child_count, dim=0) - child_count
    children = int(child_count.sum().item())
    total = parents + children

    policy = torch.zeros(total, h, device=device)
    child_policy = torch.rand(children, h, device=device)
    policy[parents:] = child_policy
    reach = torch.rand(total, players, h, device=device)
    reach[:parents] /= reach[:parents].sum(dim=-1, keepdim=True).clamp_min(1e-12)
    beliefs = torch.rand(total, players, h, device=device)
    beliefs[:parents] /= beliefs[:parents].sum(dim=-1, keepdim=True).clamp_min(1e-12)
    allowed = torch.ones(total, h, dtype=torch.bool, device=device)
    allowed_prob = torch.full((total, h), 1.0 / h, device=device)
    root_index = torch.arange(total, device=device).clamp(max=parents - 1)
    for parent in range(parents):
        first = int(child_offsets[parent].item())
        count = int(child_count[parent].item())
        root_index[first : first + count] = parent
    prev_actor = torch.randint(0, players, (total,), device=device)

    policy_generic = policy.clone().contiguous()
    reach_generic = reach.clone().contiguous()
    beliefs_generic = beliefs.clone().contiguous()
    policy_public = policy.clone().contiguous()
    reach_public = reach.clone().contiguous()
    beliefs_public = beliefs.clone().contiguous()

    fused_policy_reach_beliefs_depth_preflop_multiway_(
        policy_generic,
        reach_generic,
        beliefs_generic,
        allowed,
        allowed_prob,
        root_index.contiguous(),
        child_offsets.contiguous(),
        child_count.contiguous(),
        prev_actor.contiguous(),
        parent_base=0,
        max_children=max_children,
    )
    fused_policy_reach_beliefs_depth_preflop_public_multiway_(
        policy_public,
        reach_public,
        beliefs_public,
        root_index.contiguous(),
        child_offsets.contiguous(),
        child_count.contiguous(),
        prev_actor.contiguous(),
        parent_base=0,
        max_children=max_children,
    )
    torch.testing.assert_close(policy_public, policy_generic, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reach_public, reach_generic, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(beliefs_public, beliefs_generic, rtol=1e-5, atol=1e-6)

    leaf_slot = torch.full((total,), -1, dtype=torch.long, device=device)
    leaf_slot[parents::3] = torch.arange(
        0,
        ((total - parents) + 2) // 3,
        device=device,
        dtype=torch.long,
    )[: leaf_slot[parents::3].numel()]
    leaf_rows = int((leaf_slot >= 0).sum().item())
    leaf_generic = torch.empty(leaf_rows, players, h, device=device)
    leaf_public = torch.empty_like(leaf_generic)
    policy_generic = policy.clone().contiguous()
    reach_generic = reach.clone().contiguous()
    beliefs_generic = beliefs.clone().contiguous()
    policy_public = policy.clone().contiguous()
    reach_public = reach.clone().contiguous()
    beliefs_public = beliefs.clone().contiguous()

    fused_policy_reach_beliefs_depth_preflop_multiway_(
        policy_generic,
        reach_generic,
        beliefs_generic,
        allowed,
        allowed_prob,
        root_index.contiguous(),
        child_offsets.contiguous(),
        child_count.contiguous(),
        prev_actor.contiguous(),
        parent_base=0,
        max_children=max_children,
        leaf_slot=leaf_slot.contiguous(),
        leaf_out=leaf_generic,
    )
    fused_policy_reach_beliefs_depth_preflop_public_multiway_(
        policy_public,
        reach_public,
        beliefs_public,
        root_index.contiguous(),
        child_offsets.contiguous(),
        child_count.contiguous(),
        prev_actor.contiguous(),
        parent_base=0,
        max_children=max_children,
        leaf_slot=leaf_slot.contiguous(),
        leaf_out=leaf_public,
    )
    torch.testing.assert_close(policy_public, policy_generic, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reach_public, reach_generic, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(beliefs_public, beliefs_generic, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(leaf_public, leaf_generic, rtol=1e-5, atol=1e-6)


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
    last = torch.randn_like(h)
    old, new = 7.0, 3.0
    ref = ((old + new) * h - old * last) / new
    out = fused_model_values_mix(h.contiguous(), last.contiguous(), old, new)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_sparse_evaluator_matches_baseline_across_iterations() -> None:
    """Run N iterations of each evaluator from identical init; compare state."""
    pytest.importorskip("triton")
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    # Build one baseline evaluator and clone its init state into a fused one.
    ev_base = _build_evaluator()
    ev_fused = _clone_evaluator_state(ev_base)
    # Swap the fused evaluator's class to inherit the same initial tensors.
    ev_fused.__class__ = FusedSparseCFREvaluator
    ev_fused._fused_positive_regrets_buf = None

    # Compare the evaluator math before tiny BF16 model-forward differences at
    # leaf refreshes get amplified by regret matching over later iterations.
    ev_base.set_leaf_values = lambda t: None
    ev_fused.set_leaf_values = lambda t: None
    for t in range(1, 3):
        ev_base.cfr_iteration(t)
        ev_fused.cfr_iteration(t)
    torch.cuda.synchronize()

    for name in [
        "policy_probs",
        "cumulative_regrets",
        "beliefs",
        "self_reach",
        "values_avg",
    ]:
        a = getattr(ev_base, name)
        b = getattr(ev_fused, name)
        # Accumulated rounding over 5 iterations; generous but still tight.
        torch.testing.assert_close(
            a, b, rtol=1e-3, atol=1e-4, msg=f"mismatch on {name}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_sparse_sapdcfr_runs_past_predictive_delay() -> None:
    pytest.importorskip("triton")
    from p2.search.fused_sparse_cfr_evaluator import FusedSparseCFREvaluator

    ev = _build_evaluator()
    ev.__class__ = FusedSparseCFREvaluator
    ev._ensure_fused_attrs()
    ev.cfr_type = CFRType.sapcfr
    ev.sapcfr_alpha = 2.0
    ev.predictive_cfr_delay = 2
    ev._predictive_cfr_dcfr_hybrid = True
    ev._predictive_cfr_enabled = True
    ev.dcfr_delay = 3
    ev.set_leaf_values = lambda t: None

    for t in range(1, 6):
        ev.cfr_iteration(t)
    torch.cuda.synchronize()

    assert ev._last_instantaneous_regrets is not None
    assert torch.isfinite(ev.cumulative_regrets).all()
    assert torch.isfinite(ev.policy_probs).all()
    assert ev._last_instantaneous_regrets.abs().sum().item() > 0
    assert ev._t_scalars.predictive_scale.item() == pytest.approx(1.0 / 3.0)

    bottom = ev.depth_offsets[1]
    top = ev.depth_offsets[-2]
    for parent in range(top):
        count = int(ev.child_count[parent].item())
        if count == 0:
            continue
        first = int(ev.child_offsets[parent].item())
        sibling_sum = ev.policy_probs[first : first + count].sum(dim=0)
        active = sibling_sum > 0
        if active.any():
            torch.testing.assert_close(
                sibling_sum[active],
                torch.ones_like(sibling_sum[active]),
                rtol=1e-4,
                atol=1e-5,
            )
    assert bottom < ev.total_nodes


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

    ev = _build_evaluator()
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
        # capture() executes two real iters (t_warmup, t_capture) that mutate
        # state. We restore pre-state afterward and replay so the captured
        # kernels (which write into the original tensors) run starting from
        # the same pre-state as the baseline cfr_iteration(T) above.
        runner.capture(t_warmup=T - 1, t_capture=T)

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
    ev_base = _build_evaluator()
    ev_graph = _clone_evaluator_state(ev_base)
    ev_base.__class__ = FusedSparseCFREvaluator
    ev_graph.__class__ = FusedSparseCFREvaluator
    ev_base._ensure_fused_attrs()
    ev_graph._ensure_fused_attrs()

    # Prime both past early-t branches so subsequent iters share Python branch.
    for t in range(1, 3):
        ev_base.cfr_iteration(t)
        ev_graph.cfr_iteration(t)
    torch.cuda.synchronize()

    # Stub _record_stats on both so comparison isn't polluted by .item() paths.
    ev_base._record_stats = lambda t, old: None
    ev_graph._record_stats = lambda t, old: None

    # capture()'s warmup at t=3 mutates state; the captured body for t=4
    # doesn't execute until the first replay below.
    runner = GraphedCFRIteration(ev_graph)
    runner.capture(t_warmup=3, t_capture=4)
    # Step baseline forward once to match the warmup mutation.
    ev_base.cfr_iteration(3)
    torch.cuda.synchronize()

    for replay_t in [4, 5, 6, 7]:
        ev_base.cfr_iteration(replay_t)
        runner.replay(t=replay_t)
        torch.cuda.synchronize()
        a = _EvaluatorStateSnapshot.from_evaluator(ev_base)
        b = _EvaluatorStateSnapshot.from_evaluator(ev_graph)
        diffs = a.max_abs_diff(b)
        worst = max(diffs.values())
        assert worst < 1e-2, f"t={replay_t}: replay diverges from baseline: {diffs}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_sparse_evaluate_cfr_captures_pre_and_post_dcfr_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("triton")
    from p2.search.fused_cfr_triton import GraphedCFRIteration
    import p2.search.fused_sparse_cfr_evaluator as fused_sparse_module

    captures: list[tuple[int, int, str | None]] = []

    class CountingGraphedCFRIteration(GraphedCFRIteration):
        def capture(self, t_warmup: int, t_capture: int) -> None:
            captures.append(
                (
                    t_warmup,
                    t_capture,
                    self.evaluator._graph_capture_regime(t_capture),
                )
            )
            super().capture(t_warmup=t_warmup, t_capture=t_capture)

    monkeypatch.setattr(
        fused_sparse_module, "GraphedCFRIteration", CountingGraphedCFRIteration
    )

    ev = _build_evaluator(num_envs=1, depth=2)
    ev.__class__ = fused_sparse_module.FusedSparseCFREvaluator
    ev._ensure_fused_attrs()
    ev.cfr_iterations = 14
    ev.warm_start_iterations = 1
    ev.dcfr_delay = 5

    pbs = ev.evaluate_cfr(training_mode=False)
    torch.cuda.synchronize()
    pbs.env.sanity_check(label="sampled public state")
    assert not bool(pbs.env.has_folded.any().item())

    assert [regime for _, _, regime in captures] == [
        "pre_dcfr_delay",
        "post_dcfr_delay",
    ]
    assert captures[0][0] <= ev.dcfr_delay
    assert captures[0][1] <= ev.dcfr_delay
    assert captures[1][0] > ev.dcfr_delay
    assert captures[1][1] > ev.dcfr_delay
    sampled_nodes = ev._sample_leaf_indices_padded[: ev.root_nodes]
    continue_mask = (
        ev._sample_leaf_ready_padded[: ev.root_nodes]
        & (sampled_nodes >= ev.root_nodes)
        & (~ev.env.done[sampled_nodes])
        & ev.new_street_mask[sampled_nodes]
    )
    root_indices = torch.where(continue_mask)[0]
    torch.testing.assert_close(
        pbs.beliefs,
        ev._sample_leaf_beliefs_padded[root_indices],
    )
    if pbs.env.N > 0:
        next_roots = torch.arange(pbs.env.N, dtype=torch.long, device=ev.device)
        ev.initialize_subgame(pbs.env, next_roots, pbs.beliefs)
        torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# Kernel 12-14 correctness tests: reach weights and deep beliefs.
# ---------------------------------------------------------------------------


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

    allowed_mask = torch.rand(total, h, device=device) > 0.2

    # Reference: for each child c in [4, 10) and player p:
    # reach[c, p, h] = reach[parent, p, h] *
    #     (policy[c] if p==prev_actor[c] else 1) * allowed_mask[c]
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
            ref[c, player] = torch.where(
                allowed_mask[c], ref[c, player], torch.zeros_like(ref[c, player])
            )

    out = reach.clone().contiguous()
    fused_reach_weights_depth_(
        reach=out,
        policy=policy.contiguous(),
        allowed_mask=allowed_mask.contiguous(),
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

    # Caller is responsible for pre-blocking reach (the real reach kernel does
    # this); apply it here so the kernel's no-allowed-mask contract holds.
    reach = reach * allowed_mask[:, None, :].float()
    # Reference: fan_out_deep * reach + normalize (reach is already blocked).
    fanned = root_beliefs[root_index]  # [total, 2, h]
    ref = fanned * reach
    denom = ref.sum(dim=-1, keepdim=True)
    ref = torch.where(denom > 1e-5, ref / denom, allowed_prob[:, None, :])

    out = torch.zeros_like(ref).contiguous()
    # Roots are skipped by the kernel (idempotent under real usage). Pre-fill
    # them so the comparison covers the full tensor.
    out[:n] = ref[:n]
    fused_deep_beliefs_(
        out=out,
        root_beliefs=root_beliefs.contiguous(),
        reach_weights=reach.contiguous(),
        allowed_prob=allowed_prob.contiguous(),
        root_index=root_index.contiguous(),
    )
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
