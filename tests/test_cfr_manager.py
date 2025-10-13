import torch
import torch.nn as nn

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.search.cfr_manager import CFRManager, SearchConfig
from alphaholdem.search.dcfr import run_dcfr
from alphaholdem.models.transformer.structured_embedding_data import (
    StructuredEmbeddingData,
)
from alphaholdem.models.transformer.tokens import (
    Special,
    get_card_token_id_offset,
)


class DummyModel(nn.Module):
    def __init__(self, num_actions=10):
        super().__init__()
        self.num_actions = num_actions

    def forward(self, embedding_data):
        class Output:
            pass

        out = Output()
        out.policy_logits = torch.randn(
            embedding_data.token_ids.shape[0],
            self.num_actions,
            device=embedding_data.token_ids.device,
        )
        out.value = torch.randn(
            embedding_data.token_ids.shape[0], 1, device=embedding_data.token_ids.device
        )
        return out


def make_env(N: int = 4) -> HUNLTensorEnv:
    device = torch.device("cpu")
    rng = torch.Generator(device=device)
    env = HUNLTensorEnv(
        num_envs=N,
        starting_stack=20000,
        sb=50,
        bb=100,
        device=device,
        rng=rng,
        float_dtype=torch.float32,
        debug_step_table=False,
        flop_showdown=False,
    )
    env.reset()
    return env


def test_copy_and_clone_state_roundtrip():
    src = make_env(4)
    dst = make_env(4)
    idx = torch.tensor([0, 1, 2, 3])
    dst.copy_state_from(src, idx, idx, copy_deck=True)
    # Compare a few key fields
    torch.testing.assert_close(dst.pot[idx].float(), src.pot[idx].float())
    torch.testing.assert_close(dst.stacks[idx].float(), src.stacks[idx].float())
    torch.testing.assert_close(
        dst.board_indices[idx].float(), src.board_indices[idx].float()
    )


def test_manager_seed_and_expand():
    base = make_env(8)
    bet_bins = [1.0]
    mgr = CFRManager(
        batch_size=2,
        env_proto=base,
        bet_bins=bet_bins,
        sequence_length=8,
        device=base.device,
        float_dtype=base.float_dtype,
        cfg=SearchConfig(depth=1, iterations=2, branching=4),
    )
    roots = mgr.seed_roots(
        base,
        torch.tensor([0, 1], device=base.device),
        StructuredEmbeddingData.empty(2, 8, len(bet_bins) + 3, base.device),
    )
    assert roots.shape[0] == 2
    children = mgr.expand_children(roots, depth=0)
    assert children.shape[0] == 8


def test_seed_roots_copies_structured_tokens():
    base = make_env(2)
    bet_bins = base.default_bet_bins
    mgr = CFRManager(
        batch_size=2,
        env_proto=base,
        bet_bins=bet_bins,
        sequence_length=10,
        device=base.device,
        float_dtype=base.float_dtype,
        cfg=SearchConfig(depth=1, iterations=1, branching=4),
    )

    src_indices = torch.tensor([0, 1], device=base.device)
    # Build structured data with custom tokens
    seq_len = mgr.tsb.sequence_length
    num_bins = len(bet_bins) + 3
    data = StructuredEmbeddingData.empty(2, seq_len, num_bins, base.device)

    # Populate minimal sequences (CLS, GAME, HOLE0, HOLE1)
    card_offset = get_card_token_id_offset()
    data.lengths[:] = 4
    data.token_ids[:, 0] = Special.CLS.value
    data.token_ids[:, 1] = Special.GAME.value
    data.token_ids[:, 2] = (card_offset + torch.tensor([3, 17], dtype=torch.int8)).to(
        data.token_ids.dtype
    )
    data.token_ids[:, 3] = (card_offset + torch.tensor([25, 31], dtype=torch.int8)).to(
        data.token_ids.dtype
    )
    data.card_ranks[:, 2] = torch.tensor([3, 4], dtype=torch.uint8)
    data.card_ranks[:, 3] = torch.tensor([12, 5], dtype=torch.uint8)
    data.card_suits[:, 2] = torch.tensor([0, 1], dtype=torch.uint8)
    data.card_suits[:, 3] = torch.tensor([1, 2], dtype=torch.uint8)
    data.action_legal_masks[:, 0, 0] = True

    roots = mgr.seed_roots(base, src_indices, data)

    torch.testing.assert_close(
        mgr.tsb.lengths[roots],
        data.lengths.to(mgr.tsb.lengths.dtype),
    )
    torch.testing.assert_close(
        mgr.tsb.token_ids[roots, :4],
        data.token_ids.to(mgr.tsb.token_ids.dtype)[:, :4],
    )
    torch.testing.assert_close(
        mgr.tsb.card_ranks[roots, :4],
        data.card_ranks.to(mgr.tsb.card_ranks.dtype)[:, :4],
    )
    torch.testing.assert_close(
        mgr.tsb.card_suits[roots, :4],
        data.card_suits.to(mgr.tsb.card_suits.dtype)[:, :4],
    )


def test_cfr_integration_pipeline():
    """Test complete CFR pipeline: CFRManager + DCFR + policy collapse."""
    batch_size = 2
    depth = 1
    iterations = 5

    # Create test environment
    base = make_env(batch_size * 4)  # Extra space for tree
    bet_bins = [1.0, 2.0]

    # Create CFR manager
    mgr = CFRManager(
        batch_size=batch_size,
        env_proto=base,
        bet_bins=bet_bins,
        sequence_length=8,
        device=base.device,
        float_dtype=base.float_dtype,
        cfg=SearchConfig(depth=depth, iterations=iterations, branching=4),
    )

    # Seed roots from test states
    src_indices = torch.tensor([0, 1], device=base.device)
    roots = mgr.seed_roots(
        base,
        src_indices,
        StructuredEmbeddingData.empty(2, 8, len(bet_bins) + 3, base.device),
    )

    # Build tree and run CFR
    dummy_model = DummyModel(num_actions=len(bet_bins) + 3)
    logits_full, legal_full, values_full, to_act = mgr.build_tree_tensors(dummy_model)
    dcfr_res = run_dcfr(
        logits_full=logits_full,
        legal_mask_full=legal_full,
        values=values_full,
        to_act=to_act,
        depth_offsets=mgr.depth_offsets,
        depth=depth,
        iterations=iterations,
    )

    # Verify results
    assert dcfr_res.root_policy_collapsed.shape == (batch_size, 4)
    assert (dcfr_res.root_policy_collapsed >= 0).all()  # Non-negative probabilities
    assert dcfr_res.root_policy_collapsed.sum(dim=-1).allclose(
        torch.ones(batch_size)
    )  # Sum to 1
