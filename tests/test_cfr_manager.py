import torch
import torch.nn as nn

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.search.cfr_manager import CFRManager, SearchConfig
from alphaholdem.search.dcfr import run_dcfr


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
    roots = mgr.seed_roots(base, torch.tensor([0, 1], device=base.device))
    assert roots.shape[0] == 2
    children = mgr.expand_children(roots, depth=0)
    assert children.shape[0] == 8


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
    roots = mgr.seed_roots(base, src_indices)

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
