import torch

from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv


def test_legal_mask_bins_for_subset():
    env = HUNLTensorEnv(
        num_envs=3,
        starting_stack=20000,
        sb=50,
        bb=100,
        device=torch.device("cpu"),
        rng=torch.Generator(device=torch.device("cpu")),
    )
    env.reset()
    mask_full = env.legal_bins_mask()
    subset = torch.tensor([0, 2])
    mask_subset = env.legal_mask_bins_for(subset)
    assert mask_subset.shape[0] == 2
    torch.testing.assert_close(mask_subset[0].float(), mask_full[0].float())
    torch.testing.assert_close(mask_subset[1].float(), mask_full[2].float())


def test_clone_states_delegates_to_copy():
    env = HUNLTensorEnv(
        num_envs=4,
        starting_stack=20000,
        sb=50,
        bb=100,
        device=torch.device("cpu"),
        rng=torch.Generator(device=torch.device("cpu")),
    )
    env.reset()
    parents = torch.tensor([0, 1])
    children = torch.tensor([2, 3])
    # Modify some fields so clone is observable
    env.pot[parents] = torch.tensor([10, 20])
    env.clone_states(children, parents)
    torch.testing.assert_close(env.pot[children].float(), env.pot[parents].float())
