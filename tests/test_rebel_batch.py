import torch

from p2.env.card_utils import (
    NUM_HANDS,
    combo_suit_permutation_inverse_tensor,
    suit_permutations_tensor,
)
from p2.models.mlp.mlp_features import MLPFeatures
from p2.rl.rebel_batch import RebelBatch


def test_with_permuted_targets_reorders_policy_and_value():
    device = torch.device("cpu")
    batch_size = 2
    num_actions = 3
    num_players = 2

    features = MLPFeatures(
        context=torch.zeros(batch_size, 4, device=device),
        street=torch.zeros(batch_size, dtype=torch.long, device=device),
        to_act=torch.zeros(batch_size, dtype=torch.long, device=device),
        board=torch.zeros(batch_size, 5, dtype=torch.long, device=device),
        beliefs=torch.zeros(batch_size, 2 * NUM_HANDS, device=device),
    )

    value_targets = torch.arange(
        batch_size * num_players * NUM_HANDS, device=device, dtype=torch.float32
    ).reshape(batch_size, num_players, NUM_HANDS)
    policy_targets = torch.arange(
        batch_size * NUM_HANDS * num_actions, device=device, dtype=torch.float32
    ).reshape(batch_size, NUM_HANDS, num_actions)
    legal_masks = torch.ones(batch_size, num_actions, dtype=torch.bool, device=device)

    batch = RebelBatch(
        features=features,
        legal_masks=legal_masks,
        policy_targets=policy_targets,
        value_targets=value_targets,
    )

    suit_permutation_idxs = torch.tensor([0, 1], device=device)
    suit_permutations = suit_permutations_tensor(device=device)[suit_permutation_idxs]
    permuted_batch, returned_idxs = batch.with_permuted_targets(
        suit_permutations=suit_permutations, num_players=num_players
    )

    torch.testing.assert_close(returned_idxs, suit_permutation_idxs)

    combo_permutations_inverse = combo_suit_permutation_inverse_tensor(device=device)[
        suit_permutation_idxs
    ]

    expected_value_targets = torch.gather(
        value_targets,
        2,
        combo_permutations_inverse[:, None, :].expand(-1, num_players, -1),
    )
    expected_policy_targets = torch.gather(
        policy_targets,
        1,
        combo_permutations_inverse[:, :, None].expand(-1, -1, num_actions),
    )

    torch.testing.assert_close(permuted_batch.value_targets, expected_value_targets)
    torch.testing.assert_close(permuted_batch.policy_targets, expected_policy_targets)
