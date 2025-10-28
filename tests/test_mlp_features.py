import torch

from alphaholdem.env.card_utils import (
    HAND_EQUITY_ORDERING,
    IDX_TO_RANK,
    NUM_HANDS,
    hand_combos_tensor,
)
from alphaholdem.env.hunl_tensor_env import HUNLTensorEnv
from alphaholdem.models.mlp.rebel_feature_encoder import RebelFeatureEncoder


def make_env(num_envs: int = 4) -> HUNLTensorEnv:
    env = HUNLTensorEnv(
        num_envs=num_envs,
        starting_stack=1000,
        sb=5,
        bb=10,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
        flop_showdown=False,
    )
    env.reset()
    return env


def build_hand_name_to_combos(device: torch.device) -> dict[str, list[int]]:
    """Build a mapping from hand name to list of combo indices."""
    combos = hand_combos_tensor(device=device)
    hand_name_to_combos: dict[str, list[int]] = {}

    for idx in range(NUM_HANDS):
        c1, c2 = combos[idx]
        r1, r2 = c1 % 13, c2 % 13
        is_suited = c1 // 13 == c2 // 13

        # Create a canonical hand name from this combo
        if r1 == r2:
            hand_name = (
                f"{IDX_TO_RANK[int(r1)]}{IDX_TO_RANK[int(r1)]}"  # Pair like "AA"
            )
        else:
            rank_char_1 = IDX_TO_RANK[int(max(r1, r2))]
            rank_char_2 = IDX_TO_RANK[int(min(r1, r2))]
            suffix = "s" if is_suited else "o"
            hand_name = f"{rank_char_1}{rank_char_2}{suffix}"  # Like "AKs" or "AKo"

        if hand_name not in hand_name_to_combos:
            hand_name_to_combos[hand_name] = []
        hand_name_to_combos[hand_name].append(idx)

    return hand_name_to_combos


def test_permute_suits():
    """Test that permute_suits correctly permutes board cards and beliefs."""
    env = make_env(4)
    encoder = RebelFeatureEncoder(env, device=env.device, dtype=torch.float32)

    # Create random beliefs for each player
    generator = torch.Generator(device=env.device)
    generator.manual_seed(123)
    beliefs = torch.rand((4, 2, NUM_HANDS), generator=generator, device=env.device)
    # Normalize to sum to 1.0
    beliefs = beliefs / beliefs.sum(dim=2, keepdim=True)

    # Create features
    features = encoder.encode(beliefs)

    # Store original values
    original_board = features.board.clone()
    original_beliefs = features.beliefs.clone()

    # Verify beliefs sum to 1.0
    torch.testing.assert_close(
        original_beliefs[:, :NUM_HANDS].sum(dim=1),
        torch.ones(4, device=env.device),
    )

    # Apply suit permutation
    features.permute_suits(generator)

    # Verify beliefs still sum to 1.0 (values are just reordered)
    torch.testing.assert_close(
        features.beliefs[:, :NUM_HANDS].sum(dim=1),
        torch.ones(4, device=env.device),
    )

    # Verify board cards were permuted
    # Board should be different unless permutation happened to be identity
    # Check that ranks are preserved but suits changed
    original_ranks = original_board % 13
    new_ranks = features.board % 13
    # Ranks should be the same (permutation only affects suits)
    torch.testing.assert_close(original_ranks, new_ranks)

    # Verify beliefs total sum is unchanged
    torch.testing.assert_close(original_beliefs.sum(), features.beliefs.sum())

    # For each hand class in HAND_EQUITY_ORDERING, verify suit permutation is correct
    hand_name_to_combos = build_hand_name_to_combos(env.device)

    # Check each player's beliefs
    for player_idx in range(2):
        player_beliefs_start = player_idx * NUM_HANDS
        player_beliefs_end = (player_idx + 1) * NUM_HANDS

        # Get original and permuted beliefs for this player
        orig_player_beliefs = original_beliefs[
            :, player_beliefs_start:player_beliefs_end
        ]
        permuted_player_beliefs = features.beliefs[
            :, player_beliefs_start:player_beliefs_end
        ]

        # For each hand class, check that the sum of beliefs across all suit combinations is preserved
        for hand_name in HAND_EQUITY_ORDERING:
            combo_indices = hand_name_to_combos[hand_name]
            orig_player_beliefs_combo = (
                orig_player_beliefs[:, combo_indices].sort(dim=1).values
            )
            permuted_player_beliefs_combo = (
                permuted_player_beliefs[:, combo_indices].sort(dim=1).values
            )

            # The sum should be the same (suit permutation just reorders within the hand class)
            torch.testing.assert_close(
                orig_player_beliefs_combo, permuted_player_beliefs_combo
            )
