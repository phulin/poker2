import torch

from p2.env.card_utils import (
    HAND_EQUITY_ORDERING,
    IDX_TO_RANK,
    NUM_HANDS,
    combo_lookup_tensor,
    hand_combos_tensor,
)
from p2.env.hunl_tensor_env import HUNLTensorEnv
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.mlp.rebel_feature_encoder import RebelFeatureEncoder


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
    features.permute_suits(generator=generator)

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


def _expected_remap(suit_permutation: torch.Tensor) -> torch.Tensor:
    device = suit_permutation.device
    hand_combos = hand_combos_tensor(device=device)
    combo_lookup = combo_lookup_tensor(device=device)

    all_ranks = (hand_combos % 13).unsqueeze(0)
    all_suits = (hand_combos // 13).unsqueeze(0)

    suit_perm_expanded = suit_permutation[:, None, :].expand(-1, NUM_HANDS, -1)
    new_suits = torch.gather(suit_perm_expanded, 2, all_suits)

    new_cards = all_ranks + new_suits * 13
    min_cards = torch.min(new_cards, dim=2).values
    max_cards = torch.max(new_cards, dim=2).values

    remap = combo_lookup[min_cards, max_cards].to(torch.long)
    return torch.argsort(remap, dim=1)


def test_permute_suits_permuted_board_and_beliefs():
    device = torch.device("cpu")
    batch_size = 1

    context = torch.zeros((batch_size, 1), device=device)
    street = torch.zeros(batch_size, device=device, dtype=torch.long)
    board = torch.tensor([[0, 13, 26, 39, -1]], device=device)
    board_clone = board.clone()

    # Beliefs must be (batch_size, 2 * NUM_HANDS)
    beliefs_raw = torch.rand(1, 2 * NUM_HANDS, dtype=torch.float32, device=device)
    beliefs = beliefs_raw / beliefs_raw.sum(dim=1, keepdim=True)  # Normalize
    beliefs_clone = beliefs.clone()

    generator = torch.Generator(device=device)
    generator.manual_seed(123)
    state = generator.get_state()
    suit_permutation = torch.argsort(
        torch.rand((batch_size, 4), generator=generator), dim=-1
    )
    generator.set_state(state)

    features = MLPFeatures(
        context=context.clone(),
        street=street.clone(),
        to_act=torch.zeros(1, device=device, dtype=torch.long),
        board=board.clone(),
        beliefs=beliefs.clone(),
    )
    features.permute_suits(generator=generator)

    board_valid = board_clone >= 0
    ranks = (board_clone % 13).clamp(0, 12)
    suits = (board_clone // 13).clamp(0, 3)
    expected_suits = torch.gather(
        suit_permutation.unsqueeze(1).expand(-1, 5, -1),
        dim=2,
        index=suits.unsqueeze(2).to(torch.long),
    ).squeeze(2)
    expected_board = torch.where(board_valid, expected_suits * 13 + ranks, board_clone)
    assert torch.equal(features.board, expected_board)

    inverse_remap = _expected_remap(suit_permutation)
    # Split beliefs into two players, permute each, then concatenate
    p0_beliefs = beliefs_clone[:, :NUM_HANDS]
    p1_beliefs = beliefs_clone[:, NUM_HANDS:]
    expected_p0 = torch.gather(p0_beliefs, 1, inverse_remap)
    expected_p1 = torch.gather(p1_beliefs, 1, inverse_remap)
    expected_beliefs = torch.cat([expected_p0, expected_p1], dim=1)
    torch.testing.assert_close(features.beliefs, expected_beliefs)

    # Two-player belief tensor path.
    beliefs_two = torch.arange(
        2 * NUM_HANDS, dtype=torch.float32, device=device
    ).unsqueeze(0)
    generator_two = torch.Generator(device=device)
    generator_two.set_state(state)
    features_two = MLPFeatures(
        context=context.clone(),
        street=street.clone(),
        to_act=torch.zeros(1, device=device, dtype=torch.long),
        board=board.clone(),
        beliefs=beliefs_two.clone(),
    )
    features_two.permute_suits(generator=generator_two)
    assert torch.equal(features_two.board, expected_board)

    expected_two = torch.cat(
        [
            torch.gather(beliefs_two[:, :NUM_HANDS], 1, inverse_remap),
            torch.gather(beliefs_two[:, NUM_HANDS:], 1, inverse_remap),
        ],
        dim=1,
    )
    torch.testing.assert_close(features_two.beliefs, expected_two)
