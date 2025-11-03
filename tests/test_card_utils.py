from __future__ import annotations

import pytest
import torch

from alphaholdem.env.card_utils import (
    combo_blocking_tensor,
    combo_index,
    combo_lookup_tensor,
    combo_to_onehot_tensor,
    combo_to_range_grid,
    combo_suit_permutation_tensor,
    combo_suit_permutation_inverse_tensor,
    hand_combos_tensor,
    mask_conflicting_combos,
    suit_permutations_tensor,
)


class TestHandCombosTensor:
    """Test hand_combos_tensor function."""

    def test_basic_properties(self):
        """Test basic properties of hand combinations tensor."""
        combos = hand_combos_tensor()

        # Should have 1326 combinations (52 choose 2)
        assert combos.shape == (1326, 2)
        assert combos.dtype == torch.long

        # All combinations should be sorted (first card < second card)
        assert torch.all(combos[:, 0] < combos[:, 1])

        # Should contain all possible pairs
        unique_pairs = set()
        for i in range(combos.shape[0]):
            pair = tuple(combos[i].tolist())
            unique_pairs.add(pair)

        assert len(unique_pairs) == 1326

        # Verify first few combinations
        expected_first = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        for i, expected in enumerate(expected_first):
            assert combos[i].tolist() == list(expected)

    def test_device_handling(self):
        """Test device parameter handling."""
        cpu_combos = hand_combos_tensor(device=torch.device("cpu"))
        assert cpu_combos.device.type == "cpu"

        # Test with None device (should default to CPU)
        default_combos = hand_combos_tensor(device=None)
        assert default_combos.device.type == "cpu"

    def test_caching(self):
        """Test that function is properly cached."""
        combos1 = hand_combos_tensor()
        combos2 = hand_combos_tensor()

        # Should be the same tensor object due to caching
        assert combos1 is combos2


class TestComboLookupTensor:
    """Test combo_lookup_tensor function."""

    def test_lookup_structure(self):
        """Test basic structure of lookup tensor."""
        lookup = combo_lookup_tensor()

        assert lookup.shape == (52, 52)
        assert lookup.dtype == torch.long

        # Diagonal should be -1 (no self-pairs)
        assert torch.all(torch.diag(lookup) == -1)

    def test_lookup_symmetry(self):
        """Test that lookup is symmetric."""
        lookup = combo_lookup_tensor()

        # Should be symmetric: lookup[i,j] == lookup[j,i]
        for i in range(52):
            for j in range(52):
                if i != j:
                    assert lookup[i, j] == lookup[j, i]

    def test_lookup_correctness(self):
        """Test that lookup correctly maps card pairs to combo indices."""
        lookup = combo_lookup_tensor()
        combos = hand_combos_tensor()

        # Test a few known combinations
        test_cases = [
            (0, 1, 0),  # First combo should be (0,1) -> index 0
            (0, 2, 1),  # Second combo should be (0,2) -> index 1
            (1, 2, 51),  # (1,2) should be at index 51 (not 13)
        ]

        for card1, card2, expected_idx in test_cases:
            assert lookup[card1, card2] == expected_idx
            assert lookup[card2, card1] == expected_idx

    def test_device_handling(self):
        """Test device parameter handling."""
        cpu_lookup = combo_lookup_tensor(device=torch.device("cpu"))
        assert cpu_lookup.device.type == "cpu"


class TestComboIndex:
    """Test combo_index function."""

    def test_basic_functionality(self):
        """Test basic combo index calculation."""
        # Test known combinations
        assert combo_index(0, 1) == 0
        assert combo_index(1, 0) == 0  # Should be symmetric
        assert combo_index(0, 2) == 1
        assert combo_index(2, 0) == 1  # Should be symmetric
        assert combo_index(1, 2) == 51  # Corrected expected value

    def test_edge_cases(self):
        """Test edge cases."""
        # Test highest cards
        assert combo_index(50, 51) == 1325  # Last combination
        assert combo_index(51, 50) == 1325  # Symmetric

        # Test middle cards
        assert combo_index(25, 26) > 0
        assert combo_index(26, 25) == combo_index(25, 26)

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Test same card (should be invalid)
        with pytest.raises(ValueError):
            combo_index(0, 0)

        # Note: The function doesn't validate card ranges, so these don't raise errors
        # but they return unexpected results. This is a limitation of the current implementation.
        # combo_index(-1, 0) returns 50 (unexpected but not an error)
        # combo_index(52, 0) would also not raise an error

    def test_type_conversion(self):
        """Test that function handles different input types."""
        # Should work with different integer types
        assert combo_index(0, 1) == combo_index(torch.tensor(0), torch.tensor(1))


class TestMaskConflictingCombos:
    """Test mask_conflicting_combos function."""

    def test_empty_occupied_cards(self):
        """Test with no occupied cards."""
        occupied = torch.tensor([])
        mask = mask_conflicting_combos(occupied)

        # All combinations should be available
        assert mask.shape == (1326,)
        assert torch.all(mask)

    def test_single_occupied_card(self):
        """Test with single occupied card."""
        occupied = torch.tensor([0])  # Card 0 is occupied
        mask = mask_conflicting_combos(occupied)

        # Combinations containing card 0 should be masked out
        combos = hand_combos_tensor()
        conflicts_with_0 = torch.isin(combos, occupied).any(dim=1)
        expected_mask = ~conflicts_with_0

        torch.testing.assert_close(mask, expected_mask)

    def test_multiple_occupied_cards(self):
        """Test with multiple occupied cards."""
        occupied = torch.tensor([0, 1, 2])
        mask = mask_conflicting_combos(occupied)

        # Combinations containing any of these cards should be masked out
        combos = hand_combos_tensor()
        conflicts = torch.isin(combos, occupied).any(dim=1)
        expected_mask = ~conflicts

        torch.testing.assert_close(mask, expected_mask)

    def test_negative_cards_ignored(self):
        """Test that negative card indices are ignored."""
        occupied = torch.tensor([0, -1, 1, -2])
        mask = mask_conflicting_combos(occupied)

        # Should only consider cards 0 and 1
        occupied_clean = torch.tensor([0, 1])
        mask_clean = mask_conflicting_combos(occupied_clean)

        torch.testing.assert_close(mask, mask_clean)

    def test_device_handling(self):
        """Test device parameter handling."""
        occupied = torch.tensor([0, 1])
        cpu_mask = mask_conflicting_combos(occupied, device=torch.device("cpu"))
        assert cpu_mask.device.type == "cpu"

    def test_all_cards_occupied(self):
        """Test edge case where all cards are occupied."""
        occupied = torch.arange(52)
        mask = mask_conflicting_combos(occupied)

        # No combinations should be available
        assert torch.all(~mask)


class TestSuitPermutationHelpers:
    """Test suit permutation helper tensors."""

    def test_suit_permutations_tensor_basic(self):
        perms = suit_permutations_tensor()
        assert perms.shape == (24, 4)
        assert perms.dtype == torch.long

        sorted_rows, _ = torch.sort(perms, dim=1)
        expected = torch.arange(4, dtype=torch.long).expand_as(sorted_rows)
        torch.testing.assert_close(sorted_rows, expected)

        assert len({tuple(row.tolist()) for row in perms}) == 24

    def test_combo_suit_permutation_tensor_properties(self):
        combo_perms = combo_suit_permutation_tensor()
        assert combo_perms.shape == (24, 1326)
        assert combo_perms.dtype == torch.long

        sorted_rows, _ = torch.sort(combo_perms, dim=1)
        expected = torch.arange(1326, dtype=torch.long).expand_as(sorted_rows)
        torch.testing.assert_close(sorted_rows, expected)

    def test_combo_permutation_matches_expected_mapping(self):
        perms = suit_permutations_tensor()
        combo_perms = combo_suit_permutation_tensor()
        combos = hand_combos_tensor()

        target_perm = torch.tensor([1, 2, 3, 0], dtype=torch.long)
        perm_index = torch.nonzero(
            (perms == target_perm).all(dim=1), as_tuple=False
        ).squeeze()
        assert perm_index.numel() == 1
        perm_index = int(perm_index.item())

        combo_idx = combo_index(0, 5)
        mapped_idx = int(combo_perms[perm_index, combo_idx].item())

        original_cards = combos[combo_idx]
        expected_suits = target_perm[original_cards // 13]
        expected_cards = (expected_suits * 13) + (original_cards % 13)
        expected_cards, _ = torch.sort(expected_cards)

        actual_cards = combos[mapped_idx]
        torch.testing.assert_close(actual_cards, expected_cards)

    def test_combo_permutation_inverse_roundtrip(self):
        combo_perms = combo_suit_permutation_tensor()
        combo_perms_inv = combo_suit_permutation_inverse_tensor()
        identity = torch.arange(1326, dtype=torch.long)

        for idx in range(combo_perms.shape[0]):
            torch.testing.assert_close(
                combo_perms[idx][combo_perms_inv[idx]],
                identity,
            )
            torch.testing.assert_close(
                combo_perms_inv[idx][combo_perms[idx]],
                identity,
            )


class TestComboToOnehotTensor:
    """Test combo_to_onehot_tensor function."""

    def test_basic_structure(self):
        """Test basic structure of one-hot tensor."""
        onehot = combo_to_onehot_tensor()

        assert onehot.shape == (1326, 52)
        assert onehot.dtype == torch.bool

    def test_onehot_correctness(self):
        """Test that one-hot encoding is correct."""
        onehot = combo_to_onehot_tensor()
        combos = hand_combos_tensor()

        # Each row should have exactly 2 True values
        assert torch.all(onehot.sum(dim=1) == 2)

        # Check specific combinations
        for i in range(min(10, combos.shape[0])):  # Check first 10
            combo = combos[i]
            row = onehot[i]

            # Exactly the two cards in this combo should be True
            assert row[combo[0]] == True
            assert row[combo[1]] == True

            # All other cards should be False
            other_cards = torch.cat(
                [
                    torch.arange(combo[0]),
                    torch.arange(combo[0] + 1, combo[1]),
                    torch.arange(combo[1] + 1, 52),
                ]
            )
            assert torch.all(~row[other_cards])

    def test_device_handling(self):
        """Test device parameter handling."""
        cpu_onehot = combo_to_onehot_tensor(device=torch.device("cpu"))
        assert cpu_onehot.device.type == "cpu"


class TestComboToRangeGrid:
    """Test combo_to_range_grid function."""

    def test_basic_structure(self):
        """Test basic structure of range grid tensor."""
        grid = combo_to_range_grid()

        assert grid.shape == (1326, 2)
        assert grid.dtype == torch.long

    def test_suited_hands(self):
        """Test range grid for suited hands."""
        grid = combo_to_range_grid()
        combos = hand_combos_tensor()

        # Find suited hands (same suit)
        suited_mask = combos[:, 0] // 13 == combos[:, 1] // 13
        suited_combos = combos[suited_mask]
        suited_grid = grid[suited_mask]

        # For suited hands, ranks should be sorted ascending, then 12 - ranks
        suited_ranks = suited_combos % 13
        sorted_ranks = torch.sort(suited_ranks, dim=1, descending=True).values
        expected_grid = 12 - sorted_ranks

        torch.testing.assert_close(suited_grid, expected_grid)

    def test_offsuit_hands(self):
        """Test range grid for offsuit hands."""
        grid = combo_to_range_grid()
        combos = hand_combos_tensor()

        # Find offsuit hands (different suits)
        offsuit_mask = combos[:, 0] // 13 != combos[:, 1] // 13
        offsuit_combos = combos[offsuit_mask]
        offsuit_grid = grid[offsuit_mask]

        # For offsuit hands, ranks should be sorted ascending, then 12 - ranks
        offsuit_ranks = offsuit_combos % 13
        sorted_ranks = torch.sort(offsuit_ranks, dim=1).values
        expected_grid = 12 - sorted_ranks

        torch.testing.assert_close(offsuit_grid, expected_grid)

    def test_device_handling(self):
        """Test device parameter handling."""
        cpu_grid = combo_to_range_grid(device=torch.device("cpu"))
        assert cpu_grid.device.type == "cpu"


class TestComboBlockingTensor:
    """Test combo_blocking_tensor function."""

    def test_basic_structure(self):
        """Test basic structure of blocking tensor."""
        blocking = combo_blocking_tensor()

        assert blocking.shape == (1326, 1326)
        assert blocking.dtype == torch.bool

    def test_self_blocking(self):
        """Test that each combo blocks itself."""
        blocking = combo_blocking_tensor()

        # Diagonal should be True (each combo blocks itself)
        assert torch.all(torch.diag(blocking))

    def test_symmetry(self):
        """Test that blocking relationship is symmetric."""
        blocking = combo_blocking_tensor()

        # Should be symmetric: if combo A blocks combo B, then B blocks A
        assert torch.all(blocking == blocking.T)

    def test_blocking_logic(self):
        """Test blocking logic with known examples."""
        blocking = combo_blocking_tensor()
        combos = hand_combos_tensor()

        # Find combos that share cards
        for i in range(min(10, combos.shape[0])):  # Check first 10
            combo_i = combos[i]
            for j in range(combos.shape[0]):
                combo_j = combos[j]

                # Check if they share any cards
                shares_cards = torch.any(torch.isin(combo_i, combo_j))
                expected_blocking = shares_cards
                actual_blocking = blocking[i, j]

                assert actual_blocking == expected_blocking

    def test_device_handling(self):
        """Test device parameter handling."""
        cpu_blocking = combo_blocking_tensor(device=torch.device("cpu"))
        assert cpu_blocking.device.type == "cpu"


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_combo_index_and_lookup_consistency(self):
        """Test that combo_index and lookup tensor are consistent."""
        lookup = combo_lookup_tensor()

        # Test random combinations
        import random

        random.seed(42)

        for _ in range(100):
            card1 = random.randint(0, 51)
            card2 = random.randint(0, 51)
            if card1 != card2:
                idx_from_function = combo_index(card1, card2)
                idx_from_lookup = lookup[card1, card2].item()
                assert idx_from_function == idx_from_lookup

    def test_mask_and_onehot_consistency(self):
        """Test consistency between mask and one-hot functions."""
        occupied = torch.tensor([0, 1, 2])
        mask = mask_conflicting_combos(occupied)
        onehot = combo_to_onehot_tensor()

        # Check that masked combinations don't intersect with occupied cards
        for i in range(1326):
            if mask[i]:  # If combo is not masked
                combo_cards = onehot[i]
                # Should not intersect with occupied cards
                assert not torch.any(combo_cards[occupied])

    def test_range_grid_and_combos_consistency(self):
        """Test consistency between range grid and combo tensor."""
        grid = combo_to_range_grid()
        combos = hand_combos_tensor()

        # Check that grid indices correspond to actual card ranks
        for i in range(min(50, combos.shape[0])):  # Check first 50
            combo = combos[i]

            # Determine if suited or offsuit
            suited = combo[0] // 13 == combo[1] // 13

            if suited:
                assert grid[i][0] < grid[i][1]
            elif combo[0] % 13 == combo[1] % 13:
                assert grid[i][0] == grid[i][1]
            else:
                assert grid[i][0] > grid[i][1]


if __name__ == "__main__":
    pytest.main([__file__])
