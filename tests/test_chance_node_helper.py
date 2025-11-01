"""Test suite for ChanceNodeHelper."""

from __future__ import annotations

from collections import Counter

import pytest
import random
import torch

from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.models.mlp.mlp_features import MLPFeatures
from alphaholdem.models.model_output import ModelOutput
from alphaholdem.search.chance_node_helper import ChanceNodeHelper


class MockModel:
    """Mock model for testing ChanceNodeHelper."""

    def __init__(self, device: torch.device, float_dtype: torch.dtype):
        self.device = device
        self.float_dtype = float_dtype

    def __call__(self, features: MLPFeatures) -> ModelOutput:
        """Return mock hand values."""
        batch_size = features.context.shape[0]
        hand_values = torch.zeros(
            batch_size, 2, NUM_HANDS, device=self.device, dtype=self.float_dtype
        )
        policy_logits = torch.zeros(
            batch_size, 3, device=self.device, dtype=self.float_dtype
        )
        value = torch.zeros(batch_size, device=self.device, dtype=self.float_dtype)
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=hand_values,
        )


class TestChanceNodeHelper:
    """Test suite for ChanceNodeHelper."""

    @pytest.fixture
    def device(self) -> torch.device:
        """Get test device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @pytest.fixture
    def helper(self, device: torch.device) -> ChanceNodeHelper:
        """Create a ChanceNodeHelper instance."""
        model = MockModel(device, torch.float32)
        return ChanceNodeHelper(
            device=device,
            float_dtype=torch.float32,
            num_players=2,
            model=model,
        )

    def test_cache_initialization(self, helper: ChanceNodeHelper):
        """Test that cache tensors are initialized correctly."""
        assert helper.board_to_flop_id is not None
        assert helper.flop_id_to_canonical is not None
        assert helper.flop_id_to_count is not None
        assert helper.total_flop_count > 0

    def test_tensor_shapes(self, helper: ChanceNodeHelper):
        """Test tensor shapes are correct."""
        # board_to_flop_id: [52, 52, 52]
        assert helper.board_to_flop_id.shape == (52, 52, 52)
        assert helper.board_to_flop_id.dtype == torch.long

        # flop_id_to_canonical: [num_flops, 3]
        num_flops = helper.flop_id_to_canonical.shape[0]
        assert helper.flop_id_to_canonical.shape == (num_flops, 3)
        assert helper.flop_id_to_canonical.dtype == torch.long

        # flop_id_to_count: [num_flops]
        assert helper.flop_id_to_count.shape == (num_flops,)
        assert helper.flop_id_to_count.dtype == torch.long

        # Should have 1755 unique canonical flops
        assert num_flops == 1755

    def test_total_flop_count(self, helper: ChanceNodeHelper):
        """Test total flop count is correct."""
        # Total should be sum of all counts
        expected_total = helper.flop_id_to_count.sum().item()
        assert helper.total_flop_count == expected_total
        assert helper.total_flop_count == 22100

    def test_board_to_flop_id_mapping(self, helper: ChanceNodeHelper):
        """Test board_to_flop_id tensor maps correctly."""
        # Test a few known flops
        test_cases = [
            (0, 1, 2),
            (0, 13, 26),
            (12, 25, 38),
            (1, 14, 27),
        ]

        for board in test_cases:
            flop_id = helper.board_to_flop_id[board[0], board[1], board[2]].item()
            assert flop_id >= 0, f"Board {board} should map to a valid flop_id"
            assert flop_id < helper.flop_id_to_canonical.shape[0]

    def test_flop_id_to_canonical_mapping(self, helper: ChanceNodeHelper):
        """Test flop_id maps to correct canonical flop."""
        # Get canonical for a known flop
        board = (0, 1, 2)
        flop_id = helper.board_to_flop_id[board[0], board[1], board[2]].item()
        canonical = helper.flop_id_to_canonical[flop_id]

        # Canonical should have 3 cards, all valid (0-51)
        assert canonical.shape == (3,)
        assert torch.all(canonical >= 0)
        assert torch.all(canonical < 52)

    def test_flop_id_to_count_consistency(self, helper: ChanceNodeHelper):
        """Test that counts are consistent with total."""
        total = helper.flop_id_to_count.sum().item()
        assert total == helper.total_flop_count
        assert total == 22100

        # All counts should be positive
        assert torch.all(helper.flop_id_to_count > 0)

    def test_canonical_flop_uniqueness(self, helper: ChanceNodeHelper):
        """Test that canonical flops are unique."""
        canonical_flops = helper.flop_id_to_canonical
        # Convert to tuples for comparison
        canonical_set = set()
        for i in range(canonical_flops.shape[0]):
            canonical_tuple = tuple(canonical_flops[i].tolist())
            assert (
                canonical_tuple not in canonical_set
            ), f"Duplicate canonical flop {canonical_tuple}"
            canonical_set.add(canonical_tuple)

        assert len(canonical_set) == canonical_flops.shape[0]

    def test_board_to_flop_id_inverse_mapping(self, helper: ChanceNodeHelper):
        """Test that board_to_flop_id and flop_id_to_canonical are consistent."""
        # Sample some flops and verify consistency
        random.seed(42)

        # Test random sample of flops
        num_samples = 100
        all_flops = torch.combinations(torch.arange(52, dtype=torch.long), r=3)
        sample_indices = random.sample(
            range(len(all_flops)), min(num_samples, len(all_flops))
        )

        for idx in sample_indices:
            flop = all_flops[idx]
            flop_id = helper.board_to_flop_id[flop[0], flop[1], flop[2]].item()
            assert flop_id >= 0, f"Flop {flop.tolist()} should map to valid flop_id"

            # The canonical for this flop_id should be a valid canonical representation
            canonical = helper.flop_id_to_canonical[flop_id]
            assert canonical.shape == (3,)
            assert torch.all(canonical >= 0)
            assert torch.all(canonical < 52)

    def test_flop_chance_values_empty_input(self, helper: ChanceNodeHelper):
        """Test flop_chance_values with empty input."""
        device = helper.device
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(0, 1, device=device),
            street=torch.zeros(0, device=device, dtype=torch.long),
            board=torch.zeros(0, 3, device=device, dtype=torch.long),
            beliefs=torch.zeros(0, NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.zeros(0, 1, NUM_HANDS, device=device)
        reach_weights = torch.zeros(0, 2, NUM_HANDS, device=device)

        result = helper.flop_chance_values(
            empty_indices, root_features, pre_chance_beliefs, reach_weights
        )

        assert result.shape == (0, 2, NUM_HANDS)
        assert result.device.type == device.type

    def test_flop_chance_values_basic(self, helper: ChanceNodeHelper):
        """Test flop_chance_values with basic input."""
        device = helper.device
        dtype = helper.float_dtype
        B = 4

        root_indices = torch.arange(B, dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(B, 1, device=device),
            street=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.zeros(B, 3, device=device, dtype=torch.long),
            beliefs=torch.zeros(B, NUM_HANDS, device=device),
        )
        pre_chance_beliefs = (
            torch.ones(B, 1, NUM_HANDS, device=device, dtype=dtype) / NUM_HANDS
        )
        reach_weights = torch.ones(B, 2, NUM_HANDS, device=device, dtype=dtype) / 2

        result = helper.flop_chance_values(
            root_indices, root_features, pre_chance_beliefs, reach_weights
        )

        assert result.shape == (B, 2, NUM_HANDS)
        assert result.device.type == device.type
        assert result.dtype == dtype

    def test_single_card_chance_values_empty_input(self, helper: ChanceNodeHelper):
        """Test single_card_chance_values with empty input."""
        device = helper.device
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(0, 1, device=device),
            street=torch.zeros(0, device=device, dtype=torch.long),
            board=torch.zeros(0, 3, device=device, dtype=torch.long),
            beliefs=torch.zeros(0, NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.zeros(0, 1, NUM_HANDS, device=device)
        reach_weights = torch.zeros(0, 2, NUM_HANDS, device=device)
        board_pre = torch.full((0, 3), -1, dtype=torch.long, device=device)

        result = helper.single_card_chance_values(
            empty_indices, root_features, pre_chance_beliefs, reach_weights, board_pre
        )

        assert result.shape == (0, 2, NUM_HANDS)
        assert result.device.type == device.type

    def test_single_card_chance_values_basic(self, helper: ChanceNodeHelper):
        """Test single_card_chance_values with basic input."""
        device = helper.device
        dtype = helper.float_dtype
        B = 4

        root_indices = torch.arange(B, dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(B, 1, device=device),
            street=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.zeros(B, 3, device=device, dtype=torch.long),
            beliefs=torch.zeros(B, NUM_HANDS, device=device),
        )
        pre_chance_beliefs = (
            torch.ones(B, 1, NUM_HANDS, device=device, dtype=dtype) / NUM_HANDS
        )
        reach_weights = torch.ones(B, 2, NUM_HANDS, device=device, dtype=dtype) / 2
        board_pre = torch.full((B, 3), -1, dtype=torch.long, device=device)

        result = helper.single_card_chance_values(
            root_indices, root_features, pre_chance_beliefs, reach_weights, board_pre
        )

        assert result.shape == (B, 2, NUM_HANDS)
        assert result.device.type == device.type
        assert result.dtype == dtype

    def test_combo_onehot_bool_property(self, helper: ChanceNodeHelper):
        """Test combo_onehot_bool property."""
        combo_onehot = helper.combo_onehot_bool

        assert combo_onehot.shape == (NUM_HANDS, 52)
        assert combo_onehot.dtype == torch.bool
        assert combo_onehot.device.type == helper.device.type

    def test_cache_consistency(self, helper: ChanceNodeHelper):
        """Test that all cache components are consistent."""
        # Verify that all flop IDs map to valid canonical flops
        num_flops = helper.flop_id_to_canonical.shape[0]
        for flop_id in range(num_flops):
            canonical = helper.flop_id_to_canonical[flop_id]
            count = helper.flop_id_to_count[flop_id]

            # Canonical should be valid cards
            assert torch.all(canonical >= 0)
            assert torch.all(canonical < 52)

            # Count should be positive
            assert count > 0

    def test_board_to_flop_id_coverage(self, helper: ChanceNodeHelper):
        """Test that board_to_flop_id covers all valid sorted flops."""
        # All sorted flops (c0 < c1 < c2) should map to valid flop_id
        all_flops = torch.combinations(torch.arange(52, dtype=torch.long), r=3)
        device = helper.device
        all_flops = all_flops.to(device)

        flop_ids = helper.board_to_flop_id[
            all_flops[:, 0], all_flops[:, 1], all_flops[:, 2]
        ]

        # All should be valid (>= 0)
        assert torch.all(flop_ids >= 0), "All sorted flops should map to valid flop_id"

        # All should be within range
        assert torch.all(flop_ids < helper.flop_id_to_canonical.shape[0])

        # Should have exactly 22100 valid entries
        assert (flop_ids >= 0).sum().item() == 22100

    def test_canonical_flop_validity(self, helper: ChanceNodeHelper):
        """Test that all canonical flops are valid."""
        canonical_flops = helper.flop_id_to_canonical

        for i in range(canonical_flops.shape[0]):
            canonical = canonical_flops[i]
            # All cards should be valid
            assert torch.all(canonical >= 0)
            assert torch.all(canonical < 52)

            # Cards should be sorted (c0 < c1 < c2) - actually no, they're canonical, not necessarily sorted
            # But they should be unique
            assert (
                len(torch.unique(canonical)) == 3
            ), "Canonical flop should have 3 unique cards"

    def test_count_sum_matches_total(self, helper: ChanceNodeHelper):
        """Test that sum of counts equals total flop count."""
        count_sum = helper.flop_id_to_count.sum().item()
        assert count_sum == helper.total_flop_count
        assert count_sum == 22100

    def test_update_model(self, helper: ChanceNodeHelper):
        """Test update_model method."""
        old_model = helper.model
        new_model = MockModel(helper.device, helper.float_dtype)

        helper.update_model(new_model)

        assert helper.model is new_model
        assert helper.model is not old_model

    def test_flop_chance_values_device_handling(self, helper: ChanceNodeHelper):
        """Test flop_chance_values handles device correctly."""
        device = helper.device
        B = 2

        root_indices = torch.arange(B, dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(B, 1, device=device),
            street=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.zeros(B, 3, device=device, dtype=torch.long),
            beliefs=torch.zeros(B, NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.ones(B, 1, NUM_HANDS, device=device) / NUM_HANDS
        reach_weights = torch.ones(B, 2, NUM_HANDS, device=device) / 2

        result = helper.flop_chance_values(
            root_indices, root_features, pre_chance_beliefs, reach_weights
        )

        assert result.device.type == device.type

    def test_single_card_chance_values_device_handling(self, helper: ChanceNodeHelper):
        """Test single_card_chance_values handles device correctly."""
        device = helper.device
        B = 2

        root_indices = torch.arange(B, dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(B, 1, device=device),
            street=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.zeros(B, 3, device=device, dtype=torch.long),
            beliefs=torch.zeros(B, NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.ones(B, 1, NUM_HANDS, device=device) / NUM_HANDS
        reach_weights = torch.ones(B, 2, NUM_HANDS, device=device) / 2
        board_pre = torch.full((B, 3), -1, dtype=torch.long, device=device)

        result = helper.single_card_chance_values(
            root_indices, root_features, pre_chance_beliefs, reach_weights, board_pre
        )

        assert result.device.type == device.type

    def test_canonical_flop_basic_properties(self, helper: ChanceNodeHelper):
        """Test that canonical flops have basic expected properties."""
        # Test a few flops to ensure canonicalization is reasonable
        test_flops = [
            (0, 1, 2),
            (0, 13, 26),
            (12, 25, 38),
            (1, 14, 27),
        ]

        for flop in test_flops:
            # Get flop_id from tensor
            flop_id = helper.board_to_flop_id[flop[0], flop[1], flop[2]].item()
            canonical_from_tensor = helper.flop_id_to_canonical[flop_id]

            # Canonical should have 3 valid cards
            assert canonical_from_tensor.shape == (3,)
            assert torch.all(canonical_from_tensor >= 0)
            assert torch.all(canonical_from_tensor < 52)

            # All cards should be unique
            assert len(torch.unique(canonical_from_tensor)) == 3

    def test_flop_id_counts_are_correct(self, helper: ChanceNodeHelper):
        """Test that flop_id counts accurately reflect number of flops per canonical."""
        # Verify that each canonical flop's count matches the number of flops mapping to it
        all_flops = torch.combinations(torch.arange(52, dtype=torch.long), r=3)
        device = helper.device
        all_flops = all_flops.to(device)

        flop_ids = helper.board_to_flop_id[
            all_flops[:, 0], all_flops[:, 1], all_flops[:, 2]
        ]

        # Count how many flops map to each canonical flop_id
        flop_id_counts = Counter(flop_ids.cpu().tolist())

        # Verify counts match
        for flop_id, expected_count in flop_id_counts.items():
            actual_count = helper.flop_id_to_count[flop_id].item()
            assert actual_count == expected_count, (
                f"Count mismatch for flop_id {flop_id}: "
                f"expected={expected_count}, actual={actual_count}"
            )
