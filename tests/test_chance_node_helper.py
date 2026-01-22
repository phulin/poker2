"""Test suite for ChanceNodeHelper."""

from __future__ import annotations

import random
from collections import Counter

import pytest
import torch

from p2.env.card_utils import NUM_HANDS
from p2.models.mlp.mlp_features import MLPFeatures
from p2.models.model_output import ModelOutput
from p2.search.chance_node_helper import ChanceNodeHelper


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

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


class BeliefEchoModel:
    """Model that echoes normalized beliefs for testing permutation logic."""

    def __init__(self, device: torch.device, float_dtype: torch.dtype):
        self.device = device
        self.float_dtype = float_dtype

    def __call__(self, features: MLPFeatures) -> ModelOutput:
        belief_tensor = features.beliefs.view(-1, 2, NUM_HANDS).to(
            device=self.device, dtype=self.float_dtype
        )
        batch_size = belief_tensor.shape[0]
        policy_logits = torch.zeros(
            batch_size, 3, device=self.device, dtype=self.float_dtype
        )
        value = torch.zeros(batch_size, device=self.device, dtype=self.float_dtype)
        return ModelOutput(
            policy_logits=policy_logits,
            value=value,
            hand_values=belief_tensor,
        )

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


class BoardBasedModel:
    """Model that returns hand values based on board card ranks only (commutes with permutations)."""

    def __init__(self, device: torch.device, float_dtype: torch.dtype):
        self.device = device
        self.float_dtype = float_dtype

    def __call__(self, features: MLPFeatures) -> ModelOutput:
        batch_size = features.context.shape[0]
        # Return hand values based on sum of board card ranks (not suits)
        # This commutes with suit permutations because it only depends on ranks
        board = features.board  # [B, 5]
        valid_mask = board >= 0
        ranks = (board % 13).float()  # Extract ranks (0-12)
        ranks = torch.where(valid_mask, ranks, torch.zeros_like(ranks))
        board_sum = ranks.sum(dim=1, keepdim=True)  # [B, 1]
        hand_values = (
            (board_sum % NUM_HANDS)
            .unsqueeze(1)
            .expand(-1, 2, NUM_HANDS)
            .to(device=self.device, dtype=self.float_dtype)
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

    def eval(self) -> None:
        pass

    def train(self) -> None:
        pass


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

    def test_flop_perm_counts_consistency(self, helper: ChanceNodeHelper):
        """Ensure per-permutation counts sum to total canonical counts."""
        perm_counts = helper.flop_id_perm_counts
        summed = perm_counts.sum(dim=1)
        assert torch.equal(summed, helper.flop_id_to_count)

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
            to_act=torch.zeros(0, device=device, dtype=torch.long),
            board=torch.zeros(0, 5, device=device, dtype=torch.long),
            beliefs=torch.zeros(0, 2 * NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.zeros(0, 2, NUM_HANDS, device=device)

        result = helper.flop_chance_values(
            empty_indices, root_features, pre_chance_beliefs
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
            to_act=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.cat(
                [
                    torch.zeros(B, 3, device=device, dtype=torch.long),
                    torch.full((B, 2), -1, device=device, dtype=torch.long),
                ],
                dim=1,
            ),
            beliefs=torch.zeros(B, 2 * NUM_HANDS, device=device),
        )
        pre_chance_beliefs = (
            torch.ones(B, 2, NUM_HANDS, device=device, dtype=dtype) / NUM_HANDS
        )

        result = helper.flop_chance_values(
            root_indices, root_features, pre_chance_beliefs
        )

        assert result.shape == (B, 2, NUM_HANDS)
        assert result.device.type == device.type
        assert result.dtype == dtype

    def test_flop_chance_values_matches_bruteforce(self, device: torch.device):
        """Evaluate model on all 22100 flops and check it matches flop_chance_values."""
        model = BoardBasedModel(device, torch.float32)
        helper = ChanceNodeHelper(
            device=device,
            float_dtype=torch.float32,
            num_players=2,
            model=model,
        )

        B = 1
        root_indices = torch.arange(B, device=device, dtype=torch.long)
        root_features = MLPFeatures(
            context=torch.zeros(B, 1, device=device),
            street=torch.zeros(B, device=device, dtype=torch.long),
            to_act=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.cat(
                [
                    torch.zeros(B, 3, device=device, dtype=torch.long),
                    torch.full((B, 2), -1, device=device, dtype=torch.long),
                ],
                dim=1,
            ),
            beliefs=torch.zeros(B, 2 * NUM_HANDS, device=device),
        )

        base0 = torch.linspace(1.0, 2.0, NUM_HANDS, device=device)
        base1 = torch.linspace(2.0, 3.0, NUM_HANDS, device=device)
        pre_chance_beliefs = torch.stack([base0, base1], dim=0).unsqueeze(0)

        result = helper.flop_chance_values(
            root_indices, root_features, pre_chance_beliefs
        )

        # Evaluate model on all 22100 flops
        all_flops = helper.all_flops  # (22100, 3)
        num_flops = all_flops.shape[0]
        assert num_flops == 22100

        pre_beliefs = pre_chance_beliefs[root_indices].to(dtype=helper.float_dtype)
        context_root = root_features.context[root_indices]
        street_root = root_features.street[root_indices]
        to_act_root = root_features.to_act[root_indices]

        values_sum = torch.zeros(
            B, helper.num_players, NUM_HANDS, device=device, dtype=helper.float_dtype
        )

        model.eval()
        chunk_size = 128
        for start in range(0, num_flops, chunk_size):
            end = min(start + chunk_size, num_flops)
            chunk_len = end - start

            flop_chunk = all_flops[start:end]
            board_chunk = torch.cat(
                [
                    flop_chunk,
                    torch.full((chunk_len, 2), -1, device=device, dtype=torch.long),
                ],
                dim=1,
            )

            # Build allowed mask for each flop
            board_onehot = torch.zeros(
                chunk_len, 52, dtype=helper.float_dtype, device=device
            )
            board_onehot.scatter_(
                1,
                flop_chunk,
                torch.ones(chunk_len, 3, dtype=helper.float_dtype, device=device),
            )
            conflict_matrix = helper.combo_onehot_float @ board_onehot.T
            allowed_chunk = (conflict_matrix == 0).T

            allowed_broadcast = (
                allowed_chunk.unsqueeze(0)
                .unsqueeze(2)
                .expand(B, chunk_len, helper.num_players, NUM_HANDS)
            )

            post_beliefs = (
                pre_beliefs.unsqueeze(1).expand(-1, chunk_len, -1, -1).clone()
            )
            post_beliefs[..., ~allowed_broadcast] = 0.0

            sums = post_beliefs.sum(dim=-1, keepdim=True)
            uniform = allowed_broadcast.to(helper.float_dtype)
            uniform_sum = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = uniform / uniform_sum

            normalized_beliefs = torch.where(
                sums > 1e-12, post_beliefs / sums.clamp(min=1e-12), uniform
            )

            belief_features = normalized_beliefs.reshape(B * chunk_len, -1)
            board_samples_flat = (
                board_chunk.unsqueeze(0).expand(B, -1, -1).reshape(-1, 5)
            )

            context_expand = (
                context_root.unsqueeze(1)
                .expand(-1, chunk_len, -1)
                .reshape(-1, context_root.shape[1])
            )
            street_expand = street_root.unsqueeze(1).expand(-1, chunk_len).reshape(-1)
            to_act_expand = to_act_root.unsqueeze(1).expand(-1, chunk_len).reshape(-1)

            synthetic_features = MLPFeatures(
                context=context_expand,
                street=street_expand,
                to_act=to_act_expand,
                board=board_samples_flat,
                beliefs=belief_features,
            )
            hand_values = model(synthetic_features).hand_values.to(
                dtype=helper.float_dtype
            )
            hand_values = hand_values.view(B, chunk_len, helper.num_players, NUM_HANDS)

            values_sum += hand_values.sum(dim=1)

        expected = values_sum / num_flops
        torch.testing.assert_close(result, expected)

    def test_single_card_chance_values_empty_input(self, helper: ChanceNodeHelper):
        """Test single_card_chance_values with empty input."""
        device = helper.device
        empty_indices = torch.tensor([], dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(0, 1, device=device),
            street=torch.zeros(0, device=device, dtype=torch.long),
            to_act=torch.zeros(0, device=device, dtype=torch.long),
            board=torch.zeros(0, 5, device=device, dtype=torch.long),
            beliefs=torch.zeros(0, 2 * NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.zeros(0, 2, NUM_HANDS, device=device)
        board_pre = torch.full((0, 5), -1, dtype=torch.long, device=device)

        result = helper.single_card_chance_values(
            empty_indices, root_features, pre_chance_beliefs, board_pre
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
            to_act=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.cat(
                [
                    torch.zeros(B, 3, device=device, dtype=torch.long),
                    torch.full((B, 2), -1, device=device, dtype=torch.long),
                ],
                dim=1,
            ),
            beliefs=torch.zeros(B, 2 * NUM_HANDS, device=device),
        )
        pre_chance_beliefs = (
            torch.ones(B, 2, NUM_HANDS, device=device, dtype=dtype) / NUM_HANDS
        )
        board_pre = torch.full((B, 5), -1, dtype=torch.long, device=device)

        result = helper.single_card_chance_values(
            root_indices, root_features, pre_chance_beliefs, board_pre
        )

        assert result.shape == (B, 2, NUM_HANDS)
        assert result.device.type == device.type
        assert result.dtype == dtype

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

    def test_flop_chance_values_device_handling(self, helper: ChanceNodeHelper):
        """Test flop_chance_values handles device correctly."""
        device = helper.device
        B = 2

        root_indices = torch.arange(B, dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(B, 1, device=device),
            street=torch.zeros(B, device=device, dtype=torch.long),
            to_act=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.cat(
                [
                    torch.zeros(B, 3, device=device, dtype=torch.long),
                    torch.full((B, 2), -1, device=device, dtype=torch.long),
                ],
                dim=1,
            ),
            beliefs=torch.zeros(B, 2 * NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.ones(B, 2, NUM_HANDS, device=device) / NUM_HANDS

        result = helper.flop_chance_values(
            root_indices, root_features, pre_chance_beliefs
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
            to_act=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.cat(
                [
                    torch.zeros(B, 3, device=device, dtype=torch.long),
                    torch.full((B, 2), -1, device=device, dtype=torch.long),
                ],
                dim=1,
            ),
            beliefs=torch.zeros(B, 2 * NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.ones(B, 2, NUM_HANDS, device=device) / NUM_HANDS
        board_pre = torch.full((B, 5), -1, dtype=torch.long, device=device)

        result = helper.single_card_chance_values(
            root_indices, root_features, pre_chance_beliefs, board_pre
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
