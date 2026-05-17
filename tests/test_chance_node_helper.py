"""Test suite for ChanceNodeHelper."""

from __future__ import annotations

import pytest
import torch

from p2.env.card_utils import (
    NUM_HANDS,
    calculate_unblocked_mass,
    combo_to_onehot_tensor,
)
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


class LegalHandMaskModel:
    """Model that returns 1 for hands compatible with the board and 0 otherwise."""

    def __init__(self, device: torch.device, float_dtype: torch.dtype):
        self.device = device
        self.float_dtype = float_dtype
        self.combo_onehot = combo_to_onehot_tensor(device=device).float()

    def __call__(self, features: MLPFeatures) -> ModelOutput:
        batch_size = features.context.shape[0]
        board = features.board
        board_onehot = torch.zeros(
            batch_size, 52, dtype=torch.float32, device=self.device
        )
        valid_rows, valid_slots = torch.nonzero(board >= 0, as_tuple=True)
        if valid_rows.numel() > 0:
            board_onehot[valid_rows, board[valid_rows, valid_slots]] = 1.0

        allowed = (self.combo_onehot @ board_onehot.T < 0.5).T
        hand_values = allowed[:, None, :].expand(-1, 2, -1).to(self.float_dtype)
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

    def test_raw_flop_initialization(self, helper: ChanceNodeHelper):
        """Test that raw flop enumeration is initialized correctly."""
        assert helper.all_flops.shape == (22100, 3)
        assert helper.all_flops.dtype == torch.long
        assert helper.all_flops.device.type == helper.device.type

    def test_raw_flop_values_are_valid(self, helper: ChanceNodeHelper):
        """Test that raw flops contain sorted unique card triples."""
        all_flops = helper.all_flops
        assert torch.all(all_flops >= 0)
        assert torch.all(all_flops < 52)
        assert torch.all(all_flops[:, 0] < all_flops[:, 1])
        assert torch.all(all_flops[:, 1] < all_flops[:, 2])

    def test_raw_flop_enumeration_matches_torch_combinations(
        self, helper: ChanceNodeHelper
    ):
        """Test that helper flops cover every 3-card combination exactly once."""
        expected = torch.combinations(
            torch.arange(52, dtype=torch.long, device=helper.device),
            r=3,
            with_replacement=False,
        )
        torch.testing.assert_close(helper.all_flops, expected)

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
        helper.FLOP_SAMPLE_SIZE = 0

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
        weight_sum = torch.zeros_like(values_sum)

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

            post_unnorm = (
                pre_beliefs.unsqueeze(1).expand(-1, chunk_len, -1, -1).clone()
            )
            post_unnorm[..., ~allowed_broadcast] = 0.0

            sums = post_unnorm.sum(dim=-1, keepdim=True)
            uniform = allowed_broadcast.to(helper.float_dtype)
            uniform_sum = uniform.sum(dim=-1, keepdim=True).clamp(min=1.0)
            uniform = uniform / uniform_sum

            normalized_beliefs = torch.where(
                sums > 1e-12, post_unnorm / sums.clamp(min=1e-12), uniform
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

            weights = calculate_unblocked_mass(post_unnorm).flip(dims=[-2])
            weights = weights * allowed_chunk.unsqueeze(0).unsqueeze(2).to(
                dtype=weights.dtype
            )
            values_sum += (hand_values * weights).sum(dim=1)
            weight_sum += weights.sum(dim=1)

        expected = torch.where(
            weight_sum > 1e-12,
            values_sum / weight_sum.clamp(min=1e-12),
            torch.zeros_like(values_sum),
        )
        torch.testing.assert_close(result, expected)

    def test_flop_chance_values_excludes_hand_colliding_flops(
        self, device: torch.device
    ):
        """Legal post-flop hand slots should average to one, not be diluted."""
        model = LegalHandMaskModel(device, torch.float32)
        helper = ChanceNodeHelper(
            device=device,
            float_dtype=torch.float32,
            num_players=2,
            model=model,
        )
        helper.FLOP_SAMPLE_SIZE = 0

        B = 1
        root_indices = torch.arange(B, device=device, dtype=torch.long)
        root_features = MLPFeatures(
            context=torch.zeros(B, 1, device=device),
            street=torch.zeros(B, device=device, dtype=torch.long),
            to_act=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.full((B, 5), -1, device=device, dtype=torch.long),
            beliefs=torch.zeros(B, 2 * NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.ones(B, 2, NUM_HANDS, device=device) / NUM_HANDS

        result = helper.flop_chance_values(
            root_indices, root_features, pre_chance_beliefs
        )

        torch.testing.assert_close(result, torch.ones_like(result), atol=1e-6, rtol=0)

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

    def test_single_card_chance_values_excludes_hand_colliding_cards(
        self, device: torch.device
    ):
        """Legal post-card hand slots should average to one, not count invalid cards."""
        model = LegalHandMaskModel(device, torch.float32)
        helper = ChanceNodeHelper(
            device=device,
            float_dtype=torch.float32,
            num_players=2,
            model=model,
        )

        B = 1
        root_indices = torch.arange(B, dtype=torch.long, device=device)
        root_features = MLPFeatures(
            context=torch.zeros(B, 1, device=device),
            street=torch.ones(B, device=device, dtype=torch.long),
            to_act=torch.zeros(B, device=device, dtype=torch.long),
            board=torch.tensor([[0, 1, 2, 3, -1]], dtype=torch.long, device=device),
            beliefs=torch.zeros(B, 2 * NUM_HANDS, device=device),
        )
        pre_chance_beliefs = torch.ones(B, 2, NUM_HANDS, device=device) / NUM_HANDS
        board_pre = torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.long, device=device)

        result = helper.single_card_chance_values(
            root_indices, root_features, pre_chance_beliefs, board_pre
        )

        board_onehot = torch.zeros(1, 52, dtype=torch.float32, device=device)
        board_onehot[0, :3] = 1.0
        pre_allowed = (helper.combo_onehot_float @ board_onehot.T < 0.5).T[0]

        torch.testing.assert_close(
            result[:, :, pre_allowed],
            torch.ones_like(result[:, :, pre_allowed]),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            result[:, :, ~pre_allowed],
            torch.zeros_like(result[:, :, ~pre_allowed]),
            atol=1e-6,
            rtol=0,
        )

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
