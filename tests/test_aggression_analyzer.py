from __future__ import annotations

import torch

from alphaholdem.env.aggression_analyzer import (
    IDX_TO_RANK,
    AggressionAnalyzer,
    build_hand_to_group_mapping,
    hand_combos_tensor,
)
from alphaholdem.env.card_utils import NUM_HANDS
from alphaholdem.rl.rebel_replay import RebelBatch


def test_aggression_analyzer_singleton() -> None:
    """Test that AggressionAnalyzer is a singleton."""
    analyzer1 = AggressionAnalyzer()
    analyzer2 = AggressionAnalyzer()

    assert analyzer1 is analyzer2


def test_aggression_analyzer_no_policy() -> None:
    """Test analyzer with a batch that has no policy targets."""
    analyzer = AggressionAnalyzer()

    batch = RebelBatch(
        features=torch.randn(10, 100),
        legal_masks=torch.ones(10, 5, dtype=torch.bool),
        value_targets=torch.randn(10),  # Required by RebelBatch
        statistics={"bet_amounts": torch.randn(10, 5) * 10 + 50},
    )

    result = analyzer.analyze_batch(batch)

    assert "group_avg_bets" in result
    assert "group_counts" in result
    assert "overall_avg" in result
    assert "overall_std" in result


def test_aggression_analyzer_with_policy() -> None:
    """Test analyzer with policy targets."""
    analyzer = AggressionAnalyzer()

    batch_size = 10
    num_actions = 5

    # Create a simple batch with policy targets
    batch = RebelBatch(
        features=torch.randn(batch_size, 100),
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
        policy_targets=torch.randn(batch_size, NUM_HANDS, num_actions),
        statistics={"bet_amounts": torch.randn(batch_size, num_actions) * 10 + 50},
    )

    result = analyzer.analyze_batch(batch)

    assert "group_avg_bets" in result
    assert "group_counts" in result
    assert "overall_avg" in result
    assert "overall_std" in result

    # Check that we have 5 groups
    assert result["group_avg_bets"].shape == (5,)
    assert result["group_counts"].shape == (5,)


def test_build_hand_to_group_mapping() -> None:
    """Test that build_hand_to_group_mapping creates correct group structure."""
    # Build the mapping
    chunks = build_hand_to_group_mapping()
    assert len(chunks) == 5, "Should have 5 groups"

    # Flatten to get the complete mapping
    all_combos = torch.cat([chunk for chunk in chunks])
    assert len(all_combos) == NUM_HANDS, "Should have all 1326 combos"

    combos = hand_combos_tensor()

    # Check that AA combos come first (6 combos for 4 suits choose 2)
    aa_indices = all_combos[:6]
    for idx in aa_indices:
        c1, c2 = combos[idx]
        r1, r2 = c1 % 13, c2 % 13
        assert r1 == r2 == 12, f"Should be AA, got {IDX_TO_RANK[r1]}{IDX_TO_RANK[r2]}"

    # Check that KK combos come next (6 combos)
    kk_indices = all_combos[6:12]
    for idx in kk_indices:
        c1, c2 = combos[idx]
        r1, r2 = c1 % 13, c2 % 13
        assert r1 == r2 == 11, f"Should be KK, got {IDX_TO_RANK[r1]}{IDX_TO_RANK[r2]}"

    # Check that 32o combos come last (12 offsuit combinations)
    last_indices = all_combos[-12:]
    for idx in last_indices:
        c1, c2 = combos[idx]
        r1, r2 = c1 % 13, c2 % 13
        is_suited = c1 // 13 == c2 // 13
        # Should be 32o: rank 1 (3) and rank 0 (2), not suited
        assert sorted([r1, r2]) == [0, 1], f"Should be 32, got ranks {r1},{r2}"
        assert not is_suited, "Should be offsuit"


def test_analyze_batch_correctness() -> None:
    """Test that analyze_batch computes correct statistics with group-specific policies."""

    analyzer = AggressionAnalyzer()

    batch_size = 20
    num_actions = 5

    # Get the group chunks to know which hands are in which groups
    chunk_tuples = build_hand_to_group_mapping()

    # Define bet amounts for each action (same for all states)
    # Action indices: 0=fold(0), 1=call(5), 2=bet1(20), 3=bet2(50), 4=allin(100)
    bet_amounts_per_action = torch.tensor([0.0, 5.0, 20.0, 50.0, 100.0])
    bet_amounts = bet_amounts_per_action.unsqueeze(0).expand(batch_size, -1)

    # Create policy targets with group-specific actions
    policy_targets = torch.zeros(batch_size, NUM_HANDS, num_actions)

    # Define action for each group
    group_actions = [4, 3, 3, 1, 0]  # all-in, bet, bet, call, fold

    for group_idx, chunk in enumerate(chunk_tuples):
        action_idx = group_actions[group_idx]
        # Set policy to concentrate all probability on the chosen action for all hands in this chunk
        policy_targets[:, chunk, action_idx] = 1.0

    batch = RebelBatch(
        features=torch.randn(batch_size, 100),
        legal_masks=torch.ones(batch_size, num_actions, dtype=torch.bool),
        policy_targets=policy_targets,
        statistics={"bet_amounts": bet_amounts},
    )

    result = analyzer.analyze_batch(batch)

    # Check structure
    assert "group_avg_bets" in result
    assert "group_counts" in result
    assert "overall_avg" in result
    assert "overall_std" in result

    # Check dimensions
    assert result["group_avg_bets"].shape == (5,)
    assert result["group_counts"].shape == (5,)

    # Verify that total count equals number of states * hands
    total_states_hands = batch_size * NUM_HANDS
    assert result["group_counts"].sum().item() == total_states_hands

    # Verify group-specific averages are correct
    expected_avgs = torch.tensor([100.0, 50.0, 50.0, 5.0, 0.0])
    torch.testing.assert_close(result["group_avg_bets"], expected_avgs)
