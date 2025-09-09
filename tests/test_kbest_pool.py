from __future__ import annotations

import torch
from alphaholdem.rl.k_best_pool import KBestOpponentPool, AgentSnapshot
from alphaholdem.models.siamese_convnet import SiameseConvNetV1


def test_kbest_add_and_sample():
    pool = KBestOpponentPool(k=3, min_elo_diff=25.0)

    # Add three snapshots with increasing ELO
    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    pool.add_snapshot(model, 10, rating=1200)
    pool.add_snapshot(model, 20, rating=1250)
    pool.add_snapshot(model, 30, rating=1300)

    stats = pool.get_pool_stats()
    assert stats["pool_size"] == 3
    assert stats["best_snapshot_elo"] == 1300

    # Sampling should return up to k opponents
    sampled = pool.sample(k=2)
    assert 1 <= len(sampled) <= 2


def test_kbest_save_and_load(tmp_path):
    pool = KBestOpponentPool(k=2, min_elo_diff=25.0)
    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    pool.add_snapshot(model, 100, rating=1275)

    path = tmp_path / "pool.pt"
    pool.save_pool(str(path))
    assert path.exists()

    new_pool = KBestOpponentPool(k=2)
    new_pool.load_pool(
        str(path),
        lambda: SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=256,
            actions_hidden=256,
            fusion_hidden=[1024, 1024],
            num_actions=8,
        ),
    )

    stats = new_pool.get_pool_stats()
    assert stats["pool_size"] == 1
    assert stats["best_snapshot_elo"] == 1275


def test_min_step_diff():
    """Test that min_step_diff prevents rapid snapshot additions when pool is full."""
    pool = KBestOpponentPool(
        k=2, min_elo_diff=50.0, min_step_diff=100
    )  # Smaller pool for easier testing

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Add first snapshot at step 0
    pool.add_snapshot(model, 0, rating=1200)
    assert len(pool.snapshots) == 1

    # Set current ELO to 1200 (same as first snapshot)
    pool.current_elo = 1200

    # Pool is not full yet, so should always allow adding snapshots
    assert pool.should_add_snapshot(50)  # Should be True (pool not full)

    # Add second snapshot at step 50 to fill the pool
    pool.add_snapshot(model, 50, rating=1250)
    assert len(pool.snapshots) == 2

    # Set current ELO to 1250 (same as second snapshot)
    pool.current_elo = 1250

    # Now pool is full, so constraints apply
    # Try to add snapshot at step 100 - should be rejected (step diff = 50 < 100 from latest snapshot)
    assert not pool.should_add_snapshot(100)

    # Try to add snapshot at step 200 - should be accepted (step diff = 150 >= 100, ELO diff = 50 >= 50 with snapshot 0)
    assert pool.should_add_snapshot(200)

    # Set current ELO to 1300 (ELO diff = 50 >= 50 with snapshot 0)
    pool.current_elo = 1300

    # Try to add snapshot at step 200 - should be accepted (ELO diff >= 50 and step diff >= 100)
    assert pool.should_add_snapshot(200)

    # Add third snapshot at step 200 (this will replace the worst one)
    pool.add_snapshot(model, 200, rating=1300)
    assert len(pool.snapshots) == 2  # Still 2 because k=2

    # Verify snapshots are ordered by ELO
    assert pool.snapshots[0].elo == 1300
    assert pool.snapshots[1].elo == 1250


def test_min_step_diff_with_elo_diff():
    """Test that both min_step_diff and min_elo_diff must be satisfied."""
    pool = KBestOpponentPool(k=3, min_elo_diff=50.0, min_step_diff=100)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Add first snapshot at step 0 with ELO 1200
    pool.add_snapshot(model, 0, rating=1200)
    assert len(pool.snapshots) == 1

    # Set current ELO to 1200 (same as first snapshot)
    pool.current_elo = 1200

    # Try to add snapshot at step 150 - should be accepted (step diff OK, ELO diff = 50 >= 50 with snapshot 0)
    assert pool.should_add_snapshot(150)

    # Set current ELO to 1250 (ELO diff = 50)
    pool.current_elo = 1250

    # Try to add snapshot at step 150 - should be accepted (ELO diff OK, step diff = 150 >= 100)
    assert pool.should_add_snapshot(150)

    # Try to add snapshot at step 200 - should be accepted (both conditions met)
    assert pool.should_add_snapshot(200)

    # Add second snapshot
    pool.add_snapshot(model, 200, rating=1250)
    assert len(pool.snapshots) == 2


def test_min_step_diff_edge_cases():
    """Test edge cases for min_step_diff."""
    pool = KBestOpponentPool(k=2, min_elo_diff=25.0, min_step_diff=100)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Test with empty pool - should always allow first snapshot
    assert pool.should_add_snapshot(0)
    pool.add_snapshot(model, 0, rating=1200)

    # Test exact step difference threshold
    assert pool.should_add_snapshot(99)  # Pool not full, so always allow
    assert pool.should_add_snapshot(100)  # Pool not full, so always allow

    # Test with multiple snapshots - should check against ALL existing snapshots
    pool.add_snapshot(model, 100, rating=1250)

    # Try to add at step 150 - should be rejected (only 50 steps from last snapshot)
    assert not pool.should_add_snapshot(150)

    # Try to add at step 250 - should be accepted (200 steps from first, 150 from second)
    assert pool.should_add_snapshot(250)
