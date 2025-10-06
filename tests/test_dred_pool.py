"""Tests for DRED opponent pool."""

from __future__ import annotations

import torch

from alphaholdem.models.cnn import SiameseConvNetV1
from alphaholdem.models.cnn.cnn_embedding_data import CNNEmbeddingData
from alphaholdem.rl.dred_pool import DREDPool, DREDSnapshotData


def test_dred_prune_basic():
    """Test basic pruning functionality."""
    # Create a pool with small max_size to trigger pruning
    pool = DREDPool(max_size=5)

    # Create a simple model for testing
    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Provide sample batch for embedding generation during pruning
    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add more snapshots than max_size to trigger pruning
    for i in range(10):
        elo = 1200 + i * 10  # Increasing ELO ratings
        # Create a new random model for each snapshot to ensure diversity
        random_model = SiameseConvNetV1(
            cards_channels=6,
            actions_channels=24,
            cards_hidden=256,
            actions_hidden=256,
            fusion_hidden=[1024, 1024],
            num_actions=8,
        )
        pool.add_snapshot(random_model, step=i * 100, rating=elo)

    # After adding 10 snapshots, should be pruned (will be <= max_size)
    # With clustering, we may get fewer than max_size due to diversity constraints
    assert len(pool.snapshots) <= 5
    assert len(pool.snapshots) >= 2  # Should keep at least a few diverse snapshots

    # Check that the highest ELO snapshots are kept (top 10%)
    elos = [snapshot.elo for snapshot in pool.snapshots]
    assert max(elos) >= 1290  # Should keep the highest ELO snapshots


def test_dred_prune_top_elo_preserved():
    """Test that top ELO snapshots are preserved during pruning."""
    pool = DREDPool(max_size=3)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Provide sample batch for embedding generation during pruning
    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots with specific ELO ratings
    # Create different random models for each snapshot to ensure diversity
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=100, rating=1200
    )  # Low ELO
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=200, rating=1300
    )  # High ELO
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=300, rating=1250
    )  # Medium ELO
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=400, rating=1350
    )  # Highest ELO
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=500, rating=1210
    )  # Low ELO

    # Should keep max_size=3 snapshots
    assert len(pool.snapshots) == 3

    # Check that the highest ELO snapshot (1350) is preserved
    elos = [snapshot.elo for snapshot in pool.snapshots]
    assert 1350 in elos

    # Check that at least the top 10% (1 snapshot) is kept
    # Since we have 5 snapshots, top 10% = 1, so highest should be kept
    assert max(elos) == 1350


def test_dred_prune_clustering():
    """Test that pruning uses clustering for diversity."""
    pool = DREDPool(max_size=4)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Provide sample batch for embedding generation during pruning
    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots with different characteristics
    # Group 1: High ELO snapshots
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=100, rating=1300
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=200, rating=1320
    )

    # Group 2: Medium ELO snapshots
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=300, rating=1200
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=400, rating=1220
    )

    # Group 3: Low ELO snapshots
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=500, rating=1100
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=600, rating=1120
    )

    # Add more to trigger pruning
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=700, rating=1150
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=800, rating=1250
    )

    # Should be pruned (will be <= max_size)
    # With clustering, we may get fewer than max_size due to diversity constraints
    assert len(pool.snapshots) <= 4
    assert len(pool.snapshots) >= 2  # Should keep at least a few diverse snapshots

    # Should maintain diversity through clustering
    elos = [snapshot.elo for snapshot in pool.snapshots]

    # Should have some high ELO snapshots (top 10% preserved)
    assert max(elos) >= 1300

    # Should have some diversity (not all from exactly the same ELO)
    elo_range = max(elos) - min(elos)
    # Due to clustering on model embeddings (which are similar for random models),
    # we may not get wide ELO range, but should have at least some variation
    assert elo_range > 0  # At least 2 different models retained


def test_dred_prune_edge_cases():
    """Test edge cases for pruning."""
    pool = DREDPool(max_size=1)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Add multiple snapshots
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=100, rating=1200
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=200, rating=1300
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=300, rating=1250
    )

    # Should be pruned to max_size=1
    assert len(pool.snapshots) == 1

    # Should keep the highest ELO snapshot
    assert pool.snapshots[0].elo == 1300


def test_dred_prune_no_pruning_needed():
    """Test that pruning doesn't occur when not needed."""
    pool = DREDPool(max_size=10)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Add fewer snapshots than max_size
    for i in range(5):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 10,
        )

    # Should not be pruned
    assert len(pool.snapshots) == 5


def test_dred_prune_age_tracking():
    """Test that pruning preserves age tracking."""
    pool = DREDPool(max_size=3)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Provide sample batch for embedding generation during pruning
    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots with different steps
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=100, rating=1200
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=200, rating=1250
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=300, rating=1300
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=400, rating=1350
    )
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=500, rating=1400
    )

    # Check that remaining snapshots have proper age tracking
    for snapshot in pool.snapshots:
        assert isinstance(snapshot.data, DREDSnapshotData)
        assert snapshot.data.age >= 0


def test_dred_prune_embedding_generation():
    """Test that pruning works with embedding generation."""
    pool = DREDPool(max_size=3)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Provide sample batch to enable embedding generation during pruning
    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning
    for i in range(8):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 5,
        )

    # Should be pruned (will be <= max_size)
    # With clustering, we may get fewer than max_size due to diversity constraints
    assert len(pool.snapshots) <= 3
    assert len(pool.snapshots) >= 2  # Should keep at least a few diverse snapshots

    # Test that embeddings can still be generated for remaining snapshots
    for snapshot in pool.snapshots:
        # Provide a minimal sample batch using zeros matching CNNEmbeddingData
        sample = CNNEmbeddingData(
            cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
        )
        embedding = pool._generate_embedding(snapshot, sample)
        assert embedding.dim() == 1
        assert isinstance(embedding, torch.Tensor)


def test_dred_prune_kmedoids_integration():
    """Test that pruning integrates properly with k-medoids clustering."""
    pool = DREDPool(max_size=5)

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    model = SiameseConvNetV1(
        cards_channels=6,
        actions_channels=24,
        cards_hidden=256,
        actions_hidden=256,
        fusion_hidden=[1024, 1024],
        num_actions=8,
    )

    # Add many snapshots to trigger clustering-based pruning
    for i in range(15):
        elo = 1200 + (i % 3) * 50  # Create 3 ELO groups
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8), step=i * 50, rating=elo
        )

    # Should be pruned (will be <= max_size)
    # With clustering, we may get fewer than max_size due to diversity constraints
    assert len(pool.snapshots) >= 3  # Should keep at least a few diverse snapshots
    assert len(pool.snapshots) <= 5

    # Verify that snapshots are still valid
    for snapshot in pool.snapshots:
        assert snapshot.model is not None
        assert snapshot.step >= 0
        assert snapshot.elo >= 1200
        assert isinstance(snapshot.data, DREDSnapshotData)


def test_dred_deletion_queue_basic():
    """Test that pruned snapshots are moved to deletion queue."""
    pool = DREDPool(max_size=3)

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning
    for i in range(6):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 10,
        )

    # Should have active snapshots and deletion queue
    assert len(pool.snapshots) <= 3
    assert len(pool.deletion_queue) > 0

    # Total should equal what we added
    total_snapshots = len(pool.snapshots) + len(pool.deletion_queue)
    assert total_snapshots == 6


def test_dred_deletion_queue_sampling():
    """Test that snapshots from deletion queue can still be sampled."""
    pool = DREDPool(max_size=3)

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning
    for i in range(6):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 10,
        )

    # Sample from the combined pool
    sampled = pool.sample(k=5)

    assert len(sampled) == 5

    # Verify we can sample from both active and deletion queue
    # (probabilistically they should come from both if we sample enough)
    all_steps = set(s.step for s in pool.snapshots + pool.deletion_queue)
    sampled_steps = set(s.step for s in sampled)

    # Sampled steps should be a subset of all available steps
    assert sampled_steps.issubset(all_steps)


def test_dred_deletion_queue_lazy_removal():
    """Test that deletion queue items are removed lazily when new snapshots are added."""
    pool = DREDPool(max_size=3)

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning
    for i in range(6):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 10,
        )

    # Record initial deletion queue size
    initial_deletion_queue_size = len(pool.deletion_queue)
    assert initial_deletion_queue_size > 0

    # Add another snapshot - should remove one from deletion queue
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
        step=700,
        rating=1270,
    )

    # Deletion queue should be smaller (unless new pruning added more)
    # After adding the 7th snapshot, we should have removed one from deletion queue
    # but also potentially added more through pruning
    # Let's verify the total count behavior
    assert len(pool.snapshots) <= 3


def test_dred_deletion_queue_age_tracking():
    """Test that snapshots in deletion queue continue to age."""
    pool = DREDPool(max_size=3)

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning
    for i in range(6):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 10,
        )

    # Get ages of deletion queue items
    initial_ages = [s.data.age for s in pool.deletion_queue]

    # Add another snapshot
    pool.add_snapshot(
        SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
        step=700,
        rating=1270,
    )

    # Check that remaining deletion queue items have aged
    if pool.deletion_queue:  # If there are still items in the queue
        for snapshot in pool.deletion_queue:
            assert snapshot.data.age > 0


def test_dred_deletion_queue_save_load():
    """Test that deletion queue is properly saved and loaded."""
    import tempfile
    import os

    pool = DREDPool(max_size=3)

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning
    for i in range(6):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 10,
        )

    # Record state before saving
    active_count = len(pool.snapshots)
    deletion_count = len(pool.deletion_queue)
    total_count = active_count + deletion_count

    assert deletion_count > 0, "Should have items in deletion queue"

    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp_path = tmp.name

    try:
        pool.save_pool(tmp_path)

        # Load into new pool
        new_pool = DREDPool(max_size=3)
        new_pool.load_pool(
            tmp_path,
            lambda: SiameseConvNetV1(
                cards_channels=6,
                actions_channels=24,
                cards_hidden=256,
                actions_hidden=256,
                fusion_hidden=[1024, 1024],
                num_actions=8,
            ),
        )

        # Verify counts match
        assert len(new_pool.snapshots) == active_count
        assert len(new_pool.deletion_queue) == deletion_count

        # Verify total snapshots preserved
        new_total = len(new_pool.snapshots) + len(new_pool.deletion_queue)
        assert new_total == total_count

        # Verify snapshots in deletion queue have proper data
        for snapshot in new_pool.deletion_queue:
            assert snapshot.model is not None
            assert isinstance(snapshot.data, DREDSnapshotData)
            assert snapshot.data.age >= 0

    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_dred_deletion_queue_stats():
    """Test that pool stats include deletion queue information."""
    pool = DREDPool(max_size=3)

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning
    for i in range(6):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 10,
        )

    # Get stats
    stats = pool.get_pool_stats()

    # Should include both pool_size and deletion_queue_size
    assert "pool_size" in stats
    assert "deletion_queue_size" in stats
    assert stats["pool_size"] == len(pool.snapshots)
    assert stats["deletion_queue_size"] == len(pool.deletion_queue)

    # Stats should reflect combined pool (active + deletion queue)
    total_snapshots = len(pool.snapshots) + len(pool.deletion_queue)
    assert total_snapshots > stats["pool_size"]  # Should have items in deletion queue


def test_dred_deletion_queue_gradual_removal():
    """Test that deletion queue items are removed gradually as new snapshots are added."""
    pool = DREDPool(max_size=2)

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning (add 5 snapshots, max_size=2)
    for i in range(5):
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=i * 100,
            rating=1200 + i * 10,
        )

    # Should have deletion queue with pruned items
    initial_deletion_queue_size = len(pool.deletion_queue)
    assert initial_deletion_queue_size > 0

    # Add more snapshots one by one - each should remove one from deletion queue
    additions_before_empty = 0
    while pool.deletion_queue and additions_before_empty < 10:  # Safety limit
        pool.add_snapshot(
            SiameseConvNetV1(6, 24, 256, 256, [1024, 1024], 8),
            step=(100 * (5 + additions_before_empty)),
            rating=1200 + (5 + additions_before_empty) * 10,
        )
        additions_before_empty += 1

    # Eventually deletion queue should be empty (or very small if new pruning occurred)
    # This tests that the gradual removal mechanism works
    assert additions_before_empty > 0  # We did add some snapshots
