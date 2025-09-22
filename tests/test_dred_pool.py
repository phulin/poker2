"""Tests for DRED opponent pool."""

from __future__ import annotations

import torch

from alphaholdem.models.cnn import SiameseConvNetV1
from alphaholdem.rl.agent_snapshot import AgentSnapshot
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
    from alphaholdem.models.cnn_embedding_data import CNNEmbeddingData

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add more snapshots than max_size to trigger pruning
    for i in range(10):
        elo = 1200 + i * 10  # Increasing ELO ratings
        pool.add_snapshot(model, step=i * 100, rating=elo)

    # After adding 10 snapshots, should be pruned to max_size=5
    assert len(pool.snapshots) == 5

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
    from alphaholdem.models.cnn_embedding_data import CNNEmbeddingData

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots with specific ELO ratings
    pool.add_snapshot(model, step=100, rating=1200)  # Low ELO
    pool.add_snapshot(model, step=200, rating=1300)  # High ELO
    pool.add_snapshot(model, step=300, rating=1250)  # Medium ELO
    pool.add_snapshot(model, step=400, rating=1350)  # Highest ELO
    pool.add_snapshot(model, step=500, rating=1210)  # Low ELO

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
    from alphaholdem.models.cnn_embedding_data import CNNEmbeddingData

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots with different characteristics
    # Group 1: High ELO snapshots
    pool.add_snapshot(model, step=100, rating=1300)
    pool.add_snapshot(model, step=200, rating=1320)

    # Group 2: Medium ELO snapshots
    pool.add_snapshot(model, step=300, rating=1200)
    pool.add_snapshot(model, step=400, rating=1220)

    # Group 3: Low ELO snapshots
    pool.add_snapshot(model, step=500, rating=1100)
    pool.add_snapshot(model, step=600, rating=1120)

    # Add more to trigger pruning
    pool.add_snapshot(model, step=700, rating=1150)
    pool.add_snapshot(model, step=800, rating=1250)

    # Should be pruned to max_size=4
    assert len(pool.snapshots) == 4

    # Should maintain diversity through clustering
    elos = [snapshot.elo for snapshot in pool.snapshots]

    # Should have some high ELO snapshots (top 10% preserved)
    assert max(elos) >= 1300

    # Should have some diversity (not all from same ELO range)
    elo_range = max(elos) - min(elos)
    assert elo_range > 50  # Should have reasonable diversity


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
    pool.add_snapshot(model, step=100, rating=1200)
    pool.add_snapshot(model, step=200, rating=1300)
    pool.add_snapshot(model, step=300, rating=1250)

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
        pool.add_snapshot(model, step=i * 100, rating=1200 + i * 10)

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
    from alphaholdem.models.cnn_embedding_data import CNNEmbeddingData

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots with different steps
    pool.add_snapshot(model, step=100, rating=1200)
    pool.add_snapshot(model, step=200, rating=1250)
    pool.add_snapshot(model, step=300, rating=1300)
    pool.add_snapshot(model, step=400, rating=1350)
    pool.add_snapshot(model, step=500, rating=1400)

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
    from alphaholdem.models.cnn_embedding_data import CNNEmbeddingData

    sample = CNNEmbeddingData(
        cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
    )
    pool.set_last_batch_data(sample)

    # Add snapshots to trigger pruning
    for i in range(8):
        pool.add_snapshot(model, step=i * 100, rating=1200 + i * 5)

    # Should be pruned to max_size=3
    assert len(pool.snapshots) == 3

    # Test that embeddings can still be generated for remaining snapshots
    for snapshot in pool.snapshots:
        # Provide a minimal sample batch using zeros matching CNNEmbeddingData
        from alphaholdem.models.cnn_embedding_data import CNNEmbeddingData

        sample = CNNEmbeddingData(
            cards=torch.zeros(1, 6, 4, 13), actions=torch.zeros(1, 24, 4, 8)
        )
        embedding = pool._generate_embedding(snapshot, sample)
        assert embedding.dim() == 1
        assert isinstance(embedding, torch.Tensor)


def test_dred_prune_kmedoids_integration():
    """Test that pruning integrates properly with k-medoids clustering."""
    pool = DREDPool(max_size=5)
    from alphaholdem.models.cnn_embedding_data import CNNEmbeddingData

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
        pool.add_snapshot(model, step=i * 50, rating=elo)

    # Should be pruned to max_size=5
    assert len(pool.snapshots) == 5

    # Verify that snapshots are still valid
    for snapshot in pool.snapshots:
        assert snapshot.model is not None
        assert snapshot.step >= 0
        assert snapshot.elo >= 1200
        assert isinstance(snapshot.data, DREDSnapshotData)
