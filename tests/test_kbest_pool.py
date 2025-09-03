from __future__ import annotations

import torch
from alphaholdem.rl.k_best_pool import KBestOpponentPool, AgentSnapshot
from alphaholdem.models.siamese_convnet import SiameseConvNetV1


def test_kbest_add_and_sample():
    pool = KBestOpponentPool(k=3, min_elo_diff=25.0)

    # Add three snapshots with increasing ELO
    model = SiameseConvNetV1()

    class Dummy:
        def __init__(self, model, ep):
            self.model = model
            self.episode_count = ep

    pool.add_snapshot(Dummy(model, 10), rating=1200)
    pool.add_snapshot(Dummy(model, 20), rating=1250)
    pool.add_snapshot(Dummy(model, 30), rating=1300)

    stats = pool.get_pool_stats()
    assert stats["pool_size"] == 3
    assert stats["best_snapshot_elo"] == 1300

    # Sampling should return up to k opponents
    sampled = pool.sample(k=2)
    assert 1 <= len(sampled) <= 2


def test_kbest_save_and_load(tmp_path):
    pool = KBestOpponentPool(k=2, min_elo_diff=25.0)
    model = SiameseConvNetV1()

    class Dummy:
        def __init__(self, model, ep):
            self.model = model
            self.episode_count = ep

    pool.add_snapshot(Dummy(model, 100), rating=1275)

    path = tmp_path / "pool.pt"
    pool.save_pool(str(path))
    assert path.exists()

    new_pool = KBestOpponentPool(k=2)
    new_pool.load_pool(str(path), SiameseConvNetV1)

    stats = new_pool.get_pool_stats()
    assert stats["pool_size"] == 1
    assert stats["best_snapshot_elo"] == 1275
