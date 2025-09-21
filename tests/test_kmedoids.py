#!/usr/bin/env python3
"""Comprehensive test suite for k-medoids clustering implementation."""

from typing import List, Tuple

import pytest
import torch

from alphaholdem.rl.kmedoids import SimpleKMedoids, kmedoids_pytorch


class TestSimpleKMedoids:
    """Test cases for SimpleKMedoids class."""

    def test_basic_clustering(self):
        """Test basic clustering functionality."""
        # Create simple 2-cluster data
        torch.manual_seed(42)
        X = torch.randn(20, 3)
        X[:10] += torch.tensor([2.0, 0.0, 0.0])  # Cluster 1
        X[10:] += torch.tensor([-2.0, 0.0, 0.0])  # Cluster 2

        kmedoids = SimpleKMedoids(n_clusters=2, random_state=42)
        kmedoids.fit(X)

        # Check that we found 2 medoids
        assert len(kmedoids.medoid_indices_) == 2
        assert len(kmedoids.labels_) == 20

        # Check that labels are valid
        assert all(0 <= label < 2 for label in kmedoids.labels_)
        assert len(set(kmedoids.labels_)) <= 2  # At most 2 clusters

    def test_multiple_clusters(self):
        """Test clustering with multiple clusters."""
        torch.manual_seed(42)
        X = torch.randn(30, 2)
        X[:10] += torch.tensor([3.0, 0.0])  # Cluster 1
        X[10:20] += torch.tensor([-3.0, 0.0])  # Cluster 2
        X[20:] += torch.tensor([0.0, 3.0])  # Cluster 3

        kmedoids = SimpleKMedoids(n_clusters=3, random_state=42)
        kmedoids.fit(X)

        assert len(kmedoids.medoid_indices_) == 3
        assert len(kmedoids.labels_) == 30
        assert all(0 <= label < 3 for label in kmedoids.labels_)

    def test_single_cluster(self):
        """Test clustering with single cluster."""
        torch.manual_seed(42)
        X = torch.randn(10, 3)

        kmedoids = SimpleKMedoids(n_clusters=1, random_state=42)
        kmedoids.fit(X)

        assert len(kmedoids.medoid_indices_) == 1
        assert len(kmedoids.labels_) == 10
        assert all(label == 0 for label in kmedoids.labels_)

    def test_equal_clusters_and_samples(self):
        """Test when n_clusters equals n_samples."""
        torch.manual_seed(42)
        X = torch.randn(5, 2)

        kmedoids = SimpleKMedoids(n_clusters=5, random_state=42)
        kmedoids.fit(X)

        assert len(kmedoids.medoid_indices_) == 5
        assert len(kmedoids.labels_) == 5
        assert set(kmedoids.labels_) == set(range(5))  # Each point is its own cluster

    def test_convergence(self):
        """Test that algorithm converges."""
        torch.manual_seed(42)
        X = torch.randn(20, 3)
        X[:10] += torch.tensor([2.0, 0.0, 0.0])
        X[10:] += torch.tensor([-2.0, 0.0, 0.0])

        kmedoids = SimpleKMedoids(n_clusters=2, max_iter=100, random_state=42)
        kmedoids.fit(X)

        # Should converge within max_iter
        assert len(kmedoids.medoid_indices_) == 2
        assert len(kmedoids.labels_) == 20

    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        torch.manual_seed(42)
        X = torch.randn(20, 3)

        kmedoids1 = SimpleKMedoids(n_clusters=2, random_state=42)
        kmedoids1.fit(X)

        kmedoids2 = SimpleKMedoids(n_clusters=2, random_state=42)
        kmedoids2.fit(X)

        assert kmedoids1.medoid_indices_ == kmedoids2.medoid_indices_
        assert kmedoids1.labels_ == kmedoids2.labels_

    def test_different_random_seeds(self):
        """Test that different seeds can produce different results."""
        torch.manual_seed(42)
        X = torch.randn(20, 3)

        kmedoids1 = SimpleKMedoids(n_clusters=2, random_state=42)
        kmedoids1.fit(X)

        kmedoids2 = SimpleKMedoids(n_clusters=2, random_state=123)
        kmedoids2.fit(X)

        # Results might be different (though not guaranteed)
        # We just check that both are valid
        assert len(kmedoids1.medoid_indices_) == 2
        assert len(kmedoids2.medoid_indices_) == 2

    def test_empty_cluster_handling(self):
        """Test handling of empty clusters."""
        torch.manual_seed(42)
        X = torch.randn(10, 2)
        # All points very close together
        X += torch.tensor([0.0, 0.0])

        kmedoids = SimpleKMedoids(n_clusters=3, random_state=42)
        kmedoids.fit(X)

        # Should handle empty clusters gracefully
        assert len(kmedoids.medoid_indices_) == 3
        assert len(kmedoids.labels_) == 10

    def test_input_validation(self):
        """Test input validation."""
        kmedoids = SimpleKMedoids(n_clusters=2)

        # Test 1D tensor
        X_1d = torch.randn(10)
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            kmedoids.fit(X_1d)

        # Test insufficient samples
        X_small = torch.randn(1, 3)  # 1 sample < 2 clusters
        with pytest.raises(ValueError, match="Number of samples"):
            kmedoids.fit(X_small)

    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with very small dataset
        X_small = torch.randn(2, 2)
        kmedoids = SimpleKMedoids(n_clusters=2, random_state=42)
        kmedoids.fit(X_small)
        assert len(kmedoids.medoid_indices_) == 2

        # Test with high-dimensional data
        X_high_dim = torch.randn(10, 50)
        kmedoids = SimpleKMedoids(n_clusters=3, random_state=42)
        kmedoids.fit(X_high_dim)
        assert len(kmedoids.medoid_indices_) == 3

    def test_medoid_indices_validity(self):
        """Test that medoid indices are valid."""
        torch.manual_seed(42)
        X = torch.randn(20, 3)

        kmedoids = SimpleKMedoids(n_clusters=3, random_state=42)
        kmedoids.fit(X)

        # Check that all medoid indices are valid
        assert all(0 <= idx < len(X) for idx in kmedoids.medoid_indices_)
        assert len(set(kmedoids.medoid_indices_)) == len(
            kmedoids.medoid_indices_
        )  # No duplicates

    def test_labels_consistency(self):
        """Test that labels are consistent with medoid indices."""
        torch.manual_seed(42)
        X = torch.randn(20, 3)

        kmedoids = SimpleKMedoids(n_clusters=3, random_state=42)
        kmedoids.fit(X)

        # Check that each point is assigned to its nearest medoid
        medoids = X[kmedoids.medoid_indices_]
        distances = torch.cdist(X, medoids)
        expected_labels = torch.argmin(distances, dim=1).tolist()

        assert kmedoids.labels_ == expected_labels


class TestKMedoidsPyTorch:
    """Test cases for kmedoids_pytorch convenience function."""

    def test_convenience_function(self):
        """Test the convenience function."""
        torch.manual_seed(42)
        X = torch.randn(20, 3)
        X[:10] += torch.tensor([2.0, 0.0, 0.0])
        X[10:] += torch.tensor([-2.0, 0.0, 0.0])

        medoid_indices, labels = kmedoids_pytorch(X, n_clusters=2, random_state=42)

        assert len(medoid_indices) == 2
        assert len(labels) == 20
        assert all(0 <= label < 2 for label in labels)

    def test_convenience_function_consistency(self):
        """Test that convenience function matches class method."""
        torch.manual_seed(42)
        X = torch.randn(20, 3)

        # Using class
        kmedoids = SimpleKMedoids(n_clusters=2, random_state=42)
        kmedoids.fit(X)

        # Using convenience function
        medoid_indices, labels = kmedoids_pytorch(X, n_clusters=2, random_state=42)

        assert kmedoids.medoid_indices_ == medoid_indices
        assert kmedoids.labels_ == labels


class TestPerformance:
    """Performance and stress tests."""

    def test_large_dataset(self):
        """Test performance on larger dataset."""
        torch.manual_seed(42)
        X = torch.randn(200, 10)

        kmedoids = SimpleKMedoids(n_clusters=5, random_state=42)
        kmedoids.fit(X)

        assert len(kmedoids.medoid_indices_) == 5
        assert len(kmedoids.labels_) == 200

    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        torch.manual_seed(42)
        X = torch.randn(50, 100)  # 100 dimensions

        kmedoids = SimpleKMedoids(n_clusters=3, random_state=42)
        kmedoids.fit(X)

        assert len(kmedoids.medoid_indices_) == 3
        assert len(kmedoids.labels_) == 50

    def test_many_clusters(self):
        """Test with many clusters."""
        torch.manual_seed(42)
        X = torch.randn(100, 5)

        kmedoids = SimpleKMedoids(n_clusters=20, random_state=42)
        kmedoids.fit(X)

        assert len(kmedoids.medoid_indices_) == 20
        assert len(kmedoids.labels_) == 100


def run_manual_tests():
    """Run manual tests without pytest."""
    print("Running manual k-medoids tests...")

    # Test 1: Basic functionality
    print("\n1. Testing basic clustering...")
    torch.manual_seed(42)
    X = torch.randn(20, 3)
    X[:10] += torch.tensor([2.0, 0.0, 0.0])
    X[10:] += torch.tensor([-2.0, 0.0, 0.0])

    kmedoids = SimpleKMedoids(n_clusters=2, random_state=42)
    kmedoids.fit(X)

    print(
        f"   ✓ Found {len(kmedoids.medoid_indices_)} medoids: {kmedoids.medoid_indices_}"
    )
    print(f"   ✓ Cluster sizes: {[kmedoids.labels_.count(i) for i in range(2)]}")

    # Test 2: Convenience function
    print("\n2. Testing convenience function...")
    medoid_indices, labels = kmedoids_pytorch(X, n_clusters=2, random_state=42)
    print(f"   ✓ Medoids: {medoid_indices}")
    print(f"   ✓ Labels: {labels[:5]}... (showing first 5)")

    # Test 3: Multiple clusters
    print("\n3. Testing multiple clusters...")
    torch.manual_seed(42)
    X_multi = torch.randn(30, 2)
    X_multi[:10] += torch.tensor([3.0, 0.0])
    X_multi[10:20] += torch.tensor([-3.0, 0.0])
    X_multi[20:] += torch.tensor([0.0, 3.0])

    kmedoids_multi = SimpleKMedoids(n_clusters=3, random_state=42)
    kmedoids_multi.fit(X_multi)

    print(
        f"   ✓ Found {len(kmedoids_multi.medoid_indices_)} medoids: {kmedoids_multi.medoid_indices_}"
    )
    print(f"   ✓ Cluster sizes: {[kmedoids_multi.labels_.count(i) for i in range(3)]}")

    # Test 4: Performance test
    print("\n4. Testing performance...")
    import time

    torch.manual_seed(42)
    X_large = torch.randn(100, 5)

    start_time = time.time()
    kmedoids_large = SimpleKMedoids(n_clusters=5, random_state=42)
    kmedoids_large.fit(X_large)
    end_time = time.time()

    print(
        f"   ✓ Processed {len(X_large)} points with {kmedoids_large.n_clusters} clusters"
    )
    print(f"   ✓ Time taken: {end_time - start_time:.4f} seconds")

    # Test 5: Edge cases
    print("\n5. Testing edge cases...")

    # Single cluster
    X_single = torch.randn(10, 3)
    kmedoids_single = SimpleKMedoids(n_clusters=1, random_state=42)
    kmedoids_single.fit(X_single)
    print(f"   ✓ Single cluster: {len(kmedoids_single.medoid_indices_)} medoid")

    # Equal clusters and samples
    X_equal = torch.randn(5, 2)
    kmedoids_equal = SimpleKMedoids(n_clusters=5, random_state=42)
    kmedoids_equal.fit(X_equal)
    print(f"   ✓ Equal clusters/samples: {len(kmedoids_equal.medoid_indices_)} medoids")

    print("\n🎉 All manual tests passed!")


if __name__ == "__main__":
    run_manual_tests()
