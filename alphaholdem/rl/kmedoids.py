"""Simple k-medoids clustering implementation using PyTorch.

This provides a lightweight alternative to sklearn's KMedoids for use in DREDPool.
"""

import torch
from typing import List, Tuple


class SimpleKMedoids:
    """
    Simple k-medoids clustering implementation.

    Uses the PAM (Partitioning Around Medoids) algorithm with PyTorch tensors.
    """

    def __init__(self, n_clusters: int, max_iter: int = 100, random_state: int = 0):
        """
        Initialize k-medoids clustering.

        Args:
            n_clusters: Number of clusters to find
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.medoid_indices_: List[int] = []
        self.labels_: List[int] = []

    def fit(self, X: torch.Tensor) -> "SimpleKMedoids":
        """
        Fit k-medoids clustering to data.

        Args:
            X: Input data tensor of shape (n_samples, n_features)

        Returns:
            Self for method chaining
        """
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {X.dim()}D")

        n_samples, n_features = X.shape

        if n_samples < self.n_clusters:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= n_clusters ({self.n_clusters})"
            )

        # Set random seed
        torch.manual_seed(self.random_state)

        # Initialize medoids randomly
        medoid_indices = torch.randperm(n_samples)[: self.n_clusters].tolist()

        # PAM algorithm
        for iteration in range(self.max_iter):
            # Assign each point to nearest medoid
            labels = self._assign_labels(X, medoid_indices)

            # Update medoids
            new_medoid_indices = self._update_medoids(X, labels)

            # Check convergence
            if set(new_medoid_indices) == set(medoid_indices):
                break

            medoid_indices = new_medoid_indices

        self.medoid_indices_ = medoid_indices
        self.labels_ = self._assign_labels(X, medoid_indices)

        return self

    def _assign_labels(self, X: torch.Tensor, medoid_indices: List[int]) -> List[int]:
        """Assign each point to the nearest medoid."""
        medoids = X[medoid_indices]  # Shape: (n_clusters, n_features)

        # Compute distances from each point to each medoid
        distances = torch.cdist(X, medoids)  # Shape: (n_samples, n_clusters)

        # Assign to nearest medoid
        labels = torch.argmin(distances, dim=1).tolist()

        return labels

    def _update_medoids(self, X: torch.Tensor, labels: List[int]) -> List[int]:
        """Update medoids to minimize total cost within each cluster (fully vectorized)."""
        labels_tensor = torch.tensor(labels)
        new_medoid_indices = []

        for cluster_id in range(self.n_clusters):
            # Find points in this cluster
            cluster_mask = labels_tensor == cluster_id
            cluster_indices = torch.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                # Empty cluster, keep current medoid
                current_medoid = torch.where(labels_tensor == cluster_id)[0]
                if len(current_medoid) > 0:
                    new_medoid_indices.append(current_medoid[0].item())
                continue

            cluster_points = X[cluster_indices]

            # Fully vectorized: compute all pairwise distances within cluster
            # Shape: (cluster_size, cluster_size)
            pairwise_distances = torch.cdist(cluster_points, cluster_points)

            # Sum of distances for each point as potential medoid
            # Shape: (cluster_size,)
            total_costs = pairwise_distances.sum(dim=1)

            # Find the point with minimum total cost
            best_idx = torch.argmin(total_costs)
            best_medoid_idx = cluster_indices[best_idx].item()

            new_medoid_indices.append(best_medoid_idx)

        return new_medoid_indices


def kmedoids_pytorch(
    X: torch.Tensor, n_clusters: int, max_iter: int = 100, random_state: int = 0
) -> Tuple[List[int], List[int]]:
    """
    Convenience function for k-medoids clustering.

    Args:
        X: Input data tensor of shape (n_samples, n_features)
        n_clusters: Number of clusters to find
        max_iter: Maximum number of iterations
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (medoid_indices, labels)
    """
    kmedoids = SimpleKMedoids(
        n_clusters=n_clusters, max_iter=max_iter, random_state=random_state
    )
    kmedoids.fit(X)
    return kmedoids.medoid_indices_, kmedoids.labels_
