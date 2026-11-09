"""
Imbalance handling module for highly skewed fraud data.
Implements:
1. Class Weighting (cost-sensitive scaling)
2. Random Undersampling of majority class
3. SMOTE (Synthetic Minority Over-sampling Technique)
All resampling methods operate strictly on training folds to prevent data leakage.
"""

from typing import Tuple
import numpy as np


class ImbalanceHandler:
    @staticmethod
    def compute_class_weights(y: np.ndarray) -> Tuple[float, float]:
        """Returns (weight_neg, weight_pos) where weight_pos scales with minority scarcity."""
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        assert n_pos > 0, "No positive instances found."
        weight_pos = n_neg / n_pos
        weight_neg = 1.0
        return weight_neg, weight_pos

    @staticmethod
    def random_undersample(
        X: np.ndarray,
        y: np.ndarray,
        ratio: float = 0.1,  # n_pos / n_neg target ratio (e.g. 10:1 neg to pos)
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undersamples the majority (class 0) instances to achieve target ratio.
        """
        rng = np.random.RandomState(random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        n_pos = len(pos_idx)
        n_neg_target = int(n_pos / ratio)
        n_neg_target = min(len(neg_idx), max(n_pos, n_neg_target))
        
        sampled_neg_idx = rng.choice(neg_idx, size=n_neg_target, replace=False)
        selected_idx = np.concatenate([pos_idx, sampled_neg_idx])
        rng.shuffle(selected_idx)
        
        return X[selected_idx], y[selected_idx]

    @staticmethod
    def smote(
        X: np.ndarray,
        y: np.ndarray,
        k_neighbors: int = 5,
        target_ratio: float = 0.2,  # minority to majority ratio after synthesis
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized SMOTE implementation in pure NumPy.
        Generates synthetic minority instances along line segments connecting k-nearest neighbors.
        """
        rng = np.random.RandomState(random_state)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        n_pos = len(pos_idx)
        n_neg = len(neg_idx)
        
        n_pos_target = int(n_neg * target_ratio)
        n_synthetic_needed = max(0, n_pos_target - n_pos)
        
        if n_synthetic_needed == 0 or n_pos < 2:
            return X.copy(), y.copy()
            
        X_pos = X[pos_idx]
        k = min(k_neighbors, n_pos - 1)
        
        # Compute pairwise Euclidean distances among positive instances
        dists = np.linalg.norm(X_pos[:, np.newaxis, :] - X_pos[np.newaxis, :, :], axis=2)
        # Exclude self distance
        np.fill_diagonal(dists, np.inf)
        # Find k nearest neighbors for each positive instance
        nn_indices = np.argsort(dists, axis=1)[:, :k]
        
        # Generate synthetic samples
        synthetic_samples = []
        for _ in range(n_synthetic_needed):
            base_idx = rng.randint(0, n_pos)
            neighbor_choice = rng.randint(0, k)
            neighbor_idx = nn_indices[base_idx, neighbor_choice]
            
            diff = X_pos[neighbor_idx] - X_pos[base_idx]
            gap = rng.uniform(0.0, 1.0)
            synthetic = X_pos[base_idx] + gap * diff
            synthetic_samples.append(synthetic)
            
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.ones(n_synthetic_needed, dtype=int)
        
        X_resampled = np.vstack([X, X_synthetic])
        y_resampled = np.concatenate([y, y_synthetic])
        
        # Shuffle combined dataset
        perm = rng.permutation(len(y_resampled))
        return X_resampled[perm], y_resampled[perm]
