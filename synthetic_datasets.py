"""Synthetic datasets for probing optimizer behaviour.

Each generator returns (train_loader, test_loader) with features standardised
to zero mean and unit variance (fit on the training split only).

Datasets
--------
illcond     Ill-conditioned Gaussian classification — tests curvature handling
sparse      Sparse signal classification           — tests coordinate adaptivity
noisy_grad  Noisy-label classification             — tests stochastic robustness
manifold    Nonlinear manifold (make_moons + noise) — tests nonconvex landscapes
saddle      Bimodal saddle-point classification    — tests escape from flat regions
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    test_split: float = 0.2,
    random_state: int = 42,
):
    """Standardise, split train/test, and return DataLoaders."""
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_split)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit standardisation on training set only
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long(),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_illcond_loaders(batch_size: int = 128, random_state: int = 42):
    """Ill-conditioned Gaussian classification (condition number = 1 000).

    Features are sampled from a multivariate Gaussian whose covariance matrix
    has eigenvalues log-spaced from 1 to 1 000, creating extreme curvature
    differences across directions.  Optimizers with curvature awareness
    (Adam, Shampoo) should outperform plain SGD.

    Shape : (n=2 000, d=64)   Classes : 2
    """
    rng = np.random.RandomState(random_state)
    n, d, kappa = 2_000, 64, 1_000

    eigvals = np.logspace(0, np.log10(kappa), d)
    Q, _    = np.linalg.qr(rng.randn(d, d))
    cov     = (Q * eigvals) @ Q.T

    X = rng.multivariate_normal(np.zeros(d), cov, n).astype(np.float32)
    w = rng.randn(d).astype(np.float32)
    y = ((X @ w + 0.1 * rng.randn(n)) > 0).astype(np.int64)

    return _to_loaders(X, y, batch_size=batch_size, random_state=random_state)


def make_sparse_loaders(batch_size: int = 128, random_state: int = 42):
    """Sparse signal classification (5 informative out of 100 features).

    Only 5 features carry signal; the remaining 95 are pure noise.  Adaptive
    per-coordinate optimizers (Adagrad, Adam) should adapt faster than SGD.

    Shape : (n=2 000, d=100)   Classes : 2
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2_000,
        n_features=100,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    return _to_loaders(
        X.astype(np.float32), y.astype(np.int64),
        batch_size=batch_size, random_state=random_state,
    )


def make_noisy_grad_loaders(batch_size: int = 128, random_state: int = 42):
    """Noisy-label classification (30 % random label flips).

    High gradient noise from mislabelled examples.  Optimizers with adaptive
    momentum or variance reduction (Adam, NAdam) should be more robust than
    pure SGD.

    Shape : (n=2 000, d=64)   Classes : 2
    """
    from sklearn.datasets import make_classification

    rng  = np.random.RandomState(random_state)
    X, y = make_classification(
        n_samples=2_000,
        n_features=64,
        n_informative=32,
        n_redundant=16,
        random_state=random_state,
    )
    flip    = rng.rand(len(y)) < 0.3
    y[flip] = 1 - y[flip]

    return _to_loaders(
        X.astype(np.float32), y.astype(np.int64),
        batch_size=batch_size, random_state=random_state,
    )


def make_manifold_loaders(batch_size: int = 128, random_state: int = 42):
    """Nonlinear manifold classification (make_moons embedded in 64-D).

    The two classes lie on interleaving half-moons in 2-D; 62 noise dimensions
    are added so the MLP must discover the low-dimensional structure.  Tests
    optimizers on nonconvex, curved decision boundaries.

    Shape : (n=2 000, d=64)   Classes : 2
    """
    from sklearn.datasets import make_moons

    rng         = np.random.RandomState(random_state)
    X_2d, y    = make_moons(n_samples=2_000, noise=0.1, random_state=random_state)
    noise_dims  = rng.randn(len(X_2d), 62) * 0.1
    X           = np.hstack([X_2d, noise_dims]).astype(np.float32)

    return _to_loaders(X, y.astype(np.int64),
                       batch_size=batch_size, random_state=random_state)


def make_saddle_loaders(batch_size: int = 128, random_state: int = 42):
    """Saddle-point landscape classification.

    Positive class is bimodal (two clusters at ±2), negative class is
    unimodal near the origin.  The symmetric structure creates saddle points
    in the loss surface.  Optimizers that use momentum or second-order
    information (Adam, Shampoo) should escape faster.

    Shape : (n=2 000, d=64)   Classes : 2
    """
    rng    = np.random.RandomState(random_state)
    n, d   = 2_000, 64
    n_pos  = n // 2
    n_neg  = n - n_pos

    # Positive: two clusters far from origin (bimodal)
    X_pos = np.vstack([
        rng.randn(n_pos // 2,        d) + 2.0,
        rng.randn(n_pos - n_pos // 2, d) - 2.0,
    ])
    # Negative: tight cluster at origin (sits at the saddle)
    X_neg = rng.randn(n_neg, d) * 0.3

    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

    perm = rng.permutation(n)
    return _to_loaders(X[perm], y[perm],
                       batch_size=batch_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Registry (mirrors train.py pattern for easy lookup)
# ---------------------------------------------------------------------------

SYNTHETIC_LOADERS = {
    "illcond":    make_illcond_loaders,
    "sparse":     make_sparse_loaders,
    "noisy_grad": make_noisy_grad_loaders,
    "manifold":   make_manifold_loaders,
    "saddle":     make_saddle_loaders,
}
