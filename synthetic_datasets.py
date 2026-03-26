"""Synthetic tabular datasets for probing optimizer behaviour on controlled tasks.

Unlike real image datasets, these synthetic generators let us *design* the
difficulty: we can set the exact condition number, noise level, and geometry
of the loss surface and directly test whether each optimizer handles the
corresponding challenge.

Each generator function returns (train_loader, test_loader) of feature tensors
that have been standardised to zero mean / unit variance (fit on the training
split only, to avoid data leakage).  All datasets produce a binary
classification task and use MLP models — ResNet/ViT are not meaningful here.

Dataset catalogue
-----------------
illcond     Ill-conditioned Gaussian — condition number 1 000.
            Tests curvature-aware optimizers (Adam, Shampoo) vs. plain SGD.

sparse      Sparse signal (5 of 100 features informative).
            Tests per-coordinate adaptivity: Adagrad / Adam find the few
            informative directions while SGD spreads step budget uniformly.

noisy_grad  30 % random label flips introduce high gradient variance.
            Tests momentum / variance reduction (Adam, NAdam) vs. vanilla SGD.

manifold    Interleaving half-moons embedded in 64-D (62 noise dims).
            Tests handling of nonconvex decision boundaries.

saddle      Bimodal positive class vs. unimodal negative at origin.
            Symmetric structure creates saddle points; tests momentum-based
            saddle escape (Adam, SGD+Momentum vs. ScheduleFreeAdamW, LAMB).

checkerboard 2-D checkerboard pattern (4×4 grid) embedded in 32-D.
            Highly nonlinear XOR-like decision boundary; tests whether
            optimizers can drive deep enough representation learning.

plateau     Wide flat loss region followed by a sharp drop.
            Gradient magnitude is near-zero in the plateau and large on the
            cliff edge; tests optimizers that depend on gradient magnitude
            (sign-based methods like Lion/Tiger fare well).

correlated  Highly correlated features (ρ ≈ 0.95 between all pairs).
            Complements illcond with a different kind of ill-conditioning:
            many near-duplicate directions rather than extreme scale spread.

imbalanced  10:1 class imbalance (binary).
            Reveals whether optimizers treat the minority class differently;
            adaptive methods tend to assign larger steps to the rare-class
            gradient signals.

All nine are registered in SYNTHETIC_LOADERS for easy lookup by name.
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
) -> tuple[DataLoader, DataLoader]:
    """Standardise, split train/test, wrap in DataLoaders.

    Standardisation (z-scoring) is fit on the training split *only* and
    applied to both splits.  This prevents information from the test set
    from leaking into the normalisation statistics — the same rule that
    scikit-learn's Pipeline enforces.

    Parameters
    ----------
    X            : (n, d) float32 feature matrix
    y            : (n,) int64 label vector
    batch_size   : mini-batch size (default 128)
    test_split   : fraction of data held out for testing (default 0.2)
    random_state : seed for the shuffled train/test split (default 42)

    Returns
    -------
    (train_loader, test_loader)
    """
    rng = np.random.RandomState(random_state)
    # Shuffle all indices, then take the first n_test as the test set
    idx    = rng.permutation(len(X))
    n_test = int(len(X) * test_split)
    test_idx,  train_idx  = idx[:n_test], idx[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit mean/std on training split — +1e-8 guards against constant features
    mean   = X_train.mean(axis=0)               # shape (d,)
    std    = X_train.std(axis=0) + 1e-8          # shape (d,); +1e-8 avoids /0
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std             # use TRAIN statistics here

    # Wrap numpy arrays in TensorDataset then DataLoader
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).long(),
    )
    test_ds  = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test).long(),
    )

    # shuffle=True for training (stochastic gradient), False for test evaluation
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------

def make_illcond_loaders(batch_size: int = 128, random_state: int = 42):
    """Ill-conditioned Gaussian classification (condition number κ = 1 000).

    Generates a binary classification task whose feature covariance has
    eigenvalues log-spaced from 1 to 1 000.  This creates extreme curvature
    differences across directions in parameter space — a small step in the
    high-curvature direction can overshoot, while an equally small step in
    the low-curvature direction makes almost no progress.

    WHY log-spacing: gives equal representation to all orders of magnitude
    of curvature (1, 10, 100, 1000) — a more realistic model of neural
    network loss surfaces than uniform spacing.

    Expected behaviour:
      SGD (plain)     — converges slowly; optimal step size is 1/κ ≈ 1e-3
      SGD+Momentum    — modest improvement via acceleration
      Adam / Shampoo  — adapt per-coordinate; converge much faster

    Shape : (n=2 000, d=64)   Classes : 2
    """
    rng     = np.random.RandomState(random_state)
    n, d, kappa = 2_000, 64, 1_000

    # Build covariance: Q diag(λ) Q^T where λ are log-spaced from 1 to κ
    eigvals = np.logspace(0, np.log10(kappa), d)   # shape (d,); λ₁=1, λ_d=1000
    Q, _    = np.linalg.qr(rng.randn(d, d))        # random orthonormal basis
    cov     = (Q * eigvals) @ Q.T                   # (Q diag(λ) Q^T): shape (d,d)

    # Sample features from the ill-conditioned Gaussian
    X = rng.multivariate_normal(np.zeros(d), cov, n).astype(np.float32)

    # Linear label rule: y = sign(w^T x + noise) where w is a random direction
    w = rng.randn(d).astype(np.float32)
    y = ((X @ w + 0.1 * rng.randn(n)) > 0).astype(np.int64)   # binary {0,1}

    return _to_loaders(X, y, batch_size=batch_size, random_state=random_state)


def make_sparse_loaders(batch_size: int = 128, random_state: int = 42):
    """Sparse signal classification (5 informative out of 100 features).

    Only 5 features carry class information; the remaining 95 are pure i.i.d.
    Gaussian noise.  An optimizer that treats all coordinates equally (SGD)
    must spread its step budget across all 100 directions, while adaptive
    methods (Adagrad, Adam) allocate more trust to the high-gradient (informative)
    coordinates and less to the noisy ones.

    WHY sklearn.make_classification: well-tested generator that controls
    n_informative precisely; it also handles cluster placement and scaling.

    Shape : (n=2 000, d=100)   Classes : 2
    """
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=2_000,
        n_features=100,
        n_informative=5,      # only 5 features matter
        n_redundant=0,        # no derived features
        n_repeated=0,         # no duplicate features
        n_clusters_per_class=1,
        random_state=random_state,
    )
    return _to_loaders(
        X.astype(np.float32), y.astype(np.int64),
        batch_size=batch_size, random_state=random_state,
    )


def make_noisy_grad_loaders(batch_size: int = 128, random_state: int = 42):
    """Noisy-label classification (30 % uniform random label flips).

    30 % of training labels are randomly flipped, injecting a large constant
    variance into every gradient estimate regardless of batch size.  Optimizers
    with adaptive momentum (Adam's β₂ moment) or explicit variance reduction
    are expected to navigate toward the true signal despite the noise.

    WHY 30 %: noise below ~20 % has little effect on well-tuned SGD; above
    ~40 % the task becomes unlearnable.  30 % is a challenging but realistic
    level (e.g. labelling disagreements in crowdsourced annotation).

    Shape : (n=2 000, d=64)   Classes : 2
    """
    from sklearn.datasets import make_classification

    rng  = np.random.RandomState(random_state)
    X, y = make_classification(
        n_samples=2_000,
        n_features=64,
        n_informative=32,   # more informative features than sparse — harder for noise to dominate
        n_redundant=16,     # correlated noise dimensions
        random_state=random_state,
    )

    # Flip 30 % of labels: for binary {0,1} flipping = 1 - y
    flip    = rng.rand(len(y)) < 0.3     # boolean mask, ~30 % True
    y[flip] = 1 - y[flip]                # 0→1 and 1→0 for selected indices

    return _to_loaders(
        X.astype(np.float32), y.astype(np.int64),
        batch_size=batch_size, random_state=random_state,
    )


def make_manifold_loaders(batch_size: int = 128, random_state: int = 42):
    """Nonlinear manifold classification (make_moons embedded in 64-D).

    The two classes form interleaving half-moons in 2-D (a nonlinearly
    separable pattern that no linear classifier can solve).  62 additional
    noise dimensions are concatenated so the MLP must discover the 2-D
    structure from 64 features — a toy version of the manifold hypothesis
    in representation learning.

    WHY noise=0.1: a small amount of class overlap makes the task harder
    than a perfectly clean moon (which an MLP can memorise in one epoch).
    WHY 62 noise dimensions at scale 0.1: keeps the noise-to-signal ratio
    moderate while ensuring the informative dimensions are not trivially
    identified by their variance.

    Shape : (n=2 000, d=64)   Classes : 2
    """
    from sklearn.datasets import make_moons

    rng        = np.random.RandomState(random_state)
    X_2d, y   = make_moons(n_samples=2_000, noise=0.1, random_state=random_state)
    # X_2d: (2000, 2); append 62 columns of Gaussian noise scaled to 0.1
    noise_dims = rng.randn(len(X_2d), 62) * 0.1    # (2000, 62)
    X          = np.hstack([X_2d, noise_dims]).astype(np.float32)  # (2000, 64)

    return _to_loaders(X, y.astype(np.int64),
                       batch_size=batch_size, random_state=random_state)


def make_saddle_loaders(batch_size: int = 128, random_state: int = 42):
    """Saddle-point landscape classification.

    Class 1 (positive) is bimodal: two equal-sized clusters at ±2 along all
    dimensions.  Class 0 (negative) is a tight unimodal cluster near the
    origin.  The symmetric bimodal structure means the gradient along the
    axis connecting the two positive clusters is near zero at the centroid,
    creating a saddle-like region in the loss surface.

    WHY ±2 for positive class: far enough from origin (where the negative
    cluster sits) that the classes are separable, but the symmetry of the
    two positive clusters creates a degenerate gradient near the midpoint.

    Expected behaviour:
      SGD (no momentum) — may stall at the saddle (gradient is ≈0)
      SGD+Momentum / Adam — escape via accumulated momentum
      Shampoo — second-order info helps detect and escape the flat region

    Shape : (n=2 000, d=64)   Classes : 2
    """
    rng   = np.random.RandomState(random_state)
    n, d  = 2_000, 64
    n_pos = n // 2      # 1000 positive examples
    n_neg = n - n_pos   # 1000 negative examples

    # Positive class: two clusters of size n_pos//2 at +2 and -2 (bimodal)
    X_pos = np.vstack([
        rng.randn(n_pos // 2,         d) + 2.0,   # cluster near +2 in all dims
        rng.randn(n_pos - n_pos // 2, d) - 2.0,   # cluster near -2 in all dims
    ])
    # Negative class: tight unimodal cluster at origin (scale 0.3 ≪ ±2)
    X_neg = rng.randn(n_neg, d) * 0.3

    # Concatenate and assign labels (1 = positive, 0 = negative)
    X = np.vstack([X_pos, X_neg]).astype(np.float32)
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.int64)

    # Shuffle so that class 1 and 0 are interleaved in the loaders
    perm = rng.permutation(n)
    return _to_loaders(X[perm], y[perm],
                       batch_size=batch_size, random_state=random_state)


def make_checkerboard_loaders(batch_size: int = 128, random_state: int = 42):
    """Checkerboard classification (4×4 grid in 2-D, embedded in 32-D).

    Points are drawn uniformly from [-4, 4]² and labelled by a 4×4
    checkerboard pattern (XOR of which grid cell each point falls into).
    30 additional Gaussian noise dimensions are appended so the network
    must discover the relevant 2-D subspace.

    This is a highly nonlinear, multi-modal decision boundary — no shallow
    linear or quadratic classifier can solve it.  Optimizers that drive
    deep representation learning (Adam, Adan) should outperform those that
    rely on gradient magnitude scaling alone (SignSGD, Lion).

    WHY 4×4 grid: 16 cells, alternating labels — hard enough that a 2-layer
    MLP needs many epochs but easy enough that convergence is detectable
    within a short benchmark run.

    Shape : (n=2 000, d=32)   Classes : 2
    """
    rng = np.random.RandomState(random_state)
    n   = 2_000
    # 2-D coordinates in [-4, 4]
    xy  = rng.uniform(-4.0, 4.0, size=(n, 2)).astype(np.float32)
    # Cell index along each axis: 0..3 (4 cells per axis, each width 2)
    cell_x = np.floor((xy[:, 0] + 4.0) / 2.0).astype(int).clip(0, 3)
    cell_y = np.floor((xy[:, 1] + 4.0) / 2.0).astype(int).clip(0, 3)
    # Checkerboard label: XOR of the two cell indices (even XOR = 0, odd XOR = 1)
    y = ((cell_x + cell_y) % 2).astype(np.int64)
    # Append 30 Gaussian noise dimensions (scaled small so signal dominates)
    noise = rng.randn(n, 30).astype(np.float32) * 0.5
    X = np.hstack([xy, noise])  # (n, 32)

    return _to_loaders(X, y, batch_size=batch_size, random_state=random_state)


def make_plateau_loaders(batch_size: int = 128, random_state: int = 42):
    """Plateau loss-landscape classification.

    The two classes are separated by a wide, nearly-flat sigmoid transition:
    class 1 when a linear score s > threshold, class 0 otherwise, where the
    score is fed through a very flat sigmoid (temperature τ = 0.05) before
    the label is drawn.  This means most of the input space lies in the
    plateau region where the expected gradient is near zero, with a sharp
    cliff at the decision boundary.

    Optimizers that rely on gradient magnitude (Adagrad, RMSprop accumulating
    small gradients) slow down severely in the plateau; sign-based methods
    (Lion, Tiger, SignSGD) maintain a fixed step size and can cross it
    faster.  Adam's second-moment normalisation also helps escape plateaus.

    WHY temperature 0.05: creates an extremely flat sigmoid
    (≈ a step function), so 95%+ of samples land in near-zero gradient
    territory.  Lowering it further makes the task nearly unlearnable.

    Shape : (n=2 000, d=64)   Classes : 2
    """
    rng = np.random.RandomState(random_state)
    n, d = 2_000, 64

    X = rng.randn(n, d).astype(np.float32)
    # Linear scoring direction (random unit vector)
    w = rng.randn(d).astype(np.float32)
    w /= np.linalg.norm(w) + 1e-8
    scores = X @ w  # (n,), mean ≈ 0, std ≈ 1

    # Very-flat sigmoid probability for class 1
    tau = 0.05   # low temperature → near-step function → wide plateau
    prob_pos = 1.0 / (1.0 + np.exp(-scores / tau))
    y = (rng.rand(n) < prob_pos).astype(np.int64)

    return _to_loaders(X, y, batch_size=batch_size, random_state=random_state)


def make_correlated_loaders(batch_size: int = 128, random_state: int = 42):
    """Highly correlated features classification (ρ ≈ 0.95 between all pairs).

    All 64 features share a single latent factor z ~ N(0,1) plus small
    independent noise: x_i = z + 0.23·ε_i.  This produces pairwise
    correlations of ≈ 1/(1 + 0.23²) ≈ 0.95.  The label is determined by z,
    so in principle one feature is sufficient — but the optimizer must
    discover this from a highly redundant, near-rank-1 covariance.

    This differs from illcond (scale spread) by testing a different
    structural challenge: many near-duplicate gradient directions rather than
    extreme scale differences.  Second-order methods (Shampoo, SOAP) and
    methods with per-coordinate adaptivity (Adam) should identify the shared
    factor faster than SGD, which diffuses the step across all 64 directions.

    WHY ρ = 0.95: high enough that naive PCA-like reasoning fails within a
    few epochs, but low enough that the task remains solvable by all
    optimizers given enough time.

    Shape : (n=2 000, d=64)   Classes : 2
    """
    rng    = np.random.RandomState(random_state)
    n, d   = 2_000, 64
    noise_scale = 0.23   # gives pairwise ρ ≈ 0.95

    # Shared latent factor z
    z = rng.randn(n).astype(np.float32)           # (n,)
    # Independent per-feature noise
    eps = rng.randn(n, d).astype(np.float32) * noise_scale
    X = z[:, None] + eps                           # (n, d); broadcast z

    # Label: class 1 when z > 0 (the latent factor is above median)
    y = (z > 0).astype(np.int64)

    return _to_loaders(X, y, batch_size=batch_size, random_state=random_state)


def make_imbalanced_loaders(batch_size: int = 128, random_state: int = 42):
    """Class-imbalanced binary classification (10:1 majority-to-minority ratio).

    The minority class (class 1) comprises only ~9% of samples; the majority
    class (class 0) comprises ~91%.  Both classes are drawn from isotropic
    Gaussians with different means so the task is learnable, but the gradient
    signal from the minority class is swamped by the majority class in each
    mini-batch.

    Adaptive methods (Adam, Adagrad) implicitly upweight rare-class updates
    because their per-coordinate second moments track gradient frequency —
    infrequent minority-class gradients get divided by a smaller v_t, giving
    them a relatively larger effective step.  Plain SGD has no such mechanism.

    WHY 10:1: a common real-world ratio (fraud detection, medical diagnosis).
    More extreme ratios (100:1) make the task nearly degenerate without
    specialised loss functions; 10:1 is challenging but learnable.

    Shape : (n=2 200, d=64)   Classes : 2  (2 000 majority + 200 minority)
    """
    rng    = np.random.RandomState(random_state)
    d      = 64
    n_maj  = 2_000   # majority class (class 0)
    n_min  = 200     # minority class (class 1) — 10:1 ratio

    # Majority class: isotropic Gaussian centred at origin
    X_maj = rng.randn(n_maj, d).astype(np.float32)
    y_maj = np.zeros(n_maj, dtype=np.int64)

    # Minority class: shifted by 2 along a random direction
    direction = rng.randn(d).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    X_min = rng.randn(n_min, d).astype(np.float32) + 2.0 * direction
    y_min = np.ones(n_min, dtype=np.int64)

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([y_maj, y_min])

    # Shuffle before splitting so both classes appear in train and test
    perm = rng.permutation(len(X))
    return _to_loaders(X[perm], y[perm],
                       batch_size=batch_size, random_state=random_state)


# ---------------------------------------------------------------------------
# Registry (mirrors train.py pattern for easy lookup by name)
# ---------------------------------------------------------------------------

# Maps the dataset name used in DATASET_INFO and on the command line to the
# corresponding loader factory function.  Add new synthetic datasets here.
SYNTHETIC_LOADERS: dict = {
    "illcond":      make_illcond_loaders,
    "sparse":       make_sparse_loaders,
    "noisy_grad":   make_noisy_grad_loaders,
    "manifold":     make_manifold_loaders,
    "saddle":       make_saddle_loaders,
    "checkerboard": make_checkerboard_loaders,
    "plateau":      make_plateau_loaders,
    "correlated":   make_correlated_loaders,
    "imbalanced":   make_imbalanced_loaders,
}
