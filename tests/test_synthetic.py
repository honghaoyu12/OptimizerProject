"""Tests for synthetic_datasets.py."""

import pytest
import torch
from torch.utils.data import DataLoader

from synthetic_datasets import (
    make_illcond_loaders,
    make_manifold_loaders,
    make_noisy_grad_loaders,
    make_saddle_loaders,
    make_sparse_loaders,
    make_checkerboard_loaders,
    make_plateau_loaders,
    make_correlated_loaders,
    make_imbalanced_loaders,
    SYNTHETIC_LOADERS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_MAKERS = [
    ("illcond",      make_illcond_loaders),
    ("sparse",       make_sparse_loaders),
    ("noisy_grad",   make_noisy_grad_loaders),
    ("manifold",     make_manifold_loaders),
    ("saddle",       make_saddle_loaders),
    ("checkerboard", make_checkerboard_loaders),
    ("plateau",      make_plateau_loaders),
    ("correlated",   make_correlated_loaders),
    ("imbalanced",   make_imbalanced_loaders),
]

EXPECTED_INPUT_SIZES = {
    "illcond":      64,
    "sparse":       100,
    "noisy_grad":   64,
    "manifold":     64,
    "saddle":       64,
    "checkerboard": 32,
    "plateau":      64,
    "correlated":   64,
    "imbalanced":   64,
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_all_keys_present(self):
        assert set(SYNTHETIC_LOADERS.keys()) == {
            "illcond", "sparse", "noisy_grad", "manifold", "saddle",
            "checkerboard", "plateau", "correlated", "imbalanced",
        }

    def test_all_values_callable(self):
        for key, fn in SYNTHETIC_LOADERS.items():
            assert callable(fn), f"SYNTHETIC_LOADERS['{key}'] is not callable"


# ---------------------------------------------------------------------------
# Per-dataset loader tests
# ---------------------------------------------------------------------------

class TestLoaderOutputs:
    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_returns_two_loaders(self, name, maker):
        result = maker(batch_size=64)
        assert isinstance(result, tuple) and len(result) == 2

    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_loaders_are_dataloaders(self, name, maker):
        train_loader, test_loader = maker(batch_size=64)
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader,  DataLoader)

    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_batch_feature_shape(self, name, maker):
        train_loader, _ = maker(batch_size=64)
        X_batch, y_batch = next(iter(train_loader))
        expected_d = EXPECTED_INPUT_SIZES[name]
        assert X_batch.ndim == 2,              f"{name}: expected 2-D features, got {X_batch.shape}"
        assert X_batch.shape[1] == expected_d, f"{name}: expected {expected_d} features, got {X_batch.shape[1]}"

    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_labels_are_binary(self, name, maker):
        train_loader, test_loader = maker(batch_size=512)
        for loader in (train_loader, test_loader):
            for _, y in loader:
                assert y.min() >= 0 and y.max() <= 1, \
                    f"{name}: labels out of range [0, 1]: min={y.min()}, max={y.max()}"

    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_no_nan_in_features(self, name, maker):
        train_loader, test_loader = maker(batch_size=512)
        for loader in (train_loader, test_loader):
            for X, _ in loader:
                assert not torch.isnan(X).any(), f"{name}: NaN found in features"
                assert not torch.isinf(X).any(), f"{name}: Inf found in features"

    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_train_larger_than_test(self, name, maker):
        train_loader, test_loader = maker(batch_size=512)
        n_train = sum(len(y) for _, y in train_loader)
        n_test  = sum(len(y) for _, y in test_loader)
        assert n_train > n_test, \
            f"{name}: expected more train samples than test, got {n_train} vs {n_test}"

    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_features_approximately_standardised(self, name, maker):
        """Training features should be close to zero mean and unit std after normalisation."""
        train_loader, _ = maker(batch_size=2048)
        all_X = torch.cat([X for X, _ in train_loader], dim=0)
        mean = all_X.mean(dim=0).abs().mean().item()
        std  = all_X.std(dim=0).mean().item()
        assert mean < 0.1, f"{name}: training mean not near zero: {mean:.4f}"
        assert 0.5 < std < 2.0, f"{name}: training std far from 1: {std:.4f}"

    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_reproducible_with_same_seed(self, name, maker):
        # Use the test loader (shuffle=False) so batch order is deterministic
        _, test1 = maker(batch_size=64, random_state=0)
        _, test2 = maker(batch_size=64, random_state=0)
        X1, _ = next(iter(test1))
        X2, _ = next(iter(test2))
        assert torch.allclose(X1, X2), f"{name}: not reproducible with same random_state"

    @pytest.mark.parametrize("name,maker", ALL_MAKERS)
    def test_different_seeds_differ(self, name, maker):
        _, test1 = maker(batch_size=64, random_state=0)
        _, test2 = maker(batch_size=64, random_state=99)
        X1, _ = next(iter(test1))
        X2, _ = next(iter(test2))
        assert not torch.allclose(X1, X2), f"{name}: different seeds produced identical data"


# ---------------------------------------------------------------------------
# Dataset-specific property tests
# ---------------------------------------------------------------------------

class TestIllcond:
    def test_both_classes_present(self):
        train_loader, _ = make_illcond_loaders(batch_size=2048)
        all_y = torch.cat([y for _, y in train_loader])
        assert set(all_y.tolist()) == {0, 1}


class TestSparse:
    def test_feature_count(self):
        train_loader, _ = make_sparse_loaders(batch_size=64)
        X, _ = next(iter(train_loader))
        assert X.shape[1] == 100


class TestNoisyGrad:
    def test_label_noise_present(self):
        """With 30% noise the Bayes error should be detectable: not all samples trivially correct."""
        train_loader, _ = make_noisy_grad_loaders(batch_size=2048)
        all_y = torch.cat([y for _, y in train_loader])
        # Both classes must appear (noise didn't collapse everything to one class)
        assert set(all_y.tolist()) == {0, 1}


class TestManifold:
    def test_embedding_dimension(self):
        train_loader, _ = make_manifold_loaders(batch_size=64)
        X, _ = next(iter(train_loader))
        assert X.shape[1] == 64   # 2 signal + 62 noise


class TestSaddle:
    def test_both_classes_present(self):
        train_loader, _ = make_saddle_loaders(batch_size=2048)
        all_y = torch.cat([y for _, y in train_loader])
        assert set(all_y.tolist()) == {0, 1}


class TestCheckerboard:
    def test_feature_dimension(self):
        train_loader, _ = make_checkerboard_loaders(batch_size=64)
        X, _ = next(iter(train_loader))
        assert X.shape[1] == 32   # 2 signal + 30 noise

    def test_both_classes_present(self):
        train_loader, _ = make_checkerboard_loaders(batch_size=2048)
        all_y = torch.cat([y for _, y in train_loader])
        assert set(all_y.tolist()) == {0, 1}

    def test_roughly_balanced(self):
        """Checkerboard should be roughly 50/50 by construction."""
        train_loader, _ = make_checkerboard_loaders(batch_size=2048)
        all_y = torch.cat([y for _, y in train_loader])
        frac = all_y.float().mean().item()
        assert 0.35 < frac < 0.65, f"Class balance {frac:.2f} too extreme"


class TestPlateau:
    def test_feature_dimension(self):
        train_loader, _ = make_plateau_loaders(batch_size=64)
        X, _ = next(iter(train_loader))
        assert X.shape[1] == 64

    def test_both_classes_present(self):
        train_loader, _ = make_plateau_loaders(batch_size=2048)
        all_y = torch.cat([y for _, y in train_loader])
        assert set(all_y.tolist()) == {0, 1}


class TestCorrelated:
    def test_feature_dimension(self):
        train_loader, _ = make_correlated_loaders(batch_size=64)
        X, _ = next(iter(train_loader))
        assert X.shape[1] == 64

    def test_high_pairwise_correlation(self):
        """Mean pairwise correlation between features should be high (> 0.7)."""
        train_loader, _ = make_correlated_loaders(batch_size=2048)
        all_X = torch.cat([X for X, _ in train_loader], dim=0)
        # Compute correlation matrix via standardised features
        X_std = (all_X - all_X.mean(0)) / (all_X.std(0) + 1e-8)
        # Sample a few feature pairs instead of full O(d²) computation
        corr_01 = (X_std[:, 0] * X_std[:, 1]).mean().item()
        corr_23 = (X_std[:, 2] * X_std[:, 3]).mean().item()
        assert corr_01 > 0.7, f"Expected high correlation, got {corr_01:.3f}"
        assert corr_23 > 0.7, f"Expected high correlation, got {corr_23:.3f}"


class TestImbalanced:
    def test_feature_dimension(self):
        train_loader, _ = make_imbalanced_loaders(batch_size=64)
        X, _ = next(iter(train_loader))
        assert X.shape[1] == 64

    def test_class_imbalance_ratio(self):
        """Minority class (1) should be roughly 9% of training data."""
        train_loader, _ = make_imbalanced_loaders(batch_size=2048)
        all_y = torch.cat([y for _, y in train_loader])
        minority_frac = all_y.float().mean().item()
        # 200 minority / 2200 total ≈ 0.091 (test split may shift this slightly)
        assert 0.05 < minority_frac < 0.20, \
            f"Expected ~9% minority, got {minority_frac:.2%}"

    def test_both_classes_present(self):
        train_loader, _ = make_imbalanced_loaders(batch_size=2048)
        all_y = torch.cat([y for _, y in train_loader])
        assert set(all_y.tolist()) == {0, 1}
