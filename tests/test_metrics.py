"""Tests for metrics.py — Hessian trace, sharpness, dead neurons, weight rank,
gradient noise scale, and Fisher trace estimators."""

import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from metrics import (compute_hessian_trace, compute_sharpness,
                     compute_dead_neurons, compute_weight_rank,
                     compute_gradient_noise_scale, compute_fisher_trace)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_model_and_batch():
    """Return a tiny MLP, a small batch, and a criterion."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    images = torch.randn(16, 1, 4, 4)   # 16 samples, 1×4×4
    labels = torch.randint(0, 4, (16,))
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    return model, (images, labels), criterion, device


def tiny_loader(n=64, in_features=16, num_classes=4, batch_size=16):
    """Return a simple DataLoader over random tabular data."""
    X = torch.randn(n, 1, 4, 4)   # image-style for compatibility with tiny_model
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Hessian trace
# ---------------------------------------------------------------------------

class TestHessianTrace:
    def test_returns_float(self):
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_hessian_trace(model, batch, criterion, device)
        assert isinstance(result, float)

    def test_finite_on_valid_model(self):
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_hessian_trace(model, batch, criterion, device, n_samples=2)
        assert math.isfinite(result)

    def test_positive(self):
        # Hessian trace of a convex loss should be positive
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_hessian_trace(model, batch, criterion, device, n_samples=5)
        assert result > 0

    def test_nan_on_broken_model(self):
        # A model with no parameters should return nan gracefully
        model = nn.Identity()
        batch = (torch.randn(4, 4), torch.randint(0, 2, (4,)))
        criterion = nn.CrossEntropyLoss()
        result = compute_hessian_trace(model, batch, criterion, torch.device("cpu"))
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------

class TestSharpness:
    def test_returns_float(self):
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_sharpness(model, batch, criterion, device)
        assert isinstance(result, float)

    def test_non_negative(self):
        # Sharpness is max loss increase — should be >= 0
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_sharpness(model, batch, criterion, device, n_directions=3)
        assert result >= 0.0

    def test_finite(self):
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_sharpness(model, batch, criterion, device)
        assert math.isfinite(result)

    def test_weights_restored(self):
        """Weights must be identical before and after compute_sharpness."""
        model, batch, criterion, device = tiny_model_and_batch()
        weights_before = [p.data.clone() for p in model.parameters()]
        compute_sharpness(model, batch, criterion, device, n_directions=3)
        for before, p in zip(weights_before, model.parameters()):
            assert torch.allclose(before, p.data), "Weights were not restored"

    def test_larger_epsilon_increases_sharpness(self):
        """Bigger perturbation radius should generally yield equal or larger sharpness."""
        model, batch, criterion, device = tiny_model_and_batch()
        torch.manual_seed(0)
        small = compute_sharpness(model, batch, criterion, device, epsilon=1e-4, n_directions=10)
        torch.manual_seed(0)
        large = compute_sharpness(model, batch, criterion, device, epsilon=1.0, n_directions=10)
        assert large >= small


# ---------------------------------------------------------------------------
# Dead neurons
# ---------------------------------------------------------------------------

class TestDeadNeurons:
    def test_returns_dict(self):
        model, _, _, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, device)
        assert isinstance(result, dict)

    def test_keys_are_relu_layers(self):
        """Keys should correspond to ReLU module names in the model."""
        model, _, _, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, device)
        relu_names = {name for name, mod in model.named_modules()
                      if isinstance(mod, nn.ReLU)}
        assert set(result.keys()) == relu_names

    def test_fractions_in_range(self):
        """Dead fractions must be in [0, 1]."""
        model, _, _, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, device)
        for name, frac in result.items():
            assert 0.0 <= frac <= 1.0, f"{name}: {frac} out of [0,1]"

    def test_always_dead_relu_has_fraction_one(self):
        """A model where all pre-activations are forced negative → fraction=1."""
        # Build a model that always produces negative pre-relu values:
        # Linear(-1, ...) so all outputs are negative → ReLU kills everything.
        model = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU())
        with torch.no_grad():
            model[1].weight.fill_(-1.0)   # all negative weights
            model[1].bias.fill_(-10.0)    # large negative bias
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, torch.device("cpu"))
        assert len(result) == 1
        key = list(result.keys())[0]
        assert result[key] == 1.0

    def test_no_relu_returns_empty(self):
        """Model without any ReLU should return an empty dict."""
        model = nn.Sequential(nn.Flatten(), nn.Linear(16, 4))
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, torch.device("cpu"))
        assert result == {}

    def test_weights_unchanged_after_call(self):
        """compute_dead_neurons must not modify model weights."""
        model, _, _, device = tiny_model_and_batch()
        loader = tiny_loader()
        weights_before = [p.data.clone() for p in model.parameters()]
        compute_dead_neurons(model, loader, device)
        for before, p in zip(weights_before, model.parameters()):
            assert torch.allclose(before, p.data)


# ---------------------------------------------------------------------------
# Weight rank
# ---------------------------------------------------------------------------

class TestWeightRank:
    def test_returns_dict(self):
        model, _, _, device = tiny_model_and_batch()
        result = compute_weight_rank(model)
        assert isinstance(result, dict)

    def test_keys_are_linear_layers(self):
        """Keys should correspond to Linear layer names in the model."""
        model, _, _, device = tiny_model_and_batch()
        result = compute_weight_rank(model)
        linear_names = {name for name, mod in model.named_modules()
                        if isinstance(mod, nn.Linear)}
        assert set(result.keys()) == linear_names

    def test_rank_at_least_one(self):
        """Effective rank must be ≥ 1 for any non-degenerate matrix."""
        model, _, _, device = tiny_model_and_batch()
        result = compute_weight_rank(model)
        for name, rank in result.items():
            if not math.isnan(rank):
                assert rank >= 1.0 - 1e-6, f"{name}: rank={rank} < 1"

    def test_rank_bounded_by_min_dim(self):
        """Effective rank must be ≤ min(rows, cols)."""
        model = nn.Linear(8, 4)   # weight shape 4×8, min dim = 4
        result = compute_weight_rank(model)
        for name, rank in result.items():
            if not math.isnan(rank):
                assert rank <= 4.0 + 1e-4, f"{name}: rank={rank} > min_dim=4"

    def test_rank_one_matrix(self):
        """An exactly rank-1 matrix should have effective rank ≈ 1."""
        model = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            # Outer product v·v^T is exactly rank 1
            v = torch.randn(4)
            model.weight.data = v.unsqueeze(1) * v.unsqueeze(0)
        result = compute_weight_rank(model)
        rank = list(result.values())[0]
        assert abs(rank - 1.0) < 0.05, f"Rank-1 matrix gave erank={rank}"

    def test_empty_model_returns_empty(self):
        """A model with no Linear or Conv2d layers returns empty dict."""
        model = nn.Sequential(nn.ReLU())
        result = compute_weight_rank(model)
        assert result == {}


# ---------------------------------------------------------------------------
# Gradient noise scale
# ---------------------------------------------------------------------------

class TestGradientNoiseScale:
    def test_returns_float(self):
        result = compute_gradient_noise_scale(1.0, 2.0)
        assert isinstance(result, float)

    def test_value_in_range(self):
        """GNS = signal² / total² must be in [0, 1] when signal ≤ total."""
        # ||E[g]||² ≤ E[||g||²] by Jensen's inequality
        result = compute_gradient_noise_scale(0.5, 2.0)
        assert 0.0 <= result <= 1.0

    def test_perfect_agreement(self):
        """When signal equals total (no noise), GNS should be 1."""
        result = compute_gradient_noise_scale(3.0, 3.0)
        assert abs(result - 1.0) < 1e-9

    def test_zero_denominator_returns_nan(self):
        """Zero total power should return NaN."""
        result = compute_gradient_noise_scale(0.0, 0.0)
        assert math.isnan(result)

    def test_proportional_to_signal_power(self):
        """Higher signal power → higher GNS (for fixed total power)."""
        low  = compute_gradient_noise_scale(0.1, 1.0)
        high = compute_gradient_noise_scale(0.9, 1.0)
        assert high > low


# ---------------------------------------------------------------------------
# Fisher trace
# ---------------------------------------------------------------------------

class TestFisherTrace:
    def test_returns_float(self):
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        assert isinstance(result, float)

    def test_finite_on_valid_model(self):
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        assert math.isfinite(result)

    def test_non_negative(self):
        """Fisher trace = E[||∇||²] — must be ≥ 0."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        assert result >= 0.0

    def test_weights_unchanged_after_call(self):
        """compute_fisher_trace must not modify model weights."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        weights_before = [p.data.clone() for p in model.parameters()]
        compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        for before, p in zip(weights_before, model.parameters()):
            assert torch.allclose(before, p.data)

    def test_gradients_zeroed_after_call(self):
        """Gradients should be zeroed after the call so training is not affected."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        for p in model.parameters():
            if p.grad is not None:
                assert (p.grad == 0).all()
