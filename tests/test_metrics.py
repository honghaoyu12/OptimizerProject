"""Tests for metrics.py — Hessian trace and sharpness estimators."""

import math
import torch
import torch.nn as nn
from metrics import compute_hessian_trace, compute_sharpness


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
