"""Tests for all optimizers — construction and one step with a tiny model."""

import pytest
import torch
import torch.nn as nn
from optimizers import VanillaSGD, Lion, LAMB, Shampoo
from train import build_optimizer, OPTIMIZER_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model_and_loss():
    """Return a tiny Linear model, a random loss, and its backward pass."""
    model = nn.Linear(16, 4)
    x = torch.randn(8, 16)
    loss = model(x).sum()
    loss.backward()
    return model, loss


def one_step(optimizer_name: str, lr: float = 1e-3):
    """Build the named optimizer, run one step, return the model."""
    model = nn.Linear(16, 4)
    opt = build_optimizer(optimizer_name, model.parameters(), lr=lr)
    x = torch.randn(8, 16)
    loss = model(x).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return model


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestOptimizerRegistry:
    def test_all_keys_present(self):
        expected = {
            "adam", "adamw", "nadam", "radam", "adagrad",
            "sgd", "rmsprop", "vanilla_sgd", "lion", "lamb", "shampoo",
        }
        assert expected == set(OPTIMIZER_REGISTRY.keys())

    @pytest.mark.parametrize("name", list(OPTIMIZER_REGISTRY.keys()))
    def test_each_optimizer_steps(self, name):
        lr = 1e-4 if name == "lion" else 1e-3
        model = one_step(name, lr=lr)
        # Weights should be finite after one step
        for p in model.parameters():
            assert torch.isfinite(p).all(), f"{name}: non-finite weights after one step"


# ---------------------------------------------------------------------------
# VanillaSGD
# ---------------------------------------------------------------------------

class TestVanillaSGD:
    def test_update_direction(self):
        """Weights should decrease when gradient is positive."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 1.0)
        opt = VanillaSGD(model.parameters(), lr=0.1)
        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        opt.step()
        assert (model.weight < 1.0).all()

    def test_zero_grad_clears_grad(self):
        model, _ = make_model_and_loss()
        opt = VanillaSGD(model.parameters(), lr=0.01)
        opt.zero_grad()
        for p in model.parameters():
            assert p.grad is None or (p.grad == 0).all()


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------

class TestLion:
    def test_momentum_buffer_initialized(self):
        model = nn.Linear(8, 4)
        opt = Lion(model.parameters(), lr=1e-4)
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()
        opt.step()
        for p in model.parameters():
            assert "exp_avg" in opt.state[p]

    def test_weight_decay_applied(self):
        """With large weight decay, weights should shrink toward zero."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 1.0)
        opt = Lion(model.parameters(), lr=1e-4, weight_decay=1.0)
        loss = model(torch.zeros(1, 4)).sum()   # zero input → zero grad
        loss.backward()
        opt.step()
        # weight decay still pushes weights down even with zero gradient
        assert (model.weight < 1.0).all()


# ---------------------------------------------------------------------------
# LAMB
# ---------------------------------------------------------------------------

class TestLAMB:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = LAMB(model.parameters(), lr=1e-3)
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]

    def test_step_counter_increments(self):
        model = nn.Linear(8, 4)
        opt = LAMB(model.parameters(), lr=1e-3)
        for i in range(1, 4):
            loss = model(torch.randn(4, 8)).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 3


# ---------------------------------------------------------------------------
# Shampoo
# ---------------------------------------------------------------------------

class TestShampoo:
    def test_kronecker_state_initialized(self):
        model = nn.Linear(8, 4)   # weight: 4×8 — fits within max_precond_size
        opt = Shampoo(model.parameters(), lr=0.01)
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()
        opt.step()
        w = model.weight
        assert "L" in opt.state[w]
        assert "R" in opt.state[w]
        assert "L_inv_root" in opt.state[w]

    def test_diagonal_fallback_for_oversized(self):
        # max_precond_size=4 forces the 4×8 weight to use diagonal fallback
        model = nn.Linear(8, 4)
        opt = Shampoo(model.parameters(), lr=0.01, max_precond_size=4)
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()
        opt.step()
        w = model.weight
        assert opt.state[w]["use_kronecker"] is False
        assert "sum_sq" in opt.state[w]

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = Shampoo(model.parameters(), lr=0.01)
        for _ in range(25):   # crosses the update_freq=10 boundary twice
            loss = model(torch.randn(4, 8)).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()
