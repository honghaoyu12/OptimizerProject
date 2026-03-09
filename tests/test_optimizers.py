"""Tests for all optimizers — construction and one step with a tiny model."""

import pytest
import torch
import torch.nn as nn
from optimizers import VanillaSGD, Lion, LAMB, Shampoo, Muon, Adan, AdaHessian
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
    opt = build_optimizer(optimizer_name, model, lr=lr)
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
            "muon", "adan", "adahessian",
        }
        assert expected == set(OPTIMIZER_REGISTRY.keys())

    @pytest.mark.parametrize("name", list(OPTIMIZER_REGISTRY.keys()))
    def test_each_optimizer_steps(self, name):
        lr_map = {"lion": 1e-4, "muon": 0.02, "adahessian": 0.1}
        lr = lr_map.get(name, 1e-3)
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


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------

class TestMuon:
    def test_momentum_buffer_initialized(self):
        model = nn.Linear(8, 4)
        opt = Muon(model.parameters(), lr=0.02)
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()
        opt.step()
        for p in model.parameters():
            assert "momentum_buffer" in opt.state[p]

    def test_2d_weights_updated(self):
        """2-D weight matrix should change after one step."""
        model = nn.Linear(8, 4, bias=False)
        w_before = model.weight.data.clone()
        opt = Muon(model.parameters(), lr=0.02)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        assert not torch.allclose(model.weight.data, w_before)

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = Muon(model.parameters(), lr=0.02)
        for _ in range(10):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_weight_decay_shrinks_params(self):
        """Large weight_decay should produce smaller param norm than wd=0."""
        def _run(wd):
            torch.manual_seed(0)
            model = nn.Linear(4, 2, bias=False)
            nn.init.constant_(model.weight, 1.0)
            opt = Muon(model.parameters(), lr=0.02, weight_decay=wd)
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)


# ---------------------------------------------------------------------------
# Adan
# ---------------------------------------------------------------------------

class TestAdan:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = Adan(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_diff" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert "prev_grad" in opt.state[p]

    def test_step_counter_increments(self):
        model = nn.Linear(8, 4)
        opt = Adan(model.parameters(), lr=1e-3)
        for _ in range(3):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 3

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = Adan(model.parameters(), lr=1e-3)
        for _ in range(10):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_weight_decay_shrinks_params(self):
        def _run(wd):
            torch.manual_seed(0)
            model = nn.Linear(4, 2, bias=False)
            nn.init.constant_(model.weight, 1.0)
            opt = Adan(model.parameters(), lr=1e-3, weight_decay=wd)
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)


# ---------------------------------------------------------------------------
# AdaHessian
# ---------------------------------------------------------------------------

class TestAdaHessian:
    def test_requires_create_graph_flag(self):
        """AdaHessian must advertise its need for create_graph."""
        assert AdaHessian.requires_create_graph is True

    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = AdaHessian(model.parameters(), lr=0.1)
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()   # no create_graph → fallback to squared grad
        opt.step()
        for p in model.parameters():
            assert "exp_avg" in opt.state[p]
            assert "exp_hess" in opt.state[p]

    def test_weights_finite_without_create_graph(self):
        """Falls back to Adam-like behaviour when graph is not retained."""
        model = nn.Linear(8, 4)
        opt = AdaHessian(model.parameters(), lr=0.1)
        for _ in range(5):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_weights_finite_with_create_graph(self):
        """True AdaHessian (with HVP) also produces finite weights."""
        model = nn.Linear(8, 4)
        opt = AdaHessian(model.parameters(), lr=0.1)
        for _ in range(5):
            loss = model(torch.randn(4, 8)).sum()
            loss.backward(create_graph=True)
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_hvp_path_taken_with_differentiable_grads(self):
        """When gradients are differentiable nodes, HVP path is used.

        We verify this by checking that exp_hess differs from the squared
        gradient on a model whose Hessian is non-zero (quadratic loss on a
        linear model gives a non-zero Hessian).
        """
        torch.manual_seed(42)
        model = nn.Linear(4, 2, bias=False)
        opt = AdaHessian(model.parameters(), lr=0.1)
        x = torch.randn(3, 4)

        # Use autograd.grad so p.grad has grad_fn (differentiable)
        loss = model(x).pow(2).sum()   # quadratic → non-zero Hessian
        trainable = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, trainable, create_graph=True)
        for p, g in zip(trainable, grads):
            p.grad = g

        opt.step()
        exp_hess = opt.state[model.weight]["exp_hess"]

        # Reference: squared gradient (what the fallback would use)
        g = grads[0].detach()
        sq_grad = g.pow(2)

        # HVP-based estimate differs from pure squared gradient
        assert not torch.allclose(exp_hess / (1 - 0.999), sq_grad, atol=1e-5)
