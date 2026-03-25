"""Tests for all optimizers — construction and one step with a tiny model."""

import pytest
import torch
import torch.nn as nn
from optimizers import VanillaSGD, Lion, LAMB, Shampoo, Muon, Adan, AdaHessian, AdaBelief, SignSGD, AdaFactor, Sophia, Prodigy, ScheduleFreeAdamW, SOAP, CautiousAdam, MARS, GrokFast, Tiger, AdanNesterov
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
            "muon", "adan", "adahessian", "adabelief", "signsgd", "adafactor",
            "sophia", "prodigy", "sf_adamw",
            # New optimizers (2024)
            "soap", "cautious_adam", "mars", "grokfast",
            # New optimizers (2025)
            "tiger", "adan_nesterov",
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


# ---------------------------------------------------------------------------
# AdaBelief
# ---------------------------------------------------------------------------

class TestAdaBelief:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = AdaBelief(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_belief" in opt.state[p]
            assert "step" in opt.state[p]

    def test_step_counter_increments(self):
        model = nn.Linear(8, 4)
        opt = AdaBelief(model.parameters(), lr=1e-3)
        for _ in range(3):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 3

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = AdaBelief(model.parameters(), lr=1e-3)
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
            opt = AdaBelief(model.parameters(), lr=1e-3, weight_decay=wd)
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)


# ---------------------------------------------------------------------------
# SignSGD
# ---------------------------------------------------------------------------

class TestSignSGD:
    def test_update_uses_sign(self):
        """All weight changes should be exactly ±lr (sign update)."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 0.0)
        lr = 0.01
        opt = SignSGD(model.parameters(), lr=lr)
        model(torch.ones(1, 4)).sum().backward()
        w_before = model.weight.data.clone()
        opt.step()
        delta = (model.weight.data - w_before).abs()
        # Every element should change by exactly lr
        assert torch.allclose(delta, torch.full_like(delta, lr))

    def test_momentum_buffer_initialized(self):
        model = nn.Linear(8, 4)
        opt = SignSGD(model.parameters(), lr=0.01, momentum=0.9)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "momentum_buffer" in opt.state[p]

    def test_no_momentum_buffer_without_momentum(self):
        model = nn.Linear(8, 4)
        opt = SignSGD(model.parameters(), lr=0.01, momentum=0.0)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "momentum_buffer" not in opt.state[p]

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = SignSGD(model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(10):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# AdaFactor
# ---------------------------------------------------------------------------

class TestAdaFactor:
    def test_factored_state_for_2d(self):
        """2-D weight matrix should use row/col factors, not full second moment."""
        model = nn.Linear(8, 4)   # weight: 4×8
        opt = AdaFactor(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        w = model.weight
        assert opt.state[w]["factored"] is True
        assert "exp_avg_sq_row" in opt.state[w]
        assert "exp_avg_sq_col" in opt.state[w]
        assert "exp_avg_sq" not in opt.state[w]

    def test_non_factored_state_for_1d(self):
        """1-D bias should use the full second moment (no factoring)."""
        model = nn.Linear(8, 4)   # bias: (4,)
        opt = AdaFactor(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        b = model.bias
        assert opt.state[b]["factored"] is False
        assert "exp_avg_sq" in opt.state[b]

    def test_row_col_factor_shapes(self):
        """Row factor shape == (out,), col factor shape == (in,) for Linear weight."""
        model = nn.Linear(8, 4, bias=False)   # weight: 4×8
        opt = AdaFactor(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        w = model.weight
        assert opt.state[w]["exp_avg_sq_row"].shape == (4,)
        assert opt.state[w]["exp_avg_sq_col"].shape == (8,)

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = AdaFactor(model.parameters(), lr=1e-3)
        for _ in range(10):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_relative_step_mode(self):
        """lr=None (relative step) should also produce finite weights."""
        model = nn.Linear(8, 4)
        opt = AdaFactor(model.parameters(), lr=None)
        for _ in range(5):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# Sophia
# ---------------------------------------------------------------------------

class TestSophia:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = Sophia(model.parameters(), lr=1e-4)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "exp_avg" in opt.state[p]
            assert "hess" in opt.state[p]
            assert "step" in opt.state[p]

    def test_hessian_updated_at_update_freq(self):
        """hess should change on step 1 and step update_freq+1, not in between."""
        model = nn.Linear(4, 2, bias=False)
        opt = Sophia(model.parameters(), lr=1e-4, update_freq=5)
        # Step 1 — hess is updated (step % update_freq == 1)
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
        h_after_1 = opt.state[model.weight]["hess"].clone()
        # Steps 2–5 — hess should not change
        for _ in range(4):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            opt.zero_grad()
        h_after_5 = opt.state[model.weight]["hess"].clone()
        assert torch.allclose(h_after_1, h_after_5)
        # Step 6 — hess updated again
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
        h_after_6 = opt.state[model.weight]["hess"].clone()
        assert not torch.allclose(h_after_5, h_after_6)

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = Sophia(model.parameters(), lr=1e-4, update_freq=5)
        for _ in range(25):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_weights_finite_with_hvp(self):
        """With create_graph=True the full HVP path is used and weights stay finite."""
        model = nn.Linear(8, 4)
        opt = Sophia(model.parameters(), lr=1e-4, update_freq=3)
        for _ in range(10):
            loss = model(torch.randn(4, 8)).sum()
            loss.backward(create_graph=True)
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()


# ---------------------------------------------------------------------------
# Prodigy
# ---------------------------------------------------------------------------

class TestProdigy:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = Prodigy(model.parameters())
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "x0" in opt.state[p]
            assert "s" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]

    def test_d_grows_over_time(self):
        """d should be non-decreasing (Prodigy's monotone guarantee)."""
        model = nn.Linear(8, 4)
        opt = Prodigy(model.parameters(), d0=1e-6)
        d_values = []
        for _ in range(10):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
            d_values.append(opt.param_groups[0]["d"])
        # d must never decrease
        for i in range(1, len(d_values)):
            assert d_values[i] >= d_values[i - 1]

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = Prodigy(model.parameters())
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
            opt = Prodigy(model.parameters(), weight_decay=wd)
            for _ in range(5):
                model(torch.randn(2, 4)).sum().backward()
                opt.step()
                opt.zero_grad()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)


# ---------------------------------------------------------------------------
# ScheduleFreeAdamW
# ---------------------------------------------------------------------------

class TestScheduleFreeAdamW:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "x" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert "step" in opt.state[p]

    def test_params_differ_from_x_after_steps(self):
        """After >1 steps, params (y) should differ from x (the average)."""
        model = nn.Linear(8, 4, bias=False)
        opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3)
        for _ in range(5):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        w = model.weight
        x = opt.state[w]["x"]
        # y = (1-beta1)*x + beta1*z; after several steps y != x in general
        assert not torch.allclose(w.data, x)

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3)
        for _ in range(10):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_get_averaged_params_returns_x(self):
        """get_averaged_params() should return tensors close to state['x']."""
        model = nn.Linear(4, 2, bias=False)
        opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3)
        for _ in range(3):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            opt.zero_grad()
        averaged = opt.get_averaged_params()
        for avg, p in zip(averaged, model.parameters()):
            assert torch.allclose(avg, opt.state[p]["x"])


# ---------------------------------------------------------------------------
# SOAP
# ---------------------------------------------------------------------------

class TestSOAP:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = SOAP(model.parameters(), lr=3e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        w = model.weight
        assert "L" in opt.state[w]
        assert "R" in opt.state[w]
        assert "exp_avg" in opt.state[w]
        assert "exp_avg_sq" in opt.state[w]

    def test_diagonal_fallback_for_oversized(self):
        """Matrices exceeding max_precond_size should use plain Adam (no L/R)."""
        model = nn.Linear(8, 4)   # weight shape 4×8; set limit below both dims
        opt = SOAP(model.parameters(), lr=3e-3, max_precond_size=3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        w = model.weight
        # Falls back to plain Adam — no Kronecker factors stored
        assert "L" not in opt.state[w]
        assert "R" not in opt.state[w]

    def test_kronecker_updated_at_update_freq(self):
        """Q_L/Q_R eigenvectors should be refreshed at multiples of update_freq."""
        model = nn.Linear(8, 4, bias=False)
        opt = SOAP(model.parameters(), lr=3e-3, update_freq=5)
        # Run enough steps to trigger an eigenvector refresh
        for _ in range(6):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        w = model.weight
        # Q_L and Q_R should exist after a refresh
        assert "Q_L" in opt.state[w]
        assert "Q_R" in opt.state[w]

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = SOAP(model.parameters(), lr=3e-3)
        for _ in range(25):
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
            opt = SOAP(model.parameters(), lr=3e-3, weight_decay=wd)
            for _ in range(5):
                model(torch.randn(2, 4)).sum().backward()
                opt.step()
                opt.zero_grad()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)


# ---------------------------------------------------------------------------
# CautiousAdam
# ---------------------------------------------------------------------------

class TestCautiousAdam:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = CautiousAdam(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]

    def test_step_counter_increments(self):
        model = nn.Linear(8, 4)
        opt = CautiousAdam(model.parameters(), lr=1e-3)
        for i in range(1, 4):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 3

    def test_masked_update_direction(self):
        """With a large positive constant weight and input, gradient is positive,
        so the Adam update is negative.  The mask should keep the update (sign
        agrees) and the weight should decrease."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 1.0)
        opt = CautiousAdam(model.parameters(), lr=0.5)
        model(torch.ones(1, 4)).sum().backward()
        opt.step()
        assert (model.weight < 1.0).all()

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = CautiousAdam(model.parameters(), lr=1e-3)
        for _ in range(20):
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
            opt = CautiousAdam(model.parameters(), lr=1e-3, weight_decay=wd)
            for _ in range(5):
                model(torch.randn(2, 4)).sum().backward()
                opt.step()
                opt.zero_grad()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)


# ---------------------------------------------------------------------------
# MARS
# ---------------------------------------------------------------------------

class TestMARS:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = MARS(model.parameters(), lr=3e-4)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert "prev_grad" in opt.state[p]

    def test_prev_grad_stored(self):
        """prev_grad at step t should equal the gradient used at step t."""
        model = nn.Linear(4, 2, bias=False)
        opt = MARS(model.parameters(), lr=3e-4)
        model(torch.ones(2, 4)).sum().backward()
        g_step1 = model.weight.grad.clone()
        opt.step()
        # After step 1, prev_grad should be the gradient from step 1
        assert torch.allclose(opt.state[model.weight]["prev_grad"], g_step1)

    def test_step_counter_increments(self):
        model = nn.Linear(8, 4)
        opt = MARS(model.parameters(), lr=3e-4)
        for _ in range(3):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 3

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = MARS(model.parameters(), lr=3e-4)
        for _ in range(20):
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
            opt = MARS(model.parameters(), lr=3e-4, weight_decay=wd)
            for _ in range(5):
                model(torch.randn(2, 4)).sum().backward()
                opt.step()
                opt.zero_grad()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)


# ---------------------------------------------------------------------------
# GrokFast
# ---------------------------------------------------------------------------

class TestGrokFast:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = GrokFast(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert "grad_ema" in opt.state[p]

    def test_grad_ema_initialized_to_first_gradient(self):
        """On first step, grad_ema should be initialised to the gradient."""
        model = nn.Linear(4, 2, bias=False)
        opt = GrokFast(model.parameters(), lr=1e-3)
        model(torch.ones(2, 4)).sum().backward()
        g_init = model.weight.grad.clone()
        opt.step()
        # After step 1, grad_ema should equal the first gradient
        assert torch.allclose(opt.state[model.weight]["grad_ema"], g_init, atol=1e-6)

    def test_grad_ema_decays_over_steps(self):
        """grad_ema should change across steps (EMA update)."""
        model = nn.Linear(4, 2, bias=False)
        opt = GrokFast(model.parameters(), lr=1e-3, alpha=0.9)
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
        ema_after_1 = opt.state[model.weight]["grad_ema"].clone()
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
        ema_after_2 = opt.state[model.weight]["grad_ema"]
        # EMA must have changed (different random gradients each step)
        assert not torch.allclose(ema_after_1, ema_after_2)

    def test_step_counter_increments(self):
        model = nn.Linear(8, 4)
        opt = GrokFast(model.parameters(), lr=1e-3)
        for _ in range(3):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 3

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = GrokFast(model.parameters(), lr=1e-3)
        for _ in range(20):
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
            opt = GrokFast(model.parameters(), lr=1e-3, weight_decay=wd)
            for _ in range(5):
                model(torch.randn(2, 4)).sum().backward()
                opt.step()
                opt.zero_grad()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)


# ---------------------------------------------------------------------------
# Tiger
# ---------------------------------------------------------------------------

class TestTiger:
    def test_momentum_buffer_initialized(self):
        model = nn.Linear(8, 4)
        opt = Tiger(model.parameters(), lr=1e-4)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "exp_avg" in opt.state[p]

    def test_update_uses_sign(self):
        """Weight change per coordinate should be exactly ±lr when wd=0."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 1.0)
        lr = 0.1
        opt = Tiger(model.parameters(), lr=lr)
        model(torch.ones(1, 4)).sum().backward()
        w_before = model.weight.data.clone()
        opt.step()
        delta = (model.weight.data - w_before).abs()
        assert torch.allclose(delta, torch.full_like(delta, lr), atol=1e-6)

    def test_weight_decay_applied(self):
        """With large weight decay, weights should shrink toward zero."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 1.0)
        opt = Tiger(model.parameters(), lr=1e-4, weight_decay=1.0)
        model(torch.zeros(1, 4)).sum().backward()
        opt.step()
        assert (model.weight < 1.0).all()

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = Tiger(model.parameters(), lr=1e-4)
        for _ in range(20):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p).all()

    def test_momentum_updates_after_step(self):
        """exp_avg should change across steps."""
        model = nn.Linear(4, 2, bias=False)
        opt = Tiger(model.parameters(), lr=1e-4)
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
        m_after_1 = opt.state[model.weight]["exp_avg"].clone()
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
        m_after_2 = opt.state[model.weight]["exp_avg"]
        assert not torch.allclose(m_after_1, m_after_2)


# ---------------------------------------------------------------------------
# AdanNesterov
# ---------------------------------------------------------------------------

class TestAdanNesterov:
    def test_state_initialized(self):
        model = nn.Linear(8, 4)
        opt = AdanNesterov(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_diff" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert "prev_grad" in opt.state[p]

    def test_step_counter_increments(self):
        model = nn.Linear(8, 4)
        opt = AdanNesterov(model.parameters(), lr=1e-3)
        for _ in range(3):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 3

    def test_weights_finite_after_many_steps(self):
        model = nn.Linear(8, 4)
        opt = AdanNesterov(model.parameters(), lr=1e-3)
        for _ in range(20):
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
            opt = AdanNesterov(model.parameters(), lr=1e-3, weight_decay=wd)
            for _ in range(5):
                model(torch.randn(2, 4)).sum().backward()
                opt.step()
                opt.zero_grad()
            return model.weight.norm().item()

        assert _run(1.0) < _run(0.0)

    def test_nesterov_gamma_zero_matches_adan(self):
        """With nesterov_gamma=0 the first step should match standard Adan."""
        from optimizers import Adan
        torch.manual_seed(42)
        model_n = nn.Linear(4, 2, bias=False)
        torch.manual_seed(42)
        model_a = nn.Linear(4, 2, bias=False)
        model_a.weight.data.copy_(model_n.weight.data)

        opt_n = AdanNesterov(model_n.parameters(), lr=1e-3, nesterov_gamma=0.0)
        opt_a = Adan(model_a.parameters(), lr=1e-3)

        torch.manual_seed(0)
        x = torch.randn(4, 4)
        model_n(x).sum().backward()
        g = model_n.weight.grad.clone()
        model_a.weight.grad = g.clone()

        opt_n.step()
        opt_a.step()

        # First step: no prior momentum, gamma=0 → no Nesterov correction
        assert torch.allclose(model_n.weight.data, model_a.weight.data, atol=1e-6)
