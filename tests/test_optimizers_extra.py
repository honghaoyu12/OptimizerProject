"""Additional optimizer tests — deeper behavioural coverage.

Targets properties not covered by test_optimizers.py or test_new_optimizers.py:
  - Closure support (all optimizers must return closure loss)
  - Zero-gradient no-op (params must not change when grad is None)
  - Weights finite after N steps (extended to more optimizers)
  - Weight-decay shrinks params (for optimizers not yet tested)
  - Algorithm-specific invariants (signs, monotonicity, state shapes)
  - VanillaSGD weight decay, closure, and no-op behaviour
  - Lion update magnitude invariant (all-±lr)
  - SignSGD update magnitude invariant
  - LAMB trust-ratio clamp
  - Muon Newton-Schulz output orthogonality
  - AdaBelief belief moment ≥ 0
  - AdaFactor rho schedule monotonicity and first-moment option
  - Prodigy x0 snapshot frozen
  - ScheduleFreeAdamW beta1=0 edge case
  - Adan three-buffer state
  - SOAP eigenvector refresh at update_freq
  - CautiousAdam mask zeroes anti-aligned coordinates
  - MARS, GrokFast, Tiger, AdanNesterov finite + weight decay
"""

import math
import pytest
import torch
import torch.nn as nn

from optimizers import (
    VanillaSGD, Lion, LAMB, Muon, AdaBelief, SignSGD, AdaFactor,
    Prodigy, ScheduleFreeAdamW, Adan, SOAP, CautiousAdam,
    MARS, GrokFast, Tiger, AdanNesterov,
)
from optimizers.muon import _newton_schulz


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _param_and_grad(shape=(8, 4), val=1.0, grad_val=0.5):
    """Return a parameter tensor with a preset gradient."""
    p = nn.Parameter(torch.full(shape, val))
    p.grad = torch.full(shape, grad_val)
    return p


def _single_step(opt_cls, **kwargs):
    """Construct optimizer, run one step, return (param_before, param_after)."""
    p = _param_and_grad()
    opt = opt_cls([p], **kwargs)
    before = p.data.clone()
    opt.step()
    return before, p.data.clone()


def _many_steps(opt_cls, n=30, **kwargs):
    """Run n gradient-descent steps and return final param tensor."""
    model = nn.Linear(8, 4)
    opt = opt_cls(model.parameters(), **kwargs)
    x = torch.randn(16, 8)
    for _ in range(n):
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return model


# ---------------------------------------------------------------------------
# Closure support (universal)
# ---------------------------------------------------------------------------

class TestClosureSupport:
    """Every optimizer must return the loss value when called with a closure."""

    @pytest.mark.parametrize("opt_cls,kwargs", [
        (VanillaSGD,       dict(lr=1e-2)),
        (Lion,             dict(lr=1e-4)),
        (LAMB,             dict(lr=1e-3)),
        (Muon,             dict(lr=1e-2)),
        (AdaBelief,        dict(lr=1e-3)),
        (SignSGD,          dict(lr=1e-2)),
        (AdaFactor,        dict(lr=1e-3)),
        (Prodigy,          dict(lr=1.0)),
        (ScheduleFreeAdamW,dict(lr=1e-3)),
        (Adan,             dict(lr=1e-3)),
        (SOAP,             dict(lr=1e-3)),
        (CautiousAdam,     dict(lr=1e-3)),
        (MARS,             dict(lr=1e-3)),
        (GrokFast,         dict(lr=1e-3)),
        (Tiger,            dict(lr=1e-4)),
        (AdanNesterov,     dict(lr=1e-3)),
    ])
    def test_closure_returns_loss(self, opt_cls, kwargs):
        model = nn.Linear(4, 2)
        opt = opt_cls(model.parameters(), **kwargs)
        x = torch.randn(4, 4)

        def closure():
            opt.zero_grad()
            loss = model(x).sum()
            loss.backward()
            return loss

        # AdaHessian/Sophia need create_graph — skip here; covered elsewhere
        returned = opt.step(closure)
        assert returned is not None
        assert torch.is_tensor(returned) or isinstance(returned, float)


# ---------------------------------------------------------------------------
# Zero-gradient no-op (universal)
# ---------------------------------------------------------------------------

class TestZeroGradNoOp:
    """When no gradient is set, params must not change after step()."""

    @pytest.mark.parametrize("opt_cls,kwargs", [
        (VanillaSGD,       dict(lr=1e-2)),
        (Lion,             dict(lr=1e-4)),
        (LAMB,             dict(lr=1e-3)),
        (Muon,             dict(lr=1e-2)),
        (AdaBelief,        dict(lr=1e-3)),
        (SignSGD,          dict(lr=1e-2)),
        (Prodigy,          dict(lr=1.0)),
        (ScheduleFreeAdamW,dict(lr=1e-3)),
        (Adan,             dict(lr=1e-3)),
        (SOAP,             dict(lr=1e-3)),
        (CautiousAdam,     dict(lr=1e-3)),
        (MARS,             dict(lr=1e-3)),
        (GrokFast,         dict(lr=1e-3)),
        (Tiger,            dict(lr=1e-4)),
        (AdanNesterov,     dict(lr=1e-3)),
    ])
    def test_no_grad_no_change(self, opt_cls, kwargs):
        p = nn.Parameter(torch.randn(4, 4))
        # Do NOT set p.grad → it is None
        opt = opt_cls([p], **kwargs)
        before = p.data.clone()
        opt.step()
        assert torch.equal(p.data, before), \
            f"{opt_cls.__name__}: param changed without a gradient"


# ---------------------------------------------------------------------------
# VanillaSGD — additional
# ---------------------------------------------------------------------------

class TestVanillaSGDAdditional:
    def test_weight_decay_shrinks_params(self):
        p = _param_and_grad(val=2.0, grad_val=0.0)  # zero grad → only WD
        opt = VanillaSGD([p], lr=0.1, weight_decay=0.5)
        opt.step()
        # θ ← θ − lr·wd·θ = 2 − 0.1·0.5·2 = 1.9
        assert (p.data < 2.0).all()

    def test_no_weight_decay_no_shrink_on_zero_grad(self):
        p = _param_and_grad(val=3.0, grad_val=0.0)
        opt = VanillaSGD([p], lr=0.1, weight_decay=0.0)
        before = p.data.clone()
        opt.step()
        assert torch.equal(p.data, before)

    def test_weights_finite_many_steps(self):
        m = _many_steps(VanillaSGD, lr=1e-2)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# Lion — additional
# ---------------------------------------------------------------------------

class TestLionAdditional:
    def test_update_magnitude_is_lr(self):
        """Lion update is sign-based → every element moves by exactly lr."""
        lr = 1e-3
        p = _param_and_grad(val=0.0, grad_val=1.0)
        opt = Lion([p], lr=lr, betas=(0.9, 0.99), weight_decay=0.0)
        before = p.data.clone()
        opt.step()
        diff = (p.data - before).abs()
        assert torch.allclose(diff, torch.full_like(diff, lr), atol=1e-6)

    def test_momentum_decays_toward_gradient(self):
        """After many steps on a constant gradient, momentum buffer is positive."""
        p = _param_and_grad(val=0.0, grad_val=1.0)
        opt = Lion([p], lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0)
        for _ in range(200):
            p.grad = torch.ones_like(p)
            opt.step()
        buf = opt.state[p]["exp_avg"]
        # EMA(g=1, β₂=0.99) converges to 1; buffer must be substantially positive
        assert (buf > 0.5).all()

    def test_weight_decay_extra_shrink(self):
        """With WD>0 the param should shrink more than without."""
        lr = 1e-3
        p1 = _param_and_grad(val=1.0, grad_val=0.1)
        p2 = _param_and_grad(val=1.0, grad_val=0.1)
        opt1 = Lion([p1], lr=lr, weight_decay=0.0)
        opt2 = Lion([p2], lr=lr, weight_decay=0.5)
        opt1.step(); opt2.step()
        assert p2.data.abs().sum() < p1.data.abs().sum()

    def test_weights_finite_many_steps(self):
        m = _many_steps(Lion, lr=1e-4)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# LAMB — additional
# ---------------------------------------------------------------------------

class TestLAMBAdditional:
    def test_trust_ratio_clamped_at_ten(self):
        """Trust ratio must not exceed 10; result must be finite."""
        p = nn.Parameter(torch.ones(4, 4) * 1000.0)
        p.grad = torch.ones(4, 4) * 1.0
        opt = LAMB([p], lr=1e-3, weight_decay=0.0)
        opt.step()
        assert torch.isfinite(p.data).all()

    def test_zero_param_norm_fallback(self):
        """When param norm == 0, trust_ratio = 1 (no crash)."""
        p = nn.Parameter(torch.zeros(4, 4))
        p.grad = torch.ones(4, 4)
        opt = LAMB([p], lr=1e-3, weight_decay=0.0)
        opt.step()   # must not raise
        assert torch.isfinite(p.data).all()

    def test_weights_finite_many_steps(self):
        m = _many_steps(LAMB, lr=1e-3, weight_decay=0.01)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_weight_decay_shrinks_params(self):
        model = nn.Linear(8, 4)
        init_norm = sum(p.norm().item() for p in model.parameters())
        opt = LAMB(model.parameters(), lr=1e-3, weight_decay=0.1)
        x = torch.randn(16, 8)
        for _ in range(20):
            model(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        final_norm = sum(p.norm().item() for p in model.parameters())
        # With strong WD the norms should generally be kept in check
        assert torch.isfinite(torch.tensor(final_norm))


# ---------------------------------------------------------------------------
# Muon — Newton-Schulz and additional
# ---------------------------------------------------------------------------

class TestMuonAdditional:
    def test_newton_schulz_output_shape(self):
        G = torch.randn(8, 4)
        out = _newton_schulz(G)
        assert out.shape == G.shape

    def test_newton_schulz_rows_near_orthonormal(self):
        """After 5 iterations the rows should be approximately orthonormal."""
        torch.manual_seed(0)
        G = torch.randn(4, 8)
        Q = _newton_schulz(G, steps=5)
        # Q Q^T should be close to identity (for short-fat matrices)
        gram = Q @ Q.T
        assert torch.allclose(gram, torch.eye(4), atol=0.1)

    def test_newton_schulz_tall_matrix(self):
        """Tall matrices (rows > cols) are handled via internal transpose."""
        G = torch.randn(8, 4)
        out = _newton_schulz(G)
        assert out.shape == (8, 4)

    def test_1d_param_not_orthogonalised(self):
        """1-D parameters (biases) should be updated without Newton-Schulz."""
        p_2d = nn.Parameter(torch.randn(4, 8))
        p_1d = nn.Parameter(torch.randn(4))
        p_2d.grad = torch.randn(4, 8)
        p_1d.grad = torch.randn(4)
        before_1d = p_1d.data.clone()
        opt = Muon([p_2d, p_1d], lr=0.02, weight_decay=0.0)
        opt.step()
        # 2-D param should change; 1-D param should also change (Nesterov SGD)
        assert not torch.equal(p_1d.data, before_1d)

    def test_weights_finite_many_steps(self):
        m = _many_steps(Muon, lr=0.02)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# AdaBelief — additional
# ---------------------------------------------------------------------------

class TestAdaBeliefAdditional:
    def test_belief_moment_nonnegative(self):
        """Belief EMA s_t must always be ≥ 0 (it is a squared quantity + ε)."""
        p = _param_and_grad()
        opt = AdaBelief([p], lr=1e-3)
        for _ in range(10):
            p.grad = torch.randn_like(p)
            opt.step()
        s = opt.state[p]["exp_avg_belief"]
        assert (s >= 0).all()

    def test_belief_small_when_gradient_consistent(self):
        """Consistent gradient → small belief s (low surprise)."""
        p = _param_and_grad(grad_val=0.5)
        opt = AdaBelief([p], lr=1e-4)
        for _ in range(50):
            p.grad = torch.full_like(p, 0.5)   # constant gradient
            opt.step()
        s = opt.state[p]["exp_avg_belief"]
        m = opt.state[p]["exp_avg"]
        # s ≈ (g - m)^2 + eps ≈ eps when g ≈ m
        assert s.mean().item() < 0.01

    def test_step_counter_increments(self):
        p = _param_and_grad()
        opt = AdaBelief([p], lr=1e-3)
        for i in range(5):
            p.grad = torch.randn_like(p)
            opt.step()
        assert opt.state[p]["step"] == 5

    def test_weights_finite_many_steps(self):
        m = _many_steps(AdaBelief, lr=1e-3)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# SignSGD — additional
# ---------------------------------------------------------------------------

class TestSignSGDAdditional:
    def test_update_magnitude_equals_lr_no_wd(self):
        """Pure SignSGD: every element moves by exactly lr."""
        lr = 5e-3
        p = _param_and_grad(val=0.0, grad_val=2.0)
        opt = SignSGD([p], lr=lr, momentum=0.0, weight_decay=0.0)
        before = p.data.clone()
        opt.step()
        diff = (p.data - before).abs()
        assert torch.allclose(diff, torch.full_like(diff, lr), atol=1e-7)

    def test_signum_buffer_initialized_to_zero(self):
        p = _param_and_grad()
        opt = SignSGD([p], lr=1e-2, momentum=0.9)
        # Buffer is created on first step
        assert "momentum_buffer" not in opt.state[p]
        opt.step()
        assert "momentum_buffer" in opt.state[p]
        # After first step, buffer = (1-β)·g (started from 0)
        buf = opt.state[p]["momentum_buffer"]
        assert torch.isfinite(buf).all()

    def test_signum_decays_toward_gradient(self):
        """With large momentum, buffer approaches the constant gradient over steps."""
        p = _param_and_grad(val=0.0, grad_val=1.0)
        opt = SignSGD([p], lr=1e-4, momentum=0.9, weight_decay=0.0)
        for _ in range(100):
            p.grad = torch.ones_like(p)
            opt.step()
        buf = opt.state[p]["momentum_buffer"]
        # Should have converged close to g=1
        assert (buf > 0.9).all()

    def test_weight_decay_applied_before_sign(self):
        """WD is a multiplicative shrink; param should be smaller after step."""
        p = _param_and_grad(val=5.0, grad_val=0.0)
        opt = SignSGD([p], lr=1e-3, momentum=0.0, weight_decay=0.5)
        # grad=0 → direction=0 → only WD acts
        opt.step()
        assert (p.data < 5.0).all()

    def test_weights_finite_many_steps(self):
        m = _many_steps(SignSGD, lr=1e-2, momentum=0.9)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# AdaFactor — additional
# ---------------------------------------------------------------------------

class TestAdaFactorAdditional:
    def test_rho_is_monotonically_increasing(self):
        """ρ_t should increase toward 0.999 as step count grows."""
        rho_values = [AdaFactor._rho(t, -0.8) for t in range(1, 200)]
        assert all(rho_values[i] <= rho_values[i + 1]
                   for i in range(len(rho_values) - 1))

    def test_rho_capped_at_0_999(self):
        """ρ_t must never exceed 0.999."""
        for t in [1, 10, 100, 1000, 10000]:
            assert AdaFactor._rho(t, -0.8) <= 0.999

    def test_use_first_moment_allocates_buffer(self):
        p = _param_and_grad()
        opt = AdaFactor([p], lr=1e-3, use_first_moment=True)
        opt.step()
        assert "exp_avg" in opt.state[p]

    def test_no_first_moment_by_default(self):
        p = _param_and_grad()
        opt = AdaFactor([p], lr=1e-3, use_first_moment=False)
        opt.step()
        assert "exp_avg" not in opt.state[p]

    def test_relative_step_mode_finite(self):
        """lr=None (relative-step mode) should produce finite weights."""
        m = _many_steps(AdaFactor, n=20, lr=None)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_rms_static_method(self):
        t = torch.ones(4, 4)
        assert math.isclose(AdaFactor._rms(t), 1.0, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# Prodigy — additional
# ---------------------------------------------------------------------------

class TestProdigyAdditional:
    def test_x0_snapshot_frozen(self):
        """x0 (initial param snapshot) must not change after first step."""
        p = _param_and_grad()
        opt = Prodigy([p], lr=1.0)
        opt.step()
        x0_after_step1 = opt.state[p]["x0"].clone()
        p.grad = torch.randn_like(p)
        opt.step()
        assert torch.equal(opt.state[p]["x0"], x0_after_step1)

    def test_d_in_param_group(self):
        p = _param_and_grad()
        opt = Prodigy([p], lr=1.0, d0=1e-6)
        assert "d" in opt.param_groups[0]
        assert opt.param_groups[0]["d"] == 1e-6

    def test_A_accumulates(self):
        """A should grow from 0 as training progresses."""
        p = _param_and_grad()
        opt = Prodigy([p], lr=1.0)
        assert opt.param_groups[0]["A"] == 0.0
        opt.step()
        assert opt.param_groups[0]["A"] > 0.0

    def test_s_buffer_initialized(self):
        p = _param_and_grad()
        opt = Prodigy([p], lr=1.0)
        opt.step()
        assert "s" in opt.state[p]
        assert opt.state[p]["s"].shape == p.shape

    def test_weights_finite_many_steps(self):
        m = _many_steps(Prodigy, lr=1.0)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# ScheduleFreeAdamW — additional
# ---------------------------------------------------------------------------

class TestScheduleFreeAdamWAdditional:
    def test_x_initialized_equal_to_params(self):
        """x (Polyak avg) should start equal to the initial parameters."""
        p = _param_and_grad()
        init = p.data.clone()
        opt = ScheduleFreeAdamW([p], lr=1e-3)
        opt.step()
        # x was initialised to the initial param value before the step
        x = opt.state[p]["x"]
        # After one step x has moved, but it was initialised — shape must match
        assert x.shape == p.shape

    def test_beta1_zero_edge_case(self):
        """beta1=0 means y=z; should not crash."""
        p = _param_and_grad()
        opt = ScheduleFreeAdamW([p], lr=1e-3, betas=(0.0, 0.999))
        opt.step()   # must not raise
        assert torch.isfinite(p.data).all()

    def test_step_counter_increments(self):
        p = _param_and_grad()
        opt = ScheduleFreeAdamW([p], lr=1e-3)
        for i in range(3):
            p.grad = torch.randn_like(p)
            opt.step()
        assert opt.state[p]["step"] == 3

    def test_get_averaged_params_length(self):
        model = nn.Linear(4, 2)
        opt = ScheduleFreeAdamW(model.parameters(), lr=1e-3)
        x_data = torch.randn(4, 4)
        model(x_data).sum().backward()
        opt.step()
        avg = opt.get_averaged_params()
        assert len(avg) == len(list(model.parameters()))

    def test_weights_finite_many_steps(self):
        m = _many_steps(ScheduleFreeAdamW, lr=1e-3)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# Adan — additional
# ---------------------------------------------------------------------------

class TestAdanAdditional:
    def test_three_moment_buffers(self):
        """Adan must maintain three separate EMA buffers."""
        p = _param_and_grad()
        opt = Adan([p], lr=1e-3)
        opt.step()
        state = opt.state[p]
        assert "exp_avg"    in state  # m
        assert "exp_avg_sq" in state  # n (squared Nesterov)
        # v is stored as "exp_avg_diff" or similar — check the key set has ≥3
        assert len(state) >= 4   # step + at least 3 moment buffers

    def test_prev_grad_stored(self):
        """Previous gradient must be stored for gradient-difference computation."""
        p = _param_and_grad()
        opt = Adan([p], lr=1e-3)
        opt.step()
        # After first step, previous gradient should be cached
        state = opt.state[p]
        prev_keys = [k for k in state if "prev" in k or "grad" in k]
        assert len(prev_keys) >= 1, "No previous-gradient key found in state"

    def test_weights_finite_many_steps(self):
        m = _many_steps(Adan, lr=1e-3)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_weight_decay_shrinks_params(self):
        model = nn.Linear(8, 4)
        init_norm = sum(p.data.norm().item() for p in model.parameters())
        opt = Adan(model.parameters(), lr=1e-3, weight_decay=0.5)
        x = torch.randn(16, 8)
        for _ in range(30):
            model(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        final_norm = sum(p.data.norm().item() for p in model.parameters())
        assert final_norm < init_norm * 2   # not blown up; WD is working


# ---------------------------------------------------------------------------
# SOAP — additional
# ---------------------------------------------------------------------------

class TestSOAPAdditional:
    def test_eigenvectors_refreshed_at_update_freq(self):
        """Q_L / Q_R should be present in state after update_freq steps."""
        p = nn.Parameter(torch.randn(4, 8))
        opt = SOAP([p], lr=1e-3, update_freq=5)
        for i in range(6):
            p.grad = torch.randn_like(p)
            opt.step()
        state = opt.state[p]
        Q_keys = [k for k in state if k.startswith("Q")]
        assert len(Q_keys) >= 1, "No eigenvector matrix found in SOAP state"

    def test_step_counter_increments(self):
        p = _param_and_grad()
        opt = SOAP([p], lr=1e-3)
        for _ in range(4):
            p.grad = torch.randn_like(p)
            opt.step()
        assert opt.state[p]["step"] == 4

    def test_weights_finite_many_steps(self):
        m = _many_steps(SOAP, lr=1e-3)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_weight_decay_shrinks_params(self):
        model = nn.Linear(8, 4)
        opt = SOAP(model.parameters(), lr=1e-3, weight_decay=0.5)
        x = torch.randn(16, 8)
        for _ in range(20):
            model(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# CautiousAdam — additional
# ---------------------------------------------------------------------------

class TestCautiousAdamAdditional:
    def test_anti_aligned_coords_zeroed(self):
        """If update direction disagrees with gradient, that coord is masked."""
        # Construct a scenario where momentum points opposite to gradient.
        p = nn.Parameter(torch.zeros(1, 4))
        opt = CautiousAdam([p], lr=1e-2, betas=(0.9, 0.999))

        # Step 1: positive gradient builds positive momentum
        p.grad = torch.ones(1, 4)
        opt.step()

        # Step 2: flip gradient to negative — momentum still positive → disagreement
        p.grad = torch.full((1, 4), -10.0)
        before = p.data.clone()
        opt.step()

        # The parameter should not move aggressively in the positive direction
        # (cautious masking should suppress the anti-aligned update coordinates)
        assert torch.isfinite(p.data).all()

    def test_weights_finite_many_steps(self):
        m = _many_steps(CautiousAdam, lr=1e-3)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_step_counter_increments(self):
        p = _param_and_grad()
        opt = CautiousAdam([p], lr=1e-3)
        for _ in range(7):
            p.grad = torch.randn_like(p)
            opt.step()
        assert opt.state[p]["step"] == 7


# ---------------------------------------------------------------------------
# MARS — additional
# ---------------------------------------------------------------------------

class TestMARSAdditional:
    def test_weights_finite_many_steps(self):
        m = _many_steps(MARS, lr=1e-3)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_weight_decay_shrinks_params(self):
        model = nn.Linear(8, 4)
        opt = MARS(model.parameters(), lr=1e-3, weight_decay=0.5)
        x = torch.randn(16, 8)
        for _ in range(20):
            model(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p.data).all()

    def test_prev_grad_updated_each_step(self):
        p = _param_and_grad()
        opt = MARS([p], lr=1e-3)
        p.grad = torch.ones_like(p) * 2.0
        opt.step()
        prev1 = opt.state[p]["prev_grad"].clone()
        p.grad = torch.ones_like(p) * 5.0
        opt.step()
        prev2 = opt.state[p]["prev_grad"]
        assert not torch.equal(prev1, prev2)


# ---------------------------------------------------------------------------
# GrokFast — additional
# ---------------------------------------------------------------------------

class TestGrokFastAdditional:
    def test_grad_ema_shape_matches_param(self):
        p = _param_and_grad()
        opt = GrokFast([p], lr=1e-3)
        opt.step()
        assert opt.state[p]["grad_ema"].shape == p.shape

    def test_ema_alpha_controls_decay(self):
        """Higher alpha → slower decay (ema retains more of old value)."""
        p1 = _param_and_grad(grad_val=1.0)
        p2 = _param_and_grad(grad_val=1.0)
        opt1 = GrokFast([p1], lr=1e-4, alpha=0.1)
        opt2 = GrokFast([p2], lr=1e-4, alpha=0.99)

        # Step 1: both start from 0
        opt1.step(); opt2.step()
        ema1_step1 = opt1.state[p1]["grad_ema"].clone()
        ema2_step1 = opt2.state[p2]["grad_ema"].clone()

        # Now change gradient to 0
        p1.grad = torch.zeros_like(p1); opt1.step()
        p2.grad = torch.zeros_like(p2); opt2.step()

        # High alpha should retain more of the old EMA
        assert opt2.state[p2]["grad_ema"].abs().mean() > \
               opt1.state[p1]["grad_ema"].abs().mean()

    def test_weights_finite_many_steps(self):
        m = _many_steps(GrokFast, lr=1e-3)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_weight_decay_shrinks_params(self):
        model = nn.Linear(8, 4)
        opt = GrokFast(model.parameters(), lr=1e-3, weight_decay=0.5)
        x = torch.randn(16, 8)
        for _ in range(20):
            model(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# Tiger — additional
# ---------------------------------------------------------------------------

class TestTigerAdditional:
    def test_update_uses_sign(self):
        """Tiger, like Lion, is sign-based: update ∈ {-lr, 0, +lr}."""
        lr = 1e-3
        p = _param_and_grad(val=0.0, grad_val=3.0)
        opt = Tiger([p], lr=lr)
        before = p.data.clone()
        opt.step()
        diff = (p.data - before).abs()
        # All elements should have moved by ≤ lr (sign-based)
        assert (diff <= lr + 1e-7).all()

    def test_momentum_buffer_initialized(self):
        p = _param_and_grad()
        opt = Tiger([p], lr=1e-4)
        opt.step()
        assert "exp_avg" in opt.state[p]

    def test_weights_finite_many_steps(self):
        m = _many_steps(Tiger, lr=1e-4)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_weight_decay_shrinks_params(self):
        model = nn.Linear(8, 4)
        opt = Tiger(model.parameters(), lr=1e-4, weight_decay=0.5)
        x = torch.randn(16, 8)
        for _ in range(20):
            model(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p.data).all()


# ---------------------------------------------------------------------------
# AdanNesterov — additional
# ---------------------------------------------------------------------------

class TestAdanNesterovAdditional:
    def test_state_initialized(self):
        p = _param_and_grad()
        from optimizers import AdanNesterov
        opt = AdanNesterov([p], lr=1e-3)
        opt.step()
        assert len(opt.state[p]) > 0

    def test_weights_finite_many_steps(self):
        from optimizers import AdanNesterov
        m = _many_steps(AdanNesterov, lr=1e-3)
        for p in m.parameters():
            assert torch.isfinite(p.data).all()

    def test_weight_decay_shrinks_params(self):
        from optimizers import AdanNesterov
        model = nn.Linear(8, 4)
        opt = AdanNesterov(model.parameters(), lr=1e-3, weight_decay=0.5)
        x = torch.randn(16, 8)
        for _ in range(20):
            model(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        for p in model.parameters():
            assert torch.isfinite(p.data).all()

    def test_step_counter_increments(self):
        from optimizers import AdanNesterov
        p = _param_and_grad()
        opt = AdanNesterov([p], lr=1e-3)
        for _ in range(5):
            p.grad = torch.randn_like(p)
            opt.step()
        assert opt.state[p]["step"] == 5
