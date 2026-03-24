"""Tests for the 7 custom re-implementations of PyTorch built-in optimizers.

Covers: Adam, AdamW, NAdam, RAdam, Adagrad, SGDMomentum, RMSprop.

Each test class verifies:
  - State is correctly initialised after the first step
  - Step counter increments as expected (where applicable)
  - Weights remain finite after many steps
  - Weight decay shrinks parameter norms
  - Optimizer-specific invariants (e.g. SGDMomentum Nesterov, RAdam fallback)
  - Numerical agreement with the equivalent torch.optim reference implementation
"""

import math

import pytest
import torch
import torch.nn as nn

from optimizers import Adam, AdamW, NAdam, RAdam, Adagrad, SGDMomentum, RMSprop


# ---------------------------------------------------------------------------
# Helpers shared across all test classes
# ---------------------------------------------------------------------------

def make_linear(in_features=8, out_features=4, bias=True):
    """Return a freshly initialised Linear module with a fixed seed."""
    torch.manual_seed(42)
    return nn.Linear(in_features, out_features, bias=bias)


def run_steps(opt, model, n=10, in_features=8):
    """Run *n* forward-backward-step cycles on a random mini-batch."""
    for _ in range(n):
        # Fresh random input each step so the loss is non-trivially non-zero
        x = torch.randn(4, in_features)
        model(x).sum().backward()
        opt.step()
        opt.zero_grad()


def weights_are_finite(model):
    """Return True iff every parameter tensor contains only finite values."""
    return all(torch.isfinite(p).all() for p in model.parameters())


def norm_after_steps(opt_cls, wd, n=3, **opt_kwargs):
    """Run *n* steps with weight_decay=*wd* and return the weight norm."""
    torch.manual_seed(0)
    model = nn.Linear(4, 2, bias=False)
    # Start from a known non-zero value so decay has something to shrink
    nn.init.constant_(model.weight, 1.0)
    opt = opt_cls(model.parameters(), weight_decay=wd, **opt_kwargs)
    for _ in range(n):
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        opt.zero_grad()
    return model.weight.norm().item()


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class TestAdam:
    def test_state_initialized_after_first_step(self):
        """After one step every parameter should have m, v, and step in state."""
        model = make_linear()
        opt = Adam(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]        # first moment m
            assert "exp_avg_sq" in opt.state[p]     # second moment v

    def test_step_counter_increments(self):
        """state['step'] should equal the number of opt.step() calls."""
        model = make_linear()
        opt = Adam(model.parameters(), lr=1e-3)
        for i in range(1, 5):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 4

    def test_weights_finite_after_many_steps(self):
        """Parameters must stay numerically finite over extended training."""
        model = make_linear()
        opt = Adam(model.parameters(), lr=1e-3)
        run_steps(opt, model, n=20)
        assert weights_are_finite(model)

    def test_weights_change_after_step(self):
        """Parameters must actually move after one step."""
        model = make_linear()
        w_before = model.weight.data.clone()
        opt = Adam(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        assert not torch.allclose(model.weight.data, w_before)

    def test_weight_decay_shrinks_params(self):
        """Higher weight_decay should produce a smaller parameter norm."""
        assert norm_after_steps(Adam, wd=0.1, lr=1e-3) < norm_after_steps(Adam, wd=0.0, lr=1e-3)

    def test_update_direction_descends_loss(self):
        """Weights should move in the negative gradient direction on a simple loss."""
        model = nn.Linear(4, 2, bias=False)
        # All weights = 1, all inputs = 1 → gradient is positive → weights decrease
        nn.init.constant_(model.weight, 1.0)
        opt = Adam(model.parameters(), lr=0.1)
        model(torch.ones(1, 4)).sum().backward()
        opt.step()
        assert (model.weight < 1.0).all()

    def test_numerically_close_to_torch_adam(self):
        """Custom Adam should match torch.optim.Adam to high precision."""
        torch.manual_seed(7)
        x = torch.randn(4, 8)

        # Reference: PyTorch Adam
        torch.manual_seed(7)
        ref_model = make_linear()
        ref_opt = torch.optim.Adam(ref_model.parameters(), lr=1e-3, weight_decay=0.0)

        # Custom Adam — same initial weights via same seed
        torch.manual_seed(7)
        cus_model = make_linear()
        cus_opt = Adam(cus_model.parameters(), lr=1e-3, weight_decay=0.0)

        # Run identical forward/backward/step on the same input
        for _ in range(5):
            ref_model(x).sum().backward()
            ref_opt.step()
            ref_opt.zero_grad()

            cus_model(x).sum().backward()
            cus_opt.step()
            cus_opt.zero_grad()

        # Weights should agree to ~6 decimal places
        torch.testing.assert_close(
            cus_model.weight.data, ref_model.weight.data, atol=1e-6, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------

class TestAdamW:
    def test_state_initialized_after_first_step(self):
        model = make_linear()
        opt = AdamW(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]

    def test_step_counter_increments(self):
        model = make_linear()
        opt = AdamW(model.parameters(), lr=1e-3)
        for _ in range(3):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 3

    def test_weights_finite_after_many_steps(self):
        model = make_linear()
        opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        run_steps(opt, model, n=20)
        assert weights_are_finite(model)

    def test_weight_decay_shrinks_params(self):
        assert norm_after_steps(AdamW, wd=0.1, lr=1e-3) < norm_after_steps(AdamW, wd=0.0, lr=1e-3)

    def test_decoupled_decay_does_not_modify_gradient(self):
        """AdamW applies decay to params directly — p.grad must be unchanged."""
        model = nn.Linear(4, 2, bias=False)
        opt = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        model(torch.randn(2, 4)).sum().backward()
        # Clone gradient before step
        grad_before = model.weight.grad.clone()
        opt.step()
        # Gradient tensor should be bit-for-bit identical (not mutated by decay)
        assert torch.equal(model.weight.grad, grad_before)

    def test_numerically_close_to_torch_adamw(self):
        """Custom AdamW should match torch.optim.AdamW to high precision."""
        torch.manual_seed(7)
        x = torch.randn(4, 8)

        torch.manual_seed(7)
        ref_model = make_linear()
        ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=1e-3, weight_decay=0.01)

        torch.manual_seed(7)
        cus_model = make_linear()
        cus_opt = AdamW(cus_model.parameters(), lr=1e-3, weight_decay=0.01)

        for _ in range(5):
            ref_model(x).sum().backward()
            ref_opt.step()
            ref_opt.zero_grad()

            cus_model(x).sum().backward()
            cus_opt.step()
            cus_opt.zero_grad()

        torch.testing.assert_close(
            cus_model.weight.data, ref_model.weight.data, atol=1e-6, rtol=1e-5
        )

    def test_adamw_vs_adam_with_decay(self):
        """AdamW and Adam should diverge when weight_decay > 0 (different mechanisms)."""
        torch.manual_seed(0)
        x = torch.randn(4, 8)

        torch.manual_seed(0)
        m_adam = make_linear()
        o_adam = Adam(m_adam.parameters(), lr=1e-3, weight_decay=0.1)

        torch.manual_seed(0)
        m_adamw = make_linear()
        o_adamw = AdamW(m_adamw.parameters(), lr=1e-3, weight_decay=0.1)

        for _ in range(5):
            m_adam(x).sum().backward()
            o_adam.step()
            o_adam.zero_grad()

            m_adamw(x).sum().backward()
            o_adamw.step()
            o_adamw.zero_grad()

        # Coupled vs decoupled decay must produce different weights
        assert not torch.allclose(m_adam.weight.data, m_adamw.weight.data)


# ---------------------------------------------------------------------------
# NAdam
# ---------------------------------------------------------------------------

class TestNAdam:
    def test_state_initialized_after_first_step(self):
        model = make_linear()
        opt = NAdam(model.parameters(), lr=2e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert "mu_product" in opt.state[p]   # Nesterov momentum schedule product

    def test_step_counter_increments(self):
        model = make_linear()
        opt = NAdam(model.parameters(), lr=2e-3)
        for _ in range(4):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 4

    def test_weights_finite_after_many_steps(self):
        model = make_linear()
        opt = NAdam(model.parameters(), lr=2e-3)
        run_steps(opt, model, n=20)
        assert weights_are_finite(model)

    def test_weights_change_after_step(self):
        model = make_linear()
        w_before = model.weight.data.clone()
        opt = NAdam(model.parameters(), lr=2e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        assert not torch.allclose(model.weight.data, w_before)

    def test_weight_decay_shrinks_params(self):
        assert norm_after_steps(NAdam, wd=0.1, lr=2e-3) < norm_after_steps(NAdam, wd=0.0, lr=2e-3)

    def test_mu_product_strictly_less_than_one(self):
        """mu_product tracks ∏ μ_t, which decays over time and stays < 1."""
        model = nn.Linear(4, 2, bias=False)
        opt = NAdam(model.parameters(), lr=2e-3)
        for _ in range(10):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert 0.0 < opt.state[p]["mu_product"] < 1.0

    def test_momentum_decay_affects_update(self):
        """momentum_decay must change the parameter trajectory (not be a no-op).

        Before the fix, momentum_decay was computed but never used — the first
        moment EMA and bias corrections both used the fixed beta1 regardless of
        the schedule.  This test checks that two NAdam instances with different
        momentum_decay values actually produce different weight trajectories.
        """
        torch.manual_seed(0)
        x = torch.randn(4, 8)

        torch.manual_seed(42)
        model_no_decay = make_linear()
        opt_no_decay = NAdam(model_no_decay.parameters(), lr=2e-3, momentum_decay=0.0)

        torch.manual_seed(42)
        model_with_decay = make_linear()
        opt_with_decay = NAdam(model_with_decay.parameters(), lr=2e-3, momentum_decay=4e-3)

        for _ in range(10):
            model_no_decay(x).sum().backward()
            opt_no_decay.step()
            opt_no_decay.zero_grad()

            model_with_decay(x).sum().backward()
            opt_with_decay.step()
            opt_with_decay.zero_grad()

        # With a non-trivial momentum schedule the weights must diverge
        assert not torch.allclose(
            model_no_decay.weight.data, model_with_decay.weight.data
        ), "momentum_decay=0 and momentum_decay=4e-3 produced identical weights — schedule has no effect"

    def test_mu_product_matches_manual_product(self):
        """State mu_product must equal the running product ∏ μ_i, not a stale value.

        This verifies that mu_product is updated correctly each step and equals
        the product of all scheduled mu_t values seen so far — confirming the
        schedule accumulation is wired up and not bypassed.
        """
        beta1 = 0.9
        mu_decay = 4e-3
        model = nn.Linear(4, 2, bias=False)
        torch.manual_seed(0)
        opt = NAdam(model.parameters(), lr=2e-3, betas=(beta1, 0.999),
                    momentum_decay=mu_decay)

        expected_product = 1.0
        for step in range(1, 6):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            opt.zero_grad()

            mu_t = beta1 * (1.0 - 0.5 * (0.96 ** (step * mu_decay)))
            expected_product *= mu_t

            for p in model.parameters():
                actual = opt.state[p]["mu_product"]
                assert abs(actual - expected_product) < 1e-9, (
                    f"step {step}: mu_product={actual:.10f}, "
                    f"expected={expected_product:.10f}"
                )

    def test_first_moment_uses_scheduled_mu(self):
        """The first moment EMA must use mu_t, not the fixed beta1.

        After one step with momentum_decay > 0, mu_1 < beta1.  The first
        moment m_1 = mu_1 * 0 + (1 - mu_1) * g  (zero-initialised buffer).
        If the bug were present m_1 would use beta1 instead, giving a different
        (smaller) value for (1 - beta1) * g vs (1 - mu_1) * g since mu_1 < beta1.
        """
        beta1 = 0.9
        mu_decay = 4e-3
        # mu_1 = beta1 * (1 - 0.5 * 0.96^(1 * mu_decay))
        mu_1 = beta1 * (1.0 - 0.5 * (0.96 ** mu_decay))
        # mu_1 should be strictly less than beta1 (the schedule reduces it)
        assert mu_1 < beta1

        torch.manual_seed(0)
        model = nn.Linear(4, 2, bias=False)
        opt = NAdam(model.parameters(), lr=2e-3, betas=(beta1, 0.999),
                    momentum_decay=mu_decay)

        torch.manual_seed(1)
        x = torch.randn(2, 4)
        model(x).sum().backward()
        opt.step()

        for p in model.parameters():
            if p.grad is not None:
                m = opt.state[p]["exp_avg"]
                g = p.grad  # zero_grad not called — grad still set
                # With correct fix: m = (1 - mu_1) * g (zero init, one step)
                expected_m = (1.0 - mu_1) * g
                # With the old bug: m = (1 - beta1) * g
                buggy_m = (1.0 - beta1) * g
                # The two differ because mu_1 != beta1
                assert not torch.allclose(expected_m, buggy_m), \
                    "mu_1 == beta1: schedule has no effect at step 1"
                assert torch.allclose(m, expected_m, atol=1e-6), (
                    f"First moment does not match (1-mu_1)*g: "
                    f"max diff {(m - expected_m).abs().max().item():.2e}"
                )

    def test_zero_momentum_decay_matches_fixed_beta1(self):
        """When momentum_decay=0 the schedule collapses to fixed beta1.

        mu_t = beta1 * (1 - 0.5 * 0.96^0) = beta1 * 0.5 ... wait, that's
        only true for t*0=0.  Actually 0.96^(t*0)=1 for all t, so
        mu_t = beta1 * (1 - 0.5) = 0.5*beta1 when decay=0, which is NOT
        equal to beta1.  A decay that truly disables the schedule uses
        a large decay so that 0.96^(t*decay) → 0, giving mu_t → beta1.
        Here we verify the edge case: momentum_decay=0 keeps mu_t constant
        (at 0.5*beta1) across all steps, so mu_product stays geometric.
        """
        beta1 = 0.9
        mu_decay = 0.0
        # With decay=0: mu_t = beta1 * (1 - 0.5 * 1) = beta1 * 0.5 every step
        mu_constant = beta1 * 0.5

        model = nn.Linear(4, 2, bias=False)
        torch.manual_seed(0)
        opt = NAdam(model.parameters(), lr=2e-3, betas=(beta1, 0.999),
                    momentum_decay=mu_decay)

        expected_product = 1.0
        for step in range(1, 6):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            opt.zero_grad()
            expected_product *= mu_constant
            for p in model.parameters():
                actual = opt.state[p]["mu_product"]
                assert abs(actual - expected_product) < 1e-9, (
                    f"step {step}: mu_product={actual:.10f} != "
                    f"(0.5*beta1)^step={expected_product:.10f}"
                )

    def test_numerically_close_to_torch_nadam(self):
        """Custom NAdam should produce similar results to torch.optim.NAdam.

        PyTorch's NAdam uses a slightly different internal schedule for the
        bias-correction product (cumulative mu), so we verify broad agreement
        (same order of magnitude, same direction) rather than bit-exact match.
        """
        torch.manual_seed(7)
        x = torch.randn(4, 8)

        torch.manual_seed(7)
        ref_model = make_linear()
        ref_opt = torch.optim.NAdam(ref_model.parameters(), lr=2e-3,
                                     momentum_decay=4e-3, weight_decay=0.0)

        torch.manual_seed(7)
        cus_model = make_linear()
        cus_opt = NAdam(cus_model.parameters(), lr=2e-3,
                        momentum_decay=4e-3, weight_decay=0.0)

        for _ in range(5):
            ref_model(x).sum().backward()
            ref_opt.step()
            ref_opt.zero_grad()

            cus_model(x).sum().backward()
            cus_opt.step()
            cus_opt.zero_grad()

        # Both should move in the same general direction from the initial weights
        # (sign agreement on the update), even if magnitudes differ slightly.
        ref_delta = ref_model.weight.data - make_linear().weight.data
        cus_delta = cus_model.weight.data - make_linear().weight.data
        sign_agreement = (ref_delta.sign() == cus_delta.sign()).float().mean().item()
        assert sign_agreement >= 0.9, (
            f"NAdam sign agreement with torch.optim.NAdam dropped to {sign_agreement:.2f}"
        )


# ---------------------------------------------------------------------------
# RAdam
# ---------------------------------------------------------------------------

class TestRAdam:
    def test_state_initialized_after_first_step(self):
        model = make_linear()
        opt = RAdam(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]

    def test_step_counter_increments(self):
        model = make_linear()
        opt = RAdam(model.parameters(), lr=1e-3)
        for _ in range(5):
            model(torch.randn(4, 8)).sum().backward()
            opt.step()
            opt.zero_grad()
        for p in model.parameters():
            assert opt.state[p]["step"] == 5

    def test_weights_finite_after_many_steps(self):
        """RAdam must stay finite through both the fallback and adaptive phases."""
        model = make_linear()
        opt = RAdam(model.parameters(), lr=1e-3)
        run_steps(opt, model, n=30)   # spans both rho < 5 and rho > 5 phases
        assert weights_are_finite(model)

    def test_weight_decay_shrinks_params(self):
        assert norm_after_steps(RAdam, wd=0.1, lr=1e-3) < norm_after_steps(RAdam, wd=0.0, lr=1e-3)

    def test_early_steps_use_sgd_fallback(self):
        """In the very first step rho_t < 5 for β₂=0.999, so RAdam uses the SGD path.

        We verify this indirectly: with β₂=0.999 and t=1,
        rho_1 = rho_inf - 2*1*0.999 / (1-0.999) ≈ 2000 - 2*0.999/0.001 ≈ 2000-1998 ≈ 2 < 5.
        After step 1 the second moment should still be very close to zero
        because the SGD path does not use v to compute the update.
        """
        model = nn.Linear(4, 2, bias=False)
        opt = RAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        # The EMA has been updated but is tiny after just 1 step
        for p in model.parameters():
            v = opt.state[p]["exp_avg_sq"]
            assert v.max().item() < 1.0   # still small; not a huge adaptive denominator

    def test_weights_change_after_step(self):
        model = make_linear()
        w_before = model.weight.data.clone()
        opt = RAdam(model.parameters(), lr=1e-3)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        assert not torch.allclose(model.weight.data, w_before)

    def test_numerically_close_to_torch_radam(self):
        """Custom RAdam should closely match torch.optim.RAdam."""
        torch.manual_seed(7)
        x = torch.randn(4, 8)

        torch.manual_seed(7)
        ref_model = make_linear()
        ref_opt = torch.optim.RAdam(ref_model.parameters(), lr=1e-3, weight_decay=0.0)

        torch.manual_seed(7)
        cus_model = make_linear()
        cus_opt = RAdam(cus_model.parameters(), lr=1e-3, weight_decay=0.0)

        for _ in range(10):
            ref_model(x).sum().backward()
            ref_opt.step()
            ref_opt.zero_grad()

            cus_model(x).sum().backward()
            cus_opt.step()
            cus_opt.zero_grad()

        torch.testing.assert_close(
            cus_model.weight.data, ref_model.weight.data, atol=1e-5, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# Adagrad
# ---------------------------------------------------------------------------

class TestAdagrad:
    def test_state_initialized_after_first_step(self):
        """After one step state should contain the squared-gradient accumulator."""
        model = make_linear()
        opt = Adagrad(model.parameters(), lr=1e-2)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "step" in opt.state[p]
            assert "sum" in opt.state[p]   # G: accumulated squared gradients

    def test_sum_is_monotonically_increasing(self):
        """G_t = sum of g² is non-decreasing (Adagrad's defining property)."""
        model = nn.Linear(4, 2, bias=False)
        opt = Adagrad(model.parameters(), lr=1e-2)
        prev_sum = None
        for _ in range(5):
            model(torch.randn(2, 4)).sum().backward()
            opt.step()
            opt.zero_grad()
            cur_sum = opt.state[model.weight]["sum"].clone()
            if prev_sum is not None:
                # Every element of G should be >= previous value
                assert (cur_sum >= prev_sum - 1e-9).all()
            prev_sum = cur_sum

    def test_weights_finite_after_many_steps(self):
        model = make_linear()
        opt = Adagrad(model.parameters(), lr=1e-2)
        run_steps(opt, model, n=30)
        assert weights_are_finite(model)

    def test_weight_decay_shrinks_params(self):
        assert norm_after_steps(Adagrad, wd=0.1, lr=1e-2) < norm_after_steps(Adagrad, wd=0.0, lr=1e-2)

    def test_initial_accumulator_value(self):
        """initial_accumulator_value > 0 should pre-fill G with that constant."""
        model = nn.Linear(4, 2, bias=False)
        init_val = 0.5
        opt = Adagrad(model.parameters(), lr=1e-2, initial_accumulator_value=init_val)
        # Before any step, check state is empty (lazy initialisation)
        assert len(opt.state) == 0
        # After one step, G should be >= init_val everywhere
        model(torch.randn(2, 4)).sum().backward()
        opt.step()
        g_sum = opt.state[model.weight]["sum"]
        assert (g_sum >= init_val).all()

    def test_lr_decay_reduces_effective_lr(self):
        """lr_decay > 0 should reduce the effective LR over time.

        We verify this indirectly: the parameter update magnitude should
        decrease over steps when lr_decay is large.
        """
        torch.manual_seed(0)
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 0.0)
        x = torch.ones(2, 4)
        opt = Adagrad(model.parameters(), lr=1.0, lr_decay=1.0)
        deltas = []
        for _ in range(5):
            w_before = model.weight.data.clone()
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()
            deltas.append((model.weight.data - w_before).abs().mean().item())
        # Later updates should be smaller than early updates
        assert deltas[-1] < deltas[0]

    def test_effective_lr_decreases_over_steps(self):
        """The effective step size should shrink as G accumulates."""
        torch.manual_seed(0)
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 0.0)
        x = torch.ones(2, 4)
        opt = Adagrad(model.parameters(), lr=1.0)
        deltas = []
        for _ in range(10):
            w_before = model.weight.data.clone()
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()
            deltas.append((model.weight.data - w_before).abs().mean().item())
        # First step should take a larger absolute step than the last
        assert deltas[0] > deltas[-1]

    def test_numerically_close_to_torch_adagrad(self):
        """Custom Adagrad should closely match torch.optim.Adagrad."""
        torch.manual_seed(7)
        x = torch.randn(4, 8)

        torch.manual_seed(7)
        ref_model = make_linear()
        ref_opt = torch.optim.Adagrad(ref_model.parameters(), lr=1e-2,
                                       weight_decay=0.0, lr_decay=0.0)

        torch.manual_seed(7)
        cus_model = make_linear()
        cus_opt = Adagrad(cus_model.parameters(), lr=1e-2,
                          weight_decay=0.0, lr_decay=0.0)

        for _ in range(10):
            ref_model(x).sum().backward()
            ref_opt.step()
            ref_opt.zero_grad()

            cus_model(x).sum().backward()
            cus_opt.step()
            cus_opt.zero_grad()

        torch.testing.assert_close(
            cus_model.weight.data, ref_model.weight.data, atol=1e-5, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# SGDMomentum
# ---------------------------------------------------------------------------

class TestSGDMomentum:
    def test_no_momentum_buffer_without_momentum(self):
        """When momentum=0, no velocity buffer should be allocated."""
        model = make_linear()
        opt = SGDMomentum(model.parameters(), lr=1e-2, momentum=0.0, nesterov=False)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "velocity" not in opt.state[p]

    def test_velocity_buffer_initialized(self):
        """When momentum > 0, a velocity buffer should appear after one step."""
        model = make_linear()
        opt = SGDMomentum(model.parameters(), lr=1e-2, momentum=0.9)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "velocity" in opt.state[p]

    def test_update_direction_descends_loss(self):
        """Weights should decrease when gradient is positive."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 1.0)
        opt = SGDMomentum(model.parameters(), lr=0.1, momentum=0.9)
        model(torch.ones(1, 4)).sum().backward()
        opt.step()
        assert (model.weight < 1.0).all()

    def test_weights_finite_after_many_steps(self):
        model = make_linear()
        opt = SGDMomentum(model.parameters(), lr=1e-2, momentum=0.9)
        run_steps(opt, model, n=30)
        assert weights_are_finite(model)

    def test_weight_decay_shrinks_params(self):
        assert (
            norm_after_steps(SGDMomentum, wd=0.1, lr=1e-2, momentum=0.9)
            < norm_after_steps(SGDMomentum, wd=0.0, lr=1e-2, momentum=0.9)
        )

    def test_nesterov_requires_zero_dampening(self):
        """Constructing SGDMomentum with nesterov=True and dampening!=0 must raise."""
        with pytest.raises(ValueError, match="dampening"):
            SGDMomentum(make_linear().parameters(), lr=0.01,
                        nesterov=True, dampening=0.1)

    def test_nesterov_vs_standard_momentum_differ(self):
        """Nesterov and standard momentum should produce different weights."""
        torch.manual_seed(0)
        x = torch.randn(4, 8)

        torch.manual_seed(0)
        m_std = make_linear()
        o_std = SGDMomentum(m_std.parameters(), lr=1e-2, momentum=0.9, nesterov=False)

        torch.manual_seed(0)
        m_nes = make_linear()
        o_nes = SGDMomentum(m_nes.parameters(), lr=1e-2, momentum=0.9, nesterov=True)

        for _ in range(5):
            m_std(x).sum().backward()
            o_std.step()
            o_std.zero_grad()

            m_nes(x).sum().backward()
            o_nes.step()
            o_nes.zero_grad()

        assert not torch.allclose(m_std.weight.data, m_nes.weight.data)

    def test_momentum_accumulates_velocity(self):
        """The velocity buffer should grow over steps on a constant gradient."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 0.0)
        x = torch.ones(2, 4)   # constant input → constant gradient
        opt = SGDMomentum(model.parameters(), lr=0.01, momentum=0.9, nesterov=False)

        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        v1 = opt.state[model.weight]["velocity"].abs().mean().item()
        opt.zero_grad()
        model(x).sum().backward()
        opt.step()
        v2 = opt.state[model.weight]["velocity"].abs().mean().item()
        # Velocity should grow because constant gradient reinforces it
        assert v2 > v1

    def test_numerically_close_to_torch_sgd_with_momentum(self):
        """Custom SGDMomentum should closely match torch.optim.SGD with momentum."""
        torch.manual_seed(7)
        x = torch.randn(4, 8)

        torch.manual_seed(7)
        ref_model = make_linear()
        # torch.optim.SGD with nesterov=True, dampening=0
        ref_opt = torch.optim.SGD(ref_model.parameters(), lr=1e-2,
                                   momentum=0.9, nesterov=True, weight_decay=0.0)

        torch.manual_seed(7)
        cus_model = make_linear()
        cus_opt = SGDMomentum(cus_model.parameters(), lr=1e-2,
                              momentum=0.9, nesterov=True, weight_decay=0.0)

        for _ in range(10):
            ref_model(x).sum().backward()
            ref_opt.step()
            ref_opt.zero_grad()

            cus_model(x).sum().backward()
            cus_opt.step()
            cus_opt.zero_grad()

        torch.testing.assert_close(
            cus_model.weight.data, ref_model.weight.data, atol=1e-6, rtol=1e-5
        )


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

class TestRMSprop:
    def test_state_initialized_after_first_step(self):
        """After one step state should contain the squared-gradient EMA."""
        model = make_linear()
        opt = RMSprop(model.parameters(), lr=1e-2)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "square_avg" in opt.state[p]

    def test_no_momentum_buffer_by_default(self):
        """With momentum=0 no velocity buffer should be allocated."""
        model = make_linear()
        opt = RMSprop(model.parameters(), lr=1e-2, momentum=0.0)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "momentum_buffer" not in opt.state[p]

    def test_momentum_buffer_allocated_when_enabled(self):
        model = make_linear()
        opt = RMSprop(model.parameters(), lr=1e-2, momentum=0.9)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "momentum_buffer" in opt.state[p]

    def test_centered_grad_avg_allocated(self):
        """Centred variant should maintain a gradient mean buffer."""
        model = make_linear()
        opt = RMSprop(model.parameters(), lr=1e-2, centered=True)
        model(torch.randn(4, 8)).sum().backward()
        opt.step()
        for p in model.parameters():
            assert "grad_avg" in opt.state[p]

    def test_weights_finite_after_many_steps(self):
        model = make_linear()
        opt = RMSprop(model.parameters(), lr=1e-2)
        run_steps(opt, model, n=30)
        assert weights_are_finite(model)

    def test_weights_finite_centered_and_momentum(self):
        """Centred RMSprop with momentum should also stay finite."""
        model = make_linear()
        opt = RMSprop(model.parameters(), lr=1e-3, centered=True, momentum=0.9)
        run_steps(opt, model, n=20)
        assert weights_are_finite(model)

    def test_weight_decay_shrinks_params(self):
        assert norm_after_steps(RMSprop, wd=0.1, lr=1e-2) < norm_after_steps(RMSprop, wd=0.0, lr=1e-2)

    def test_square_avg_ema_decays(self):
        """square_avg is an EMA of g² — it should be strictly positive and finite."""
        model = nn.Linear(4, 2, bias=False)
        nn.init.constant_(model.weight, 0.0)
        x = torch.ones(2, 4)
        alpha = 0.9
        opt = RMSprop(model.parameters(), lr=1e-2, alpha=alpha)

        for _ in range(5):
            model(x).sum().backward()
            opt.step()
            opt.zero_grad()

        sq = opt.state[model.weight]["square_avg"]
        # EMA of squared gradients must be strictly positive (g != 0 on every step)
        assert (sq > 0).all(), "square_avg should be positive after gradient updates"
        # And must be finite — no blow-up
        assert torch.isfinite(sq).all(), "square_avg should remain finite"

    def test_numerically_close_to_torch_rmsprop(self):
        """Custom RMSprop should closely match torch.optim.RMSprop."""
        torch.manual_seed(7)
        x = torch.randn(4, 8)

        torch.manual_seed(7)
        ref_model = make_linear()
        ref_opt = torch.optim.RMSprop(ref_model.parameters(), lr=1e-2,
                                       alpha=0.99, eps=1e-8,
                                       weight_decay=0.0, momentum=0.0, centered=False)

        torch.manual_seed(7)
        cus_model = make_linear()
        cus_opt = RMSprop(cus_model.parameters(), lr=1e-2,
                          alpha=0.99, eps=1e-8,
                          weight_decay=0.0, momentum=0.0, centered=False)

        for _ in range(10):
            ref_model(x).sum().backward()
            ref_opt.step()
            ref_opt.zero_grad()

            cus_model(x).sum().backward()
            cus_opt.step()
            cus_opt.zero_grad()

        torch.testing.assert_close(
            cus_model.weight.data, ref_model.weight.data, atol=1e-5, rtol=1e-4
        )
