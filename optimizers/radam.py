"""Custom RAdam optimizer.

From "On the Variance of the Adaptive Learning Rate and Beyond"
(Liu et al., ICLR 2020).  https://arxiv.org/abs/1908.03265

Motivation
----------
Adam's second moment estimate v_t is initialised to zero.  In the first few
training steps, v_t is very small — not because the gradient variance is small,
but because the EMA hasn't had time to accumulate.  Dividing by √v_t when v_t
is near zero produces an extremely large effective learning rate, which can
cause the model to diverge or get stuck in a bad local minimum right at the
start of training.

A common workaround is learning-rate warmup: manually ramp α from 0 over the
first N steps.  RAdam avoids the need for a separate warmup schedule by
*automatically* detecting whether the adaptive learning rate has sufficiently
low variance (using a closed-form approximation to ρ, the degrees of freedom
of the Student-t distribution of the second moment).

When ρ_t is large enough (reliable estimate), RAdam applies the full adaptive
update.  When ρ_t is too small (unreliable second moment), RAdam falls back to
SGD with first-moment momentum — exactly like a warm-up phase, but derived
from the statistics rather than a manually chosen schedule.

Algorithm
---------
Precompute: ρ_∞ = 2 / (1 − β₂) − 1   (maximum achievable degrees of freedom)

For each parameter p with gradient g at step t:

  1. m_t = β₁ · m_{t-1} + (1 − β₁) · g_t          [first moment]
  2. v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²          [second moment]
  3. m̂_t = m_t / (1 − β₁^t)                         [bias-corrected mean]
  4. ρ_t = ρ_∞ − 2·t·β₂^t / (1 − β₂^t)             [approx. SMA length]

  If ρ_t > 5 (second moment is reliable):
    5a. r_t = √[(ρ_t−4)(ρ_t−2)ρ_∞ / ((ρ_∞−4)(ρ_∞−2)ρ_t)]   [variance correction]
    5b. v̂_t = v_t / (1 − β₂^t)
    5c. θ_t = θ_{t-1} − α · r_t · m̂_t / (√v̂_t + ε)
  Else (unreliable; fall back to SGD momentum):
    5d. θ_t = θ_{t-1} − α · m̂_t

Practical tips
--------------
  • RAdam eliminates the need for a separate warmup schedule in most cases.
  • Works especially well at large learning rates where Adam would diverge.
  • Weight decay (coupled L2 form) is applied before the moment update.
"""

import math

import torch
from .base import BaseOptimizer


class RAdam(BaseOptimizer):
    """RAdam: Rectified Adam optimizer.

    Parameters
    ----------
    params       : iterable of parameters or param groups
    lr           : global learning rate α (default 1e-3)
    betas        : (β₁, β₂) — EMA decay rates (default (0.9, 0.999))
    eps          : numerical stability constant (default 1e-8)
    weight_decay : L2 regularisation (coupled form, default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one RAdam step across all parameter groups.

        Returns
        -------
        float | None
            Loss from the closure, or None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr            = group["lr"]
            beta1, beta2  = group["betas"]
            eps           = group["eps"]
            wd            = group["weight_decay"]

            # ρ_∞: the maximum SMA length (degrees of freedom when t → ∞).
            # Derived from the stationary distribution of the second moment EMA.
            # Formula: ρ_∞ = 2/(1−β₂) − 1
            rho_inf = 2.0 / (1.0 - beta2) - 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    state["step"]       = 0
                    state["exp_avg"]    = torch.zeros_like(p)  # m
                    state["exp_avg_sq"] = torch.zeros_like(p)  # v

                state["step"] += 1
                t    = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # ── Coupled weight decay ──────────────────────────────────
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # ── Steps 1 & 2: Moment updates ───────────────────────────
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # ── Step 3: Bias-corrected first moment ───────────────────
                bias_corr1 = 1.0 - beta1 ** t
                m_hat      = m / bias_corr1

                # ── Step 4: Compute approximate SMA length ────────────────
                # ρ_t estimates how many effective samples the second moment
                # EMA represents after t steps.
                # Formula: ρ_t = ρ_∞ − 2·t·β₂^t / (1 − β₂^t)
                beta2_t    = beta2 ** t
                bias_corr2 = 1.0 - beta2_t
                rho_t      = rho_inf - 2.0 * t * beta2_t / bias_corr2

                # ── Step 5a–c: Adaptive update (variance is reliable) ─────
                if rho_t > 5.0:
                    # The second moment has enough samples to be a reliable
                    # variance estimate.  Use the full adaptive update with
                    # a variance correction factor r_t.

                    # r_t corrects for the bias in √v̂_t as a variance estimator.
                    # Derived from the ratio of expected values of the
                    # actual vs. asymptotic second moment distribution.
                    r_t = math.sqrt(
                        (rho_t - 4) * (rho_t - 2) * rho_inf
                        / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )

                    # Bias-corrected second moment
                    v_hat = v / bias_corr2

                    # Adaptive update: θ ← θ − α · r_t · m̂ / (√v̂ + ε)
                    denom = v_hat.sqrt().add_(eps)
                    p.addcdiv_(m_hat, denom, value=-lr * r_t)

                else:
                    # ── Step 5d: SGD-momentum fallback ────────────────────
                    # Second moment is too noisy — skip the adaptive scaling.
                    # This is effectively warm-up: gradient magnitude is not
                    # normalised, so the learning rate is smaller in effect.
                    p.add_(m_hat, alpha=-lr)

        return loss
