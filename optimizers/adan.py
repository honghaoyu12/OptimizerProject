"""Adan optimizer.

From "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing
Deep Models" (Xie et al., 2022).  https://arxiv.org/abs/2208.06677

Motivation
----------
Nesterov momentum (NAG) is theoretically faster than classical momentum on
smooth convex problems because it evaluates the gradient at the "look-ahead"
point θ + β·Δθ rather than the current point.  However, combining NAG with
adaptive step sizes (like Adam) is non-trivial — Adam's EMA update must be
reformulated to avoid a circular dependency between the look-ahead point and
the gradient used to update the moments.

Adan solves this by tracking the *gradient difference* Δg_t = g_t − g_{t-1}
as a direct proxy for gradient curvature (the finite-difference Hessian-vector
product along the training trajectory).  Three independent EMAs capture:

  m  — gradient trend (where are we heading?)
  v  — gradient curvature (how fast is the direction changing?)
  n  — squared Nesterov gradient  (how large is the adaptive denominator?)

The Nesterov direction is then: m̂_t + (1−β₂)·v̂_t,
which combines the mean gradient trend with a scaled curvature correction.

Algorithm (bias-corrected, decoupled weight decay)
---------------------------------------------------
Given g_t, previous gradient g_{t−1} (treated as 0 at t=1):

  Δg_t = g_t − g_{t−1}                         [gradient difference / curvature proxy]
  m_t  = β₁·m_{t−1} + (1−β₁)·g_t              [EMA of gradients]
  v_t  = β₂·v_{t−1} + (1−β₂)·Δg_t             [EMA of gradient differences]
  ñ_t  = g_t + (1−β₂)·Δg_t                    [Nesterov-adjusted gradient]
  n_t  = β₃·n_{t−1} + (1−β₃)·ñ_t²            [EMA of squared Nesterov gradient]

  Bias-correct: m̂, v̂, n̂ (divide by (1 − β^t))

  η_t  = (m̂_t + (1−β₂)·v̂_t) / (√n̂_t + ε)    [normalised update direction]
  θ_t  = θ_{t−1}·(1 − lr·λ) − lr·η_t          [decoupled WD + update]
  g_{t−1} ← g_t                                 [save for next step]
"""

import torch
from .base import BaseOptimizer


class Adan(BaseOptimizer):
    """Adan: Adaptive Nesterov Momentum optimizer.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-3)
    betas        : (β₁, β₂, β₃) — EMA rates for m, v, and n respectively.
                   Default (0.98, 0.92, 0.99) from the paper.
                   β₂ must be shared between the v update and the η formula.
    eps          : numerical stability constant added to √n̂_t (default 1e-8)
    weight_decay : decoupled L2 coefficient (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr              = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            eps             = group["eps"]
            wd              = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad      # current gradient g_t
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    state["step"]         = 0
                    state["exp_avg"]      = torch.zeros_like(p)   # m: gradient EMA
                    state["exp_avg_diff"] = torch.zeros_like(p)   # v: gradient-diff EMA
                    state["exp_avg_sq"]   = torch.zeros_like(p)   # n: sq. Nesterov EMA
                    # Store g_{t-1} for Δg computation; initialise to current g
                    # so Δg = 0 at the very first step (conservative start).
                    state["prev_grad"]    = g.clone()

                state["step"] += 1
                t      = state["step"]
                m      = state["exp_avg"]
                v      = state["exp_avg_diff"]
                n      = state["exp_avg_sq"]
                g_prev = state["prev_grad"]    # g_{t-1}

                # ── Gradient difference (finite-difference curvature) ──────
                # Δg_t = g_t − g_{t-1} approximates d/dt g_t, i.e., how fast
                # the gradient is changing along the trajectory.
                delta_g = g - g_prev

                # ── Three EMA updates ─────────────────────────────────────
                # m_t = β₁·m_{t-1} + (1-β₁)·g_t     — trend of gradient
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)

                # v_t = β₂·v_{t-1} + (1-β₂)·Δg_t    — trend of gradient change
                v.mul_(beta2).add_(delta_g, alpha=1.0 - beta2)

                # Nesterov-adjusted gradient: ñ_t = g_t + (1-β₂)·Δg_t
                # This is the gradient evaluated at an approximate look-ahead
                # point, without actually computing a second forward pass.
                nesterov_g = g.add(delta_g, alpha=1.0 - beta2)

                # n_t = β₃·n_{t-1} + (1-β₃)·ñ_t²   — EMA of squared Nesterov grad
                n.mul_(beta3).addcmul_(nesterov_g, nesterov_g, value=1.0 - beta3)

                # ── Bias corrections ──────────────────────────────────────
                # Divide each EMA by (1 − β^t) to correct the zero-initialisation
                # bias that shrinks EMAs toward zero early in training.
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                bc3 = 1.0 - beta3 ** t

                m_hat = m / bc1     # bias-corrected gradient EMA
                v_hat = v / bc2     # bias-corrected gradient-diff EMA
                n_hat = n / bc3     # bias-corrected squared Nesterov EMA

                # ── Nesterov-combined update direction ────────────────────
                # η_t = (m̂_t + (1−β₂)·v̂_t) / (√n̂_t + ε)
                # WHY (1-β₂) scalar: mirrors the coefficient used when forming
                # ñ_t above, keeping the numerator consistent with the denominator.
                eta = m_hat.add(v_hat, alpha=1.0 - beta2)   # numerator
                eta.div_(n_hat.sqrt().add_(eps))             # adaptive normalisation

                # ── Decoupled weight decay + parameter update ─────────────
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)   # shrink weights before the gradient step
                p.add_(eta, alpha=-lr)      # θ_t = θ_{t-1} − lr·η_t

                # ── Save current gradient for the next step ───────────────
                # Must copy into the existing buffer (not reassign) so state
                # tracking works correctly across multiple step() calls.
                state["prev_grad"].copy_(g)

        return loss
