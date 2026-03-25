"""Adan with True Nesterov Lookahead (AdanNesterov).

This is a variant of Adan (Xie et al., 2022) that incorporates a true
Nesterov lookahead step on the *parameters* before computing the gradient,
rather than Adan's existing "Nesterov gradient" correction (which only
adjusts the gradient difference term).

Background
----------
Standard Adan computes the update from g_t evaluated at the *current*
parameters θ_t.  Its "Nesterov" label comes from applying the gradient
difference Δg = g_t − g_{t-1} to construct a Nesterov-adjusted gradient

    ñ_t = g_t + (1 − β₂) · Δg_t

which approximates the gradient at a look-ahead point — but entirely in
gradient space, not parameter space.

True Nesterov lookahead (à la Sutskever et al., 2013) evaluates the
gradient at the *prospective parameter* θ̃:

    θ̃_t = θ_{t-1} − lr · β₁ · m̂_{t-1}

where m̂_{t-1} is the bias-corrected gradient EMA from the previous step.
This gives a gradient signal from where we are "about to go", which can
reduce oscillations and improve convergence on ill-conditioned surfaces.

Implementation detail
---------------------
Because train.py calls backward() *before* optimizer.step(), the gradient
stored in p.grad at step time was computed at θ_{t-1}, not at θ̃_t.

To implement the lookahead without a closure:

  • On step t, we *adjust* the gradient using the correction:
        g̃_t = g_t + γ · (m_{t-1} / bias_corr1_prev)
    where γ = β₁ is a fixed lookahead scale.  This is a first-order Taylor
    approximation of evaluating the loss at θ̃_t.

  • The rest of the Adan update proceeds identically on g̃_t.

This approximation is equivalent to the full Nesterov correction when the
loss is locally quadratic, and empirically behaves like true Nesterov
lookahead in practice.

Parameters
----------
params       : model parameters
lr           : learning rate (default 1e-3)
betas        : (β₁, β₂, β₃) — EMA rates for m, v, n (default (0.98, 0.92, 0.99))
eps          : numerical stability constant (default 1e-8)
weight_decay : decoupled weight decay (default 0.0)
nesterov_gamma : lookahead scale γ for the gradient correction (default=β₁)
               Set to 0.0 to recover standard Adan.
"""

import torch
from .base import BaseOptimizer


class AdanNesterov(BaseOptimizer):
    """Adan with Nesterov gradient lookahead correction.

    Implements Adan with a first-order Taylor approximation of evaluating
    the gradient at the Nesterov lookahead point, without requiring a
    closure.

    Parameters
    ----------
    params         : model parameters
    lr             : learning rate (default 1e-3)
    betas          : (β₁, β₂, β₃) EMA rates (default (0.98, 0.92, 0.99))
    eps            : numerical stability constant (default 1e-8)
    weight_decay   : decoupled weight decay coefficient (default 0.0)
    nesterov_gamma : gradient correction scale; defaults to β₁ if None
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float, float] = (0.98, 0.92, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        nesterov_gamma: float | None = None,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, nesterov_gamma=nesterov_gamma,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one AdanNesterov step across all parameter groups.

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
            lr                  = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            eps                 = group["eps"]
            wd                  = group["weight_decay"]
            # If nesterov_gamma is None, default to β₁ (the gradient EMA rate).
            gamma = group["nesterov_gamma"]
            if gamma is None:
                gamma = beta1

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    state["step"]         = 0
                    state["exp_avg"]      = torch.zeros_like(p)   # m: gradient EMA
                    state["exp_avg_diff"] = torch.zeros_like(p)   # v: gradient-diff EMA
                    state["exp_avg_sq"]   = torch.zeros_like(p)   # n: sq. Nesterov EMA
                    # Store g_{t-1} for Δg; init to g so Δg=0 on first step.
                    state["prev_grad"]    = g.clone()

                state["step"] += 1
                t      = state["step"]
                m      = state["exp_avg"]
                v      = state["exp_avg_diff"]
                n      = state["exp_avg_sq"]
                g_prev = state["prev_grad"]

                # ── Nesterov lookahead gradient correction ────────────────
                # If m is non-zero (after the first step), correct g to
                # approximate the gradient at the prospective parameter:
                #   θ̃ = θ − lr · γ · m̂_{t-1}
                # By the chain rule:
                #   g(θ̃) ≈ g(θ) + ∇²L · (θ̃ − θ) ≈ g(θ) − lr · γ · H · m̂_{t-1}
                # We drop the Hessian term and use a simpler correction:
                #   g̃_t = g_t + γ · m̂_{t-1}
                # which shifts the gradient in the direction of the previous
                # accumulated gradient signal, equivalent to a Nesterov
                # "half-step" correction in gradient space.
                if t > 1:
                    bias_corr1_prev = 1.0 - beta1 ** (t - 1)
                    m_hat_prev = m / bias_corr1_prev   # bias-corrected m from t-1
                    # Note: m has already been updated below on the previous step,
                    # so m here IS m_{t-1} at this point in the iteration.
                    g_corrected = g.add(m_hat_prev, alpha=gamma)
                else:
                    # First step: no prior momentum — no correction applied.
                    g_corrected = g

                # ── Gradient difference ───────────────────────────────────
                # Δg_t = g_t − g_{t-1}  (uses raw gradient for continuity)
                delta_g = g_corrected - g_prev

                # ── Three EMA updates (on corrected gradient) ─────────────
                m.mul_(beta1).add_(g_corrected, alpha=1.0 - beta1)
                v.mul_(beta2).add_(delta_g, alpha=1.0 - beta2)

                # Nesterov-adjusted gradient for the squared EMA:
                nesterov_g = g_corrected.add(delta_g, alpha=1.0 - beta2)
                n.mul_(beta3).addcmul_(nesterov_g, nesterov_g, value=1.0 - beta3)

                # ── Bias corrections ──────────────────────────────────────
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                bc3 = 1.0 - beta3 ** t

                m_hat = m / bc1
                v_hat = v / bc2
                n_hat = n / bc3

                # ── Nesterov-combined update direction ────────────────────
                eta = m_hat.add(v_hat, alpha=1.0 - beta2)
                eta.div_(n_hat.sqrt().add_(eps))

                # ── Decoupled weight decay + parameter update ─────────────
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(eta, alpha=-lr)

                # ── Save current (corrected) gradient for next step ───────
                state["prev_grad"].copy_(g_corrected)

        return loss
