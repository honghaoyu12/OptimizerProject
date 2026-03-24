"""AdaBelief optimizer.

From "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed
Gradients" (Zhuang et al., NeurIPS 2020).  https://arxiv.org/abs/2010.07468

Core Idea
---------
Adam's second moment v_t = EMA(g_t²) measures how large the gradients are
on average.  The step size is inversely proportional to √v_t — large gradients
→ small steps, small gradients → large steps.

AdaBelief replaces v_t with the EMA of the *prediction error*:
    s_t = EMA((g_t − m_t)²)

where m_t = EMA(g_t) is the first moment.  (g_t − m_t)² measures *surprise*:
  - If g_t ≈ m_t (gradient matches its prediction), (g_t − m_t)² ≈ 0, so s_t
    is small → large step (we "believe" the gradient is reliable).
  - If g_t ≫ m_t (noisy or rapidly changing), (g_t − m_t)² is large, so s_t
    is large → small step (low confidence → be cautious).

This gives AdaBelief a key advantage over Adam: it adapts to gradient
*reliability*, not just gradient *magnitude*.  On flat loss surfaces where
gradients are small but consistent, Adam's steps shrink (small √v_t →
small denom → actually large steps — Adam's "step amplification" problem).
AdaBelief behaves like SGD in these regions because small gradients that
are *predictable* (g ≈ m → s ≈ 0) lead to aggressive steps anyway.

Algorithm
---------
Given momentum m_{t-1}, belief s_{t-1}, gradient g_t, parameters θ_{t-1}:

  1. m_t = β₁ · m_{t-1} + (1−β₁) · g_t               [gradient EMA]
  2. s_t = β₂ · s_{t-1} + (1−β₂) · (g_t − m_t)² + ε  [belief EMA + stability]
  3. m̂_t = m_t / (1−β₁ᵗ)                              [bias correction]
  4. ŝ_t = s_t / (1−β₂ᵗ)                              [bias correction]
  5. θ_t = θ_{t-1} − α · m̂_t / (√ŝ_t + ε)           [parameter update]

The +ε in step 2 ensures s_t ≥ ε always, preventing division by zero even
when the gradient is perfectly predictable (s_t → 0).  Using a smaller ε
here (1e-16) than Adam (1e-8) allows the denominator to be smaller and the
effective step larger on well-predicted gradients.
"""

import math

import torch

from .base import BaseOptimizer


class AdaBelief(BaseOptimizer):
    """AdaBelief: adapts step sizes by belief in observed gradients.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-3, same as Adam)
    betas        : (β₁, β₂) — EMA coefficients for gradient and belief moment
                   (default (0.9, 0.999), same as Adam)
    eps          : added to belief EMA (step 2) and to the denominator (step 5).
                   Should be smaller than Adam's 1e-8 to allow near-zero belief
                   to produce large steps on reliable gradient directions.
                   Default 1e-16 (paper recommendation).
    weight_decay : decoupled weight decay (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-16,
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
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            wd           = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad      # current gradient tensor
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    state["step"]            = 0
                    # m: first moment (gradient EMA) — same as Adam's m
                    state["exp_avg"]         = torch.zeros_like(p)
                    # s: second moment based on prediction error (the "belief")
                    state["exp_avg_belief"]  = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]     # current step count (for bias correction)
                m = state["exp_avg"]
                s = state["exp_avg_belief"]

                # ── Decoupled weight decay (before gradient steps) ─────────
                # Decoupled = applied as θ ← θ(1 − α·λ) rather than modifying g,
                # which avoids interaction between weight decay and the adaptive
                # denominator (see AdamW paper, Loshchilov & Hutter 2019).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ── Step 1: Gradient EMA ──────────────────────────────────
                # m_t = β₁ · m_{t-1} + (1−β₁) · g_t
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)

                # ── Step 2: Belief EMA ────────────────────────────────────
                # diff = g_t − m_t  (prediction error after EMA update)
                # s_t = β₂ · s_{t-1} + (1−β₂) · diff² + ε
                # addcmul_(diff, diff, value=...) computes += value * diff * diff
                diff = g - m          # element-wise prediction error
                s.mul_(beta2).addcmul_(diff, diff, value=1.0 - beta2).add_(eps)

                # ── Steps 3 & 4: Bias corrections ─────────────────────────
                # Early in training (small t) the EMAs are biased toward zero
                # because they are initialised to 0.  Dividing by (1 − β^t)
                # corrects this bias, matching the true moment expectation.
                bc1 = 1.0 - beta1 ** t   # correction for m (first moment)
                bc2 = 1.0 - beta2 ** t   # correction for s (second moment)

                # ── Step 5: Parameter update ──────────────────────────────
                # θ_t ← θ_{t-1} − (lr/bc1) · m / (√(s/bc2) + ε)
                # Equivalent to: −lr · m̂_t / (√ŝ_t + ε)
                step_size = lr / bc1
                # (s / bc2).sqrt() + eps: bias-corrected belief std + numerical floor
                denom     = (s / bc2).sqrt().add_(eps)
                # addcdiv_(numerator, denominator, value): p += value * num / denom
                p.addcdiv_(m, denom, value=-step_size)

        return loss
