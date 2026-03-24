"""MARS optimizer.

From "MARS: Unleashing the Power of Variance Reduction for Training Large Models"
(Yuan et al., 2024).  https://arxiv.org/abs/2411.10438

Motivation
----------
Adam converges in practice but is not theoretically optimal for non-convex
problems because its gradient estimate has high variance from mini-batching.
MARS (Make vARiance-reduction Simple) adds a *variance-reduction correction*
to the gradient before the Adam update.

The idea is borrowed from STORM / SARAH variance-reduction methods: instead of
using the raw mini-batch gradient g_t directly, MARS uses a *corrected*
gradient that remembers the direction from the previous step and subtracts
the drift caused by parameter movement:

    c_t = g_t + γ · (g_t - g_{t-1}) * (w_t - w_{t-1}) / (lr * g_{t-1})

In the simplified form implemented here (MARS-AdamW variant):
    c_t = g_t + γ · (g_t - g_{t-1})

where γ controls the strength of the variance-reduction correction.  With
γ = 0, MARS reduces exactly to Adam.  With γ > 0, the corrected gradient
tracks gradient changes across steps, reducing the variance that comes from
parameter movement between iterations.

Algorithm (MARS-AdamW, simplified)
------------------------------------
At step t, given gradient g_t and previous gradient g_{t-1}:

  1. Compute corrected gradient:
       c_t = g_t + γ · (g_t - g_{t-1})
           = (1 + γ) · g_t − γ · g_{t-1}

  2. Normalise correction (clip to prevent explosion):
       c_t = c_t / max(||c_t||, ||g_t||) * ||g_t||

  3. Adam updates with the corrected gradient c_t:
       m_t = β₁ · m_{t-1} + (1 − β₁) · c_t
       v_t = β₂ · v_{t-1} + (1 − β₂) · c_t²
       m̂_t = m_t / (1 − β₁^t)
       v̂_t = v_t / (1 − β₂^t)

  4. Decoupled weight decay:
       w_t ← w_t * (1 − α·λ)

  5. Parameter update:
       w_t ← w_{t-1} − α · m̂_t / (√v̂_t + ε)

  6. Store g_{t-1} ← g_t for the next step.

Practical tips
--------------
  • γ (gamma) in (0, 0.01) works well for most tasks; 0.005 is a safe default.
  • γ = 0 reduces to AdamW exactly; use this as a sanity check.
  • γ too large can make the corrected gradient unstable; the norm-clip in
    step 2 prevents divergence but some gamma tuning is still needed.
  • Same lr/betas/eps defaults as AdamW.
"""

import math

import torch
from .base import BaseOptimizer


class MARS(BaseOptimizer):
    """MARS: Variance-reduced Adam for large-model training.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate α (default 3e-4)
    betas        : (β₁, β₂) — EMA rates for first and second moments
                   (default (0.9, 0.99))
    eps          : numerical stability constant (default 1e-8)
    weight_decay : decoupled weight decay coefficient (default 0.1)
    gamma        : variance-reduction correction strength.  0 = plain AdamW.
                   (default 0.005)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        gamma: float = 0.005,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, gamma=gamma,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one MARS step across all parameter groups.

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
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            wd           = group["weight_decay"]
            gamma        = group["gamma"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on first step ─────────────────────────
                if len(state) == 0:
                    state["step"]       = 0
                    state["exp_avg"]    = torch.zeros_like(p)   # first moment m
                    state["exp_avg_sq"] = torch.zeros_like(p)   # second moment v
                    # Previous gradient: needed for the variance-reduction term.
                    # Initialised to the first gradient (so first-step correction = 0).
                    state["prev_grad"]  = g.clone()

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                g_prev = state["prev_grad"]

                # ── Step 1: Variance-reduction corrected gradient ──────────
                # c_t = g_t + γ · (g_t - g_{t-1})
                #      = (1 + γ) · g_t − γ · g_{t-1}
                #
                # Intuition: if the gradient has shifted from g_{t-1} to g_t,
                # the correction amplifies this shift, "looking ahead" in the
                # gradient direction and reducing variance from stale estimates.
                #
                # On the first step g_prev == g_t, so the correction term is
                # zero and c_t = g_t (plain Adam for the first step — safe).
                c_t = g.add(g - g_prev, alpha=gamma)

                # ── Step 2: Safety norm-clip to prevent explosion ──────────
                # If the correction amplifies the gradient beyond the original
                # magnitude, we rescale back to the original norm.
                # This is the key stability guard: without it, large γ values
                # can cause the corrected gradient to explode.
                norm_g  = g.norm()
                norm_ct = c_t.norm()
                # Only clip if the corrected gradient is strictly larger.
                # Avoids division by zero with the 1e-12 floor.
                if norm_ct > norm_g and norm_ct > 1e-12:
                    c_t = c_t.mul_(norm_g / norm_ct)

                # ── Step 3: Adam moment updates with corrected gradient ─────
                # Standard Adam EMA updates, but using c_t instead of g_t.
                # This is what makes MARS variance-reduced: the Adam second
                # moment v_t now tracks the variance of the *corrected* gradient,
                # which has lower variance than the raw mini-batch gradient.
                m.mul_(beta1).add_(c_t, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(c_t, c_t, value=1.0 - beta2)

                # Bias correction for early-step underestimation.
                bias_corr1 = 1.0 - beta1 ** t
                bias_corr2 = 1.0 - beta2 ** t

                # ── Step 4: Decoupled weight decay ─────────────────────────
                # AdamW-style: scale parameters directly before the gradient
                # update so decay is independent of the learning rate denominator.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ── Step 5: Parameter update ───────────────────────────────
                # Standard Adam step using bias-corrected moments.
                denom = (v.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                step_size = lr / bias_corr1
                p.addcdiv_(m, denom, value=-step_size)

                # ── Step 6: Store current gradient for next step ───────────
                # g_prev is needed next step to compute the correction term.
                # clone() + detach() ensures we own a clean copy not tied to
                # the autograd graph (which may be freed between steps).
                state["prev_grad"].copy_(g)

        return loss
