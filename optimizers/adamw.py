"""Custom AdamW optimizer.

From "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, ICLR 2019).
https://arxiv.org/abs/1711.05101

Motivation
----------
In standard Adam, weight decay is implemented by adding λ·θ to the gradient
*before* the moment update.  This means the regularisation term gets adapted
by the second moment estimate v_t: effectively, parameters with large gradient
variance receive a weaker regularisation signal.

AdamW fixes this by *decoupling* the weight decay from the gradient — the
decay is applied directly to the parameter after the Adam update, with a
constant effective magnitude regardless of the gradient statistics.

The difference is subtle but matters:
  - Adam + L2 reg:  u_t = m̂_t / (√v̂_t + ε) + λ·θ  (decay scaled by 1/√v̂)
  - AdamW:          θ_t = (θ_{t-1} − α · m̂_t / (√v̂_t + ε)) − α·λ·θ_{t-1}

Because the decay in AdamW is multiplied by the same α as the gradient term,
the effective weight decay strength is α·λ rather than just λ, which keeps
the regularisation scale consistent when the learning rate changes (e.g. with
a cosine schedule).

Algorithm
---------
For each parameter p with gradient g at step t:

  1. Decouple weight decay:  p ← p · (1 − α·λ)    [shrink toward origin first]
  2. First moment:           m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
  3. Second moment:          v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²
  4. Bias-correct:           m̂_t = m_t / (1 − β₁^t),  v̂_t = v_t / (1 − β₂^t)
  5. Gradient update:        θ_t = p − α · m̂_t / (√v̂_t + ε)

Practical tips
--------------
  • The decoupled form makes weight_decay comparable across LR schedules —
    a value of 0.01–0.1 is typical for most vision tasks.
  • AdamW generally outperforms Adam + L2 reg when a cosine or warmup
    schedule is used, because the decay strength stays proportional to α.
"""

import math

import torch
from .base import BaseOptimizer


class AdamW(BaseOptimizer):
    """AdamW: Adam with decoupled weight decay.

    Parameters
    ----------
    params       : iterable of parameters or param groups
    lr           : global learning rate α (default 1e-3)
    betas        : (β₁, β₂) — EMA decay rates for first and second moments
                   (default (0.9, 0.999))
    eps          : numerical stability constant in the adaptive denominator
                   (default 1e-8)
    weight_decay : decoupled weight decay coefficient λ (default 1e-2).
                   Applied directly to parameters, not through the gradients.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one AdamW step across all parameter groups.

        Parameters
        ----------
        closure : callable | None
            Optional re-evaluation closure. Rarely needed for AdamW.

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

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on the first step ────────────────────
                if len(state) == 0:
                    state["step"]       = 0
                    # m: first moment (mean of gradient)
                    state["exp_avg"]    = torch.zeros_like(p)
                    # v: second moment (uncentred variance of gradient)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t    = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # ── Step 1: Decoupled weight decay ────────────────────────
                # Shrink the parameter directly by factor (1 − α·λ).
                # This is the KEY difference from Adam: the decay is NOT
                # added to the gradient, so it bypasses the second moment
                # scaling entirely.  Both large and small gradient parameters
                # get the same proportional shrinkage.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ── Steps 2 & 3: Update moment estimates (on raw gradient) ─
                # Note: gradient g is NOT modified by weight decay here.
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # ── Step 4: Bias correction ───────────────────────────────
                # Remove the initialisation-at-zero bias from early steps.
                bias_corr1 = 1.0 - beta1 ** t
                bias_corr2 = 1.0 - beta2 ** t

                # ── Step 5: Adaptive gradient update ─────────────────────
                # θ ← θ − α · m̂ / (√v̂ + ε)
                step_size = lr / bias_corr1
                denom     = v.sqrt().div_(math.sqrt(bias_corr2)).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

        return loss
