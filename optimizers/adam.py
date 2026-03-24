"""Custom Adam optimizer.

From "Adam: A Method for Stochastic Optimization" (Kingma & Ba, ICLR 2015).
https://arxiv.org/abs/1412.6980

Motivation
----------
Plain SGD uses the same learning rate for every parameter regardless of how
frequently or how noisily each parameter's gradient appears.  Adam fixes both
problems by maintaining:
  - A first moment (mean)  m_t: the running average of the gradient direction.
  - A second moment (variance) v_t: the running average of squared gradients,
    which captures per-parameter gradient magnitude.

The update divides the (bias-corrected) mean by the square root of the
(bias-corrected) variance, giving each parameter an individually scaled
effective learning rate.  Parameters with consistently large gradients take
small steps; parameters with small or infrequent gradients take large steps.

Algorithm
---------
For each parameter p with gradient g at step t:

  1. Update first moment:   m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
  2. Update second moment:  v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²
  3. Bias correction:       m̂_t = m_t / (1 − β₁^t)
                            v̂_t = v_t / (1 − β₂^t)
  4. Parameter update:      θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)

Bias correction (step 3) is needed because both moment buffers are initialised
to zero — without it, early updates would be systematically under-estimated.

Weight decay (L2 regularisation) is implemented here as the classical coupled
form: the decay term λ·θ is added to the gradient before the moment update,
so the regulariser is also adapted by the second moment.  For decoupled weight
decay use AdamW instead.

Practical tips
--------------
  • Default hyperparameters (α=1e-3, β₁=0.9, β₂=0.999, ε=1e-8) work
    surprisingly well across a wide range of tasks without tuning.
  • If training loss oscillates, reduce β₁ (e.g. 0.8) or reduce α.
  • ε is rarely worth tuning; increase it (1e-6) if you see NaN in early steps.
"""

import math

import torch
from .base import BaseOptimizer


class Adam(BaseOptimizer):
    """Adam: Adaptive Moment Estimation optimizer.

    Parameters
    ----------
    params       : iterable of parameters or param groups
    lr           : global learning rate α (default 1e-3)
    betas        : (β₁, β₂) — EMA decay rates for first and second moments
                   (default (0.9, 0.999))
    eps          : numerical stability constant added to the denominator
                   (default 1e-8)
    weight_decay : L2 regularisation coefficient — added to gradient before
                   moment update (coupled form, default 0.0)
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
        """Perform one Adam step across all parameter groups.

        Parameters
        ----------
        closure : callable | None
            Optional re-evaluation closure (re-computes loss + backward).
            Required by the Optimizer API but rarely used with Adam.

        Returns
        -------
        float | None
            Loss from the closure, or None.
        """
        loss = None
        if closure is not None:
            # Re-enable grad so the closure can call .backward()
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr            = group["lr"]
            beta1, beta2  = group["betas"]
            eps           = group["eps"]
            wd            = group["weight_decay"]

            for p in group["params"]:
                # Skip parameters that received no gradient this step
                # (e.g. unused embedding rows, frozen layers)
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on the very first step ───────────────
                if len(state) == 0:
                    state["step"]       = 0
                    # m: first moment — tracks the mean gradient direction
                    state["exp_avg"]    = torch.zeros_like(p)
                    # v: second moment — tracks uncentred gradient variance
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t    = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # ── Coupled weight decay ──────────────────────────────────
                # Add λ·θ to the raw gradient so the decay flows through the
                # moment estimators (this is the "coupled" / L2 form).
                # For decoupled weight decay, use AdamW instead.
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # ── Step 1 & 2: Update moment estimates ───────────────────
                # m_t ← β₁·m_{t-1} + (1−β₁)·g
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                # v_t ← β₂·v_{t-1} + (1−β₂)·g²  (addcmul_ = in-place multiply-accumulate)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # ── Step 3: Bias-corrected moment estimates ───────────────
                # Both m and v are initialised to zero, so early steps are
                # biased toward zero.  Dividing by (1 − β^t) corrects this.
                # The correction factor shrinks to 1 as t → ∞.
                bias_corr1 = 1.0 - beta1 ** t   # correction for first moment
                bias_corr2 = 1.0 - beta2 ** t   # correction for second moment

                # ── Step 4: Parameter update ──────────────────────────────
                # θ ← θ − α · (m/bias_corr1) / (√(v/bias_corr2) + ε)
                # Rearranged to avoid allocating bias-corrected tensors:
                #   step_size = α / bias_corr1
                #   denom     = √v / √bias_corr2 + ε  (addcmul_ is not available here)
                step_size = lr / bias_corr1
                # √bias_corr2 applied to denominator instead of numerator — same result
                denom = v.sqrt().div_(math.sqrt(bias_corr2)).add_(eps)
                # In-place parameter update: θ ← θ − step_size · m / denom
                p.addcdiv_(m, denom, value=-step_size)

        return loss
