"""Custom Adagrad optimizer.

From "Adaptive Subgradient Methods for Online Learning and Stochastic
Optimization" (Duchi et al., JMLR 2011).
https://jmlr.org/papers/v12/duchi11a.html

Motivation
----------
Different parameters in a neural network can have very different gradient
magnitudes — for example, word embeddings for rare words receive infrequent
but large gradients, while common words receive small but frequent updates.
Using a single global learning rate is a poor fit for this heterogeneity.

Adagrad assigns each parameter its own effective learning rate by dividing
the global α by the square root of the sum of all squared gradients seen so
far.  Parameters with consistently large gradients accumulate a large sum and
get a small effective LR; parameters with small or infrequent gradients
accumulate less and get a larger effective LR.

This property made Adagrad particularly effective for sparse problems (NLP,
recommendation systems) where most parameters rarely receive gradients.

Limitation
----------
The sum-of-squares accumulator G_t is monotonically non-decreasing, so the
effective learning rate can only shrink.  In long training runs, Adagrad can
"run out of steam" and stop learning entirely.  AdaRMSProp/RMSprop and Adam
address this by using an exponential moving average instead of a sum.

Algorithm
---------
For each parameter p with gradient g at step t:

  1. Accumulate squared gradients:  G_t = G_{t-1} + g_t²
  2. Parameter update:              θ_t = θ_{t-1} − α · g_t / (√G_t + ε)

The ε prevents division by zero when G_t is near zero (uninitialised or
sparse parameters).

Practical tips
--------------
  • Works well for sparse inputs; often the preferred baseline for NLP tasks.
  • Tune α in the range 0.01–0.1; Adagrad is less sensitive to α than SGD.
  • For dense problems (CNNs, MLPs) prefer Adam/AdamW which don't degrade.
  • initial_accumulator_value > 0 can prevent very large updates at step 1.
"""

import torch
from .base import BaseOptimizer


class Adagrad(BaseOptimizer):
    """Adagrad: Adaptive Gradient optimizer.

    Parameters
    ----------
    params                    : iterable of parameters or param groups
    lr                        : global learning rate α (default 1e-2)
    eps                       : stability constant in the denominator
                                (default 1e-10; Adagrad's denominator can be
                                very large, so a small ε suffices)
    weight_decay              : L2 regularisation strength, added to gradient
                                before accumulation (default 0.0)
    initial_accumulator_value : starting value for G_t (default 0.0).
                                Setting this > 0 prevents a very large
                                effective LR on the first step for parameters
                                that have never been updated.
    lr_decay                  : rate at which the base LR α itself decays
                                with step count: α_t = α / (1 + t·lr_decay)
                                (default 0.0, disabled)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        lr_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, eps=eps, weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            lr_decay=lr_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Adagrad step across all parameter groups.

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
            lr          = group["lr"]
            eps         = group["eps"]
            wd          = group["weight_decay"]
            init_acc    = group["initial_accumulator_value"]
            lr_decay    = group["lr_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    # G: accumulated sum of squared gradients.
                    # Initialised to initial_accumulator_value to dampen the
                    # very first step's effective learning rate.
                    state["sum"] = torch.full_like(p, fill_value=init_acc)

                state["step"] += 1
                t   = state["step"]
                G   = state["sum"]

                # ── Optional coupled weight decay ─────────────────────────
                # Add λ·θ to the gradient before accumulation, so the decay
                # also contributes to the squared-gradient sum.
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # ── Optional LR decay ─────────────────────────────────────
                # Gradually reduce α over training steps.
                # α_t = α / (1 + (t−1) · lr_decay)
                # (t−1 because the first step should use the full α)
                if lr_decay != 0.0:
                    clr = lr / (1.0 + (t - 1) * lr_decay)
                else:
                    clr = lr

                # ── Step 1: Accumulate squared gradients ──────────────────
                # G_t = G_{t-1} + g²
                # In-place element-wise multiply-accumulate:
                G.addcmul_(g, g, value=1.0)

                # ── Step 2: Adaptive parameter update ─────────────────────
                # θ ← θ − α_t · g / (√G_t + ε)
                # The denominator grows over time → effective LR shrinks.
                denom = G.sqrt().add_(eps)
                p.addcdiv_(g, denom, value=-clr)

        return loss
