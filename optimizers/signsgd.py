"""SignSGD / Signum optimizer.

From "signSGD: Compressed Optimisation for Non-Convex Problems"
(Bernstein et al., ICML 2018).  https://arxiv.org/abs/1802.04434

Core Idea
---------
Every gradient-based update can be decomposed into a *direction* and a
*magnitude*.  SignSGD throws away the magnitude entirely:

    θ_t = θ_{t-1} − α · sign(g_t)

Each parameter moves exactly ±α per step, regardless of how large or small
its gradient was.  This has three practical consequences:

  1. Implicit gradient clipping: no exploding-gradient problem.
  2. Bandwidth efficiency: the update is 1 bit per parameter (communicated
     as +1/−1), making it attractive for distributed/federated learning.
  3. Insensitivity to gradient scale: large and small gradients contribute
     equal updates, which can hurt on sparse or highly imbalanced tasks.

The Signum variant (β > 0) smooths the gradient before taking the sign:

    m_t = β · m_{t-1} + (1−β) · g_t
    θ_t = θ_{t-1} − α · sign(m_t)

This reduces the variance of the sign estimate (noisy gradients cancel out
in the EMA) at the cost of introducing momentum lag.

Relationship to other optimizers in this project
-------------------------------------------------
  Lion (lion.py): sign(β₁·m + (1−β₁)·g) — two-beta interpolation before sign.
  SignSGD:        sign(g)                  — sign of raw gradient (β=0) or EMA.

Lion is generally stronger; SignSGD exists as a simpler baseline.
"""

import torch

from .base import BaseOptimizer


class SignSGD(BaseOptimizer):
    """SignSGD / Signum: gradient-sign-based optimizer.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate / step magnitude (default 0.01).
                   Because every step is ±lr, this directly controls the
                   per-parameter displacement — unlike Adam where lr interacts
                   with the adaptive denominator.
    momentum     : EMA coefficient β for the Signum variant (default 0.0).
                   Set to 0.0 for pure SignSGD; set to 0.9 for Signum
                   (similar performance, smoother trajectory).
    weight_decay : decoupled weight decay (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr   = group["lr"]
            beta = group["momentum"]    # 0 = pure SignSGD, >0 = Signum
            wd   = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Decoupled weight decay ────────────────────────────────
                # Applied as multiplicative shrink before the sign step so
                # it doesn't interact with the gradient direction.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ── Compute sign direction ────────────────────────────────
                if beta > 0.0:
                    # Signum variant: EMA-smooth the gradient first
                    if "momentum_buffer" not in state:
                        # Initialise to zero — first step uses (1-β)·g only
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    # m_t = β·m_{t-1} + (1-β)·g_t  (in-place)
                    buf.mul_(beta).add_(g, alpha=1.0 - beta)
                    # Take sign of the smoothed momentum buffer
                    direction = buf.sign()
                else:
                    # Pure SignSGD: sign of the raw gradient
                    # No state needed — zero memory overhead.
                    direction = g.sign()

                # ── Parameter update ──────────────────────────────────────
                # θ ← θ − α · sign(...)
                # Every element moves by exactly α regardless of gradient magnitude.
                p.add_(direction, alpha=-lr)

        return loss
