"""Adan optimizer.

From "Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing
Deep Models" (Xie et al., 2022).  https://arxiv.org/abs/2208.06677

Adan maintains three exponential moving averages to implicitly exploit
Nesterov momentum while remaining adaptive:

  m  — EMA of gradients                       (like Adam's 1st moment)
  v  — EMA of gradient differences g_t−g_{t−1}
  n  — EMA of squared "Nesterov gradient"  (g_t + (1−β₂)·Δg_t)²

The update direction combines m and v (Nesterov-style) and is scaled
element-wise by 1/√n — giving an AdaGrad-like adaptive step that also
uses gradient curvature information via the difference term.

Algorithm (bias-corrected, decoupled weight decay)
---------------------------------------------------
Given g_t, previous gradient g_{t−1} (0 at t=1):

  Δg_t = g_t − g_{t−1}
  m_t  = β₁·m_{t−1} + (1−β₁)·g_t
  v_t  = β₂·v_{t−1} + (1−β₂)·Δg_t
  n_t  = β₃·n_{t−1} + (1−β₃)·(g_t + (1−β₂)·Δg_t)²

  Bias-correct each moment, then:
  η_t  = (m̂_t + (1−β₂)·v̂_t) / (√n̂_t + ε)
  θ_t  = θ_{t−1}·(1 − lr·λ) − lr·η_t
"""

import torch
from .base import BaseOptimizer


class Adan(BaseOptimizer):
    """Adan: Adaptive Nesterov Momentum optimizer.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-3)
    betas        : (β₁, β₂, β₃) EMA decay rates (default (0.98, 0.92, 0.99))
    eps          : numerical stability constant  (default 1e-8)
    weight_decay : decoupled L2 coefficient      (default 0.0)
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
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            eps = group["eps"]
            wd  = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"]       = 0
                    state["exp_avg"]    = torch.zeros_like(p)   # m
                    state["exp_avg_diff"] = torch.zeros_like(p) # v
                    state["exp_avg_sq"] = torch.zeros_like(p)   # n
                    state["prev_grad"]  = g.clone()             # g_{t-1}

                state["step"] += 1
                t = state["step"]
                m  = state["exp_avg"]
                v  = state["exp_avg_diff"]
                n  = state["exp_avg_sq"]
                g_prev = state["prev_grad"]

                # Gradient difference
                delta_g = g - g_prev

                # Update three moments
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).add_(delta_g, alpha=1.0 - beta2)
                # Nesterov gradient: g + (1-β₂)·Δg
                nesterov_g = g.add(delta_g, alpha=1.0 - beta2)
                n.mul_(beta3).addcmul_(nesterov_g, nesterov_g, value=1.0 - beta3)

                # Bias corrections
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t
                bc3 = 1.0 - beta3 ** t

                m_hat = m / bc1
                v_hat = v / bc2
                n_hat = n / bc3

                # Nesterov-combined update direction
                eta = m_hat.add(v_hat, alpha=1.0 - beta2)
                eta.div_(n_hat.sqrt().add_(eps))

                # Decoupled weight decay + parameter update
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(eta, alpha=-lr)

                # Store current gradient for next step
                state["prev_grad"].copy_(g)

        return loss
