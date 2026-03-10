"""AdaBelief optimizer.

From "AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients"
(Zhuang et al., NeurIPS 2020).  https://arxiv.org/abs/2010.07468

The key insight: instead of adapting the step size by the raw magnitude of the
second moment (like Adam), AdaBelief adapts by the *belief* — how closely the
current gradient matches its own running average.  When the gradient is
well-predicted by the EMA (low surprise), the denominator is small and the step
is large.  When the gradient is noisy and unpredictable (high surprise), the
denominator grows and the step shrinks.

In practice this behaves like Adam on smooth losses and like SGD on noisy ones,
often achieving better generalization without extra hyperparameter tuning.

Algorithm
---------
Given momentum m_{t-1}, belief s_{t-1}, gradient g_t, parameters θ_{t-1}:

  1. m_t = β₁ · m_{t-1} + (1−β₁) · g_t               [gradient EMA]
  2. s_t = β₂ · s_{t-1} + (1−β₂) · (g_t − m_t)² + ε  [belief EMA]
  3. m̂_t = m_t / (1−β₁ᵗ)                              [bias correction]
  4. ŝ_t = s_t / (1−β₂ᵗ)                              [bias correction]
  5. θ_t = θ_{t-1} − α · m̂_t / (√ŝ_t + ε)           [parameter update]
"""

import math

import torch

from .base import BaseOptimizer


class AdaBelief(BaseOptimizer):
    """AdaBelief: adapts step sizes by belief in observed gradients.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-3)
    betas        : (β₁, β₂) — EMA coefficients for gradient and belief
                   (default (0.9, 0.999))
    eps          : added to belief EMA and denominator for numerical stability
                   (default 1e-16; smaller than Adam's 1e-8 is intentional)
    weight_decay : decoupled weight decay coefficient (default 0.0)
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
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_belief"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                s = state["exp_avg_belief"]

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Step 1: gradient EMA
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)

                # Step 2: belief EMA — (g - m)² + ε
                diff = g - m
                s.mul_(beta2).addcmul_(diff, diff, value=1.0 - beta2).add_(eps)

                # Steps 3 & 4: bias correction
                bc1 = 1.0 - beta1 ** t
                bc2 = 1.0 - beta2 ** t

                # Step 5: parameter update
                step_size = lr / bc1
                denom = (s / bc2).sqrt().add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

        return loss
