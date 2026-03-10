"""Schedule-Free AdamW optimizer.

From "The Road Less Scheduled" (Defazio, Yang, Khaled, Mishchenko,
Defazio & Cutkosky, NeurIPS 2024).  https://arxiv.org/abs/2405.15682

Standard optimizers require a carefully tuned learning-rate schedule that
decays the LR toward the end of training (cosine, step, etc.).  Schedule-Free
replaces the schedule with a combination of:
  1. Polyak-Ruppert averaging (maintain a running average x of the iterates)
  2. Momentum interpolation (evaluate gradients at y = (1−β₁)·x + β₁·z,
     halfway between the average x and the latest gradient-updated iterate z)

This gives the best of both worlds — the convergence speed of momentum and
the generalization of averaging — without specifying a schedule in advance.

The three sequences maintained conceptually:
  z  — "primal" sequence, updated by Adam steps (not stored explicitly)
  x  — Polyak average of z (stored in optimizer state)
  y  — interpolation of x and z, set as the model weights (params = y)

Note on evaluation: during training the model weights are y (a mix of x and z).
For final evaluation you should switch to x (the averaged weights), which tends
to be slightly better.  A convenience method `get_averaged_params()` is provided
for this purpose.

Algorithm (one step, params = y_k)
-----------------------------------
  g_k  = ∇f(y_k)                                   [gradient at y]
  v_k  = β₂·v_{k−1} + (1−β₂)·g_k²                 [second moment at y]
  z_k  = (y_k − (1−β₁)·x_{k−1}) / β₁              [reconstruct z from y and x]
  z_k  = z_k − α·g_k/(√v_k + ε) − α·λ·z_k         [Adam step on z]
  x_k  = (1 − 1/k)·x_{k−1} + (1/k)·z_k            [Polyak average]
  y_{k+1} = (1−β₁)·x_k + β₁·z_k                   [set params for next step]
"""

import torch
from .base import BaseOptimizer


class ScheduleFreeAdamW(BaseOptimizer):
    """Schedule-Free AdamW: adaptive optimizer with built-in iterate averaging.

    Parameters
    ----------
    params       : model parameters (will be set to y, the interpolation point)
    lr           : learning rate (default 1e-3)
    betas        : (β₁, β₂) — interpolation weight and second-moment EMA rate
                   (default (0.9, 0.999))
    eps          : numerical stability constant (default 1e-8)
    weight_decay : decoupled weight decay applied to z (default 0.0)
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
                    # x starts equal to the initial params (= y_1 = z_1)
                    state["x"] = p.data.clone()
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                x = state["x"]
                v = state["exp_avg_sq"]

                # Step 1: update second moment (computed at y = p)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Step 2: reconstruct z from current params (y) and stored x
                #   y = (1−β₁)·x + β₁·z  ⟹  z = (y − (1−β₁)·x) / β₁
                if beta1 > 0.0:
                    z = (p.data - (1.0 - beta1) * x) / beta1
                else:
                    z = p.data.clone()

                # Step 3: Adam step on z (decoupled weight decay applied to z)
                if wd != 0.0:
                    z.mul_(1.0 - lr * wd)
                denom = v.sqrt().add_(eps)
                z.addcdiv_(g, denom, value=-lr)

                # Step 4: Polyak-Ruppert update of x  (1/t weighting)
                #   x_k = x_{k−1} + (z_k − x_{k−1}) / k
                x.add_(z - x, alpha=1.0 / t)

                # Step 5: set params to new y = (1−β₁)·x + β₁·z
                p.data.copy_((1.0 - beta1) * x + beta1 * z)

        return loss

    @torch.no_grad()
    def get_averaged_params(self) -> list[torch.Tensor]:
        """Return the Polyak-averaged weights x (better for final evaluation).

        Example usage after training::

            averaged = optimizer.get_averaged_params()
            for p, x in zip(model.parameters(), averaged):
                p.data.copy_(x)
        """
        result = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "x" in state:
                    result.append(state["x"].clone())
                else:
                    result.append(p.data.clone())
        return result
