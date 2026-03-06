"""Lion optimizer.

From "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023).
https://arxiv.org/abs/2302.06675

Unlike Adam, Lion tracks only one momentum buffer (no second moment) and
updates weights using only the *sign* of the interpolated gradient, making
each step the same magnitude.  This also means the effective learning rate
should be roughly 10× smaller than you would use with Adam.

Algorithm
---------
Given momentum m_{t-1}, gradient g_t, parameters θ_{t-1}:

  1. Compute signed update:  c_t = sign(β₁ · m_{t-1} + (1−β₁) · g_t)
  2. Apply weight decay:     c_t = c_t + λ · θ_{t-1}
  3. Update parameters:      θ_t = θ_{t-1} − α · c_t
  4. Update momentum:        m_t = β₂ · m_{t-1} + (1−β₂) · g_t
"""

import torch
from .base import BaseOptimizer


class Lion(BaseOptimizer):
    """Lion: EvoLved Sign Momentum optimizer.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-4; ~10× smaller than Adam is typical)
    betas        : (β₁, β₂) — β₁ controls the interpolation for the sign update,
                   β₂ controls the momentum EMA  (default (0.9, 0.99))
    weight_decay : decoupled weight decay coefficient (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]

                # Step 1: signed interpolation
                update = (beta1 * m + (1.0 - beta1) * g).sign()

                # Step 2: decoupled weight decay
                if wd != 0.0:
                    update.add_(p, alpha=wd)

                # Step 3: parameter update
                p.add_(update, alpha=-lr)

                # Step 4: momentum EMA
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss
