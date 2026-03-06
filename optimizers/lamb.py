"""LAMB optimizer.

From "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
(You et al., 2019).  https://arxiv.org/abs/1904.00962

LAMB extends Adam with a *layer-wise trust ratio* that scales the update for
each parameter tensor by the ratio of its weight norm to its update norm.
This keeps each layer's effective learning rate bounded, which stabilises
training at very large batch sizes without requiring per-layer LR tuning.

Algorithm
---------
For each parameter tensor p with gradient g:

  1. Compute Adam moments:
       m_t = β₁ · m_{t-1} + (1−β₁) · g_t
       v_t = β₂ · v_{t-1} + (1−β₂) · g_t²

  2. Bias-correct:
       m̂_t = m_t / (1 − β₁^t)
       v̂_t = v_t / (1 − β₂^t)

  3. Adam-style update direction with weight decay:
       u_t = m̂_t / (√v̂_t + ε) + λ · p_{t-1}

  4. Layer-wise trust ratio:
       r_t = ‖p_{t-1}‖₂ / ‖u_t‖₂   (clipped to [0, 10] for stability)

  5. Parameter update:
       p_t = p_{t-1} − α · r_t · u_t
"""

import torch
from .base import BaseOptimizer


class LAMB(BaseOptimizer):
    """LAMB: Layer-wise Adaptive Moments for Batch training.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-3)
    betas        : (β₁, β₂) Adam moment decay rates  (default (0.9, 0.999))
    eps          : numerical stability constant       (default 1e-6)
    weight_decay : weight decay coefficient (applied inside the update, not as
                   a separate gradient penalty)       (default 0.01)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
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
                    state["exp_avg"]    = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # Steps 1–2: bias-corrected Adam moments
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)

                # Step 3: Adam update direction + weight decay
                u = m_hat / (v_hat.sqrt() + eps)
                if wd != 0.0:
                    u.add_(p, alpha=wd)

                # Step 4: layer-wise trust ratio (clip to [0, 10])
                p_norm = p.norm()
                u_norm = u.norm()
                if p_norm == 0.0 or u_norm == 0.0:
                    trust_ratio = 1.0
                else:
                    trust_ratio = float((p_norm / u_norm).clamp(max=10.0))

                # Step 5: update
                p.add_(u, alpha=-lr * trust_ratio)

        return loss
