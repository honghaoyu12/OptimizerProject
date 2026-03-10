"""SignSGD optimizer.

From "signSGD: Compressed Optimisation for Non-Convex Problems"
(Bernstein et al., ICML 2018).  https://arxiv.org/abs/1802.04434

Each weight update uses only the *sign* of the gradient (or a momentum-smoothed
gradient), making every step exactly ±lr regardless of gradient magnitude.
This costs just 1 bit per parameter in communication, and removes the need to
tune a gradient scale.

With β = 0 (default) this is pure SignSGD:
    θ_t = θ_{t-1} − α · sign(g_t)

With β > 0 this becomes the "Signum" variant:
    m_t = β · m_{t-1} + (1−β) · g_t
    θ_t = θ_{t-1} − α · sign(m_t)

Note: Lion (already in this project) is closely related — it uses a two-beta
scheme where the sign is taken of a *convex interpolation* between momentum and
gradient.  SignSGD is simpler and serves as a useful baseline alongside Lion.
"""

import torch

from .base import BaseOptimizer


class SignSGD(BaseOptimizer):
    """SignSGD / Signum optimizer.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 0.01)
    momentum     : EMA coefficient β for Signum variant (default 0.0 = pure SignSGD)
    weight_decay : decoupled weight decay coefficient (default 0.0)
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
            lr = group["lr"]
            beta = group["momentum"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                if beta > 0.0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    buf = state["momentum_buffer"]
                    buf.mul_(beta).add_(g, alpha=1.0 - beta)
                    direction = buf.sign()
                else:
                    direction = g.sign()

                p.add_(direction, alpha=-lr)

        return loss
