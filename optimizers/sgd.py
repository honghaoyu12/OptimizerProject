import torch
from .base import BaseOptimizer


class VanillaSGD(BaseOptimizer):
    """Plain stochastic gradient descent (no momentum).

    This serves as a minimal reference implementation showing how to build
    a custom optimizer on top of BaseOptimizer.

    Parameters
    ----------
    params : iterable
        Model parameters to optimize.
    lr : float
        Learning rate.
    """

    def __init__(self, params, lr: float = 1e-2):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-lr)

        return loss
