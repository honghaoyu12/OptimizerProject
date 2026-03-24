"""Vanilla SGD optimizer — minimal reference implementation.

Plain stochastic gradient descent without momentum.  This is the simplest
possible gradient-based optimizer:

    θ_t = θ_{t-1} − α · (g_t + λ · θ_{t-1})

where:
  α  = learning rate
  g_t = ∇L(θ_{t-1}) on the current mini-batch
  λ  = weight decay coefficient

This file also serves as the recommended template for adding new optimizers:
  1. Copy this file to optimizers/my_optimizer.py
  2. Rename VanillaSGD → MyOptimizer
  3. Modify __init__ to add hyperparameters to `defaults`
  4. Implement your update rule in step()
  5. Import and register it in train.py and benchmark.py
"""

import torch
from .base import BaseOptimizer


class VanillaSGD(BaseOptimizer):
    """Plain stochastic gradient descent — no momentum, no adaptive LR.

    Serves as a minimal reference implementation and a performance baseline.
    Any optimizer that cannot outperform this on a given task likely has a
    bug or needs LR tuning.

    Note: torch.optim.SGD with momentum=0.0 is equivalent but slightly faster
    (C++ kernel).  This Python implementation exists to demonstrate the
    BaseOptimizer pattern and to be instrumented by diagnostic tools.

    Parameters
    ----------
    params       : model parameters (as returned by model.parameters())
    lr           : learning rate (default 1e-2; high by Adam standards but
                   typical for momentum-free SGD on simple datasets)
    weight_decay : L2 regularisation strength (default 0.0, disabled)
    """

    def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0):
        # Pack hyperparameters into the defaults dict that PyTorch's Optimizer
        # base class distributes to each param group.
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()   # disable autograd inside step() for speed and correctness
    def step(self, closure=None):
        """Perform one SGD step.

        Parameters
        ----------
        closure : callable | None
            Optional re-evaluation closure (used by some line-search methods).
            Rarely needed for SGD but required by the Optimizer API.

        Returns
        -------
        float | None
            Loss value from the closure, or None if no closure was provided.
        """
        loss = None
        if closure is not None:
            # Re-enable grad temporarily so the closure can call .backward()
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                # Skip parameters that did not participate in the forward pass
                # (e.g. embedding rows that were never looked up)
                if p.grad is None:
                    continue

                grad = p.grad

                # L2 weight decay: add λ·θ to the gradient.
                # WHY not out-of-place: we don't want to corrupt p.grad in
                # case the caller inspects it after step().  grad.add() creates
                # a new tensor; the original p.grad is unchanged.
                if wd != 0.0:
                    grad = grad.add(p.data, alpha=wd)

                # Core SGD update: θ ← θ − α · (g + λθ)
                p.add_(grad, alpha=-lr)

        return loss
