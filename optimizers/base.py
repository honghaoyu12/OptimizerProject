from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    """Base class for custom optimizers.

    Subclass this and implement `step()` to create your own optimizer.
    All PyTorch optimizer machinery (param groups, state dict, etc.) is
    inherited for free.

    Example
    -------
    class MyOptimizer(BaseOptimizer):
        def __init__(self, params, lr=1e-3):
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
                    # ---- your update rule here ----
                    p.add_(p.grad, alpha=-lr)
            return loss
    """

    def step(self, closure=None):
        raise NotImplementedError("Subclasses must implement step()")
