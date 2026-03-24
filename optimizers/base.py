from torch.optim import Optimizer


class BaseOptimizer(Optimizer):
    """Abstract base class for all custom optimizers in this project.

    Subclassing this instead of torch.optim.Optimizer directly provides:
      - A consistent inheritance chain so isinstance checks work uniformly.
      - A clear contract: subclasses must implement step() and nothing else.
      - All standard PyTorch Optimizer infrastructure for free:
          state_dict() / load_state_dict() — checkpoint save/load
          zero_grad()                      — gradient zeroing
          add_param_group()               — adding parameter groups at runtime
          param_groups                    — list of {params, lr, ...} dicts

    How to implement a new optimizer
    ---------------------------------
    1. Create optimizers/<name>.py containing a class that subclasses this.
    2. Define __init__ to collect hyperparameters and call super().__init__
       with a `defaults` dict.
    3. Implement step() with the @torch.no_grad() decorator.
    4. Import the class in optimizers/__init__.py.
    5. Add a lambda entry to OPTIMIZER_REGISTRY in train.py (and benchmark.py).

    Example skeleton
    ----------------
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

    Special class attributes (optional)
    ------------------------------------
    requires_create_graph : bool
        Set to True on second-order optimizers (AdaHessian, Sophia) that need
        gradients computed with create_graph=True.  train_one_epoch() in
        train.py checks this attribute and switches the backward call
        accordingly.  Also automatically disables AMP for these optimizers.
    """

    def step(self, closure=None):
        # Subclasses must override this; calling the base raises immediately
        # rather than silently doing nothing — a loud failure is easier to debug.
        raise NotImplementedError("Subclasses must implement step()")
