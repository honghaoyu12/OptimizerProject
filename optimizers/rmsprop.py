"""Custom RMSprop optimizer.

Introduced by Geoffrey Hinton in his Coursera lecture, slide 29:
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

(Unpublished but widely cited; predates Adam.)

Motivation
----------
Adagrad's key problem is that it accumulates the *sum* of squared gradients
forever.  In long training runs, this sum grows without bound, causing the
effective learning rate to shrink to zero and training to stall.

RMSprop replaces the cumulative sum with an Exponential Moving Average (EMA)
of squared gradients, giving recent gradients more influence than old ones.
The EMA "forgets" distant history with rate (1 − α), where α is the decay
(historically called `alpha` or `rho`).

This allows RMSprop to adapt to non-stationary loss landscapes (e.g. when the
gradient scale shifts dramatically at different training phases) and prevents
the learning rate from collapsing to zero.

Algorithm
---------
For each parameter p with gradient g at step t:

  1. EMA of squared gradients: E[g²]_t = α · E[g²]_{t-1} + (1 − α) · g_t²
  2. Parameter update:          θ_t = θ_{t-1} − lr · g_t / (√E[g²]_t + ε)

Optional centred variant
------------------------
The centred variant normalises by the *variance* of the gradient rather than
its uncentred second moment:

  E[g]_t  = α · E[g]_{t-1} + (1 − α) · g_t
  Var[g]_t = E[g²]_t − E[g]_t²
  θ_t     = θ_{t-1} − lr · g_t / (√Var[g]_t + ε)

The centred form is more expensive (one extra buffer) but can be more stable
when the gradient distribution changes rapidly.

Optional momentum
-----------------
Momentum can be layered on top of the adaptive scaling:

  v_t = μ · v_{t-1} + lr · g_t / (√E[g²]_t + ε)
  θ_t = θ_{t-1} − v_t

Practical tips
--------------
  • RMSprop is particularly popular for RNNs, where gradient magnitudes vary
    greatly across time steps.
  • Typical α (decay): 0.99; typical LR: 1e-3 to 1e-2.
  • For image classification, Adam generally outperforms RMSprop, but the
    centred variant with momentum can close much of the gap.
"""

import torch
from .base import BaseOptimizer


class RMSprop(BaseOptimizer):
    """RMSprop: Root Mean Square Propagation optimizer.

    Parameters
    ----------
    params       : iterable of parameters or param groups
    lr           : global learning rate (default 1e-2)
    alpha        : EMA decay factor for squared gradient (default 0.99).
                   Higher α = longer memory = smoother effective LR.
    eps          : numerical stability constant (default 1e-8)
    weight_decay : L2 regularisation (coupled form, default 0.0)
    momentum     : momentum coefficient for the velocity buffer (default 0.0,
                   disabled).  When > 0, accumulates scaled updates in a buffer.
    centered     : if True, use the centred variant — normalise by gradient
                   variance instead of uncentred second moment (default False)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
    ):
        defaults = dict(
            lr=lr, alpha=alpha, eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one RMSprop step across all parameter groups.

        Returns
        -------
        float | None
            Loss from the closure, or None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            alpha    = group["alpha"]
            eps      = group["eps"]
            wd       = group["weight_decay"]
            mu       = group["momentum"]
            centered = group["centered"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    # E[g²]: running EMA of squared gradients
                    state["square_avg"] = torch.zeros_like(p)

                    if centered:
                        # E[g]: running EMA of the gradient itself (for centring)
                        state["grad_avg"] = torch.zeros_like(p)

                    if mu > 0.0:
                        # buf: velocity buffer for optional momentum
                        state["momentum_buffer"] = torch.zeros_like(p)

                sq_avg = state["square_avg"]

                # ── Coupled weight decay ──────────────────────────────────
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # ── Step 1: Update EMA of squared gradients ───────────────
                # E[g²]_t = α · E[g²]_{t-1} + (1−α) · g²
                sq_avg.mul_(alpha).addcmul_(g, g, value=1.0 - alpha)

                # ── Compute adaptive denominator ──────────────────────────
                if centered:
                    # Centred variant: normalise by variance instead of E[g²]
                    # Var[g] = E[g²] − E[g]²
                    grad_avg = state["grad_avg"]
                    # Update running mean of gradient
                    grad_avg.mul_(alpha).add_(g, alpha=1.0 - alpha)
                    # Variance = second moment minus squared first moment
                    # Use addcmul_ to compute E[g²] − E[g]² element-wise
                    avg = sq_avg.addcmul(grad_avg, grad_avg, value=-1.0).sqrt_().add_(eps)
                else:
                    # Standard variant: denominator is √E[g²] + ε
                    avg = sq_avg.sqrt().add_(eps)

                # ── Step 2: Parameter update ──────────────────────────────
                if mu > 0.0:
                    # With momentum: accumulate scaled updates in a buffer
                    # v_t = μ · v_{t-1} + lr · g / avg
                    buf = state["momentum_buffer"]
                    buf.mul_(mu).addcdiv_(g, avg, value=lr)
                    # θ ← θ − v_t
                    p.add_(buf, alpha=-1.0)
                else:
                    # Without momentum: direct adaptive update
                    # θ ← θ − lr · g / (√E[g²] + ε)
                    p.addcdiv_(g, avg, value=-lr)

        return loss
