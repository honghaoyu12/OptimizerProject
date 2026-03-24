"""SGD with Momentum (and optional Nesterov acceleration).

Classical SGD with momentum is one of the most widely used optimizers in deep
learning.  It stores a velocity buffer that accumulates gradient history with
exponential decay, giving the optimizer "inertia" that helps it traverse
flat regions and dampen oscillations in narrow valleys.

Motivation
----------
Plain SGD takes a step exactly in the negative gradient direction each iteration.
In a narrow valley (like the bottom of a long, curved ravine), the gradient
points almost perpendicular to the optimal direction — the optimizer zigzags
across the valley rather than travelling along it.  Momentum accumulates the
consistent components across steps (the along-valley direction) and cancels
the oscillating components (the cross-valley direction), leading to faster and
smoother convergence.

Standard momentum
-----------------
  v_t = μ · v_{t-1} + g_t          [velocity: weighted sum of past gradients]
  θ_t = θ_{t-1} − α · v_t          [update: step along velocity direction]

Nesterov momentum
-----------------
Instead of computing the gradient at θ_{t-1} and then applying momentum,
Nesterov first applies momentum and computes the gradient at the lookahead
position.  In SGD's mini-batch setting (where we can't truly re-compute the
gradient), this is approximated by modifying the update:

  v_t = μ · v_{t-1} + g_t
  θ_t = θ_{t-1} − α · (μ · v_t + g_t)

The Nesterov form is equivalent to adding an extra momentum step before the
gradient step.  It often gives faster convergence because the gradient
evaluation is closer to the next iterate.

Weight decay
------------
Weight decay is implemented here as the standard *coupled* L2 form — adding
λ·θ to the gradient before the velocity update.  This means the decay term
also accumulates in the velocity buffer (unlike AdamW's decoupled form).

Practical tips
--------------
  • Momentum μ = 0.9 is a near-universal default; try 0.95–0.99 for very
    smooth loss surfaces (e.g. wide-batch training).
  • Nesterov is almost always preferable to standard momentum — enable it.
  • SGD+Nesterov + cosine schedule is competitive with or better than Adam on
    image classification (ResNets, ViTs) when properly tuned.
  • Learning rate is more sensitive than in Adam; typical range 0.01–0.1.
"""

import torch
from .base import BaseOptimizer


class SGDMomentum(BaseOptimizer):
    """SGD with momentum and optional Nesterov acceleration.

    Parameters
    ----------
    params       : iterable of parameters or param groups
    lr           : learning rate α (default 1e-2; SGD typically uses higher
                   LRs than Adam because there is no per-parameter scaling)
    momentum     : momentum coefficient μ (default 0.9).
                   0.0 = plain SGD (same as VanillaSGD but with Nesterov option)
    weight_decay : L2 regularisation (coupled form, default 0.0)
    nesterov     : if True, use Nesterov momentum (default True; usually better)
    dampening    : reduces the gradient contribution to the velocity update:
                   v_t = μ·v_{t-1} + (1−dampening)·g_t.
                   Has no effect when nesterov=True (must be 0 with Nesterov).
                   (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        dampening: float = 0.0,
    ):
        # Nesterov requires dampening=0 (otherwise the lookahead is inconsistent)
        if nesterov and dampening != 0.0:
            raise ValueError("Nesterov momentum requires dampening=0.0")

        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            nesterov=nesterov, dampening=dampening,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one SGD+Momentum step across all parameter groups.

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
            mu       = group["momentum"]
            wd       = group["weight_decay"]
            nesterov = group["nesterov"]
            dampening = group["dampening"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Coupled weight decay ──────────────────────────────────
                # Add λ·θ to the gradient so that the regularisation signal
                # also flows into the velocity buffer.
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # ── Velocity update ───────────────────────────────────────
                if mu != 0.0:
                    if "velocity" not in state:
                        # First step: initialise velocity with the current gradient
                        # (no prior momentum to mix with yet).
                        state["velocity"] = g.clone()
                        buf = state["velocity"]
                    else:
                        buf = state["velocity"]
                        # v_t = μ · v_{t-1} + (1 − dampening) · g_t
                        # dampening < 1 reduces the gradient's influence on velocity
                        # (smooths the trajectory at the cost of slower convergence)
                        buf.mul_(mu).add_(g, alpha=1.0 - dampening)

                    if nesterov:
                        # Nesterov lookahead: use the gradient at the next position
                        # approximated as g_t + μ·v_t (one extra momentum step)
                        # This effectively looks one step ahead along the velocity.
                        g = g.add(buf, alpha=mu)
                    else:
                        # Standard momentum: update direction = current velocity
                        g = buf

                # ── Parameter update ──────────────────────────────────────
                # θ ← θ − α · g   (where g is either raw grad or Nesterov update)
                p.add_(g, alpha=-lr)

        return loss
