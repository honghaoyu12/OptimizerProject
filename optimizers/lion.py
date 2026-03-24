"""Lion optimizer.

From "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023).
https://arxiv.org/abs/2302.06675

Lion was found by a symbolic search over possible optimizers and differs from
Adam in two key ways:
  1. Only one momentum buffer (m) — no second-moment (v).  Memory footprint
     is half of Adam for the same model.
  2. The update is the *sign* of an interpolated gradient — every parameter
     receives a step of the same absolute magnitude each iteration.  This
     makes the effective step size completely controlled by the learning rate.

The two momentum roles (interpolation vs. EMA decay) are *deliberately
different* (β₁ ≠ β₂):
  β₁ governs the interpolation used to form the signed update direction.
  β₂ governs the EMA that carries the momentum forward.
Using β₁ < β₂ means the update direction is more responsive to the current
gradient than the stored momentum — a form of "look-ahead" similar to
Nesterov momentum.

Algorithm
---------
Given momentum m_{t-1}, gradient g_t, parameters θ_{t-1}:

  1. Compute signed update:  c_t = sign(β₁ · m_{t-1} + (1−β₁) · g_t)
  2. Apply weight decay:     c_t = c_t + λ · θ_{t-1}
  3. Update parameters:      θ_t = θ_{t-1} − α · c_t
  4. Update momentum:        m_t = β₂ · m_{t-1} + (1−β₂) · g_t

Note: weight decay is applied AFTER the sign (decoupled), keeping its
magnitude proportional to the parameter norm rather than the gradient norm.

Practical tips
--------------
  • Use ~10× smaller LR than Adam (sign updates are much larger in magnitude).
  • Works best with substantial weight decay (0.01–1.0) to prevent divergence.
  • Particularly effective for large language and vision models.
"""

import torch
from .base import BaseOptimizer


class Lion(BaseOptimizer):
    """Lion: EvoLved Sign Momentum optimizer.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-4; roughly 10× smaller than Adam
                   is a typical starting point because sign steps are ≈10×
                   larger than Adam's normalised gradient updates)
    betas        : (β₁, β₂)
                   β₁ = interpolation weight for the sign update direction
                   β₂ = EMA decay for the momentum buffer
                   Default (0.9, 0.99) from the paper.
    weight_decay : decoupled weight decay applied after the sign (default 0.0)
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
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            wd           = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad                  # current gradient
                state = self.state[p]

                # ── Initialise momentum buffer on first call ──────────────
                if len(state) == 0:
                    # Zeros: equivalent to no prior momentum before the first step
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]            # momentum buffer from previous step

                # ── Step 1: Signed update direction ───────────────────────
                # Interpolate between stored momentum and current gradient,
                # then take the sign.  The sign discards magnitude information —
                # every update moves by exactly lr in the ±1 direction per element.
                # WHY sign: normalises per-coordinate step sizes; provably optimal
                # under the "linf ball" trust region (Lion paper, Theorem 1).
                update = (beta1 * m + (1.0 - beta1) * g).sign()

                # ── Step 2: Decoupled weight decay ────────────────────────
                # Added to the signed update so its effect scales with
                # parameter magnitude, not gradient magnitude.
                if wd != 0.0:
                    update.add_(p, alpha=wd)

                # ── Step 3: Parameter update ──────────────────────────────
                # θ ← θ − α · update
                p.add_(update, alpha=-lr)

                # ── Step 4: Momentum EMA (AFTER the parameter update) ─────
                # WHY β₂ > β₁: the momentum carries a longer memory than the
                # update uses — the update looks at "what is the gradient doing
                # now?" while the momentum asks "what has it been doing?".
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss
