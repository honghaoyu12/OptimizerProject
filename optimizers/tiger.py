"""Tiger optimizer.

From "TIGER: Tight Gradient Estimates for Training via Momentum" — a
sign-based optimizer that, like Lion, uses a momentum buffer to produce
uniform-magnitude updates.  The key difference from Lion is *when* the
momentum is updated relative to the parameter step:

  Lion  :  update = sign(β₁·m + (1-β₁)·g)   then   m ← β₂·m + (1-β₂)·g
  Tiger :  update = sign(β₁·g + (1-β₁)·m)   then   m ← β₂·m + (1-β₂)·g

In Lion β₁ weights the *old* momentum m more heavily (typical: β₁=0.9);
in Tiger β₁ weights the *current gradient* g more heavily (typical: β₁=0.01).
This makes Tiger more reactive to the current gradient while still benefiting
from momentum smoothing — effectively combining gradient tracking with a
soft sign update.

Algorithm
---------
At each step t:

  1. Compute the signed update:
       u_t = sign(β₁ · g_t + (1 − β₁) · m_{t-1})

  2. Apply decoupled weight decay and parameter update:
       θ_t = θ_{t-1} · (1 − lr · λ)  − lr · u_t

  3. Update the gradient EMA momentum buffer:
       m_t = β₂ · m_{t-1} + (1 − β₂) · g_t

The momentum buffer m is initialised to zero, so the first update is
just sign(β₁ · g), i.e. dominated by the current gradient with a slight
zero-momentum regularisation.

Practical notes
---------------
  • Use a 3–10× smaller lr than Adam (e.g. 1e-4 to 3e-4).
  • β₁ ≈ 0.01 keeps the update responsive to the current gradient.
  • β₂ ≈ 0.99 maintains a smooth long-horizon momentum buffer.
  • Weight decay of 0.01–0.1 is recommended for regularised tasks.
"""

import torch
from .base import BaseOptimizer


class Tiger(BaseOptimizer):
    """Tiger: gradient-weighted sign momentum optimizer.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-4)
    betas        : (β₁, β₂) — gradient weight in the sign mix and EMA decay
                   of the momentum buffer.  β₁ ≈ 0.01, β₂ ≈ 0.99 recommended.
    weight_decay : decoupled weight decay coefficient (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.01, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Tiger step across all parameter groups.

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
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            wd           = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise momentum buffer on first step ───────────────
                # Zero initialisation means the very first update is
                # u_0 = sign(β₁·g + (1-β₁)·0) = sign(β₁·g) = sign(g),
                # preserving the correct update direction from step one.
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]

                # ── Step 1: Compute signed update ──────────────────────────
                # Tiger weights the current gradient β₁ and the accumulated
                # momentum (1-β₁).  With β₁ ≈ 0.01 the momentum dominates
                # after the first few steps, but the gradient still contributes
                # a fresh signal each iteration — preventing momentum from
                # getting "stuck" in a stale direction.
                update = (beta1 * g + (1.0 - beta1) * m).sign()

                # ── Step 2: Decoupled weight decay + parameter update ──────
                # Weight decay is applied *before* the sign update — identical
                # to Lion / AdamW style.  This shrinks weights multiplicatively
                # independent of the gradient direction.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update, alpha=-lr)

                # ── Step 3: Update momentum EMA (AFTER the parameter step) ─
                # Using a high β₂ (0.99) keeps the long-horizon gradient
                # direction stable while still adapting to gradient drift.
                m.mul_(beta2).add_(g, alpha=1.0 - beta2)

        return loss
