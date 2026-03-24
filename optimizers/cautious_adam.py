"""Cautious Adam optimizer.

From "Cautious Optimizers: Improving Training with One Line of Code"
(Zhu et al., 2024).  https://arxiv.org/abs/2411.16085

Motivation
----------
Standard Adam applies its momentum-based update even when the update direction
disagrees with the current gradient.  In such cases the update moves the
parameters *away* from where the loss is locally decreasing, potentially
overshooting or destabilising training.

Cautious Adam adds a single masking step: before applying the update, check
whether each coordinate of the Adam update has the same sign as the
corresponding raw gradient.  If not — i.e., momentum is pointing against the
current gradient — zero out that coordinate's update.

This "caution" prevents momentum from overriding a clear local gradient signal.
The cost is essentially zero (one element-wise comparison), and the benefit can
be large in early training or on tasks where gradient directions shift rapidly.

Algorithm
---------
For each parameter p with gradient g at step t:

  1. Standard Adam moments:
       m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
       v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²
       m̂_t = m_t / (1 − β₁^t)
       v̂_t = v_t / (1 − β₂^t)

  2. Compute raw update direction:
       u_t = m̂_t / (√v̂_t + ε)

  3. Agreement mask (the "cautious" step):
       mask = (u_t · g_t > 0)      ← element-wise sign agreement
       u_t = u_t * mask            ← zero out disagreeing coordinates

  4. Re-scale by mask density to maintain the same expected update magnitude:
       u_t = u_t / max(mean(mask), δ)   where δ = 1e-8 to avoid division by zero

  5. Apply update:
       θ_t = θ_{t-1} − α · u_t

The rescaling in step 4 is important: without it, zeroing out a fraction of
the update coordinates would artificially shrink the effective learning rate.
Dividing by mean(mask) restores the original expected update magnitude among
the agreeing coordinates.

Weight decay is applied in decoupled form (AdamW-style): the parameter is
scaled by (1 − α·λ) before the gradient update.

Practical tips
--------------
  • Use the same lr/betas/eps as Adam — cautious masking does not require
    retuning these hyperparameters.
  • Most beneficial on tasks with noisy gradients or in early training epochs
    where momentum direction can diverge from the current gradient.
  • Conceptually similar to gradient clipping but acts per-coordinate rather
    than globally.
"""

import math

import torch
from .base import BaseOptimizer


class CautiousAdam(BaseOptimizer):
    """Cautious Adam — momentum masking for safer updates.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate α (default 1e-3)
    betas        : (β₁, β₂) — EMA rates for first and second moments
                   (default (0.9, 0.999))
    eps          : numerical stability constant (default 1e-8)
    weight_decay : decoupled weight decay coefficient (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one Cautious Adam step across all parameter groups.

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
            eps          = group["eps"]
            wd           = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on first step ─────────────────────────
                if len(state) == 0:
                    state["step"]        = 0
                    state["exp_avg"]     = torch.zeros_like(p)  # first moment m
                    state["exp_avg_sq"]  = torch.zeros_like(p)  # second moment v

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # ── Decoupled weight decay (AdamW-style) ───────────────────
                # Scale parameters by (1 − α·λ) before the gradient update.
                # Decoupled means decay is independent of the second moment v_t,
                # unlike coupled L2 which adds λ·θ to the gradient.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # ── Steps 1: Standard Adam moment updates ──────────────────
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias correction: early steps would otherwise underestimate
                # both moments because m and v start at zero.
                bias_corr1 = 1.0 - beta1 ** t
                bias_corr2 = 1.0 - beta2 ** t
                m_hat = m / bias_corr1   # unbiased first moment
                v_hat = v / bias_corr2   # unbiased second moment

                # ── Step 2: Raw Adam update direction ──────────────────────
                # u_t = m̂ / (√v̂ + ε)  — the usual Adam step, not yet masked.
                denom = v_hat.sqrt().add_(eps)
                update = m_hat / denom   # shape matches p

                # ── Step 3: Caution mask — sign agreement check ────────────
                # A coordinate of the update "agrees" with the gradient if they
                # have the same sign, i.e. their product is positive.
                # Disagreement means momentum is pointing against the current
                # gradient signal; we suppress those coordinates entirely.
                #
                # mask[i] = 1  if update[i] * g[i] > 0   (same direction)
                # mask[i] = 0  if update[i] * g[i] <= 0  (opposite / orthogonal)
                mask = (update * g > 0).to(dtype=p.dtype)

                # Apply mask: zero out updates that conflict with the gradient.
                update = update * mask

                # ── Step 4: Rescale by mask density ───────────────────────
                # Without rescaling, zeroing out coordinates would shrink the
                # effective update norm by a factor of mean(mask) — equivalent
                # to an implicit reduction in learning rate proportional to how
                # many coordinates were masked.
                #
                # Dividing by mean(mask) restores the original magnitude among
                # the agreeing coordinates.  The 1e-8 floor prevents division by
                # zero in the degenerate case where the entire update is masked.
                mask_density = mask.mean().clamp(min=1e-8)
                update = update / mask_density

                # ── Step 5: Apply update ───────────────────────────────────
                p.add_(update, alpha=-lr)

        return loss
