"""Prodigy optimizer.

From "Prodigy: An Expeditiously Adaptive Parameter-Free Learning Rate Method"
(Mishchenko & Defazio, NeurIPS 2023).  https://arxiv.org/abs/2306.06101

Prodigy automatically estimates the distance D from the initial parameters to
the optimum and uses it to set the effective learning rate — no LR tuning
required.  The user-supplied `lr` acts as a global multiplier (default 1.0).

The estimator tracks two global scalars:
  A_k  — exponentially-weighted sum of squared gradient norms, scaled by d²
  d_k  — the current step-size estimate (monotonically non-decreasing)

And one per-parameter vector:
  s_k  — exponentially-weighted gradient accumulator, scaled by d

The d update rule is:
  d_num = Σ_i <s_k,i, x_0,i − x_k,i>   (inner product across all params)
  d_k   = max(d_{k−1}, d_num / √(A_k + ε))

The parameter update is Adam-like with effective LR = d_k · lr:
  m_k = β₁ · m_{k−1} + (1−β₁) · d_k · lr · g_k
  v_k = β₂ · v_{k−1} + (1−β₂) · (d_k · lr)² · g_k²
  θ_k = θ_{k−1} − m_k / (√v_k + d_k · lr · ε) − λ · d_k · lr · θ_{k−1}

Key properties:
  • d grows monotonically (max ensures it never shrinks).
  • On quadratic problems it recovers the exact step size D/√k asymptotically.
  • The inner product d_num measures how well the gradients are "aligned" with
    the direction from the initial point to the current point — a proxy for
    progress toward the optimum.
"""

import torch
from .base import BaseOptimizer


class Prodigy(BaseOptimizer):
    """Prodigy: parameter-free adaptive optimizer.

    Parameters
    ----------
    params       : model parameters
    lr           : global LR multiplier on top of the estimated d (default 1.0;
                   leave at 1.0 to let Prodigy self-tune; scale up/down to bias)
    betas        : (β₁, β₂) — momentum and second-moment EMA rates
                   (default (0.9, 0.999))
    eps          : numerical stability constant (default 1e-8)
    d0           : initial step-size estimate — very small so d can only grow
                   (default 1e-6)
    weight_decay : decoupled weight decay coefficient (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        d0: float = 1e-6,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, d0=d0,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        # Initialise d and A in each param group (survive state_dict via param_groups)
        for group in self.param_groups:
            group["d"] = d0
            group["A"] = 0.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            d = group["d"]
            A = group["A"]

            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue

            # ── Initialise per-param state ────────────────────────────────
            for p in params_with_grad:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["x0"] = p.data.clone()       # initial weights (fixed reference)
                    state["s"] = torch.zeros_like(p)   # d-weighted gradient accumulator
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

            # ── d estimation (pass 1: accumulate global quantities) ───────
            # Update s and compute d_num and g_sq_sum before updating params
            d_num = 0.0
            g_sq_sum = 0.0
            new_s: dict[int, torch.Tensor] = {}

            for p in params_with_grad:
                g = p.grad
                state = self.state[p]
                # s_k = β₂·s_{k−1} + (1−β₂)·d·g
                s_new = state["s"].mul(beta2).add_(g, alpha=(1.0 - beta2) * d)
                new_s[id(p)] = s_new
                # Inner product contribution: <s_new, x_0 − x_k>
                d_num += (s_new * (state["x0"] - p.data)).sum().item()
                # Squared gradient norm contribution: d²·‖g‖²
                g_sq_sum += (d ** 2) * g.norm().item() ** 2

            # Update global A and d
            A_new = beta2 * A + (1.0 - beta2) * g_sq_sum
            d_new = max(d, d_num / (A_new ** 0.5 + eps)) if A_new > 0 else d
            group["d"] = d_new
            group["A"] = A_new

            # Commit updated s into state
            for p in params_with_grad:
                self.state[p]["s"].copy_(new_s[id(p)])

            # ── Adam update with effective LR = d_new · lr ────────────────
            eff_lr = d_new * lr

            for p in params_with_grad:
                g = p.grad
                state = self.state[p]
                state["step"] += 1

                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # First and second moment updates (scaled by effective LR)
                m.mul_(beta1).add_(g, alpha=(1.0 - beta1) * eff_lr)
                v.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2) * eff_lr ** 2)

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - wd * eff_lr)

                # Parameter update: −m / (√v + d·lr·ε)
                denom = v.sqrt().add_(eff_lr * eps)
                p.addcdiv_(m, denom, value=-1.0)

        return loss
