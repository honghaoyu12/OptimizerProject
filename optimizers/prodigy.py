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
            # We are under @torch.no_grad(); enable_grad lets the closure call
            # backward() while still keeping the outer no_grad context intact.
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr    = group["lr"]
            beta1, beta2 = group["betas"]
            eps   = group["eps"]
            wd    = group["weight_decay"]
            # d: the current step-size estimate (lives in param_group, not state,
            # so it is shared across all parameters in the group and survives
            # state_dict() / load_state_dict() round-trips).
            d = group["d"]
            # A: exponentially-weighted sum of d²·‖g‖² — the denominator of the
            # d-update formula.  Also lives in param_group for the same reason.
            A = group["A"]

            # Only update parameters that actually received a gradient
            params_with_grad = [p for p in group["params"] if p.grad is not None]
            if not params_with_grad:
                continue

            # ── Pass 0: Initialise per-parameter state on the very first step ─
            for p in params_with_grad:
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # x0: snapshot of the initial parameter values.
                    # Frozen after the first step — never updated again.
                    # Used in d_num = <s, x0 - x> to measure how far the
                    # parameters have moved from their starting point.
                    state["x0"] = p.data.clone()
                    # s: exponentially-weighted gradient accumulator, scaled by d.
                    # Tracks how much gradient signal has been accumulated in
                    # each parameter direction (the "momentum of d").
                    state["s"] = torch.zeros_like(p)
                    # m: first moment buffer (mean of d-scaled gradients)
                    state["exp_avg"] = torch.zeros_like(p)
                    # v: second moment buffer (mean squared d-scaled gradients)
                    state["exp_avg_sq"] = torch.zeros_like(p)

            # ── Pass 1: d estimation (global quantities over all parameters) ──
            # Prodigy's key novelty: it uses a running estimate of the *distance*
            # to the optimum (d) instead of a fixed learning rate.  d is estimated
            # from the inner product between the gradient accumulator s and the
            # displacement from the initial parameters (x0 − x).
            #
            # Theory: on a quadratic f(x) with minimiser x*, the true step-size
            # that minimises loss in one step is ‖x - x*‖.  The numerator d_num
            # approximates this distance using the EMA-accumulated gradient
            # direction s (which points toward x* on quadratics).
            d_num    = 0.0    # numerator:   Σ_i <s_new,i, x0,i − x_i>
            g_sq_sum = 0.0    # denominator: d² · Σ_i ‖g_i‖²  (normaliser for A)
            new_s: dict[int, torch.Tensor] = {}   # staged s updates (committed after)

            for p in params_with_grad:
                g     = p.grad
                state = self.state[p]

                # ── Update s: s_k = β₂ · s_{k-1} + (1−β₂) · d · g ──────────
                # s accumulates a running d-scaled gradient sum with EMA decay β₂.
                # Multiplying g by d before accumulating is critical: it ensures
                # that s grows proportionally to the effective LR, which makes
                # d_num scale correctly.
                # We stage the update in new_s rather than writing directly to
                # state["s"] so that d_num below uses the updated s value
                # (consistent with the paper's update order).
                s_new = state["s"].mul(beta2).add_(g, alpha=(1.0 - beta2) * d)
                new_s[id(p)] = s_new

                # ── Inner product contribution: <s_new, x_0 − x_k> ───────────
                # This measures how well the accumulated gradient direction s
                # aligns with the displacement from the initial point.
                # On a convex quadratic, this equals approximately d · ‖x* − x‖²,
                # which is the distance-to-optimum signal Prodigy exploits.
                d_num += (s_new * (state["x0"] - p.data)).sum().item()

                # ── Squared gradient norm: d² · ‖g‖² ─────────────────────────
                # This becomes the denominator of d_new = d_num / √(A + ε).
                # Scaling by d² is necessary because s is d-scaled, so d_num
                # scales as d · (distance), and we want d_new = (distance).
                g_sq_sum += (d ** 2) * g.norm().item() ** 2

            # ── Update global A (EMA of d²·‖g‖²) ─────────────────────────────
            # A_k = β₂ · A_{k-1} + (1−β₂) · d² · ‖g‖²
            # A acts as a normaliser: √A ≈ d · ‖g‖_RMS, which cancels the
            # gradient-scale dependence in d_new so d estimates pure distance.
            A_new = beta2 * A + (1.0 - beta2) * g_sq_sum

            # ── Update d (monotonically non-decreasing) ───────────────────────
            # d_new = max(d_old, d_num / (√A_new + ε))
            # The max ensures d never decreases — once we have discovered
            # that the optimum is at least d away, we never shrink our step size.
            # Guard: if A_new == 0 (first step, no gradient yet), keep d as-is.
            d_new = max(d, d_num / (A_new ** 0.5 + eps)) if A_new > 0 else d

            # Write updated d and A back to the param_group (shared state)
            group["d"] = d_new
            group["A"] = A_new

            # ── Commit staged s values into per-parameter state ───────────────
            # We waited until after d_new is computed to write s because the
            # d_num computation needed s_new but d_new is computed after the loop.
            # Using copy_() is equivalent to an in-place assignment but works
            # correctly even if s and new_s[id(p)] share underlying storage.
            for p in params_with_grad:
                self.state[p]["s"].copy_(new_s[id(p)])

            # ── Pass 2: Adam-style parameter update with effective LR = d·lr ──
            # Once d is updated, the effective learning rate for this step is
            # d_new * lr.  The user-supplied lr is a global multiplier (default 1.0)
            # that biases the self-tuned d.  Setting lr > 1.0 makes Prodigy
            # more aggressive; lr < 1.0 makes it more conservative.
            eff_lr = d_new * lr

            for p in params_with_grad:
                g     = p.grad
                state = self.state[p]
                state["step"] += 1     # increment step counter (used nowhere yet, for logging)

                m = state["exp_avg"]       # first moment buffer
                v = state["exp_avg_sq"]    # second moment buffer

                # ── First moment: m_k = β₁ · m_{k-1} + (1−β₁) · eff_lr · g ──
                # Critically, g is scaled by eff_lr *before* the EMA, not after.
                # This means m absorbs the scale of eff_lr, so the denominator
                # below (√v + eff_lr · ε) is in the same units.
                m.mul_(beta1).add_(g, alpha=(1.0 - beta1) * eff_lr)

                # ── Second moment: v_k = β₂ · v_{k-1} + (1−β₂) · (eff_lr · g)² ─
                # Similarly, g is weighted by eff_lr² = (d·lr)², so v tracks
                # the expected squared magnitude of the effective gradient signal.
                v.mul_(beta2).addcmul_(g, g, value=(1.0 - beta2) * eff_lr ** 2)

                # ── Decoupled weight decay ─────────────────────────────────────
                # Scale by eff_lr so decay strength stays proportional to the
                # effective learning rate (analogous to AdamW's lr · λ decay).
                if wd != 0.0:
                    p.mul_(1.0 - wd * eff_lr)

                # ── Parameter update: θ ← θ − m / (√v + eff_lr · ε) ──────────
                # The denominator eff_lr · ε is in the same units as √v
                # (both are scaled by eff_lr), preventing the numeric floor ε
                # from becoming irrelevantly tiny when eff_lr is large.
                # addcdiv_(m, denom, value=-1.0) performs: θ ← θ − 1.0 · m/denom
                denom = v.sqrt().add_(eff_lr * eps)
                p.addcdiv_(m, denom, value=-1.0)

        return loss
