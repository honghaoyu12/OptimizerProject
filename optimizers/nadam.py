"""Custom NAdam optimizer.

From "Incorporating Nesterov Momentum into Adam"
(Dozat, ICLR 2016 Workshop).  https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ

Motivation
----------
Nesterov Accelerated Gradient (NAG) improves over classical momentum by
evaluating the gradient at the *lookahead* position (where the momentum would
take the parameters) rather than at the current position.  This gives a more
informed gradient estimate and often results in faster convergence.

NAdam incorporates the Nesterov idea into Adam by replacing the standard
first-moment term in the update with a lookahead version.  Concretely, the
update uses the *next* step's bias-corrected first moment (m̂_{t+1}) rather
than the current step's (m̂_t), which is equivalent to computing the gradient
one step ahead along the momentum direction.

Algorithm
---------
For each parameter p with gradient g at step t:

  1. First moment:   m_t  = β₁ · m_{t-1} + (1 − β₁) · g_t
  2. Second moment:  v_t  = β₂ · v_{t-1} + (1 − β₂) · g_t²
  3. Bias-correct:   v̂_t = v_t / (1 − β₂^t)
  4. Nesterov update direction (lookahead first moment):
       ĝ_t = β₁ · m_t / (1 − β₁^{t+1}) + (1 − β₁) · g_t / (1 − β₁^t)
  5. Parameter update:
       θ_t = θ_{t-1} − α · ĝ_t / (√v̂_t + ε)

Step 4 is derived by noticing that in NAG, the update uses g_{t} evaluated at
θ_{t-1} − α·β₁·m_{t-1}.  Substituting into Adam's update and simplifying
gives this closed-form expression in terms of m_t and g_t.

Practical tips
--------------
  • Often slightly faster convergence than Adam on smooth loss surfaces.
  • Performs well on RNNs and transformers where gradient direction is stable.
  • Same default hyperparameters as Adam work well; ε can be left at 1e-8.
"""

import math

import torch
from .base import BaseOptimizer


class NAdam(BaseOptimizer):
    """NAdam: Adam with Nesterov momentum.

    Parameters
    ----------
    params           : iterable of parameters or param groups
    lr               : global learning rate α (default 2e-3; NAdam often
                       tolerates a slightly larger LR than Adam)
    betas            : (β₁, β₂) — EMA decay rates (default (0.9, 0.999))
    eps              : numerical stability constant (default 1e-8)
    weight_decay     : L2 regularisation (coupled form, default 0.0)
    momentum_decay   : per-step decay applied to β₁ each step, giving
                       schedule μ_t = β₁ · (1 − 0.5 · 0.96^{t/250}).
                       Matches the original paper's "momentum schedule"
                       (default 4e-3).  Set to 0 to disable.
    """

    def __init__(
        self,
        params,
        lr: float = 2e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum_decay: float = 4e-3,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one NAdam step across all parameter groups.

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
            lr             = group["lr"]
            beta1, beta2   = group["betas"]
            eps            = group["eps"]
            wd             = group["weight_decay"]
            mu_decay       = group["momentum_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    state["step"]              = 0
                    state["exp_avg"]           = torch.zeros_like(p)
                    state["exp_avg_sq"]        = torch.zeros_like(p)
                    # mu_product tracks ∏_{i=1}^{t} μ_i, needed for exact
                    # bias correction when momentum is scheduled.
                    state["mu_product"]        = 1.0

                state["step"] += 1
                t    = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # ── Coupled weight decay ──────────────────────────────────
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                # ── Momentum schedule ─────────────────────────────────────
                # μ_t is a per-step effective β₁ that decays over time.
                # The schedule (from Sutskever et al. / original NAdam paper):
                #   μ_t = β₁ · (1 − 0.5 · 0.96^{t/250})
                # This gradually reduces momentum, stabilising later training.
                mu_t      = beta1 * (1.0 - 0.5 * (0.96 ** (t * mu_decay)))
                # μ_{t+1}: the momentum at the NEXT step — needed for the
                # Nesterov lookahead update direction.
                mu_t1     = beta1 * (1.0 - 0.5 * (0.96 ** ((t + 1) * mu_decay)))

                # Running product of all μ values up to step t (and t+1)
                mu_prod_t  = state["mu_product"] * mu_t
                mu_prod_t1 = mu_prod_t * mu_t1
                state["mu_product"] = mu_prod_t   # save for next iteration

                # ── Steps 1 & 2: Moment updates ───────────────────────────
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # ── Step 3: Second moment bias correction ─────────────────
                bias_corr2 = 1.0 - beta2 ** t
                v_hat      = v / bias_corr2   # unbiased variance estimate

                # ── Step 4: Nesterov update direction ─────────────────────
                # Combines lookahead first moment (m_t biased toward t+1)
                # with the current raw gradient scaled by its own bias term.
                # This is mathematically equivalent to evaluating the gradient
                # one momentum step ahead.
                bias_corr1      = 1.0 - beta1 ** t    # bias correction for step t
                bias_corr1_next = 1.0 - beta1 ** (t + 1)  # bias correction for t+1

                # Nesterov gradient direction:
                #   ĝ = (β₁ · m_t / (1−β₁^{t+1})) + ((1−β₁) · g / (1−β₁^t))
                nesterov_grad = m.mul(beta1 / bias_corr1_next).add(
                    g, alpha=(1.0 - beta1) / bias_corr1
                )

                # ── Step 5: Parameter update ──────────────────────────────
                # θ ← θ − α · ĝ / (√v̂ + ε)
                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(nesterov_grad, denom, value=-lr)

        return loss
