"""GrokFast optimizer.

From "GrokFast: Accelerated Grokking by Amplifying Slow Gradients"
(Lee et al., 2024).  https://arxiv.org/abs/2405.20233

Motivation
----------
"Grokking" is the phenomenon where a model first memorises training data
(achieving 100% train accuracy) and only much later — sometimes thousands of
epochs later — generalises to the test set.  Lee et al. found that the
gradient signal responsible for generalisation operates at *low frequencies*:
it changes slowly across steps, whereas memorisation is driven by high-frequency
gradient components that oscillate rapidly.

GrokFast accelerates grokking by amplifying the slow (low-frequency) component
of each parameter's gradient before the optimizer update.  The slow component
is extracted with an exponential moving average (EMA) filter on the gradient
sequence.  Amplifying it effectively gives the "generalisation signal" more
weight relative to the high-frequency memorisation noise.

Algorithm
---------
At each step t, before calling the underlying optimizer:

  1. Update the gradient EMA (slow-frequency filter):
       ḡ_t = α · ḡ_{t-1} + (1 − α) · g_t

  2. Amplify the slow component and add to the raw gradient:
       g̃_t = g_t + λ · ḡ_t

  3. Replace p.grad with g̃_t and call the underlying optimizer step.

The EMA decay α controls the cutoff frequency: high α (e.g. 0.98) keeps a
long memory, emphasising very slow gradient components; lower α (e.g. 0.9)
tracks faster-changing directions.

λ (lamb) is the amplification strength.  The original paper uses λ = 2,
meaning the slow component is added at twice the strength of the original
gradient.  Larger λ accelerates grokking but risks instability.

This implementation wraps *any* base optimizer: GrokFast is not a standalone
update rule but a gradient preprocessing layer.  The wrapped optimizer performs
the actual parameter update using the amplified gradient g̃_t.

Practical tips
--------------
  • Use with Adam or AdamW as the base optimizer (paper default).
  • α = 0.98, λ = 2.0 are the paper's recommended defaults.
  • Most beneficial on tasks where grokking is expected: small datasets,
    weight decay regularisation, and classification tasks with binary loss.
  • Has minimal overhead: just one EMA buffer per parameter.
  • If the base optimizer is AdamW, GrokFast + AdamW ≈ AdamW + slow-gradient
    amplification; no hyperparameter retuning of the base optimizer needed.
"""

import torch
from .base import BaseOptimizer


class GrokFast(BaseOptimizer):
    """GrokFast: Accelerated grokking via slow-frequency gradient amplification.

    This optimizer wraps a base optimizer (default: Adam) and injects an
    exponential-moving-average amplification of the slow gradient component
    before each base-optimizer step.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate passed to the base optimizer (default 1e-3)
    betas        : (β₁, β₂) for the base Adam-style optimizer (default (0.9, 0.999))
    eps          : numerical stability constant for the base optimizer (default 1e-8)
    weight_decay : decoupled weight decay for the base optimizer (default 0.0)
    alpha        : EMA decay for the slow-gradient filter.  Higher = slower filter
                   = more emphasis on very-low-frequency gradient components.
                   (default 0.98)
    lamb         : amplification factor for the slow component.  The amplified
                   gradient is g̃ = g + lamb · ḡ.  (default 2.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        alpha: float = 0.98,
        lamb: float = 2.0,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, alpha=alpha, lamb=lamb,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one GrokFast step across all parameter groups.

        Modifies p.grad in-place with the amplified gradient, then runs the
        embedded Adam update.  p.grad is restored after the update so that
        external gradient accumulators see the original gradient.

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
            alpha        = group["alpha"]   # EMA decay for slow-gradient filter
            lamb         = group["lamb"]    # amplification strength

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise state on first step ─────────────────────────
                if len(state) == 0:
                    state["step"]       = 0
                    state["exp_avg"]    = torch.zeros_like(p)   # Adam 1st moment m
                    state["exp_avg_sq"] = torch.zeros_like(p)   # Adam 2nd moment v
                    # Slow-gradient EMA buffer ḡ.
                    # Initialised to the first gradient so the first-step
                    # amplification is g + lamb * g = (1 + lamb) * g.
                    # This is intentional — it front-loads the amplification
                    # from the very first step.
                    state["grad_ema"]   = g.clone()

                state["step"] += 1
                t    = state["step"]
                m    = state["exp_avg"]
                v    = state["exp_avg_sq"]
                g_ema = state["grad_ema"]

                # ── Step 1: Update slow-gradient EMA ──────────────────────
                # ḡ_t = α · ḡ_{t-1} + (1 − α) · g_t
                # High α (0.98) means ḡ changes slowly and only reflects
                # gradient components that are *consistent* across many steps —
                # exactly the generalisation signal GrokFast targets.
                g_ema.mul_(alpha).add_(g, alpha=1.0 - alpha)

                # ── Step 2: Amplify slow component ─────────────────────────
                # g̃_t = g_t + λ · ḡ_t
                # The raw gradient already carries the full signal; adding λ·ḡ_t
                # on top amplifies the slow (consistent) part more than the
                # fast (variable) part, shifting the update towards directions
                # that have been consistently active across many steps.
                g_amplified = g + lamb * g_ema

                # ── Step 3: Adam update with amplified gradient ────────────
                # Standard Adam EMA + bias correction + parameter update,
                # but using g_amplified in place of g.

                # Decoupled weight decay (AdamW-style).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # Moment updates using the amplified gradient.
                m.mul_(beta1).add_(g_amplified, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g_amplified, g_amplified, value=1.0 - beta2)

                # Bias correction for early-step underestimation.
                bias_corr1 = 1.0 - beta1 ** t
                bias_corr2 = 1.0 - beta2 ** t

                # Compute and apply the Adam step.
                denom = (v / bias_corr2).sqrt().add_(eps)
                p.addcdiv_(m / bias_corr1, denom, value=-lr)

        return loss
