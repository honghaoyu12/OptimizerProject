"""LAMB optimizer.

From "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"
(You et al., ICLR 2020).  https://arxiv.org/abs/1904.00962

Motivation
----------
When training with very large mini-batches (e.g. 32 768 BERT samples), plain
Adam becomes numerically unstable: the adaptive denominator √v effectively
normalises gradient directions, but the resulting step sizes become highly
non-uniform across layers.  Some layers receive huge effective learning rates
and diverge while others stagnate.

LAMB adds a *layer-wise trust ratio* r_t that re-scales each tensor's update
by the ratio of its weight norm to its update norm:

    r_t = ‖p_{t-1}‖₂ / ‖u_t‖₂

This keeps the effective step size proportional to the current weight magnitude
for every layer simultaneously, even when the base LR is scaled up ×1 000.

Algorithm
---------
For each parameter tensor p with gradient g:

  1. Compute Adam moments:
       m_t = β₁ · m_{t-1} + (1−β₁) · g_t        [first moment EMA]
       v_t = β₂ · v_{t-1} + (1−β₂) · g_t²       [second moment EMA]

  2. Bias-correct:
       m̂_t = m_t / (1 − β₁^t)
       v̂_t = v_t / (1 − β₂^t)

  3. Adam-style update direction including weight decay:
       u_t = m̂_t / (√v̂_t + ε) + λ · p_{t-1}

  4. Layer-wise trust ratio:
       r_t = ‖p_{t-1}‖₂ / ‖u_t‖₂   (clipped to [0, 10] for stability)
       If either norm is zero, r_t = 1 (no scaling).

  5. Parameter update:
       p_t = p_{t-1} − α · r_t · u_t

Notes
-----
  - The clip at 10 prevents the trust ratio from exploding when u_t is very
    small (e.g. near the end of training when gradients are tiny).
  - LAMB was designed for large-batch training; at small batch sizes it
    behaves very similarly to Adam with weight decay.
"""

import torch
from .base import BaseOptimizer


class LAMB(BaseOptimizer):
    """LAMB: Layer-wise Adaptive Moments for Batch training.

    Parameters
    ----------
    params       : model parameters
    lr           : global learning rate (default 1e-3)
    betas        : (β₁, β₂) — Adam moment EMA rates (default (0.9, 0.999))
    eps          : numerical stability constant in the adaptive denominator
                   (default 1e-6; slightly larger than Adam's 1e-8 for stability
                   at large LR)
    weight_decay : L2 regularisation coefficient, included in the update
                   direction u_t before computing the trust ratio (default 0.01;
                   non-zero by default because LAMB is designed for regularised
                   training)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
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
            eps          = group["eps"]
            wd           = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise Adam moments on first step ─────────────────
                if len(state) == 0:
                    state["step"]        = 0
                    state["exp_avg"]     = torch.zeros_like(p)   # m (first moment)
                    state["exp_avg_sq"]  = torch.zeros_like(p)   # v (second moment)

                state["step"] += 1
                t    = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]

                # ── Steps 1–2: Adam moments with bias correction ───────────
                # m_t = β₁·m_{t-1} + (1-β₁)·g_t
                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                # v_t = β₂·v_{t-1} + (1-β₂)·g²   (element-wise square via addcmul)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # Bias-corrected moments: divide by (1 - β^t) to remove the
                # initialisation-at-zero bias from the EMA.
                m_hat = m / (1.0 - beta1 ** t)   # shape same as p
                v_hat = v / (1.0 - beta2 ** t)   # shape same as p

                # ── Step 3: Adam update direction + weight decay ───────────
                # u_t = m̂_t / (√v̂_t + ε) + λ·p
                # Weight decay is folded into u before the trust ratio so that
                # the trust ratio can scale *both* the gradient and the decay term.
                u = m_hat / (v_hat.sqrt() + eps)
                if wd != 0.0:
                    u.add_(p, alpha=wd)

                # ── Step 4: Layer-wise trust ratio ────────────────────────
                # r_t = ‖p‖ / ‖u‖  — measures how big u is relative to p.
                # When u is large (aggressive step), r < 1 → conservative update.
                # When u is small (conservative step), r > 1 → amplify.
                # Clamp at 10 to prevent blow-up when u is near zero.
                p_norm = p.norm()    # scalar: ‖p_{t-1}‖₂
                u_norm = u.norm()    # scalar: ‖u_t‖₂

                if p_norm == 0.0 or u_norm == 0.0:
                    # If either norm is zero the ratio is undefined → fallback to 1
                    trust_ratio = 1.0
                else:
                    trust_ratio = float((p_norm / u_norm).clamp(max=10.0))

                # ── Step 5: Parameter update ──────────────────────────────
                # p_t = p_{t-1} − α · r_t · u_t
                p.add_(u, alpha=-lr * trust_ratio)

        return loss
