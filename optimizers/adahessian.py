"""AdaHessian optimizer.

From "ADAHESSIAN: An Adaptive Second Order Optimizer for Machine
Learning" (Yao et al., 2020).  https://arxiv.org/abs/2006.00719

AdaHessian is an Adam-like optimizer that replaces the squared gradient
(v_t) with an EMA of the *diagonal Hessian estimate*, computed cheaply
via Hutchinson's randomised estimator:

  For a random Rademacher vector z ~ {−1,+1}^d:
    E[z ⊙ (H·z)] = diag(H)

  where H is the Hessian matrix and ⊙ is element-wise product.

The Hessian-vector product H·z is obtained by differentiating the
first-order gradients, which requires those gradients to have been
computed with ``create_graph=True``.  When the computation graph is
not available (``p.grad.requires_grad is False``), AdaHessian falls
back to using the squared gradient — identical to Adam — so the
optimizer still runs correctly inside the standard training loop.

To unlock true AdaHessian behaviour, set the ``requires_create_graph``
class attribute; ``train_one_epoch()`` detects this and calls
``loss.backward(create_graph=True)`` automatically.

Algorithm
---------
  z_t  ~ Rademacher({−1,+1}^d)
  hz_t  = ∇(g_t · z_t)   [Hessian-vector product via autograd]
  d_t   = hz_t ⊙ z_t      [diagonal Hessian estimate, E[d_t] = diag(H)]

  m_t = β₁·m_{t-1} + (1−β₁)·g_t
  v_t = β₂·v_{t-1} + (1−β₂)·d_t²

  Bias-correct, then:
  θ_t = θ_{t-1} − lr · (m̂_t / (√v̂_t + ε) + λ·θ_{t-1})
"""

import torch
from .base import BaseOptimizer


class AdaHessian(BaseOptimizer):
    """AdaHessian: Adam with diagonal Hessian preconditioning.

    Parameters
    ----------
    params         : model parameters
    lr             : learning rate (default 0.1)
    betas          : (β₁, β₂) EMA decay rates (default (0.9, 0.999))
    eps            : numerical stability constant (default 1e-4)
    weight_decay   : L2 regularisation coefficient (default 0.0)
    """

    # Signal to train_one_epoch() that it should call
    # loss.backward(create_graph=True) for this optimizer.
    requires_create_graph: bool = True

    def __init__(
        self,
        params,
        lr: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-4,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Re-enable autograd so the closure can call .backward().
            # We are inside @torch.no_grad(), so this context manager is
            # necessary to allow gradient computation inside the closure.
            with torch.enable_grad():
                loss = closure()

        # ── Collect all parameters that received a gradient this step ────────
        # We gather them into a flat list so that the HVP loop below can
        # iterate over every parameter once and share the same computation graph.
        # Parameters without gradients (frozen layers, unused embeddings) are
        # skipped — they have no contribution to the Hessian diagonal.
        params_grads: list[tuple] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params_grads.append((p, p.grad, group))

        # ── Decide whether to compute true HVPs or fall back to g² ───────────
        # When loss.backward(create_graph=True) was called, PyTorch keeps the
        # computation graph alive and the gradient tensors carry a grad_fn
        # (they are non-leaf intermediate nodes in the graph).
        # With plain loss.backward() (create_graph=False, the default), the
        # gradients are materialised as leaf tensors with no grad_fn — we
        # cannot differentiate through them to get H·z.
        # Checking grad_fn on the first parameter's gradient is a fast proxy
        # for "was create_graph=True used?" — all parameters will agree.
        hessian_diags: dict[int, torch.Tensor] = {}
        use_hvp = params_grads and params_grads[0][1].grad_fn is not None

        if use_hvp:
            # ── Full Hutchinson estimator: one HVP per parameter ─────────────
            # We iterate over parameters one at a time so we can call
            # autograd.grad() with retain_graph=False on the last parameter,
            # freeing the computation graph from memory as soon as we are done.
            # For all earlier parameters, retain_graph=True keeps it alive so
            # the subsequent iterations can still use it.
            for i, (p, g, _) in enumerate(params_grads):
                # ── Draw a Rademacher vector z ~ Uniform({-1, +1}) ───────────
                # randint(0, 2) → {0, 1}; * 2.0 → {0.0, 2.0}; - 1.0 → {-1.0, +1.0}
                # Each element is independently ±1 with equal probability.
                # The key property: E[z ⊙ (H·z)] = diag(H) (see module docstring).
                z = torch.randint_like(g, 0, 2).float().mul_(2.0).sub_(1.0)
                is_last = (i == len(params_grads) - 1)

                # ── Hessian-vector product via autograd ───────────────────────
                # (g * z).sum() is a scalar equal to g^T z.
                # Its gradient w.r.t. p is ∂(g^T z)/∂p = (∂g/∂p)^T z = H·z,
                # because g = ∂L/∂p and H = ∂²L/∂p² = ∂g/∂p.
                # torch.enable_grad() is required because we are inside the
                # @torch.no_grad() decorator, but autograd.grad needs grad to be on.
                with torch.enable_grad():
                    hz = torch.autograd.grad(
                        (g * z).sum(),
                        p,
                        retain_graph=not is_last,  # free graph after the last param
                        create_graph=False,          # we don't need 3rd-order grads
                    )[0]

                # ── Diagonal Hessian estimate: d_t = hz ⊙ z ──────────────────
                # By the Hutchinson identity: E[z ⊙ (H·z)] = diag(H).
                # We take the absolute value because the true diagonal Hessian
                # of a well-scaled loss is non-negative — negatives arise from
                # sampling noise, and |·| makes the preconditioner more stable.
                # .detach() detaches from the autograd graph so we can safely
                # use this inside the @torch.no_grad() region.
                hessian_diags[id(p)] = (hz * z).detach().abs()
        else:
            # ── Fallback: squared gradient ≈ diagonal Hessian ────────────────
            # For squared loss (MSE), the Gauss-Newton approximation gives
            # diag(H) ≈ g² exactly.  For cross-entropy it is an approximation,
            # but it matches Adam's second moment — so AdaHessian degrades
            # gracefully to Adam behaviour when create_graph is unavailable.
            for p, g, _ in params_grads:
                hessian_diags[id(p)] = g.detach().pow(2)

        # ── Main parameter update loop ────────────────────────────────────────
        # Now that we have the Hessian diagonal estimates for all parameters,
        # iterate again to update moments and apply the AdaHessian update.
        # This is a separate loop from the HVP computation so that the
        # computation graph is fully released before we start writing to parameters.
        for p, g, group in params_grads:
            lr    = group["lr"]
            beta1, beta2 = group["betas"]
            eps   = group["eps"]
            wd    = group["weight_decay"]

            # .detach() ensures the gradient tensor is not accidentally included
            # in any future autograd graphs if create_graph were used upstream.
            g_detached = g.detach()
            diag_h = hessian_diags[id(p)]   # shape matches p

            # ── Initialise per-parameter state on the very first step ────────
            state = self.state[p]
            if len(state) == 0:
                state["step"]       = 0
                # m: first moment (mean gradient direction), same as in Adam
                state["exp_avg"]    = torch.zeros_like(p)
                # v: second moment, but tracking the Hessian diagonal instead
                # of the squared gradient (the key AdaHessian modification)
                state["exp_hess"]   = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]
            m = state["exp_avg"]   # alias — in-place ops update the state dict
            v = state["exp_hess"]

            # ── Step 1: First moment update (identical to Adam) ───────────────
            # m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
            m.mul_(beta1).add_(g_detached, alpha=1.0 - beta1)

            # ── Step 2: Second moment update (Hessian EMA) ───────────────────
            # v_t = β₂ · v_{t-1} + (1 − β₂) · |diag(H)|
            # Here diag_h replaces g_t² from Adam.  The EMA smooths
            # the noisy single-sample Hessian estimate over many steps.
            v.mul_(beta2).add_(diag_h, alpha=1.0 - beta2)

            # ── Step 3: Bias correction ───────────────────────────────────────
            # Both m and v are initialised to zero, making early values too
            # small.  Dividing by (1 − β^t) undoes this initialisation bias.
            # The bias correction factor → 1 as t → ∞ (no correction needed
            # once the EMAs have accumulated enough history).
            bc1 = 1.0 - beta1 ** t   # first moment bias correction
            bc2 = 1.0 - beta2 ** t   # Hessian EMA bias correction
            m_hat = m / bc1           # bias-corrected mean gradient
            v_hat = v / bc2           # bias-corrected Hessian diagonal estimate

            # ── Step 4: AdaHessian parameter update ──────────────────────────
            # θ ← θ − lr · m̂ / (√v̂ + ε) − lr · λ · θ
            # Unlike Adam (which uses √v̂ from the squared gradient), here
            # √v̂ is derived from the Hessian diagonal — directions with high
            # curvature get a smaller effective LR, which is more principled
            # than adapting to gradient magnitude alone.
            # ε prevents division by zero when the Hessian estimate is near 0.
            update = m_hat / (v_hat.sqrt().add_(eps))

            if wd != 0.0:
                # L2 regularisation added to the update (coupled form — the
                # decay term is also scaled by the Hessian preconditioner
                # because it is added to m̂ before the division).
                update.add_(p.detach(), alpha=wd)

            # In-place subtraction: θ ← θ − lr · update
            p.add_(update, alpha=-lr)

        return loss
