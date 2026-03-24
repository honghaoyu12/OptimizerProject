"""Sophia optimizer.

From "Sophia: A Scalable Stochastic Second-order Optimizer for Language Model
Pre-training" (Liu et al., 2023).  https://arxiv.org/abs/2305.14342

Sophia maintains a momentum term (like Adam's first moment) and a *clipped*
diagonal Hessian estimate, updating the Hessian only every `update_freq` steps
to amortise the cost of the Hessian-vector product (HVP).

The critical difference from AdaHessian (also in this repo):
  • AdaHessian divides by √ĥ and uses bias-corrected second moment — Adam-like.
  • Sophia divides by ĥ directly (no sqrt) and *clips* the result to [−ρ, ρ],
    preventing catastrophically large steps when the Hessian is near zero.
    This clipping acts as an adaptive trust-region.

Sophia-H variant (implemented here): Hutchinson randomised Hessian estimator.
    z  ~ Rademacher({−1,+1})
    ĥ  = |z ⊙ (H·z)|   (element-wise; absolute value for stability)

When `create_graph=True` is not used (the gradient graph is unavailable),
the optimizer falls back to using g² as the Hessian estimate — identical to
a clipped version of Adam.  Set `requires_create_graph = True` on the class
so that `train_one_epoch()` automatically enables the full Sophia-H path.

Algorithm
---------
Every step t:
  m_t = β₁·m_{t-1} + (1−β₁)·g_t                       [momentum EMA]

Every `update_freq` steps (starting at t=1):
  z  ~ Rademacher; ĥ = |z ⊙ H·z| via HVP
  h_t = β₂·h_{t-1} + (1−β₂)·ĥ                          [Hessian EMA]

Every step (after the above):
  θ_t = θ_{t-1} − lr·clip(m_t / max(h_t, ε), −ρ, ρ)   [clipped update]
        − lr·λ·θ_{t-1}                                   [weight decay]
"""

import torch
from .base import BaseOptimizer


class Sophia(BaseOptimizer):
    """Sophia: second-order optimizer with clipped Hessian preconditioning.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 1e-4; Sophia needs a smaller lr than Adam
                   because it divides by the raw Hessian rather than its sqrt)
    betas        : (β₁, β₂) — momentum and Hessian EMA rates
                   (default (0.965, 0.99) from the paper)
    rho          : clip threshold for the normalised update (default 0.04)
    weight_decay : decoupled weight decay coefficient (default 0.0)
    update_freq  : how often (in steps) to recompute the Hessian estimate
                   (default 10; paper uses 10 for GPT-2 scale)
    """

    # Signal to train_one_epoch() to call loss.backward(create_graph=True)
    # so that HVPs can be computed.  Falls back to g² if graph is unavailable.
    requires_create_graph: bool = True

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
        update_freq: int = 10,
    ):
        defaults = dict(lr=lr, betas=betas, rho=rho,
                        weight_decay=weight_decay, update_freq=update_freq)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Re-enable autograd inside the closure so backward() can run.
            # We are under @torch.no_grad(), so this context manager is required.
            with torch.enable_grad():
                loss = closure()

        # ── Collect all parameters that have gradients this step ─────────────
        # List comprehension form for conciseness; semantically identical to
        # the loop form in AdaHessian.  Only parameters with non-None gradients
        # participate in the update — frozen/unused parameters are skipped.
        all_pg: list[tuple] = [
            (p, p.grad, group)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not all_pg:
            # Nothing to update — return early (avoids indexing into empty list below)
            return loss

        # ── Initialise per-parameter state and increment step counters ────────
        # We do the step increment here (before deciding whether to update the
        # Hessian) so that `first_step` below reflects the current step number
        # and can be used to determine the update schedule correctly.
        for p, g, group in all_pg:
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                # m: standard Adam-like first moment (EMA of the gradient)
                state["exp_avg"] = torch.zeros_like(p)
                # h: Hessian diagonal EMA — initialised to ones so the very
                # first clipped update is bounded by ρ even before any HVP
                # is computed.  Using ones instead of zeros avoids a divide-by-zero
                # on the first call when h is all zeros.
                state["hess"] = torch.ones_like(p)
            state["step"] += 1

        # ── Decide whether to recompute the Hessian on this step ─────────────
        # The Hessian update is amortised: we only recompute it every
        # `update_freq` steps.  The condition `step % update_freq == 1` fires
        # on steps 1, 11, 21, … (with the default update_freq=10).
        # Starting at step 1 (not 0) ensures we always get a Hessian estimate
        # before the first momentum-only update.
        # All parameters share the same step counter (first param is the proxy).
        first_step = self.state[all_pg[0][0]]["step"]
        update_freq = all_pg[0][2]["update_freq"]
        update_hessian = (first_step % update_freq == 1)  # steps 1, 11, 21, ...

        # ── Hutchinson Hessian diagonal estimates (computed only when scheduled) ─
        # hessian_diags maps id(p) → estimated |diag(H)| tensor, same shape as p.
        # On non-update steps this dict stays empty; the existing h values in
        # state["hess"] continue to be used without modification.
        hessian_diags: dict[int, torch.Tensor] = {}
        if update_hessian:
            # Check whether the computation graph is available (i.e., was
            # loss.backward(create_graph=True) called?).  If grad_fn is not None,
            # the gradient tensor is a non-leaf node in the autograd graph and
            # we can differentiate through it to get H·z.
            use_hvp = all_pg[0][1].grad_fn is not None

            if use_hvp:
                # ── True Hessian-vector products via Hutchinson ───────────────
                # For each parameter p:
                #   1. Draw z ~ Rademacher({-1, +1})
                #   2. Compute H·z = ∂(g^T z)/∂p via autograd
                #   3. Estimate diag(H) ≈ |z ⊙ H·z|  (absolute value for stability)
                # retain_graph=True for all but the last parameter so the graph
                # stays alive for subsequent iterations; the last frees it.
                for i, (p, g, _) in enumerate(all_pg):
                    # Rademacher vector: ±1 with equal probability
                    # randint(0,2) gives {0,1}; *2-1 maps to {-1,+1}
                    z = torch.randint_like(g, 0, 2).float().mul_(2.0).sub_(1.0)
                    with torch.enable_grad():
                        hz = torch.autograd.grad(
                            (g * z).sum(), p,
                            retain_graph=(i < len(all_pg) - 1),  # keep graph for next param
                            create_graph=False,                    # no 3rd-order grads needed
                        )[0]
                    # |z ⊙ H·z| — absolute value because the true diagonal of
                    # a positive semi-definite Hessian should be non-negative;
                    # absolute value removes sampling noise sign flips.
                    hessian_diags[id(p)] = (hz * z).detach().abs()
            else:
                # ── Fallback: g² ≈ diag(H) (Gauss-Newton approximation) ──────
                # For squared loss this is exact; for cross-entropy it is
                # approximate but keeps Sophia behaving like a clipped version
                # of Adam when the full second-order path is unavailable.
                for p, g, _ in all_pg:
                    hessian_diags[id(p)] = g.detach().pow(2)

        # ── Main parameter update ─────────────────────────────────────────────
        # Applied every step (momentum update always; Hessian EMA only on
        # scheduled steps).  The clipped update is Sophia's defining feature.
        for p, g, group in all_pg:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]      # clip threshold for the normalised update
            wd  = group["weight_decay"]

            # .detach() prevents the gradient from being tracked in any future
            # autograd graph.  Important because g may still have a grad_fn
            # if create_graph=True was used.
            g_det = g.detach()
            state = self.state[p]
            m = state["exp_avg"]   # first moment buffer (alias, modified in-place)
            h = state["hess"]      # Hessian diagonal EMA buffer (alias)

            # ── Step 1: Momentum EMA (every step) ────────────────────────────
            # m_t = β₁ · m_{t-1} + (1 − β₁) · g_t
            # Note: Sophia does NOT bias-correct m (unlike Adam) — the clip
            # bound ρ provides an implicit normalisation that makes the bias
            # correction less critical for stability.
            m.mul_(beta1).add_(g_det, alpha=1.0 - beta1)

            # ── Step 2: Hessian EMA (only on scheduled steps) ────────────────
            # h_t = β₂ · h_{t-1} + (1 − β₂) · |diag(H)|
            # Only executed when id(p) is in hessian_diags, i.e., on Hessian
            # update steps.  Between updates, h retains its previous value —
            # the EMA already smooths out the stale estimate gracefully.
            if id(p) in hessian_diags:
                h.mul_(beta2).add_(hessian_diags[id(p)], alpha=1.0 - beta2)

            # ── Step 3: Decoupled weight decay (applied before the gradient update) ─
            # p ← p · (1 − lr · λ)
            # Sophia uses decoupled decay (like AdamW) so the regularisation
            # strength is not inadvertently scaled by the Hessian preconditioning.
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)

            # ── Step 4: Clipped Hessian-normalised update ────────────────────
            # The raw update direction is m_t / max(h_t, ε).
            # Dividing by h_t (not √h_t like Adam) uses the *full* curvature
            # information, so directions with large curvature get proportionally
            # smaller steps.  The clip to [-ρ, ρ] serves as the trust region:
            # it prevents catastrophically large steps when h_t is near zero
            # (e.g. very flat directions or early in training before the Hessian
            # EMA has accumulated).
            # h.clamp(min=1e-8): numerical guard in case h underflows to 0.0.
            update = m / h.clamp(min=1e-8)
            update.clamp_(-rho, rho)   # in-place clip to [-ρ, ρ]

            # θ ← θ − lr · clip(m / max(h, ε), -ρ, ρ)
            p.add_(update, alpha=-lr)

        return loss
