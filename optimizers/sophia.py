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
            with torch.enable_grad():
                loss = closure()

        # Collect all params that have gradients
        all_pg: list[tuple] = [
            (p, p.grad, group)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not all_pg:
            return loss

        # Initialise state and increment step counters
        for p, g, group in all_pg:
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                # Initialise Hessian to ones so the first clipped update is
                # bounded by ρ even before the first HVP is computed.
                state["hess"] = torch.ones_like(p)
            state["step"] += 1

        # Decide whether to recompute the Hessian this step (shared across all params)
        first_step = self.state[all_pg[0][0]]["step"]
        update_freq = all_pg[0][2]["update_freq"]
        update_hessian = (first_step % update_freq == 1)  # steps 1, 11, 21, ...

        # Compute diagonal Hessian estimates via Hutchinson HVP
        hessian_diags: dict[int, torch.Tensor] = {}
        if update_hessian:
            use_hvp = all_pg[0][1].grad_fn is not None  # True when create_graph was used
            if use_hvp:
                for i, (p, g, _) in enumerate(all_pg):
                    z = torch.randint_like(g, 0, 2).float().mul_(2.0).sub_(1.0)
                    with torch.enable_grad():
                        hz = torch.autograd.grad(
                            (g * z).sum(), p,
                            retain_graph=(i < len(all_pg) - 1),
                            create_graph=False,
                        )[0]
                    hessian_diags[id(p)] = (hz * z).detach().abs()
            else:
                # Fallback: g² is an unbiased estimate of the diagonal Hessian
                # for squared loss (Gauss-Newton approximation).
                for p, g, _ in all_pg:
                    hessian_diags[id(p)] = g.detach().pow(2)

        # Main parameter update
        for p, g, group in all_pg:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            wd = group["weight_decay"]

            g_det = g.detach()
            state = self.state[p]
            m = state["exp_avg"]
            h = state["hess"]

            # Step 1: momentum EMA
            m.mul_(beta1).add_(g_det, alpha=1.0 - beta1)

            # Step 2: Hessian EMA (only on update steps)
            if id(p) in hessian_diags:
                h.mul_(beta2).add_(hessian_diags[id(p)], alpha=1.0 - beta2)

            # Step 3: clipped update
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            update = m / h.clamp(min=1e-8)
            update.clamp_(-rho, rho)
            p.add_(update, alpha=-lr)

        return loss
