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
            with torch.enable_grad():
                loss = closure()

        # Collect parameters and gradients
        params_grads: list[tuple] = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    params_grads.append((p, p.grad, group))

        # Compute Hessian diagonal estimates (one HVP per parameter).
        # Differentiable gradients have a grad_fn; plain backward() produces
        # leaf tensors (grad_fn is None) and we fall back to squared gradient.
        hessian_diags: dict[int, torch.Tensor] = {}
        use_hvp = params_grads and params_grads[0][1].grad_fn is not None

        if use_hvp:
            for i, (p, g, _) in enumerate(params_grads):
                z = torch.randint_like(g, 0, 2).float().mul_(2.0).sub_(1.0)
                is_last = (i == len(params_grads) - 1)
                # Differentiate g·z w.r.t. p to get H·z
                with torch.enable_grad():
                    hz = torch.autograd.grad(
                        (g * z).sum(),
                        p,
                        retain_graph=not is_last,
                        create_graph=False,
                    )[0]
                hessian_diags[id(p)] = (hz * z).detach().abs()
        else:
            # Fallback: use squared gradient (Adam behaviour)
            for p, g, _ in params_grads:
                hessian_diags[id(p)] = g.detach().pow(2)

        # Optimizer update
        for p, g, group in params_grads:
            lr    = group["lr"]
            beta1, beta2 = group["betas"]
            eps   = group["eps"]
            wd    = group["weight_decay"]

            g_detached = g.detach()
            diag_h = hessian_diags[id(p)]

            state = self.state[p]
            if len(state) == 0:
                state["step"]       = 0
                state["exp_avg"]    = torch.zeros_like(p)
                state["exp_hess"]   = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]
            m = state["exp_avg"]
            v = state["exp_hess"]

            # Update moments
            m.mul_(beta1).add_(g_detached, alpha=1.0 - beta1)
            v.mul_(beta2).add_(diag_h, alpha=1.0 - beta2)

            # Bias correction
            bc1 = 1.0 - beta1 ** t
            bc2 = 1.0 - beta2 ** t
            m_hat = m / bc1
            v_hat = v / bc2

            # Update
            update = m_hat / (v_hat.sqrt().add_(eps))
            if wd != 0.0:
                update.add_(p.detach(), alpha=wd)
            p.add_(update, alpha=-lr)

        return loss
