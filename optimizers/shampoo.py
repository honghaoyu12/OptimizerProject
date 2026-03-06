"""Shampoo optimizer.

From "Shampoo: Preconditioned Stochastic Tensor Optimization"
(Gupta et al., 2018).  https://arxiv.org/abs/1802.09568

Shampoo maintains Kronecker-factored preconditioners for each parameter
tensor and applies a preconditioned gradient update.  For a 2-D weight
matrix W ∈ ℝ^{m×n} the update is:

    L_t = decay · L_{t-1} + (1−decay) · G_t G_t^T   (left EMA factor)
    R_t = decay · R_{t-1} + (1−decay) · G_t^T G_t   (right EMA factor)
    Δ_t = L_t^{-1/4} G_t R_t^{-1/4}                 (preconditioned gradient)

Using an exponential moving average (controlled by `precond_decay`) instead
of a pure running sum prevents the matrices from growing without bound, which
keeps them well-conditioned throughout training.

For efficiency the matrix inverse 4th-roots are recomputed every
`update_freq` steps (default 10) rather than every step.

Parameters with more than `max_precond_size` rows or columns fall back to
a diagonal (Adagrad-style) preconditioner to avoid memory/compute explosion
on large layers (e.g. the FC head of a ResNet).

Higher-dimensional parameters (Conv2d, etc.) are reshaped to 2-D:
    shape (C_out, C_in, kH, kW)  →  (C_out, C_in * kH * kW)
before the Kronecker factors are computed, then reshaped back.
"""

import torch
from .base import BaseOptimizer


class Shampoo(BaseOptimizer):
    """Shampoo: Preconditioned Stochastic Tensor Optimization.

    Parameters
    ----------
    params          : model parameters
    lr              : learning rate                       (default 0.01)
    momentum        : momentum coefficient                (default 0.9)
    weight_decay    : L2 weight decay                     (default 0.0)
    epsilon         : regulariser added to the preconditioner diagonal
                      before computing the inverse root   (default 1e-4)
    precond_decay   : EMA decay for Kronecker factors; controls how quickly
                      old gradients are forgotten (0 = no memory,
                      1 = pure sum).  Default 0.95.
    update_freq     : steps between recomputing the matrix inverse roots
                      (default 10; lower = more accurate but slower)
    max_precond_size: maximum side-length of a Kronecker factor; params
                      with larger dimensions use diagonal preconditioning
                      (default 512)
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        precond_decay: float = 0.95,
        update_freq: int = 10,
        max_precond_size: int = 512,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            epsilon=epsilon, precond_decay=precond_decay,
            update_freq=update_freq, max_precond_size=max_precond_size,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _matrix_inv_pth_root(
        A: torch.Tensor, p: int, epsilon: float, fallback: torch.Tensor
    ) -> torch.Tensor:
        """Compute A^{-1/p} via eigendecomposition.

        Runs on CPU in float32 for device compatibility (torch.linalg.eigh is
        not supported on MPS).  Returns `fallback` on any numerical failure.
        """
        orig_dtype = A.dtype
        orig_device = A.device
        try:
            A_cpu = A.float().cpu()
            # Regularise to ensure positive-definiteness
            A_cpu = A_cpu + epsilon * torch.eye(A_cpu.shape[0])
            eigenvalues, eigenvectors = torch.linalg.eigh(A_cpu)
            eigenvalues = eigenvalues.clamp(min=epsilon)
            inv_root = (
                eigenvectors
                @ torch.diag(eigenvalues.pow(-1.0 / p))
                @ eigenvectors.mT
            )
            return inv_root.to(dtype=orig_dtype, device=orig_device)
        except Exception:
            return fallback

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr            = group["lr"]
            momentum      = group["momentum"]
            wd            = group["weight_decay"]
            epsilon       = group["epsilon"]
            decay         = group["precond_decay"]
            update_freq   = group["update_freq"]
            max_size      = group["max_precond_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.clone()
                if wd != 0.0:
                    g.add_(p, alpha=wd)

                state = self.state[p]

                # ── Determine whether to use Kronecker or diagonal ──────────
                if p.dim() >= 2:
                    rows = p.shape[0]
                    cols = p.numel() // rows
                    use_kronecker = rows <= max_size and cols <= max_size
                else:
                    rows = cols = None
                    use_kronecker = False

                # ── Initialise state on first call ──────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["use_kronecker"] = use_kronecker
                    if use_kronecker:
                        # Initialise as identity — safe starting point
                        state["L"] = torch.eye(rows, device=p.device, dtype=p.dtype)
                        state["R"] = torch.eye(cols, device=p.device, dtype=p.dtype)
                        state["L_inv_root"] = torch.eye(rows, device=p.device, dtype=p.dtype)
                        state["R_inv_root"] = torch.eye(cols, device=p.device, dtype=p.dtype)
                    else:
                        state["sum_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                # ── Compute preconditioned update ───────────────────────────
                if state["use_kronecker"]:
                    g_2d = g.reshape(rows, cols)

                    # EMA update of Kronecker factors (keeps matrices bounded)
                    outer_L = g_2d @ g_2d.mT
                    outer_R = g_2d.mT @ g_2d
                    state["L"].mul_(decay).add_(outer_L, alpha=1.0 - decay)
                    state["R"].mul_(decay).add_(outer_R, alpha=1.0 - decay)

                    # Recompute inverse roots periodically
                    if t == 1 or t % update_freq == 0:
                        eye_L = torch.eye(rows, device=p.device, dtype=p.dtype)
                        eye_R = torch.eye(cols, device=p.device, dtype=p.dtype)
                        state["L_inv_root"] = self._matrix_inv_pth_root(
                            state["L"], 4, epsilon, fallback=eye_L)
                        state["R_inv_root"] = self._matrix_inv_pth_root(
                            state["R"], 4, epsilon, fallback=eye_R)

                    precond = state["L_inv_root"] @ g_2d @ state["R_inv_root"]
                    update = precond.reshape(p.shape)
                else:
                    # Diagonal (Adagrad-style) fallback
                    state["sum_sq"].addcmul_(g, g)
                    update = g / (state["sum_sq"].sqrt() + epsilon)

                # ── Momentum + parameter update ─────────────────────────────
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(update)
                p.add_(buf, alpha=-lr)

        return loss
