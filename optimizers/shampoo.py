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
            # Move to CPU float32: torch.linalg.eigh is only reliable on CPU,
            # and float32 is sufficient for the 4th-root preconditioning accuracy.
            A_cpu = A.float().cpu()

            # Regularise the matrix to ensure strict positive-definiteness.
            # Without this, near-zero eigenvalues would produce infinite inverse roots.
            # ε·I shifts all eigenvalues up by ε, bounding the inverse root from above.
            A_cpu = A_cpu + epsilon * torch.eye(A_cpu.shape[0])

            # Symmetric eigendecomposition: A = Q Λ Q^T, where Λ = diag(eigenvalues)
            # torch.linalg.eigh exploits symmetry — more numerically stable and
            # faster than torch.linalg.eig for symmetric matrices.
            eigenvalues, eigenvectors = torch.linalg.eigh(A_cpu)

            # Clamp eigenvalues to [ε, ∞) to prevent negative eigenvalues (which
            # can arise from numerical noise in nearly-singular matrices) from
            # producing complex roots.
            eigenvalues = eigenvalues.clamp(min=epsilon)

            # A^{-1/p} = Q · diag(λ_i^{-1/p}) · Q^T
            # The formula uses the eigendecomposition A = Q Λ Q^T, so
            # A^{-1/p} = Q Λ^{-1/p} Q^T = Q · diag(λ_i^{-1/p}) · Q^T.
            # p=4 gives the fourth root, which is the Shampoo preconditioner.
            inv_root = (
                eigenvectors
                @ torch.diag(eigenvalues.pow(-1.0 / p))
                @ eigenvectors.mT   # .mT is the conjugate transpose (= .T for real matrices)
            )
            # Move result back to the original device and dtype for use in the update
            return inv_root.to(dtype=orig_dtype, device=orig_device)
        except Exception:
            # Any numerical failure (non-convergence, NaN, etc.) falls back to
            # the identity matrix — this gives an unscaled gradient step for
            # this parameter, which is safe even if suboptimal.
            return fallback

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            # Re-enable autograd so the closure can call backward().
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr          = group["lr"]
            momentum    = group["momentum"]
            wd          = group["weight_decay"]
            epsilon     = group["epsilon"]       # regulariser for preconditioner diagonal
            decay       = group["precond_decay"] # EMA decay for Kronecker factors
            update_freq = group["update_freq"]   # how often to recompute inverse roots
            max_size    = group["max_precond_size"]  # size threshold for Kronecker vs diagonal

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Clone the gradient so we can add weight decay without
                # modifying p.grad in-place (which would affect other users
                # of the gradient, e.g. gradient accumulation scenarios).
                g = p.grad.clone()

                # ── Coupled L2 weight decay ───────────────────────────────────
                # Add λ·θ to the gradient so the regularisation signal also
                # flows through the Kronecker preconditioner (unlike decoupled
                # forms where decay bypasses the preconditioner entirely).
                if wd != 0.0:
                    g.add_(p, alpha=wd)

                state = self.state[p]

                # ── Determine Kronecker vs. diagonal preconditioner ───────────
                # Kronecker factoring requires the parameter to be at least 2-D
                # (has both row and column dimensions).  For very large layers
                # (e.g. the 512×10 final FC head of a ResNet), the Kronecker
                # factors would be 512×512 and 10×10 — manageable.  For
                # embedding layers that might be 50000×512, the left factor
                # would be 50000×50000 — prohibitive.  max_precond_size guards
                # against this memory/compute explosion.
                if p.dim() >= 2:
                    rows = p.shape[0]
                    cols = p.numel() // rows   # flatten all dims except the first
                    use_kronecker = rows <= max_size and cols <= max_size
                else:
                    # 1-D parameters (biases, LayerNorm scales) get diagonal treatment:
                    # no meaningful Kronecker structure to exploit.
                    rows = cols = None
                    use_kronecker = False

                # ── Initialise state on the very first call ───────────────────
                if len(state) == 0:
                    state["step"] = 0
                    # momentum_buffer: velocity buffer for the SGD-like outer update.
                    # Initialised to zeros — first step applies pure preconditioned gradient.
                    state["momentum_buffer"] = torch.zeros_like(p)
                    # Record the preconditioner type so we can assert consistency later.
                    state["use_kronecker"] = use_kronecker

                    if use_kronecker:
                        # L: left Kronecker factor, shape (rows, rows)
                        # R: right Kronecker factor, shape (cols, cols)
                        # Both initialised to identity — safe starting point that
                        # gives an unscaled gradient on the first step.
                        state["L"] = torch.eye(rows, device=p.device, dtype=p.dtype)
                        state["R"] = torch.eye(cols, device=p.device, dtype=p.dtype)
                        # L_inv_root, R_inv_root: cached inverse 4th-roots of L and R.
                        # Recomputed every `update_freq` steps via eigendecomposition.
                        # Initialised to identity so the first step is unscaled.
                        state["L_inv_root"] = torch.eye(rows, device=p.device, dtype=p.dtype)
                        state["R_inv_root"] = torch.eye(cols, device=p.device, dtype=p.dtype)
                    else:
                        # sum_sq: diagonal second-moment accumulator (Adagrad-style).
                        # shape matches p — each element tracks Σ g_i² over all steps.
                        state["sum_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                # ── Compute preconditioned gradient ───────────────────────────
                if state["use_kronecker"]:
                    # Reshape p to 2-D for Kronecker arithmetic.
                    # Higher-dim tensors (Conv2d weight: C_out × C_in × kH × kW)
                    # are flattened to (C_out, C_in * kH * kW) — the standard
                    # Shampoo convention for convolutional layers.
                    g_2d = g.reshape(rows, cols)

                    # ── EMA update of Kronecker factors ───────────────────────
                    # L_t = decay · L_{t-1} + (1-decay) · G G^T    (left factor)
                    # R_t = decay · R_{t-1} + (1-decay) · G^T G    (right factor)
                    # Using EMA (rather than a pure running sum) prevents L and R
                    # from growing without bound.  The EMA effectively forgets
                    # old gradient statistics at rate (1-decay) per step,
                    # keeping the preconditioner adapted to recent curvature.
                    outer_L = g_2d @ g_2d.mT    # (rows, rows): left Gram matrix
                    outer_R = g_2d.mT @ g_2d    # (cols, cols): right Gram matrix
                    state["L"].mul_(decay).add_(outer_L, alpha=1.0 - decay)
                    state["R"].mul_(decay).add_(outer_R, alpha=1.0 - decay)

                    # ── Recompute inverse roots periodically ──────────────────
                    # Eigendecomposition is expensive (O(n³)), so we only
                    # recompute every `update_freq` steps.  The cached inverse
                    # root from the previous update step is used in between.
                    # t == 1: always recompute on the first step (from identity).
                    # t % update_freq == 0: recompute every `update_freq` steps.
                    if t == 1 or t % update_freq == 0:
                        # Identity fallback: used if eigendecomposition fails
                        eye_L = torch.eye(rows, device=p.device, dtype=p.dtype)
                        eye_R = torch.eye(cols, device=p.device, dtype=p.dtype)
                        # L^{-1/4}: inverse fourth root of the left factor
                        state["L_inv_root"] = self._matrix_inv_pth_root(
                            state["L"], 4, epsilon, fallback=eye_L)
                        # R^{-1/4}: inverse fourth root of the right factor
                        state["R_inv_root"] = self._matrix_inv_pth_root(
                            state["R"], 4, epsilon, fallback=eye_R)

                    # ── Apply Kronecker preconditioner ────────────────────────
                    # precond = L^{-1/4} · G · R^{-1/4}
                    # This is the matrix equivalent of dividing each gradient
                    # element by the square root of its curvature.  The 4th-root
                    # is used (rather than the square root) because we have TWO
                    # factors (left and right), each contributing a square root,
                    # and their product gives the full square root of the Kronecker
                    # approximation to H: (L⊗R)^{-1/2} ≈ L^{-1/4} ⊗ R^{-1/4}.
                    precond = state["L_inv_root"] @ g_2d @ state["R_inv_root"]
                    # Reshape back to the original parameter shape before the
                    # momentum + parameter update below.
                    update = precond.reshape(p.shape)

                else:
                    # ── Diagonal (Adagrad-style) fallback ────────────────────
                    # G_t = G_{t-1} + g²   (running sum of squared gradients)
                    # This is the standard Adagrad accumulator — no EMA, just
                    # a monotonically growing sum.  The preconditioned gradient
                    # is g / (√G_t + ε), which shrinks over time.
                    state["sum_sq"].addcmul_(g, g)   # addcmul_ does: sum_sq += g * g
                    update = g / (state["sum_sq"].sqrt() + epsilon)

                # ── Momentum + parameter update ───────────────────────────────
                # Apply standard SGD momentum on top of the preconditioned gradient.
                # buf_t = μ · buf_{t-1} + update_t
                # θ_t   = θ_{t-1} − lr · buf_t
                #
                # Note: this is the "heavy-ball" form of momentum (not Nesterov).
                # The preconditioned update enters the velocity buffer directly,
                # which means the momentum accumulates pre-conditioned directions.
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(update)   # velocity: exponentially decay + add new
                p.add_(buf, alpha=-lr)            # parameter: step along accumulated velocity

        return loss
