"""SOAP optimizer.

From "SOAP: Improving and Stabilizing Shampoo using Adam"
(Vyas et al., 2024).  https://arxiv.org/abs/2409.11321

Motivation
----------
Shampoo maintains Kronecker-factored preconditioners (L, R) and applies the
update  Δ = L^{-1/4} G R^{-1/4}.  The key insight of SOAP is that this
preconditioned gradient lives in the *eigenbasis* of L and R — the same
basis used by Adam if you first project into that subspace, run Adam, then
project back.

SOAP therefore:
  1. Maintains the same Kronecker factors L, R as Shampoo (EMA of outer
     products).
  2. Projects the gradient into the eigenbasis of (L, R) via the cached
     eigenvectors Q_L, Q_R.
  3. Runs *Adam momentum updates in that projected space* (first and second
     moments tracked in the eigenbasis rather than the parameter space).
  4. Projects the Adam update back to the original parameter space and applies
     it.

This gives Shampoo-level curvature adaptation (full Kronecker preconditioning)
with Adam-style adaptive learning rates *per principal direction*, making it
more numerically stable than raw Shampoo and often faster than plain Adam.

Algorithm (2-D weight W ∈ ℝ^{m×n})
--------------------------------------
  L_t  = β₂ · L_{t-1} + (1 − β₂) · G G^T          (left EMA factor, m×m)
  R_t  = β₂ · R_{t-1} + (1 − β₂) · G^T G           (right EMA factor, n×n)

  Every `update_freq` steps: recompute eigenvectors
    Q_L, Λ_L = eigh(L_t + ε I)
    Q_R, Λ_R = eigh(R_t + ε I)

  Project gradient into eigenbasis:
    G̃ = Q_L^T · G · Q_R                              (m×n, projected)

  Adam update in projected space (standard Adam with bias correction):
    m̃_t = β₁ · m̃_{t-1} + (1 − β₁) · G̃              (first moment in eigenbasis)
    ṽ_t = β₂ · ṽ_{t-1} + (1 − β₂) · G̃²             (second moment in eigenbasis)
    m̂   = m̃_t / (1 − β₁^t)
    v̂   = ṽ_t / (1 − β₂^t)
    Δ̃   = m̂ / (√v̂ + ε)                              (Adam step in eigenbasis)

  Project update back and apply:
    Δ   = Q_L · Δ̃ · Q_R^T
    W_t = W_{t-1} − α · Δ

For 1-D parameters (biases, LayerNorm) there is no meaningful Kronecker
structure, so SOAP falls back to plain Adam.

Practical tips
--------------
  • lr=1e-3, betas=(0.95, 0.95), eps=1e-8 are good starting points.
  • β₂ for the Kronecker factors is the same as for Adam's second moment;
    0.95 is recommended in the paper (slightly higher than Adam's 0.999
    to make the preconditioner adapt faster).
  • update_freq=10 balances the O(n³) eigh cost with preconditioning quality.
  • max_precond_size guards against huge embedding/output layers (same role
    as in Shampoo).
"""

import math

import torch
from .base import BaseOptimizer


class SOAP(BaseOptimizer):
    """SOAP: Shampoo-as-Adam Preconditioner.

    Parameters
    ----------
    params           : model parameters
    lr               : learning rate α (default 3e-3)
    betas            : (β₁, β₂) — momentum and second-moment EMA rates.
                       Also used as the EMA rate for Kronecker factors.
                       (default (0.95, 0.95))
    eps              : numerical stability constant (default 1e-8)
    weight_decay     : decoupled L2 regularisation (default 0.0)
    update_freq      : steps between eigenvector recomputations (default 10)
    max_precond_size : parameters whose row or column count exceeds this fall
                       back to plain Adam (default 512)
    """

    def __init__(
        self,
        params,
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_freq: int = 10,
        max_precond_size: int = 512,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            update_freq=update_freq, max_precond_size=max_precond_size,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _eigh_cpu(A: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric eigendecomposition of A on CPU float32.

        Returns (eigenvalues, eigenvectors) after regularising A with ε·I.
        Runs on CPU because torch.linalg.eigh is unreliable on MPS.

        Returns the identity eigenvectors and unit eigenvalues on failure so
        that the caller can proceed with an unscaled update rather than crash.
        """
        orig_device = A.device
        orig_dtype  = A.dtype
        try:
            # Regularise to ensure positive definiteness before decomposition.
            # Without this, near-zero eigenvalues would produce NaN in Adam's √v.
            A_cpu = A.float().cpu() + epsilon * torch.eye(A.shape[0])
            eigenvalues, eigenvectors = torch.linalg.eigh(A_cpu)
            # Clamp eigenvalues to ε to prevent negative values from floating-point noise.
            eigenvalues = eigenvalues.clamp(min=epsilon)
            return (
                eigenvalues.to(dtype=orig_dtype, device=orig_device),
                eigenvectors.to(dtype=orig_dtype, device=orig_device),
            )
        except Exception:
            # Safe fallback: identity eigenvectors (no rotation) + unit eigenvalues.
            n = A.shape[0]
            return (
                torch.ones(n, dtype=orig_dtype, device=orig_device),
                torch.eye(n,  dtype=orig_dtype, device=orig_device),
            )

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one SOAP step across all parameter groups.

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
            update_freq  = group["update_freq"]
            max_size     = group["max_precond_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Decide preconditioner type ─────────────────────────────
                # Kronecker preconditioning requires at least 2 dimensions and
                # both dimensions within the max_precond_size limit.
                if p.dim() >= 2:
                    rows = p.shape[0]
                    cols = p.numel() // rows   # flatten higher dims into cols
                    use_kronecker = rows <= max_size and cols <= max_size
                else:
                    rows = cols = None
                    use_kronecker = False

                # ── Initialise state on first step ─────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    state["use_kronecker"] = use_kronecker

                    if use_kronecker:
                        # Kronecker EMA factors (same role as Shampoo's L/R).
                        state["L"] = torch.eye(rows, device=p.device, dtype=p.dtype)
                        state["R"] = torch.eye(cols, device=p.device, dtype=p.dtype)
                        # Cached eigenvectors — recomputed every update_freq steps.
                        # Initialised to identity so the first projected step is
                        # equivalent to plain Adam (no rotation yet).
                        state["Q_L"] = torch.eye(rows, device=p.device, dtype=p.dtype)
                        state["Q_R"] = torch.eye(cols, device=p.device, dtype=p.dtype)
                        # Adam buffers in the *projected eigenbasis* (m×n, same shape as G).
                        state["exp_avg"]    = torch.zeros(rows, cols, device=p.device, dtype=p.dtype)
                        state["exp_avg_sq"] = torch.zeros(rows, cols, device=p.device, dtype=p.dtype)
                    else:
                        # Fallback: standard Adam buffers in parameter space.
                        state["exp_avg"]    = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]

                # ── Decoupled weight decay ─────────────────────────────────
                # Applied directly to parameters before the gradient update.
                # Decoupled means decay does NOT flow through the preconditioner,
                # matching AdamW's design rather than coupled L2 regularisation.
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                if state["use_kronecker"]:
                    # ── Reshape gradient to 2-D ────────────────────────────
                    # Conv2d: (C_out, C_in, kH, kW) → (C_out, C_in*kH*kW)
                    g_2d = g.reshape(rows, cols)

                    # ── Update Kronecker EMA factors ───────────────────────
                    # L accumulates the left Gram matrix G G^T;
                    # R accumulates the right Gram matrix G^T G.
                    # EMA decay β₂ forgets old curvature at rate (1-β₂) per step,
                    # keeping the preconditioner adapted to recent gradient structure.
                    state["L"].mul_(beta2).add_(g_2d @ g_2d.mT, alpha=1.0 - beta2)
                    state["R"].mul_(beta2).add_(g_2d.mT @ g_2d, alpha=1.0 - beta2)

                    # ── Periodically recompute eigenvectors ────────────────
                    # eigh is O(n³) — expensive, so we cache and refresh every
                    # update_freq steps rather than every step.
                    # Step 1 always recomputes (first useful eigenbasis).
                    if t == 1 or t % update_freq == 0:
                        _, state["Q_L"] = self._eigh_cpu(state["L"], eps)
                        _, state["Q_R"] = self._eigh_cpu(state["R"], eps)

                    Q_L = state["Q_L"]   # (rows, rows) — eigenvectors of L
                    Q_R = state["Q_R"]   # (cols, cols) — eigenvectors of R

                    # ── Project gradient into eigenbasis ───────────────────
                    # G̃ = Q_L^T · G · Q_R
                    # In this basis, the Kronecker preconditioner becomes diagonal
                    # (Λ_L ⊗ Λ_R), which is exactly what Adam's v_t approximates
                    # — one adaptive scale per principal direction.
                    g_proj = Q_L.mT @ g_2d @ Q_R   # (rows, cols), projected

                    # ── Adam update in projected space ─────────────────────
                    m = state["exp_avg"]     # first moment in eigenbasis
                    v = state["exp_avg_sq"]  # second moment in eigenbasis

                    # Standard EMA updates for both moments.
                    m.mul_(beta1).add_(g_proj, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g_proj, g_proj, value=1.0 - beta2)

                    # Bias correction: compensates for zero initialisation of m, v.
                    bias_corr1 = 1.0 - beta1 ** t
                    bias_corr2 = 1.0 - beta2 ** t
                    m_hat = m / bias_corr1    # unbiased first moment
                    v_hat = v / bias_corr2    # unbiased second moment

                    # Adam step in the eigenbasis:
                    #   Δ̃ = m̂ / (√v̂ + ε)
                    step_proj = m_hat / (v_hat.sqrt().add_(eps))   # (rows, cols)

                    # ── Project update back to parameter space ─────────────
                    # Δ = Q_L · Δ̃ · Q_R^T
                    # This reverses the rotation, giving the update in the
                    # original (weight-matrix) coordinate system.
                    update = Q_L @ step_proj @ Q_R.mT   # (rows, cols)

                    # Reshape back to original parameter shape and apply.
                    p.add_(update.reshape(p.shape), alpha=-lr)

                else:
                    # ── Fallback: plain Adam in parameter space ────────────
                    m = state["exp_avg"]
                    v = state["exp_avg_sq"]

                    m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bias_corr1 = 1.0 - beta1 ** t
                    bias_corr2 = 1.0 - beta2 ** t
                    step = (m / bias_corr1) / ((v / bias_corr2).sqrt().add_(eps))
                    p.add_(step, alpha=-lr)

        return loss
