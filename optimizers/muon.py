"""Muon optimizer.

From "Muon: An optimizer for hidden layers in neural networks"
(Kostrikov, 2024).  https://kellerjordan.github.io/posts/muon/

Motivation
----------
Adam normalises each gradient by its running RMS, which prevents large updates
but also *destroys gradient direction information* (all coordinates have ≈unit
step size regardless of their relative importance).

Muon instead keeps the gradient direction — using Nesterov momentum for
stability — but orthogonalises the resulting update matrix so that different
neurons in the same layer receive independent, non-redundant updates.

Mathematically, the Newton-Schulz iterations approximate the orthogonal polar
factor of the gradient matrix G:

    G ≈ U Σ V^T  (SVD)
    polar(G) = U V^T   (orthogonal polar factor)

The polar factor has all singular values exactly 1, so every neuron direction
is treated equally — like SGD after normalising each row.  This is closely
related to a steepest-descent step under the *spectral norm* rather than the
ℓ₂ norm, connecting Muon to low-rank second-order methods.

Algorithm (per parameter p with gradient g)
-------------------------------------------
  1. m_t = β · m_{t-1} + g_t                  [classical momentum buffer]
  2. u_t = g_t + β · m_t                       [Nesterov "look-ahead" direction]
  3. For 2-D p: U_t = NewtonSchulz5(u_t)       [orthogonalise rows]
               U_t *= max(1, rows/cols)^0.5    [RMS normalisation: keeps ‖U‖_F]
     For 1-D p: U_t = u_t                      [fallback: ordinary Nesterov SGD]
  4. θ_t = θ_{t-1} · (1 − lr·λ) − lr · U_t   [decoupled weight decay + update]

Newton-Schulz iteration (degree-5 polynomial approximation of the matrix sign)
converges cubically and avoids any SVD computation, making it O(mn) per step.
"""

import torch
from .base import BaseOptimizer


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal polar factor of G ∈ ℝ^{m×n} via Newton-Schulz.

    The Newton-Schulz iteration for the matrix sign function is:
        X_{k+1} = 1.5 · X_k − 0.5 · X_k · X_k^T · X_k

    Applied to G/‖G‖ (normalised), this converges cubically to the orthogonal
    polar factor U (where G = U Σ V^T in the SVD).

    Implementation details:
      - The standard iteration requires rows ≤ cols (works for "short-fat"
        matrices).  We transpose G if rows > cols and transpose back at the end.
      - ‖G‖ normalisation (by Frobenius norm) ensures the initial singular
        values are ≈1, which puts the iteration in the quadratic convergence
        regime immediately.
      - 5 iterations achieves ≈machine-precision on well-conditioned G; 3
        iterations is sufficient for most practical gradient matrices.

    Parameters
    ----------
    G     : (m, n) gradient matrix (not modified in-place; called with .clone())
    steps : number of Newton-Schulz iterations (default 5)
    eps   : small constant to avoid dividing by zero when ‖G‖≈0 (default 1e-7)

    Returns
    -------
    (m, n) tensor with (approximately) orthonormal rows
    """
    assert G.ndim == 2, "Newton-Schulz only applies to 2-D matrices"

    # Transpose so rows ≤ cols (iteration defined for this case)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T                             # now (min, max)

    # Normalise: singular values start near 1 for fast convergence
    G = G / (G.norm() + eps)               # Frobenius norm normalisation

    # Fixed point coefficients for the cubic iteration X ← 1.5X − 0.5 X Xᵀ X
    a, b = 1.5, -0.5
    for _ in range(steps):
        # One Newton-Schulz step: (3I − X^T X) X / 2  ≡  aX + b·X Xᵀ X
        G = a * G + b * (G @ G.T @ G)

    # Undo the transpose so the output shape matches the original G shape
    if transposed:
        G = G.T

    return G


class Muon(BaseOptimizer):
    """Muon: Nesterov Momentum + Newton-Schulz orthogonalisation.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 0.02; typically 5–10× larger than
                   Adam because the orthogonal update is naturally normalised)
    momentum     : Nesterov momentum coefficient β (default 0.95; higher than
                   typical because orthogonalisation already controls step magnitude)
    weight_decay : decoupled L2 regularisation coefficient (default 0.0)
    ns_steps     : Newton-Schulz iteration count (default 5; reduce to 3 for
                   speed with negligible accuracy loss)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            beta     = group["momentum"]    # Nesterov coefficient
            wd       = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad
                state = self.state[p]

                # ── Initialise momentum buffer on first step ───────────────
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]    # m_{t-1}

                # ── Steps 1–2: Nesterov momentum ──────────────────────────
                # Classical:  m_t = β·m_{t-1} + g_t    update = m_t
                # Nesterov:   m_t = β·m_{t-1} + g_t    update = g_t + β·m_t
                # Nesterov's update "looks ahead" by one momentum step, giving
                # faster convergence on smooth loss surfaces (Sutskever et al., 2013).
                buf.mul_(beta).add_(g)            # m_t in-place: β·m + g
                update = g.add(buf, alpha=beta)   # u_t = g_t + β·m_t (Nesterov)

                # ── Step 3: Orthogonalise 2-D update matrices ─────────────
                if update.ndim == 2:
                    # clone() so Newton-Schulz can work in-place without
                    # modifying the gradient tensors that train.py may still read
                    update = _newton_schulz(update.clone(), steps=ns_steps)

                    # RMS normalisation: scale up rectangular matrices so that
                    # ‖U‖_F / sqrt(rows * cols) ≈ 1/sqrt(cols).  Without this,
                    # tall matrices (rows > cols) would have smaller per-row norms.
                    # max(1, rows/cols)^0.5 keeps the total update norm consistent
                    # across different layer shapes.
                    scale  = max(1.0, update.shape[0] / update.shape[1]) ** 0.5
                    update = update.mul_(scale)
                # 1-D parameters (biases, norms): use Nesterov update as-is

                # ── Step 4: Decoupled weight decay + parameter update ──────
                # Apply weight decay as multiplicative shrink (decoupled: does
                # not interact with the orthogonal update direction).
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                # θ ← θ − α · U_t
                p.add_(update, alpha=-lr)

        return loss
