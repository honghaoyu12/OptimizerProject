"""Muon optimizer.

From "Muon: An optimizer for hidden layers in neural networks"
(Kostrikov, 2024).  https://kellerjordan.github.io/posts/muon/

Muon applies Nesterov momentum and then orthogonalises the resulting
update matrix (for 2-D weight tensors) via Newton-Schulz iterations.
Orthogonalisation decorrelates the update directions across rows and
columns, effectively acting as an approximate steepest-descent step
under the spectral norm — without ever forming the full Hessian.

For 1-D parameters (biases, LayerNorm scales, etc.) Muon falls back
to ordinary Nesterov SGD.

Algorithm (per parameter p with gradient g)
-------------------------------------------
  1. m_t = β · m_{t-1} + g_t                  [momentum buffer]
  2. u_t = g_t + β · m_t                       [Nesterov look-ahead]
  3. For 2-D p: U_t = NewtonSchulz(u_t)        [orthogonalise]
     Scale: U_t *= max(1, rows/cols)^0.5       [RMS normalisation]
  4. θ_t = θ_{t-1} · (1 − lr · λ) − lr · U_t [decoupled weight decay]
"""

import torch
from .base import BaseOptimizer


def _newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate the orthogonal polar factor of G via Newton-Schulz.

    For G ∈ R^{m×n}:
      - normalise so the largest singular value ≈ 1
      - iterate  X ← 1.5·X − 0.5·X·Xᵀ·X  (cubic convergence to polar factor)
      - handles non-square matrices by transposing so rows ≤ cols

    Returns a tensor of the same shape as G with (approximately) orthonormal
    rows (or columns if G was transposed internally).
    """
    assert G.ndim == 2
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        G = G.T                       # ensure rows ≤ cols
    G = G / (G.norm() + eps)         # normalise
    a, b = 1.5, -0.5
    for _ in range(steps):
        G = a * G + b * (G @ G.T @ G)
    if transposed:
        G = G.T
    return G


class Muon(BaseOptimizer):
    """Muon: Momentum + Newton-Schulz orthogonalisation.

    Parameters
    ----------
    params       : model parameters
    lr           : learning rate (default 0.02)
    momentum     : Nesterov momentum coefficient β (default 0.95)
    weight_decay : decoupled L2 regularisation (default 0.0)
    ns_steps     : Newton-Schulz iteration count (default 5)
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group["lr"]
            beta     = group["momentum"]
            wd       = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # Nesterov update direction
                buf.mul_(beta).add_(g)
                update = g.add(buf, alpha=beta)      # g + β·m

                # Orthogonalise 2-D weight matrices
                if update.ndim == 2:
                    update = _newton_schulz(update.clone(), steps=ns_steps)
                    # RMS normalisation: scale so updates have consistent magnitude
                    scale = max(1.0, update.shape[0] / update.shape[1]) ** 0.5
                    update = update.mul_(scale)

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss
