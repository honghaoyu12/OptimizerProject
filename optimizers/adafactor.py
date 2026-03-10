"""AdaFactor optimizer.

From "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
(Shazeer & Stern, ICML 2018).  https://arxiv.org/abs/1805.09843

AdaFactor's central idea: for a weight matrix of shape (m, n), the full Adam
second moment V of size O(mn) is approximated by the outer product of a row
factor r ∈ ℝᵐ and a column factor c ∈ ℝⁿ, reducing storage to O(m + n).

    V̂_{i,j} ≈ (r_i · c_j) / Σ_k r_k

Row and column marginals are updated with a time-varying EMA:
    r_t = ρ_t · r_{t-1} + (1−ρ_t) · (G_t² 1 + ε₁)   [row mean of G²]
    c_t = ρ_t · c_{t-1} + (1−ρ_t) · (1ᵀ G_t² + ε₁)  [col mean of G²]

where ρ_t = min(1 − t^β₂_decay, 0.999) grows toward 1 as training proceeds.

For 1-D parameters (biases) the full second moment is stored instead (like Adam).

This implementation supports a fixed learning rate (default, for easy
benchmarking) or the paper's original relative/scale-free step sizing
(lr=None).  Gradient updates are clipped by their RMS norm before being applied.
"""

import torch

from .base import BaseOptimizer


class AdaFactor(BaseOptimizer):
    """AdaFactor: adaptive learning rates with sublinear memory cost.

    Parameters
    ----------
    params           : model parameters
    lr               : fixed learning rate, or None for relative step sizing
                       (default 1e-3)
    beta2_decay      : exponent for the time-varying ρ_t schedule; negative
                       values make ρ_t grow toward 1 (default -0.8)
    eps1             : floor added to squared gradients for numerical stability
                       (default 1e-30)
    eps2             : floor for relative step sizing and parameter RMS
                       (default 1e-3)
    clip_threshold   : RMS clip threshold applied to the raw update before the
                       learning rate is applied (default 1.0)
    weight_decay     : decoupled weight decay coefficient (default 0.0)
    use_first_moment : store and use a momentum EMA (increases memory by one
                       full copy of the parameters; default False)
    """

    def __init__(
        self,
        params,
        lr: float | None = 1e-3,
        beta2_decay: float = -0.8,
        eps1: float = 1e-30,
        eps2: float = 1e-3,
        clip_threshold: float = 1.0,
        weight_decay: float = 0.0,
        use_first_moment: bool = False,
    ):
        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps1=eps1,
            eps2=eps2,
            clip_threshold=clip_threshold,
            weight_decay=weight_decay,
            use_first_moment=use_first_moment,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _rho(step: int, beta2_decay: float) -> float:
        """Second-moment EMA decay — grows toward 1 as training proceeds."""
        return min(1.0 - step ** beta2_decay, 0.999)

    @staticmethod
    def _rms(t: torch.Tensor) -> float:
        return (t.norm() / (t.numel() ** 0.5)).item()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta2_decay = group["beta2_decay"]
            eps1 = group["eps1"]
            eps2 = group["eps2"]
            clip_threshold = group["clip_threshold"]
            wd = group["weight_decay"]
            use_first_moment = group["use_first_moment"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Work in float32 for numerical stability (important on MPS/fp16)
                g = p.grad.float()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    factored = p.dim() >= 2
                    state["factored"] = factored
                    if factored:
                        # Row factor shape: (..., m); col factor shape: (..., n)
                        state["exp_avg_sq_row"] = torch.zeros(
                            p.shape[:-1], dtype=torch.float32, device=p.device
                        )
                        state["exp_avg_sq_col"] = torch.zeros(
                            p.shape[:-2] + p.shape[-1:], dtype=torch.float32, device=p.device
                        )
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(g)
                    if use_first_moment:
                        state["exp_avg"] = torch.zeros_like(g)

                state["step"] += 1
                t = state["step"]
                rho = self._rho(t, beta2_decay)

                # Decoupled weight decay
                if wd != 0.0:
                    alpha_wd = lr if lr is not None else eps2
                    p.mul_(1.0 - wd * alpha_wd)

                # Squared gradient with stability floor
                g_sq = g.pow(2).add_(eps1)

                # Second moment update and approximate V
                if state["factored"]:
                    row = state["exp_avg_sq_row"]
                    col = state["exp_avg_sq_col"]
                    # Row factor: mean of g² over last axis
                    row.mul_(rho).add_(g_sq.mean(dim=-1), alpha=1.0 - rho)
                    # Col factor: mean of g² over second-to-last axis
                    col.mul_(rho).add_(g_sq.mean(dim=-2), alpha=1.0 - rho)
                    # Reconstruct V̂ from outer product, normalised by row sum
                    row_norm = row.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=eps1)
                    v_approx = row.unsqueeze(-1) * col.unsqueeze(-2) / row_norm
                    update = g / v_approx.sqrt().clamp(min=eps1)
                else:
                    v = state["exp_avg_sq"]
                    v.mul_(rho).add_(g_sq, alpha=1.0 - rho)
                    update = g / v.sqrt().clamp(min=eps1)

                # RMS clipping of the raw update
                rms = self._rms(update)
                if rms > clip_threshold:
                    update.mul_(clip_threshold / rms)

                # Optional first moment (momentum smoothing)
                if use_first_moment:
                    m = state["exp_avg"]
                    m.mul_(0.9).add_(update, alpha=0.1)
                    update = m

                # Step size: fixed lr or relative (scale-free)
                if lr is not None:
                    alpha = lr
                else:
                    param_rms = max(self._rms(p.float()), eps2)
                    alpha = param_rms * min(1.0 / (t ** 0.5), 0.01)

                p.add_(update.to(p.dtype), alpha=-alpha)

        return loss
