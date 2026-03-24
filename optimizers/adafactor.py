"""AdaFactor optimizer.

From "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
(Shazeer & Stern, ICML 2018).  https://arxiv.org/abs/1805.09843

Memory Problem with Adam
------------------------
Adam maintains two state tensors per parameter (m and v), each the same size
as the parameter.  For a model with P parameters, Adam needs 3P floats of
memory (model + m + v).  For a T5-11B model (11 billion parameters), that's
~130 GB just for the optimizer states.

AdaFactor's Solution: Factored Second Moment
--------------------------------------------
For a 2-D weight matrix W ∈ ℝ^{m×n}, instead of storing the full
(m×n) second moment V, AdaFactor stores only:
  r ∈ ℝᵐ  (row factor)
  c ∈ ℝⁿ  (column factor)

And reconstructs the full V on the fly:
    V̂_{i,j} ≈ (r_i · c_j) / (Σ_k r_k)

This is a rank-1 approximation to the true V.  The denominator normalises
the outer product so V̂ has the same total magnitude as V.  Memory drops from
O(mn) to O(m + n) — for large matrices this is a massive saving.

Row and column factors are updated with a time-varying decay ρ_t:
    r_t = ρ_t · r_{t-1} + (1−ρ_t) · (G_t² @ 1 + ε₁)   [row mean of G²]
    c_t = ρ_t · c_{t-1} + (1−ρ_t) · (1^T @ G_t² + ε₁)  [col mean of G²]

where ρ_t = min(1 − t^{β₂_decay}, 0.999) grows toward 1 as training progresses
(more memory of the past = more stable estimates).

For 1-D parameters (biases, norms) the full second moment is stored — no
factoring is needed or useful.

This implementation supports:
  - Fixed LR (lr=<value>): for easy benchmarking against other optimizers.
  - Relative/scale-free LR (lr=None): paper's original mode where the step
    size is proportional to the parameter RMS, giving a true "no LR tuning"
    experience.
"""

import torch

from .base import BaseOptimizer


class AdaFactor(BaseOptimizer):
    """AdaFactor: adaptive learning rates with sublinear memory cost.

    Parameters
    ----------
    params           : model parameters
    lr               : fixed learning rate, or None for relative step sizing.
                       Default 1e-3 (fixed) for easy benchmark comparison.
    beta2_decay      : exponent for the time-varying ρ_t schedule.
                       Negative values make ρ_t grow toward 1 over training.
                       Default -0.8 from the paper.
    eps1             : floor added to squared gradients before row/col EMA.
                       Prevents zero-initialised factors from staying at zero.
                       Default 1e-30 (very small; keeps factors meaningful).
    eps2             : minimum threshold for the relative LR mode and for the
                       parameter RMS floor.  Default 1e-3.
    clip_threshold   : RMS clip threshold applied to the raw update vector
                       before scaling by the learning rate.  Prevents large
                       single-step weight changes.  Default 1.0.
    weight_decay     : decoupled weight decay coefficient (default 0.0).
    use_first_moment : if True, also maintain a momentum EMA (like Adam's m).
                       Increases memory by P floats but may improve convergence
                       on noisy gradients.  Default False.
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
        """Compute the time-varying second-moment decay ρ_t.

        ρ_t = min(1 − step^{beta2_decay}, 0.999)

        With beta2_decay=-0.8 and step=1000: 1 − 1000^{-0.8} ≈ 0.994.
        This grows toward 1 monotonically, giving the second moment more
        "memory" as training progresses — crucial for stability late in training.
        The cap at 0.999 prevents the EMA from freezing (ρ→1 would mean no
        new information is ever incorporated).
        """
        return min(1.0 - step ** beta2_decay, 0.999)

    @staticmethod
    def _rms(t: torch.Tensor) -> float:
        """Root mean square of tensor elements: ‖t‖_F / √(numel(t))."""
        return (t.norm() / (t.numel() ** 0.5)).item()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr               = group["lr"]
            beta2_decay      = group["beta2_decay"]
            eps1             = group["eps1"]
            eps2             = group["eps2"]
            clip_threshold   = group["clip_threshold"]
            wd               = group["weight_decay"]
            use_first_moment = group["use_first_moment"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Work in float32 throughout: AdaFactor uses very small eps1
                # values that would underflow in float16.  The result is cast
                # back to p.dtype before the final parameter update.
                g     = p.grad.float()
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    state["step"]    = 0
                    # 2-D parameters: use factored second moment
                    # 1-D parameters: use full second moment (like Adam)
                    factored         = p.dim() >= 2
                    state["factored"] = factored

                    if factored:
                        # Row factor r: shape is all dims except the last
                        # e.g. for (out, in) → shape (out,)
                        state["exp_avg_sq_row"] = torch.zeros(
                            p.shape[:-1], dtype=torch.float32, device=p.device
                        )
                        # Col factor c: shape is all dims except second-to-last
                        # e.g. for (out, in) → shape (in,)
                        state["exp_avg_sq_col"] = torch.zeros(
                            p.shape[:-2] + p.shape[-1:], dtype=torch.float32, device=p.device
                        )
                    else:
                        # Full second moment for 1-D (biases etc.)
                        state["exp_avg_sq"] = torch.zeros_like(g)

                    if use_first_moment:
                        # Optional: adds momentum like Adam's first moment
                        state["exp_avg"] = torch.zeros_like(g)

                state["step"] += 1
                t   = state["step"]
                rho = self._rho(t, beta2_decay)   # time-varying EMA decay

                # ── Decoupled weight decay ────────────────────────────────
                if wd != 0.0:
                    # Use the actual lr for the decay step; fall back to eps2
                    # in relative-step mode (lr=None) so decay is still applied.
                    alpha_wd = lr if lr is not None else eps2
                    p.mul_(1.0 - wd * alpha_wd)

                # ── Squared gradient with numerical floor ─────────────────
                # g_sq = g² + ε₁  — prevents zero row/col factors from
                # remaining at zero when a gradient is exactly zero.
                g_sq = g.pow(2).add_(eps1)    # shape same as g

                # ── Second moment update and V̂ reconstruction ─────────────
                if state["factored"]:
                    row = state["exp_avg_sq_row"]   # r_{t-1}
                    col = state["exp_avg_sq_col"]   # c_{t-1}

                    # Row factor: mean of g² over the last dimension (columns)
                    # r_t = ρ·r_{t-1} + (1-ρ)·g_sq.mean(dim=-1)
                    row.mul_(rho).add_(g_sq.mean(dim=-1), alpha=1.0 - rho)

                    # Col factor: mean of g² over the second-to-last dim (rows)
                    # c_t = ρ·c_{t-1} + (1-ρ)·g_sq.mean(dim=-2)
                    col.mul_(rho).add_(g_sq.mean(dim=-2), alpha=1.0 - rho)

                    # Reconstruct V̂ from the outer product, normalised by Σr
                    # V̂_{i,j} = r_i · c_j / (Σ_k r_k)
                    # row.sum() normalises so V̂ has the right total magnitude.
                    row_norm  = row.sum(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=eps1)
                    v_approx  = row.unsqueeze(-1) * col.unsqueeze(-2) / row_norm

                    # Adaptive update: g / √V̂   (like Adam's g / √v)
                    update = g / v_approx.sqrt().clamp(min=eps1)

                else:
                    # Full second moment (Adam-style) for 1-D tensors
                    v = state["exp_avg_sq"]
                    v.mul_(rho).add_(g_sq, alpha=1.0 - rho)
                    update = g / v.sqrt().clamp(min=eps1)

                # ── RMS clipping ──────────────────────────────────────────
                # Clip the update vector so its RMS ≤ clip_threshold.
                # Prevents catastrophically large steps in any direction.
                # WHY RMS not L2: RMS normalises by dimension count, making
                # the threshold independent of parameter tensor size.
                rms = self._rms(update)
                if rms > clip_threshold:
                    update.mul_(clip_threshold / rms)

                # ── Optional first moment (momentum smoothing) ────────────
                if use_first_moment:
                    m = state["exp_avg"]
                    # Fixed EMA with β₁=0.9 (not exposed as a hyperparameter
                    # since AdaFactor's design philosophy avoids extra tuning)
                    m.mul_(0.9).add_(update, alpha=0.1)
                    update = m

                # ── Step size ─────────────────────────────────────────────
                if lr is not None:
                    # Fixed LR mode: simple global step size
                    alpha = lr
                else:
                    # Relative/scale-free mode: step ∝ ‖p‖_rms / √t
                    # Automatically scales with parameter magnitude and decays
                    # over time without any user-specified schedule.
                    param_rms = max(self._rms(p.float()), eps2)  # floor at eps2
                    alpha     = param_rms * min(1.0 / (t ** 0.5), 0.01)

                # Cast update back to original dtype (e.g. float16 on GPU)
                p.add_(update.to(p.dtype), alpha=-alpha)

        return loss
