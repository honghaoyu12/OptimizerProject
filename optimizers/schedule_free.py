"""Schedule-Free AdamW optimizer.

From "The Road Less Scheduled" (Defazio, Yang, Khaled, Mishchenko,
Defazio & Cutkosky, NeurIPS 2024).  https://arxiv.org/abs/2405.15682

Problem with Learning Rate Schedules
-------------------------------------
All standard schedules (cosine, step, warmup-cosine) require the user to
specify the total training budget T in advance.  If training is extended or
cut short, the schedule is no longer calibrated.  Moreover, the "optimal"
schedule is task-dependent and must be re-tuned for each new experiment.

Schedule-Free Optimisation
--------------------------
Schedule-Free methods achieve the convergence benefits of a decaying LR by
replacing the schedule with *iterate averaging* (Polyak-Ruppert averaging).

The core observation: a constant-LR optimizer converges to the neighbourhood
of the optimum, but the iterates *oscillate* around it.  Averaging iterates
θ₁, θ₂, ..., θ_k produces a sequence that converges toward the true minimum
*in expectation*, effectively mimicking a decaying LR for the averaged iterate
without ever changing the actual LR.

The three sequences:
  z  — "primal" iterate: receives the Adam gradient step
       (not stored separately; reconstructed from y and x each step)
  x  — Polyak-Ruppert average of z: x_k = (1 − 1/k)·x_{k-1} + (1/k)·z_k
       (accumulated in state["x"])
  y  — interpolation of x and z, stored in model.parameters():
       y_{k+1} = (1−β₁)·x_k + β₁·z_k

The model always holds y (gradient is evaluated at y), but at inference time
x (the averaged iterate) should be used — it is more stable and typically
achieves better accuracy.

Algorithm (one step, params = y_k)
-----------------------------------
  g_k  = ∇f(y_k)                                    [gradient at y]
  v_k  = β₂·v_{k−1} + (1−β₂)·g_k²                  [second moment at y]
  z_k  = (y_k − (1−β₁)·x_{k−1}) / β₁               [reconstruct z from y,x]
  z_k  = z_k − α·(g_k/(√v_k + ε) + λ·z_k)          [Adam step on z]
  x_k  = (1−1/k)·x_{k−1} + (1/k)·z_k               [Polyak average]
  y_{k+1} = (1−β₁)·x_k + β₁·z_k                    [new params]
"""

import torch
from .base import BaseOptimizer


class ScheduleFreeAdamW(BaseOptimizer):
    """Schedule-Free AdamW: Adam with built-in Polyak-Ruppert averaging.

    Note on evaluation:
    During training, model.parameters() holds y (the interpolated iterate).
    For final evaluation, load the averaged weights x via get_averaged_params()
    — they are more stable and usually achieve better accuracy.

    Parameters
    ----------
    params       : model parameters — these will be y during training
    lr           : constant learning rate (default 1e-3; no schedule needed)
    betas        : (β₁, β₂)
                   β₁ = interpolation weight between x and z (controls
                        how much the params stay close to the average vs.
                        the latest gradient step)
                   β₂ = Adam second-moment EMA decay rate
                   Default (0.9, 0.999)
    eps          : numerical stability constant for Adam denominator (1e-8)
    weight_decay : decoupled weight decay applied to z (the primal iterate),
                   not to x or y (default 0.0)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
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

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            wd           = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g     = p.grad      # gradient at y_k (the current params)
                state = self.state[p]

                # ── Initialise state on first step ────────────────────────
                if len(state) == 0:
                    state["step"] = 0
                    # x starts equal to the initial params (x_0 = y_0 = z_0).
                    # WHY clone(): state["x"] must be an independent tensor;
                    # without clone() it would share storage with p.data and
                    # get overwritten when we update y at the end of step().
                    state["x"]           = p.data.clone()
                    # v: Adam second moment, initialised to zero (standard)
                    state["exp_avg_sq"]  = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]   # 1-based step counter for Polyak averaging
                x = state["x"]      # running average x_{k-1}
                v = state["exp_avg_sq"]

                # ── Step 1: Second moment update (at y = p) ───────────────
                # v_k = β₂·v_{k-1} + (1−β₂)·g_k²
                # The second moment is evaluated at y (current params), not z.
                # This is consistent because the gradient was also computed at y.
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                # ── Step 2: Reconstruct z from y and x ────────────────────
                # y_k = (1−β₁)·x_{k-1} + β₁·z_k
                # ⟹  z_k = (y_k − (1−β₁)·x_{k-1}) / β₁
                #
                # WHY reconstruct z rather than store it: storing z would
                # require an extra full copy of the parameters.  Reconstructing
                # from y and x costs O(1) extra memory (just arithmetic).
                if beta1 > 0.0:
                    z = (p.data - (1.0 - beta1) * x) / beta1
                else:
                    # β₁=0 means y=x=z (no interpolation); z is just a copy
                    z = p.data.clone()

                # ── Step 3: Adam step on z ────────────────────────────────
                # Decoupled weight decay on z (not on y or x)
                if wd != 0.0:
                    z.mul_(1.0 - lr * wd)

                # Adam update: z ← z − lr · g / (√v + ε)
                # No bias correction for v here (same as AdamW; simplifies
                # the schedule-free analysis in the paper).
                denom = v.sqrt().add_(eps)
                z.addcdiv_(g, denom, value=-lr)   # z += −lr · g/denom

                # ── Step 4: Polyak-Ruppert averaging ──────────────────────
                # x_k = x_{k-1} + (z_k − x_{k-1}) / k
                #      = (1 − 1/k) · x_{k-1} + (1/k) · z_k
                #
                # WHY divide by k (not use a fixed EMA β): equal-weight
                # averaging of all past z iterates is theoretically optimal
                # for convex problems.  The effective weight of old iterates
                # decays as 1/k naturally without specifying β.
                x.add_(z - x, alpha=1.0 / t)     # in-place Polyak update

                # ── Step 5: Update params to new y_{k+1} ──────────────────
                # y_{k+1} = (1−β₁)·x_k + β₁·z_k
                # This places the model "between" the stable average x and
                # the latest gradient step z — gradient evaluation at y
                # thus uses the stability of x while staying responsive to z.
                p.data.copy_((1.0 - beta1) * x + beta1 * z)

        return loss

    @torch.no_grad()
    def get_averaged_params(self) -> list[torch.Tensor]:
        """Return the Polyak-averaged weight tensors x for final evaluation.

        During training model.parameters() holds y (interpolation of x and z).
        For evaluation or inference, load x into the model — it is the
        statistically optimal estimate of the minimum.

        Example
        -------
        averaged = optimizer.get_averaged_params()
        for p, x in zip(model.parameters(), averaged):
            p.data.copy_(x)
        # evaluate or save the model here
        # then optionally restore y for continued training

        Returns
        -------
        list[torch.Tensor] — one tensor per parameter, in the same order
                             as model.parameters()
        """
        result = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "x" in state:
                    # Clone so the caller can modify without corrupting state
                    result.append(state["x"].clone())
                else:
                    # Parameter was never updated (no gradient); return current value
                    result.append(p.data.clone())
        return result
