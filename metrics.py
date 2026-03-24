"""Optimizer diagnostic metrics: Hessian trace and loss-surface sharpness.

Both metrics probe the local geometry of the loss surface around the current
model weights θ.  They are computed on a *fixed* mini-batch grabbed at the
start of training so that the values are comparable across epochs.

  Hessian trace   Tr(H) — measures total curvature.  High values indicate
                  a sharp minimum that may be sensitive to weight perturbations
                  and may generalise poorly.  Computed with Hutchinson's
                  stochastic estimator to avoid forming the full O(p²) Hessian.

  Sharpness       max_{‖δ‖≤ε} L(θ+δ) − L(θ) — the worst-case loss increase
                  under a bounded parameter perturbation.  Closely related to
                  the PAC-Bayes generalisation bound motivating SAM
                  (Foret et al., 2021; https://arxiv.org/abs/2010.01412).

Both functions catch all exceptions and return float('nan') on failure so that
a single incompatible device (e.g. MPS limitations on create_graph) does not
crash the entire training run — the value simply shows as NaN in plots.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Hessian trace estimator
# ---------------------------------------------------------------------------

def compute_hessian_trace(
    model: nn.Module,
    batch: tuple,
    criterion: nn.Module,
    device: torch.device,
    n_samples: int = 3,
) -> float:
    """Hutchinson estimator of the Hessian trace Tr(H).

    The true Hessian H ∈ ℝ^{p×p} is prohibitively expensive to materialise
    for modern networks.  Hutchinson's trick obtains an unbiased Monte Carlo
    estimate using random Rademacher vectors v ∈ {-1,+1}^p:

        Tr(H) = 𝔼_v[v^T H v]

    Each evaluation of v^T H v requires one HVP (Hessian-vector product),
    which costs the same as a second-order backward pass.  The estimate
    improves as n_samples grows; n_samples=3 is a cheap but reasonable default
    for monitoring trends over training.

    Implementation detail: HVPs are computed via the double-backward trick —
    first-order grads are computed with create_graph=True so that the
    autograd engine records the computation graph of the gradient, then a
    second backward pass differentiates g(θ)·v with respect to θ.

    Parameters
    ----------
    model     : nn.Module — evaluated in eval() mode (no BatchNorm updates)
    batch     : (images, labels) tuple — fixed training batch from the loop
    criterion : loss function; should be plain CrossEntropyLoss (no smoothing)
    device    : torch.device
    n_samples : number of Rademacher vectors to average (default 3)

    Returns
    -------
    float
        Estimated Tr(H).  Returns float('nan') on any failure (e.g. MPS
        autograd limitations with create_graph=True).
    """
    try:
        model.eval()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        # Only leaf tensors that participate in the loss contribute to H
        params = [p for p in model.parameters() if p.requires_grad]
        trace_estimates = []

        for _ in range(n_samples):
            # ── Draw a Rademacher random vector v ~ {-1, +1}^p ───────────
            # Each element is ±1 with equal probability, sampled via
            # randint(0,2) ∈ {0,1} then mapped to {-1,+1} via * 2 - 1.
            vs = [torch.randint(0, 2, p.shape, device=device).float() * 2 - 1
                  for p in params]

            # ── Forward pass — must record computation graph for HVP ──────
            logits = model(images)
            loss   = criterion(logits, labels)

            # ── First-order gradients (with graph retained) ───────────────
            # create_graph=True builds a higher-order graph so we can
            # differentiate through the gradient computation itself.
            grads = torch.autograd.grad(loss, params, create_graph=True)

            # ── Hessian-vector product: grad of (g·v) w.r.t. θ ───────────
            # g·v is a scalar, so its gradient is exactly H·v (chain rule).
            gv  = sum((g * v).sum() for g, v in zip(grads, vs))
            hvp = torch.autograd.grad(gv, params, retain_graph=False)

            # ── v^T H v — the Hutchinson estimator for this sample ────────
            vhv = sum((v * h).sum().item() for v, h in zip(vs, hvp))
            trace_estimates.append(vhv)

        # Average over n_samples; each sample is an unbiased estimate
        return float(sum(trace_estimates) / len(trace_estimates))

    except Exception:
        # create_graph is not supported on all backends (e.g. MPS < PyTorch 2.x)
        return float("nan")


# ---------------------------------------------------------------------------
# SAM-style sharpness estimator
# ---------------------------------------------------------------------------

def compute_sharpness(
    model: nn.Module,
    batch: tuple,
    criterion: nn.Module,
    device: torch.device,
    epsilon: float = 1e-2,
    n_directions: int = 5,
) -> float:
    """SAM-style sharpness: max loss increase under an ε-ball perturbation.

    Computes an approximate solution to the inner maximisation problem from
    Sharpness-Aware Minimisation (SAM, Foret et al., 2021):

        max_{‖δ‖_2 ≤ ε}  L(θ + δ) - L(θ)

    Rather than the gradient ascent step used by SAM, this function samples
    n_directions random unit perturbations scaled to exactly ‖δ‖_2 = ε and
    returns the maximum observed loss increase — a faster but slightly less
    tight estimate of the true sharpness.

    The original model weights are restored after each direction so the
    function is non-destructive (calling it has no effect on training).

    WHY n_directions=5 and not more: sharpness is an epoch-level diagnostic.
    5 directions is sufficient to catch obvious sharp minima while keeping the
    overhead small relative to the training step.  The true sharpness (gradient
    ascent solution) would require multiple forward+backward passes.

    Parameters
    ----------
    model        : nn.Module — evaluated in eval() mode
    batch        : (images, labels) — fixed training batch from the training loop
    criterion    : loss function (plain CrossEntropyLoss recommended)
    device       : torch.device
    epsilon      : perturbation ball radius ‖δ‖_2 ≤ ε (default 0.01)
    n_directions : number of random perturbation directions to try (default 5)

    Returns
    -------
    float
        Maximum observed loss increase L(θ+δ) − L(θ) over all sampled δ.
        Returns float('nan') on failure.
    """
    try:
        model.eval()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        params = [p for p in model.parameters() if p.requires_grad]

        # ── Compute baseline loss at current θ ────────────────────────────
        with torch.no_grad():
            logits    = model(images)
            base_loss = criterion(logits, labels).item()

        # ── Save a deep copy of all parameter tensors ─────────────────────
        # clone() is essential — without it 'original' would be a list of
        # views into parameter storage that get overwritten during add_().
        original = [p.data.clone() for p in params]

        max_increase = 0.0
        for _ in range(n_directions):
            # ── Sample a random direction and normalise to length ε ───────
            # torch.randn draws from N(0,I), giving a uniformly random
            # direction on the sphere (isotropic Gaussian).
            noise      = [torch.randn_like(p) for p in params]
            total_norm = (sum(n.norm() ** 2 for n in noise) ** 0.5)
            # scale maps the random vector to have Euclidean norm exactly ε
            scale      = epsilon / (total_norm + 1e-12)   # +1e-12 avoids div-by-zero

            # ── Apply perturbation δ = scale * noise ──────────────────────
            with torch.no_grad():
                for p, n in zip(params, noise):
                    p.data.add_(n * scale)

            # ── Evaluate perturbed loss L(θ + δ) ──────────────────────────
            with torch.no_grad():
                logits         = model(images)
                perturbed_loss = criterion(logits, labels).item()

            max_increase = max(max_increase, perturbed_loss - base_loss)

            # ── Restore original weights (in-place copy for speed) ────────
            with torch.no_grad():
                for p, orig in zip(params, original):
                    p.data.copy_(orig)

        return float(max_increase)

    except Exception:
        return float("nan")
