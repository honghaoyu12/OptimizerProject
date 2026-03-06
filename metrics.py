"""Optimizer diagnostic metrics: Hessian trace and sharpness estimators."""

import torch
import torch.nn as nn


def compute_hessian_trace(
    model: nn.Module,
    batch: tuple,
    criterion: nn.Module,
    device: torch.device,
    n_samples: int = 3,
) -> float:
    """Hutchinson estimator of the Hessian trace.

    Tr(H) ≈ (1/S) Σ v^T H v,  v ~ Rademacher(±1)

    Uses torch.autograd.grad with create_graph=True for Hessian-vector products.
    Returns float('nan') on failure (e.g. MPS autograd limitations).
    """
    try:
        model.eval()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        trace_estimates = []

        for _ in range(n_samples):
            # Draw Rademacher random vector
            vs = [torch.randint(0, 2, p.shape, device=device).float() * 2 - 1 for p in params]

            # Forward pass with graph
            logits = model(images)
            loss = criterion(logits, labels)

            # First-order gradients
            grads = torch.autograd.grad(loss, params, create_graph=True)

            # HVP: grad of (g·v) w.r.t. params
            gv = sum((g * v).sum() for g, v in zip(grads, vs))
            hvp = torch.autograd.grad(gv, params, retain_graph=False)

            # v^T H v
            vhv = sum((v * h).sum().item() for v, h in zip(vs, hvp))
            trace_estimates.append(vhv)

        return float(sum(trace_estimates) / len(trace_estimates))

    except Exception:
        return float("nan")


def compute_sharpness(
    model: nn.Module,
    batch: tuple,
    criterion: nn.Module,
    device: torch.device,
    epsilon: float = 1e-2,
    n_directions: int = 5,
) -> float:
    """SAM-style sharpness estimate.

    max_{‖δ‖ ≤ ε} L(θ + δ) - L(θ)

    Samples n_directions random unit perturbations scaled to epsilon, returns
    the maximum observed loss increase.  Saves and restores model weights.
    Returns float('nan') on failure.
    """
    try:
        model.eval()
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        params = [p for p in model.parameters() if p.requires_grad]

        # Baseline loss
        with torch.no_grad():
            logits = model(images)
            base_loss = criterion(logits, labels).item()

        # Save original weights
        original = [p.data.clone() for p in params]

        max_increase = 0.0
        for _ in range(n_directions):
            # Random unit perturbation scaled to epsilon
            noise = [torch.randn_like(p) for p in params]
            total_norm = (sum(n.norm() ** 2 for n in noise) ** 0.5)
            scale = epsilon / (total_norm + 1e-12)
            with torch.no_grad():
                for p, n in zip(params, noise):
                    p.data.add_(n * scale)

            # Perturbed loss
            with torch.no_grad():
                logits = model(images)
                perturbed_loss = criterion(logits, labels).item()

            max_increase = max(max_increase, perturbed_loss - base_loss)

            # Restore weights
            with torch.no_grad():
                for p, orig in zip(params, original):
                    p.data.copy_(orig)

        return float(max_increase)

    except Exception:
        return float("nan")
