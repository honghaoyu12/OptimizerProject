"""Optimizer diagnostic metrics: Hessian trace, loss-surface sharpness,
dead neuron tracking, weight effective rank, gradient noise scale, Fisher
information trace, loss of plasticity, gradient alignment, spectral norm,
and optimizer state entropy.

All metrics probe the current training state from different angles:

  Hessian trace       Tr(H) — total curvature via Hutchinson's estimator.
  Sharpness           max_{‖δ‖≤ε} L(θ+δ) − L(θ) — SAM-style worst-case
                      loss increase under a bounded perturbation.
  Dead neurons        Fraction of ReLU units that never fire on a full
                      dataset pass — a proxy for wasted model capacity.
  Weight rank         Effective rank of each weight matrix via SVD entropy —
                      low rank indicates redundant or collapsed representations.
  Grad noise scale    ||E[g]||² / E[||g||²] — ratio of signal power to total
                      gradient power; large = low noise, small = dominated by
                      mini-batch variance.
  Fisher trace        E[||∇_θ log p(y|x)||²] — expected squared gradient norm,
                      a lightweight curvature proxy requiring only first-order
                      backprop.
  Plasticity score    Recovery delta-acc after resetting the last layer and
                      briefly retraining — high score = still plastic.
  Gradient alignment  Cosine similarity between the first and last linear
                      layer gradient directions — near +1 means end-to-end
                      cooperative update; near -1 means gradient reversal.
  Spectral norm       Largest singular value σ_max(W) per weight matrix —
                      proxy for the Lipschitz constant; high values risk
                      training instability.
  Optimizer entropy   Shannon entropy of the momentum / variance buffers —
                      collapses near convergence as state concentrates.

All functions catch exceptions and return float('nan') / empty dicts on
failure so that a single incompatible device does not crash the training run.
"""

import math

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


# ---------------------------------------------------------------------------
# Dead neuron tracker
# ---------------------------------------------------------------------------

def compute_dead_neurons(
    model: nn.Module,
    loader,
    device: torch.device,
    threshold: float = 0.0,
) -> dict[str, float]:
    """Fraction of ReLU units that never produce a positive activation.

    A "dead" neuron is one whose pre-activation output is ≤ *threshold*
    for *every* sample in the loader.  These units contribute nothing to
    the forward pass and cannot receive a useful gradient — they represent
    wasted model capacity that the optimizer has failed to utilise.

    Common causes of dead neurons:
      • Too-high learning rate (large negative weight updates after init)
      • Dying ReLU problem (gradient never flows back through a unit)
      • Poor weight initialisation (all pre-activations start negative)

    Implementation
    --------------
    Forward hooks are attached to every nn.ReLU module in the model.  Each
    hook records a boolean *ever-active* mask: True if the output was > 0
    at least once across all samples.  After the full loader pass, the
    dead fraction is (1 − mean(ever_active)) per layer.

    The model is run in eval() mode (no BatchNorm/Dropout side-effects) and
    all gradient computation is disabled.  Hooks are removed unconditionally
    on exit (success or exception).

    Parameters
    ----------
    model     : nn.Module — ReLU activations are detected by isinstance check
    loader    : DataLoader providing (images, labels) batches
    device    : torch.device
    threshold : activation value below which a unit is considered "not fired"
                (default 0.0 — standard ReLU dead zone)

    Returns
    -------
    dict[str, float]
        Mapping from ReLU layer name to dead neuron fraction in [0, 1].
        Returns an empty dict if no ReLU modules are found or on failure.
    """
    try:
        model.eval()

        # ── Find all ReLU submodules ───────────────────────────────────────
        relu_layers: dict[str, nn.Module] = {
            name: mod
            for name, mod in model.named_modules()
            if isinstance(mod, nn.ReLU)
        }
        if not relu_layers:
            return {}

        # ── ever_active[name] accumulates a boolean mask across all batches ─
        # Initialised to None; set to zeros_like(output) on first batch, then
        # OR'd with (output > threshold) for each subsequent batch.
        ever_active: dict[str, torch.Tensor | None] = {n: None for n in relu_layers}

        handles = []
        def _make_hook(name: str):
            def _hook(module, inp, output):
                # output shape: (batch, *feature_dims)
                # Reduce over the batch dimension to get a per-unit bool mask.
                # For a Linear output shape is (B, H); for Conv it is (B, C, H, W).
                fired = (output > threshold)   # (B, ...)
                # Collapse batch dim: a unit "fired" if it fired for ANY sample.
                fired_any = fired.any(dim=0)   # (*feature_dims,)
                if ever_active[name] is None:
                    ever_active[name] = fired_any
                else:
                    ever_active[name] = ever_active[name] | fired_any
            return _hook

        for name, mod in relu_layers.items():
            handles.append(mod.register_forward_hook(_make_hook(name)))

        try:
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    model(images)
        finally:
            # Always remove hooks, even if forward pass raises an exception
            for h in handles:
                h.remove()

        # ── Compute dead fraction per layer ───────────────────────────────
        result: dict[str, float] = {}
        for name, mask in ever_active.items():
            if mask is None:
                result[name] = float("nan")
            else:
                total  = mask.numel()
                active = mask.sum().item()
                result[name] = float(1.0 - active / total) if total > 0 else float("nan")
        return result

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Weight effective rank
# ---------------------------------------------------------------------------

def compute_weight_rank(model: nn.Module) -> dict[str, float]:
    """Effective rank of each Linear (and Conv2d) weight matrix via SVD entropy.

    The effective rank of a matrix W with singular values σ₁ ≥ … ≥ σ_k is:

        erank(W) = exp( −∑_i p_i · log(p_i) )

    where p_i = σ_i / ∑_j σ_j are the normalised singular values (treated as
    a probability distribution over the spectrum).  This equals the exponential
    of the Shannon entropy of the singular value distribution.

    Interpretation
    --------------
    • erank = 1 : all weight magnitude concentrated in one direction (extreme
      collapse / almost rank-1 matrix)
    • erank = k : all singular values equal (the matrix is effectively full-rank)

    A *decreasing* erank over training means the weight matrix is collapsing to
    a lower-dimensional subspace — often a sign that the optimizer is not
    exploring the full capacity of the layer.  An *increasing* erank means the
    layer is differentiating its representations.

    For Conv2d layers, the weight tensor (out_c, in_c, kH, kW) is reshaped to
    a 2-D matrix (out_c, in_c × kH × kW) before computing the SVD.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    dict[str, float]
        Mapping from layer name to effective rank (float ≥ 1).
        Layers with a single singular value return erank = 1.0.
        Returns an empty dict on failure.
    """
    try:
        result: dict[str, float] = {}

        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Linear, nn.Conv2d)):
                W = mod.weight.data

                # Reshape Conv2d to 2-D: (out_channels, in_channels * kH * kW)
                if W.dim() > 2:
                    W = W.reshape(W.shape[0], -1)

                # SVD on CPU float32 (same pattern as Shampoo / SOAP) to ensure
                # device-independence and numerical stability.
                W_cpu = W.detach().float().cpu()

                # torch.linalg.svdvals returns only the singular values (faster
                # than full SVD since we don't need the singular vectors).
                sv = torch.linalg.svdvals(W_cpu)   # shape (k,), k = min(m, n)

                sv_sum = sv.sum().item()
                if sv_sum < 1e-12:
                    # Degenerate zero matrix — rank is effectively 0 / undefined
                    result[name] = float("nan")
                    continue

                # ── Compute entropy of normalised singular value distribution ─
                # p_i = σ_i / ∑σ; entropy H = −∑ p_i · log(p_i)
                p = sv / sv_sum          # normalised weights, sum to 1
                # Add eps to avoid log(0) for zero singular values (which have
                # p_i = 0 and should contribute 0·log(0) = 0 to the entropy).
                log_p = torch.log(p.clamp(min=1e-10))
                entropy = -(p * log_p).sum().item()

                # erank = exp(H) — ranges from 1 (rank-1) to k (full-rank)
                result[name] = float(math.exp(entropy))

        return result

    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Gradient noise scale
# ---------------------------------------------------------------------------

def compute_gradient_noise_scale(
    grad_mean_sq_norm: float,
    grad_sq_mean_norm: float,
) -> float:
    """Gradient noise scale: ratio of signal power to total gradient power.

    The gradient noise scale (GNS) measures how much of the per-batch
    gradient norm is "signal" (i.e., direction that agrees across batches)
    versus "noise" (direction that cancels across batches).

    Definition
    ----------
    Given mini-batch gradients g₁, …, g_B collected within an epoch:

      GNS = ||E[g]||² / E[||g||²]

    where the expectation is over mini-batches.

    Interpretation
    --------------
    • GNS ≈ 1 : gradients are nearly identical across batches — very low
               noise.  The full-batch and mini-batch gradients agree.
    • GNS ≈ 0 : gradient directions cancel across batches — high noise.
               The mini-batch gradient is a poor estimate of the true gradient.
    • GNS = B  (batch size) at the optimal batch size from the OpenAI
               scaling paper (McCandlish et al., 2018) — used as a guide for
               choosing the right batch size relative to gradient noise.

    This function takes pre-computed aggregates (already available from the
    per-batch accumulation in train_one_epoch) and returns the scalar GNS.

    Parameters
    ----------
    grad_mean_sq_norm : ||mean(g)||²  — squared norm of the mean gradient
                        (computed as ||mean_grad_vec||² over all parameters
                        after averaging per-batch gradients)
    grad_sq_mean_norm : mean(||g||²)  — mean of the squared gradient norms
                        across mini-batches within the epoch

    Returns
    -------
    float
        GNS in [0, 1].  Returns float('nan') if grad_sq_mean_norm ≤ 0.
    """
    if grad_sq_mean_norm <= 0.0:
        return float("nan")
    return float(grad_mean_sq_norm / grad_sq_mean_norm)


# ---------------------------------------------------------------------------
# Fisher information trace
# ---------------------------------------------------------------------------

def compute_fisher_trace(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    n_samples: int = 50,
) -> float:
    """Fisher information trace: expected squared gradient norm.

    The Fisher information matrix F is defined as:

        F = E_{(x,y)~p_data}[ ∇_θ log p(y|x) · ∇_θ log p(y|x)^T ]

    Its trace Tr(F) = E[||∇_θ log p(y|x)||²] equals the *expected squared
    gradient norm* over the data distribution — a cheap, first-order-only
    proxy for the curvature seen by the optimizer.

    Unlike the Hessian trace (which requires second-order backprop), the
    Fisher trace requires only standard first-order gradients and is
    therefore O(p) in memory and O(1 backward pass per sample).

    Interpretation
    --------------
    • High Fisher trace → steep gradients on average → the loss surface is
      well-conditioned for gradient-based optimisation (there is strong
      gradient signal).
    • Decreasing Fisher trace during training often indicates that the
      model is converging (loss gradients shrink near a minimum).
    • Differences across optimizers at the same epoch reflect how each
      optimizer shapes the effective loss landscape.

    Implementation
    --------------
    We sample *n_samples* mini-batches from the loader (or fewer if the
    loader has fewer batches), compute the gradient norm² for each, and
    return the average.  Zero-grad is called before each backward so
    gradients don't accumulate across batches.

    Parameters
    ----------
    model     : nn.Module — evaluated in train() mode so BatchNorm uses
                running stats correctly
    loader    : DataLoader providing (images, labels) batches
    criterion : loss function (plain CrossEntropyLoss recommended)
    device    : torch.device
    n_samples : number of mini-batches to average over (default 50)

    Returns
    -------
    float
        Estimated Tr(F).  Returns float('nan') on failure.
    """
    try:
        model.train()   # keep BatchNorm in train mode for correct behaviour
        params = [p for p in model.parameters() if p.requires_grad]

        sq_norm_accum = 0.0
        count = 0

        for images, labels in loader:
            if count >= n_samples:
                break

            images, labels = images.to(device), labels.to(device)

            # Zero existing gradients so each backward is independent.
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            # Compute total squared gradient norm across all parameters.
            # ||∇_θ L||² = ∑_i ||∇_{θ_i} L||² (Frobenius norm squared for matrices)
            batch_sq_norm = sum(
                p.grad.norm() ** 2 for p in params if p.grad is not None
            ).item()
            sq_norm_accum += batch_sq_norm
            count += 1

        if count == 0:
            return float("nan")

        # Zero gradients after we're done so training is not affected.
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        return float(sq_norm_accum / count)

    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Plasticity score
# ---------------------------------------------------------------------------

def compute_plasticity_score(
    model: "nn.Module",
    train_loader,
    test_loader,
    criterion: "nn.Module",
    device: "torch.device",
    n_epochs: int = 3,
) -> float:
    """Estimate the network's remaining plasticity via last-layer reset.

    Resets the last Linear layer's weights to their initialisation defaults,
    then fine-tunes for *n_epochs* epochs (all other layers frozen) and
    measures the accuracy recovery.  A high score means the network can
    still reorganise its final representations quickly — i.e. it retains
    plasticity.  A score near 0 means the hidden representations are too
    rigid to support rapid re-adaptation (loss of plasticity).

    The returned value is the test accuracy achieved after recovery, so it
    lives in [0, 1] and can be compared directly across optimizers and
    epochs.

    Parameters
    ----------
    model       : nn.Module — the *currently trained* model (not modified
                  in-place; a deep copy is used internally)
    train_loader: DataLoader for training data
    test_loader : DataLoader for evaluation
    criterion   : loss function (e.g. CrossEntropyLoss)
    device      : torch.device
    n_epochs    : number of fine-tuning epochs after the last-layer reset
                  (default 3 — enough to measure recovery without being
                  expensive)

    Returns
    -------
    float
        Test accuracy after recovery in [0, 1], or float('nan') on failure.
    """
    import copy

    try:
        # Work on a deep copy so the original model is never touched.
        m = copy.deepcopy(model).to(device)

        # Locate the last nn.Linear layer.
        last_linear: nn.Linear | None = None
        for mod in m.modules():
            if isinstance(mod, nn.Linear):
                last_linear = mod
        if last_linear is None:
            return float("nan")

        # Reset the last linear layer to default initialisation (Kaiming
        # uniform for weight, uniform for bias — matches nn.Linear.__init__).
        nn.init.kaiming_uniform_(last_linear.weight, a=math.sqrt(5))
        if last_linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(last_linear.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(last_linear.bias, -bound, bound)

        # Freeze all layers except the last linear.
        for p in m.parameters():
            p.requires_grad_(False)
        for p in last_linear.parameters():
            p.requires_grad_(True)

        # Fine-tune with a fresh Adam on the last layer only.
        opt = torch.optim.Adam(last_linear.parameters(), lr=1e-3)
        m.train()
        for _ in range(n_epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad()
                criterion(m(images), labels).backward()
                opt.step()

        # Measure recovery accuracy on the test set.
        m.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = m(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        return float(correct / total) if total > 0 else float("nan")

    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Gradient alignment (first ↔ last layer)
# ---------------------------------------------------------------------------

def compute_gradient_alignment(
    mean_grad_vec: "dict[str, torch.Tensor | None]",
) -> float:
    """Cosine similarity between the first and last linear layer's mean
    gradient directions.

    A value near +1 means both ends of the network are pushing in the same
    direction — a healthy, end-to-end cooperative update.  Values near 0
    indicate orthogonal updates (the layers are learning independently), and
    negative values suggest gradient reversal (the two ends are fighting each
    other), which is a warning sign of vanishing or exploding gradient flow.

    Parameters
    ----------
    mean_grad_vec : dict mapping layer name → mean gradient tensor (or None)
                    as returned by ``train_one_epoch``'s
                    ``mean_grad_vec_per_layer`` key.  Must have at least two
                    entries with non-None tensors.

    Returns
    -------
    float
        Cosine similarity in [-1, 1], or float('nan') if fewer than two
        layers have valid gradient vectors.
    """
    try:
        # Collect layers that have a valid (non-None) gradient vector.
        valid = [
            v for v in mean_grad_vec.values()
            if v is not None
        ]
        if len(valid) < 2:
            return float("nan")

        first = valid[0].flatten()
        last  = valid[-1].flatten()

        # Cosine similarity requires matching dimensions only through the
        # dot product — no reshaping needed as both are already 1-D.
        cos_sim = torch.nn.functional.cosine_similarity(
            first.unsqueeze(0), last.unsqueeze(0)
        ).item()
        return float(cos_sim)

    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Spectral norm of weight matrices
# ---------------------------------------------------------------------------

def compute_spectral_norm(model: "nn.Module") -> "dict[str, float]":
    """Largest singular value σ_max(W) for each Linear layer's weight matrix.

    The spectral norm is the tightest upper bound on how much a linear
    transformation can stretch a vector — it equals the Lipschitz constant
    of that layer.  High spectral norms can cause training instability and
    are particularly relevant for understanding gradient explosion.

    Uses ``torch.linalg.svdvals`` on a CPU float32 copy of each weight
    matrix (same device-agnostic pattern as ``compute_weight_rank``).

    Parameters
    ----------
    model : nn.Module — any module; only ``nn.Linear`` submodules are
            inspected.

    Returns
    -------
    dict[str, float]
        Mapping from layer name (from ``named_modules``) to σ_max ≥ 0.
        Empty dict if the model has no Linear layers.  Returns float('nan')
        for a specific layer if SVD fails.
    """
    result: dict[str, float] = {}
    try:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                try:
                    # Compute on CPU float32 for numerical stability and
                    # device independence (same pattern as weight rank).
                    W_cpu = mod.weight.detach().float().cpu()
                    sv = torch.linalg.svdvals(W_cpu)
                    result[name] = float(sv[0].item())
                except Exception:
                    result[name] = float("nan")
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Optimizer state entropy
# ---------------------------------------------------------------------------

def compute_optimizer_state_entropy(optimizer: "torch.optim.Optimizer") -> float:
    """Shannon entropy of the momentum / variance buffers across all parameters.

    Inspects the optimizer's ``state_dict()`` for buffer tensors
    (``exp_avg``, ``exp_avg_sq``, ``momentum_buffer``, ``m``, ``v``,
    ``exp_avg_sq_row``, ``exp_avg_sq_col``).  All found buffers are
    flattened, concatenated, converted to probabilities via ``softmax``,
    and their Shannon entropy is returned in nats.

    Interpretation
    --------------
    High entropy  → the buffers are spread across many different values,
                    meaning the optimizer is still actively exploring the
                    loss landscape.
    Low entropy   → the buffers have concentrated (many near-zero, a few
                    dominating) which is typical of convergence or collapse.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer — any optimizer that stores
                first/second moment tensors in ``param_groups`` state.

    Returns
    -------
    float
        Shannon entropy in nats ≥ 0, or float('nan') if no buffers are found.
    """
    # Keys used by common optimizers for their moment buffers.
    _BUFFER_KEYS = {
        "exp_avg",        # Adam, AdamW, NAdam, RAdam, Lion, Adan, ...
        "exp_avg_sq",     # Adam, AdamW, NAdam, RAdam, ...
        "momentum_buffer",# SGDMomentum
        "m",              # Tiger, custom
        "v",              # custom
        "exp_avg_sq_row", # AdaFactor (factored second moment rows)
        "exp_avg_sq_col", # AdaFactor (factored second moment cols)
    }
    try:
        fragments: list[torch.Tensor] = []
        state = optimizer.state_dict()["state"]
        for param_state in state.values():
            for key, val in param_state.items():
                if key in _BUFFER_KEYS and isinstance(val, torch.Tensor):
                    fragments.append(val.detach().float().cpu().reshape(-1))

        if not fragments:
            return float("nan")

        # Concatenate all buffers, compute softmax probabilities, then
        # Shannon entropy H = -Σ p_i · log(p_i).
        all_vals = torch.cat(fragments)
        # Take absolute values so negative momenta are treated symmetrically,
        # then normalise to a probability distribution via softmax.
        probs = torch.softmax(all_vals.abs(), dim=0)
        # Clamp to avoid log(0); probs are already in (0,1] after softmax.
        entropy = -(probs * probs.clamp(min=1e-10).log()).sum().item()
        return float(entropy)

    except Exception:
        return float("nan")
