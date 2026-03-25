"""Tests for metrics.py — Hessian trace, sharpness, dead neurons, weight rank,
gradient noise scale, Fisher trace, plasticity score, gradient alignment,
spectral norm, and optimizer state entropy estimators."""

import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from metrics import (compute_hessian_trace, compute_sharpness,
                     compute_dead_neurons, compute_weight_rank,
                     compute_gradient_noise_scale, compute_fisher_trace,
                     compute_plasticity_score, compute_gradient_alignment,
                     compute_spectral_norm, compute_optimizer_state_entropy)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_model_and_batch():
    """Return a tiny MLP, a small batch, and a criterion."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
    images = torch.randn(16, 1, 4, 4)   # 16 samples, 1×4×4
    labels = torch.randint(0, 4, (16,))
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")
    return model, (images, labels), criterion, device


def tiny_loader(n=64, in_features=16, num_classes=4, batch_size=16):
    """Return a simple DataLoader over random tabular data."""
    X = torch.randn(n, 1, 4, 4)   # image-style for compatibility with tiny_model
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size)


# ---------------------------------------------------------------------------
# Hessian trace
# ---------------------------------------------------------------------------

class TestHessianTrace:
    def test_returns_float(self):
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_hessian_trace(model, batch, criterion, device)
        assert isinstance(result, float)

    def test_finite_on_valid_model(self):
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_hessian_trace(model, batch, criterion, device, n_samples=2)
        assert math.isfinite(result)

    def test_positive(self):
        # Hessian trace of a convex loss should be positive
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_hessian_trace(model, batch, criterion, device, n_samples=5)
        assert result > 0

    def test_nan_on_broken_model(self):
        # A model with no parameters should return nan gracefully
        model = nn.Identity()
        batch = (torch.randn(4, 4), torch.randint(0, 2, (4,)))
        criterion = nn.CrossEntropyLoss()
        result = compute_hessian_trace(model, batch, criterion, torch.device("cpu"))
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------

class TestSharpness:
    def test_returns_float(self):
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_sharpness(model, batch, criterion, device)
        assert isinstance(result, float)

    def test_non_negative(self):
        # Sharpness is max loss increase — should be >= 0
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_sharpness(model, batch, criterion, device, n_directions=3)
        assert result >= 0.0

    def test_finite(self):
        model, batch, criterion, device = tiny_model_and_batch()
        result = compute_sharpness(model, batch, criterion, device)
        assert math.isfinite(result)

    def test_weights_restored(self):
        """Weights must be identical before and after compute_sharpness."""
        model, batch, criterion, device = tiny_model_and_batch()
        weights_before = [p.data.clone() for p in model.parameters()]
        compute_sharpness(model, batch, criterion, device, n_directions=3)
        for before, p in zip(weights_before, model.parameters()):
            assert torch.allclose(before, p.data), "Weights were not restored"

    def test_larger_epsilon_increases_sharpness(self):
        """Bigger perturbation radius should generally yield equal or larger sharpness."""
        model, batch, criterion, device = tiny_model_and_batch()
        torch.manual_seed(0)
        small = compute_sharpness(model, batch, criterion, device, epsilon=1e-4, n_directions=10)
        torch.manual_seed(0)
        large = compute_sharpness(model, batch, criterion, device, epsilon=1.0, n_directions=10)
        assert large >= small


# ---------------------------------------------------------------------------
# Dead neurons
# ---------------------------------------------------------------------------

class TestDeadNeurons:
    def test_returns_dict(self):
        model, _, _, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, device)
        assert isinstance(result, dict)

    def test_keys_are_relu_layers(self):
        """Keys should correspond to ReLU module names in the model."""
        model, _, _, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, device)
        relu_names = {name for name, mod in model.named_modules()
                      if isinstance(mod, nn.ReLU)}
        assert set(result.keys()) == relu_names

    def test_fractions_in_range(self):
        """Dead fractions must be in [0, 1]."""
        model, _, _, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, device)
        for name, frac in result.items():
            assert 0.0 <= frac <= 1.0, f"{name}: {frac} out of [0,1]"

    def test_always_dead_relu_has_fraction_one(self):
        """A model where all pre-activations are forced negative → fraction=1."""
        # Build a model that always produces negative pre-relu values:
        # Linear(-1, ...) so all outputs are negative → ReLU kills everything.
        model = nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU())
        with torch.no_grad():
            model[1].weight.fill_(-1.0)   # all negative weights
            model[1].bias.fill_(-10.0)    # large negative bias
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, torch.device("cpu"))
        assert len(result) == 1
        key = list(result.keys())[0]
        assert result[key] == 1.0

    def test_no_relu_returns_empty(self):
        """Model without any ReLU should return an empty dict."""
        model = nn.Sequential(nn.Flatten(), nn.Linear(16, 4))
        loader = tiny_loader()
        result = compute_dead_neurons(model, loader, torch.device("cpu"))
        assert result == {}

    def test_weights_unchanged_after_call(self):
        """compute_dead_neurons must not modify model weights."""
        model, _, _, device = tiny_model_and_batch()
        loader = tiny_loader()
        weights_before = [p.data.clone() for p in model.parameters()]
        compute_dead_neurons(model, loader, device)
        for before, p in zip(weights_before, model.parameters()):
            assert torch.allclose(before, p.data)


# ---------------------------------------------------------------------------
# Weight rank
# ---------------------------------------------------------------------------

class TestWeightRank:
    def test_returns_dict(self):
        model, _, _, device = tiny_model_and_batch()
        result = compute_weight_rank(model)
        assert isinstance(result, dict)

    def test_keys_are_linear_layers(self):
        """Keys should correspond to Linear layer names in the model."""
        model, _, _, device = tiny_model_and_batch()
        result = compute_weight_rank(model)
        linear_names = {name for name, mod in model.named_modules()
                        if isinstance(mod, nn.Linear)}
        assert set(result.keys()) == linear_names

    def test_rank_at_least_one(self):
        """Effective rank must be ≥ 1 for any non-degenerate matrix."""
        model, _, _, device = tiny_model_and_batch()
        result = compute_weight_rank(model)
        for name, rank in result.items():
            if not math.isnan(rank):
                assert rank >= 1.0 - 1e-6, f"{name}: rank={rank} < 1"

    def test_rank_bounded_by_min_dim(self):
        """Effective rank must be ≤ min(rows, cols)."""
        model = nn.Linear(8, 4)   # weight shape 4×8, min dim = 4
        result = compute_weight_rank(model)
        for name, rank in result.items():
            if not math.isnan(rank):
                assert rank <= 4.0 + 1e-4, f"{name}: rank={rank} > min_dim=4"

    def test_rank_one_matrix(self):
        """An exactly rank-1 matrix should have effective rank ≈ 1."""
        model = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            # Outer product v·v^T is exactly rank 1
            v = torch.randn(4)
            model.weight.data = v.unsqueeze(1) * v.unsqueeze(0)
        result = compute_weight_rank(model)
        rank = list(result.values())[0]
        assert abs(rank - 1.0) < 0.05, f"Rank-1 matrix gave erank={rank}"

    def test_empty_model_returns_empty(self):
        """A model with no Linear or Conv2d layers returns empty dict."""
        model = nn.Sequential(nn.ReLU())
        result = compute_weight_rank(model)
        assert result == {}


# ---------------------------------------------------------------------------
# Gradient noise scale
# ---------------------------------------------------------------------------

class TestGradientNoiseScale:
    def test_returns_float(self):
        result = compute_gradient_noise_scale(1.0, 2.0)
        assert isinstance(result, float)

    def test_value_in_range(self):
        """GNS = signal² / total² must be in [0, 1] when signal ≤ total."""
        # ||E[g]||² ≤ E[||g||²] by Jensen's inequality
        result = compute_gradient_noise_scale(0.5, 2.0)
        assert 0.0 <= result <= 1.0

    def test_perfect_agreement(self):
        """When signal equals total (no noise), GNS should be 1."""
        result = compute_gradient_noise_scale(3.0, 3.0)
        assert abs(result - 1.0) < 1e-9

    def test_zero_denominator_returns_nan(self):
        """Zero total power should return NaN."""
        result = compute_gradient_noise_scale(0.0, 0.0)
        assert math.isnan(result)

    def test_proportional_to_signal_power(self):
        """Higher signal power → higher GNS (for fixed total power)."""
        low  = compute_gradient_noise_scale(0.1, 1.0)
        high = compute_gradient_noise_scale(0.9, 1.0)
        assert high > low


# ---------------------------------------------------------------------------
# Fisher trace
# ---------------------------------------------------------------------------

class TestFisherTrace:
    def test_returns_float(self):
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        assert isinstance(result, float)

    def test_finite_on_valid_model(self):
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        assert math.isfinite(result)

    def test_non_negative(self):
        """Fisher trace = E[||∇||²] — must be ≥ 0."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        assert result >= 0.0

    def test_weights_unchanged_after_call(self):
        """compute_fisher_trace must not modify model weights."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        weights_before = [p.data.clone() for p in model.parameters()]
        compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        for before, p in zip(weights_before, model.parameters()):
            assert torch.allclose(before, p.data)

    def test_gradients_zeroed_after_call(self):
        """Gradients should be zeroed after the call so training is not affected."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        compute_fisher_trace(model, loader, criterion, device, n_samples=5)
        for p in model.parameters():
            if p.grad is not None:
                assert (p.grad == 0).all()


# ---------------------------------------------------------------------------
# Plasticity score
# ---------------------------------------------------------------------------

class TestPlasticityScore:
    def test_returns_float(self):
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_plasticity_score(model, loader, loader, criterion, device, n_epochs=1)
        assert isinstance(result, float)

    def test_in_zero_one_range(self):
        """Recovery accuracy must be in [0, 1]."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_plasticity_score(model, loader, loader, criterion, device, n_epochs=1)
        if not math.isnan(result):
            assert 0.0 <= result <= 1.0

    def test_original_model_unchanged(self):
        """compute_plasticity_score must not modify the original model weights."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        weights_before = [p.data.clone() for p in model.parameters()]
        compute_plasticity_score(model, loader, loader, criterion, device, n_epochs=1)
        for before, p in zip(weights_before, model.parameters()):
            assert torch.allclose(before, p.data), "Original model weights were modified"

    def test_no_linear_returns_nan(self):
        """A model with no Linear layers should return NaN gracefully."""
        model = nn.Sequential(nn.Flatten(), nn.ReLU())
        criterion = nn.CrossEntropyLoss()
        loader = tiny_loader()
        result = compute_plasticity_score(model, loader, loader, criterion,
                                          torch.device("cpu"), n_epochs=1)
        assert math.isnan(result)

    def test_finite_on_valid_model(self):
        """Standard tiny model should give a finite score."""
        model, _, criterion, device = tiny_model_and_batch()
        loader = tiny_loader()
        result = compute_plasticity_score(model, loader, loader, criterion, device, n_epochs=2)
        assert math.isfinite(result)


# ---------------------------------------------------------------------------
# Gradient alignment
# ---------------------------------------------------------------------------

class TestGradientAlignment:
    def _make_grad_vec(self, *shapes):
        """Return a mean_grad_vec dict with random tensors for given shapes."""
        names = [f"layer_{i}" for i in range(len(shapes))]
        return {name: torch.randn(*shape) for name, shape in zip(names, shapes)}

    def test_returns_float(self):
        gv = self._make_grad_vec((4, 8), (2, 4))
        result = compute_gradient_alignment(gv)
        assert isinstance(result, float)

    def test_value_in_range(self):
        """Cosine similarity must be in [-1, 1]."""
        gv = self._make_grad_vec((4, 8), (2, 4))
        result = compute_gradient_alignment(gv)
        if not math.isnan(result):
            assert -1.0 - 1e-6 <= result <= 1.0 + 1e-6

    def test_identical_vectors_gives_one(self):
        """If first and last layer have identical gradient vectors, similarity = 1."""
        v = torch.ones(4, 8)
        gv = {"layer_0": v, "layer_1": v}
        result = compute_gradient_alignment(gv)
        assert abs(result - 1.0) < 1e-5

    def test_opposite_vectors_gives_minus_one(self):
        """If last layer grad is the negation of the first, similarity = -1."""
        v = torch.ones(4, 8)
        gv = {"layer_0": v, "layer_1": -v}
        result = compute_gradient_alignment(gv)
        assert abs(result + 1.0) < 1e-5

    def test_single_layer_returns_nan(self):
        """Only one layer with valid grad → cannot compare first vs last → NaN."""
        gv = {"layer_0": torch.randn(4, 8), "layer_1": None}
        result = compute_gradient_alignment(gv)
        assert math.isnan(result)

    def test_empty_dict_returns_nan(self):
        """Empty grad dict → NaN."""
        result = compute_gradient_alignment({})
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# Spectral norm
# ---------------------------------------------------------------------------

class TestSpectralNorm:
    def test_returns_dict(self):
        model, _, _, _ = tiny_model_and_batch()
        result = compute_spectral_norm(model)
        assert isinstance(result, dict)

    def test_keys_are_linear_layers(self):
        model, _, _, _ = tiny_model_and_batch()
        result = compute_spectral_norm(model)
        linear_names = {name for name, mod in model.named_modules()
                        if isinstance(mod, nn.Linear)}
        assert set(result.keys()) == linear_names

    def test_non_negative(self):
        """Largest singular value must be ≥ 0."""
        model, _, _, _ = tiny_model_and_batch()
        result = compute_spectral_norm(model)
        for name, sigma in result.items():
            if not math.isnan(sigma):
                assert sigma >= 0.0, f"{name}: sigma={sigma} < 0"

    def test_rank_one_spectral_norm(self):
        """σ_max of a rank-1 matrix u·v^T equals ||u||·||v|| / ||u||·||v||... = ||w||₂ norm."""
        # For a rank-1 matrix W = u·v^T, σ_max = ||u||_2 · ||v||_2.
        model = nn.Linear(4, 4, bias=False)
        u = torch.randn(4)
        v = torch.randn(4)
        with torch.no_grad():
            model.weight.data = u.unsqueeze(1) * v.unsqueeze(0)
        expected = u.norm().item() * v.norm().item()
        result = compute_spectral_norm(model)
        sigma = list(result.values())[0]
        assert abs(sigma - expected) < 1e-3, f"Expected σ_max≈{expected:.4f}, got {sigma:.4f}"

    def test_empty_model_returns_empty(self):
        """A model with no Linear layers returns empty dict."""
        model = nn.Sequential(nn.ReLU())
        result = compute_spectral_norm(model)
        assert result == {}

    def test_values_finite(self):
        """Spectral norms should be finite for a randomly initialised model."""
        model, _, _, _ = tiny_model_and_batch()
        result = compute_spectral_norm(model)
        for name, sigma in result.items():
            assert math.isfinite(sigma), f"{name}: sigma={sigma} is not finite"


# ---------------------------------------------------------------------------
# Optimizer state entropy
# ---------------------------------------------------------------------------

class TestOptimizerStateEntropy:
    def _adam_with_state(self):
        """Return an Adam optimizer that has taken at least one step (has state)."""
        model = nn.Linear(8, 4)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        # One forward + backward + step to populate the state buffers.
        loss = model(torch.randn(4, 8)).sum()
        loss.backward()
        opt.step()
        return opt

    def test_returns_float(self):
        opt = self._adam_with_state()
        result = compute_optimizer_state_entropy(opt)
        assert isinstance(result, float)

    def test_non_negative(self):
        """Entropy is always ≥ 0."""
        opt = self._adam_with_state()
        result = compute_optimizer_state_entropy(opt)
        if not math.isnan(result):
            assert result >= 0.0

    def test_finite_for_adam(self):
        """Adam buffers should give a finite entropy."""
        opt = self._adam_with_state()
        result = compute_optimizer_state_entropy(opt)
        assert math.isfinite(result)

    def test_no_state_returns_nan(self):
        """An optimizer that has never stepped has no state → NaN."""
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters())
        result = compute_optimizer_state_entropy(opt)
        assert math.isnan(result)

    def test_entropy_decreases_with_concentrated_state(self):
        """Entropy of nearly uniform buffers > entropy of a single spike."""
        # Two 1-parameter models: one near-uniform state, one concentrated.
        model_uniform = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            model_uniform.weight.fill_(1.0)
        opt_uniform = torch.optim.Adam(model_uniform.parameters(), lr=1e-3)
        loss = model_uniform(torch.ones(4, 8)).sum()
        loss.backward()
        opt_uniform.step()
        ent_uniform = compute_optimizer_state_entropy(opt_uniform)

        model_spike = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            model_spike.weight.zero_()
            model_spike.weight[0, 0] = 1e6   # one very large gradient component
        opt_spike = torch.optim.Adam(model_spike.parameters(), lr=1e-3)
        g = torch.zeros_like(model_spike.weight)
        g[0, 0] = 1e6
        loss_spike = (model_spike.weight * g).sum()
        loss_spike.backward()
        opt_spike.step()
        ent_spike = compute_optimizer_state_entropy(opt_spike)

        # Uniform buffers should have higher entropy than a single spike.
        assert ent_uniform > ent_spike
