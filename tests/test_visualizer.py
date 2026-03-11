"""Tests for visualizer.py plot utilities."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any pyplot import

import pytest
from visualizer import plot_lr_sensitivity, plot_grad_flow_heatmap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lr_results(lrs=(0.1, 0.01), opt_name="Adam"):
    """Minimal results dict with config_lr set, as run_benchmark() would produce."""
    results = {}
    for lr in lrs:
        series = f"{opt_name} (lr={lr:g})"
        results[("MNIST", "MLP", series)] = {
            "test_acc":  [0.80, 0.85, 0.90],
            "train_acc": [0.85, 0.90, 0.95],
            "config_lr": lr,
        }
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_runs_without_error(tmp_path):
    """plot_lr_sensitivity() completes without raising on valid data."""
    results = _make_lr_results()
    plot_lr_sensitivity(results, save_path=str(tmp_path / "lr_sens.png"))


def test_saves_file(tmp_path):
    """Figure file is created at the requested path."""
    results = _make_lr_results()
    out = tmp_path / "lr_sens.png"
    plot_lr_sensitivity(results, save_path=str(out))
    assert out.exists()


def test_skips_when_no_config_lr(tmp_path, capsys):
    """Results without config_lr → skip message printed, no file written."""
    results = {("MNIST", "MLP", "Adam"): {"test_acc": [0.9]}}
    plot_lr_sensitivity(results, save_path=str(tmp_path / "lr_sens.png"))
    captured = capsys.readouterr()
    assert "skipped" in captured.out.lower()
    assert not (tmp_path / "lr_sens.png").exists()


def test_multi_optimizer(tmp_path):
    """Multiple optimizers in results → file saved without error."""
    results = _make_lr_results(lrs=(0.1, 0.01), opt_name="Adam")
    results.update(_make_lr_results(lrs=(0.1, 0.01), opt_name="SGD"))
    out = tmp_path / "lr_sens.png"
    plot_lr_sensitivity(results, save_path=str(out))
    assert out.exists()


def test_std_error_bands_no_error(tmp_path):
    """test_acc_std present → fill_between bands drawn without error."""
    results = _make_lr_results()
    for hist in results.values():
        hist["test_acc_std"] = [0.01, 0.01, 0.01]
    out = tmp_path / "lr_sens.png"
    plot_lr_sensitivity(results, save_path=str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# plot_grad_flow_heatmap
# ---------------------------------------------------------------------------

def _make_grad_results(series_names=("Adam",), n_epochs=3, layers=("fc1", "fc2")):
    """Minimal results dict with grad_norms, as run_benchmark() would produce."""
    results = {}
    for series in series_names:
        results[("MNIST", "MLP", series)] = {
            "test_acc": [0.8 + 0.05 * e for e in range(n_epochs)],
            "grad_norms": {ln: [0.1 * (e + 1) for e in range(n_epochs)] for ln in layers},
        }
    return results


class TestGradFlowHeatmap:
    def test_runs_without_error(self, tmp_path):
        """plot_grad_flow_heatmap() completes without raising on valid data."""
        results = _make_grad_results()
        plot_grad_flow_heatmap(results, save_path=str(tmp_path / "grad.png"))

    def test_saves_file(self, tmp_path):
        """Figure file is created at the requested path."""
        results = _make_grad_results()
        out = tmp_path / "grad.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        assert out.exists()

    def test_skips_when_no_grad_norms(self, tmp_path, capsys):
        """Results without grad_norms → skip message printed, no file written."""
        results = {("MNIST", "MLP", "Adam"): {"test_acc": [0.9]}}
        out = tmp_path / "grad.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()
        assert not out.exists()

    def test_multi_optimizer(self, tmp_path):
        """Multiple optimizer series → file saved without error."""
        results = _make_grad_results(series_names=("Adam", "SGD", "Lion"))
        out = tmp_path / "grad_multi.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        assert out.exists()

    def test_multi_layer(self, tmp_path):
        """Four layers → file saved without error; ordering handled correctly."""
        results = _make_grad_results(layers=("fc1", "fc2", "fc3", "fc4"), n_epochs=5)
        out = tmp_path / "grad_layers.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        assert out.exists()

    def test_zero_gradient_values(self, tmp_path):
        """Zero gradient values (vanishing grads) handled without error."""
        results = _make_grad_results()
        for hist in results.values():
            hist["grad_norms"]["fc1"] = [0.0, 0.0, 0.0]
        out = tmp_path / "grad_zero.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        assert out.exists()
