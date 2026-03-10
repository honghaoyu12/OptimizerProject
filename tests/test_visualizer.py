"""Tests for plot_lr_sensitivity() in visualizer.py."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any pyplot import

import pytest
from visualizer import plot_lr_sensitivity


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
