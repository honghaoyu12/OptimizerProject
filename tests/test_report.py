"""Tests for report.py — generate_report()."""

import os
import pytest

from report import generate_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_results(
    datasets=("MNIST",),
    models=("MLP",),
    optimizers=("Adam",),
    epochs=3,
):
    """Build a minimal synthetic results dict (no training required)."""
    results = {}
    for ds in datasets:
        for mdl in models:
            for opt in optimizers:
                history = {
                    "train_loss":  [0.5, 0.4, 0.3],
                    "train_acc":   [0.80, 0.85, 0.90],
                    "test_acc":    [0.75, 0.80, 0.85],
                    "time_elapsed": [1.0, 2.0, 3.0],
                    "early_stopped_epoch": None,
                }
                results[(ds, mdl, opt)] = history
    return results


_DEFAULT_CONFIG = {
    "epochs": 3,
    "batch_size": 128,
    "scheduler": "none",
    "weight_decays": [0.0],
    "lrs": None,
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_string():
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert isinstance(out, str)


def test_file_created(tmp_path):
    results = _make_results()
    report_file = tmp_path / "report.md"
    generate_report(results, _DEFAULT_CONFIG, save_path=str(report_file))
    assert report_file.exists()


def test_file_content_matches_return_value(tmp_path):
    results = _make_results()
    report_file = tmp_path / "report.md"
    returned = generate_report(results, _DEFAULT_CONFIG, save_path=str(report_file))
    written = report_file.read_text()
    assert written == returned


def test_markdown_has_header():
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert out.startswith("# Benchmark Report")


def test_markdown_has_setup_section():
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "## Setup" in out


def test_markdown_has_results_section():
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "## Results" in out


def test_markdown_has_per_optimizer_section():
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "## Per-Optimizer" in out


def test_markdown_has_observations_section():
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "## Key Observations" in out


def test_empty_save_path_skips_file(tmp_path):
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert isinstance(out, str) and len(out) > 0
    # No files should have been written to tmp_path (we never passed it)
    assert list(tmp_path.iterdir()) == []


def test_multiple_optimizers_rankings_present():
    results = _make_results(optimizers=("Adam", "SGD"))
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "Best final accuracy" in out
    assert "Fastest" in out


def test_report_convergence_columns_present():
    """Report Results table includes 'Epochs to X%' and 'Time to X%' column headers."""
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "Epochs to" in out
    assert "Time to" in out


def test_report_convergence_dash_when_none():
    """target_accuracy_epoch absent from history → '—' appears in the report."""
    results = _make_results()
    # _make_results() does not set target_accuracy_epoch, so it defaults to None
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "—" in out


def test_report_shows_std_when_present():
    """history with 'test_acc_std' key → '±' appears in the report output."""
    results = _make_results()
    key = next(iter(results))
    results[key]["test_acc_std"] = [0.01, 0.01, 0.015]
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "±" in out


def test_report_lr_sensitivity_section_present_when_swept():
    """Results with config_lr and ≥2 LRs → 'LR Sensitivity' section in report."""
    results = {}
    for lr, acc in [(1e-3, 0.97), (1e-2, 0.90), (1e-1, 0.70)]:
        series = f"Adam (lr={lr:g})"
        results[("MNIST", "MLP", series)] = {
            "train_loss":   [0.3],
            "train_acc":    [0.95],
            "test_acc":     [acc],
            "time_elapsed": [2.0],
            "config_lr":    lr,
        }
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "LR Sensitivity" in out
    assert "Range" in out
    assert "Best LR" in out


def test_report_lr_sensitivity_section_absent_without_sweep():
    """Results with no config_lr → 'LR Sensitivity' section not in report."""
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "LR Sensitivity" not in out


def test_report_ema_columns_present_when_ema_data_exists():
    """history with 'test_acc_ema' key → 'EMA Acc' and 'EMA Gap' columns appear."""
    results = _make_results()
    key = next(iter(results))
    results[key]["test_acc_ema"] = [0.74, 0.79, 0.84]
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "EMA Acc" in out
    assert "EMA Gap" in out


def test_report_ema_columns_absent_without_ema_data():
    """Without 'test_acc_ema', EMA columns not in report."""
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "EMA Acc" not in out
    assert "EMA Gap" not in out


def test_report_label_smoothing_row_present():
    """config with label_smoothing key → 'Label smoothing' row appears in Setup."""
    results = _make_results()
    cfg = {**_DEFAULT_CONFIG, "label_smoothing": 0.1}
    out = generate_report(results, cfg, save_path="")
    assert "Label smoothing" in out
    assert "0.1" in out


def test_report_label_smoothing_disabled_row():
    """config with label_smoothing=0.0 → row shows '0.0 (disabled)'."""
    results = _make_results()
    out = generate_report(results, _DEFAULT_CONFIG, save_path="")
    assert "Label smoothing" in out
    assert "disabled" in out
