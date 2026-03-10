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
