"""Tests for benchmark.py — LR sweep feature."""

import torch
import pytest

from benchmark import run_benchmark, DATASET_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DATASET_NAMES  = ["Ill-Conditioned (synth)"]
MODEL_NAMES    = ["MLP"]
OPT_NAMES      = ["SGD"]
HIDDEN_SIZES   = [32]
EPOCHS         = 1
BATCH_SIZE     = 64
DATA_DIR       = "./data"
DEVICE         = torch.device("cpu")


def _run(**kwargs):
    """Thin wrapper with sensible defaults — only override what the test needs."""
    defaults = dict(
        dataset_names=DATASET_NAMES,
        model_names=MODEL_NAMES,
        optimizer_names=OPT_NAMES,
        hidden_sizes=HIDDEN_SIZES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        data_dir=DATA_DIR,
        device=DEVICE,
        lrs=None,
        logger=None,
    )
    defaults.update(kwargs)
    return run_benchmark(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lrs_none_uses_per_optimizer_default():
    """lrs=None → series key is the bare optimizer name (no lr= suffix)."""
    results = _run(lrs=None)
    assert len(results) == 1
    key = next(iter(results))
    ds_name, mdl_name, series_name = key
    assert series_name == "SGD"


def test_single_lr_no_suffix():
    """lrs=[0.1] (single value) → series key has no lr= suffix."""
    results = _run(lrs=[0.1])
    assert len(results) == 1
    key = next(iter(results))
    ds_name, mdl_name, series_name = key
    assert series_name == "SGD"


def test_lrs_sweep_creates_two_series():
    """lrs=[0.1, 0.01] → two distinct result entries."""
    results = _run(lrs=[0.1, 0.01])
    assert len(results) == 2


def test_lr_sweep_series_names_contain_lr():
    """LR sweep series names include 'lr=<value>'."""
    results = _run(lrs=[0.1, 0.01])
    series_names = [key[2] for key in results]
    assert any("lr=0.1" in n for n in series_names)
    assert any("lr=0.01" in n for n in series_names)


def test_lr_and_wd_sweep_combined_suffix():
    """lrs=[0.1, 0.01] + weight_decays=[0.0, 1e-4] → 4 series, all with both lr= and wd=."""
    results = _run(lrs=[0.1, 0.01], weight_decays=[0.0, 1e-4])
    assert len(results) == 4
    series_names = [key[2] for key in results]
    for name in series_names:
        assert "lr=" in name
        assert "wd=" in name


def test_wd_only_sweep_no_lr_suffix():
    """lrs=None + weight_decays=[0.0, 1e-4] → 2 series, none with lr= in the name."""
    results = _run(lrs=None, weight_decays=[0.0, 1e-4])
    assert len(results) == 2
    series_names = [key[2] for key in results]
    for name in series_names:
        assert "lr=" not in name
