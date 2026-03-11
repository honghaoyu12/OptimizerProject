"""Tests for benchmark.py — LR sweep feature."""

import json
import torch
import pytest

from benchmark import (
    run_benchmark, DATASET_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY,
    _save_results, _load_results,
)


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


def test_num_seeds_single_result_per_combo():
    """num_seeds=2 still produces 1 entry per (ds, model, series) in results."""
    results = _run(num_seeds=2, seed=0)
    assert len(results) == 1


def test_num_seeds_adds_std_keys():
    """Aggregated history contains 'test_acc_std' and 'train_loss_std'."""
    results = _run(num_seeds=2, seed=0)
    hist = next(iter(results.values()))
    assert "test_acc_std" in hist
    assert "train_loss_std" in hist


def test_target_acc_low_populates_convergence_epoch():
    """target_acc=0.0 guarantees epoch 1 is recorded for every run."""
    results = _run(target_acc=0.0)
    hist = next(iter(results.values()))
    assert hist.get("target_accuracy_epoch") is not None


def test_target_acc_high_gives_none():
    """target_acc=1.0 (impossible) gives None convergence epoch."""
    results = _run(target_acc=1.0)
    hist = next(iter(results.values()))
    assert hist.get("target_accuracy_epoch") is None


def test_save_results_creates_file(tmp_path):
    """_save_results() writes a JSON file that exists on disk."""
    results = _run()
    save_path = str(tmp_path / "results.json")
    _save_results(results, save_path)
    assert (tmp_path / "results.json").exists()


def test_save_results_valid_json(tmp_path):
    """The written file is valid JSON (no tuple keys that would break json.load)."""
    results = _run()
    save_path = str(tmp_path / "results.json")
    _save_results(results, save_path)
    with open(save_path) as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert len(data) == len(results)


def test_load_results_round_trip(tmp_path):
    """save → load preserves tuple keys and float list values."""
    results = _run()
    save_path = str(tmp_path / "results.json")
    _save_results(results, save_path)
    loaded = _load_results(save_path)

    assert set(loaded.keys()) == set(results.keys())
    for key in results:
        orig_acc  = results[key]["test_acc"]
        loaded_acc = loaded[key]["test_acc"]
        assert len(orig_acc) == len(loaded_acc)
        assert all(abs(a - b) < 1e-9 for a, b in zip(orig_acc, loaded_acc))


def test_load_results_missing_file_raises():
    """_load_results() raises FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError):
        _load_results("/nonexistent/path/results.json")


def test_save_results_creates_parent_dirs(tmp_path):
    """_save_results() creates nested parent directories automatically."""
    save_path = str(tmp_path / "nested" / "dir" / "results.json")
    results = _run()
    _save_results(results, save_path)
    import os
    assert os.path.exists(save_path)


def test_num_seeds_one_no_std_keys():
    """num_seeds=1 produces no aggregation '_std' keys (identical to current behaviour).

    Note: ``grad_norm_std`` is a native per-epoch key from ``train_one_epoch``
    (within-epoch per-batch gradient-norm std) and is therefore excluded from
    the check.
    """
    results = _run(num_seeds=1)
    hist = next(iter(results.values()))
    # Exclude the pre-existing native grad_norm_std key
    agg_std_keys = [k for k in hist if k.endswith("_std") and k != "grad_norm_std"]
    assert agg_std_keys == []
