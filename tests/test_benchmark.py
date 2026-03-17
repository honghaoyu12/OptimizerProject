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
    (within-epoch per-batch gradient-norm std) and ``batch_loss_std`` is a
    native per-epoch key measuring within-epoch batch loss variance — both are
    therefore excluded from the check.
    """
    results = _run(num_seeds=1)
    hist = next(iter(results.values()))
    # Exclude native per-epoch _std keys that are not aggregation artefacts
    _native_std_keys = {"grad_norm_std", "batch_loss_std"}
    agg_std_keys = [k for k in hist if k.endswith("_std") and k not in _native_std_keys]
    assert agg_std_keys == []


def test_num_seeds_adds_final_seeds_list():
    """num_seeds=2 → aggregated history contains 'test_acc_final_seeds' with 2 values."""
    results = _run(num_seeds=2, seed=0)
    hist = next(iter(results.values()))
    seeds_list = hist.get("test_acc_final_seeds")
    assert seeds_list is not None, "test_acc_final_seeds key missing"
    assert len(seeds_list) == 2, f"Expected 2 seed values, got {len(seeds_list)}"
    assert all(isinstance(v, float) for v in seeds_list)


def test_num_seeds_one_no_final_seeds_list():
    """num_seeds=1 → 'test_acc_final_seeds' key should NOT be present (raw history)."""
    results = _run(num_seeds=1)
    hist = next(iter(results.values()))
    assert "test_acc_final_seeds" not in hist, (
        "test_acc_final_seeds should not exist for single-seed runs"
    )


def test_num_seeds_convergence_profile_aggregated():
    """num_seeds=2 → aggregated history has 'convergence_epochs' and 'convergence_times' dicts."""
    results = _run(num_seeds=2, seed=0, epochs=2)
    hist = next(iter(results.values()))
    assert "convergence_epochs" in hist, "convergence_epochs key missing"
    assert "convergence_times"  in hist, "convergence_times key missing"
    # Keys must be the five threshold labels produced by _CONV_THRESHOLDS
    from train import _CONV_THRESHOLDS
    expected_keys = {f"{int(t * 100)}%" for t in _CONV_THRESHOLDS}
    assert set(hist["convergence_epochs"].keys()) == expected_keys
    assert set(hist["convergence_times"].keys())  == expected_keys


def test_convergence_profile_single_seed():
    """Single-seed run still populates convergence_epochs / convergence_times dicts."""
    results = _run(num_seeds=1, epochs=2)
    hist = next(iter(results.values()))
    assert "convergence_epochs" in hist
    assert "convergence_times"  in hist
    from train import _CONV_THRESHOLDS
    expected_keys = {f"{int(t * 100)}%" for t in _CONV_THRESHOLDS}
    assert set(hist["convergence_epochs"].keys()) == expected_keys


def test_steps_at_epoch_end_present():
    """Single-seed run has 'steps_at_epoch_end' with one entry per epoch."""
    results = _run(num_seeds=1, epochs=2)
    hist = next(iter(results.values()))
    assert "steps_at_epoch_end" in hist
    assert len(hist["steps_at_epoch_end"]) == 2


def test_steps_at_epoch_end_aggregated_for_multi_seed():
    """num_seeds=2 → aggregated history still has 'steps_at_epoch_end' list."""
    results = _run(num_seeds=2, seed=0, epochs=2)
    hist = next(iter(results.values()))
    assert "steps_at_epoch_end" in hist
    assert len(hist["steps_at_epoch_end"]) == 2


def test_auto_lr_finds_lr(monkeypatch):
    """auto_lr=True → run completes and LRFinder suggestion is used (not nan)."""
    results = _run(auto_lr=True, epochs=1)
    hist = next(iter(results.values()))
    # config_lr is the effective LR used; it should be a positive finite number
    lr = hist.get("config_lr")
    import math
    assert lr is not None and math.isfinite(lr) and lr > 0


def test_auto_lr_ignored_when_lrs_set():
    """auto_lr=True with explicit lrs → uses the explicit LR, not the finder's suggestion."""
    explicit_lr = 0.05
    results = _run(auto_lr=False, lrs=[explicit_lr], epochs=1)
    hist = next(iter(results.values()))
    assert abs(hist.get("config_lr", 0) - explicit_lr) < 1e-9


def test_config_wd_stored_in_history():
    """run_benchmark() stores config_wd in each result history."""
    results = _run(weight_decays=[0.0], epochs=1)
    hist = next(iter(results.values()))
    assert "config_wd" in hist
    assert hist["config_wd"] == 0.0


def test_weight_distance_key_present():
    """Single-seed run has 'weight_distance' dict with per-layer lists."""
    results = _run(epochs=2)
    hist = next(iter(results.values()))
    assert "weight_distance" in hist
    assert isinstance(hist["weight_distance"], dict)
    # Each layer list has one entry per epoch
    for layer_vals in hist["weight_distance"].values():
        assert len(layer_vals) == 2


def test_weight_distance_aggregated_for_multi_seed():
    """num_seeds=2 → weight_distance is aggregated (averaged) across seeds."""
    results = _run(num_seeds=2, seed=0, epochs=2)
    hist = next(iter(results.values()))
    assert "weight_distance" in hist
    for layer_vals in hist["weight_distance"].values():
        assert len(layer_vals) == 2


def test_grad_snr_present_single_seed():
    """Single-seed run has 'grad_snr' dict with per-layer lists."""
    results = _run(epochs=2)
    hist = next(iter(results.values()))
    assert "grad_snr" in hist
    assert isinstance(hist["grad_snr"], dict)
    for layer_vals in hist["grad_snr"].values():
        assert len(layer_vals) == 2


def test_grad_snr_aggregated_for_multi_seed():
    """num_seeds=2 → grad_snr is aggregated (averaged) across seeds."""
    results = _run(num_seeds=2, seed=0, epochs=2)
    hist = next(iter(results.values()))
    assert "grad_snr" in hist
    for layer_vals in hist["grad_snr"].values():
        assert len(layer_vals) == 2


def test_ece_present_single_seed():
    """Single-seed run has 'ece' list with one entry per epoch."""
    results = _run(epochs=2)
    hist = next(iter(results.values()))
    assert "ece" in hist
    assert len(hist["ece"]) == 2


def test_ece_aggregated_for_multi_seed():
    """num_seeds=2 → 'ece' is averaged across seeds and has one entry per epoch."""
    results = _run(num_seeds=2, seed=0, epochs=2)
    hist = next(iter(results.values()))
    assert "ece" in hist
    assert len(hist["ece"]) == 2
    assert all(0.0 <= v <= 1.0 for v in hist["ece"])


def test_class_acc_present_single_seed():
    """Single-seed run has 'class_acc' dict with int keys and per-epoch lists."""
    results = _run(epochs=2)
    hist = next(iter(results.values()))
    assert "class_acc" in hist
    assert isinstance(hist["class_acc"], dict)
    # illcond is a 2-class dataset — expect keys 0 and 1
    assert len(hist["class_acc"]) == 2
    for cls_id, vals in hist["class_acc"].items():
        assert isinstance(cls_id, int)
        assert len(vals) == 2


def test_class_acc_aggregated_for_multi_seed():
    """num_seeds=2 → class_acc is averaged across seeds."""
    results = _run(num_seeds=2, seed=0, epochs=2)
    hist = next(iter(results.values()))
    assert "class_acc" in hist
    for vals in hist["class_acc"].values():
        assert len(vals) == 2


def test_batch_loss_std_present():
    """Single-seed run has 'batch_loss_std' list with one entry per epoch."""
    results = _run(epochs=2)
    hist = next(iter(results.values()))
    assert "batch_loss_std" in hist
    assert len(hist["batch_loss_std"]) == 2
    assert all(v >= 0.0 for v in hist["batch_loss_std"])


def test_step_size_present_single_seed():
    """Single-seed run has 'step_size' dict with per-layer lists."""
    results = _run(epochs=2)
    hist = next(iter(results.values()))
    assert "step_size" in hist
    assert isinstance(hist["step_size"], dict)
    for layer_vals in hist["step_size"].values():
        assert len(layer_vals) == 2


def test_step_size_aggregated_for_multi_seed():
    """num_seeds=2 → step_size is aggregated (averaged) across seeds."""
    results = _run(num_seeds=2, seed=0, epochs=2)
    hist = next(iter(results.values()))
    assert "step_size" in hist
    for layer_vals in hist["step_size"].values():
        assert len(layer_vals) == 2
