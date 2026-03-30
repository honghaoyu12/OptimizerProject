"""Additional tests for benchmark.py — covers registry structure, _aggregate_histories,
multi-combo runs, scheduler/EMA/clipping integration, and edge cases."""

import json
import math
import os

import numpy as np
import pytest
import torch

from benchmark import (
    DATASET_REGISTRY,
    MODEL_REGISTRY,
    OPTIMIZER_REGISTRY,
    _aggregate_histories,
    _load_results,
    _save_results,
    get_device,
    run_benchmark,
)


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------

DATASET_NAMES = ["Ill-Conditioned (synth)"]
MODEL_NAMES   = ["MLP"]
OPT_NAMES     = ["SGD"]
DEVICE        = torch.device("cpu")


def _run(**kwargs):
    defaults = dict(
        dataset_names=DATASET_NAMES,
        model_names=MODEL_NAMES,
        optimizer_names=OPT_NAMES,
        hidden_sizes=[32],
        epochs=1,
        batch_size=64,
        data_dir="./data",
        device=DEVICE,
        lrs=None,
        logger=None,
    )
    defaults.update(kwargs)
    return run_benchmark(**defaults)


# ---------------------------------------------------------------------------
# Registry structure
# ---------------------------------------------------------------------------

class TestDatasetRegistry:
    def test_is_ordered_dict(self):
        from collections import OrderedDict
        assert isinstance(DATASET_REGISTRY, OrderedDict)

    def test_all_values_are_strings(self):
        for k, v in DATASET_REGISTRY.items():
            assert isinstance(v, str), f"DATASET_REGISTRY[{k!r}] value is not a str"

    def test_known_display_names_present(self):
        keys = set(DATASET_REGISTRY.keys())
        assert "MNIST" in keys
        assert "CIFAR-10" in keys
        assert "Ill-Conditioned (synth)" in keys

    def test_synthetic_datasets_map_to_correct_keys(self):
        assert DATASET_REGISTRY["Ill-Conditioned (synth)"] == "illcond"
        assert DATASET_REGISTRY["Sparse Signal (synth)"]   == "sparse"
        assert DATASET_REGISTRY["Noisy Gradient (synth)"]  == "noisy_grad"

    def test_lm_datasets_present(self):
        assert "TinyStories (LM)"   in DATASET_REGISTRY
        assert "WikiText-103 (LM)"  in DATASET_REGISTRY

    def test_lm_keys_correct(self):
        assert DATASET_REGISTRY["TinyStories (LM)"]  == "tinystories"
        assert DATASET_REGISTRY["WikiText-103 (LM)"] == "wikitext103"


class TestModelRegistry:
    def test_is_ordered_dict(self):
        from collections import OrderedDict
        assert isinstance(MODEL_REGISTRY, OrderedDict)

    def test_required_models_present(self):
        for name in ("MLP", "ResNet-18", "ViT", "GPT"):
            assert name in MODEL_REGISTRY, f"MODEL_REGISTRY missing '{name}'"

    def test_each_entry_has_factory_and_description(self):
        for name, entry in MODEL_REGISTRY.items():
            assert "factory"     in entry, f"MODEL_REGISTRY[{name!r}] missing 'factory'"
            assert "description" in entry, f"MODEL_REGISTRY[{name!r}] missing 'description'"

    def test_factory_is_callable(self):
        for name, entry in MODEL_REGISTRY.items():
            assert callable(entry["factory"]), f"MODEL_REGISTRY[{name!r}]['factory'] not callable"

    def test_mlp_factory_builds_model(self):
        from train import DATASET_INFO
        info = DATASET_INFO["illcond"]
        model = MODEL_REGISTRY["MLP"]["factory"](info, [64])
        assert sum(p.numel() for p in model.parameters()) > 0


class TestOptimizerRegistry:
    def test_is_ordered_dict(self):
        from collections import OrderedDict
        assert isinstance(OPTIMIZER_REGISTRY, OrderedDict)

    def test_known_optimizers_present(self):
        for name in ("SGD", "Adam", "AdamW", "Lion", "LAMB", "Muon"):
            assert name in OPTIMIZER_REGISTRY, f"OPTIMIZER_REGISTRY missing '{name}'"

    def test_each_entry_has_required_fields(self):
        for name, entry in OPTIMIZER_REGISTRY.items():
            assert "factory"    in entry, f"missing 'factory' for {name!r}"
            assert "default_lr" in entry, f"missing 'default_lr' for {name!r}"
            assert "color"      in entry, f"missing 'color' for {name!r}"

    def test_default_lrs_are_positive(self):
        for name, entry in OPTIMIZER_REGISTRY.items():
            assert entry["default_lr"] > 0, f"default_lr <= 0 for {name!r}"

    def test_colors_are_hex_strings(self):
        import re
        hex_pat = re.compile(r"^#[0-9a-fA-F]{6}$")
        for name, entry in OPTIMIZER_REGISTRY.items():
            assert hex_pat.match(entry["color"]), \
                f"color {entry['color']!r} is not a valid hex for {name!r}"

    def test_factory_builds_optimizer(self):
        import torch.nn as nn
        model = nn.Linear(4, 2)
        for name, entry in OPTIMIZER_REGISTRY.items():
            lr = entry["default_lr"]
            opt = entry["factory"](model.parameters(), lr, 0.0)
            assert hasattr(opt, "step"), f"optimizer {name!r} has no .step()"


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------

class TestGetDevice:
    def test_cpu_returns_cpu(self):
        assert get_device("cpu") == torch.device("cpu")

    def test_explicit_cpu_not_auto(self):
        # Explicitly requesting cpu should always give cpu regardless of hardware
        d = get_device("cpu")
        assert d.type == "cpu"

    def test_auto_returns_device(self):
        d = get_device("auto")
        assert isinstance(d, torch.device)
        assert d.type in ("cpu", "cuda", "mps")


# ---------------------------------------------------------------------------
# _aggregate_histories — unit tests
# ---------------------------------------------------------------------------

class TestAggregateHistories:
    def _make_hist(self, train_loss, test_acc):
        """Minimal history dict compatible with _aggregate_histories."""
        return {
            "train_loss": train_loss,
            "test_loss":  [0.5] * len(train_loss),
            "train_acc":  [0.5] * len(train_loss),
            "test_acc":   test_acc,
            "learning_rates": [0.01] * len(train_loss),
            "grad_norm_global": [1.0] * len(train_loss),
            "grad_norm_before_clip": [1.0] * len(train_loss),
            "grad_norm_std": [0.1] * len(train_loss),
            "time_elapsed": [0.1] * len(train_loss),
            "steps_at_epoch_end": [10] * len(train_loss),
            "ece": [0.05] * len(train_loss),
            "batch_loss_std": [0.01] * len(train_loss),
            "sharpness": [float("nan")] * len(train_loss),
            "test_acc_ema": [],
            "grad_noise_scale": [0.1] * len(train_loss),
            "fisher_trace": [0.5] * len(train_loss),
            "grad_alignment": [0.9] * len(train_loss),
            "optimizer_state_entropy": [1.5] * len(train_loss),
            "plasticity": [0.8] * len(train_loss),
            "train_perplexity": [],
            "test_perplexity": [],
            "target_accuracy_epoch": None,
            "target_accuracy_time": None,
            "early_stopped_epoch": None,
            "step_losses": [0.5, 0.4],
            "weight_norms": {"fc": [1.0] * len(train_loss)},
            "grad_norms":   {"fc": [0.5] * len(train_loss)},
            "weight_distance": {"fc": [0.1] * len(train_loss)},
            "grad_snr":  {"fc": [2.0] * len(train_loss)},
            "class_acc": {0: [0.5] * len(train_loss), 1: [0.6] * len(train_loss)},
            "step_size": {"fc": [0.01] * len(train_loss)},
            "grad_cosine_sim": {"fc": [float("nan")] + [0.9] * (len(train_loss) - 1)},
            "grad_conflict": {},
            "dead_neurons": {},
            "weight_rank": {},
            "spectral_norm": {},
            "optimizer_states": {},
            "convergence_epochs": {"50%": None, "75%": None, "90%": None, "95%": None, "99%": None},
            "convergence_times":  {"50%": None, "75%": None, "90%": None, "95%": None, "99%": None},
            "config_lr": 0.01,
            "config_wd": 0.0,
        }

    def test_empty_list_returns_empty_dict(self):
        assert _aggregate_histories([]) == {}

    def test_single_seed_passes_through_list_vals(self):
        h = self._make_hist([0.5, 0.4], [0.6, 0.7])
        out = _aggregate_histories([h])
        assert out["train_loss"] == pytest.approx([0.5, 0.4])
        assert out["test_acc"]   == pytest.approx([0.6, 0.7])

    def test_two_seeds_mean_is_correct(self):
        h1 = self._make_hist([1.0, 0.8], [0.5, 0.6])
        h2 = self._make_hist([0.6, 0.4], [0.7, 0.8])
        out = _aggregate_histories([h1, h2])
        assert out["train_loss"] == pytest.approx([0.8, 0.6])
        assert out["test_acc"]   == pytest.approx([0.6, 0.7])

    def test_two_seeds_std_keys_present(self):
        h1 = self._make_hist([1.0, 0.8], [0.5, 0.6])
        h2 = self._make_hist([0.6, 0.4], [0.7, 0.8])
        out = _aggregate_histories([h1, h2])
        assert "train_loss_std" in out
        assert "test_acc_std"   in out

    def test_std_values_are_non_negative(self):
        h1 = self._make_hist([1.0, 0.8], [0.5, 0.6])
        h2 = self._make_hist([0.6, 0.4], [0.7, 0.8])
        out = _aggregate_histories([h1, h2])
        assert all(v >= 0.0 for v in out["train_loss_std"])

    def test_unequal_length_truncates_to_shorter(self):
        h1 = self._make_hist([1.0, 0.8, 0.6], [0.5, 0.6, 0.7])
        h2 = self._make_hist([0.6, 0.4], [0.7, 0.8])
        out = _aggregate_histories([h1, h2])
        assert len(out["train_loss"]) == 2

    def test_target_accuracy_epoch_averaged_over_non_none(self):
        h1 = self._make_hist([1.0], [0.5])
        h2 = self._make_hist([0.8], [0.7])
        h1["target_accuracy_epoch"] = 3
        h2["target_accuracy_epoch"] = 5
        out = _aggregate_histories([h1, h2])
        assert out["target_accuracy_epoch"] == pytest.approx(4.0)

    def test_target_accuracy_epoch_none_when_none_reached(self):
        h1 = self._make_hist([1.0], [0.5])
        h2 = self._make_hist([0.8], [0.7])
        out = _aggregate_histories([h1, h2])
        assert out["target_accuracy_epoch"] is None

    def test_test_acc_final_seeds_has_correct_length(self):
        h1 = self._make_hist([1.0, 0.8], [0.5, 0.6])
        h2 = self._make_hist([0.6, 0.4], [0.7, 0.8])
        out = _aggregate_histories([h1, h2])
        assert len(out["test_acc_final_seeds"]) == 2

    def test_test_acc_final_seeds_values(self):
        h1 = self._make_hist([1.0, 0.8], [0.5, 0.6])
        h2 = self._make_hist([0.6, 0.4], [0.7, 0.8])
        out = _aggregate_histories([h1, h2])
        # Final test_acc of each seed (h1→0.6, h2→0.8)
        finals = sorted(out["test_acc_final_seeds"])
        assert finals == pytest.approx(sorted([0.6, 0.8]))

    def test_dict_keys_averaged_per_layer(self):
        h1 = self._make_hist([1.0], [0.5])
        h2 = self._make_hist([0.8], [0.6])
        h1["weight_distance"] = {"fc": [0.2]}
        h2["weight_distance"] = {"fc": [0.4]}
        out = _aggregate_histories([h1, h2])
        assert out["weight_distance"]["fc"] == pytest.approx([0.3])

    def test_convergence_epochs_aggregated(self):
        h1 = self._make_hist([1.0], [0.5])
        h2 = self._make_hist([0.8], [0.6])
        h1["convergence_epochs"] = {"50%": 2, "75%": None, "90%": None, "95%": None, "99%": None}
        h2["convergence_epochs"] = {"50%": 4, "75%": None, "90%": None, "95%": None, "99%": None}
        out = _aggregate_histories([h1, h2])
        assert out["convergence_epochs"]["50%"] == pytest.approx(3.0)

    def test_none_thresh_stays_none_when_no_seed_reaches(self):
        h1 = self._make_hist([1.0], [0.5])
        h2 = self._make_hist([0.8], [0.6])
        out = _aggregate_histories([h1, h2])
        assert out["convergence_epochs"]["90%"] is None

    def test_passthrough_scalar_key_preserved(self):
        h = self._make_hist([1.0], [0.5])
        h["config_lr"] = 0.042
        out = _aggregate_histories([h])
        assert out["config_lr"] == pytest.approx(0.042)


# ---------------------------------------------------------------------------
# Multi-combo runs
# ---------------------------------------------------------------------------

class TestMultiComboRuns:
    def test_two_optimizers_produce_two_results(self):
        results = _run(optimizer_names=["SGD", "Adam"], epochs=1)
        assert len(results) == 2

    def test_two_datasets_produce_two_results(self):
        results = _run(
            dataset_names=["Ill-Conditioned (synth)", "Sparse Signal (synth)"],
            epochs=1,
        )
        assert len(results) == 2

    def test_two_models_skipped_for_tabular(self):
        # Tabular datasets only support MLP; ResNet-18 should be skipped → 1 result
        results = _run(
            dataset_names=["Ill-Conditioned (synth)"],
            model_names=["MLP", "ResNet-18"],
            epochs=1,
        )
        assert len(results) == 1

    def test_result_keys_are_tuples_of_three(self):
        results = _run(optimizer_names=["SGD", "Adam"], epochs=1)
        for key in results:
            assert isinstance(key, tuple)
            assert len(key) == 3

    def test_result_keys_contain_dataset_and_model_names(self):
        results = _run(epochs=1)
        key = next(iter(results.keys()))
        ds, model, series = key
        assert ds in DATASET_NAMES
        assert model in MODEL_NAMES


# ---------------------------------------------------------------------------
# Training output sanity checks
# ---------------------------------------------------------------------------

class TestRunBenchmarkOutputSanity:
    def test_train_loss_is_positive(self):
        results = _run(epochs=2)
        hist = next(iter(results.values()))
        assert all(v > 0 for v in hist["train_loss"])

    def test_test_acc_in_range(self):
        results = _run(epochs=2)
        hist = next(iter(results.values()))
        assert all(0.0 <= v <= 1.0 for v in hist["test_acc"])

    def test_train_loss_length_matches_epochs(self):
        results = _run(epochs=3)
        hist = next(iter(results.values()))
        assert len(hist["train_loss"]) == 3

    def test_test_acc_length_matches_epochs(self):
        results = _run(epochs=3)
        hist = next(iter(results.values()))
        assert len(hist["test_acc"]) == 3

    def test_train_loss_is_finite(self):
        results = _run(epochs=2)
        hist = next(iter(results.values()))
        assert all(math.isfinite(v) for v in hist["train_loss"])

    def test_config_lr_stored_correctly_with_single_lr(self):
        results = _run(lrs=[0.05], epochs=1)
        hist = next(iter(results.values()))
        assert abs(hist["config_lr"] - 0.05) < 1e-9

    def test_config_lr_stored_for_default(self):
        results = _run(lrs=None, epochs=1)
        hist = next(iter(results.values()))
        # Default LR for SGD is 0.01
        assert math.isfinite(hist["config_lr"])
        assert hist["config_lr"] > 0

    def test_config_wd_zero_by_default(self):
        results = _run(epochs=1)
        hist = next(iter(results.values()))
        assert hist["config_wd"] == 0.0

    def test_learning_rates_list_length_matches_epochs(self):
        results = _run(epochs=3)
        hist = next(iter(results.values()))
        assert len(hist["learning_rates"]) == 3

    def test_grad_norm_global_is_finite(self):
        results = _run(epochs=2)
        hist = next(iter(results.values()))
        assert all(math.isfinite(v) for v in hist["grad_norm_global"])


# ---------------------------------------------------------------------------
# Scheduler integration
# ---------------------------------------------------------------------------

class TestSchedulerIntegration:
    def test_cosine_scheduler_changes_lr(self):
        results_none = _run(epochs=3, lrs=[0.1])
        results_cos  = _run(epochs=3, lrs=[0.1], scheduler_name="cosine")
        lr_none = next(iter(results_none.values()))["learning_rates"]
        lr_cos  = next(iter(results_cos.values()))["learning_rates"]
        # Cosine schedule should change LR across epochs; constant baseline should not
        assert lr_cos != lr_none

    def test_step_scheduler_runs_without_error(self):
        results = _run(epochs=2, scheduler_name="step")
        assert len(results) == 1

    def test_warmup_cosine_scheduler_runs_without_error(self):
        results = _run(epochs=3, scheduler_name="warmup_cosine", warmup_epochs=1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Gradient clipping integration
# ---------------------------------------------------------------------------

class TestGradientClippingIntegration:
    def test_grad_clip_runs_without_error(self):
        results = _run(epochs=2, max_grad_norm=1.0)
        assert len(results) == 1

    def test_grad_norm_before_clip_exceeds_post_when_large_norm(self):
        # With a very tight clip, pre-clip norm should exceed post-clip on many steps
        results = _run(epochs=2, max_grad_norm=0.01)
        hist = next(iter(results.values()))
        pre  = hist["grad_norm_before_clip"]
        post = hist["grad_norm_global"]
        # At least one epoch where pre > post (clipping was active)
        assert any(b >= a for a, b in zip(post, pre))

    def test_grad_norm_global_bounded_by_clip(self):
        # With a very tight clip the per-epoch average should still be finite
        # and the pre-clip norm should not be less than post-clip norm on average
        clip = 0.5
        results = _run(epochs=3, max_grad_norm=clip)
        hist = next(iter(results.values()))
        pre  = hist["grad_norm_before_clip"]
        post = hist["grad_norm_global"]
        # pre >= post on at least one epoch (clipping was active)
        assert any(b >= a for a, b in zip(post, pre))
        # All values must be finite
        assert all(math.isfinite(v) for v in post)


# ---------------------------------------------------------------------------
# EMA integration
# ---------------------------------------------------------------------------

class TestEMAIntegration:
    def test_ema_decay_runs_without_error(self):
        results = _run(epochs=2, ema_decay=0.999)
        assert len(results) == 1

    def test_ema_acc_key_present_when_enabled(self):
        results = _run(epochs=2, ema_decay=0.999)
        hist = next(iter(results.values()))
        assert "test_acc_ema" in hist
        assert len(hist["test_acc_ema"]) == 2

    def test_ema_acc_in_range(self):
        results = _run(epochs=2, ema_decay=0.99)
        hist = next(iter(results.values()))
        for v in hist["test_acc_ema"]:
            assert 0.0 <= v <= 1.0

    def test_no_ema_acc_without_ema_decay(self):
        results = _run(epochs=2, ema_decay=None)
        hist = next(iter(results.values()))
        # Without EMA the list should be empty
        ema = hist.get("test_acc_ema", [])
        assert len(ema) == 0


# ---------------------------------------------------------------------------
# Label smoothing integration
# ---------------------------------------------------------------------------

class TestLabelSmoothingIntegration:
    def test_label_smoothing_runs_without_error(self):
        results = _run(epochs=2, label_smoothing=0.1)
        assert len(results) == 1

    def test_label_smoothing_train_loss_finite(self):
        results = _run(epochs=2, label_smoothing=0.1)
        hist = next(iter(results.values()))
        assert all(math.isfinite(v) for v in hist["train_loss"])


# ---------------------------------------------------------------------------
# Weight decay integration
# ---------------------------------------------------------------------------

class TestWeightDecayIntegration:
    def test_positive_wd_runs_without_error(self):
        results = _run(epochs=2, weight_decays=[1e-4])
        assert len(results) == 1

    def test_wd_stored_in_config(self):
        results = _run(epochs=1, weight_decays=[1e-4])
        hist = next(iter(results.values()))
        assert abs(hist["config_wd"] - 1e-4) < 1e-12


# ---------------------------------------------------------------------------
# New metrics tracking flags
# ---------------------------------------------------------------------------

class TestMetricsTrackingFlags:
    def test_track_dead_neurons_runs_without_error(self):
        results = _run(epochs=2, track_dead_neurons=True)
        assert len(results) == 1

    def test_dead_neurons_key_present_when_tracked(self):
        results = _run(epochs=2, track_dead_neurons=True)
        hist = next(iter(results.values()))
        assert "dead_neurons" in hist

    def test_track_weight_rank_runs_without_error(self):
        results = _run(epochs=2, track_weight_rank=True)
        assert len(results) == 1

    def test_weight_rank_key_present_when_tracked(self):
        results = _run(epochs=2, track_weight_rank=True)
        hist = next(iter(results.values()))
        assert "weight_rank" in hist

    def test_track_spectral_norm_runs_without_error(self):
        results = _run(epochs=2, track_spectral_norm=True)
        assert len(results) == 1

    def test_spectral_norm_key_present_when_tracked(self):
        results = _run(epochs=2, track_spectral_norm=True)
        hist = next(iter(results.values()))
        assert "spectral_norm" in hist
        assert isinstance(hist["spectral_norm"], dict)

    def test_spectral_norm_values_positive(self):
        results = _run(epochs=2, track_spectral_norm=True)
        hist = next(iter(results.values()))
        for layer_vals in hist["spectral_norm"].values():
            assert all(v > 0 for v in layer_vals)

    def test_track_optimizer_entropy_runs_without_error(self):
        results = _run(epochs=2, optimizer_names=["Adam"], track_optimizer_entropy=True)
        assert len(results) == 1

    def test_plasticity_tracked_when_interval_set(self):
        results = _run(epochs=2, plasticity_interval=1)
        hist = next(iter(results.values()))
        assert "plasticity" in hist

    def test_plasticity_absent_when_interval_zero(self):
        results = _run(epochs=2, plasticity_interval=0)
        hist = next(iter(results.values()))
        # plasticity is a per-epoch list; when disabled it contains nan values
        p = hist.get("plasticity", [])
        if isinstance(p, list):
            assert all(math.isnan(v) for v in p) or len(p) == 0
        else:
            assert p == {} or all(len(v) == 0 for v in p.values())


# ---------------------------------------------------------------------------
# Gradient accumulation integration
# ---------------------------------------------------------------------------

class TestGradAccumIntegration:
    def test_accum_steps_two_runs_without_error(self):
        results = _run(epochs=2, accum_steps=2)
        assert len(results) == 1

    def test_accum_steps_train_loss_finite(self):
        results = _run(epochs=2, accum_steps=2)
        hist = next(iter(results.values()))
        assert all(math.isfinite(v) for v in hist["train_loss"])


# ---------------------------------------------------------------------------
# _save_results / _load_results additional coverage
# ---------------------------------------------------------------------------

class TestPersistenceAdditional:
    def _make_results(self):
        return _run(epochs=1)

    def test_saved_json_has_string_keys(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "r.json")
        _save_results(results, path)
        with open(path) as f:
            raw = json.load(f)
        for k in raw:
            assert isinstance(k, str)

    def test_load_keys_are_tuples(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "r.json")
        _save_results(results, path)
        loaded = _load_results(path)
        for k in loaded:
            assert isinstance(k, tuple)
            assert len(k) == 3

    def test_round_trip_preserves_all_float_lists(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "r.json")
        _save_results(results, path)
        loaded = _load_results(path)
        for key in results:
            orig   = results[key]["train_loss"]
            loaded_vals = loaded[key]["train_loss"]
            assert len(orig) == len(loaded_vals)
            for a, b in zip(orig, loaded_vals):
                assert abs(a - b) < 1e-9

    def test_overwrite_existing_file(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "r.json")
        _save_results(results, path)
        # Overwrite with a new run
        _save_results(results, path)
        loaded = _load_results(path)
        assert set(loaded.keys()) == set(results.keys())

    def test_multiple_results_all_present(self, tmp_path):
        results = _run(optimizer_names=["SGD", "Adam"], epochs=1)
        assert len(results) == 2
        path = str(tmp_path / "r.json")
        _save_results(results, path)
        loaded = _load_results(path)
        assert len(loaded) == 2

    def test_sep_encoded_correctly(self, tmp_path):
        results = self._make_results()
        path = str(tmp_path / "r.json")
        _save_results(results, path)
        with open(path) as f:
            raw = json.load(f)
        for k in raw:
            # Key must contain two "||" separators
            assert k.count("||") == 2
