"""Tests for plot_from_logs.py — load_session() and reconstruct_results()."""

import json
import os

import pytest

from plot_from_logs import load_session, reconstruct_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_HISTORY = {
    "train_loss": [1.0, 0.8],
    "train_acc":  [0.5, 0.6],
    "test_loss":  [1.1, 0.9],
    "test_acc":   [0.45, 0.55],
    "time_elapsed": [1.0, 2.0],
}


def make_session_dir(tmp_path, runs: list[dict]) -> str:
    """Write a minimal run_summary.json and return the directory path."""
    session_dir = str(tmp_path)
    data = {
        "session": {
            "started":   "2026-01-01 00:00:00",
            "finished":  "2026-01-01 00:01:00",
            "elapsed_s": 60.0,
            "num_runs":  len(runs),
        },
        "runs": runs,
    }
    with open(os.path.join(session_dir, "run_summary.json"), "w") as f:
        json.dump(data, f)
    return session_dir


def sample_run(dataset="mnist", model="MLP", optimizer="Adam", wd=0.0):
    return {
        "name":    f"{dataset}_{model}_{optimizer}",
        "config":  {
            "dataset":      dataset,
            "model":        model,
            "optimizer":    optimizer,
            "weight_decay": wd,
            "lr":           0.001,
            "epochs":       2,
        },
        "summary": {
            "best_test_acc": 0.55,
            "best_test_epoch": 2,
        },
        "history": MINIMAL_HISTORY,
        "files": {
            "log":        f"{dataset}_{model}_{optimizer}.log",
            "epochs_csv": f"{dataset}_{model}_{optimizer}_epochs.csv",
        },
    }


# ---------------------------------------------------------------------------
# load_session()
# ---------------------------------------------------------------------------

class TestLoadSession:
    def test_returns_run_list(self, tmp_path):
        runs = [sample_run(), sample_run(optimizer="AdamW")]
        sd   = make_session_dir(tmp_path, runs)
        loaded = load_session(sd)
        assert len(loaded) == 2

    def test_missing_json_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_session(str(tmp_path))

    def test_run_keys_present(self, tmp_path):
        sd = make_session_dir(tmp_path, [sample_run()])
        run = load_session(sd)[0]
        for key in ("name", "config", "history"):
            assert key in run


# ---------------------------------------------------------------------------
# reconstruct_results()
# ---------------------------------------------------------------------------

class TestReconstructResults:
    def _runs(self):
        return [
            sample_run(dataset="mnist",         optimizer="Adam"),
            sample_run(dataset="mnist",         optimizer="AdamW"),
            sample_run(dataset="fashion_mnist",  optimizer="Adam"),
        ]

    def test_result_keys_are_3tuples(self):
        results, *_ = reconstruct_results(self._runs())
        for key in results:
            assert len(key) == 3
            ds_name, mdl_name, series_name = key
            assert isinstance(ds_name,    str)
            assert isinstance(mdl_name,   str)
            assert isinstance(series_name, str)

    def test_history_preserved(self):
        results, *_ = reconstruct_results(self._runs())
        # Pick one run and check its train_loss is intact
        key = ("MNIST", "MLP", "Adam")
        assert key in results
        assert results[key]["train_loss"] == MINIMAL_HISTORY["train_loss"]

    def test_dataset_key_mapped_to_display_name(self):
        _, dataset_names, *_ = reconstruct_results(self._runs())
        assert "MNIST" in dataset_names
        assert "Fashion MNIST" in dataset_names

    def test_sweep_wd_adds_suffix(self):
        runs = [
            sample_run(optimizer="Adam", wd=0.0),
            sample_run(optimizer="Adam", wd=1e-4),
        ]
        _, _, _, series_names, _ = reconstruct_results(runs)
        assert any("wd=" in n for n in series_names)

    def test_no_sweep_wd_no_suffix(self):
        _, _, _, series_names, _ = reconstruct_results(self._runs())
        assert not any("wd=" in n for n in series_names)

    def test_all_series_have_color(self):
        _, _, _, series_names, opt_colors = reconstruct_results(self._runs())
        for name in series_names:
            assert name in opt_colors
