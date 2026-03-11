"""Tests for logger.py — TrainingLogger output files and schema."""

import csv
import json
import os
import tempfile

import pytest

from logger import TrainingLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(dataset="mnist", model="mlp", optimizer="adam", **extra):
    base = {"dataset": dataset, "model": model, "optimizer": optimizer,
            "lr": 1e-3, "epochs": 3, "batch_size": 128}
    base.update(extra)
    return base


def _make_history(n_epochs=3):
    return {
        "train_loss":   [0.5 - i * 0.1 for i in range(n_epochs)],
        "train_acc":    [0.8 + i * 0.05 for i in range(n_epochs)],
        "test_loss":    [0.6 - i * 0.08 for i in range(n_epochs)],
        "test_acc":     [0.75 + i * 0.04 for i in range(n_epochs)],
        "time_elapsed": [1.0 + i * 0.5 for i in range(n_epochs)],
        "step_losses":  [0.55, 0.50, 0.48, 0.45, 0.42, 0.40],
        "learning_rates": [1e-3] * n_epochs,
        "grad_norm_global": [1.2, 1.1, 1.0],
        "grad_norm_before_clip": [1.2, 1.1, 1.0],
        "grad_norm_std": [0.1, 0.09, 0.08],
        "early_stopped_epoch": None,
        "weight_norms": {"L1": [1.0, 1.1, 1.2]},
        "grad_norms":   {"L1": [0.5, 0.4, 0.3]},
    }


@pytest.fixture
def logger_and_dir():
    """Return a TrainingLogger that writes into a temp directory."""
    with tempfile.TemporaryDirectory() as tmp:
        logger = TrainingLogger(log_dir=tmp)
        yield logger, logger.run_dir


# ---------------------------------------------------------------------------
# log_run — per-run files
# ---------------------------------------------------------------------------

class TestLogRun:
    def test_returns_log_filename(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        fname = logger.log_run(_make_config(), _make_history())
        assert fname == "mnist_mlp_adam.log"

    def test_log_file_created(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        fname = logger.log_run(_make_config(), _make_history())
        assert os.path.exists(os.path.join(run_dir, fname))

    def test_epochs_csv_created(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        assert os.path.exists(os.path.join(run_dir, "mnist_mlp_adam_epochs.csv"))

    def test_epochs_csv_is_clean(self, logger_and_dir):
        """epochs CSV must have no comment lines or section markers."""
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        path = os.path.join(run_dir, "mnist_mlp_adam_epochs.csv")
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            assert not line.startswith("#"), f"Unexpected comment line: {line!r}"
            assert not line.startswith("["), f"Unexpected section marker: {line!r}"

    def test_epochs_csv_header(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        path = os.path.join(run_dir, "mnist_mlp_adam_epochs.csv")
        with open(path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == {
                "epoch", "train_loss", "train_acc",
                "test_loss", "test_acc", "time_elapsed_s", "learning_rate",
            }

    def test_epochs_csv_learning_rate_column(self, logger_and_dir):
        """learning_rate column is present and matches history['learning_rates']."""
        logger, run_dir = logger_and_dir
        hist = _make_history(n_epochs=3)
        logger.log_run(_make_config(), hist)
        path = os.path.join(run_dir, "mnist_mlp_adam_epochs.csv")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        lr_vals = [float(r["learning_rate"]) for r in rows]
        assert lr_vals == pytest.approx(hist["learning_rates"])

    def test_epochs_csv_learning_rate_absent_when_no_lr_history(self, logger_and_dir):
        """learning_rate column is empty string when history has no learning_rates key."""
        logger, run_dir = logger_and_dir
        hist = _make_history(n_epochs=2)
        del hist["learning_rates"]
        logger.log_run(_make_config(), hist)
        path = os.path.join(run_dir, "mnist_mlp_adam_epochs.csv")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert all(r["learning_rate"] == "" for r in rows)

    def test_epochs_csv_row_count(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        n = 4
        logger.log_run(_make_config(), _make_history(n_epochs=n))
        path = os.path.join(run_dir, "mnist_mlp_adam_epochs.csv")
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == n

    def test_epochs_csv_values_finite(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        path = os.path.join(run_dir, "mnist_mlp_adam_epochs.csv")
        with open(path) as f:
            for row in csv.DictReader(f):
                assert float(row["train_loss"]) > 0
                assert 0.0 <= float(row["train_acc"]) <= 1.0
                assert float(row["time_elapsed_s"]) >= 0

    def test_log_file_has_comment_header(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        fname = logger.log_run(_make_config(), _make_history())
        path = os.path.join(run_dir, fname)
        with open(path) as f:
            content = f.read()
        assert "# Run" in content
        assert "# dataset" in content

    def test_log_file_has_batch_section(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        fname = logger.log_run(_make_config(), _make_history())
        path = os.path.join(run_dir, fname)
        with open(path) as f:
            content = f.read()
        assert "[batch-wise]" in content


# ---------------------------------------------------------------------------
# close — summary files
# ---------------------------------------------------------------------------

class TestClose:
    def test_run_summary_log_created(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        logger.close()
        assert os.path.exists(os.path.join(run_dir, "run_summary.log"))

    def test_run_summary_csv_created(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        logger.close()
        assert os.path.exists(os.path.join(run_dir, "run_summary.csv"))

    def test_run_summary_json_created(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        logger.close()
        assert os.path.exists(os.path.join(run_dir, "run_summary.json"))

    def test_close_returns_log_path(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        path = logger.close()
        assert path.endswith("run_summary.log")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# run_summary.csv schema and content
# ---------------------------------------------------------------------------

class TestSummaryCSV:
    def _load(self, logger_and_dir, configs=None, histories=None):
        logger, run_dir = logger_and_dir
        configs   = configs   or [_make_config()]
        histories = histories or [_make_history()]
        for cfg, hist in zip(configs, histories):
            logger.log_run(cfg, hist)
        logger.close()
        path = os.path.join(run_dir, "run_summary.csv")
        with open(path) as f:
            return list(csv.DictReader(f))

    def test_one_row_per_run(self, logger_and_dir):
        rows = self._load(logger_and_dir)
        assert len(rows) == 1

    def test_multiple_runs_multiple_rows(self, logger_and_dir):
        rows = self._load(
            logger_and_dir,
            configs=[_make_config("mnist"), _make_config("cifar10")],
            histories=[_make_history(), _make_history()],
        )
        assert len(rows) == 2

    def test_required_metric_columns_present(self, logger_and_dir):
        rows = self._load(logger_and_dir)
        required = {
            "run", "dataset", "model", "optimizer",
            "final_train_loss", "final_train_acc",
            "final_test_loss",  "final_test_acc",
            "best_test_acc",    "best_test_epoch",
            "total_time_s",     "log_file", "epochs_csv",
        }
        assert required.issubset(set(rows[0].keys()))

    def test_best_test_acc_geq_final(self, logger_and_dir):
        rows = self._load(logger_and_dir)
        assert float(rows[0]["best_test_acc"]) >= float(rows[0]["final_test_acc"])

    def test_best_test_epoch_in_range(self, logger_and_dir):
        rows = self._load(logger_and_dir)
        n = 3
        epoch = int(rows[0]["best_test_epoch"])
        assert 1 <= epoch <= n

    def test_config_columns_populated(self, logger_and_dir):
        rows = self._load(logger_and_dir)
        assert rows[0]["dataset"] == "mnist"
        assert rows[0]["optimizer"] == "adam"
        assert float(rows[0]["lr"]) == pytest.approx(1e-3)

    def test_epochs_csv_column_matches_filename(self, logger_and_dir):
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        logger.close()
        csv_path = os.path.join(run_dir, "run_summary.csv")
        with open(csv_path) as f:
            row = next(csv.DictReader(f))
        epochs_file = os.path.join(run_dir, row["epochs_csv"])
        assert os.path.exists(epochs_file)


# ---------------------------------------------------------------------------
# run_summary.json schema and content
# ---------------------------------------------------------------------------

class TestSummaryJSON:
    def _load(self, logger_and_dir, configs=None, histories=None):
        logger, run_dir = logger_and_dir
        configs   = configs   or [_make_config()]
        histories = histories or [_make_history()]
        for cfg, hist in zip(configs, histories):
            logger.log_run(cfg, hist)
        logger.close()
        path = os.path.join(run_dir, "run_summary.json")
        with open(path) as f:
            return json.load(f)

    def test_top_level_keys(self, logger_and_dir):
        data = self._load(logger_and_dir)
        assert set(data.keys()) == {"session", "runs"}

    def test_session_metadata_keys(self, logger_and_dir):
        data = self._load(logger_and_dir)
        assert {"started", "finished", "elapsed_s", "num_runs"} == set(data["session"].keys())

    def test_num_runs_matches(self, logger_and_dir):
        data = self._load(logger_and_dir)
        assert data["session"]["num_runs"] == len(data["runs"])

    def test_run_keys(self, logger_and_dir):
        data = self._load(logger_and_dir)
        run = data["runs"][0]
        assert {"name", "config", "summary", "history", "files"} == set(run.keys())

    def test_summary_keys(self, logger_and_dir):
        data = self._load(logger_and_dir)
        summary = data["runs"][0]["summary"]
        assert {"final_train_loss", "final_train_acc", "final_test_loss", "final_test_acc",
                "best_test_acc", "best_test_epoch", "total_time_s"} == set(summary.keys())

    def test_history_contains_epoch_lists(self, logger_and_dir):
        data = self._load(logger_and_dir)
        history = data["runs"][0]["history"]
        assert "train_loss" in history
        assert isinstance(history["train_loss"], list)
        assert len(history["train_loss"]) == 3

    def test_best_test_acc_correct(self, logger_and_dir):
        hist = _make_history(n_epochs=3)
        # test_acc = [0.75, 0.79, 0.83] — best is last
        data = self._load(logger_and_dir, histories=[hist])
        assert data["runs"][0]["summary"]["best_test_acc"] == pytest.approx(max(hist["test_acc"]))

    def test_files_keys(self, logger_and_dir):
        data = self._load(logger_and_dir)
        assert {"log", "epochs_csv"} == set(data["runs"][0]["files"].keys())

    def test_json_is_valid(self, logger_and_dir):
        """Ensure the file is valid JSON (no serialisation errors)."""
        logger, run_dir = logger_and_dir
        logger.log_run(_make_config(), _make_history())
        logger.close()
        path = os.path.join(run_dir, "run_summary.json")
        with open(path) as f:
            data = json.load(f)   # raises if invalid
        assert data is not None

    def test_multiple_runs_in_json(self, logger_and_dir):
        data = self._load(
            logger_and_dir,
            configs=[_make_config("mnist"), _make_config("cifar10")],
            histories=[_make_history(), _make_history()],
        )
        assert len(data["runs"]) == 2
        names = {r["name"] for r in data["runs"]}
        assert "mnist_mlp_adam" in names
        assert "cifar10_mlp_adam" in names
