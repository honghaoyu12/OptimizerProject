"""Training logger — structured output for every benchmark run.

Creates a timestamped session directory under `log_dir` and writes three
categories of files:

Per-run files (one pair per log_run() call):
  <dataset>_<model>_<optimizer>.log          — human-readable epoch table
                                               + per-batch step losses
  <dataset>_<model>_<optimizer>_epochs.csv  — clean epoch CSV (no comment
                                               lines; suitable for pandas)

Session summary files (written by close()):
  run_summary.log   — human-readable table of all runs in this session
  run_summary.csv   — machine-readable, one row per run (pandas-friendly)
  run_summary.json  — fully structured JSON with epoch histories embedded

Directory layout
----------------
logs/
  2026-03-09_14-30-00/
    run_summary.log
    run_summary.csv
    run_summary.json
    mnist_mlp_adam.log
    mnist_mlp_adam_epochs.csv
    fashion_mnist_mlp_adamw.log
    fashion_mnist_mlp_adamw_epochs.csv
    ...

Design decisions
----------------
  - Timestamp in directory name: guarantees no session ever overwrites another.
  - Two formats per run (.log and _epochs.csv): the .log is human-readable
    with a header block; the CSV has no comments so pandas/numpy can load it
    directly without the `comment='#'` workaround.
  - JSON summary keeps nested list histories for programmatic access without
    re-parsing individual CSVs.  Dict-typed history keys (weight_norms, etc.)
    are omitted from the JSON to keep file size manageable.
"""

import csv
import json
import os
from datetime import datetime


class TrainingLogger:
    """Logs training runs to a timestamped directory under `log_dir`.

    Typical usage
    -------------
    logger = TrainingLogger(log_dir="logs")         # creates the session dir
    logger.log_run(config_dict, history_dict)        # call once per training run
    logger.log_run(another_config, another_history)
    logger.close()                                   # writes summary files

    The logger is safe to call from benchmark.py across multiple seeds and
    optimizer combinations — each log_run() call is independent and appends
    to the internal summary list.
    """

    def __init__(self, log_dir: str = "logs"):
        # Record session start time for elapsed time in summary and as the
        # unique timestamp in the directory name.
        self._start_time = datetime.now()
        timestamp = self._start_time.strftime("%Y-%m-%d_%H-%M-%S")

        # Session directory: logs/<timestamp>/ — one per TrainingLogger instance
        self.run_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)

        # Accumulates one entry per log_run() call; used by close() to build
        # summary files.  Each entry is a dict with keys: name, config,
        # final_* / best_* metrics, log_file, epochs_csv, history.
        self._runs: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_run(self, config: dict, history: dict) -> str:
        """Write per-run log files and record a summary entry.

        Two files are written for each call:
          1. <name>.log          — human-readable header + epoch CSV + step losses
          2. <name>_epochs.csv   — clean epoch CSV (header only, no comments)

        Parameters
        ----------
        config  : dict with at minimum keys dataset, model, optimizer.
                  Any additional keys (lr, epochs, batch_size, …) are written
                  to the .log header block.  Avoid nested dicts as values.
        history : dict as returned by run_training() or collected by main().
                  Required top-level list keys:
                    train_loss, train_acc, test_loss, test_acc,
                    time_elapsed, step_losses.
                  Optional keys (written when present):
                    learning_rates   — per-epoch LR column in the epoch CSV/log
                    test_acc_ema     — EMA model accuracy (summary only)
                    swa_final_acc    — scalar SWA accuracy (summary only)

        Returns
        -------
        str — filename of the written .log file (relative to run_dir).
        """
        # Build a unique filename from dataset + model + optimizer keys
        name = (
            f"{config.get('dataset', 'unknown')}"
            f"_{config.get('model', 'unknown')}"
            f"_{config.get('optimizer', 'unknown')}"
        )
        log_filename  = f"{name}.log"
        log_path      = os.path.join(self.run_dir, log_filename)

        # Derive summary statistics from the history lists
        n_epochs       = len(history["train_loss"])
        test_accs      = history["test_acc"]
        best_test_acc  = max(test_accs) if test_accs else float("nan")
        # Epoch index is 0-based in the list; +1 converts to 1-based epoch number
        best_test_epoch = (test_accs.index(best_test_acc) + 1) if test_accs else None

        # ── 1. Human-readable .log file ──────────────────────────────────
        # Format: comment-prefixed header block, then a CSV epoch table,
        # then an optional CSV step-loss table.
        with open(log_path, "w", newline="") as f:
            # Header block: key-value pairs from config for quick inspection
            f.write(f"# Run      : {name}\n")
            f.write(f"# Logged   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, val in config.items():
                f.write(f"# {key:<12}: {val}\n")
            f.write("\n")

            # Epoch-wise CSV section
            f.write("[epoch-wise]\n")
            writer = csv.writer(f)
            learning_rates = history.get("learning_rates", [])
            # Column order matches the clean _epochs.csv for consistency
            writer.writerow([
                "epoch", "train_loss", "train_acc",
                "test_loss", "test_acc", "time_elapsed_s", "learning_rate",
            ])
            for i in range(n_epochs):
                # LR is optional — empty string if not recorded for this epoch
                lr_val = f"{learning_rates[i]:.8g}" if i < len(learning_rates) else ""
                writer.writerow([
                    i + 1,
                    f"{history['train_loss'][i]:.6f}",
                    f"{history['train_acc'][i]:.6f}",
                    f"{history['test_loss'][i]:.6f}",
                    f"{history['test_acc'][i]:.6f}",
                    f"{history['time_elapsed'][i]:.2f}",
                    lr_val,
                ])

            # Step-wise loss section — only written if step losses were recorded
            step_losses = history.get("step_losses", [])
            if step_losses:
                f.write("\n[batch-wise]\n")
                writer.writerow(["step", "loss"])
                for step, loss in enumerate(step_losses, 1):
                    writer.writerow([step, f"{loss:.6f}"])

        # ── 2. Clean epoch CSV ────────────────────────────────────────────
        # Identical columns to the epoch section above but without any
        # comment lines or section markers — directly loadable by pandas:
        #     pd.read_csv("mnist_mlp_adam_epochs.csv")
        epochs_csv_filename = f"{name}_epochs.csv"
        epochs_csv_path     = os.path.join(self.run_dir, epochs_csv_filename)
        with open(epochs_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc",
                "test_loss", "test_acc", "time_elapsed_s", "learning_rate",
            ])
            for i in range(n_epochs):
                lr_val = f"{learning_rates[i]:.8g}" if i < len(learning_rates) else ""
                writer.writerow([
                    i + 1,
                    f"{history['train_loss'][i]:.6f}",
                    f"{history['train_acc'][i]:.6f}",
                    f"{history['test_loss'][i]:.6f}",
                    f"{history['test_acc'][i]:.6f}",
                    f"{history['time_elapsed'][i]:.2f}",
                    lr_val,
                ])

        # ── 3. Record summary entry for close() ───────────────────────────
        # Stores the full history dict so run_summary.json can embed it.
        # Only scalar-list fields are serialised to JSON (see close()).
        self._runs.append({
            "name":             name,
            "config":           config,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float("nan"),
            "final_train_acc":  history["train_acc"][-1]  if history["train_acc"]  else float("nan"),
            "final_test_loss":  history["test_loss"][-1]  if history["test_loss"]  else float("nan"),
            "final_test_acc":   history["test_acc"][-1]   if history["test_acc"]   else float("nan"),
            "best_test_acc":    best_test_acc,
            "best_test_epoch":  best_test_epoch,
            "total_time_s":     history["time_elapsed"][-1] if history.get("time_elapsed") else float("nan"),
            "log_file":         log_filename,
            "epochs_csv":       epochs_csv_filename,
            "history":          history,  # kept in memory only; written to JSON by close()
        })
        return log_filename

    def close(self) -> str:
        """Flush session summary files.

        Writes three files to run_dir:
          run_summary.log  — human-readable table of every run in the session
          run_summary.csv  — one row per run (unions all config keys as columns)
          run_summary.json — full JSON with epoch histories embedded

        Returns
        -------
        str — path to run_summary.log (unchanged from the pre-CSV era for
              backward compatibility with any external tooling).
        """
        end_time = datetime.now()
        elapsed  = (end_time - self._start_time).total_seconds()

        # ── run_summary.log (human-readable table) ────────────────────────
        summary_path = os.path.join(self.run_dir, "run_summary.log")
        with open(summary_path, "w") as f:
            f.write("# Training Session Summary\n")
            f.write(f"# Started  : {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Finished : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Elapsed  : {elapsed:.1f}s\n")
            f.write(f"# Runs     : {len(self._runs)}\n")
            f.write("\n")

            if self._runs:
                col = 48   # column width for run names
                header = (
                    f"{'Run':<{col}}"
                    f"{'TrainLoss':>10}"
                    f"{'TrainAcc%':>10}"
                    f"{'TestLoss':>10}"
                    f"{'TestAcc%':>10}"
                    f"{'BestAcc%':>10}"
                )
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                for r in self._runs:
                    f.write(
                        f"{r['name']:<{col}}"
                        f"{r['final_train_loss']:>10.4f}"
                        f"{r['final_train_acc'] * 100:>9.2f}%"
                        f"{r['final_test_loss']:>10.4f}"
                        f"{r['final_test_acc'] * 100:>9.2f}%"
                        f"{r['best_test_acc'] * 100:>9.2f}%\n"
                    )

                f.write("\n# Individual log files:\n")
                for r in self._runs:
                    f.write(f"#   {r['log_file']}\n")

        # ── run_summary.csv (machine-readable, one row per run) ───────────
        # Build the union of all config keys across all runs as dynamic columns.
        # This handles runs with different config keys (e.g. different schedulers)
        # in the same session — missing keys get an empty cell rather than an error.
        csv_path = os.path.join(self.run_dir, "run_summary.csv")
        if self._runs:
            all_config_keys: list[str] = []
            seen: set = set()
            for r in self._runs:
                for k in r["config"]:
                    if k not in seen:
                        all_config_keys.append(k)   # preserve insertion order
                        seen.add(k)

            metric_cols = [
                "final_train_loss", "final_train_acc",
                "final_test_loss",  "final_test_acc",
                "best_test_acc",    "best_test_epoch",
                "total_time_s",
            ]
            # Column order: run name | all config keys | metrics | file references
            fieldnames = ["run"] + all_config_keys + metric_cols + ["log_file", "epochs_csv"]

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in self._runs:
                    row = {"run": r["name"]}
                    row.update(r["config"])
                    for col in metric_cols:
                        row[col] = r[col]
                    row["log_file"]   = r["log_file"]
                    row["epochs_csv"] = r["epochs_csv"]
                    writer.writerow(row)

        # ── run_summary.json (full structured export) ─────────────────────
        # Embeds per-epoch scalar-list histories for programmatic access.
        # Dict-typed history entries (weight_norms, grad_norms, etc.) are
        # intentionally omitted to keep the JSON file a manageable size;
        # they can always be reconstructed from the per-run .log files.
        json_path = os.path.join(self.run_dir, "run_summary.json")
        session_data = {
            "session": {
                "started":   self._start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "finished":  end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_s": round(elapsed, 2),
                "num_runs":  len(self._runs),
            },
            "runs": [],
        }
        for r in self._runs:
            history = r["history"]
            # Keep only flat list histories (no nested dicts like weight_norms)
            # and no nested list-of-dicts.  This is the minimal footprint that
            # still allows plotting all per-epoch scalar curves from JSON alone.
            serialisable_history = {
                k: v for k, v in history.items()
                if isinstance(v, list) and (not v or not isinstance(v[0], dict))
            }
            session_data["runs"].append({
                "name":   r["name"],
                "config": r["config"],
                "summary": {
                    "final_train_loss": r["final_train_loss"],
                    "final_train_acc":  r["final_train_acc"],
                    "final_test_loss":  r["final_test_loss"],
                    "final_test_acc":   r["final_test_acc"],
                    "best_test_acc":    r["best_test_acc"],
                    "best_test_epoch":  r["best_test_epoch"],
                    "total_time_s":     r["total_time_s"],
                },
                "history": serialisable_history,
                "files": {
                    "log":        r["log_file"],
                    "epochs_csv": r["epochs_csv"],
                },
            })

        # default=str handles any non-serialisable values (e.g. device objects)
        with open(json_path, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

        print(f"\nLogs saved to  {self.run_dir}/")
        return summary_path
