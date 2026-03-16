"""Training logger.

Creates a timestamped directory under `log_dir` for each session.
Writes one log file per training run (epoch-wise + batch-wise losses)
and a summary file covering all runs in the session.

Directory layout
----------------
logs/
  2026-03-09_14-30-00/
    run_summary.log          ← human-readable session overview (unchanged)
    run_summary.csv          ← machine-readable, one row per run (pandas-friendly)
    run_summary.json         ← full structured export incl. epoch histories
    mnist_mlp_adam.log       ← epoch + batch detail for one run (unchanged)
    mnist_mlp_adam_epochs.csv← clean epoch CSV (no comment lines / section markers)
    fashion_mnist_mlp_adamw.log
    fashion_mnist_mlp_adamw_epochs.csv
    ...
"""

import csv
import json
import os
from datetime import datetime


class TrainingLogger:
    """Logs training runs to a timestamped directory.

    Usage
    -----
    logger = TrainingLogger(log_dir="logs")
    logger.log_run(config_dict, history_dict)   # call once per run
    logger.close()                               # writes summary files
    """

    def __init__(self, log_dir: str = "logs"):
        self._start_time = datetime.now()
        timestamp = self._start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        self._runs: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_run(self, config: dict, history: dict) -> str:
        """Write per-run log files and record a summary entry.

        Parameters
        ----------
        config  : dict with at minimum keys dataset, model, optimizer.
                  Any additional keys (lr, epochs, batch_size, …) are
                  written to the log header.
        history : dict as returned by run_training() or collected by
                  main().  Required keys:
                    train_loss, train_acc, test_loss, test_acc,
                    time_elapsed, step_losses.
                  Optional keys (written when present):
                    learning_rates   — per-epoch LR recorded to epoch CSV
                    test_acc_ema     — EMA model accuracy per epoch
                    swa_final_acc    — scalar SWA accuracy (written to summary).

        Returns
        -------
        str — filename of the written .log file (relative to run_dir).
        """
        name = (
            f"{config.get('dataset', 'unknown')}"
            f"_{config.get('model', 'unknown')}"
            f"_{config.get('optimizer', 'unknown')}"
        )
        log_filename = f"{name}.log"
        log_path = os.path.join(self.run_dir, log_filename)

        n_epochs = len(history["train_loss"])
        test_accs = history["test_acc"]
        best_test_acc   = max(test_accs) if test_accs else float("nan")
        best_test_epoch = (test_accs.index(best_test_acc) + 1) if test_accs else None

        # ── Original .log (unchanged format) ──────────────────────────
        with open(log_path, "w", newline="") as f:
            f.write(f"# Run      : {name}\n")
            f.write(f"# Logged   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, val in config.items():
                f.write(f"# {key:<12}: {val}\n")
            f.write("\n")

            f.write("[epoch-wise]\n")
            writer = csv.writer(f)
            learning_rates = history.get("learning_rates", [])
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

            step_losses = history.get("step_losses", [])
            if step_losses:
                f.write("\n[batch-wise]\n")
                writer.writerow(["step", "loss"])
                for step, loss in enumerate(step_losses, 1):
                    writer.writerow([step, f"{loss:.6f}"])

        # ── Clean epoch CSV ────────────────────────────────────────────
        epochs_csv_filename = f"{name}_epochs.csv"
        epochs_csv_path = os.path.join(self.run_dir, epochs_csv_filename)
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

        # Record summary entry (used by close())
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
            "history":          history,
        })
        return log_filename

    def close(self) -> str:
        """Write run_summary.log, run_summary.csv, run_summary.json.

        Returns the path of run_summary.log (unchanged from before).
        """
        end_time = datetime.now()
        elapsed  = (end_time - self._start_time).total_seconds()

        # ── run_summary.log (human-readable, unchanged) ────────────────
        summary_path = os.path.join(self.run_dir, "run_summary.log")
        with open(summary_path, "w") as f:
            f.write("# Training Session Summary\n")
            f.write(f"# Started  : {self._start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Finished : {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Elapsed  : {elapsed:.1f}s\n")
            f.write(f"# Runs     : {len(self._runs)}\n")
            f.write("\n")

            if self._runs:
                col = 48
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

        # ── run_summary.csv (machine-readable, one row per run) ────────
        csv_path = os.path.join(self.run_dir, "run_summary.csv")
        if self._runs:
            # Collect all config keys across all runs (union) for columns
            all_config_keys: list[str] = []
            seen: set = set()
            for r in self._runs:
                for k in r["config"]:
                    if k not in seen:
                        all_config_keys.append(k)
                        seen.add(k)

            metric_cols = [
                "final_train_loss", "final_train_acc",
                "final_test_loss",  "final_test_acc",
                "best_test_acc",    "best_test_epoch",
                "total_time_s",
            ]
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

        # ── run_summary.json (full structured export) ──────────────────
        json_path = os.path.join(self.run_dir, "run_summary.json")
        session_data = {
            "session": {
                "started":    self._start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "finished":   end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_s":  round(elapsed, 2),
                "num_runs":   len(self._runs),
            },
            "runs": [],
        }
        for r in self._runs:
            history = r["history"]
            # Serialise only the scalar-list fields (skip nested dicts like
            # weight_norms / grad_norms to keep the JSON reasonably compact;
            # they can always be loaded from the per-run .log)
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

        with open(json_path, "w") as f:
            json.dump(session_data, f, indent=2, default=str)

        print(f"\nLogs saved to  {self.run_dir}/")
        return summary_path
