"""Training logger.

Creates a timestamped directory under `log_dir` for each session.
Writes one log file per training run (epoch-wise + batch-wise losses)
and a summary file covering all runs in the session.

Directory layout
----------------
logs/
  2026-03-09_14-30-00/
    run_summary.log          ← session overview: config, all results
    mnist_mlp_adam.log       ← epoch + batch detail for one run
    fashion_mnist_mlp_adamw.log
    ...
"""

import csv
import os
from datetime import datetime


class TrainingLogger:
    """Logs training runs to a timestamped directory.

    Usage
    -----
    logger = TrainingLogger(log_dir="logs")
    logger.log_run(config_dict, history_dict)   # call once per run
    logger.close()                               # writes run_summary.log
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
        """Write the per-run log file and record a summary entry.

        Parameters
        ----------
        config  : dict with at minimum keys dataset, model, optimizer.
                  Any additional keys (lr, epochs, batch_size, …) are
                  written to the log header.
        history : dict as returned by run_training() or collected by
                  main().  Expected keys:
                    train_loss, train_acc, test_loss, test_acc,
                    time_elapsed, step_losses.

        Returns
        -------
        str — filename of the written log (relative to run_dir).
        """
        name = (
            f"{config.get('dataset', 'unknown')}"
            f"_{config.get('model', 'unknown')}"
            f"_{config.get('optimizer', 'unknown')}"
        )
        log_filename = f"{name}.log"
        log_path = os.path.join(self.run_dir, log_filename)

        with open(log_path, "w", newline="") as f:
            # ── Header ──────────────────────────────────────────────────
            f.write(f"# Run      : {name}\n")
            f.write(f"# Logged   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for key, val in config.items():
                f.write(f"# {key:<12}: {val}\n")
            f.write("\n")

            # ── Epoch-wise table ─────────────────────────────────────────
            f.write("[epoch-wise]\n")
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc",
                "test_loss", "test_acc", "time_elapsed_s",
            ])
            n_epochs = len(history["train_loss"])
            for i in range(n_epochs):
                writer.writerow([
                    i + 1,
                    f"{history['train_loss'][i]:.6f}",
                    f"{history['train_acc'][i]:.6f}",
                    f"{history['test_loss'][i]:.6f}",
                    f"{history['test_acc'][i]:.6f}",
                    f"{history['time_elapsed'][i]:.2f}",
                ])

            # ── Batch-wise step losses ───────────────────────────────────
            step_losses = history.get("step_losses", [])
            if step_losses:
                f.write("\n[batch-wise]\n")
                writer.writerow(["step", "loss"])
                for step, loss in enumerate(step_losses, 1):
                    writer.writerow([step, f"{loss:.6f}"])

        # Record summary entry
        self._runs.append({
            "name":             name,
            "config":           config,
            "final_train_loss": history["train_loss"][-1],
            "final_train_acc":  history["train_acc"][-1],
            "final_test_loss":  history["test_loss"][-1],
            "final_test_acc":   history["test_acc"][-1],
            "log_file":         log_filename,
        })
        return log_filename

    def close(self) -> str:
        """Write run_summary.log and return its path."""
        end_time = datetime.now()
        elapsed  = (end_time - self._start_time).total_seconds()

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
                )
                f.write(header + "\n")
                f.write("-" * len(header) + "\n")
                for r in self._runs:
                    f.write(
                        f"{r['name']:<{col}}"
                        f"{r['final_train_loss']:>10.4f}"
                        f"{r['final_train_acc'] * 100:>9.2f}%"
                        f"{r['final_test_loss']:>10.4f}"
                        f"{r['final_test_acc'] * 100:>9.2f}%\n"
                    )

                f.write("\n# Individual log files:\n")
                for r in self._runs:
                    f.write(f"#   {r['log_file']}\n")

        print(f"\nLogs saved to  {self.run_dir}/")
        return summary_path
