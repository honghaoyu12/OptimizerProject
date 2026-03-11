"""Live training dashboard.

Shows six panels that update after every epoch:
  ┌───────────────────┬───────────────────┐
  │   Loss vs Epoch   │  Accuracy vs Epoch│
  ├───────────────────┼───────────────────┤
  │   Loss vs Steps   │  Global Grad Norm │
  │  (batch-level)    │  (mean ± std band)│
  ├───────────────────┼───────────────────┤
  │  Weight L2 Norm   │ Hessian Trace &   │
  │  per Layer        │  Sharpness        │
  └───────────────────┴───────────────────┘
"""

import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Manages a 3×2 matplotlib figure updated each epoch.

    Parameters
    ----------
    layer_names : list[str]
        Names of the layers to track (used for legend labels).
    live : bool
        If True, open an interactive window and update it each epoch.
        If False, accumulate data silently and only render when `close()` is called.
    save_path : str | None
        If given, the final figure is saved to this path when `close()` is called.
    """

    _PALETTE = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
    ]

    def __init__(
        self,
        layer_names: list[str],
        live: bool = True,
        save_path: str | None = "training_curves.png",
    ):
        self.layer_names = layer_names
        self.live = live
        self.save_path = save_path

        # History buffers
        self.epochs: list[int] = []
        self.train_losses: list[float] = []
        self.test_losses:  list[float] = []
        self.train_accs:   list[float] = []
        self.test_accs:    list[float] = []
        self.weight_norms: dict[str, list[float]] = {n: [] for n in layer_names}
        self.grad_norms:   dict[str, list[float]] = {n: [] for n in layer_names}
        self.grad_norm_globals: list[float] = []
        self.grad_norm_stds:    list[float] = []
        self.all_step_losses:   list[float] = []
        self.hessian_traces:    list[float] = []
        self.sharpnesses:       list[float] = []
        self._lrs:              list[float] = []

        if live:
            plt.ion()
        self.fig, axes = plt.subplots(3, 2, figsize=(14, 14))
        self.fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")

        self.ax_loss   = axes[0][0]
        self.ax_acc    = axes[0][1]
        self.ax_steps  = axes[1][0]
        self.ax_ggrad  = axes[1][1]
        self.ax_weight = axes[2][0]
        self.ax_hess   = axes[2][1]

        self._setup_axes()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if live:
            self._flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        epoch: int,
        train_loss: float,
        test_loss: float,
        train_acc: float,
        test_acc: float,
        weight_norms: dict[str, float],
        grad_norms: dict[str, float],
        grad_norm_global: float = float("nan"),
        grad_norm_std: float = float("nan"),
        step_losses: list[float] | None = None,
        hessian_trace: float = float("nan"),
        sharpness: float = float("nan"),
        learning_rate: float = float("nan"),
    ) -> None:
        """Append one epoch of metrics and redraw the figure."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.train_accs.append(train_acc * 100)
        self.test_accs.append(test_acc * 100)
        self.grad_norm_globals.append(grad_norm_global)
        self.grad_norm_stds.append(grad_norm_std)
        self.hessian_traces.append(hessian_trace)
        self.sharpnesses.append(sharpness)
        self._lrs.append(learning_rate)
        if step_losses:
            self.all_step_losses.extend(step_losses)
        for name in self.layer_names:
            self.weight_norms[name].append(weight_norms.get(name, float("nan")))
            self.grad_norms[name].append(grad_norms.get(name, float("nan")))

        self._redraw()
        if self.live:
            self._flush()

    def close(self) -> None:
        """Render final figure, save to file, and close interactive mode."""
        self._redraw()
        if self.save_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
            self.fig.savefig(self.save_path, dpi=150, bbox_inches="tight")
            print(f"\nFigure saved to {self.save_path}")
        if self.live:
            plt.ioff()
            plt.show()
        else:
            plt.close(self.fig)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_axes(self) -> None:
        self.ax_loss.set_title("Loss vs Epoch")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Cross-entropy loss")

        self.ax_acc.set_title("Accuracy vs Epoch")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy (%)")

        self.ax_steps.set_title("Loss vs Steps (batch-level)")
        self.ax_steps.set_xlabel("Step")
        self.ax_steps.set_ylabel("Cross-entropy loss")

        self.ax_ggrad.set_title("Global Gradient Norm")
        self.ax_ggrad.set_xlabel("Epoch")
        self.ax_ggrad.set_ylabel("‖∇θ‖₂")

        self.ax_weight.set_title("Weight L2 Norm per Layer")
        self.ax_weight.set_xlabel("Epoch")
        self.ax_weight.set_ylabel("‖W‖₂")

        self.ax_hess.set_title("Hessian Trace & Sharpness")
        self.ax_hess.set_xlabel("Epoch")
        self.ax_hess.set_ylabel("Hessian Trace")

    def _redraw(self) -> None:
        ep = self.epochs

        # ---- Loss vs Epoch ----
        ax = self.ax_loss
        ax.clear()
        ax.set_title("Loss vs Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
        ax.plot(ep, self.train_losses, "o-", color="#2196F3", label="Train")
        ax.plot(ep, self.test_losses,  "s--", color="#F44336", label="Test")
        ax.legend(fontsize=8)
        finite_lrs = [lr for lr in self._lrs if not math.isnan(lr)]
        if finite_lrs:
            ax2_lr = ax.twinx()
            ax2_lr.plot(ep, self._lrs, "--", color="#9E9E9E", linewidth=1.2, label="LR")
            ax2_lr.set_ylabel("Learning Rate", color="#9E9E9E", fontsize=8)
            ax2_lr.tick_params(axis="y", labelcolor="#9E9E9E", labelsize=7)
            ax2_lr.legend(loc="upper right", fontsize=7)

        # ---- Accuracy vs Epoch ----
        ax = self.ax_acc
        ax.clear()
        ax.set_title("Accuracy vs Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.plot(ep, self.train_accs, "o-", color="#2196F3", label="Train")
        ax.plot(ep, self.test_accs,  "s--", color="#F44336", label="Test")
        ax.set_ylim(0, 101)
        ax.legend(fontsize=8)

        # ---- Loss vs Steps ----
        ax = self.ax_steps
        ax.clear()
        ax.set_title("Loss vs Steps (batch-level)"); ax.set_xlabel("Step"); ax.set_ylabel("Cross-entropy loss")
        if self.all_step_losses:
            steps = list(range(1, len(self.all_step_losses) + 1))
            ax.plot(steps, self.all_step_losses, color="#9C27B0", linewidth=0.8, alpha=0.7)

        # ---- Global Gradient Norm ----
        ax = self.ax_ggrad
        ax.clear()
        ax.set_title("Global Gradient Norm"); ax.set_xlabel("Epoch"); ax.set_ylabel("‖∇θ‖₂")
        valid = [(e, g, s) for e, g, s in zip(ep, self.grad_norm_globals, self.grad_norm_stds)
                 if not math.isnan(g)]
        if valid:
            ev, gv, sv = zip(*valid)
            gv_arr = np.array(gv)
            sv_arr = np.array(sv)
            ax.plot(ev, gv_arr, "o-", color="#009688", label="mean")
            ax.fill_between(ev, gv_arr - sv_arr, gv_arr + sv_arr,
                            alpha=0.25, color="#009688", label="±std")
            ax.legend(fontsize=8)

        # ---- Weight norms per layer ----
        ax = self.ax_weight
        ax.clear()
        ax.set_title("Weight L2 Norm per Layer"); ax.set_xlabel("Epoch"); ax.set_ylabel("‖W‖₂")
        for i, name in enumerate(self.layer_names):
            color = self._PALETTE[i % len(self._PALETTE)]
            ax.plot(ep, self.weight_norms[name], "o-", color=color, label=name)
        ax.legend(fontsize=7)

        # ---- Hessian Trace & Sharpness ----
        ax = self.ax_hess
        ax.clear()
        ax.set_title("Hessian Trace & Sharpness"); ax.set_xlabel("Epoch")

        ht_valid = [(e, h) for e, h in zip(ep, self.hessian_traces) if not math.isnan(h)]
        sh_valid = [(e, s) for e, s in zip(ep, self.sharpnesses)    if not math.isnan(s)]

        if not ht_valid and not sh_valid:
            ax.text(0.5, 0.5, "not computed\n(use --hessian)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="gray", style="italic")
        else:
            ax.set_ylabel("Hessian Trace")
            if ht_valid:
                ev, hv = zip(*ht_valid)
                ax.plot(ev, hv, "o-", color="#FF5722", label="Hessian Trace")
                ax.legend(loc="upper left", fontsize=8)
            if sh_valid:
                ax2 = ax.twinx()
                ax2.set_ylabel("Sharpness")
                ev, sv = zip(*sh_valid)
                ax2.plot(ev, sv, "s--", color="#795548", label="Sharpness")
                ax2.legend(loc="upper right", fontsize=8)

        self.fig.canvas.draw()

    def _flush(self) -> None:
        try:
            self.fig.canvas.flush_events()
            plt.pause(0.01)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Benchmark comparison plot
# ---------------------------------------------------------------------------

def plot_benchmark(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "benchmark.png",
) -> None:
    """Plot a comparison grid after a multi-optimizer benchmark run.

    Layout
    ------
    One row per (dataset × model) combination, four columns:
      Train Loss | Test Loss | Train Accuracy | Test Accuracy

    Parameters
    ----------
    results        : {(dataset_name, model_name, optimizer_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of optimizer names
    opt_colors     : {optimizer_name: hex_color_string}
    save_path      : file to save the figure (None = don't save)
    """
    row_labels = [f"{ds} / {mdl}" for ds in dataset_names for mdl in model_names]
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, 5, figsize=(22, 4.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle("Optimizer Benchmark", fontsize=15, fontweight="bold")

    col_titles = ["Train Loss", "Test Loss", "Train Accuracy (%)", "Test Accuracy (%)", "LR vs Epoch"]
    col_keys   = ["train_loss", "test_loss", "train_acc", "test_acc", "learning_rates"]
    is_acc     = [False, False, True, True, False]

    for row, (ds_name, mdl_name) in enumerate(
        (ds, mdl) for ds in dataset_names for mdl in model_names
    ):
        row_label = f"{ds_name} / {mdl_name}"
        for col, (title, key, acc) in enumerate(zip(col_titles, col_keys, is_acc)):
            ax = axes[row][col]
            ax.set_title(f"{row_label}\n{title}", fontsize=9)
            ax.set_xlabel("Epoch", fontsize=8)
            if key == "learning_rates":
                ax.set_ylabel("Learning Rate", fontsize=8)
            else:
                ax.set_ylabel("Accuracy (%)" if acc else "Cross-entropy loss", fontsize=8)
            if acc:
                ax.set_ylim(0, 101)

            for opt_name in optimizer_names:
                key3 = (ds_name, mdl_name, opt_name)
                if key3 not in results:
                    continue
                history = results[key3]
                values  = history.get(key, [])
                if not values:
                    continue
                if acc:
                    values = [v * 100 for v in values]
                epochs = list(range(1, len(values) + 1))
                color = opt_colors[opt_name]
                ax.plot(epochs, values, "o-",
                        color=color,
                        label=opt_name,
                        linewidth=1.8, markersize=4)

                std_vals = history.get(f"{key}_std", [])
                if std_vals and len(std_vals) == len(values):
                    if acc:
                        std_vals = [v * 100 for v in std_vals]
                    lo = [v - s for v, s in zip(values, std_vals)]
                    hi = [v + s for v, s in zip(values, std_vals)]
                    ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)

                if key == "test_acc":
                    tae = history.get("target_accuracy_epoch")
                    if tae is not None:
                        ax.axvline(x=tae, color=color, linestyle=":", linewidth=1.2, alpha=0.7)

            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nBenchmark figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Per-layer gradient flow heatmap
# ---------------------------------------------------------------------------

def plot_grad_flow_heatmap(
    results: dict,
    save_path: str | None = "plots/grad_flow.png",
) -> None:
    """Plot a per-layer gradient magnitude heatmap for each benchmark run.

    Each subplot is a 2-D heatmap where:
      - **x-axis** = epoch
      - **y-axis** = layer (input → output, bottom to top)
      - **colour** = log10 of the mean gradient L2 norm for that layer/epoch

    One row of subplots per (dataset, model) combination; one column per
    optimizer series.  Useful for spotting vanishing or exploding gradients
    across layers and epochs.

    Silently skips runs whose history contains no ``"grad_norms"`` data.

    Parameters
    ----------
    results : {(dataset_name, model_name, series_name): history_dict}
        Same format as returned by ``run_benchmark()``.
    save_path : str | None
        File path to save the figure.  ``None`` or ``''`` disables saving.
    """
    # Filter to runs that have per-layer grad norm data
    valid = {k: v for k, v in results.items() if v.get("grad_norms")}
    if not valid:
        print("\n  (Gradient flow heatmap skipped: no grad_norms found in results)")
        return

    combos = sorted({(k[0], k[1]) for k in valid})
    series_per_combo: dict[tuple, list[str]] = {}
    for ds, mdl, ser in valid:
        series_per_combo.setdefault((ds, mdl), []).append(ser)
    for combo in series_per_combo:
        series_per_combo[combo] = sorted(series_per_combo[combo])

    n_rows = len(combos)
    n_cols = max(len(v) for v in series_per_combo.values())

    fig_w = max(5, 3.5 * n_cols)
    fig_h = max(4, 3.0 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle("Per-Layer Gradient Flow (log₁₀ |grad|)", fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        series_list = series_per_combo[(ds_name, mdl_name)]
        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(series_list):
                ax.set_visible(False)
                continue

            series = series_list[col]
            history = valid[(ds_name, mdl_name, series)]
            grad_norms: dict[str, list[float]] = history["grad_norms"]

            # Layer order: as stored (train.py records input→output)
            layer_names = list(grad_norms.keys())
            n_layers = len(layer_names)
            n_epochs = max(len(grad_norms[ln]) for ln in layer_names)

            # Build matrix (layers × epochs), log10 scale, clip floor at -6
            matrix = np.full((n_layers, n_epochs), float("nan"))
            for i, ln in enumerate(layer_names):
                vals = grad_norms[ln]
                for j, v in enumerate(vals):
                    if v is not None and v > 0:
                        matrix[i, j] = math.log10(v)
                    elif v == 0:
                        matrix[i, j] = -6.0

            # Flip so input layer is at the bottom
            matrix = matrix[::-1]
            layer_labels = layer_names[::-1]

            vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else -6
            vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 0

            im = ax.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log₁₀|grad|")

            ax.set_title(f"{ds_name}/{mdl_name}\n{series}", fontsize=8)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.set_ylabel("Layer", fontsize=8)
            ax.set_xticks(range(n_epochs))
            ax.set_xticklabels([str(e + 1) for e in range(n_epochs)], fontsize=7)
            ax.set_yticks(range(n_layers))
            ax.set_yticklabels(layer_labels, fontsize=7)
            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Gradient flow heatmap saved to {save_path}")

    plt.show()

# ---------------------------------------------------------------------------
# LR sensitivity plot
# ---------------------------------------------------------------------------

def plot_lr_sensitivity(
    results: dict,
    save_path: str | None = "plots/lr_sensitivity.png",
) -> None:
    """Plot final test accuracy vs learning rate for each optimizer.

    One subplot per (dataset × model) combination; one line per optimizer.
    Error bars are drawn when ``test_acc_std`` is present (multi-seed runs).
    Silently skips when no history contains a ``"config_lr"`` key.

    Parameters
    ----------
    results : {(dataset_name, model_name, series_name): history_dict}
        Same format as returned by ``run_benchmark()``.  Each history should
        contain a ``"config_lr"`` key (set automatically by ``run_benchmark()``).
    save_path : str | None
        File path to save the figure.  ``None`` or ``''`` disables saving.
    """
    lr_entries = {k: v for k, v in results.items() if v.get("config_lr") is not None}
    if not lr_entries:
        print("\n  (LR sensitivity plot skipped: no config_lr found in results)")
        return

    combos = sorted({(k[0], k[1]) for k in lr_entries})
    n_rows = len(combos)

    fig, axes = plt.subplots(n_rows, 1, figsize=(8, 4 * n_rows), squeeze=False)
    fig.suptitle("LR Sensitivity", fontsize=14, fontweight="bold")

    base_opts = sorted({k[2].split(" (")[0] for k in lr_entries})
    cmap = plt.get_cmap("tab10")
    opt_colors = {name: cmap(i % 10) for i, name in enumerate(base_opts)}

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Learning Rate (log scale)", fontsize=9)
        ax.set_ylabel("Final Test Accuracy (%)", fontsize=9)
        ax.set_xscale("log")

        by_opt: dict[str, list[tuple]] = {}
        for (d, m, series), hist in lr_entries.items():
            if d != ds_name or m != mdl_name:
                continue
            base_opt = series.split(" (")[0]
            lr = hist["config_lr"]
            test_acc = hist.get("test_acc", [])
            test_acc_std = hist.get("test_acc_std", [])
            final_acc = test_acc[-1] * 100 if test_acc else float("nan")
            final_std = test_acc_std[-1] * 100 if test_acc_std else float("nan")
            by_opt.setdefault(base_opt, []).append((lr, final_acc, final_std))

        for base_opt, points in sorted(by_opt.items()):
            points.sort(key=lambda t: t[0])
            lrs_p, accs, stds = zip(*points)
            color = opt_colors.get(base_opt, "gray")
            ax.plot(lrs_p, accs, "o-", color=color, label=base_opt,
                    linewidth=1.8, markersize=5)
            valid_stds = [s for s in stds if not math.isnan(s)]
            if valid_stds:
                lo = [a - s for a, s in zip(accs, stds)]
                hi = [a + s for a, s in zip(accs, stds)]
                ax.fill_between(lrs_p, lo, hi, color=color, alpha=0.15)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  LR sensitivity figure saved to {save_path}")

    plt.show()
