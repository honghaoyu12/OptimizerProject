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
        # ep is the list of epoch numbers (x-axis for all epoch-level panels).
        # All axes are cleared and redrawn from scratch each call — this
        # avoids accumulation of old lines when the figure is updated live.
        ep = self.epochs

        # ── Panel 1: Loss vs Epoch ────────────────────────────────────────────
        # Shows train (blue) and test (red) cross-entropy loss on the same axis.
        # A secondary y-axis (right side, gray) shows the learning rate schedule
        # overlaid so we can visually correlate LR drops with loss decreases.
        ax = self.ax_loss
        ax.clear()
        ax.set_title("Loss vs Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
        ax.plot(ep, self.train_losses, "o-", color="#2196F3", label="Train")
        ax.plot(ep, self.test_losses,  "s--", color="#F44336", label="Test")
        ax.legend(fontsize=8)

        # Only draw LR axis if we have at least one finite LR value.
        # LR may be NaN if the optimizer didn't record it (e.g. Prodigy self-tunes).
        # twinx() creates a second y-axis that shares the same x-axis range,
        # allowing two differently-scaled quantities on the same plot.
        finite_lrs = [lr for lr in self._lrs if not math.isnan(lr)]
        if finite_lrs:
            ax2_lr = ax.twinx()
            ax2_lr.plot(ep, self._lrs, "--", color="#9E9E9E", linewidth=1.2, label="LR")
            ax2_lr.set_ylabel("Learning Rate", color="#9E9E9E", fontsize=8)
            ax2_lr.tick_params(axis="y", labelcolor="#9E9E9E", labelsize=7)
            ax2_lr.legend(loc="upper right", fontsize=7)

        # ── Panel 2: Accuracy vs Epoch ────────────────────────────────────────
        # Same colour coding as the loss panel (blue=train, red=test).
        # y-axis forced to [0, 101] so small improvements are visually meaningful
        # and the scale is consistent across multiple redraws.
        ax = self.ax_acc
        ax.clear()
        ax.set_title("Accuracy vs Epoch"); ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
        ax.plot(ep, self.train_accs, "o-", color="#2196F3", label="Train")
        ax.plot(ep, self.test_accs,  "s--", color="#F44336", label="Test")
        ax.set_ylim(0, 101)   # fixed range so the plot doesn't rescale every epoch
        ax.legend(fontsize=8)

        # ── Panel 3: Loss vs Steps (batch-level) ─────────────────────────────
        # all_step_losses accumulates every mini-batch loss throughout training
        # (not just per-epoch means), so the x-axis is the global step count.
        # linewidth=0.8 and alpha=0.7 keep the high-frequency noise readable
        # without dominating the visual; thin + semi-transparent = detail without clutter.
        ax = self.ax_steps
        ax.clear()
        ax.set_title("Loss vs Steps (batch-level)"); ax.set_xlabel("Step"); ax.set_ylabel("Cross-entropy loss")
        if self.all_step_losses:
            steps = list(range(1, len(self.all_step_losses) + 1))
            ax.plot(steps, self.all_step_losses, color="#9C27B0", linewidth=0.8, alpha=0.7)

        # ── Panel 4: Global Gradient Norm ────────────────────────────────────
        # grad_norm_globals is the L2 norm of the full concatenated gradient
        # vector (computed after clipping, if any).  grad_norm_stds is the
        # standard deviation of per-layer norms — wider std bands indicate that
        # some layers receive very different gradient magnitudes than others,
        # which can signal training instability or dead layers.
        # NaN filtering: gradient norms may be NaN if no valid parameter received
        # a gradient (e.g. the model is frozen).  We skip those epochs.
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
            # fill_between shows the ±std band — alpha=0.25 makes it translucent
            # so the mean line remains clearly visible through the shaded region.
            ax.fill_between(ev, gv_arr - sv_arr, gv_arr + sv_arr,
                            alpha=0.25, color="#009688", label="±std")
            ax.legend(fontsize=8)

        # ── Panel 5: Weight L2 Norm per Layer ────────────────────────────────
        # One line per named layer, using the _PALETTE colour list (cycling when
        # there are more layers than colours).  Weight norm growth over training
        # indicates that the optimizer is not effectively regularising; a flat
        # or shrinking norm suggests weight decay or Lion's sign update is active.
        ax = self.ax_weight
        ax.clear()
        ax.set_title("Weight L2 Norm per Layer"); ax.set_xlabel("Epoch"); ax.set_ylabel("‖W‖₂")
        for i, name in enumerate(self.layer_names):
            color = self._PALETTE[i % len(self._PALETTE)]  # cycle colours for >8 layers
            ax.plot(ep, self.weight_norms[name], "o-", color=color, label=name)
        ax.legend(fontsize=7)

        # ── Panel 6: Hessian Trace & Sharpness ───────────────────────────────
        # Both metrics are expensive to compute every epoch (they require
        # additional forward/backward passes), so they may be all-NaN if
        # --hessian / --sharpness flags were not passed.
        # When both are available, Sharpness uses a secondary y-axis (twinx)
        # because its scale (0.001–0.1) is different from Hessian Trace (10³–10⁶).
        ax = self.ax_hess
        ax.clear()
        ax.set_title("Hessian Trace & Sharpness"); ax.set_xlabel("Epoch")

        # Filter out NaN entries (epochs where the metric wasn't computed)
        ht_valid = [(e, h) for e, h in zip(ep, self.hessian_traces) if not math.isnan(h)]
        sh_valid = [(e, s) for e, s in zip(ep, self.sharpnesses)    if not math.isnan(s)]

        if not ht_valid and not sh_valid:
            # Neither metric was computed — show an informative placeholder message
            # instead of a blank panel.  transform=ax.transAxes uses axes coordinates
            # (0–1) so the text is centred regardless of the data range.
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
                # Secondary y-axis for Sharpness: different scale from Hessian Trace,
                # so we use a twin axis rather than plotting on the same scale.
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
    One row per (dataset × model) combination, eight columns:
      Train Loss | Test Loss | Train Accuracy | Test Accuracy | LR vs Epoch
      | Gen Gap (%) | Test Acc vs Steps | ECE

    New columns:

    * **Gen Gap (%)** — ``(train_acc − test_acc) × 100`` per epoch.  A rising
      gap signals overfitting; a flat or shrinking gap is healthy.  Error bands
      show ±1 std of the gap when seed-averaged data is available.

    * **Test Acc vs Steps** — test accuracy plotted against cumulative gradient
      steps (``steps_at_epoch_end``) instead of epochs.  This is a fairer
      comparison when optimizers take different-sized steps or when you care
      about compute budget rather than epoch count.

    * **ECE** — Expected Calibration Error per epoch.  Lower is better (0 =
      perfectly calibrated).  Adaptive methods tend to be better-calibrated
      than SGD on classification tasks.

    Parameters
    ----------
    results        : {(dataset_name, model_name, optimizer_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of optimizer names (used for colour lookup)
    opt_colors     : {optimizer_name: hex_color_string}
    save_path      : file to save the figure (None = don't save)
    """
    row_labels = [f"{ds} / {mdl}" for ds in dataset_names for mdl in model_names]
    n_rows = len(row_labels)

    fig, axes = plt.subplots(n_rows, 8, figsize=(36, 4.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    fig.suptitle("Optimizer Benchmark", fontsize=15, fontweight="bold")

    # ── Columns 0-4: standard epoch-based panels ──────────────────────────
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

                    ema_vals = history.get("test_acc_ema", [])
                    if ema_vals:
                        ema_pct = [v * 100 for v in ema_vals]
                        ema_epochs = list(range(1, len(ema_pct) + 1))
                        ax.plot(ema_epochs, ema_pct, "--",
                                color=color, linewidth=1.2, alpha=0.7,
                                label=f"{opt_name} (EMA)")

            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        # ── Column 5: Generalisation gap (train_acc − test_acc) × 100 ────
        ax_gap = axes[row][5]
        ax_gap.set_title(f"{row_label}\nGen Gap (%)", fontsize=9)
        ax_gap.set_xlabel("Epoch", fontsize=8)
        ax_gap.set_ylabel("Train − Test Acc (pp)", fontsize=8)
        # Dashed zero line so it's easy to spot zero-gap
        ax_gap.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history   = results[key3]
            train_acc = history.get("train_acc", [])
            test_acc  = history.get("test_acc", [])
            if not train_acc or not test_acc:
                continue
            n = min(len(train_acc), len(test_acc))
            gap    = [(tr - te) * 100 for tr, te in zip(train_acc[:n], test_acc[:n])]
            epochs = list(range(1, n + 1))
            color  = opt_colors[opt_name]
            ax_gap.plot(epochs, gap, "o-", color=color, label=opt_name,
                        linewidth=1.8, markersize=4)

            # Error band: gap std = sqrt(train_std² + test_std²)
            tr_std = history.get("train_acc_std", [])
            te_std = history.get("test_acc_std", [])
            if tr_std and te_std and len(tr_std) >= n and len(te_std) >= n:
                gap_std = [math.sqrt(ts ** 2 + es ** 2) * 100
                           for ts, es in zip(tr_std[:n], te_std[:n])]
                lo = [g - s for g, s in zip(gap, gap_std)]
                hi = [g + s for g, s in zip(gap, gap_std)]
                ax_gap.fill_between(epochs, lo, hi, color=color, alpha=0.15)

        ax_gap.legend(fontsize=7)
        ax_gap.tick_params(labelsize=7)

        # ── Column 6: Test accuracy vs cumulative gradient steps ──────────
        ax_steps = axes[row][6]
        ax_steps.set_title(f"{row_label}\nTest Acc vs Steps", fontsize=9)
        ax_steps.set_xlabel("Gradient steps", fontsize=8)
        ax_steps.set_ylabel("Test Accuracy (%)", fontsize=8)
        ax_steps.set_ylim(0, 101)

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history  = results[key3]
            test_acc = history.get("test_acc", [])
            steps    = history.get("steps_at_epoch_end", [])
            if not test_acc or not steps or len(test_acc) != len(steps):
                continue
            test_pct = [v * 100 for v in test_acc]
            color    = opt_colors[opt_name]
            ax_steps.plot(steps, test_pct, "o-", color=color, label=opt_name,
                          linewidth=1.8, markersize=4)

            std_vals = history.get("test_acc_std", [])
            if std_vals and len(std_vals) == len(test_pct):
                std_pct = [v * 100 for v in std_vals]
                lo = [v - s for v, s in zip(test_pct, std_pct)]
                hi = [v + s for v, s in zip(test_pct, std_pct)]
                ax_steps.fill_between(steps, lo, hi, color=color, alpha=0.15)

        ax_steps.legend(fontsize=7)
        ax_steps.tick_params(labelsize=7)

        # ── Column 7: Expected Calibration Error vs epoch ─────────────────
        ax_ece = axes[row][7]
        ax_ece.set_title(f"{row_label}\nECE vs Epoch", fontsize=9)
        ax_ece.set_xlabel("Epoch", fontsize=8)
        ax_ece.set_ylabel("ECE (lower is better)", fontsize=8)
        ax_ece.set_ylim(bottom=0)

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history  = results[key3]
            ece_vals = history.get("ece", [])
            if not ece_vals:
                continue
            epochs = list(range(1, len(ece_vals) + 1))
            color  = opt_colors[opt_name]
            ax_ece.plot(epochs, ece_vals, "o-", color=color, label=opt_name,
                        linewidth=1.8, markersize=4)

            std_vals = history.get("ece_std", [])
            if std_vals and len(std_vals) == len(ece_vals):
                lo = [max(0.0, v - s) for v, s in zip(ece_vals, std_vals)]
                hi = [v + s for v, s in zip(ece_vals, std_vals)]
                ax_ece.fill_between(epochs, lo, hi, color=color, alpha=0.15)

        ax_ece.legend(fontsize=7)
        ax_ece.tick_params(labelsize=7)

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
# LR sensitivity score helper + bar chart
# ---------------------------------------------------------------------------

def _compute_lr_sensitivity(results: dict) -> dict:
    """Compute per-optimizer LR sensitivity scores from a multi-LR sweep.

    Groups results by ``(dataset, model, base_optimizer)`` and, for each
    group with ≥2 LR values, computes:

    * ``range``    — ``max(final_acc) - min(final_acc)`` in percentage points
    * ``std``      — standard deviation of final accs in percentage points
    * ``best_lr``  — LR value that achieved the highest final accuracy
    * ``worst_lr`` — LR value that achieved the lowest final accuracy
    * ``n_lrs``    — number of LR values swept

    Only entries with ``config_lr`` in their history are considered.
    Returns an empty dict when fewer than 2 distinct LRs are present.

    Parameters
    ----------
    results : {(dataset_name, model_name, series_name): history_dict}
        Same format as returned by ``run_benchmark()``.
    """
    lr_entries = {k: v for k, v in results.items() if v.get("config_lr") is not None}

    groups: dict = {}
    for (ds, model, series), hist in lr_entries.items():
        base_opt = series.split(" (")[0]
        lr = hist["config_lr"]
        test_acc = hist.get("test_acc", [])
        if not test_acc:
            continue
        groups.setdefault((ds, model, base_opt), []).append((lr, test_acc[-1] * 100))

    scores: dict = {}
    for (ds, model, base_opt), points in groups.items():
        if len(points) < 2:
            continue
        lrs, accs = zip(*points)
        lrs, accs = list(lrs), list(accs)
        max_acc, min_acc = max(accs), min(accs)
        scores[(ds, model, base_opt)] = {
            "range":    round(max_acc - min_acc, 3),
            "std":      round(float(np.std(accs)), 3),
            "best_lr":  lrs[accs.index(max_acc)],
            "worst_lr": lrs[accs.index(min_acc)],
            "n_lrs":    len(points),
        }

    return scores


def plot_lr_sensitivity_scores(
    results: dict,
    save_path: str | None = "plots/lr_sensitivity_scores.png",
) -> None:
    """Horizontal bar chart of per-optimizer LR sensitivity scores.

    Each bar shows the **range** of final test accuracy (in percentage points)
    across all LR values swept for that optimizer.  Bars are sorted from most
    robust (smallest range, top) to most fragile (largest range, bottom).
    A ``±`` annotation shows the standard deviation alongside the range.

    One subplot per (dataset × model) combination.  Silently skips when fewer
    than two LR values were swept.

    Parameters
    ----------
    results : {(dataset_name, model_name, series_name): history_dict}
        Same format as returned by ``run_benchmark()``.
    save_path : str | None
        File path to save the figure.  ``None`` or ``''`` disables saving.
    """
    scores = _compute_lr_sensitivity(results)
    if not scores:
        print("\n  (LR sensitivity score plot skipped: need ≥2 LR values per optimizer)")
        return

    combos = sorted({(k[0], k[1]) for k in scores})
    n_rows = len(combos)
    fig_w = 8
    fig_h = max(4, 2 + 0.5 * max(
        sum(1 for k in scores if k[0] == ds and k[1] == mdl)
        for ds, mdl in combos
    ) * n_rows)

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle("LR Sensitivity Score\n(accuracy range across swept LRs — lower = more robust)",
                 fontsize=12, fontweight="bold")

    cmap = plt.get_cmap("RdYlGn_r")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        subset = {k[2]: v for k, v in scores.items()
                  if k[0] == ds_name and k[1] == mdl_name}
        # Sort: most robust (smallest range) at top
        opts = sorted(subset, key=lambda o: subset[o]["range"])
        ranges = [subset[o]["range"] for o in opts]
        stds   = [subset[o]["std"]   for o in opts]

        # Colour: green (low range) → red (high range)
        max_r = max(ranges) if max(ranges) > 0 else 1.0
        colors = [cmap(r / max_r) for r in ranges]

        bars = ax.barh(opts, ranges, color=colors, edgecolor="white", linewidth=0.5)
        # Annotate each bar with "X.X ± Y.Y pp"
        for bar, r, s in zip(bars, ranges, stds):
            ax.text(
                bar.get_width() + max_r * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{r:.2f} ± {s:.2f} pp",
                va="center", ha="left", fontsize=8,
            )

        ax.set_xlim(0, max_r * 1.35)
        ax.set_xlabel("Accuracy range (percentage points)", fontsize=9)
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()  # most robust at top

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  LR sensitivity score chart saved to {save_path}")

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


# ---------------------------------------------------------------------------
# Optimizer internal state plot
# ---------------------------------------------------------------------------

_STATE_LABELS = {
    "exp_avg_sq": "2nd Moment v_t",
    "exp_avg":    "Momentum |m|",
    "hessian":    "Hessian diagonal",
    "d":          "LR estimate d",
}


def plot_optimizer_states(
    results: dict,
    save_path: str | None = "plots/opt_states.png",
) -> None:
    """Plot optimizer internal state trajectories for each benchmark run.

    Two panel types depending on the state:

    * **Scalar trajectory** (e.g. Prodigy's ``d``): line plot showing how
      the scalar evolves over epochs.
    * **Per-layer heatmap** (e.g. Adam ``exp_avg_sq``, Lion ``exp_avg``,
      Sophia ``hessian``): ``imshow`` grid with x=epoch, y=layer
      (input at bottom), colour=log₁₀ of mean absolute state magnitude.
      Uses the ``plasma`` colourmap to distinguish it from the gradient
      flow heatmap (``viridis``).

    Layout: rows = (dataset, model, state_key); columns = optimizer series.

    Silently skips when no ``"optimizer_states"`` data is found.

    Parameters
    ----------
    results : {(dataset_name, model_name, series_name): history_dict}
        Same format as returned by ``run_benchmark()``.
    save_path : str | None
        File path to save the figure.  ``None`` or ``''`` disables saving.
    """
    # Group: (ds, model, state_key) → [(series, history), ...]
    grouped: dict = {}
    for (ds, model, series), hist in results.items():
        opt_states = hist.get("optimizer_states") or {}
        for key, val in opt_states.items():
            has_data = any(v for v in val.values()) if isinstance(val, dict) else bool(val)
            if has_data:
                grouped.setdefault((ds, model, key), []).append((series, hist))

    if not grouped:
        print("\n  (Optimizer state plot skipped: no optimizer_states found in results)")
        return

    row_keys = sorted(grouped.keys())
    n_rows = len(row_keys)
    n_cols = max(len(v) for v in grouped.values())

    fig_w = max(5, 4.0 * n_cols)
    fig_h = max(4, 3.5 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle("Optimizer Internal State", fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name, state_key) in enumerate(row_keys):
        label = _STATE_LABELS.get(state_key, state_key)
        series_list = grouped[(ds_name, mdl_name, state_key)]

        for col in range(n_cols):
            ax = axes[row][col]
            if col >= len(series_list):
                ax.set_visible(False)
                continue

            series, hist = series_list[col]
            state_data = (hist.get("optimizer_states") or {}).get(state_key)
            title = f"{ds_name}/{mdl_name}\n{series}\n{label}"

            if isinstance(state_data, list):
                # Scalar trajectory (e.g. Prodigy's d)
                epochs_range = list(range(1, len(state_data) + 1))
                ax.plot(epochs_range, state_data, color="#377eb8", linewidth=1.8,
                        marker="o", markersize=4)
                ax.set_xlabel("Epoch", fontsize=8)
                ax.set_ylabel(label, fontsize=8)
                ax.set_title(title, fontsize=8)
                ax.tick_params(labelsize=7)

            elif isinstance(state_data, dict) and state_data:
                # Per-layer heatmap
                layer_names_ordered = list(state_data.keys())
                n_layers = len(layer_names_ordered)
                n_epochs = max(len(v) for v in state_data.values())

                matrix = np.full((n_layers, n_epochs), float("nan"))
                for i, ln in enumerate(layer_names_ordered):
                    for j, v in enumerate(state_data[ln]):
                        if v is not None and v > 0:
                            matrix[i, j] = math.log10(v)
                        elif v == 0:
                            matrix[i, j] = -10.0

                # Flip so input layer is at the bottom
                matrix = matrix[::-1]
                layer_labels = layer_names_ordered[::-1]

                vmin = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else -10
                vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 0

                im = ax.imshow(
                    matrix, aspect="auto", origin="lower", cmap="plasma",
                    vmin=vmin, vmax=vmax, interpolation="nearest",
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                             label=f"log₁₀({label})")
                ax.set_title(title, fontsize=8)
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
        print(f"\n  Optimizer state figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Efficiency frontier plot
# ---------------------------------------------------------------------------

def _pareto_frontier(points: list[tuple]) -> list[tuple]:
    """Return the Pareto-optimal subset of (time, acc, label) points.

    A point is on the frontier if no other point is both faster *and* more
    accurate.  The returned list is sorted by time (ascending).
    """
    sorted_pts = sorted(points, key=lambda p: p[0])
    frontier = []
    best_acc = -float("inf")
    for t, acc, label in sorted_pts:
        if acc > best_acc:
            frontier.append((t, acc, label))
            best_acc = acc
    return frontier


def plot_efficiency_frontier(
    results: dict,
    save_path: str | None = "plots/efficiency_frontier.png",
) -> None:
    """Scatter plot of final test accuracy vs total training time per optimizer.

    Each point represents one optimizer series.  The **Pareto frontier** —
    the set of runs where no alternative is both faster *and* more accurate —
    is highlighted with a step line, making it easy to read off the best
    optimizer for any given time budget.

    One subplot per (dataset, model) combination.  Error bars are drawn when
    ``test_acc_std`` is present (multi-seed runs).

    Silently skips when no history contains ``time_elapsed`` data.

    Parameters
    ----------
    results : {(dataset_name, model_name, series_name): history_dict}
        Same format as returned by ``run_benchmark()``.
    save_path : str | None
        File path to save the figure.  ``None`` or ``''`` disables saving.
    """
    valid = {k: v for k, v in results.items()
             if v.get("time_elapsed") and v.get("test_acc")}
    if not valid:
        print("\n  (Efficiency frontier plot skipped: no time_elapsed data in results)")
        return

    combos = sorted({(k[0], k[1]) for k in valid})
    n_rows = len(combos)

    fig_w = max(6, 6 * min(n_rows, 2))
    fig_h = max(5, 5 * math.ceil(n_rows / 2))
    n_cols = min(n_rows, 2)
    n_plot_rows = math.ceil(n_rows / n_cols)
    fig, axes = plt.subplots(n_plot_rows, n_cols,
                             figsize=(fig_w, fig_h), squeeze=False)
    fig.suptitle("Efficiency Frontier\n(accuracy vs training time)",
                 fontsize=13, fontweight="bold")

    # Consistent colour per base optimizer name across subplots
    all_series = sorted({k[2] for k in valid})
    cmap = plt.get_cmap("tab20")
    series_colors = {s: cmap(i % 20 / 20) for i, s in enumerate(all_series)}

    for idx, (ds_name, mdl_name) in enumerate(combos):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        points: list[tuple] = []  # (time, acc, label, std)
        for (d, m, series), hist in valid.items():
            if d != ds_name or m != mdl_name:
                continue
            t = hist["time_elapsed"][-1]
            acc = hist["test_acc"][-1] * 100
            std_list = hist.get("test_acc_std", [])
            std = std_list[-1] * 100 if std_list else float("nan")
            points.append((t, acc, series, std))

        # Plot each point
        for t, acc, series, std in points:
            color = series_colors.get(series, "gray")
            if not math.isnan(std):
                ax.errorbar(t, acc, yerr=std, fmt="o", color=color,
                            capsize=4, markersize=7, linewidth=1.5)
            else:
                ax.scatter(t, acc, color=color, s=60, zorder=3)
            ax.annotate(
                series,
                xy=(t, acc),
                xytext=(5, 3),
                textcoords="offset points",
                fontsize=7,
                color=color,
            )

        # Pareto frontier
        frontier = _pareto_frontier([(t, acc, s) for t, acc, s, _ in points])
        if len(frontier) >= 2:
            ft, fa, _ = zip(*frontier)
            # Step line: extend to the right edge for readability
            max_t = max(t for t, _, _, _ in points)
            step_t = list(ft) + [max_t]
            step_a = list(fa) + [fa[-1]]
            ax.step(step_t, step_a, where="post", color="black",
                    linewidth=1.2, linestyle="--", alpha=0.5,
                    label="Pareto frontier")
            ax.legend(fontsize=7, loc="lower right")

        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Training time (s)", fontsize=9)
        ax.set_ylabel("Final test accuracy (%)", fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide unused subplots
    for idx in range(len(combos), n_plot_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Efficiency frontier saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Weight distance from initialisation
# ---------------------------------------------------------------------------

def plot_weight_distance(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/weight_distance.png",
) -> None:
    """Plot how far each optimizer moves the weights from their initial values.

    One subplot per (dataset × model) combination.  Each line shows the
    **global** weight distance (sum of per-layer L2 distances) vs epoch.
    Error bands are drawn when seed-averaged per-layer std data is present.

    A rapidly rising curve early in training indicates large, aggressive
    updates (typical of adaptive methods at high LR).  A slow, steady rise
    is characteristic of SGD with momentum.  Optimisers that find flat minima
    often end up *further* from init at convergence.

    Silently skips runs whose ``history["weight_distance"]`` is empty.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of optimizer/series names (for colour lookup)
    opt_colors     : {series_name: colour}
    save_path      : file path to save the figure; ``None`` / ``''`` disables.
    """
    combos = [(ds, mdl) for ds in dataset_names for mdl in model_names]
    n_rows = len(combos)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Weight Distance from Initialisation (global L2)", fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Σ ||w_t − w_0||₂ (all layers)", fontsize=9)

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history = results[key3]
            wd_dict = history.get("weight_distance", {})
            if not wd_dict:
                continue

            # Sum per-layer distances into one global trace
            n_epochs = max(len(v) for v in wd_dict.values() if v)
            global_dist = [0.0] * n_epochs
            for layer_vals in wd_dict.values():
                for e, v in enumerate(layer_vals[:n_epochs]):
                    if not math.isnan(v):
                        global_dist[e] += v

            epochs = list(range(1, n_epochs + 1))
            color  = opt_colors[opt_name]
            ax.plot(epochs, global_dist, "o-", color=color, label=opt_name,
                    linewidth=1.8, markersize=4)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Weight distance figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Gradient signal-to-noise ratio (per layer, vs epoch)
# ---------------------------------------------------------------------------


def plot_grad_snr(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/grad_snr.png",
) -> None:
    """Plot the per-layer gradient signal-to-noise ratio (SNR) vs epoch.

    SNR is defined as ``mean_norm / std_norm`` where the mean and std are
    taken over the per-batch gradient L2 norms of each Linear layer within
    a training epoch.

    A high SNR (>> 1) indicates consistent gradient directions across
    mini-batches — the gradient signal is strong relative to its stochastic
    noise.  SNR ≈ 1 indicates a highly variable, noisy gradient.

    One subplot per (dataset × model) combination.  Each line shows the
    **mean SNR across all layers** vs epoch (global view).  Per-layer detail
    is averaged out because individual layer SNRs are highly correlated and a
    single trace is easier to compare across optimizers.

    Silently skips runs whose ``history["grad_snr"]`` is empty.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of optimizer/series names (for colour lookup)
    opt_colors     : {series_name: colour}
    save_path      : file path to save the figure; ``None`` / ``''`` disables.
    """
    combos = [(ds, mdl) for ds in dataset_names for mdl in model_names]
    n_rows = len(combos)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Gradient SNR (mean across layers)", fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("mean_norm / std_norm (higher = lower grad noise)", fontsize=9)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history = results[key3]
            snr_dict = history.get("grad_snr", {})
            if not snr_dict:
                continue

            # Compute mean SNR across all layers for each epoch
            n_epochs = max(len(v) for v in snr_dict.values() if v)
            mean_snr = []
            for e in range(n_epochs):
                layer_vals = [
                    snr_dict[lyr][e]
                    for lyr in snr_dict
                    if e < len(snr_dict[lyr]) and not math.isnan(snr_dict[lyr][e])
                ]
                mean_snr.append(sum(layer_vals) / len(layer_vals) if layer_vals else float("nan"))

            epochs = list(range(1, n_epochs + 1))
            color  = opt_colors[opt_name]
            ax.plot(epochs, mean_snr, "o-", color=color, label=opt_name,
                    linewidth=1.8, markersize=4)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Gradient SNR figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Hyperparameter robustness heatmap (LR × WD → final test accuracy)
# ---------------------------------------------------------------------------

def plot_hp_heatmap(
    results: dict,
    save_path: str | None = "plots/hp_heatmap.png",
) -> None:
    """2-D heatmap of final test accuracy over an LR × weight-decay grid.

    One subplot per (dataset, model, base-optimizer) triple.  Each cell shows
    the final test accuracy for that (lr, wd) combination, colour-coded from
    low (purple) to high (yellow, viridis).  Optimisers whose accuracy stays
    high across the full grid are *robust* to hyperparameter choice; those
    with a narrow bright ridge are *sensitive*.

    Requires ``history["config_lr"]`` and ``history["config_wd"]`` to be set
    (``run_benchmark()`` stores these automatically).  Silently skips combos
    that have fewer than 2 unique LR values *or* fewer than 2 unique WD values.

    Parameters
    ----------
    results  : {(dataset_name, model_name, series_name): history_dict}
    save_path: file path to save the figure; ``None`` / ``''`` disables.
    """
    from collections import defaultdict

    # Group by (ds, mdl, base_opt) → {(lr, wd): final_acc}
    # Base optimizer name is the part before the first " (" in the series name.
    groups: dict = defaultdict(dict)
    for (ds, mdl, series), hist in results.items():
        lr = hist.get("config_lr")
        wd = hist.get("config_wd")
        if lr is None or wd is None:
            continue
        final_acc = hist.get("test_acc", [])
        if not final_acc:
            continue
        base_opt = series.split(" (")[0]
        groups[(ds, mdl, base_opt)][(lr, wd)] = final_acc[-1] * 100

    if not groups:
        print("\n  (HP heatmap skipped: config_lr / config_wd not found in results)")
        return

    # Keep only combos with >= 2 unique values on each axis
    valid = {
        k: v for k, v in groups.items()
        if len({lr for lr, _ in v}) >= 2 and len({wd for _, wd in v}) >= 2
    }
    if not valid:
        print("\n  (HP heatmap skipped: need ≥ 2 LR values and ≥ 2 WD values)")
        return

    n_plots = len(valid)
    n_cols  = min(3, n_plots)
    n_rows  = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    fig.suptitle("HP Robustness: LR × Weight Decay → Final Test Accuracy (%)",
                 fontsize=13, fontweight="bold")

    for idx, ((ds, mdl, base_opt), cell_map) in enumerate(sorted(valid.items())):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        lrs = sorted({lr for lr, _ in cell_map})
        wds = sorted({wd for _, wd in cell_map})

        # Build 2-D accuracy matrix (rows = LR, cols = WD)
        matrix = np.full((len(lrs), len(wds)), float("nan"))
        for (lr, wd), acc in cell_map.items():
            r = lrs.index(lr)
            c = wds.index(wd)
            matrix[r, c] = acc

        im = ax.imshow(matrix, aspect="auto", origin="lower",
                       cmap="viridis", vmin=0, vmax=100)
        fig.colorbar(im, ax=ax, label="Test Acc (%)", fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(wds)))
        ax.set_xticklabels([f"{w:g}" for w in wds], fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{l:g}" for l in lrs], fontsize=8)
        ax.set_xlabel("Weight Decay", fontsize=9)
        ax.set_ylabel("Learning Rate", fontsize=9)
        ax.set_title(f"{ds} / {mdl}\n{base_opt}", fontsize=9)

        # Annotate each cell with its accuracy value
        for r in range(len(lrs)):
            for c in range(len(wds)):
                val = matrix[r, c]
                if not math.isnan(val):
                    ax.text(c, r, f"{val:.1f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if val < 60 else "black")

    # Hide unused subplot cells
    for idx in range(len(valid), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  HP heatmap saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Per-class accuracy comparison (final epoch)
# ---------------------------------------------------------------------------


def plot_class_accuracy(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/class_accuracy.png",
) -> None:
    """Bar/line chart of final test accuracy broken down by class.

    Shows which classes each optimizer handles best and worst.  Useful for
    spotting optimizers that boost overall accuracy at the cost of hard or
    minority classes.

    Layout
    ------
    One subplot per (dataset × model) combination.

    * **≤ 20 classes** — grouped bar chart: one cluster per class, one bar
      per optimizer, coloured by optimizer.
    * **> 20 classes** — line plot: x = class ID, y = final accuracy.  Bar
      clusters become illegible for CIFAR-100 (100 classes) or Tiny ImageNet
      (200 classes), so lines are used instead.

    Silently skips runs whose ``history["class_acc"]`` is empty.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of optimizer/series names (for colour lookup)
    opt_colors     : {series_name: colour}
    save_path      : file path to save; ``None`` / ``''`` disables saving.
    """
    combos = [(ds, mdl) for ds in dataset_names for mdl in model_names]
    n_rows = len(combos)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 5 * n_rows), squeeze=False)
    fig.suptitle("Per-Class Test Accuracy (final epoch)", fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Class ID", fontsize=9)
        ax.set_ylabel("Test Accuracy (%)", fontsize=9)
        ax.set_ylim(0, 105)

        # Collect final per-class accuracies for each optimizer
        opt_class_accs: dict[str, dict[int, float]] = {}
        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history = results[key3]
            ca = history.get("class_acc", {})
            if not ca:
                continue
            # Take the last epoch value for each class
            opt_class_accs[opt_name] = {
                cls_id: vals[-1] * 100
                for cls_id, vals in ca.items()
                if vals
            }

        if not opt_class_accs:
            continue

        all_classes = sorted({c for d in opt_class_accs.values() for c in d})
        n_classes   = len(all_classes)
        n_opts      = len(opt_class_accs)

        if n_classes <= 20:
            # Grouped bars
            bar_width = 0.8 / max(n_opts, 1)
            for i, (opt_name, class_acc) in enumerate(opt_class_accs.items()):
                x_offsets = [
                    j + (i - n_opts / 2 + 0.5) * bar_width
                    for j, _ in enumerate(all_classes)
                ]
                heights = [class_acc.get(c, 0.0) for c in all_classes]
                ax.bar(x_offsets, heights, width=bar_width * 0.9,
                       color=opt_colors[opt_name], alpha=0.85, label=opt_name)
            ax.set_xticks(range(n_classes))
            ax.set_xticklabels([str(c) for c in all_classes], fontsize=8)
        else:
            # Line plot for many classes
            for opt_name, class_acc in opt_class_accs.items():
                xs = all_classes
                ys = [class_acc.get(c, float("nan")) for c in xs]
                ax.plot(xs, ys, "-", color=opt_colors[opt_name],
                        label=opt_name, linewidth=1.5, alpha=0.85)
            ax.set_xlim(all_classes[0] - 0.5, all_classes[-1] + 0.5)

        ax.axhline(y=100 / n_classes * 100 / 100,   # chance level — always 1/n_classes
                   color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Per-class accuracy figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Training instability (batch loss std vs epoch)
# ---------------------------------------------------------------------------


def plot_instability(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/instability.png",
) -> None:
    """Plot training instability (within-epoch batch loss std) vs epoch.

    A high value at a given epoch means per-batch losses were highly variable
    within that epoch — indicative of a rough loss surface or an overly large
    learning rate.  A smoothly declining curve suggests the optimizer is
    settling into a stable region.

    One subplot per (dataset × model) combination.  A dashed y=0 reference
    line is drawn since std = 0 would mean every batch had identical loss.

    Silently skips runs whose ``history["batch_loss_std"]`` is empty.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of optimizer/series names (for colour lookup)
    opt_colors     : {series_name: colour}
    save_path      : file path to save; ``None`` / ``''`` disables saving.
    """
    combos = [(ds, mdl) for ds in dataset_names for mdl in model_names]
    n_rows = len(combos)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Training Instability (within-epoch batch loss std)", fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Batch loss std (within epoch)", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history = results[key3]
            vals = history.get("batch_loss_std", [])
            if not vals:
                continue

            epochs = list(range(1, len(vals) + 1))
            color  = opt_colors[opt_name]
            ax.plot(epochs, vals, "o-", color=color, label=opt_name,
                    linewidth=1.8, markersize=4)

            # Error bands when seed-averaged std is available
            std_vals = history.get("batch_loss_std_std", [])
            if std_vals and len(std_vals) == len(vals):
                lo = [max(0.0, v - s) for v, s in zip(vals, std_vals)]
                hi = [v + s for v, s in zip(vals, std_vals)]
                ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Instability figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Effective step size (||w_t - w_{t-1}|| per layer, vs epoch)
# ---------------------------------------------------------------------------


def plot_step_size(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/step_size.png",
) -> None:
    """Plot the per-epoch effective step size (weight-space displacement).

    Each epoch's value is the mean across all Linear layers of
    ``||w_t - w_{t-1}||_2`` — the actual distance the optimizer moved the
    weights in a single epoch.  This is distinct from the gradient norm
    (which measures how large the gradient is *before* the optimizer applies
    its update rule) and from weight distance from init (which is cumulative).

    Adaptive methods like Adam typically show a *decreasing* step size as
    the second-moment estimates grow and dampen the effective learning rate.
    SGD+Momentum tends to keep a more constant step size relative to the
    gradient norm.

    One subplot per (dataset × model) combination.  Silently skips runs
    whose ``history["step_size"]`` is empty.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of optimizer/series names (for colour lookup)
    opt_colors     : {series_name: colour}
    save_path      : file path to save; ``None`` / ``''`` disables saving.
    """
    combos = [(ds, mdl) for ds in dataset_names for mdl in model_names]
    n_rows = len(combos)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Effective Step Size (mean ||w_t − w_{t-1}||₂ across layers)",
                 fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("mean ||w_t − w_{t-1}||₂", fontsize=9)

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history = results[key3]
            ss_dict = history.get("step_size", {})
            if not ss_dict:
                continue

            # Average step size across all layers
            n_epochs  = max(len(v) for v in ss_dict.values() if v)
            mean_step = []
            for e in range(n_epochs):
                layer_vals = [
                    ss_dict[lyr][e]
                    for lyr in ss_dict
                    if e < len(ss_dict[lyr]) and not math.isnan(ss_dict[lyr][e])
                ]
                mean_step.append(sum(layer_vals) / len(layer_vals) if layer_vals else float("nan"))

            epochs = list(range(1, n_epochs + 1))
            color  = opt_colors[opt_name]
            ax.plot(epochs, mean_step, "o-", color=color, label=opt_name,
                    linewidth=1.8, markersize=4)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Step size figure saved to {save_path}")

    plt.show()


def plot_sharpness(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/sharpness.png",
) -> None:
    """Plot SAM-style sharpness vs epoch for each optimizer.

    Sharpness = max loss increase under a random perturbation of radius ε.
    Only plotted for series where sharpness was actually computed (non-nan
    values present); series with all-nan sharpness are silently skipped.

    One subplot per (dataset × model) combination.
    """
    combos = [(ds, mdl) for ds in dataset_names for mdl in model_names]
    n_rows = len(combos)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Loss Landscape Sharpness (SAM-style, ε=0.01)",
                 fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Sharpness", fontsize=9)

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history = results[key3]
            values = history.get("sharpness", [])
            # Skip series where sharpness was never computed (all nan)
            finite_vals = [v for v in values if not math.isnan(v)]
            if not finite_vals:
                continue

            epochs = list(range(1, len(values) + 1))
            color  = opt_colors[opt_name]
            ax.plot(epochs, values, "o-", color=color, label=opt_name,
                    linewidth=1.8, markersize=4)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Sharpness figure saved to {save_path}")

    plt.show()


def plot_grad_cosine_sim(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/grad_cosine_sim.png",
) -> None:
    """Plot cosine similarity between consecutive epoch gradient directions.

    Shows how consistent the gradient direction is across epochs for each
    optimizer.  Values near 1 = very stable gradient direction; near 0 =
    rotating / oscillating; negative = reversed direction.

    Epoch 1 has no prior direction (nan) and is excluded from the plot.
    The global mean across all Linear layers is shown per epoch.

    One subplot per (dataset × model) combination.
    """
    combos = [(ds, mdl) for ds in dataset_names for mdl in model_names]
    n_rows = len(combos)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Gradient Cosine Similarity (consecutive epochs, mean across layers)",
                 fontsize=13, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Cosine similarity", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history = results[key3]
            cs_dict = history.get("grad_cosine_sim", {})
            if not cs_dict:
                continue

            # Mean cosine similarity across all layers per epoch
            n_epochs  = max(len(v) for v in cs_dict.values() if v)
            mean_cs = []
            for e in range(n_epochs):
                layer_vals = [
                    cs_dict[lyr][e]
                    for lyr in cs_dict
                    if e < len(cs_dict[lyr]) and not math.isnan(cs_dict[lyr][e])
                ]
                mean_cs.append(
                    sum(layer_vals) / len(layer_vals) if layer_vals else float("nan")
                )

            # Only plot the epochs where we have a valid value (skip epoch 1 nan)
            valid_epochs = [e + 1 for e, v in enumerate(mean_cs) if not math.isnan(v)]
            valid_vals   = [v for v in mean_cs if not math.isnan(v)]
            if not valid_vals:
                continue

            color = opt_colors[opt_name]
            ax.plot(valid_epochs, valid_vals, "o-", color=color, label=opt_name,
                    linewidth=1.8, markersize=4)

        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Gradient cosine similarity figure saved to {save_path}")

    plt.show()


def plot_grad_conflict(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/grad_conflict.png",
) -> None:
    """Plot gradient conflict (cosine similarity) between adjacent Linear layers.

    For each adjacent pair (L_i, L_{i+1}) sharing hidden dimension h_i, the
    conflict score is cosine_similarity(G_i.mean(dim=1), G_{i+1}.mean(dim=0))
    where G_i is the mean gradient matrix of L_i.  Near +1 = layers cooperate
    on the same hidden units; near -1 = gradient conflict (opposing pressures).

    Silently skipped for single-layer networks (no adjacent pairs) or results
    whose ``grad_conflict`` dict is empty.  One subplot per (dataset × model).
    """
    combos = [(ds, mdl) for ds in dataset_names for mdl in model_names]
    n_rows = len(combos)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 1, figsize=(9, 4.5 * n_rows), squeeze=False)
    fig.suptitle("Gradient Conflict Between Adjacent Layers\n"
                 "(cosine similarity along shared hidden dimension; −1 = full conflict)",
                 fontsize=12, fontweight="bold")

    for row, (ds_name, mdl_name) in enumerate(combos):
        ax = axes[row][0]
        ax.set_title(f"{ds_name} / {mdl_name}", fontsize=10)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Cosine similarity", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        for opt_name in optimizer_names:
            key3 = (ds_name, mdl_name, opt_name)
            if key3 not in results:
                continue
            history = results[key3]
            gc_dict = history.get("grad_conflict", {})
            if not gc_dict:
                continue

            color = opt_colors[opt_name]
            # One line per adjacent pair; distinguishable via linestyle when many pairs
            linestyles = ["-", "--", "-.", ":"]
            for li, (pair_key, vals) in enumerate(gc_dict.items()):
                if not vals:
                    continue
                valid_epochs = [e + 1 for e, v in enumerate(vals) if not math.isnan(v)]
                valid_vals   = [v for v in vals if not math.isnan(v)]
                if not valid_vals:
                    continue
                ls = linestyles[li % len(linestyles)]
                label = f"{opt_name} ({pair_key})" if len(gc_dict) > 1 else opt_name
                ax.plot(valid_epochs, valid_vals, color=color, linestyle=ls,
                        label=label, linewidth=1.8, marker="o", markersize=4)

        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Gradient conflict figure saved to {save_path}")

    plt.show()


def plot_lr_sensitivity_curves(
    results: dict,
    save_path: str | None = "plots/lr_curves.png",
) -> None:
    """Plot full training curves overlaid for each LR value per optimizer.

    Unlike ``plot_lr_sensitivity`` (which shows only final accuracy vs LR),
    this overlays the complete test-accuracy and train-loss epoch trajectories
    for every LR value, letting you compare *how fast* each LR converges, not
    just where it ends up.

    Grouping: series names are split on ``" lr="`` to find the base optimizer
    name. All series sharing the same base optimizer are overlaid in one subplot,
    coloured by LR via a sequential colormap.  Silently skipped when no
    ``config_lr`` is found in results.

    Parameters
    ----------
    results   : {(dataset_name, model_name, series_name): history_dict}
    save_path : file path to save; ``None`` / ``''`` disables saving.
    """
    lr_entries = {k: v for k, v in results.items() if v.get("config_lr") is not None}
    if not lr_entries:
        print("\n  (LR sensitivity curves skipped: no config_lr found in results)")
        return

    # Group by (dataset, model, base_optimizer)
    groups: dict[tuple, list[tuple]] = {}
    for (ds, mdl, series), hist in lr_entries.items():
        base_opt = series.split(" lr=")[0].split(" wd=")[0]
        key = (ds, mdl, base_opt)
        groups.setdefault(key, []).append((hist.get("config_lr", 0.0), series, hist))

    if not groups:
        return

    # Two columns: train_loss | test_acc
    n_rows = len(groups)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows), squeeze=False)
    fig.suptitle("LR Sensitivity: Full Training Curves", fontsize=13, fontweight="bold")

    for row, ((ds_name, mdl_name, base_opt), entries) in enumerate(sorted(groups.items())):
        entries.sort(key=lambda t: t[0])  # sort by LR ascending
        lrs_in_group = [e[0] for e in entries]

        # Colormap: map LR index → colour (blue shades or tab colormap)
        n = len(entries)
        cmap = plt.get_cmap("viridis" if n > 2 else "Set1")
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

        ax_loss = axes[row][0]
        ax_acc  = axes[row][1]
        ax_loss.set_title(f"{ds_name} / {mdl_name} / {base_opt} — Train Loss", fontsize=9)
        ax_acc.set_title(f"{ds_name} / {mdl_name} / {base_opt} — Test Acc (%)", fontsize=9)
        ax_loss.set_xlabel("Epoch", fontsize=9)
        ax_acc.set_xlabel("Epoch", fontsize=9)
        ax_loss.set_ylabel("Train Loss", fontsize=9)
        ax_acc.set_ylabel("Test Accuracy (%)", fontsize=9)

        for (lr_val, series, hist), color in zip(entries, colors):
            train_loss = hist.get("train_loss", [])
            test_acc   = hist.get("test_acc", [])
            epochs     = list(range(1, max(len(train_loss), len(test_acc)) + 1))
            lr_label   = f"lr={lr_val:.0e}" if lr_val < 0.01 else f"lr={lr_val}"

            if train_loss:
                ax_loss.plot(epochs[:len(train_loss)], train_loss, "o-",
                             color=color, label=lr_label, linewidth=1.8, markersize=4)
                std_vals = hist.get("train_loss_std", [])
                if std_vals and len(std_vals) == len(train_loss):
                    lo = [v - s for v, s in zip(train_loss, std_vals)]
                    hi = [v + s for v, s in zip(train_loss, std_vals)]
                    ax_loss.fill_between(epochs[:len(train_loss)], lo, hi,
                                         color=color, alpha=0.15)

            if test_acc:
                acc_pct = [v * 100 for v in test_acc]
                ax_acc.plot(epochs[:len(acc_pct)], acc_pct, "o-",
                            color=color, label=lr_label, linewidth=1.8, markersize=4)
                std_vals = hist.get("test_acc_std", [])
                if std_vals and len(std_vals) == len(test_acc):
                    lo = [(v - s) * 100 for v, s in zip(test_acc, std_vals)]
                    hi = [(v + s) * 100 for v, s in zip(test_acc, std_vals)]
                    ax_acc.fill_between(epochs[:len(acc_pct)], lo, hi,
                                        color=color, alpha=0.15)

        ax_loss.legend(fontsize=8)
        ax_acc.legend(fontsize=8)
        ax_loss.tick_params(labelsize=8)
        ax_acc.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  LR sensitivity curves saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Dead neuron fraction
# ---------------------------------------------------------------------------

def plot_dead_neurons(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/dead_neurons.png",
) -> None:
    """Plot the dead ReLU neuron fraction per epoch.

    A dead neuron is a ReLU unit that produces zero output for every sample
    in the validation set.  A high dead fraction means a large portion of
    the network's capacity is wasted.

    The per-ReLU-layer fractions are averaged across all ReLU layers to give
    a single scalar per epoch per optimizer, making comparison easy.  Runs
    with no ``dead_neurons`` data are silently skipped.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of series names (optimizer labels)
    opt_colors     : series_name → hex color
    save_path      : file path to save the figure (None = don't save)
    """
    n_ds  = len(dataset_names)
    n_mdl = len(model_names)
    n_cols = max(1, n_ds * n_mdl)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
    fig.suptitle("Dead ReLU Neuron Fraction vs Epoch", fontsize=13, fontweight="bold")

    for col, (ds, mdl) in enumerate(
        (ds, mdl) for ds in dataset_names for mdl in model_names
    ):
        ax = axes[0][col]
        ax.set_title(f"{ds} / {mdl}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dead neuron fraction")
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.grid(True, alpha=0.3)

        for series in optimizer_names:
            hist = results.get((ds, mdl, series), {})
            dn_dict = hist.get("dead_neurons", {})
            if not dn_dict:
                continue
            # Average across all ReLU layers per epoch
            n_layers = max(len(v) for v in dn_dict.values()) if dn_dict else 0
            if n_layers == 0:
                continue
            avg_per_epoch = []
            for ep_idx in range(n_layers):
                vals = [v[ep_idx] for v in dn_dict.values()
                        if ep_idx < len(v) and not (v[ep_idx] != v[ep_idx])]
                avg_per_epoch.append(sum(vals) / len(vals) if vals else float("nan"))
            epochs = list(range(1, len(avg_per_epoch) + 1))
            color  = opt_colors.get(series, "#333333")
            ax.plot(epochs, avg_per_epoch, "o-", color=color, label=series,
                    linewidth=1.8, markersize=4)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Dead neuron figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Weight effective rank
# ---------------------------------------------------------------------------

def plot_weight_rank(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/weight_rank.png",
) -> None:
    """Plot the effective weight matrix rank vs epoch.

    The effective rank of a weight matrix W is computed as exp(H(σ)) where
    H(σ) is the Shannon entropy of the normalised singular values.  Higher
    rank = more diverse (full-capacity) representations; lower rank = more
    collapsed or redundant representations.

    The per-layer ranks are averaged across all tracked layers to give a
    single scalar per epoch per optimizer.  Runs with no ``weight_rank``
    data are silently skipped.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of series names (optimizer labels)
    opt_colors     : series_name → hex color
    save_path      : file path to save the figure (None = don't save)
    """
    n_ds  = len(dataset_names)
    n_mdl = len(model_names)
    n_cols = max(1, n_ds * n_mdl)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
    fig.suptitle("Effective Weight Matrix Rank vs Epoch", fontsize=13, fontweight="bold")

    for col, (ds, mdl) in enumerate(
        (ds, mdl) for ds in dataset_names for mdl in model_names
    ):
        ax = axes[0][col]
        ax.set_title(f"{ds} / {mdl}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Effective rank (exp(H(σ)))")
        ax.grid(True, alpha=0.3)

        for series in optimizer_names:
            hist = results.get((ds, mdl, series), {})
            wr_dict = hist.get("weight_rank", {})
            if not wr_dict:
                continue
            n_epochs = max(len(v) for v in wr_dict.values()) if wr_dict else 0
            if n_epochs == 0:
                continue
            avg_per_epoch = []
            for ep_idx in range(n_epochs):
                vals = [v[ep_idx] for v in wr_dict.values()
                        if ep_idx < len(v) and not (v[ep_idx] != v[ep_idx])]
                avg_per_epoch.append(sum(vals) / len(vals) if vals else float("nan"))
            epochs = list(range(1, len(avg_per_epoch) + 1))
            color  = opt_colors.get(series, "#333333")
            ax.plot(epochs, avg_per_epoch, "o-", color=color, label=series,
                    linewidth=1.8, markersize=4)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Weight rank figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Gradient noise scale
# ---------------------------------------------------------------------------

def plot_grad_noise_scale(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/grad_noise_scale.png",
) -> None:
    """Plot the gradient noise scale per epoch.

    Gradient noise scale = ||E[g]||² / E[||g||²].  Values near 1 indicate
    low noise (gradient directions agree across mini-batches); values near 0
    indicate high noise (mini-batch gradients cancel each other).

    Runs with no ``grad_noise_scale`` data or all-NaN values are silently
    skipped.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of series names (optimizer labels)
    opt_colors     : series_name → hex color
    save_path      : file path to save the figure (None = don't save)
    """
    import math as _math
    n_ds  = len(dataset_names)
    n_mdl = len(model_names)
    n_cols = max(1, n_ds * n_mdl)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
    fig.suptitle("Gradient Noise Scale vs Epoch", fontsize=13, fontweight="bold")

    for col, (ds, mdl) in enumerate(
        (ds, mdl) for ds in dataset_names for mdl in model_names
    ):
        ax = axes[0][col]
        ax.set_title(f"{ds} / {mdl}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Grad noise scale (||E[g]||² / E[||g||²])")
        ax.set_ylim(-0.02, 1.05)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.grid(True, alpha=0.3)

        for series in optimizer_names:
            hist = results.get((ds, mdl, series), {})
            gns  = hist.get("grad_noise_scale", [])
            if not gns:
                continue
            # Skip if all values are NaN
            finite = [v for v in gns if not _math.isnan(v)]
            if not finite:
                continue
            epochs = list(range(1, len(gns) + 1))
            color  = opt_colors.get(series, "#333333")
            ax.plot(epochs, gns, "o-", color=color, label=series,
                    linewidth=1.8, markersize=4)
            # Error band if std key present
            std_vals = hist.get("grad_noise_scale_std", [])
            if std_vals and len(std_vals) == len(gns):
                lo = [max(0, v - s) for v, s in zip(gns, std_vals)]
                hi = [min(1, v + s) for v, s in zip(gns, std_vals)]
                ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)

        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Gradient noise scale figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Fisher information trace
# ---------------------------------------------------------------------------

def plot_fisher_trace(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict[str, str],
    save_path: str | None = "plots/fisher_trace.png",
) -> None:
    """Plot the Fisher information trace per epoch.

    Fisher trace = E[||∇_θ log p(y|x)||²] — the expected squared gradient
    norm over the data distribution.  It measures the average curvature seen
    by the optimizer without requiring second-order backprop.

    A decreasing Fisher trace during training often indicates convergence
    (the model is approaching a flat region of the loss).

    Runs with no ``fisher_trace`` data or all-NaN values are silently skipped.

    Parameters
    ----------
    results        : {(dataset_name, model_name, series_name): history_dict}
    dataset_names  : ordered list of dataset names
    model_names    : ordered list of model names
    optimizer_names: ordered list of series names (optimizer labels)
    opt_colors     : series_name → hex color
    save_path      : file path to save the figure (None = don't save)
    """
    import math as _math
    n_ds  = len(dataset_names)
    n_mdl = len(model_names)
    n_cols = max(1, n_ds * n_mdl)

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), squeeze=False)
    fig.suptitle("Fisher Information Trace vs Epoch", fontsize=13, fontweight="bold")

    for col, (ds, mdl) in enumerate(
        (ds, mdl) for ds in dataset_names for mdl in model_names
    ):
        ax = axes[0][col]
        ax.set_title(f"{ds} / {mdl}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fisher trace (E[||∇ log p||²])")
        ax.grid(True, alpha=0.3)

        any_plotted = False
        for series in optimizer_names:
            hist = results.get((ds, mdl, series), {})
            ft   = hist.get("fisher_trace", [])
            if not ft:
                continue
            finite = [v for v in ft if not _math.isnan(v)]
            if not finite:
                continue
            epochs = list(range(1, len(ft) + 1))
            color  = opt_colors.get(series, "#333333")
            ax.plot(epochs, ft, "o-", color=color, label=series,
                    linewidth=1.8, markersize=4)
            std_vals = hist.get("fisher_trace_std", [])
            if std_vals and len(std_vals) == len(ft):
                lo = [max(0, v - s) for v, s in zip(ft, std_vals)]
                hi = [v + s for v, s in zip(ft, std_vals)]
                ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)
            any_plotted = True

        if any_plotted:
            ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Fisher trace figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Gradient alignment (first ↔ last layer cosine similarity)
# ---------------------------------------------------------------------------

def plot_grad_alignment(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict,
    save_path: str | None = "plots/grad_alignment.png",
) -> None:
    """Plot per-epoch cosine similarity between the first and last linear
    layer gradient directions.

    Values near +1 indicate end-to-end cooperative updates; near -1 suggest
    gradient reversal.  NaN points (e.g. epoch 1 or single-layer models) are
    silently omitted.  The y-axis is fixed to [-1.05, 1.05] with a dashed
    zero reference line for easy interpretation.

    Parameters
    ----------
    results        : dict mapping (dataset, model, series) → history
    dataset_names  : list of dataset display names to include
    model_names    : list of model names to include
    optimizer_names: list of series names to include
    opt_colors     : {series_name: color_str}
    save_path      : file path to save the figure (None = do not save)
    """
    n_ds  = len(dataset_names)
    n_mdl = len(model_names)
    fig, axes = plt.subplots(n_ds, n_mdl, figsize=(6 * n_mdl, 4 * n_ds),
                             squeeze=False)
    fig.suptitle("Gradient Alignment (first ↔ last layer cosine similarity)",
                 fontsize=13, fontweight="bold")

    for row, ds in enumerate(dataset_names):
        for col, mdl in enumerate(model_names):
            ax = axes[row][col]
            ax.set_title(f"{ds} / {mdl}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cosine similarity")
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylim(-1.05, 1.05)

            for opt_name in optimizer_names:
                key = (ds, mdl, opt_name)
                if key not in results:
                    continue
                vals = results[key].get("grad_alignment", [])
                if not vals:
                    continue
                # Filter out NaN points
                epochs = [i + 1 for i, v in enumerate(vals)
                          if not (isinstance(v, float) and v != v)]
                valid  = [v for v in vals
                          if not (isinstance(v, float) and v != v)]
                if not epochs:
                    continue
                color = opt_colors.get(opt_name, "#333333")
                ax.plot(epochs, valid, color=color, label=opt_name,
                        linewidth=1.8, marker="o", markersize=4)

                # Error bands if multi-seed std is present
                std_vals = results[key].get("grad_alignment_std", [])
                if std_vals:
                    std_valid = [std_vals[i - 1] for i in epochs
                                 if (i - 1) < len(std_vals)]
                    valid_np  = [valid[k] for k in range(len(epochs))]
                    if len(std_valid) == len(valid_np):
                        ax.fill_between(
                            epochs,
                            [v - s for v, s in zip(valid_np, std_valid)],
                            [v + s for v, s in zip(valid_np, std_valid)],
                            alpha=0.15, color=color,
                        )

            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Gradient alignment figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Spectral norm (σ_max per layer per epoch)
# ---------------------------------------------------------------------------

def plot_spectral_norm(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict,
    save_path: str | None = "plots/spectral_norm.png",
) -> None:
    """Plot per-epoch largest singular value (σ_max) averaged across all
    Linear layers for each optimizer.

    High spectral norm = potentially unstable layer (large Lipschitz constant).
    Declining spectral norm across epochs suggests progressive regularisation.

    Parameters
    ----------
    results        : dict mapping (dataset, model, series) → history
    dataset_names  : list of dataset display names to include
    model_names    : list of model names to include
    optimizer_names: list of series names to include
    opt_colors     : {series_name: color_str}
    save_path      : file path to save the figure (None = do not save)
    """
    import math as _math

    n_ds  = len(dataset_names)
    n_mdl = len(model_names)
    fig, axes = plt.subplots(n_ds, n_mdl, figsize=(6 * n_mdl, 4 * n_ds),
                             squeeze=False)
    fig.suptitle("Spectral Norm σ_max(W) — mean across Linear layers",
                 fontsize=13, fontweight="bold")

    for row, ds in enumerate(dataset_names):
        for col, mdl in enumerate(model_names):
            ax = axes[row][col]
            ax.set_title(f"{ds} / {mdl}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("σ_max (mean over layers)")

            for opt_name in optimizer_names:
                key = (ds, mdl, opt_name)
                if key not in results:
                    continue
                sn_dict = results[key].get("spectral_norm", {})
                if not sn_dict:
                    continue

                # Average σ_max across all layers per epoch.
                num_epochs = max(len(v) for v in sn_dict.values())
                avg_vals = []
                for ep_idx in range(num_epochs):
                    ep_vals = [
                        v[ep_idx] for v in sn_dict.values()
                        if ep_idx < len(v) and not _math.isnan(v[ep_idx])
                    ]
                    avg_vals.append(
                        sum(ep_vals) / len(ep_vals) if ep_vals else float("nan")
                    )

                epochs = list(range(1, num_epochs + 1))
                color  = opt_colors.get(opt_name, "#333333")
                valid_e = [e for e, v in zip(epochs, avg_vals)
                           if not _math.isnan(v)]
                valid_v = [v for v in avg_vals if not _math.isnan(v)]
                if valid_e:
                    ax.plot(valid_e, valid_v, color=color, label=opt_name,
                            linewidth=1.8, marker="o", markersize=4)

            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Spectral norm figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Optimizer state entropy
# ---------------------------------------------------------------------------

def plot_optimizer_state_entropy(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict,
    save_path: str | None = "plots/optimizer_entropy.png",
) -> None:
    """Plot per-epoch Shannon entropy of the optimizer's momentum/variance
    buffers.

    High entropy indicates the optimizer is still exploring; entropy collapse
    near convergence is a useful early-stopping signal and a sign that the
    model has settled into a local minimum.

    NaN epochs (disabled or unsupported optimizers) are silently skipped.
    Error bands are shown when multi-seed std data is available.

    Parameters
    ----------
    results        : dict mapping (dataset, model, series) → history
    dataset_names  : list of dataset display names to include
    model_names    : list of model names to include
    optimizer_names: list of series names to include
    opt_colors     : {series_name: color_str}
    save_path      : file path to save the figure (None = do not save)
    """
    import math as _math

    n_ds  = len(dataset_names)
    n_mdl = len(model_names)
    fig, axes = plt.subplots(n_ds, n_mdl, figsize=(6 * n_mdl, 4 * n_ds),
                             squeeze=False)
    fig.suptitle("Optimizer State Entropy (Shannon entropy of momentum/variance buffers)",
                 fontsize=13, fontweight="bold")

    for row, ds in enumerate(dataset_names):
        for col, mdl in enumerate(model_names):
            ax = axes[row][col]
            ax.set_title(f"{ds} / {mdl}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Entropy (nats)")

            for opt_name in optimizer_names:
                key = (ds, mdl, opt_name)
                if key not in results:
                    continue
                vals = results[key].get("optimizer_state_entropy", [])
                if not vals:
                    continue
                # Skip series that are all-NaN (optimizer has no state buffers).
                if all(_math.isnan(v) for v in vals if isinstance(v, float)):
                    continue
                epochs = [i + 1 for i, v in enumerate(vals)
                          if not (isinstance(v, float) and _math.isnan(v))]
                valid  = [v for v in vals
                          if not (isinstance(v, float) and _math.isnan(v))]
                if not epochs:
                    continue
                color = opt_colors.get(opt_name, "#333333")
                ax.plot(epochs, valid, color=color, label=opt_name,
                        linewidth=1.8, marker="o", markersize=4)

                std_vals = results[key].get("optimizer_state_entropy_std", [])
                if std_vals:
                    std_valid = [std_vals[i - 1] for i in epochs
                                 if (i - 1) < len(std_vals)]
                    if len(std_valid) == len(valid):
                        ax.fill_between(
                            epochs,
                            [v - s for v, s in zip(valid, std_valid)],
                            [v + s for v, s in zip(valid, std_valid)],
                            alpha=0.15, color=color,
                        )

            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Optimizer state entropy figure saved to {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Plasticity score
# ---------------------------------------------------------------------------

def plot_plasticity(
    results: dict,
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    opt_colors: dict,
    save_path: str | None = "plots/plasticity.png",
) -> None:
    """Plot per-epoch plasticity score (recovery accuracy after last-layer
    reset + brief fine-tuning).

    Epochs where the plasticity test was not run (NaN values) are omitted,
    producing a sparse x-axis tick pattern when ``plasticity_interval > 1``.
    A score near the network's test accuracy means the model is still fully
    plastic; a score that is much lower indicates the representations have
    become rigid (loss of plasticity).

    Parameters
    ----------
    results        : dict mapping (dataset, model, series) → history
    dataset_names  : list of dataset display names to include
    model_names    : list of model names to include
    optimizer_names: list of series names to include
    opt_colors     : {series_name: color_str}
    save_path      : file path to save the figure (None = do not save)
    """
    import math as _math

    n_ds  = len(dataset_names)
    n_mdl = len(model_names)
    fig, axes = plt.subplots(n_ds, n_mdl, figsize=(6 * n_mdl, 4 * n_ds),
                             squeeze=False)
    fig.suptitle("Plasticity Score (last-layer reset recovery accuracy)",
                 fontsize=13, fontweight="bold")

    for row, ds in enumerate(dataset_names):
        for col, mdl in enumerate(model_names):
            ax = axes[row][col]
            ax.set_title(f"{ds} / {mdl}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Recovery accuracy")
            ax.set_ylim(0.0, 1.05)

            for opt_name in optimizer_names:
                key = (ds, mdl, opt_name)
                if key not in results:
                    continue
                vals = results[key].get("plasticity", [])
                if not vals:
                    continue
                # Filter out NaN epochs (where score was not measured).
                epochs = [i + 1 for i, v in enumerate(vals)
                          if not (isinstance(v, float) and _math.isnan(v))]
                valid  = [v for v in vals
                          if not (isinstance(v, float) and _math.isnan(v))]
                if not epochs:
                    continue
                color = opt_colors.get(opt_name, "#333333")
                ax.plot(epochs, valid, color=color, label=opt_name,
                        linewidth=1.8, marker="o", markersize=6)

                std_vals = results[key].get("plasticity_std", [])
                if std_vals:
                    std_valid = [std_vals[i - 1] for i in epochs
                                 if (i - 1) < len(std_vals)]
                    if len(std_valid) == len(valid):
                        ax.fill_between(
                            epochs,
                            [max(0, v - s) for v, s in zip(valid, std_valid)],
                            [min(1, v + s) for v, s in zip(valid, std_valid)],
                            alpha=0.15, color=color,
                        )

            ax.legend(fontsize=8)
            ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plasticity score figure saved to {save_path}")

    plt.show()
