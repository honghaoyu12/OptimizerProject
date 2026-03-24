"""Benchmark report generator.

Produces a human-readable Markdown summary of benchmark results.

Usage::

    from report import generate_report
    md = generate_report(results, config, save_path="reports/benchmark_report.md")
"""

import os
import statistics
from datetime import datetime

from visualizer import _compute_lr_sensitivity


def _paired_ttest(a: list[float], b: list[float]) -> float | None:
    """Run a paired t-test between two lists of per-seed accuracies.

    Returns the two-tailed p-value, or None when scipy is unavailable or the
    lists are too short / identical.  A small p-value (< 0.05) means the
    difference in means is unlikely to be due to random seed variation alone.

    Parameters
    ----------
    a, b:
        Per-seed final test accuracies for two optimizers.  Both lists must
        have the same length and at least 2 elements.

    Returns
    -------
    float | None
        Two-tailed p-value in [0, 1], or None on failure.
    """
    if len(a) != len(b) or len(a) < 2:
        return None
    try:
        from scipy.stats import ttest_rel  # optional dependency
        _, p = ttest_rel(a, b)
        return float(p)
    except (ImportError, ValueError):
        return None


def generate_report(
    results: dict,
    config: dict,
    save_path: str,
) -> str:
    """Generate a Markdown benchmark report.

    Parameters
    ----------
    results:
        Mapping of ``(dataset_name, model_name, series_name)`` → history dict.
        Each history dict must contain at least::

            train_loss    : list[float]  (one entry per epoch)
            train_acc     : list[float]
            test_acc      : list[float]
            time_elapsed  : list[float]  (cumulative seconds)

        Optional keys: ``early_stopped_epoch`` (int | None),
        ``test_acc_ema`` (list[float]), ``swa_final_acc`` (float | None),
        ``test_acc_std`` (list[float] — present after multi-seed aggregation),
        ``test_acc_final_seeds`` (list[float] — per-seed final accuracies for
        statistical significance testing),
        ``convergence_epochs`` / ``convergence_times`` (dict[str, float|None]
        — epoch / seconds to reach each accuracy milestone, e.g. ``"90%"``),
        ``ece`` (list[float] — Expected Calibration Error per epoch; 0 = perfect
        calibration, 1 = maximally miscalibrated).

    config:
        Benchmark settings. Recognised keys:
        ``epochs``, ``batch_size``, ``scheduler``, ``weight_decays``,
        ``lrs``, ``seed``, ``num_seeds``, ``target_acc``,
        ``label_smoothing``, ``swa_start``.
    save_path:
        File path to write the report.  Pass ``''`` to skip writing.

    Returns
    -------
    str
        The complete Markdown text (always returned, regardless of save_path).
    """
    lines: list[str] = []

    # ------------------------------------------------------------------
    # Section 1 — Header
    # ------------------------------------------------------------------
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines += [
        "# Benchmark Report",
        f"Generated: {now}",
        "",
    ]

    # ------------------------------------------------------------------
    # Section 2 — Setup
    # ------------------------------------------------------------------
    # Derive the unique dataset, model, and optimizer names from the
    # results dict keys, which are (dataset, model, series) 3-tuples.
    # sorted() ensures consistent ordering independent of insertion order.
    datasets   = sorted({k[0] for k in results})
    models     = sorted({k[1] for k in results})
    optimizers = sorted({k[2] for k in results})

    # Pull benchmark config values with safe defaults for optional fields.
    epochs           = config.get("epochs", "N/A")
    batch_size       = config.get("batch_size", "N/A")
    scheduler        = config.get("scheduler", "N/A")
    wds              = config.get("weight_decays", [0.0])
    lrs              = config.get("lrs", None)
    seed             = config.get("seed", None)
    num_seeds        = config.get("num_seeds", 1)
    target_acc       = config.get("target_acc", 0.95)
    label_smoothing  = config.get("label_smoothing", 0.0)
    # Convert the [0.0, 1.0] float target accuracy to an integer percentage
    # for display (e.g. 0.95 → 95).
    target_pct       = int(round(target_acc * 100))

    # Format config values as human-readable strings for the Markdown table.
    wd_str   = ", ".join(str(w) for w in (wds or [0.0]))
    # lrs=None means "use each optimizer's own default LR" (no global sweep)
    lr_str   = ", ".join(str(l) for l in lrs) if lrs else "per-optimizer defaults"
    seed_str = f"{num_seeds} (averaged)" if num_seeds > 1 else "1 (single run)"
    ls_str   = str(label_smoothing) if label_smoothing > 0.0 else "0.0 (disabled)"

    lines += [
        "## Setup",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Datasets | {', '.join(datasets)} |",
        f"| Models | {', '.join(models)} |",
        f"| Optimizers | {', '.join(optimizers)} |",
        f"| Epochs | {epochs} |",
        f"| Batch size | {batch_size} |",
        f"| Scheduler | {scheduler} |",
        f"| Weight decays | {wd_str} |",
        f"| LRs | {lr_str} |",
        f"| Label smoothing | {ls_str} |",
        f"| Seed | {seed} |",
        f"| Seeds | {seed_str} |",
        f"| Target acc | {target_pct}% |",
        "",
    ]

    # ------------------------------------------------------------------
    # Section 3 — Results Overview
    # ------------------------------------------------------------------
    lines += ["## Results", ""]

    # combos: all unique (dataset, model) pairs in the results.
    # We iterate over them in sorted order to produce one table per combo.
    combos = sorted({(k[0], k[1]) for k in results})
    for ds, mdl in combos:
        lines += [f"### {ds} / {mdl}", ""]

        # Check whether ANY run in this (dataset, model) combo used EMA or SWA.
        # We only add those columns to the table header when at least one run
        # has the corresponding data, keeping the table compact otherwise.
        combo_has_ema = any(
            bool(hist.get("test_acc_ema"))
            for (d, m, _), hist in results.items()
            if d == ds and m == mdl
        )
        combo_has_swa = any(
            hist.get("swa_final_acc") is not None
            for (d, m, _), hist in results.items()
            if d == ds and m == mdl
        )
        # Build the optional column header fragments:
        # "EMA Acc (%)" = Polyak-averaged model accuracy
        # "EMA Gap (pp)" = improvement over the base model in percentage points
        ema_header = " EMA Acc (%) | EMA Gap (pp) |" if combo_has_ema else ""
        ema_sep    = "---|---|" if combo_has_ema else ""
        swa_header = " SWA Acc (%) |" if combo_has_swa else ""
        swa_sep    = "---|" if combo_has_swa else ""
        lines += [
            f"| Optimizer | Final Test Acc (%) | Best Test Acc (%) | Epochs to {target_pct}% | Time to {target_pct}% (s) | Epochs Run | Time (s) | Train-Test Gap (%) | Final ECE |{ema_header}{swa_header}",
            f"|---|---|---|---|---|---|---|---|---|{ema_sep}{swa_sep}",
        ]

        # Collect one row per series (optimizer) for this combo,
        # then sort by final_acc descending so the best optimizer is at the top.
        rows = []
        for (d, m, series), hist in results.items():
            if d != ds or m != mdl:
                continue
            test_acc     = hist.get("test_acc", [])
            test_acc_std = hist.get("test_acc_std", [])   # present after multi-seed averaging
            test_acc_ema = hist.get("test_acc_ema", [])   # EMA-smoothed accuracy per epoch
            train_acc    = hist.get("train_acc", [])
            train_loss   = hist.get("train_loss", [])
            elapsed      = hist.get("time_elapsed", [])   # cumulative seconds per epoch

            # Final accuracy = last epoch's test accuracy (×100 for %)
            final_acc  = test_acc[-1] * 100  if test_acc  else float("nan")
            # best_idx: index (epoch) at which test accuracy was highest.
            # Used to look up the corresponding std at the best epoch (not the last).
            best_idx   = test_acc.index(max(test_acc)) if test_acc else 0
            best_acc   = max(test_acc) * 100  if test_acc  else float("nan")
            ep_run     = len(train_loss)
            total_time = elapsed[-1]           if elapsed   else float("nan")
            # Train-test gap in percentage points: positive = overfitting
            gap        = ((train_acc[-1] - test_acc[-1]) * 100
                          if train_acc and test_acc else float("nan"))

            # std_present: True when we have seed-averaged std data.
            # final_std is the std at the last epoch; best_std at the best epoch.
            # Using best_idx for best_std (not just max of std list) correctly
            # pairs the best accuracy with its corresponding uncertainty.
            std_present = bool(test_acc_std)
            final_std   = test_acc_std[-1] * 100   if std_present else None
            best_std    = test_acc_std[best_idx] * 100 if std_present else None

            # Format as "mean ± std" when std data is available, or plain "mean" otherwise
            final_str = f"{final_acc:.2f} ± {final_std:.2f}" if final_std is not None else f"{final_acc:.2f}"
            best_str  = f"{best_acc:.2f} ± {best_std:.2f}"   if best_std  is not None else f"{best_acc:.2f}"

            # Convergence epoch/time: first epoch at which test_acc ≥ target_acc.
            # Stored as a scalar (averaged over seeds).  "—" when never reached.
            conv_epoch = hist.get("target_accuracy_epoch")
            conv_time  = hist.get("target_accuracy_time")
            conv_ep_str   = f"{conv_epoch:.1f}" if conv_epoch is not None else "—"
            conv_time_str = f"{conv_time:.1f}"  if conv_time  is not None else "—"

            # EMA accuracy: accuracy of the Polyak-averaged model on the test set.
            # ema_gap_pp: how many percentage points the EMA model exceeds the
            # raw model (positive = EMA is better; large gaps suggest the
            # raw model was oscillating and EMA smoothed it out).
            ema_acc_pct = test_acc_ema[-1] * 100 if test_acc_ema else None
            ema_gap_pp  = (ema_acc_pct - final_acc) if ema_acc_pct is not None else None

            # SWA accuracy: single scalar computed at the end of training by
            # averaging weights from the cosine-restart peaks.
            swa_acc_raw = hist.get("swa_final_acc")
            swa_acc_pct = swa_acc_raw * 100 if swa_acc_raw is not None else None

            # ECE: Expected Calibration Error at the last epoch.
            # 0.0 = perfectly calibrated; values > 0.15 are typically poor.
            ece_list = hist.get("ece", [])
            final_ece = ece_list[-1] if ece_list else None

            rows.append((series, final_acc, final_str, best_str, conv_ep_str, conv_time_str, ep_run, total_time, gap, final_ece, ema_acc_pct, ema_gap_pp, swa_acc_pct))

        # Sort rows by final_acc descending so the best optimizer is first in the table.
        # The second element of each row tuple (index 1) is the float final_acc.
        rows.sort(key=lambda r: r[1], reverse=True)
        for series, _, final_str, best_str, conv_ep_str, conv_time_str, ep_run, total_time, gap, final_ece, ema_acc_pct, ema_gap_pp, swa_acc_pct in rows:
            ece_str = f"{final_ece:.4f}" if final_ece is not None else "—"
            # Build optional EMA and SWA column fragments only if those columns exist.
            # Using an f-string fragment avoids shifting column count for tables
            # that don't have EMA/SWA data at all.
            ema_part = ""
            if combo_has_ema:
                ema_acc_str = f"{ema_acc_pct:.2f}" if ema_acc_pct is not None else "—"
                # :+.2f prints the sign explicitly (e.g. "+0.31" or "-0.12")
                # making it obvious whether EMA helped or hurt
                ema_gap_str = f"{ema_gap_pp:+.2f}" if ema_gap_pp is not None else "—"
                ema_part = f" {ema_acc_str} | {ema_gap_str} |"
            swa_part = ""
            if combo_has_swa:
                swa_str = f"{swa_acc_pct:.2f}" if swa_acc_pct is not None else "—"
                swa_part = f" {swa_str} |"
            lines.append(
                f"| {series} | {final_str} | {best_str} | {conv_ep_str} | {conv_time_str} | {ep_run} | {total_time:.1f} | {gap:.2f} | {ece_str} |{ema_part}{swa_part}"
            )
        lines.append("")

    # ------------------------------------------------------------------
    # Section 4 — Rankings
    # Single-sentence medals for the best optimizer on each axis.
    # ------------------------------------------------------------------
    lines += ["## Rankings", ""]

    for ds, mdl in combos:
        lines += [f"### {ds} / {mdl}", ""]

        # subset: {series_name: history_dict} for this (dataset, model) combo
        subset = {
            series: hist
            for (d, m, series), hist in results.items()
            if d == ds and m == mdl
        }
        if not subset:
            continue

        # Helper lambdas evaluate a specific scalar from a history dict.
        # float("-inf") / float("inf") are safe sentinels for max/min selection
        # when a particular history key is missing.

        def _final_acc(h):
            # Final test accuracy (last epoch) as a percentage
            ta = h.get("test_acc", [])
            return ta[-1] * 100 if ta else float("-inf")

        def _best_acc(h):
            # Peak test accuracy across all epochs
            ta = h.get("test_acc", [])
            return max(ta) * 100 if ta else float("-inf")

        def _time(h):
            # Total wall-clock training time (cumulative at the last epoch)
            el = h.get("time_elapsed", [])
            return el[-1] if el else float("inf")

        def _stability(h):
            # Standard deviation of the last 3 test accuracy values.
            # Low std = stable convergence; high std = still oscillating.
            # Tail of 3 is a heuristic: long enough to capture oscillations,
            # short enough not to be dominated by early-epoch noise.
            ta = h.get("test_acc", [])
            tail = ta[-3:] if len(ta) >= 3 else ta
            if len(tail) < 2:
                return 0.0   # single epoch — no variance to measure
            return statistics.stdev(tail)

        # Find the winner on each axis using the helper lambdas
        best_final   = max(subset, key=lambda s: _final_acc(subset[s]))
        fastest      = min(subset, key=lambda s: _time(subset[s]))
        best_peak    = max(subset, key=lambda s: _best_acc(subset[s]))
        most_stable  = min(subset, key=lambda s: _stability(subset[s]))

        lines += [
            f"- 🥇 **Best final accuracy**: {best_final} ({_final_acc(subset[best_final]):.2f}%)",
            f"- 🚀 **Fastest**: {fastest} ({_time(subset[fastest]):.1f} s)",
            f"- 📈 **Best peak accuracy**: {best_peak} ({_best_acc(subset[best_peak]):.2f}%)",
            f"- 🎯 **Most stable**: {most_stable} (std={_stability(subset[most_stable]):.4f})",
            "",
        ]

    # ------------------------------------------------------------------
    # Section 4.5 — LR Sensitivity (only when ≥2 LRs were swept)
    # Shows how much each optimizer's accuracy varies across LR choices.
    # The range (max acc − min acc) is the primary metric; lower = more robust.
    # ------------------------------------------------------------------
    # _compute_lr_sensitivity() returns an empty dict when no LR sweep was done
    # (i.e., when all series_names lack the "lr=X" suffix), so this section
    # is automatically omitted for single-LR or default-LR benchmarks.
    lr_scores = _compute_lr_sensitivity(results)
    if lr_scores:
        lines += ["## LR Sensitivity", ""]
        lines += [
            "Ranks optimizers by how much their final accuracy varies across the swept LR values.",
            "**Lower range = more robust to LR choice.**",
            "",
        ]

        for ds, mdl in combos:
            # Filter lr_scores to only the current (dataset, model) combo
            subset = {k[2]: v for k, v in lr_scores.items()
                      if k[0] == ds and k[1] == mdl}
            if not subset:
                continue
            lines += [f"### {ds} / {mdl}", ""]
            lines += [
                "| Optimizer | Range (pp) | Std (pp) | Best LR | Worst LR | LRs tested |",
                "|---|---|---|---|---|---|",
            ]
            # Sort by range ascending so the most robust optimizer (smallest range) is first
            for opt in sorted(subset, key=lambda o: subset[o]["range"]):
                s = subset[opt]
                lines.append(
                    f"| {opt} | {s['range']:.2f} | {s['std']:.2f} "
                    f"| {s['best_lr']:g} | {s['worst_lr']:g} | {s['n_lrs']} |"
                )
            lines.append("")

    # ------------------------------------------------------------------
    # Section 4.7 — Convergence Profile
    # Rows: one per optimizer.  Columns: epoch first reached each threshold
    # (50%, 75%, 90%, 95%, 99%).  "—" when the threshold was never reached.
    # ------------------------------------------------------------------
    # _CONV_THRESHOLDS is the same list used during training to record
    # convergence milestones; importing here ensures the labels always match.
    from train import _CONV_THRESHOLDS  # noqa: PLC0415 (localised import)
    thresh_labels = [f"{int(t * 100)}%" for t in _CONV_THRESHOLDS]

    # Only emit this section when at least one run has convergence_epochs data.
    # A run with no convergence_epochs simply had history["convergence_epochs"] = {}
    # (an empty dict), so bool({}) = False.
    any_conv = any(bool(hist.get("convergence_epochs")) for hist in results.values())
    if any_conv:
        lines += ["## Convergence Profile", ""]
        lines += [
            "Epoch at which each optimizer **first** reached each accuracy milestone.",
            "Values averaged over seeds when `--num-seeds > 1`.  '—' = never reached.",
            "",
        ]
        for ds, mdl in combos:
            lines += [f"### {ds} / {mdl}", ""]
            # Dynamic header: one column per accuracy threshold
            header = "| Optimizer |" + "".join(f" Ep to {t} |" for t in thresh_labels)
            sep    = "|---|" + "---|" * len(thresh_labels)
            lines += [header, sep]
            for (d, m, series), hist in sorted(results.items()):
                if d != ds or m != mdl:
                    continue
                conv_eps = hist.get("convergence_epochs", {})
                cells = ""
                for t in thresh_labels:
                    val = conv_eps.get(t)
                    # f"{val:.1f}" for numeric epoch; "—" when never reached
                    cells += f" {val:.1f} |" if val is not None else " — |"
                lines.append(f"| {series} |{cells}")
            lines.append("")

    # ------------------------------------------------------------------
    # Section 4.8 — Statistical Significance
    # Only shown when num_seeds > 1 and per-seed final accuracies are stored.
    # Produces a p-value matrix: for every ordered pair of optimizers on every
    # (dataset, model) combo, shows the paired t-test p-value.
    # ------------------------------------------------------------------
    # has_seeds_data: True only when _aggregate_histories() stored the per-seed
    # final accuracies in "test_acc_final_seeds".  This is always present when
    # --num-seeds > 1 and the benchmark ran successfully.
    has_seeds_data = any(
        bool(hist.get("test_acc_final_seeds"))
        for hist in results.values()
    )
    if num_seeds > 1 and has_seeds_data:
        lines += ["## Statistical Significance", ""]
        lines += [
            f"Paired t-test p-values for every optimizer pair ({num_seeds} seeds).",
            "**p < 0.05** indicates the performance difference is unlikely due to "
            "random initialisation alone.",
            "",
        ]
        for ds, mdl in combos:
            # All series (optimizer names) in this combo, sorted alphabetically
            combo_series = sorted(
                {s for (d, m, s) in results if d == ds and m == mdl}
            )
            if len(combo_series) < 2:
                continue   # need at least 2 optimizers to compare
            lines += [f"### {ds} / {mdl}", ""]
            # Square matrix header: column labels are the series names
            header = "| |" + "".join(f" {s} |" for s in combo_series)
            sep    = "|---|" + "---|" * len(combo_series)
            lines += [header, sep]
            for s_a in combo_series:
                row = f"| **{s_a}** |"
                for s_b in combo_series:
                    if s_a == s_b:
                        # Diagonal: comparing an optimizer to itself — skip
                        row += " — |"
                    else:
                        # Off-diagonal: retrieve per-seed final accuracies for each optimizer.
                        # .get() with a fallback dict and .get("...", []) guards against
                        # missing keys without raising KeyError.
                        seeds_a = (results.get((ds, mdl, s_a)) or {}).get("test_acc_final_seeds", [])
                        seeds_b = (results.get((ds, mdl, s_b)) or {}).get("test_acc_final_seeds", [])
                        p = _paired_ttest(seeds_a, seeds_b)
                        if p is None:
                            # scipy unavailable, or fewer than 2 seeds
                            row += " N/A |"
                        elif p < 0.001:
                            # Highly significant: bold the p-value for emphasis
                            row += f" **{p:.3f}** |"
                        elif p < 0.05:
                            # Significant: italicise (conventional threshold)
                            row += f" *{p:.3f}* |"
                        else:
                            # Not significant: plain text
                            row += f" {p:.3f} |"
                lines.append(row)
            lines.append("")

    # ------------------------------------------------------------------
    # Section 5 — Per-Optimizer Summary
    # One bullet point per (dataset, model) combo for each optimizer.
    # Flags overfitting (gap > 5%) and early stopping with inline markers.
    # ------------------------------------------------------------------
    lines += ["## Per-Optimizer Summary", ""]

    for series in optimizers:
        lines += [f"### {series}", ""]
        for (d, m, s), hist in sorted(results.items()):
            if s != series:
                continue
            test_acc     = hist.get("test_acc", [])
            test_acc_std = hist.get("test_acc_std", [])
            train_acc    = hist.get("train_acc", [])
            elapsed      = hist.get("time_elapsed", [])
            train_loss   = hist.get("train_loss", [])

            final_acc  = test_acc[-1] * 100  if test_acc   else float("nan")
            best_acc   = max(test_acc) * 100  if test_acc   else float("nan")
            ep_run     = len(train_loss)
            total_time = elapsed[-1]          if elapsed    else float("nan")
            # Train-test gap: large positive values suggest overfitting
            gap        = ((train_acc[-1] - test_acc[-1]) * 100
                          if train_acc and test_acc else float("nan"))
            early_ep   = hist.get("early_stopped_epoch", None)

            std_present = bool(test_acc_std)
            final_std   = test_acc_std[-1] * 100      if std_present else None
            # best_idx finds the epoch with the highest test accuracy, so best_std
            # is the uncertainty at that same epoch (not at the last epoch).
            best_idx    = test_acc.index(max(test_acc)) if test_acc else 0
            best_std    = test_acc_std[best_idx] * 100 if std_present else None

            final_str = f"{final_acc:.1f} ± {final_std:.1f}" if final_std is not None else f"{final_acc:.1f}"
            best_str  = f"{best_acc:.1f} ± {best_std:.1f}"   if best_std  is not None else f"{best_acc:.1f}"

            conv_epoch = hist.get("target_accuracy_epoch")
            conv_time  = hist.get("target_accuracy_time")

            # Start building the bullet-point line for this combo
            line = (
                f"- **{d} / {m}**: {final_str}% final, {best_str}% peak "
                f"({ep_run} epochs, {total_time:.1f} s)"
            )
            # Append optional suffixes in a consistent order
            if conv_epoch is not None:
                line += f", reached {target_pct}% at epoch {conv_epoch:.1f} ({conv_time:.1f} s)"
            ece_list = hist.get("ece", [])
            if ece_list:
                line += f", ECE {ece_list[-1]:.4f}"
            if gap > 5.0:
                # ⚠️ flag draws attention to potential overfitting
                line += f" ⚠️ high train-test gap ({gap:.1f}%)"
            if early_ep is not None:
                line += f" (early stopped at epoch {early_ep})"
            lines.append(line)
        lines.append("")

    # ------------------------------------------------------------------
    # Section 6 — Key Observations
    # Single-sentence global highlights derived from the full results dict.
    # ------------------------------------------------------------------
    lines += ["## Key Observations", ""]

    # Best overall test accuracy: pick the result with the highest final test_acc.
    # The default [float("-inf")] sentinel ensures the key function never crashes
    # on runs with empty test_acc lists.
    best_series = max(
        results,
        key=lambda k: (results[k].get("test_acc") or [float("-inf")])[-1],
    )
    best_series_name  = best_series[2]   # series name (optimizer or "opt lr=X" suffix)
    best_series_acc   = (results[best_series].get("test_acc") or [float("nan")])[-1] * 100
    lines.append(
        f"- **Highest test accuracy overall**: {best_series_name} "
        f"on {best_series[0]} / {best_series[1]} ({best_series_acc:.2f}%)"
    )

    # Fastest overall: the series with the lowest *mean* training time across all combos.
    # Mean is used rather than min so a series that was fast on one easy dataset but
    # slow on a hard one doesn't unfairly dominate.
    series_times: dict[str, list[float]] = {}
    for (d, m, s), hist in results.items():
        el = hist.get("time_elapsed", [])
        if el:
            series_times.setdefault(s, []).append(el[-1])
    if series_times:
        fastest_series = min(series_times, key=lambda s: statistics.mean(series_times[s]))
        mean_t = statistics.mean(series_times[fastest_series])
        lines.append(f"- **Fastest overall** (mean time): {fastest_series} ({mean_t:.1f} s)")

    # Most stable overall: the series with the lowest mean last-3 test_acc stdev.
    # "Last 3 epochs" captures late-training oscillations without being distorted
    # by early-training variance (which is always high regardless of optimizer).
    series_stds: dict[str, list[float]] = {}
    for (d, m, s), hist in results.items():
        ta = hist.get("test_acc", [])
        tail = ta[-3:] if len(ta) >= 3 else ta
        if len(tail) >= 2:
            series_stds.setdefault(s, []).append(statistics.stdev(tail))
    if series_stds:
        stable_series = min(series_stds, key=lambda s: statistics.mean(series_stds[s]))
        mean_std = statistics.mean(series_stds[stable_series])
        lines.append(f"- **Most stable overall**: {stable_series} (mean last-3 std={mean_std:.4f})")

    # Count runs where early stopping fired — indicates optimizer converged
    # faster than the epoch budget, or patience was set aggressively.
    early_count = sum(
        1 for hist in results.values()
        if hist.get("early_stopped_epoch") is not None
    )
    if early_count:
        lines.append(f"- **Early-stopped runs**: {early_count}")

    # Count runs where train-test gap > 5 percentage points at the final epoch.
    # A large gap is a strong signal of overfitting; useful for flagging
    # which optimizers need more regularisation or a smaller LR.
    gap_count = 0
    for hist in results.values():
        ta = hist.get("test_acc", [])
        tr = hist.get("train_acc", [])
        if ta and tr and (tr[-1] - ta[-1]) * 100 > 5.0:
            gap_count += 1
    if gap_count:
        lines.append(f"- **Runs with train-test gap > 5%**: {gap_count}")

    # Remind the reader about seed averaging so they know accuracy figures are means.
    if num_seeds > 1:
        lines.append(f"- **Seed averaging**: results averaged over {num_seeds} seeds")

    lines.append("")

    # Join all lines with newlines to form the final Markdown string.
    md = "\n".join(lines)

    # Write to disk if a save_path was provided.
    # os.makedirs with exist_ok=True creates intermediate directories silently.
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(md)

    return md
