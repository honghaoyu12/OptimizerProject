"""Replay benchmark plots from saved JSON logs without re-training.

Usage
-----
    # Single session
    python plot_from_logs.py logs/2026-03-09_14-30-00/

    # Merge multiple sessions into one comparison plot
    python plot_from_logs.py logs/session1/ logs/session2/ --save plots/replay.png

Each session directory must contain a ``run_summary.json`` file written by
``TrainingLogger.close()``.  The script reconstructs the ``results`` dict
expected by ``plot_benchmark()`` and opens (or saves) the comparison figure.
"""

import argparse
import json
import os
import sys

from benchmark import DATASET_REGISTRY, OPTIMIZER_REGISTRY
from visualizer import plot_benchmark


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_session(session_dir: str) -> list[dict]:
    """Return the list of run dicts from a session's ``run_summary.json``.

    Parameters
    ----------
    session_dir : path to a session directory produced by TrainingLogger

    Returns
    -------
    list of run dicts, each with keys ``name``, ``config``, ``summary``,
    ``history``, and ``files``.

    Raises
    ------
    FileNotFoundError if ``run_summary.json`` does not exist.
    """
    json_path = os.path.join(session_dir, "run_summary.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"run_summary.json not found in {session_dir!r}. "
            "Make sure the session was closed with TrainingLogger.close()."
        )
    with open(json_path) as f:
        data = json.load(f)
    return data["runs"]


def reconstruct_results(
    all_runs: list[dict],
) -> tuple[dict, list[str], list[str], list[str], dict]:
    """Build inputs for ``plot_benchmark()`` from a flat list of run dicts.

    Parameters
    ----------
    all_runs : concatenated list of run dicts from one or more sessions

    Returns
    -------
    results        : dict mapping (dataset_name, model_name, series_name) → history
    dataset_names  : ordered list of dataset display names
    model_names    : ordered list of model names
    series_names   : ordered list of series (optimizer) names
    opt_colors     : dict mapping series_name → matplotlib colour spec
    """
    # Reverse-map dataset key → display name  (e.g. "mnist" → "MNIST")
    ds_key_to_name = {v: k for k, v in DATASET_REGISTRY.items()}

    # Decide whether to use wd-suffixed series names:
    # if any (dataset, model, optimizer) combination appears with more than
    # one distinct weight_decay value, we suffix every series with (wd=…).
    from collections import defaultdict
    wd_sets: defaultdict = defaultdict(set)
    for run in all_runs:
        cfg = run["config"]
        key = (cfg.get("dataset", ""), cfg.get("model", ""), cfg.get("optimizer", ""))
        wd_sets[key].add(float(cfg.get("weight_decay", 0.0)))
    sweep_wd = any(len(wds) > 1 for wds in wd_sets.values())

    results: dict = {}
    dataset_names: list[str] = []
    model_names: list[str] = []
    series_names: list[str] = []

    for run in all_runs:
        cfg     = run["config"]
        history = run["history"]

        ds_key   = cfg.get("dataset", "unknown")
        mdl_name = cfg.get("model", "unknown")
        opt_name = cfg.get("optimizer", "unknown")
        wd       = float(cfg.get("weight_decay", 0.0))

        ds_name     = ds_key_to_name.get(ds_key, ds_key)
        series_name = f"{opt_name} (wd={wd:g})" if sweep_wd else opt_name

        results[(ds_name, mdl_name, series_name)] = history

        if ds_name not in dataset_names:
            dataset_names.append(ds_name)
        if mdl_name not in model_names:
            model_names.append(mdl_name)
        if series_name not in series_names:
            series_names.append(series_name)

    # Assign colours: use registered colours where available, fall back to tab10
    import matplotlib.pyplot as plt
    if sweep_wd:
        cmap = plt.get_cmap("tab20")
        opt_colors = {n: cmap(i % 20 / 20) for i, n in enumerate(series_names)}
    else:
        cmap = plt.get_cmap("tab10")
        opt_colors: dict = {}
        tab_i = 0
        for name in series_names:
            if name in OPTIMIZER_REGISTRY and "color" in OPTIMIZER_REGISTRY[name]:
                opt_colors[name] = OPTIMIZER_REGISTRY[name]["color"]
            else:
                opt_colors[name] = cmap(tab_i % 10 / 10)
                tab_i += 1

    return results, dataset_names, model_names, series_names, opt_colors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot benchmark results from saved JSON logs"
    )
    p.add_argument(
        "session_dirs", nargs="+", metavar="SESSION_DIR",
        help="One or more session log directories containing run_summary.json",
    )
    p.add_argument(
        "--save", default=None, metavar="PATH",
        help="Save the figure to this path instead of showing it interactively",
    )
    return p.parse_args()


def main():
    args = parse_args()

    all_runs: list[dict] = []
    for sd in args.session_dirs:
        runs = load_session(sd)
        all_runs.extend(runs)
        print(f"  Loaded {len(runs)} run(s) from {sd}")

    if not all_runs:
        print("No runs found. Exiting.", file=sys.stderr)
        sys.exit(1)

    results, dataset_names, model_names, series_names, opt_colors = reconstruct_results(all_runs)

    print(f"\n  Datasets : {', '.join(dataset_names)}")
    print(f"  Models   : {', '.join(model_names)}")
    print(f"  Series   : {', '.join(series_names)}")

    plot_benchmark(
        results, dataset_names, model_names, series_names, opt_colors,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
