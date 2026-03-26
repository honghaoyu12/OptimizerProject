"""Interactive optimizer benchmark — compare multiple optimizers in one run.

Run with:
    python benchmark.py

You are first prompted to select one or more datasets, then one or more
models, then one or more optimizers.  Every (dataset × model × optimizer)
combination is trained in sequence; results are plotted together for
easy comparison and saved to a Markdown report.

Key CLI flags
-------------
Training:
    --epochs, --batch-size, --hidden-sizes, --device
    --scheduler  (none | cosine | step | warmup_cosine | cosine_wr)
    --warmup-epochs, --patience, --min-delta, --max-grad-norm
    --ema-decay, --label-smoothing, --swa-start

LR / WD sweep:
    --lrs         (omit = per-optimizer defaults; one value = global override;
                  multiple values = LR sweep with lr= suffix in series names)
    --weight-decays  (similar multi-value sweep)

Seed averaging:
    --num-seeds   (default 1; >1 trains each combo N times and reports mean ± std)
    --seed        (base seed; consecutive seeds base+0, base+1, … are used)

Persistence:
    --save-results PATH   write results dict to JSON after training
    --load-results PATH   skip training; load a previous JSON and go straight
                          to plotting and report generation

Output:
    --save-plot, --save-lr-plot, --save-lr-scores, --save-grad-heatmap,
    --save-opt-states, --save-frontier, --report-path
"""

import argparse
import json
import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from logger import TrainingLogger
from lr_finder import LRFinder
from model import MLP, ResNet18, ViT
from optimizers import Lion, LAMB, Shampoo, Muon, Adan, AdaHessian, AdaBelief, SignSGD, AdaFactor, Sophia, Prodigy, ScheduleFreeAdamW
# Custom re-implementations of PyTorch built-in optimizers
from optimizers import Adam, AdamW, NAdam, RAdam, Adagrad, SGDMomentum, RMSprop
# New optimizers (2024)
from optimizers import SOAP, CautiousAdam, MARS, GrokFast
from optimizers import Tiger, AdanNesterov
from train import (DATASET_INFO, SCHEDULER_REGISTRY, get_dataloaders,
                   linear_layer_names, make_param_groups, run_training,
                   save_checkpoint, set_seed)
from report import generate_report
from visualizer import (plot_benchmark, plot_lr_sensitivity, plot_lr_sensitivity_scores,
                        plot_grad_flow_heatmap, plot_optimizer_states, plot_efficiency_frontier,
                        plot_weight_distance, plot_hp_heatmap, plot_grad_snr,
                        plot_class_accuracy, plot_instability, plot_step_size,
                        plot_sharpness, plot_grad_cosine_sim,
                        plot_grad_conflict, plot_lr_sensitivity_curves,
                        plot_dead_neurons, plot_weight_rank,
                        plot_grad_noise_scale, plot_fisher_trace,
                        plot_grad_alignment, plot_spectral_norm,
                        plot_optimizer_state_entropy, plot_plasticity)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
# Each registry is an OrderedDict so that items stay in insertion order when
# displayed in the interactive menus (numbered lists).
#
# DATASET_REGISTRY: human-readable display name -> internal dataset key
#   The internal key must match a key in train.DATASET_INFO.
#
# MODEL_REGISTRY: display name -> {factory, description}
#   factory(info, hidden_sizes) -> nn.Module
#
# OPTIMIZER_REGISTRY: display name -> {factory, default_lr, color}
#   factory(params, lr, weight_decay) -> torch.optim.Optimizer
#   default_lr: used when lrs=None (no explicit LR sweep)
#   color:      matplotlib hex color for plots (hand-picked for distinguishability)
# ---------------------------------------------------------------------------
DATASET_REGISTRY: OrderedDict = OrderedDict([
    ("MNIST",                     "mnist"),
    ("Fashion MNIST",             "fashion_mnist"),
    ("CIFAR-10",                  "cifar10"),
    ("CIFAR-100",                 "cifar100"),
    ("Tiny ImageNet",             "tiny_imagenet"),
    # Synthetic datasets (MLP only)
    ("Ill-Conditioned (synth)",   "illcond"),
    ("Sparse Signal (synth)",     "sparse"),
    ("Noisy Gradient (synth)",    "noisy_grad"),
    ("Nonlinear Manifold (synth)","manifold"),
    ("Saddle Point (synth)",      "saddle"),
    ("Checkerboard (synth)",      "checkerboard"),
    ("Plateau (synth)",           "plateau"),
    ("Correlated (synth)",        "correlated"),
    ("Imbalanced (synth)",        "imbalanced"),
])

MODEL_REGISTRY: OrderedDict = OrderedDict([
    ("MLP", {
        "factory": lambda info, hs: MLP(
            input_size=info["input_size"],
            hidden_sizes=hs,
            num_classes=info["num_classes"],
        ),
        "description": "Fully Connected (width & depth via --hidden-sizes)",
    }),
    ("ResNet-18", {
        "factory": lambda info, hs: ResNet18(
            in_channels=info["in_channels"],
            num_classes=info["num_classes"],
        ),
        "description": "ResNet-18 (adapted for small images)",
    }),
    ("ViT", {
        "factory": lambda info, hs: ViT(
            image_size=int((info["input_size"] / info["in_channels"]) ** 0.5),
            in_channels=info["in_channels"],
            num_classes=info["num_classes"],
        ),
        "description": "Vision Transformer (patch=4, dim=128, depth=6, heads=8)",
    }),
])

# Each entry: factory function + sensible default LR + display colour
OPTIMIZER_REGISTRY: OrderedDict = OrderedDict([
    ("SGD",          {
        # Custom VanillaSGD equivalent — plain SGD without momentum
        "factory":    lambda p, lr, wd: SGDMomentum(p, lr=lr, momentum=0.0, nesterov=False, weight_decay=wd),
        "default_lr": 0.01,
        "color":      "#e41a1c",   # red
    }),
    ("SGD+Momentum", {
        # Custom SGDMomentum with Nesterov acceleration (matches torch.optim.SGD momentum=0.9)
        "factory":    lambda p, lr, wd: SGDMomentum(p, lr=lr, momentum=0.9, weight_decay=wd),
        "default_lr": 0.01,
        "color":      "#ff7f00",   # orange
    }),
    ("Adam",         {
        # Custom Adam — decoupled bias-corrected adaptive moment estimation
        "factory":    lambda p, lr, wd: Adam(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#377eb8",   # blue
    }),
    ("Adagrad",      {
        # Custom Adagrad — cumulative squared gradient scaling
        "factory":    lambda p, lr, wd: Adagrad(p, lr=lr, weight_decay=wd),
        "default_lr": 0.01,
        "color":      "#4daf4a",   # green
    }),
    ("AdamW",        {
        # Custom AdamW — Adam with decoupled (not coupled) weight decay
        "factory":    lambda p, lr, wd: AdamW(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#984ea3",   # purple
    }),
    ("NAdam",        {
        # Custom NAdam — Adam with Nesterov momentum lookahead
        "factory":    lambda p, lr, wd: NAdam(p, lr=lr, weight_decay=wd),
        "default_lr": 0.002,
        "color":      "#a65628",   # brown
    }),
    ("RAdam",        {
        # Custom RAdam — Rectified Adam with automatic warm-up via SMA variance check
        "factory":    lambda p, lr, wd: RAdam(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#f781bf",   # pink
    }),
    ("Lion",         {
        "factory":    lambda p, lr, wd: Lion(p, lr=lr, weight_decay=wd),
        "default_lr": 0.0001,
        "color":      "#17becf",   # teal
    }),
    ("LAMB",         {
        "factory":    lambda p, lr, wd: LAMB(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#8c564b",   # dark red
    }),
    ("Shampoo",      {
        "factory":    lambda p, lr, wd: Shampoo(p, lr=lr, weight_decay=wd),
        "default_lr": 0.01,
        "color":      "#bcbd22",   # yellow-green
    }),
    ("Muon",         {
        "factory":    lambda p, lr, wd: Muon(p, lr=lr, weight_decay=wd),
        "default_lr": 0.02,
        "color":      "#e7298a",   # magenta
    }),
    ("Adan",         {
        "factory":    lambda p, lr, wd: Adan(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#66a61e",   # olive
    }),
    ("AdaHessian",   {
        "factory":    lambda p, lr, wd: AdaHessian(p, lr=lr, weight_decay=wd),
        "default_lr": 0.1,
        "color":      "#e6ab02",   # gold
    }),
    ("AdaBelief",    {
        "factory":    lambda p, lr, wd: AdaBelief(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#7f7f7f",   # grey
    }),
    ("SignSGD",      {
        "factory":    lambda p, lr, wd: SignSGD(p, lr=lr, weight_decay=wd),
        "default_lr": 0.01,
        "color":      "#aec7e8",   # light blue
    }),
    ("AdaFactor",    {
        "factory":    lambda p, lr, wd: AdaFactor(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#ffbb78",   # light orange
    }),
    ("Sophia",       {
        "factory":    lambda p, lr, wd: Sophia(p, lr=lr, weight_decay=wd),
        "default_lr": 0.0001,
        "color":      "#9467bd",   # purple
    }),
    ("Prodigy",      {
        "factory":    lambda p, lr, wd: Prodigy(p, lr=lr, weight_decay=wd),
        "default_lr": 1.0,
        "color":      "#d62728",   # crimson
    }),
    ("SF-AdamW",     {
        "factory":    lambda p, lr, wd: ScheduleFreeAdamW(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#2ca02c",   # green
    }),
    # New optimizers (2024)
    ("SOAP",         {
        "factory":    lambda p, lr, wd: SOAP(p, lr=lr, weight_decay=wd),
        "default_lr": 0.003,
        "color":      "#1f77b4",   # steel blue
    }),
    ("CautiousAdam", {
        "factory":    lambda p, lr, wd: CautiousAdam(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#ff9896",   # light red
    }),
    ("MARS",         {
        "factory":    lambda p, lr, wd: MARS(p, lr=lr, weight_decay=wd),
        "default_lr": 0.0003,
        "color":      "#98df8a",   # light green
    }),
    ("GrokFast",     {
        "factory":    lambda p, lr, wd: GrokFast(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#c5b0d5",   # lavender
    }),
    ("Tiger",        {
        "factory":    lambda p, lr, wd: Tiger(p, lr=lr, weight_decay=wd),
        "default_lr": 0.0001,
        "color":      "#e377c2",   # pink
    }),
    ("AdanNesterov", {
        "factory":    lambda p, lr, wd: AdanNesterov(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#bcbd22",   # yellow-green
    }),
])


# ---------------------------------------------------------------------------
# Interactive selection menus
# ---------------------------------------------------------------------------

def prompt_multiselect(title: str, options: list[str], descriptions: list[str] | None = None) -> list[str]:
    """Print a numbered menu and return the user-selected subset (multi-select).

    The user types comma-separated indices (e.g. "1,3,5") or "all" to
    select everything.  Duplicate indices are silently de-duplicated.
    Loops until valid input is received.

    Parameters
    ----------
    title        : section header shown above the menu
    options      : list of option display strings
    descriptions : optional per-option description strings (shown in parens)

    Returns
    -------
    list[str] — selected option strings, in the order the user entered them
    """
    width = 56
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"  Enter numbers separated by commas, or 'all'")
    print(f"{'─' * width}")
    for i, opt in enumerate(options, 1):
        desc = f"  ({descriptions[i-1]})" if descriptions else ""
        print(f"  [{i}]  {opt}{desc}")
    print()

    while True:
        raw = input("  > ").strip()
        if raw.lower() == "all":
            return list(options)
        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
            if indices and all(1 <= idx <= len(options) for idx in indices):
                seen: set = set()
                return [
                    options[idx - 1]
                    for idx in indices
                    if not (options[idx - 1] in seen or seen.add(options[idx - 1]))  # type: ignore[func-returns-value]
                ]
        except ValueError:
            pass
        print(f"  Invalid input. Enter numbers 1–{len(options)} separated by commas.")


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device(preference: str) -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

_KEY_SEP = "||"


def _save_results(results: dict, path: str) -> None:
    """Serialize the results dict to JSON at *path*.

    Tuple keys ``(dataset, model, series)`` are encoded as
    ``"dataset||model||series"`` strings so they survive JSON round-trips.
    All values must already be JSON-compatible (lists of floats / ints /
    None, nested dicts).  The parent directory is created automatically.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    serializable = {
        _KEY_SEP.join(k): v
        for k, v in results.items()
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def _load_results(path: str) -> dict:
    """Deserialize a results JSON file written by :func:`_save_results`.

    Returns a dict with tuple keys ``(dataset, model, series)``.
    Raises ``FileNotFoundError`` if *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return {tuple(k.split(_KEY_SEP)): v for k, v in raw.items()}


def _aggregate_histories(histories: list[dict]) -> dict:
    """Aggregate N per-seed histories into mean ± std.

    Each entry in ``histories`` is the raw dict returned by run_training()
    for one random seed.  The aggregated result has the same keys as a
    single-seed history, plus ``<key>_std`` companions for every list-
    valued metric (e.g. ``test_acc_std``).

    How each key type is handled
    ----------------------------
    * **List-valued metrics** (train_loss, test_acc, …) — stacked into a
      2-D array of shape (num_seeds × epochs), then reduced along axis=0
      to produce mean and std lists.  Truncated to the shortest seed when
      early stopping caused unequal lengths.
    * **Scalar convergence keys** (target_accuracy_epoch, target_accuracy_time)
      — averaged across seeds that actually reached the target; None if none did.
    * **early_stopped_epoch** — mean epoch across seeds that were stopped,
      rounded to int; None if early stopping was never triggered.
    * **step_losses** — batch-level losses concatenated then averaged
      (useful for step-level loss curves after seed averaging).
    * **weight_norms / grad_norms** — per-layer dicts, each layer averaged
      across seeds.
    * **optimizer_states** — per-state-key, per-layer trajectories averaged
      across seeds; scalar trajectories (e.g. Prodigy's d) also averaged.
    * **All other keys** (config, flags, non-list scalars) — taken from
      histories[0] unchanged.
    """
    import numpy as np

    if not histories:
        return {}

    result: dict = {}

    # ── List-valued per-epoch metrics ─────────────────────────────────────
    list_keys = [
        "train_loss", "test_loss", "train_acc", "test_acc", "learning_rates",
        "grad_norm_global", "grad_norm_before_clip", "grad_norm_std", "time_elapsed",
        "test_acc_ema", "steps_at_epoch_end",
        # ECE and instability are scalars per epoch — averaged like accuracy
        "ece", "batch_loss_std",
        # Sharpness is a scalar per epoch (nan when track_sharpness=False)
        "sharpness",
        # Gradient noise scale and Fisher trace are scalars per epoch
        "grad_noise_scale",
        "fisher_trace",
        # New scalar-per-epoch metrics
        "grad_alignment",
        "optimizer_state_entropy",
        "plasticity",
    ]
    for key in list_keys:
        vals = [h.get(key) or [] for h in histories]
        non_empty = [v for v in vals if v]
        if not non_empty:
            continue
        # Truncate to shortest run (early stopping may cause different lengths)
        min_len = min(len(v) for v in non_empty)
        arr = np.array([v[:min_len] for v in non_empty], dtype=float)
        result[key] = arr.mean(axis=0).tolist()
        result[f"{key}_std"] = arr.std(axis=0).tolist()

    for key in ("target_accuracy_epoch", "target_accuracy_time"):
        vals = [h.get(key) for h in histories if h.get(key) is not None]
        result[key] = float(np.mean(vals)) if vals else None

    early_vals = [h.get("early_stopped_epoch") for h in histories
                  if h.get("early_stopped_epoch") is not None]
    result["early_stopped_epoch"] = round(float(np.mean(early_vals))) if early_vals else None

    step_vals = [h.get("step_losses") or [] for h in histories]
    non_empty = [v for v in step_vals if v]
    if non_empty:
        min_len = min(len(v) for v in non_empty)
        arr = np.array([v[:min_len] for v in non_empty], dtype=float)
        result["step_losses"] = arr.mean(axis=0).tolist()

    for key in ("weight_norms", "grad_norms", "weight_distance", "grad_snr",
                "class_acc", "step_size", "grad_cosine_sim", "grad_conflict",
                "dead_neurons", "weight_rank", "spectral_norm"):
        dicts = [h.get(key, {}) for h in histories]
        all_subkeys: set = set()
        for d in dicts:
            all_subkeys.update(d.keys())
        merged: dict = {}
        for subkey in all_subkeys:
            sub_vals = [d.get(subkey) or [] for d in dicts]
            non_empty_sub = [v for v in sub_vals if v]
            if non_empty_sub:
                min_len = min(len(v) for v in non_empty_sub)
                arr = np.array([v[:min_len] for v in non_empty_sub], dtype=float)
                merged[subkey] = arr.mean(axis=0).tolist()
            else:
                merged[subkey] = []
        result[key] = merged

    # optimizer_states: {state_key: {layer: [floats]} or [floats]}
    opt_states_list = [h.get("optimizer_states") or {} for h in histories]
    all_state_keys: set = set()
    for d in opt_states_list:
        all_state_keys.update(d.keys())
    aggregated_opt: dict = {}
    for state_key in all_state_keys:
        vals = [d[state_key] for d in opt_states_list if state_key in d]
        if not vals:
            continue
        if isinstance(vals[0], dict):
            all_layers: set = set()
            for v in vals:
                all_layers.update(v.keys())
            merged: dict = {}
            for layer in all_layers:
                sub_vals = [v.get(layer) or [] for v in vals]
                non_empty_sub = [v for v in sub_vals if v]
                if non_empty_sub:
                    min_len = min(len(v) for v in non_empty_sub)
                    arr = np.array([v[:min_len] for v in non_empty_sub], dtype=float)
                    merged[layer] = arr.mean(axis=0).tolist()
                else:
                    merged[layer] = []
            aggregated_opt[state_key] = merged
        else:
            non_empty = [v for v in vals if v]
            if non_empty:
                min_len = min(len(v) for v in non_empty)
                arr = np.array([v[:min_len] for v in non_empty], dtype=float)
                aggregated_opt[state_key] = arr.mean(axis=0).tolist()
    result["optimizer_states"] = aggregated_opt

    # ── Per-seed final accuracies (for statistical significance testing) ───
    # Store the raw final test_acc from each seed as a plain list so that
    # callers can run paired t-tests or other statistics on them later.
    final_accs = [
        h.get("test_acc", [])[-1] if h.get("test_acc") else float("nan")
        for h in histories
    ]
    result["test_acc_final_seeds"] = final_accs

    # ── Convergence profile aggregation ───────────────────────────────────
    # For each threshold key ("50%", "75%", …), average the epoch / time
    # values over seeds that actually reached that milestone.  Seeds that
    # never reached a threshold contribute None and are excluded from the
    # mean.  If no seed reached a threshold the result is None.
    for key in ("convergence_epochs", "convergence_times"):
        seed_dicts = [h.get(key, {}) for h in histories]
        all_thresh_keys: set = set()
        for d in seed_dicts:
            all_thresh_keys.update(d.keys())
        merged: dict = {}
        for thresh_key in sorted(all_thresh_keys):
            vals = [
                d[thresh_key]
                for d in seed_dicts
                if thresh_key in d and d[thresh_key] is not None
            ]
            if vals:
                avg = float(np.mean(vals))
                # Round epoch counts to one decimal; times keep full precision
                merged[thresh_key] = round(avg, 1) if key == "convergence_epochs" else round(avg, 2)
            else:
                merged[thresh_key] = None
        result[key] = merged

    for key in histories[0]:
        if key not in result:
            result[key] = histories[0][key]

    return result


def run_benchmark(
    dataset_names: list[str],
    model_names: list[str],
    optimizer_names: list[str],
    hidden_sizes: list[int],
    epochs: int,
    batch_size: int,
    data_dir: str,
    device: torch.device,
    lrs: list[float] | None,
    logger: TrainingLogger | None = None,
    scheduler_name: str = "none",
    warmup_epochs: int = 5,
    patience: int = 0,
    min_delta: float = 0.0,
    max_grad_norm: float | None = None,
    weight_decays: list[float] | None = None,
    seed: int | None = None,
    checkpoint_dir: str | None = None,
    num_seeds: int = 1,
    target_acc: float = 0.95,
    amp: bool = False,
    compile_model: bool = False,
    ema_decay: float | None = None,
    label_smoothing: float = 0.0,
    swa_start: int | None = None,
    auto_lr: bool = False,
    track_sharpness: bool = False,
    track_dead_neurons: bool = False,
    track_weight_rank: bool = False,
    track_fisher: bool = False,
    track_spectral_norm: bool = False,
    track_optimizer_entropy: bool = False,
    plasticity_interval: int = 0,
    accum_steps: int = 1,
) -> dict:
    """Train every (dataset, model, optimizer, weight_decay) combination and collect histories.

    When ``num_seeds > 1`` each combination is trained ``num_seeds`` times
    with consecutive seeds and the histories are aggregated via
    :func:`_aggregate_histories` (mean ± std for list-valued keys).

    Returns
    -------
    dict mapping (dataset_name, model_name, series_name) -> history_dict
    """
    if weight_decays is None:
        weight_decays = [0.0]

    # Optimizers incompatible with torch.compile (create_graph=True in step())
    _COMPILE_INCOMPATIBLE_NAMES = {"sophia", "adahessian"}

    results: dict = {}
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    sweep_lr = lrs is not None and len(lrs) > 1
    sweep_wd = len(weight_decays) > 1
    n_lr_vals = len(lrs) if lrs is not None else 1
    total_runs = (len(dataset_names) * len(model_names)
                  * len(optimizer_names) * n_lr_vals * len(weight_decays))
    run_idx = 0

    for ds_name in dataset_names:
        ds_key = DATASET_REGISTRY[ds_name]
        ds_info = DATASET_INFO[ds_key]
        print(f"\n  Loading {ds_name}...")
        train_loader, test_loader = get_dataloaders(ds_key, batch_size, data_dir)

        for mdl_name in model_names:
            if ds_info.get("tabular") and mdl_name != "MLP":
                print(f"\n  Skipping {mdl_name} for tabular dataset '{ds_name}' (MLP only).")
                run_idx += len(optimizer_names) * n_lr_vals * len(weight_decays)
                continue
            mdl_info = MODEL_REGISTRY[mdl_name]

            for opt_name in optimizer_names:
                opt_info = OPTIMIZER_REGISTRY[opt_name]
                lr_list = lrs if lrs is not None else [opt_info["default_lr"]]

                for lr in lr_list:
                    for wd in weight_decays:
                        run_idx += 1
                        suffixes = []
                        if sweep_lr:
                            suffixes.append(f"lr={lr:g}")
                        if sweep_wd:
                            suffixes.append(f"wd={wd:g}")
                        series_name = f"{opt_name} ({', '.join(suffixes)})" if suffixes else opt_name

                        # ── Auto-LR: run LR range test before the seed loop ───────────
                        # When --auto-lr is active and the user has not specified explicit
                        # LR values, a fresh temporary model is built, LRFinder sweeps LRs
                        # on the training data, and the point of steepest loss descent is
                        # used as the effective LR for all seeds of this run.  The temporary
                        # model and optimizer are discarded immediately after the search.
                        effective_lr = lr
                        if auto_lr and lrs is None:
                            _tmp_model = mdl_info["factory"](ds_info, hidden_sizes).to(device)
                            if wd > 0.0:
                                _tmp_params = make_param_groups(_tmp_model, wd)
                                _tmp_opt = opt_info["factory"](_tmp_params, opt_info["default_lr"], 0.0)
                            else:
                                _tmp_opt = opt_info["factory"](_tmp_model.parameters(), opt_info["default_lr"], 0.0)
                            _finder = LRFinder(_tmp_model, _tmp_opt, nn.CrossEntropyLoss(), device)
                            _finder.run(train_loader)
                            _found_lr = _finder.suggestion()
                            del _tmp_model, _tmp_opt, _finder
                            if not math.isnan(_found_lr) and _found_lr > 0:
                                effective_lr = _found_lr
                            else:
                                print(f"  ⚠  Auto-LR: LRFinder returned nan for {opt_name}, "
                                      f"using default {lr:.2e}")

                        lr_tag = f"{effective_lr:.3g}{'  [auto]' if auto_lr and lrs is None else ''}"
                        print(f"\n{'═' * 60}")
                        print(f"  Run {run_idx}/{total_runs}: {ds_name} | {mdl_name} | {series_name}  (lr={lr_tag})")
                        print(f"{'═' * 60}")

                        # Warn once per (opt, lr, wd) if compile is incompatible
                        if compile_model and opt_name.lower() in _COMPILE_INCOMPATIBLE_NAMES:
                            print(
                                f"\n  ⚠  torch.compile disabled for {opt_name}: "
                                f"this optimizer uses backward(create_graph=True) to "
                                f"estimate Hessian-vector products. torch.compile's "
                                f"static graph tracing cannot capture dynamic higher-order "
                                f"graphs correctly. Running {opt_name} in eager mode."
                            )

                        base_seed = seed if seed is not None else 0
                        seed_histories: list[dict] = []
                        for seed_idx in range(num_seeds):
                            if num_seeds > 1 or seed is not None:
                                set_seed(base_seed + seed_idx)
                            model = mdl_info["factory"](ds_info, hidden_sizes).to(device)
                            if compile_model and opt_name.lower() not in _COMPILE_INCOMPATIBLE_NAMES:
                                try:
                                    model = torch.compile(model)
                                except Exception as e:
                                    if seed_idx == 0:
                                        print(f"\n  ⚠  torch.compile failed: {e}. "
                                              f"Running {opt_name} in eager mode.")
                            if wd > 0.0:
                                params = make_param_groups(model, wd)
                                optimizer = opt_info["factory"](params, effective_lr, 0.0)
                            else:
                                optimizer = opt_info["factory"](model.parameters(), effective_lr, 0.0)
                            layer_names = linear_layer_names(model)
                            scheduler = SCHEDULER_REGISTRY[scheduler_name](optimizer, epochs, warmup_epochs)

                            run_ckpt_dir = None
                            if checkpoint_dir:
                                run_ckpt_dir = os.path.join(
                                    checkpoint_dir,
                                    f"{ds_key}_{mdl_name.lower()}_{opt_name.lower().replace('+', '_')}",
                                )
                            run_cfg = {
                                "dataset": ds_key, "model": mdl_name, "optimizer": opt_name,
                                "lr": effective_lr, "epochs": epochs, "weight_decay": wd,
                            }
                            history = run_training(
                                model, train_loader, test_loader,
                                optimizer, criterion, device,
                                epochs, layer_names, verbose=True,
                                scheduler=scheduler,
                                patience=patience,
                                min_delta=min_delta,
                                max_grad_norm=max_grad_norm,
                                target_acc=target_acc,
                                checkpoint_dir=run_ckpt_dir,
                                checkpoint_config=run_cfg,
                                amp=amp,
                                ema_decay=ema_decay,
                                swa_start=swa_start,
                                track_sharpness=track_sharpness,
                                track_dead_neurons=track_dead_neurons,
                                track_weight_rank=track_weight_rank,
                                track_fisher=track_fisher,
                                track_spectral_norm=track_spectral_norm,
                                track_optimizer_entropy=track_optimizer_entropy,
                                plasticity_interval=plasticity_interval,
                                accum_steps=accum_steps,
                            )
                            seed_histories.append(history)

                            if logger is not None:
                                log_cfg = {
                                    "dataset": ds_key, "model": mdl_name, "optimizer": opt_name,
                                    "lr": effective_lr, "epochs": epochs, "batch_size": batch_size,
                                    "scheduler": scheduler_name, "patience": patience,
                                    "min_delta": min_delta, "max_grad_norm": max_grad_norm,
                                    "weight_decay": wd, "seed": base_seed + seed_idx,
                                }
                                logger.log_run(log_cfg, history)

                        aggregated = (
                            _aggregate_histories(seed_histories) if num_seeds > 1
                            else seed_histories[0]
                        )
                        aggregated["config_lr"] = effective_lr
                        aggregated["config_wd"] = wd
                        results[(ds_name, mdl_name, series_name)] = aggregated

    return results


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Interactive optimizer benchmark")
    p.add_argument("--epochs",       default=10,         type=int,
                   help="Training epochs per run")
    p.add_argument("--batch-size",   default=128,        type=int)
    p.add_argument("--hidden-sizes", default=[256, 128], type=int, nargs="+", metavar="N",
                   help="Hidden layer widths for MLP, e.g. --hidden-sizes 256 128")
    p.add_argument("--lrs",          default=None,       type=float, nargs="+", metavar="LR",
                   help="LR values to sweep (omit = per-optimizer defaults; "
                        "one value = global override; multiple = LR sweep)")
    p.add_argument("--data-dir",     default="./data")
    p.add_argument("--device",       default="auto",     help="cpu | cuda | mps | auto")
    p.add_argument("--save-plot",    default="plots/benchmark.png",
                   help="Path to save comparison figure ('' to disable)")
    p.add_argument("--log-dir",      default="logs",
                   help="Root directory for training logs (default: logs/)")
    p.add_argument("--scheduler",    default="none",
                   choices=list(SCHEDULER_REGISTRY.keys()),
                   help="LR scheduler: none | cosine | step | warmup_cosine | cosine_wr")
    p.add_argument("--warmup-epochs", default=5, type=int,
                   help="Linear warmup epochs for warmup_cosine (default: 5)")
    p.add_argument("--patience",  default=0,   type=int,
                   help="Early stopping patience epochs (0 = disabled)")
    p.add_argument("--min-delta", default=0.0, type=float,
                   help="Min val-loss improvement to reset patience counter")
    p.add_argument("--max-grad-norm", default=None, type=float,
                   help="Clip global gradient norm to this value (None = disabled)")
    p.add_argument("--weight-decays", default=[0.0], type=float, nargs="+",
                   help="Weight decay values to sweep (e.g. --weight-decays 0 1e-4 1e-3)")
    p.add_argument("--seed", default=None, type=int,
                   help="Random seed for reproducibility (default: None, non-deterministic)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Save best.pt and final.pt under this directory per run (default: disabled)")
    p.add_argument("--report-path", default="reports/benchmark_report.md",
                   help="Save path for the Markdown benchmark report ('' to disable)")
    p.add_argument("--num-seeds", default=1, type=int,
                   help="Seeds to average over per run (default: 1 = single run)")
    p.add_argument("--target-acc", default=0.95, type=float,
                   help="Accuracy threshold for convergence speed columns in report (default: 0.95)")
    p.add_argument("--amp", action="store_true",
                   help="Enable automatic mixed precision: float16 autocast + GradScaler "
                        "on CUDA; float16 autocast only on MPS; no-op on CPU. "
                        "Auto-disabled for Sophia and AdaHessian (incompatible with GradScaler).")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile() to each model before training. Auto-disabled "
                        "per-run for Sophia and AdaHessian (create_graph=True incompatible "
                        "with static graph tracing). Falls back to eager if compilation fails.")
    p.add_argument("--save-lr-plot", default="plots/lr_sensitivity.png",
                   help="Path to save LR sensitivity figure ('' to disable; "
                        "only generated when --lrs has ≥2 values)")
    p.add_argument("--save-lr-scores", default="plots/lr_sensitivity_scores.png",
                   help="Path to save LR sensitivity score bar chart ('' to disable; "
                        "only generated when --lrs has ≥2 values)")
    p.add_argument("--save-grad-heatmap", default="plots/grad_flow.png",
                   help="Path to save per-layer gradient flow heatmap ('' to disable)")
    p.add_argument("--save-opt-states", default="plots/opt_states.png",
                   help="Path to save optimizer internal state figure ('' to disable)")
    p.add_argument("--save-frontier", default="plots/efficiency_frontier.png",
                   help="Path to save efficiency frontier plot ('' to disable)")
    p.add_argument("--ema-decay", default=None, type=float,
                   help="Exponential moving average decay for model weights "
                        "(e.g. 0.999). EMA weights are evaluated each epoch alongside "
                        "raw weights. None = disabled (default).")
    p.add_argument("--label-smoothing", default=0.0, type=float,
                   help="Label smoothing epsilon for CrossEntropyLoss (default: 0.0 = disabled). "
                        "Typical value: 0.1.")
    p.add_argument("--save-results", default="", metavar="PATH",
                   help="Save benchmark results dict to a JSON file after training "
                        "(e.g. results/run.json). Skipped when empty (default).")
    p.add_argument("--load-results", default="", metavar="PATH",
                   help="Load a previously saved results JSON and skip training entirely. "
                        "Plots and report are still generated from the loaded data.")
    p.add_argument("--swa-start", default=None, type=int, metavar="EPOCH",
                   help="Epoch to begin Stochastic Weight Averaging (default: disabled). "
                        "SWA is evaluated once after training completes.")
    p.add_argument("--save-weight-distance", default="plots/weight_distance.png",
                   help="Path to save weight-distance-from-init figure ('' to disable)")
    p.add_argument("--save-hp-heatmap", default="plots/hp_heatmap.png",
                   help="Path to save LR × WD robustness heatmap ('' to disable; "
                        "only generated when both --lrs and --weight-decays have ≥2 values)")
    p.add_argument("--save-grad-snr", default="plots/grad_snr.png",
                   help="Path to save gradient SNR figure ('' to disable)")
    p.add_argument("--save-class-accuracy", default="plots/class_accuracy.png",
                   help="Path to save per-class accuracy figure ('' to disable)")
    p.add_argument("--save-instability", default="plots/instability.png",
                   help="Path to save training instability figure ('' to disable)")
    p.add_argument("--save-step-size", default="plots/step_size.png",
                   help="Path to save effective step-size figure ('' to disable)")
    p.add_argument("--sharpness", action="store_true",
                   help="Compute SAM-style sharpness every epoch (small overhead: "
                        "5 random forward passes on a fixed batch per epoch)")
    p.add_argument("--save-sharpness", default="plots/sharpness.png",
                   help="Path to save sharpness-vs-epoch figure ('' to disable)")
    p.add_argument("--save-grad-cosine-sim", default="plots/grad_cosine_sim.png",
                   help="Path to save gradient cosine similarity figure ('' to disable)")
    p.add_argument("--save-grad-conflict", default="plots/grad_conflict.png",
                   help="Path to save gradient conflict figure ('' to disable)")
    p.add_argument("--save-lr-curves", default="plots/lr_curves.png",
                   help="Path to save LR sensitivity training curves ('' to disable)")
    p.add_argument("--dead-neurons", action="store_true",
                   help="Track dead ReLU neuron fraction each epoch (cheap forward-pass hook)")
    p.add_argument("--save-dead-neurons", default="plots/dead_neurons.png",
                   help="Path to save dead neuron fraction figure ('' to disable)")
    p.add_argument("--weight-rank", action="store_true",
                   help="Track effective weight matrix rank via SVD each epoch")
    p.add_argument("--save-weight-rank", default="plots/weight_rank.png",
                   help="Path to save weight rank figure ('' to disable)")
    p.add_argument("--save-grad-noise", default="plots/grad_noise_scale.png",
                   help="Path to save gradient noise scale figure ('' to disable)")
    p.add_argument("--fisher", action="store_true",
                   help="Track Fisher information trace each epoch")
    p.add_argument("--save-fisher", default="plots/fisher_trace.png",
                   help="Path to save Fisher trace figure ('' to disable)")
    p.add_argument("--spectral-norm", action="store_true",
                   help="Track largest singular value σ_max(W) per Linear layer each epoch")
    p.add_argument("--save-spectral-norm", default="plots/spectral_norm.png",
                   help="Path to save spectral norm figure ('' to disable)")
    p.add_argument("--optimizer-entropy", action="store_true",
                   help="Track Shannon entropy of optimizer momentum/variance buffers each epoch")
    p.add_argument("--save-optimizer-entropy", default="plots/optimizer_entropy.png",
                   help="Path to save optimizer state entropy figure ('' to disable)")
    p.add_argument("--plasticity-interval", type=int, default=0,
                   help="Compute loss-of-plasticity score every N epochs (0 = disabled)")
    p.add_argument("--save-plasticity", default="plots/plasticity.png",
                   help="Path to save plasticity score figure ('' to disable)")
    p.add_argument("--save-grad-alignment", default="plots/grad_alignment.png",
                   help="Path to save gradient alignment figure ('' to disable)")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps (simulate larger batch; default=1)")
    p.add_argument("--auto-lr", action="store_true",
                   help="Run a learning-rate range test (LR finder) before each "
                        "(optimizer, dataset, model) combination and use the found LR "
                        "instead of the optimizer's default.  Ignored when --lrs is "
                        "also set (explicit LRs take precedence).")
    return p.parse_args()


def main():
    args = parse_args()

    device = get_device(args.device)

    print(f"\n  Device       : {device}")
    print(f"  Epochs       : {args.epochs}")
    if args.num_seeds > 1:
        base_seed = args.seed if args.seed is not None else 0
        print(f"  Seed averaging : {args.num_seeds} seeds (base {base_seed})")
    if args.amp:
        print(f"  AMP            : enabled (auto-disabled per-run for Sophia/AdaHessian)")
    if args.compile:
        print(f"  torch.compile  : enabled (auto-disabled per-run for Sophia/AdaHessian)")
    if args.ema_decay is not None:
        print(f"  EMA decay      : {args.ema_decay}")
    if args.label_smoothing > 0.0:
        print(f"  Label smoothing: {args.label_smoothing}")
    if args.swa_start is not None:
        print(f"  SWA start      : epoch {args.swa_start}")
    if args.auto_lr:
        if args.lrs is not None:
            print(f"  Auto-LR        : --lrs is set; explicit LR values take precedence over auto-LR")
        else:
            print(f"  Auto-LR        : enabled (LR range test run before each combination)")
    if args.lrs is not None and len(args.lrs) == 1:
        print(f"  LR (global)  : {args.lrs[0]}")
    elif args.lrs is not None:
        print(f"  LR sweep     : {args.lrs}")
    else:
        print(f"  LR           : per-optimizer defaults")

    # ── Load or train ─────────────────────────────────────────────────────
    if args.load_results:
        print(f"\n  Loading results from {args.load_results} (skipping training)...")
        results = _load_results(args.load_results)
        # Recover dataset / model / series names from keys
        dataset_names   = list(dict.fromkeys(k[0] for k in results))
        model_names     = list(dict.fromkeys(k[1] for k in results))
        series_names    = list(dict.fromkeys(k[2] for k in results))
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("tab20")
        opt_colors = {n: cmap(i % 20 / 20) for i, n in enumerate(series_names)}
        print(f"  Loaded {len(results)} result(s): "
              f"{', '.join(dataset_names)} | {', '.join(model_names)} | "
              f"{', '.join(series_names)}")
    else:
        # ── Interactive selection ─────────────────────────────────────────
        dataset_names   = prompt_multiselect("Select Datasets",   list(DATASET_REGISTRY.keys()))
        model_descs     = [MODEL_REGISTRY[m]["description"] for m in MODEL_REGISTRY]
        model_names     = prompt_multiselect("Select Models",     list(MODEL_REGISTRY.keys()), model_descs)
        optimizer_names = prompt_multiselect("Select Optimizers", list(OPTIMIZER_REGISTRY.keys()))

        print(f"\n  Selected datasets   : {', '.join(dataset_names)}")
        print(f"  Selected models     : {', '.join(model_names)}")
        print(f"  Selected optimizers : {', '.join(optimizer_names)}")

        # ── Determine series names and colors ─────────────────────────────
        sweep_lr = args.lrs is not None and len(args.lrs) > 1
        sweep_wd = len(args.weight_decays) > 1
        if sweep_lr or sweep_wd:
            lr_vals = args.lrs if args.lrs is not None else [None]  # None = per-optimizer default
            series_names = []
            for opt in optimizer_names:
                for lr in lr_vals:
                    for wd in args.weight_decays:
                        suffixes = []
                        if sweep_lr and lr is not None:
                            suffixes.append(f"lr={lr:g}")
                        if sweep_wd:
                            suffixes.append(f"wd={wd:g}")
                        series_names.append(f"{opt} ({', '.join(suffixes)})" if suffixes else opt)
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("tab20")
            opt_colors = {n: cmap(i % 20 / 20) for i, n in enumerate(series_names)}
        else:
            series_names = optimizer_names
            opt_colors = {n: OPTIMIZER_REGISTRY[n]["color"] for n in optimizer_names}

        # ── Run ───────────────────────────────────────────────────────────
        logger = TrainingLogger(log_dir=args.log_dir)
        results = run_benchmark(
            dataset_names, model_names, optimizer_names,
            hidden_sizes=args.hidden_sizes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            device=device,
            lrs=args.lrs,
            logger=logger,
            scheduler_name=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
            max_grad_norm=args.max_grad_norm,
            weight_decays=args.weight_decays,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
            num_seeds=args.num_seeds,
            target_acc=args.target_acc,
            amp=args.amp,
            compile_model=args.compile,
            ema_decay=args.ema_decay,
            label_smoothing=args.label_smoothing,
            swa_start=args.swa_start,
            auto_lr=args.auto_lr and args.lrs is None,
            track_sharpness=args.sharpness,
            track_dead_neurons=args.dead_neurons,
            track_weight_rank=args.weight_rank,
            track_fisher=args.fisher,
            track_spectral_norm=args.spectral_norm,
            track_optimizer_entropy=args.optimizer_entropy,
            plasticity_interval=args.plasticity_interval,
            accum_steps=args.accum_steps,
        )
        logger.close()

        # ── Save results ──────────────────────────────────────────────────
        if args.save_results:
            _save_results(results, args.save_results)
            print(f"\n  Results saved to {args.save_results}")

    # ── Plot ──────────────────────────────────────────────────────────────
    save_path = args.save_plot if args.save_plot else None
    plot_benchmark(results, dataset_names, model_names, series_names, opt_colors, save_path=save_path)

    # ── LR Sensitivity Plot ───────────────────────────────────────────────
    if args.lrs is not None and len(args.lrs) > 1 and args.save_lr_plot:
        plot_lr_sensitivity(results, save_path=args.save_lr_plot)

    # ── LR Sensitivity Score Chart ────────────────────────────────────────
    if args.lrs is not None and len(args.lrs) > 1 and args.save_lr_scores:
        plot_lr_sensitivity_scores(results, save_path=args.save_lr_scores)

    # ── Gradient Flow Heatmap ─────────────────────────────────────────────
    if args.save_grad_heatmap:
        plot_grad_flow_heatmap(results, save_path=args.save_grad_heatmap)

    # ── Optimizer State Plot ───────────────────────────────────────────────
    if args.save_opt_states:
        plot_optimizer_states(results, save_path=args.save_opt_states)

    # ── Efficiency Frontier ────────────────────────────────────────────────
    if args.save_frontier:
        plot_efficiency_frontier(results, save_path=args.save_frontier)

    # ── Weight Distance from Init ──────────────────────────────────────────
    if args.save_weight_distance:
        plot_weight_distance(results, dataset_names, model_names, series_names,
                             opt_colors, save_path=args.save_weight_distance)

    # ── Gradient SNR ───────────────────────────────────────────────────────
    if args.save_grad_snr:
        plot_grad_snr(results, dataset_names, model_names, series_names,
                      opt_colors, save_path=args.save_grad_snr)

    # ── Per-Class Accuracy ─────────────────────────────────────────────────
    if args.save_class_accuracy:
        plot_class_accuracy(results, dataset_names, model_names, series_names,
                            opt_colors, save_path=args.save_class_accuracy)

    # ── Training Instability ───────────────────────────────────────────────
    if args.save_instability:
        plot_instability(results, dataset_names, model_names, series_names,
                         opt_colors, save_path=args.save_instability)

    # ── Effective Step Size ────────────────────────────────────────────────
    if args.save_step_size:
        plot_step_size(results, dataset_names, model_names, series_names,
                       opt_colors, save_path=args.save_step_size)

    # ── Sharpness ─────────────────────────────────────────────────────────
    if args.save_sharpness:
        plot_sharpness(results, dataset_names, model_names, series_names,
                       opt_colors, save_path=args.save_sharpness)

    # ── Gradient Cosine Similarity ────────────────────────────────────────
    if args.save_grad_cosine_sim:
        plot_grad_cosine_sim(results, dataset_names, model_names, series_names,
                             opt_colors, save_path=args.save_grad_cosine_sim)

    # ── Gradient Conflict ─────────────────────────────────────────────────
    if args.save_grad_conflict:
        plot_grad_conflict(results, dataset_names, model_names, series_names,
                           opt_colors, save_path=args.save_grad_conflict)

    # ── LR Sensitivity Curves ─────────────────────────────────────────────
    if args.save_lr_curves:
        plot_lr_sensitivity_curves(results, save_path=args.save_lr_curves)

    # ── Dead Neurons ──────────────────────────────────────────────────────
    if args.save_dead_neurons and args.dead_neurons:
        plot_dead_neurons(results, dataset_names, model_names, series_names,
                          opt_colors, save_path=args.save_dead_neurons)

    # ── Weight Rank ───────────────────────────────────────────────────────
    if args.save_weight_rank and args.weight_rank:
        plot_weight_rank(results, dataset_names, model_names, series_names,
                         opt_colors, save_path=args.save_weight_rank)

    # ── Gradient Noise Scale ──────────────────────────────────────────────
    if args.save_grad_noise:
        plot_grad_noise_scale(results, dataset_names, model_names, series_names,
                              opt_colors, save_path=args.save_grad_noise)

    # ── Fisher Trace ──────────────────────────────────────────────────────
    if args.save_fisher and args.fisher:
        plot_fisher_trace(results, dataset_names, model_names, series_names,
                          opt_colors, save_path=args.save_fisher)

    # ── Gradient Alignment ────────────────────────────────────────────────
    if args.save_grad_alignment:
        plot_grad_alignment(results, dataset_names, model_names, series_names,
                            opt_colors, save_path=args.save_grad_alignment)

    # ── Spectral Norm ─────────────────────────────────────────────────────
    if args.save_spectral_norm and args.spectral_norm:
        plot_spectral_norm(results, dataset_names, model_names, series_names,
                           opt_colors, save_path=args.save_spectral_norm)

    # ── Optimizer State Entropy ───────────────────────────────────────────
    if args.save_optimizer_entropy and args.optimizer_entropy:
        plot_optimizer_state_entropy(results, dataset_names, model_names, series_names,
                                     opt_colors, save_path=args.save_optimizer_entropy)

    # ── Plasticity Score ──────────────────────────────────────────────────
    if args.save_plasticity and args.plasticity_interval > 0:
        plot_plasticity(results, dataset_names, model_names, series_names,
                        opt_colors, save_path=args.save_plasticity)

    # ── HP Robustness Heatmap ─────────────────────────────────────────────
    if (args.save_hp_heatmap
            and args.lrs is not None and len(args.lrs) >= 2
            and len(args.weight_decays) >= 2):
        plot_hp_heatmap(results, save_path=args.save_hp_heatmap)

    # ── Report ────────────────────────────────────────────────────────────
    if args.report_path:
        report_cfg = {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "scheduler": args.scheduler,
            "weight_decays": args.weight_decays,
            "lrs": args.lrs,
            "seed": args.seed,
            "num_seeds": args.num_seeds,
            "target_acc": args.target_acc,
            "label_smoothing": args.label_smoothing,
            "swa_start": args.swa_start,
        }
        generate_report(results, report_cfg, save_path=args.report_path)
        print(f"\n  Report saved to {args.report_path}")


if __name__ == "__main__":
    main()
