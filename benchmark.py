"""Interactive optimizer benchmark.

Run with:
    python benchmark.py

You are first prompted to select one or more datasets, then one or more
models, then one or more optimizers.  Every (dataset, model, optimizer)
triple is trained in sequence and the results are plotted together for
easy comparison.

Optional CLI flags let you set shared hyperparameters without re-running:
    --epochs, --batch-size, --hidden-sizes, --lrs, --device, --save-plot
"""

import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from logger import TrainingLogger
from model import MLP, ResNet18, ViT
from optimizers import Lion, LAMB, Shampoo, Muon, Adan, AdaHessian, AdaBelief, SignSGD, AdaFactor, Sophia, Prodigy, ScheduleFreeAdamW
from train import (DATASET_INFO, SCHEDULER_REGISTRY, get_dataloaders,
                   linear_layer_names, make_param_groups, run_training,
                   save_checkpoint, set_seed)
from report import generate_report
from visualizer import plot_benchmark, plot_lr_sensitivity


# ---------------------------------------------------------------------------
# Registries
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
        "factory":    lambda p, lr, wd: torch.optim.SGD(p, lr=lr, weight_decay=wd),
        "default_lr": 0.01,
        "color":      "#e41a1c",   # red
    }),
    ("SGD+Momentum", {
        "factory":    lambda p, lr, wd: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd),
        "default_lr": 0.01,
        "color":      "#ff7f00",   # orange
    }),
    ("Adam",         {
        "factory":    lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#377eb8",   # blue
    }),
    ("Adagrad",      {
        "factory":    lambda p, lr, wd: torch.optim.Adagrad(p, lr=lr, weight_decay=wd),
        "default_lr": 0.01,
        "color":      "#4daf4a",   # green
    }),
    ("AdamW",        {
        "factory":    lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
        "default_lr": 0.001,
        "color":      "#984ea3",   # purple
    }),
    ("NAdam",        {
        "factory":    lambda p, lr, wd: torch.optim.NAdam(p, lr=lr, weight_decay=wd),
        "default_lr": 0.002,
        "color":      "#a65628",   # brown
    }),
    ("RAdam",        {
        "factory":    lambda p, lr, wd: torch.optim.RAdam(p, lr=lr, weight_decay=wd),
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
    # --- ADD YOUR CUSTOM OPTIMIZER HERE ---
    # ("MyOptimizer", {
    #     "factory":    lambda p, lr, wd: MyOptimizer(p, lr=lr, weight_decay=wd),
    #     "default_lr": 0.001,
    #     "color":      "#e377c2",
    # }),
])


# ---------------------------------------------------------------------------
# Interactive selection menus
# ---------------------------------------------------------------------------

def prompt_multiselect(title: str, options: list[str], descriptions: list[str] | None = None) -> list[str]:
    """Print a numbered menu and return the user-selected subset (multi-select)."""
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

def _aggregate_histories(histories: list[dict]) -> dict:
    """Aggregate N per-seed histories into mean ± std.

    List-valued keys are stacked and reduced along the seed axis.
    Scalar and special keys are handled individually.
    """
    import numpy as np

    if not histories:
        return {}

    result: dict = {}

    list_keys = [
        "train_loss", "test_loss", "train_acc", "test_acc", "learning_rates",
        "grad_norm_global", "grad_norm_before_clip", "grad_norm_std", "time_elapsed",
    ]
    for key in list_keys:
        vals = [h.get(key) or [] for h in histories]
        non_empty = [v for v in vals if v]
        if not non_empty:
            continue
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

    for key in ("weight_norms", "grad_norms"):
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
    criterion = nn.CrossEntropyLoss()
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

                        print(f"\n{'═' * 60}")
                        print(f"  Run {run_idx}/{total_runs}: {ds_name} | {mdl_name} | {series_name}  (lr={lr})")
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
                                optimizer = opt_info["factory"](params, lr, 0.0)
                            else:
                                optimizer = opt_info["factory"](model.parameters(), lr, 0.0)
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
                                "lr": lr, "epochs": epochs, "weight_decay": wd,
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
                            )
                            seed_histories.append(history)

                            if logger is not None:
                                log_cfg = {
                                    "dataset": ds_key, "model": mdl_name, "optimizer": opt_name,
                                    "lr": lr, "epochs": epochs, "batch_size": batch_size,
                                    "scheduler": scheduler_name, "patience": patience,
                                    "min_delta": min_delta, "max_grad_norm": max_grad_norm,
                                    "weight_decay": wd, "seed": base_seed + seed_idx,
                                }
                                logger.log_run(log_cfg, history)

                        aggregated = (
                            _aggregate_histories(seed_histories) if num_seeds > 1
                            else seed_histories[0]
                        )
                        aggregated["config_lr"] = lr
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
                   help="LR scheduler: none | cosine | step | warmup_cosine")
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
    if args.lrs is not None and len(args.lrs) == 1:
        print(f"  LR (global)  : {args.lrs[0]}")
    elif args.lrs is not None:
        print(f"  LR sweep     : {args.lrs}")
    else:
        print(f"  LR           : per-optimizer defaults")

    # ── Interactive selection ─────────────────────────────────────────────
    dataset_names   = prompt_multiselect("Select Datasets",   list(DATASET_REGISTRY.keys()))
    model_descs     = [MODEL_REGISTRY[m]["description"] for m in MODEL_REGISTRY]
    model_names     = prompt_multiselect("Select Models",     list(MODEL_REGISTRY.keys()), model_descs)
    optimizer_names = prompt_multiselect("Select Optimizers", list(OPTIMIZER_REGISTRY.keys()))

    print(f"\n  Selected datasets   : {', '.join(dataset_names)}")
    print(f"  Selected models     : {', '.join(model_names)}")
    print(f"  Selected optimizers : {', '.join(optimizer_names)}")

    # ── Determine series names and colors ─────────────────────────────────
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

    # ── Run ───────────────────────────────────────────────────────────────
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
    )
    logger.close()

    # ── Plot ──────────────────────────────────────────────────────────────
    save_path = args.save_plot if args.save_plot else None
    plot_benchmark(results, dataset_names, model_names, series_names, opt_colors, save_path=save_path)

    # ── LR Sensitivity Plot ───────────────────────────────────────────────
    if args.lrs is not None and len(args.lrs) > 1 and args.save_lr_plot:
        plot_lr_sensitivity(results, save_path=args.save_lr_plot)

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
        }
        generate_report(results, report_cfg, save_path=args.report_path)
        print(f"\n  Report saved to {args.report_path}")


if __name__ == "__main__":
    main()
