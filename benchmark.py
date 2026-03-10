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
from optimizers import Lion, LAMB, Shampoo, Muon, Adan, AdaHessian
from train import (DATASET_INFO, SCHEDULER_REGISTRY, get_dataloaders,
                   linear_layer_names, make_param_groups, run_training,
                   save_checkpoint, set_seed)
from visualizer import plot_benchmark


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
) -> dict:
    """Train every (dataset, model, optimizer, weight_decay) combination and collect histories.

    Returns
    -------
    dict mapping (dataset_name, model_name, series_name) -> history_dict
    """
    if weight_decays is None:
        weight_decays = [0.0]

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

                        model = mdl_info["factory"](ds_info, hidden_sizes).to(device)
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
                            checkpoint_dir=run_ckpt_dir,
                            checkpoint_config=run_cfg,
                        )
                        results[(ds_name, mdl_name, series_name)] = history

                        if logger is not None:
                            config = {
                                "dataset": ds_key, "model": mdl_name, "optimizer": opt_name,
                                "lr": lr, "epochs": epochs, "batch_size": batch_size,
                                "scheduler": scheduler_name, "patience": patience,
                                "min_delta": min_delta, "max_grad_norm": max_grad_norm,
                                "weight_decay": wd, "seed": seed,
                            }
                            logger.log_run(config, history)

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
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = get_device(args.device)

    print(f"\n  Device       : {device}")
    print(f"  Epochs       : {args.epochs}")
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
    )
    logger.close()

    # ── Plot ──────────────────────────────────────────────────────────────
    save_path = args.save_plot if args.save_plot else None
    plot_benchmark(results, dataset_names, model_names, series_names, opt_colors, save_path=save_path)


if __name__ == "__main__":
    main()
