"""Interactive optimizer benchmark.

Run with:
    python benchmark.py

You are first prompted to select one or more datasets, then one or more
models, then one or more optimizers.  Every (dataset, model, optimizer)
triple is trained in sequence and the results are plotted together for
easy comparison.

Optional CLI flags let you set shared hyperparameters without re-running:
    --epochs, --batch-size, --hidden-sizes, --lr, --device, --save-plot
"""

import argparse
from collections import OrderedDict

import torch
import torch.nn as nn

from logger import TrainingLogger
from model import MLP, ResNet18, ViT
from optimizers import Lion, LAMB, Shampoo
from train import DATASET_INFO, SCHEDULER_REGISTRY, get_dataloaders, linear_layer_names, run_training
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
        "factory":    lambda p, lr: torch.optim.SGD(p, lr=lr),
        "default_lr": 0.01,
        "color":      "#e41a1c",   # red
    }),
    ("SGD+Momentum", {
        "factory":    lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
        "default_lr": 0.01,
        "color":      "#ff7f00",   # orange
    }),
    ("Adam",         {
        "factory":    lambda p, lr: torch.optim.Adam(p, lr=lr),
        "default_lr": 0.001,
        "color":      "#377eb8",   # blue
    }),
    ("Adagrad",      {
        "factory":    lambda p, lr: torch.optim.Adagrad(p, lr=lr),
        "default_lr": 0.01,
        "color":      "#4daf4a",   # green
    }),
    ("AdamW",        {
        "factory":    lambda p, lr: torch.optim.AdamW(p, lr=lr),
        "default_lr": 0.001,
        "color":      "#984ea3",   # purple
    }),
    ("NAdam",        {
        "factory":    lambda p, lr: torch.optim.NAdam(p, lr=lr),
        "default_lr": 0.002,
        "color":      "#a65628",   # brown
    }),
    ("RAdam",        {
        "factory":    lambda p, lr: torch.optim.RAdam(p, lr=lr),
        "default_lr": 0.001,
        "color":      "#f781bf",   # pink
    }),
    ("Lion",         {
        "factory":    lambda p, lr: Lion(p, lr=lr),
        "default_lr": 0.0001,
        "color":      "#17becf",   # teal
    }),
    ("LAMB",         {
        "factory":    lambda p, lr: LAMB(p, lr=lr),
        "default_lr": 0.001,
        "color":      "#8c564b",   # dark red
    }),
    ("Shampoo",      {
        "factory":    lambda p, lr: Shampoo(p, lr=lr),
        "default_lr": 0.01,
        "color":      "#bcbd22",   # yellow-green
    }),
    # --- ADD YOUR CUSTOM OPTIMIZER HERE ---
    # ("MyOptimizer", {
    #     "factory":    lambda p, lr: MyOptimizer(p, lr=lr),
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
    lr_override: float | None,
    logger: TrainingLogger | None = None,
    scheduler_name: str = "none",
    warmup_epochs: int = 5,
    patience: int = 0,
    min_delta: float = 0.0,
    max_grad_norm: float | None = None,
) -> dict:
    """Train every (dataset, model, optimizer) triple and collect histories.

    Returns
    -------
    dict mapping (dataset_name, model_name, optimizer_name) -> history_dict
    """
    results: dict = {}
    criterion = nn.CrossEntropyLoss()
    total_runs = len(dataset_names) * len(model_names) * len(optimizer_names)
    run_idx = 0

    for ds_name in dataset_names:
        ds_key = DATASET_REGISTRY[ds_name]
        ds_info = DATASET_INFO[ds_key]
        print(f"\n  Loading {ds_name}...")
        train_loader, test_loader = get_dataloaders(ds_key, batch_size, data_dir)

        for mdl_name in model_names:
            if ds_info.get("tabular") and mdl_name != "MLP":
                print(f"\n  Skipping {mdl_name} for tabular dataset '{ds_name}' (MLP only).")
                run_idx += len(optimizer_names)
                continue
            mdl_info = MODEL_REGISTRY[mdl_name]

            for opt_name in optimizer_names:
                run_idx += 1
                opt_info = OPTIMIZER_REGISTRY[opt_name]
                lr = lr_override if lr_override is not None else opt_info["default_lr"]

                print(f"\n{'═' * 60}")
                print(f"  Run {run_idx}/{total_runs}: {ds_name} | {mdl_name} | {opt_name}  (lr={lr})")
                print(f"{'═' * 60}")

                model = mdl_info["factory"](ds_info, hidden_sizes).to(device)
                optimizer = opt_info["factory"](model.parameters(), lr)
                layer_names = linear_layer_names(model)
                scheduler = SCHEDULER_REGISTRY[scheduler_name](optimizer, epochs, warmup_epochs)

                history = run_training(
                    model, train_loader, test_loader,
                    optimizer, criterion, device,
                    epochs, layer_names, verbose=True,
                    scheduler=scheduler,
                    patience=patience,
                    min_delta=min_delta,
                    max_grad_norm=max_grad_norm,
                )
                results[(ds_name, mdl_name, opt_name)] = history

                if logger is not None:
                    config = {
                        "dataset":       ds_key,
                        "model":         mdl_name,
                        "optimizer":     opt_name,
                        "lr":            lr,
                        "epochs":        epochs,
                        "batch_size":    batch_size,
                        "scheduler":     scheduler_name,
                        "patience":      patience,
                        "min_delta":     min_delta,
                        "max_grad_norm": max_grad_norm,
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
    p.add_argument("--lr",           default=None,       type=float,
                   help="Override all optimizer default LRs (omit to use per-optimizer defaults)")
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
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    print(f"\n  Device       : {device}")
    print(f"  Epochs       : {args.epochs}")
    if args.lr is not None:
        print(f"  LR (global)  : {args.lr}")
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

    # ── Run ───────────────────────────────────────────────────────────────
    logger = TrainingLogger(log_dir=args.log_dir)
    results = run_benchmark(
        dataset_names, model_names, optimizer_names,
        hidden_sizes=args.hidden_sizes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        device=device,
        lr_override=args.lr,
        logger=logger,
        scheduler_name=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        max_grad_norm=args.max_grad_norm,
    )
    logger.close()

    # ── Plot ──────────────────────────────────────────────────────────────
    opt_colors = {name: OPTIMIZER_REGISTRY[name]["color"] for name in optimizer_names}
    save_path  = args.save_plot if args.save_plot else None
    plot_benchmark(results, dataset_names, model_names, optimizer_names, opt_colors, save_path=save_path)


if __name__ == "__main__":
    main()
