"""Training script for MNIST / FashionMNIST / CIFAR-10 / Tiny ImageNet classification.

Swap the optimizer by changing the `build_optimizer` function or by passing
`--optimizer` on the command line.

Built-in choices  : adam | sgd | rmsprop | vanilla_sgd
Custom optimizer  : add your class to optimizers/ and register it below.
"""

import argparse
import os
import shutil
import time
import urllib.request
import zipfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from logger import TrainingLogger
from model import MLP, ResNet18, ViT
from visualizer import Visualizer

# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------
# To plug in your own optimizer:
#   1. Implement it in optimizers/<your_file>.py (subclass BaseOptimizer).
#   2. Import it here.
#   3. Add an entry to OPTIMIZER_REGISTRY.
# ---------------------------------------------------------------------------

from optimizers import VanillaSGD, Lion, LAMB, Shampoo

OPTIMIZER_REGISTRY: dict = {
    # PyTorch built-ins
    "adam":        lambda p, lr: torch.optim.Adam(p, lr=lr),
    "adamw":       lambda p, lr: torch.optim.AdamW(p, lr=lr),
    "nadam":       lambda p, lr: torch.optim.NAdam(p, lr=lr),
    "radam":       lambda p, lr: torch.optim.RAdam(p, lr=lr),
    "adagrad":     lambda p, lr: torch.optim.Adagrad(p, lr=lr),
    "sgd":         lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
    "rmsprop":     lambda p, lr: torch.optim.RMSprop(p, lr=lr),
    # Custom implementations
    "vanilla_sgd": lambda p, lr: VanillaSGD(p, lr=lr),
    "lion":        lambda p, lr: Lion(p, lr=lr),
    "lamb":        lambda p, lr: LAMB(p, lr=lr),
    "shampoo":     lambda p, lr: Shampoo(p, lr=lr),
    # --- ADD YOUR OPTIMIZER HERE ---
    # "my_optimizer": lambda p, lr: MyOptimizer(p, lr=lr),
}

# ---------------------------------------------------------------------------
# Scheduler registry
# ---------------------------------------------------------------------------

SCHEDULER_REGISTRY: dict = {
    "none": lambda opt, epochs, warmup: None,
    "cosine": lambda opt, epochs, warmup: torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=0.0
    ),
    "step": lambda opt, epochs, warmup: torch.optim.lr_scheduler.StepLR(
        opt, step_size=max(1, epochs // 3), gamma=0.1
    ),
    "warmup_cosine": lambda opt, epochs, warmup: torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, end_factor=1.0, total_iters=max(1, warmup)
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, epochs - warmup), eta_min=0.0
            ),
        ],
        milestones=[max(1, warmup)],
    ),
}


def build_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    return OPTIMIZER_REGISTRY[name](params, lr)


# ---------------------------------------------------------------------------
# Dataset info (exported for benchmark.py)
# ---------------------------------------------------------------------------

DATASET_INFO: dict = {
    # Image datasets
    "mnist":          {"in_channels": 1, "input_size": 784,   "num_classes": 10},
    "fashion_mnist":  {"in_channels": 1, "input_size": 784,   "num_classes": 10},
    "cifar10":        {"in_channels": 3, "input_size": 3072,  "num_classes": 10},
    "cifar100":       {"in_channels": 3, "input_size": 3072,  "num_classes": 100},
    "tiny_imagenet":  {"in_channels": 3, "input_size": 12288, "num_classes": 200},
    # Synthetic tabular datasets (MLP only)
    "illcond":        {"in_channels": 1, "input_size": 64,    "num_classes": 2,  "tabular": True},
    "sparse":         {"in_channels": 1, "input_size": 100,   "num_classes": 2,  "tabular": True},
    "noisy_grad":     {"in_channels": 1, "input_size": 64,    "num_classes": 2,  "tabular": True},
    "manifold":       {"in_channels": 1, "input_size": 64,    "num_classes": 2,  "tabular": True},
    "saddle":         {"in_channels": 1, "input_size": 64,    "num_classes": 2,  "tabular": True},
}

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_DATASET_STATS = {
    "mnist":         ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10":       ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100":      ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "tiny_imagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}

_TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _setup_tiny_imagenet(data_dir: str) -> str:
    """Download and reorganise Tiny ImageNet, returning the dataset root path.

    The raw archive has two quirks that prevent ImageFolder from working
    out-of-the-box:
    - Train images live in  train/<class>/images/<file>.JPEG  (extra subdirectory)
    - Val images are flat:  val/images/<file>.JPEG  with labels in val_annotations.txt

    This function fixes both, then writes a .setup_done marker so subsequent
    calls skip all work.
    """
    root = os.path.join(data_dir, "tiny-imagenet-200")
    marker = os.path.join(root, ".setup_done")
    if os.path.exists(marker):
        return root

    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "tiny-imagenet-200.zip")

    if not os.path.exists(zip_path):
        print("Downloading Tiny ImageNet (~236 MB) ...")
        urllib.request.urlretrieve(_TINY_IMAGENET_URL, zip_path)
        print("Download complete.")

    if not os.path.exists(root):
        print("Extracting archive ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print("Extraction complete.")

    # ── Reorganise training set ──────────────────────────────────────────
    # Before: train/<class>/images/<file>.JPEG
    # After:  train/<class>/<file>.JPEG
    train_dir = os.path.join(root, "train")
    print("Reorganising training set ...")
    for class_id in os.listdir(train_dir):
        img_subdir = os.path.join(train_dir, class_id, "images")
        if os.path.isdir(img_subdir):
            for fname in os.listdir(img_subdir):
                shutil.move(os.path.join(img_subdir, fname),
                            os.path.join(train_dir, class_id, fname))
            os.rmdir(img_subdir)

    # ── Reorganise validation set ────────────────────────────────────────
    # Before: val/images/<file>.JPEG  (flat, class in val_annotations.txt)
    # After:  val/<class>/<file>.JPEG
    val_dir = os.path.join(root, "val")
    val_img_dir = os.path.join(val_dir, "images")
    val_ann_file = os.path.join(val_dir, "val_annotations.txt")
    if os.path.isdir(val_img_dir):
        print("Reorganising validation set ...")
        with open(val_ann_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, class_id = parts[0], parts[1]
                class_dir = os.path.join(val_dir, class_id)
                os.makedirs(class_dir, exist_ok=True)
                src = os.path.join(val_img_dir, fname)
                if os.path.exists(src):
                    shutil.move(src, os.path.join(class_dir, fname))
        shutil.rmtree(val_img_dir)

    open(marker, "w").close()
    print("Tiny ImageNet setup complete.")
    return root


def get_dataloaders(dataset: str = "mnist", batch_size: int = 128, data_dir: str = "./data"):
    """Return (train_loader, test_loader) for the named dataset."""
    dataset = dataset.lower()
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from {list(DATASET_INFO)}")

    # Synthetic tabular datasets are self-normalising — delegate immediately
    if DATASET_INFO[dataset].get("tabular"):
        from synthetic_datasets import SYNTHETIC_LOADERS
        return SYNTHETIC_LOADERS[dataset](batch_size=batch_size)

    mean, std = _DATASET_STATS[dataset]

    if dataset in ("cifar10", "cifar100"):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
        train_set = cls(data_dir, train=True,  download=True, transform=train_transform)
        test_set  = cls(data_dir, train=False, download=True, transform=test_transform)
    elif dataset == "tiny_imagenet":
        root = _setup_tiny_imagenet(data_dir)
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        train_set = datasets.ImageFolder(os.path.join(root, "train"), transform=train_transform)
        test_set  = datasets.ImageFolder(os.path.join(root, "val"),   transform=test_transform)
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        train_set = cls(data_dir, train=True,  download=True, transform=transform)
        test_set  = cls(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_name: str, dataset_info: dict, hidden_sizes: list[int]) -> nn.Module:
    """Instantiate a model from its name and dataset metadata.

    Parameters
    ----------
    model_name   : "mlp" or "resnet18"
    dataset_info : dict with keys in_channels, input_size, num_classes
    hidden_sizes : hidden layer widths (only used by MLP)
    """
    name = model_name.lower()
    if dataset_info.get("tabular") and name != "mlp":
        raise ValueError(
            f"Model '{model_name}' is not supported for tabular datasets. Use MLP."
        )
    if name == "mlp":
        return MLP(
            input_size=dataset_info["input_size"],
            hidden_sizes=hidden_sizes,
            num_classes=dataset_info["num_classes"],
        )
    elif name == "resnet18":
        return ResNet18(
            in_channels=dataset_info["in_channels"],
            num_classes=dataset_info["num_classes"],
        )
    elif name == "vit":
        image_size = int((dataset_info["input_size"] / dataset_info["in_channels"]) ** 0.5)
        return ViT(
            image_size=image_size,
            in_channels=dataset_info["in_channels"],
            num_classes=dataset_info["num_classes"],
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: mlp, resnet18, vit")


# ---------------------------------------------------------------------------
# Parameter tracking helpers
# ---------------------------------------------------------------------------

def linear_layer_names(model: nn.Module) -> list[str]:
    """Return short display names for each Linear layer's weight tensor."""
    names = []
    idx = 1
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            in_f, out_f = module.in_features, module.out_features
            names.append(f"L{idx}({in_f}→{out_f})")
            idx += 1
    return names


def weight_norms(model: nn.Module, layer_names: list[str]) -> dict[str, float]:
    """L2 norm of each Linear layer's weight matrix."""
    norms = {}
    idx = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            norms[layer_names[idx]] = module.weight.norm().item()
            idx += 1
    return norms


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, device, layer_names,
    global_step_offset: int = 0,
) -> dict:
    """Run one training epoch.

    Returns
    -------
    dict with keys:
        loss                : float  — average per-sample cross-entropy
        acc                 : float  — fraction of correct predictions
        grad_norms_per_layer: dict[str, float]  — mean grad L2 norm per linear layer
        grad_norm_global    : float  — L2 norm across all parameter gradients
        grad_norm_std       : float  — std of per-parameter gradient norms
        step_losses         : list[float]  — per-batch loss values
    """
    model.train()
    total_loss, correct = 0.0, 0
    grad_accum = {n: 0.0 for n in layer_names}
    num_batches = 0
    step_losses: list[float] = []
    all_grad_norms: list[float] = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Accumulate gradient norms per linear layer
        idx = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    grad_accum[layer_names[idx]] += module.weight.grad.norm().item()
                idx += 1

        # Global gradient norm across all parameters
        param_grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        all_grad_norms.extend(param_grad_norms)

        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        step_losses.append(loss.item())
        num_batches += 1

    n = len(loader.dataset)
    avg_grad_norms = {name: total / num_batches for name, total in grad_accum.items()}

    import statistics
    grad_norm_global = float(sum(g ** 2 for g in all_grad_norms) ** 0.5) if all_grad_norms else 0.0
    grad_norm_std = float(statistics.stdev(all_grad_norms)) if len(all_grad_norms) > 1 else 0.0

    return {
        "loss": total_loss / n,
        "acc": correct / n,
        "grad_norms_per_layer": avg_grad_norms,
        "grad_norm_global": grad_norm_global,
        "grad_norm_std": grad_norm_std,
        "step_losses": step_losses,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Multi-epoch training (shared by train.py and benchmark.py)
# ---------------------------------------------------------------------------

def run_training(
    model, train_loader, test_loader, optimizer, criterion,
    device, epochs, layer_names, verbose=True,
    compute_hessian: bool = False,
    target_acc: float = 0.95,
    scheduler=None,
) -> dict:
    """Train for *epochs* and return a history dict with per-epoch metrics.

    Returns
    -------
    dict with keys:
        train_loss, test_loss      : list[float]
        train_acc,  test_acc       : list[float]  (fractions, not percentages)
        weight_norms               : dict[layer_name, list[float]]
        grad_norms                 : dict[layer_name, list[float]]
        time_elapsed               : list[float]  — seconds since start per epoch
        target_accuracy_epoch      : int | None
        target_accuracy_time       : float | None
        grad_norm_global           : list[float]
        grad_norm_std              : list[float]
        hessian_trace              : list[float]  — NaN if compute_hessian=False
        sharpness                  : list[float]  — NaN if compute_hessian=False
        step_losses                : list[float]  — all batch-level losses across epochs
        learning_rates             : list[float]  — LR at start of each epoch
    """
    from metrics import compute_hessian_trace, compute_sharpness

    history: dict = {
        "train_loss": [], "test_loss": [],
        "train_acc":  [], "test_acc":  [],
        "weight_norms": {n: [] for n in layer_names},
        "grad_norms":   {n: [] for n in layer_names},
        "time_elapsed": [],
        "target_accuracy_epoch": None,
        "target_accuracy_time": None,
        "grad_norm_global": [],
        "grad_norm_std": [],
        "hessian_trace": [],
        "sharpness": [],
        "step_losses": [],
        "learning_rates": [],
    }

    # Fixed batch for Hessian/sharpness (reuse same batch each epoch)
    hessian_batch = None
    if compute_hessian:
        for batch in train_loader:
            hessian_batch = batch
            break

    criterion_no_reduce = nn.CrossEntropyLoss()
    start_time = time.time()
    global_step_offset = 0

    for epoch in range(1, epochs + 1):
        # Record LR before this epoch's training
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        epoch_result = train_one_epoch(
            model, train_loader, optimizer, criterion, device, layer_names,
            global_step_offset=global_step_offset,
        )
        global_step_offset += len(epoch_result["step_losses"])

        train_loss = epoch_result["loss"]
        train_acc  = epoch_result["acc"]
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        w_norms = weight_norms(model, layer_names)
        elapsed = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["time_elapsed"].append(elapsed)
        history["grad_norm_global"].append(epoch_result["grad_norm_global"])
        history["grad_norm_std"].append(epoch_result["grad_norm_std"])
        history["step_losses"].extend(epoch_result["step_losses"])

        for n in layer_names:
            history["weight_norms"][n].append(w_norms.get(n, float("nan")))
            history["grad_norms"][n].append(epoch_result["grad_norms_per_layer"].get(n, float("nan")))

        if history["target_accuracy_epoch"] is None and test_acc >= target_acc:
            history["target_accuracy_epoch"] = epoch
            history["target_accuracy_time"] = elapsed

        # Hessian / sharpness
        if compute_hessian and hessian_batch is not None:
            ht = compute_hessian_trace(model, hessian_batch, criterion_no_reduce, device)
            sh = compute_sharpness(model, hessian_batch, criterion_no_reduce, device)
        else:
            ht = float("nan")
            sh = float("nan")
        history["hessian_trace"].append(ht)
        history["sharpness"].append(sh)

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(
                f"  Epoch {epoch:>3}/{epochs} | "
                f"train loss {train_loss:.4f}  acc {train_acc*100:.2f}% | "
                f"test  loss {test_loss:.4f}  acc {test_acc*100:.2f}%"
            )

    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Single-run trainer (MNIST / FashionMNIST / CIFAR-10)")
    p.add_argument("--dataset",      default="mnist",    help="mnist | fashion_mnist | cifar10 | cifar100 | tiny_imagenet | illcond | sparse | noisy_grad | manifold | saddle")
    p.add_argument("--model",        default="mlp",      help="mlp | resnet18 | vit")
    p.add_argument("--optimizer",    default="adam",     help="Optimizer name (see OPTIMIZER_REGISTRY)")
    p.add_argument("--lr",           default=1e-3,       type=float, help="Learning rate")
    p.add_argument("--epochs",       default=10,         type=int)
    p.add_argument("--batch-size",   default=128,        type=int)
    p.add_argument("--hidden-sizes", default=[256, 128], type=int, nargs="+",
                   metavar="N",
                   help="Width of each hidden layer (MLP only). "
                        "Example: --hidden-sizes 512 256 128")
    p.add_argument("--data-dir",     default="./data")
    p.add_argument("--device",       default="auto",     help="cpu | cuda | mps | auto")
    p.add_argument("--no-plot",      action="store_true", help="Disable live visualisation")
    p.add_argument("--save-plot",    default="plots/training_curves.png",
                   help="Path to save the final training figure ('' to disable)")
    p.add_argument("--log-dir",      default="logs",
                   help="Root directory for training logs (default: logs/)")
    p.add_argument("--hessian",      action="store_true",
                   help="Compute Hessian trace and sharpness each epoch (slow)")
    p.add_argument("--scheduler",    default="none",
                   choices=list(SCHEDULER_REGISTRY.keys()),
                   help="LR scheduler: none | cosine | step | warmup_cosine")
    p.add_argument("--warmup-epochs", default=5, type=int,
                   help="Linear warmup epochs for warmup_cosine (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()

    # Device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device : {device}")
    print(f"Dataset      : {args.dataset}")
    print(f"Model        : {args.model}")

    # Build model
    ds_info = DATASET_INFO[args.dataset.lower()]
    model = build_model(args.model, ds_info, args.hidden_sizes).to(device)
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    layer_names = linear_layer_names(model)

    # Components
    train_loader, test_loader = get_dataloaders(args.dataset, args.batch_size, args.data_dir)
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = SCHEDULER_REGISTRY[args.scheduler](optimizer, args.epochs, args.warmup_epochs)
    print(f"Optimizer    : {optimizer.__class__.__name__}  (lr={args.lr})")
    print(f"Scheduler    : {args.scheduler}")
    print()

    # Visualizer
    save_path = args.save_plot if args.save_plot else None
    use_vis = (not args.no_plot) or (save_path is not None)
    vis = Visualizer(layer_names, live=not args.no_plot, save_path=save_path) if use_vis else None

    # Hessian batch (fixed subset for diagnostics)
    hessian_batch = None
    if args.hessian:
        for batch in train_loader:
            hessian_batch = batch
            break

    from metrics import compute_hessian_trace, compute_sharpness
    criterion_diag = nn.CrossEntropyLoss()

    # History collected for logging
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
        "time_elapsed": [], "step_losses": [],
        "learning_rates": [],
    }
    run_start = time.time()

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        epoch_result = train_one_epoch(
            model, train_loader, optimizer, criterion, device, layer_names
        )
        train_loss = epoch_result["loss"]
        train_acc  = epoch_result["acc"]
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        w_norms = weight_norms(model, layer_names)

        ht = float("nan")
        sh = float("nan")
        if args.hessian and hessian_batch is not None:
            ht = compute_hessian_trace(model, hessian_batch, criterion_diag, device)
            sh = compute_sharpness(model, hessian_batch, criterion_diag, device)

        print(
            f"Epoch {epoch:>3}/{args.epochs} | "
            f"train loss {train_loss:.4f}  acc {train_acc*100:.2f}% | "
            f"test  loss {test_loss:.4f}  acc {test_acc*100:.2f}%"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["time_elapsed"].append(time.time() - run_start)
        history["step_losses"].extend(epoch_result["step_losses"])

        if vis is not None:
            vis.update(
                epoch, train_loss, test_loss, train_acc, test_acc,
                w_norms, epoch_result["grad_norms_per_layer"],
                grad_norm_global=epoch_result["grad_norm_global"],
                grad_norm_std=epoch_result["grad_norm_std"],
                step_losses=epoch_result["step_losses"],
                hessian_trace=ht,
                sharpness=sh,
                learning_rate=current_lr,
            )

        if scheduler is not None:
            scheduler.step()

    if vis is not None:
        vis.close()

    # Write logs
    config = {
        "dataset":      args.dataset,
        "model":        args.model,
        "optimizer":    args.optimizer,
        "lr":           args.lr,
        "epochs":       args.epochs,
        "batch_size":   args.batch_size,
        "hidden_sizes": args.hidden_sizes,
        "scheduler":    args.scheduler,
    }
    logger = TrainingLogger(log_dir=args.log_dir)
    logger.log_run(config, history)
    logger.close()


if __name__ == "__main__":
    main()
