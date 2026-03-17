"""Single-run trainer for image and tabular classification.

Supports datasets  : mnist | fashion_mnist | cifar10 | cifar100 | tiny_imagenet
                     + 5 synthetic tabular datasets (illcond, sparse, noisy_grad,
                     manifold, saddle)
Supports models    : mlp | resnet18 | vit
Supports optimizers: 20 choices — see OPTIMIZER_REGISTRY below for the full list.

Quickstart
----------
    python train.py                                  # Adam on MNIST (defaults)
    python train.py --optimizer sgd --lr 0.01        # SGD on MNIST
    python train.py --dataset cifar10 --model resnet18 --scheduler cosine
    python train.py --ema-decay 0.999 --swa-start 5 --num-epochs 10

To add a custom optimizer
-------------------------
    1. Implement it in optimizers/<your_file>.py (subclass BaseOptimizer).
    2. Import it in this file.
    3. Add a lambda entry to OPTIMIZER_REGISTRY below.
"""

import argparse
import contextlib
import os
import random
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
from lr_finder import LRFinder

# ---------------------------------------------------------------------------
# Optimizer registry
# ---------------------------------------------------------------------------
# To plug in your own optimizer:
#   1. Implement it in optimizers/<your_file>.py (subclass BaseOptimizer).
#   2. Import it here.
#   3. Add an entry to OPTIMIZER_REGISTRY.
# ---------------------------------------------------------------------------

import numpy as np

from optimizers import VanillaSGD, Lion, LAMB, Shampoo, Muon, Adan, AdaHessian, AdaBelief, SignSGD, AdaFactor, Sophia, Prodigy, ScheduleFreeAdamW

# Optimizers that call loss.backward(create_graph=True) inside their own step()
# to estimate Hessian-vector products.  GradScaler's loss scaling corrupts the
# second-order gradient graph, so AMP must be disabled when these are in use.
_AMP_INCOMPATIBLE = (Sophia, AdaHessian)

# Accuracy thresholds used for the convergence profile.  For each threshold T
# the training loop records the first epoch and wall-clock time at which
# test_acc >= T.  All five milestones are tracked simultaneously so that a
# single run tells you "when did it hit 50%, 75%, 90%, 95%, 99%?".
_CONV_THRESHOLDS = [0.50, 0.75, 0.90, 0.95, 0.99]

# torch.compile traces the computation graph statically.  Optimizers that call
# backward(create_graph=True) build a dynamic higher-order graph that the
# compiler cannot capture correctly, leading to incorrect gradients or crashes.
_COMPILE_INCOMPATIBLE = (Sophia, AdaHessian)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all relevant random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

OPTIMIZER_REGISTRY: dict = {
    # PyTorch built-ins
    "adam":        lambda p, lr, wd: torch.optim.Adam(p, lr=lr, weight_decay=wd),
    "adamw":       lambda p, lr, wd: torch.optim.AdamW(p, lr=lr, weight_decay=wd),
    "nadam":       lambda p, lr, wd: torch.optim.NAdam(p, lr=lr, weight_decay=wd),
    "radam":       lambda p, lr, wd: torch.optim.RAdam(p, lr=lr, weight_decay=wd),
    "adagrad":     lambda p, lr, wd: torch.optim.Adagrad(p, lr=lr, weight_decay=wd),
    "sgd":         lambda p, lr, wd: torch.optim.SGD(p, lr=lr, momentum=0.9, weight_decay=wd),
    "rmsprop":     lambda p, lr, wd: torch.optim.RMSprop(p, lr=lr, weight_decay=wd),
    # Custom implementations
    "vanilla_sgd": lambda p, lr, wd: VanillaSGD(p, lr=lr, weight_decay=wd),
    "lion":        lambda p, lr, wd: Lion(p, lr=lr, weight_decay=wd),
    "lamb":        lambda p, lr, wd: LAMB(p, lr=lr, weight_decay=wd),
    "shampoo":     lambda p, lr, wd: Shampoo(p, lr=lr, weight_decay=wd),
    "muon":        lambda p, lr, wd: Muon(p, lr=lr, weight_decay=wd),
    "adan":        lambda p, lr, wd: Adan(p, lr=lr, weight_decay=wd),
    "adahessian":  lambda p, lr, wd: AdaHessian(p, lr=lr, weight_decay=wd),
    "adabelief":   lambda p, lr, wd: AdaBelief(p, lr=lr, weight_decay=wd),
    "signsgd":     lambda p, lr, wd: SignSGD(p, lr=lr, weight_decay=wd),
    "adafactor":   lambda p, lr, wd: AdaFactor(p, lr=lr, weight_decay=wd),
    "sophia":      lambda p, lr, wd: Sophia(p, lr=lr, weight_decay=wd),
    "prodigy":     lambda p, lr, wd: Prodigy(p, lr=lr, weight_decay=wd),
    "sf_adamw":    lambda p, lr, wd: ScheduleFreeAdamW(p, lr=lr, weight_decay=wd),
    # --- ADD YOUR OPTIMIZER HERE ---
    # "my_optimizer": lambda p, lr, wd: MyOptimizer(p, lr=lr, weight_decay=wd),
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
    # T_0 = epochs // 3 gives ~3 restarts over a full run (standard SGDR setup).
    # T_mult = 1 keeps the restart period constant.
    "cosine_wr": lambda opt, epochs, warmup: torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(1, epochs // 3), T_mult=1, eta_min=0.0
    ),
}


class EarlyStopping:
    """Monitors val loss; signals when to stop and saves the best model state."""
    def __init__(self, patience: int = 0, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update state. Returns True if training should stop."""
        if not self.enabled:
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module, device: torch.device) -> None:
        """Load best weights back into model (no-op if no checkpoint saved)."""
        if self.best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})


# Optimizer state keys to track (mean abs value per linear layer per epoch)
_OPT_STATE_KEYS = ("exp_avg_sq", "exp_avg", "hessian")


def _extract_optimizer_states(
    model: nn.Module,
    optimizer,
    layer_names: list[str],
) -> dict:
    """Read per-layer optimizer state scalars after a training step.

    Returns a dict with entries of two shapes:
      ``state_key -> {layer_name: float}``  — per-parameter tensor states
      ``"d"       -> float``                — Prodigy's scalar LR estimate

    Parameters are matched to ``layer_names`` by position (the same ordering
    that ``linear_layer_names()`` and ``weight_norms()`` use), so display
    names like ``"L1(64→32)"`` are correctly assigned.
    Only each layer's weight matrix (ndim ≥ 2) is read; biases are skipped.
    """
    result: dict = {}

    layer_idx = 0
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if layer_idx >= len(layer_names):
            break
        layer_name = layer_names[layer_idx]
        layer_idx += 1

        param = module.weight
        if param not in optimizer.state:
            continue
        param_state = optimizer.state[param]
        for key in _OPT_STATE_KEYS:
            if key not in param_state:
                continue
            val = param_state[key].detach().float().abs().mean().item()
            result.setdefault(key, {})[layer_name] = val

    # Prodigy stores its effective-LR estimate d in param_groups (not per-param state)
    for pg in optimizer.param_groups:
        if "d" in pg:
            result["d"] = float(pg["d"])
            break

    return result


def save_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    metrics: dict,
    config: dict | None = None,
) -> None:
    """Save a training checkpoint to *path*.

    The saved dict contains:
      epoch, model_state_dict, optimizer_state_dict, metrics, config
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":              metrics,
        "config":               config or {},
    }, path)


def make_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters into decay / no-decay groups.

    1-D parameters (biases, LayerNorm and BatchNorm weights/biases) are
    excluded from weight decay. All other parameters (weight matrices,
    conv filters, embeddings) are placed in the decay group.
    """
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float = 0.0):
    """Instantiate an optimizer by name.

    When weight_decay > 0, parameters are split into two groups via
    make_param_groups() so that 1-D params (biases, norms) are exempt.
    The weight_decay argument is passed to the decay group only; the
    factory always receives wd=0.0 because the per-group dicts already
    carry the correct values.

    Parameters
    ----------
    name         : key from OPTIMIZER_REGISTRY (case-insensitive)
    model        : the nn.Module to optimise
    lr           : learning rate
    weight_decay : L2 penalty on weight matrices / conv filters (default 0)
    """
    name = name.lower()
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )
    if weight_decay > 0.0:
        params = make_param_groups(model, weight_decay)
        return OPTIMIZER_REGISTRY[name](params, lr, 0.0)  # wd set per-group
    return OPTIMIZER_REGISTRY[name](model.parameters(), lr, 0.0)


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
    model_name   : "mlp" | "resnet18" | "vit"
    dataset_info : dict with keys in_channels, input_size, num_classes.
                   Tabular datasets must use "mlp"; others raise ValueError.
    hidden_sizes : hidden layer widths (only used by MLP, ignored for others)
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


def weight_distances(
    model: nn.Module,
    layer_names: list[str],
    init_weights: dict[str, torch.Tensor],
) -> dict[str, float]:
    """L2 distance of each Linear layer's weight from its initial value.

    Parameters
    ----------
    model       : current model (weights have been updated by the optimizer)
    layer_names : ordered layer display names (same list used by weight_norms)
    init_weights: {layer_name: weight_tensor_at_epoch_0} — snapshot taken
                  before training begins; must be on the same device as model

    Returns
    -------
    dict mapping layer_name -> ||w_t - w_0||_2  (scalar float)
    """
    dists = {}
    idx = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            name = layer_names[idx]
            dists[name] = (module.weight - init_weights[name]).norm().item()
            idx += 1
    return dists


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, loader, optimizer, criterion, device, layer_names,
    global_step_offset: int = 0,
    max_grad_norm: float | None = None,
    amp: bool = False,
    scaler=None,
    ema_state: dict | None = None,
    ema_decay: float = 0.999,
) -> dict:
    """Run one training epoch and return per-epoch metrics.

    Parameters
    ----------
    model            : nn.Module to train (set to train() internally)
    loader           : DataLoader providing (images, labels) batches
    optimizer        : the torch.optim.Optimizer to step
    criterion        : loss function (e.g. CrossEntropyLoss with label smoothing)
    device           : torch.device
    layer_names      : list of display names from linear_layer_names(); used for
                       per-layer gradient accumulation
    global_step_offset: step index at the start of this epoch (for multi-epoch
                       step counting; not used by this function but kept for
                       external callers that need absolute step numbers)
    max_grad_norm    : if set, clip global gradient L2 norm to this value before
                       the optimizer step; None = no clipping
    amp              : enable float16 autocast (GPU/MPS only)
    scaler           : GradScaler for CUDA AMP (None if not using AMP)
    ema_state        : if not None, a dict {param_name: tensor} that is updated
                       in-place after each optimizer step using exponential moving
                       average: ema = ema * decay + param * (1 - decay)
    ema_decay        : EMA coefficient (only used when ema_state is not None)

    Returns
    -------
    dict with keys:
        loss                 : float  — average per-sample cross-entropy
        acc                  : float  — fraction of correct predictions in [0, 1]
        grad_norms_per_layer : dict[str, float]  — mean grad L2 norm per linear layer
        grad_norm_global     : float  — global L2 norm across all params (post-clip)
        grad_norm_std        : float  — std of per-parameter gradient norms
        grad_norm_before_clip: float  — global L2 norm (pre-clip; equals
                               grad_norm_global when max_grad_norm is None)
        step_losses          : list[float]  — per-batch loss values
    """
    model.train()
    total_loss, correct = 0.0, 0
    grad_accum    = {n: 0.0 for n in layer_names}
    # Tracks the sum-of-squares of per-batch gradient norms per layer.
    # Together with grad_accum (sum of norms), this lets us compute
    # SNR = mean_norm / std_norm without storing all per-batch tensors.
    grad_sq_accum = {n: 0.0 for n in layer_names}
    # Accumulates the actual gradient tensor per layer (not just the norm).
    # Divided by num_batches after the loop to give the mean gradient direction,
    # which is used upstream to compute cosine similarity between epochs.
    grad_vec_accum: dict[str, torch.Tensor | None] = {n: None for n in layer_names}
    num_batches = 0
    step_losses: list[float] = []
    all_raw_norms: list[float] = []   # pre-clip per-param norms across all batches
    all_grad_norms: list[float] = []  # post-clip per-param norms across all batches

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if amp else contextlib.nullcontext()
        )
        with amp_ctx:
            logits = model(images)
            loss = criterion(logits, labels)

        if getattr(optimizer, "requires_create_graph", False):
            # Use autograd.grad so gradients are differentiable graph nodes
            # (needed for HVP computation in second-order optimizers).
            # AMP is always disabled upstream when this path is taken.
            trainable = [p for p in model.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, trainable, create_graph=True)
            for p, g in zip(trainable, grads):
                p.grad = g
        elif scaler is not None:
            # CUDA AMP: scaled backward, then unscale so gradient norms are
            # in full precision before we read or clip them.
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Accumulate gradient norms and gradient vectors per linear layer.
        # Norms: both sum and sum-of-squares for SNR = mean/std computation.
        # Vectors: running sum of weight.grad tensors; divided by num_batches
        # after the loop to give the mean gradient direction for this epoch.
        idx = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if module.weight.grad is not None:
                    _g = module.weight.grad.detach()
                    _gnorm = _g.norm().item()
                    grad_accum[layer_names[idx]]    += _gnorm
                    grad_sq_accum[layer_names[idx]] += _gnorm * _gnorm
                    if grad_vec_accum[layer_names[idx]] is None:
                        grad_vec_accum[layer_names[idx]] = _g.clone()
                    else:
                        grad_vec_accum[layer_names[idx]].add_(_g)
                idx += 1

        # Pre-clip global gradient norm for this batch
        param_grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        all_raw_norms.extend(param_grad_norms)

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            post_clip_norms = [
                p.grad.norm().item()
                for p in model.parameters()
                if p.grad is not None
            ]
            all_grad_norms.extend(post_clip_norms)
        else:
            all_grad_norms.extend(param_grad_norms)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # EMA weight update (after every optimizer step)
        if ema_state is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if k in ema_state:
                        ema_state[k].mul_(ema_decay).add_(v, alpha=1.0 - ema_decay)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        step_losses.append(loss.item())
        num_batches += 1

    n = len(loader.dataset)
    avg_grad_norms = {name: total / num_batches for name, total in grad_accum.items()}

    # Mean gradient vector per layer (for cosine similarity between epochs).
    # Divide accumulated sum by number of batches; leave as None if no grad.
    mean_grad_vec: dict[str, torch.Tensor | None] = {}
    for name in layer_names:
        acc = grad_vec_accum[name]
        mean_grad_vec[name] = (acc / num_batches) if (acc is not None and num_batches > 0) else None

    import statistics
    grad_norm_global = float(sum(g ** 2 for g in all_grad_norms) ** 0.5) if all_grad_norms else 0.0
    grad_norm_before_clip = float(sum(g ** 2 for g in all_raw_norms) ** 0.5) if all_raw_norms else 0.0
    grad_norm_std = float(statistics.stdev(all_grad_norms)) if len(all_grad_norms) > 1 else 0.0

    # Gradient SNR per layer: mean_norm / std_norm.
    # Computed from the accumulated sum and sum-of-squares of per-batch norms
    # (Welford-style; avoids storing all per-batch tensors).
    # SNR > 1 means the gradient magnitude is consistent across mini-batches
    # (the signal is stronger than the noise).  SNR ≈ 1 indicates a very
    # stochastic gradient.
    grad_snr: dict[str, float] = {}
    for name in layer_names:
        mean_n  = grad_accum[name] / num_batches if num_batches > 0 else 0.0
        mean_sq = grad_sq_accum[name] / num_batches if num_batches > 0 else 0.0
        # Clamp to avoid floating-point negatives near zero
        var_n   = max(0.0, mean_sq - mean_n * mean_n)
        std_n   = var_n ** 0.5
        grad_snr[name] = mean_n / (std_n + 1e-10)

    return {
        "loss": total_loss / n,
        "acc": correct / n,
        "grad_norms_per_layer": avg_grad_norms,
        "grad_norm_global": grad_norm_global,
        "grad_norm_std": grad_norm_std,
        "grad_norm_before_clip": grad_norm_before_clip,
        "step_losses": step_losses,
        # Per-layer gradient signal-to-noise ratio (see above).
        "grad_snr_per_layer": grad_snr,
        # Mean gradient vector per Linear layer (for cosine similarity tracking).
        "mean_grad_vec_per_layer": mean_grad_vec,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """Evaluate *model* on *loader* and return (mean_loss, accuracy).

    Runs in torch.no_grad() context; sets model to eval() mode.
    Accuracy is returned as a fraction in [0, 1], not a percentage.
    """
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


@torch.no_grad()
def compute_ece(
    model: nn.Module,
    loader: "DataLoader",
    device: "torch.device",
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE) on the test set.

    Bins the test samples by their maximum predicted softmax probability
    (confidence) and measures how well that confidence predicts actual
    accuracy.  A perfectly calibrated model achieves ECE = 0.

    Formula
    -------
    ECE = Σ_b  (|B_b| / N)  ×  |acc(B_b) − conf(B_b)|

    where the sum runs over the *n_bins* equally-spaced bins, N is the
    total number of samples, and |B_b| is the count in bin b.

    Parameters
    ----------
    model  : nn.Module — evaluated in eval() mode
    loader : DataLoader providing (images, labels) batches
    device : torch.device
    n_bins : number of equal-width confidence bins in [0, 1] (default 10)

    Returns
    -------
    float
        ECE in [0, 1].  0 = perfectly calibrated; 1 = maximally miscalibrated.
    """
    model.eval()
    bin_acc_sum  = [0.0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    bin_count    = [0]   * n_bins

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        probs = torch.softmax(model(images), dim=1)
        conf, preds = probs.max(dim=1)
        correct = (preds == labels).float()

        for i in range(len(labels)):
            # Map confidence in [0, 1] to a bin index; clamp to [0, n_bins-1]
            b = min(int(conf[i].item() * n_bins), n_bins - 1)
            bin_conf_sum[b] += conf[i].item()
            bin_acc_sum[b]  += correct[i].item()
            bin_count[b]    += 1

    n_total = sum(bin_count)
    if n_total == 0:
        return 0.0

    ece = sum(
        abs(bin_acc_sum[b] / bin_count[b] - bin_conf_sum[b] / bin_count[b])
        * bin_count[b] / n_total
        for b in range(n_bins)
        if bin_count[b] > 0
    )
    return ece


@torch.no_grad()
def per_class_accuracy(
    model: nn.Module,
    loader: "DataLoader",
    device: "torch.device",
) -> dict[int, float]:
    """Per-class test accuracy in a single forward pass.

    For each class present in *loader*, computes the fraction of samples
    predicted correctly.  Class IDs are inferred from the labels seen in
    the loader — no explicit ``num_classes`` argument is required.

    Parameters
    ----------
    model  : nn.Module — set to eval() mode internally
    loader : DataLoader providing (images, labels) batches
    device : torch.device

    Returns
    -------
    dict[int, float]
        Mapping from class ID (int) to accuracy in [0, 1].  Only classes
        that appear in the loader are included as keys.
    """
    from collections import defaultdict
    model.eval()
    correct: dict[int, int] = defaultdict(int)
    total:   dict[int, int] = defaultdict(int)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        for pred, label in zip(preds, labels):
            c = label.item()
            total[c]   += 1
            correct[c] += int(pred.item() == c)

    return {c: correct[c] / total[c] for c in sorted(total.keys())}


# ---------------------------------------------------------------------------
# Multi-epoch training (shared by train.py and benchmark.py)
# ---------------------------------------------------------------------------

def run_training(
    model, train_loader, test_loader, optimizer, criterion,
    device, epochs, layer_names, verbose=True,
    compute_hessian: bool = False,
    target_acc: float = 0.95,
    scheduler=None,
    patience: int = 0,
    min_delta: float = 0.0,
    max_grad_norm: float | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_config: dict | None = None,
    amp: bool = False,
    resume_from: str | None = None,
    ema_decay: float | None = None,
    swa_start: int | None = None,
    track_sharpness: bool = False,
) -> dict:
    """Train *model* for *epochs* and return a history dict.

    Parameters
    ----------
    model            : nn.Module to train
    train_loader     : DataLoader for training data
    test_loader      : DataLoader for evaluation
    optimizer        : configured torch.optim.Optimizer
    criterion        : loss function (CrossEntropyLoss, optionally with label smoothing)
    device           : torch.device
    epochs           : total number of training epochs
    layer_names      : display names from linear_layer_names()
    verbose          : print epoch summaries to stdout
    compute_hessian  : compute Hessian trace + sharpness each epoch (slow)
    target_acc       : accuracy threshold to record convergence speed (fraction, e.g. 0.95)
    scheduler        : LR scheduler; step() is called once per epoch after training
    patience         : early stopping patience in epochs (0 = disabled)
    min_delta        : minimum val-loss improvement to reset patience counter
    max_grad_norm    : global gradient clip norm (None = disabled)
    checkpoint_dir   : if set, saves best.pt and final.pt to this directory
    checkpoint_config: extra metadata dict written into checkpoint files
    amp              : enable automatic mixed precision
    resume_from      : path to a .pt checkpoint to resume from
    ema_decay        : if not None, maintain an EMA shadow copy of model weights
                       (float, e.g. 0.999) and evaluate it alongside raw weights
    swa_start        : if not None, start Stochastic Weight Averaging at this epoch

    Returns
    -------
    dict with keys:
        train_loss, test_loss      : list[float]
        train_acc,  test_acc       : list[float]  — fractions in [0, 1], not percentages
        weight_norms               : dict[layer_name, list[float]]  — per-layer L2 norms
        grad_norms                 : dict[layer_name, list[float]]  — per-layer mean grad norms
        time_elapsed               : list[float]  — cumulative wall-clock seconds per epoch
        target_accuracy_epoch      : int | None   — first epoch where test_acc >= target_acc
        target_accuracy_time       : float | None — wall-clock seconds at that epoch
        grad_norm_global           : list[float]  — post-clip global gradient norm per epoch
        grad_norm_std              : list[float]  — std of per-param gradient norms
        grad_norm_before_clip      : list[float]  — pre-clip global norm (same as above when
                                     max_grad_norm is None)
        hessian_trace              : list[float]  — NaN per epoch if compute_hessian=False
        sharpness                  : list[float]  — NaN per epoch if compute_hessian=False
        step_losses                : list[float]  — every batch loss across all epochs
        learning_rates             : list[float]  — LR recorded at the start of each epoch
        early_stopped_epoch        : int | None   — epoch where early stopping fired
        optimizer_states           : dict         — per-layer optimizer state trajectories
        test_acc_ema               : list[float]  — EMA model test accuracy each epoch
                                     (empty list when ema_decay is None)
        swa_final_acc              : float | None — SWA model accuracy after training
                                     (None when swa_start is None)
    """
    from metrics import compute_hessian_trace, compute_sharpness

    # ── AMP setup ──────────────────────────────────────────────────────────
    scaler = None
    if amp:
        if isinstance(optimizer, _AMP_INCOMPATIBLE):
            print(
                f"\n  ⚠  AMP auto-disabled: {type(optimizer).__name__} computes "
                f"Hessian-vector products via loss.backward(create_graph=True). "
                f"GradScaler's loss scaling corrupts the second-order gradient "
                f"graph, producing incorrect curvature estimates. "
                f"Falling back to full precision (float32)."
            )
            amp = False
        elif device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
        elif device.type == "mps":
            pass  # autocast only — GradScaler is CUDA-specific
        else:
            print(
                "\n  ⚠  AMP not enabled: device is CPU. "
                "float16 autocast provides no speed benefit on CPU. "
                "Training in float32."
            )
            amp = False

    history: dict = {
        "train_loss": [], "test_loss": [],
        "train_acc":  [], "test_acc":  [],
        "weight_norms":    {n: [] for n in layer_names},
        "grad_norms":      {n: [] for n in layer_names},
        # ||w_t - w_0||_2 per layer — how far the optimizer has moved each
        # layer's weights from their random initialisation.  High values early
        # in training indicate large, aggressive updates (typical of adaptive
        # methods); a slow, steady rise is characteristic of SGD.
        "weight_distance": {n: [] for n in layer_names},
        # Gradient signal-to-noise ratio per Linear layer — ratio of the
        # mean gradient L2 norm to its standard deviation across mini-batches
        # within the epoch.  High SNR (>> 1) means the gradient direction is
        # consistent across batches (low noise); SNR ≈ 1 indicates a noisy,
        # highly stochastic signal.  Adaptive methods tend to show higher SNR
        # because their adaptive scaling reduces effective gradient variance.
        "grad_snr": {n: [] for n in layer_names},
        # Expected Calibration Error on the test set, computed each epoch.
        # Ranges from 0 (perfect calibration) to 1 (worst possible).  A
        # well-calibrated model's stated confidence matches its accuracy: if
        # it predicts 80% confidence on 100 examples, ~80 should be correct.
        "ece": [],
        "time_elapsed": [],
        "target_accuracy_epoch": None,
        "target_accuracy_time": None,
        "grad_norm_global": [],
        "grad_norm_std": [],
        "grad_norm_before_clip": [],
        "hessian_trace": [],
        "sharpness": [],
        "step_losses": [],
        "learning_rates": [],
        "early_stopped_epoch": None,
        "optimizer_states": {},
        "test_acc_ema": [],
        "swa_final_acc": None,
        # Per-threshold convergence profile (see _CONV_THRESHOLDS).
        # Keys are "50%", "75%", "90%", "95%", "99%"; values are the epoch
        # (1-based int) / wall-clock seconds at which that accuracy was first
        # reached, or None if never reached.
        "convergence_epochs": {f"{int(t * 100)}%": None for t in _CONV_THRESHOLDS},
        "convergence_times":  {f"{int(t * 100)}%": None for t in _CONV_THRESHOLDS},
        # Cumulative gradient steps at the end of each epoch.  Equals
        # sum(len(batch) for batch in all epochs up to and including this one).
        # Useful for plotting accuracy vs compute (steps) rather than epochs,
        # which gives a fairer comparison when batch sizes or epoch counts differ.
        "steps_at_epoch_end": [],
        # Per-class test accuracy each epoch.  Keys are class IDs (ints);
        # values are per-epoch accuracy lists.  Populated lazily via setdefault
        # so no prior knowledge of num_classes is needed.  Useful for spotting
        # optimizers that sacrifice hard or minority classes for overall accuracy.
        "class_acc": {},
        # Standard deviation of per-batch training losses within each epoch.
        # A high value means the loss surface is rough and the per-batch updates
        # are inconsistent (typical of high-LR or noisy-gradient regimes).
        # A low value indicates stable, consistent training.
        "batch_loss_std": [],
        # Per-Linear-layer effective step size: ||w_t - w_{t-1}||_2 per epoch.
        # Measures how large each optimizer step actually is in weight space,
        # independent of the gradient magnitude.  Adaptive methods show a
        # decreasing step size as second moments grow; SGD keeps a roughly
        # constant ratio to the gradient norm.
        "step_size": {n: [] for n in layer_names},
        # Cosine similarity between consecutive epochs' mean gradient directions,
        # per Linear layer.  Epoch 1 has no prior direction → float('nan').
        # Values near 1 = very consistent gradient direction across epochs (stable
        # training); near 0 = the gradient direction is rotating (oscillating);
        # negative = the gradient has reversed direction (diverging or oscillating).
        "grad_cosine_sim": {n: [] for n in layer_names},
    }

    # ── EMA setup ──────────────────────────────────────────────────────────
    # EMA keeps a shadow copy of every float parameter, updated in-place after
    # each optimizer step: ema = ema * decay + param * (1 - decay).
    # Only floating-point tensors are tracked (int buffers like running_mean
    # counters are excluded — they are updated by the model directly).
    ema_state: dict | None = None
    if ema_decay is not None:
        ema_state = {k: v.clone() for k, v in model.state_dict().items()
                     if v.is_floating_point()}

    # ── Initial weight snapshot (for weight_distance tracking) ────────────
    # Capture each Linear layer's weight *before* any training step so that
    # we can compute ||w_t - w_0||_2 each epoch.  Stored on the same device
    # as the model so subtraction later is device-safe.
    _init_linear_weights: dict[str, torch.Tensor] = {}
    _widx = 0
    for _mod in model.modules():
        if isinstance(_mod, nn.Linear):
            _init_linear_weights[layer_names[_widx]] = _mod.weight.detach().clone()
            _widx += 1

    # ── Previous-epoch weight snapshot (for step_size tracking) ───────────
    # Updated at the end of every epoch so that on the *next* epoch we can
    # compute ||w_t - w_{t-1}||_2 (the actual per-epoch parameter update
    # magnitude in weight space).  Initialised to the same values as
    # _init_linear_weights so epoch-1's step_size == weight_distance[1].
    # IMPORTANT: use detach().clone() — without detach() we hold a live
    # autograd reference; without clone() we'd store a view that gets
    # silently overwritten when the model updates its weights.
    _prev_linear_weights: dict[str, torch.Tensor] = {
        n: _init_linear_weights[n].clone() for n in layer_names
    }

    # ── Previous-epoch mean gradient direction (for cosine similarity) ────
    # Stores the mean gradient vector from the previous epoch per layer.
    # Initialised to None; epoch 1 records nan since there is no prior epoch.
    _prev_mean_grad: dict[str, torch.Tensor | None] = {n: None for n in layer_names}

    # ── SWA setup ──────────────────────────────────────────────────────────
    # AveragedModel maintains a running uniform average of model weights.
    # update_parameters() is called after each training epoch >= swa_start.
    # After the loop, update_bn() re-estimates BatchNorm statistics (needed
    # because SWA weights have different statistics than any single checkpoint).
    # For MLP (no BatchNorm), update_bn() is skipped entirely.
    swa_model = None
    if swa_start is not None:
        swa_model = torch.optim.swa_utils.AveragedModel(model)

    es = EarlyStopping(patience, min_delta)
    best_test_acc: float = -1.0

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 1
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if verbose:
            print(f"\n  Resumed from {resume_from} (epoch {ckpt['epoch']}). "
                  f"Continuing from epoch {start_epoch}.")

    # Fixed batch for Hessian/sharpness (reuse same batch each epoch).
    # Grabbed whenever either the Hessian trace or sharpness tracking is active.
    hessian_batch = None
    if compute_hessian or track_sharpness:
        for batch in train_loader:
            hessian_batch = batch
            break

    # Plain CE loss (no label smoothing) used exclusively for Hessian / sharpness
    # estimation — the diagnostics should measure the true loss curvature, not
    # the smoothed training objective.
    criterion_hessian = nn.CrossEntropyLoss()
    start_time = time.time()
    global_step_offset = 0

    for epoch in range(start_epoch, epochs + 1):
        # Record LR before this epoch's training
        history["learning_rates"].append(optimizer.param_groups[0]["lr"])

        epoch_result = train_one_epoch(
            model, train_loader, optimizer, criterion, device, layer_names,
            global_step_offset=global_step_offset,
            max_grad_norm=max_grad_norm,
            amp=amp,
            scaler=scaler,
            ema_state=ema_state,
            ema_decay=ema_decay if ema_decay is not None else 0.999,
        )
        global_step_offset += len(epoch_result["step_losses"])

        if swa_model is not None and epoch >= swa_start:
            swa_model.update_parameters(model)

        train_loss = epoch_result["loss"]
        train_acc  = epoch_result["acc"]
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # EMA evaluation: temporarily load EMA weights, evaluate, then restore.
        # IMPORTANT — state_dict() returns detached views sharing parameter storage,
        # so .clone() is required; without it 'orig_state' would be silently
        # overwritten when load_state_dict() writes new values into that storage.
        if ema_state is not None:
            orig_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict({**orig_state, **ema_state})
            _, ema_acc = evaluate(model, test_loader, criterion, device)
            model.load_state_dict(orig_state)
            history["test_acc_ema"].append(ema_acc)

        w_norms  = weight_norms(model, layer_names)
        w_dists  = weight_distances(model, layer_names, _init_linear_weights)
        elapsed  = time.time() - start_time

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["time_elapsed"].append(elapsed)
        history["grad_norm_global"].append(epoch_result["grad_norm_global"])
        history["grad_norm_std"].append(epoch_result["grad_norm_std"])
        history["grad_norm_before_clip"].append(epoch_result["grad_norm_before_clip"])
        history["step_losses"].extend(epoch_result["step_losses"])
        history["steps_at_epoch_end"].append(len(history["step_losses"]))

        # Effective step size: ||w_t - w_{t-1}||_2 per Linear layer.
        # Measures the actual weight-space displacement from the previous epoch
        # to this one.  Computed before updating _prev_linear_weights so that
        # the comparison is always "current vs immediately prior epoch".
        _sidx = 0
        for _mod in model.modules():
            if isinstance(_mod, nn.Linear):
                _name = layer_names[_sidx]
                _step = (_mod.weight.detach() - _prev_linear_weights[_name]).norm().item()
                history["step_size"][_name].append(_step)
                # Update snapshot for next epoch — clone() prevents a dangling
                # view into parameter storage (same safety rule as _init_linear_weights)
                _prev_linear_weights[_name] = _mod.weight.detach().clone()
                _sidx += 1

        for n in layer_names:
            history["weight_norms"][n].append(w_norms.get(n, float("nan")))
            history["grad_norms"][n].append(epoch_result["grad_norms_per_layer"].get(n, float("nan")))
            history["weight_distance"][n].append(w_dists.get(n, float("nan")))
            history["grad_snr"][n].append(epoch_result["grad_snr_per_layer"].get(n, float("nan")))

        # Training instability: std of per-batch losses within this epoch.
        # A large value means individual batches produced very different losses —
        # a signal of rough loss landscape or overly large learning rate.
        import statistics as _stats
        _epoch_step_losses = epoch_result["step_losses"]
        _batch_loss_std = (
            _stats.stdev(_epoch_step_losses) if len(_epoch_step_losses) > 1 else 0.0
        )
        history["batch_loss_std"].append(_batch_loss_std)

        # Calibration — one-pass ECE evaluation on the test set.
        # Computed on the *raw* model weights (not EMA / SWA) so the value
        # tracks how calibration evolves with the primary optimisation trajectory.
        ece_val = compute_ece(model, test_loader, device)
        history["ece"].append(ece_val)

        # Per-class accuracy — single forward pass over the test set.
        # Keys are class IDs (int); populated lazily so no num_classes needed.
        cls_acc = per_class_accuracy(model, test_loader, device)
        for cls_id, cls_val in cls_acc.items():
            history["class_acc"].setdefault(cls_id, []).append(cls_val)

        # Optimizer internal state snapshot
        _states = _extract_optimizer_states(model, optimizer, layer_names)
        for key, val in _states.items():
            if isinstance(val, dict):
                state_entry = history["optimizer_states"].setdefault(key, {})
                for layer, scalar in val.items():
                    state_entry.setdefault(layer, []).append(scalar)
            else:
                history["optimizer_states"].setdefault(key, []).append(val)

        # Best checkpoint
        if checkpoint_dir and test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(
                os.path.join(checkpoint_dir, "best.pt"),
                epoch, model, optimizer,
                metrics={"train_loss": train_loss, "train_acc": train_acc,
                         "test_loss": test_loss,   "test_acc": test_acc},
                config=checkpoint_config,
            )

        if history["target_accuracy_epoch"] is None and test_acc >= target_acc:
            history["target_accuracy_epoch"] = epoch
            history["target_accuracy_time"] = elapsed

        # Convergence profile — record the first epoch each milestone is crossed.
        for _thresh in _CONV_THRESHOLDS:
            _tkey = f"{int(_thresh * 100)}%"
            if history["convergence_epochs"][_tkey] is None and test_acc >= _thresh:
                history["convergence_epochs"][_tkey] = epoch
                history["convergence_times"][_tkey]  = elapsed

        # Hessian trace (expensive: requires create_graph backward)
        if compute_hessian and hessian_batch is not None:
            ht = compute_hessian_trace(model, hessian_batch, criterion_hessian, device)
        else:
            ht = float("nan")
        history["hessian_trace"].append(ht)

        # Sharpness: max loss increase under a small random weight perturbation.
        # Decoupled from the Hessian trace — much cheaper (no backward pass).
        # Enabled either by the full --hessian flag or the lighter --sharpness flag.
        if (compute_hessian or track_sharpness) and hessian_batch is not None:
            sh = compute_sharpness(model, hessian_batch, criterion_hessian, device)
        else:
            sh = float("nan")
        history["sharpness"].append(sh)

        # Gradient cosine similarity: cosine(mean_grad_t, mean_grad_{t-1}) per layer.
        # First epoch has no prior direction → float('nan').  Values near 1 indicate
        # the gradient direction is highly consistent across epochs; values near 0
        # suggest the gradient is rotating (oscillatory or chaotic training).
        curr_grad_vecs = epoch_result["mean_grad_vec_per_layer"]
        for n in layer_names:
            curr = curr_grad_vecs.get(n)
            prev = _prev_mean_grad.get(n)
            if curr is not None and prev is not None:
                cos_sim = torch.nn.functional.cosine_similarity(
                    curr.flatten().unsqueeze(0),
                    prev.flatten().unsqueeze(0),
                ).item()
            else:
                cos_sim = float("nan")
            history["grad_cosine_sim"][n].append(cos_sim)
            _prev_mean_grad[n] = curr.detach().clone() if curr is not None else None

        if scheduler is not None:
            scheduler.step()

        if verbose:
            print(
                f"  Epoch {epoch:>3}/{epochs} | "
                f"train loss {train_loss:.4f}  acc {train_acc*100:.2f}% | "
                f"test  loss {test_loss:.4f}  acc {test_acc*100:.2f}%"
            )

        if es.step(test_loss, model):
            history["early_stopped_epoch"] = epoch
            es.restore(model, device)
            if verbose:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val loss {es.best_loss:.4f})")
            break

    # Final checkpoint
    if checkpoint_dir and history["train_loss"]:
        last = len(history["train_loss"]) - 1
        save_checkpoint(
            os.path.join(checkpoint_dir, "final.pt"),
            last + 1, model, optimizer,
            metrics={"train_loss": history["train_loss"][last],
                     "train_acc":  history["train_acc"][last],
                     "test_loss":  history["test_loss"][last],
                     "test_acc":   history["test_acc"][last]},
            config=checkpoint_config,
        )

    # SWA final evaluation: fix BatchNorm stats (if any), then evaluate once.
    if swa_model is not None:
        _has_bn = any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in model.modules()
        )
        if _has_bn:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        _, swa_acc = evaluate(swa_model, test_loader, criterion, device)
        history["swa_final_acc"] = swa_acc
        if verbose:
            print(f"  SWA final accuracy : {swa_acc * 100:.2f}%")

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
                   help="LR scheduler: none | cosine | step | warmup_cosine | cosine_wr")
    p.add_argument("--warmup-epochs", default=5, type=int,
                   help="Linear warmup epochs for warmup_cosine (default: 5)")
    p.add_argument("--patience",  default=0,   type=int,
                   help="Early stopping patience epochs (0 = disabled)")
    p.add_argument("--min-delta", default=0.0, type=float,
                   help="Min val-loss improvement to reset patience counter")
    p.add_argument("--max-grad-norm", default=None, type=float,
                   help="Clip global gradient norm to this value (None = disabled)")
    p.add_argument("--weight-decay", default=0.0, type=float,
                   help="L2 weight decay coefficient (default: 0.0, disabled)")
    p.add_argument("--seed", default=None, type=int,
                   help="Random seed for reproducibility (default: None, non-deterministic)")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Directory to save best.pt and final.pt checkpoints (default: disabled)")
    p.add_argument("--find-lr",       action="store_true",
                   help="Run LR range test before training and print suggested LR")
    p.add_argument("--find-lr-iters", default=100, type=int,
                   help="Mini-batch steps for the LR range test (default: 100)")
    p.add_argument("--find-lr-plot",  default=None,
                   help="Save path for LR range test plot (default: no plot)")
    p.add_argument("--amp", action="store_true",
                   help="Enable automatic mixed precision: float16 autocast + GradScaler "
                        "on CUDA (~30%% faster, half memory); float16 autocast only on MPS; "
                        "no-op on CPU. Auto-disabled for Sophia and AdaHessian (their "
                        "Hessian estimators call backward(create_graph=True), which is "
                        "incompatible with GradScaler loss scaling).")
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile() to the model before training for additional "
                        "speed (~20-40%% on CUDA/MPS via kernel fusion). Auto-disabled for "
                        "Sophia and AdaHessian (create_graph=True is incompatible with "
                        "static graph tracing). Falls back to eager if compilation fails.")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint (.pt) to resume training from. "
                        "Restores model weights, optimizer state, and continues from the "
                        "next epoch.")
    p.add_argument("--ema-decay", default=None, type=float,
                   help="Exponential moving average decay for model weights "
                        "(e.g. 0.999). EMA weights are evaluated each epoch alongside "
                        "raw weights. None = disabled (default).")
    p.add_argument("--label-smoothing", default=0.0, type=float,
                   help="Label smoothing epsilon for CrossEntropyLoss (default: 0.0 = disabled). "
                        "Typical value: 0.1.")
    p.add_argument("--swa-start", default=None, type=int, metavar="EPOCH",
                   help="Epoch to begin Stochastic Weight Averaging (default: disabled). "
                        "After this epoch weights are averaged across all subsequent epochs; "
                        "a single evaluation is run after training completes.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        set_seed(args.seed)

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
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scheduler = SCHEDULER_REGISTRY[args.scheduler](optimizer, args.epochs, args.warmup_epochs)
    print(f"Optimizer    : {optimizer.__class__.__name__}  (lr={args.lr})")
    print(f"Scheduler    : {args.scheduler}")
    if args.patience > 0:
        print(f"Early stopping: patience={args.patience}, min_delta={args.min_delta}")
    if args.max_grad_norm is not None:
        print(f"Grad clipping : max_norm={args.max_grad_norm}")
    if args.weight_decay > 0.0:
        print(f"Weight decay  : {args.weight_decay:g}")
    if args.seed is not None:
        print(f"Seed          : {args.seed}")
    if args.ema_decay is not None:
        print(f"EMA decay     : {args.ema_decay}")
    if args.label_smoothing > 0.0:
        print(f"Label smoothing: {args.label_smoothing}")
    if args.swa_start is not None:
        print(f"SWA start      : epoch {args.swa_start}")
    if args.checkpoint_dir:
        print(f"Checkpoints   : {args.checkpoint_dir}/")

    # torch.compile
    if args.compile:
        if isinstance(optimizer, _COMPILE_INCOMPATIBLE):
            print(
                f"  compile    : disabled ⚠  {optimizer.__class__.__name__} uses "
                f"backward(create_graph=True) to estimate the Hessian — "
                f"torch.compile's static graph tracing cannot capture dynamic "
                f"higher-order graphs correctly. Running in eager mode."
            )
        else:
            try:
                model = torch.compile(model)
                print(f"  compile    : enabled  (torch.compile)")
            except Exception as e:
                print(f"  compile    : failed ({e}) — running in eager mode")

    # AMP setup
    amp = args.amp
    scaler = None
    if amp:
        if isinstance(optimizer, _AMP_INCOMPATIBLE):
            print(
                f"  AMP        : disabled ⚠  {optimizer.__class__.__name__} uses "
                f"backward(create_graph=True) to estimate the Hessian — "
                f"GradScaler loss scaling corrupts the second-order gradient graph. "
                f"Training in float32."
            )
            amp = False
        elif device.type == "cuda":
            scaler = torch.cuda.amp.GradScaler()
            print(f"  AMP        : enabled  (CUDA — float16 autocast + GradScaler)")
        elif device.type == "mps":
            print(f"  AMP        : enabled  (MPS — float16 autocast, no GradScaler)")
        else:
            print(f"  AMP        : disabled (CPU — float16 provides no speed benefit)")
            amp = False
    print()

    if args.find_lr:
        print("Running LR range test...")
        finder = LRFinder(model, optimizer, criterion, device,
                          num_iter=args.find_lr_iters)
        finder.run(train_loader)
        suggested_lr = finder.suggestion()
        print(f"  Suggested LR : {suggested_lr:.2e}")
        if args.find_lr_plot:
            finder.plot(save_path=args.find_lr_plot)
            print(f"  LR plot saved: {args.find_lr_plot}")
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
        "early_stopped_epoch": None,
        "test_acc_ema": [],
        "swa_final_acc": None,
    }

    # Initialise EMA shadow weights (float tensors only)
    ema_state: dict | None = None
    if args.ema_decay is not None:
        ema_state = {k: v.clone() for k, v in model.state_dict().items()
                     if v.is_floating_point()}

    swa_model_main = None
    if args.swa_start is not None:
        swa_model_main = torch.optim.swa_utils.AveragedModel(model)

    es = EarlyStopping(args.patience, args.min_delta)
    best_test_acc: float = -1.0
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    run_config = {
        "dataset": args.dataset, "model": args.model, "optimizer": args.optimizer,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "scheduler": args.scheduler, "weight_decay": args.weight_decay, "seed": args.seed,
    }
    run_start = time.time()

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"\n  Resumed from {args.resume} (epoch {ckpt['epoch']}). "
              f"Continuing from epoch {start_epoch}.")
        if start_epoch > args.epochs:
            print(f"  Nothing to do: checkpoint epoch {ckpt['epoch']} >= "
                  f"--epochs {args.epochs}. Exiting.")
            return

    for epoch in range(start_epoch, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        history["learning_rates"].append(current_lr)

        epoch_result = train_one_epoch(
            model, train_loader, optimizer, criterion, device, layer_names,
            max_grad_norm=args.max_grad_norm,
            amp=amp,
            scaler=scaler,
            ema_state=ema_state,
            ema_decay=args.ema_decay if args.ema_decay is not None else 0.999,
        )
        if swa_model_main is not None and epoch >= args.swa_start:
            swa_model_main.update_parameters(model)

        train_loss = epoch_result["loss"]
        train_acc  = epoch_result["acc"]
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # EMA evaluation: temporarily load EMA weights, evaluate, then restore.
        # IMPORTANT — state_dict() returns detached views sharing parameter storage,
        # so .clone() is required; without it 'orig_state' would be silently
        # overwritten when load_state_dict() writes new values into that storage.
        if ema_state is not None:
            orig_state = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict({**orig_state, **ema_state})
            _, ema_acc = evaluate(model, test_loader, criterion, device)
            model.load_state_dict(orig_state)
            history["test_acc_ema"].append(ema_acc)
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

        if args.checkpoint_dir and test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best.pt"),
                epoch, model, optimizer,
                metrics={"train_loss": train_loss, "train_acc": train_acc,
                         "test_loss": test_loss,   "test_acc": test_acc},
                config=run_config,
            )

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

        if es.step(test_loss, model):
            history["early_stopped_epoch"] = epoch
            es.restore(model, device)
            print(f"Early stopping at epoch {epoch} "
                  f"(best val loss {es.best_loss:.4f})")
            break

    if vis is not None:
        vis.close()

    # Final checkpoint
    if args.checkpoint_dir and history["train_loss"]:
        last = len(history["train_loss"]) - 1
        save_checkpoint(
            os.path.join(args.checkpoint_dir, "final.pt"),
            last + 1, model, optimizer,
            metrics={"train_loss": history["train_loss"][last],
                     "train_acc":  history["train_acc"][last],
                     "test_loss":  history["test_loss"][last],
                     "test_acc":   history["test_acc"][last]},
            config=run_config,
        )

    # SWA final evaluation
    if swa_model_main is not None:
        _has_bn = any(
            isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
            for m in model.modules()
        )
        if _has_bn:
            torch.optim.swa_utils.update_bn(train_loader, swa_model_main, device=device)
        _, swa_acc = evaluate(swa_model_main, test_loader, criterion, device)
        history["swa_final_acc"] = swa_acc
        print(f"SWA final accuracy : {swa_acc * 100:.2f}%")

    # Write logs
    config = {
        "dataset":       args.dataset,
        "model":         args.model,
        "optimizer":     args.optimizer,
        "lr":            args.lr,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "hidden_sizes":  args.hidden_sizes,
        "scheduler":     args.scheduler,
        "patience":      args.patience,
        "min_delta":     args.min_delta,
        "max_grad_norm":    args.max_grad_norm,
        "weight_decay":     args.weight_decay,
        "seed":             args.seed,
        "label_smoothing":  args.label_smoothing,
    }
    logger = TrainingLogger(log_dir=args.log_dir)
    logger.log_run(config, history)
    logger.close()


if __name__ == "__main__":
    main()
