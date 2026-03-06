# OptimizerProject

[![CI](https://github.com/honghaoyu12/OptimizerProject/actions/workflows/ci.yml/badge.svg)](https://github.com/honghaoyu12/OptimizerProject/actions/workflows/ci.yml)

A testing ground for custom numerical optimizers. Train a neural network on MNIST, FashionMNIST, CIFAR-10, or Tiny ImageNet, plug in your own optimizer, and watch detailed diagnostics update live.

---

## Project Structure

```
OptimizerProject/
├── model.py              # MLP, ResNet-18, and ViT model definitions
├── metrics.py            # Hessian trace and sharpness estimators
├── train.py              # Training loop, data loading, CLI entry point
├── visualizer.py         # Live dashboard and benchmark comparison plots
├── benchmark.py          # Interactive multi-(dataset × model × optimizer) benchmark
├── optimizers/
│   ├── __init__.py       # Exports all custom optimizers
│   ├── base.py           # Abstract base class for custom optimizers
│   ├── sgd.py            # VanillaSGD — minimal reference implementation
│   ├── lion.py           # Lion — EvoLved Sign Momentum
│   ├── lamb.py           # LAMB — Layer-wise Adaptive Moments
│   └── shampoo.py        # Shampoo — Kronecker-factored second-order
├── tests/
│   ├── __init__.py
│   ├── test_models.py    # 17 tests for MLP, ResNet18, and ViT
│   ├── test_optimizers.py# 18 tests for all optimizers
│   ├── test_metrics.py   # 9 tests for Hessian trace and sharpness
│   └── test_train.py     # 28 tests for DATASET_INFO, build_model, and training loop
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI — runs all tests on push/PR
├── requirements.txt      # Python dependencies
├── data/                 # Datasets downloaded here automatically
├── CHANGELOG.md          # History of all changes
└── README.md
```

---

## Running from the Terminal

### 1. Open a terminal and navigate to the project

```bash
cd /path/to/OptimizerProject
```

### 2. Install dependencies (one-time)

```bash
pip install torch torchvision matplotlib
```

### 3. Run a single training job

```bash
python train.py
```

This uses all defaults: MLP, MNIST, Adam, 10 epochs. A live 3×2 dashboard opens and updates each epoch. The final figure is saved to `training_curves.png`.

**Common variations:**

```bash
# Change dataset
python train.py --dataset fashion_mnist
python train.py --dataset cifar10
python train.py --dataset tiny_imagenet   # downloads ~236 MB on first run

# Change model
python train.py --model resnet18
python train.py --model vit

# Change optimizer
python train.py --optimizer adamw
python train.py --optimizer nadam
python train.py --optimizer lion
python train.py --optimizer shampoo

# Change MLP architecture (depth = number of values, width = each value)
python train.py --hidden-sizes 512 256 128   # 3 hidden layers
python train.py --hidden-sizes 64            # 1 hidden layer

# Change training length
python train.py --epochs 20 --batch-size 256

# Disable the live plot window and save the figure to a file instead
python train.py --no-plot --save-plot my_run.png

# Combine flags freely
python train.py --model vit --dataset cifar10 --optimizer adamw --epochs 15 --no-plot --save-plot vit_cifar10.png

# Enable Hessian trace + sharpness diagnostics (adds the bottom-right panel, slower)
python train.py --hessian
```

**All flags for `train.py`:**

```
--dataset        mnist | fashion_mnist | cifar10 | tiny_imagenet   (default: mnist)
--model          mlp | resnet18 | vit                              (default: mlp)
--optimizer      adam | adamw | nadam | radam | adagrad |
                 sgd | rmsprop | vanilla_sgd |
                 lion | lamb | shampoo                             (default: adam)
--lr             learning rate                                     (default: 1e-3)
--epochs         number of epochs                                  (default: 10)
--batch-size     mini-batch size                                   (default: 128)
--hidden-sizes   MLP hidden layer widths, space-separated          (default: 256 128)
--data-dir       where to store downloaded datasets                (default: ./data)
--device         cpu | cuda | mps | auto                          (default: auto)
--no-plot        disable the live plot window
--save-plot      path to save the final figure                     (default: training_curves.png)
--hessian        enable Hessian trace + sharpness computation
```

### 4. Run the interactive benchmark

```bash
python benchmark.py
```

Three sequential menus appear. Type numbers separated by commas, or `all`:

```
────────────────────────────────────────────────────────
  Select Datasets
  Enter numbers separated by commas, or 'all'
────────────────────────────────────────────────────────
  [1]  MNIST
  [2]  Fashion MNIST
  [3]  CIFAR-10
  [4]  Tiny ImageNet

  > 1,2         ← type this, press Enter

────────────────────────────────────────────────────────
  Select Models
  Enter numbers separated by commas, or 'all'
────────────────────────────────────────────────────────
  [1]  MLP
  [2]  ResNet-18
  [3]  ViT

  > 1           ← type this, press Enter

────────────────────────────────────────────────────────
  Select Optimizers
  Enter numbers separated by commas, or 'all'
────────────────────────────────────────────────────────
  [1]  SGD
  [2]  SGD+Momentum
  [3]  Adam
  [4]  Adagrad
  [5]  AdamW
  [6]  NAdam
  [7]  RAdam
  [8]  Lion
  [9]  LAMB
  [10] Shampoo

  > all         ← type this, press Enter
```

Every `(dataset, model, optimizer)` combination is then trained in sequence and a comparison figure is saved to `benchmark.png`.

You can also pipe the selections in to skip the prompts entirely:

```bash
# MNIST only, MLP only, Adam and AdamW — pipe answers straight in
printf "1\n1\n3,5\n" | python benchmark.py --epochs 5 --save-plot my_benchmark.png
```

**All flags for `benchmark.py`:**

```
--epochs         epochs per run                        (default: 10)
--batch-size     mini-batch size                       (default: 128)
--hidden-sizes   MLP hidden layer widths               (default: 256 128)
--lr             override all optimizer LRs            (default: per-optimizer defaults)
--data-dir       dataset location                      (default: ./data)
--device         cpu | cuda | mps | auto              (default: auto)
--save-plot      output figure path                    (default: benchmark.png)
```

### 5. Device selection

The `--device auto` default picks the best available hardware automatically:
**CUDA GPU → Apple Silicon (MPS) → CPU**

To force a specific device:

```bash
python train.py --device cpu
python train.py --device cuda    # NVIDIA GPU
python train.py --device mps     # Apple Silicon
```

### 6. If the live plot window freezes

Some systems don't support interactive matplotlib windows well. Use `--no-plot` to skip it and save to a file instead:

```bash
python train.py --no-plot --save-plot output.png
```

---

## Example Commands

```bash
# Quick sanity check (2 epochs, no window, discard figure)
python train.py --epochs 2 --no-plot --save-plot ""

# MLP on MNIST with Adam — default run
python train.py

# ResNet-18 on CIFAR-10 with SGD and Hessian diagnostics
python train.py --model resnet18 --dataset cifar10 --optimizer sgd --lr 0.01 --hessian --no-plot --save-plot resnet_cifar.png

# ViT on CIFAR-10 with AdamW, 15 epochs
python train.py --model vit --dataset cifar10 --optimizer adamw --epochs 15 --no-plot --save-plot vit_cifar.png

# Wide MLP on FashionMNIST
python train.py --dataset fashion_mnist --hidden-sizes 512 256 128 --no-plot --save-plot wide_mlp.png

# Benchmark: MNIST + FashionMNIST, all 3 models, all 10 optimizers, 5 epochs
printf "1,2\nall\nall\n" | python benchmark.py --epochs 5 --save-plot full_benchmark.png

# Benchmark: CIFAR-10 only, MLP + ViT, Adam family only (Adam, AdamW, NAdam, RAdam)
printf "3\n1,3\n3,5,6,7\n" | python benchmark.py --epochs 10 --save-plot adam_family.png

# Benchmark with a global LR override
python benchmark.py --lr 0.001 --epochs 10
```

---

## Testing

The project ships with 72 pytest tests covering every major component.

### Run the tests locally

```bash
# Install pytest (one-time)
pip install pytest

# Run all tests with verbose output
pytest tests/ -v

# Run a single test file
pytest tests/test_models.py -v

# Run a single test class or function
pytest tests/test_optimizers.py::TestShampoo -v
pytest tests/test_train.py::TestTrainingLoop::test_loss_decreases_over_epochs -v
```

Expected output: `72 passed` in a few seconds (all on CPU, no downloads needed).

### Test files

| File | Tests | What it covers |
|---|---|---|
| `tests/test_models.py` | 17 | MLP, ResNet18, ViT forward passes; output shapes; patch size assertion |
| `tests/test_optimizers.py` | 18 | Registry completeness; finite weights after one step for all 11 optimizers; optimizer-specific state and behaviour |
| `tests/test_metrics.py` | 9 | Hessian trace (type, finiteness, positivity, NaN on no-param model); sharpness (type, non-negativity, weight restoration, epsilon monotonicity) |
| `tests/test_train.py` | 28 | `DATASET_INFO` metadata; `build_model` for all model×dataset combos; `linear_layer_names`/`weight_norms`; `train_one_epoch` return dict; loss decrease over 5 epochs |

### CI pipeline

Every push and pull request to `main` triggers the GitHub Actions workflow defined in `.github/workflows/ci.yml`:

- **Runner**: `ubuntu-latest`, Python 3.12
- **PyTorch**: CPU-only install (`--index-url https://download.pytorch.org/whl/cpu`) for fast execution
- **Cache**: pip dependencies cached on `requirements.txt` hash
- **Command**: `pytest tests/ -v`

The badge below reflects the current status of the `main` branch:

[![CI](https://github.com/honghaoyu12/OptimizerProject/actions/workflows/ci.yml/badge.svg)](https://github.com/honghaoyu12/OptimizerProject/actions/workflows/ci.yml)

---

## Files in Detail

### `model.py`

Defines three model classes.

**`MLP(input_size=784, hidden_sizes=[256,128], num_classes=10)`**

A fully-connected network. `hidden_sizes` controls both depth (number of values) and width (each value).

```bash
# From the terminal, set hidden sizes with --hidden-sizes
python train.py --hidden-sizes 256 128        # default: 784→256→128→10
python train.py --hidden-sizes 512            # one wide layer
python train.py --hidden-sizes 512 256 128 64 # four layers
```

**`ResNet18(in_channels=3, num_classes=10)`**

Wraps `torchvision.models.resnet18` with two adaptations for small images:
- `conv1` changed from 7×7/stride-2 → 3×3/stride-1 (preserves spatial resolution)
- `maxpool` replaced with `nn.Identity()` (avoids over-downsampling on 28–32 px inputs)

```bash
python train.py --model resnet18
python train.py --model resnet18 --dataset cifar10
```

**`ViT(image_size=32, in_channels=3, patch_size=4, num_classes=10, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1)`**

Vision Transformer from "An Image is Worth 16×16 Words" (Dosovitskiy et al., 2020), adapted for small images. The default `patch_size=4` divides both 28 (MNIST) and 32 (CIFAR-10) evenly:

| Dataset | Image | Patches |
|---|---|---|
| MNIST / FashionMNIST | 28×28 | 7×7 = 49 patches |
| CIFAR-10 | 32×32 | 8×8 = 64 patches |
| Tiny ImageNet | 64×64 | 16×16 = 256 patches |

Architecture overview:
1. **Patch embedding** — each patch is flattened and projected to `dim` tokens via a linear layer with pre/post LayerNorm
2. **CLS token** — a learnable `[class]` token is prepended to the sequence
3. **Positional embeddings** — learnable, added to all tokens
4. **Transformer encoder** — `depth` layers of multi-head self-attention + MLP (Pre-LN for training stability)
5. **Classification head** — LayerNorm → Linear applied to the final CLS token

```bash
python train.py --model vit
python train.py --model vit --dataset cifar10
```

---

### `metrics.py`

Two optimizer diagnostic functions. Both accept a pre-loaded `batch` tuple `(images, labels)` and return `float("nan")` on any failure.

**`compute_hessian_trace(model, batch, criterion, device, n_samples=3) → float`**

Estimates the trace of the loss Hessian using the Hutchinson estimator:

```
Tr(H) ≈ (1/S) Σ vᵀHv,   v ~ Rademacher(±1)
```

Hessian-vector products are computed via `autograd.grad(create_graph=True)`. High values indicate a sharp, ill-conditioned loss landscape.

**`compute_sharpness(model, batch, criterion, device, epsilon=1e-2, n_directions=5) → float`**

SAM-style sharpness: the maximum loss increase over random unit perturbations scaled to radius `epsilon`:

```
sharpness = max_{‖δ‖ ≤ ε} L(θ + δ) − L(θ)
```

Enable both via the `--hessian` flag:

```bash
python train.py --hessian --no-plot --save-plot hessian_run.png
```

---

### `train.py`

The core training module and the CLI entry point for single runs.

#### Key exports (used by `benchmark.py`)

| Symbol | Description |
|---|---|
| `DATASET_INFO` | Dict of per-dataset metadata (`in_channels`, `input_size`, `num_classes`) |
| `OPTIMIZER_REGISTRY` | Dict mapping name → optimizer factory lambda |
| `get_dataloaders()` | Returns `(train_loader, test_loader)` for a named dataset |
| `build_model()` | Instantiates MLP, ResNet18, or ViT from a name + dataset metadata |
| `linear_layer_names()` | Returns display labels for each linear layer |
| `weight_norms()` | Returns `{layer_name: L2_norm}` dict |
| `train_one_epoch()` | Runs one epoch; returns a metrics dict |
| `run_training()` | Runs all epochs; returns a full history dict |

#### Supported datasets

| Key | Source | Images | Classes |
|---|---|---|---|
| `"mnist"` | `torchvision.datasets.MNIST` | 28×28 gray | 10 |
| `"fashion_mnist"` | `FashionMNIST` | 28×28 gray | 10 |
| `"cifar10"` | `CIFAR10` | 32×32 RGB | 10 |
| `"tiny_imagenet"` | `ImageFolder` (auto-downloaded) | 64×64 RGB | 200 |

Training augmentations:
- **CIFAR-10**: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
- **Tiny ImageNet**: RandomCrop(64, padding=8) + RandomHorizontalFlip + Normalize
- **MNIST / FashionMNIST**: ToTensor + Normalize only

#### Tiny ImageNet setup

The first time `--dataset tiny_imagenet` is used, the dataset is downloaded and organised automatically (~236 MB). A `.setup_done` marker means this only ever runs once.

#### Available optimizers

| Key (`--optimizer`) | Display name | Source | Default LR |
|---|---|---|---|
| `adam` | Adam | `torch.optim.Adam` | 1e-3 |
| `adamw` | AdamW | `torch.optim.AdamW` | 1e-3 |
| `nadam` | NAdam | `torch.optim.NAdam` | 2e-3 |
| `radam` | RAdam | `torch.optim.RAdam` | 1e-3 |
| `adagrad` | Adagrad | `torch.optim.Adagrad` | 1e-2 |
| `sgd` | SGD+Momentum | `torch.optim.SGD(momentum=0.9)` | 1e-2 |
| `rmsprop` | — | `torch.optim.RMSprop` | 1e-3 |
| `vanilla_sgd` | — | `optimizers/sgd.py` | 1e-2 |
| `lion` | Lion | `optimizers/lion.py` | 1e-4 |
| `lamb` | LAMB | `optimizers/lamb.py` | 1e-3 |
| `shampoo` | Shampoo | `optimizers/shampoo.py` | 1e-2 |

**Lion** — EvoLved Sign Momentum (Chen et al., 2023). Tracks one momentum buffer; every update is the sign of a β₁-interpolated gradient. Use ~10× smaller LR than Adam. Memory-efficient: no second moment.

**LAMB** — Layer-wise Adaptive Moments for Batch training (You et al., 2019). Adam moments + a per-tensor trust ratio `‖θ‖ / ‖u‖`. Designed for large-batch training.

**Shampoo** — Kronecker-factored second-order optimizer (Gupta et al., 2018). Applies L^{-1/4} G R^{-1/4} as the preconditioned update. Eigendecomposition runs on CPU for MPS compatibility.

#### `train_one_epoch()` return dict

```python
{
    "loss":                 float,         # mean cross-entropy over the epoch
    "acc":                  float,         # fraction correct (not percentage)
    "grad_norms_per_layer": dict,          # mean grad L2 norm per linear layer
    "grad_norm_global":     float,         # L2 norm across all parameter gradients
    "grad_norm_std":        float,         # std of individual parameter grad norms
    "step_losses":          list[float],   # per-batch loss values
}
```

#### `run_training()` history dict

```python
{
    "train_loss":            list[float],
    "test_loss":             list[float],
    "train_acc":             list[float],
    "test_acc":              list[float],
    "weight_norms":          dict[str, list[float]],   # per layer, per epoch
    "grad_norms":            dict[str, list[float]],   # per layer, per epoch
    "time_elapsed":          list[float],              # seconds since start
    "target_accuracy_epoch": int | None,               # first epoch >= target_acc
    "target_accuracy_time":  float | None,             # wall-clock time at that epoch
    "grad_norm_global":      list[float],
    "grad_norm_std":         list[float],
    "hessian_trace":         list[float],              # NaN if --hessian not passed
    "sharpness":             list[float],              # NaN if --hessian not passed
    "step_losses":           list[float],              # all batches, all epochs
}
```

#### Adding a custom optimizer

**Step 1** — Create `optimizers/my_optimizer.py`:

```python
import torch
from .base import BaseOptimizer

class MyOptimizer(BaseOptimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # ---- your update rule here ----
                p.add_(p.grad, alpha=-lr)
        return loss
```

**Step 2** — Register it in `train.py`:

```python
from optimizers.my_optimizer import MyOptimizer

OPTIMIZER_REGISTRY = {
    ...
    "my_optimizer": lambda p, lr: MyOptimizer(p, lr=lr),
}
```

**Step 3** — Run it:

```bash
python train.py --optimizer my_optimizer --lr 0.001
```

---

### `visualizer.py`

#### `Visualizer` class

Manages a live 3×2 matplotlib figure that updates after each epoch.

```
┌──────────────────────┬──────────────────────┐
│   Loss vs Epoch      │  Accuracy vs Epoch   │
│   (train + test)     │  (train + test)      │
├──────────────────────┼──────────────────────┤
│   Loss vs Steps      │  Global Grad Norm    │
│   (batch-level)      │  (mean ± std band)   │
├──────────────────────┼──────────────────────┤
│  Weight L2 Norm      │  Hessian Trace &     │
│  per Layer           │  Sharpness           │
└──────────────────────┴──────────────────────┘
```

- **Loss / Accuracy** — train (blue) and test (red) curves per epoch
- **Loss vs Steps** — every individual batch loss; useful for spotting instability
- **Global Grad Norm** — L2 norm of all gradients with a ±std band; useful for detecting explosion or vanishing
- **Weight L2 Norm per Layer** — how each layer's weights grow or shrink
- **Hessian Trace & Sharpness** — dual y-axis; shows "not computed (use --hessian)" if the flag was not passed

#### `plot_benchmark()` function

Produces a static comparison grid: one row per `(dataset × model)` combination, four columns (train loss, test loss, train accuracy, test accuracy). Each optimizer is a differently coloured line.

---

### `benchmark.py`

Interactive script that trains every `(dataset, model, optimizer)` triple and plots results side-by-side.

#### Registries

**`DATASET_REGISTRY`**
```
[1] MNIST          → "mnist"
[2] Fashion MNIST  → "fashion_mnist"
[3] CIFAR-10       → "cifar10"
[4] Tiny ImageNet  → "tiny_imagenet"
```

**`MODEL_REGISTRY`**
```
[1] MLP        → MLP(input_size=..., hidden_sizes=..., num_classes=...)
[2] ResNet-18  → ResNet18(in_channels=..., num_classes=...)
[3] ViT        → ViT(image_size=..., in_channels=..., num_classes=...)
```

**`OPTIMIZER_REGISTRY`**
```
[1]  SGD           red          lr=0.01
[2]  SGD+Momentum  orange       lr=0.01
[3]  Adam          blue         lr=0.001
[4]  Adagrad       green        lr=0.01
[5]  AdamW         purple       lr=0.001
[6]  NAdam         brown        lr=0.002
[7]  RAdam         pink         lr=0.001
[8]  Lion          teal         lr=0.0001
[9]  LAMB          dark red     lr=0.001
[10] Shampoo       yellow-green lr=0.01
```

---

### `optimizers/`

#### `base.py` — `BaseOptimizer`

Extends `torch.optim.Optimizer`. Subclassing gives you:
- `self.param_groups` — list of dicts, each with `"params"` and your hyperparameters
- `self.state` — per-parameter dict for momentum buffers, second moments, etc.
- `state_dict()` / `load_state_dict()` — checkpointing
- `zero_grad()` — gradient clearing

The only method you must implement is `step()`.

#### `sgd.py` — `VanillaSGD`

Plain gradient descent, no momentum: `θ ← θ − lr · ∇θ L`. The minimal reference implementation — all the boilerplate is shown explicitly.

#### `lion.py` — `Lion`

Sign-based momentum. One buffer, no second moment. Default lr=1e-4.

#### `lamb.py` — `LAMB`

Adam + layer-wise trust ratio. Default weight_decay=0.01, lr=1e-3.

#### `shampoo.py` — `Shampoo`

Kronecker-factored preconditioner. Inverse roots recomputed every 10 steps on CPU. Default lr=0.01.
