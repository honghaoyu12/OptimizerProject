# OptimizerProject

[![CI](https://github.com/honghaoyu12/OptimizerProject/actions/workflows/ci.yml/badge.svg)](https://github.com/honghaoyu12/OptimizerProject/actions/workflows/ci.yml)

A testing ground for custom numerical optimizers. Train a neural network on MNIST, FashionMNIST, CIFAR-10, CIFAR-100, Tiny ImageNet, or one of five synthetic tabular datasets, plug in your own optimizer, and watch detailed diagnostics update live.

---

## Project Structure

```
OptimizerProject/
├── model.py              # MLP, ResNet-18, and ViT model definitions
├── metrics.py            # Hessian trace and sharpness estimators
├── train.py              # Training loop, data loading, CLI entry point
├── visualizer.py         # Live dashboard and benchmark comparison plots
├── benchmark.py          # Interactive multi-(dataset × model × optimizer) benchmark
├── lr_finder.py          # LR range test (Leslie Smith, 2018)
├── logger.py             # Structured training logger (timestamped session folders)
├── synthetic_datasets.py # Five synthetic tabular datasets for optimizer stress-testing
├── optimizers/
│   ├── __init__.py       # Exports all custom optimizers
│   ├── base.py           # Abstract base class for custom optimizers
│   ├── sgd.py            # VanillaSGD — minimal reference implementation
│   ├── lion.py           # Lion — EvoLved Sign Momentum
│   ├── lamb.py           # LAMB — Layer-wise Adaptive Moments
│   ├── shampoo.py        # Shampoo — Kronecker-factored second-order
│   ├── muon.py           # Muon — Momentum + Nesterov + orthogonalisation
│   ├── adan.py           # Adan — Adaptive Nesterov Momentum
│   └── adahessian.py     # AdaHessian — second-order adaptive (Hessian diagonal)
├── tests/
│   ├── __init__.py
│   ├── test_models.py        # 17 tests for MLP, ResNet18, and ViT
│   ├── test_optimizers.py    # 37 tests for all 14 optimizers
│   ├── test_metrics.py       # 9 tests for Hessian trace and sharpness
│   ├── test_train.py         # 73 tests for datasets, build_model, training loop, schedulers, early stopping, grad clipping, weight decay
│   ├── test_synthetic.py     # 52 tests for the five synthetic datasets
│   ├── test_logger.py        # 30 tests for TrainingLogger
│   ├── test_checkpoints.py   # 5 tests for checkpoint save/restore
│   ├── test_plot_from_logs.py# 9 tests for log-replay plotting
│   └── test_lr_finder.py     # 7 tests for LRFinder
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI — runs all tests on push/PR
├── requirements.txt      # Python dependencies
├── data/                 # Datasets downloaded here automatically
├── CHANGELOG.md          # Technical history of all file changes
├── SESSION_LOG.md        # Narrative log of all project discussions and decisions
├── LR_GUIDE.md           # Guide to LR finder and LR scheduler — what they do and how to use them together
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
pip install torch torchvision matplotlib scikit-learn
```

### 3. Run a single training job

```bash
python train.py
```

This uses all defaults: MLP, MNIST, Adam, 10 epochs. A live 3×2 dashboard opens and updates each epoch. The final figure is saved to `plots/training_curves.png`.

**Common variations:**

```bash
# Change dataset
python train.py --dataset fashion_mnist
python train.py --dataset cifar10
python train.py --dataset cifar100
python train.py --dataset tiny_imagenet   # downloads ~236 MB on first run

# Synthetic tabular datasets (MLP only)
python train.py --dataset illcond
python train.py --dataset sparse
python train.py --dataset noisy_grad
python train.py --dataset manifold
python train.py --dataset saddle

# Change model
python train.py --model resnet18
python train.py --model vit

# Change optimizer
python train.py --optimizer adamw
python train.py --optimizer nadam
python train.py --optimizer lion
python train.py --optimizer muon
python train.py --optimizer adan
python train.py --optimizer adahessian
python train.py --optimizer shampoo

# Change MLP architecture (depth = number of values, width = each value)
python train.py --hidden-sizes 512 256 128   # 3 hidden layers
python train.py --hidden-sizes 64            # 1 hidden layer

# Change training length
python train.py --epochs 20 --batch-size 256

# LR scheduling
python train.py --scheduler cosine
python train.py --scheduler warmup_cosine --warmup-epochs 5
python train.py --scheduler step

# Early stopping (stops if val loss doesn't improve by min-delta for patience epochs)
python train.py --patience 5 --min-delta 1e-4

# Gradient clipping
python train.py --max-grad-norm 1.0

# Weight decay (biases and norm-layer parameters are automatically excluded)
python train.py --weight-decay 1e-4

# Reproducibility
python train.py --seed 42

# Save checkpoints (best.pt and final.pt)
python train.py --checkpoint-dir checkpoints/

# LR range test before training
python train.py --find-lr
python train.py --find-lr --find-lr-iters 100 --find-lr-plot plots/lr_range.png

# Disable the live plot window and save the figure to a file instead
python train.py --no-plot --save-plot my_run.png

# Enable Hessian trace + sharpness diagnostics (adds the bottom-right panel, slower)
python train.py --hessian

# Combine flags freely
python train.py --model vit --dataset cifar10 --optimizer adamw --epochs 15 \
  --scheduler cosine --patience 5 --no-plot --save-plot vit_cifar10.png
```

**All flags for `train.py`:**

```
--dataset        mnist | fashion_mnist | cifar10 | cifar100 | tiny_imagenet |
                 illcond | sparse | noisy_grad | manifold | saddle  (default: mnist)
--model          mlp | resnet18 | vit                                (default: mlp)
--optimizer      adam | adamw | nadam | radam | adagrad |
                 sgd | rmsprop | vanilla_sgd |
                 lion | lamb | shampoo | muon | adan | adahessian    (default: adam)
--lr             learning rate                                        (default: 1e-3)
--epochs         number of epochs                                    (default: 10)
--batch-size     mini-batch size                                     (default: 128)
--hidden-sizes   MLP hidden layer widths, space-separated            (default: 256 128)
--data-dir       where to store downloaded datasets                  (default: ./data)
--device         cpu | cuda | mps | auto                            (default: auto)
--no-plot        disable the live plot window
--save-plot      path to save the final figure                       (default: plots/training_curves.png)
--log-dir        root directory for training logs                    (default: logs/)
--hessian        enable Hessian trace + sharpness computation
--scheduler      none | cosine | step | warmup_cosine               (default: none)
--warmup-epochs  linear warmup epochs for warmup_cosine              (default: 5)
--patience       early stopping patience (0 = disabled)             (default: 0)
--min-delta      min val-loss improvement to reset patience counter  (default: 0.0)
--max-grad-norm  clip global gradient norm (None = disabled)
--weight-decay   L2 weight decay (biases and norm params excluded)   (default: 0.0)
--seed           random seed for reproducibility
--checkpoint-dir directory to save best.pt and final.pt
--find-lr        run LR range test before training
--find-lr-iters  mini-batch steps for the LR range test             (default: 100)
--find-lr-plot   save path for the LR range test plot
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
  [4]  CIFAR-100
  [5]  Tiny ImageNet
  [6]  illcond (synth)
  ...

  > 1,2         ← type this, press Enter

────────────────────────────────────────────────────────
  Select Models
────────────────────────────────────────────────────────
  [1]  MLP
  [2]  ResNet-18
  [3]  ViT

  > 1           ← type this, press Enter

────────────────────────────────────────────────────────
  Select Optimizers
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
  [11] Muon
  [12] Adan
  [13] AdaHessian

  > all         ← type this, press Enter
```

Every `(dataset, model, optimizer)` combination is then trained in sequence and a comparison figure is saved to `plots/benchmark.png`.

You can also pipe the selections in to skip the prompts entirely:

```bash
# MNIST only, MLP only, Adam and AdamW — pipe answers straight in
printf "1\n1\n3,5\n" | python benchmark.py --epochs 5 --save-plot my_benchmark.png
```

**All flags for `benchmark.py`:**

```
--epochs         epochs per run                              (default: 10)
--batch-size     mini-batch size                             (default: 128)
--hidden-sizes   MLP hidden layer widths                     (default: 256 128)
--lr             override all optimizer LRs                  (default: per-optimizer defaults)
--data-dir       dataset location                            (default: ./data)
--device         cpu | cuda | mps | auto                    (default: auto)
--save-plot      output figure path                          (default: plots/benchmark.png)
--log-dir        root directory for training logs            (default: logs/)
--scheduler      none | cosine | step | warmup_cosine        (default: none)
--warmup-epochs  linear warmup epochs for warmup_cosine      (default: 5)
--patience       early stopping patience (0 = disabled)      (default: 0)
--min-delta      min val-loss improvement to reset patience  (default: 0.0)
--max-grad-norm  clip global gradient norm (None = disabled)
--weight-decays  one or more weight decay values (sweep)     (default: 0.0)
--seed           random seed for reproducibility
--checkpoint-dir directory for per-run checkpoints
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
python train.py --model resnet18 --dataset cifar10 --optimizer sgd --lr 0.01 \
  --hessian --no-plot --save-plot resnet_cifar.png

# ViT on CIFAR-10 with AdamW, cosine LR schedule, 15 epochs
python train.py --model vit --dataset cifar10 --optimizer adamw \
  --epochs 15 --scheduler cosine --no-plot --save-plot vit_cifar.png

# Wide MLP on FashionMNIST with early stopping and grad clipping
python train.py --dataset fashion_mnist --hidden-sizes 512 256 128 \
  --patience 5 --max-grad-norm 1.0 --no-plot --save-plot wide_mlp.png

# Muon on synthetic ill-conditioned dataset
python train.py --dataset illcond --optimizer muon --lr 0.02 --no-plot

# AdaHessian on MNIST
python train.py --optimizer adahessian --lr 0.1 --no-plot

# LR range test before training, then train
python train.py --dataset illcond --optimizer adam \
  --find-lr --find-lr-iters 100 --find-lr-plot plots/lr_range.png \
  --epochs 10 --no-plot

# Reproducible run with checkpoints
python train.py --seed 42 --checkpoint-dir checkpoints/ --no-plot

# Benchmark: MNIST + FashionMNIST, all 3 models, all 13 optimizers, 5 epochs
printf "1,2\nall\nall\n" | python benchmark.py --epochs 5 --save-plot full_benchmark.png

# Benchmark: CIFAR-10 only, MLP + ViT, Adam family only (Adam, AdamW, NAdam, RAdam)
printf "3\n1,3\n3,5,6,7\n" | python benchmark.py --epochs 10 --save-plot adam_family.png

# Benchmark with a global LR override and cosine scheduling
python benchmark.py --lr 0.001 --epochs 10 --scheduler cosine
```

---

## Testing

The project ships with **234 pytest tests** covering every major component.

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

Expected output: `234 passed` in a few seconds (all on CPU, no downloads needed).

### Test files

| File | Tests | What it covers |
|---|---|---|
| `tests/test_models.py` | 17 | MLP, ResNet18, ViT forward passes; output shapes; patch size assertion |
| `tests/test_optimizers.py` | 37 | Registry completeness; finite weights after one step for all 14 optimizers; optimizer-specific state and behaviour |
| `tests/test_metrics.py` | 9 | Hessian trace (type, finiteness, positivity, NaN on no-param model); sharpness (type, non-negativity, weight restoration, epsilon monotonicity) |
| `tests/test_train.py` | 68 | `DATASET_INFO` metadata; `build_model` for all model×dataset combos; `train_one_epoch` return dict; schedulers; early stopping; gradient clipping; seed reproducibility |
| `tests/test_synthetic.py` | 52 | Registry completeness; loader shapes; label validity; NaN checks; standardisation; reproducibility; dataset-specific properties |
| `tests/test_logger.py` | 30 | `TrainingLogger` session creation; `log_run` file format; `close` summary; edge cases |
| `tests/test_checkpoints.py` | 5 | Checkpoint save and restore for best and final model states |
| `tests/test_plot_from_logs.py` | 9 | Log-replay plotting from saved session directories |
| `tests/test_lr_finder.py` | 7 | History keys/shape; monotone LR schedule; weight/LR state restoration; suggestion range; AdaHessian compatibility |

### CI pipeline

Every push and pull request to `main` triggers the GitHub Actions workflow defined in `.github/workflows/ci.yml`:

- **Runner**: `ubuntu-latest`, Python 3.12
- **PyTorch**: CPU-only install (`--index-url https://download.pytorch.org/whl/cpu`) for fast execution
- **Cache**: pip dependencies cached on `requirements.txt` hash
- **Command**: `pytest tests/ -v`

The badge at the top of this file reflects the current status of the `main` branch.

---

## Files in Detail

### `model.py`

Defines three model classes.

**`MLP(input_size=784, hidden_sizes=[256,128], num_classes=10)`**

A fully-connected network. `hidden_sizes` controls both depth (number of values) and width (each value).

```bash
python train.py --hidden-sizes 256 128        # default: 784→256→128→10
python train.py --hidden-sizes 512            # one wide layer
python train.py --hidden-sizes 512 256 128 64 # four layers
```

**`ResNet18(in_channels=3, num_classes=10)`**

Wraps `torchvision.models.resnet18` with two adaptations for small images:
- `conv1` changed from 7×7/stride-2 → 3×3/stride-1 (preserves spatial resolution)
- `maxpool` replaced with `nn.Identity()` (avoids over-downsampling on 28–32 px inputs)

**`ViT(image_size=32, in_channels=3, patch_size=4, num_classes=10, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1)`**

Vision Transformer adapted for small images. The default `patch_size=4` divides 28, 32, and 64 evenly:

| Dataset | Image | Patches |
|---|---|---|
| MNIST / FashionMNIST | 28×28 | 7×7 = 49 patches |
| CIFAR-10 / CIFAR-100 | 32×32 | 8×8 = 64 patches |
| Tiny ImageNet | 64×64 | 16×16 = 256 patches |

---

### `train.py`

The core training module and the CLI entry point for single runs.

#### Supported datasets

| Key | Source | Images | Classes |
|---|---|---|---|
| `"mnist"` | `torchvision.datasets.MNIST` | 28×28 gray | 10 |
| `"fashion_mnist"` | `FashionMNIST` | 28×28 gray | 10 |
| `"cifar10"` | `CIFAR10` | 32×32 RGB | 10 |
| `"cifar100"` | `CIFAR100` | 32×32 RGB | 100 |
| `"tiny_imagenet"` | `ImageFolder` (auto-downloaded) | 64×64 RGB | 200 |
| `"illcond"` | `synthetic_datasets.py` | 64-D tabular | 2 |
| `"sparse"` | `synthetic_datasets.py` | 100-D tabular | 2 |
| `"noisy_grad"` | `synthetic_datasets.py` | 64-D tabular | 2 |
| `"manifold"` | `synthetic_datasets.py` | 64-D tabular | 2 |
| `"saddle"` | `synthetic_datasets.py` | 64-D tabular | 2 |

Synthetic tabular datasets are MLP-only. ResNet-18 and ViT raise a `ValueError` when a tabular dataset is selected.

#### Available optimizers

| Key (`--optimizer`) | Display name | Source | Default LR |
|---|---|---|---|
| `adam` | Adam | `torch.optim.Adam` | 1e-3 |
| `adamw` | AdamW | `torch.optim.AdamW` | 1e-3 |
| `nadam` | NAdam | `torch.optim.NAdam` | 2e-3 |
| `radam` | RAdam | `torch.optim.RAdam` | 1e-3 |
| `adagrad` | Adagrad | `torch.optim.Adagrad` | 1e-2 |
| `sgd` | SGD+Momentum | `torch.optim.SGD(momentum=0.9)` | 1e-2 |
| `rmsprop` | RMSprop | `torch.optim.RMSprop` | 1e-3 |
| `vanilla_sgd` | VanillaSGD | `optimizers/sgd.py` | 1e-2 |
| `lion` | Lion | `optimizers/lion.py` | 1e-4 |
| `lamb` | LAMB | `optimizers/lamb.py` | 1e-3 |
| `shampoo` | Shampoo | `optimizers/shampoo.py` | 1e-2 |
| `muon` | Muon | `optimizers/muon.py` | 2e-2 |
| `adan` | Adan | `optimizers/adan.py` | 1e-3 |
| `adahessian` | AdaHessian | `optimizers/adahessian.py` | 1e-1 |

**Lion** — EvoLved Sign Momentum (Chen et al., 2023). One momentum buffer; every update is the sign of a β₁-interpolated gradient. Use ~10× smaller LR than Adam.

**LAMB** — Layer-wise Adaptive Moments for Batch training (You et al., 2019). Adam moments + per-tensor trust ratio `‖θ‖ / ‖u‖`. Designed for large-batch training.

**Shampoo** — Kronecker-factored second-order optimizer (Gupta et al., 2018). Eigendecomposition runs on CPU for MPS compatibility.

**Muon** — Momentum with Nesterov update and Gram-Schmidt orthogonalisation of the gradient matrix (Kosson et al., 2024). Effective for weight matrices; falls back to SGD for 1-D params. Default lr=0.02.

**Adan** — Adaptive Nesterov Momentum (Xie et al., 2023). Tracks three moment terms (gradient, gradient difference, gradient difference squared) for more accurate curvature estimates. Default lr=1e-3.

**AdaHessian** — Second-order adaptive optimizer (Yao et al., 2021). Uses the diagonal of the Hessian (estimated via Hutchinson) as a preconditioner. Requires `create_graph=True` during backward. Default lr=0.1.

#### Schedulers

| Key (`--scheduler`) | Behaviour |
|---|---|
| `none` | No scheduling (default) |
| `cosine` | CosineAnnealingLR from initial LR to 0 |
| `step` | StepLR — multiply by 0.1 every `epochs/3` steps |
| `warmup_cosine` | Linear warmup for `--warmup-epochs` then cosine decay |

#### Early stopping

Pass `--patience N` to stop training when validation loss has not improved by more than `--min-delta` for N consecutive epochs. The best-epoch weights are automatically restored after stopping.

#### Gradient clipping

Pass `--max-grad-norm F` to clip the global gradient L2 norm to F before each optimizer step.

#### `train_one_epoch()` return dict

```python
{
    "loss":                 float,         # mean cross-entropy over the epoch
    "acc":                  float,         # fraction correct (not percentage)
    "grad_norms_per_layer": dict,          # mean grad L2 norm per linear layer
    "grad_norm_global":     float,         # post-clip L2 norm across all gradients
    "grad_norm_before_clip":float,         # pre-clip L2 norm (== post-clip when no clipping)
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
    "target_accuracy_epoch": int | None,
    "target_accuracy_time":  float | None,
    "grad_norm_global":      list[float],
    "grad_norm_before_clip": list[float],              # pre-clip global norm per epoch
    "grad_norm_std":         list[float],
    "hessian_trace":         list[float],              # NaN if --hessian not passed
    "sharpness":             list[float],              # NaN if --hessian not passed
    "step_losses":           list[float],              # all batches, all epochs
    "learning_rates":        list[float],              # LR at the start of each epoch
    "early_stopped_epoch":   int | None,               # epoch where training stopped early
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
    "my_optimizer": lambda p, lr, wd: MyOptimizer(p, lr=lr),
}
```

**Step 3** — Run it:

```bash
python train.py --optimizer my_optimizer --lr 0.001
```

---

### `lr_finder.py`

`LRFinder` class — runs a learning-rate range test before training to suggest a good starting LR. For a full explanation of how the finder and scheduler differ and how to combine them, see [`LR_GUIDE.md`](LR_GUIDE.md).

```python
finder = LRFinder(model, optimizer, criterion, device,
                  start_lr=1e-7, end_lr=10.0, num_iter=100)
finder.run(train_loader)
print(finder.suggestion())   # LR at steepest loss decrease
finder.plot(save_path="plots/lr_range.png")
```

The model and optimizer are fully restored to their original state after `run()`. AdaHessian (`requires_create_graph=True`) is supported transparently.

---

### `synthetic_datasets.py`

Five synthetic tabular datasets designed to stress-test specific optimizer properties.

| Key | Features | What it tests |
|---|---|---|
| `illcond` | 64-D Gaussian, κ=1000 | Curvature handling |
| `sparse` | 100-D, 5 informative | Coordinate adaptivity |
| `noisy_grad` | 64-D, 30% label noise | Stochastic robustness |
| `manifold` | 64-D (make_moons + noise) | Nonconvex landscapes |
| `saddle` | 64-D bimodal | Saddle-point escape |

All return `(train_loader, test_loader)` with features standardised on the training split.

---

### `logger.py`

`TrainingLogger` class — writes structured logs for every training session.

- Creates `logs/<YYYY-MM-DD_HH-MM-SS>/` per session.
- `log_run(config, history)` writes a `.log` file per run with epoch-wise CSV and batch-wise step losses.
- `close()` writes `run_summary.log` with session timing and a final results table.

Logs are written automatically on every `train.py` and `benchmark.py` run. Use `--log-dir` to change the root directory.

---

### `visualizer.py`

#### `Visualizer` class

Manages a live 3×2 matplotlib figure that updates after each epoch.

```
┌──────────────────────┬──────────────────────┐
│   Loss vs Epoch      │  Accuracy vs Epoch   │
│   (train + test      │  (train + test)      │
│    + LR trace)       │                      │
├──────────────────────┼──────────────────────┤
│   Loss vs Steps      │  Global Grad Norm    │
│   (batch-level)      │  (mean ± std band)   │
├──────────────────────┼──────────────────────┤
│  Weight L2 Norm      │  Hessian Trace &     │
│  per Layer           │  Sharpness           │
└──────────────────────┴──────────────────────┘
```

#### `plot_benchmark()` function

Produces a static comparison grid: one row per `(dataset × model)` combination, five columns (train loss, test loss, train accuracy, test accuracy, LR vs epoch). Each optimizer is a differently coloured line.

---

### `benchmark.py`

Interactive script that trains every `(dataset, model, optimizer)` triple and plots results side-by-side.

#### `OPTIMIZER_REGISTRY`

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
[11] Muon          magenta      lr=0.02
[12] Adan          olive        lr=0.001
[13] AdaHessian    gold         lr=0.1
```

---

### `optimizers/`

#### `base.py` — `BaseOptimizer`

Extends `torch.optim.Optimizer`. The only method you must implement is `step()`.

#### `sgd.py` — `VanillaSGD`

Plain gradient descent: `θ ← θ − lr · ∇θ L`. Minimal reference implementation.

#### `lion.py` — `Lion`

Sign-based momentum. One buffer, no second moment. Default lr=1e-4.

#### `lamb.py` — `LAMB`

Adam + layer-wise trust ratio. Default weight_decay=0.01, lr=1e-3.

#### `shampoo.py` — `Shampoo`

Kronecker-factored preconditioner. Inverse roots recomputed every 10 steps on CPU. Default lr=0.01.

#### `muon.py` — `Muon`

Nesterov momentum with Gram-Schmidt orthogonalisation of the gradient matrix before the update step. Designed for weight matrices; 1-D params (bias, BN) use plain SGD. Default lr=0.02.

#### `adan.py` — `Adan`

Adaptive Nesterov Momentum (Xie et al., 2023). Maintains three EMA terms — gradient, gradient difference, and squared gradient difference — giving more accurate curvature information than Adam. Default lr=1e-3.

#### `adahessian.py` — `AdaHessian`

Second-order adaptive optimizer (Yao et al., 2021). Estimates the Hessian diagonal using the Hutchinson estimator (random ±1 vectors) and uses it as a per-parameter preconditioner. Requires `create_graph=True` during backward (set via `requires_create_graph = True` flag). Default lr=0.1.

---

## Session Log

A narrative record of how the project was built, session by session. The full log is in [`SESSION_LOG.md`](SESSION_LOG.md).

| Session | Topic | Key additions |
|---|---|---|
| 1 | Initial setup | `model.py` (MLP), `train.py`, `optimizers/` (BaseOptimizer, VanillaSGD) |
| 2 | Live visualisation and benchmark runner | `visualizer.py` (2×2 dashboard), `benchmark.py`, `run_training()` |
| 3 | Diagnostics, CIFAR-10, model selection | `metrics.py`, ResNet-18, CIFAR-10, 3×2 dashboard, richer training metrics |
| 4 | Vision Transformer | ViT in `model.py` and `build_model()` |
| 5 | Tiny ImageNet | Auto-download and reorganisation in `train.py` |
| 6 | 7 new optimizers | Lion, LAMB, Shampoo (custom); AdamW, NAdam, RAdam, Adagrad (PyTorch) |
| 7 | README, CHANGELOG, Git, GitHub | `CHANGELOG.md`, `README.md`, `requirements.txt`, pushed to GitHub |
| 8 | Test suite and CI pipeline | 72 pytest tests across 4 files, `.github/workflows/ci.yml`, all green |
| 9 | Version control strategy | Discussed branches, tags, and safe upgrade workflow |
| 10 | Housekeeping and branch setup | `SESSION_LOG.md`, `MEMORY.md` updated, `new_feature` branch created |
| 11 | CIFAR-100 dataset | `cifar100` added to `DATASET_INFO`, benchmark, and tests |
| 12 | Synthetic datasets | `synthetic_datasets.py` (5 datasets), `test_synthetic.py` (45 tests) |
| 13 | Bug fix and training validation | `visualizer.py` FileNotFoundError fix; 6 clean training runs |
| 14 | Training logger and plot directory enforcement | `logger.py`, `--log-dir`, default plot paths into `plots/` |
| 15 | Debug and test pass | 5 clean runs verified; all 140 tests passing |
| 16 | Muon, Adan, AdaHessian optimizers | 3 new custom optimizers; 14 total; 37 optimizer tests |
| 17 | Schedulers, early stopping, grad clipping, checkpoints, log-replay | 4 schedulers; EarlyStopping; grad clipping; `--seed`; `--checkpoint-dir`; `test_checkpoints.py`; `test_plot_from_logs.py` |
| 18 | LR range test | `lr_finder.py`, `--find-lr` flags in `train.py`, `test_lr_finder.py` (7 tests) |
