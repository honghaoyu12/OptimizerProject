# Changelog

All notable changes to this project are documented here, in reverse-chronological order.

---

## Round 10 — Bug Fix and Training Validation

### `visualizer.py`
- **Bug fix**: `Visualizer.close()` and `plot_benchmark()` both raised `FileNotFoundError` when `--save-plot` pointed to a subdirectory (e.g. `plots/foo.png`) that did not yet exist. Fixed by adding `os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)` before each `savefig` call.
- Added `import os` to module imports.

### Training validation
Six end-to-end training runs confirmed clean after the fix:

| Config | Final test acc |
|---|---|
| MNIST / MLP / Adam | 98.0% |
| FashionMNIST / MLP / AdamW | 87.8% |
| MNIST / ResNet-18 / SGD | 98.6% |
| MNIST / MLP / Lion | 96.5% |
| FashionMNIST / MLP / Shampoo | 87.8% |
| illcond (synthetic) / MLP / Adam | 92.8% |

Plots saved to `plots/`. All 140 tests continue to pass.

---

## Round 9 — Synthetic Datasets

### New file: `synthetic_datasets.py`

Five synthetic tabular datasets for probing specific optimizer weaknesses. All datasets produce binary classification tasks, standardise features on the training split, and return `(train_loader, test_loader)` pairs.

| Key | Shape | What it tests |
|---|---|---|
| `illcond` | (2000, 64) | **Curvature handling** — Gaussian features with covariance condition number = 1 000 (log-spaced eigenvalues). Adaptive curvature methods (Adam, Shampoo) should outperform SGD. |
| `sparse` | (2000, 100) | **Coordinate adaptivity** — only 5 out of 100 features are informative (via `sklearn.make_classification`). Per-coordinate optimizers (Adagrad, Adam) should adapt faster. |
| `noisy_grad` | (2000, 64) | **Stochastic robustness** — 30% random label flips injected after data generation. Momentum-based optimizers (Adam, NAdam) should be more robust. |
| `manifold` | (2000, 64) | **Nonconvex landscapes** — `sklearn.make_moons` (2-D nonlinear) embedded in 64-D with 62 noise dimensions. Tests optimizers on curved decision boundaries. |
| `saddle` | (2000, 64) | **Escape from flat regions** — bimodal positive class (±2) vs. tight negative cluster at origin. Creates saddle-point structure. Momentum and second-order methods should escape faster. |

Internal `_to_loaders()` helper: standardises with train-split mean/std, splits 80/20 train/test, returns `DataLoader` pairs.
`SYNTHETIC_LOADERS` registry mirrors the `OPTIMIZER_REGISTRY` pattern for easy lookup.

### `train.py`
- `DATASET_INFO` extended with all 5 synthetic keys; each entry includes `"tabular": True`.
- `get_dataloaders()`: synthetic datasets are delegated to `SYNTHETIC_LOADERS` before the `_DATASET_STATS` lookup.
- `build_model()`: raises `ValueError` if a non-MLP model is requested for a tabular dataset.
- `--dataset` CLI help updated.

### `benchmark.py`
- `DATASET_REGISTRY` extended with all 5 synthetic datasets (labelled with `(synth)` suffix).
- `run_benchmark()`: ResNet-18 and ViT are skipped with a printed message when a tabular dataset is selected; run counter advances correctly.

### `requirements.txt`
- Added `scikit-learn>=1.3.0` (used by `make_classification` and `make_moons`).

### `tests/test_synthetic.py` (new)
- 45 tests across 7 classes:
  - `TestRegistry` — all keys present, all values callable.
  - `TestLoaderOutputs` (parametrized across all 5 datasets) — returns two `DataLoader`s; correct feature shape; labels in `{0, 1}`; no NaN/Inf; train > test size; features approximately standardised; reproducible with same seed; different seeds differ.
  - `TestIllcond`, `TestSparse`, `TestNoisyGrad`, `TestManifold`, `TestSaddle` — dataset-specific property checks.

### `tests/test_train.py`
- `test_all_datasets_present`: expected key set updated to include all 5 synthetic keys.
- `test_dataset_metadata`: 5 new parametrize cases added.
- `test_mlp_all_datasets`: all 5 synthetic datasets added.
- `test_tabular_rejects_non_mlp`: new parametrized test — `resnet18` and `vit` raise `ValueError` for tabular datasets.
- Total tests: 140 (up from 76).

---

## Round 8 — CIFAR-100 Dataset

### `train.py`
- Added `"cifar100"` to `DATASET_INFO`: `{in_channels: 3, input_size: 3072, num_classes: 100}` (32×32 RGB, 100 classes).
- Added CIFAR-100 normalization stats to `_DATASET_STATS`: mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761).
- `get_dataloaders()`: CIFAR-100 shares the CIFAR-10 augmentation pipeline (RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize) using `datasets.CIFAR100`.
- `--dataset` CLI help updated to include `cifar100`.

### `benchmark.py`
- `DATASET_REGISTRY` extended with `("CIFAR-100", "cifar100")`.

### `tests/test_train.py`
- `test_all_datasets_present`: updated expected key set to include `"cifar100"`.
- `test_dataset_metadata`: added `("cifar100", 3, 3072, 100)` parametrize case.
- `test_mlp_all_datasets`, `test_resnet18_all_datasets`, `test_vit_all_datasets`: `"cifar100"` added to each.
- Total tests: 76 (up from 72), all passing.

---

## Round 7 — Test Suite and CI Pipeline

### New files: `tests/test_models.py`, `tests/test_optimizers.py`, `tests/test_metrics.py`, `tests/test_train.py`, `tests/__init__.py`

**72 pytest tests** covering every major component:

- **`test_models.py`** (17 tests) — MLP, ResNet18, ViT forward passes across all supported image sizes and class counts; patch size assertion; positional embedding shape check.
- **`test_optimizers.py`** (18 tests) — Registry completeness (11 entries); parametrized one-step finite-weight check for every optimizer; `VanillaSGD` update direction and zero-grad; `Lion` momentum buffer and weight decay; `LAMB` state fields and step counter; `Shampoo` Kronecker state, diagonal fallback, and multi-step stability.
- **`test_metrics.py`** (9 tests) — `compute_hessian_trace`: return type, finiteness, positivity, graceful NaN on parameter-free model. `compute_sharpness`: return type, non-negativity, finiteness, weight restoration, monotonicity with epsilon.
- **`test_train.py`** (28 tests) — `DATASET_INFO` keys and metadata; `build_model` for all model×dataset combos; `linear_layer_names` / `weight_norms` helpers; `train_one_epoch` return keys, finite loss, accuracy range, step-loss count, per-layer grad-norm keys; `evaluate` finiteness; loss-decreases-over-5-epochs integration test.

### New file: `.github/workflows/ci.yml`

- GitHub Actions workflow triggered on every push and pull request to `main`.
- Runs on `ubuntu-latest` with Python 3.12.
- Installs CPU-only PyTorch (`--index-url https://download.pytorch.org/whl/cpu`) for fast CI execution.
- Pip cache keyed on `requirements.txt` to speed up subsequent runs.
- Executes `pytest tests/ -v`; all 72 tests pass in ~48 s.

---

## Round 6 — Optimizer Expansion (7 new optimizers)

### New files: `optimizers/lion.py`, `optimizers/lamb.py`, `optimizers/shampoo.py`

**`Lion`** (EvoLved Sign Momentum, Chen et al. 2023)
- Maintains one momentum buffer (no second moment, ~half the memory of Adam).
- Each step direction is purely the sign of a β₁-interpolated gradient, so all updates have equal magnitude.
- Default lr=1e-4 (sign-only updates are stronger; typical rule: use ~10× smaller LR than Adam).
- `betas=(0.9, 0.99)`: β₁ controls interpolation for the sign, β₂ controls the EMA update.

**`LAMB`** (Layer-wise Adaptive Moments for Batch training, You et al. 2019)
- Adam moments + bias correction (identical to Adam up to this point).
- Adds a per-tensor trust ratio `‖θ‖ / ‖u‖` (clipped at 10) that scales the update layer-wise, stabilising training at large batch sizes.
- Default weight_decay=0.01 (applied inside the update direction, not as a gradient penalty).

**`Shampoo`** (Gupta et al. 2018)
- Maintains Kronecker-factored preconditioners L ∈ ℝ^{m×m} and R ∈ ℝ^{n×n} for each 2-D parameter.
- Preconditioned update: L^{-1/4} G R^{-1/4}.
- Matrix inverse 4th-roots computed via eigendecomposition every `update_freq` steps (default 10). Eigendecomposition runs on CPU (float32) for device compatibility (MPS does not support `torch.linalg.eigh`).
- Parameters with any dimension > `max_precond_size` (default 512) fall back to a diagonal Adagrad preconditioner.
- Higher-dimensional params (Conv2d) are reshaped to 2-D before factoring.

### `optimizers/__init__.py`
- Exports `Lion`, `LAMB`, `Shampoo`.

### `train.py`
- `OPTIMIZER_REGISTRY` extended with: `adagrad`, `adamw`, `nadam`, `radam`, `lion`, `lamb`, `shampoo`.
- Import updated.

### `benchmark.py`
- `OPTIMIZER_REGISTRY` extended with: `AdamW` (purple), `NAdam` (brown), `RAdam` (pink), `Lion` (teal), `LAMB` (dark red), `Shampoo` (yellow-green).
- Import updated.

---

## Round 5 — Tiny ImageNet Dataset

### `train.py`
- Added `"tiny_imagenet"` to `DATASET_INFO`: `{in_channels: 3, input_size: 12288, num_classes: 200}` (64×64 RGB, 200 classes).
- Added Tiny ImageNet normalization stats to `_DATASET_STATS`: mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821).
- Added `_TINY_IMAGENET_URL` constant and `_setup_tiny_imagenet(data_dir) → str` helper:
  - Downloads `tiny-imagenet-200.zip` (~236 MB) from the Stanford CS231n URL if not already present.
  - Extracts the archive.
  - Reorganises the **training set**: moves images from `train/<class>/images/<file>.JPEG` → `train/<class>/<file>.JPEG` so `ImageFolder` can read it.
  - Reorganises the **validation set**: reads `val_annotations.txt` and moves images from the flat `val/images/` directory into per-class subdirectories `val/<class>/<file>.JPEG`.
  - Writes a `.setup_done` marker so all subsequent calls return immediately.
- `get_dataloaders()` gains a `tiny_imagenet` branch: calls `_setup_tiny_imagenet()`, then uses `datasets.ImageFolder` for both train and val. Training transform uses `RandomCrop(64, padding=8)` + `RandomHorizontalFlip` + `Normalize`.
- Added new imports: `os`, `shutil`, `urllib.request`, `zipfile`.
- `--dataset` CLI help updated.

### `benchmark.py`
- `DATASET_REGISTRY` extended with `("Tiny ImageNet", "tiny_imagenet")`.

---

## Round 4 — Vision Transformer

### `model.py`
- Added `ViT(image_size, in_channels, patch_size=4, num_classes=10, dim=128, depth=6, heads=8, mlp_dim=256, dropout=0.1)`.
  - Image is split into non-overlapping `patch_size×patch_size` patches (patch_size=4 divides both 28 and 32 evenly).
  - Patches are linearly projected to `dim`-dimensional tokens via a small MLP with pre- and post-LayerNorm.
  - A learnable CLS token and learnable positional embeddings are added.
  - Tokens are processed by `depth` Transformer encoder layers using `nn.TransformerEncoderLayer` (Pre-LN via `norm_first=True`, `batch_first=True`).
  - The CLS token output is normalized and passed to a linear classification head.
  - Weights initialized with truncated normal (std=0.02) following the original ViT paper.

### `train.py`
- `build_model()` extended with a `"vit"` branch: derives `image_size` from `dataset_info` as `sqrt(input_size / in_channels)`.
- `--model` CLI help text updated to include `vit`.

### `benchmark.py`
- `MODEL_REGISTRY` extended with a `"ViT"` entry using the same `image_size` derivation.
- Import updated to include `ViT`.

---

## Round 3 — Diagnostics, CIFAR-10, and Model Selection

### New Files
- **`metrics.py`** — Two optimizer diagnostic functions:
  - `compute_hessian_trace(model, batch, criterion, device, n_samples=3)` — Hutchinson estimator of the Hessian trace: Tr(H) ≈ (1/S) Σ vᵀHv, v ~ Rademacher(±1). Uses `autograd.grad(create_graph=True)` for Hessian-vector products.
  - `compute_sharpness(model, batch, criterion, device, epsilon=1e-2, n_directions=5)` — SAM-style sharpness: max‖δ‖≤ε L(θ+δ) − L(θ), sampled over random unit perturbations. Saves and restores model weights.
  - Both return `float("nan")` on failure (e.g. MPS autograd limitations).

### `model.py`
- `MLP` now accepts `input_size: int = 784` and `num_classes: int = 10`, replacing the hardcoded `28*28` and `10` constants. CIFAR-10 can be handled with `input_size=3072`.
- Added `ResNet18(in_channels=3, num_classes=10)` — wraps `torchvision.models.resnet18` with two small-image adaptations: `conv1` changed from 7×7/stride-2 to 3×3/stride-1; `maxpool` replaced with `nn.Identity()` to avoid over-downsampling on 28×32 px inputs.

### `train.py`
- Added `DATASET_INFO` dict (exported for `benchmark.py`):
  ```python
  {"mnist": ..., "fashion_mnist": ..., "cifar10": {"in_channels": 3, "input_size": 3072, "num_classes": 10}}
  ```
- `get_dataloaders()` now supports CIFAR-10. Training transform includes `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` + `Normalize`. Test transform is `Normalize` only.
- Added `build_model(model_name, dataset_info, hidden_sizes) → nn.Module` helper that dispatches to `MLP` or `ResNet18` based on the name string.
- `train_one_epoch()` return value changed from `(loss, acc, grad_norms_per_layer)` tuple to a dict with keys:
  - `loss`, `acc` — unchanged
  - `grad_norms_per_layer` — renamed from `grad_norms` (mean per linear layer)
  - `grad_norm_global` — L2 norm across all parameter gradients (new)
  - `grad_norm_std` — std of individual parameter gradient norms (new)
  - `step_losses` — list of per-batch loss values for the epoch (new)
- `run_training()` history dict expanded with:
  - `time_elapsed` — wall-clock seconds since start, recorded per epoch
  - `target_accuracy_epoch` / `target_accuracy_time` — first epoch/time where test accuracy ≥ `target_acc`
  - `grad_norm_global`, `grad_norm_std` — per-epoch global gradient statistics
  - `hessian_trace`, `sharpness` — per-epoch diagnostic values (NaN when `compute_hessian=False`)
  - `step_losses` — all batch-level losses across all epochs concatenated
- `run_training()` signature additions: `compute_hessian=False`, `target_acc=0.95`.
- `main()` CLI additions: `--model mlp|resnet18`, `--hessian` flag.
- `main()` now calls `build_model()` and passes all new metrics to `Visualizer.update()`.

### `benchmark.py`
- Added `MODEL_REGISTRY` (`OrderedDict`) with entries for `"MLP"` and `"ResNet-18"`, each with a `factory` lambda and `description`.
- `DATASET_REGISTRY` extended with `("CIFAR-10", "cifar10")`.
- Interactive selection now has three steps (was two): **Datasets → Models → Optimizers**.
- `run_benchmark()` now iterates over all `(dataset, model, optimizer)` triples. Results are keyed by 3-tuples `(ds_name, model_name, opt_name)` instead of 2-tuples.
- `plot_benchmark()` call updated to pass `model_names`.

### `visualizer.py`
- `Visualizer` expanded from a 2×2 to a 3×2 grid (`figsize` 12×8 → 14×14). New panels:
  - **Loss vs Steps** (row 2, left) — batch-level loss trace across all steps
  - **Global Gradient Norm** (row 2, right) — mean ± std band per epoch
  - **Hessian Trace & Sharpness** (row 3, right) — dual y-axis (twinx); shows "not computed (use --hessian)" placeholder when data is NaN
- `update()` signature extended: `grad_norm_global`, `grad_norm_std`, `step_losses`, `hessian_trace`, `sharpness`.
- `plot_benchmark()` signature updated: added `model_names`; rows now labelled `"{Dataset} / {Model}"`; keys into `results` changed from 2-tuple to 3-tuple.

---

## Round 2 — Live Visualisation and Configurable Architecture

### `train.py`
- Added `--hidden-sizes` CLI flag (space-separated integers); controls MLP depth and width.
- Added `--no-plot` and `--save-plot` CLI flags.
- Added `run_training()` function — shared multi-epoch loop used by both `train.py` and `benchmark.py`.
- Added `linear_layer_names()` and `weight_norms()` helpers.

### `visualizer.py` (new)
- Created `Visualizer` class with a 2×2 live matplotlib dashboard:
  - Loss vs Epoch (train + test)
  - Accuracy vs Epoch (train + test)
  - Weight L2 norm per linear layer
  - Gradient L2 norm per linear layer
- `update()` called each epoch; `close()` saves to file and ends interactive mode.
- Created `plot_benchmark()` — static grid plot for multi-optimizer comparison runs (one row per dataset, four columns: train loss / test loss / train acc / test acc).

### `benchmark.py` (new)
- Created `DATASET_REGISTRY` (MNIST, Fashion MNIST) and `OPTIMIZER_REGISTRY` (SGD, SGD+Momentum, Adam, Adagrad) as `OrderedDict`s.
- `prompt_multiselect()` — numbered menu for interactive multi-selection.
- `run_benchmark()` — trains every (dataset, optimizer) pair, returns `{(ds, opt): history}`.
- Results passed to `plot_benchmark()` and optionally saved.

---

## Round 1 — Initial Project

### `model.py` (new)
- Created `MLP` with a hardcoded 784-input and configurable `hidden_sizes` list.

### `train.py` (new)
- MNIST training loop with `OPTIMIZER_REGISTRY` (adam, sgd, rmsprop, vanilla_sgd).
- `get_dataloaders()` for MNIST and FashionMNIST.
- `train_one_epoch()` and `evaluate()` functions.
- CLI flags: `--dataset`, `--optimizer`, `--lr`, `--epochs`, `--batch-size`, `--data-dir`, `--device`.

### `optimizers/` (new)
- `base.py` — `BaseOptimizer(torch.optim.Optimizer)`: abstract base requiring `step()` to be implemented. Inherits param groups, state dict, and `zero_grad()` for free.
- `sgd.py` — `VanillaSGD`: plain gradient descent (`p ← p − lr · ∇p`), serves as a minimal reference implementation.
- `__init__.py` — exports `BaseOptimizer`, `VanillaSGD`.
