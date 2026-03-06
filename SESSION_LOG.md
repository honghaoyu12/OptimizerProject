# Session Log

A chronological record of everything discussed and built across all conversations on this project.

---

## Session 1 — Initial Setup

**What we discussed:**
- Set up the project from scratch as a testing ground for custom numerical optimizers
- Decided on PyTorch as the core dependency
- Chose MNIST as the starting dataset

**What was built:**
- `model.py` — MLP with configurable `hidden_sizes`
- `train.py` — training loop with `OPTIMIZER_REGISTRY`, `get_dataloaders()`, `train_one_epoch()`, `evaluate()`; CLI flags: `--dataset`, `--optimizer`, `--lr`, `--epochs`, `--batch-size`, `--data-dir`, `--device`
- `optimizers/base.py` — `BaseOptimizer` extending `torch.optim.Optimizer`; subclasses only need to implement `step()`
- `optimizers/sgd.py` — `VanillaSGD`, a minimal reference implementation (`θ ← θ − lr · ∇θ`)
- `optimizers/__init__.py` — exports `BaseOptimizer`, `VanillaSGD`

---

## Session 2 — Live Visualisation and Benchmark Runner

**What we discussed:**
- Wanted to see training progress visually in real time
- Wanted to compare multiple optimizers side by side

**What was built:**
- `visualizer.py` — `Visualizer` class: live 2×2 matplotlib dashboard (loss/epoch, accuracy/epoch, weight norms per layer, grad norms per layer); `plot_benchmark()` for static comparison grids
- `benchmark.py` — `DATASET_REGISTRY`, `OPTIMIZER_REGISTRY`, `prompt_multiselect()` (numbered interactive menu), `run_benchmark()`, `plot_benchmark()` call
- `train.py` updates — `--hidden-sizes`, `--no-plot`, `--save-plot` CLI flags; `run_training()` shared loop; `linear_layer_names()`, `weight_norms()` helpers

---

## Session 3 — Diagnostics, CIFAR-10, and Model Selection

**What we discussed:**
- Wanted deeper optimizer diagnostics beyond loss and accuracy
- Wanted CIFAR-10 as a more challenging dataset
- Wanted to switch between MLP and ResNet-18 from the CLI

**What was built:**
- `metrics.py` (new) — `compute_hessian_trace()` (Hutchinson estimator: Tr(H) ≈ (1/S) Σ vᵀHv); `compute_sharpness()` (SAM-style: max‖δ‖≤ε L(θ+δ) − L(θ)); both return `float("nan")` on failure
- `model.py` updates — MLP gains `input_size` and `num_classes` params; `ResNet18` class wrapping torchvision with 3×3 conv1 and Identity maxpool for small images
- `train.py` updates — `DATASET_INFO` dict exported; CIFAR-10 in `get_dataloaders()` with RandomCrop+HorizontalFlip; `build_model()` helper; `train_one_epoch()` return changed from tuple to dict (adds `grad_norm_global`, `grad_norm_std`, `step_losses`); `run_training()` history expanded (adds `time_elapsed`, `target_accuracy_epoch/time`, `grad_norm_global/std`, `hessian_trace`, `sharpness`, `step_losses`); `--model`, `--hessian` CLI flags
- `benchmark.py` updates — `MODEL_REGISTRY`; CIFAR-10 in `DATASET_REGISTRY`; 3-step selection menu (Datasets → Models → Optimizers); results keyed by 3-tuple `(dataset_name, model_name, optimizer_name)`
- `visualizer.py` updates — expanded from 2×2 to 3×2 dashboard (adds batch-level loss, global grad norm with ±std band, hessian/sharpness dual y-axis panel); `update()` and `plot_benchmark()` signatures extended

---

## Session 4 — Vision Transformer

**What we discussed:**
- Wanted a third model option alongside MLP and ResNet-18

**What was built:**
- `model.py` — `ViT` class: patch embedding with pre/post LayerNorm, learnable CLS token and positional embeddings, Pre-LN TransformerEncoder (`norm_first=True`, `enable_nested_tensor=False`), classification from CLS token; `patch_size=4` chosen to divide 28, 32, and 64 evenly
- `train.py` — `build_model()` extended with `"vit"` branch; derives `image_size` from `dataset_info` automatically
- `benchmark.py` — `MODEL_REGISTRY` extended with ViT entry

**Issues encountered and fixed:**
- `nn.TransformerEncoder` with `norm_first=True` triggers a warning → fixed with `enable_nested_tensor=False`

---

## Session 5 — Tiny ImageNet Dataset

**What we discussed:**
- Wanted a larger, harder dataset (200 classes, 64×64 images)

**What was built:**
- `train.py` — `"tiny_imagenet"` added to `DATASET_INFO` and `_DATASET_STATS`; `_setup_tiny_imagenet()` helper that downloads the zip (~236 MB), extracts it, reorganises train (removes extra `images/` subdirectory level) and val (flat → per-class subdirs using `val_annotations.txt`), writes `.setup_done` marker; `get_dataloaders()` handles `ImageFolder` for Tiny ImageNet with RandomCrop(64, padding=8) + HorizontalFlip
- `benchmark.py` — Tiny ImageNet added to `DATASET_REGISTRY`

---

## Session 6 — 7 New Optimizers

**What we discussed:**
- Wanted to add Adagrad, NAdam, RAdam, AdamW (PyTorch built-ins) and Lion, LAMB, Shampoo (custom implementations)

**What was built:**
- `optimizers/lion.py` — Lion (Chen et al. 2023): `c = sign(β₁·m + (1-β₁)·g)`, update = `lr·c + weight_decay·θ`, then EMA momentum update; lr=1e-4 default
- `optimizers/lamb.py` — LAMB (You et al. 2019): Adam moments + bias correction + per-layer trust ratio `‖θ‖/‖u‖` clipped at 10; weight_decay=0.01 default
- `optimizers/shampoo.py` — Shampoo (Gupta et al. 2018): Kronecker-factored EMA preconditioners L and R; inverse 4th-roots via eigendecomp every 10 steps; eigendecomp runs on CPU (float32) for MPS compatibility; diagonal Adagrad fallback for params larger than `max_precond_size`
- `optimizers/__init__.py` — exports Lion, LAMB, Shampoo
- `train.py` — `OPTIMIZER_REGISTRY` extended to 11 entries
- `benchmark.py` — `OPTIMIZER_REGISTRY` extended to 10 display entries with colours

**Issues encountered and fixed:**
- Shampoo: `torch.linalg.eigh` not supported on MPS → fixed by moving eigendecomp to CPU
- Shampoo: eigendecomp "failed to converge" with unbounded matrix accumulation → fixed by switching from pure sum (`L += GGᵀ`) to EMA (`L = decay·L + (1-decay)·GGᵀ`), changing epsilon 1e-6 → 1e-4, and wrapping eigendecomp in try/except with identity fallback

**Benchmark run:**
- Ran all 10 optimizers on MNIST and FashionMNIST with MLP, 10 epochs
- Saved comparison figure to `benchmark.png`

---

## Session 7 — README, CHANGELOG, Git, GitHub

**What we discussed:**
- Wanted a record of all changes and a full project explanation
- Wanted the project on GitHub
- Wanted a `requirements.txt`

**What was built / done:**
- `CHANGELOG.md` — 6 rounds documented in reverse-chronological order
- `README.md` — full project documentation: structure, all CLI flags, example commands, dataset table, optimizer table, model descriptions, adding a custom optimizer guide
- `requirements.txt` — `torch>=2.0.0`, `torchvision>=0.15.0`, `matplotlib>=3.7.0`
- `.gitignore` — excludes `__pycache__/`, `data/`, `*.png`, `.DS_Store`, venv directories
- Git initialised, committed, and pushed to `https://github.com/honghaoyu12/OptimizerProject`

---

## Session 8 — Test Suite and CI Pipeline

**What we discussed:**
- Wanted automated tests for every component
- Wanted a GitHub Actions CI pipeline to run them on every push

**What was built:**
- `tests/__init__.py`
- `tests/test_models.py` — 17 tests: MLP (7), ResNet18 (4), ViT (6)
- `tests/test_optimizers.py` — 18 tests: registry completeness, one-step finite-weight check for all 11 optimizers, VanillaSGD/Lion/LAMB/Shampoo-specific behaviour
- `tests/test_metrics.py` — 9 tests: hessian trace (returns float, finite, positive, nan on no-param model), sharpness (returns float, non-negative, finite, weights restored, monotone in epsilon)
- `tests/test_train.py` — 28 tests: DATASET_INFO keys and metadata, build_model across all model×dataset combos, linear_layer_names/weight_norms, train_one_epoch return dict, evaluate, loss decreases over 5 epochs
- `.github/workflows/ci.yml` — GitHub Actions: ubuntu-latest, Python 3.12, CPU-only PyTorch, pip cache, `pytest tests/ -v`

**Results:**
- All 72 tests passed locally in 2.40s
- First CI run on GitHub Actions passed in 48s
- CI badge added to README

---

## Session 9 — Version Control Strategy (this conversation)

**What we discussed:**
- How to safely upgrade the project in the future without breaking the current working state
- Whether git handles this automatically
- Whether a local backup is needed

**Key points covered:**
- Every git commit is a permanent snapshot — you can always restore any previous state
- **Branches** are the right tool for significant upgrades: work on a branch, keep `main` stable
- **Tags** (`git tag v1.0-stable`) are useful named restore points before big changes
- GitHub is already an off-site backup; no separate local backup needed
- The CI pipeline automatically runs tests on every push — regressions are caught before merging
- Recommended workflow: branch → develop → run tests → merge → tag milestone

---

## Project State at End of Session 9

| Component | Status |
|---|---|
| Models | MLP, ResNet-18, ViT |
| Datasets | MNIST, FashionMNIST, CIFAR-10, Tiny ImageNet |
| Optimizers | 11 total (4 PyTorch built-ins + 4 additional built-ins + VanillaSGD + Lion + LAMB + Shampoo) |
| Tests | 72 passing |
| CI | GitHub Actions, green |
| GitHub | https://github.com/honghaoyu12/OptimizerProject |
