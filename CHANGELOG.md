# Changelog

All notable changes to this project are documented here, in reverse-chronological order.

---

## Round 19 — Benchmark Report Generator

### New file: `report.py`

`generate_report(results, config, save_path) -> str`

Produces a six-section Markdown report from the `results` dict returned by `run_benchmark()`.

**Section 1 — Header**: generation timestamp.

**Section 2 — Setup**: table of benchmark settings (datasets, models, optimizers, epochs, batch size, scheduler, weight decays, LRs, seed). Datasets/models/optimizers extracted from `results` keys; other settings from the `config` dict.

**Section 3 — Results Overview**: one sub-table per `(dataset, model)` combination, sorted by dataset then model. Columns: `Optimizer | Final Test Acc (%) | Best Test Acc (%) | Epochs Run | Time (s) | Train-Test Gap (%)`. Rows sorted descending by final test accuracy.

**Section 4 — Rankings**: one `###` subsection per `(dataset, model)` combination. Four ranked bullet points: 🥇 best final accuracy, 🚀 fastest, 📈 best peak accuracy, 🎯 most stable (lowest `std(test_acc[-3:])`, fallback to full list).

**Section 5 — Per-Optimizer Summary**: one `###` subsection per series name. Per-combo bullet with final/peak accuracy, epoch count, and elapsed time. Appends `⚠️ high train-test gap (X.X%)` when gap > 5% and `(early stopped at epoch N)` when applicable.

**Section 6 — Key Observations**: auto-generated bullets identifying highest overall accuracy, fastest optimizer (by mean time), most stable optimizer (by mean last-3 std), early-stop count, and high-gap count.

File writing follows the same `os.makedirs(..., exist_ok=True)` pattern as `visualizer.py`. Returns the Markdown string regardless of `save_path`; skips writing when `save_path=''`.

### `benchmark.py`

- New import: `from report import generate_report`.
- New CLI flag: `--report-path` (default: `reports/benchmark_report.md`; `''` to disable).
- `main()` calls `generate_report(results, report_cfg, save_path=args.report_path)` immediately after `plot_benchmark()` and prints the save path.

### `tests/test_report.py` (new)

10 tests:
- `test_returns_string` — `generate_report(..., save_path='')` returns a `str`
- `test_file_created` — file is written when `save_path` is given
- `test_file_content_matches_return_value` — file content equals the return value
- `test_markdown_has_header` — output starts with `# Benchmark Report`
- `test_markdown_has_setup_section` — `## Setup` in output
- `test_markdown_has_results_section` — `## Results` in output
- `test_markdown_has_per_optimizer_section` — `## Per-Optimizer` in output
- `test_markdown_has_observations_section` — `## Key Observations` in output
- `test_empty_save_path_skips_file` — non-empty string returned; no file written
- `test_multiple_optimizers_rankings_present` — `Best final accuracy` and `Fastest` in output

Total tests: 286 (up from 276).

---

## Round 18 — Sophia, Prodigy, and Schedule-Free AdamW

### New files: `optimizers/sophia.py`, `optimizers/prodigy.py`, `optimizers/schedule_free.py`

**`Sophia`** (Liu et al., 2023)
- Second-order optimizer using a clipped Hutchinson diagonal Hessian estimate as a preconditioner.
- `requires_create_graph = True` class attribute — `train_one_epoch()` uses `autograd.grad(create_graph=True)` automatically.
- HVP computed every `update_freq=10` steps (at steps 1, 11, 21, …); falls back to `g²` when `create_graph` is unavailable.
- Update: `p -= lr * clip(m / h.clamp(min=1e-8), -ρ, ρ)` where `h` is the EMA Hessian diagonal.
- Default lr=1e-4, rho=0.04.

**`Prodigy`** (Mishchenko & Defazio, 2023)
- Parameter-free: automatically scales the step size without manual LR tuning.
- Maintains `d` (monotonically non-decreasing scale) and `A` (accumulator) in `param_groups` so they survive `state_dict()` / `load_state_dict()`.
- Per-parameter state: `x0` (initial weights), `s` (gradient accumulator), `exp_avg`, `exp_avg_sq`.
- Two-pass step: first pass updates `s` and computes `d_num`/`g_sq_sum`; second pass applies Adam with `eff_lr = d * lr`.
- Default lr=1.0 (treated as a global multiplier on `d`).

**`ScheduleFreeAdamW`** (Defazio et al., NeurIPS 2024)
- Replaces LR schedules with Polyak-Ruppert iterate averaging.
- Maintains `x` (Polyak average) in state; params are set to `y = (1-β₁)x + β₁z` at each step.
- `z` reconstructed from `y` and `x` each step: `z = (y - (1-β₁)x) / β₁`.
- `get_averaged_params()` method returns `x` tensors for final evaluation.
- Default lr=1e-3.

### `optimizers/__init__.py`
- Exports `Sophia`, `Prodigy`, `ScheduleFreeAdamW`.

### `train.py`
- `OPTIMIZER_REGISTRY` extended: `"sophia"`, `"prodigy"`, `"sf_adamw"` added (20 total).
- Import updated.

### `benchmark.py`
- `OPTIMIZER_REGISTRY` extended: Sophia (purple, lr=1e-4), Prodigy (crimson, lr=1.0), SF-AdamW (green, lr=1e-3). Total: 19 entries.

### `tests/test_optimizers.py`
- Added `TestSophia` (4 tests), `TestProdigy` (4 tests), `TestScheduleFreeAdamW` (4 tests).
- Total: 55 tests (up from 43).

Total tests: 276 (up from 258).

---

## Round 17 — AdaBelief, SignSGD, and AdaFactor

### New files: `optimizers/adabelief.py`, `optimizers/signsgd.py`, `optimizers/adafactor.py`

**`AdaBelief`** (Zhang et al., 2020)
- Adapts step size by how much the gradient deviates from its running average.
- Belief EMA: `s = β₂·s + (1-β₂)·(g-m)² + ε` — small when gradient matches momentum (large step), large when surprised (small step).
- Bias-corrected denominator: `sqrt(s/bc2) + ε`.
- Decoupled weight decay: `p *= (1 - lr * wd)` before the gradient step.
- Default lr=1e-3.

**`SignSGD`** (Bernstein et al., 2018)
- Pure sign update: every parameter moves `±lr` regardless of gradient magnitude.
- Optional momentum buffer (Signum variant): `direction = (β·buf + (1-β)·g).sign()`.
- No buffer stored when `β = 0` (plain sign mode).
- Default lr=1e-2.

**`AdaFactor`** (Shazeer & Stern, 2018)
- Memory-efficient: for 2-D+ tensors, factors the second-moment matrix into row and column marginals (`O(m+n)` storage instead of `O(mn)`).
- `_rho(step, beta2_decay)` — `min(1 - step^β₂_decay, 0.999)` gives annealing second moment.
- Factored reconstruction: `v_approx = row.unsqueeze(-1) * col.unsqueeze(-2) / row_norm`.
- RMS-based update clipping; optional first moment; supports `lr=None` for relative step.
- Default lr=1e-3.

### `optimizers/__init__.py`
- Exports `AdaBelief`, `SignSGD`, `AdaFactor`.

### `train.py`
- `OPTIMIZER_REGISTRY` extended: `"adabelief"`, `"signsgd"`, `"adafactor"` added (17 total at this point).
- Import updated.

### `benchmark.py`
- `OPTIMIZER_REGISTRY` extended: AdaBelief (grey, lr=0.001), SignSGD (light blue, lr=0.01), AdaFactor (light orange, lr=0.001). Total: 16 entries at this point.

### `tests/test_optimizers.py`
- Added `TestAdaBelief` (4 tests), `TestSignSGD` (4 tests), `TestAdaFactor` (5 tests).
- Total: 43 tests (up from 30).

Total tests: 258 (up from 245).

---

## Round 16 — LR Sweep in benchmark.py

### `benchmark.py`
- **`--lr` renamed to `--lrs`** (nargs="+", metavar="LR") — supports one value (global override, no suffix) or multiple values (LR sweep, adds `lr=<val>` suffix to series names). Single-value behavior is fully backward-compatible with the old `--lr`.
- **`run_benchmark()` signature** — `lr_override: float | None` → `lrs: list[float] | None`.
- **Inner loop restructured** — added `sweep_lr` flag and `n_lr_vals` counter; body iterates `for lr in lr_list` inside the optimizer loop.
- **Param groups fixed** — `run_benchmark()` now calls `make_param_groups(model, wd)` when `wd > 0`, matching the `train.py` pattern. Factory is called with `0.0` as weight_decay.
- **Series naming** — suffixes list built conditionally: `lr=<val>` added when sweeping LR, `wd=<val>` added when sweeping WD; combined when both sweep.
- **`main()` LR banner** — three-way display: global override / LR sweep / per-optimizer defaults.
- **`main()` colors** — `sweep_lr or sweep_wd` block uses `tab20` colormap for series colors; single-optimizer single-LR still uses registry colors.
- Added `make_param_groups` to the `from train import` line.

### `tests/test_benchmark.py` (new)
- 6 tests:
  - `test_lrs_none_uses_per_optimizer_default` — series name = bare optimizer name, no suffix
  - `test_single_lr_no_suffix` — single `lrs=[0.1]` produces no `lr=` suffix
  - `test_lrs_sweep_creates_two_series` — `lrs=[0.1, 0.01]` → 2 result entries
  - `test_lr_sweep_series_names_contain_lr` — result keys contain `"lr=0.1"` and `"lr=0.01"`
  - `test_lr_and_wd_sweep_combined_suffix` — `lrs=[0.1, 0.01]`, `weight_decays=[0.0, 1e-4]` → 4 results, all with both `lr=` and `wd=`
  - `test_wd_only_sweep_no_lr_suffix` — `lrs=None`, `weight_decays=[0.0, 1e-4]` → 2 results, none with `lr=`

Total tests: 245 (up from 239).

---

## Round 15 — Per-Parameter Group Weight Decay

### `train.py`
- **New helper `make_param_groups(model, weight_decay)`** — splits trainable parameters into two groups: `ndim >= 2` (weight matrices, conv filters) receive the specified `weight_decay`; `ndim < 2` (biases, LayerNorm and BatchNorm weights/biases) always receive `weight_decay=0.0`.
- **`build_optimizer()` signature change** — first positional argument changed from a parameter iterator (`params`) to `nn.Module` (`model`). When `weight_decay > 0`, the optimizer is constructed with per-group params from `make_param_groups()`; when `weight_decay == 0`, falls back to `model.parameters()` unchanged.
- Updated call site in `main()` accordingly.

### `tests/test_train.py`
- All existing `build_optimizer(name, model.parameters(), ...)` calls updated to `build_optimizer(name, model, ...)`.
- Added `make_param_groups` to imports.
- Added 5 new tests in `TestWeightDecay`:
  - `test_make_param_groups_two_groups` — returns exactly 2 dicts with `params` and `weight_decay` keys
  - `test_make_param_groups_no_overlap` — no parameter id appears in both groups
  - `test_make_param_groups_covers_all_params` — union of both groups equals all trainable parameter ids
  - `test_bias_params_have_zero_decay` — all `ndim < 2` params land in the no-decay group
  - `test_weight_matrices_have_nonzero_decay` — all `ndim >= 2` params land in the decay group

### `tests/test_optimizers.py`
- Updated the one call site that passed `model.parameters()` to `build_optimizer`.

Total tests: 239 (up from 234).

---

## Round 14 — LR Range Test (LRFinder)

### New file: `lr_finder.py`

`LRFinder` class — self-contained, no dependency on `train.py`.

- `__init__(model, optimizer, criterion, device, start_lr=1e-7, end_lr=10.0, num_iter=100, smooth_f=0.05, diverge_th=5.0)`
- `run(loader)` — ramps LR exponentially, records EMA-smoothed loss, stops early on divergence; model and optimizer state unconditionally restored via `finally` block; supports AdaHessian via `requires_create_graph` flag
- `suggestion()` — returns LR at the steepest negative loss gradient
- `plot(save_path=None)` — log-scale loss vs LR figure with red dashed vertical line at suggested LR; returns suggestion

### `train.py`
- Added `from lr_finder import LRFinder` import.
- Three new CLI flags: `--find-lr` (store_true), `--find-lr-iters` (default 100), `--find-lr-plot` (default None).
- Pre-training block in `main()`: when `--find-lr` is set, constructs `LRFinder`, runs it, prints suggested LR, optionally saves the plot — all before the training loop starts.

### `tests/test_lr_finder.py` (new)
- 7 tests across 4 classes:
  - `TestLRFinderBasic` — history keys, length match, monotone LR schedule
  - `TestLRFinderStateRestoration` — model weights and optimizer LR restored after run
  - `TestLRFinderSuggestion` — suggestion is a float within `[start_lr, end_lr]`
  - `TestLRFinderAdaHessian` — completes with AdaHessian; all losses finite

Total tests: 234 (up from 227).

---

## Round 13 — Schedulers, Early Stopping, Gradient Clipping, Checkpoints, and Log-Replay Plotting

### `train.py`
- **Scheduler registry** — `SCHEDULER_REGISTRY` with four entries: `none`, `cosine` (CosineAnnealingLR), `step` (StepLR, gamma=0.1 every epochs//3 steps), `warmup_cosine` (SequentialLR: LinearLR warmup + CosineAnnealingLR decay).
- **`EarlyStopping` class** — `patience=0` disables (default). `step(val_loss, model)` saves best-epoch weights and returns True when patience is exhausted. `restore(model, device)` loads best-epoch weights back.
- **`run_training()` updates** — accepts `scheduler`, `patience`, `min_delta`, `max_grad_norm` args; history gains `"learning_rates"` (LR at epoch start) and `"early_stopped_epoch"` (int or None) and `"grad_norm_before_clip"` (pre-clip global norm per epoch).
- **`train_one_epoch()` updates** — accepts `max_grad_norm`; accumulates pre-clip per-param norms; calls `clip_grad_norm_` when set; returns both `grad_norm_before_clip` and `grad_norm_global`.
- **New CLI flags** — `--scheduler`, `--warmup-epochs`, `--patience`, `--min-delta`, `--max-grad-norm`, `--weight-decay`, `--seed`, `--checkpoint-dir`.
- **`Visualizer.update()`** — accepts `learning_rate` kwarg; overlays LR trace on the Loss vs Epoch panel.

### `benchmark.py`
- `run_benchmark()` accepts `scheduler_name`, `warmup_epochs`, `patience`, `min_delta`, `max_grad_norm`, `weight_decays`, `seed`, `checkpoint_dir`.
- New CLI flags mirroring `train.py`: `--scheduler`, `--warmup-epochs`, `--patience`, `--min-delta`, `--max-grad-norm`, `--weight-decays` (nargs="+", allows weight-decay sweep), `--seed`, `--checkpoint-dir`.

### `visualizer.py`
- `plot_benchmark()` gains a fifth column: **LR vs Epoch**.

### `tests/test_train.py`
- Added `TestSchedulers` (4 tests), `TestEarlyStopping` (5 tests), `TestGradientClipping` (4 tests), `TestSeed` (3 tests).
- Total tests: 68 (up from ~52 after previous additions).

### `tests/test_checkpoints.py` (new)
- 5 tests: checkpoint directory created on run; `best.pt` and `final.pt` written; restored model produces same outputs; partial-epoch best tracking.

### `tests/test_plot_from_logs.py` (new)
- 9 tests: log directory discovery; CSV parsing; per-run plot generation; multi-run overlay; edge cases (empty log dir, missing keys).

Total tests: 227 (up from 190).

---

## Round 12 — Muon, Adan, and AdaHessian Optimizers

### New files: `optimizers/muon.py`, `optimizers/adan.py`, `optimizers/adahessian.py`

**`Muon`** (Kosson et al., 2024)
- Nesterov momentum with Gram-Schmidt orthogonalisation of the update matrix before applying it.
- Per-parameter orthogonalisation via `torch.linalg.svd`; 1-D params (bias, BatchNorm) fall back to plain SGD.
- Default lr=0.02; weight_decay applied to non-1-D params only.

**`Adan`** (Adaptive Nesterov Momentum, Xie et al., 2023)
- Maintains three EMA terms: gradient (`m1`), gradient difference (`m2`), squared gradient difference (`m3`).
- Update direction incorporates Nesterov look-ahead from the difference term; element-wise preconditioned by `m3`.
- Default lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8.

**`AdaHessian`** (Yao et al., 2021)
- Estimates the Hessian diagonal using the Hutchinson estimator (random ±1 vectors, `create_graph=True` backward).
- Per-parameter EMA Hessian diagonal used as preconditioner (similar to RMSprop but second-order).
- `requires_create_graph: bool = True` class attribute — detected by `train_one_epoch()` and `LRFinder.run()` to use `autograd.grad(create_graph=True)` instead of `.backward()`.
- Default lr=0.1.

### `optimizers/__init__.py`
- Exports `Muon`, `Adan`, `AdaHessian`.

### `train.py`
- `OPTIMIZER_REGISTRY` extended: `"muon"`, `"adan"`, `"adahessian"` entries added (14 total).
- `train_one_epoch()` checks `getattr(optimizer, "requires_create_graph", False)` to use `autograd.grad` path.

### `benchmark.py`
- `OPTIMIZER_REGISTRY` extended: `Muon` (magenta, lr=0.02), `Adan` (olive, lr=0.001), `AdaHessian` (gold, lr=0.1). Total: 13 entries.

### `tests/test_optimizers.py`
- Registry completeness test updated to expect 14 entries.
- 19 new tests added for Muon, Adan, and AdaHessian (one-step finite-weight; optimizer-specific state; AdaHessian `create_graph` path).
- Total: 37 tests (up from 18).

Total tests: 190 (up from 140).

---

## Round 11 — Training Logger and Plot Directory Enforcement

### New file: `logger.py`

`TrainingLogger` class for structured logging of training sessions.

- `__init__(log_dir)` — creates a timestamped subdirectory `logs/<YYYY-MM-DD_HH-MM-SS>/` at session start.
- `log_run(config, history)` — writes one `<dataset>_<model>_<optimizer>.log` per run containing:
  - Header: run name, timestamp, all config key-value pairs.
  - `[epoch-wise]` CSV section: `epoch, train_loss, train_acc, test_loss, test_acc, time_elapsed_s`.
  - `[batch-wise]` CSV section: `step, loss` for every training batch across all epochs.
- `close()` — writes `run_summary.log` with session start/end times, elapsed seconds, total run count, and a formatted results table (train loss, train acc%, test loss, test acc% per run).

### `train.py`
- Default `--save-plot` changed from `training_curves.png` → `plots/training_curves.png`.
- New `--log-dir` flag (default: `logs/`).
- `main()` now collects a `history` dict (train/test loss and acc per epoch, cumulative `time_elapsed`, all `step_losses`) during the epoch loop, then calls `TrainingLogger.log_run()` and `close()` after training.

### `benchmark.py`
- Default `--save-plot` changed from `benchmark.png` → `plots/benchmark.png`.
- New `--log-dir` flag (default: `logs/`).
- `run_benchmark()` accepts an optional `logger: TrainingLogger` argument; calls `log_run()` after each (dataset, model, optimizer) run.
- `main()` creates a `TrainingLogger`, passes it to `run_benchmark()`, and calls `close()` after all runs.

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
