# Project Architecture

## Goal

OptimizerProject is a testing ground for custom numerical optimizers. The aim is to make it
easy to implement a new optimizer, plug it in, train it on a real dataset, and immediately
see diagnostic information — gradient norms, weight norms, loss landscape sharpness,
learning rate schedules — side-by-side with reference optimizers. Every component exists
to serve that core loop.

---

## High-Level Picture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          USER ENTRY POINTS                                   │
│                                                                              │
│   python train.py [flags]          python benchmark.py [flags]              │
│   (single run, one optimizer)      (multi-run, all combinations)            │
└───────────────┬──────────────────────────────────┬───────────────────────────┘
                │                                  │
                ▼                                  ▼
┌──────────────────────────┐       ┌─────────────────────────────┐
│        train.py          │       │        benchmark.py          │
│  ─────────────────────   │       │  ─────────────────────────   │
│  build_model()           │◄──────│  imports run_training()      │
│  build_optimizer()       │       │  imports get_dataloaders()   │
│  get_dataloaders()       │       │  imports DATASET_INFO        │
│  train_one_epoch()       │       │  builds its own optimizer    │
│  evaluate()              │       │  instances per combo         │
│  run_training()          │       └──────────────┬──────────────┘
│  EarlyStopping           │                      │
│  make_param_groups()     │                      │
│  save_checkpoint()       │                      │
└──────┬───────────────────┘                      │
       │                                           │
       │  imports from                             │  imports from
       ▼                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SUPPORT LAYER                                       │
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐ │
│  │   model.py   │  │  metrics.py  │  │  logger.py   │  │  visualizer.py  │ │
│  │ ──────────── │  │ ──────────── │  │ ──────────── │  │ ─────────────── │ │
│  │ MLP          │  │ hessian_     │  │ Training-    │  │ Visualizer      │ │
│  │ ResNet18     │  │   trace()    │  │   Logger     │  │ (live 3×2       │ │
│  │ ViT          │  │ sharpness()  │  │ log_run()    │  │  dashboard)     │ │
│  └──────────────┘  └──────────────┘  │ close()      │  │ plot_benchmark()│ │
│                                       └──────────────┘  └─────────────────┘ │
│  ┌────────────────────────┐  ┌───────────────────────────────────────────┐  │
│  │   synthetic_datasets   │  │              optimizers/                  │  │
│  │ ─────────────────────  │  │ ────────────────────────────────────────  │  │
│  │ make_illcond_loaders   │  │ base.py → BaseOptimizer                   │  │
│  │ make_sparse_loaders    │  │ sgd.py  → VanillaSGD (reference)          │  │
│  │ make_noisy_grad_loaders│  │ lion.py → Lion                            │  │
│  │ make_manifold_loaders  │  │ lamb.py → LAMB                            │  │
│  │ make_saddle_loaders    │  │ shampoo.py → Shampoo                      │  │
│  │ SYNTHETIC_LOADERS dict │  │ muon.py → Muon                            │  │
│  └────────────────────────┘  │ adan.py → Adan                            │  │
│                               │ adahessian.py → AdaHessian                │  │
│                               │ adabelief.py → AdaBelief                  │  │
│                               │ signsgd.py → SignSGD                      │  │
│                               │ adafactor.py → AdaFactor                  │  │
│                               │ sophia.py → Sophia                        │  │
│                               │ prodigy.py → Prodigy                      │  │
│                               │ schedule_free.py → ScheduleFreeAdamW      │  │
│  ┌────────────────────────┐  └───────────────────────────────────────────┘  │
│  │     lr_finder.py       │                                                  │
│  │ ─────────────────────  │                                                  │
│  │ LRFinder               │                                                  │
│  │ run() / suggestion()   │                                                  │
│  │ plot()                 │                                                  │
│  └────────────────────────┘                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### Entry Points

#### `train.py`
The primary script. Runs a single training job from the command line.

What it does in `main()`:
1. Parse CLI flags
2. `build_model()` — construct MLP, ResNet-18, or ViT
3. `get_dataloaders()` — load a torchvision or synthetic dataset
4. `build_optimizer()` — construct the chosen optimizer with per-parameter-group weight decay
5. Build scheduler from `SCHEDULER_REGISTRY`
6. Optionally run `LRFinder` and print suggested LR (`--find-lr`)
7. Create a `Visualizer` for the live dashboard
8. `run_training()` — the main loop: train, evaluate, log, visualize, optionally checkpoint
9. `TrainingLogger.log_run()` + `close()` — persist all metrics to disk

It also exports stable utilities (`DATASET_INFO`, `OPTIMIZER_REGISTRY`, `run_training`,
`get_dataloaders`, etc.) so that `benchmark.py` can reuse them without duplication.

#### `benchmark.py`
Interactive multi-run comparison tool. Prompts the user to choose datasets, models, and
optimizers, then runs every combination and produces a single comparison plot.

What it does in `main()`:
1. Parse CLI flags (shared hyperparameters for all runs)
2. Interactive `prompt_multiselect()` menus for datasets, models, and optimizers
3. `run_benchmark()` — nested loop over all combinations; inner seed loop (controlled by `--num-seeds`) rebuilds model + optimizer fresh each seed and calls `run_training()`; `_aggregate_histories()` merges N seed histories into mean ± std when `num_seeds > 1`
4. `plot_benchmark()` — render a multi-panel comparison figure (one row per dataset/model pair, one line per optimizer, five metric columns); draws `fill_between` error bands when `_std` keys are present
5. `generate_report()` — write a Markdown summary report to `--report-path`; shows ± std when `test_acc_std` is present

`benchmark.py` deliberately has its own `OPTIMIZER_REGISTRY`, `MODEL_REGISTRY`, and
`DATASET_REGISTRY` so it can control colors, display names, and default LRs independently
of `train.py`.

---

### Models — `model.py`

Three architectures, all adapted for small-image classification:

| Class | Architecture | Key adaptation |
|---|---|---|
| `MLP` | Linear → ReLU stacks | `Flatten` on input; width/depth configurable |
| `ResNet18` | torchvision ResNet-18 | 3×3 conv1 + stride-1 + Identity maxpool for 28–64 px images |
| `ViT` | Vision Transformer | Pre-LN TransformerEncoder; patch_size=4; learnable CLS token |

All three are registered in `train.py`'s `build_model()` and `benchmark.py`'s `MODEL_REGISTRY`.

**Contribution to the goal:** Provides the models that the optimizers actually train. Three
architecturally distinct networks stress-test optimizers across different gradient landscapes
(dense linear, convolutional, attention-based).

---

### Optimizers — `optimizers/`

```
optimizers/
├── base.py        ← BaseOptimizer extends torch.optim.Optimizer
│                    subclasses only need to implement step()
├── sgd.py         ← VanillaSGD — plain SGD; reference baseline
├── adam.py        ← Adam — bias-corrected adaptive moments (custom, replaces torch.optim.Adam)
├── adamw.py       ← AdamW — Adam with decoupled weight decay (custom, replaces torch.optim.AdamW)
├── nadam.py       ← NAdam — Adam + Nesterov lookahead with momentum schedule (custom)
├── radam.py       ← RAdam — Rectified Adam; automatic warm-up via SMA variance check (custom)
├── adagrad.py     ← Adagrad — cumulative squared gradient scaling (custom, replaces torch.optim.Adagrad)
├── sgd_momentum.py← SGDMomentum — velocity buffer + optional Nesterov (custom, replaces torch.optim.SGD)
├── rmsprop.py     ← RMSprop — EMA squared gradient; centred + momentum variants (custom)
├── lion.py        ← Lion (Chen et al. 2023); sign-based momentum updates
├── lamb.py        ← LAMB; per-layer trust ratio, scales step by ‖θ‖/‖u‖
├── shampoo.py     ← Shampoo; Kronecker-factored preconditioners (CPU eigendecomp)
├── muon.py        ← Muon; Nesterov + orthogonalization of the update matrix
├── adan.py        ← Adan; adaptive Nesterov momentum
├── adahessian.py  ← AdaHessian; Hessian diagonal approximation (needs create_graph=True)
├── adabelief.py   ← AdaBelief; belief EMA adapts step size by gradient surprise
├── signsgd.py     ← SignSGD; pure sign updates with optional Signum momentum
├── adafactor.py   ← AdaFactor; factored second moment (O(m+n) memory for 2-D params)
├── sophia.py      ← Sophia; Hutchinson Hessian clipping (needs create_graph=True)
├── prodigy.py     ← Prodigy; parameter-free auto-scaling via inner product estimates
└── schedule_free.py ← ScheduleFreeAdamW; Polyak averaging replaces LR schedule
```

All are registered in `train.py`'s `OPTIMIZER_REGISTRY` (20 entries — all custom implementations) and available in `benchmark.py`'s registry (19 entries with display colors). No optimizer delegates to `torch.optim.*` any longer.

**Contribution to the goal:** The primary subject under study. Adding a new optimizer means
implementing one file extending `BaseOptimizer`, importing it in `train.py`, and adding one
line to `OPTIMIZER_REGISTRY`. All diagnostics, logging, and visualization then work for free.

---

### Training Engine — `train.py` internal functions

| Function | Role |
|---|---|
| `make_param_groups(model, wd)` | Splits parameters: `ndim≥2` → decay, `ndim<2` → no decay |
| `build_optimizer(name, model, lr, wd)` | Constructs optimizer; uses param groups when `wd>0` |
| `train_one_epoch(...)` | One pass over the training loader; tracks per-layer grad norms, supports gradient clipping; returns loss, accuracy, grad norm metrics |
| `evaluate(...)` | One pass over the test loader; returns loss and accuracy |
| `EarlyStopping` | Patience-based; saves best-epoch weights, restores them on stop |
| `run_training(...)` | Outer epoch loop; calls `train_one_epoch` + `evaluate`; handles scheduler step, early stopping, checkpointing, Hessian/sharpness diagnostics, LR recording |
| `save_checkpoint(...)` | Saves `best.pt` / `final.pt` with full state |

---

### Diagnostics — `metrics.py`

| Function | What it computes | When it runs |
|---|---|---|
| `compute_hessian_trace()` | Hutchinson trace estimator (Rademacher probe vectors) | Every epoch when `--hessian` is set |
| `compute_sharpness()` | SAM-style: max loss increase within an ε-ball around current weights | Every epoch when `--hessian` is set |

Both operate on a fixed mini-batch each epoch so the values are comparable across epochs.
The Hessian trace measures curvature; sharpness measures sensitivity to perturbation.
Together they let you compare whether two optimizers converge to flat vs sharp minima —
a key indicator of generalization quality.

**Contribution to the goal:** Provides the "why" behind accuracy differences. A flat minimum
(low trace, low sharpness) generalizes better; these metrics make that visible.

---

### LR Range Test — `lr_finder.py`

`LRFinder` implements Leslie Smith's Learning Rate Range Test (2018):

1. Temporarily ramp the LR exponentially from `start_lr` to `end_lr` over `num_iter` mini-batches
2. Track EMA-smoothed loss at each step
3. Stop early if loss diverges (`diverge_th × best_loss`)
4. Restore model weights and optimizer state unconditionally via `finally` block
5. `suggestion()` — return the LR at the steepest descent point of the loss curve

**Contribution to the goal:** Removes guesswork from LR selection. Before starting a long
experiment, users can run `--find-lr` to identify where the loss falls fastest for their
chosen optimizer/dataset/model combination.

---

### Datasets — `synthetic_datasets.py` + torchvision

#### Torchvision datasets (via `train.py` `get_dataloaders()`)
| Dataset | Size | Classes | Use case |
|---|---|---|---|
| MNIST | 28×28 gray | 10 | Basic digit classification |
| FashionMNIST | 28×28 gray | 10 | Slightly harder image classification |
| CIFAR-10 | 32×32 RGB | 10 | Standard benchmark with augmentation |
| CIFAR-100 | 32×32 RGB | 100 | Fine-grained, harder than CIFAR-10 |
| Tiny ImageNet | 64×64 RGB | 200 | Near-real-world scale (~236 MB) |

#### Synthetic tabular datasets (MLP only)
| Dataset | Design purpose |
|---|---|
| `illcond` | 64-D Gaussian with κ=1000 — tests curvature handling |
| `sparse` | 100-D, 5 informative features — tests coordinate adaptivity |
| `noisy_grad` | 64-D, 30% label noise — tests stochastic robustness |
| `manifold` | 64-D make_moons + noise — tests nonconvex landscape navigation |
| `saddle` | 64-D bimodal positives — tests saddle-point escape |

**Contribution to the goal:** Each dataset creates a qualitatively different optimization
problem. An optimizer that does well on `illcond` handles curvature; one that does well on
`noisy_grad` is stochastically robust. The suite turns "does my optimizer work?" into a
structured, multi-dimensional answer.

---

### Logging — `logger.py`

```
logs/
└── 2026-03-09_14-30-00/          ← one folder per session
    ├── run_summary.log            ← human-readable aggregate (all runs)
    ├── run_summary.csv            ← one row per run (machine-readable)
    ├── run_summary.json           ← full structured export with histories
    ├── mnist_mlp_adam.log         ← per-run detail (epoch + batch losses)
    └── mnist_mlp_adam_epochs.csv  ← clean epoch-level CSV
```

`TrainingLogger` is created at the start of a session. Each call to `log_run(config, history)`
writes one pair of files. `close()` aggregates everything into the three summary files.

**Contribution to the goal:** Provides a persistent, structured record of every experiment.
The JSON summary enables programmatic analysis and plot replay (`plot_from_logs.py`).

---

### Visualization — `visualizer.py`

#### Live dashboard (`Visualizer` class)
A 3×2 matplotlib figure updated each epoch:

```
┌──────────────────┬──────────────────┐
│  Loss vs Epoch   │  Acc vs Epoch    │
│  + LR trace (→)  │                  │
├──────────────────┼──────────────────┤
│  Step Loss       │  Grad Norm       │
│  (batch-level)   │  (per layer)     │
├──────────────────┼──────────────────┤
│  Weight Norms    │  Hessian / Sharp │
│  (per layer)     │  (when --hessian)│
└──────────────────┴──────────────────┘
```

#### Benchmark comparison (`plot_benchmark()`)
N rows × 5 columns (one row per dataset/model combination, one line per optimizer):

```
Train Loss | Test Loss | Train Acc% | Test Acc% | LR vs Epoch
```

**Contribution to the goal:** Makes optimizer differences immediately visible. Watching grad
norms diverge, weight norms collapse, or sharpness spike during training gives instant
feedback on optimizer stability — far more informative than final accuracy alone.

---

## Data Flow: Single Training Run

```
CLI args
    │
    ▼
build_model()  ──────────────────────────────────► model.py
get_dataloaders()  ───── torchvision or ──────────► synthetic_datasets.py
build_optimizer()  ──── make_param_groups() ──────► optimizers/
SCHEDULER_REGISTRY[name] ────────────────────────► (torch.optim.lr_scheduler)
    │
    ▼
[optional] LRFinder.run()  ──────────────────────► lr_finder.py
    │
    ▼
run_training()
    │
    ├─ each epoch:
    │     train_one_epoch()  ─── gradient clipping, per-layer norms
    │     evaluate()
    │     scheduler.step()
    │     [optional] compute_hessian_trace()  ────► metrics.py
    │     [optional] compute_sharpness()  ────────► metrics.py
    │     Visualizer.update()  ───────────────────► visualizer.py
    │     [optional] save_checkpoint()
    │     [optional] EarlyStopping.step()
    │
    └─ TrainingLogger.log_run()  ────────────────► logger.py
       TrainingLogger.close()
```

---

## Data Flow: Benchmark Run

```
CLI args + interactive prompts
    │
    ▼
for each (dataset, model, optimizer, lr, wd) combination:
    │
    ├─ get_dataloaders()  ────────────────────────► synthetic_datasets.py / torchvision
    │
    └─ for seed_idx in range(num_seeds):           ← num_seeds=1 by default
          set_seed(base_seed + seed_idx)
          model factory()  ────────────────────────► model.py
          optimizer factory()  ────────────────────► optimizers/
          run_training()  ─────────────────────────── (from train.py)
          TrainingLogger.log_run()  ───────────────► logger.py
       _aggregate_histories(seed_histories)        ← skipped when num_seeds=1
    │
    ▼
plot_benchmark(results, ...)  ───────────────────► visualizer.py  (fill_between bands when _std keys present)
generate_report(results, cfg, path)  ───────────► report.py  (± std when test_acc_std present)
TrainingLogger.close()  ────────────────────────► logger.py
```

---

## Import Graph

```
train.py
 ├── model.py
 ├── metrics.py
 ├── logger.py
 ├── visualizer.py
 ├── lr_finder.py
 ├── synthetic_datasets.py
 └── optimizers/__init__.py
       ├── base.py
       ├── sgd.py
       ├── lion.py
       ├── lamb.py
       ├── shampoo.py
       ├── muon.py
       ├── adan.py
       ├── adahessian.py
       ├── adabelief.py
       ├── signsgd.py
       ├── adafactor.py
       ├── sophia.py
       ├── prodigy.py
       └── schedule_free.py

benchmark.py
 ├── train.py  (run_training, get_dataloaders, DATASET_INFO, SCHEDULER_REGISTRY,
 │              linear_layer_names, save_checkpoint, set_seed)
 ├── model.py
 ├── logger.py
 ├── report.py  (generate_report)
 ├── visualizer.py  (plot_benchmark)
 └── optimizers/__init__.py

plot_from_logs.py
 ├── benchmark.py  (DATASET_REGISTRY, OPTIMIZER_REGISTRY)
 └── visualizer.py  (plot_benchmark)

visualizer.py      ── no internal imports (matplotlib, numpy only)
logger.py          ── no internal imports (stdlib only)
metrics.py         ── no internal imports (torch only)
lr_finder.py       ── no internal imports (torch, matplotlib)
model.py           ── no internal imports (torch, torchvision.models)
synthetic_datasets ── no internal imports (numpy, torch, sklearn)
optimizers/*.py    ── only optimizers/base.py (+ torch)
```

---

## Extension Points

### Adding a new optimizer
1. Create `optimizers/myopt.py` extending `BaseOptimizer`; implement `step()`
2. Export it in `optimizers/__init__.py`
3. Add one entry to `OPTIMIZER_REGISTRY` in `train.py`
4. Optionally add to `benchmark.py`'s registry with a color and default LR

All diagnostics, logging, visualization, and tests work automatically.

### Adding a new dataset
1. Add an entry to `DATASET_INFO` in `train.py` with `in_channels`, `input_size`, `num_classes`
2. Add loading logic in `get_dataloaders()` (or add a generator to `synthetic_datasets.py` and register it in `SYNTHETIC_LOADERS`)
3. Add to `benchmark.py`'s `DATASET_REGISTRY` if it should appear in interactive benchmarks

### Adding a new model
1. Implement the class in `model.py`
2. Register it in `build_model()` in `train.py`
3. Add to `benchmark.py`'s `MODEL_REGISTRY`

---

## Test Coverage

| File | Tests | What is verified |
|---|---|---|
| `test_train.py` | 126 | DATASET_INFO, build_model, train_one_epoch, run_training, EarlyStopping, gradient clipping, AMP, compile, checkpoint resume, scheduler integration, weight decay, gradient SNR, ECE, per-class accuracy, instability, step size, sharpness, grad cosine sim, grad conflict |
| `test_new_optimizers.py` | 54 | Adam, AdamW, NAdam, RAdam, Adagrad, SGDMomentum, RMSprop: state init, step counter, finite weights, weight decay, optimizer-specific invariants, numerical agreement with torch.optim |
| `test_optimizers.py` | 49 | VanillaSGD, Lion, LAMB, Shampoo, Muon, Adan, AdaHessian, AdaBelief, SignSGD, AdaFactor, Sophia, Prodigy, ScheduleFreeAdamW: factory, one-step finite weights, state dict, optimizer-specific state |
| `test_visualizer.py` | 86 | All plot functions including sharpness, grad cosine sim, grad conflict, LR sensitivity curves |
| `test_logger.py` | 32 | log_run(), close(), CSV/JSON schema, multi-run aggregation |
| `test_benchmark.py` | 44 | run_benchmark() LR/WD sweep, multi-seed aggregation, sharpness, grad_cosine_sim, grad_conflict, lr_sensitivity_curves |
| `test_report.py` | 31 | generate_report(): content, sections, ± std, EMA/SWA/ECE columns |
| `test_synthetic.py` | 16 | SYNTHETIC_LOADERS registry, tensor shapes, label ranges, NaN checks |
| `test_models.py` | 17 | MLP, ResNet18, ViT: forward pass, param counts, device transfer |
| `test_metrics.py` | 9 | Hutchinson trace, sharpness (epsilon, directions, weight restoration) |
| `test_plot_from_logs.py` | 9 | load_session(), reconstruct_results(), series naming, weight-decay sweeps |
| `test_lr_finder.py` | 7 | history keys/length, monotone LR, state restoration, suggestion range |
| `test_checkpoints.py` | 5 | save_checkpoint(), load, config keys, best vs final checkpoint logic |
| **Total** | **564** | |

---

## How Everything Serves the Goal

```
GOAL: Make it easy to implement a new optimizer, plug it in, and immediately
      understand how it behaves relative to existing optimizers.

├── optimizers/         ← WHERE you put new optimizers (one file, one class)
│
├── train.py            ← HOW you run them (single job, any dataset/model)
├── benchmark.py        ← HOW you compare them (interactive, all combos at once)
│
├── model.py            ← WHAT they train on (three architecturally distinct models)
├── synthetic_datasets  ← WHAT stress-tests they face (5 targeted problem types)
│
├── metrics.py          ← WHY differences occur (curvature and sharpness)
├── visualizer.py       ← WHERE you see it (live dashboard + comparison plots)
├── lr_finder.py        ← HOW you choose the LR (range test, no guessing)
│
├── logger.py           ← HOW you keep it (structured logs for every run)
└── report.py           ← HOW you share it (Markdown report after benchmark runs)
```
