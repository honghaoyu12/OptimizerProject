# Learning Rate Tools — A Quick Guide

This project has two complementary learning rate features. They solve different problems and are designed to be used together.

---

## LR Finder (`--find-lr`)

**What it does:** Helps you pick a good *starting* learning rate before training begins.

**How it works:**
1. Runs a short diagnostic (~100 mini-batches) before training.
2. Ramps the learning rate exponentially from very small (1e-7) to very large (10.0).
3. Records the smoothed loss at each step.
4. Reports the LR where loss was falling the fastest — that is the suggested starting point.
5. Fully restores the model and optimizer to their original state, so training starts clean.

**When to use it:** Any time you are unsure what LR to set — especially when trying a new optimizer, dataset, or model combination where the default LR may not be ideal.

```bash
# Print the suggested LR and save a plot
python train.py --find-lr --find-lr-iters 100 --find-lr-plot plots/lr_range.png --no-plot
```

---

## LR Scheduler (`--scheduler`)

**What it does:** Controls how the learning rate *changes over the course of training*, starting from whatever LR you set with `--lr`.

**Available schedules:**

| Key | Behaviour |
|---|---|
| `none` | LR stays constant throughout (default) |
| `cosine` | Smoothly decays from initial LR to 0 following a cosine curve |
| `step` | Drops by 10× every `epochs / 3` epochs |
| `warmup_cosine` | Linearly ramps up for `--warmup-epochs` epochs, then cosine decays |

**When to use it:** When you want the LR to shrink as training converges, instead of staying fixed. Cosine decay is a good default for most runs.

```bash
# Cosine decay from lr=3e-3 over 20 epochs
python train.py --lr 3e-3 --scheduler cosine --epochs 20 --no-plot
```

---

## Key Difference

| | LR Finder | LR Scheduler |
|---|---|---|
| **When it runs** | Before training (one-time diagnostic) | During training (every epoch) |
| **What it controls** | Which LR to start at | How LR evolves from that starting point |
| **Output** | A suggested LR value (and optional plot) | Modified LR values fed to the optimizer each epoch |
| **Effect on training** | None — model is fully restored afterward | Directly affects all weight updates |

---

## Using Them Together

They are complementary and can be combined in a single command:

```bash
python train.py \
  --find-lr --find-lr-iters 100 --find-lr-plot plots/lr_range.png \
  --lr 3e-3 --scheduler cosine --epochs 20 \
  --no-plot
```

**What happens:**
1. The LR finder runs for 100 batches and prints the suggested LR.
2. Training starts from `--lr 3e-3` (you can adjust this based on the suggestion).
3. The cosine scheduler smoothly decays that LR to 0 over the 20 epochs.

**Recommended workflow:**
1. Run `--find-lr` once to get a suggested starting LR.
2. Set `--lr` to that value (or the nearest round number).
3. Add `--scheduler cosine` (or `warmup_cosine`) to decay it during training.
4. On subsequent runs you can drop `--find-lr` since the good LR is now known.
