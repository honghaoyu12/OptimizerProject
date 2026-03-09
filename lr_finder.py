"""Learning Rate Range Test (Leslie Smith, 2018).

Ramps the learning rate exponentially over a fixed number of mini-batches,
records the smoothed loss, and identifies the LR where loss falls fastest.
"""

import copy
import itertools
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class LRFinder:
    """Runs a learning-rate range test on a model/optimizer pair.

    Parameters
    ----------
    model       : nn.Module
    optimizer   : torch.optim.Optimizer
    criterion   : callable(output, target) -> scalar loss
    device      : torch.device
    start_lr    : float  — first LR tried (default 1e-7)
    end_lr      : float  — last LR tried  (default 10.0)
    num_iter    : int    — number of mini-batch steps (default 100)
    smooth_f    : float  — EMA factor for loss smoothing (default 0.05)
    diverge_th  : float  — stop if loss > diverge_th * best_loss (default 5.0)
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.smooth_f = smooth_f
        self.diverge_th = diverge_th
        self.history: dict = {"lrs": [], "losses": []}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, loader) -> dict:
        """Run the LR range test on *loader* and return ``self.history``.

        The model and optimizer are restored to their original states
        unconditionally (even if an exception is raised).
        """
        # Snapshot original state before touching anything
        _opt_state = copy.deepcopy(self.optimizer.state_dict())
        _mdl_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        try:
            self.history = {"lrs": [], "losses": []}

            # Set starting LR
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.start_lr

            # Multiplicative step factor
            if self.num_iter > 1:
                q = (self.end_lr / self.start_lr) ** (1.0 / (self.num_iter - 1))
            else:
                q = 1.0

            smoothed = None
            best_loss = math.inf
            data_iter = itertools.cycle(loader)

            for step in range(self.num_iter):
                current_lr = self.optimizer.param_groups[0]["lr"]

                x, y = next(data_iter)
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)

                # Support optimizers that need create_graph (e.g. AdaHessian)
                if getattr(self.optimizer, "requires_create_graph", False):
                    trainable = [p for p in self.model.parameters() if p.requires_grad]
                    grads = torch.autograd.grad(loss, trainable, create_graph=True)
                    for p, g in zip(trainable, grads):
                        p.grad = g
                else:
                    loss.backward()

                self.optimizer.step()

                # EMA smoothing
                loss_val = loss.item()
                if step == 0:
                    smoothed = loss_val
                else:
                    smoothed = self.smooth_f * loss_val + (1.0 - self.smooth_f) * smoothed

                best_loss = min(best_loss, smoothed)

                self.history["lrs"].append(current_lr)
                self.history["losses"].append(smoothed)

                # Divergence check
                if smoothed > self.diverge_th * best_loss:
                    break

                # Advance LR for next step
                for pg in self.optimizer.param_groups:
                    pg["lr"] *= q

        finally:
            # Always restore original model and optimizer state
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in _mdl_state.items()}
            )
            self.optimizer.load_state_dict(_opt_state)

        return self.history

    def suggestion(self) -> float:
        """Return the LR at the point of steepest loss decrease."""
        losses = self.history["losses"]
        if len(losses) < 2:
            return float("nan")
        diffs = [losses[i + 1] - losses[i] for i in range(len(losses) - 1)]
        return self.history["lrs"][diffs.index(min(diffs))]

    def plot(self, save_path: str = None) -> float:
        """Plot loss vs LR (log scale) and return the suggested LR.

        Parameters
        ----------
        save_path : str or None
            If provided, the figure is saved to this path (directories are
            created as needed).  Pass ``None`` to skip saving.
        """
        sugg = self.suggestion()

        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.plot(self.history["lrs"], self.history["losses"], label="Smoothed loss")
        if not math.isnan(sugg):
            ax.axvline(sugg, color="red", linestyle="--",
                       label=f"Suggested: {sugg:.2e}")
        ax.set_xlabel("Learning rate (log scale)")
        ax.set_ylabel("Smoothed loss")
        ax.set_title("LR Range Test")
        ax.legend()

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path)

        plt.close(fig)
        return sugg
