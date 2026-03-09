"""Tests for lr_finder.LRFinder."""

import math
import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lr_finder import LRFinder
from optimizers import AdaHessian


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _simple_model():
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))


def _simple_loader(n=64, batch_size=16):
    x = torch.randn(n, 8)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _make_finder(model, optimizer, num_iter=20, diverge_th=1e9):
    return LRFinder(
        model, optimizer, nn.CrossEntropyLoss(), torch.device("cpu"),
        start_lr=1e-7, end_lr=1.0,
        num_iter=num_iter, diverge_th=diverge_th,
    )


# ---------------------------------------------------------------------------
# TestLRFinderBasic
# ---------------------------------------------------------------------------

class TestLRFinderBasic:
    def test_run_returns_history_keys(self):
        model = _simple_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        finder = _make_finder(model, opt)
        history = finder.run(_simple_loader())
        assert set(history.keys()) == {"lrs", "losses"}

    def test_history_lengths_match(self):
        model = _simple_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        num_iter = 15
        finder = _make_finder(model, opt, num_iter=num_iter)
        history = finder.run(_simple_loader())
        assert len(history["lrs"]) == len(history["losses"])
        assert 1 <= len(history["lrs"]) <= num_iter

    def test_lr_schedule_monotone_increasing(self):
        """LRs must be strictly increasing when diverge_th is huge."""
        model = _simple_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        finder = _make_finder(model, opt, num_iter=20, diverge_th=1e9)
        history = finder.run(_simple_loader())
        lrs = history["lrs"]
        for i in range(1, len(lrs)):
            assert lrs[i] > lrs[i - 1], f"LR not increasing at step {i}"


# ---------------------------------------------------------------------------
# TestLRFinderStateRestoration
# ---------------------------------------------------------------------------

class TestLRFinderStateRestoration:
    def test_model_weights_restored_after_run(self):
        model = _simple_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        finder = _make_finder(model, opt)
        finder.run(_simple_loader())
        after = model.state_dict()
        for k in before:
            assert torch.allclose(before[k], after[k]), \
                f"Weight '{k}' changed after LR finder run"

    def test_optimizer_lr_restored_after_run(self):
        original_lr = 3e-4
        model = _simple_model()
        opt = torch.optim.Adam(model.parameters(), lr=original_lr)
        finder = _make_finder(model, opt)
        finder.run(_simple_loader())
        restored_lr = opt.param_groups[0]["lr"]
        assert math.isclose(restored_lr, original_lr, rel_tol=1e-9), \
            f"Optimizer LR not restored: expected {original_lr}, got {restored_lr}"


# ---------------------------------------------------------------------------
# TestLRFinderSuggestion
# ---------------------------------------------------------------------------

class TestLRFinderSuggestion:
    def test_suggestion_returns_float_in_range(self):
        model = _simple_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        start_lr, end_lr = 1e-7, 1.0
        finder = LRFinder(
            model, opt, nn.CrossEntropyLoss(), torch.device("cpu"),
            start_lr=start_lr, end_lr=end_lr, num_iter=20, diverge_th=1e9,
        )
        finder.run(_simple_loader())
        sugg = finder.suggestion()
        assert isinstance(sugg, float)
        assert start_lr <= sugg <= end_lr, \
            f"Suggestion {sugg} outside [{start_lr}, {end_lr}]"


# ---------------------------------------------------------------------------
# TestLRFinderAdaHessian
# ---------------------------------------------------------------------------

class TestLRFinderAdaHessian:
    def test_run_with_adahessian(self):
        """LRFinder completes with AdaHessian (requires_create_graph=True)."""
        model = _simple_model()
        opt = AdaHessian(model.parameters(), lr=1e-3)
        finder = LRFinder(
            model, opt, nn.CrossEntropyLoss(), torch.device("cpu"),
            start_lr=1e-7, end_lr=1.0, num_iter=10,
        )
        history = finder.run(_simple_loader())
        assert len(history["losses"]) >= 1
        assert all(math.isfinite(l) for l in history["losses"]), \
            "Some losses are non-finite during AdaHessian LR finder run"
