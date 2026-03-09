"""Tests for checkpoint saving — save_checkpoint() and run_training() integration."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import linear_layer_names, run_training, save_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_model():
    return nn.Linear(8, 4)


def tiny_loaders(batch_size=16):
    x = torch.randn(32, 8)
    y = torch.randint(0, 4, (32,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, loader


# ---------------------------------------------------------------------------
# save_checkpoint()
# ---------------------------------------------------------------------------

class TestSaveCheckpoint:
    def test_creates_file(self, tmp_path):
        model = tiny_model()
        opt   = torch.optim.SGD(model.parameters(), lr=0.01)
        path  = str(tmp_path / "ckpt" / "best.pt")
        save_checkpoint(path, epoch=3, model=model, optimizer=opt,
                        metrics={"val_acc": 0.9})
        assert os.path.isfile(path)

    def test_contains_required_keys(self, tmp_path):
        model = tiny_model()
        opt   = torch.optim.SGD(model.parameters(), lr=0.01)
        path  = str(tmp_path / "best.pt")
        save_checkpoint(path, epoch=5, model=model, optimizer=opt,
                        metrics={"train_loss": 0.5})
        ckpt = torch.load(path, weights_only=False)
        assert {"epoch", "model_state_dict", "optimizer_state_dict",
                "metrics", "config"} == set(ckpt.keys())

    def test_epoch_and_metrics_stored_correctly(self, tmp_path):
        model = tiny_model()
        opt   = torch.optim.Adam(model.parameters())
        path  = str(tmp_path / "ckpt.pt")
        metrics = {"train_acc": 0.75, "test_acc": 0.70}
        save_checkpoint(path, epoch=7, model=model, optimizer=opt,
                        metrics=metrics, config={"lr": 0.001})
        ckpt = torch.load(path, weights_only=False)
        assert ckpt["epoch"] == 7
        assert ckpt["metrics"]["train_acc"] == pytest.approx(0.75)
        assert ckpt["config"]["lr"] == pytest.approx(0.001)

    def test_model_state_restores_weights(self, tmp_path):
        torch.manual_seed(0)
        model = tiny_model()
        original_weight = model.weight.data.clone()
        opt  = torch.optim.SGD(model.parameters(), lr=0.01)
        path = str(tmp_path / "ckpt.pt")
        save_checkpoint(path, epoch=1, model=model, optimizer=opt,
                        metrics={})
        # Load into a fresh model
        new_model = tiny_model()
        ckpt = torch.load(path, weights_only=False)
        new_model.load_state_dict(ckpt["model_state_dict"])
        assert torch.allclose(new_model.weight.data, original_weight)

    def test_run_training_saves_best_and_final(self, tmp_path):
        train_loader, test_loader = tiny_loaders()
        model     = tiny_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        ckpt_dir  = str(tmp_path / "checkpoints")

        run_training(
            model, train_loader, test_loader,
            optimizer, criterion, torch.device("cpu"),
            epochs=2, layer_names=linear_layer_names(model),
            checkpoint_dir=ckpt_dir,
            checkpoint_config={"dataset": "test", "optimizer": "adam"},
        )
        assert os.path.isfile(os.path.join(ckpt_dir, "final.pt"))
        # best.pt should also exist (accuracy improves at least once in 2 epochs)
        assert os.path.isfile(os.path.join(ckpt_dir, "best.pt"))
