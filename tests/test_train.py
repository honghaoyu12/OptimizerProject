"""Tests for train.py — dataset info, model builder, and training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    DATASET_INFO,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    build_model,
    build_optimizer,
    evaluate,
    linear_layer_names,
    run_training,
    train_one_epoch,
    weight_norms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dummy_loader(n=64, in_channels=1, image_size=28, num_classes=10, batch_size=32):
    images = torch.randn(n, in_channels, image_size, image_size)
    labels = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


# ---------------------------------------------------------------------------
# DATASET_INFO
# ---------------------------------------------------------------------------

class TestDatasetInfo:
    def test_all_datasets_present(self):
        assert set(DATASET_INFO.keys()) == {
            "mnist", "fashion_mnist", "cifar10", "cifar100", "tiny_imagenet",
            "illcond", "sparse", "noisy_grad", "manifold", "saddle",
        }

    @pytest.mark.parametrize("key,in_ch,input_sz,n_cls", [
        ("mnist",         1, 784,   10),
        ("fashion_mnist", 1, 784,   10),
        ("cifar10",       3, 3072,  10),
        ("cifar100",      3, 3072,  100),
        ("tiny_imagenet", 3, 12288, 200),
        ("illcond",       1, 64,    2),
        ("sparse",        1, 100,   2),
        ("noisy_grad",    1, 64,    2),
        ("manifold",      1, 64,    2),
        ("saddle",        1, 64,    2),
    ])
    def test_dataset_metadata(self, key, in_ch, input_sz, n_cls):
        info = DATASET_INFO[key]
        assert info["in_channels"]  == in_ch
        assert info["input_size"]   == input_sz
        assert info["num_classes"]  == n_cls


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------

class TestBuildModel:
    @pytest.mark.parametrize("dataset_key", [
        "mnist", "fashion_mnist", "cifar10", "cifar100", "tiny_imagenet",
        "illcond", "sparse", "noisy_grad", "manifold", "saddle",
    ])
    def test_mlp_all_datasets(self, dataset_key):
        info = DATASET_INFO[dataset_key]
        model = build_model("mlp", info, hidden_sizes=[64, 32])
        image_size = int((info["input_size"] / info["in_channels"]) ** 0.5)
        x = torch.randn(2, info["in_channels"], image_size, image_size)
        out = model(x)
        assert out.shape == (2, info["num_classes"])

    @pytest.mark.parametrize("dataset_key", ["mnist", "cifar10", "cifar100", "tiny_imagenet"])
    def test_resnet18_all_datasets(self, dataset_key):
        info = DATASET_INFO[dataset_key]
        model = build_model("resnet18", info, hidden_sizes=[])
        image_size = int((info["input_size"] / info["in_channels"]) ** 0.5)
        x = torch.randn(2, info["in_channels"], image_size, image_size)
        out = model(x)
        assert out.shape == (2, info["num_classes"])

    @pytest.mark.parametrize("dataset_key", ["mnist", "cifar10", "cifar100"])
    def test_vit_all_datasets(self, dataset_key):
        info = DATASET_INFO[dataset_key]
        model = build_model("vit", info, hidden_sizes=[])
        image_size = int((info["input_size"] / info["in_channels"]) ** 0.5)
        x = torch.randn(2, info["in_channels"], image_size, image_size)
        out = model(x)
        assert out.shape == (2, info["num_classes"])

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("transformer_xl", DATASET_INFO["mnist"], hidden_sizes=[])

    @pytest.mark.parametrize("model_name", ["resnet18", "vit"])
    def test_tabular_rejects_non_mlp(self, model_name):
        with pytest.raises(ValueError, match="not supported for tabular"):
            build_model(model_name, DATASET_INFO["illcond"], hidden_sizes=[])


# ---------------------------------------------------------------------------
# linear_layer_names / weight_norms
# ---------------------------------------------------------------------------

class TestLayerHelpers:
    def test_linear_layer_names_mlp(self):
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[64, 32])
        names = linear_layer_names(model)
        assert len(names) == 3   # 784→64, 64→32, 32→10

    def test_weight_norms_keys_match_names(self):
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[64, 32])
        names = linear_layer_names(model)
        norms = weight_norms(model, names)
        assert set(norms.keys()) == set(names)

    def test_weight_norms_positive(self):
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[64])
        names = linear_layer_names(model)
        norms = weight_norms(model, names)
        for v in norms.values():
            assert v > 0


# ---------------------------------------------------------------------------
# train_one_epoch / evaluate
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    def _setup(self, model_name="mlp"):
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model(model_name, info, hidden_sizes=[32, 16]).to(device)
        optimizer = build_optimizer("adam", model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=64, in_channels=1, image_size=28, num_classes=10)
        layer_names = linear_layer_names(model)
        return model, optimizer, criterion, device, loader, layer_names

    def test_train_one_epoch_return_keys(self):
        model, opt, crit, device, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, device, names)
        assert set(result.keys()) == {
            "loss", "acc", "grad_norms_per_layer",
            "grad_norm_global", "grad_norm_std", "step_losses",
        }

    def test_train_one_epoch_loss_finite(self):
        model, opt, crit, device, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, device, names)
        assert 0.0 < result["loss"] < 1e6

    def test_train_one_epoch_acc_in_range(self):
        model, opt, crit, device, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, device, names)
        assert 0.0 <= result["acc"] <= 1.0

    def test_step_losses_length(self):
        model, opt, crit, device, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, device, names)
        # 64 samples / batch_size 32 = 2 batches
        assert len(result["step_losses"]) == 2

    def test_grad_norms_per_layer_keys(self):
        model, opt, crit, device, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, device, names)
        assert set(result["grad_norms_per_layer"].keys()) == set(names)

    def test_evaluate_returns_finite(self):
        model, _, crit, device, loader, _ = self._setup()
        loss, acc = evaluate(model, loader, crit, device)
        assert 0.0 < loss < 1e6
        assert 0.0 <= acc <= 1.0

    def test_loss_decreases_over_epochs(self):
        """Loss should trend down over several epochs on a small fixed dataset."""
        model, opt, crit, device, loader, names = self._setup()
        losses = []
        for _ in range(5):
            r = train_one_epoch(model, loader, opt, crit, device, names)
            losses.append(r["loss"])
        assert losses[-1] < losses[0], "Loss did not decrease over 5 epochs"


# ---------------------------------------------------------------------------
# Scheduler registry
# ---------------------------------------------------------------------------

class TestSchedulers:
    def _make_opt(self, lr=0.1):
        """Return a tiny model and SGD optimizer for scheduler tests."""
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        return optimizer

    def test_registry_has_all_keys(self):
        assert set(SCHEDULER_REGISTRY.keys()) == {"none", "cosine", "step", "warmup_cosine"}

    def test_none_returns_none(self):
        opt = self._make_opt()
        result = SCHEDULER_REGISTRY["none"](opt, 10, 5)
        assert result is None

    def test_cosine_lr_decreases(self):
        opt = self._make_opt(lr=0.1)
        scheduler = SCHEDULER_REGISTRY["cosine"](opt, 10, 0)
        initial_lr = opt.param_groups[0]["lr"]
        for _ in range(10):
            scheduler.step()
        final_lr = opt.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_warmup_lr_increases_then_decreases(self):
        opt = self._make_opt(lr=0.1)
        epochs, warmup = 10, 3
        scheduler = SCHEDULER_REGISTRY["warmup_cosine"](opt, epochs, warmup)
        # Step through warmup — LR should rise
        for _ in range(warmup):
            scheduler.step()
        peak_lr = opt.param_groups[0]["lr"]
        # Step through remaining epochs — LR should then decay
        for _ in range(epochs - warmup):
            scheduler.step()
        final_lr = opt.param_groups[0]["lr"]
        assert final_lr < peak_lr

    def test_run_training_records_learning_rates(self):
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = SCHEDULER_REGISTRY["cosine"](optimizer, 3, 0)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=32, in_channels=1, image_size=28, num_classes=10)
        layer_names = linear_layer_names(model)

        history = run_training(
            model, loader, loader, optimizer, criterion,
            device, epochs=3, layer_names=layer_names,
            verbose=False, scheduler=scheduler,
        )
        assert len(history["learning_rates"]) == 3
        # Cosine decay: first LR >= last LR
        assert history["learning_rates"][0] >= history["learning_rates"][-1]

    def test_run_training_no_scheduler_constant_lr(self):
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=32, in_channels=1, image_size=28, num_classes=10)
        layer_names = linear_layer_names(model)

        history = run_training(
            model, loader, loader, optimizer, criterion,
            device, epochs=3, layer_names=layer_names,
            verbose=False, scheduler=None,
        )
        lrs = history["learning_rates"]
        assert len(lrs) == 3
        assert all(lr == lrs[0] for lr in lrs), "LR should be constant with no scheduler"
