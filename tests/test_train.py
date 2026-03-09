"""Tests for train.py — dataset info, model builder, and training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train import (
    DATASET_INFO,
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    EarlyStopping,
    build_model,
    build_optimizer,
    evaluate,
    linear_layer_names,
    run_training,
    set_seed,
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
            "grad_norm_global", "grad_norm_std", "grad_norm_before_clip", "step_losses",
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


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_disabled_when_patience_zero(self):
        es = EarlyStopping(patience=0)
        assert not es.enabled
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        for loss in [1.0, 2.0, 3.0]:
            assert es.step(loss, model) is False

    def test_triggers_after_patience_epochs(self):
        es = EarlyStopping(patience=3)
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        # First step improves → saves best, counter stays 0
        assert es.step(1.0, model) is False
        # Three non-improving steps → should trigger on the 3rd
        assert es.step(1.5, model) is False
        assert es.step(1.5, model) is False
        assert es.step(1.5, model) is True

    def test_no_trigger_when_improving(self):
        es = EarlyStopping(patience=2)
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert es.step(loss, model) is False

    def test_best_state_saved_on_improvement(self):
        es = EarlyStopping(patience=2)
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        assert es.best_state is None
        es.step(1.0, model)
        assert es.best_state is not None
        assert es.best_loss == 1.0

    def test_restore_loads_best_weights(self):
        device = torch.device("cpu")
        es = EarlyStopping(patience=2)
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16]).to(device)
        # Save weights at best epoch
        es.step(1.0, model)
        best_param = next(iter(es.best_state.values())).clone()
        # Mutate model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        # Restore and verify weights match the saved best
        es.restore(model, device)
        restored_param = next(iter(model.state_dict().values()))
        assert torch.allclose(restored_param, best_param)

    def test_run_training_stops_early(self):
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = build_optimizer("adam", model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=32, in_channels=1, image_size=28, num_classes=10, batch_size=32)
        layer_names = linear_layer_names(model)

        # min_delta=1e9 means no realistic improvement satisfies the threshold,
        # so the counter increments every epoch and patience=2 fires after 3 epochs.
        history = run_training(
            model, loader, loader, optimizer, criterion,
            device, epochs=50, layer_names=layer_names,
            verbose=False, patience=2, min_delta=1e9,
        )
        assert history["early_stopped_epoch"] is not None
        assert len(history["train_loss"]) < 50

    def test_run_training_no_early_stop(self):
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = build_optimizer("adam", model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=32, in_channels=1, image_size=28, num_classes=10, batch_size=32)
        layer_names = linear_layer_names(model)

        history = run_training(
            model, loader, loader, optimizer, criterion,
            device, epochs=3, layer_names=layer_names,
            verbose=False, patience=0,
        )
        assert history["early_stopped_epoch"] is None
        assert len(history["train_loss"]) == 3


# ---------------------------------------------------------------------------
# Gradient Clipping
# ---------------------------------------------------------------------------

class TestGradientClipping:
    def _setup(self, n=32, batch_size=32):
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = build_optimizer("adam", model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=n, in_channels=1, image_size=28, num_classes=10, batch_size=batch_size)
        layer_names = linear_layer_names(model)
        return model, optimizer, criterion, device, loader, layer_names

    def test_no_clipping_when_none(self):
        model, opt, crit, device, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, device, names, max_grad_norm=None)
        assert "grad_norm_before_clip" in result
        assert result["grad_norm_before_clip"] == pytest.approx(result["grad_norm_global"])

    def test_clipping_caps_norm(self):
        # Single batch so accumulated post-clip global norm == per-batch clipped norm
        model, opt, crit, device, loader, names = self._setup(n=32, batch_size=32)
        max_norm = 1e-6
        result = train_one_epoch(model, loader, opt, crit, device, names, max_grad_norm=max_norm)
        assert result["grad_norm_global"] <= max_norm + 1e-8

    def test_no_clip_when_norm_below_threshold(self):
        model, opt, crit, device, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, device, names, max_grad_norm=1e9)
        assert result["grad_norm_before_clip"] == pytest.approx(result["grad_norm_global"])

    def test_run_training_history_key_present(self):
        model, opt, crit, device, loader, names = self._setup()
        history = run_training(
            model, loader, loader, opt, crit,
            device, epochs=3, layer_names=names,
            verbose=False, max_grad_norm=1.0,
        )
        assert "grad_norm_before_clip" in history
        assert len(history["grad_norm_before_clip"]) == 3

    def test_run_training_no_clipping_parity(self):
        model, opt, crit, device, loader, names = self._setup()
        history = run_training(
            model, loader, loader, opt, crit,
            device, epochs=3, layer_names=names,
            verbose=False, max_grad_norm=None,
        )
        for before, after in zip(history["grad_norm_before_clip"], history["grad_norm_global"]):
            assert before == pytest.approx(after)


# ---------------------------------------------------------------------------
# Weight Decay
# ---------------------------------------------------------------------------

class TestWeightDecay:
    def test_build_optimizer_accepts_weight_decay(self):
        """All 11 optimizers in OPTIMIZER_REGISTRY build without error with wd=1e-4."""
        from train import OPTIMIZER_REGISTRY
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        for name in OPTIMIZER_REGISTRY:
            opt = build_optimizer(name, model.parameters(), lr=1e-3, weight_decay=1e-4)
            assert opt is not None

    def test_weight_decay_stored_in_param_groups(self):
        """build_optimizer with weight_decay=1e-4 stores it in param_groups."""
        opt = build_optimizer("adam", build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16]).parameters(),
                              lr=1e-3, weight_decay=1e-4)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)

    def test_weight_decay_zero_default(self):
        """build_optimizer without weight_decay defaults to 0.0."""
        opt = build_optimizer("adam", build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16]).parameters(),
                              lr=1e-3)
        assert opt.param_groups[0]["weight_decay"] == 0.0

    def test_train_one_epoch_with_weight_decay(self):
        """train_one_epoch with weight_decay=1e-3 completes and returns finite loss."""
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = build_optimizer("adam", model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=32, in_channels=1, image_size=28, num_classes=10)
        layer_names = linear_layer_names(model)
        result = train_one_epoch(model, loader, optimizer, criterion, device, layer_names)
        assert 0.0 < result["loss"] < 1e6

    def test_vanilla_sgd_weight_decay_shrinks_params(self):
        """VanillaSGD with wd=1.0 produces smaller param norm than wd=0 after one step."""
        from optimizers import VanillaSGD

        def _run(wd):
            torch.manual_seed(0)
            model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
            optimizer = VanillaSGD(model.parameters(), lr=0.01, weight_decay=wd)
            criterion = nn.CrossEntropyLoss()
            loader = dummy_loader(n=32, in_channels=1, image_size=28, num_classes=10, batch_size=32)
            images, labels = next(iter(loader))
            optimizer.zero_grad()
            criterion(model(images), labels).backward()
            optimizer.step()
            return sum(p.norm().item() for p in model.parameters())

        norm_no_wd = _run(0.0)
        norm_with_wd = _run(1.0)
        assert norm_with_wd < norm_no_wd


# ---------------------------------------------------------------------------
# Seed / reproducibility
# ---------------------------------------------------------------------------

class TestSeed:
    def _run_one_epoch(self, seed=None):
        """Set seed, build model+optimizer from scratch, return first epoch loss."""
        if seed is not None:
            set_seed(seed)
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[32, 16]).to(device)
        optimizer = build_optimizer("adam", model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=64, in_channels=1, image_size=28, num_classes=10)
        layer_names = linear_layer_names(model)
        result = train_one_epoch(model, loader, optimizer, criterion, device, layer_names)
        return result["loss"]

    def test_same_seed_same_loss(self):
        """Two runs with the same seed produce identical losses."""
        loss_a = self._run_one_epoch(seed=42)
        loss_b = self._run_one_epoch(seed=42)
        assert loss_a == pytest.approx(loss_b)

    def test_different_seeds_different_loss(self):
        """Two runs with different seeds produce different losses."""
        loss_a = self._run_one_epoch(seed=0)
        loss_b = self._run_one_epoch(seed=99)
        assert loss_a != pytest.approx(loss_b)

    def test_no_seed_does_not_raise(self):
        """Calling without a seed completes without error."""
        loss = self._run_one_epoch(seed=None)
        assert 0.0 < loss < 1e6

    def test_set_seed_is_idempotent(self):
        """set_seed with the same value twice gives the same RNG state."""
        set_seed(7)
        t1 = torch.randn(4)
        set_seed(7)
        t2 = torch.randn(4)
        assert torch.allclose(t1, t2)
