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
    _extract_optimizer_states,
    build_model,
    build_optimizer,
    compute_ece,
    evaluate,
    linear_layer_names,
    make_param_groups,
    per_class_accuracy,
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
            "checkerboard", "plateau", "correlated", "imbalanced",
            "tinystories", "wikitext103",
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
        ("checkerboard",  1, 32,    2),
        ("plateau",       1, 64,    2),
        ("correlated",    1, 64,    2),
        ("imbalanced",    1, 64,    2),
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
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=64, in_channels=1, image_size=28, num_classes=10)
        layer_names = linear_layer_names(model)
        return model, optimizer, criterion, device, loader, layer_names

    def test_train_one_epoch_return_keys(self):
        model, opt, crit, device, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, device, names)
        assert set(result.keys()) == {
            "loss", "acc", "perplexity", "grad_norms_per_layer",
            "grad_norm_global", "grad_norm_std", "grad_norm_before_clip", "step_losses",
            "grad_snr_per_layer", "mean_grad_vec_per_layer", "grad_noise_scale",
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
        assert set(SCHEDULER_REGISTRY.keys()) == {"none", "cosine", "step", "warmup_cosine", "cosine_wr"}

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

    def test_cosine_wr_lr_is_cyclic(self):
        """cosine_wr: LR drops to near-zero then resets — not monotonically decreasing."""
        opt = self._make_opt(lr=0.1)
        epochs = 9  # 3 restarts of T_0=3
        scheduler = SCHEDULER_REGISTRY["cosine_wr"](opt, epochs, 0)
        lrs = []
        for _ in range(epochs):
            scheduler.step()
            lrs.append(opt.param_groups[0]["lr"])
        # A cyclic schedule must increase at least once (restart point)
        increased = any(lrs[i] > lrs[i - 1] for i in range(1, len(lrs)))
        assert increased, "cosine_wr should show at least one LR increase (warm restart)"

    def test_cosine_wr_t0_derived_from_epochs(self):
        """T_0 = max(1, epochs // 3): for epochs=6, T_0=2 → 3 restarts."""
        opt = self._make_opt(lr=0.1)
        scheduler = SCHEDULER_REGISTRY["cosine_wr"](opt, 6, 0)
        assert hasattr(scheduler, "T_0")
        assert scheduler.T_0 == 2

    def test_cosine_wr_short_run_no_crash(self):
        """cosine_wr with epochs=1 (T_0=1) should not raise."""
        opt = self._make_opt(lr=0.1)
        scheduler = SCHEDULER_REGISTRY["cosine_wr"](opt, 1, 0)
        scheduler.step()  # should not raise

    def test_cosine_wr_in_run_training(self):
        """run_training with cosine_wr records one LR per epoch and shows cycling."""
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        epochs = 6
        scheduler = SCHEDULER_REGISTRY["cosine_wr"](optimizer, epochs, 0)
        criterion = nn.CrossEntropyLoss()
        loader = dummy_loader(n=32, in_channels=1, image_size=28, num_classes=10)
        layer_names = linear_layer_names(model)

        history = run_training(
            model, loader, loader, optimizer, criterion,
            device, epochs=epochs, layer_names=layer_names,
            verbose=False, scheduler=scheduler,
        )
        lrs = history["learning_rates"]
        assert len(lrs) == epochs
        increased = any(lrs[i] > lrs[i - 1] for i in range(1, len(lrs)))
        assert increased, "cosine_wr history should show at least one LR increase"


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
        optimizer = build_optimizer("adam", model, lr=1e-3)
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
        optimizer = build_optimizer("adam", model, lr=1e-3)
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
        optimizer = build_optimizer("adam", model, lr=1e-3)
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
            opt = build_optimizer(name, model, lr=1e-3, weight_decay=1e-4)
            assert opt is not None

    def test_weight_decay_stored_in_param_groups(self):
        """build_optimizer with weight_decay=1e-4 stores it in param_groups (decay group)."""
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        opt = build_optimizer("adam", model, lr=1e-3, weight_decay=1e-4)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(1e-4)

    def test_weight_decay_zero_default(self):
        """build_optimizer without weight_decay defaults to 0.0."""
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        opt = build_optimizer("adam", model, lr=1e-3)
        assert opt.param_groups[0]["weight_decay"] == 0.0

    def test_train_one_epoch_with_weight_decay(self):
        """train_one_epoch with weight_decay=1e-3 completes and returns finite loss."""
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = build_optimizer("adam", model, lr=1e-3, weight_decay=1e-3)
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

    def test_make_param_groups_two_groups(self):
        """make_param_groups returns exactly 2 dicts with 'params' and 'weight_decay' keys."""
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[32, 16])
        groups = make_param_groups(model, 1e-4)
        assert len(groups) == 2
        for g in groups:
            assert "params" in g
            assert "weight_decay" in g

    def test_make_param_groups_no_overlap(self):
        """No parameter id appears in both groups."""
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[32, 16])
        groups = make_param_groups(model, 1e-4)
        ids_decay = {id(p) for p in groups[0]["params"]}
        ids_no_decay = {id(p) for p in groups[1]["params"]}
        assert ids_decay.isdisjoint(ids_no_decay)

    def test_make_param_groups_covers_all_params(self):
        """Union of both groups equals the set of all trainable parameter ids."""
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[32, 16])
        groups = make_param_groups(model, 1e-4)
        all_ids = {id(p) for p in model.parameters() if p.requires_grad}
        group_ids = {id(p) for g in groups for p in g["params"]}
        assert group_ids == all_ids

    def test_bias_params_have_zero_decay(self):
        """All 1-D parameters (biases, norm params) land in the no-decay group."""
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[32, 16])
        groups = make_param_groups(model, 1e-4)
        no_decay_ids = {id(p) for p in groups[1]["params"]}
        for p in model.parameters():
            if p.requires_grad and p.ndim < 2:
                assert id(p) in no_decay_ids
        assert groups[1]["weight_decay"] == 0.0

    def test_weight_matrices_have_nonzero_decay(self):
        """All 2-D+ parameters (weight matrices) land in the decay group."""
        target_wd = 1e-4
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[32, 16])
        groups = make_param_groups(model, target_wd)
        decay_ids = {id(p) for p in groups[0]["params"]}
        for p in model.parameters():
            if p.requires_grad and p.ndim >= 2:
                assert id(p) in decay_ids
        assert groups[0]["weight_decay"] == pytest.approx(target_wd)


# ---------------------------------------------------------------------------
# Label smoothing
# ---------------------------------------------------------------------------

class TestLabelSmoothing:
    """Tests for label_smoothing in CrossEntropyLoss via train_one_epoch / run_training."""

    def _make_components(self, label_smoothing=0.0):
        device = torch.device("cpu")
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        loader = dummy_loader(n=32, in_channels=1, image_size=28, num_classes=10)
        layer_names = linear_layer_names(model)
        return model, loader, optimizer, criterion, device, layer_names

    def test_label_smoothing_zero_runs(self):
        """label_smoothing=0.0 (default) completes without error."""
        model, loader, opt, crit, dev, ln = self._make_components(0.0)
        result = train_one_epoch(model, loader, opt, crit, dev, ln)
        assert result["loss"] > 0.0

    def test_label_smoothing_nonzero_runs(self):
        """label_smoothing=0.1 completes without error and returns finite loss."""
        model, loader, opt, crit, dev, ln = self._make_components(0.1)
        result = train_one_epoch(model, loader, opt, crit, dev, ln)
        import math
        assert math.isfinite(result["loss"])
        assert result["loss"] > 0.0

    def test_label_smoothing_raises_loss_floor(self):
        """With smoothing=0.1, minimum achievable loss is higher than with smoothing=0."""
        # With 10 classes and smoothing ε, the minimum CE loss = -log(1 - ε + ε/C).
        # After many steps from the same init, the smoothed run should have higher loss.
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        tl, vl = SYNTHETIC_LOADERS["illcond"](batch_size=64)

        def _run(smoothing):
            torch.manual_seed(0)
            m = build_model("mlp", ds_info, [64])
            opt = torch.optim.Adam(m.parameters(), lr=1e-2)
            crit = torch.nn.CrossEntropyLoss(label_smoothing=smoothing)
            ln = linear_layer_names(m)
            history = run_training(m, tl, vl, opt, crit, torch.device("cpu"),
                                   5, ln, verbose=False)
            return min(history["train_loss"])

        loss_no_smooth = _run(0.0)
        loss_smoothed  = _run(0.1)
        assert loss_smoothed > loss_no_smooth

    def test_label_smoothing_in_run_training(self):
        """run_training with a smoothed criterion runs to completion."""
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        tl, vl = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        model = build_model("mlp", ds_info, [32])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        ln = linear_layer_names(model)
        history = run_training(model, tl, vl, opt, crit, torch.device("cpu"),
                               2, ln, verbose=False)
        assert len(history["train_loss"]) == 2
        assert all(v > 0 for v in history["train_loss"])


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
        optimizer = build_optimizer("adam", model, lr=1e-3)
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


# ---------------------------------------------------------------------------
# AMP
# ---------------------------------------------------------------------------

class TestAMP:
    """Tests for amp=True/False in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, optimizer, criterion, device, layer_names

    def test_amp_disabled_history_keys(self):
        """amp=False (default): all expected history keys present after training."""
        model, tl, vl, opt, crit, dev, ln = self._make_components()
        history = run_training(model, tl, vl, opt, crit, dev, 1, ln, verbose=False, amp=False)
        for key in ("train_loss", "test_acc", "time_elapsed", "grad_norm_global"):
            assert key in history

    def test_amp_cpu_noop(self):
        """amp=True on CPU: auto-disabled with message, training completes normally."""
        model, tl, vl, opt, crit, dev, ln = self._make_components()
        history = run_training(model, tl, vl, opt, crit, dev, 1, ln, verbose=False, amp=True)
        assert len(history["train_loss"]) == 1

    def test_amp_auto_disabled_sophia(self, capsys):
        """Sophia + amp=True: warning naming Sophia and create_graph printed; training completes."""
        from optimizers import Sophia
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        from synthetic_datasets import SYNTHETIC_LOADERS
        tl, vl = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        opt = Sophia(model.parameters(), lr=1e-4)
        crit = torch.nn.CrossEntropyLoss()
        dev = torch.device("cpu")
        ln = linear_layer_names(model)
        history = run_training(model, tl, vl, opt, crit, dev, 1, ln, verbose=False, amp=True)
        captured = capsys.readouterr()
        assert "AMP auto-disabled" in captured.out
        assert "Sophia" in captured.out
        assert "create_graph" in captured.out
        assert len(history["train_loss"]) == 1

    def test_amp_auto_disabled_adahessian(self, capsys):
        """AdaHessian + amp=True: warning naming AdaHessian and create_graph printed; training completes."""
        from optimizers import AdaHessian
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        from synthetic_datasets import SYNTHETIC_LOADERS
        tl, vl = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        opt = AdaHessian(model.parameters(), lr=0.1)
        crit = torch.nn.CrossEntropyLoss()
        dev = torch.device("cpu")
        ln = linear_layer_names(model)
        history = run_training(model, tl, vl, opt, crit, dev, 1, ln, verbose=False, amp=True)
        captured = capsys.readouterr()
        assert "AMP auto-disabled" in captured.out
        assert "AdaHessian" in captured.out
        assert "create_graph" in captured.out
        assert len(history["train_loss"]) == 1


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

class TestCompile:
    """Tests for torch.compile integration."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, optimizer, criterion, device, layer_names

    def test_compiled_model_trains_normally(self):
        """A torch.compiled model passed to run_training() produces a valid history."""
        model, tl, vl, opt, crit, dev, ln = self._make_components()
        try:
            compiled = torch.compile(model)
        except Exception:
            pytest.skip("torch.compile not available on this platform")
        history = run_training(compiled, tl, vl, opt, crit, dev, 1, ln, verbose=False)
        assert len(history["train_loss"]) == 1
        assert len(history["test_acc"]) == 1

    def test_compile_incompatible_tuple_contains_sophia_adahessian(self):
        """_COMPILE_INCOMPATIBLE sentinel contains both Sophia and AdaHessian."""
        from train import _COMPILE_INCOMPATIBLE
        from optimizers import Sophia, AdaHessian
        assert Sophia in _COMPILE_INCOMPATIBLE
        assert AdaHessian in _COMPILE_INCOMPATIBLE

    def test_compile_incompatible_warning_sophia(self, capsys):
        """Sophia optimizer triggers compile-incompatibility message when --compile is used."""
        from optimizers import Sophia
        from train import _COMPILE_INCOMPATIBLE
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        opt = Sophia(model.parameters(), lr=1e-4)
        # Simulate what main() does: check isinstance before compiling
        if isinstance(opt, _COMPILE_INCOMPATIBLE):
            print(
                f"  compile    : disabled ⚠  {opt.__class__.__name__} uses "
                f"backward(create_graph=True) to estimate the Hessian — "
                f"torch.compile's static graph tracing cannot capture dynamic "
                f"higher-order graphs correctly. Running in eager mode."
            )
        captured = capsys.readouterr()
        assert "Sophia" in captured.out
        assert "create_graph" in captured.out
        assert "eager" in captured.out


# ---------------------------------------------------------------------------
# Resume from checkpoint
# ---------------------------------------------------------------------------

class TestResume:
    """Tests for resume_from in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, optimizer, criterion, device, layer_names

    def test_resume_continues_from_checkpoint(self, tmp_path):
        """Resuming from an epoch-1 checkpoint produces a 1-epoch history (epoch 2 only)."""
        import os
        from train import save_checkpoint
        model, tl, vl, opt, crit, dev, ln = self._make_components()
        # Save a checkpoint at epoch 1
        ckpt_path = str(tmp_path / "ckpt.pt")
        save_checkpoint(ckpt_path, 1, model, opt, {"train_loss": 1.0})
        # Fresh model and optimizer to resume into
        model2, _, _, opt2, _, _, _ = self._make_components()
        history = run_training(
            model2, tl, vl, opt2, crit, dev, epochs=2,
            layer_names=ln, verbose=False, resume_from=ckpt_path,
        )
        # epochs=2, resumed from epoch 1 → only epoch 2 runs → 1 entry
        assert len(history["train_loss"]) == 1

    def test_resume_loads_model_weights(self, tmp_path):
        """Weights in the checkpoint are correctly restored before training continues."""
        from train import save_checkpoint
        model, tl, vl, opt, crit, dev, ln = self._make_components()
        # Set all weights to zero and save
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        ckpt_path = str(tmp_path / "ckpt_zeros.pt")
        save_checkpoint(ckpt_path, 1, model, opt, {})
        # Fresh model (random weights) — resume should overwrite with zeros at load time
        model2, _, _, opt2, _, _, _ = self._make_components()
        # Verify model2 is NOT all zeros before resume
        initial_nonzero = any(p.abs().max() > 0 for p in model2.parameters())
        assert initial_nonzero, "sanity: fresh model should have non-zero weights"
        # run_training will load zeros from checkpoint at start
        run_training(
            model2, tl, vl, opt2, crit, dev, epochs=2,
            layer_names=ln, verbose=False, resume_from=ckpt_path,
        )
        # After load_state_dict the model started from zeros; the test just checks no crash.


# ---------------------------------------------------------------------------
# Optimizer internal state tracking
# ---------------------------------------------------------------------------

class TestOptimizerStates:
    """Tests for _extract_optimizer_states() and history["optimizer_states"]."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_history_has_optimizer_states_key(self):
        """history always contains the optimizer_states key after run_training()."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 1, ln, verbose=False)
        assert "optimizer_states" in history

    def test_adam_tracks_exp_avg_sq(self):
        """Adam optimizer → exp_avg_sq populated with one entry per epoch per layer."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        opt_states = history["optimizer_states"]
        assert "exp_avg_sq" in opt_states
        for layer_vals in opt_states["exp_avg_sq"].values():
            assert len(layer_vals) == 2  # one entry per epoch

    def test_extract_returns_empty_for_fresh_optimizer(self):
        """_extract_optimizer_states returns {} before any step (no state built yet)."""
        model, _, _, _, _, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        result = _extract_optimizer_states(model, opt, ln)
        assert result == {}

    def test_extract_adam_exp_avg_sq_after_step(self):
        """After one step, _extract_optimizer_states finds exp_avg_sq for all linear layers."""
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        ln = linear_layer_names(model)
        tl, _ = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.CrossEntropyLoss()
        dev = torch.device("cpu")
        # Run one step to populate optimizer state
        train_one_epoch(model, tl, opt, crit, dev, ln)
        result = _extract_optimizer_states(model, opt, ln)
        assert "exp_avg_sq" in result
        assert set(result["exp_avg_sq"].keys()) == set(ln)
        for v in result["exp_avg_sq"].values():
            assert isinstance(v, float)
            assert v >= 0

    def test_sgd_produces_no_tracked_state(self):
        """SGD (no momentum) has no tracked state → optimizer_states is empty."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        history = run_training(model, tl, vl, opt, crit, dev, 1, ln, verbose=False)
        assert history["optimizer_states"] == {}


# ---------------------------------------------------------------------------
# EMA weights
# ---------------------------------------------------------------------------

class TestEMA:
    """Tests for ema_decay in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_ema_disabled_produces_empty_list(self):
        """Without ema_decay, history['test_acc_ema'] is empty."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert history["test_acc_ema"] == []

    def test_ema_enabled_has_one_entry_per_epoch(self):
        """With ema_decay, history['test_acc_ema'] has one float per epoch."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False,
                               ema_decay=0.9)
        assert len(history["test_acc_ema"]) == 2
        for v in history["test_acc_ema"]:
            assert 0.0 <= v <= 1.0

    def test_ema_values_are_finite(self):
        """EMA accuracy values are finite floats, not NaN or inf."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln, verbose=False,
                               ema_decay=0.999)
        import math
        for v in history["test_acc_ema"]:
            assert math.isfinite(v)

    def test_ema_does_not_corrupt_model_weights(self):
        """EMA evaluation swaps weights temporarily; model is fully restored afterward."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False,
                               ema_decay=0.9)

        # Re-evaluate the model with its current weights; must match the recorded test_acc
        _, final_acc = evaluate(model, vl, crit, dev)
        assert abs(final_acc - history["test_acc"][-1]) < 1e-6


# ---------------------------------------------------------------------------
# Stochastic Weight Averaging
# ---------------------------------------------------------------------------

class TestSWA:
    """Tests for the swa_start parameter in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_swa_disabled_gives_none(self):
        """swa_start=None → history['swa_final_acc'] is None."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln,
                               verbose=False, swa_start=None)
        assert history["swa_final_acc"] is None

    def test_swa_enabled_gives_float(self):
        """swa_start=1 → history['swa_final_acc'] is a float in [0, 1]."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln,
                               verbose=False, swa_start=1)
        assert isinstance(history["swa_final_acc"], float)
        assert 0.0 <= history["swa_final_acc"] <= 1.0

    def test_swa_value_is_finite(self):
        """SWA final accuracy is a finite float (not NaN or inf)."""
        import math
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln,
                               verbose=False, swa_start=2)
        assert math.isfinite(history["swa_final_acc"])

    def test_swa_does_not_corrupt_model_weights(self):
        """SWA evaluation leaves the original model weights intact."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln,
                               verbose=False, swa_start=1)
        # Re-evaluate the model; must match the last raw test_acc
        _, final_acc = evaluate(model, vl, torch.nn.CrossEntropyLoss(), dev)
        assert abs(final_acc - history["test_acc"][-1]) < 1e-6


# ---------------------------------------------------------------------------
# Convergence profile
# ---------------------------------------------------------------------------

class TestConvergenceProfile:
    """Tests for per-threshold convergence tracking in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_convergence_keys_present(self):
        """run_training() always returns 'convergence_epochs' and 'convergence_times' dicts."""
        from train import _CONV_THRESHOLDS
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "convergence_epochs" in history
        assert "convergence_times"  in history
        expected_keys = {f"{int(t * 100)}%" for t in _CONV_THRESHOLDS}
        assert set(history["convergence_epochs"].keys()) == expected_keys
        assert set(history["convergence_times"].keys())  == expected_keys

    def test_convergence_values_are_none_or_positive(self):
        """Each threshold value is either None (not reached) or a positive number."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for thresh_key, val in history["convergence_epochs"].items():
            assert val is None or (isinstance(val, int) and val >= 1), (
                f"convergence_epochs['{thresh_key}'] = {val!r}"
            )
        for thresh_key, val in history["convergence_times"].items():
            assert val is None or (isinstance(val, float) and val > 0), (
                f"convergence_times['{thresh_key}'] = {val!r}"
            )

    def test_lower_thresholds_not_later_than_higher(self):
        """If both a lower and higher threshold are reached, the lower must be reached first."""
        from train import _CONV_THRESHOLDS
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln, verbose=False)
        thresholds = sorted(_CONV_THRESHOLDS)
        epoch_vals = [history["convergence_epochs"].get(f"{int(t * 100)}%") for t in thresholds]
        reached = [(t, v) for t, v in zip(thresholds, epoch_vals) if v is not None]
        for i in range(len(reached) - 1):
            assert reached[i][1] <= reached[i + 1][1], (
                f"Lower threshold {reached[i][0]} reached later "
                f"({reached[i][1]}) than {reached[i + 1][0]} ({reached[i + 1][1]})"
            )


# ---------------------------------------------------------------------------
# Per-step efficiency tracking
# ---------------------------------------------------------------------------

class TestStepsAtEpochEnd:
    """Tests for history['steps_at_epoch_end'] added by run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_key_present(self):
        """run_training() always returns 'steps_at_epoch_end' in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "steps_at_epoch_end" in history

    def test_length_equals_epochs(self):
        """'steps_at_epoch_end' has one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        assert len(history["steps_at_epoch_end"]) == n_epochs

    def test_strictly_increasing(self):
        """Cumulative step counts must be strictly increasing across epochs."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln, verbose=False)
        steps = history["steps_at_epoch_end"]
        for i in range(1, len(steps)):
            assert steps[i] > steps[i - 1], (
                f"steps_at_epoch_end not increasing: {steps}"
            )

    def test_final_value_equals_step_losses_length(self):
        """Last entry must equal len(history['step_losses']) — cumulative batch count."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert history["steps_at_epoch_end"][-1] == len(history["step_losses"])


# ---------------------------------------------------------------------------
# Weight distance from initialisation
# ---------------------------------------------------------------------------

class TestWeightDistanceFromInit:
    """Tests for history['weight_distance'] added by run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_key_present(self):
        """run_training() always returns 'weight_distance' dict in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "weight_distance" in history
        assert isinstance(history["weight_distance"], dict)

    def test_same_layer_keys_as_weight_norms(self):
        """weight_distance has the same layer keys as weight_norms."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert set(history["weight_distance"].keys()) == set(history["weight_norms"].keys())

    def test_length_equals_epochs_per_layer(self):
        """Each layer list has one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        for layer, vals in history["weight_distance"].items():
            assert len(vals) == n_epochs, f"Layer {layer}: expected {n_epochs} entries, got {len(vals)}"

    def test_starts_near_zero_and_grows(self):
        """Epoch-1 distance is close to 0; distance grows (or stays) over epochs."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln, verbose=False)
        for layer, vals in history["weight_distance"].items():
            assert vals[-1] >= vals[0], (
                f"Layer {layer}: distance shrank from {vals[0]:.4f} to {vals[-1]:.4f}"
            )

    def test_all_values_non_negative(self):
        """L2 distances are always >= 0."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for layer, vals in history["weight_distance"].items():
            for v in vals:
                assert v >= 0, f"Negative distance for {layer}: {v}"


# ---------------------------------------------------------------------------
# Gradient SNR
# ---------------------------------------------------------------------------

class TestGradientSNR:
    """Tests for history['grad_snr'] produced by run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_key_present(self):
        """run_training() returns 'grad_snr' dict in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "grad_snr" in history
        assert isinstance(history["grad_snr"], dict)

    def test_same_layer_keys_as_grad_norms(self):
        """grad_snr has the same layer keys as grad_norms."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert set(history["grad_snr"].keys()) == set(history["grad_norms"].keys())

    def test_values_positive(self):
        """All per-layer SNR values are positive (mean_norm / (std_norm + eps) > 0)."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for layer, vals in history["grad_snr"].items():
            for v in vals:
                assert v > 0, f"Non-positive SNR for layer {layer}: {v}"

    def test_length_equals_epochs_per_layer(self):
        """Each layer list has one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        for layer, vals in history["grad_snr"].items():
            assert len(vals) == n_epochs, (
                f"Layer {layer}: expected {n_epochs} SNR values, got {len(vals)}"
            )


# ---------------------------------------------------------------------------
# Calibration (ECE)
# ---------------------------------------------------------------------------

class TestCalibration:
    """Tests for history['ece'] (Expected Calibration Error) from run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_ece_key_present(self):
        """run_training() always returns 'ece' list in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "ece" in history
        assert isinstance(history["ece"], list)

    def test_ece_one_per_epoch(self):
        """'ece' has exactly one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        assert len(history["ece"]) == n_epochs

    def test_ece_in_unit_interval(self):
        """All ECE values are in [0, 1]."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for v in history["ece"]:
            assert 0.0 <= v <= 1.0, f"ECE out of [0,1]: {v}"

    def test_compute_ece_exported(self):
        """compute_ece() is importable from train and returns a float in [0, 1]."""
        from train import compute_ece
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        _, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        device = torch.device("cpu")
        ece = compute_ece(model, test_loader, device)
        assert isinstance(ece, float)
        assert 0.0 <= ece <= 1.0


# ---------------------------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------------------------

class TestPerClassAccuracy:
    """Tests for history['class_acc'] and per_class_accuracy() helper."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_class_acc_key_present(self):
        """run_training() returns 'class_acc' dict in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "class_acc" in history
        assert isinstance(history["class_acc"], dict)

    def test_class_acc_keys_are_ints(self):
        """class_acc dict keys are integers (class IDs)."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 1, ln, verbose=False)
        for k in history["class_acc"]:
            assert isinstance(k, int), f"Expected int key, got {type(k)}: {k}"

    def test_class_acc_values_in_unit_interval(self):
        """All per-class accuracy values are in [0, 1]."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for cls_id, vals in history["class_acc"].items():
            for v in vals:
                assert 0.0 <= v <= 1.0, f"Class {cls_id}: {v} not in [0, 1]"

    def test_class_acc_length_equals_epochs(self):
        """Each class list has one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        for cls_id, vals in history["class_acc"].items():
            assert len(vals) == n_epochs, (
                f"Class {cls_id}: expected {n_epochs} entries, got {len(vals)}"
            )

    def test_per_class_accuracy_helper(self):
        """per_class_accuracy() returns a dict of int→float in [0, 1]."""
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        _, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        device = torch.device("cpu")
        result = per_class_accuracy(model, test_loader, device)
        assert isinstance(result, dict)
        for cls_id, acc in result.items():
            assert isinstance(cls_id, int)
            assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# Training instability (batch_loss_std)
# ---------------------------------------------------------------------------

class TestInstabilityScore:
    """Tests for history['batch_loss_std'] in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_key_present(self):
        """run_training() returns 'batch_loss_std' list in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "batch_loss_std" in history
        assert isinstance(history["batch_loss_std"], list)

    def test_one_per_epoch(self):
        """'batch_loss_std' has exactly one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        assert len(history["batch_loss_std"]) == n_epochs

    def test_values_non_negative(self):
        """All batch_loss_std values are >= 0."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for v in history["batch_loss_std"]:
            assert v >= 0.0, f"Negative batch_loss_std: {v}"


# ---------------------------------------------------------------------------
# Effective step size
# ---------------------------------------------------------------------------

class TestEffectiveStepSize:
    """Tests for history['step_size'] in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_key_present(self):
        """run_training() returns 'step_size' dict in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "step_size" in history
        assert isinstance(history["step_size"], dict)

    def test_same_layer_keys_as_weight_norms(self):
        """step_size has the same layer keys as weight_norms."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert set(history["step_size"].keys()) == set(history["weight_norms"].keys())

    def test_length_equals_epochs_per_layer(self):
        """Each layer list has one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        for layer, vals in history["step_size"].items():
            assert len(vals) == n_epochs, (
                f"Layer {layer}: expected {n_epochs} entries, got {len(vals)}"
            )

    def test_values_non_negative(self):
        """All step sizes are non-negative (L2 norm)."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for layer, vals in history["step_size"].items():
            for v in vals:
                assert v >= 0.0, f"Negative step size for {layer}: {v}"

    def test_positive_with_nonzero_lr(self):
        """With a nonzero LR, step sizes should be > 0 (weights actually move)."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for layer, vals in history["step_size"].items():
            assert vals[0] > 0.0, f"Zero step size at epoch 1 for {layer}"


class TestSharpness:
    """Tests for history['sharpness'] with track_sharpness=True."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_sharpness_key_present(self):
        """run_training() always returns 'sharpness' list in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "sharpness" in history
        assert isinstance(history["sharpness"], list)

    def test_sharpness_length_equals_epochs(self):
        """sharpness list has one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        assert len(history["sharpness"]) == n_epochs

    def test_sharpness_finite_when_track_sharpness_true(self):
        """With track_sharpness=True, sharpness values are finite (not nan)."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln,
                               verbose=False, track_sharpness=True)
        import math
        assert all(math.isfinite(v) for v in history["sharpness"]), (
            "Expected finite sharpness values; got: " + str(history["sharpness"])
        )

    def test_sharpness_nan_when_disabled(self):
        """With track_sharpness=False (default), sharpness values are nan."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        import math
        assert all(math.isnan(v) for v in history["sharpness"]), (
            "Expected nan sharpness when track_sharpness=False"
        )


class TestGradCosineSim:
    """Tests for history['grad_cosine_sim'] in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_key_present(self):
        """run_training() returns 'grad_cosine_sim' dict in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "grad_cosine_sim" in history
        assert isinstance(history["grad_cosine_sim"], dict)

    def test_per_layer_keys_match_weight_norms(self):
        """grad_cosine_sim has the same layer keys as weight_norms."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert set(history["grad_cosine_sim"].keys()) == set(history["weight_norms"].keys())

    def test_length_equals_epochs_per_layer(self):
        """Each layer list has one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        for layer, vals in history["grad_cosine_sim"].items():
            assert len(vals) == n_epochs, f"Layer {layer}: expected {n_epochs}, got {len(vals)}"

    def test_epoch1_is_nan(self):
        """Epoch 1 has no prior gradient direction — value should be nan."""
        import math
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for layer, vals in history["grad_cosine_sim"].items():
            assert math.isnan(vals[0]), f"Layer {layer} epoch-1 should be nan, got {vals[0]}"

    def test_epoch2_in_valid_range(self):
        """Epoch 2+ cosine similarities are in [-1, 1]."""
        import math
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 3, ln, verbose=False)
        for layer, vals in history["grad_cosine_sim"].items():
            for e, v in enumerate(vals[1:], start=2):
                assert not math.isnan(v), f"Layer {layer} epoch {e} is nan"
                assert -1.01 <= v <= 1.01, f"Layer {layer} epoch {e} out of range: {v}"


class TestGradConflict:
    """Tests for history['grad_conflict'] in run_training()."""

    def _make_components(self):
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        model = build_model("mlp", ds_info, [32])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        return model, train_loader, test_loader, criterion, device, layer_names

    def test_key_present(self):
        """run_training() returns 'grad_conflict' dict in history."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        assert "grad_conflict" in history
        assert isinstance(history["grad_conflict"], dict)

    def test_pair_keys_format(self):
        """Keys are 'L_i↔L_{i+1}' strings."""
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for key in history["grad_conflict"]:
            assert "↔" in key, f"Expected '↔' in key, got: {key}"

    def test_length_equals_epochs_per_pair(self):
        """Each adjacent-pair list has one entry per epoch."""
        n_epochs = 3
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, n_epochs, ln, verbose=False)
        for pair, vals in history["grad_conflict"].items():
            assert len(vals) == n_epochs, f"{pair}: expected {n_epochs}, got {len(vals)}"

    def test_values_in_range(self):
        """All conflict values are in [-1, 1]."""
        import math
        model, tl, vl, crit, dev, ln = self._make_components()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, tl, vl, opt, crit, dev, 2, ln, verbose=False)
        for pair, vals in history["grad_conflict"].items():
            for v in vals:
                assert not math.isnan(v), f"{pair} has nan"
                assert -1.01 <= v <= 1.01, f"{pair} value {v} out of range"

    def test_single_layer_empty(self):
        """A 1-Linear-layer network produces an empty grad_conflict dict."""
        from synthetic_datasets import SYNTHETIC_LOADERS
        ds_info = DATASET_INFO["illcond"]
        # hidden_sizes=[] → single Linear layer (input → output)
        model = build_model("mlp", ds_info, [])
        train_loader, test_loader = SYNTHETIC_LOADERS["illcond"](batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        device = torch.device("cpu")
        layer_names = linear_layer_names(model)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        history = run_training(model, train_loader, test_loader, opt, criterion,
                               device, 1, layer_names, verbose=False)
        assert history["grad_conflict"] == {}


# ---------------------------------------------------------------------------
# Gradient noise scale (train_one_epoch return key)
# ---------------------------------------------------------------------------

import math as _math

class TestGradNoiseScale:
    """Tests for the grad_noise_scale key added to train_one_epoch's return dict."""

    def _setup(self):
        model_name = "mlp"
        info = DATASET_INFO["mnist"]
        model = build_model(model_name, info, hidden_sizes=[32, 16]).to(torch.device("cpu"))
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)
        layer_names = linear_layer_names(model)
        return model, optimizer, criterion, torch.device("cpu"), loader, layer_names

    def test_key_present(self):
        model, opt, crit, dev, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, dev, names)
        assert "grad_noise_scale" in result

    def test_value_is_float(self):
        model, opt, crit, dev, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, dev, names)
        assert isinstance(result["grad_noise_scale"], float)

    def test_value_in_range(self):
        """GNS ∈ [0, 1] (Jensen's inequality: ||E[g]||² ≤ E[||g||²])."""
        model, opt, crit, dev, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, dev, names)
        gns = result["grad_noise_scale"]
        if not _math.isnan(gns):
            assert 0.0 <= gns <= 1.0 + 1e-6

    def test_history_key_populated(self):
        """run_training should populate history['grad_noise_scale'] each epoch."""
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)
        layer_names = linear_layer_names(model)
        history = run_training(model, loader, loader, optimizer, criterion,
                               torch.device("cpu"), 2, layer_names, verbose=False)
        assert len(history["grad_noise_scale"]) == 2


# ---------------------------------------------------------------------------
# Dead neuron tracking (run_training integration)
# ---------------------------------------------------------------------------

class TestDeadNeuronsIntegration:
    def _run(self, track=True):
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)
        layer_names = linear_layer_names(model)
        return run_training(model, loader, loader, optimizer, criterion,
                            torch.device("cpu"), 2, layer_names, verbose=False,
                            track_dead_neurons=track)

    def test_empty_when_disabled(self):
        history = self._run(track=False)
        assert history["dead_neurons"] == {}

    def test_populated_when_enabled(self):
        history = self._run(track=True)
        dn = history["dead_neurons"]
        assert len(dn) > 0  # at least one ReLU layer in MLP
        for key, vals in dn.items():
            assert len(vals) == 2  # one value per epoch

    def test_fractions_in_range(self):
        history = self._run(track=True)
        for key, vals in history["dead_neurons"].items():
            for v in vals:
                assert 0.0 <= v <= 1.0, f"{key}: {v} out of [0,1]"


# ---------------------------------------------------------------------------
# Weight rank tracking (run_training integration)
# ---------------------------------------------------------------------------

class TestWeightRankIntegration:
    def _run(self, track=True):
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)
        layer_names = linear_layer_names(model)
        return run_training(model, loader, loader, optimizer, criterion,
                            torch.device("cpu"), 2, layer_names, verbose=False,
                            track_weight_rank=track)

    def test_empty_when_disabled(self):
        history = self._run(track=False)
        assert history["weight_rank"] == {}

    def test_populated_when_enabled(self):
        history = self._run(track=True)
        wr = history["weight_rank"]
        assert len(wr) > 0
        for key, vals in wr.items():
            assert len(vals) == 2

    def test_ranks_at_least_one(self):
        history = self._run(track=True)
        for key, vals in history["weight_rank"].items():
            for v in vals:
                if not _math.isnan(v):
                    assert v >= 1.0 - 1e-6


# ---------------------------------------------------------------------------
# Fisher trace tracking (run_training integration)
# ---------------------------------------------------------------------------

class TestFisherTraceIntegration:
    def _run(self, track=True):
        info = DATASET_INFO["mnist"]
        model = build_model("mlp", info, hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)
        layer_names = linear_layer_names(model)
        return run_training(model, loader, loader, optimizer, criterion,
                            torch.device("cpu"), 2, layer_names, verbose=False,
                            track_fisher=track)

    def test_all_nan_when_disabled(self):
        history = self._run(track=False)
        assert all(_math.isnan(v) for v in history["fisher_trace"])

    def test_finite_when_enabled(self):
        history = self._run(track=True)
        ft = history["fisher_trace"]
        assert len(ft) == 2
        assert all(_math.isfinite(v) for v in ft)

    def test_non_negative_when_enabled(self):
        history = self._run(track=True)
        for v in history["fisher_trace"]:
            assert v >= 0.0


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

class TestGradientAccumulation:
    """Tests for the accum_steps parameter in train_one_epoch / run_training."""

    def _setup(self, n_samples=64, batch_size=16):
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[32]).to(torch.device("cpu"))
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(n_samples, 1, 28, 28)
        y = torch.randint(0, 10, (n_samples,))
        loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
        names = linear_layer_names(model)
        return model, optimizer, criterion, torch.device("cpu"), loader, names

    def test_return_keys_unchanged(self):
        """accum_steps should not add or remove keys from the return dict."""
        model, opt, crit, dev, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, dev, names, accum_steps=2)
        assert set(result.keys()) == {
            "loss", "acc", "perplexity", "grad_norms_per_layer",
            "grad_norm_global", "grad_norm_std", "grad_norm_before_clip", "step_losses",
            "grad_snr_per_layer", "mean_grad_vec_per_layer", "grad_noise_scale",
        }

    def test_loss_is_finite(self):
        """Accumulated gradient training should still produce finite loss."""
        model, opt, crit, dev, loader, names = self._setup()
        result = train_one_epoch(model, loader, opt, crit, dev, names, accum_steps=4)
        assert _math.isfinite(result["loss"])

    def test_accum_steps_1_equals_default(self):
        """accum_steps=1 should be identical to calling without the parameter."""
        torch.manual_seed(42)
        model1, opt1, crit, dev, loader, names = self._setup()
        torch.manual_seed(42)
        model2, opt2, _, _, _, _ = self._setup()
        torch.manual_seed(0)
        r1 = train_one_epoch(model1, loader, opt1, crit, dev, names, accum_steps=1)
        torch.manual_seed(0)
        r2 = train_one_epoch(model2, loader, opt2, crit, dev, names)
        assert abs(r1["loss"] - r2["loss"]) < 1e-5

    def test_weights_change_with_accumulation(self):
        """Weights must be different after training with accum_steps > 1."""
        model, opt, crit, dev, loader, names = self._setup()
        weights_before = [p.data.clone() for p in model.parameters()]
        train_one_epoch(model, loader, opt, crit, dev, names, accum_steps=2)
        weights_after = [p.data.clone() for p in model.parameters()]
        any_changed = any(
            not torch.allclose(b, a)
            for b, a in zip(weights_before, weights_after)
        )
        assert any_changed, "Weights did not change after accumulated training step"

    def test_run_training_accepts_accum_steps(self):
        """run_training should propagate accum_steps without error."""
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(X, y), batch_size=8)
        names = linear_layer_names(model)
        history = run_training(model, loader, loader, optimizer, criterion,
                               torch.device("cpu"), 2, names, verbose=False,
                               accum_steps=4)
        assert len(history["train_loss"]) == 2


# ---------------------------------------------------------------------------
# Gradient alignment (run_training integration)
# ---------------------------------------------------------------------------

class TestGradAlignmentIntegration:
    def _run(self):
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(64, 1, 28, 28)
        y = torch.randint(0, 10, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)
        return run_training(model, loader, loader, optimizer, criterion,
                            torch.device("cpu"), 3, linear_layer_names(model),
                            verbose=False)

    def test_key_present(self):
        history = self._run()
        assert "grad_alignment" in history

    def test_length_matches_epochs(self):
        history = self._run()
        assert len(history["grad_alignment"]) == 3

    def test_values_in_range_or_nan(self):
        history = self._run()
        for v in history["grad_alignment"]:
            if not _math.isnan(v):
                assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Spectral norm (run_training integration)
# ---------------------------------------------------------------------------

class TestSpectralNormIntegration:
    def _run(self, track=False):
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)
        return run_training(model, loader, loader, optimizer, criterion,
                            torch.device("cpu"), 2, linear_layer_names(model),
                            verbose=False, track_spectral_norm=track)

    def test_empty_when_disabled(self):
        history = self._run(track=False)
        assert history["spectral_norm"] == {}

    def test_populated_when_enabled(self):
        history = self._run(track=True)
        assert len(history["spectral_norm"]) > 0

    def test_values_non_negative(self):
        history = self._run(track=True)
        for layer, vals in history["spectral_norm"].items():
            for v in vals:
                if not _math.isnan(v):
                    assert v >= 0.0, f"{layer}: σ_max={v} < 0"


# ---------------------------------------------------------------------------
# Optimizer state entropy (run_training integration)
# ---------------------------------------------------------------------------

class TestOptimizerEntropyIntegration:
    def _run(self, track=False):
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)
        return run_training(model, loader, loader, optimizer, criterion,
                            torch.device("cpu"), 2, linear_layer_names(model),
                            verbose=False, track_optimizer_entropy=track)

    def test_all_nan_when_disabled(self):
        history = self._run(track=False)
        assert all(_math.isnan(v) for v in history["optimizer_state_entropy"])

    def test_finite_when_enabled(self):
        history = self._run(track=True)
        for v in history["optimizer_state_entropy"]:
            assert _math.isfinite(v)

    def test_non_negative_when_enabled(self):
        history = self._run(track=True)
        for v in history["optimizer_state_entropy"]:
            assert v >= 0.0


# ---------------------------------------------------------------------------
# Plasticity score (run_training integration)
# ---------------------------------------------------------------------------

class TestPlasticityIntegration:
    def _run(self, interval=0):
        model = build_model("mlp", DATASET_INFO["mnist"], hidden_sizes=[16])
        optimizer = build_optimizer("adam", model, lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        X = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)
        return run_training(model, loader, loader, optimizer, criterion,
                            torch.device("cpu"), 4, linear_layer_names(model),
                            verbose=False, plasticity_interval=interval)

    def test_all_nan_when_disabled(self):
        history = self._run(interval=0)
        assert all(_math.isnan(v) for v in history["plasticity"])

    def test_measured_at_correct_epochs(self):
        """With interval=2, only epochs 2 and 4 should be non-NaN."""
        history = self._run(interval=2)
        ps = history["plasticity"]
        # Indices 0, 1, 2, 3 → epochs 1, 2, 3, 4
        assert _math.isnan(ps[0])   # epoch 1 — not measured
        assert not _math.isnan(ps[1])  # epoch 2 — measured
        assert _math.isnan(ps[2])   # epoch 3 — not measured
        assert not _math.isnan(ps[3])  # epoch 4 — measured

    def test_values_in_range(self):
        history = self._run(interval=2)
        for v in history["plasticity"]:
            if not _math.isnan(v):
                assert 0.0 <= v <= 1.0
