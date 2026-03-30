"""Additional tests for train.py — covering gaps in test_train.py.

Areas covered:
  - DATASET_INFO: LM dataset entries
  - build_model: GPT construction, non-gpt rejected for LM, empty model_name
  - build_optimizer: unknown name raises, case-insensitive, weight_decay param groups
  - linear_layer_names: GPT model, ResNet18, format correctness
  - weight_norms / weight_distances: correctness and types
  - EarlyStopping: min_delta, counter resets, best_loss tracking, restore no-op
  - _extract_optimizer_states: Prodigy d key, missing state graceful
  - save_checkpoint: file saved, keys present, round-trip load
  - LM training loop: train_one_epoch task_type=lm, perplexity key, no ECE/class_acc
  - run_training: LM dataset end-to-end
  - OPTIMIZER_REGISTRY: all entries are callable lambdas
  - SCHEDULER_REGISTRY: step scheduler reduces LR
  - set_seed: fixes torch random state
  - make_param_groups: frozen params excluded, no weight_decay → single group skipped
"""

import math
import os
import tempfile

import numpy as np
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
    linear_layer_names,
    make_param_groups,
    save_checkpoint,
    set_seed,
    weight_norms,
    weight_distances,
    _extract_optimizer_states,
    run_training,
    train_one_epoch,
)
from model import MLP, GPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_loaders(n=256, in_features=16, num_classes=2, batch_size=32):
    x = torch.randn(n, in_features)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, loader


def _lm_loaders(n_tokens=2048, seq_len=16, batch_size=8):
    """Tiny in-memory LM loader — avoids network access."""
    tokens = torch.randint(0, 256, (n_tokens // seq_len, seq_len))
    ds = TensorDataset(tokens)
    loader = DataLoader(ds, batch_size=batch_size)
    return loader, loader


def _make_lm_loader_pair(seq_len=16, batch_size=4, vocab_size=64):
    """Tiny in-memory LM loader yielding raw (B, seq_len) tensors."""
    tokens = torch.randint(0, vocab_size, (64, seq_len))

    class _RawDS(torch.utils.data.Dataset):
        def __getitem__(self, i): return tokens[i]
        def __len__(self): return len(tokens)

    ds = _RawDS()
    return DataLoader(ds, batch_size=batch_size), DataLoader(ds, batch_size=batch_size)


# ---------------------------------------------------------------------------
# DATASET_INFO — LM entries
# ---------------------------------------------------------------------------

class TestDatasetInfoLM:
    def test_lm_datasets_present(self):
        assert "tinystories" in DATASET_INFO
        assert "wikitext103" in DATASET_INFO

    def test_lm_task_type(self):
        for ds in ("tinystories", "wikitext103"):
            assert DATASET_INFO[ds].get("task_type") == "lm"

    def test_lm_has_vocab_size(self):
        for ds in ("tinystories", "wikitext103"):
            assert "vocab_size" in DATASET_INFO[ds]
            assert DATASET_INFO[ds]["vocab_size"] == 256

    def test_lm_has_seq_len(self):
        for ds in ("tinystories", "wikitext103"):
            assert "seq_len" in DATASET_INFO[ds]
            assert DATASET_INFO[ds]["seq_len"] > 0

    def test_lm_no_num_classes(self):
        """LM datasets do not have num_classes (they're not classifiers)."""
        for ds in ("tinystories", "wikitext103"):
            assert "num_classes" not in DATASET_INFO[ds]


# ---------------------------------------------------------------------------
# build_model — GPT and LM paths
# ---------------------------------------------------------------------------

class TestBuildModelGPT:
    _lm_info = {"vocab_size": 64, "seq_len": 32, "task_type": "lm"}

    def test_gpt_built_for_lm_dataset(self):
        model = build_model("gpt", self._lm_info, hidden_sizes=[])
        assert isinstance(model, GPT)

    def test_empty_name_builds_gpt_for_lm(self):
        model = build_model("", self._lm_info, hidden_sizes=[])
        assert isinstance(model, GPT)

    def test_non_gpt_rejected_for_lm(self):
        for name in ("mlp", "resnet18", "vit"):
            with pytest.raises(ValueError):
                build_model(name, self._lm_info, hidden_sizes=[])

    def test_gpt_vocab_size_propagated(self):
        info = {"vocab_size": 128, "seq_len": 16, "task_type": "lm"}
        model = build_model("gpt", info, hidden_sizes=[])
        assert model.token_emb.num_embeddings == 128

    def test_gpt_seq_len_propagated(self):
        info = {"vocab_size": 64, "seq_len": 32, "task_type": "lm"}
        model = build_model("gpt", info, hidden_sizes=[])
        assert model.seq_len == 32

    def test_gpt_case_insensitive(self):
        model = build_model("GPT", self._lm_info, hidden_sizes=[])
        assert isinstance(model, GPT)


# ---------------------------------------------------------------------------
# build_optimizer — additional
# ---------------------------------------------------------------------------

class TestBuildOptimizerAdditional:
    def test_unknown_name_raises(self):
        model = nn.Linear(4, 2)
        with pytest.raises(ValueError):
            build_optimizer("nonexistent_optimizer", model, lr=1e-3)

    def test_case_insensitive(self):
        model = nn.Linear(4, 2)
        opt = build_optimizer("ADAM", model, lr=1e-3)
        assert opt is not None

    def test_weight_decay_zero_skips_param_groups(self):
        """With wd=0 all params go through a single group (no split)."""
        model = nn.Linear(4, 2)
        opt = build_optimizer("adam", model, lr=1e-3, weight_decay=0.0)
        # All params in a single iterable (no split into decay/no-decay)
        all_params = list(model.parameters())
        opt_params = [p for g in opt.param_groups for p in g["params"]]
        assert len(opt_params) == len(all_params)

    def test_weight_decay_positive_splits_param_groups(self):
        """With wd>0 make_param_groups is called → two groups."""
        model = nn.Linear(4, 2)
        opt = build_optimizer("adam", model, lr=1e-3, weight_decay=0.1)
        assert len(opt.param_groups) == 2

    def test_all_registry_entries_callable(self):
        model = nn.Linear(4, 2)
        for name in OPTIMIZER_REGISTRY:
            # Sophia and AdaHessian need create_graph; just check construction
            try:
                opt = build_optimizer(name, model, lr=1e-3)
                assert opt is not None
            except Exception as e:
                pytest.fail(f"build_optimizer('{name}') raised: {e}")


# ---------------------------------------------------------------------------
# OPTIMIZER_REGISTRY
# ---------------------------------------------------------------------------

class TestOptimizerRegistry:
    def test_all_values_are_callable(self):
        for name, factory in OPTIMIZER_REGISTRY.items():
            assert callable(factory), f"OPTIMIZER_REGISTRY['{name}'] is not callable"

    def test_registry_includes_new_optimizers(self):
        for name in ("soap", "cautious_adam", "mars", "grokfast", "tiger", "adan_nesterov"):
            assert name in OPTIMIZER_REGISTRY, f"'{name}' missing from OPTIMIZER_REGISTRY"


# ---------------------------------------------------------------------------
# SCHEDULER_REGISTRY — additional
# ---------------------------------------------------------------------------

class TestSchedulerRegistryAdditional:
    def _make_opt(self, lr=0.1):
        model = nn.Linear(4, 2)
        return torch.optim.SGD(model.parameters(), lr=lr)

    def test_step_scheduler_reduces_lr(self):
        opt = self._make_opt(lr=0.1)
        scheduler = SCHEDULER_REGISTRY["step"](opt, epochs=9, warmup=0)
        initial_lr = opt.param_groups[0]["lr"]
        # After step_size=3 scheduler steps, LR should have dropped
        for _ in range(3):
            scheduler.step()
        assert opt.param_groups[0]["lr"] < initial_lr

    def test_all_registry_keys_present(self):
        expected = {"none", "cosine", "step", "warmup_cosine", "cosine_wr"}
        assert set(SCHEDULER_REGISTRY.keys()) == expected

    def test_none_scheduler_returns_none(self):
        opt = self._make_opt()
        result = SCHEDULER_REGISTRY["none"](opt, epochs=10, warmup=2)
        assert result is None

    def test_cosine_wr_t0_at_least_one(self):
        """cosine_wr T_0 = max(1, epochs//3); must not crash for small epochs."""
        opt = self._make_opt()
        sched = SCHEDULER_REGISTRY["cosine_wr"](opt, epochs=2, warmup=0)
        sched.step()   # must not raise


# ---------------------------------------------------------------------------
# linear_layer_names — additional
# ---------------------------------------------------------------------------

class TestLinearLayerNamesAdditional:
    def test_format_arrow_separator(self):
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        for name in names:
            assert "→" in name

    def test_index_starts_at_one(self):
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        assert names[0].startswith("L1(")

    def test_gpt_has_linear_layers(self):
        model = GPT(vocab_size=64, seq_len=16, dim=32, depth=1, heads=4, mlp_dim=64)
        names = linear_layer_names(model)
        assert len(names) > 0

    def test_resnet18_has_linear_layers(self):
        from model import ResNet18
        model = ResNet18(in_channels=3, num_classes=10)
        names = linear_layer_names(model)
        assert len(names) >= 1

    def test_count_matches_nn_linear_count(self):
        model = MLP(input_size=16, hidden_sizes=[8, 4], num_classes=2)
        names = linear_layer_names(model)
        n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        assert len(names) == n_linear


# ---------------------------------------------------------------------------
# weight_norms / weight_distances
# ---------------------------------------------------------------------------

class TestWeightNormsAdditional:
    def test_returns_dict_of_floats(self):
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        norms = weight_norms(model, names)
        assert isinstance(norms, dict)
        for v in norms.values():
            assert isinstance(v, float)

    def test_keys_match_layer_names(self):
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        norms = weight_norms(model, names)
        assert set(norms.keys()) == set(names)

    def test_all_positive(self):
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        for v in weight_norms(model, names).values():
            assert v >= 0.0


class TestWeightDistancesAdditional:
    def test_zero_at_init(self):
        """Distance from init to itself should be zero."""
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        init = {n: w.detach().clone()
                for n, w in zip(names, [m.weight for m in model.modules()
                                        if isinstance(m, nn.Linear)])}
        dists = weight_distances(model, names, init)
        for v in dists.values():
            assert math.isclose(v, 0.0, abs_tol=1e-6)

    def test_positive_after_update(self):
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        init = {n: w.detach().clone()
                for n, w in zip(names, [m.weight for m in model.modules()
                                        if isinstance(m, nn.Linear)])}
        # Perturb weights
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.weight.add_(torch.ones_like(m.weight))
        dists = weight_distances(model, names, init)
        for v in dists.values():
            assert v > 0.0


# ---------------------------------------------------------------------------
# EarlyStopping — additional
# ---------------------------------------------------------------------------

class TestEarlyStoppingAdditional:
    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=3)
        model = nn.Linear(2, 1)
        es.step(1.0, model)
        es.step(1.5, model)   # no improvement → counter=1
        es.step(0.8, model)   # improvement → counter resets to 0
        assert es.counter == 0

    def test_best_loss_updated_on_improvement(self):
        es = EarlyStopping(patience=5)
        model = nn.Linear(2, 1)
        es.step(1.0, model)
        assert es.best_loss == 1.0
        es.step(0.5, model)
        assert es.best_loss == 0.5

    def test_min_delta_prevents_update_on_tiny_improvement(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        model = nn.Linear(2, 1)
        es.step(1.0, model)    # best_loss = 1.0
        # 0.95 < 1.0 but not by min_delta=0.1 → no reset
        es.step(0.95, model)
        assert es.counter == 1

    def test_restore_noop_when_no_best_state(self):
        """restore() must not crash when step() was never called."""
        es = EarlyStopping(patience=3)
        model = nn.Linear(2, 1)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        es.restore(model, torch.device("cpu"))
        for k in original:
            assert torch.equal(model.state_dict()[k], original[k])

    def test_enabled_property(self):
        assert not EarlyStopping(patience=0).enabled
        assert EarlyStopping(patience=1).enabled

    def test_stop_returns_false_when_disabled(self):
        es = EarlyStopping(patience=0)
        model = nn.Linear(2, 1)
        for _ in range(100):
            assert es.step(999.0, model) is False


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------

class TestSaveCheckpointAdditional:
    def test_file_is_created(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "ckpt.pt")
            save_checkpoint(path, epoch=1, model=model, optimizer=opt, metrics={})
            assert os.path.exists(path)

    def test_saved_dict_has_required_keys(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(path, epoch=3, model=model, optimizer=opt,
                            metrics={"loss": 0.5}, config={"lr": 1e-3})
            ckpt = torch.load(path, weights_only=False)
            for key in ("epoch", "model_state_dict", "optimizer_state_dict",
                        "metrics", "config"):
                assert key in ckpt

    def test_epoch_stored_correctly(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(path, epoch=7, model=model, optimizer=opt, metrics={})
            ckpt = torch.load(path, weights_only=False)
            assert ckpt["epoch"] == 7

    def test_model_weights_round_trip(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters())
        original_w = model.weight.data.clone()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(path, epoch=1, model=model, optimizer=opt, metrics={})
            ckpt = torch.load(path, weights_only=False)
            loaded_w = ckpt["model_state_dict"]["weight"]
            assert torch.allclose(original_w, loaded_w)

    def test_config_none_stored_as_empty_dict(self):
        model = nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(path, epoch=1, model=model, optimizer=opt,
                            metrics={}, config=None)
            ckpt = torch.load(path, weights_only=False)
            assert ckpt["config"] == {}


# ---------------------------------------------------------------------------
# _extract_optimizer_states
# ---------------------------------------------------------------------------

class TestExtractOptimizerStates:
    def test_prodigy_d_key_extracted(self):
        from optimizers import Prodigy
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        opt = Prodigy(model.parameters(), lr=1.0)
        # Run one step to populate state
        x = torch.randn(4, 16)
        model(x).sum().backward()
        opt.step(); opt.zero_grad()
        result = _extract_optimizer_states(model, opt, names)
        assert "d" in result
        assert isinstance(result["d"], float)

    def test_adam_exp_avg_sq_extracted(self):
        from optimizers import Adam
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        opt = Adam(model.parameters(), lr=1e-3)
        x = torch.randn(4, 16)
        model(x).sum().backward()
        opt.step(); opt.zero_grad()
        result = _extract_optimizer_states(model, opt, names)
        assert "exp_avg_sq" in result

    def test_empty_for_vanilla_sgd(self):
        from optimizers import VanillaSGD
        model = MLP(input_size=16, hidden_sizes=[8], num_classes=4)
        names = linear_layer_names(model)
        opt = VanillaSGD(model.parameters(), lr=1e-2)
        x = torch.randn(4, 16)
        model(x).sum().backward()
        opt.step(); opt.zero_grad()
        result = _extract_optimizer_states(model, opt, names)
        # VanillaSGD has no tracked moments
        moment_keys = [k for k in result if k != "d"]
        assert len(moment_keys) == 0


# ---------------------------------------------------------------------------
# make_param_groups — additional
# ---------------------------------------------------------------------------

class TestMakeParamGroupsAdditional:
    def test_frozen_params_excluded(self):
        model = nn.Linear(4, 2)
        model.bias.requires_grad_(False)
        groups = make_param_groups(model, weight_decay=0.1)
        all_params = [p for g in groups for p in g["params"]]
        # bias is frozen → should not appear in any group
        assert not any(p is model.bias for p in all_params)

    def test_ndim_gte_2_gets_weight_decay(self):
        model = nn.Linear(4, 2)
        groups = make_param_groups(model, weight_decay=0.5)
        decay_group = groups[0]
        for p in decay_group["params"]:
            assert p.ndim >= 2
        assert decay_group["weight_decay"] == 0.5

    def test_ndim_1_gets_zero_decay(self):
        model = nn.Linear(4, 2)
        groups = make_param_groups(model, weight_decay=0.5)
        no_decay_group = groups[1]
        for p in no_decay_group["params"]:
            assert p.ndim < 2
        assert no_decay_group["weight_decay"] == 0.0


# ---------------------------------------------------------------------------
# set_seed — additional
# ---------------------------------------------------------------------------

class TestSetSeedAdditional:
    def test_same_seed_same_random_tensor(self):
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_different_seeds_different_tensors(self):
        set_seed(1)
        t1 = torch.randn(10)
        set_seed(2)
        t2 = torch.randn(10)
        assert not torch.allclose(t1, t2)

    def test_fixes_numpy_random(self):
        set_seed(99)
        a = np.random.rand(5)
        set_seed(99)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# LM training loop integration
# ---------------------------------------------------------------------------

class TestLMTrainingLoop:
    """Test train_one_epoch and run_training with a tiny in-memory LM loader."""

    def _make_gpt(self, vocab_size=64, seq_len=16):
        return GPT(vocab_size=vocab_size, seq_len=seq_len,
                   dim=32, depth=1, heads=4, mlp_dim=64, dropout=0.0)

    def _setup(self, seq_len=16, batch_size=4):
        model = self._make_gpt(seq_len=seq_len)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        names = linear_layer_names(model)
        train_l, val_l = _make_lm_loader_pair(seq_len=seq_len, batch_size=batch_size)
        device = torch.device("cpu")
        return model, opt, criterion, names, train_l, val_l, device

    def test_train_one_epoch_lm_returns_loss_and_perplexity(self):
        model, opt, criterion, names, train_l, _, device = self._setup()
        result = train_one_epoch(
            model, train_l, opt, criterion, device, names, task_type="lm"
        )
        assert "loss" in result
        assert "perplexity" in result
        assert math.isfinite(result["loss"])
        assert math.isfinite(result["perplexity"])

    def test_train_one_epoch_lm_loss_positive(self):
        model, opt, criterion, names, train_l, _, device = self._setup()
        result = train_one_epoch(
            model, train_l, opt, criterion, device, names, task_type="lm"
        )
        assert result["loss"] > 0

    def test_train_one_epoch_lm_perplexity_gte_one(self):
        model, opt, criterion, names, train_l, _, device = self._setup()
        result = train_one_epoch(
            model, train_l, opt, criterion, device, names, task_type="lm"
        )
        assert result["perplexity"] >= 1.0

    def test_train_one_epoch_lm_has_token_acc(self):
        """LM mode returns 'acc' as token-level accuracy."""
        model, opt, criterion, names, train_l, _, device = self._setup()
        result = train_one_epoch(
            model, train_l, opt, criterion, device, names, task_type="lm"
        )
        assert "acc" in result
        assert 0.0 <= result["acc"] <= 1.0

    def test_run_training_lm_history_keys(self):
        """run_training with task_type=lm should record train_loss and perplexity."""
        model, opt, criterion, names, train_l, val_l, device = self._setup()
        history = run_training(
            model=model,
            train_loader=train_l,
            test_loader=val_l,
            optimizer=opt,
            criterion=criterion,
            device=device,
            epochs=2,
            layer_names=names,
            verbose=False,
            task_type="lm",
        )
        assert "train_loss" in history
        assert "train_perplexity" in history
        assert len(history["train_loss"]) == 2

    def test_run_training_lm_no_ece(self):
        """ECE calibration is undefined for LM — should not be populated."""
        model, opt, criterion, names, train_l, val_l, device = self._setup()
        history = run_training(
            model=model,
            train_loader=train_l,
            test_loader=val_l,
            optimizer=opt,
            criterion=criterion,
            device=device,
            epochs=2,
            layer_names=names,
            verbose=False,
            task_type="lm",
        )
        assert "ece" not in history or all(math.isnan(v) for v in history["ece"])

    def test_run_training_lm_perplexity_finite(self):
        model, opt, criterion, names, train_l, val_l, device = self._setup()
        history = run_training(
            model=model,
            train_loader=train_l,
            test_loader=val_l,
            optimizer=opt,
            criterion=criterion,
            device=device,
            epochs=3,
            layer_names=names,
            verbose=False,
            task_type="lm",
        )
        for v in history["train_perplexity"]:
            assert math.isfinite(v)
