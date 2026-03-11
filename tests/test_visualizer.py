"""Tests for visualizer.py plot utilities."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before any pyplot import

import pytest
from visualizer import (plot_benchmark, plot_lr_sensitivity, plot_lr_sensitivity_scores,
                        plot_grad_flow_heatmap, plot_optimizer_states,
                        plot_efficiency_frontier, _pareto_frontier,
                        _compute_lr_sensitivity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lr_results(lrs=(0.1, 0.01), opt_name="Adam"):
    """Minimal results dict with config_lr set, as run_benchmark() would produce."""
    results = {}
    for lr in lrs:
        series = f"{opt_name} (lr={lr:g})"
        results[("MNIST", "MLP", series)] = {
            "test_acc":  [0.80, 0.85, 0.90],
            "train_acc": [0.85, 0.90, 0.95],
            "config_lr": lr,
        }
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_runs_without_error(tmp_path):
    """plot_lr_sensitivity() completes without raising on valid data."""
    results = _make_lr_results()
    plot_lr_sensitivity(results, save_path=str(tmp_path / "lr_sens.png"))


def test_saves_file(tmp_path):
    """Figure file is created at the requested path."""
    results = _make_lr_results()
    out = tmp_path / "lr_sens.png"
    plot_lr_sensitivity(results, save_path=str(out))
    assert out.exists()


def test_skips_when_no_config_lr(tmp_path, capsys):
    """Results without config_lr → skip message printed, no file written."""
    results = {("MNIST", "MLP", "Adam"): {"test_acc": [0.9]}}
    plot_lr_sensitivity(results, save_path=str(tmp_path / "lr_sens.png"))
    captured = capsys.readouterr()
    assert "skipped" in captured.out.lower()
    assert not (tmp_path / "lr_sens.png").exists()


def test_multi_optimizer(tmp_path):
    """Multiple optimizers in results → file saved without error."""
    results = _make_lr_results(lrs=(0.1, 0.01), opt_name="Adam")
    results.update(_make_lr_results(lrs=(0.1, 0.01), opt_name="SGD"))
    out = tmp_path / "lr_sens.png"
    plot_lr_sensitivity(results, save_path=str(out))
    assert out.exists()


def test_std_error_bands_no_error(tmp_path):
    """test_acc_std present → fill_between bands drawn without error."""
    results = _make_lr_results()
    for hist in results.values():
        hist["test_acc_std"] = [0.01, 0.01, 0.01]
    out = tmp_path / "lr_sens.png"
    plot_lr_sensitivity(results, save_path=str(out))
    assert out.exists()


# ---------------------------------------------------------------------------
# plot_grad_flow_heatmap
# ---------------------------------------------------------------------------

def _make_grad_results(series_names=("Adam",), n_epochs=3, layers=("fc1", "fc2")):
    """Minimal results dict with grad_norms, as run_benchmark() would produce."""
    results = {}
    for series in series_names:
        results[("MNIST", "MLP", series)] = {
            "test_acc": [0.8 + 0.05 * e for e in range(n_epochs)],
            "grad_norms": {ln: [0.1 * (e + 1) for e in range(n_epochs)] for ln in layers},
        }
    return results


class TestGradFlowHeatmap:
    def test_runs_without_error(self, tmp_path):
        """plot_grad_flow_heatmap() completes without raising on valid data."""
        results = _make_grad_results()
        plot_grad_flow_heatmap(results, save_path=str(tmp_path / "grad.png"))

    def test_saves_file(self, tmp_path):
        """Figure file is created at the requested path."""
        results = _make_grad_results()
        out = tmp_path / "grad.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        assert out.exists()

    def test_skips_when_no_grad_norms(self, tmp_path, capsys):
        """Results without grad_norms → skip message printed, no file written."""
        results = {("MNIST", "MLP", "Adam"): {"test_acc": [0.9]}}
        out = tmp_path / "grad.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()
        assert not out.exists()

    def test_multi_optimizer(self, tmp_path):
        """Multiple optimizer series → file saved without error."""
        results = _make_grad_results(series_names=("Adam", "SGD", "Lion"))
        out = tmp_path / "grad_multi.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        assert out.exists()

    def test_multi_layer(self, tmp_path):
        """Four layers → file saved without error; ordering handled correctly."""
        results = _make_grad_results(layers=("fc1", "fc2", "fc3", "fc4"), n_epochs=5)
        out = tmp_path / "grad_layers.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        assert out.exists()

    def test_zero_gradient_values(self, tmp_path):
        """Zero gradient values (vanishing grads) handled without error."""
        results = _make_grad_results()
        for hist in results.values():
            hist["grad_norms"]["fc1"] = [0.0, 0.0, 0.0]
        out = tmp_path / "grad_zero.png"
        plot_grad_flow_heatmap(results, save_path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_optimizer_states
# ---------------------------------------------------------------------------

def _make_opt_state_results(series_names=("Adam",), n_epochs=3,
                             layers=("fc1", "fc2"), state_key="exp_avg_sq"):
    """Minimal results dict with optimizer_states, as run_training() would produce."""
    results = {}
    for series in series_names:
        results[("MNIST", "MLP", series)] = {
            "test_acc": [0.8 + 0.05 * e for e in range(n_epochs)],
            "optimizer_states": {
                state_key: {ln: [1e-4 * (e + 1) for e in range(n_epochs)] for ln in layers},
            },
        }
    return results


class TestOptimizerStatePlot:
    def test_runs_without_error(self, tmp_path):
        """plot_optimizer_states() completes without raising on valid heatmap data."""
        results = _make_opt_state_results()
        plot_optimizer_states(results, save_path=str(tmp_path / "opt.png"))

    def test_saves_file(self, tmp_path):
        """Figure file is created at the requested path."""
        results = _make_opt_state_results()
        out = tmp_path / "opt.png"
        plot_optimizer_states(results, save_path=str(out))
        assert out.exists()

    def test_skips_when_no_optimizer_states(self, tmp_path, capsys):
        """Results without optimizer_states → skip message printed, no file written."""
        results = {("MNIST", "MLP", "SGD"): {"test_acc": [0.9]}}
        out = tmp_path / "opt.png"
        plot_optimizer_states(results, save_path=str(out))
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()
        assert not out.exists()

    def test_scalar_trajectory(self, tmp_path):
        """Scalar state (Prodigy's d) renders as line plot without error."""
        results = {
            ("MNIST", "MLP", "Prodigy"): {
                "test_acc": [0.8, 0.85, 0.9],
                "optimizer_states": {"d": [1e-4, 5e-4, 1e-3]},
            }
        }
        out = tmp_path / "opt_scalar.png"
        plot_optimizer_states(results, save_path=str(out))
        assert out.exists()

    def test_mixed_state_types(self, tmp_path):
        """Adam (heatmap) and Prodigy (scalar) in same results → single file saved."""
        results = {
            ("MNIST", "MLP", "Adam"): {
                "test_acc": [0.9],
                "optimizer_states": {"exp_avg_sq": {"fc1": [1e-4], "fc2": [2e-4]}},
            },
            ("MNIST", "MLP", "Prodigy"): {
                "test_acc": [0.85],
                "optimizer_states": {"d": [1e-4]},
            },
        }
        out = tmp_path / "opt_mixed.png"
        plot_optimizer_states(results, save_path=str(out))
        assert out.exists()

    def test_multi_optimizer_same_state(self, tmp_path):
        """Adam and AdamW both have exp_avg_sq → both columns rendered."""
        results = _make_opt_state_results(series_names=("Adam", "AdamW"))
        out = tmp_path / "opt_multi.png"
        plot_optimizer_states(results, save_path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_efficiency_frontier / _pareto_frontier
# ---------------------------------------------------------------------------

def _make_frontier_results(series=("Adam", "SGD"), times=(5.0, 10.0),
                            accs=(0.95, 0.92)):
    """Minimal results dict with time_elapsed and test_acc."""
    results = {}
    for s, t, a in zip(series, times, accs):
        results[("MNIST", "MLP", s)] = {
            "test_acc":    [a],
            "time_elapsed": [t],
        }
    return results


class TestParetoFrontier:
    def test_single_point_is_frontier(self):
        """A single point is always on the frontier."""
        pts = [(3.0, 0.9, "Adam")]
        assert _pareto_frontier(pts) == pts

    def test_faster_and_better_dominates(self):
        """Point A that is both faster and better leaves point B off the frontier."""
        pts = [(2.0, 0.95, "A"), (5.0, 0.90, "B")]
        frontier = _pareto_frontier(pts)
        assert len(frontier) == 1
        assert frontier[0][2] == "A"

    def test_tradeoff_both_on_frontier(self):
        """A (fast, lower acc) and B (slow, higher acc) are both Pareto-optimal."""
        pts = [(2.0, 0.90, "A"), (8.0, 0.96, "B")]
        frontier = _pareto_frontier(pts)
        assert len(frontier) == 2

    def test_dominated_point_excluded(self):
        """Three points; the middle one dominated by the rightmost is excluded."""
        pts = [(1.0, 0.80, "A"), (3.0, 0.78, "B"), (5.0, 0.95, "C")]
        frontier = _pareto_frontier(pts)
        labels = [p[2] for p in frontier]
        assert "B" not in labels
        assert "A" in labels and "C" in labels

    def test_sorted_by_time(self):
        """Frontier is returned sorted by time ascending."""
        pts = [(10.0, 0.95, "C"), (1.0, 0.80, "A"), (5.0, 0.90, "B")]
        frontier = _pareto_frontier(pts)
        times = [p[0] for p in frontier]
        assert times == sorted(times)


class TestEfficiencyFrontier:
    def test_runs_without_error(self, tmp_path):
        """plot_efficiency_frontier() completes without raising on valid data."""
        results = _make_frontier_results()
        plot_efficiency_frontier(results, save_path=str(tmp_path / "frontier.png"))

    def test_saves_file(self, tmp_path):
        """Figure file is created at the requested path."""
        results = _make_frontier_results()
        out = tmp_path / "frontier.png"
        plot_efficiency_frontier(results, save_path=str(out))
        assert out.exists()

    def test_skips_when_no_time_data(self, tmp_path, capsys):
        """Results without time_elapsed → skip message, no file written."""
        results = {("MNIST", "MLP", "Adam"): {"test_acc": [0.95]}}
        out = tmp_path / "frontier.png"
        plot_efficiency_frontier(results, save_path=str(out))
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()
        assert not out.exists()

    def test_single_optimizer(self, tmp_path):
        """Single optimizer (no frontier line) renders without error."""
        results = _make_frontier_results(series=("Adam",), times=(5.0,), accs=(0.95,))
        out = tmp_path / "frontier_single.png"
        plot_efficiency_frontier(results, save_path=str(out))
        assert out.exists()

    def test_with_std_error_bars(self, tmp_path):
        """test_acc_std present → error bars drawn without error."""
        results = _make_frontier_results()
        for hist in results.values():
            hist["test_acc_std"] = [0.01]
        out = tmp_path / "frontier_std.png"
        plot_efficiency_frontier(results, save_path=str(out))
        assert out.exists()

    def test_multi_dataset_model(self, tmp_path):
        """Multiple (dataset, model) combos → multiple subplots, file saved."""
        results = {
            ("MNIST",        "MLP",     "Adam"): {"test_acc": [0.97], "time_elapsed": [4.0]},
            ("MNIST",        "MLP",     "SGD"):  {"test_acc": [0.95], "time_elapsed": [3.0]},
            ("FashionMNIST", "MLP",     "Adam"): {"test_acc": [0.89], "time_elapsed": [5.0]},
        }
        out = tmp_path / "frontier_multi.png"
        plot_efficiency_frontier(results, save_path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# _compute_lr_sensitivity / plot_lr_sensitivity_scores
# ---------------------------------------------------------------------------

def _make_sweep_results(opts=("Adam", "SGD"), lrs=(1e-3, 1e-2, 1e-1),
                        accs=None):
    """Multi-LR sweep results. accs: {opt: [acc_per_lr]}."""
    if accs is None:
        accs = {
            "Adam": [0.97, 0.95, 0.80],
            "SGD":  [0.90, 0.92, 0.60],
        }
    results = {}
    for opt in opts:
        for i, lr in enumerate(lrs):
            series = f"{opt} (lr={lr:g})"
            results[("MNIST", "MLP", series)] = {
                "test_acc":  [accs[opt][i]],
                "config_lr": lr,
            }
    return results


class TestComputeLrSensitivity:
    def test_returns_scores_for_swept_optimizers(self):
        """Two optimizers × 3 LRs → scores present for both."""
        results = _make_sweep_results()
        scores = _compute_lr_sensitivity(results)
        keys = [(ds, mdl, opt) for ds, mdl, opt in scores]
        assert any(opt == "Adam" for _, _, opt in keys)
        assert any(opt == "SGD"  for _, _, opt in keys)

    def test_range_correct(self):
        """Range = max - min across LR values in pp."""
        results = _make_sweep_results(opts=("Adam",), lrs=(1e-3, 1e-2),
                                      accs={"Adam": [0.90, 0.80]})
        scores = _compute_lr_sensitivity(results)
        s = scores[("MNIST", "MLP", "Adam")]
        assert abs(s["range"] - 10.0) < 1e-6

    def test_best_and_worst_lr(self):
        """best_lr and worst_lr point to the correct LR values."""
        results = _make_sweep_results(opts=("Adam",), lrs=(1e-3, 1e-2),
                                      accs={"Adam": [0.95, 0.80]})
        scores = _compute_lr_sensitivity(results)
        s = scores[("MNIST", "MLP", "Adam")]
        assert s["best_lr"]  == pytest.approx(1e-3)
        assert s["worst_lr"] == pytest.approx(1e-2)

    def test_single_lr_excluded(self):
        """Only one LR value → no score (can't compute sensitivity)."""
        results = {("MNIST", "MLP", "Adam"): {"test_acc": [0.95], "config_lr": 1e-3}}
        scores = _compute_lr_sensitivity(results)
        assert scores == {}

    def test_no_config_lr_excluded(self):
        """Entries without config_lr are ignored."""
        results = {("MNIST", "MLP", "Adam"): {"test_acc": [0.95]}}
        scores = _compute_lr_sensitivity(results)
        assert scores == {}


class TestLrSensitivityScores:
    def test_runs_without_error(self, tmp_path):
        """plot_lr_sensitivity_scores() completes on valid sweep data."""
        results = _make_sweep_results()
        plot_lr_sensitivity_scores(results, save_path=str(tmp_path / "scores.png"))

    def test_saves_file(self, tmp_path):
        """Figure file is created at the requested path."""
        results = _make_sweep_results()
        out = tmp_path / "scores.png"
        plot_lr_sensitivity_scores(results, save_path=str(out))
        assert out.exists()

    def test_skips_when_single_lr(self, tmp_path, capsys):
        """Single LR per optimizer → skip message, no file."""
        results = {("MNIST", "MLP", "Adam"): {"test_acc": [0.95], "config_lr": 1e-3}}
        out = tmp_path / "scores.png"
        plot_lr_sensitivity_scores(results, save_path=str(out))
        captured = capsys.readouterr()
        assert "skipped" in captured.out.lower()
        assert not out.exists()

    def test_multi_optimizer(self, tmp_path):
        """Multiple optimizers → all bars rendered, file saved."""
        results = _make_sweep_results(opts=("Adam", "SGD"))
        out = tmp_path / "scores_multi.png"
        plot_lr_sensitivity_scores(results, save_path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# EMA line in plot_benchmark
# ---------------------------------------------------------------------------

def _make_benchmark_results_with_ema(n_epochs=3):
    """Minimal results dict with test_acc_ema populated."""
    return {
        ("MNIST", "MLP", "Adam"): {
            "train_loss":    [0.5, 0.4, 0.3],
            "test_loss":     [0.6, 0.5, 0.4],
            "train_acc":     [0.80, 0.85, 0.90],
            "test_acc":      [0.75, 0.80, 0.85],
            "test_acc_ema":  [0.76, 0.81, 0.86],
            "learning_rates": [1e-3, 1e-3, 1e-3],
        }
    }


class TestEMAPlot:
    """Tests for EMA dashed line in plot_benchmark()."""

    def test_ema_runs_without_error(self, tmp_path):
        """plot_benchmark() does not raise when test_acc_ema is present."""
        results = _make_benchmark_results_with_ema()
        plot_benchmark(results, ["MNIST"], ["MLP"], ["Adam"],
                       {"Adam": "#1f77b4"}, save_path=str(tmp_path / "bench.png"))

    def test_ema_saves_file(self, tmp_path):
        """Figure file is created even when test_acc_ema is present."""
        results = _make_benchmark_results_with_ema()
        out = tmp_path / "bench_ema.png"
        plot_benchmark(results, ["MNIST"], ["MLP"], ["Adam"],
                       {"Adam": "#1f77b4"}, save_path=str(out))
        assert out.exists()

    def test_no_ema_data_no_error(self, tmp_path):
        """plot_benchmark() without test_acc_ema still works (backward-compatible)."""
        results = {
            ("MNIST", "MLP", "Adam"): {
                "train_loss":     [0.5, 0.4],
                "test_loss":      [0.6, 0.5],
                "train_acc":      [0.80, 0.85],
                "test_acc":       [0.75, 0.80],
                "learning_rates": [1e-3, 1e-3],
            }
        }
        out = tmp_path / "bench_no_ema.png"
        plot_benchmark(results, ["MNIST"], ["MLP"], ["Adam"],
                       {"Adam": "#1f77b4"}, save_path=str(out))
        assert out.exists()
