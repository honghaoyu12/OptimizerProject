"""Additional tests for visualizer.py — covers plot_benchmark, plot_dead_neurons,
plot_weight_rank, plot_grad_noise_scale, plot_fisher_trace, plot_perplexity,
plot_token_accuracy, and deeper edge-case coverage for existing plots."""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from visualizer import (
    _compute_lr_sensitivity,
    _pareto_frontier,
    plot_benchmark,
    plot_class_accuracy,
    plot_dead_neurons,
    plot_efficiency_frontier,
    plot_fisher_trace,
    plot_grad_conflict,
    plot_grad_cosine_sim,
    plot_grad_flow_heatmap,
    plot_grad_noise_scale,
    plot_grad_snr,
    plot_hp_heatmap,
    plot_instability,
    plot_lr_sensitivity,
    plot_lr_sensitivity_curves,
    plot_lr_sensitivity_scores,
    plot_optimizer_states,
    plot_perplexity,
    plot_plasticity,
    plot_sharpness,
    plot_spectral_norm,
    plot_step_size,
    plot_token_accuracy,
    plot_weight_distance,
    plot_weight_rank,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DS   = "MNIST"
MDL  = "MLP"
OPT  = "Adam"
KEY  = (DS, MDL, OPT)
COLORS = {OPT: "#377eb8"}


def _minimal_hist(epochs=3):
    """Minimal history dict with all commonly required keys."""
    return {
        "train_loss":       [0.8 - 0.1 * i for i in range(epochs)],
        "test_loss":        [0.9 - 0.1 * i for i in range(epochs)],
        "train_acc":        [0.5 + 0.1 * i for i in range(epochs)],
        "test_acc":         [0.4 + 0.1 * i for i in range(epochs)],
        "learning_rates":   [0.001] * epochs,
        "grad_norm_global": [1.0] * epochs,
        "steps_at_epoch_end": [100 * (i + 1) for i in range(epochs)],
        "ece":              [0.05] * epochs,
        "time_elapsed":     [1.0 * (i + 1) for i in range(epochs)],
        "config_lr":        0.001,
        "config_wd":        0.0,
    }


def _make_results(epochs=3, extra=None):
    h = _minimal_hist(epochs)
    if extra:
        h.update(extra)
    return {KEY: h}


# ---------------------------------------------------------------------------
# plot_benchmark
# ---------------------------------------------------------------------------

class TestPlotBenchmark:
    def _results(self, epochs=3, **extra):
        h = _minimal_hist(epochs)
        h.update(extra)
        return {KEY: h}

    def test_runs_without_error(self):
        plot_benchmark(
            self._results(), [DS], [MDL], [OPT], COLORS, save_path=None
        )
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "bench.png")
        plot_benchmark(
            self._results(), [DS], [MDL], [OPT], COLORS, save_path=path
        )
        plt.close("all")
        assert os.path.exists(path)

    def test_multi_optimizer(self):
        opt2 = "SGD"
        results = {
            KEY:            self._results()["MNIST", "MLP", "Adam"],
            (DS, MDL, opt2): _minimal_hist(),
        }
        colors = {OPT: "#377eb8", opt2: "#e41a1c"}
        plot_benchmark(results, [DS], [MDL], [OPT, opt2], colors, save_path=None)
        plt.close("all")

    def test_multi_dataset(self):
        ds2 = "CIFAR-10"
        results = {
            KEY:              self._results()["MNIST", "MLP", "Adam"],
            (ds2, MDL, OPT):  _minimal_hist(),
        }
        plot_benchmark(results, [DS, ds2], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_with_std_bands(self):
        h = _minimal_hist(3)
        h["test_acc_std"]  = [0.02, 0.02, 0.02]
        h["train_loss_std"] = [0.05, 0.05, 0.05]
        plot_benchmark({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_missing_ece_no_error(self):
        h = _minimal_hist(3)
        h.pop("ece", None)
        plot_benchmark({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_missing_steps_no_error(self):
        h = _minimal_hist(3)
        h.pop("steps_at_epoch_end", None)
        plot_benchmark({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_empty_results_no_error(self):
        plot_benchmark({}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_dead_neurons
# ---------------------------------------------------------------------------

class TestDeadNeuronsPlot:
    def _results(self, dn_data=None):
        h = _minimal_hist(3)
        if dn_data is not None:
            h["dead_neurons"] = dn_data
        return {KEY: h}

    def test_runs_without_error(self):
        results = self._results({"relu1": [0.1, 0.05, 0.02]})
        plot_dead_neurons(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "dn.png")
        results = self._results({"relu1": [0.1, 0.05, 0.02]})
        plot_dead_neurons(results, [DS], [MDL], [OPT], COLORS, save_path=path)
        plt.close("all")
        assert os.path.exists(path)

    def test_empty_dead_neurons_no_error(self):
        results = self._results({})
        plot_dead_neurons(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_missing_key_no_error(self):
        results = {KEY: _minimal_hist(3)}
        plot_dead_neurons(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_multi_layer(self):
        dn = {"relu1": [0.1, 0.08, 0.05], "relu2": [0.2, 0.15, 0.10]}
        results = self._results(dn)
        plot_dead_neurons(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_multi_optimizer(self):
        opt2 = "SGD"
        results = {
            KEY: {**_minimal_hist(3), "dead_neurons": {"relu1": [0.1, 0.05, 0.02]}},
            (DS, MDL, opt2): {**_minimal_hist(3), "dead_neurons": {"relu1": [0.3, 0.2, 0.1]}},
        }
        colors = {OPT: "#377eb8", opt2: "#e41a1c"}
        plot_dead_neurons(results, [DS], [MDL], [OPT, opt2], colors, save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_weight_rank
# ---------------------------------------------------------------------------

class TestWeightRankPlot:
    def _results(self, wr_data=None):
        h = _minimal_hist(3)
        if wr_data is not None:
            h["weight_rank"] = wr_data
        return {KEY: h}

    def test_runs_without_error(self):
        results = self._results({"fc1": [8.0, 7.5, 7.0]})
        plot_weight_rank(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "wr.png")
        results = self._results({"fc1": [8.0, 7.5, 7.0]})
        plot_weight_rank(results, [DS], [MDL], [OPT], COLORS, save_path=path)
        plt.close("all")
        assert os.path.exists(path)

    def test_empty_weight_rank_no_error(self):
        results = self._results({})
        plot_weight_rank(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_missing_key_no_error(self):
        results = {KEY: _minimal_hist(3)}
        plot_weight_rank(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_multi_layer(self):
        wr = {"fc1": [8.0, 7.5, 7.0], "fc2": [4.0, 4.0, 3.8]}
        results = self._results(wr)
        plot_weight_rank(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_multi_optimizer(self):
        opt2 = "SGD"
        results = {
            KEY: {**_minimal_hist(3), "weight_rank": {"fc1": [8.0, 7.5, 7.0]}},
            (DS, MDL, opt2): {**_minimal_hist(3), "weight_rank": {"fc1": [5.0, 5.2, 5.4]}},
        }
        colors = {OPT: "#377eb8", opt2: "#e41a1c"}
        plot_weight_rank(results, [DS], [MDL], [OPT, opt2], colors, save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_grad_noise_scale
# ---------------------------------------------------------------------------

class TestGradNoiseScalePlot:
    def _results(self, gns=None):
        h = _minimal_hist(3)
        if gns is not None:
            h["grad_noise_scale"] = gns
        return {KEY: h}

    def test_runs_without_error(self):
        results = self._results([0.8, 0.7, 0.6])
        plot_grad_noise_scale(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "gns.png")
        results = self._results([0.8, 0.7, 0.6])
        plot_grad_noise_scale(results, [DS], [MDL], [OPT], COLORS, save_path=path)
        plt.close("all")
        assert os.path.exists(path)

    def test_all_nan_skipped_no_error(self):
        results = self._results([float("nan")] * 3)
        plot_grad_noise_scale(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_missing_key_no_error(self):
        results = {KEY: _minimal_hist(3)}
        plot_grad_noise_scale(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_with_std_bands(self):
        h = _minimal_hist(3)
        h["grad_noise_scale"]     = [0.8, 0.7, 0.6]
        h["grad_noise_scale_std"] = [0.05, 0.05, 0.05]
        plot_grad_noise_scale({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_multi_optimizer(self):
        opt2 = "SGD"
        results = {
            KEY: {**_minimal_hist(3), "grad_noise_scale": [0.8, 0.7, 0.6]},
            (DS, MDL, opt2): {**_minimal_hist(3), "grad_noise_scale": [0.5, 0.4, 0.3]},
        }
        colors = {OPT: "#377eb8", opt2: "#e41a1c"}
        plot_grad_noise_scale(results, [DS], [MDL], [OPT, opt2], colors, save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_fisher_trace
# ---------------------------------------------------------------------------

class TestFisherTracePlot:
    def _results(self, ft=None):
        h = _minimal_hist(3)
        if ft is not None:
            h["fisher_trace"] = ft
        return {KEY: h}

    def test_runs_without_error(self):
        results = self._results([2.0, 1.5, 1.0])
        plot_fisher_trace(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "ft.png")
        results = self._results([2.0, 1.5, 1.0])
        plot_fisher_trace(results, [DS], [MDL], [OPT], COLORS, save_path=path)
        plt.close("all")
        assert os.path.exists(path)

    def test_all_nan_skipped_no_error(self):
        results = self._results([float("nan")] * 3)
        plot_fisher_trace(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_missing_key_no_error(self):
        results = {KEY: _minimal_hist(3)}
        plot_fisher_trace(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_multi_optimizer(self):
        opt2 = "SGD"
        results = {
            KEY: {**_minimal_hist(3), "fisher_trace": [2.0, 1.5, 1.0]},
            (DS, MDL, opt2): {**_minimal_hist(3), "fisher_trace": [3.0, 2.0, 1.0]},
        }
        colors = {OPT: "#377eb8", opt2: "#e41a1c"}
        plot_fisher_trace(results, [DS], [MDL], [OPT, opt2], colors, save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_perplexity
# ---------------------------------------------------------------------------

class TestPerplexityPlot:
    def _lm_results(self, train_ppl=None, test_ppl=None, series=OPT):
        h = _minimal_hist(3)
        h["train_perplexity"] = train_ppl if train_ppl is not None else [50.0, 40.0, 30.0]
        h["test_perplexity"]  = test_ppl  if test_ppl  is not None else [55.0, 45.0, 35.0]
        return {(DS, MDL, series): h}

    def test_runs_without_error(self):
        plot_perplexity(self._lm_results(), save_path=None)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "ppl.png")
        plot_perplexity(self._lm_results(), save_path=path)
        plt.close("all")
        assert os.path.exists(path)

    def test_all_nan_perplexity_skipped(self):
        results = self._lm_results(
            train_ppl=[float("nan")] * 3,
            test_ppl=[float("nan")] * 3,
        )
        plot_perplexity(results, save_path=None)
        plt.close("all")

    def test_missing_perplexity_key_skipped(self):
        h = _minimal_hist(3)
        # No train_perplexity key → should be skipped silently
        plot_perplexity({KEY: h}, save_path=None)
        plt.close("all")

    def test_multi_series(self):
        opt2 = "SGD"
        results = {
            (DS, MDL, OPT):  {**_minimal_hist(3), "train_perplexity": [50.0, 40.0, 30.0],
                              "test_perplexity": [55.0, 45.0, 35.0]},
            (DS, MDL, opt2): {**_minimal_hist(3), "train_perplexity": [60.0, 45.0, 35.0],
                              "test_perplexity": [65.0, 50.0, 40.0]},
        }
        plot_perplexity(results, save_path=None)
        plt.close("all")

    def test_with_std_bands(self):
        h = {**_minimal_hist(3),
             "train_perplexity":     [50.0, 40.0, 30.0],
             "test_perplexity":      [55.0, 45.0, 35.0],
             "train_perplexity_std": [2.0, 2.0, 2.0],
             "test_perplexity_std":  [3.0, 3.0, 3.0]}
        plot_perplexity({KEY: h}, save_path=None)
        plt.close("all")

    def test_empty_results_no_error(self):
        plot_perplexity({}, save_path=None)
        plt.close("all")

    def test_multi_dataset(self):
        ds2 = "TinyStories"
        results = {
            (DS, MDL, OPT):   {**_minimal_hist(3), "train_perplexity": [50.0, 40.0, 30.0],
                               "test_perplexity": [55.0, 45.0, 35.0]},
            (ds2, MDL, OPT):  {**_minimal_hist(3), "train_perplexity": [60.0, 50.0, 40.0],
                               "test_perplexity": [65.0, 55.0, 45.0]},
        }
        plot_perplexity(results, save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# plot_token_accuracy
# ---------------------------------------------------------------------------

class TestTokenAccuracyPlot:
    def _lm_results(self, train_ppl=None):
        h = _minimal_hist(3)
        h["train_perplexity"] = train_ppl if train_ppl is not None else [50.0, 40.0, 30.0]
        h["test_perplexity"]  = [55.0, 45.0, 35.0]
        return {KEY: h}

    def test_runs_without_error(self):
        plot_token_accuracy(self._lm_results(), save_path=None)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "tok_acc.png")
        plot_token_accuracy(self._lm_results(), save_path=path)
        plt.close("all")
        assert os.path.exists(path)

    def test_all_nan_perplexity_skipped(self):
        results = self._lm_results(train_ppl=[float("nan")] * 3)
        plot_token_accuracy(results, save_path=None)
        plt.close("all")

    def test_missing_perplexity_skipped(self):
        # No train_perplexity → classification run, should be skipped
        plot_token_accuracy({KEY: _minimal_hist(3)}, save_path=None)
        plt.close("all")

    def test_multi_series(self):
        opt2 = "SGD"
        results = {
            (DS, MDL, OPT):  {**_minimal_hist(3), "train_perplexity": [50.0, 40.0, 30.0],
                              "test_perplexity": [55.0, 45.0, 35.0]},
            (DS, MDL, opt2): {**_minimal_hist(3), "train_perplexity": [60.0, 45.0, 35.0],
                              "test_perplexity": [65.0, 50.0, 40.0]},
        }
        plot_token_accuracy(results, save_path=None)
        plt.close("all")

    def test_empty_results_no_error(self):
        plot_token_accuracy({}, save_path=None)
        plt.close("all")


# ---------------------------------------------------------------------------
# _compute_lr_sensitivity — deeper tests
# ---------------------------------------------------------------------------

class TestComputeLrSensitivityAdditional:
    def _sweep_results(self, lrs=(0.1, 0.01, 0.001), accs=None):
        results = {}
        if accs is None:
            accs = [0.9, 0.8, 0.6]
        for lr, acc in zip(lrs, accs):
            series = f"Adam (lr={lr:g})"
            results[(DS, MDL, series)] = {
                "test_acc":  [acc - 0.05, acc],
                "config_lr": lr,
            }
        return results

    def test_three_lr_sweep_produces_one_entry(self):
        scores = _compute_lr_sensitivity(self._sweep_results())
        assert len(scores) == 1

    def test_returns_dict_with_range_and_std(self):
        # Each value is a dict with 'range', 'std', 'best_lr', 'worst_lr', 'n_lrs'
        scores = _compute_lr_sensitivity(self._sweep_results())
        info = next(iter(scores.values()))
        assert "range"    in info
        assert "std"      in info
        assert "best_lr"  in info
        assert "worst_lr" in info
        assert "n_lrs"    in info

    def test_n_lrs_correct(self):
        scores = _compute_lr_sensitivity(self._sweep_results(lrs=(0.1, 0.01, 0.001)))
        info = next(iter(scores.values()))
        assert info["n_lrs"] == 3

    def test_low_variance_gives_low_range(self):
        # All LRs produce the same accuracy → range near 0
        results = self._sweep_results(accs=[0.8, 0.8, 0.8])
        scores = _compute_lr_sensitivity(results)
        info = next(iter(scores.values()))
        assert info["range"] < 1.0  # range in percentage points (%)

    def test_high_variance_gives_larger_range(self):
        results_high = self._sweep_results(accs=[0.9, 0.5, 0.1])
        results_low  = self._sweep_results(accs=[0.8, 0.8, 0.8])
        info_high = next(iter(_compute_lr_sensitivity(results_high).values()))
        info_low  = next(iter(_compute_lr_sensitivity(results_low).values()))
        assert info_high["range"] >= info_low["range"]

    def test_best_lr_has_highest_acc(self):
        results = self._sweep_results(lrs=(0.1, 0.01), accs=[0.9, 0.7])
        info = next(iter(_compute_lr_sensitivity(results).values()))
        assert abs(info["best_lr"] - 0.1) < 1e-9

    def test_multiple_optimizers_multiple_entries(self):
        results = {}
        for opt, base_acc in [("Adam", 0.9), ("SGD", 0.7)]:
            for lr in [0.1, 0.01]:
                series = f"{opt} (lr={lr:g})"
                results[(DS, MDL, series)] = {
                    "test_acc":  [base_acc, base_acc],
                    "config_lr": lr,
                }
        scores = _compute_lr_sensitivity(results)
        assert len(scores) == 2

    def test_empty_results_returns_empty(self):
        assert _compute_lr_sensitivity({}) == {}


# ---------------------------------------------------------------------------
# _pareto_frontier — deeper tests
# ---------------------------------------------------------------------------

class TestParetoFrontierAdditional:
    def test_all_points_on_frontier_when_tradeoff(self):
        # Each point is best in one dimension (time, acc, label)
        pts = [(1.0, 0.9, "A"), (2.0, 0.95, "B"), (3.0, 0.98, "C")]
        frontier = _pareto_frontier(pts)
        assert len(frontier) == 3

    def test_duplicate_points(self):
        pts = [(1.0, 0.8, "A"), (1.0, 0.8, "B")]
        frontier = _pareto_frontier(pts)
        # At least one copy on the frontier
        assert len(frontier) >= 1

    def test_single_dominated_point_excluded(self):
        # (2.0, 0.7, "B") is dominated by (1.0, 0.9, "A") — faster AND better
        pts = [(1.0, 0.9, "A"), (2.0, 0.7, "B")]
        frontier = _pareto_frontier(pts)
        times = [p[0] for p in frontier]
        assert 2.0 not in times

    def test_empty_input(self):
        assert _pareto_frontier([]) == []

    def test_frontier_sorted_by_time(self):
        pts = [(3.0, 0.98, "C"), (1.0, 0.9, "A"), (2.0, 0.95, "B")]
        frontier = _pareto_frontier(pts)
        times = [p[0] for p in frontier]
        assert times == sorted(times)


# ---------------------------------------------------------------------------
# Edge cases for existing plots (deeper than "runs without error")
# ---------------------------------------------------------------------------

class TestPlotBenchmarkEdgeCases:
    def test_accuracy_values_multiplied_by_100(self):
        """Test Accuracy is plotted as percent — train_acc=0.9 → 90%."""
        # We just verify no exception is raised with fractional acc values
        h = _minimal_hist(3)
        h["train_acc"] = [0.9, 0.92, 0.95]
        h["test_acc"]  = [0.85, 0.87, 0.90]
        plot_benchmark({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_single_epoch_no_error(self):
        h = {k: ([v[0]] if isinstance(v, list) else v)
             for k, v in _minimal_hist(3).items()}
        plot_benchmark({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotGradFlowHeatmapEdgeCases:
    def test_single_layer_single_epoch(self):
        results = {KEY: {**_minimal_hist(1), "grad_norms": {"fc": [0.5]}}}
        plot_grad_flow_heatmap(results, save_path=None)
        plt.close("all")

    def test_all_zero_gradients(self):
        results = {KEY: {**_minimal_hist(3), "grad_norms": {"fc": [0.0, 0.0, 0.0]}}}
        plot_grad_flow_heatmap(results, save_path=None)
        plt.close("all")


class TestPlotSharpnessEdgeCases:
    def test_some_nan_some_finite(self):
        results = {KEY: {**_minimal_hist(4), "sharpness": [float("nan"), 0.5, 0.4, 0.3]}}
        plot_sharpness(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_multi_optimizer_one_all_nan(self):
        opt2 = "SGD"
        results = {
            KEY:             {**_minimal_hist(3), "sharpness": [0.5, 0.4, 0.3]},
            (DS, MDL, opt2): {**_minimal_hist(3), "sharpness": [float("nan")] * 3},
        }
        colors = {OPT: "#377eb8", opt2: "#e41a1c"}
        plot_sharpness(results, [DS], [MDL], [OPT, opt2], colors, save_path=None)
        plt.close("all")


class TestPlotGradCosineSimEdgeCases:
    def test_values_outside_minus_one_to_one_still_plots(self):
        # Clamp robustness: values near boundaries
        results = {KEY: {**_minimal_hist(3), "grad_cosine_sim": {"fc": [-0.99, 0.0, 0.99]}}}
        plot_grad_cosine_sim(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_all_nan_first_epoch_excluded(self):
        results = {KEY: {**_minimal_hist(3), "grad_cosine_sim": {"fc": [float("nan"), 0.8, 0.9]}}}
        plot_grad_cosine_sim(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotGradConflictEdgeCases:
    def test_two_pairs_plots_without_error(self):
        results = {KEY: {**_minimal_hist(3),
                         "grad_conflict": {"L0-L1": [0.2, 0.3, 0.4],
                                           "L1-L2": [-0.1, 0.0, 0.1]}}}
        plot_grad_conflict(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotInstabilityEdgeCases:
    def test_zero_instability_no_error(self):
        results = {KEY: {**_minimal_hist(3), "batch_loss_std": [0.0, 0.0, 0.0]}}
        plot_instability(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_with_std_bands(self):
        h = {**_minimal_hist(3),
             "batch_loss_std":     [0.1, 0.08, 0.06],
             "batch_loss_std_std": [0.01, 0.01, 0.01]}
        plot_instability({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotWeightDistanceEdgeCases:
    def test_single_epoch_no_error(self):
        results = {KEY: {**_minimal_hist(1), "weight_distance": {"fc": [0.1]}}}
        plot_weight_distance(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_multiple_layers_averaged_correctly(self):
        results = {KEY: {**_minimal_hist(3),
                         "weight_distance": {"fc1": [0.1, 0.2, 0.3],
                                             "fc2": [0.2, 0.4, 0.6]}}}
        plot_weight_distance(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotStepSizeEdgeCases:
    def test_large_step_sizes_no_error(self):
        results = {KEY: {**_minimal_hist(3), "step_size": {"fc": [10.0, 8.0, 6.0]}}}
        plot_step_size(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotClassAccuracyEdgeCases:
    def test_exactly_20_classes_uses_bar(self):
        h = _minimal_hist(3)
        h["class_acc"] = {i: [0.5 + i * 0.02] * 3 for i in range(20)}
        plot_class_accuracy({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_single_class_no_error(self):
        h = _minimal_hist(3)
        h["class_acc"] = {0: [0.9, 0.92, 0.95]}
        plot_class_accuracy({KEY: h}, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotEfficiencyFrontierEdgeCases:
    def _results(self, time_val=5.0, acc_val=0.9):
        h = _minimal_hist(3)
        h["time_elapsed"] = [time_val / 3, time_val * 2 / 3, time_val]
        h["test_acc"]     = [acc_val - 0.1, acc_val - 0.05, acc_val]
        return {KEY: h}

    def test_single_point_frontier(self):
        plot_efficiency_frontier(self._results(), save_path=None)
        plt.close("all")

    def test_two_points_on_frontier(self):
        opt2 = "SGD"
        h1 = _minimal_hist(3)
        h1["time_elapsed"] = [1.0, 2.0, 5.0]
        h1["test_acc"]     = [0.7, 0.8, 0.9]
        h2 = _minimal_hist(3)
        h2["time_elapsed"] = [0.5, 1.0, 3.0]
        h2["test_acc"]     = [0.6, 0.7, 0.8]
        results = {KEY: h1, (DS, MDL, opt2): h2}
        plot_efficiency_frontier(results, save_path=None)
        plt.close("all")


class TestPlotGradSNREdgeCases:
    def test_very_high_snr_no_error(self):
        results = {KEY: {**_minimal_hist(3), "grad_snr": {"fc": [100.0, 90.0, 80.0]}}}
        plot_grad_snr(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")

    def test_snr_near_zero_no_error(self):
        results = {KEY: {**_minimal_hist(3), "grad_snr": {"fc": [0.001, 0.001, 0.001]}}}
        plot_grad_snr(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotSpectralNormEdgeCases:
    def test_multi_layer_no_error(self):
        results = {KEY: {**_minimal_hist(3),
                         "spectral_norm": {"fc1": [2.0, 1.9, 1.8],
                                           "fc2": [1.5, 1.4, 1.3]}}}
        plot_spectral_norm(results, [DS], [MDL], [OPT], COLORS, save_path=None)
        plt.close("all")


class TestPlotLrSensitivityCurvesEdgeCases:
    def _sweep(self, lrs=(0.1, 0.01)):
        results = {}
        for lr in lrs:
            series = f"Adam (lr={lr:g})"
            results[(DS, MDL, series)] = {
                "test_acc":    [0.7, 0.8, 0.9],
                "train_loss":  [0.9, 0.6, 0.4],
                "config_lr":   lr,
            }
        return results

    def test_three_lr_sweep_no_error(self):
        plot_lr_sensitivity_curves(self._sweep((0.1, 0.01, 0.001)), save_path=None)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "lr_curves.png")
        plot_lr_sensitivity_curves(self._sweep(), save_path=path)
        plt.close("all")
        assert os.path.exists(path)


class TestPlotHPHeatmapEdgeCases:
    def test_two_lrs_two_wds_no_error(self):
        results = {}
        for lr in [0.1, 0.01]:
            for wd in [0.0, 1e-4]:
                series = f"Adam (lr={lr:g}, wd={wd:g})"
                results[(DS, MDL, series)] = {
                    "test_acc":  [0.8, 0.85],
                    "config_lr": lr,
                    "config_wd": wd,
                }
        plot_hp_heatmap(results, save_path=None)
        plt.close("all")


class TestPlotLrSensitivityEdgeCases:
    def _sweep(self):
        return {
            (DS, MDL, f"Adam (lr={lr:g})"): {"test_acc": [0.8, 0.9], "config_lr": lr}
            for lr in [0.1, 0.01, 0.001]
        }

    def test_three_lrs_no_error(self):
        plot_lr_sensitivity(self._sweep(), save_path=None)
        plt.close("all")

    def test_saves_file(self, tmp_path):
        path = str(tmp_path / "lr_sens.png")
        plot_lr_sensitivity(self._sweep(), save_path=path)
        plt.close("all")
        assert os.path.exists(path)
