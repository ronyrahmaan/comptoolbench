"""Tests for the statistical rigor module.

Covers:
  - bootstrap_ci: basic CI computation, edge cases, seed determinism
  - bootstrap_paired_test: paired significance, degenerate inputs
  - cohens_d: standard effect sizes, edge cases
  - compute_model_cis: integration with BenchmarkResults
  - compute_gap_cis: gap bootstrapping
  - pairwise_significance: Bonferroni correction, paired task matching
  - generate_statistical_report: full report generation

Uses synthetic benchmark results from the shared test fixture pattern
established in test_analysis.py.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from comptoolbench.analysis.loader import BenchmarkResults, load_results
from comptoolbench.analysis.statistical import (
    LEVELS,
    bootstrap_ci,
    bootstrap_paired_test,
    cohens_d,
    compute_gap_cis,
    compute_model_cis,
    generate_statistical_report,
    pairwise_significance,
)

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_model_result(
    name: str,
    provider: str,
    l0: float,
    l1: float,
    l2: float,
    l3: float,
    total_tokens: int = 50000,
) -> dict:
    """Create a synthetic model result dict matching runner output format."""
    gap_l1 = l0 - l1
    gap_l2 = l0 - l2
    gap_l3 = l0 - l3
    gap_overall = 0.30 * gap_l1 + 0.30 * gap_l2 + 0.40 * gap_l3
    overall = (l0 + l1 + l2 + l3) / 4

    task_results = []
    levels = [
        ("L0_node", l0, 20),
        ("L1_chain", l1, 15),
        ("L2_parallel", l2, 10),
        ("L3_dag", l3, 8),
    ]
    task_id = 0
    for level, acc, count in levels:
        for i in range(count):
            task_id += 1
            score = max(0.0, min(1.0, acc + (i % 3 - 1) * 0.05))
            error = None if score > 0.7 else "E4_wrong_arguments"
            if score < 0.3:
                error = "E8_partial_completion"
            task_results.append(
                {
                    "task_id": f"task_{task_id:04d}",
                    "model": name,
                    "score": {
                        "task_id": f"task_{task_id:04d}",
                        "level": level,
                        "overall": round(score, 4),
                        "tool_sequence_score": round(score * 0.95, 4),
                        "argument_score": round(score * 0.9, 4),
                        "completeness_score": round(min(1.0, score * 1.1), 4),
                        "data_flow_score": round(score * 0.85, 4),
                        "error_type": error,
                    },
                    "tokens": {
                        "input": total_tokens // (count * 4),
                        "output": total_tokens // (count * 8),
                    },
                    "latency_ms": 200 + i * 10,
                    "num_tool_calls": 1 if "L0" in level else 3,
                }
            )

    error_dist = {}
    for tr in task_results:
        et = tr["score"]["error_type"]
        if et:
            error_dist[et] = error_dist.get(et, 0) + 1

    return {
        "model": {
            "name": name,
            "litellm_id": f"provider/{name}",
            "provider": provider,
            "supports_tools": True,
        },
        "composition_gap": {
            "model": name,
            "headline_metrics": {
                "overall_accuracy": round(overall, 4),
                "composition_gap_overall": round(gap_overall, 4),
                "composition_gap_L1": round(gap_l1, 4),
                "composition_gap_L2": round(gap_l2, 4),
                "composition_gap_L3": round(gap_l3, 4),
            },
            "per_level_accuracy": {
                "L0_node": round(l0, 4),
                "L1_chain": round(l1, 4),
                "L2_parallel": round(l2, 4),
                "L3_dag": round(l3, 4),
            },
            "diagnostic_metrics": {
                "tool_selection_accuracy": round(overall * 0.95, 4),
                "argument_accuracy": round(overall * 0.9, 4),
                "data_flow_accuracy": round(overall * 0.85, 4),
                "completion_rate": round(min(1.0, overall * 1.1), 4),
                "hallucinated_tool_rate": round(max(0, 0.15 - overall * 0.1), 4),
                "early_termination_rate": round(max(0, 0.2 - overall * 0.15), 4),
            },
            "error_distribution": error_dist,
            "per_tool_l0_accuracy": {},
            "task_counts": {
                "L0_node": 20,
                "L1_chain": 15,
                "L2_parallel": 10,
                "L3_dag": 8,
            },
        },
        "summary": {
            "total_tasks": len(task_results),
            "total_input_tokens": total_tokens,
            "total_output_tokens": total_tokens // 2,
            "total_latency_ms": sum(tr["latency_ms"] for tr in task_results),
            "avg_latency_ms": 250.0,
            "errors": sum(1 for tr in task_results if tr["score"]["error_type"]),
        },
        "start_time": "2026-02-22T00:00:00Z",
        "end_time": "2026-02-22T01:00:00Z",
        "task_results": task_results,
    }


def _make_synthetic_results_json() -> dict:
    """Create a full synthetic results JSON with 5 models."""
    return {
        "benchmark": {
            "name": "CompToolBench",
            "version": "0.1.0",
            "stats": {
                "total_tasks": 53,
                "by_level": {
                    "L0_node": 20,
                    "L1_chain": 15,
                    "L2_parallel": 10,
                    "L3_dag": 8,
                },
                "unique_tools": 30,
            },
        },
        "models": {
            "gpt-4o": _make_model_result(
                "gpt-4o", "openai", 0.95, 0.82, 0.73, 0.58, 80000
            ),
            "claude-sonnet": _make_model_result(
                "claude-sonnet", "anthropic", 0.92, 0.85, 0.76, 0.61, 60000
            ),
            "gemini-flash": _make_model_result(
                "gemini-flash", "google", 0.88, 0.74, 0.65, 0.42, 40000
            ),
            "llama-3.1-8b": _make_model_result(
                "llama-3.1-8b", "ollama", 0.78, 0.55, 0.48, 0.25, 20000
            ),
            "qwen-2.5-7b": _make_model_result(
                "qwen-2.5-7b", "ollama", 0.82, 0.60, 0.50, 0.30, 25000
            ),
        },
        "generated_at": "20260222_120000",
    }


@pytest.fixture()
def synthetic_results_path(tmp_path: Path) -> Path:
    """Write synthetic results to a temp file and return path."""
    path = tmp_path / "results_test.json"
    path.write_text(json.dumps(_make_synthetic_results_json(), indent=2))
    return path


@pytest.fixture()
def results(synthetic_results_path: Path) -> BenchmarkResults:
    """Load synthetic results."""
    return load_results(synthetic_results_path)


# ---------------------------------------------------------------------------
# Tests: bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for the bootstrap_ci function."""

    def test_basic_ci(self):
        """Known scores should produce a CI containing the true mean."""
        scores = np.array([0.8, 0.9, 0.85, 0.75, 0.95, 0.88, 0.92, 0.78])
        point, ci_lo, ci_hi = bootstrap_ci(scores, n_bootstrap=5000)
        assert ci_lo <= point <= ci_hi
        assert abs(point - np.mean(scores)) < 1e-10

    def test_ci_width_increases_with_variance(self):
        """Higher variance should produce wider CIs."""
        low_var = np.array([0.50, 0.51, 0.49, 0.50, 0.50, 0.51, 0.49, 0.50])
        high_var = np.array([0.10, 0.90, 0.20, 0.80, 0.30, 0.70, 0.15, 0.85])

        _, lo1, hi1 = bootstrap_ci(low_var, n_bootstrap=5000)
        _, lo2, hi2 = bootstrap_ci(high_var, n_bootstrap=5000)

        width1 = hi1 - lo1
        width2 = hi2 - lo2
        assert width2 > width1

    def test_ci_narrows_with_more_samples(self):
        """More data points should produce narrower CIs."""
        rng = np.random.default_rng(42)
        small_n = rng.uniform(0.3, 0.7, size=10)
        large_n = rng.uniform(0.3, 0.7, size=200)

        _, lo1, hi1 = bootstrap_ci(small_n, n_bootstrap=5000)
        _, lo2, hi2 = bootstrap_ci(large_n, n_bootstrap=5000)

        assert (hi2 - lo2) < (hi1 - lo1)

    def test_single_value_returns_degenerate(self):
        """Single observation should return degenerate CI (point = lower = upper)."""
        scores = np.array([0.75])
        point, ci_lo, ci_hi = bootstrap_ci(scores)
        assert point == 0.75
        assert ci_lo == 0.75
        assert ci_hi == 0.75

    def test_constant_values_return_degenerate(self):
        """All-same values should return degenerate CI."""
        scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        point, ci_lo, ci_hi = bootstrap_ci(scores)
        assert point == 0.5
        assert ci_lo == 0.5
        assert ci_hi == 0.5

    def test_empty_raises_value_error(self):
        """Empty array should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_ci(np.array([]))

    def test_seed_determinism(self):
        """Same seed should produce identical CIs."""
        scores = np.random.default_rng(99).uniform(0, 1, size=50)
        result1 = bootstrap_ci(scores, seed=123)
        result2 = bootstrap_ci(scores, seed=123)
        assert result1 == result2

    def test_different_seeds_differ(self):
        """Different seeds should (with high probability) produce different CIs."""
        scores = np.random.default_rng(99).uniform(0, 1, size=50)
        result1 = bootstrap_ci(scores, seed=1)
        result2 = bootstrap_ci(scores, seed=2)
        # Point estimates are the same (not affected by seed), but CIs differ
        assert result1[0] == result2[0]
        assert result1[1] != result2[1] or result1[2] != result2[2]

    def test_custom_alpha(self):
        """99% CI should be wider than 90% CI."""
        scores = np.random.default_rng(42).uniform(0.3, 0.7, size=100)
        _, lo_90, hi_90 = bootstrap_ci(scores, alpha=0.10, n_bootstrap=5000)
        _, lo_99, hi_99 = bootstrap_ci(scores, alpha=0.01, n_bootstrap=5000)
        assert (hi_99 - lo_99) > (hi_90 - lo_90)

    def test_binary_scores(self):
        """Should work with binary 0/1 scores (common for accuracy)."""
        scores = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 1])
        point, ci_lo, ci_hi = bootstrap_ci(scores)
        assert 0.0 <= ci_lo <= point <= ci_hi <= 1.0


# ---------------------------------------------------------------------------
# Tests: bootstrap_paired_test
# ---------------------------------------------------------------------------


class TestBootstrapPairedTest:
    """Tests for the paired bootstrap significance test."""

    def test_identical_scores_high_pvalue(self):
        """Identical paired scores should yield p >= 0.05."""
        scores = np.array([0.8, 0.9, 0.85, 0.75, 0.95])
        diff, p_val = bootstrap_paired_test(scores, scores)
        assert abs(diff) < 1e-10
        assert p_val >= 0.05

    def test_clearly_different_scores_low_pvalue(self):
        """Clearly different distributions should yield low p-value."""
        a = np.array([0.9, 0.95, 0.88, 0.92, 0.91, 0.93, 0.89, 0.94, 0.90, 0.92])
        b = np.array([0.3, 0.35, 0.28, 0.32, 0.31, 0.33, 0.29, 0.34, 0.30, 0.32])
        diff, p_val = bootstrap_paired_test(a, b, n_bootstrap=5000)
        assert diff > 0.5
        assert p_val < 0.01

    def test_diff_sign_correct(self):
        """Positive diff means A > B."""
        a = np.array([0.9, 0.8, 0.85])
        b = np.array([0.5, 0.4, 0.45])
        diff, _ = bootstrap_paired_test(a, b)
        assert diff > 0

    def test_mismatched_lengths_raises(self):
        """Different-length arrays should raise ValueError."""
        with pytest.raises(ValueError, match="equal-length"):
            bootstrap_paired_test(np.array([0.5, 0.6]), np.array([0.5]))

    def test_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            bootstrap_paired_test(np.array([]), np.array([]))

    def test_seed_determinism(self):
        """Same seed should produce identical results."""
        a = np.random.default_rng(1).uniform(0.5, 1.0, size=30)
        b = np.random.default_rng(2).uniform(0.3, 0.8, size=30)
        result1 = bootstrap_paired_test(a, b, seed=42)
        result2 = bootstrap_paired_test(a, b, seed=42)
        assert result1 == result2

    def test_pvalue_between_0_and_1(self):
        """p-value should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1, size=20)
        b = rng.uniform(0, 1, size=20)
        _, p_val = bootstrap_paired_test(a, b)
        assert 0.0 <= p_val <= 1.0


# ---------------------------------------------------------------------------
# Tests: cohens_d
# ---------------------------------------------------------------------------


class TestCohensD:
    """Tests for the Cohen's d effect size computation."""

    def test_identical_distributions_zero(self):
        """Same distributions should yield d = 0."""
        scores = np.array([0.5, 0.6, 0.7, 0.4, 0.5])
        assert cohens_d(scores, scores) == 0.0

    def test_known_effect_size(self):
        """Two distributions with known separation should yield expected d."""
        # Two normal-ish distributions separated by 1 sd
        a = np.array([5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9])
        b = np.array([4.0, 4.1, 3.9, 4.2, 3.8, 4.0, 4.1, 3.9])
        d = cohens_d(a, b)
        # Separation is ~1.0 with std ~0.13, so d should be large (>5)
        assert d > 3.0

    def test_sign_reflects_direction(self):
        """Positive d when A > B, negative when A < B."""
        a = np.array([0.9, 0.8, 0.85, 0.95])
        b = np.array([0.3, 0.2, 0.25, 0.35])
        assert cohens_d(a, b) > 0
        assert cohens_d(b, a) < 0

    def test_small_effect(self):
        """Close distributions should have small d."""
        rng = np.random.default_rng(42)
        a = rng.normal(0.50, 0.10, size=100)
        b = rng.normal(0.52, 0.10, size=100)
        d = cohens_d(a, b)
        assert abs(d) < 0.5  # small or negligible

    def test_empty_raises(self):
        """Empty arrays should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            cohens_d(np.array([]), np.array([0.5]))

    def test_single_element_each(self):
        """Two single-element arrays: denom = 0, should return 0 if equal."""
        assert cohens_d(np.array([0.5]), np.array([0.5])) == 0.0

    def test_zero_variance_different_means(self):
        """Zero variance with different means should return inf."""
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([0.0, 0.0, 0.0])
        d = cohens_d(a, b)
        assert d == float("inf")


# ---------------------------------------------------------------------------
# Tests: compute_model_cis (integration)
# ---------------------------------------------------------------------------


class TestComputeModelCIs:
    """Integration tests for compute_model_cis with BenchmarkResults."""

    def test_returns_dataframe(self, results: BenchmarkResults):
        df = compute_model_cis(results, n_bootstrap=500)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_expected_columns(self, results: BenchmarkResults):
        df = compute_model_cis(results, n_bootstrap=500)
        expected = {"model", "level", "accuracy", "ci_lower", "ci_upper", "n_tasks"}
        assert expected.issubset(set(df.columns))

    def test_all_models_present(self, results: BenchmarkResults):
        df = compute_model_cis(results, n_bootstrap=500)
        models_in_df = set(df["model"].unique())
        expected_models = set(results.model_names)
        assert models_in_df == expected_models

    def test_all_levels_plus_overall(self, results: BenchmarkResults):
        df = compute_model_cis(results, n_bootstrap=500)
        expected_levels = set(LEVELS) | {"overall"}
        for model in results.model_names:
            model_levels = set(df.loc[df["model"] == model, "level"].values)
            assert model_levels == expected_levels

    def test_ci_contains_point_estimate(self, results: BenchmarkResults):
        df = compute_model_cis(results, n_bootstrap=500)
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["accuracy"] <= row["ci_upper"]

    def test_n_tasks_is_positive(self, results: BenchmarkResults):
        df = compute_model_cis(results, n_bootstrap=500)
        assert (df["n_tasks"] > 0).all()

    def test_empty_results(self):
        """Empty task_df should return empty DataFrame gracefully."""
        empty_results = BenchmarkResults(
            task_df=pd.DataFrame(),
            level_df=pd.DataFrame(),
            model_df=pd.DataFrame(),
            error_df=pd.DataFrame(),
            raw={},
        )
        df = compute_model_cis(empty_results, n_bootstrap=100)
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: compute_gap_cis
# ---------------------------------------------------------------------------


class TestComputeGapCIs:
    """Tests for composition gap CI computation."""

    def test_returns_dataframe(self, results: BenchmarkResults):
        df = compute_gap_cis(results, n_bootstrap=500)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_expected_columns(self, results: BenchmarkResults):
        df = compute_gap_cis(results, n_bootstrap=500)
        expected = {"model", "gap_level", "gap", "ci_lower", "ci_upper"}
        assert expected.issubset(set(df.columns))

    def test_all_models_present(self, results: BenchmarkResults):
        df = compute_gap_cis(results, n_bootstrap=500)
        models_in_df = set(df["model"].unique())
        expected_models = set(results.model_names)
        assert models_in_df == expected_models

    def test_gap_levels_include_overall(self, results: BenchmarkResults):
        df = compute_gap_cis(results, n_bootstrap=500)
        for model in results.model_names:
            levels = set(df.loc[df["model"] == model, "gap_level"].values)
            assert "overall" in levels
            assert "L1" in levels
            assert "L2" in levels
            assert "L3" in levels

    def test_ci_contains_point_estimate(self, results: BenchmarkResults):
        df = compute_gap_cis(results, n_bootstrap=500)
        for _, row in df.iterrows():
            assert row["ci_lower"] <= row["gap"] <= row["ci_upper"]

    def test_gap_positive_for_degrading_models(self, results: BenchmarkResults):
        """Models with higher L0 than composed levels should have positive gaps."""
        df = compute_gap_cis(results, n_bootstrap=500)
        # All our synthetic models have L0 > Lx, so gaps should be positive
        overall_gaps = df[df["gap_level"] == "overall"]["gap"]
        assert (overall_gaps > 0).all()

    def test_empty_results(self):
        empty_results = BenchmarkResults(
            task_df=pd.DataFrame(),
            level_df=pd.DataFrame(),
            model_df=pd.DataFrame(),
            error_df=pd.DataFrame(),
            raw={},
        )
        df = compute_gap_cis(empty_results, n_bootstrap=100)
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: pairwise_significance
# ---------------------------------------------------------------------------


class TestPairwiseSignificance:
    """Tests for pairwise significance testing."""

    def test_returns_dataframe(self, results: BenchmarkResults):
        df = pairwise_significance(results, n_bootstrap=500)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_expected_columns(self, results: BenchmarkResults):
        df = pairwise_significance(results, n_bootstrap=500)
        expected = {
            "model_a",
            "model_b",
            "diff",
            "p_value",
            "significant",
            "effect_size",
            "effect_interpretation",
        }
        assert expected.issubset(set(df.columns))

    def test_correct_number_of_pairs(self, results: BenchmarkResults):
        """n_models choose 2 pairs should be present."""
        df = pairwise_significance(results, n_bootstrap=500)
        n = results.n_models
        expected_pairs = n * (n - 1) // 2
        assert len(df) == expected_pairs

    def test_pvalues_in_valid_range(self, results: BenchmarkResults):
        df = pairwise_significance(results, n_bootstrap=500)
        assert (df["p_value"] >= 0.0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_significant_is_boolean(self, results: BenchmarkResults):
        df = pairwise_significance(results, n_bootstrap=500)
        assert df["significant"].dtype == bool

    def test_effect_interpretation_valid(self, results: BenchmarkResults):
        df = pairwise_significance(results, n_bootstrap=500)
        valid_interpretations = {"negligible", "small", "medium", "large"}
        assert set(df["effect_interpretation"].unique()).issubset(valid_interpretations)

    def test_bonferroni_correction_applied(self, results: BenchmarkResults):
        """With 5 models, we have 10 pairs, so alpha = 0.05/10 = 0.005.
        Marginally significant pairs (p ~ 0.03) should not be significant."""
        df = pairwise_significance(results, n_bootstrap=500)
        n = results.n_models
        n_comparisons = n * (n - 1) // 2
        adjusted_alpha = 0.05 / n_comparisons

        for _, row in df.iterrows():
            if row["p_value"] >= adjusted_alpha:
                assert not row["significant"]

    def test_single_model_returns_empty(self):
        """Single model should return empty DataFrame."""
        task_data = {
            "model": ["only_model"] * 5,
            "task_id": [f"t{i}" for i in range(5)],
            "level": ["L0_node"] * 5,
            "overall": [0.8] * 5,
        }
        single = BenchmarkResults(
            task_df=pd.DataFrame(task_data),
            level_df=pd.DataFrame(),
            model_df=pd.DataFrame({"model": ["only_model"], "overall_accuracy": [0.8]}),
            error_df=pd.DataFrame(),
            raw={},
        )
        df = pairwise_significance(single, n_bootstrap=100)
        assert df.empty

    def test_empty_results(self):
        empty_results = BenchmarkResults(
            task_df=pd.DataFrame(),
            level_df=pd.DataFrame(),
            model_df=pd.DataFrame(),
            error_df=pd.DataFrame(),
            raw={},
        )
        df = pairwise_significance(empty_results, n_bootstrap=100)
        assert df.empty


# ---------------------------------------------------------------------------
# Tests: generate_statistical_report
# ---------------------------------------------------------------------------


class TestGenerateStatisticalReport:
    """Tests for the markdown report generator."""

    def test_returns_string(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_section_headers(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        assert "# CompToolBench Statistical Analysis Report" in report
        assert "## 1. Per-Model Accuracy" in report
        assert "## 2. Composition Gap" in report
        assert "## 3. Pairwise Significance" in report
        assert "## 4. Notable Effect Sizes" in report

    def test_contains_all_model_names(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        for model in results.model_names:
            assert model in report

    def test_contains_ci_notation(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        # Should contain CI bracket notation like [XX.X%, YY.Y%]
        assert "[" in report and "]" in report
        assert "%" in report

    def test_contains_table_headers(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        assert "| Level |" in report
        assert "| Gap Level |" in report

    def test_contains_pvalue_column(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        assert "p-value" in report

    def test_contains_effect_size_column(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        assert "Cohen" in report

    def test_report_mentions_bonferroni(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        assert "Bonferroni" in report

    def test_report_mentions_bootstrap_count(self, results: BenchmarkResults):
        report = generate_statistical_report(results, n_bootstrap=500)
        assert "500" in report  # n_bootstrap appears in footer


# ---------------------------------------------------------------------------
# Tests: Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_two_model_benchmark(self, tmp_path: Path):
        """Should work with just 2 models."""
        data = {
            "benchmark": {"name": "Test", "version": "0.1.0", "stats": {}},
            "models": {
                "model-a": _make_model_result("model-a", "test", 0.9, 0.8, 0.7, 0.6),
                "model-b": _make_model_result("model-b", "test", 0.7, 0.5, 0.4, 0.3),
            },
            "generated_at": "20260222",
        }
        path = tmp_path / "two_model.json"
        path.write_text(json.dumps(data))
        results = load_results(path)

        ci_df = compute_model_cis(results, n_bootstrap=200)
        assert not ci_df.empty

        gap_df = compute_gap_cis(results, n_bootstrap=200)
        assert not gap_df.empty

        sig_df = pairwise_significance(results, n_bootstrap=200)
        assert len(sig_df) == 1  # Only one pair

        report = generate_statistical_report(results, n_bootstrap=200)
        assert len(report) > 100

    def test_bootstrap_ci_large_n(self):
        """Should handle large arrays efficiently."""
        scores = np.random.default_rng(42).uniform(0, 1, size=5000)
        point, ci_lo, ci_hi = bootstrap_ci(scores, n_bootstrap=1000)
        # Mean of uniform(0,1) is 0.5; CI should be tight around it
        assert abs(point - 0.5) < 0.03
        assert ci_hi - ci_lo < 0.06

    def test_all_perfect_scores(self):
        """All 1.0 scores should have degenerate CI."""
        perfect = np.ones(20)
        point, ci_lo, ci_hi = bootstrap_ci(perfect)
        assert point == 1.0
        assert ci_lo == 1.0
        assert ci_hi == 1.0

    def test_all_zero_scores(self):
        """All 0.0 scores should have degenerate CI."""
        zeros = np.zeros(20)
        point, ci_lo, ci_hi = bootstrap_ci(zeros)
        assert point == 0.0
        assert ci_lo == 0.0
        assert ci_hi == 0.0
