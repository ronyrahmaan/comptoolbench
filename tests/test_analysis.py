"""Tests for the analysis and visualization pipeline.

Uses synthetic benchmark results to verify:
  1. loader.py correctly parses results JSON into DataFrames
  2. All figure functions render without errors
  3. LaTeX table generation produces valid output
  4. Style configuration applies correctly
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

# Use non-interactive backend for CI/headless environments
matplotlib.use("Agg")

from comptoolbench.analysis.figures import (
    fig_accuracy_by_level,
    fig_accuracy_heatmap,
    fig_benchmark_overview,
    fig_composition_gap,
    fig_cost_accuracy,
    fig_degradation_curves,
    fig_diagnostic_radar,
    fig_error_distribution,
    fig_gap_vs_baseline,
    generate_all_figures,
    generate_latex_table,
)
from comptoolbench.analysis.loader import BenchmarkResults, load_results
from comptoolbench.analysis.style import (
    LEVEL_COLORS,
    OKABE_ITO,
    PALETTE,
    apply_style,
    figure_size,
    get_level_color,
    get_model_color,
)

# ---------------------------------------------------------------------------
# Fixtures: Synthetic benchmark results
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
            # Vary scores around the mean
            score = max(0.0, min(1.0, acc + (i % 3 - 1) * 0.05))
            error = None if score > 0.7 else "E4_wrong_arguments"
            if score < 0.3:
                error = "E8_partial_completion"
            task_results.append({
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
            })

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
    """Create a full synthetic results JSON matching runner output."""
    return {
        "benchmark": {
            "name": "CompToolBench",
            "version": "0.1.0",
            "stats": {
                "total_tasks": 53,
                "by_level": {"L0_node": 20, "L1_chain": 15, "L2_parallel": 10, "L3_dag": 8},
                "unique_tools": 30,
            },
        },
        "models": {
            "gpt-4o": _make_model_result("gpt-4o", "openai", 0.95, 0.82, 0.73, 0.58, 80000),
            "claude-sonnet": _make_model_result("claude-sonnet", "anthropic", 0.92, 0.85, 0.76, 0.61, 60000),
            "gemini-flash": _make_model_result("gemini-flash", "google", 0.88, 0.74, 0.65, 0.42, 40000),
            "llama-3.1-8b": _make_model_result("llama-3.1-8b", "ollama", 0.78, 0.55, 0.48, 0.25, 20000),
            "qwen-2.5-7b": _make_model_result("qwen-2.5-7b", "ollama", 0.82, 0.60, 0.50, 0.30, 25000),
        },
        "generated_at": "20260222_120000",
    }


@pytest.fixture
def synthetic_results_path(tmp_path: Path) -> Path:
    """Write synthetic results to a temp file and return path."""
    path = tmp_path / "results_test.json"
    path.write_text(json.dumps(_make_synthetic_results_json(), indent=2))
    return path


@pytest.fixture
def results(synthetic_results_path: Path) -> BenchmarkResults:
    """Load synthetic results."""
    return load_results(synthetic_results_path)


# ---------------------------------------------------------------------------
# Tests: Style
# ---------------------------------------------------------------------------

class TestStyle:
    def test_okabe_ito_has_8_colors(self):
        assert len(OKABE_ITO) == 8

    def test_palette_has_8_colors(self):
        assert len(PALETTE) == 8

    def test_level_colors_has_4_levels(self):
        assert set(LEVEL_COLORS.keys()) == {"L0", "L1", "L2", "L3"}

    def test_apply_style_sets_rcparams(self):
        apply_style()
        assert "serif" in plt.rcParams["font.family"]
        assert plt.rcParams["axes.spines.top"] is False

    def test_figure_size_golden_ratio(self):
        w, h = figure_size(5.5)
        assert abs(w - 5.5) < 0.01
        assert abs(h / w - 0.618) < 0.01

    def test_get_model_color_known(self):
        assert get_model_color("gpt-4o") == OKABE_ITO[4]
        assert get_model_color("claude-opus-4") == OKABE_ITO[5]

    def test_get_model_color_unknown_returns_palette(self):
        color = get_model_color("unknown-model-xyz")
        assert color in PALETTE

    def test_get_level_color(self):
        assert get_level_color("L0") == OKABE_ITO[4]
        assert get_level_color("L3") == OKABE_ITO[5]


# ---------------------------------------------------------------------------
# Tests: Loader
# ---------------------------------------------------------------------------

class TestLoader:
    def test_load_creates_dataframes(self, results: BenchmarkResults):
        assert not results.task_df.empty
        assert not results.model_df.empty
        assert not results.level_df.empty

    def test_model_count(self, results: BenchmarkResults):
        assert results.n_models == 5

    def test_model_names_sorted_by_accuracy(self, results: BenchmarkResults):
        names = results.model_names
        assert len(names) == 5
        # First model should have highest overall accuracy
        df = results.model_df
        top_model = df.loc[df["overall_accuracy"].idxmax(), "model"]
        assert names[0] == top_model

    def test_task_df_has_expected_columns(self, results: BenchmarkResults):
        expected_cols = {
            "model", "task_id", "level", "overall",
            "tool_sequence_score", "argument_score",
        }
        assert expected_cols.issubset(set(results.task_df.columns))

    def test_model_df_has_gap_columns(self, results: BenchmarkResults):
        expected_cols = {
            "model", "provider", "overall_accuracy",
            "gap_overall", "gap_l1", "gap_l2", "gap_l3",
            "accuracy_l0", "accuracy_l1", "accuracy_l2", "accuracy_l3",
        }
        assert expected_cols.issubset(set(results.model_df.columns))

    def test_level_df_has_aggregated_scores(self, results: BenchmarkResults):
        assert "accuracy" in results.level_df.columns
        assert "n_tasks" in results.level_df.columns

    def test_error_df_has_counts(self, results: BenchmarkResults):
        if not results.error_df.empty:
            assert "error_type" in results.error_df.columns
            assert "count" in results.error_df.columns


# ---------------------------------------------------------------------------
# Tests: Figures (verify they render without errors)
# ---------------------------------------------------------------------------

class TestFigures:
    """Test that each figure function produces a valid Figure object."""

    def setup_method(self):
        apply_style()

    def teardown_method(self):
        plt.close("all")

    def test_fig_accuracy_by_level(self, results: BenchmarkResults):
        fig = fig_accuracy_by_level(results)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1

    def test_fig_composition_gap(self, results: BenchmarkResults):
        fig = fig_composition_gap(results)
        assert isinstance(fig, plt.Figure)

    def test_fig_degradation_curves(self, results: BenchmarkResults):
        fig = fig_degradation_curves(results)
        assert isinstance(fig, plt.Figure)

    def test_fig_error_distribution(self, results: BenchmarkResults):
        fig = fig_error_distribution(results)
        assert isinstance(fig, plt.Figure)

    def test_fig_diagnostic_radar(self, results: BenchmarkResults):
        fig = fig_diagnostic_radar(results)
        assert isinstance(fig, plt.Figure)

    def test_fig_cost_accuracy(self, results: BenchmarkResults):
        fig = fig_cost_accuracy(results)
        assert isinstance(fig, plt.Figure)

    def test_fig_accuracy_heatmap(self, results: BenchmarkResults):
        fig = fig_accuracy_heatmap(results)
        assert isinstance(fig, plt.Figure)

    def test_fig_gap_vs_baseline(self, results: BenchmarkResults):
        fig = fig_gap_vs_baseline(results)
        assert isinstance(fig, plt.Figure)

    def test_fig_benchmark_overview(self, results: BenchmarkResults):
        fig = fig_benchmark_overview(results)
        assert isinstance(fig, plt.Figure)

    def test_fig_saves_png_and_pdf(self, results: BenchmarkResults, tmp_path: Path):
        png_path = tmp_path / "test_fig.png"
        fig_accuracy_by_level(results, output_path=png_path)
        assert png_path.exists()
        assert png_path.with_suffix(".pdf").exists()


# ---------------------------------------------------------------------------
# Tests: LaTeX Table
# ---------------------------------------------------------------------------

class TestLatexTable:
    def test_generates_valid_latex(self, results: BenchmarkResults):
        latex = generate_latex_table(results)
        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert r"\toprule" in latex
        assert r"\bottomrule" in latex
        assert r"\textbf{" in latex  # At least one bold value

    def test_contains_all_models(self, results: BenchmarkResults):
        latex = generate_latex_table(results)
        for model in results.model_names:
            assert model in latex

    def test_saves_to_file(self, results: BenchmarkResults, tmp_path: Path):
        path = tmp_path / "table.tex"
        generate_latex_table(results, output_path=path)
        assert path.exists()
        content = path.read_text()
        assert r"\begin{table}" in content


# ---------------------------------------------------------------------------
# Tests: Generate All Figures
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_generate_all_creates_files(self, results: BenchmarkResults, tmp_path: Path):
        outputs = generate_all_figures(results, output_dir=tmp_path)
        assert len(outputs) >= 9  # 9 figures + 1 table
        for name, path in outputs.items():
            assert path.exists(), f"Missing: {name} → {path}"
