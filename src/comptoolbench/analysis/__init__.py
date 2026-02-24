"""Analysis and visualization pipeline for CompToolBench results.

Usage:
    from comptoolbench.analysis import load_results, generate_all_figures, apply_style

    apply_style()
    results = load_results("results/run_001/results_20260222.json")
    generate_all_figures(results, output_dir="figures/")

    # Statistical rigor
    from comptoolbench.analysis import compute_model_cis, pairwise_significance
    ci_df = compute_model_cis(results)
    sig_df = pairwise_significance(results)

    # Deep failure analysis
    from comptoolbench.analysis import generate_failure_report
    report = generate_failure_report(Path("results/run_001"), results)
"""

from comptoolbench.analysis.failure_analysis import (
    analyze_error_cascade,
    analyze_error_patterns_by_level,
    analyze_scaling,
    analyze_step_position_failures,
    analyze_tool_difficulty,
    generate_failure_report,
)
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
from comptoolbench.analysis.statistical import (
    bootstrap_ci,
    bootstrap_paired_test,
    cohens_d,
    compute_gap_cis,
    compute_model_cis,
    generate_statistical_report,
    pairwise_significance,
)
from comptoolbench.analysis.style import apply_style

__all__ = [
    "BenchmarkResults",
    "analyze_error_cascade",
    "analyze_error_patterns_by_level",
    "analyze_scaling",
    "analyze_step_position_failures",
    "analyze_tool_difficulty",
    "apply_style",
    "bootstrap_ci",
    "bootstrap_paired_test",
    "cohens_d",
    "compute_gap_cis",
    "compute_model_cis",
    "fig_accuracy_by_level",
    "fig_accuracy_heatmap",
    "fig_benchmark_overview",
    "fig_composition_gap",
    "fig_cost_accuracy",
    "fig_degradation_curves",
    "fig_diagnostic_radar",
    "fig_error_distribution",
    "fig_gap_vs_baseline",
    "generate_all_figures",
    "generate_failure_report",
    "generate_latex_table",
    "generate_statistical_report",
    "load_results",
    "pairwise_significance",
]
