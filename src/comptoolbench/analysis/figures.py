"""Publication-quality figure generation for CompToolBench paper.

Generates all figures needed for the benchmark paper:
  - Figure 1: Accuracy by composition level (grouped bars) — the headline figure
  - Figure 2: Composition gap comparison (grouped bars)
  - Figure 3: Accuracy degradation curves (line plot L0→L3)
  - Figure 4: Error type distribution (stacked horizontal bars)
  - Figure 5: Diagnostic radar chart (multi-model capability comparison)
  - Figure 6: Cost-accuracy Pareto frontier (scatter)
  - Figure 7: Model x Level accuracy heatmap
  - Figure 8: Gap vs. L0 accuracy scatter (does baseline predict composability?)
  - Figure 9: Benchmark overview / dataset statistics (NeurIPS D&B requirement)
  - Table 1: LaTeX leaderboard table

All figures follow NeurIPS formatting:
  - 5.5" text width, serif fonts, 300 DPI
  - Okabe-Ito colorblind-safe palette
  - PDF output (vector) with PNG fallback

Usage:
    from comptoolbench.analysis.loader import load_results
    from comptoolbench.analysis.figures import generate_all_figures
    from comptoolbench.analysis.style import apply_style

    apply_style()
    results = load_results("results/run_001/results_20260222.json")
    generate_all_figures(results, output_dir="figures/")
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from comptoolbench.analysis.loader import BenchmarkResults
from comptoolbench.analysis.style import (
    FIG_FULL,
    FIG_HALF,
    GOLDEN,
    PALETTE,
    figure_size,
    get_level_color,
    get_model_color,
)

logger = logging.getLogger(__name__)

# Human-readable level names for axis labels
LEVEL_LABELS: dict[str, str] = {
    "L0_node": "L0\n(Single)",
    "L1_chain": "L1\n(Chain)",
    "L2_parallel": "L2\n(Parallel)",
    "L3_dag": "L3\n(DAG)",
}

LEVEL_ORDER = ["L0_node", "L1_chain", "L2_parallel", "L3_dag"]

# Error type labels (shorter for figures)
ERROR_LABELS: dict[str, str] = {
    "E1_wrong_tool": "Wrong tool",
    "E3_wrong_order": "Wrong order",
    "E4_wrong_arguments": "Wrong args",
    "E6_hallucinated_tool": "Hallucinated",
    "E7_unnecessary_tool": "Extra tool",
    "E8_partial_completion": "Incomplete",
    "E10_format_error": "Format error",
}


# ---------------------------------------------------------------------------
# Helper: sort models by overall accuracy for consistent ordering
# ---------------------------------------------------------------------------

def _sort_models(model_df: pd.DataFrame) -> list[str]:
    """Return model names sorted by overall accuracy descending."""
    return (
        model_df.sort_values("overall_accuracy", ascending=False)["model"].tolist()
    )


def _short_name(model: str) -> str:
    """Shorten model name for tick labels (e.g., 'Llama 3.1 8B' → 'Llama-3.1-8B')."""
    return model.replace(" ", "\n") if len(model) > 15 else model


# ---------------------------------------------------------------------------
# Figure 1: Accuracy by Composition Level (THE headline figure)
# ---------------------------------------------------------------------------

def fig_accuracy_by_level(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: accuracy at L0/L1/L2/L3 for each model.

    This is the paper's headline figure. It shows the "cliff" where
    accuracy drops as composition complexity increases.

    Layout: models on x-axis, bars grouped by level, y-axis = accuracy.
    """
    df = results.model_df.copy()
    models = _sort_models(df)
    n_models = len(models)
    n_levels = 4

    fig, ax = plt.subplots(figsize=figure_size(FIG_FULL, 0.55))

    bar_width = 0.8 / n_levels
    x = np.arange(n_models)

    for i, (level_key, label) in enumerate(LEVEL_LABELS.items()):
        col = f"accuracy_{level_key.split('_')[0].lower()}"
        values = [df.loc[df["model"] == m, col].values[0] for m in models]
        color = get_level_color(level_key.split("_")[0].upper())
        offset = (i - (n_levels - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width * 0.9,
            label=label.replace("\n", " "),
            color=color, edgecolor="white", linewidth=0.3,
            zorder=3,
        )
        # Value annotations on bars
        for bar, val in zip(bars, values, strict=True):
            if val > 0.05:  # Only label visible bars
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.0%}",
                    ha="center", va="bottom",
                    fontsize=5, rotation=90 if n_models > 6 else 0,
                )

    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(m) for m in models], rotation=45, ha="right")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(
        loc="upper right", ncol=4, frameon=True, fancybox=False,
        edgecolor="#CCCCCC", facecolor="white",
    )
    ax.set_title("Accuracy by Composition Level", fontweight="bold", pad=10)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_accuracy_by_level → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 2: Composition Gap Comparison
# ---------------------------------------------------------------------------

def fig_composition_gap(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: composition gap at L1/L2/L3 for each model.

    Shows how much accuracy DROPS when moving from single-tool to composed.
    Higher bars = worse at composition.
    """
    df = results.model_df.copy()
    models = _sort_models(df)
    n_models = len(models)

    fig, ax = plt.subplots(figsize=figure_size(FIG_FULL, 0.50))

    gap_cols = [("gap_l1", "L1 (Chain)"), ("gap_l2", "L2 (Parallel)"), ("gap_l3", "L3 (DAG)")]
    n_gaps = len(gap_cols)
    bar_width = 0.8 / n_gaps
    x = np.arange(n_models)

    colors = [get_level_color("L1"), get_level_color("L2"), get_level_color("L3")]

    for i, ((col, label), color) in enumerate(zip(gap_cols, colors, strict=True)):
        values = [df.loc[df["model"] == m, col].values[0] for m in models]
        offset = (i - (n_gaps - 1) / 2) * bar_width
        ax.bar(
            x + offset, values, bar_width * 0.9,
            label=label, color=color, edgecolor="white", linewidth=0.3,
            zorder=3,
        )

    # Reference line at 0
    ax.axhline(y=0, color="#666666", linewidth=0.6, linestyle="-", zorder=2)

    ax.set_ylabel("Composition Gap")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_name(m) for m in models], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(loc="upper right", ncol=3)
    ax.set_title("Composition Gap by Level", fontweight="bold", pad=10)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_composition_gap → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 3: Accuracy Degradation Curves
# ---------------------------------------------------------------------------

def fig_degradation_curves(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> plt.Figure:
    """Line plot: accuracy L0→L1→L2→L3 per model, showing the drop.

    Each model is a line. The steeper the drop, the worse at composition.
    This gives a visual sense of which models degrade gracefully.
    """
    df = results.model_df.copy()
    models = _sort_models(df)

    fig, ax = plt.subplots(figsize=figure_size(FIG_FULL, GOLDEN))

    levels = ["accuracy_l0", "accuracy_l1", "accuracy_l2", "accuracy_l3"]
    level_labels = ["L0\nSingle", "L1\nChain", "L2\nParallel", "L3\nDAG"]
    x = np.arange(len(levels))

    for model in models:
        row = df[df["model"] == model].iloc[0]
        values = [row[col] for col in levels]
        color = get_model_color(model)
        ax.plot(
            x, values, "o-",
            color=color, label=model,
            markersize=5, linewidth=1.4,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Shade the degradation zone
    ax.fill_between(
        x, 0, 1, alpha=0.03, color="#FF0000",
        zorder=1,
    )

    # Legend outside if many models, inside if few
    if len(models) > 6:
        ax.legend(
            bbox_to_anchor=(1.02, 1), loc="upper left",
            fontsize=6, borderaxespad=0,
        )
    else:
        ax.legend(loc="lower left", fontsize=7)

    ax.set_title("Accuracy Degradation: L0 → L3", fontweight="bold", pad=10)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_degradation_curves → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 4: Error Type Distribution
# ---------------------------------------------------------------------------

def fig_error_distribution(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> plt.Figure:
    """Stacked horizontal bar chart: error types per model.

    Shows WHAT goes wrong, not just that it does. Helps identify whether
    models fail at tool selection, argument passing, ordering, etc.
    """
    df = results.error_df.copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figure_size(FIG_FULL, 0.4))
        ax.text(0.5, 0.5, "No errors recorded", ha="center", va="center", fontsize=10)
        return fig

    models = _sort_models(results.model_df)
    error_types = sorted(df["error_type"].unique())

    fig, ax = plt.subplots(figsize=figure_size(FIG_FULL, max(0.35, 0.08 * len(models))))

    # Pivot: models x error_types
    pivot = df.pivot_table(
        index="model", columns="error_type", values="count",
        fill_value=0, aggfunc="sum",
    ).reindex(models)

    # Ensure all error types are present
    for et in error_types:
        if et not in pivot.columns:
            pivot[et] = 0

    # Stack bars
    left = np.zeros(len(models))
    y = np.arange(len(models))
    colors_iter = iter(PALETTE)

    for et in error_types:
        color = next(colors_iter)
        label = ERROR_LABELS.get(et, et)
        values = pivot[et].values.astype(float)
        ax.barh(
            y, values, left=left, height=0.7,
            label=label, color=color, edgecolor="white", linewidth=0.3,
            zorder=3,
        )
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(models)
    ax.set_xlabel("Number of Errors")
    ax.legend(
        loc="lower right", fontsize=6, ncol=2,
        frameon=True, fancybox=False, edgecolor="#CCCCCC",
    )
    ax.set_title("Error Type Distribution", fontweight="bold", pad=10)
    ax.invert_yaxis()  # Best model on top

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_error_distribution → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 5: Diagnostic Radar Chart
# ---------------------------------------------------------------------------

def fig_diagnostic_radar(
    results: BenchmarkResults,
    top_n: int = 6,
    output_path: Path | None = None,
) -> plt.Figure:
    """Radar/spider chart comparing model capabilities across diagnostics.

    Axes: Tool Selection, Argument Accuracy, Data Flow, Completion Rate,
          1 - Hallucination Rate.
    Each model is a polygon. Useful for understanding failure modes.
    """
    df = results.model_df.copy()
    models = _sort_models(df)[:top_n]

    categories = [
        ("tool_selection", "Tool\nSelection"),
        ("argument_accuracy", "Argument\nAccuracy"),
        ("data_flow", "Data\nFlow"),
        ("completion_rate", "Completion\nRate"),
    ]
    cat_cols = [c[0] for c in categories]
    cat_labels = [c[1] for c in categories]
    n_cats = len(categories)

    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(
        figsize=figure_size(FIG_HALF * 1.3, 1.0),
        subplot_kw={"projection": "polar"},
    )

    for model in models:
        row = df[df["model"] == model].iloc[0]
        values = [row[col] for col in cat_cols]
        values += values[:1]  # Close
        color = get_model_color(model)
        ax.plot(angles, values, "o-", color=color, label=model, linewidth=1.2, markersize=3)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cat_labels, fontsize=7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=5, color="#888888")
    ax.set_rlabel_position(30)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.3, 1.1),
        fontsize=6, frameon=False,
    )
    ax.set_title("Diagnostic Capabilities", fontweight="bold", pad=20, fontsize=9)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_diagnostic_radar → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 6: Cost-Accuracy Pareto Frontier
# ---------------------------------------------------------------------------

def fig_cost_accuracy(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot: overall accuracy vs total tokens (cost proxy).

    Models on the Pareto frontier are highlighted. Shows which models
    give the best accuracy per token — important for practical deployment.
    """
    df = results.model_df.copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figure_size(FIG_HALF, GOLDEN))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figure_size(FIG_HALF, GOLDEN))

    # Plot each model
    for _, row in df.iterrows():
        color = get_model_color(str(row["model"]))
        ax.scatter(
            row["total_tokens"], row["overall_accuracy"],
            color=color, s=50, zorder=3, edgecolors="white", linewidths=0.5,
        )
        ax.annotate(
            str(row["model"]),
            (row["total_tokens"], row["overall_accuracy"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=5.5, color=color, fontweight="bold",
        )

    # Pareto frontier
    sorted_df = df.sort_values("total_tokens")
    pareto_x, pareto_y = [], []
    best_acc = -1.0
    for _, row in sorted_df.iterrows():
        if row["overall_accuracy"] > best_acc:
            pareto_x.append(row["total_tokens"])
            pareto_y.append(row["overall_accuracy"])
            best_acc = row["overall_accuracy"]

    if len(pareto_x) > 1:
        ax.plot(
            pareto_x, pareto_y, "--",
            color="#AAAAAA", linewidth=0.8, zorder=2,
            label="Pareto frontier",
        )

    ax.set_xlabel("Total Tokens (cost proxy)")
    ax.set_ylabel("Overall Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # Log scale for tokens if range is large
    if df["total_tokens"].max() > 10 * df["total_tokens"].min() > 0:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    if pareto_x:
        ax.legend(loc="lower right", fontsize=6)

    ax.set_title("Accuracy vs. Cost", fontweight="bold", pad=10)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_cost_accuracy → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 7: Model x Level Heatmap
# ---------------------------------------------------------------------------

def fig_accuracy_heatmap(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> plt.Figure:
    """Heatmap: rows = models, columns = levels, cells = accuracy.

    Color intensity shows accuracy (green = high, red = low).
    Annotated with exact percentages. Good for at-a-glance comparison.
    """
    df = results.model_df.copy()
    models = _sort_models(df)

    level_cols = ["accuracy_l0", "accuracy_l1", "accuracy_l2", "accuracy_l3"]
    level_labels = ["L0 (Single)", "L1 (Chain)", "L2 (Parallel)", "L3 (DAG)"]

    matrix = np.array([
        [df.loc[df["model"] == m, col].values[0] for col in level_cols]
        for m in models
    ])

    fig, ax = plt.subplots(
        figsize=(FIG_HALF * 1.2, max(1.5, 0.35 * len(models))),
    )

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    # Annotations
    for i in range(len(models)):
        for j in range(len(level_cols)):
            val = matrix[i, j]
            text_color = "white" if val < 0.35 or val > 0.85 else "black"
            ax.text(
                j, i, f"{val:.0%}",
                ha="center", va="center",
                fontsize=7, fontweight="bold", color=text_color,
            )

    ax.set_xticks(range(len(level_labels)))
    ax.set_xticklabels(level_labels, fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Accuracy", fontsize=7)

    ax.set_title("Accuracy Heatmap", fontweight="bold", pad=10)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_accuracy_heatmap → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 8: Gap vs. L0 Accuracy Scatter
# ---------------------------------------------------------------------------

def fig_gap_vs_baseline(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> plt.Figure:
    """Scatter plot: composition gap vs. L0 accuracy.

    Tests the hypothesis: "Do stronger models have smaller composition gaps?"
    Each point is a model, annotated with name. Linear trend line overlaid.
    Source: Standard correlation analysis figure in ML benchmark papers.
    """
    df = results.model_df.copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figure_size(FIG_HALF, GOLDEN))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figure_size(FIG_HALF * 1.3, 0.85))

    l0_accs = df["accuracy_l0"].values
    gaps = df["gap_overall"].values

    for _, row in df.iterrows():
        color = get_model_color(str(row["model"]))
        ax.scatter(
            row["accuracy_l0"], row["gap_overall"],
            color=color, s=55, zorder=4,
            edgecolors="white", linewidths=0.5,
        )
        ax.annotate(
            _short_name(str(row["model"])).replace("\n", " "),
            (row["accuracy_l0"], row["gap_overall"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=5.5, color=color, fontweight="bold", zorder=5,
        )

    # Linear trend line (if enough points)
    valid = ~np.isnan(l0_accs) & ~np.isnan(gaps)
    if valid.sum() >= 3:
        z = np.polyfit(l0_accs[valid], gaps[valid], 1)
        p = np.poly1d(z)
        x_range = np.linspace(l0_accs[valid].min(), l0_accs[valid].max(), 100)
        ax.plot(
            x_range, p(x_range), "--",
            color="#999999", linewidth=0.8, alpha=0.7,
            label="Linear fit", zorder=3,
        )

    ax.axhline(y=0, color="#AAAAAA", linewidth=0.6, linestyle=":", zorder=2)
    ax.set_xlabel("L0 Accuracy (Individual Tool Use)")
    ax.set_ylabel("Composition Gap (Overall)")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.legend(fontsize=6, frameon=False)
    ax.set_title("Does Baseline Accuracy Predict Composability?", fontweight="bold", pad=10)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_gap_vs_baseline → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Figure 9: Benchmark Overview / Dataset Statistics
# ---------------------------------------------------------------------------

def fig_benchmark_overview(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> plt.Figure:
    """Multi-panel figure showing benchmark dataset statistics.

    Required by NeurIPS Datasets & Benchmarks track checklist.
    Panel (a): Task count per level. Panel (b): Avg steps per level.
    """
    from comptoolbench.analysis.style import LEVEL_COLORS as LC

    # Get task counts from the data
    task_df = results.task_df
    if task_df.empty:
        fig, ax = plt.subplots(figsize=figure_size(FIG_FULL, 0.4))
        ax.text(0.5, 0.5, "No task data", ha="center", va="center")
        return fig

    # Count tasks per level (from first model — all models have same tasks)
    first_model = task_df["model"].iloc[0]
    model_tasks = task_df[task_df["model"] == first_model]
    level_counts = model_tasks.groupby("level").size()

    levels = ["L0_node", "L1_chain", "L2_parallel", "L3_dag"]
    level_short = ["L0", "L1", "L2", "L3"]
    colors = [LC.get(ls, PALETTE[0]) for ls in level_short]

    fig, axes = plt.subplots(1, 2, figsize=figure_size(FIG_FULL, 0.42))

    # Panel (a): Task count per level
    ax = axes[0]
    counts = [level_counts.get(lv, 0) for lv in levels]
    bars = ax.bar(
        level_short, counts, color=colors,
        edgecolor="white", linewidth=0.3, width=0.6, zorder=3,
    )
    for bar, val in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(val), ha="center", va="bottom",
            fontsize=7, fontweight="bold",
        )
    ax.set_ylabel("Number of Tasks")
    ax.set_title(f"(a) Tasks per Level (Total: {sum(counts)})", fontsize=8, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.2 if counts else 10)

    # Panel (b): Average score per level across all models
    ax = axes[1]
    avg_scores = [
        model_tasks[model_tasks["level"] == lv]["overall"].mean()
        for lv in levels
    ]
    bars2 = ax.bar(
        level_short, avg_scores, color=colors,
        edgecolor="white", linewidth=0.3, width=0.6, zorder=3,
    )
    for bar, val in zip(bars2, avg_scores, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.0%}", ha="center", va="bottom",
            fontsize=7, fontweight="bold",
        )
    ax.set_ylabel("Average Accuracy")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_title("(b) Average Accuracy per Level", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.15)

    fig.suptitle("CompToolBench — Dataset Statistics", fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        fig.savefig(output_path.with_suffix(".pdf"))
        logger.info("Saved fig_benchmark_overview → %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# Table: LaTeX Leaderboard
# ---------------------------------------------------------------------------

def generate_latex_table(
    results: BenchmarkResults,
    output_path: Path | None = None,
) -> str:
    """Generate a LaTeX table for the paper's main results.

    Format matches NeurIPS table style with booktabs.
    Bold-faces the best value in each column.
    """
    df = results.model_df.copy()
    models = _sort_models(df)

    cols = [
        ("accuracy_l0", "L0", True),
        ("accuracy_l1", "L1", True),
        ("accuracy_l2", "L2", True),
        ("accuracy_l3", "L3", True),
        ("overall_accuracy", "Overall", True),
        ("gap_overall", "Gap↓", False),  # Lower is better
    ]

    # Find best values for bolding
    best: dict[str, float] = {}
    for col, _, higher_better in cols:
        vals = df[col].values
        best[col] = float(vals.max() if higher_better else vals.min())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{CompToolBench leaderboard. Best values \textbf{bolded}. "
        r"Gap = min(L0 per-tool) $-$ composed accuracy (lower is better).}",
        r"\label{tab:leaderboard}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        r"Model & " + " & ".join(label for _, label, _ in cols) + r" \\",
        r"\midrule",
    ]

    for model in models:
        row = df[df["model"] == model].iloc[0]
        cells = [model]
        for col, _, _ in cols:
            val = row[col]
            formatted = f"{val:.1%}"
            if abs(val - best[col]) < 1e-4:
                formatted = r"\textbf{" + formatted + "}"
            cells.append(formatted)
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(latex)
        logger.info("Saved LaTeX table → %s", output_path)

    return latex


# ---------------------------------------------------------------------------
# Generate ALL figures at once
# ---------------------------------------------------------------------------

def generate_all_figures(
    results: BenchmarkResults,
    output_dir: str | Path = "figures",
) -> dict[str, Path]:
    """Generate all paper figures and save to output_dir.

    Args:
        results: Loaded benchmark results.
        output_dir: Directory to save figures (created if needed).

    Returns:
        Dict mapping figure name to output path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # Figure 1: Headline accuracy bars
    fig_accuracy_by_level(results, out / "fig1_accuracy_by_level.png")
    outputs["accuracy_by_level"] = out / "fig1_accuracy_by_level.pdf"
    plt.close()

    # Figure 2: Composition gap
    fig_composition_gap(results, out / "fig2_composition_gap.png")
    outputs["composition_gap"] = out / "fig2_composition_gap.pdf"
    plt.close()

    # Figure 3: Degradation curves
    fig_degradation_curves(results, out / "fig3_degradation_curves.png")
    outputs["degradation_curves"] = out / "fig3_degradation_curves.pdf"
    plt.close()

    # Figure 4: Error distribution
    fig_error_distribution(results, out / "fig4_error_distribution.png")
    outputs["error_distribution"] = out / "fig4_error_distribution.pdf"
    plt.close()

    # Figure 5: Diagnostic radar
    fig_diagnostic_radar(results, output_path=out / "fig5_diagnostic_radar.png")
    outputs["diagnostic_radar"] = out / "fig5_diagnostic_radar.pdf"
    plt.close()

    # Figure 6: Cost-accuracy
    fig_cost_accuracy(results, out / "fig6_cost_accuracy.png")
    outputs["cost_accuracy"] = out / "fig6_cost_accuracy.pdf"
    plt.close()

    # Figure 7: Heatmap
    fig_accuracy_heatmap(results, out / "fig7_accuracy_heatmap.png")
    outputs["accuracy_heatmap"] = out / "fig7_accuracy_heatmap.pdf"
    plt.close()

    # Figure 8: Gap vs baseline scatter
    fig_gap_vs_baseline(results, out / "fig8_gap_vs_baseline.png")
    outputs["gap_vs_baseline"] = out / "fig8_gap_vs_baseline.pdf"
    plt.close()

    # Figure 9: Benchmark overview
    fig_benchmark_overview(results, out / "fig9_benchmark_overview.png")
    outputs["benchmark_overview"] = out / "fig9_benchmark_overview.pdf"
    plt.close()

    # Table 1: LaTeX leaderboard
    latex_path = out / "table1_leaderboard.tex"
    generate_latex_table(results, latex_path)
    outputs["leaderboard_table"] = latex_path

    logger.info("All figures generated in %s", out)
    return outputs
