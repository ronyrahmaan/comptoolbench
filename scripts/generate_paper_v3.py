#!/usr/bin/env python3
"""Generate all paper figures and tables from V3 benchmark results.

Reads checkpoint files from multiple result directories, computes metrics,
and generates publication-quality figures + LaTeX tables for the paper.

Usage:
    uv run python scripts/generate_paper_v3.py

Outputs go to paper/figures/ and paper/tables/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comptoolbench.evaluation.metrics import compute_composition_gap, CompositionGapResult
from comptoolbench.evaluation.model_adapter import AVAILABLE_MODELS
from comptoolbench.evaluation.scorers import CallScore, TaskScore
from comptoolbench.tasks.models import CompositionLevel, Task

# ── Publication settings ──────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Okabe-Ito colorblind-safe palette
COLORS = {
    "L0": "#0072B2",
    "L1": "#009E73",
    "L2": "#E69F00",
    "L3": "#D55E00",
    "gap": "#CC79A7",
    "cloud": "#56B4E9",
    "local": "#999999",
}

# ── Configuration ─────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent.parent
PAPER_DIR = PROJECT_DIR / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"

# V3 result directories (all use the same task_suite.json with same hash)
V3_RESULT_DIRS = [
    PROJECT_DIR / "results" / "unified_v3",
    PROJECT_DIR / "results" / "cerebras_v2",
    PROJECT_DIR / "results" / "cohere_v2",
    PROJECT_DIR / "results" / "local_v3",
]

# Task suite (any directory works — they all have the same suite)
TASK_SUITE_PATH = PROJECT_DIR / "results" / "unified_v3" / "task_suite.json"

# Cloud model keys (for cloud/local split analysis)
CLOUD_PROVIDERS = {"groq", "mistral", "openrouter", "cerebras", "cohere", "sambanova", "gemini"}

# Short display names for figures (full names are too long for x-axis)
SHORT_NAMES = {
    "Llama 3.1 8B (Groq)": "Llama 8B\n(Groq)",
    "Llama 3.1 8B (Cerebras)": "Llama 8B\n(Cerebras)",
    "Llama 4 Scout 17B (Groq)": "Scout 17B\n(Groq)",
    "Mistral Small": "Mistral\nSmall",
    "Mistral Medium": "Mistral\nMedium",
    "Mistral Large": "Mistral\nLarge",
    "Gemini 2.0 Flash (OpenRouter)": "Gemini\nFlash",
    "GPT-OSS 120B (Cerebras)": "GPT-OSS\n120B",
    "Command A": "Command A",
    "Command R+": "Command R+",
    # Local models
    "Qwen3 8B": "Qwen3 8B",
    "Qwen 2.5 7B": "Qwen 2.5 7B",
    "Mistral Nemo 12B": "Nemo 12B",
    "Mistral 7B": "Mistral 7B",
    "Mistral Small 24B": "Mistral\nSmall 24B",
    "Llama 3.1 8B": "Llama 8B",
    "Granite4 3B": "Granite4 3B",
    "Granite4 1B": "Granite4 1B",
}


def short_name(full_name: str) -> str:
    """Get short display name for figures."""
    return SHORT_NAMES.get(full_name, full_name)


# ── Data Loading ──────────────────────────────────────────────────────

def load_tasks(suite_path: Path) -> list[Task]:
    """Load tasks from a task suite JSON file."""
    data = json.loads(suite_path.read_text())
    return [Task.model_validate(t) for t in data["tasks"]]


def load_checkpoint(cp_path: Path) -> list[dict]:
    """Load entries from a checkpoint JSONL file."""
    entries = []
    for line in cp_path.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def build_task_scores(tasks: list[Task], entries: list[dict]) -> list[TaskScore]:
    """Reconstruct TaskScore objects from checkpoint entries."""
    entry_map = {e["task_id"]: e for e in entries}
    scores = []
    for task in tasks:
        entry = entry_map.get(task.task_id)
        if entry is None:
            scores.append(TaskScore(
                task_id=task.task_id, level=task.level, call_scores=[], overall=0.0,
            ))
            continue

        sd = entry["score"]
        call_scores = [
            CallScore(
                step_id=cs.get("step", ""),
                expected_tool=cs.get("expected_tool", ""),
                actual_tool=cs.get("actual_tool"),
                tool_correct=cs.get("tool_correct", False),
                args_score=cs.get("args_score", 0.0),
                matched=cs.get("matched", False),
            )
            for cs in sd.get("call_scores", [])
        ]
        scores.append(TaskScore(
            task_id=sd["task_id"],
            level=CompositionLevel(sd["level"]),
            call_scores=call_scores,
            overall=sd["overall"],
            tool_sequence_score=sd.get("tool_sequence_score", 0.0),
            argument_score=sd.get("argument_score", 0.0),
            completeness_score=sd.get("completeness_score", 0.0),
            data_flow_score=sd.get("data_flow_score", 0.0),
            error_type=sd.get("error_type"),
        ))
    return scores


def load_all_results(tasks: list[Task]) -> dict[str, dict]:
    """Load results from all V3 directories.

    Returns: {model_key: {"config": ModelConfig, "gap": CompositionGapResult,
              "entries": [...], "task_scores": [...]}}
    """
    results = {}
    for results_dir in V3_RESULT_DIRS:
        if not results_dir.exists():
            print(f"  [skip] {results_dir} does not exist")
            continue

        for cp_path in sorted(results_dir.glob("checkpoint_*.jsonl")):
            model_key = cp_path.stem.replace("checkpoint_", "")
            config = AVAILABLE_MODELS.get(model_key)
            if config is None:
                print(f"  [skip] Unknown model key: {model_key}")
                continue

            entries = load_checkpoint(cp_path)
            if len(entries) != len(tasks):
                print(f"  [skip] {model_key}: {len(entries)}/{len(tasks)} tasks (incomplete)")
                continue

            # Don't duplicate models
            if model_key in results:
                print(f"  [skip] {model_key}: already loaded from another directory")
                continue

            task_scores = build_task_scores(tasks, entries)
            gap = compute_composition_gap(config.name, tasks, task_scores)

            # Compute summary stats
            total_tokens = sum(
                e.get("tokens", {}).get("input", 0) + e.get("tokens", {}).get("output", 0)
                for e in entries
            )
            avg_latency = sum(e.get("latency_ms", 0) for e in entries) / len(entries)

            results[model_key] = {
                "config": config,
                "gap": gap,
                "entries": entries,
                "task_scores": task_scores,
                "total_tokens": total_tokens,
                "avg_latency_ms": avg_latency,
            }
            print(f"  [ok] {config.name} ({config.provider.value}): "
                  f"overall={gap.overall_accuracy:.1%}, L0={gap.accuracy_l0:.1%}")

    return results


# ── Analysis helpers ──────────────────────────────────────────────────

def is_cloud(config) -> bool:
    """Check if a model config is a cloud provider."""
    return config.provider.value in CLOUD_PROVIDERS


def sort_models(results: dict) -> list[str]:
    """Sort models: cloud by accuracy desc, then local by accuracy desc."""
    cloud_keys = [k for k, r in results.items() if is_cloud(r["config"])]
    local_keys = [k for k, r in results.items() if not is_cloud(r["config"])]
    cloud_keys.sort(key=lambda k: -results[k]["gap"].overall_accuracy)
    local_keys.sort(key=lambda k: -results[k]["gap"].overall_accuracy)
    return cloud_keys + local_keys


def compute_selection_gap(gap: CompositionGapResult) -> dict:
    """Check if model exhibits Selection Gap (L0 < all of L1, L2, L3)."""
    l0 = gap.accuracy_l0
    composed_min = min(gap.accuracy_l1, gap.accuracy_l2, gap.accuracy_l3)
    composed_avg = (gap.accuracy_l1 + gap.accuracy_l2 + gap.accuracy_l3) / 3
    return {
        "l0": l0,
        "composed_min": composed_min,
        "composed_avg": composed_avg,
        "selection_gap_strict": l0 < composed_min,  # L0 < min(L1,L2,L3)
        "selection_gap_avg": l0 < composed_avg,  # L0 < avg(L1,L2,L3)
        "gap_from_avg": composed_avg - l0,  # how much worse L0 is
    }


# ── Figure Generation ─────────────────────────────────────────────────

def fig_accuracy_by_level(results: dict, ordered_keys: list[str]) -> None:
    """Bar chart: accuracy at each level for all models."""
    n = len(ordered_keys)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.75), 3.8))

    x = np.arange(n)
    w = 0.19
    levels = [
        ("L$_0$", "accuracy_l0", COLORS["L0"]),
        ("L$_1$", "accuracy_l1", COLORS["L1"]),
        ("L$_2$", "accuracy_l2", COLORS["L2"]),
        ("L$_3$", "accuracy_l3", COLORS["L3"]),
    ]

    for i, (label, attr, color) in enumerate(levels):
        vals = [getattr(results[k]["gap"], attr) * 100 for k in ordered_keys]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color, edgecolor="white", linewidth=0.5)

    # Add cloud/local separator
    cloud_count = sum(1 for k in ordered_keys if is_cloud(results[k]["config"]))
    if 0 < cloud_count < n:
        ax.axvline(cloud_count - 0.5, color="#333", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(cloud_count / 2, 96, "Cloud", ha="center", fontsize=7, color="#555", style="italic")
        ax.text((cloud_count + n) / 2, 96, "Local", ha="center", fontsize=7, color="#555", style="italic")

    names = [short_name(results[k]["gap"].model_name) for k in ordered_keys]
    ax.set_xticks(x)
    ax.set_xticklabels(names, ha="center", fontsize=7.5, linespacing=0.9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 102)
    ax.legend(loc="upper right", ncol=4, framealpha=0.9, fontsize=7.5)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig1_accuracy_by_level.{fmt}")
    plt.close(fig)
    print("  [fig] fig1_accuracy_by_level")


def fig_degradation_curves(results: dict, ordered_keys: list[str]) -> None:
    """Line chart: accuracy trajectory from L0 → L3 for each model."""
    n = len(ordered_keys)
    fig, ax = plt.subplots(figsize=(5.5, 4))

    levels = ["L$_0$", "L$_1$", "L$_2$", "L$_3$"]
    x = np.arange(4)

    cloud_keys = [k for k in ordered_keys if is_cloud(results[k]["config"])]
    local_keys = [k for k in ordered_keys if not is_cloud(results[k]["config"])]

    # Use distinct colormaps for cloud vs local
    cloud_cm = plt.cm.Blues(np.linspace(0.4, 0.9, max(len(cloud_keys), 1)))
    local_cm = plt.cm.Oranges(np.linspace(0.3, 0.8, max(len(local_keys), 1)))

    for i, k in enumerate(cloud_keys):
        gap = results[k]["gap"]
        vals = [gap.accuracy_l0 * 100, gap.accuracy_l1 * 100, gap.accuracy_l2 * 100, gap.accuracy_l3 * 100]
        label = short_name(gap.model_name).replace("\n", " ")
        ax.plot(x, vals, "o-", color=cloud_cm[i], label=label, markersize=4, linewidth=1.5)

    for i, k in enumerate(local_keys):
        gap = results[k]["gap"]
        vals = [gap.accuracy_l0 * 100, gap.accuracy_l1 * 100, gap.accuracy_l2 * 100, gap.accuracy_l3 * 100]
        label = short_name(gap.model_name).replace("\n", " ")
        ax.plot(x, vals, "s--", color=local_cm[i], label=label, markersize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Composition Level")
    ax.set_ylim(0, 100)
    # Use 2 columns for legend if many models
    ncol = 2 if n > 8 else 1
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, framealpha=0.9, ncol=ncol)
    ax.grid(alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig3_degradation_curves.{fmt}")
    plt.close(fig)
    print("  [fig] fig3_degradation_curves")


def fig_selection_gap(results: dict, ordered_keys: list[str]) -> None:
    """Bar chart comparing L0 accuracy vs. avg(L1,L2,L3) — the Selection Gap."""
    n = len(ordered_keys)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.75), 3.8))

    x = np.arange(n)
    w = 0.35

    l0_vals = []
    composed_vals = []
    names = []
    gap_present = []

    for k in ordered_keys:
        gap = results[k]["gap"]
        sg = compute_selection_gap(gap)
        l0_vals.append(sg["l0"] * 100)
        composed_vals.append(sg["composed_avg"] * 100)
        names.append(short_name(gap.model_name))
        gap_present.append(sg["selection_gap_avg"])

    ax.bar(x - w/2, l0_vals, w, label="L$_0$ (Single Tool)", color=COLORS["L0"],
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w/2, composed_vals, w, label="Avg(L$_1$, L$_2$, L$_3$)", color=COLORS["L2"],
           edgecolor="white", linewidth=0.5)

    # Draw gap arrows for models with Selection Gap
    for i, has_gap in enumerate(gap_present):
        if has_gap:
            gap_size = composed_vals[i] - l0_vals[i]
            ax.annotate(
                "", xy=(x[i], composed_vals[i] - 1), xytext=(x[i], l0_vals[i] + 1),
                arrowprops=dict(arrowstyle="->", color=COLORS["L3"], lw=1.5),
            )
            ax.text(x[i] + 0.25, (l0_vals[i] + composed_vals[i]) / 2,
                    f"+{gap_size:.0f}", fontsize=6, color=COLORS["L3"],
                    ha="left", va="center", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, ha="center", fontsize=7.5, linespacing=0.9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig_selection_gap.{fmt}")
    plt.close(fig)
    print("  [fig] fig_selection_gap")


def fig_heatmap(results: dict, ordered_keys: list[str]) -> None:
    """Heatmap of accuracy: models × levels."""
    names = [short_name(results[k]["gap"].model_name).replace("\n", " ") for k in ordered_keys]
    levels = ["L$_0$", "L$_1$", "L$_2$", "L$_3$", "Overall"]
    data = []
    for k in ordered_keys:
        g = results[k]["gap"]
        data.append([g.accuracy_l0, g.accuracy_l1, g.accuracy_l2, g.accuracy_l3, g.overall_accuracy])

    data = np.array(data) * 100
    n = len(names)

    fig, ax = plt.subplots(figsize=(5, max(3.5, n * 0.32)))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=7)

    # Annotate cells
    for i in range(n):
        for j in range(len(levels)):
            val = data[i, j]
            color = "white" if val < 35 or val > 80 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=7, color=color,
                    fontweight="bold" if j == 4 else "normal")

    # Cloud/local separator
    cloud_count = sum(1 for k in ordered_keys if is_cloud(results[k]["config"]))
    if 0 < cloud_count < n:
        ax.axhline(cloud_count - 0.5, color="black", linewidth=1.5)
        ax.text(-0.8, (cloud_count - 1) / 2, "Cloud", ha="right", va="center",
                fontsize=6, color="#555", style="italic", rotation=90)
        ax.text(-0.8, (cloud_count + n - 1) / 2, "Local", ha="right", va="center",
                fontsize=6, color="#555", style="italic", rotation=90)

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig7_accuracy_heatmap.{fmt}")
    plt.close(fig)
    print("  [fig] fig7_accuracy_heatmap")


def fig_gap_vs_baseline(results: dict, ordered_keys: list[str]) -> None:
    """Scatter: L0 accuracy vs. overall accuracy."""
    fig, ax = plt.subplots(figsize=(4.5, 4))

    for k in ordered_keys:
        g = results[k]["gap"]
        c = results[k]["config"]
        color = COLORS["cloud"] if is_cloud(c) else COLORS["local"]
        marker = "o" if is_cloud(c) else "s"
        ax.scatter(g.accuracy_l0 * 100, g.overall_accuracy * 100,
                   c=color, marker=marker, s=50, zorder=3, edgecolors="white", linewidth=0.5)
        sn = short_name(g.model_name).replace("\n", " ")
        ax.annotate(sn, (g.accuracy_l0 * 100 + 0.5, g.overall_accuracy * 100 + 0.5),
                    fontsize=5.5, alpha=0.8)

    # Diagonal (perfect transfer)
    ax.plot([0, 100], [0, 100], "--", color="#aaa", linewidth=0.8, label="L$_0$ = Overall")

    # Add cloud/local legend markers
    ax.scatter([], [], c=COLORS["cloud"], marker="o", s=40, label="Cloud")
    ax.scatter([], [], c=COLORS["local"], marker="s", s=40, label="Local")

    ax.set_xlabel("L$_0$ Accuracy (%)")
    ax.set_ylabel("Overall Accuracy (%)")
    # Dynamic axis limits based on data
    all_l0 = [results[k]["gap"].accuracy_l0 * 100 for k in ordered_keys]
    all_ov = [results[k]["gap"].overall_accuracy * 100 for k in ordered_keys]
    lo = max(0, min(min(all_l0), min(all_ov)) - 10)
    hi = min(100, max(max(all_l0), max(all_ov)) + 10)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    ax.grid(alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig8_gap_vs_baseline.{fmt}")
    plt.close(fig)
    print("  [fig] fig8_gap_vs_baseline")


def fig_error_distribution(results: dict, ordered_keys: list[str]) -> None:
    """Stacked bar: error type distribution by level (averaged across models)."""
    # Aggregate error types by level across all models
    level_errors: dict[str, dict[str, int]] = {
        "L0": {}, "L1": {}, "L2": {}, "L3": {},
    }
    level_totals: dict[str, int] = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}

    level_map = {
        "L0_node": "L0", "L1_chain": "L1", "L2_parallel": "L2", "L3_dag": "L3",
    }

    for k in ordered_keys:
        for entry in results[k]["entries"]:
            sd = entry["score"]
            level_str = level_map.get(sd["level"], sd["level"])
            et = sd.get("error_type")
            level_totals[level_str] = level_totals.get(level_str, 0) + 1
            if et:
                level_errors[level_str][et] = level_errors[level_str].get(et, 0) + 1

    # Simplify error types (match actual strings from scorer output)
    simple_map = {
        "E1_no_calls": "No calls",
        "E2_wrong_tool": "Wrong tool",
        "E3_extra_calls": "Extra calls",
        "E3_wrong_order": "Wrong order",
        "E4_wrong_args": "Wrong args",
        "E4_wrong_arguments": "Wrong args",
        "E5_missing_args": "Missing args",
        "E6_hallucinated_tool": "Hallucinated",
        "E7_wrong_sequence": "Wrong sequence",
        "E7_unnecessary_tool": "Unnecessary tool",
        "E8_partial_completion": "Incomplete",
        "E9_data_flow_error": "Data flow",
        "E10_format_error": "Format error",
    }

    # Get all error types
    all_errors = set()
    for errs in level_errors.values():
        all_errors.update(errs.keys())

    # Build data
    fig, ax = plt.subplots(figsize=(6, 4))
    levels = ["L0", "L1", "L2", "L3"]
    x = np.arange(4)
    bottom = np.zeros(4)

    error_colors = plt.cm.Set3(np.linspace(0, 1, len(all_errors)))

    for i, et in enumerate(sorted(all_errors)):
        vals = []
        for lv in levels:
            total = level_totals[lv]
            count = level_errors[lv].get(et, 0)
            vals.append(count / total * 100 if total > 0 else 0)
        vals = np.array(vals)
        label = simple_map.get(et, et)
        ax.bar(x, vals, bottom=bottom, label=label, color=error_colors[i], width=0.6)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(["L$_0$", "L$_1$", "L$_2$", "L$_3$"])
    ax.set_ylabel("% of Tasks with Error")
    ax.set_xlabel("Composition Level")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig4_error_distribution.{fmt}")
    plt.close(fig)
    print("  [fig] fig4_error_distribution")


def fig_composition_gap_bars(results: dict, ordered_keys: list[str]) -> None:
    """Horizontal bars: composition gap (L0 - L3 delta) per model."""
    n = len(ordered_keys)
    fig, ax = plt.subplots(figsize=(5.5, max(3, n * 0.3)))

    names = []
    deltas = []
    colors = []
    for k in ordered_keys:
        g = results[k]["gap"]
        delta = (g.accuracy_l0 - g.accuracy_l3) * 100
        names.append(short_name(g.model_name).replace("\n", " "))
        deltas.append(delta)
        colors.append(COLORS["cloud"] if is_cloud(results[k]["config"]) else COLORS["local"])

    y = np.arange(n)
    bars = ax.barh(y, deltas, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

    # Color negative gaps (inverted — L0 worse than L3) differently
    for bar, d in zip(bars, deltas):
        if d < 0:
            bar.set_color(COLORS["gap"])

    # Value labels on each bar
    for i, d in enumerate(deltas):
        ha = "left" if d >= 0 else "right"
        offset = 0.5 if d >= 0 else -0.5
        ax.text(d + offset, y[i], f"{d:+.1f}", va="center", ha=ha, fontsize=6.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("L$_0$ $-$ L$_3$ Gap (pp)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    for fmt in ["pdf", "png"]:
        fig.savefig(FIGURES_DIR / f"fig2_composition_gap.{fmt}")
    plt.close(fig)
    print("  [fig] fig2_composition_gap")


# ── LaTeX Table Generation ────────────────────────────────────────────

def generate_leaderboard_table(results: dict, ordered_keys: list[str]) -> str:
    """Generate the main leaderboard LaTeX table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results on \bench{}. "
                 r"Models ranked by overall accuracy. "
                 r"\textbf{Bold} = best per column. "
                 r"$\Delta$ = \lzero{} $-$ \lthree{} gap (positive $=$ degradation). "
                 r"$\dagger$ = exhibits Selection Gap (\lzero{} $<$ avg of \lone{}--\lthree{}). "
                 r"All models achieve 100\% tool \emph{selection} accuracy (when they issue a call, they name the correct tool).}")
    lines.append(r"\label{tab:leaderboard}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Provider & \lzero{} & \lone{} & \ltwo{} & \lthree{} & Overall & $\Delta$\,$\downarrow$ \\")
    lines.append(r"\midrule")

    # Compute column-wise bests
    all_gaps = [results[k]["gap"] for k in ordered_keys]
    best_l0 = max(g.accuracy_l0 for g in all_gaps)
    best_l1 = max(g.accuracy_l1 for g in all_gaps)
    best_l2 = max(g.accuracy_l2 for g in all_gaps)
    best_l3 = max(g.accuracy_l3 for g in all_gaps)
    best_overall = max(g.overall_accuracy for g in all_gaps)
    best_delta = min((g.accuracy_l0 - g.accuracy_l3) for g in all_gaps)

    cloud_keys = [k for k in ordered_keys if is_cloud(results[k]["config"])]
    local_keys = [k for k in ordered_keys if not is_cloud(results[k]["config"])]

    def fmt(val: float, best: float, pct: bool = True) -> str:
        s = f"{val * 100:.1f}" if pct else f"{val:.1f}"
        return rf"\textbf{{{s}}}" if abs(val - best) < 1e-4 else s

    def fmt_delta(val: float, best: float) -> str:
        s = f"{val * 100:.1f}"
        if val < 0:
            s = f"$-${abs(val) * 100:.1f}"
        return rf"\textbf{{{s}}}" if abs(val - best) < 1e-4 else s

    for group_keys in [cloud_keys, local_keys]:
        for k in group_keys:
            g = results[k]["gap"]
            sg = compute_selection_gap(g)
            delta = g.accuracy_l0 - g.accuracy_l3

            provider = results[k]["config"].provider.value.capitalize()
            if provider == "Openrouter":
                provider = "OpenRouter"

            # Strip redundant provider parenthetical from model name
            # e.g. "Llama 3.1 8B (Groq)" → "Llama 3.1 8B" since Provider column has "Groq"
            name = re.sub(r"\s*\([^)]*\)\s*$", "", g.model_name)
            if sg["selection_gap_avg"]:
                name += r"$^\dagger$"

            lines.append(
                f"{name} & {provider} & "
                f"{fmt(g.accuracy_l0, best_l0)} & "
                f"{fmt(g.accuracy_l1, best_l1)} & "
                f"{fmt(g.accuracy_l2, best_l2)} & "
                f"{fmt(g.accuracy_l3, best_l3)} & "
                f"{fmt(g.overall_accuracy, best_overall)} & "
                f"{fmt_delta(delta, best_delta)} \\\\"
            )
        if group_keys == cloud_keys and local_keys:
            lines.append(r"\midrule")

    # Averages
    lines.append(r"\midrule")

    def avg_stat(keys: list[str], attr: str) -> float:
        vals = [getattr(results[k]["gap"], attr) for k in keys]
        return sum(vals) / len(vals) if vals else 0.0

    all_keys = cloud_keys + local_keys

    summary_rows = [(r"\textit{All models avg.}", all_keys)]
    # Only add cloud/local split rows if both groups exist
    if cloud_keys and local_keys:
        summary_rows.append((r"\textit{Cloud avg.}", cloud_keys))
        summary_rows.append((r"\textit{Local avg.}", local_keys))

    for label, keys in summary_rows:
        if not keys:
            continue
        l0 = avg_stat(keys, "accuracy_l0")
        l1 = avg_stat(keys, "accuracy_l1")
        l2 = avg_stat(keys, "accuracy_l2")
        l3 = avg_stat(keys, "accuracy_l3")
        ov = avg_stat(keys, "overall_accuracy")
        delta = l0 - l3
        lines.append(
            f"{label} & & "
            f"\\textit{{{l0*100:.1f}}} & \\textit{{{l1*100:.1f}}} & "
            f"\\textit{{{l2*100:.1f}}} & \\textit{{{l3*100:.1f}}} & "
            f"\\textit{{{ov*100:.1f}}} & \\textit{{{delta*100:.1f}}} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_leaderboard_csv(results: dict, ordered_keys: list[str]) -> str:
    """Generate a CSV leaderboard."""
    header = "model,provider,overall_accuracy,L0_accuracy,L1_accuracy,L2_accuracy,L3_accuracy,delta,selection_gap,argument_accuracy,avg_latency_ms"
    rows = [header]
    for k in ordered_keys:
        g = results[k]["gap"]
        c = results[k]["config"]
        sg = compute_selection_gap(g)
        delta = g.accuracy_l0 - g.accuracy_l3
        rows.append(",".join([
            g.model_name,
            c.provider.value,
            f"{g.overall_accuracy:.4f}",
            f"{g.accuracy_l0:.4f}",
            f"{g.accuracy_l1:.4f}",
            f"{g.accuracy_l2:.4f}",
            f"{g.accuracy_l3:.4f}",
            f"{delta:.4f}",
            str(sg["selection_gap_avg"]),
            f"{g.argument_accuracy:.4f}",
            f"{results[k]['avg_latency_ms']:.1f}",
        ]))
    return "\n".join(rows) + "\n"


# ── Summary Statistics ────────────────────────────────────────────────

def print_summary(results: dict, ordered_keys: list[str]) -> None:
    """Print paper-ready summary statistics."""
    cloud_keys = [k for k in ordered_keys if is_cloud(results[k]["config"])]
    local_keys = [k for k in ordered_keys if not is_cloud(results[k]["config"])]
    all_keys = cloud_keys + local_keys

    print("\n" + "=" * 70)
    print("PAPER SUMMARY STATISTICS")
    print("=" * 70)

    print(f"\nModels: {len(all_keys)} total ({len(cloud_keys)} cloud, {len(local_keys)} local)")
    print(f"Providers: {len(set(results[k]['config'].provider.value for k in all_keys))}")

    # Overall stats
    def avg(keys, attr):
        return sum(getattr(results[k]["gap"], attr) for k in keys) / len(keys)

    print(f"\nOverall accuracy: {avg(all_keys, 'overall_accuracy'):.1%}")
    print(f"  Cloud: {avg(cloud_keys, 'overall_accuracy'):.1%}")
    if local_keys:
        print(f"  Local: {avg(local_keys, 'overall_accuracy'):.1%}")

    print(f"\nL0 accuracy avg: {avg(all_keys, 'accuracy_l0'):.1%}")
    print(f"L1 accuracy avg: {avg(all_keys, 'accuracy_l1'):.1%}")
    print(f"L2 accuracy avg: {avg(all_keys, 'accuracy_l2'):.1%}")
    print(f"L3 accuracy avg: {avg(all_keys, 'accuracy_l3'):.1%}")

    # Composition gap (L0 - L3)
    deltas = [(results[k]["gap"].accuracy_l0 - results[k]["gap"].accuracy_l3) for k in all_keys]
    print(f"\nAvg L0→L3 delta: {np.mean(deltas)*100:.1f} pp")

    # Selection Gap analysis
    print("\n--- Selection Gap Analysis ---")
    for k in all_keys:
        g = results[k]["gap"]
        sg = compute_selection_gap(g)
        marker = " ← SELECTION GAP" if sg["selection_gap_avg"] else ""
        print(f"  {g.model_name}: L0={sg['l0']:.1%}, avg(L1,L2,L3)={sg['composed_avg']:.1%}, "
              f"gap={sg['gap_from_avg']*100:+.1f}pp{marker}")

    # Best/worst models
    best_k = max(all_keys, key=lambda k: results[k]["gap"].overall_accuracy)
    worst_k = min(all_keys, key=lambda k: results[k]["gap"].overall_accuracy)
    print(f"\nBest overall: {results[best_k]['gap'].model_name} ({results[best_k]['gap'].overall_accuracy:.1%})")
    print(f"Worst overall: {results[worst_k]['gap'].model_name} ({results[worst_k]['gap'].overall_accuracy:.1%})")

    # L2 > L1 analysis
    l2_gt_l1 = sum(1 for k in all_keys if results[k]["gap"].accuracy_l2 > results[k]["gap"].accuracy_l1)
    print(f"\nL2 > L1 (parallel easier than sequential): {l2_gt_l1}/{len(all_keys)} models")

    if cloud_keys:
        cloud_l2_gt_l1 = sum(1 for k in cloud_keys if results[k]["gap"].accuracy_l2 > results[k]["gap"].accuracy_l1)
        print(f"  Cloud: {cloud_l2_gt_l1}/{len(cloud_keys)}")
    if local_keys:
        local_l2_gt_l1 = sum(1 for k in local_keys if results[k]["gap"].accuracy_l2 > results[k]["gap"].accuracy_l1)
        print(f"  Local: {local_l2_gt_l1}/{len(local_keys)}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    """Generate all paper figures and tables."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading V3 task suite...")
    tasks = load_tasks(TASK_SUITE_PATH)
    print(f"  {len(tasks)} tasks loaded")

    print("\nLoading V3 results from all directories...")
    results = load_all_results(tasks)
    print(f"\n  Total models: {len(results)}")

    if not results:
        print("ERROR: No complete results found!")
        sys.exit(1)

    ordered_keys = sort_models(results)

    print("\nGenerating figures...")
    fig_accuracy_by_level(results, ordered_keys)
    fig_degradation_curves(results, ordered_keys)
    fig_selection_gap(results, ordered_keys)
    fig_heatmap(results, ordered_keys)
    fig_gap_vs_baseline(results, ordered_keys)
    fig_error_distribution(results, ordered_keys)
    fig_composition_gap_bars(results, ordered_keys)

    print("\nGenerating tables...")
    table_tex = generate_leaderboard_table(results, ordered_keys)
    (TABLES_DIR / "leaderboard.tex").write_text(table_tex)
    print("  [table] leaderboard.tex")

    csv_text = generate_leaderboard_csv(results, ordered_keys)
    (TABLES_DIR / "leaderboard.csv").write_text(csv_text)
    print("  [table] leaderboard.csv")

    print_summary(results, ordered_keys)

    print(f"\nDone! Figures: {FIGURES_DIR}")
    print(f"Tables: {TABLES_DIR}")


if __name__ == "__main__":
    main()
