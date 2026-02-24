#!/usr/bin/env python3
"""Scoring weight ablation study for CompToolBench.

Re-scores all V3 benchmark results with different weight configurations
to demonstrate that the Selection Gap finding and model rankings are
robust to the choice of scoring weights.

This is a critical ablation for the paper — it proves that our ad-hoc
weights (0.40/0.35/0.25) don't drive the results.

Usage:
    uv run python scripts/ablation_weights.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Weight configurations to ablate
# ---------------------------------------------------------------------------

@dataclass
class WeightConfig:
    """A named set of scoring weights for L1/L2/L3 tasks."""

    name: str
    # L1 (chain): tool_seq, args, completeness
    l1_weights: tuple[float, float, float]
    # L2 (parallel): tool_seq, args, data_flow, completeness
    l2_weights: tuple[float, float, float, float]
    # L3 (DAG): tool_seq, args, data_flow, completeness
    l3_weights: tuple[float, float, float, float]
    # For binary configs, threshold for pass/fail (None = use weighted)
    binary_threshold: float | None = None


WEIGHT_CONFIGS: list[WeightConfig] = [
    WeightConfig(
        name="Default (paper)",
        l1_weights=(0.40, 0.35, 0.25),
        l2_weights=(0.35, 0.35, 0.15, 0.15),
        l3_weights=(0.30, 0.30, 0.25, 0.15),
    ),
    WeightConfig(
        name="Uniform",
        l1_weights=(1/3, 1/3, 1/3),
        l2_weights=(0.25, 0.25, 0.25, 0.25),
        l3_weights=(0.25, 0.25, 0.25, 0.25),
    ),
    WeightConfig(
        name="Sequence-heavy",
        l1_weights=(0.60, 0.20, 0.20),
        l2_weights=(0.50, 0.20, 0.15, 0.15),
        l3_weights=(0.50, 0.20, 0.15, 0.15),
    ),
    WeightConfig(
        name="Args-heavy",
        l1_weights=(0.20, 0.60, 0.20),
        l2_weights=(0.15, 0.55, 0.15, 0.15),
        l3_weights=(0.15, 0.55, 0.15, 0.15),
    ),
    WeightConfig(
        name="Completeness-heavy",
        l1_weights=(0.20, 0.20, 0.60),
        l2_weights=(0.15, 0.15, 0.15, 0.55),
        l3_weights=(0.15, 0.15, 0.15, 0.55),
    ),
    WeightConfig(
        name="Data-flow-heavy",
        l1_weights=(0.25, 0.25, 0.50),  # L1 has no data_flow, use completeness
        l2_weights=(0.15, 0.15, 0.55, 0.15),
        l3_weights=(0.15, 0.15, 0.55, 0.15),
    ),
    WeightConfig(
        name="Binary (≥0.50)",
        l1_weights=(0.40, 0.35, 0.25),
        l2_weights=(0.35, 0.35, 0.15, 0.15),
        l3_weights=(0.30, 0.30, 0.25, 0.15),
        binary_threshold=0.50,
    ),
    WeightConfig(
        name="Binary (≥0.70)",
        l1_weights=(0.40, 0.35, 0.25),
        l2_weights=(0.35, 0.35, 0.15, 0.15),
        l3_weights=(0.30, 0.30, 0.25, 0.15),
        binary_threshold=0.70,
    ),
]


# ---------------------------------------------------------------------------
# Score recomputation
# ---------------------------------------------------------------------------

def rescore_task(
    entry: dict[str, Any],
    config: WeightConfig,
) -> float:
    """Recompute overall score for a task using the given weight config."""
    score = entry["score"]
    level = score["level"]

    seq = score.get("tool_sequence_score", 0.0)
    args = score.get("argument_score", 0.0)
    comp = score.get("completeness_score", 0.0)
    df = score.get("data_flow_score", 1.0)

    if level == "L0_node":
        # L0 is always binary: tool correct AND args >= 0.85
        # This doesn't change with weight configs
        return score["overall"]

    if level == "L1_chain":
        w_seq, w_args, w_comp = config.l1_weights
        overall = seq * w_seq + args * w_args + comp * w_comp
    elif level == "L2_parallel":
        w_seq, w_args, w_df, w_comp = config.l2_weights
        overall = seq * w_seq + args * w_args + df * w_df + comp * w_comp
    elif level == "L3_dag":
        w_seq, w_args, w_df, w_comp = config.l3_weights
        overall = seq * w_seq + args * w_args + df * w_df + comp * w_comp
    else:
        overall = score["overall"]

    if config.binary_threshold is not None:
        return 1.0 if overall >= config.binary_threshold else 0.0

    return overall


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

V3_CHECKPOINT_DIRS = [
    Path("results/unified_v3"),
    Path("results/cerebras_v2"),
    Path("results/cohere_v2"),
]

# Map from checkpoint filename stems to display names
MODEL_DISPLAY_NAMES = {
    "groq-llama3.1-8b": "Llama 3.1 8B (Groq)",
    "groq-llama4-scout": "Llama 4 Scout (Groq)",
    "mistral-small": "Mistral Small",
    "mistral-medium": "Mistral Medium",
    "mistral-large": "Mistral Large",
    "or-gemini-2.0-flash": "Gemini 2.0 Flash",
    "cerebras-llama3.1-8b": "Llama 3.1 8B (Cerebras)",
    "cerebras-gpt-oss-120b": "GPT-OSS 120B (Cerebras)",
    "cohere-command-a": "Command A (Cohere)",
    "cohere-command-r-plus": "Command R+ (Cohere)",
}


def load_all_checkpoints() -> dict[str, list[dict[str, Any]]]:
    """Load all V3 checkpoint files. Returns {model_key: [entries]}."""
    all_data: dict[str, list[dict[str, Any]]] = {}

    for dir_path in V3_CHECKPOINT_DIRS:
        if not dir_path.exists():
            continue
        for ckpt_file in sorted(dir_path.glob("checkpoint_*.jsonl")):
            model_key = ckpt_file.stem.replace("checkpoint_", "")
            entries = []
            for line in ckpt_file.read_text().strip().split("\n"):
                if line.strip():
                    entries.append(json.loads(line))
            if entries:
                all_data[model_key] = entries
                print(f"  Loaded {model_key}: {len(entries)} tasks")

    return all_data


# ---------------------------------------------------------------------------
# Ablation analysis
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result of rescoring one model with one weight config."""

    model_key: str
    config_name: str
    overall_accuracy: float
    l0_accuracy: float
    l1_accuracy: float
    l2_accuracy: float
    l3_accuracy: float
    selection_gap_present: bool  # L0 < min(L1, L2, L3)?


def run_ablation(
    all_data: dict[str, list[dict[str, Any]]],
) -> list[AblationResult]:
    """Run the full ablation across all models and weight configs."""
    results: list[AblationResult] = []

    for config in WEIGHT_CONFIGS:
        for model_key, entries in all_data.items():
            # Rescore all tasks
            by_level: dict[str, list[float]] = defaultdict(list)
            all_scores: list[float] = []

            for entry in entries:
                new_score = rescore_task(entry, config)
                level = entry["score"]["level"]
                by_level[level].append(new_score)
                all_scores.append(new_score)

            # Compute per-level averages
            l0 = np.mean(by_level.get("L0_node", [0.0]))
            l1 = np.mean(by_level.get("L1_chain", [0.0]))
            l2 = np.mean(by_level.get("L2_parallel", [0.0]))
            l3 = np.mean(by_level.get("L3_dag", [0.0]))
            overall = np.mean(all_scores) if all_scores else 0.0

            # Is the Selection Gap present? (L0 < min of composed levels)
            selection_gap = l0 < min(l1, l2, l3)

            results.append(AblationResult(
                model_key=model_key,
                config_name=config.name,
                overall_accuracy=float(overall),
                l0_accuracy=float(l0),
                l1_accuracy=float(l1),
                l2_accuracy=float(l2),
                l3_accuracy=float(l3),
                selection_gap_present=selection_gap,
            ))

    return results


def compute_rank_correlation(
    results: list[AblationResult],
    all_data: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Compute Spearman rank correlation between weight configs.

    Returns: {config_pair: {spearman_rho, p_value}}
    """
    from scipy import stats

    models = sorted(all_data.keys())
    configs = [c.name for c in WEIGHT_CONFIGS]

    # Build accuracy matrix: config × model
    accuracy_matrix: dict[str, dict[str, float]] = defaultdict(dict)
    for r in results:
        accuracy_matrix[r.config_name][r.model_key] = r.overall_accuracy

    # Compute pairwise Spearman correlations
    correlations: dict[str, dict[str, float]] = {}
    baseline = configs[0]  # Default weights

    baseline_ranks = [accuracy_matrix[baseline].get(m, 0.0) for m in models]

    for config_name in configs[1:]:
        config_ranks = [accuracy_matrix[config_name].get(m, 0.0) for m in models]
        rho, p_val = stats.spearmanr(baseline_ranks, config_ranks)
        correlations[f"{baseline} vs {config_name}"] = {
            "spearman_rho": float(rho),
            "p_value": float(p_val),
        }

    return correlations


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_ablation_table(
    results: list[AblationResult],
    all_data: dict[str, list[dict[str, Any]]],
) -> str:
    """Format ablation results as a readable table."""
    configs = [c.name for c in WEIGHT_CONFIGS]
    models = sorted(all_data.keys())

    lines: list[str] = []
    lines.append("=" * 120)
    lines.append("SCORING WEIGHT ABLATION STUDY — CompToolBench V3")
    lines.append("=" * 120)
    lines.append("")

    # Table 1: Overall accuracy by weight config
    lines.append("Table 1: Overall Accuracy (%) by Weight Configuration")
    lines.append("-" * 120)

    # Header
    header = f"{'Model':<30}"
    for config in configs:
        header += f" {config[:15]:>15}"
    lines.append(header)
    lines.append("-" * 120)

    # Build lookup
    lookup: dict[tuple[str, str], AblationResult] = {}
    for r in results:
        lookup[(r.model_key, r.config_name)] = r

    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        row = f"{display:<30}"
        for config in configs:
            r = lookup.get((model, config))
            if r:
                row += f" {r.overall_accuracy * 100:>14.1f}%"
            else:
                row += f" {'N/A':>15}"
        lines.append(row)

    lines.append("")

    # Table 2: Selection Gap persistence
    lines.append("Table 2: Selection Gap Persistence (L0 < min(L1, L2, L3)?)")
    lines.append("-" * 120)

    header2 = f"{'Model':<30}"
    for config in configs:
        header2 += f" {config[:15]:>15}"
    lines.append(header2)
    lines.append("-" * 120)

    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        row = f"{display:<30}"
        for config in configs:
            r = lookup.get((model, config))
            if r:
                symbol = "YES" if r.selection_gap_present else "no"
                row += f" {symbol:>15}"
            else:
                row += f" {'N/A':>15}"
        lines.append(row)

    lines.append("")

    # Table 3: Per-level accuracy for default vs each alternative
    lines.append("Table 3: Per-Level Accuracy for Default vs Uniform Weights")
    lines.append("-" * 80)
    lines.append(f"{'Model':<30} {'Cfg':<20} {'L0':>8} {'L1':>8} {'L2':>8} {'L3':>8}")
    lines.append("-" * 80)

    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)
        for config_name in ["Default (paper)", "Uniform"]:
            r = lookup.get((model, config_name))
            if r:
                tag = "Default" if config_name == "Default (paper)" else "Uniform"
                lines.append(
                    f"{display:<30} {tag:<20} "
                    f"{r.l0_accuracy * 100:>7.1f}% "
                    f"{r.l1_accuracy * 100:>7.1f}% "
                    f"{r.l2_accuracy * 100:>7.1f}% "
                    f"{r.l3_accuracy * 100:>7.1f}%"
                )
        lines.append("")

    return "\n".join(lines)


def generate_ablation_figure(
    results: list[AblationResult],
    all_data: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> Path:
    """Generate a publication-quality ablation figure."""
    import matplotlib.pyplot as plt

    configs = [c.name for c in WEIGHT_CONFIGS]
    models = sorted(all_data.keys(), key=lambda m: -next(
        r.overall_accuracy for r in results
        if r.model_key == m and r.config_name == "Default (paper)"
    ))

    # Build accuracy matrix
    acc_matrix = np.zeros((len(models), len(configs)))
    for i, model in enumerate(models):
        for j, config in enumerate(configs):
            r = next(
                (r for r in results if r.model_key == model and r.config_name == config),
                None,
            )
            if r:
                acc_matrix[i, j] = r.overall_accuracy * 100

    # Figure 1: Heatmap of overall accuracy across configs
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Heatmap
    ax = axes[0]
    im = ax.imshow(acc_matrix, cmap="YlOrRd", aspect="auto", vmin=30, vmax=70)

    display_names = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]
    short_configs = [c.split("(")[0].strip()[:12] for c in configs]

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(short_configs, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(display_names, fontsize=9)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(configs)):
            val = acc_matrix[i, j]
            color = "white" if val > 55 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title("Overall Accuracy (%) Across Weight Configurations", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Accuracy (%)")

    # Ranking stability bar chart
    ax2 = axes[1]
    default_rank = np.argsort(-acc_matrix[:, 0])  # Descending rank for default
    rank_changes = []
    for j in range(1, len(configs)):
        config_rank = np.argsort(-acc_matrix[:, j])
        # Count how many models changed rank
        changes = sum(1 for a, b in zip(default_rank, config_rank) if a != b)
        rank_changes.append(changes)

    ax2.barh(range(len(configs) - 1), rank_changes, color="#E69F00", edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(configs) - 1))
    ax2.set_yticklabels(short_configs[1:], fontsize=9)
    ax2.set_xlabel("Models that changed rank", fontsize=10)
    ax2.set_title("Ranking Stability", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(0, len(models))

    plt.tight_layout()
    output_path = output_dir / "ablation_weight_heatmap.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Figure 2: Selection Gap persistence across weight configs
    fig2, ax3 = plt.subplots(figsize=(10, 5))

    gap_data: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        gap_data[r.config_name].append(r.selection_gap_present)

    gap_rates = [sum(gap_data[c]) / len(gap_data[c]) * 100 for c in configs]

    bars = ax3.bar(range(len(configs)), gap_rates, color="#0072B2", edgecolor="black", linewidth=0.5)
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels(short_configs, rotation=45, ha="right", fontsize=9)
    ax3.set_ylabel("Models showing Selection Gap (%)", fontsize=11)
    ax3.set_title("Selection Gap Persistence Across Weight Configurations", fontsize=12, fontweight="bold")
    ax3.set_ylim(0, 110)
    ax3.axhline(y=100, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add value labels
    for bar, val in zip(bars, gap_rates):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    output_path2 = output_dir / "ablation_selection_gap_persistence.pdf"
    fig2.savefig(output_path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {output_path2}")

    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full ablation study."""
    print("=" * 70)
    print("  CompToolBench — Scoring Weight Ablation Study")
    print("=" * 70)
    print()

    # Load checkpoints
    print("Loading V3 checkpoint files...")
    all_data = load_all_checkpoints()
    print(f"\nLoaded {len(all_data)} models total\n")

    if not all_data:
        print("ERROR: No checkpoint files found. Run benchmarks first.")
        sys.exit(1)

    # Run ablation
    print("Running ablation across", len(WEIGHT_CONFIGS), "weight configurations...")
    results = run_ablation(all_data)
    print(f"  Generated {len(results)} model × config results\n")

    # Print tables
    table = print_ablation_table(results, all_data)
    print(table)

    # Rank correlations
    try:
        print("\nRank Correlations (Spearman's ρ vs Default weights):")
        print("-" * 60)
        correlations = compute_rank_correlation(results, all_data)
        for pair, vals in correlations.items():
            rho = vals["spearman_rho"]
            p = vals["p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {pair}: ρ = {rho:.4f} (p = {p:.4f}) {sig}")
    except ImportError:
        print("  (scipy not available — skipping rank correlations)")

    # Generate figures
    output_dir = Path("results/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures in {output_dir}...")
    generate_ablation_figure(results, all_data, output_dir)

    # Save results as JSON
    json_path = output_dir / "ablation_results.json"
    json_output = {
        "weight_configs": [
            {
                "name": c.name,
                "l1_weights": list(c.l1_weights),
                "l2_weights": list(c.l2_weights),
                "l3_weights": list(c.l3_weights),
                "binary_threshold": c.binary_threshold,
            }
            for c in WEIGHT_CONFIGS
        ],
        "results": [
            {
                "model": r.model_key,
                "config": r.config_name,
                "overall": round(r.overall_accuracy, 4),
                "l0": round(r.l0_accuracy, 4),
                "l1": round(r.l1_accuracy, 4),
                "l2": round(r.l2_accuracy, 4),
                "l3": round(r.l3_accuracy, 4),
                "selection_gap": bool(r.selection_gap_present),
            }
            for r in results
        ],
    }
    try:
        json_output["rank_correlations"] = correlations
    except NameError:
        pass
    json_path.write_text(json.dumps(json_output, indent=2))
    print(f"\nResults saved to {json_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  ABLATION SUMMARY")
    print("=" * 70)

    # Count Selection Gap persistence
    for config in WEIGHT_CONFIGS:
        gap_count = sum(
            1 for r in results
            if r.config_name == config.name and r.selection_gap_present
        )
        total = sum(1 for r in results if r.config_name == config.name)
        print(f"  {config.name:<25}: Selection Gap in {gap_count}/{total} models")

    print()
    print("CONCLUSION: If Selection Gap persists across ALL weight configs,")
    print("then it is a genuine finding, not an artifact of weight choices.")
    print("=" * 70)


if __name__ == "__main__":
    main()
