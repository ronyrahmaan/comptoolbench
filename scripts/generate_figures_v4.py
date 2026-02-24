#!/usr/bin/env python3
"""Generate publication-quality figures from the final leaderboard CSV.

Uses leaderboard_final.csv (all 27 models) for the main figures.
Generates:
  - fig_selection_gap.pdf: Selection Gap visualization
  - fig1_accuracy_by_level.pdf: Per-level accuracy bars
  - fig2_composition_gap.pdf: L0→L3 delta bars
  - fig7_accuracy_heatmap.pdf: Heatmap of model × level accuracy

Usage:
    uv run python scripts/generate_figures_v4.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# ── Publication settings ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 7,
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
    "frontier": "#E69F00",
    "cloud": "#56B4E9",
    "local": "#999999",
}

PROJECT_DIR = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_DIR / "paper" / "figures"
CSV_PATH = PROJECT_DIR / "results" / "leaderboard_final.csv"


def load_data() -> pd.DataFrame:
    """Load the final leaderboard CSV."""
    df = pd.read_csv(CSV_PATH)
    return df


def short_name(model: str, provider: str) -> str:
    """Create short display name."""
    if provider == "Ollama":
        return model.replace(" ", "\n") if len(model) > 12 else model
    if "Cerebras" in model or "Cerebras" in provider:
        short = model.split(" (")[0] if "(" in model else model
        return f"{short}\n(Cere.)"
    if provider in ("Groq", "Cohere", "Mistral", "OpenRouter"):
        return f"{model}\n({provider[:4]})" if len(model) > 8 else f"{model} ({provider[:4]})"
    return model.replace(" ", "\n") if len(model) > 10 else model


def fig_selection_gap(df: pd.DataFrame) -> None:
    """Bar chart: L0 vs composed avg for each model."""
    df = df.copy()
    df["composed_avg"] = df[["L1", "L2", "L3"]].mean(axis=1)
    df["short"] = [short_name(r["model"], r["provider"]) for _, r in df.iterrows()]

    n = len(df)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), 4))
    x = np.arange(n)
    w = 0.35

    bars_l0 = ax.bar(x - w / 2, df["L0"], w, label=r"L$_0$ (single)", color=COLORS["L0"], alpha=0.85)
    bars_comp = ax.bar(x + w / 2, df["composed_avg"], w, label=r"Avg(L$_1$,L$_2$,L$_3$)", color=COLORS["L2"], alpha=0.85)

    # Add gap arrows
    for i, (l0, comp) in enumerate(zip(df["L0"], df["composed_avg"])):
        if comp > l0:
            ax.annotate("", xy=(i + w / 2, comp), xytext=(i - w / 2, l0),
                        arrowprops=dict(arrowstyle="->", color=COLORS["gap"], lw=1.2))

    # Deployment regime separators
    frontier_n = len(df[df["deployment"] == "cloud"])  # frontier + free cloud
    local_start = len(df[df["deployment"] != "local"])
    if local_start < n:
        ax.axvline(local_start - 0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.text(local_start - 0.3, ax.get_ylim()[1] * 0.95, "Local →", fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Selection Gap: L₀ vs. Composed Average")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    path = FIGURES_DIR / "fig_selection_gap"
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")


def fig_accuracy_by_level(df: pd.DataFrame) -> None:
    """Grouped bar chart: accuracy at each level for all models."""
    df = df.copy()
    df["short"] = [short_name(r["model"], r["provider"]) for _, r in df.iterrows()]

    n = len(df)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 4))
    x = np.arange(n)
    w = 0.2

    for i, (level, color) in enumerate([("L0", COLORS["L0"]), ("L1", COLORS["L1"]),
                                         ("L2", COLORS["L2"]), ("L3", COLORS["L3"])]):
        ax.bar(x + (i - 1.5) * w, df[level], w, label=f"L$_{level[1]}$", color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Level Accuracy Across 27 Models")
    ax.legend(loc="upper right", ncol=4)
    ax.set_ylim(0, 105)

    path = FIGURES_DIR / "fig1_accuracy_by_level"
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")


def fig_composition_gap(df: pd.DataFrame) -> None:
    """Bar chart: L0 → L3 delta for each model."""
    df = df.copy()
    df["short"] = [short_name(r["model"], r["provider"]) for _, r in df.iterrows()]

    n = len(df)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.45), 3.5))
    x = np.arange(n)

    colors = [COLORS["frontier"] if r["deployment"] == "cloud" and r["provider"] in ("OpenAI", "Anthropic", "Google")
              else COLORS["cloud"] if r["deployment"] == "cloud"
              else COLORS["local"]
              for _, r in df.iterrows()]

    ax.bar(x, df["delta"], color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["short"], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(r"$\Delta$ = L$_0$ − L$_3$ (pp)")
    ax.set_title("L₀ → L₃ Composition Gap")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["frontier"], alpha=0.85, label="Frontier"),
        Patch(facecolor=COLORS["cloud"], alpha=0.85, label="Free Cloud"),
        Patch(facecolor=COLORS["local"], alpha=0.85, label="Local"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    path = FIGURES_DIR / "fig2_composition_gap"
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")


def fig_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of model × level accuracy."""
    df = df.copy()
    df["short"] = [short_name(r["model"], r["provider"]) for _, r in df.iterrows()]

    data = df[["L0", "L1", "L2", "L3"]].values
    labels = df["short"].values

    fig, ax = plt.subplots(figsize=(4, max(6, len(df) * 0.3)))
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)

    ax.set_xticks(range(4))
    ax.set_xticklabels([r"L$_0$", r"L$_1$", r"L$_2$", r"L$_3$"])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=6)

    # Add text annotations
    for i in range(len(df)):
        for j in range(4):
            val = data[i, j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5, color=color)

    plt.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    ax.set_title("Accuracy Heatmap: Model × Level")

    path = FIGURES_DIR / "fig7_accuracy_heatmap"
    fig.savefig(f"{path}.pdf")
    fig.savefig(f"{path}.png")
    plt.close(fig)
    print(f"  Saved: {path}.pdf")


def main() -> None:
    """Generate all figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} models loaded")

    print("\nGenerating figures...")
    fig_selection_gap(df)
    fig_accuracy_by_level(df)
    fig_composition_gap(df)
    fig_heatmap(df)

    print("\nDone! All figures saved to paper/figures/")


if __name__ == "__main__":
    main()
