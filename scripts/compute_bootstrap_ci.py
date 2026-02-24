#!/usr/bin/env python3
"""Compute bootstrap 95% confidence intervals for the Selection Gap.

Reads leaderboard_final.csv and computes:
  - Overall Selection Gap mean and 95% CI (10,000 resamples)
  - Per-regime (frontier, cloud, local) gap mean and 95% CI
  - Count of models showing the gap with CI

Usage:
    uv run python scripts/compute_bootstrap_ci.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
CSV_PATH = RESULTS_DIR / "leaderboard_final.csv"
N_BOOTSTRAP = 10_000
SEED = 42

FRONTIER_PROVIDERS = {"OpenAI", "Anthropic", "Google"}


def compute_selection_gap(df: pd.DataFrame) -> pd.Series:
    """Compute per-model Selection Gap = composed_avg - L0."""
    composed_avg = df[["L1", "L2", "L3"]].mean(axis=1)
    return composed_avg - df["L0"]


def bootstrap_ci(
    values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via bootstrap resampling."""
    rng = np.random.default_rng(seed)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(values)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bootstrap_proportion(
    flags: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float, float]:
    """Bootstrap CI for proportion (count/total)."""
    rng = np.random.default_rng(seed)
    n = len(flags)
    proportions = np.array([
        np.mean(rng.choice(flags, size=n, replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(flags)), float(np.percentile(proportions, 2.5)), float(np.percentile(proportions, 97.5))


def main() -> None:
    """Compute and print bootstrap CIs."""
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} models from {CSV_PATH.name}\n")

    # Per-model selection gap
    df["gap"] = compute_selection_gap(df)
    df["has_gap"] = df["gap"] > 0

    # Overall
    print("=" * 70)
    print("SELECTION GAP BOOTSTRAP 95% CIs (10,000 resamples)")
    print("=" * 70)

    all_gaps = df["gap"].values
    mean, lo, hi = bootstrap_ci(all_gaps)
    print(f"\nAll models (n={len(df)}):")
    print(f"  Mean gap:  {mean:.1f} pp  (95% CI: [{lo:.1f}, {hi:.1f}])")

    prop_mean, prop_lo, prop_hi = bootstrap_proportion(df["has_gap"].values.astype(float))
    n_gap = int(df["has_gap"].sum())
    print(f"  Models with gap: {n_gap}/{len(df)} ({prop_mean*100:.0f}%, 95% CI: [{prop_lo*100:.0f}%, {prop_hi*100:.0f}%])")

    # Per regime
    for label, mask in [
        ("Frontier", df["provider"].isin(FRONTIER_PROVIDERS)),
        ("Free cloud", ~df["provider"].isin(FRONTIER_PROVIDERS) & (df["deployment"] == "cloud")),
        ("Local", df["deployment"] == "local"),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        gaps = sub["gap"].values
        mean, lo, hi = bootstrap_ci(gaps)
        n_gap = int(sub["has_gap"].sum())
        print(f"\n{label} (n={len(sub)}):")
        print(f"  Mean gap:  {mean:.1f} pp  (95% CI: [{lo:.1f}, {hi:.1f}])")
        print(f"  Models with gap: {n_gap}/{len(sub)}")

    # Per-level averages with CIs
    print(f"\n{'='*70}")
    print("PER-LEVEL ACCURACY BOOTSTRAP 95% CIs")
    print(f"{'='*70}")

    for level in ["L0", "L1", "L2", "L3"]:
        vals = df[level].values
        mean, lo, hi = bootstrap_ci(vals)
        print(f"  {level}: {mean:.1f}%  (95% CI: [{lo:.1f}, {hi:.1f}])")

    overall = df["overall"].values
    mean, lo, hi = bootstrap_ci(overall)
    print(f"  Overall: {mean:.1f}%  (95% CI: [{lo:.1f}, {hi:.1f}])")

    # Composed average
    composed = df[["L1", "L2", "L3"]].mean(axis=1).values
    mean, lo, hi = bootstrap_ci(composed)
    print(f"  Composed avg (L1-L3): {mean:.1f}%  (95% CI: [{lo:.1f}, {hi:.1f}])")

    print(f"\n{'='*70}")
    print("EXACT NUMBERS FOR PAPER (copy-paste ready)")
    print(f"{'='*70}")

    # Compute exact values
    all_l0 = df["L0"].mean()
    all_composed = df[["L1", "L2", "L3"]].mean(axis=1).mean()
    gap_mean, gap_lo, gap_hi = bootstrap_ci(df["gap"].values)

    print(f"\nAbstract/Intro gap: {gap_mean:.1f} pp (95% CI: [{gap_lo:.1f}, {gap_hi:.1f}])")
    print(f"L0 avg: {all_l0:.1f}%")
    print(f"Composed avg: {all_composed:.1f}%")
    print(f"Gap = composed - L0 = {all_composed - all_l0:.1f} pp")

    # Per regime gaps
    for label, mask in [
        ("Frontier", df["provider"].isin(FRONTIER_PROVIDERS)),
        ("Cloud (free)", ~df["provider"].isin(FRONTIER_PROVIDERS) & (df["deployment"] == "cloud")),
        ("Local", df["deployment"] == "local"),
    ]:
        sub = df[mask]
        if len(sub) == 0:
            continue
        gap_mean, gap_lo, gap_hi = bootstrap_ci(sub["gap"].values)
        print(f"{label} gap: {gap_mean:.1f} pp (95% CI: [{gap_lo:.1f}, {gap_hi:.1f}])")

    # Level averages for each regime
    print(f"\nPer-regime level averages:")
    for label, mask in [
        ("Frontier", df["provider"].isin(FRONTIER_PROVIDERS)),
        ("Cloud (free)", ~df["provider"].isin(FRONTIER_PROVIDERS) & (df["deployment"] == "cloud")),
        ("Local", df["deployment"] == "local"),
        ("All", pd.Series([True]*len(df), index=df.index)),
    ]:
        sub = df[mask]
        print(f"  {label}: L0={sub['L0'].mean():.1f}, L1={sub['L1'].mean():.1f}, L2={sub['L2'].mean():.1f}, L3={sub['L3'].mean():.1f}, Overall={sub['overall'].mean():.1f}")

    # Llama 3.1 8B Groq special case
    llama = df[df["model"].str.contains("Llama 3.1 8B") & (df["provider"] == "Groq")]
    if not llama.empty:
        row = llama.iloc[0]
        composed = (row["L1"] + row["L2"] + row["L3"]) / 3
        print(f"\nLlama 3.1 8B (Groq): L0={row['L0']:.1f}%, composed avg={composed:.1f}%")

    # Models showing gap
    print(f"\nSelection Gap count: {int(df['has_gap'].sum())}/{len(df)}")
    no_gap = df[~df["has_gap"]]
    if not no_gap.empty:
        print(f"Models WITHOUT gap: {', '.join(no_gap['model'].tolist())}")


if __name__ == "__main__":
    main()
