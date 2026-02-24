"""Statistical rigor module for CompToolBench benchmark.

Provides confidence intervals and significance tests for all reported metrics,
following best practices from ML benchmark evaluation literature:

  - Bootstrap CIs (Efron & Tibshirani, 1993) for all per-model, per-level metrics
  - Paired bootstrap tests (Berg-Kirkpatrick et al., 2012) for model comparisons
  - Cohen's d effect sizes for practical significance
  - Bonferroni correction for multiple pairwise comparisons

All methods use non-parametric bootstrap to avoid distributional assumptions,
which is appropriate for accuracy-like metrics on discrete task outcomes.

Usage:
    from comptoolbench.analysis.loader import load_results
    from comptoolbench.analysis.statistical import (
        compute_model_cis,
        pairwise_significance,
        generate_statistical_report,
    )

    results = load_results("results/run_001/results_20260222.json")
    ci_df = compute_model_cis(results)
    sig_df = pairwise_significance(results)
    report = generate_statistical_report(results)
"""

from __future__ import annotations

import logging
from itertools import combinations
from textwrap import dedent
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from comptoolbench.analysis.loader import BenchmarkResults

logger = logging.getLogger(__name__)

# Composition level constants — canonical order used throughout the benchmark
LEVELS = ["L0_node", "L1_chain", "L2_parallel", "L3_dag"]
COMPOSED_LEVELS = ["L1_chain", "L2_parallel", "L3_dag"]

# Cohen's d interpretation thresholds (Cohen, 1988)
_EFFECT_THRESHOLDS: list[tuple[float, str]] = [
    (0.2, "negligible"),
    (0.5, "small"),
    (0.8, "medium"),
]


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d magnitude following Cohen (1988) conventions.

    Args:
        d: Absolute value of Cohen's d.

    Returns:
        Human-readable interpretation string.
    """
    abs_d = abs(d)
    for threshold, label in _EFFECT_THRESHOLDS:
        if abs_d < threshold:
            return label
    return "large"


# ---------------------------------------------------------------------------
# 1. Bootstrap Confidence Interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval for the mean of *scores*.

    Uses the percentile method: resample with replacement, compute the mean
    for each bootstrap replicate, then take the alpha/2 and 1-alpha/2
    percentiles of the bootstrap distribution as CI bounds.

    Args:
        scores: 1-D array of metric values (e.g., per-task accuracy scores).
        n_bootstrap: Number of bootstrap resamples (default 10,000).
        alpha: Significance level; CI covers (1-alpha)*100% (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper) where point_estimate = mean(scores).

    Raises:
        ValueError: If *scores* is empty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot compute bootstrap CI on empty array.")

    point_estimate = float(np.mean(scores))

    # Degenerate case: single observation or zero variance
    if scores.size == 1 or np.std(scores) == 0.0:
        return (point_estimate, point_estimate, point_estimate)

    rng = np.random.default_rng(seed)
    # Vectorised bootstrap: draw (n_bootstrap, n) index matrix in one shot
    n = scores.size
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = scores[boot_indices].mean(axis=1)

    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    return (point_estimate, ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# 2. Paired Bootstrap Significance Test
# ---------------------------------------------------------------------------


def bootstrap_paired_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap test for the difference in means between two models.

    Tests H0: mean(scores_a) == mean(scores_b) using the paired bootstrap
    approach of Berg-Kirkpatrick et al. (2012, EMNLP).

    Procedure:
      1. Compute observed_diff = mean(scores_a) - mean(scores_b).
      2. Centre the paired differences under H0 (subtract observed mean diff).
      3. Resample centred differences with replacement and compute the mean
         for each replicate.
      4. Two-sided p-value = fraction of resampled means at least as extreme
         as observed_diff.

    Args:
        scores_a: Per-task scores for Model A.
        scores_b: Per-task scores for Model B (must be same length, paired by task).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        (observed_diff, p_value) where observed_diff = mean(A) - mean(B).

    Raises:
        ValueError: If input arrays have different lengths or are empty.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if scores_a.size == 0 or scores_b.size == 0:
        raise ValueError("Cannot run paired test on empty arrays.")
    if scores_a.size != scores_b.size:
        raise ValueError(
            f"Paired test requires equal-length arrays, "
            f"got {scores_a.size} vs {scores_b.size}."
        )

    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))

    # Paired differences, centred under H0
    diffs = scores_a - scores_b
    centred_diffs = diffs - np.mean(diffs)  # shift to mean-zero under H0

    rng = np.random.default_rng(seed)
    n = centred_diffs.size
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = centred_diffs[boot_indices].mean(axis=1)

    # Two-sided p-value: fraction of bootstrap replicates as extreme as observed
    p_value = float(np.mean(np.abs(boot_means) >= abs(observed_diff)))

    return (observed_diff, p_value)


# ---------------------------------------------------------------------------
# 3. Cohen's d Effect Size
# ---------------------------------------------------------------------------


def cohens_d(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """Compute Cohen's d effect size between two score distributions.

    Uses the pooled standard deviation:
        d = (mean_a - mean_b) / s_pooled

    where s_pooled = sqrt(((n_a - 1)*s_a^2 + (n_b - 1)*s_b^2) / (n_a + n_b - 2)).

    Interpretation thresholds (Cohen, 1988):
        |d| < 0.2  = negligible
        |d| < 0.5  = small
        |d| < 0.8  = medium
        |d| >= 0.8 = large

    Args:
        scores_a: Scores for group A.
        scores_b: Scores for group B.

    Returns:
        Cohen's d (positive means A > B).

    Raises:
        ValueError: If either array is empty.
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if scores_a.size == 0 or scores_b.size == 0:
        raise ValueError("Cannot compute Cohen's d on empty arrays.")

    n_a, n_b = scores_a.size, scores_b.size
    var_a = np.var(scores_a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(scores_b, ddof=1) if n_b > 1 else 0.0

    # Pooled standard deviation
    denom = n_a + n_b - 2
    if denom <= 0:
        # Both groups have a single element
        return 0.0

    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / denom
    pooled_std = float(np.sqrt(pooled_var))

    if pooled_std == 0.0:
        # Zero variance in both groups: difference is either 0 or infinite
        mean_diff = float(np.mean(scores_a) - np.mean(scores_b))
        return 0.0 if mean_diff == 0.0 else float("inf") * np.sign(mean_diff)

    return float((np.mean(scores_a) - np.mean(scores_b)) / pooled_std)


# ---------------------------------------------------------------------------
# 4. Per-Model Confidence Intervals by Level
# ---------------------------------------------------------------------------


def compute_model_cis(
    results: BenchmarkResults,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute bootstrap CIs for accuracy at every (model, level) combination.

    For each model and each composition level (L0, L1, L2, L3, overall),
    bootstraps the task-level ``overall`` scores and reports point estimate
    with percentile CI.

    Args:
        results: Loaded benchmark results.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        DataFrame with columns: model, level, accuracy, ci_lower, ci_upper, n_tasks.
    """
    task_df = results.task_df
    if task_df.empty:
        logger.warning("task_df is empty; returning empty CI DataFrame.")
        return pd.DataFrame(
            columns=["model", "level", "accuracy", "ci_lower", "ci_upper", "n_tasks"]
        )

    rows: list[dict[str, object]] = []
    models = results.model_names

    for model in models:
        model_tasks = task_df[task_df["model"] == model]

        # Per-level CIs
        for level in LEVELS:
            level_scores = model_tasks.loc[
                model_tasks["level"] == level, "overall"
            ].values
            if level_scores.size == 0:
                continue
            acc, ci_lo, ci_hi = bootstrap_ci(
                level_scores, n_bootstrap=n_bootstrap, alpha=alpha
            )
            rows.append(
                {
                    "model": model,
                    "level": level,
                    "accuracy": acc,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "n_tasks": level_scores.size,
                }
            )

        # Overall CI (all levels combined)
        all_scores = model_tasks["overall"].values
        if all_scores.size > 0:
            acc, ci_lo, ci_hi = bootstrap_ci(
                all_scores, n_bootstrap=n_bootstrap, alpha=alpha
            )
            rows.append(
                {
                    "model": model,
                    "level": "overall",
                    "accuracy": acc,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "n_tasks": all_scores.size,
                }
            )

    df = pd.DataFrame(rows)
    logger.info("Computed CIs for %d (model, level) combinations.", len(df))
    return df


# ---------------------------------------------------------------------------
# 5. Composition Gap Confidence Intervals
# ---------------------------------------------------------------------------


def compute_gap_cis(
    results: BenchmarkResults,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute bootstrap CIs for the composition gap metric.

    The composition gap is defined as:
        gap(Lx) = mean(L0 scores) - mean(Lx scores)

    For each bootstrap resample we independently resample L0 and Lx tasks,
    compute their means, and form the gap. This captures the sampling
    uncertainty in *both* the numerator and denominator of the gap.

    Args:
        results: Loaded benchmark results.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        DataFrame with columns: model, gap_level, gap, ci_lower, ci_upper.
    """
    task_df = results.task_df
    if task_df.empty:
        logger.warning("task_df is empty; returning empty gap CI DataFrame.")
        return pd.DataFrame(
            columns=["model", "gap_level", "gap", "ci_lower", "ci_upper"]
        )

    rows: list[dict[str, object]] = []
    models = results.model_names

    for model in models:
        model_tasks = task_df[task_df["model"] == model]
        l0_scores = model_tasks.loc[model_tasks["level"] == "L0_node", "overall"].values

        if l0_scores.size == 0:
            continue

        rng = np.random.default_rng(42)

        for level in COMPOSED_LEVELS:
            lx_scores = model_tasks.loc[model_tasks["level"] == level, "overall"].values
            if lx_scores.size == 0:
                continue

            # Observed gap
            observed_gap = float(np.mean(l0_scores) - np.mean(lx_scores))

            # Bootstrap: resample L0 and Lx independently, compute gap each time
            n_l0 = l0_scores.size
            n_lx = lx_scores.size
            boot_l0 = l0_scores[rng.integers(0, n_l0, size=(n_bootstrap, n_l0))]
            boot_lx = lx_scores[rng.integers(0, n_lx, size=(n_bootstrap, n_lx))]
            boot_gaps = boot_l0.mean(axis=1) - boot_lx.mean(axis=1)

            lower_pct = (alpha / 2) * 100
            upper_pct = (1 - alpha / 2) * 100
            ci_lower = float(np.percentile(boot_gaps, lower_pct))
            ci_upper = float(np.percentile(boot_gaps, upper_pct))

            # Map level to short label (L1, L2, L3)
            gap_label = level.split("_")[0]  # "L1_chain" -> "L1"
            rows.append(
                {
                    "model": model,
                    "gap_level": gap_label,
                    "gap": observed_gap,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

        # Overall gap: L0 vs mean of all composed levels
        composed_scores = model_tasks.loc[
            model_tasks["level"].isin(COMPOSED_LEVELS), "overall"
        ].values
        if composed_scores.size > 0:
            observed_overall_gap = float(np.mean(l0_scores) - np.mean(composed_scores))
            n_composed = composed_scores.size
            boot_l0 = l0_scores[
                rng.integers(0, l0_scores.size, size=(n_bootstrap, l0_scores.size))
            ]
            boot_composed = composed_scores[
                rng.integers(0, n_composed, size=(n_bootstrap, n_composed))
            ]
            boot_overall_gaps = boot_l0.mean(axis=1) - boot_composed.mean(axis=1)

            lower_pct = (alpha / 2) * 100
            upper_pct = (1 - alpha / 2) * 100
            ci_lower = float(np.percentile(boot_overall_gaps, lower_pct))
            ci_upper = float(np.percentile(boot_overall_gaps, upper_pct))

            rows.append(
                {
                    "model": model,
                    "gap_level": "overall",
                    "gap": observed_overall_gap,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                }
            )

    df = pd.DataFrame(rows)
    logger.info("Computed gap CIs for %d (model, level) combinations.", len(df))
    return df


# ---------------------------------------------------------------------------
# 6. Pairwise Significance Tests
# ---------------------------------------------------------------------------


def pairwise_significance(
    results: BenchmarkResults,
    metric: str = "overall",
    n_bootstrap: int = 5_000,
) -> pd.DataFrame:
    """Run pairwise paired bootstrap tests between all model pairs.

    For each ordered pair (A, B), tests whether A's mean task-level score
    is significantly different from B's. Applies Bonferroni correction for
    the number of comparisons.

    Tasks are matched by task_id to ensure paired testing. Only tasks present
    in both models' results are compared.

    Args:
        results: Loaded benchmark results.
        metric: Column in task_df to compare (default "overall").
        n_bootstrap: Number of bootstrap resamples per pair.

    Returns:
        DataFrame with columns: model_a, model_b, diff, p_value,
        significant, effect_size, effect_interpretation.
    """
    task_df = results.task_df
    if task_df.empty:
        logger.warning("task_df is empty; returning empty significance DataFrame.")
        return pd.DataFrame(
            columns=[
                "model_a",
                "model_b",
                "diff",
                "p_value",
                "significant",
                "effect_size",
                "effect_interpretation",
            ]
        )

    models = results.model_names
    if len(models) < 2:
        logger.warning("Fewer than 2 models; no pairwise comparisons possible.")
        return pd.DataFrame(
            columns=[
                "model_a",
                "model_b",
                "diff",
                "p_value",
                "significant",
                "effect_size",
                "effect_interpretation",
            ]
        )

    # Number of comparisons for Bonferroni correction
    n_comparisons = len(models) * (len(models) - 1) // 2
    adjusted_alpha = 0.05 / n_comparisons

    rows: list[dict[str, object]] = []

    for model_a, model_b in combinations(models, 2):
        # Get paired task scores — align by task_id
        df_a = (
            task_df[task_df["model"] == model_a]
            .set_index("task_id")[[metric]]
            .rename(columns={metric: "score_a"})
        )
        df_b = (
            task_df[task_df["model"] == model_b]
            .set_index("task_id")[[metric]]
            .rename(columns={metric: "score_b"})
        )

        merged = df_a.join(df_b, how="inner")
        if merged.empty:
            logger.warning(
                "No shared tasks between %s and %s; skipping.", model_a, model_b
            )
            continue

        scores_a = merged["score_a"].values
        scores_b = merged["score_b"].values

        diff, p_val = bootstrap_paired_test(scores_a, scores_b, n_bootstrap=n_bootstrap)
        d = cohens_d(scores_a, scores_b)
        interpretation = _interpret_effect_size(d)

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "diff": round(diff, 6),
                "p_value": round(p_val, 6),
                "significant": p_val < adjusted_alpha,
                "effect_size": round(d, 4),
                "effect_interpretation": interpretation,
            }
        )

    df = pd.DataFrame(rows)
    logger.info(
        "Completed %d pairwise comparisons (Bonferroni alpha=%.4f).",
        len(df),
        adjusted_alpha,
    )
    return df


# ---------------------------------------------------------------------------
# 7. Markdown Statistical Report
# ---------------------------------------------------------------------------


def generate_statistical_report(
    results: BenchmarkResults,
    n_bootstrap: int = 10_000,
) -> str:
    """Generate a full markdown report of statistical analyses.

    Includes:
      - Per-model accuracy with 95% CIs at every level
      - Composition gap with 95% CIs
      - Pairwise significance for top 5 models
      - Effect sizes for notable differences

    The report is formatted for direct inclusion in a paper appendix
    or supplementary material.

    Args:
        results: Loaded benchmark results.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Markdown-formatted string.
    """
    sections: list[str] = []

    # Header
    sections.append(
        dedent("""\
        # CompToolBench Statistical Analysis Report

        All confidence intervals are 95% non-parametric bootstrap percentile intervals
        (10,000 resamples). Pairwise tests use paired bootstrap with Bonferroni correction
        for multiple comparisons.

        ---
    """)
    )

    # --- Section 1: Per-Model Accuracy CIs ---
    ci_df = compute_model_cis(results, n_bootstrap=n_bootstrap)

    if not ci_df.empty:
        sections.append("## 1. Per-Model Accuracy with 95% Confidence Intervals\n")

        for model in results.model_names:
            model_ci = ci_df[ci_df["model"] == model]
            sections.append(f"### {model}\n")
            sections.append(
                "| Level | Accuracy | 95% CI | n |\n|-------|----------|--------|---|\n"
            )
            for _, row in model_ci.iterrows():
                level = row["level"]
                acc = row["accuracy"]
                ci_lo = row["ci_lower"]
                ci_hi = row["ci_upper"]
                n = int(row["n_tasks"])
                sections.append(
                    f"| {level} | {acc:.1%} | [{ci_lo:.1%}, {ci_hi:.1%}] | {n} |\n"
                )
            sections.append("\n")
    else:
        sections.append("## 1. Per-Model Accuracy\n\n*No data available.*\n\n")

    sections.append("---\n\n")

    # --- Section 2: Composition Gap CIs ---
    gap_df = compute_gap_cis(results, n_bootstrap=n_bootstrap)

    if not gap_df.empty:
        sections.append("## 2. Composition Gap with 95% Confidence Intervals\n\n")
        sections.append(
            "The composition gap measures the drop in accuracy from L0 (single tool) "
            "to composed levels. Positive values indicate degradation.\n\n"
        )

        for model in results.model_names:
            model_gap = gap_df[gap_df["model"] == model]
            if model_gap.empty:
                continue
            sections.append(f"### {model}\n")
            sections.append(
                "| Gap Level | Gap | 95% CI |\n|-----------|-----|--------|\n"
            )
            for _, row in model_gap.iterrows():
                gap_level = row["gap_level"]
                gap = row["gap"]
                ci_lo = row["ci_lower"]
                ci_hi = row["ci_upper"]
                sections.append(
                    f"| {gap_level} | {gap:+.1%} | [{ci_lo:+.1%}, {ci_hi:+.1%}] |\n"
                )
            sections.append("\n")
    else:
        sections.append("## 2. Composition Gap\n\n*No data available.*\n\n")

    sections.append("---\n\n")

    # --- Section 3: Pairwise Significance (top 5 models) ---
    top_models = results.model_names[:5]
    sig_df = pairwise_significance(results, n_bootstrap=n_bootstrap)

    if not sig_df.empty:
        # Filter to top-5 model pairs
        sig_top = sig_df[
            sig_df["model_a"].isin(top_models) & sig_df["model_b"].isin(top_models)
        ].copy()

        sections.append("## 3. Pairwise Significance Tests (Top 5 Models)\n\n")

        n_models = len(top_models)
        n_comparisons = n_models * (n_models - 1) // 2
        adj_alpha = 0.05 / max(n_comparisons, 1)
        sections.append(
            f"Bonferroni-corrected significance threshold: "
            f"alpha = 0.05 / {n_comparisons} = {adj_alpha:.4f}\n\n"
        )
        sections.append(
            "| Model A | Model B | Diff | p-value | Sig. | Cohen's d | Effect |\n"
            "|---------|---------|------|---------|------|-----------|--------|\n"
        )
        for _, row in sig_top.iterrows():
            sig_marker = "Yes" if row["significant"] else "No"
            sections.append(
                f"| {row['model_a']} | {row['model_b']} "
                f"| {row['diff']:+.3f} | {row['p_value']:.4f} "
                f"| {sig_marker} | {row['effect_size']:.3f} "
                f"| {row['effect_interpretation']} |\n"
            )
        sections.append("\n")
    else:
        sections.append(
            "## 3. Pairwise Significance Tests\n\n*Fewer than 2 models.*\n\n"
        )

    sections.append("---\n\n")

    # --- Section 4: Notable Effect Sizes ---
    if not sig_df.empty:
        sections.append("## 4. Notable Effect Sizes\n\n")
        notable = sig_df[sig_df["effect_size"].abs() >= 0.5].sort_values(
            "effect_size", key=abs, ascending=False
        )
        if not notable.empty:
            sections.append(
                "Pairs with medium or larger effect sizes (|d| >= 0.5):\n\n"
            )
            for _, row in notable.iterrows():
                direction = "outperforms" if row["diff"] > 0 else "underperforms vs"
                sections.append(
                    f"- **{row['model_a']}** {direction} **{row['model_b']}**: "
                    f"d = {row['effect_size']:.3f} ({row['effect_interpretation']}), "
                    f"diff = {row['diff']:+.3f}, p = {row['p_value']:.4f}\n"
                )
            sections.append("\n")
        else:
            sections.append("No pairs with medium or larger effect sizes found.\n\n")
    else:
        sections.append("## 4. Notable Effect Sizes\n\n*No data available.*\n\n")

    # Footer
    sections.append(
        dedent("""\
        ---

        *Report generated by CompToolBench statistical analysis module.*
        *Bootstrap resamples: {n_bootstrap:,}. Seed: 42.*
    """).format(n_bootstrap=n_bootstrap)
    )

    report = "".join(sections)
    logger.info("Generated statistical report (%d characters).", len(report))
    return report
