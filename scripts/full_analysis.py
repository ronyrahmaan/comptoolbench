#!/usr/bin/env python3
"""Comprehensive analysis script for CompToolBench.

Produces six analysis outputs with bootstrap CIs, baselines, per-category
breakdowns, and qualitative error examples.

Usage:
    # Auto-detect latest run:
    uv run python scripts/full_analysis.py

    # Specify results directory:
    uv run python scripts/full_analysis.py --results-dir results/run_20260311_000147

    # Custom bootstrap resamples (faster for debugging):
    uv run python scripts/full_analysis.py --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project imports — add src/ to path so the script works standalone
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comptoolbench.analysis.loader import BenchmarkResults, load_results
from comptoolbench.analysis.statistical import (
    bootstrap_ci,
    compute_gap_cis,
    compute_model_cis,
    pairwise_significance,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_analysis")

# Error taxonomy from ARCHITECTURE.md Appendix C
ERROR_DESCRIPTIONS: dict[str, str] = {
    "E1_wrong_tool": "Model selected the wrong tool for the step.",
    "E3_wrong_order": "Tools were called in the wrong order.",
    "E4_wrong_arguments": "Correct tool selected but with incorrect arguments.",
    "E6_hallucinated_tool": "Model called a tool that does not exist in available_tools.",
    "E7_unnecessary_tool": "Model made extra, unnecessary tool calls.",
    "E8_partial_completion": "Model did not complete all required steps.",
    "E10_format_error": "Model output could not be parsed as tool calls.",
}

# Canonical level order
LEVELS: list[str] = ["L0_node", "L1_chain", "L2_parallel", "L3_dag"]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _find_latest_run(base: Path) -> Path:
    """Return the most recently modified run_* directory under *base*."""
    candidates = sorted(base.glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No run_* directories found under {base}")
    return candidates[-1]


def _find_results_json(run_dir: Path) -> Path:
    """Find the results_*.json file inside a run directory."""
    candidates = sorted(run_dir.glob("results_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No results_*.json found in {run_dir}")
    return candidates[-1]


def _find_task_suite(run_dir: Path) -> Path | None:
    """Find task_suite.json inside a run directory."""
    path = run_dir / "task_suite.json"
    return path if path.exists() else None


def _load_task_suite(path: Path) -> dict[str, Any]:
    """Load the task suite JSON and return the raw dict."""
    return json.loads(path.read_text())


def _build_task_lookup(
    suite: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build a {task_id: task_dict} lookup from the task suite."""
    return {t["task_id"]: t for t in suite.get("tasks", [])}


def _fmt_pct(val: float) -> str:
    """Format a fraction as a percentage string."""
    return f"{val:.1%}"


def _print_df(df: pd.DataFrame, title: str) -> None:
    """Pretty-print a DataFrame with a section header."""
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    with pd.option_context(
        "display.max_rows", 200,
        "display.max_columns", 20,
        "display.width", 120,
        "display.float_format", lambda x: f"{x:.4f}",
    ):
        print(df.to_string(index=False))
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 1. Bootstrap CIs (delegates to existing functions)
# ═══════════════════════════════════════════════════════════════════════════


def run_bootstrap_cis(
    results: BenchmarkResults,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute per-model CIs, gap CIs, and pairwise significance tests.

    Returns:
        (ci_df, gap_df, pairwise_df) — three DataFrames.
    """
    logger.info("Computing per-model bootstrap CIs (n=%d)...", n_bootstrap)
    ci_df = compute_model_cis(results, n_bootstrap=n_bootstrap)

    logger.info("Computing composition gap CIs...")
    gap_df = compute_gap_cis(results, n_bootstrap=n_bootstrap)

    logger.info("Running pairwise significance tests...")
    pairwise_df = pairwise_significance(
        results, metric="overall", n_bootstrap=min(n_bootstrap, 5_000)
    )

    return ci_df, gap_df, pairwise_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. Random Baseline
# ═══════════════════════════════════════════════════════════════════════════


def compute_random_baseline(
    suite: dict[str, Any],
    n_simulations: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate a random-guess baseline for each composition level.

    For each task, the "random agent" picks one tool uniformly at random from
    available_tools for each expected step.  Tool-selection accuracy and
    overall score are estimated via Monte Carlo simulation.

    Args:
        suite: Parsed task_suite.json dict.
        n_simulations: Number of Monte Carlo draws per task.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: level, random_tool_acc, random_overall,
        ci_lower, ci_upper, n_tasks.
    """
    rng = np.random.default_rng(seed)
    tasks = suite.get("tasks", [])

    level_scores: dict[str, list[float]] = {lv: [] for lv in LEVELS}

    for task in tasks:
        level = task["level"]
        available = task.get("available_tools", [])
        expected_steps = task.get("expected_trace", {}).get("steps", [])
        n_expected = len(expected_steps)

        if n_expected == 0 or len(available) == 0:
            continue

        expected_tools = [s["tool_name"] for s in expected_steps]

        # Monte Carlo: how often does a random pick match expected tools?
        match_counts = np.zeros(n_simulations)
        for sim in range(n_simulations):
            random_picks = rng.choice(available, size=n_expected)
            matches = sum(
                1 for rp, et in zip(random_picks, expected_tools) if rp == et
            )
            match_counts[sim] = matches / n_expected

        avg_score = float(match_counts.mean())
        level_scores[level].append(avg_score)

    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        scores = level_scores[level]
        if not scores:
            continue
        arr = np.array(scores)
        mean_val, ci_lo, ci_hi = bootstrap_ci(arr, n_bootstrap=5_000, seed=seed)
        rows.append({
            "level": level,
            "random_tool_acc": mean_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_tasks": len(scores),
        })

    # Overall
    all_scores = [s for lv_scores in level_scores.values() for s in lv_scores]
    if all_scores:
        arr = np.array(all_scores)
        mean_val, ci_lo, ci_hi = bootstrap_ci(arr, n_bootstrap=5_000, seed=seed)
        rows.append({
            "level": "overall",
            "random_tool_acc": mean_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_tasks": len(all_scores),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Oracle Baseline (tool-selection ceiling)
# ═══════════════════════════════════════════════════════════════════════════


def compute_oracle_baseline(
    suite: dict[str, Any],
    n_simulations: int = 1_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute an oracle baseline where tool selection is perfect.

    The oracle always picks the correct tool(s) but uses random arguments
    drawn from a uniform distribution for numeric/string args.  This gives
    an upper bound on what correct tool selection alone can achieve.

    The argument accuracy is approximated: for each expected argument, a
    random guess has ~0 probability of being exactly correct for
    unconstrained arguments.  We conservatively model it as a Bernoulli
    trial with p=0.1 per arg (reflecting limited arg-value domains in
    the benchmark's synthetic tasks).

    Args:
        suite: Parsed task_suite.json dict.
        n_simulations: Monte Carlo draws per task.
        seed: Random seed.

    Returns:
        DataFrame with columns: level, oracle_tool_acc (always 1.0),
        oracle_arg_acc, oracle_overall, ci_lower, ci_upper, n_tasks.
    """
    rng = np.random.default_rng(seed)
    tasks = suite.get("tasks", [])

    # Estimate per-arg match probability from actual argument domains
    # by counting unique values across the suite
    arg_domains: dict[str, set[str]] = {}
    for task in tasks:
        for step in task.get("expected_trace", {}).get("steps", []):
            for arg_name, arg_val in step.get("arguments", {}).items():
                key = f"{step['tool_name']}.{arg_name}"
                arg_domains.setdefault(key, set()).add(str(arg_val))

    # p(random correct) = 1/|domain| for each arg
    arg_match_probs: dict[str, float] = {}
    for key, vals in arg_domains.items():
        arg_match_probs[key] = 1.0 / max(len(vals), 1)

    level_scores: dict[str, list[float]] = {lv: [] for lv in LEVELS}

    for task in tasks:
        level = task["level"]
        expected_steps = task.get("expected_trace", {}).get("steps", [])
        n_expected = len(expected_steps)
        if n_expected == 0:
            continue

        # Oracle: correct tool always, random args
        sim_scores = np.zeros(n_simulations)
        for sim in range(n_simulations):
            step_scores = []
            for step in expected_steps:
                tool = step["tool_name"]
                args = step.get("arguments", {})
                if not args:
                    step_scores.append(1.0)
                    continue
                arg_correct = 0
                for arg_name in args:
                    key = f"{tool}.{arg_name}"
                    p = arg_match_probs.get(key, 0.05)
                    if rng.random() < p:
                        arg_correct += 1
                step_scores.append(arg_correct / len(args))
            sim_scores[sim] = np.mean(step_scores)

        avg = float(sim_scores.mean())
        level_scores[level].append(avg)

    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        scores = level_scores[level]
        if not scores:
            continue
        arr = np.array(scores)
        mean_val, ci_lo, ci_hi = bootstrap_ci(arr, n_bootstrap=5_000, seed=seed)
        rows.append({
            "level": level,
            "oracle_tool_acc": 1.0,
            "oracle_arg_acc": mean_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_tasks": len(scores),
        })

    # Overall
    all_scores = [s for lv_scores in level_scores.values() for s in lv_scores]
    if all_scores:
        arr = np.array(all_scores)
        mean_val, ci_lo, ci_hi = bootstrap_ci(arr, n_bootstrap=5_000, seed=seed)
        rows.append({
            "level": "overall",
            "oracle_tool_acc": 1.0,
            "oracle_arg_acc": mean_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_tasks": len(all_scores),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Per-Category Analysis
# ═══════════════════════════════════════════════════════════════════════════


def compute_per_category(
    results: BenchmarkResults,
    suite: dict[str, Any],
    n_bootstrap: int = 5_000,
) -> pd.DataFrame:
    """Compute accuracy + bootstrap CI per (model, category).

    Categories come from task metadata['category'] in task_suite.json.

    Args:
        results: Loaded benchmark results.
        suite: Parsed task_suite.json dict.
        n_bootstrap: Bootstrap resamples for CIs.

    Returns:
        DataFrame with columns: model, category, accuracy, ci_lower,
        ci_upper, n_tasks, gap_vs_overall.
    """
    task_lookup = _build_task_lookup(suite)
    task_df = results.task_df.copy()

    if task_df.empty:
        logger.warning("task_df is empty; cannot compute per-category analysis.")
        return pd.DataFrame(
            columns=["model", "category", "accuracy", "ci_lower",
                      "ci_upper", "n_tasks", "gap_vs_overall"]
        )

    # Map task_id to category
    task_df["category"] = task_df["task_id"].map(
        lambda tid: task_lookup.get(tid, {}).get("metadata", {}).get("category", "unknown")
    )

    # Compute per-model overall accuracy for gap calculation
    model_overall: dict[str, float] = (
        task_df.groupby("model")["overall"].mean().to_dict()
    )

    rows: list[dict[str, Any]] = []
    for (model, category), group in task_df.groupby(["model", "category"]):
        scores = group["overall"].values
        if scores.size == 0:
            continue
        acc, ci_lo, ci_hi = bootstrap_ci(scores, n_bootstrap=n_bootstrap)
        overall_acc = model_overall.get(str(model), acc)
        rows.append({
            "model": model,
            "category": category,
            "accuracy": acc,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "n_tasks": scores.size,
            "gap_vs_overall": acc - overall_acc,
        })

    df = pd.DataFrame(rows).sort_values(["model", "accuracy"], ascending=[True, True])
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 5. Qualitative Error Examples
# ═══════════════════════════════════════════════════════════════════════════


def extract_error_examples(
    results: BenchmarkResults,
    suite: dict[str, Any],
    examples_per_type: int = 3,
) -> str:
    """Extract concrete error examples for each error type.

    For each error type (E1-E10), selects up to *examples_per_type* tasks,
    showing the prompt, expected tool calls, actual outcome, and error
    classification.

    Args:
        results: Loaded benchmark results.
        suite: Parsed task_suite.json dict.
        examples_per_type: Number of examples per error type.

    Returns:
        Markdown-formatted string ready for writing to a file.
    """
    task_lookup = _build_task_lookup(suite)
    raw = results.raw
    sections: list[str] = []

    sections.append("# CompToolBench — Qualitative Error Examples\n")
    sections.append(
        "Concrete examples for each error type, showing task prompt, "
        "expected tool calls, and model output.\n"
    )
    sections.append("---\n")

    # Collect all task results grouped by error type across all models
    error_examples: dict[str, list[dict[str, Any]]] = {}

    for model_key, model_data in raw.get("models", {}).items():
        model_name = model_data["model"]["name"]
        for tr in model_data.get("task_results", []):
            error_type = tr["score"].get("error_type")
            if error_type is None:
                continue
            error_examples.setdefault(error_type, []).append({
                "model": model_name,
                "task_result": tr,
            })

    # Sort error types for deterministic output
    sorted_errors = sorted(error_examples.keys())

    for error_type in sorted_errors:
        description = ERROR_DESCRIPTIONS.get(error_type, "Unknown error type.")
        sections.append(f"## {error_type}\n")
        sections.append(f"**Description:** {description}\n")

        examples = error_examples[error_type][:examples_per_type]

        for i, ex in enumerate(examples, start=1):
            model = ex["model"]
            tr = ex["task_result"]
            task_id = tr["task_id"]
            score = tr["score"]
            task_data = task_lookup.get(task_id)

            sections.append(f"### Example {i} ({model}, {task_id})\n")

            # Task prompt
            if task_data:
                prompt = task_data.get("prompt", "_Prompt not available._")
                available_tools = task_data.get("available_tools", [])
                expected_trace = task_data.get("expected_trace", {})
                expected_steps = expected_trace.get("steps", [])

                sections.append(f"**Prompt:**\n> {prompt}\n")
                sections.append(
                    f"**Available tools:** {', '.join(available_tools)}\n"
                )

                # Expected tool calls
                sections.append("**Expected tool calls:**\n")
                for step in expected_steps:
                    tool = step["tool_name"]
                    args = step.get("arguments", {})
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    deps = step.get("depends_on", [])
                    dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
                    sections.append(
                        f"- `{step['step_id']}`: **{tool}**({args_str}){dep_str}\n"
                    )
            else:
                sections.append(f"_Task {task_id} not found in task suite._\n")

            # Actual model output (from score data)
            sections.append("\n**Model output (from scoring):**\n")
            call_scores = score.get("call_scores", [])
            if call_scores:
                for cs in call_scores:
                    step_label = cs.get("step", "?")
                    matched = cs.get("matched", False)
                    tool_ok = cs.get("tool_correct", False)
                    arg_score = cs.get("args_score", 0.0)
                    status = "MATCHED" if matched else "MISSING"
                    tool_status = "correct" if tool_ok else "WRONG"
                    sections.append(
                        f"- `{step_label}`: {status}, "
                        f"tool={tool_status}, args_score={arg_score:.2f}\n"
                    )
            else:
                sections.append("- _No tool calls parsed from model output._\n")

            # Score summary
            sections.append(
                f"\n**Score:** overall={score.get('overall', 0):.2f}, "
                f"tool_seq={score.get('tool_sequence_score', 0):.2f}, "
                f"args={score.get('argument_score', 0):.2f}, "
                f"completeness={score.get('completeness_score', 0):.2f}, "
                f"data_flow={score.get('data_flow_score', 0):.2f}\n"
            )
            sections.append(f"**Error classification:** `{error_type}`\n")
            sections.append("---\n")

    # Summary table: error frequency across models
    sections.append("## Error Frequency Summary\n")
    error_df = results.error_df
    if not error_df.empty:
        pivot = error_df.pivot_table(
            index="error_type", columns="model", values="count",
            fill_value=0, aggfunc="sum",
        )
        # Manual markdown table (avoid tabulate dependency)
        cols = list(pivot.columns)
        header = "| error_type | " + " | ".join(str(c) for c in cols) + " |"
        sep_line = "|---|" + "|".join("---" for _ in cols) + "|"
        table_rows = [header, sep_line]
        for idx, row in pivot.iterrows():
            vals = " | ".join(str(int(row[c])) for c in cols)
            table_rows.append(f"| {idx} | {vals} |")
        sections.append("\n".join(table_rows))
        sections.append("\n")
    else:
        sections.append("_No error distribution data available._\n")

    return "\n".join(sections)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Combined Baselines Table
# ═══════════════════════════════════════════════════════════════════════════


def build_baselines_table(
    results: BenchmarkResults,
    random_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine model results with random and oracle baselines.

    Returns:
        DataFrame with columns: agent, level, accuracy, ci_lower, ci_upper,
        n_tasks.
    """
    rows: list[dict[str, Any]] = []

    # Model results
    ci_df = compute_model_cis(results, n_bootstrap=5_000)
    for _, row in ci_df.iterrows():
        rows.append({
            "agent": row["model"],
            "level": row["level"],
            "accuracy": row["accuracy"],
            "ci_lower": row["ci_lower"],
            "ci_upper": row["ci_upper"],
            "n_tasks": int(row["n_tasks"]),
        })

    # Random baseline
    for _, row in random_df.iterrows():
        rows.append({
            "agent": "Random Baseline",
            "level": row["level"],
            "accuracy": row["random_tool_acc"],
            "ci_lower": row["ci_lower"],
            "ci_upper": row["ci_upper"],
            "n_tasks": int(row["n_tasks"]),
        })

    # Oracle baseline
    for _, row in oracle_df.iterrows():
        rows.append({
            "agent": "Oracle (correct tool, random args)",
            "level": row["level"],
            "accuracy": row["oracle_arg_acc"],
            "ci_lower": row["ci_lower"],
            "ci_upper": row["ci_upper"],
            "n_tasks": int(row["n_tasks"]),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis for CompToolBench results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Path to a run directory (e.g. results/run_20260311_000147). "
            "Auto-detects latest run if omitted."
        ),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples (default: 10000).",
    )
    parser.add_argument(
        "--examples-per-type",
        type=int,
        default=3,
        help="Number of qualitative error examples per error type (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full analysis pipeline."""
    args = parse_args()

    # --- Locate results ---
    results_base = Path(__file__).parent.parent / "results"

    if args.results_dir:
        run_dir = Path(args.results_dir)
    else:
        run_dir = _find_latest_run(results_base)

    results_json = _find_results_json(run_dir)
    task_suite_path = _find_task_suite(run_dir)

    logger.info("Run directory:  %s", run_dir)
    logger.info("Results JSON:   %s", results_json)
    logger.info("Task suite:     %s", task_suite_path or "NOT FOUND")

    # --- Load data ---
    results = load_results(results_json)
    logger.info(
        "Loaded %d models, %d tasks/model",
        results.n_models,
        results.n_tasks,
    )

    suite: dict[str, Any] = {}
    if task_suite_path:
        suite = _load_task_suite(task_suite_path)
        logger.info("Loaded task suite with %d tasks", len(suite.get("tasks", [])))
    else:
        logger.warning(
            "No task_suite.json found — random/oracle baselines and "
            "per-category analysis will be limited."
        )

    # --- Output directory ---
    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # ═══════════════════════════════════════════════════════════════════
    # 1. Bootstrap CIs
    # ═══════════════════════════════════════════════════════════════════
    ci_df, gap_df, pairwise_df = run_bootstrap_cis(results, args.n_bootstrap)

    ci_path = output_dir / "statistical_summary.csv"
    ci_df.to_csv(ci_path, index=False)
    _print_df(ci_df, "Per-Model Accuracy with 95% Bootstrap CIs")

    gap_path = output_dir / "gap_significance.csv"
    gap_df.to_csv(gap_path, index=False)
    _print_df(gap_df, "Composition Gap with 95% Bootstrap CIs")

    pairwise_path = output_dir / "pairwise_tests.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    _print_df(pairwise_df, "Pairwise Significance Tests (Bonferroni-corrected)")

    # ═══════════════════════════════════════════════════════════════════
    # 2 & 3. Random + Oracle Baselines
    # ═══════════════════════════════════════════════════════════════════
    if suite:
        logger.info("Computing random baseline (Monte Carlo)...")
        random_df = compute_random_baseline(suite)
        _print_df(random_df, "Random Baseline (uniform tool selection)")

        logger.info("Computing oracle baseline (perfect tool, random args)...")
        oracle_df = compute_oracle_baseline(suite)
        _print_df(oracle_df, "Oracle Baseline (correct tool, random arguments)")

        baselines_df = build_baselines_table(results, random_df, oracle_df)
        baselines_path = output_dir / "baselines.csv"
        baselines_df.to_csv(baselines_path, index=False)
        _print_df(baselines_df, "All Agents + Baselines Comparison")
    else:
        logger.warning("Skipping baselines — no task_suite.json available.")
        random_df = pd.DataFrame()
        oracle_df = pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════
    # 4. Per-Category Analysis
    # ═══════════════════════════════════════════════════════════════════
    if suite:
        logger.info("Computing per-category analysis...")
        category_df = compute_per_category(results, suite, n_bootstrap=args.n_bootstrap)
        category_path = output_dir / "category_breakdown.csv"
        category_df.to_csv(category_path, index=False)
        _print_df(category_df, "Per-Category Accuracy Breakdown")

        # Identify categories that drive the Selection Gap
        logger.info("Identifying categories that drive the Selection Gap...")
        _print_selection_gap_drivers(category_df)
    else:
        logger.warning("Skipping per-category analysis — no task_suite.json.")

    # ═══════════════════════════════════════════════════════════════════
    # 5. Qualitative Error Examples
    # ═══════════════════════════════════════════════════════════════════
    if suite:
        logger.info("Extracting qualitative error examples...")
        error_md = extract_error_examples(
            results, suite, examples_per_type=args.examples_per_type
        )
        error_path = output_dir / "error_examples.md"
        error_path.write_text(error_md)
        logger.info("Wrote error examples to %s", error_path)
    else:
        logger.warning("Skipping error examples — no task_suite.json.")

    # ═══════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════
    _print_final_summary(ci_df, gap_df, pairwise_df, random_df, oracle_df)

    logger.info("Analysis complete. All outputs saved to: %s", output_dir)
    logger.info("Files generated:")
    for f in sorted(output_dir.iterdir()):
        logger.info("  %s (%s)", f.name, _human_size(f.stat().st_size))


def _print_selection_gap_drivers(category_df: pd.DataFrame) -> None:
    """Identify and print categories that drive the largest accuracy gaps."""
    if category_df.empty:
        return

    # Pivot: category x model -> accuracy
    pivot = category_df.pivot_table(
        index="category", columns="model", values="accuracy"
    )

    if pivot.shape[1] < 2:
        print("  (Need 2+ models for gap analysis)")
        return

    # Compute per-category variance across models (high variance = gap driver)
    pivot["model_variance"] = pivot.var(axis=1)
    pivot["model_range"] = pivot.max(axis=1) - pivot.min(axis=1)
    pivot = pivot.sort_values("model_range", ascending=False)

    print("\n  Categories ranked by cross-model accuracy range:")
    print("  (Larger range = bigger driver of model differences)\n")
    for cat in pivot.index:
        rng = pivot.loc[cat, "model_range"]
        print(f"    {cat:30s}  range = {rng:.3f}")
    print()


def _print_final_summary(
    ci_df: pd.DataFrame,
    gap_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    random_df: pd.DataFrame,
    oracle_df: pd.DataFrame,
) -> None:
    """Print a concise final summary to stdout."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  FULL ANALYSIS SUMMARY")
    print(sep)

    # Best and worst models
    overall = ci_df[ci_df["level"] == "overall"].copy()
    if not overall.empty:
        best = overall.loc[overall["accuracy"].idxmax()]
        worst = overall.loc[overall["accuracy"].idxmin()]
        print(f"\n  Best model:  {best['model']}  "
              f"{best['accuracy']:.1%} [{best['ci_lower']:.1%}, {best['ci_upper']:.1%}]")
        print(f"  Worst model: {worst['model']}  "
              f"{worst['accuracy']:.1%} [{worst['ci_lower']:.1%}, {worst['ci_upper']:.1%}]")

    # Baselines
    if not random_df.empty:
        rand_overall = random_df[random_df["level"] == "overall"]
        if not rand_overall.empty:
            rv = rand_overall.iloc[0]["random_tool_acc"]
            print(f"  Random baseline: {rv:.1%}")

    if not oracle_df.empty:
        oracle_overall = oracle_df[oracle_df["level"] == "overall"]
        if not oracle_overall.empty:
            ov = oracle_overall.iloc[0]["oracle_arg_acc"]
            print(f"  Oracle baseline: {ov:.1%}")

    # Significant differences
    if not pairwise_df.empty:
        n_sig = pairwise_df["significant"].sum()
        n_total = len(pairwise_df)
        print(f"\n  Pairwise tests: {n_sig}/{n_total} significant "
              f"(Bonferroni-corrected)")

    # Gap summary
    if not gap_df.empty:
        overall_gaps = gap_df[gap_df["gap_level"] == "overall"]
        if not overall_gaps.empty:
            mean_gap = overall_gaps["gap"].mean()
            print(f"  Mean composition gap (overall): {mean_gap:+.1%}")

    print(f"\n{sep}\n")


def _human_size(nbytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ["B", "KB", "MB"]:
        if nbytes < 1024:
            return f"{nbytes:.0f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


if __name__ == "__main__":
    main()
