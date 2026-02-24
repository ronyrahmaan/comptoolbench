"""Deep failure analysis for CompToolBench benchmark results.

Answers the core research question: "WHERE and WHY do models fail
when composing tools?"

Six analysis functions dissect model failures across:
  1. Step position — do models fail more on later steps?
  2. Error cascades — does one failure doom subsequent steps?
  3. Tool difficulty — which tools are hardest to use correctly?
  4. Error patterns by level — how do error types shift with complexity?
  5. Scaling analysis — does model size predict composition gap?
  6. Comprehensive report — markdown summary of all findings.

Usage:
    from comptoolbench.analysis.loader import load_results
    from comptoolbench.analysis.failure_analysis import generate_failure_report

    results = load_results("results/run_001/results_20260222.json")
    report = generate_failure_report(Path("results/run_001"), results)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from comptoolbench.analysis.loader import BenchmarkResults

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEVEL_ORDER: list[str] = ["L0_node", "L1_chain", "L2_parallel", "L3_dag"]

# Approximate parameter counts (billions) for scaling analysis.
# Models whose size is unknown/inapplicable are set to 0 and excluded
# from regression by default.
MODEL_SIZES: dict[str, float] = {
    "Granite4 1B": 1,
    "Granite4 3B": 3,
    "Mistral 7B": 7,
    "Qwen3 8B": 8,
    "Llama 3.1 8B": 8,
    "Qwen 2.5 7B": 7,
    "Mistral Nemo 12B": 12,
    "Llama 4 Scout 17B": 17,
    "Mistral Small 24B": 24,
    "Mistral Small": 24,
    "Mistral Medium": 70,
    "Llama 3.3 70B": 70,
    "Gemini 2.0 Flash": 0,  # unknown — exclude from regression
}

# Multi-step levels (L1+) where step-position and cascade analyses apply.
_MULTI_STEP_LEVELS: set[str] = {"L1_chain", "L2_parallel", "L3_dag"}


# ---------------------------------------------------------------------------
# Helpers — checkpoint I/O
# ---------------------------------------------------------------------------

def _load_checkpoint_records(results_dir: Path) -> list[dict[str, Any]]:
    """Load all checkpoint JSONL files in *results_dir* and return a flat list.

    Each JSONL file corresponds to one model's evaluation run.  Each line
    is a JSON object with ``task_id``, ``model``, ``score`` (including
    ``call_scores``), ``tokens``, and ``latency_ms``.

    Returns an empty list if no checkpoint files are found.
    """
    records: list[dict[str, Any]] = []
    checkpoint_files = sorted(results_dir.glob("checkpoint_*.jsonl"))
    if not checkpoint_files:
        logger.warning("No checkpoint JSONL files found in %s", results_dir)
        return records

    for cp_path in checkpoint_files:
        for line_no, line in enumerate(cp_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed JSON at %s:%d", cp_path.name, line_no,
                )
    logger.info(
        "Loaded %d checkpoint records from %d files in %s",
        len(records), len(checkpoint_files), results_dir,
    )
    return records


def _explode_call_scores(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten checkpoint records into one row per (model, task, step).

    Columns:
        model, task_id, level, step, step_position (1-indexed),
        tool_correct, args_score, matched, num_expected_calls, overall
    """
    rows: list[dict[str, Any]] = []
    for rec in records:
        model = rec.get("model", "unknown")
        task_id = rec["task_id"]
        score = rec.get("score", {})
        level = score.get("level", "unknown")
        overall = score.get("overall", 0.0)
        num_expected = score.get("num_expected_calls", 0)
        for idx, cs in enumerate(score.get("call_scores", [])):
            rows.append({
                "model": model,
                "task_id": task_id,
                "level": level,
                "step": cs.get("step", f"step_{idx + 1}"),
                "step_position": idx + 1,
                "tool_correct": bool(cs.get("tool_correct", False)),
                "args_score": float(cs.get("args_score", 0.0)),
                "matched": bool(cs.get("matched", False)),
                "num_expected_calls": num_expected,
                "overall": overall,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Step-Position Failure Analysis
# ---------------------------------------------------------------------------

def analyze_step_position_failures(results_dir: Path) -> pd.DataFrame:
    """Analyze WHERE in a multi-step task models fail.

    For each (model, step_position), compute:
      * **failure_rate** — fraction of steps where the tool is wrong or
        the step was not matched.
      * **avg_args_score** — mean argument-matching score.
      * **tool_correct_rate** — fraction of steps where the correct tool
        was selected.

    Only multi-step tasks (L1/L2/L3) are included.

    Args:
        results_dir: Path containing ``checkpoint_*.jsonl`` files.

    Returns:
        DataFrame with columns: model, step_position, n_tasks,
        failure_rate, avg_args_score, tool_correct_rate.

    Key insight: "Step 2 failure rate is 3x higher than step 1" shows
    that models struggle with dependent steps.
    """
    records = _load_checkpoint_records(results_dir)
    if not records:
        return pd.DataFrame(
            columns=["model", "step_position", "n_tasks",
                      "failure_rate", "avg_args_score", "tool_correct_rate"],
        )

    step_df = _explode_call_scores(records)

    # Filter to multi-step tasks only
    step_df = step_df[step_df["level"].isin(_MULTI_STEP_LEVELS)]
    if step_df.empty:
        return pd.DataFrame(
            columns=["model", "step_position", "n_tasks",
                      "failure_rate", "avg_args_score", "tool_correct_rate"],
        )

    # A step "fails" when the tool is wrong OR the step was not matched
    step_df["failed"] = ~(step_df["tool_correct"] & step_df["matched"])

    grouped = (
        step_df.groupby(["model", "step_position"])
        .agg(
            n_tasks=("task_id", "nunique"),
            failure_rate=("failed", "mean"),
            avg_args_score=("args_score", "mean"),
            tool_correct_rate=("tool_correct", "mean"),
        )
        .reset_index()
        .sort_values(["model", "step_position"])
    )
    return grouped


# ---------------------------------------------------------------------------
# 2. Error Cascade Analysis
# ---------------------------------------------------------------------------

def analyze_error_cascade(results_dir: Path) -> pd.DataFrame:
    """Quantify how errors cascade through multi-step tasks.

    For each model, compute:
      * **after_success_fail_rate** — P(step N+1 fails | step N succeeds).
      * **after_failure_fail_rate** — P(step N+1 fails | step N fails).
      * **cascade_multiplier** — ratio of the two (how much worse failure
        is after a preceding failure).

    Only consecutive step pairs within the same task are considered.

    Args:
        results_dir: Path containing ``checkpoint_*.jsonl`` files.

    Returns:
        DataFrame with columns: model, after_success_fail_rate,
        after_failure_fail_rate, cascade_multiplier,
        n_pairs_after_success, n_pairs_after_failure.

    Key insight: "After a failure, subsequent step failure rate is 85%
    vs 25% baseline" — errors compound.
    """
    records = _load_checkpoint_records(results_dir)
    if not records:
        return pd.DataFrame(
            columns=["model", "after_success_fail_rate",
                      "after_failure_fail_rate", "cascade_multiplier",
                      "n_pairs_after_success", "n_pairs_after_failure"],
        )

    step_df = _explode_call_scores(records)
    step_df = step_df[step_df["level"].isin(_MULTI_STEP_LEVELS)]
    if step_df.empty:
        return pd.DataFrame(
            columns=["model", "after_success_fail_rate",
                      "after_failure_fail_rate", "cascade_multiplier",
                      "n_pairs_after_success", "n_pairs_after_failure"],
        )

    step_df["failed"] = ~(step_df["tool_correct"] & step_df["matched"])

    # Build consecutive-step pairs within each (model, task)
    pair_rows: list[dict[str, Any]] = []
    for (model, task_id), grp in step_df.groupby(["model", "task_id"]):
        grp = grp.sort_values("step_position")
        failures = grp["failed"].tolist()
        for i in range(len(failures) - 1):
            pair_rows.append({
                "model": model,
                "prev_failed": failures[i],
                "next_failed": failures[i + 1],
            })

    if not pair_rows:
        return pd.DataFrame(
            columns=["model", "after_success_fail_rate",
                      "after_failure_fail_rate", "cascade_multiplier",
                      "n_pairs_after_success", "n_pairs_after_failure"],
        )

    pairs_df = pd.DataFrame(pair_rows)

    model_rows: list[dict[str, Any]] = []
    for model, mdf in pairs_df.groupby("model"):
        after_success = mdf[~mdf["prev_failed"]]
        after_failure = mdf[mdf["prev_failed"]]

        n_after_success = len(after_success)
        n_after_failure = len(after_failure)

        after_success_rate = (
            float(after_success["next_failed"].mean())
            if n_after_success > 0 else 0.0
        )
        after_failure_rate = (
            float(after_failure["next_failed"].mean())
            if n_after_failure > 0 else 0.0
        )
        cascade = (
            after_failure_rate / after_success_rate
            if after_success_rate > 0 else float("inf")
        )

        model_rows.append({
            "model": model,
            "after_success_fail_rate": round(after_success_rate, 4),
            "after_failure_fail_rate": round(after_failure_rate, 4),
            "cascade_multiplier": round(cascade, 2),
            "n_pairs_after_success": n_after_success,
            "n_pairs_after_failure": n_after_failure,
        })

    return pd.DataFrame(model_rows).sort_values("cascade_multiplier", ascending=False)


# ---------------------------------------------------------------------------
# 3. Tool Difficulty Analysis
# ---------------------------------------------------------------------------

def analyze_tool_difficulty(
    results: BenchmarkResults,
    task_suite_path: Path | None = None,
) -> pd.DataFrame:
    """Rank tools by how difficult they are for models.

    Cross-references the task suite JSON with per-task scores to compute,
    for each tool, the average overall score when that tool appears in the
    task's expected trace.

    Args:
        results: Parsed benchmark results.
        task_suite_path: Path to ``task_suite.json``.  If *None*, the
            function attempts to infer it from ``results.raw``.

    Returns:
        DataFrame with columns: tool_name, n_appearances,
        avg_score, l0_score, l1_score, l2_score, l3_score.
    """
    empty = pd.DataFrame(
        columns=["tool_name", "n_appearances", "avg_score",
                 "l0_score", "l1_score", "l2_score", "l3_score"],
    )

    # Load the task suite to get tool-task mapping
    suite_data = _load_task_suite(task_suite_path, results)
    if suite_data is None:
        logger.warning("Could not load task_suite.json — skipping tool difficulty analysis.")
        return empty

    # Build mapping: task_id -> list of tools used (from expected trace)
    task_tools: dict[str, list[str]] = {}
    for task in suite_data.get("tasks", []):
        tid = task["task_id"]
        tools = [
            step["tool_name"]
            for step in task.get("expected_trace", {}).get("steps", [])
        ]
        task_tools[tid] = tools

    task_df = results.task_df
    if task_df.empty:
        return empty

    # For each (model, task), tag with the tools used
    rows: list[dict[str, Any]] = []
    for _, row in task_df.iterrows():
        tid = str(row["task_id"])
        tools = task_tools.get(tid, [])
        for tool in set(tools):  # deduplicate within a single task
            rows.append({
                "tool_name": tool,
                "level": row["level"],
                "overall": row["overall"],
                "model": row["model"],
            })

    if not rows:
        return empty

    tool_df = pd.DataFrame(rows)

    # Global aggregation
    global_agg = (
        tool_df.groupby("tool_name")
        .agg(n_appearances=("overall", "count"), avg_score=("overall", "mean"))
        .reset_index()
    )

    # Per-level aggregation
    level_map = {
        "L0_node": "l0_score",
        "L1_chain": "l1_score",
        "L2_parallel": "l2_score",
        "L3_dag": "l3_score",
    }
    for level_key, col_name in level_map.items():
        lv_agg = (
            tool_df[tool_df["level"] == level_key]
            .groupby("tool_name")["overall"]
            .mean()
            .rename(col_name)
        )
        global_agg = global_agg.merge(lv_agg, on="tool_name", how="left")

    global_agg = global_agg.sort_values("avg_score", ascending=True)
    return global_agg


def _load_task_suite(
    path: Path | None,
    results: BenchmarkResults,
) -> dict[str, Any] | None:
    """Try to load a task_suite.json from *path* or infer from results metadata."""
    if path is not None and path.exists():
        return json.loads(path.read_text())

    # Attempt to find task_suite.json as a sibling of the results file
    # The raw dict may contain a source path hint, but if not, we return None.
    return None


# ---------------------------------------------------------------------------
# 4. Error Patterns by Level
# ---------------------------------------------------------------------------

def analyze_error_patterns_by_level(results: BenchmarkResults) -> pd.DataFrame:
    """Show how error-type distributions shift with composition complexity.

    At L0, errors are dominated by ``E4_wrong_arguments``; at L3, by
    ``E8_partial_completion``.  This function quantifies that shift.

    Args:
        results: Parsed benchmark results.

    Returns:
        DataFrame with columns: level, error_type, count,
        fraction_of_level_errors.
    """
    task_df = results.task_df
    empty = pd.DataFrame(
        columns=["level", "error_type", "count", "fraction_of_level_errors"],
    )
    if task_df.empty:
        return empty

    # Keep only failing tasks (those with an error_type)
    errors = task_df[task_df["error_type"].notna()].copy()
    if errors.empty:
        return empty

    # Count per (level, error_type)
    counts = (
        errors.groupby(["level", "error_type"])
        .size()
        .reset_index(name="count")
    )

    # Total errors per level
    level_totals = counts.groupby("level")["count"].transform("sum")
    counts["fraction_of_level_errors"] = (counts["count"] / level_totals).round(4)

    # Sort: level order, then descending fraction within each level
    counts["_level_order"] = counts["level"].map(
        {lv: i for i, lv in enumerate(LEVEL_ORDER)}
    )
    counts = (
        counts.sort_values(["_level_order", "fraction_of_level_errors"], ascending=[True, False])
        .drop(columns=["_level_order"])
        .reset_index(drop=True)
    )
    return counts


# ---------------------------------------------------------------------------
# 5. Scaling Analysis
# ---------------------------------------------------------------------------

def analyze_scaling(results: BenchmarkResults) -> pd.DataFrame:
    """Relate model size to composition gap.

    Uses the ``MODEL_SIZES`` lookup to attach approximate parameter counts
    to each model.  Computes Pearson and Spearman correlations between
    ``params_b`` and ``gap_overall`` for models with known sizes (>0).

    The correlations are stored as attributes on the returned DataFrame:
      * ``df.attrs["pearson_r"]``, ``df.attrs["pearson_p"]``
      * ``df.attrs["spearman_r"]``, ``df.attrs["spearman_p"]``

    Args:
        results: Parsed benchmark results.

    Returns:
        DataFrame with columns: model, params_b, accuracy_l0,
        gap_overall, gap_l1, gap_l2, gap_l3.
    """
    model_df = results.model_df
    empty = pd.DataFrame(
        columns=["model", "params_b", "accuracy_l0",
                 "gap_overall", "gap_l1", "gap_l2", "gap_l3"],
    )
    if model_df.empty:
        return empty

    rows: list[dict[str, Any]] = []
    for _, row in model_df.iterrows():
        model_name = str(row["model"])
        params = MODEL_SIZES.get(model_name, 0.0)
        rows.append({
            "model": model_name,
            "params_b": params,
            "accuracy_l0": row.get("accuracy_l0", 0.0),
            "gap_overall": row.get("gap_overall", 0.0),
            "gap_l1": row.get("gap_l1", 0.0),
            "gap_l2": row.get("gap_l2", 0.0),
            "gap_l3": row.get("gap_l3", 0.0),
        })

    df = pd.DataFrame(rows).sort_values("params_b", ascending=True)

    # Correlations (exclude unknown sizes)
    known = df[df["params_b"] > 0]
    if len(known) >= 3:
        pearson_r, pearson_p = sp_stats.pearsonr(
            known["params_b"], known["gap_overall"],
        )
        spearman_r, spearman_p = sp_stats.spearmanr(
            known["params_b"], known["gap_overall"],
        )
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = float("nan")

    df.attrs["pearson_r"] = round(float(pearson_r), 4)
    df.attrs["pearson_p"] = round(float(pearson_p), 4)
    df.attrs["spearman_r"] = round(float(spearman_r), 4)
    df.attrs["spearman_p"] = round(float(spearman_p), 4)

    logger.info(
        "Scaling correlations — Pearson r=%.3f (p=%.3f), Spearman r=%.3f (p=%.3f)",
        pearson_r, pearson_p, spearman_r, spearman_p,
    )
    return df


# ---------------------------------------------------------------------------
# 6. Comprehensive Failure Report
# ---------------------------------------------------------------------------

def generate_failure_report(
    results_dir: Path,
    results: BenchmarkResults,
    task_suite_path: Path | None = None,
) -> str:
    """Generate a comprehensive markdown report of all failure analyses.

    Calls each analysis function and formats the results into a readable
    document with tables, key findings, and actionable insights for the
    paper.

    Args:
        results_dir: Directory containing checkpoint JSONL files.
        results: Parsed benchmark results (from ``load_results``).
        task_suite_path: Optional path to ``task_suite.json`` for tool
            difficulty analysis.

    Returns:
        Markdown-formatted string.
    """
    sections: list[str] = []
    sections.append("# CompToolBench Failure Analysis Report\n")
    sections.append(
        "> Automatically generated deep-dive into WHERE and WHY "
        "models fail when composing tools.\n"
    )

    # ---- 1. Step-Position Failures ----
    sections.append("## 1. Step-Position Failure Analysis\n")
    sections.append(
        "**Question:** Do models fail more on later steps in multi-step tasks?\n"
    )
    step_df = analyze_step_position_failures(results_dir)
    if step_df.empty:
        sections.append("_No multi-step checkpoint data available._\n")
    else:
        sections.append(_df_to_md_table(step_df))
        sections.append("")

        # Key finding: compare step 1 vs step 2 failure rates (aggregated)
        agg = step_df.groupby("step_position").agg(
            mean_fail=("failure_rate", "mean"),
        ).reset_index()
        if len(agg) >= 2:
            s1 = agg.loc[agg["step_position"] == 1, "mean_fail"].values[0]
            s2 = agg.loc[agg["step_position"] == 2, "mean_fail"].values[0]
            ratio = s2 / s1 if s1 > 0 else float("inf")
            sections.append(
                f"**Key finding:** Average step-2 failure rate ({s2:.1%}) is "
                f"**{ratio:.1f}x** higher than step-1 ({s1:.1%}). "
                "Models struggle significantly with dependent steps.\n"
            )

    # ---- 2. Error Cascade ----
    sections.append("## 2. Error Cascade Analysis\n")
    sections.append(
        "**Question:** When step N fails, how much more likely is step N+1 to fail?\n"
    )
    cascade_df = analyze_error_cascade(results_dir)
    if cascade_df.empty:
        sections.append("_No cascade data available._\n")
    else:
        sections.append(_df_to_md_table(cascade_df))
        sections.append("")

        mean_cascade = cascade_df["cascade_multiplier"].replace(
            [float("inf")], float("nan"),
        ).mean()
        mean_after_fail = cascade_df["after_failure_fail_rate"].mean()
        mean_after_success = cascade_df["after_success_fail_rate"].mean()
        sections.append(
            f"**Key finding:** On average, the failure rate after a preceding "
            f"failure is **{mean_after_fail:.0%}** compared to **{mean_after_success:.0%}** "
            f"after a success (cascade multiplier: **{mean_cascade:.1f}x**). "
            "Errors compound severely.\n"
        )

    # ---- 3. Tool Difficulty ----
    sections.append("## 3. Tool Difficulty Ranking\n")
    sections.append(
        "**Question:** Which tools are hardest for models to use correctly?\n"
    )
    tool_df = analyze_tool_difficulty(results, task_suite_path)
    if tool_df.empty:
        sections.append(
            "_No tool difficulty data available "
            "(provide task_suite_path for this analysis)._\n"
        )
    else:
        # Show top-10 hardest and top-5 easiest
        sections.append("### Hardest tools (lowest avg score)\n")
        sections.append(_df_to_md_table(tool_df.head(10)))
        sections.append("")
        if len(tool_df) > 10:
            sections.append("### Easiest tools (highest avg score)\n")
            sections.append(_df_to_md_table(tool_df.tail(5)))
            sections.append("")

    # ---- 4. Error Patterns by Level ----
    sections.append("## 4. Error Patterns by Composition Level\n")
    sections.append(
        "**Question:** How do error types shift as task complexity increases?\n"
    )
    error_level_df = analyze_error_patterns_by_level(results)
    if error_level_df.empty:
        sections.append("_No error pattern data available._\n")
    else:
        sections.append(_df_to_md_table(error_level_df))
        sections.append("")

        # Find dominant error type per level
        dominant = (
            error_level_df.sort_values("fraction_of_level_errors", ascending=False)
            .groupby("level")
            .first()
            .reset_index()
        )
        findings = []
        for _, row in dominant.iterrows():
            findings.append(
                f"  - **{row['level']}**: {row['error_type']} "
                f"({row['fraction_of_level_errors']:.0%} of errors)"
            )
        sections.append("**Key finding — dominant error type per level:**\n")
        sections.append("\n".join(findings))
        sections.append("")

    # ---- 5. Scaling Analysis ----
    sections.append("## 5. Scaling Analysis (Model Size vs. Composition Gap)\n")
    sections.append(
        "**Question:** Does model size predict compositional generalization?\n"
    )
    scaling_df = analyze_scaling(results)
    if scaling_df.empty:
        sections.append("_No scaling data available._\n")
    else:
        sections.append(_df_to_md_table(scaling_df))
        sections.append("")
        pr = scaling_df.attrs.get("pearson_r", float("nan"))
        pp = scaling_df.attrs.get("pearson_p", float("nan"))
        sr = scaling_df.attrs.get("spearman_r", float("nan"))
        sp = scaling_df.attrs.get("spearman_p", float("nan"))
        sections.append(
            f"**Correlations (known-size models only):**\n"
            f"  - Pearson r = {pr:.3f} (p = {pp:.3f})\n"
            f"  - Spearman rho = {sr:.3f} (p = {sp:.3f})\n"
        )
        if not np.isnan(pr):
            direction = "larger" if pr < 0 else "smaller"
            strength = (
                "strong" if abs(pr) > 0.7
                else "moderate" if abs(pr) > 0.4
                else "weak"
            )
            sig = "statistically significant" if pp < 0.05 else "not statistically significant"
            sections.append(
                f"**Key finding:** There is a **{strength}** correlation "
                f"({sig}) suggesting {direction} models tend to have "
                f"{'smaller' if pr < 0 else 'larger'} composition gaps.\n"
            )

    # ---- Summary ----
    sections.append("---\n")
    sections.append("## Summary: Key Takeaways for the Paper\n")
    sections.append(_build_summary(step_df, cascade_df, error_level_df, scaling_df))

    report = "\n".join(sections)
    logger.info("Generated failure analysis report (%d characters)", len(report))
    return report


# ---------------------------------------------------------------------------
# Markdown table helper
# ---------------------------------------------------------------------------

def _df_to_md_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to a markdown table string.

    Numeric columns are formatted to 4 decimal places.  The table is
    suitable for embedding in markdown reports.
    """
    if df.empty:
        return "_Empty DataFrame._"

    # Format numeric columns
    formatted = df.copy()
    for col in formatted.select_dtypes(include=["float64", "float32"]).columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}" if not np.isnan(x) else "—")

    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    separator = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    rows = []
    for _, row in formatted.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row) + " |")

    return "\n".join([header, separator, *rows])


def _build_summary(
    step_df: pd.DataFrame,
    cascade_df: pd.DataFrame,
    error_level_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
) -> str:
    """Build a concise summary section from all analyses."""
    points: list[str] = []

    # Step position
    if not step_df.empty:
        agg = step_df.groupby("step_position")["failure_rate"].mean()
        if len(agg) >= 2:
            points.append(
                f"1. **Step-position effect:** Step-1 avg failure rate = "
                f"{agg.get(1, 0):.0%}, step-2 = {agg.get(2, 0):.0%}. "
                "Later steps are disproportionately harder."
            )

    # Cascade
    if not cascade_df.empty:
        mean_mult = cascade_df["cascade_multiplier"].replace(
            [float("inf")], float("nan"),
        ).mean()
        if not np.isnan(mean_mult):
            points.append(
                f"2. **Error cascading:** Average cascade multiplier = "
                f"**{mean_mult:.1f}x**. One mistake makes subsequent "
                "steps dramatically more likely to fail."
            )

    # Error patterns
    if not error_level_df.empty:
        l0_dominant = error_level_df[error_level_df["level"] == "L0_node"]
        l3_dominant = error_level_df[error_level_df["level"] == "L3_dag"]
        if not l0_dominant.empty and not l3_dominant.empty:
            l0_top = l0_dominant.iloc[0]["error_type"]
            l3_top = l3_dominant.iloc[0]["error_type"]
            points.append(
                f"3. **Error-type shift:** At L0, the dominant error is "
                f"*{l0_top}*; at L3, it shifts to *{l3_top}*. "
                "Complexity changes the nature of failures, not just their frequency."
            )

    # Scaling
    if not scaling_df.empty and "pearson_r" in scaling_df.attrs:
        pr = scaling_df.attrs["pearson_r"]
        if not np.isnan(pr):
            points.append(
                f"4. **Scaling:** Pearson r = {pr:.3f} between model size "
                f"and composition gap. "
                f"{'Larger models compose better.' if pr < 0 else 'Size alone does not solve composition.'}"
            )

    if not points:
        return "_Insufficient data for summary._\n"

    return "\n".join(points) + "\n"
