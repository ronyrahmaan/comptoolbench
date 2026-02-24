"""Load and parse CompToolBench results into pandas DataFrames.

Takes the JSON output from BenchmarkRunner.save_results() and converts it
into analysis-ready DataFrames with one row per (model, task) or per (model, level).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Typed result containers (thin wrappers for clarity)
# ---------------------------------------------------------------------------

class BenchmarkResults:
    """Parsed benchmark results ready for analysis.

    Attributes:
        task_df: One row per (model, task) — the raw per-task scores.
        level_df: One row per (model, level) — aggregated accuracy per level.
        model_df: One row per model — headline metrics and composition gaps.
        error_df: One row per (model, error_type) — error distribution.
        raw: The original JSON dict for anything not in the DataFrames.
    """

    def __init__(
        self,
        task_df: pd.DataFrame,
        level_df: pd.DataFrame,
        model_df: pd.DataFrame,
        error_df: pd.DataFrame,
        raw: dict[str, Any],
    ) -> None:
        self.task_df = task_df
        self.level_df = level_df
        self.model_df = model_df
        self.error_df = error_df
        self.raw = raw

    @property
    def model_names(self) -> list[str]:
        """Return model names sorted by overall accuracy (descending)."""
        return (
            self.model_df
            .sort_values("overall_accuracy", ascending=False)["model"]
            .tolist()
        )

    @property
    def n_models(self) -> int:
        return len(self.model_df)

    @property
    def n_tasks(self) -> int:
        per_model = self.task_df.groupby("model").size()
        return int(per_model.iloc[0]) if len(per_model) > 0 else 0


# ---------------------------------------------------------------------------
# Loading functions
# ---------------------------------------------------------------------------

def load_results(path: str | Path) -> BenchmarkResults:
    """Load a results JSON file into structured DataFrames.

    Args:
        path: Path to results_*.json from BenchmarkRunner.save_results().

    Returns:
        BenchmarkResults with task_df, level_df, model_df, error_df.
    """
    raw = json.loads(Path(path).read_text())
    models_data = raw.get("models", {})

    task_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []

    for model_key, model_data in models_data.items():
        model_info = model_data["model"]
        model_name = model_info["name"]
        provider = model_info["provider"]
        gap = model_data.get("composition_gap") or {}
        summary = model_data.get("summary", {})

        # Per-task rows
        for tr in model_data.get("task_results", []):
            score = tr.get("score", {})
            task_rows.append({
                "model": model_name,
                "model_key": model_key,
                "provider": provider,
                "task_id": tr["task_id"],
                "level": score.get("level", "unknown"),
                "overall": score.get("overall", 0.0),
                "tool_sequence_score": score.get("tool_sequence_score", 0.0),
                "argument_score": score.get("argument_score", 0.0),
                "completeness_score": score.get("completeness_score", 0.0),
                "data_flow_score": score.get("data_flow_score", 0.0),
                "error_type": score.get("error_type"),
                "input_tokens": tr.get("tokens", {}).get("input", 0),
                "output_tokens": tr.get("tokens", {}).get("output", 0),
                "latency_ms": tr.get("latency_ms", 0.0),
                "num_tool_calls": tr.get("num_tool_calls", 0),
            })

        # Headline metrics row
        headline = gap.get("headline_metrics", {})
        per_level = gap.get("per_level_accuracy", {})
        diag = gap.get("diagnostic_metrics", {})
        model_rows.append({
            "model": model_name,
            "model_key": model_key,
            "provider": provider,
            "supports_tools": model_info.get("supports_tools", True),
            "overall_accuracy": headline.get("overall_accuracy", 0.0),
            "gap_overall": headline.get("composition_gap_overall", 0.0),
            "gap_l1": headline.get("composition_gap_L1", 0.0),
            "gap_l2": headline.get("composition_gap_L2", 0.0),
            "gap_l3": headline.get("composition_gap_L3", 0.0),
            "accuracy_l0": per_level.get("L0_node", 0.0),
            "accuracy_l1": per_level.get("L1_chain", 0.0),
            "accuracy_l2": per_level.get("L2_parallel", 0.0),
            "accuracy_l3": per_level.get("L3_dag", 0.0),
            "tool_selection": diag.get("tool_selection_accuracy", 0.0),
            "argument_accuracy": diag.get("argument_accuracy", 0.0),
            "data_flow": diag.get("data_flow_accuracy", 0.0),
            "completion_rate": diag.get("completion_rate", 0.0),
            "hallucinated_tool_rate": diag.get("hallucinated_tool_rate", 0.0),
            "total_tokens": summary.get("total_input_tokens", 0) + summary.get("total_output_tokens", 0),
            "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
        })

        # Error distribution rows
        for error_type, count in gap.get("error_distribution", {}).items():
            error_rows.append({
                "model": model_name,
                "error_type": error_type,
                "count": count,
            })

    task_df = pd.DataFrame(task_rows)
    model_df = pd.DataFrame(model_rows)
    error_df = pd.DataFrame(error_rows)

    # Build level_df: aggregate task_df by (model, level)
    if not task_df.empty:
        level_df = (
            task_df.groupby(["model", "provider", "level"])
            .agg(
                accuracy=("overall", "mean"),
                n_tasks=("overall", "count"),
                tool_seq=("tool_sequence_score", "mean"),
                args=("argument_score", "mean"),
                completeness=("completeness_score", "mean"),
                data_flow=("data_flow_score", "mean"),
                avg_tokens=("input_tokens", "mean"),
                avg_latency=("latency_ms", "mean"),
            )
            .reset_index()
        )
    else:
        level_df = pd.DataFrame()

    return BenchmarkResults(
        task_df=task_df,
        level_df=level_df,
        model_df=model_df,
        error_df=error_df,
        raw=raw,
    )


def load_leaderboard(path: str | Path) -> pd.DataFrame:
    """Load a leaderboard CSV from BenchmarkRunner.save_leaderboard()."""
    return pd.read_csv(path)
