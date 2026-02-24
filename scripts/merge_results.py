#!/usr/bin/env python3
"""Merge checkpoint files from multiple model runs into a single results JSON.

Usage:
    uv run python scripts/merge_results.py results/local_run_001/
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comptoolbench.evaluation.metrics import compute_composition_gap
from comptoolbench.evaluation.model_adapter import AVAILABLE_MODELS
from comptoolbench.evaluation.scorers import CallScore, TaskScore
from comptoolbench.tasks.models import CompositionLevel, Task

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Merge all checkpoint files into a combined results JSON."""
    if len(sys.argv) < 2:
        print("Usage: merge_results.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])

    # Load task suite
    suite_path = results_dir / "task_suite.json"
    if not suite_path.exists():
        logger.error("No task_suite.json found in %s", results_dir)
        sys.exit(1)

    with open(suite_path) as f:
        suite_data = json.load(f)

    tasks = [Task.model_validate(t) for t in suite_data["tasks"]]
    logger.info("Loaded %d tasks from suite", len(tasks))

    # Find all checkpoint files
    checkpoints = sorted(results_dir.glob("checkpoint_*.jsonl"))
    if not checkpoints:
        logger.error("No checkpoint files found")
        sys.exit(1)

    models_results: dict[str, dict] = {}

    for cp_path in checkpoints:
        model_key = cp_path.stem.replace("checkpoint_", "")
        config = AVAILABLE_MODELS.get(model_key)
        if config is None:
            logger.warning("Unknown model key: %s — skipping", model_key)
            continue

        # Read checkpoint entries
        entries = []
        with open(cp_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        if len(entries) != len(tasks):
            logger.warning(
                "%s: %d/%d tasks completed — skipping incomplete model",
                model_key, len(entries), len(tasks),
            )
            continue

        logger.info("Processing %s: %d tasks", model_key, len(entries))

        # Build TaskScore objects for composition gap calculation
        entry_map = {e["task_id"]: e for e in entries}
        task_scores: list[TaskScore] = []
        for task in tasks:
            entry = entry_map.get(task.task_id)
            if entry is None:
                task_scores.append(TaskScore(
                    task_id=task.task_id, level=task.level, call_scores=[], overall=0.0,
                ))
                continue

            score_data = entry["score"]
            # Reconstruct call scores for diagnostic metrics
            call_scores = [
                CallScore(
                    step_id=cs.get("step", ""),
                    expected_tool=cs.get("expected_tool", ""),
                    actual_tool=cs.get("actual_tool"),
                    tool_correct=cs.get("tool_correct", False),
                    args_score=cs.get("args_score", 0.0),
                    matched=cs.get("matched", False),
                )
                for cs in score_data.get("call_scores", [])
            ]
            task_scores.append(TaskScore(
                task_id=score_data["task_id"],
                level=CompositionLevel(score_data["level"]),
                call_scores=call_scores,
                overall=score_data["overall"],
                tool_sequence_score=score_data.get("tool_sequence_score", 0.0),
                argument_score=score_data.get("argument_score", 0.0),
                completeness_score=score_data.get("completeness_score", 0.0),
                data_flow_score=score_data.get("data_flow_score", 0.0),
                error_type=score_data.get("error_type"),
            ))

        # Compute composition gap
        gap = compute_composition_gap(config.name, tasks, task_scores)

        # Compute summary stats
        total_input = sum(e.get("tokens", {}).get("input", 0) for e in entries)
        total_output = sum(e.get("tokens", {}).get("output", 0) for e in entries)
        total_latency = sum(e.get("latency_ms", 0) for e in entries)
        errors = sum(1 for e in entries if e.get("error"))

        # Build model result dict
        models_results[model_key] = {
            "model": {
                "name": config.name,
                "litellm_id": config.litellm_id,
                "provider": config.provider.value,
                "supports_tools": config.supports_tools,
            },
            "composition_gap": gap.to_dict(),
            "summary": {
                "total_tasks": len(entries),
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_latency_ms": round(total_latency, 1),
                "avg_latency_ms": round(total_latency / max(len(entries), 1), 1),
                "errors": errors,
            },
            "start_time": entries[0].get("timestamp", ""),
            "end_time": entries[-1].get("timestamp", ""),
            "task_results": [
                {
                    "task_id": e["task_id"],
                    "model": config.name,
                    "score": e["score"],
                    "tokens": e.get("tokens", {"input": 0, "output": 0}),
                    "latency_ms": e.get("latency_ms", 0),
                    "error": e.get("error"),
                    "num_tool_calls": e.get("num_tool_calls", 0),
                }
                for e in entries
            ],
        }

    # Build combined results JSON
    combined = {
        "benchmark": {
            "name": suite_data["name"],
            "version": suite_data["version"],
            "stats": suite_data["stats"],
        },
        "models": models_results,
        "generated_at": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    }

    # Save
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"results_merged_{timestamp}.json"
    output_path.write_text(json.dumps(combined, indent=2, default=str))
    logger.info("Saved merged results to %s (%d models)", output_path, len(models_results))

    # Also save leaderboard CSV
    leaderboard_path = results_dir / "leaderboard.csv"
    header = "model,provider,overall_accuracy,composition_gap,L0_accuracy,L1_accuracy,L2_accuracy,L3_accuracy,gap_L1,gap_L2,gap_L3,tool_selection,argument_accuracy,total_tokens,avg_latency_ms"
    rows = [header]
    for mk, mr in sorted(models_results.items(), key=lambda x: -x[1]["composition_gap"]["headline_metrics"]["overall_accuracy"]):
        gap_data = mr["composition_gap"]["headline_metrics"]
        lvl = mr["composition_gap"]["per_level_accuracy"]
        diag = mr["composition_gap"]["diagnostic_metrics"]
        summary = mr["summary"]
        rows.append(",".join([
            mr["model"]["name"],
            mr["model"]["provider"],
            f"{gap_data['overall_accuracy']:.4f}",
            f"{gap_data['composition_gap_overall']:.4f}",
            f"{lvl.get('L0_node', 0):.4f}",
            f"{lvl.get('L1_chain', 0):.4f}",
            f"{lvl.get('L2_parallel', 0):.4f}",
            f"{lvl.get('L3_dag', 0):.4f}",
            f"{gap_data.get('composition_gap_L1', 0):.4f}",
            f"{gap_data.get('composition_gap_L2', 0):.4f}",
            f"{gap_data.get('composition_gap_L3', 0):.4f}",
            f"{diag.get('tool_selection_accuracy', 0):.4f}",
            f"{diag.get('argument_accuracy', 0):.4f}",
            str(summary["total_input_tokens"] + summary["total_output_tokens"]),
            f"{summary['avg_latency_ms']:.1f}",
        ]))
    leaderboard_path.write_text("\n".join(rows) + "\n")
    logger.info("Saved leaderboard to %s", leaderboard_path)


if __name__ == "__main__":
    main()
