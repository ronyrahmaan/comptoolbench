"""Evaluation runner for CompToolBench.

Orchestrates the full pipeline:
  1. Generate (or load) benchmark tasks
  2. Call each model on each task with tool schemas
  3. Score model responses against ground truth
  4. Compute composition gap and diagnostic metrics
  5. Save results as JSON for analysis

Supports resumption: if a run is interrupted, results are saved per-task
so the run can continue without re-evaluating completed tasks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from comptoolbench.evaluation.metrics import CompositionGapResult, compute_composition_gap
from comptoolbench.evaluation.model_adapter import (
    AVAILABLE_MODELS,
    CallResult,
    ModelAdapter,
    ModelConfig,
    tools_by_name,
    tools_to_openai_schema,
)
from comptoolbench.evaluation.scorers import ModelCall, TaskScore, score_task
from comptoolbench.generators.composition_engine import CompositionEngine
from comptoolbench.generators.task_generator import TaskGenerator
from comptoolbench.tasks.models import Task, TaskSuite
from comptoolbench.tools.base import ToolMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful assistant with access to a set of tools. To solve the user's \
request, analyze what needs to be done and call the appropriate tools with the \
correct arguments.

IMPORTANT RULES:
1. Call tools by their exact name as listed.
2. Use the correct argument names and types.
3. If a task requires multiple steps, call tools in the correct order.
4. If steps can be done in parallel (independent of each other), call them together.
5. If one step depends on another's output, use the output from the previous call \
   as input to the next.
6. Only call tools that are listed as available — do not invent tools.
"""


# ---------------------------------------------------------------------------
# Per-task result storage
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result of evaluating a single model on a single task."""

    task_id: str
    model_name: str
    task_score: TaskScore
    call_result: CallResult
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "task_id": self.task_id,
            "model": self.model_name,
            "score": self.task_score.to_dict(),
            "tokens": {
                "input": self.call_result.input_tokens,
                "output": self.call_result.output_tokens,
            },
            "latency_ms": round(self.call_result.latency_ms, 1),
            "error": self.call_result.error,
            "num_tool_calls": len(self.call_result.model_calls),
            "timestamp": self.timestamp,
        }


@dataclass
class ModelRunResult:
    """Full result of evaluating a model on the benchmark."""

    model_config: ModelConfig
    task_results: list[TaskResult] = field(default_factory=list)
    composition_gap: CompositionGapResult | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    start_time: str = ""
    end_time: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "model": {
                "name": self.model_config.name,
                "litellm_id": self.model_config.litellm_id,
                "provider": self.model_config.provider.value,
                "supports_tools": self.model_config.supports_tools,
            },
            "composition_gap": self.composition_gap.to_dict() if self.composition_gap else None,
            "summary": {
                "total_tasks": len(self.task_results),
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_latency_ms": round(self.total_latency_ms, 1),
                "avg_latency_ms": round(
                    self.total_latency_ms / max(len(self.task_results), 1), 1
                ),
                "errors": sum(1 for tr in self.task_results if tr.call_result.error),
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
            "task_results": [tr.to_dict() for tr in self.task_results],
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _suite_hash(suite: TaskSuite) -> str:
    """Deterministic hash of a task suite for cache identification."""
    content = json.dumps(
        {"name": suite.name, "version": suite.version, "n": len(suite.tasks)},
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:12]


class BenchmarkRunner:
    """Runs CompToolBench evaluation across one or more models.

    Usage:
        runner = BenchmarkRunner(output_dir="results/run_001")
        runner.generate_tasks(seed=42)  # or runner.load_tasks("suite.json")
        runner.evaluate_model("groq-llama3.1-8b")
        runner.evaluate_model("mistral-small")
        runner.save_results()
    """

    def __init__(
        self,
        output_dir: str | Path = "results",
        log_raw_responses: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_raw_responses = log_raw_responses

        self.suite: TaskSuite | None = None
        self.model_results: dict[str, ModelRunResult] = {}

    def generate_tasks(
        self,
        seed: int = 42,
        l0_count: int = 50,
        l1_count: int = 50,
        l2_count: int = 30,
        l3_count: int = 20,
    ) -> TaskSuite:
        """Generate the benchmark task suite."""
        generator = TaskGenerator(seed=seed, mode=ToolMode.SIMULATED)
        self.suite = generator.generate_suite(
            l0_count=l0_count,
            l1_count=l1_count,
            l2_count=l2_count,
            l3_count=l3_count,
        )
        logger.info(
            "Generated %d tasks (%s)",
            len(self.suite.tasks),
            self.suite.stats,
        )

        # Save the suite
        suite_path = self.output_dir / "task_suite.json"
        suite_path.write_text(
            json.dumps(
                {
                    "name": self.suite.name,
                    "version": self.suite.version,
                    "stats": self.suite.stats,
                    "metadata": self.suite.metadata,
                    "tasks": [t.model_dump() for t in self.suite.tasks],
                },
                indent=2,
                default=str,
            )
        )
        logger.info("Saved task suite to %s", suite_path)

        return self.suite

    def generate_tasks_v2(
        self,
        seed: int = 42,
        l0_count: int = 600,
        l1_count: int = 800,
        l2_count: int = 500,
        l3_count: int = 600,
    ) -> TaskSuite:
        """Generate tasks using the new CompositionEngine (v2, 106 tools)."""
        engine = CompositionEngine(seed=seed, mode=ToolMode.SIMULATED)
        self.suite = engine.generate_suite(
            l0_count=l0_count,
            l1_count=l1_count,
            l2_count=l2_count,
            l3_count=l3_count,
        )
        logger.info(
            "Generated %d tasks with CompositionEngine (%s)",
            len(self.suite.tasks),
            self.suite.stats,
        )

        # Save the suite
        suite_path = self.output_dir / "task_suite.json"
        suite_path.write_text(
            json.dumps(
                {
                    "name": self.suite.name,
                    "version": self.suite.version,
                    "stats": self.suite.stats,
                    "metadata": self.suite.metadata,
                    "tasks": [t.model_dump() for t in self.suite.tasks],
                },
                indent=2,
                default=str,
            )
        )
        logger.info("Saved task suite to %s", suite_path)

        return self.suite

    def load_tasks(self, path: str | Path) -> TaskSuite:
        """Load a previously generated task suite."""
        data = json.loads(Path(path).read_text())
        tasks = [Task.model_validate(t) for t in data["tasks"]]
        self.suite = TaskSuite(
            name=data["name"],
            version=data["version"],
            tasks=tasks,
            metadata=data.get("metadata", {}),
        )
        logger.info("Loaded %d tasks from %s", len(tasks), path)
        return self.suite

    def evaluate_model(
        self,
        model_key: str,
        config: ModelConfig | None = None,
        resume: bool = True,
    ) -> ModelRunResult:
        """Evaluate a single model on the entire benchmark suite.

        Args:
            model_key: Key in AVAILABLE_MODELS registry, or a display name.
            config: Optional explicit ModelConfig (overrides registry lookup).
            resume: If True, skip tasks that already have saved results.

        Returns:
            ModelRunResult with all scores and metrics.
        """
        if self.suite is None:
            msg = "No task suite loaded. Call generate_tasks() or load_tasks() first."
            raise RuntimeError(msg)

        if config is None:
            config = AVAILABLE_MODELS.get(model_key)
            if config is None:
                msg = f"Model '{model_key}' not found in registry. Available: {list(AVAILABLE_MODELS.keys())}"
                raise ValueError(msg)

        adapter = ModelAdapter(config=config)
        run_result = ModelRunResult(
            model_config=config,
            start_time=datetime.now(timezone.utc).isoformat(),
        )

        # Check for existing partial results (for resume)
        completed_task_ids: set[str] = set()
        checkpoint_path = self.output_dir / f"checkpoint_{model_key}.jsonl"
        if resume and checkpoint_path.exists():
            for line in checkpoint_path.read_text().strip().split("\n"):
                if line:
                    entry = json.loads(line)
                    completed_task_ids.add(entry["task_id"])
                    # Reconstruct TaskResult for the completed task
                    run_result.task_results.append(self._reconstruct_task_result(entry, config))
            logger.info(
                "Resuming %s: %d/%d tasks already completed",
                config.name, len(completed_task_ids), len(self.suite.tasks),
            )

        # Evaluate each task
        tasks_to_eval = [t for t in self.suite.tasks if t.task_id not in completed_task_ids]
        total = len(self.suite.tasks)
        done = len(completed_task_ids)

        for i, task in enumerate(tasks_to_eval, start=1):
            done += 1
            logger.info(
                "[%s] Task %d/%d (%s): %s",
                config.name, done, total, task.level.value, task.task_id,
            )

            # Get tool schemas for this task's available tools
            task_tools = tools_by_name(task.available_tools)
            tool_schemas = tools_to_openai_schema(task_tools)

            # Call the model (longer timeout for large local models)
            timeout = 180.0 if config.provider.value == "ollama" else 60.0
            call_result = adapter.call(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=task.prompt,
                tool_schemas=tool_schemas,
                timeout=timeout,
            )

            # Score the response
            task_score = score_task(task, call_result.model_calls)

            task_result = TaskResult(
                task_id=task.task_id,
                model_name=config.name,
                task_score=task_score,
                call_result=call_result,
            )
            run_result.task_results.append(task_result)

            # Token/latency tracking
            run_result.total_input_tokens += call_result.input_tokens
            run_result.total_output_tokens += call_result.output_tokens
            run_result.total_latency_ms += call_result.latency_ms

            # Save checkpoint (append)
            self._save_checkpoint(checkpoint_path, task_result)

            # Progress log
            if done % 10 == 0 or done == total:
                logger.info(
                    "[%s] Progress: %d/%d (%.1f%%) — avg latency: %.0fms",
                    config.name, done, total, 100 * done / total,
                    run_result.total_latency_ms / max(done, 1),
                )

        run_result.end_time = datetime.now(timezone.utc).isoformat()

        # Compute composition gap
        tasks_list = self.suite.tasks
        scores_list = self._align_scores(tasks_list, run_result.task_results)
        run_result.composition_gap = compute_composition_gap(
            config.name, tasks_list, scores_list,
        )

        self.model_results[model_key] = run_result
        logger.info(
            "[%s] Complete! Overall accuracy: %.3f, Composition gap: %.3f",
            config.name,
            run_result.composition_gap.overall_accuracy,
            run_result.composition_gap.gap_overall,
        )

        return run_result

    def _align_scores(
        self, tasks: list[Task], task_results: list[TaskResult],
    ) -> list[TaskScore]:
        """Align task results to the task list order."""
        result_map = {tr.task_id: tr.task_score for tr in task_results}
        return [
            result_map.get(
                task.task_id,
                TaskScore(
                    task_id=task.task_id,
                    level=task.level,
                    call_scores=[],
                    overall=0.0,
                    error_type="E10_format_error",
                ),
            )
            for task in tasks
        ]

    def _save_checkpoint(self, path: Path, task_result: TaskResult) -> None:
        """Append a single task result to the checkpoint file."""
        entry = task_result.to_dict()
        with path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def _reconstruct_task_result(
        self, entry: dict[str, Any], config: ModelConfig,
    ) -> TaskResult:
        """Reconstruct a TaskResult from a checkpoint entry."""
        score_data = entry["score"]
        task_score = TaskScore(
            task_id=score_data["task_id"],
            level=next(
                lv for lv in __import__("comptoolbench.tasks.models", fromlist=["CompositionLevel"]).CompositionLevel
                if lv.value == score_data["level"]
            ),
            call_scores=[],
            overall=score_data["overall"],
            tool_sequence_score=score_data.get("tool_sequence_score", 0.0),
            argument_score=score_data.get("argument_score", 0.0),
            completeness_score=score_data.get("completeness_score", 0.0),
            data_flow_score=score_data.get("data_flow_score", 0.0),
            error_type=score_data.get("error_type"),
        )
        call_result = CallResult(
            model_calls=[],
            raw_response={},
            input_tokens=entry.get("tokens", {}).get("input", 0),
            output_tokens=entry.get("tokens", {}).get("output", 0),
            latency_ms=entry.get("latency_ms", 0.0),
            error=entry.get("error"),
        )
        return TaskResult(
            task_id=entry["task_id"],
            model_name=config.name,
            task_score=task_score,
            call_result=call_result,
            timestamp=entry.get("timestamp", ""),
        )

    def save_results(self) -> Path:
        """Save all model results to a JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        results_path = self.output_dir / f"results_{timestamp}.json"

        output = {
            "benchmark": {
                "name": self.suite.name if self.suite else "unknown",
                "version": self.suite.version if self.suite else "0.0.0",
                "stats": self.suite.stats if self.suite else {},
            },
            "models": {
                key: result.to_dict()
                for key, result in self.model_results.items()
            },
            "generated_at": timestamp,
        }

        results_path.write_text(json.dumps(output, indent=2, default=str))
        logger.info("Results saved to %s", results_path)
        return results_path

    def save_leaderboard(self) -> Path:
        """Save a concise leaderboard CSV."""
        leaderboard_path = self.output_dir / "leaderboard.csv"

        rows = []
        for key, result in self.model_results.items():
            gap = result.composition_gap
            if gap is None:
                continue
            rows.append({
                "model": gap.model_name,
                "provider": result.model_config.provider.value,
                "overall_accuracy": round(gap.overall_accuracy, 4),
                "composition_gap": round(gap.gap_overall, 4),
                "L0_accuracy": round(gap.accuracy_l0, 4),
                "L1_accuracy": round(gap.accuracy_l1, 4),
                "L2_accuracy": round(gap.accuracy_l2, 4),
                "L3_accuracy": round(gap.accuracy_l3, 4),
                "gap_L1": round(gap.gap_l1, 4),
                "gap_L2": round(gap.gap_l2, 4),
                "gap_L3": round(gap.gap_l3, 4),
                "tool_selection": round(gap.tool_selection_accuracy, 4),
                "argument_accuracy": round(gap.argument_accuracy, 4),
                "total_tokens": result.total_input_tokens + result.total_output_tokens,
                "avg_latency_ms": round(
                    result.total_latency_ms / max(len(result.task_results), 1), 1,
                ),
            })

        # Sort by composition gap (ascending = better)
        rows.sort(key=lambda r: r["composition_gap"])

        # Write CSV
        if rows:
            header = list(rows[0].keys())
            lines = [",".join(header)]
            for row in rows:
                lines.append(",".join(str(row[h]) for h in header))
            leaderboard_path.write_text("\n".join(lines) + "\n")
            logger.info("Leaderboard saved to %s", leaderboard_path)

        return leaderboard_path
