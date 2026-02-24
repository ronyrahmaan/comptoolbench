#!/usr/bin/env python3
"""L0 rich-prompt control experiment for CompToolBench.

Tests whether the Selection Gap (L0 accuracy < composed L1-L3 accuracy)
is a real phenomenon or an artifact of L0 tasks using minimal prompts
while L1-L3 tasks use context-rich prompts.

Design:
    Three prompt conditions applied to the SAME set of L0 tasks
    (same tools, same arguments, same expected traces):

    1. **Minimal** (current behavior):
       "What is the weather in Paris?"

    2. **Tool-hinted**:
       "Use the get_weather tool to check the weather in Paris"

    3. **Context-rich**:
       "You are a helpful assistant with access to weather tools.
        A user needs to know the current weather conditions.
        Please use the appropriate tool to get the weather for Paris."

    If the Selection Gap disappears when using tool-hinted or context-rich
    prompts, it suggests the gap is a prompt-engineering artifact.
    If it persists, it validates that L0 difficulty is intrinsic.

Usage:
    # Dry run — inspect generated prompts without calling any API:
    uv run python scripts/run_l0_control.py --dry-run

    # Run with default model (gpt-4o-mini, cheapest):
    uv run python scripts/run_l0_control.py

    # Run with a specific model:
    uv run python scripts/run_l0_control.py --model gemini-2.0-flash

    # Custom task count:
    uv run python scripts/run_l0_control.py --model groq-llama3.3-70b --tasks 96

    # Full analysis with specific seed:
    uv run python scripts/run_l0_control.py --model gpt-4o-mini --tasks 48 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comptoolbench.evaluation.model_adapter import (
    AVAILABLE_MODELS,
    ModelAdapter,
    ModelConfig,
    Provider,
    tools_by_name,
    tools_to_openai_schema,
)
from comptoolbench.evaluation.scorers import score_task
from comptoolbench.generators.composition_engine import CompositionEngine
from comptoolbench.tasks.models import Task
from comptoolbench.tools.base import ToolMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("l0_control")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Provider env var mapping (mirrors run_benchmark.py)
PROVIDER_ENV_VARS: dict[Provider, str] = {
    Provider.GEMINI: "GEMINI_API_KEY",
    Provider.GROQ: "GROQ_API_KEY",
    Provider.OPENROUTER: "OPENROUTER_API_KEY",
    Provider.MISTRAL: "MISTRAL_API_KEY",
    Provider.OPENAI: "OPENAI_API_KEY",
    Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
    Provider.XAI: "XAI_API_KEY",
    Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
    Provider.CEREBRAS: "CEREBRAS_API_KEY",
    Provider.SAMBANOVA: "SAMBANOVA_API_KEY",
    Provider.COHERE: "COHERE_API_KEY",
}

# The three prompt conditions
CONDITION_MINIMAL = "minimal"
CONDITION_TOOL_HINTED = "tool_hinted"
CONDITION_CONTEXT_RICH = "context_rich"
CONDITIONS = [CONDITION_MINIMAL, CONDITION_TOOL_HINTED, CONDITION_CONTEXT_RICH]

# System prompt (identical to runner.py — must match for fair comparison)
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
# Prompt rewriting logic
# ---------------------------------------------------------------------------

def _get_tool_description_fragment(tool_name: str) -> str:
    """Build a human-readable description of the tool's purpose.

    Used in context-rich prompts to describe the tool domain without
    explicitly naming the tool.
    """
    domain_map: dict[str, str] = {
        "get_weather": "weather lookup tools",
        "calculator": "mathematical computation tools",
        "unit_convert": "unit conversion tools",
        "get_stock_price": "stock market lookup tools",
        "get_exchange_rate": "currency exchange tools",
        "sentiment_analysis": "text sentiment analysis tools",
        "web_search": "web search tools",
        "translate_text": "translation tools",
        "summarize_text": "text summarization tools",
        "extract_entities": "named entity extraction tools",
        "hash_text": "text hashing tools",
        "word_count": "text analysis tools",
        "detect_language": "language detection tools",
        "percentage_change": "mathematical computation tools",
        "format_date": "date formatting tools",
        "send_email": "email communication tools",
        "read_file": "file reading tools",
        "write_file": "file writing tools",
        "list_directory": "file system tools",
        "set_reminder": "scheduling and reminder tools",
        "create_event": "calendar and scheduling tools",
        "search_contacts": "contact management tools",
        "knowledge_base_query": "knowledge retrieval tools",
        "classify_text": "text classification tools",
        "generate_password": "security utility tools",
        "get_time": "time and timezone tools",
        "convert_timezone": "timezone conversion tools",
        "text_to_speech": "audio generation tools",
        "image_search": "image search tools",
    }
    return domain_map.get(tool_name, "various utility tools")


def _build_tool_hinted_prompt(task: Task) -> str:
    """Rewrite a task's prompt to explicitly name the tool.

    Example: "What is the weather in Paris?"
          -> "Use the get_weather tool to check the weather in Paris."
    """
    step = task.expected_trace.steps[0]
    tool_name = step.tool_name
    args = step.arguments

    # Build a readable argument string
    arg_parts = []
    for key, value in args.items():
        human_key = key.replace("_", " ")
        if isinstance(value, str) and len(value) < 100:
            arg_parts.append(f"{human_key} = \"{value}\"")
        elif isinstance(value, (int, float)):
            arg_parts.append(f"{human_key} = {value}")
        elif isinstance(value, bool):
            arg_parts.append(f"{human_key} = {str(value).lower()}")

    if arg_parts:
        detail = ", ".join(arg_parts[:4])
        return f"Use the {tool_name} tool with the following parameters: {detail}."
    return f"Use the {tool_name} tool to complete this request."


def _build_context_rich_prompt(task: Task) -> str:
    """Rewrite a task's prompt with rich context but NO tool name.

    Adds role framing, user scenario, and clear instructions while keeping
    the tool selection challenge intact.

    Example: "What is the weather in Paris?"
          -> "You are a helpful assistant with access to weather tools.
              A user needs to know the current weather conditions.
              Please use the appropriate tool to get the weather for Paris."
    """
    step = task.expected_trace.steps[0]
    tool_name = step.tool_name
    args = step.arguments

    domain = _get_tool_description_fragment(tool_name)

    # Build a detailed parameter description
    param_sentences = []
    for key, value in args.items():
        human_key = key.replace("_", " ")
        if isinstance(value, str) and len(value) < 100:
            param_sentences.append(f"The {human_key} is \"{value}\".")
        elif isinstance(value, (int, float)):
            param_sentences.append(f"The {human_key} is {value}.")
        elif isinstance(value, bool):
            param_sentences.append(f"The {human_key} should be {'enabled' if value else 'disabled'}.")
        elif isinstance(value, list):
            items = ", ".join(str(v) for v in value[:5])
            param_sentences.append(f"The {human_key} values are: {items}.")

    params_text = " ".join(param_sentences[:4]) if param_sentences else ""

    # Build the full context-rich prompt
    prompt = (
        f"You are a helpful assistant with access to {domain}. "
        f"A user needs help with a task. "
        f"Please select and use the most appropriate tool from your available tools. "
        f"{params_text}"
    )
    return prompt.strip()


def rewrite_prompt(task: Task, condition: str) -> str:
    """Rewrite a task's prompt according to the specified condition.

    Args:
        task: The original L0 task with its minimal prompt.
        condition: One of 'minimal', 'tool_hinted', or 'context_rich'.

    Returns:
        The rewritten prompt string.

    Raises:
        ValueError: If condition is not recognized.
    """
    if condition == CONDITION_MINIMAL:
        return task.prompt  # No change
    elif condition == CONDITION_TOOL_HINTED:
        return _build_tool_hinted_prompt(task)
    elif condition == CONDITION_CONTEXT_RICH:
        return _build_context_rich_prompt(task)
    else:
        raise ValueError(f"Unknown prompt condition: {condition!r}")


# ---------------------------------------------------------------------------
# Per-task result
# ---------------------------------------------------------------------------

@dataclass
class ControlTaskResult:
    """Result of evaluating one task under one prompt condition."""

    task_id: str
    condition: str
    tool_name: str
    prompt: str
    score: float  # Binary: 0.0 or 1.0 for L0
    tool_correct: bool
    args_score: float
    error_type: str | None
    latency_ms: float
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "task_id": self.task_id,
            "condition": self.condition,
            "tool_name": self.tool_name,
            "prompt": self.prompt,
            "score": self.score,
            "tool_correct": self.tool_correct,
            "args_score": round(self.args_score, 4),
            "error_type": self.error_type,
            "latency_ms": round(self.latency_ms, 1),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


# ---------------------------------------------------------------------------
# Bootstrap CI and paired test (standalone, no BenchmarkResults dependency)
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the mean of a 1-D score array.

    Args:
        scores: 1-D array of binary scores (0.0 or 1.0).
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (default 0.05 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (point_estimate, ci_lower, ci_upper).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return (0.0, 0.0, 0.0)

    point = float(np.mean(scores))
    if scores.size == 1 or np.std(scores) == 0.0:
        return (point, point, point)

    rng = np.random.default_rng(seed)
    n = scores.size
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = scores[boot_indices].mean(axis=1)

    ci_lo = float(np.percentile(boot_means, (alpha / 2) * 100))
    ci_hi = float(np.percentile(boot_means, (1 - alpha / 2) * 100))
    return (point, ci_lo, ci_hi)


def _paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap significance test (Berg-Kirkpatrick et al., 2012).

    Tests H0: mean(scores_a) == mean(scores_b) where the arrays are
    paired by task_id (same task, different prompt condition).

    Args:
        scores_a: Per-task scores for condition A.
        scores_b: Per-task scores for condition B (same length, paired).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        (observed_diff, p_value) where diff = mean(A) - mean(B).
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if scores_a.size != scores_b.size:
        raise ValueError(
            f"Paired test requires equal-length arrays, "
            f"got {scores_a.size} vs {scores_b.size}."
        )
    if scores_a.size == 0:
        return (0.0, 1.0)

    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))

    diffs = scores_a - scores_b
    centred = diffs - np.mean(diffs)

    rng = np.random.default_rng(seed)
    n = centred.size
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = centred[boot_indices].mean(axis=1)

    p_value = float(np.mean(np.abs(boot_means) >= abs(observed_diff)))
    return (observed_diff, p_value)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

@dataclass
class L0ControlExperiment:
    """Runs the L0 rich-prompt control experiment.

    Generates L0 tasks once, then evaluates them under all three
    prompt conditions against a single model.
    """

    model_config: ModelConfig
    seed: int = 42
    n_tasks: int = 48
    output_dir: Path = field(default_factory=lambda: Path("results/l0_control"))
    dry_run: bool = False

    # Internal state
    _base_tasks: list[Task] = field(default_factory=list, repr=False)
    _results: dict[str, list[ControlTaskResult]] = field(default_factory=dict, repr=False)

    def generate_base_tasks(self) -> list[Task]:
        """Generate the base set of L0 tasks using CompositionEngine.

        All three conditions will use the same tasks (same tools, same
        arguments, same expected traces) — only the prompt changes.

        Returns:
            List of L0 Task objects.
        """
        engine = CompositionEngine(seed=self.seed, mode=ToolMode.SIMULATED)
        tasks = engine.generate_l0_tasks(count=self.n_tasks)

        if len(tasks) < self.n_tasks:
            logger.warning(
                "Engine generated %d tasks (requested %d). "
                "Some tools may not have enough parameter diversity.",
                len(tasks), self.n_tasks,
            )

        self._base_tasks = tasks
        logger.info("Generated %d base L0 tasks across %d unique tools.",
                     len(tasks), len({t.expected_trace.steps[0].tool_name for t in tasks}))
        return tasks

    def run(self) -> dict[str, list[ControlTaskResult]]:
        """Run all three prompt conditions and collect results.

        Returns:
            Dict mapping condition name to list of ControlTaskResult.
        """
        if not self._base_tasks:
            self.generate_base_tasks()

        if self.dry_run:
            self._print_dry_run()
            return {}

        adapter = ModelAdapter(config=self.model_config)

        for condition in CONDITIONS:
            logger.info("")
            logger.info("=" * 60)
            logger.info("  Condition: %s", condition.upper())
            logger.info("=" * 60)

            condition_results: list[ControlTaskResult] = []

            for i, task in enumerate(self._base_tasks, start=1):
                prompt = rewrite_prompt(task, condition)
                step = task.expected_trace.steps[0]

                logger.info(
                    "[%s] Task %d/%d (%s): %s",
                    condition, i, len(self._base_tasks),
                    step.tool_name, task.task_id,
                )

                # Get tool schemas for this task
                task_tools = tools_by_name(task.available_tools)
                tool_schemas = tools_to_openai_schema(task_tools)

                # Call the model
                call_result = adapter.call(
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt,
                    tool_schemas=tool_schemas,
                    timeout=60.0,
                )

                # Score identically to the main benchmark
                task_score = score_task(task, call_result.model_calls)

                # Extract per-call details
                tool_correct = False
                args_score = 0.0
                if task_score.call_scores:
                    cs = task_score.call_scores[0]
                    tool_correct = cs.tool_correct
                    args_score = cs.args_score

                result = ControlTaskResult(
                    task_id=task.task_id,
                    condition=condition,
                    tool_name=step.tool_name,
                    prompt=prompt,
                    score=task_score.overall,
                    tool_correct=tool_correct,
                    args_score=args_score,
                    error_type=task_score.error_type,
                    latency_ms=call_result.latency_ms,
                    input_tokens=call_result.input_tokens,
                    output_tokens=call_result.output_tokens,
                )
                condition_results.append(result)

                # Progress log every 10 tasks
                if i % 10 == 0 or i == len(self._base_tasks):
                    current_acc = np.mean([r.score for r in condition_results])
                    logger.info(
                        "[%s] Progress: %d/%d — running accuracy: %.1f%%",
                        condition, i, len(self._base_tasks), current_acc * 100,
                    )

            self._results[condition] = condition_results

        return self._results

    def _print_dry_run(self) -> None:
        """Print sample prompts for each condition without calling any API."""
        n_show = min(5, len(self._base_tasks))

        print("\n" + "=" * 70)
        print("  L0 CONTROL EXPERIMENT — DRY RUN")
        print(f"  Model: {self.model_config.name}")
        print(f"  Tasks: {len(self._base_tasks)}")
        print(f"  Seed:  {self.seed}")
        print("=" * 70)

        for task in self._base_tasks[:n_show]:
            step = task.expected_trace.steps[0]
            print(f"\n{'—' * 60}")
            print(f"Task: {task.task_id}  |  Tool: {step.tool_name}")
            print(f"Args: {step.arguments}")
            print(f"Available tools: {task.available_tools}")
            print()

            for condition in CONDITIONS:
                prompt = rewrite_prompt(task, condition)
                label = condition.upper().ljust(15)
                # Truncate long prompts for display
                display = prompt if len(prompt) < 120 else prompt[:117] + "..."
                print(f"  {label}: {display}")

        remaining = len(self._base_tasks) - n_show
        if remaining > 0:
            print(f"\n  ... and {remaining} more tasks (omitted)")

        print("\n" + "=" * 70)
        print("  To run for real, remove --dry-run")
        print("=" * 70 + "\n")

    def compute_statistics(self) -> dict[str, Any]:
        """Compute comparison statistics across the three conditions.

        Returns:
            Dict with accuracy, CIs, and significance tests.
        """
        if not self._results:
            raise RuntimeError("No results to analyze. Run the experiment first.")

        stats: dict[str, Any] = {
            "model": self.model_config.name,
            "n_tasks": len(self._base_tasks),
            "seed": self.seed,
            "conditions": {},
            "significance_tests": [],
        }

        # Per-condition accuracy and CIs
        for condition in CONDITIONS:
            results = self._results[condition]
            scores = np.array([r.score for r in results])

            acc, ci_lo, ci_hi = _bootstrap_ci(scores, seed=self.seed)
            tool_acc = np.mean([1.0 if r.tool_correct else 0.0 for r in results])
            avg_args = np.mean([r.args_score for r in results])

            # Error distribution
            errors: dict[str, int] = {}
            for r in results:
                if r.error_type:
                    errors[r.error_type] = errors.get(r.error_type, 0) + 1

            stats["conditions"][condition] = {
                "accuracy": round(float(acc), 4),
                "ci_lower": round(float(ci_lo), 4),
                "ci_upper": round(float(ci_hi), 4),
                "tool_selection_accuracy": round(float(tool_acc), 4),
                "argument_accuracy": round(float(avg_args), 4),
                "n_correct": int(np.sum(scores >= 1.0)),
                "n_total": len(results),
                "error_distribution": errors,
            }

        # Paired significance tests
        # Test 1: minimal vs tool_hinted
        # Test 2: minimal vs context_rich
        for alt_condition in [CONDITION_TOOL_HINTED, CONDITION_CONTEXT_RICH]:
            scores_min = np.array([r.score for r in self._results[CONDITION_MINIMAL]])
            scores_alt = np.array([r.score for r in self._results[alt_condition]])

            diff, p_val = _paired_bootstrap_test(scores_min, scores_alt, seed=self.seed)

            stats["significance_tests"].append({
                "comparison": f"minimal vs {alt_condition}",
                "diff": round(float(diff), 4),
                "p_value": round(float(p_val), 4),
                "significant_at_005": p_val < 0.05,
                "significant_at_001": p_val < 0.01,
                "interpretation": _interpret_result(diff, p_val),
            })

        return stats

    def print_results_table(self, stats: dict[str, Any]) -> None:
        """Print a formatted comparison table to stdout."""
        print("\n" + "=" * 70)
        print("  L0 CONTROL EXPERIMENT RESULTS")
        print(f"  Model: {stats['model']}")
        print(f"  Tasks: {stats['n_tasks']}  |  Seed: {stats['seed']}")
        print("=" * 70)

        # Accuracy table
        print(f"\n{'Condition':<18} {'Accuracy':>10} {'95% CI':>20} {'Tool Sel':>10} {'Arg Acc':>10}")
        print("-" * 70)

        for condition in CONDITIONS:
            c = stats["conditions"][condition]
            acc_str = f"{c['accuracy']:.1%}"
            ci_str = f"[{c['ci_lower']:.1%}, {c['ci_upper']:.1%}]"
            tool_str = f"{c['tool_selection_accuracy']:.1%}"
            arg_str = f"{c['argument_accuracy']:.1%}"
            print(f"{condition:<18} {acc_str:>10} {ci_str:>20} {tool_str:>10} {arg_str:>10}")

        # Significance tests
        print(f"\n{'Comparison':<30} {'Diff':>8} {'p-value':>10} {'Sig (p<.05)':>12}")
        print("-" * 62)

        for test in stats["significance_tests"]:
            diff_str = f"{test['diff']:+.3f}"
            p_str = f"{test['p_value']:.4f}"
            sig_str = "YES" if test["significant_at_005"] else "no"
            print(f"{test['comparison']:<30} {diff_str:>8} {p_str:>10} {sig_str:>12}")

        # Interpretation
        print("\n" + "-" * 70)
        print("INTERPRETATION:")
        for test in stats["significance_tests"]:
            print(f"  {test['comparison']}: {test['interpretation']}")

        print("=" * 70 + "\n")

    def save_results(self, stats: dict[str, Any]) -> Path:
        """Save all results and statistics to the output directory.

        Returns:
            Path to the saved results JSON.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Full results with per-task detail
        full_output = {
            "experiment": "l0_control",
            "timestamp": timestamp,
            "statistics": stats,
            "per_task_results": {
                condition: [r.to_dict() for r in results]
                for condition, results in self._results.items()
            },
            "base_tasks": [
                {
                    "task_id": t.task_id,
                    "tool_name": t.expected_trace.steps[0].tool_name,
                    "arguments": t.expected_trace.steps[0].arguments,
                    "available_tools": t.available_tools,
                    "original_prompt": t.prompt,
                }
                for t in self._base_tasks
            ],
        }

        results_path = self.output_dir / f"l0_control_{timestamp}.json"
        results_path.write_text(json.dumps(full_output, indent=2, default=str))
        logger.info("Full results saved to %s", results_path)

        # Summary CSV for quick inspection
        csv_path = self.output_dir / f"l0_control_summary_{timestamp}.csv"
        header = "condition,accuracy,ci_lower,ci_upper,tool_selection,arg_accuracy,n_correct,n_total"
        rows = [header]
        for condition in CONDITIONS:
            c = stats["conditions"][condition]
            rows.append(
                f"{condition},{c['accuracy']},{c['ci_lower']},{c['ci_upper']},"
                f"{c['tool_selection_accuracy']},{c['argument_accuracy']},"
                f"{c['n_correct']},{c['n_total']}"
            )
        csv_path.write_text("\n".join(rows) + "\n")
        logger.info("Summary CSV saved to %s", csv_path)

        return results_path


def _interpret_result(diff: float, p_value: float) -> str:
    """Generate a plain-English interpretation of the significance test.

    Args:
        diff: Observed difference (minimal - alternative). Negative means
              alternative scored higher.
        p_value: Two-sided p-value from paired bootstrap test.

    Returns:
        Human-readable interpretation string.
    """
    if p_value >= 0.05:
        return (
            f"No significant difference (p={p_value:.3f}). "
            "The Selection Gap is NOT explained by prompt richness."
        )

    direction = "HIGHER" if diff < 0 else "LOWER"
    return (
        f"Significant difference (p={p_value:.3f}). "
        f"The richer prompt achieved {direction} accuracy "
        f"(diff={abs(diff):.3f}). "
        f"{'This suggests the gap IS partly a prompt artifact.' if diff < 0 else 'Richer prompts did not help; the gap is intrinsic.'}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _check_model_available(model_key: str) -> ModelConfig | None:
    """Verify that a model's API key is available.

    Args:
        model_key: Key in the AVAILABLE_MODELS registry.

    Returns:
        ModelConfig if available, None otherwise.
    """
    config = AVAILABLE_MODELS.get(model_key)
    if config is None:
        logger.error(
            "Unknown model key '%s'. Available: %s",
            model_key, ", ".join(sorted(AVAILABLE_MODELS.keys())),
        )
        return None

    if config.provider == Provider.OLLAMA:
        return config

    env_var = PROVIDER_ENV_VARS.get(config.provider)
    if env_var and not os.environ.get(env_var):
        logger.error(
            "Model '%s' requires %s environment variable. Set it and retry.",
            model_key, env_var,
        )
        return None

    return config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="L0 rich-prompt control experiment for CompToolBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --dry-run                    # Preview prompts\n"
            "  %(prog)s --model gpt-4o-mini           # Run cheapest cloud model\n"
            "  %(prog)s --model groq-llama3.3-70b     # Run free Groq model\n"
            "  %(prog)s --model gemini-2.0-flash --tasks 96\n"
        ),
    )

    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model key from AVAILABLE_MODELS registry (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--tasks", type=int, default=48,
        help="Number of L0 tasks per condition (default: 48)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for task generation (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/l0_control/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show generated prompts without calling any API",
    )

    return parser.parse_args()


def main() -> None:
    """Run the L0 control experiment."""
    args = parse_args()

    # Resolve model (dry-run skips API key check)
    if args.dry_run:
        config = AVAILABLE_MODELS.get(args.model)
        if config is None:
            logger.error(
                "Unknown model key '%s'. Available: %s",
                args.model, ", ".join(sorted(AVAILABLE_MODELS.keys())),
            )
            sys.exit(1)
    else:
        config = _check_model_available(args.model)
        if config is None:
            sys.exit(1)

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results/l0_control")

    # Create and run experiment
    experiment = L0ControlExperiment(
        model_config=config,
        seed=args.seed,
        n_tasks=args.tasks,
        output_dir=output_dir,
        dry_run=args.dry_run,
    )

    logger.info("=" * 70)
    logger.info("  CompToolBench — L0 Rich-Prompt Control Experiment")
    logger.info("=" * 70)
    logger.info("  Model:   %s (%s)", config.name, config.provider.value)
    logger.info("  Tasks:   %d per condition (%d total API calls)",
                args.tasks, args.tasks * len(CONDITIONS))
    logger.info("  Seed:    %d", args.seed)
    logger.info("  Output:  %s", output_dir)
    logger.info("  Dry run: %s", args.dry_run)
    logger.info("=" * 70)

    # Generate base tasks
    experiment.generate_base_tasks()

    # Run (or dry-run)
    experiment.run()

    if args.dry_run:
        return

    # Compute and display statistics
    stats = experiment.compute_statistics()
    experiment.print_results_table(stats)

    # Save results
    results_path = experiment.save_results(stats)

    logger.info("")
    logger.info("=" * 70)
    logger.info("  DONE! Results saved to: %s", results_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
