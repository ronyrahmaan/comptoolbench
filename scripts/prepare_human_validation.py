#!/usr/bin/env python3
"""Prepare a stratified subset of CompToolBench tasks for human validation.

Selects 50 tasks (12 L0, 16 L1, 10 L2, 12 L3) with diverse tool categories,
formats them for annotator review, and outputs both JSON and CSV files.

Usage:
    uv run python scripts/prepare_human_validation.py
    uv run python scripts/prepare_human_validation.py --suite results/unified_v3/task_suite.json
    uv run python scripts/prepare_human_validation.py --seed 123 --output-dir results/human_validation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from comptoolbench.tasks.models import CompositionLevel, TaskSuite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("human_validation")

# Stratified sample targets per level
SAMPLE_TARGETS: dict[CompositionLevel, int] = {
    CompositionLevel.NODE: 12,
    CompositionLevel.CHAIN: 16,
    CompositionLevel.PARALLEL: 10,
    CompositionLevel.DAG: 12,
}

TOTAL_SAMPLE = sum(SAMPLE_TARGETS.values())  # 50


def find_latest_task_suite() -> Path:
    """Find the most recent task_suite.json in results/."""
    results_dir = Path(__file__).parent.parent / "results"
    candidates: list[Path] = []

    for suite_file in results_dir.rglob("task_suite.json"):
        candidates.append(suite_file)

    if not candidates:
        logger.error("No task_suite.json found under %s", results_dir)
        sys.exit(1)

    # Sort by modification time, newest first
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_suite(path: Path) -> TaskSuite:
    """Load a TaskSuite from JSON file."""
    logger.info("Loading task suite from %s", path)
    with open(path) as f:
        data = json.load(f)
    suite = TaskSuite.model_validate(data)
    logger.info(
        "Loaded %d tasks (%d unique tools)",
        len(suite.tasks),
        suite.stats.get("unique_tools", 0),
    )
    return suite


def stratified_sample(
    suite: TaskSuite,
    targets: dict[CompositionLevel, int],
    seed: int = 42,
) -> list[dict]:
    """Select a stratified sample ensuring diverse tool categories.

    For each level, groups tasks by category and samples proportionally
    to ensure category diversity in the final selection.

    Args:
        suite: The full task suite.
        targets: Number of tasks to sample per level.
        seed: Random seed for reproducibility.

    Returns:
        List of human-readable task dicts, sorted by task_id.
    """
    rng = random.Random(seed)
    selected: list[dict] = []

    by_level = suite.by_level

    for level, count in targets.items():
        tasks_at_level = by_level.get(level, [])
        if not tasks_at_level:
            logger.warning("No tasks found for level %s", level.value)
            continue

        # Group by category for diversity
        by_category: dict[str, list] = defaultdict(list)
        for task in tasks_at_level:
            cat = task.metadata.get("category", "unknown")
            by_category[cat].append(task)

        # Round-robin sample across categories
        pool: list = []
        categories = sorted(by_category.keys())
        per_cat = max(1, count // len(categories))
        remainder = count - per_cat * len(categories)

        for i, cat in enumerate(categories):
            cat_tasks = by_category[cat]
            n = per_cat + (1 if i < remainder else 0)
            n = min(n, len(cat_tasks))
            pool.extend(rng.sample(cat_tasks, n))

        # If we still need more, sample from remaining tasks
        if len(pool) < count:
            already_selected_ids = {t.task_id for t in pool}
            remaining = [t for t in tasks_at_level if t.task_id not in already_selected_ids]
            extra = min(count - len(pool), len(remaining))
            pool.extend(rng.sample(remaining, extra))

        # Trim to exact count if we oversampled
        if len(pool) > count:
            pool = rng.sample(pool, count)

        # Convert to human-readable format
        for task in pool:
            entry = _format_task_for_review(task)
            selected.append(entry)

    selected.sort(key=lambda x: x["task_id"])
    return selected


def _format_task_for_review(task) -> dict:
    """Convert a Task model into a human-readable dict for annotators.

    Args:
        task: A Task pydantic model instance.

    Returns:
        Dictionary with annotator-friendly fields.
    """
    steps_formatted = []
    for step in task.expected_trace.steps:
        steps_formatted.append({
            "step_id": step.step_id,
            "tool": step.tool_name,
            "arguments": step.arguments,
            "depends_on": step.depends_on,
        })

    return {
        "task_id": task.task_id,
        "level": task.level.value,
        "category": task.metadata.get("category", "unknown"),
        "prompt": task.prompt,
        "available_tools": task.available_tools,
        "expected_tool_sequence": task.expected_trace.tool_sequence,
        "expected_steps": steps_formatted,
        "expected_final_answer": task.expected_final_answer,
        "final_answer_source": task.expected_trace.final_answer_source,
    }


def save_json(tasks: list[dict], output_path: Path) -> None:
    """Save the formatted tasks as a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "CompToolBench human validation tasks",
        "instructions": (
            "For each task, verify: (1) the prompt is clear and unambiguous, "
            "(2) the expected tool sequence is correct, (3) the expected arguments "
            "are reasonable, (4) the expected final answer is correct. "
            "Mark your verdict in the companion CSV file."
        ),
        "total_tasks": len(tasks),
        "level_distribution": _count_levels(tasks),
        "tasks": tasks,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved JSON: %s (%d tasks)", output_path, len(tasks))


def save_csv(tasks: list[dict], output_path: Path) -> None:
    """Save a CSV for annotators to fill in correctness judgments.

    Columns: task_id, level, category, prompt, expected_tools, is_correct, notes
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "level",
        "category",
        "prompt",
        "expected_tools",
        "is_correct",
        "notes",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for task in tasks:
            writer.writerow({
                "task_id": task["task_id"],
                "level": task["level"],
                "category": task["category"],
                "prompt": task["prompt"],
                "expected_tools": " -> ".join(task["expected_tool_sequence"]),
                "is_correct": "",
                "notes": "",
            })
    logger.info("Saved CSV: %s (%d rows)", output_path, len(tasks))


def _count_levels(tasks: list[dict]) -> dict[str, int]:
    """Count tasks per level."""
    counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        counts[t["level"]] += 1
    return dict(sorted(counts.items()))


def print_summary(tasks: list[dict]) -> None:
    """Print summary statistics to stdout."""
    level_counts = _count_levels(tasks)
    categories = defaultdict(int)
    tools_seen: set[str] = set()

    for t in tasks:
        categories[t["category"]] += 1
        tools_seen.update(t["expected_tool_sequence"])

    print("\n" + "=" * 60)
    print("  CompToolBench Human Validation — Sample Summary")
    print("=" * 60)
    print(f"  Total tasks selected: {len(tasks)}")
    print()
    print("  By level:")
    for level, count in level_counts.items():
        print(f"    {level}: {count}")
    print()
    print("  By category:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    print()
    print(f"  Unique tools in sample: {len(tools_seen)}")
    print("=" * 60)
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare stratified task sample for human validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        help="Path to task_suite.json (default: auto-detect latest)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/human_validation)",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare human validation dataset."""
    args = parse_args()

    # Resolve task suite path
    if args.suite:
        suite_path = Path(args.suite)
    else:
        suite_path = find_latest_task_suite()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "results" / "human_validation"

    # Load and sample
    suite = load_suite(suite_path)
    tasks = stratified_sample(suite, SAMPLE_TARGETS, seed=args.seed)

    if len(tasks) < TOTAL_SAMPLE:
        logger.warning(
            "Only selected %d/%d tasks (some levels may have too few tasks)",
            len(tasks),
            TOTAL_SAMPLE,
        )

    # Save outputs
    json_path = output_dir / "tasks_for_review.json"
    csv_path = output_dir / "tasks_for_review.csv"
    save_json(tasks, json_path)
    save_csv(tasks, csv_path)

    # Record provenance
    provenance = {
        "source_suite": str(suite_path),
        "seed": args.seed,
        "sample_targets": {k.value: v for k, v in SAMPLE_TARGETS.items()},
        "actual_counts": _count_levels(tasks),
        "total_selected": len(tasks),
        "total_available": len(suite.tasks),
    }
    provenance_path = output_dir / "provenance.json"
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info("Saved provenance: %s", provenance_path)

    print_summary(tasks)


if __name__ == "__main__":
    main()
