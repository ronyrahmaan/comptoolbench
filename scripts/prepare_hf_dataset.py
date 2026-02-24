#!/usr/bin/env python3
"""Prepare CompToolBench task suite for HuggingFace Datasets upload.

Reads the canonical task_suite.json, flattens nested fields for tabular
compatibility, and writes both JSONL and Parquet files to the
huggingface/ directory.  The output is ready for ``huggingface-cli upload``.

Usage:
    python scripts/prepare_hf_dataset.py
    python scripts/prepare_hf_dataset.py --input path/to/task_suite.json
    python scripts/prepare_hf_dataset.py --format parquet
    python scripts/prepare_hf_dataset.py --format both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "results" / "unified_v3" / "task_suite.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "huggingface"

LEVELS_ORDERED = ["L0_node", "L1_chain", "L2_parallel", "L3_dag"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_task_suite(path: Path) -> dict[str, Any]:
    """Load and validate the task suite JSON file."""
    if not path.exists():
        print(f"Error: task suite not found at {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    required_keys = {"name", "version", "stats", "tasks"}
    missing = required_keys - set(data.keys())
    if missing:
        print(
            f"Error: task suite missing keys: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    return data


def flatten_task(task: dict[str, Any]) -> dict[str, Any]:
    """Flatten a single task dict into a HuggingFace-friendly row.

    Complex nested objects (expected_trace, expected_final_answer,
    step arguments) are JSON-serialized so that the dataset can be
    stored as a flat table while preserving full fidelity.
    """
    trace = task["expected_trace"]
    metadata = task.get("metadata", {})

    # Keep the trace as a structured object; serialize once at the top level
    steps_clean: list[dict[str, Any]] = []
    for step in trace["steps"]:
        steps_clean.append(
            {
                "step_id": step["step_id"],
                "tool_name": step["tool_name"],
                "arguments": step["arguments"],
                "depends_on": step.get("depends_on", []),
                "output_key": step.get("output_key"),
            }
        )

    return {
        "task_id": task["task_id"],
        "level": task["level"],
        "prompt": task["prompt"],
        "available_tools": task["available_tools"],
        "expected_trace": json.dumps(
            {"steps": steps_clean, "final_answer_source": trace["final_answer_source"]},
            ensure_ascii=False,
        ),
        "expected_final_answer": json.dumps(
            task["expected_final_answer"], ensure_ascii=False
        ),
        "num_steps": len(trace["steps"]),
        "num_tools_offered": len(task["available_tools"]),
        "category": metadata.get("category", ""),
        "pattern": metadata.get("pattern", ""),
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  JSONL written: {path}  ({len(rows)} rows, {path.stat().st_size:,} bytes)")


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows as a Parquet file using pyarrow or pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            df.to_parquet(path, engine="pyarrow", index=False)
            print(
                f"  Parquet written: {path}  ({len(rows)} rows, {path.stat().st_size:,} bytes)"
            )
            return
        except ImportError:
            print(
                "Error: neither pyarrow nor pandas is installed. "
                "Install with: uv pip install pyarrow",
                file=sys.stderr,
            )
            sys.exit(1)

    # Build schema explicitly for clean column types
    schema = pa.schema(
        [
            pa.field("task_id", pa.string()),
            pa.field("level", pa.string()),
            pa.field("prompt", pa.string()),
            pa.field("available_tools", pa.list_(pa.string())),
            pa.field("expected_trace", pa.string()),
            pa.field("expected_final_answer", pa.string()),
            pa.field("num_steps", pa.int32()),
            pa.field("num_tools_offered", pa.int32()),
            pa.field("category", pa.string()),
            pa.field("pattern", pa.string()),
        ]
    )

    arrays = [
        pa.array([r["task_id"] for r in rows], type=pa.string()),
        pa.array([r["level"] for r in rows], type=pa.string()),
        pa.array([r["prompt"] for r in rows], type=pa.string()),
        pa.array([r["available_tools"] for r in rows], type=pa.list_(pa.string())),
        pa.array([r["expected_trace"] for r in rows], type=pa.string()),
        pa.array([r["expected_final_answer"] for r in rows], type=pa.string()),
        pa.array([r["num_steps"] for r in rows], type=pa.int32()),
        pa.array([r["num_tools_offered"] for r in rows], type=pa.int32()),
        pa.array([r["category"] for r in rows], type=pa.string()),
        pa.array([r["pattern"] for r in rows], type=pa.string()),
    ]

    table = pa.table(arrays, schema=schema)
    pq.write_table(table, path, compression="zstd")
    print(
        f"  Parquet written: {path}  ({len(rows)} rows, {path.stat().st_size:,} bytes)"
    )


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a summary of the prepared dataset."""
    from collections import Counter

    level_counts = Counter(r["level"] for r in rows)
    category_counts = Counter(r["category"] for r in rows)

    print("\n  Dataset Summary")
    print("  " + "-" * 40)
    print(f"  Total tasks: {len(rows)}")
    print()
    print("  By composition level:")
    for level in LEVELS_ORDERED:
        print(f"    {level}: {level_counts.get(level, 0)}")
    print()
    print(f"  Unique categories: {len(category_counts)}")
    unique_patterns = len(set(r["pattern"] for r in rows if r["pattern"]))
    print(f"  Unique patterns: {unique_patterns}")
    print()

    step_counts = [r["num_steps"] for r in rows]
    print(
        f"  Steps per task: min={min(step_counts)}, max={max(step_counts)}, avg={sum(step_counts) / len(step_counts):.1f}"
    )

    tools_offered = [r["num_tools_offered"] for r in rows]
    print(
        f"  Tools offered:  min={min(tools_offered)}, max={max(tools_offered)}, avg={sum(tools_offered) / len(tools_offered):.1f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare CompToolBench for HuggingFace upload.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to task_suite.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "parquet", "both"],
        default="both",
        help="Output format (default: both)",
    )
    args = parser.parse_args()

    print("CompToolBench Dataset Preparation")
    print(f"{'=' * 50}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output_dir}")
    print(f"  Format: {args.format}")

    # Load and validate
    suite = load_task_suite(args.input)
    tasks = suite["tasks"]
    print(f"\n  Loaded {len(tasks)} tasks (v{suite['version']})")

    # Sort tasks by level then ID for deterministic ordering
    tasks.sort(key=lambda t: (LEVELS_ORDERED.index(t["level"]), t["task_id"]))

    # Flatten
    rows = [flatten_task(t) for t in tasks]

    # Validate: every row should have a non-empty prompt and at least 1 tool
    for row in rows:
        assert row["prompt"].strip(), f"Empty prompt in {row['task_id']}"
        assert row["num_tools_offered"] >= 1, f"No tools in {row['task_id']}"
        assert row["num_steps"] >= 1, f"No steps in {row['task_id']}"

    print_summary(rows)

    # Write outputs
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Writing files...")

    if args.format in ("jsonl", "both"):
        write_jsonl(rows, output_dir / "test.jsonl")

    if args.format in ("parquet", "both"):
        write_parquet(rows, output_dir / "test.parquet")

    print(f"\n  Done. Files ready for upload at: {output_dir}/")
    print("\n  Upload with:")
    print(
        f"    huggingface-cli upload comptoolbench/CompToolBench {output_dir} --repo-type dataset"
    )
    print()


if __name__ == "__main__":
    main()
