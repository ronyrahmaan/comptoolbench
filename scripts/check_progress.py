#!/usr/bin/env python3
"""Check benchmark progress across all model checkpoints.

Usage:
    uv run python scripts/check_progress.py results/local_run_001/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    """Print benchmark progress summary."""
    if len(sys.argv) < 2:
        print("Usage: check_progress.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    # Count total tasks
    suite_path = results_dir / "task_suite.json"
    if suite_path.exists():
        with open(suite_path) as f:
            suite = json.load(f)
        total = len(suite["tasks"])
    else:
        total = "?"

    print(f"Results dir: {results_dir}")
    print(f"Total tasks: {total}")
    print("-" * 60)

    # Check each checkpoint
    checkpoints = sorted(results_dir.glob("checkpoint_*.jsonl"))
    if not checkpoints:
        print("No checkpoints found yet.")
        return

    for cp in checkpoints:
        model = cp.stem.replace("checkpoint_", "")
        with open(cp) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        done = len(lines)
        if lines:
            scores = [entry["score"]["overall"] for entry in lines]
            errors = sum(1 for entry in lines if entry.get("error"))
            timeouts = sum(1 for entry in lines if entry.get("error") and "timeout" in str(entry.get("error", "")))
            avg_score = sum(scores) / len(scores)

            # Per-level breakdown
            level_scores: dict[str, list[float]] = {}
            for entry in lines:
                level = entry["score"]["level"]
                level_key = level.split("_")[0]  # "L0_node" -> "L0"
                level_scores.setdefault(level_key, []).append(entry["score"]["overall"])

            level_str = " | ".join(
                f"{k}={sum(v)/len(v)*100:.0f}%({len(v)})"
                for k, v in sorted(level_scores.items())
            )

            print(f"{model:25s}  {done}/{total}  avg={avg_score*100:.1f}%  err={errors}  timeout={timeouts}")
            print(f"{'':25s}  {level_str}")
        else:
            print(f"{model:25s}  {done}/{total}  (no results)")

    print("-" * 60)

    # Check for final results
    result_files = list(results_dir.glob("results_*.json"))
    if result_files:
        print(f"Final results: {result_files[0].name}")

        # Show leaderboard
        lb_path = results_dir / "leaderboard.csv"
        if lb_path.exists():
            print("\nLeaderboard:")
            for line in lb_path.read_text().strip().split("\n"):
                print(f"  {line}")
    else:
        print("Final results: Not yet generated (still running)")


if __name__ == "__main__":
    main()
