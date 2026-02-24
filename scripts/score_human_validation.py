#!/usr/bin/env python3
"""Score completed human validation results for CompToolBench.

Reads the annotated CSV (with is_correct column filled Y/N), computes
per-level quality rates, flags incorrect tasks, and optionally computes
inter-annotator agreement (Cohen's kappa) when multiple CSVs are provided.

Usage:
    # Single annotator:
    uv run python scripts/score_human_validation.py results/human_validation/tasks_for_review.csv

    # Multiple annotators (inter-annotator agreement):
    uv run python scripts/score_human_validation.py annotator_1.csv annotator_2.csv

    # Custom output:
    uv run python scripts/score_human_validation.py results/human_validation/tasks_for_review.csv --output report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("score_validation")


def load_annotations(csv_path: Path) -> list[dict]:
    """Load annotated CSV and validate the is_correct column.

    Args:
        csv_path: Path to the completed CSV file.

    Returns:
        List of row dicts with normalized is_correct values (True/False/None).

    Raises:
        SystemExit: If file is missing or has no valid annotations.
    """
    if not csv_path.exists():
        logger.error("File not found: %s", csv_path)
        sys.exit(1)

    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        if "is_correct" not in (reader.fieldnames or []):
            logger.error("CSV missing 'is_correct' column: %s", csv_path)
            sys.exit(1)

        for row in reader:
            raw = row.get("is_correct", "").strip().upper()
            if raw in ("Y", "YES", "1", "TRUE", "CORRECT"):
                row["_is_correct"] = True
            elif raw in ("N", "NO", "0", "FALSE", "INCORRECT"):
                row["_is_correct"] = False
            else:
                row["_is_correct"] = None  # Unannotated
            rows.append(row)

    annotated = [r for r in rows if r["_is_correct"] is not None]
    if not annotated:
        logger.error(
            "No annotations found in %s. Fill the 'is_correct' column with Y/N.",
            csv_path,
        )
        sys.exit(1)

    unannotated = len(rows) - len(annotated)
    if unannotated > 0:
        logger.warning(
            "%d/%d tasks have no annotation (skipped in scoring)",
            unannotated,
            len(rows),
        )

    return rows


def compute_quality(rows: list[dict]) -> dict:
    """Compute overall and per-level task quality metrics.

    Args:
        rows: List of row dicts with _is_correct field.

    Returns:
        Dictionary with quality metrics and flagged tasks.
    """
    annotated = [r for r in rows if r["_is_correct"] is not None]
    total = len(annotated)
    correct = sum(1 for r in annotated if r["_is_correct"])

    # Per-level breakdown
    by_level: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in annotated:
        level = r.get("level", "unknown")
        by_level[level]["total"] += 1
        if r["_is_correct"]:
            by_level[level]["correct"] += 1

    level_quality = {}
    for level, counts in sorted(by_level.items()):
        pct = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        level_quality[level] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "incorrect": counts["total"] - counts["correct"],
            "quality_pct": round(pct * 100, 1),
        }

    # Per-category breakdown
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in annotated:
        cat = r.get("category", "unknown")
        by_category[cat]["total"] += 1
        if r["_is_correct"]:
            by_category[cat]["correct"] += 1

    category_quality = {}
    for cat, counts in sorted(by_category.items()):
        pct = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        category_quality[cat] = {
            "total": counts["total"],
            "correct": counts["correct"],
            "quality_pct": round(pct * 100, 1),
        }

    # Flag incorrect tasks
    flagged = []
    for r in annotated:
        if not r["_is_correct"]:
            flagged.append({
                "task_id": r.get("task_id", "?"),
                "level": r.get("level", "?"),
                "category": r.get("category", "?"),
                "prompt": r.get("prompt", "")[:120],
                "expected_tools": r.get("expected_tools", ""),
                "notes": r.get("notes", ""),
            })

    return {
        "overall": {
            "total_annotated": total,
            "correct": correct,
            "incorrect": total - correct,
            "quality_pct": round(correct / total * 100, 1) if total > 0 else 0.0,
        },
        "by_level": level_quality,
        "by_category": category_quality,
        "flagged_tasks": flagged,
    }


def compute_cohens_kappa(rows_a: list[dict], rows_b: list[dict]) -> float | None:
    """Compute Cohen's kappa for inter-annotator agreement.

    Only considers tasks that both annotators have labeled.

    Args:
        rows_a: Annotations from annotator A.
        rows_b: Annotations from annotator B.

    Returns:
        Cohen's kappa coefficient, or None if insufficient overlap.
    """
    # Build lookup by task_id
    map_a = {r["task_id"]: r["_is_correct"] for r in rows_a if r["_is_correct"] is not None}
    map_b = {r["task_id"]: r["_is_correct"] for r in rows_b if r["_is_correct"] is not None}

    common_ids = sorted(set(map_a.keys()) & set(map_b.keys()))
    if len(common_ids) < 2:
        logger.warning("Fewer than 2 overlapping annotations; cannot compute kappa.")
        return None

    # Count agreement table
    n = len(common_ids)
    # Observed agreement
    agree = sum(1 for tid in common_ids if map_a[tid] == map_b[tid])
    p_o = agree / n

    # Expected agreement by chance
    a_yes = sum(1 for tid in common_ids if map_a[tid]) / n
    b_yes = sum(1 for tid in common_ids if map_b[tid]) / n
    a_no = 1.0 - a_yes
    b_no = 1.0 - b_yes
    p_e = (a_yes * b_yes) + (a_no * b_no)

    if p_e == 1.0:
        return 1.0  # Perfect agreement by definition

    kappa = (p_o - p_e) / (1.0 - p_e)
    return round(kappa, 4)


def print_report(report: dict, kappa: float | None = None) -> None:
    """Print a formatted summary report to stdout."""
    overall = report["overall"]

    print("\n" + "=" * 60)
    print("  CompToolBench Human Validation — Scoring Report")
    print("=" * 60)
    print(f"  Total annotated: {overall['total_annotated']}")
    print(f"  Correct:         {overall['correct']}")
    print(f"  Incorrect:       {overall['incorrect']}")
    print(f"  Quality:         {overall['quality_pct']}%")
    print()

    print("  Per-level quality:")
    for level, stats in report["by_level"].items():
        bar = "#" * int(stats["quality_pct"] / 5)
        print(
            f"    {level:15s}  {stats['correct']}/{stats['total']}  "
            f"({stats['quality_pct']:5.1f}%)  {bar}"
        )
    print()

    print("  Per-category quality:")
    for cat, stats in report["by_category"].items():
        print(
            f"    {cat:25s}  {stats['correct']}/{stats['total']}  "
            f"({stats['quality_pct']:5.1f}%)"
        )
    print()

    if kappa is not None:
        interpretation = _interpret_kappa(kappa)
        print(f"  Inter-annotator agreement (Cohen's kappa): {kappa:.4f} ({interpretation})")
        print()

    flagged = report["flagged_tasks"]
    if flagged:
        print(f"  Flagged tasks ({len(flagged)} incorrect):")
        for t in flagged:
            print(f"    [{t['task_id']}] {t['level']} | {t['category']}")
            print(f"      Prompt: {t['prompt']}...")
            print(f"      Expected: {t['expected_tools']}")
            if t["notes"]:
                print(f"      Notes: {t['notes']}")
            print()
    else:
        print("  No tasks flagged as incorrect.")

    print("=" * 60)
    print()


def _interpret_kappa(kappa: float) -> str:
    """Return a human-readable interpretation of Cohen's kappa."""
    if kappa < 0.0:
        return "less than chance agreement"
    elif kappa < 0.21:
        return "slight agreement"
    elif kappa < 0.41:
        return "fair agreement"
    elif kappa < 0.61:
        return "moderate agreement"
    elif kappa < 0.81:
        return "substantial agreement"
    else:
        return "almost perfect agreement"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Score completed human validation annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        type=str,
        help=(
            "Path(s) to annotated CSV file(s). "
            "If two files are provided, inter-annotator agreement is computed."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the JSON report (default: alongside first CSV)",
    )
    return parser.parse_args()


def main() -> None:
    """Score human validation annotations and generate report."""
    args = parse_args()
    csv_paths = [Path(p) for p in args.csv_files]

    # Load primary annotations
    primary_rows = load_annotations(csv_paths[0])
    report = compute_quality(primary_rows)

    # Inter-annotator agreement (if two CSVs provided)
    kappa: float | None = None
    if len(csv_paths) >= 2:
        secondary_rows = load_annotations(csv_paths[1])
        kappa = compute_cohens_kappa(primary_rows, secondary_rows)
        if kappa is not None:
            report["inter_annotator_agreement"] = {
                "cohens_kappa": kappa,
                "interpretation": _interpret_kappa(kappa),
                "annotator_files": [str(p) for p in csv_paths[:2]],
            }

    # Print report
    print_report(report, kappa=kappa)

    # Save JSON report
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = csv_paths[0].parent / "validation_report.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", output_path)


if __name__ == "__main__":
    main()
