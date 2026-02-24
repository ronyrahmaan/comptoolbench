"""Argument matching for scoring model outputs against ground truth.

Three matching strategies:
- Exact match: tool names, enums, booleans
- Fuzzy match: string args (normalized Levenshtein, threshold 0.85)
- Numeric tolerance: float args (relative tolerance 1e-4)
"""

from __future__ import annotations

import json
import math
from typing import Any


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insert
                prev_row[j + 1] + 1,   # delete
                prev_row[j] + cost,    # replace
            ))
        prev_row = curr_row

    return prev_row[-1]


def normalized_similarity(s1: str, s2: str) -> float:
    """Normalized string similarity (1.0 = identical, 0.0 = completely different)."""
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein_distance(s1, s2) / max_len


def match_exact(expected: Any, actual: Any) -> bool:
    """Exact match for tool names, enums, booleans."""
    return str(expected).strip().lower() == str(actual).strip().lower()


def match_string(expected: str, actual: str, threshold: float = 0.85) -> float:
    """Fuzzy string match using normalized Levenshtein similarity.

    Returns a score in [0, 1]. Returns 1.0 for exact match, 0.0 if
    similarity falls below threshold.
    """
    if expected == actual:
        return 1.0

    # Normalize whitespace and case
    e = " ".join(expected.lower().split())
    a = " ".join(actual.lower().split())

    if e == a:
        return 1.0

    sim = normalized_similarity(e, a)
    return sim if sim >= threshold else 0.0


def match_numeric(expected: float, actual: float, rel_tol: float = 1e-4) -> bool:
    """Numeric match with relative tolerance."""
    if expected == actual:
        return True
    if expected == 0:
        return abs(actual) < rel_tol
    return math.isclose(expected, actual, rel_tol=rel_tol)


def match_value(expected: Any, actual: Any) -> float:
    """Match a single argument value, dispatching to the appropriate matcher.

    Returns a score in [0.0, 1.0].
    """
    if expected is None and actual is None:
        return 1.0
    if expected is None or actual is None:
        return 0.0

    # Booleans
    if isinstance(expected, bool):
        return 1.0 if bool(actual) == expected else 0.0

    # Numbers
    if isinstance(expected, (int, float)):
        try:
            return 1.0 if match_numeric(float(expected), float(actual)) else 0.0
        except (ValueError, TypeError):
            return 0.0

    # Lists
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return 0.0
        if len(expected) == 0 and len(actual) == 0:
            return 1.0
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        # Element-wise comparison, order-sensitive
        scores = []
        for i, exp_item in enumerate(expected):
            if i < len(actual):
                scores.append(match_value(exp_item, actual[i]))
            else:
                scores.append(0.0)
        # Penalize extra items
        length_penalty = min(len(expected), len(actual)) / max(len(expected), len(actual))
        return (sum(scores) / len(scores)) * length_penalty

    # Dicts
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0.0
        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0
        all_keys = set(expected.keys()) | set(actual.keys())
        scores = []
        for key in all_keys:
            if key in expected and key in actual:
                scores.append(match_value(expected[key], actual[key]))
            else:
                scores.append(0.0)
        return sum(scores) / len(scores)

    # Strings (default)
    return match_string(str(expected), str(actual))


def match_arguments(
    expected_args: dict[str, Any],
    actual_args: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    """Match all arguments of a tool call.

    Returns:
        Tuple of (overall score [0-1], per-argument scores).
    """
    if not expected_args and not actual_args:
        return 1.0, {}
    if not expected_args or not actual_args:
        return 0.0, {}

    per_arg: dict[str, float] = {}
    all_keys = set(expected_args.keys()) | set(actual_args.keys())

    for key in all_keys:
        if key in expected_args and key in actual_args:
            per_arg[key] = match_value(expected_args[key], actual_args[key])
        elif key in expected_args:
            per_arg[key] = 0.0  # Missing arg
        else:
            per_arg[key] = 0.0  # Extra arg (penalize lightly? for now same)

    overall = sum(per_arg.values()) / len(per_arg) if per_arg else 0.0
    return overall, per_arg
