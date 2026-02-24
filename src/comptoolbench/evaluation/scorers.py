"""Scoring functions for CompToolBench.

Scores model outputs at three levels:
1. Per-call: Did the model pick the right tool with right args?
2. Per-task: How well did the model complete the full composition?
3. Per-level: Aggregated accuracy at each composition level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from comptoolbench.evaluation.matchers import match_arguments, match_exact
from comptoolbench.tasks.models import CompositionLevel, ExpectedTrace, Task, ToolCall


@dataclass
class CallScore:
    """Score for a single tool call in the model's response."""

    step_id: str
    expected_tool: str
    actual_tool: str | None
    tool_correct: bool
    args_score: float  # 0.0 - 1.0
    per_arg_scores: dict[str, float] = field(default_factory=dict)
    matched: bool = True  # Was this expected call matched to an actual call?

    @property
    def overall(self) -> float:
        """Combined score: tool selection + argument accuracy."""
        if not self.matched:
            return 0.0
        if not self.tool_correct:
            return 0.0
        return self.args_score


@dataclass
class TaskScore:
    """Score for a complete task (all calls combined)."""

    task_id: str
    level: CompositionLevel
    call_scores: list[CallScore]
    tool_sequence_score: float = 0.0  # How well the tool order matches
    argument_score: float = 0.0       # Average argument accuracy
    completeness_score: float = 0.0   # Fraction of expected calls made
    data_flow_score: float = 0.0      # Were outputs routed correctly?
    overall: float = 0.0              # Weighted combination
    error_type: str | None = None     # Primary error classification

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging."""
        return {
            "task_id": self.task_id,
            "level": self.level.value,
            "overall": round(self.overall, 4),
            "tool_sequence_score": round(self.tool_sequence_score, 4),
            "argument_score": round(self.argument_score, 4),
            "completeness_score": round(self.completeness_score, 4),
            "data_flow_score": round(self.data_flow_score, 4),
            "error_type": self.error_type,
            "num_expected_calls": len(self.call_scores),
            "num_matched_calls": sum(1 for c in self.call_scores if c.matched),
            "call_scores": [
                {
                    "step": c.step_id,
                    "tool_correct": c.tool_correct,
                    "args_score": round(c.args_score, 4),
                    "matched": c.matched,
                }
                for c in self.call_scores
            ],
        }


@dataclass
class ModelCall:
    """A tool call extracted from a model's response."""

    tool_name: str
    arguments: dict[str, Any]
    call_index: int = 0  # Order in the model's response


def _longest_common_subsequence(seq1: list[str], seq2: list[str]) -> int:
    """Length of the longest common subsequence of two string lists."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _classify_error(
    expected: ExpectedTrace,
    actual_calls: list[ModelCall],
    call_scores: list[CallScore],
) -> str | None:
    """Classify the primary error type (see ARCHITECTURE.md Appendix C)."""
    if not actual_calls:
        return "E10_format_error"  # No calls parsed

    expected_tools = [s.tool_name for s in expected.steps]
    actual_tools = [c.tool_name for c in actual_calls]

    # Check for hallucinated tools
    expected_set = {s.tool_name for s in expected.steps}
    for at in actual_tools:
        if at not in expected_set:
            return "E6_hallucinated_tool"

    # Check completeness
    if len(actual_calls) < len(expected.steps):
        return "E8_partial_completion"

    # Check for wrong tool selection
    if any(not cs.tool_correct for cs in call_scores if cs.matched):
        return "E1_wrong_tool"

    # Check sequence (order errors)
    lcs = _longest_common_subsequence(expected_tools, actual_tools)
    if lcs < len(expected_tools):
        return "E3_wrong_order"

    # Check arguments
    low_args = [cs for cs in call_scores if cs.matched and cs.args_score < 0.85]
    if low_args:
        return "E4_wrong_arguments"

    # Check for unnecessary extra calls
    if len(actual_calls) > len(expected.steps):
        return "E7_unnecessary_tool"

    return None  # No error detected


def score_task(
    task: Task,
    actual_calls: list[ModelCall],
) -> TaskScore:
    """Score a model's output against a task's expected trace.

    This is the core scoring function. It:
    1. Matches actual calls to expected steps (greedy, order-preserving)
    2. Scores each matched pair (tool selection + argument accuracy)
    3. Computes task-level metrics (sequence, args, completeness, data flow)
    4. Combines into a weighted overall score based on composition level
    """
    expected = task.expected_trace
    expected_steps = expected.steps

    # --- Step 1: Match actual calls to expected steps (greedy) ---
    call_scores: list[CallScore] = []
    used_actual: set[int] = set()

    for step in expected_steps:
        best_match: int | None = None
        best_score = -1.0

        for i, actual in enumerate(actual_calls):
            if i in used_actual:
                continue
            if match_exact(step.tool_name, actual.tool_name):
                args_score, per_arg = match_arguments(step.arguments, actual.arguments)
                score = args_score
                if score > best_score:
                    best_score = score
                    best_match = i

        if best_match is not None:
            actual = actual_calls[best_match]
            args_score, per_arg = match_arguments(step.arguments, actual.arguments)
            call_scores.append(CallScore(
                step_id=step.step_id,
                expected_tool=step.tool_name,
                actual_tool=actual.tool_name,
                tool_correct=True,
                args_score=args_score,
                per_arg_scores=per_arg,
                matched=True,
            ))
            used_actual.add(best_match)
        else:
            # Try to find a call with the wrong tool name at this position
            # (for error classification)
            call_scores.append(CallScore(
                step_id=step.step_id,
                expected_tool=step.tool_name,
                actual_tool=None,
                tool_correct=False,
                args_score=0.0,
                matched=False,
            ))

    # --- Step 2: Compute sub-scores ---
    expected_tools = [s.tool_name for s in expected_steps]
    actual_tools = [c.tool_name for c in actual_calls]

    # Tool sequence score (LCS-based)
    if expected_tools:
        lcs = _longest_common_subsequence(expected_tools, actual_tools)
        tool_sequence_score = lcs / len(expected_tools)
    else:
        tool_sequence_score = 1.0

    # Argument score (average of matched calls)
    matched_scores = [cs.args_score for cs in call_scores if cs.matched]
    argument_score = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0

    # Completeness score
    completeness_score = sum(1 for cs in call_scores if cs.matched) / max(len(expected_steps), 1)

    # Data flow score (simplified: check that dependent steps have correct tool)
    # A more sophisticated version would check actual output routing
    deps_correct = 0
    deps_total = 0
    for step in expected_steps:
        if step.depends_on:
            deps_total += 1
            # Check if this step was matched AND its dependencies were matched
            step_score = next((cs for cs in call_scores if cs.step_id == step.step_id), None)
            dep_scores = [
                cs for cs in call_scores
                if cs.step_id in step.depends_on and cs.matched
            ]
            if step_score and step_score.matched and len(dep_scores) == len(step.depends_on):
                deps_correct += 1
    data_flow_score = deps_correct / deps_total if deps_total > 0 else 1.0

    # --- Step 3: Weighted overall score by level ---
    level = task.level
    if level == CompositionLevel.NODE:
        # Binary: pass if tool correct AND args >= 0.85
        if call_scores and call_scores[0].matched and call_scores[0].tool_correct:
            overall = 1.0 if argument_score >= 0.85 else 0.0
        else:
            overall = 0.0
    elif level == CompositionLevel.CHAIN:
        overall = (
            tool_sequence_score * 0.40
            + argument_score * 0.35
            + completeness_score * 0.25
        )
    elif level == CompositionLevel.PARALLEL:
        # For parallel, tool order within the parallel segment doesn't matter
        # Use set-based comparison for the parallel portion
        overall = (
            tool_sequence_score * 0.35
            + argument_score * 0.35
            + data_flow_score * 0.15
            + completeness_score * 0.15
        )
    else:  # DAG
        overall = (
            tool_sequence_score * 0.30
            + argument_score * 0.30
            + data_flow_score * 0.25
            + completeness_score * 0.15
        )

    # --- Step 4: Error classification ---
    error_type = _classify_error(expected, actual_calls, call_scores)

    return TaskScore(
        task_id=task.task_id,
        level=level,
        call_scores=call_scores,
        tool_sequence_score=tool_sequence_score,
        argument_score=argument_score,
        completeness_score=completeness_score,
        data_flow_score=data_flow_score,
        overall=overall,
        error_type=error_type,
    )
