"""Composition gap metrics and diagnostic analysis.

The headline metric of CompToolBench: the Composition Gap measures
how much accuracy drops when models compose tools they know individually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from comptoolbench.evaluation.scorers import TaskScore
from comptoolbench.tasks.models import CompositionLevel, Task


@dataclass
class CompositionGapResult:
    """Result of composition gap analysis for a single model."""

    model_name: str
    overall_accuracy: float = 0.0

    # Per-level accuracy
    accuracy_l0: float = 0.0
    accuracy_l1: float = 0.0
    accuracy_l2: float = 0.0
    accuracy_l3: float = 0.0

    # Composition gap (the headline metric)
    gap_l1: float = 0.0
    gap_l2: float = 0.0
    gap_l3: float = 0.0
    gap_overall: float = 0.0

    # Per-tool L0 accuracy (needed for gap calculation)
    per_tool_l0_accuracy: dict[str, float] = field(default_factory=dict)

    # Diagnostic metrics
    tool_selection_accuracy: float = 0.0
    argument_accuracy: float = 0.0
    data_flow_accuracy: float = 0.0
    completion_rate: float = 0.0
    hallucinated_tool_rate: float = 0.0
    early_termination_rate: float = 0.0

    # Error taxonomy distribution
    error_distribution: dict[str, int] = field(default_factory=dict)

    # Task counts
    task_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON export."""
        return {
            "model": self.model_name,
            "headline_metrics": {
                "overall_accuracy": round(self.overall_accuracy, 4),
                "composition_gap_overall": round(self.gap_overall, 4),
                "composition_gap_L1": round(self.gap_l1, 4),
                "composition_gap_L2": round(self.gap_l2, 4),
                "composition_gap_L3": round(self.gap_l3, 4),
            },
            "per_level_accuracy": {
                "L0_node": round(self.accuracy_l0, 4),
                "L1_chain": round(self.accuracy_l1, 4),
                "L2_parallel": round(self.accuracy_l2, 4),
                "L3_dag": round(self.accuracy_l3, 4),
            },
            "diagnostic_metrics": {
                "tool_selection_accuracy": round(self.tool_selection_accuracy, 4),
                "argument_accuracy": round(self.argument_accuracy, 4),
                "data_flow_accuracy": round(self.data_flow_accuracy, 4),
                "completion_rate": round(self.completion_rate, 4),
                "hallucinated_tool_rate": round(self.hallucinated_tool_rate, 4),
                "early_termination_rate": round(self.early_termination_rate, 4),
            },
            "error_distribution": self.error_distribution,
            "per_tool_l0_accuracy": {
                k: round(v, 4) for k, v in sorted(self.per_tool_l0_accuracy.items())
            },
            "task_counts": self.task_counts,
        }


def compute_per_tool_l0_accuracy(
    l0_tasks: list[Task],
    l0_scores: list[TaskScore],
) -> dict[str, float]:
    """Compute per-tool accuracy on L0 (single-tool) tasks.

    Returns: {tool_name: accuracy} where accuracy is the fraction of
    L0 tasks using that tool where the model scored >= 0.85.
    """
    tool_scores: dict[str, list[float]] = {}

    for task, score in zip(l0_tasks, l0_scores):
        # Each L0 task uses exactly one tool
        tool_name = task.expected_trace.steps[0].tool_name
        tool_scores.setdefault(tool_name, []).append(score.overall)

    return {
        tool: sum(1 for s in scores if s >= 0.85) / len(scores)
        for tool, scores in tool_scores.items()
        if scores
    }


def compute_composition_gap(
    model_name: str,
    tasks: list[Task],
    scores: list[TaskScore],
) -> CompositionGapResult:
    """Compute the full composition gap analysis for a model.

    This is the core metric of CompToolBench. For each L1/L2/L3 task,
    we compare the model's composed accuracy against its individual
    (L0) accuracy on the same tools, using the weakest tool as the
    bottleneck estimate.

    CompositionGap = min(L0_accuracy per tool in composition) - composed_accuracy
    """
    result = CompositionGapResult(model_name=model_name)

    # Group tasks and scores by level
    by_level: dict[CompositionLevel, list[tuple[Task, TaskScore]]] = {}
    for task, score in zip(tasks, scores):
        by_level.setdefault(task.level, []).append((task, score))

    # Per-level accuracy
    for level, pairs in by_level.items():
        level_scores = [s.overall for _, s in pairs]
        avg = sum(level_scores) / len(level_scores) if level_scores else 0.0
        if level == CompositionLevel.NODE:
            result.accuracy_l0 = avg
        elif level == CompositionLevel.CHAIN:
            result.accuracy_l1 = avg
        elif level == CompositionLevel.PARALLEL:
            result.accuracy_l2 = avg
        elif level == CompositionLevel.DAG:
            result.accuracy_l3 = avg

    # Overall accuracy
    all_scores = [s.overall for _, s in sum(by_level.values(), [])]
    result.overall_accuracy = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Per-tool L0 accuracy (needed for gap calculation)
    l0_pairs = by_level.get(CompositionLevel.NODE, [])
    if l0_pairs:
        l0_tasks_list, l0_scores_list = zip(*l0_pairs)
        result.per_tool_l0_accuracy = compute_per_tool_l0_accuracy(
            list(l0_tasks_list), list(l0_scores_list)
        )

    # Composition gap at each level
    for level, gap_attr in [
        (CompositionLevel.CHAIN, "gap_l1"),
        (CompositionLevel.PARALLEL, "gap_l2"),
        (CompositionLevel.DAG, "gap_l3"),
    ]:
        level_pairs = by_level.get(level, [])
        gaps = []
        for task, score in level_pairs:
            # Get per-tool L0 accuracy for tools in this composition
            tools_in_comp = task.expected_trace.unique_tools
            tool_accs = [
                result.per_tool_l0_accuracy.get(tool, 0.0)
                for tool in tools_in_comp
            ]
            if tool_accs:
                individual_acc = min(tool_accs)  # Bottleneck: weakest tool
                composed_acc = score.overall
                gaps.append(individual_acc - composed_acc)

        if gaps:
            setattr(result, gap_attr, sum(gaps) / len(gaps))

    # Overall composition gap (weighted: DAG counts more)
    weights = {"gap_l1": 0.30, "gap_l2": 0.30, "gap_l3": 0.40}
    weighted_sum = sum(
        getattr(result, attr) * weight
        for attr, weight in weights.items()
    )
    result.gap_overall = weighted_sum

    # Diagnostic metrics
    all_call_scores = [
        cs
        for _, task_score in sum(by_level.values(), [])
        for cs in task_score.call_scores
    ]

    if all_call_scores:
        matched = [cs for cs in all_call_scores if cs.matched]
        result.tool_selection_accuracy = (
            sum(1 for cs in matched if cs.tool_correct) / len(matched)
            if matched else 0.0
        )
        result.argument_accuracy = (
            sum(cs.args_score for cs in matched) / len(matched)
            if matched else 0.0
        )
        result.completion_rate = len(matched) / len(all_call_scores)

    # Data flow accuracy (from task-level scores)
    all_task_scores = [s for _, s in sum(by_level.values(), [])]
    df_scores = [s.data_flow_score for s in all_task_scores]
    result.data_flow_accuracy = sum(df_scores) / len(df_scores) if df_scores else 0.0

    # Error distribution
    error_counts: dict[str, int] = {}
    for task_score in all_task_scores:
        if task_score.error_type:
            error_counts[task_score.error_type] = error_counts.get(task_score.error_type, 0) + 1
    result.error_distribution = error_counts

    # Specific rates
    total_composed = sum(
        len(pairs)
        for level, pairs in by_level.items()
        if level != CompositionLevel.NODE
    )
    if total_composed > 0:
        result.hallucinated_tool_rate = (
            error_counts.get("E6_hallucinated_tool", 0) / total_composed
        )
        result.early_termination_rate = (
            error_counts.get("E8_partial_completion", 0) / total_composed
        )

    # Task counts
    result.task_counts = {
        level.value: len(pairs) for level, pairs in by_level.items()
    }

    return result
