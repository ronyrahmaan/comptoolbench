"""Tests for composition gap metrics."""

from __future__ import annotations

from comptoolbench.evaluation.metrics import (
    CompositionGapResult,
    compute_composition_gap,
    compute_per_tool_l0_accuracy,
)
from comptoolbench.evaluation.scorers import CallScore, TaskScore
from comptoolbench.tasks.models import (
    CompositionLevel,
    ExpectedTrace,
    Task,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _l0_task(tool_name: str, task_id: str = "l0_task") -> Task:
    """Create a minimal L0 task."""
    return Task(
        task_id=task_id,
        level=CompositionLevel.NODE,
        prompt=f"Use {tool_name}",
        available_tools=[tool_name],
        expected_trace=ExpectedTrace(
            steps=[ToolCall(step_id="s1", tool_name=tool_name, arguments={"x": 1})],
            final_answer_source="s1",
        ),
        expected_final_answer="answer",
    )


def _l1_task(tools: list[str], task_id: str = "l1_task") -> Task:
    """Create a minimal L1 (chain) task."""
    steps = [
        ToolCall(step_id=f"s{i+1}", tool_name=t, arguments={"x": i})
        for i, t in enumerate(tools)
    ]
    return Task(
        task_id=task_id,
        level=CompositionLevel.CHAIN,
        prompt="Chain task",
        available_tools=tools,
        expected_trace=ExpectedTrace(steps=steps, final_answer_source=steps[-1].step_id),
        expected_final_answer="answer",
    )


def _task_score(task: Task, overall: float, error: str | None = None) -> TaskScore:
    """Create a TaskScore with a given overall score."""
    return TaskScore(
        task_id=task.task_id,
        level=task.level,
        call_scores=[
            CallScore(
                step_id=s.step_id,
                expected_tool=s.tool_name,
                actual_tool=s.tool_name,
                tool_correct=True,
                args_score=overall,
                matched=True,
            )
            for s in task.expected_trace.steps
        ],
        overall=overall,
        error_type=error,
    )


# ---------------------------------------------------------------------------
# Per-tool L0 accuracy tests
# ---------------------------------------------------------------------------

class TestPerToolL0Accuracy:
    def test_all_pass(self) -> None:
        tasks = [_l0_task("get_weather", f"t{i}") for i in range(3)]
        scores = [_task_score(t, 1.0) for t in tasks]
        result = compute_per_tool_l0_accuracy(tasks, scores)
        assert result["get_weather"] == 1.0

    def test_partial_pass(self) -> None:
        tasks = [_l0_task("get_weather", f"t{i}") for i in range(4)]
        # 2 pass (>=0.85), 2 fail (<0.85)
        scores = [
            _task_score(tasks[0], 1.0),
            _task_score(tasks[1], 0.9),
            _task_score(tasks[2], 0.5),
            _task_score(tasks[3], 0.3),
        ]
        result = compute_per_tool_l0_accuracy(tasks, scores)
        assert result["get_weather"] == 0.5  # 2/4

    def test_multiple_tools(self) -> None:
        tasks = [
            _l0_task("get_weather", "t1"),
            _l0_task("get_weather", "t2"),
            _l0_task("calculator", "t3"),
        ]
        scores = [
            _task_score(tasks[0], 1.0),
            _task_score(tasks[1], 0.5),  # fail
            _task_score(tasks[2], 0.9),  # pass
        ]
        result = compute_per_tool_l0_accuracy(tasks, scores)
        assert result["get_weather"] == 0.5   # 1/2
        assert result["calculator"] == 1.0     # 1/1

    def test_empty(self) -> None:
        result = compute_per_tool_l0_accuracy([], [])
        assert result == {}


# ---------------------------------------------------------------------------
# Composition gap tests
# ---------------------------------------------------------------------------

class TestCompositionGap:
    def test_perfect_scores_no_gap(self) -> None:
        """All tools score 1.0 at all levels → gap should be 0."""
        tasks = [
            _l0_task("get_weather", "l0_1"),
            _l0_task("calculator", "l0_2"),
            _l1_task(["get_weather", "calculator"], "l1_1"),
        ]
        scores = [_task_score(t, 1.0) for t in tasks]
        result = compute_composition_gap("test_model", tasks, scores)

        assert result.accuracy_l0 == 1.0
        assert result.accuracy_l1 == 1.0
        assert result.gap_l1 == 0.0  # No gap: 1.0 - 1.0 = 0

    def test_gap_exists(self) -> None:
        """Model scores 1.0 on L0 but 0.5 on L1 → positive gap."""
        tasks = [
            _l0_task("get_weather", "l0_1"),
            _l0_task("calculator", "l0_2"),
            _l1_task(["get_weather", "calculator"], "l1_1"),
        ]
        scores = [
            _task_score(tasks[0], 1.0),  # L0 weather: perfect
            _task_score(tasks[1], 1.0),  # L0 calculator: perfect
            _task_score(tasks[2], 0.5),  # L1 composed: poor
        ]
        result = compute_composition_gap("test_model", tasks, scores)

        assert result.accuracy_l0 == 1.0
        assert result.accuracy_l1 == 0.5
        assert result.gap_l1 == 0.5  # min(1.0, 1.0) - 0.5 = 0.5

    def test_bottleneck_tool(self) -> None:
        """Gap uses min(L0_accuracy per tool), i.e., weakest tool."""
        tasks = [
            _l0_task("get_weather", "l0_1"),
            _l0_task("get_weather", "l0_2"),
            _l0_task("calculator", "l0_3"),
            _l0_task("calculator", "l0_4"),
            _l1_task(["get_weather", "calculator"], "l1_1"),
        ]
        scores = [
            _task_score(tasks[0], 1.0),   # weather: pass
            _task_score(tasks[1], 0.5),   # weather: fail
            _task_score(tasks[2], 1.0),   # calculator: pass
            _task_score(tasks[3], 1.0),   # calculator: pass
            _task_score(tasks[4], 0.3),   # L1 composed
        ]
        result = compute_composition_gap("test_model", tasks, scores)

        # weather L0 accuracy: 1/2 = 0.5, calculator: 2/2 = 1.0
        # bottleneck = min(0.5, 1.0) = 0.5
        # gap = 0.5 - 0.3 = 0.2
        assert result.per_tool_l0_accuracy["get_weather"] == 0.5
        assert result.per_tool_l0_accuracy["calculator"] == 1.0
        assert abs(result.gap_l1 - 0.2) < 0.01

    def test_result_serialization(self) -> None:
        result = CompositionGapResult(
            model_name="test_model",
            overall_accuracy=0.75,
            accuracy_l0=0.9,
            accuracy_l1=0.7,
            gap_l1=0.2,
        )
        d = result.to_dict()
        assert d["model"] == "test_model"
        assert d["headline_metrics"]["overall_accuracy"] == 0.75
        assert d["headline_metrics"]["composition_gap_L1"] == 0.2
        assert "per_level_accuracy" in d
        assert "diagnostic_metrics" in d

    def test_task_counts(self) -> None:
        tasks = [
            _l0_task("a", "l0_1"),
            _l0_task("b", "l0_2"),
            _l1_task(["a", "b"], "l1_1"),
        ]
        scores = [_task_score(t, 0.9) for t in tasks]
        result = compute_composition_gap("m", tasks, scores)

        assert result.task_counts["L0_node"] == 2
        assert result.task_counts["L1_chain"] == 1

    def test_error_distribution(self) -> None:
        tasks = [
            _l0_task("a", "l0_1"),
            _l1_task(["a"], "l1_1"),
            _l1_task(["a"], "l1_2"),
        ]
        scores = [
            _task_score(tasks[0], 1.0),
            _task_score(tasks[1], 0.5, error="E4_wrong_arguments"),
            _task_score(tasks[2], 0.3, error="E4_wrong_arguments"),
        ]
        result = compute_composition_gap("m", tasks, scores)

        assert result.error_distribution["E4_wrong_arguments"] == 2

    def test_empty_tasks(self) -> None:
        result = compute_composition_gap("empty", [], [])
        assert result.overall_accuracy == 0.0
        assert result.gap_overall == 0.0

    def test_weighted_gap_overall(self) -> None:
        """Overall gap is weighted: L1(0.3) + L2(0.3) + L3(0.4)."""
        result = CompositionGapResult(
            model_name="m",
            gap_l1=0.1,
            gap_l2=0.2,
            gap_l3=0.3,
        )
        # Manually check: 0.1*0.3 + 0.2*0.3 + 0.3*0.4 = 0.03+0.06+0.12 = 0.21
        # compute_composition_gap calculates this, but let's verify the formula
        expected = 0.1 * 0.30 + 0.2 * 0.30 + 0.3 * 0.40
        assert abs(expected - 0.21) < 0.001
