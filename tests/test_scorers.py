"""Tests for the scoring functions."""

from __future__ import annotations

from comptoolbench.evaluation.scorers import (
    CallScore,
    ModelCall,
    TaskScore,
    _classify_error,
    _longest_common_subsequence,
    score_task,
)
from comptoolbench.tasks.models import (
    CompositionLevel,
    ExpectedTrace,
    Task,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Helpers to build test fixtures
# ---------------------------------------------------------------------------

def _make_task(
    level: CompositionLevel,
    steps: list[tuple[str, str, dict]],
    depends: dict[str, list[str]] | None = None,
    task_id: str = "test_task",
) -> Task:
    """Build a minimal Task for testing.

    steps: list of (step_id, tool_name, arguments)
    depends: {step_id: [dependency_step_ids]}
    """
    depends = depends or {}
    tool_calls = [
        ToolCall(
            step_id=sid,
            tool_name=tool,
            arguments=args,
            depends_on=depends.get(sid, []),
        )
        for sid, tool, args in steps
    ]
    return Task(
        task_id=task_id,
        level=level,
        prompt="Test prompt",
        available_tools=[tool for _, tool, _ in steps],
        expected_trace=ExpectedTrace(
            steps=tool_calls,
            final_answer_source=steps[-1][0],
        ),
        expected_final_answer="test_answer",
    )


def _make_calls(calls: list[tuple[str, dict]]) -> list[ModelCall]:
    """Build ModelCall list from (tool_name, arguments) tuples."""
    return [
        ModelCall(tool_name=tool, arguments=args, call_index=i)
        for i, (tool, args) in enumerate(calls)
    ]


# ---------------------------------------------------------------------------
# LCS tests
# ---------------------------------------------------------------------------

class TestLCS:
    def test_identical(self) -> None:
        assert _longest_common_subsequence(["a", "b", "c"], ["a", "b", "c"]) == 3

    def test_empty(self) -> None:
        assert _longest_common_subsequence([], []) == 0
        assert _longest_common_subsequence(["a"], []) == 0

    def test_partial(self) -> None:
        assert _longest_common_subsequence(["a", "b", "c"], ["a", "c"]) == 2

    def test_reversed(self) -> None:
        assert _longest_common_subsequence(["a", "b", "c"], ["c", "b", "a"]) == 1


# ---------------------------------------------------------------------------
# L0 (single tool) scoring tests
# ---------------------------------------------------------------------------

class TestScoreL0:
    def test_perfect_l0(self) -> None:
        task = _make_task(
            CompositionLevel.NODE,
            [("step_1", "get_weather", {"city": "NYC"})],
        )
        calls = _make_calls([("get_weather", {"city": "NYC"})])
        result = score_task(task, calls)

        assert result.overall == 1.0
        assert result.completeness_score == 1.0
        assert result.error_type is None
        assert len(result.call_scores) == 1
        assert result.call_scores[0].tool_correct is True

    def test_wrong_tool_l0(self) -> None:
        task = _make_task(
            CompositionLevel.NODE,
            [("step_1", "get_weather", {"city": "NYC"})],
        )
        calls = _make_calls([("get_time", {"city": "NYC"})])
        result = score_task(task, calls)

        assert result.overall == 0.0
        assert result.error_type == "E6_hallucinated_tool"

    def test_wrong_args_l0(self) -> None:
        task = _make_task(
            CompositionLevel.NODE,
            [("step_1", "get_weather", {"city": "NYC"})],
        )
        calls = _make_calls([("get_weather", {"city": "London"})])
        result = score_task(task, calls)

        # Tool is correct but args don't match well enough
        assert result.overall == 0.0  # Binary: args < 0.85 → fail

    def test_no_calls_l0(self) -> None:
        task = _make_task(
            CompositionLevel.NODE,
            [("step_1", "get_weather", {"city": "NYC"})],
        )
        result = score_task(task, [])

        assert result.overall == 0.0
        assert result.error_type == "E10_format_error"


# ---------------------------------------------------------------------------
# L1 (chain) scoring tests
# ---------------------------------------------------------------------------

class TestScoreL1:
    def test_perfect_chain(self) -> None:
        task = _make_task(
            CompositionLevel.CHAIN,
            [
                ("step_1", "get_weather", {"city": "NYC"}),
                ("step_2", "translate_text", {"text": "sunny", "target_language": "es"}),
            ],
        )
        calls = _make_calls([
            ("get_weather", {"city": "NYC"}),
            ("translate_text", {"text": "sunny", "target_language": "es"}),
        ])
        result = score_task(task, calls)

        assert result.overall == 1.0
        assert result.tool_sequence_score == 1.0
        assert result.argument_score == 1.0
        assert result.completeness_score == 1.0
        assert result.error_type is None

    def test_wrong_order_chain(self) -> None:
        task = _make_task(
            CompositionLevel.CHAIN,
            [
                ("step_1", "get_weather", {"city": "NYC"}),
                ("step_2", "translate_text", {"text": "sunny", "target_language": "es"}),
            ],
        )
        calls = _make_calls([
            ("translate_text", {"text": "sunny", "target_language": "es"}),
            ("get_weather", {"city": "NYC"}),
        ])
        result = score_task(task, calls)

        # Both tools present but order wrong → LCS=1/2=0.5
        assert result.tool_sequence_score == 0.5
        assert result.error_type == "E3_wrong_order"

    def test_missing_step_chain(self) -> None:
        task = _make_task(
            CompositionLevel.CHAIN,
            [
                ("step_1", "get_weather", {"city": "NYC"}),
                ("step_2", "translate_text", {"text": "sunny", "target_language": "es"}),
            ],
        )
        calls = _make_calls([("get_weather", {"city": "NYC"})])
        result = score_task(task, calls)

        assert result.completeness_score == 0.5
        assert result.error_type == "E8_partial_completion"

    def test_extra_tool_chain(self) -> None:
        task = _make_task(
            CompositionLevel.CHAIN,
            [
                ("step_1", "get_weather", {"city": "NYC"}),
            ],
        )
        calls = _make_calls([
            ("get_weather", {"city": "NYC"}),
            ("get_weather", {"city": "London"}),  # Unnecessary extra
        ])
        result = score_task(task, calls)

        assert result.error_type == "E7_unnecessary_tool"


# ---------------------------------------------------------------------------
# L2 (parallel) scoring tests
# ---------------------------------------------------------------------------

class TestScoreL2:
    def test_perfect_parallel(self) -> None:
        task = _make_task(
            CompositionLevel.PARALLEL,
            [
                ("step_1", "get_weather", {"city": "NYC"}),
                ("step_2", "get_weather", {"city": "London"}),
                ("step_3", "compare_texts", {"text1": "sunny", "text2": "rainy"}),
            ],
            depends={"step_3": ["step_1", "step_2"]},
        )
        calls = _make_calls([
            ("get_weather", {"city": "NYC"}),
            ("get_weather", {"city": "London"}),
            ("compare_texts", {"text1": "sunny", "text2": "rainy"}),
        ])
        result = score_task(task, calls)

        assert result.overall == 1.0
        assert result.data_flow_score == 1.0


# ---------------------------------------------------------------------------
# L3 (DAG) scoring tests
# ---------------------------------------------------------------------------

class TestScoreL3:
    def test_perfect_dag(self) -> None:
        task = _make_task(
            CompositionLevel.DAG,
            [
                ("step_1", "get_weather", {"city": "NYC"}),
                ("step_2", "get_exchange_rate", {"from_currency": "USD", "to_currency": "EUR"}),
                ("step_3", "calculator", {"expression": "25 * 0.92"}),
            ],
            depends={"step_3": ["step_1", "step_2"]},
        )
        calls = _make_calls([
            ("get_weather", {"city": "NYC"}),
            ("get_exchange_rate", {"from_currency": "USD", "to_currency": "EUR"}),
            ("calculator", {"expression": "25 * 0.92"}),
        ])
        result = score_task(task, calls)

        assert result.overall == 1.0
        assert result.data_flow_score == 1.0
        assert result.error_type is None

    def test_broken_data_flow_dag(self) -> None:
        """Step 3 depends on step 2, but step 2 is missing."""
        task = _make_task(
            CompositionLevel.DAG,
            [
                ("step_1", "get_weather", {"city": "NYC"}),
                ("step_2", "get_exchange_rate", {"from_currency": "USD", "to_currency": "EUR"}),
                ("step_3", "calculator", {"expression": "25 * 0.92"}),
            ],
            depends={"step_3": ["step_1", "step_2"]},
        )
        # Missing step_2 entirely
        calls = _make_calls([
            ("get_weather", {"city": "NYC"}),
            ("calculator", {"expression": "25 * 0.92"}),
        ])
        result = score_task(task, calls)

        assert result.completeness_score < 1.0
        assert result.data_flow_score < 1.0
        assert result.error_type == "E8_partial_completion"


# ---------------------------------------------------------------------------
# Error classification tests
# ---------------------------------------------------------------------------

class TestErrorClassification:
    def test_hallucinated_tool(self) -> None:
        trace = ExpectedTrace(
            steps=[ToolCall(step_id="s1", tool_name="get_weather", arguments={})],
            final_answer_source="s1",
        )
        actual = [ModelCall(tool_name="fake_tool", arguments={})]
        call_scores = [CallScore(
            step_id="s1", expected_tool="get_weather",
            actual_tool=None, tool_correct=False, args_score=0.0, matched=False,
        )]
        assert _classify_error(trace, actual, call_scores) == "E6_hallucinated_tool"

    def test_no_calls(self) -> None:
        trace = ExpectedTrace(
            steps=[ToolCall(step_id="s1", tool_name="get_weather", arguments={})],
            final_answer_source="s1",
        )
        assert _classify_error(trace, [], []) == "E10_format_error"

    def test_no_error(self) -> None:
        trace = ExpectedTrace(
            steps=[ToolCall(step_id="s1", tool_name="get_weather", arguments={"city": "NYC"})],
            final_answer_source="s1",
        )
        actual = [ModelCall(tool_name="get_weather", arguments={"city": "NYC"})]
        call_scores = [CallScore(
            step_id="s1", expected_tool="get_weather",
            actual_tool="get_weather", tool_correct=True, args_score=1.0, matched=True,
        )]
        assert _classify_error(trace, actual, call_scores) is None


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestTaskScoreSerialization:
    def test_to_dict(self) -> None:
        ts = TaskScore(
            task_id="t1",
            level=CompositionLevel.CHAIN,
            call_scores=[
                CallScore(
                    step_id="s1", expected_tool="get_weather",
                    actual_tool="get_weather", tool_correct=True,
                    args_score=0.95, matched=True,
                ),
            ],
            tool_sequence_score=1.0,
            argument_score=0.95,
            completeness_score=1.0,
            overall=0.9575,
        )
        d = ts.to_dict()
        assert d["task_id"] == "t1"
        assert d["level"] == "L1_chain"
        assert d["overall"] == 0.9575
        assert len(d["call_scores"]) == 1
        assert d["call_scores"][0]["tool_correct"] is True
