"""Tests for the CompositionEngine (v2 task generator)."""

from __future__ import annotations

import pytest

from comptoolbench.generators.composition_engine import CompositionEngine
from comptoolbench.tasks.models import CompositionLevel, TaskSuite
from comptoolbench.tools.base import ToolMode


@pytest.fixture
def engine() -> CompositionEngine:
    """Create a fresh CompositionEngine with a fixed seed."""
    return CompositionEngine(seed=42, mode=ToolMode.SIMULATED)


class TestEngineBasics:
    """Basic engine functionality."""

    def test_engine_init(self, engine: CompositionEngine) -> None:
        assert engine.seed == 42
        assert engine.mode == ToolMode.SIMULATED
        assert len(engine._all_tool_names) > 100  # 106 tools

    def test_generator_counts(self, engine: CompositionEngine) -> None:
        """Verify we have enough generators registered."""
        assert len(engine._l0_generators) >= 10
        assert len(engine._l1_generators) >= 25   # 9 original + 18 new
        assert len(engine._l2_generators) >= 14    # 5 original + 10 new
        assert len(engine._l3_generators) >= 13    # 4 original + 10 new


class TestL0Generation:
    """L0 single-tool task generation."""

    def test_generate_l0_small(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l0_tasks(count=10)
        assert len(tasks) == 10
        for t in tasks:
            assert t.level == CompositionLevel.NODE
            assert len(t.expected_trace.steps) == 1
            assert t.expected_final_answer is not None
            assert t.prompt

    def test_l0_unique_ids(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l0_tasks(count=50)
        ids = [t.task_id for t in tasks]
        assert len(set(ids)) == len(ids), "Task IDs must be unique"

    def test_l0_has_distractors(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l0_tasks(count=10)
        for t in tasks:
            # available_tools should have the real tool + distractors
            assert len(t.available_tools) > 1
            # The expected tool must be in available_tools
            used = list(t.tools_used)[0]
            assert used in t.available_tools

    def test_l0_covers_many_tools(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l0_tasks(count=200)
        all_tools = set()
        for t in tasks:
            all_tools.update(t.tools_used)
        # With generic L0 fallback, we should cover many tools
        assert len(all_tools) > 50


class TestL1Generation:
    """L1 chain (2-tool) task generation."""

    def test_generate_l1_small(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l1_tasks(count=20)
        assert len(tasks) == 20
        for t in tasks:
            assert t.level == CompositionLevel.CHAIN
            assert len(t.expected_trace.steps) == 2
            # Step 2 must depend on step 1
            step2 = t.expected_trace.steps[1]
            assert "step_1" in step2.depends_on

    def test_l1_diverse_patterns(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l1_tasks(count=100)
        patterns = set()
        for t in tasks:
            tools = tuple(sorted(t.tools_used))
            patterns.add(tools)
        # Should use many different tool combinations
        assert len(patterns) >= 15

    def test_l1_tool_in_available(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l1_tasks(count=20)
        for t in tasks:
            for step in t.expected_trace.steps:
                assert step.tool_name in t.available_tools


class TestL2Generation:
    """L2 parallel (fork-join) task generation."""

    def test_generate_l2_small(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l2_tasks(count=15)
        assert len(tasks) == 15
        for t in tasks:
            assert t.level == CompositionLevel.PARALLEL
            assert len(t.expected_trace.steps) >= 2

    def test_l2_has_parallel_steps(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l2_tasks(count=30)
        for t in tasks:
            # At least 2 steps should have the same depends_on
            dep_groups: dict[str, list[str]] = {}
            for step in t.expected_trace.steps:
                key = ",".join(sorted(step.depends_on)) if step.depends_on else "_root"
                dep_groups.setdefault(key, []).append(step.step_id)
            parallel_groups = [g for g in dep_groups.values() if len(g) > 1]
            assert len(parallel_groups) >= 1, f"L2 task {t.task_id} has no parallel steps"


class TestL3Generation:
    """L3 DAG task generation."""

    def test_generate_l3_small(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l3_tasks(count=15)
        assert len(tasks) == 15
        for t in tasks:
            assert t.level == CompositionLevel.DAG
            assert len(t.expected_trace.steps) >= 4

    def test_l3_diverse_patterns(self, engine: CompositionEngine) -> None:
        tasks = engine.generate_l3_tasks(count=100)
        patterns = set()
        for t in tasks:
            tools = tuple(sorted(t.tools_used))
            patterns.add(tools)
        assert len(patterns) >= 8


class TestFullSuite:
    """Full suite generation."""

    def test_generate_suite(self, engine: CompositionEngine) -> None:
        suite = engine.generate_suite(l0_count=20, l1_count=20, l2_count=10, l3_count=10)
        assert isinstance(suite, TaskSuite)
        assert len(suite.tasks) == 60

    def test_suite_stats(self, engine: CompositionEngine) -> None:
        suite = engine.generate_suite(l0_count=20, l1_count=20, l2_count=10, l3_count=10)
        stats = suite.stats
        assert stats["total_tasks"] == 60
        assert stats["by_level"]["L0_node"] == 20
        assert stats["by_level"]["L1_chain"] == 20
        assert stats["by_level"]["L2_parallel"] == 10
        assert stats["by_level"]["L3_dag"] == 10
        assert stats["unique_tools"] > 10

    def test_suite_all_valid(self, engine: CompositionEngine) -> None:
        """Every task in the suite must pass basic validation."""
        suite = engine.generate_suite(l0_count=30, l1_count=30, l2_count=15, l3_count=15)
        for task in suite.tasks:
            assert task.task_id, "task_id must not be empty"
            assert task.prompt, "prompt must not be empty"
            assert task.expected_trace.steps, "trace must have steps"
            assert task.expected_final_answer is not None, "answer must not be None"
            assert task.available_tools, "available_tools must not be empty"
            for step in task.expected_trace.steps:
                assert step.tool_name in task.available_tools, (
                    f"Step uses {step.tool_name} not in available_tools for {task.task_id}"
                )

    def test_suite_unique_ids(self, engine: CompositionEngine) -> None:
        suite = engine.generate_suite(l0_count=50, l1_count=50, l2_count=25, l3_count=25)
        ids = [t.task_id for t in suite.tasks]
        assert len(set(ids)) == len(ids), "All task IDs must be unique"

    def test_deterministic_with_seed(self) -> None:
        """Same seed should produce identical suites."""
        e1 = CompositionEngine(seed=123)
        e2 = CompositionEngine(seed=123)
        s1 = e1.generate_suite(l0_count=10, l1_count=10, l2_count=5, l3_count=5)
        s2 = e2.generate_suite(l0_count=10, l1_count=10, l2_count=5, l3_count=5)
        for t1, t2 in zip(s1.tasks, s2.tasks):
            assert t1.task_id == t2.task_id
            assert t1.prompt == t2.prompt

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different suites."""
        e1 = CompositionEngine(seed=42)
        e2 = CompositionEngine(seed=99)
        s1 = e1.generate_suite(l0_count=10, l1_count=10, l2_count=5, l3_count=5)
        s2 = e2.generate_suite(l0_count=10, l1_count=10, l2_count=5, l3_count=5)
        # Not all prompts should match
        same_prompts = sum(1 for t1, t2 in zip(s1.tasks, s2.tasks) if t1.prompt == t2.prompt)
        assert same_prompts < len(s1.tasks)


class TestSchemaValidation:
    """Test that tool schemas are valid."""

    def test_all_schemas_have_items_for_arrays(self) -> None:
        """Gemini requires 'items' for array-type params."""
        from comptoolbench.tools import get_all_tools

        for name, cls in get_all_tools().items():
            tool = cls(mode=ToolMode.SIMULATED)
            schema = tool.get_openai_schema()
            props = schema["function"]["parameters"]["properties"]
            for pname, pspec in props.items():
                if pspec.get("type") == "array":
                    assert "items" in pspec, (
                        f"Tool {name} param {pname} is array but missing 'items'"
                    )
