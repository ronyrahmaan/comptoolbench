"""Tests for the task generator."""

from __future__ import annotations

from comptoolbench.generators.task_generator import TaskGenerator
from comptoolbench.tasks.models import CompositionLevel
from comptoolbench.tools.base import ToolMode


class TestTaskGenerator:
    def setup_method(self) -> None:
        self.gen = TaskGenerator(seed=42, mode=ToolMode.SIMULATED)

    def test_l0_generation(self) -> None:
        tasks = self.gen.generate_l0_tasks(count=10)
        assert len(tasks) == 10
        assert all(t.level == CompositionLevel.NODE for t in tasks)
        assert all(len(t.expected_trace.steps) == 1 for t in tasks)

    def test_l1_generation(self) -> None:
        tasks = self.gen.generate_l1_tasks(count=7)
        assert len(tasks) == 7
        assert all(t.level == CompositionLevel.CHAIN for t in tasks)
        assert all(len(t.expected_trace.steps) >= 2 for t in tasks)

    def test_l2_generation(self) -> None:
        tasks = self.gen.generate_l2_tasks(count=6)
        assert len(tasks) == 6
        assert all(t.level == CompositionLevel.PARALLEL for t in tasks)

    def test_l3_generation(self) -> None:
        tasks = self.gen.generate_l3_tasks(count=4)
        assert len(tasks) == 4
        assert all(t.level == CompositionLevel.DAG for t in tasks)
        # DAG tasks should have dependencies
        for task in tasks:
            has_deps = any(s.depends_on for s in task.expected_trace.steps)
            assert has_deps, f"DAG task {task.task_id} has no dependencies"

    def test_full_suite(self) -> None:
        suite = self.gen.generate_suite(l0_count=10, l1_count=7, l2_count=6, l3_count=4)
        assert suite.name == "CompToolBench"
        assert len(suite.tasks) == 27
        by_level = suite.by_level
        assert len(by_level[CompositionLevel.NODE]) == 10
        assert len(by_level[CompositionLevel.CHAIN]) == 7

    def test_deterministic_with_same_seed(self) -> None:
        gen1 = TaskGenerator(seed=123, mode=ToolMode.SIMULATED)
        gen2 = TaskGenerator(seed=123, mode=ToolMode.SIMULATED)
        tasks1 = gen1.generate_l0_tasks(count=5)
        tasks2 = gen2.generate_l0_tasks(count=5)
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.prompt == t2.prompt
            assert t1.expected_final_answer == t2.expected_final_answer

    def test_different_seeds_differ(self) -> None:
        gen1 = TaskGenerator(seed=1, mode=ToolMode.SIMULATED)
        gen2 = TaskGenerator(seed=999, mode=ToolMode.SIMULATED)
        tasks1 = gen1.generate_l0_tasks(count=5)
        tasks2 = gen2.generate_l0_tasks(count=5)
        # At least some prompts should differ
        prompts1 = {t.prompt for t in tasks1}
        prompts2 = {t.prompt for t in tasks2}
        assert prompts1 != prompts2

    def test_all_task_ids_unique(self) -> None:
        suite = self.gen.generate_suite(l0_count=20, l1_count=14, l2_count=9, l3_count=4)
        ids = [t.task_id for t in suite.tasks]
        assert len(ids) == len(set(ids)), "Duplicate task IDs found"

    def test_available_tools_are_strings(self) -> None:
        tasks = self.gen.generate_l0_tasks(count=5)
        for task in tasks:
            assert all(isinstance(t, str) for t in task.available_tools)
