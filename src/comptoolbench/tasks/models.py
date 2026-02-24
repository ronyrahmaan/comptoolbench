"""Data models for benchmark tasks and composition graphs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CompositionLevel(str, Enum):
    """The four levels of tool composition complexity."""

    NODE = "L0_node"         # Single tool call
    CHAIN = "L1_chain"       # Sequential: A → B → C
    PARALLEL = "L2_parallel" # Concurrent: A | B → C
    DAG = "L3_dag"           # Branching + merging


class ToolCall(BaseModel):
    """A single expected tool call in a task's solution."""

    step_id: str = Field(description="Unique step identifier, e.g. 'step_1'")
    tool_name: str = Field(description="Name of the tool to call")
    arguments: dict[str, Any] = Field(description="Expected arguments")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step IDs this call depends on (for ordering/data flow)",
    )
    output_key: str | None = Field(
        default=None,
        description="Key to reference this step's output in later steps",
    )


class ExpectedTrace(BaseModel):
    """The complete expected execution trace for a task."""

    steps: list[ToolCall]
    final_answer_source: str = Field(
        description="Which step's output forms the final answer",
    )

    @property
    def tool_sequence(self) -> list[str]:
        """Return the ordered list of tool names."""
        return [s.tool_name for s in self.steps]

    @property
    def num_tools(self) -> int:
        return len(self.steps)

    @property
    def unique_tools(self) -> set[str]:
        return {s.tool_name for s in self.steps}


class Task(BaseModel):
    """A single benchmark task."""

    task_id: str = Field(description="Unique task identifier")
    level: CompositionLevel
    prompt: str = Field(description="Natural language instruction for the model")
    available_tools: list[str] = Field(
        description="Names of tools available to the model for this task",
    )
    expected_trace: ExpectedTrace
    expected_final_answer: Any = Field(
        description="The correct final answer (for verification)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (category, difficulty, tags, etc.)",
    )

    @property
    def num_steps(self) -> int:
        return self.expected_trace.num_tools

    @property
    def tools_used(self) -> set[str]:
        return self.expected_trace.unique_tools

    @property
    def has_parallel_steps(self) -> bool:
        """Check if any steps share the same dependencies (can run in parallel)."""
        dep_groups: dict[str, list[str]] = {}
        for step in self.expected_trace.steps:
            key = ",".join(sorted(step.depends_on)) if step.depends_on else "_root"
            dep_groups.setdefault(key, []).append(step.step_id)
        return any(len(group) > 1 for group in dep_groups.values())


class TaskSuite(BaseModel):
    """A collection of tasks forming a benchmark suite."""

    name: str = "CompToolBench"
    version: str = "0.1.0"
    tasks: list[Task] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def by_level(self) -> dict[CompositionLevel, list[Task]]:
        """Group tasks by composition level."""
        groups: dict[CompositionLevel, list[Task]] = {}
        for task in self.tasks:
            groups.setdefault(task.level, []).append(task)
        return groups

    @property
    def stats(self) -> dict[str, Any]:
        """Summary statistics for the suite."""
        by_level = self.by_level
        return {
            "total_tasks": len(self.tasks),
            "by_level": {
                level.value: len(tasks)
                for level, tasks in sorted(by_level.items(), key=lambda x: x[0].value)
            },
            "unique_tools": len({t for task in self.tasks for t in task.tools_used}),
            "avg_steps_per_task": (
                sum(task.num_steps for task in self.tasks) / max(len(self.tasks), 1)
            ),
        }
