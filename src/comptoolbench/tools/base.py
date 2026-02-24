"""Base tool class and registry for CompToolBench.

Every tool in the benchmark extends BaseTool and registers itself
via the @register_tool decorator. Tools can operate in two modes:
- "live": Calls real external APIs (weather, search, etc.)
- "simulated": Returns deterministic outputs for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Global tool registry
# ---------------------------------------------------------------------------
_TOOL_REGISTRY: dict[str, type[BaseTool]] = {}


def register_tool(cls: type[BaseTool]) -> type[BaseTool]:
    """Class decorator that registers a tool in the global registry."""
    _TOOL_REGISTRY[cls.name] = cls
    return cls


def get_tool(name: str) -> type[BaseTool]:
    """Look up a registered tool by name."""
    if name not in _TOOL_REGISTRY:
        raise KeyError(
            f"Tool '{name}' not found. Available: {list(_TOOL_REGISTRY)}"
        )
    return _TOOL_REGISTRY[name]


def get_all_tools() -> dict[str, type[BaseTool]]:
    """Return a copy of the full tool registry."""
    return dict(_TOOL_REGISTRY)


# ---------------------------------------------------------------------------
# Enums & schemas
# ---------------------------------------------------------------------------


class ToolCategory(str, Enum):
    """Categories of tools in the benchmark."""

    INFORMATION_RETRIEVAL = "information_retrieval"
    COMPUTATION = "computation"
    COMMUNICATION = "communication"
    FILE_DATA = "file_data"
    EXTERNAL_SERVICES = "external_services"
    STATE_MANAGEMENT = "state_management"
    TEXT_PROCESSING = "text_processing"
    TIME_SCHEDULING = "time_scheduling"
    MEDIA = "media"


class ToolMode(str, Enum):
    """Execution mode for tools."""

    LIVE = "live"
    SIMULATED = "simulated"


class ToolParameter(BaseModel):
    """Schema for a single tool parameter."""

    name: str
    type: str  # "string", "number", "integer", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None
    items: dict[str, Any] | None = None  # For array types: {type: "string"} or nested


class ToolSchema(BaseModel):
    """Full schema for a tool, compatible with OpenAI function calling format."""

    name: str
    description: str
    category: ToolCategory
    parameters: list[ToolParameter]
    returns: str = Field(description="Description of what the tool returns")
    returns_type: str = "object"  # JSON type of the return value

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            # Array parameters need "items" (Gemini/Groq validate strictly)
            if param.type == "array":
                if param.items:
                    prop["items"] = param.items
                elif "items" not in prop:
                    prop["items"] = {"type": "string"}
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolResult(BaseModel):
    """Result returned by a tool execution."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None

    def to_message(self) -> str:
        """Format as a string suitable for feeding back to an LLM."""
        if self.success:
            return json.dumps(self.data, indent=2, default=str)
        return f"Error: {self.error}"


# ---------------------------------------------------------------------------
# Base tool class
# ---------------------------------------------------------------------------


class BaseTool(ABC):
    """Abstract base for all CompToolBench tools.

    Subclasses must define:
    - `name`: Unique string identifier.
    - `schema`: A ToolSchema describing the tool.
    - `execute_live()`: Real API implementation.
    - `execute_simulated()`: Deterministic simulation.
    """

    name: str = ""
    schema: ToolSchema

    def __init__(self, mode: ToolMode = ToolMode.LIVE) -> None:
        self.mode = mode

    def __call__(self, **kwargs: Any) -> ToolResult:
        """Run the tool with the given arguments."""
        return self.execute(**kwargs)

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool in the configured mode."""
        try:
            if self.mode == ToolMode.LIVE:
                data = self.execute_live(**kwargs)
            else:
                data = self.execute_simulated(**kwargs)
            return ToolResult(tool_name=self.name, success=True, data=data)
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False, error=str(e)
            )

    @abstractmethod
    def execute_live(self, **kwargs: Any) -> Any:
        """Execute using real external APIs."""

    @abstractmethod
    def execute_simulated(self, **kwargs: Any) -> Any:
        """Execute with deterministic simulated output."""

    @staticmethod
    def _deterministic_hash(seed: str, **kwargs: Any) -> str:
        """Generate a deterministic hash from seed + arguments.

        Useful for simulated tools that need consistent outputs.
        """
        raw = json.dumps({"seed": seed, **kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def get_openai_schema(self) -> dict[str, Any]:
        """Get the OpenAI function calling schema for this tool."""
        return self.schema.to_openai_format()
