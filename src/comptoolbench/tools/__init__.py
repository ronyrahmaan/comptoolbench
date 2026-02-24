"""CompToolBench tool registry.

Import this module to auto-register all tools.
"""

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolMode,
    ToolResult,
    ToolSchema,
    get_all_tools,
    get_tool,
)

# Import all tool modules to trigger @register_tool decorators
from comptoolbench.tools import (  # noqa: F401
    communication,
    computation,
    datetime_tools,
    external_services,
    file_data,
    information_retrieval,
    math_tools,
    media,
    nlp_tools,
    productivity,
    state_management,
    text_processing,
    time_scheduling,
    utility,
    web_tools,
)

__all__ = [
    "BaseTool",
    "ToolCategory",
    "ToolMode",
    "ToolResult",
    "ToolSchema",
    "get_all_tools",
    "get_tool",
]
