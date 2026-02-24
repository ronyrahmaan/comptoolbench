"""State management tools: memory store, retrieval, listing, session context.

These tools simulate a key-value memory store that persists within
a benchmark session. Both live and simulated modes use the same
in-memory dict, making them fully deterministic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, ClassVar

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

# Shared in-memory store (reset per evaluation session)
_MEMORY_STORE: dict[str, dict[str, Any]] = {}

# Pre-seeded memories for simulated mode
_SEED_MEMORIES: dict[str, dict[str, Any]] = {
    "user_preferences": {
        "key": "user_preferences",
        "value": {"theme": "dark", "language": "en", "notifications": True},
        "tags": ["settings", "user"],
        "created_at": "2026-03-15T10:00:00",
    },
    "project_notes": {
        "key": "project_notes",
        "value": "Current project involves building a benchmark for evaluating compositional tool-use in LLMs.",
        "tags": ["project", "research"],
        "created_at": "2026-03-15T09:00:00",
    },
    "meeting_agenda": {
        "key": "meeting_agenda",
        "value": ["Review Q1 metrics", "Discuss roadmap", "Assign action items"],
        "tags": ["work", "meeting"],
        "created_at": "2026-03-15T08:30:00",
    },
    "shopping_list": {
        "key": "shopping_list",
        "value": ["milk", "eggs", "bread", "coffee", "oranges"],
        "tags": ["personal", "shopping"],
        "created_at": "2026-03-14T18:00:00",
    },
    "api_config": {
        "key": "api_config",
        "value": {"endpoint": "https://api.example.com/v2", "timeout": 30, "retries": 3},
        "tags": ["config", "technical"],
        "created_at": "2026-03-14T12:00:00",
    },
}


def reset_memory_store() -> None:
    """Reset the memory store (call between evaluation sessions)."""
    _MEMORY_STORE.clear()
    _MEMORY_STORE.update({k: v.copy() for k, v in _SEED_MEMORIES.items()})


# Initialize with seed data
reset_memory_store()


@register_tool
class StoreMemory(BaseTool):
    """Store a key-value pair in the session memory."""

    name = "store_memory"
    schema = ToolSchema(
        name="store_memory",
        description="Store a key-value pair in the session memory. Can store strings, numbers, lists, or objects. Overwrites if key already exists.",
        category=ToolCategory.STATE_MANAGEMENT,
        parameters=[
            ToolParameter(
                name="key",
                type="string",
                description="The key to store the value under (e.g. 'user_name', 'last_result')",
            ),
            ToolParameter(
                name="value",
                type="string",
                description="The value to store (can be a string, number, JSON object, or JSON array)",
            ),
            ToolParameter(
                name="tags",
                type="array",
                description="Optional tags for organizing memories",
                required=False,
            ),
        ],
        returns="Confirmation of stored memory with key and timestamp",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._store(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._store(**kwargs)

    def _store(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["key"]
        value = kwargs["value"]
        tags = kwargs.get("tags", [])

        # Try to parse JSON values
        parsed_value: Any = value
        if isinstance(value, str):
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_value = value

        entry = {
            "key": key,
            "value": parsed_value,
            "tags": tags or [],
            "created_at": "2026-02-22T12:00:00",
        }
        overwrote = key in _MEMORY_STORE
        _MEMORY_STORE[key] = entry

        return {
            "status": "stored",
            "key": key,
            "overwrote_existing": overwrote,
            "tags": entry["tags"],
            "timestamp": entry["created_at"],
        }


@register_tool
class RetrieveMemory(BaseTool):
    """Retrieve a value from the session memory by key."""

    name = "retrieve_memory"
    schema = ToolSchema(
        name="retrieve_memory",
        description="Retrieve a previously stored value from the session memory by its key.",
        category=ToolCategory.STATE_MANAGEMENT,
        parameters=[
            ToolParameter(
                name="key",
                type="string",
                description="The key to retrieve (e.g. 'user_name', 'last_result')",
            ),
        ],
        returns="The stored value, or an error if the key is not found",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._retrieve(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._retrieve(**kwargs)

    def _retrieve(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["key"]
        if key in _MEMORY_STORE:
            entry = _MEMORY_STORE[key]
            return {
                "status": "found",
                "key": key,
                "value": entry["value"],
                "tags": entry.get("tags", []),
                "created_at": entry.get("created_at", ""),
            }
        return {
            "status": "not_found",
            "key": key,
            "value": None,
            "error": f"No memory found for key '{key}'",
        }


@register_tool
class ListMemories(BaseTool):
    """List all keys in the session memory."""

    name = "list_memories"
    schema = ToolSchema(
        name="list_memories",
        description="List all stored memory keys, optionally filtered by tag. Returns keys, tags, and creation timestamps.",
        category=ToolCategory.STATE_MANAGEMENT,
        parameters=[
            ToolParameter(
                name="tag",
                type="string",
                description="Optional tag to filter memories by",
                required=False,
            ),
        ],
        returns="List of memory keys with their tags and timestamps",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._list(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._list(**kwargs)

    def _list(self, **kwargs: Any) -> dict[str, Any]:
        tag_filter = kwargs.get("tag")
        entries = []

        for key, entry in _MEMORY_STORE.items():
            if tag_filter and tag_filter not in entry.get("tags", []):
                continue
            entries.append({
                "key": key,
                "tags": entry.get("tags", []),
                "created_at": entry.get("created_at", ""),
                "value_type": type(entry["value"]).__name__,
            })

        return {
            "memories": entries,
            "count": len(entries),
            "filter_tag": tag_filter,
        }


@register_tool
class GetSessionContext(BaseTool):
    """Get the current session context and metadata."""

    name = "get_session_context"
    schema = ToolSchema(
        name="get_session_context",
        description="Get information about the current session, including session ID, start time, stored memory count, and available tools summary.",
        category=ToolCategory.STATE_MANAGEMENT,
        parameters=[],
        returns="Session context information",
        returns_type="object",
    )

    # Fixed session data for determinism
    _SESSION_DATA: ClassVar[dict[str, Any]] = {
        "session_id": "sess_comptoolbench_v1",
        "started_at": "2026-03-15T14:30:00Z",
        "user": "benchmark_evaluator",
        "environment": "comptoolbench_v0.1.0",
    }

    def execute_live(self, **kwargs: Any) -> Any:
        return self._get_context()

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._get_context()

    def _get_context(self) -> dict[str, Any]:
        return {
            **self._SESSION_DATA,
            "memory_count": len(_MEMORY_STORE),
            "memory_keys": list(_MEMORY_STORE.keys()),
            "tags_used": sorted({
                tag
                for entry in _MEMORY_STORE.values()
                for tag in entry.get("tags", [])
            }),
        }
