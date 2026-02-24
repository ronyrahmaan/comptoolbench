"""File and data tools: read, write, list, transform, merge.

These tools simulate a virtual file system for the benchmark.
All operations act on an in-memory dict, not the real filesystem.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, ClassVar

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

# Simulated virtual file system
_VIRTUAL_FS: dict[str, dict[str, Any]] = {}

# Pre-seeded files for reproducibility
_SEED_FILES: dict[str, dict[str, Any]] = {
    "/data/employees.csv": {
        "content": "name,department,salary,years\nAlice,Engineering,95000,5\nBob,Marketing,72000,3\nCharlie,Engineering,105000,8\nDiana,Sales,68000,2\nEve,Engineering,88000,4\nFrank,Marketing,76000,6\nGrace,Sales,71000,3\nHenry,Engineering,112000,10",
        "type": "text/csv",
        "size": 284,
        "created_at": "2026-03-01T10:00:00",
    },
    "/data/config.json": {
        "content": '{"app_name": "DataPipeline", "version": "2.1.0", "max_workers": 4, "timeout_seconds": 30, "features": {"caching": true, "logging": true, "metrics": false}}',
        "type": "application/json",
        "size": 156,
        "created_at": "2026-03-10T08:00:00",
    },
    "/data/report.txt": {
        "content": "Q1 2026 Sales Report\n\nTotal revenue: $2.4M\nGrowth: 15% YoY\nTop product: Enterprise Plan ($1.2M)\nNew customers: 340\nChurn rate: 2.1%\n\nKey highlights:\n- Enterprise segment grew 28%\n- APAC region exceeded targets by 12%\n- Customer satisfaction score: 4.6/5.0",
        "type": "text/plain",
        "size": 267,
        "created_at": "2026-03-14T16:00:00",
    },
    "/data/products.json": {
        "content": '[{"id": 1, "name": "Basic Plan", "price": 29, "category": "subscription"}, {"id": 2, "name": "Pro Plan", "price": 79, "category": "subscription"}, {"id": 3, "name": "Enterprise Plan", "price": 199, "category": "subscription"}, {"id": 4, "name": "API Access", "price": 49, "category": "addon"}, {"id": 5, "name": "Premium Support", "price": 99, "category": "addon"}]',
        "type": "application/json",
        "size": 342,
        "created_at": "2026-03-12T09:00:00",
    },
    "/data/notes.txt": {
        "content": "Meeting notes - March 14, 2026\n\nAttendees: Alice, Bob, Charlie\n\nAction items:\n1. Alice: Complete API documentation by March 20\n2. Bob: Run performance benchmarks on new pipeline\n3. Charlie: Review security audit findings\n\nNext meeting: March 21, 2026 at 2:00 PM",
        "type": "text/plain",
        "size": 245,
        "created_at": "2026-03-14T15:00:00",
    },
}


def reset_virtual_fs() -> None:
    """Reset the virtual file system (call between evaluation sessions)."""
    _VIRTUAL_FS.clear()
    _VIRTUAL_FS.update({k: v.copy() for k, v in _SEED_FILES.items()})


# Initialize with seed data
reset_virtual_fs()


@register_tool
class ReadFile(BaseTool):
    """Read a file from the virtual file system."""

    name = "read_file"
    schema = ToolSchema(
        name="read_file",
        description="Read the contents of a file from the file system. Returns the file content as a string, along with metadata like file size and type.",
        category=ToolCategory.FILE_DATA,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="The file path to read (e.g. '/data/report.txt', '/data/config.json')",
            ),
        ],
        returns="File content and metadata",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._read(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._read(**kwargs)

    def _read(self, **kwargs: Any) -> dict[str, Any]:
        path = kwargs["path"]
        if path in _VIRTUAL_FS:
            entry = _VIRTUAL_FS[path]
            return {
                "path": path,
                "content": entry["content"],
                "type": entry["type"],
                "size": entry["size"],
                "exists": True,
            }
        return {
            "path": path,
            "content": None,
            "exists": False,
            "error": f"File not found: {path}",
        }


@register_tool
class WriteFile(BaseTool):
    """Write content to a file in the virtual file system."""

    name = "write_file"
    schema = ToolSchema(
        name="write_file",
        description="Write content to a file. Creates the file if it doesn't exist, or overwrites if it does.",
        category=ToolCategory.FILE_DATA,
        parameters=[
            ToolParameter(
                name="path",
                type="string",
                description="The file path to write to (e.g. '/data/output.txt')",
            ),
            ToolParameter(
                name="content",
                type="string",
                description="The content to write to the file",
            ),
        ],
        returns="Confirmation with file path and size",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._write(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._write(**kwargs)

    def _write(self, **kwargs: Any) -> dict[str, Any]:
        path = kwargs["path"]
        content = kwargs["content"]
        overwrote = path in _VIRTUAL_FS

        # Infer type from extension
        ext_map = {
            ".json": "application/json",
            ".csv": "text/csv",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".xml": "application/xml",
            ".html": "text/html",
        }
        file_type = "text/plain"
        for ext, mime in ext_map.items():
            if path.endswith(ext):
                file_type = mime
                break

        _VIRTUAL_FS[path] = {
            "content": content,
            "type": file_type,
            "size": len(content),
            "created_at": "2026-02-22T12:00:00",
        }

        return {
            "status": "written",
            "path": path,
            "size": len(content),
            "type": file_type,
            "overwrote_existing": overwrote,
        }


@register_tool
class ListFiles(BaseTool):
    """List files in the virtual file system."""

    name = "list_files"
    schema = ToolSchema(
        name="list_files",
        description="List files in a directory or the entire virtual file system. Returns file names, sizes, and types.",
        category=ToolCategory.FILE_DATA,
        parameters=[
            ToolParameter(
                name="directory",
                type="string",
                description="Directory path to list (e.g. '/data'). Use '/' for root.",
                required=False,
                default="/",
            ),
        ],
        returns="List of files with their metadata",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._list(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._list(**kwargs)

    def _list(self, **kwargs: Any) -> dict[str, Any]:
        directory = kwargs.get("directory", "/")
        if not directory.endswith("/"):
            directory += "/"

        files = []
        for path, entry in sorted(_VIRTUAL_FS.items()):
            if directory == "/" or path.startswith(directory):
                files.append({
                    "path": path,
                    "name": path.split("/")[-1],
                    "type": entry["type"],
                    "size": entry["size"],
                    "created_at": entry.get("created_at", ""),
                })

        return {
            "directory": directory,
            "files": files,
            "count": len(files),
        }


@register_tool
class TransformFormat(BaseTool):
    """Transform data between formats (JSON, CSV, text)."""

    name = "transform_format"
    schema = ToolSchema(
        name="transform_format",
        description="Transform data from one format to another. Supports conversions between JSON, CSV, and plain text formats.",
        category=ToolCategory.FILE_DATA,
        parameters=[
            ToolParameter(
                name="data",
                type="string",
                description="The data to transform (as a string)",
            ),
            ToolParameter(
                name="from_format",
                type="string",
                description="Source format",
                enum=["json", "csv", "text"],
            ),
            ToolParameter(
                name="to_format",
                type="string",
                description="Target format",
                enum=["json", "csv", "text"],
            ),
        ],
        returns="The transformed data in the target format",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._transform(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._transform(**kwargs)

    def _transform(self, **kwargs: Any) -> dict[str, Any]:
        data_str = kwargs["data"]
        from_fmt = kwargs["from_format"]
        to_fmt = kwargs["to_format"]

        # Parse input
        parsed: Any = None
        if from_fmt == "json":
            parsed = json.loads(data_str)
        elif from_fmt == "csv":
            reader = csv.DictReader(io.StringIO(data_str))
            parsed = list(reader)
        else:  # text
            parsed = data_str

        # Convert to output
        if to_fmt == "json":
            result = json.dumps(parsed, indent=2)
        elif to_fmt == "csv":
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=parsed[0].keys())
                writer.writeheader()
                writer.writerows(parsed)
                result = output.getvalue()
            else:
                result = str(parsed)
        else:  # text
            if isinstance(parsed, list):
                result = "\n".join(
                    ", ".join(f"{k}: {v}" for k, v in row.items())
                    if isinstance(row, dict)
                    else str(row)
                    for row in parsed
                )
            elif isinstance(parsed, dict):
                result = "\n".join(f"{k}: {v}" for k, v in parsed.items())
            else:
                result = str(parsed)

        return {
            "from_format": from_fmt,
            "to_format": to_fmt,
            "original_length": len(data_str),
            "transformed": result,
            "transformed_length": len(result),
        }


@register_tool
class MergeData(BaseTool):
    """Merge two datasets together."""

    name = "merge_data"
    schema = ToolSchema(
        name="merge_data",
        description="Merge two datasets (lists of records) based on a common key field. Similar to a SQL JOIN operation.",
        category=ToolCategory.FILE_DATA,
        parameters=[
            ToolParameter(
                name="dataset1",
                type="string",
                description="First dataset as a JSON array of objects",
            ),
            ToolParameter(
                name="dataset2",
                type="string",
                description="Second dataset as a JSON array of objects",
            ),
            ToolParameter(
                name="join_key",
                type="string",
                description="The field name to join on (must exist in both datasets)",
            ),
        ],
        returns="The merged dataset",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._merge(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._merge(**kwargs)

    def _merge(self, **kwargs: Any) -> dict[str, Any]:
        ds1 = json.loads(kwargs["dataset1"]) if isinstance(kwargs["dataset1"], str) else kwargs["dataset1"]
        ds2 = json.loads(kwargs["dataset2"]) if isinstance(kwargs["dataset2"], str) else kwargs["dataset2"]
        join_key = kwargs["join_key"]

        # Build lookup from dataset2
        lookup: dict[str, dict[str, Any]] = {}
        for row in ds2:
            if join_key in row:
                lookup[str(row[join_key])] = row

        # Merge
        merged = []
        matched = 0
        for row in ds1:
            key_val = str(row.get(join_key, ""))
            if key_val in lookup:
                merged_row = {**row, **lookup[key_val]}
                merged.append(merged_row)
                matched += 1
            else:
                merged.append(row)

        return {
            "merged": merged,
            "total_records": len(merged),
            "matched_records": matched,
            "unmatched_records": len(merged) - matched,
            "join_key": join_key,
        }
