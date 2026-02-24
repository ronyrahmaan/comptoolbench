"""Tests for base tool infrastructure and registry."""

from __future__ import annotations

from comptoolbench.tools import get_all_tools, get_tool
from comptoolbench.tools.base import BaseTool, ToolCategory, ToolMode, ToolResult


class TestRegistry:
    """Test the global tool registry."""

    def test_registry_not_empty(self) -> None:
        tools = get_all_tools()
        assert len(tools) >= 36, f"Expected at least 36 tools, got {len(tools)}"

    def test_all_categories_covered(self) -> None:
        tools = get_all_tools()
        categories = {cls(mode=ToolMode.SIMULATED).schema.category for cls in tools.values()}
        for cat in ToolCategory:
            assert cat in categories, f"Category {cat.value} has no tools"

    def test_get_tool_known(self) -> None:
        cls = get_tool("calculator")
        assert cls.name == "calculator"

    def test_get_tool_unknown_raises(self) -> None:
        import pytest
        with pytest.raises(KeyError, match="not_a_real_tool"):
            get_tool("not_a_real_tool")

    def test_all_tools_have_unique_names(self) -> None:
        tools = get_all_tools()
        names = list(tools.keys())
        assert len(names) == len(set(names)), "Duplicate tool names found"

    def test_all_tools_have_schemas(self) -> None:
        for name, cls in get_all_tools().items():
            tool = cls(mode=ToolMode.SIMULATED)
            schema = tool.schema
            assert schema.name == name, f"{name}: schema name mismatch"
            assert schema.description, f"{name}: missing description"
            assert schema.category in ToolCategory, f"{name}: invalid category"

    def test_all_tools_produce_openai_schema(self) -> None:
        for name, cls in get_all_tools().items():
            tool = cls(mode=ToolMode.SIMULATED)
            oai = tool.get_openai_schema()
            assert oai["type"] == "function"
            assert oai["function"]["name"] == name
            assert "parameters" in oai["function"]


class TestToolResult:
    """Test ToolResult model."""

    def test_success_result(self) -> None:
        r = ToolResult(tool_name="test", success=True, data={"key": "value"})
        assert r.success
        assert '"key"' in r.to_message()

    def test_error_result(self) -> None:
        r = ToolResult(tool_name="test", success=False, error="Something went wrong")
        assert not r.success
        assert "Something went wrong" in r.to_message()


class TestBaseTool:
    """Test BaseTool execution dispatch."""

    def test_simulated_mode(self) -> None:
        cls = get_tool("calculator")
        tool = cls(mode=ToolMode.SIMULATED)
        result = tool.execute(expression="1 + 1")
        assert result.success
        assert result.data["result"] == 2.0

    def test_callable(self) -> None:
        cls = get_tool("calculator")
        tool = cls(mode=ToolMode.SIMULATED)
        result = tool(expression="5 * 3")
        assert result.success
        assert result.data["result"] == 15.0

    def test_error_handling(self) -> None:
        cls = get_tool("calculator")
        tool = cls(mode=ToolMode.SIMULATED)
        result = tool.execute(expression="1 / 0")
        assert not result.success
        assert result.error is not None
