"""Computation tools: calculator, unit converter, data operations.

These tools are fully local — no API keys needed.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)


@register_tool
class Calculator(BaseTool):
    """Evaluate mathematical expressions safely."""

    name = "calculator"
    schema = ToolSchema(
        name="calculator",
        description="Evaluate a mathematical expression and return the result. Supports basic arithmetic, powers, square roots, and common math functions.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="The mathematical expression to evaluate, e.g. '(45 * 3) + 17' or 'sqrt(144)'",
            ),
        ],
        returns="The numerical result of the expression",
        returns_type="object",
    )

    # Safe math functions available in expressions
    _SAFE_GLOBALS: dict[str, Any] = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
    }

    def execute_live(self, **kwargs: Any) -> Any:
        expression: str = kwargs["expression"]
        result = eval(expression, self._SAFE_GLOBALS)  # noqa: S307
        return {"expression": expression, "result": round(float(result), 6)}

    def execute_simulated(self, **kwargs: Any) -> Any:
        # Calculator is deterministic — same in both modes
        return self.execute_live(**kwargs)


@register_tool
class UnitConvert(BaseTool):
    """Convert between common units."""

    name = "unit_convert"
    schema = ToolSchema(
        name="unit_convert",
        description="Convert a value from one unit to another. Supports temperature (celsius/fahrenheit/kelvin), distance (km/miles/meters/feet), weight (kg/lbs/grams/ounces), and currency-related units.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="value", type="number", description="The value to convert"
            ),
            ToolParameter(
                name="from_unit",
                type="string",
                description="The source unit (e.g. 'celsius', 'km', 'kg')",
            ),
            ToolParameter(
                name="to_unit",
                type="string",
                description="The target unit (e.g. 'fahrenheit', 'miles', 'lbs')",
            ),
        ],
        returns="The converted value with units",
        returns_type="object",
    )

    _CONVERSIONS: dict[tuple[str, str], Any] = {
        # Temperature
        ("celsius", "fahrenheit"): lambda v: v * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5 / 9,
        ("celsius", "kelvin"): lambda v: v + 273.15,
        ("kelvin", "celsius"): lambda v: v - 273.15,
        ("fahrenheit", "kelvin"): lambda v: (v - 32) * 5 / 9 + 273.15,
        ("kelvin", "fahrenheit"): lambda v: (v - 273.15) * 9 / 5 + 32,
        # Distance
        ("km", "miles"): lambda v: v * 0.621371,
        ("miles", "km"): lambda v: v * 1.60934,
        ("meters", "feet"): lambda v: v * 3.28084,
        ("feet", "meters"): lambda v: v * 0.3048,
        ("km", "meters"): lambda v: v * 1000,
        ("meters", "km"): lambda v: v / 1000,
        ("miles", "feet"): lambda v: v * 5280,
        ("feet", "miles"): lambda v: v / 5280,
        # Weight
        ("kg", "lbs"): lambda v: v * 2.20462,
        ("lbs", "kg"): lambda v: v * 0.453592,
        ("kg", "grams"): lambda v: v * 1000,
        ("grams", "kg"): lambda v: v / 1000,
        ("lbs", "ounces"): lambda v: v * 16,
        ("ounces", "lbs"): lambda v: v / 16,
    }

    def execute_live(self, **kwargs: Any) -> Any:
        value = float(kwargs["value"])
        from_unit = kwargs["from_unit"].lower().strip()
        to_unit = kwargs["to_unit"].lower().strip()

        if from_unit == to_unit:
            return {
                "original_value": value,
                "from_unit": from_unit,
                "converted_value": value,
                "to_unit": to_unit,
            }

        key = (from_unit, to_unit)
        if key not in self._CONVERSIONS:
            raise ValueError(
                f"Unsupported conversion: {from_unit} → {to_unit}"
            )

        result = self._CONVERSIONS[key](value)
        return {
            "original_value": value,
            "from_unit": from_unit,
            "converted_value": round(result, 4),
            "to_unit": to_unit,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        # Unit conversion is deterministic — same in both modes
        return self.execute_live(**kwargs)


@register_tool
class StatisticalAnalysis(BaseTool):
    """Compute statistics on a list of numbers."""

    name = "statistical_analysis"
    schema = ToolSchema(
        name="statistical_analysis",
        description="Compute statistical measures (mean, median, std dev, min, max, etc.) for a list of numbers.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="numbers",
                type="array",
                description="List of numbers to analyze",
                items={"type": "number"},
            ),
            ToolParameter(
                name="operation",
                type="string",
                description="The statistical operation to perform",
                enum=["summary", "mean", "median", "stdev", "min", "max", "sum", "count"],
                default="summary",
                required=False,
            ),
        ],
        returns="Statistical result(s)",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        numbers = [float(n) for n in kwargs["numbers"]]
        operation = kwargs.get("operation", "summary")

        if not numbers:
            raise ValueError("Cannot compute statistics on empty list")

        if operation == "mean":
            return {"operation": "mean", "result": round(statistics.mean(numbers), 4)}
        elif operation == "median":
            return {"operation": "median", "result": round(statistics.median(numbers), 4)}
        elif operation == "stdev":
            if len(numbers) < 2:
                raise ValueError("Need at least 2 numbers for standard deviation")
            return {"operation": "stdev", "result": round(statistics.stdev(numbers), 4)}
        elif operation == "min":
            return {"operation": "min", "result": min(numbers)}
        elif operation == "max":
            return {"operation": "max", "result": max(numbers)}
        elif operation == "sum":
            return {"operation": "sum", "result": round(sum(numbers), 4)}
        elif operation == "count":
            return {"operation": "count", "result": len(numbers)}
        else:  # summary
            result: dict[str, Any] = {
                "count": len(numbers),
                "mean": round(statistics.mean(numbers), 4),
                "median": round(statistics.median(numbers), 4),
                "min": min(numbers),
                "max": max(numbers),
                "sum": round(sum(numbers), 4),
            }
            if len(numbers) >= 2:
                result["stdev"] = round(statistics.stdev(numbers), 4)
            return result

    def execute_simulated(self, **kwargs: Any) -> Any:
        # Statistics is deterministic
        return self.execute_live(**kwargs)


@register_tool
class DataSort(BaseTool):
    """Sort a list of items."""

    name = "data_sort"
    schema = ToolSchema(
        name="data_sort",
        description="Sort a list of items (numbers or strings) in ascending or descending order.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="items",
                type="array",
                description="List of items to sort",
            ),
            ToolParameter(
                name="order",
                type="string",
                description="Sort order",
                enum=["ascending", "descending"],
                default="ascending",
                required=False,
            ),
            ToolParameter(
                name="key",
                type="string",
                description="If items are objects, the key to sort by",
                required=False,
            ),
        ],
        returns="The sorted list",
        returns_type="array",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        items = list(kwargs["items"])
        order = kwargs.get("order", "ascending")
        key = kwargs.get("key")
        reverse = order == "descending"

        if key and items and isinstance(items[0], dict):
            sorted_items = sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)
        else:
            sorted_items = sorted(items, reverse=reverse)

        return {"sorted": sorted_items, "order": order, "count": len(sorted_items)}

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


@register_tool
class ExecutePython(BaseTool):
    """Execute a simple Python expression safely."""

    name = "execute_python"
    schema = ToolSchema(
        name="execute_python",
        description="Execute a simple Python expression or one-liner and return the result. Supports math, string operations, list comprehensions, and basic data manipulation. No imports or file access allowed.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="code",
                type="string",
                description="Python expression to evaluate (e.g. 'sorted([3,1,2])', 'len(\"hello\")', '[x**2 for x in range(5)]')",
            ),
        ],
        returns="The result of executing the Python expression",
        returns_type="object",
    )

    _SAFE_BUILTINS: dict[str, Any] = {
        "__builtins__": {},
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "frozenset": frozenset,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }

    def execute_live(self, **kwargs: Any) -> Any:
        code = kwargs["code"]
        blocked = ["import", "exec", "eval", "open", "__", "os.", "sys.", "subprocess"]
        for word in blocked:
            if word in code:
                raise ValueError(f"Blocked operation: '{word}' is not allowed")
        result = eval(code, self._SAFE_BUILTINS)  # noqa: S307
        return {"code": code, "result": result, "type": type(result).__name__}

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


@register_tool
class DataAggregate(BaseTool):
    """Aggregate data from a list of records."""

    name = "data_aggregate"
    schema = ToolSchema(
        name="data_aggregate",
        description="Aggregate a list of records by a group key, computing sum, average, count, min, or max for a specified value field.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="items",
                type="array",
                description="List of objects/records to aggregate",
            ),
            ToolParameter(
                name="group_by",
                type="string",
                description="Field name to group records by (e.g. 'department', 'category')",
            ),
            ToolParameter(
                name="value_field",
                type="string",
                description="Field name containing the numeric value to aggregate (e.g. 'salary', 'price')",
            ),
            ToolParameter(
                name="operation",
                type="string",
                description="Aggregation operation to perform",
                enum=["sum", "average", "count", "min", "max"],
                default="sum",
                required=False,
            ),
        ],
        returns="Aggregated results grouped by the specified field",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._aggregate(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._aggregate(**kwargs)

    def _aggregate(self, **kwargs: Any) -> dict[str, Any]:
        items = list(kwargs["items"])
        group_by = kwargs["group_by"]
        value_field = kwargs["value_field"]
        operation = kwargs.get("operation", "sum")

        groups: dict[str, list[float]] = {}
        for item in items:
            if isinstance(item, dict) and group_by in item and value_field in item:
                key = str(item[group_by])
                groups.setdefault(key, []).append(float(item[value_field]))

        results: dict[str, Any] = {}
        for key, values in groups.items():
            if operation == "sum":
                results[key] = round(sum(values), 4)
            elif operation == "average":
                results[key] = round(sum(values) / len(values), 4)
            elif operation == "count":
                results[key] = len(values)
            elif operation == "min":
                results[key] = min(values)
            elif operation == "max":
                results[key] = max(values)

        return {
            "operation": operation,
            "group_by": group_by,
            "value_field": value_field,
            "groups": results,
            "total_records": len(items),
            "total_groups": len(results),
        }


@register_tool
class DataFilter(BaseTool):
    """Filter items based on a condition."""

    name = "data_filter"
    schema = ToolSchema(
        name="data_filter",
        description="Filter a list of items based on a condition. For numbers: greater_than, less_than, equals. For strings: contains, starts_with, ends_with.",
        category=ToolCategory.COMPUTATION,
        parameters=[
            ToolParameter(
                name="items",
                type="array",
                description="List of items to filter",
            ),
            ToolParameter(
                name="condition",
                type="string",
                description="Filter condition",
                enum=["greater_than", "less_than", "equals", "not_equals", "contains", "starts_with", "ends_with"],
            ),
            ToolParameter(
                name="value",
                type="string",
                description="The value to compare against",
            ),
        ],
        returns="The filtered list",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        items = list(kwargs["items"])
        condition = kwargs["condition"]
        value = kwargs["value"]

        filters = {
            "greater_than": lambda x: float(x) > float(value),
            "less_than": lambda x: float(x) < float(value),
            "equals": lambda x: str(x) == str(value),
            "not_equals": lambda x: str(x) != str(value),
            "contains": lambda x: str(value).lower() in str(x).lower(),
            "starts_with": lambda x: str(x).lower().startswith(str(value).lower()),
            "ends_with": lambda x: str(x).lower().endswith(str(value).lower()),
        }

        if condition not in filters:
            raise ValueError(f"Unknown condition: {condition}")

        filtered = [item for item in items if filters[condition](item)]
        return {
            "filtered": filtered,
            "count": len(filtered),
            "original_count": len(items),
            "condition": condition,
            "value": value,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)
