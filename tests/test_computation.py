"""Tests for computation tools."""

from __future__ import annotations

import pytest

from comptoolbench.tools.base import ToolMode
from comptoolbench.tools.computation import (
    Calculator,
    DataAggregate,
    DataFilter,
    DataSort,
    ExecutePython,
    StatisticalAnalysis,
    UnitConvert,
)


class TestCalculator:
    def setup_method(self) -> None:
        self.tool = Calculator(mode=ToolMode.SIMULATED)

    def test_addition(self) -> None:
        r = self.tool.execute(expression="10 + 25")
        assert r.success
        assert r.data["result"] == 35.0

    def test_multiplication(self) -> None:
        r = self.tool.execute(expression="7 * 8")
        assert r.success
        assert r.data["result"] == 56.0

    def test_complex_expression(self) -> None:
        r = self.tool.execute(expression="(100 + 50) * 2 / 3")
        assert r.success
        assert r.data["result"] == 100.0

    def test_sqrt(self) -> None:
        r = self.tool.execute(expression="sqrt(144)")
        assert r.success
        assert r.data["result"] == 12.0

    def test_division_by_zero(self) -> None:
        r = self.tool.execute(expression="1 / 0")
        assert not r.success

    def test_deterministic(self) -> None:
        r1 = self.tool.execute(expression="42 * 17")
        r2 = self.tool.execute(expression="42 * 17")
        assert r1.data["result"] == r2.data["result"]


class TestUnitConvert:
    def setup_method(self) -> None:
        self.tool = UnitConvert(mode=ToolMode.SIMULATED)

    def test_celsius_to_fahrenheit(self) -> None:
        r = self.tool.execute(value=0, from_unit="celsius", to_unit="fahrenheit")
        assert r.success
        assert r.data["converted_value"] == 32.0

    def test_kg_to_lbs(self) -> None:
        r = self.tool.execute(value=1, from_unit="kg", to_unit="lbs")
        assert r.success
        assert abs(r.data["converted_value"] - 2.2046) < 0.01

    def test_same_unit(self) -> None:
        r = self.tool.execute(value=42, from_unit="km", to_unit="km")
        assert r.success
        assert r.data["converted_value"] == 42

    def test_unsupported_conversion(self) -> None:
        r = self.tool.execute(value=1, from_unit="kg", to_unit="celsius")
        assert not r.success


class TestStatisticalAnalysis:
    def setup_method(self) -> None:
        self.tool = StatisticalAnalysis(mode=ToolMode.SIMULATED)

    def test_summary(self) -> None:
        r = self.tool.execute(numbers=[10, 20, 30, 40, 50])
        assert r.success
        assert r.data["mean"] == 30.0
        assert r.data["median"] == 30.0
        assert r.data["min"] == 10.0
        assert r.data["max"] == 50.0
        assert r.data["count"] == 5

    def test_mean(self) -> None:
        r = self.tool.execute(numbers=[1, 2, 3], operation="mean")
        assert r.success
        assert r.data["result"] == 2.0

    def test_empty_list(self) -> None:
        r = self.tool.execute(numbers=[])
        assert not r.success


class TestExecutePython:
    def setup_method(self) -> None:
        self.tool = ExecutePython(mode=ToolMode.SIMULATED)

    def test_sort_list(self) -> None:
        r = self.tool.execute(code="sorted([3, 1, 2])")
        assert r.success
        assert r.data["result"] == [1, 2, 3]

    def test_list_comprehension(self) -> None:
        r = self.tool.execute(code="[x**2 for x in range(5)]")
        assert r.success
        assert r.data["result"] == [0, 1, 4, 9, 16]

    def test_string_operation(self) -> None:
        r = self.tool.execute(code="len('hello world')")
        assert r.success
        assert r.data["result"] == 11

    def test_blocked_import(self) -> None:
        r = self.tool.execute(code="import os")
        assert not r.success

    def test_blocked_dunder(self) -> None:
        r = self.tool.execute(code="__builtins__")
        assert not r.success


class TestDataAggregate:
    def setup_method(self) -> None:
        self.tool = DataAggregate(mode=ToolMode.SIMULATED)

    def test_sum_by_group(self) -> None:
        items = [
            {"dept": "eng", "salary": 100},
            {"dept": "eng", "salary": 200},
            {"dept": "mkt", "salary": 80},
        ]
        r = self.tool.execute(items=items, group_by="dept", value_field="salary", operation="sum")
        assert r.success
        assert r.data["groups"]["eng"] == 300.0
        assert r.data["groups"]["mkt"] == 80.0

    def test_average(self) -> None:
        items = [
            {"cat": "A", "val": 10},
            {"cat": "A", "val": 20},
            {"cat": "B", "val": 30},
        ]
        r = self.tool.execute(items=items, group_by="cat", value_field="val", operation="average")
        assert r.success
        assert r.data["groups"]["A"] == 15.0

    def test_count(self) -> None:
        items = [{"t": "x", "v": 1}] * 5 + [{"t": "y", "v": 2}] * 3
        r = self.tool.execute(items=items, group_by="t", value_field="v", operation="count")
        assert r.success
        assert r.data["groups"]["x"] == 5
        assert r.data["groups"]["y"] == 3


class TestDataSort:
    def setup_method(self) -> None:
        self.tool = DataSort(mode=ToolMode.SIMULATED)

    def test_ascending(self) -> None:
        r = self.tool.execute(items=[3, 1, 2])
        assert r.success
        assert r.data["sorted"] == [1, 2, 3]

    def test_descending(self) -> None:
        r = self.tool.execute(items=[3, 1, 2], order="descending")
        assert r.success
        assert r.data["sorted"] == [3, 2, 1]

    def test_sort_by_key(self) -> None:
        items = [{"name": "B", "v": 2}, {"name": "A", "v": 1}]
        r = self.tool.execute(items=items, key="v")
        assert r.success
        assert r.data["sorted"][0]["name"] == "A"


class TestDataFilter:
    def setup_method(self) -> None:
        self.tool = DataFilter(mode=ToolMode.SIMULATED)

    def test_greater_than(self) -> None:
        r = self.tool.execute(items=[1, 5, 10, 15], condition="greater_than", value="7")
        assert r.success
        assert r.data["filtered"] == [10, 15]

    def test_contains(self) -> None:
        r = self.tool.execute(items=["apple", "banana", "grape"], condition="contains", value="an")
        assert r.success
        assert "banana" in r.data["filtered"]
