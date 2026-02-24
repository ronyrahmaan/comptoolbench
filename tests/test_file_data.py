"""Tests for file and data tools."""

from __future__ import annotations

from comptoolbench.tools.base import ToolMode
from comptoolbench.tools.file_data import (
    ListFiles,
    MergeData,
    ReadFile,
    TransformFormat,
    WriteFile,
)


class TestReadFile:
    def setup_method(self) -> None:
        self.tool = ReadFile(mode=ToolMode.SIMULATED)

    def test_read_existing(self) -> None:
        r = self.tool.execute(path="/data/report.txt")
        assert r.success
        assert r.data["exists"] is True
        assert "Q1 2026" in r.data["content"]

    def test_read_json(self) -> None:
        r = self.tool.execute(path="/data/config.json")
        assert r.success
        assert "DataPipeline" in r.data["content"]

    def test_read_missing(self) -> None:
        r = self.tool.execute(path="/nonexistent/file.txt")
        assert r.success  # Tool doesn't fail, returns not found
        assert r.data["exists"] is False


class TestWriteFile:
    def setup_method(self) -> None:
        self.write = WriteFile(mode=ToolMode.SIMULATED)
        self.read = ReadFile(mode=ToolMode.SIMULATED)

    def test_write_new(self) -> None:
        r = self.write.execute(path="/data/new_file.txt", content="Hello world")
        assert r.success
        assert r.data["status"] == "written"
        assert r.data["overwrote_existing"] is False

        # Verify we can read it back
        r2 = self.read.execute(path="/data/new_file.txt")
        assert r2.data["content"] == "Hello world"

    def test_overwrite_existing(self) -> None:
        r = self.write.execute(path="/data/report.txt", content="New content")
        assert r.success
        assert r.data["overwrote_existing"] is True

    def test_type_inference(self) -> None:
        r = self.write.execute(path="/data/output.json", content='{"key": "val"}')
        assert r.data["type"] == "application/json"


class TestListFiles:
    def setup_method(self) -> None:
        self.tool = ListFiles(mode=ToolMode.SIMULATED)

    def test_list_all(self) -> None:
        r = self.tool.execute(directory="/")
        assert r.success
        assert r.data["count"] >= 5  # Seed files

    def test_list_data_dir(self) -> None:
        r = self.tool.execute(directory="/data")
        assert r.success
        assert r.data["count"] >= 5
        names = [f["name"] for f in r.data["files"]]
        assert "report.txt" in names


class TestTransformFormat:
    def setup_method(self) -> None:
        self.tool = TransformFormat(mode=ToolMode.SIMULATED)

    def test_json_to_csv(self) -> None:
        data = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        r = self.tool.execute(data=data, from_format="json", to_format="csv")
        assert r.success
        assert "Alice" in r.data["transformed"]
        assert "name" in r.data["transformed"]  # Header

    def test_csv_to_json(self) -> None:
        data = "name,age\nAlice,30\nBob,25"
        r = self.tool.execute(data=data, from_format="csv", to_format="json")
        assert r.success
        assert "Alice" in r.data["transformed"]

    def test_json_to_text(self) -> None:
        data = '[{"name": "Alice", "age": "30"}]'
        r = self.tool.execute(data=data, from_format="json", to_format="text")
        assert r.success
        assert "Alice" in r.data["transformed"]


class TestMergeData:
    def setup_method(self) -> None:
        self.tool = MergeData(mode=ToolMode.SIMULATED)

    def test_basic_merge(self) -> None:
        ds1 = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
        ds2 = '[{"id": 1, "score": 95}, {"id": 2, "score": 87}]'
        r = self.tool.execute(dataset1=ds1, dataset2=ds2, join_key="id")
        assert r.success
        assert r.data["matched_records"] == 2
        assert r.data["merged"][0]["name"] == "Alice"
        assert r.data["merged"][0]["score"] == 95

    def test_partial_match(self) -> None:
        ds1 = '[{"id": 1, "name": "Alice"}, {"id": 3, "name": "Charlie"}]'
        ds2 = '[{"id": 1, "score": 95}]'
        r = self.tool.execute(dataset1=ds1, dataset2=ds2, join_key="id")
        assert r.success
        assert r.data["matched_records"] == 1
        assert r.data["unmatched_records"] == 1
