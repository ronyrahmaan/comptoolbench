"""Tests for state management tools."""

from __future__ import annotations

from comptoolbench.tools.base import ToolMode
from comptoolbench.tools.state_management import (
    GetSessionContext,
    ListMemories,
    RetrieveMemory,
    StoreMemory,
)


class TestStoreMemory:
    def setup_method(self) -> None:
        self.tool = StoreMemory(mode=ToolMode.SIMULATED)

    def test_store_string(self) -> None:
        r = self.tool.execute(key="test_key", value="test_value")
        assert r.success
        assert r.data["status"] == "stored"
        assert r.data["key"] == "test_key"

    def test_overwrite(self) -> None:
        self.tool.execute(key="k", value="v1")
        r = self.tool.execute(key="k", value="v2")
        assert r.success
        assert r.data["overwrote_existing"] is True

    def test_store_with_tags(self) -> None:
        r = self.tool.execute(key="tagged", value="data", tags=["tag1", "tag2"])
        assert r.success
        assert r.data["tags"] == ["tag1", "tag2"]

    def test_store_json_value(self) -> None:
        r = self.tool.execute(key="json_val", value='{"nested": true}')
        assert r.success


class TestRetrieveMemory:
    def setup_method(self) -> None:
        self.store = StoreMemory(mode=ToolMode.SIMULATED)
        self.retrieve = RetrieveMemory(mode=ToolMode.SIMULATED)

    def test_retrieve_existing(self) -> None:
        self.store.execute(key="my_key", value="my_value")
        r = self.retrieve.execute(key="my_key")
        assert r.success
        assert r.data["status"] == "found"
        assert r.data["value"] == "my_value"

    def test_retrieve_missing(self) -> None:
        r = self.retrieve.execute(key="nonexistent_key_xyz")
        assert r.success  # Tool succeeds, but status is not_found
        assert r.data["status"] == "not_found"

    def test_retrieve_seeded(self) -> None:
        r = self.retrieve.execute(key="user_preferences")
        assert r.success
        assert r.data["status"] == "found"
        assert r.data["value"]["theme"] == "dark"


class TestListMemories:
    def setup_method(self) -> None:
        self.tool = ListMemories(mode=ToolMode.SIMULATED)

    def test_list_all(self) -> None:
        r = self.tool.execute()
        assert r.success
        assert r.data["count"] >= 5  # Seed data has 5 entries

    def test_filter_by_tag(self) -> None:
        r = self.tool.execute(tag="settings")
        assert r.success
        assert r.data["count"] >= 1
        assert r.data["filter_tag"] == "settings"


class TestGetSessionContext:
    def setup_method(self) -> None:
        self.tool = GetSessionContext(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute()
        assert r.success
        assert "session_id" in r.data
        assert "memory_count" in r.data
        assert r.data["memory_count"] >= 5
