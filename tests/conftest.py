"""Shared test fixtures for CompToolBench."""

from __future__ import annotations

import pytest

from comptoolbench.tools.base import ToolMode
from comptoolbench.tools.file_data import reset_virtual_fs
from comptoolbench.tools.state_management import reset_memory_store


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Reset all mutable state before each test."""
    reset_virtual_fs()
    reset_memory_store()


@pytest.fixture()
def simulated_mode() -> ToolMode:
    return ToolMode.SIMULATED


@pytest.fixture()
def live_mode() -> ToolMode:
    return ToolMode.LIVE
