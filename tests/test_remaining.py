"""Tests for communication, time/scheduling, media, and info retrieval tools."""

from __future__ import annotations

from comptoolbench.tools.base import ToolMode
from comptoolbench.tools.communication import (
    CreateNotification,
    CreateTask,
    ScheduleMeeting,
    SendEmail,
    SendMessage,
)
from comptoolbench.tools.information_retrieval import (
    DatabaseQuery,
    KnowledgeBaseQuery,
    LookupEntity,
    WebPageFetch,
    WebSearch,
)
from comptoolbench.tools.media import GenerateImage, TranscribeAudio
from comptoolbench.tools.time_scheduling import (
    CalculateDateDiff,
    ConvertTimezone,
    GetCurrentTime,
)


# --- Communication ---


class TestSendEmail:
    def setup_method(self) -> None:
        self.tool = SendEmail(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(to="alice@example.com", subject="Test", body="Hello")
        assert r.success
        assert r.data["status"] == "sent"
        assert r.data["to"] == "alice@example.com"


class TestSendMessage:
    def setup_method(self) -> None:
        self.tool = SendMessage(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(recipient="alice", message="Hi there")
        assert r.success
        assert r.data["status"] == "delivered"


class TestCreateNotification:
    def setup_method(self) -> None:
        self.tool = CreateNotification(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(title="Alert", message="Server is down", priority="urgent")
        assert r.success
        assert r.data["priority"] == "urgent"


class TestCreateTask:
    def setup_method(self) -> None:
        self.tool = CreateTask(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(title="Fix bug", priority="high")
        assert r.success
        assert r.data["title"] == "Fix bug"


class TestScheduleMeeting:
    def setup_method(self) -> None:
        self.tool = ScheduleMeeting(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(
            title="Standup",
            datetime_str="2026-03-20 09:00",
            participants=["alice", "bob"],
            duration_minutes=30,
        )
        assert r.success
        assert r.data["status"] == "scheduled"
        assert r.data["participants"] == ["alice", "bob"]


# --- Information Retrieval ---


class TestWebSearch:
    def setup_method(self) -> None:
        self.tool = WebSearch(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(query="python programming", max_results=3)
        assert r.success
        assert len(r.data["results"]) == 3
        assert all("title" in res for res in r.data["results"])

    def test_deterministic(self) -> None:
        r1 = self.tool.execute(query="test", max_results=2)
        r2 = self.tool.execute(query="test", max_results=2)
        assert r1.data["results"][0]["title"] == r2.data["results"][0]["title"]


class TestDatabaseQuery:
    def setup_method(self) -> None:
        self.tool = DatabaseQuery(mode=ToolMode.SIMULATED)

    def test_filter_equals(self) -> None:
        r = self.tool.execute(
            table="countries", filter_field="continent",
            filter_op="equals", filter_value="Europe", limit=5,
        )
        assert r.success
        assert all(row["continent"] == "Europe" for row in r.data["results"])

    def test_sort(self) -> None:
        r = self.tool.execute(table="countries", sort_by="population", limit=3)
        assert r.success
        pops = [row["population"] for row in r.data["results"]]
        assert pops == sorted(pops, reverse=True)

    def test_unknown_table(self) -> None:
        r = self.tool.execute(table="countries", filter_field="continent",
                              filter_op="equals", filter_value="Atlantis")
        assert r.success
        assert r.data["count"] == 0


class TestLookupEntity:
    def setup_method(self) -> None:
        self.tool = LookupEntity(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(entity="Python programming")
        assert r.success
        assert r.data["entity"] == "Python programming"
        assert r.data["summary"]


class TestWebPageFetch:
    def setup_method(self) -> None:
        self.tool = WebPageFetch(mode=ToolMode.SIMULATED)

    def test_text_mode(self) -> None:
        r = self.tool.execute(url="https://example.com/ai-research", extract_mode="text")
        assert r.success
        assert r.data["status_code"] == 200
        assert "content" in r.data

    def test_metadata_mode(self) -> None:
        r = self.tool.execute(url="https://example.com/article", extract_mode="metadata")
        assert r.success
        assert "author" in r.data


class TestKnowledgeBaseQuery:
    def setup_method(self) -> None:
        self.tool = KnowledgeBaseQuery(mode=ToolMode.SIMULATED)

    def test_known_fact(self) -> None:
        r = self.tool.execute(query="capital of france")
        assert r.success
        assert r.data["answer"] == "Paris"
        assert r.data["confidence"] == 1.0

    def test_unknown_query(self) -> None:
        r = self.tool.execute(query="obscure topic xyz")
        assert r.success
        assert r.data["found"] is True
        assert r.data["confidence"] < 1.0


# --- Time & Scheduling ---


class TestGetCurrentTime:
    def setup_method(self) -> None:
        self.tool = GetCurrentTime(mode=ToolMode.SIMULATED)

    def test_utc(self) -> None:
        r = self.tool.execute(timezone="UTC")
        assert r.success
        assert r.data["timezone"] == "UTC"
        assert "date" in r.data

    def test_different_timezone(self) -> None:
        r = self.tool.execute(timezone="Asia/Tokyo")
        assert r.success
        assert "Asia/Tokyo" in r.data["timezone"]


class TestConvertTimezone:
    def setup_method(self) -> None:
        self.tool = ConvertTimezone(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(
            datetime_str="2026-03-15T14:00:00",
            from_timezone="UTC",
            to_timezone="EST",
        )
        assert r.success
        # EST maps to America/New_York; on March 15 DST is active (EDT = UTC-4)
        assert r.data["converted_time"] == "10:00:00"


class TestCalculateDateDiff:
    def setup_method(self) -> None:
        self.tool = CalculateDateDiff(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(date1="2026-01-01", date2="2026-01-31")
        assert r.success
        assert r.data["days"] == 30

    def test_symmetric(self) -> None:
        r1 = self.tool.execute(date1="2026-01-01", date2="2026-06-30")
        r2 = self.tool.execute(date1="2026-06-30", date2="2026-01-01")
        assert r1.data["days"] == r2.data["days"]


# --- Media ---


class TestGenerateImage:
    def setup_method(self) -> None:
        self.tool = GenerateImage(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(prompt="sunset over mountains", size="512x512")
        assert r.success
        assert r.data["status"] == "generated"
        assert r.data["image_url"].startswith("https://")

    def test_deterministic(self) -> None:
        r1 = self.tool.execute(prompt="cats")
        r2 = self.tool.execute(prompt="cats")
        assert r1.data["image_url"] == r2.data["image_url"]


class TestTranscribeAudio:
    def setup_method(self) -> None:
        self.tool = TranscribeAudio(mode=ToolMode.SIMULATED)

    def test_known_audio(self) -> None:
        r = self.tool.execute(audio_source="https://example.com/meeting_recording.mp3")
        assert r.success
        assert r.data["status"] == "completed"
        assert "quarterly review" in r.data["text"].lower()

    def test_unknown_audio(self) -> None:
        r = self.tool.execute(audio_source="https://example.com/random_audio.wav")
        assert r.success
        assert r.data["text"]
        assert r.data["word_count"] > 0
