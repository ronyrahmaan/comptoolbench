"""Tests for text processing tools."""

from __future__ import annotations

from comptoolbench.tools.base import ToolMode
from comptoolbench.tools.text_processing import (
    ClassifyText,
    CompareTexts,
    ExtractEntities,
    SentimentAnalysis,
    SummarizeText,
)


class TestSummarizeText:
    def setup_method(self) -> None:
        self.tool = SummarizeText(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        r = self.tool.execute(text=text, max_sentences=2)
        assert r.success
        assert r.data["sentences_used"] == 2
        assert r.data["compression_ratio"] < 1.0

    def test_short_text(self) -> None:
        r = self.tool.execute(text="Just one sentence.", max_sentences=5)
        assert r.success
        assert r.data["sentences_used"] == 1


class TestExtractEntities:
    def setup_method(self) -> None:
        self.tool = ExtractEntities(mode=ToolMode.SIMULATED)

    def test_email_extraction(self) -> None:
        r = self.tool.execute(text="Contact us at info@example.com or support@test.org")
        assert r.success
        emails = [e for e in r.data["entities"] if e["type"] == "email"]
        assert len(emails) == 2

    def test_date_extraction(self) -> None:
        r = self.tool.execute(text="The meeting is on 03/15/2026 and 04/20/2026.")
        assert r.success
        dates = [e for e in r.data["entities"] if e["type"] == "date"]
        assert len(dates) >= 2

    def test_capitalized_phrases(self) -> None:
        r = self.tool.execute(text="Albert Einstein worked at Princeton University.")
        assert r.success
        caps = [e for e in r.data["entities"] if e["type"] == "capitalized_phrase"]
        assert any("Albert Einstein" in e["text"] for e in caps)


class TestSentimentAnalysis:
    def setup_method(self) -> None:
        self.tool = SentimentAnalysis(mode=ToolMode.SIMULATED)

    def test_positive(self) -> None:
        r = self.tool.execute(text="This is amazing and wonderful, absolutely love it!")
        assert r.success
        assert r.data["sentiment"] == "positive"

    def test_negative(self) -> None:
        r = self.tool.execute(text="Terrible experience, the worst service ever.")
        assert r.success
        assert r.data["sentiment"] == "negative"

    def test_neutral(self) -> None:
        r = self.tool.execute(text="The meeting is at 3pm in room 204.")
        assert r.success
        assert r.data["sentiment"] == "neutral"


class TestClassifyText:
    def setup_method(self) -> None:
        self.tool = ClassifyText(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(
            text="The stock market rallied today with tech stocks leading gains",
            categories=["finance", "sports", "technology"],
        )
        assert r.success
        assert r.data["predicted_category"] in ["finance", "technology"]

    def test_all_scores_present(self) -> None:
        cats = ["A", "B", "C"]
        r = self.tool.execute(text="test", categories=cats)
        assert r.success
        assert set(r.data["all_scores"].keys()) == set(cats)


class TestCompareTexts:
    def setup_method(self) -> None:
        self.tool = CompareTexts(mode=ToolMode.SIMULATED)

    def test_identical(self) -> None:
        r = self.tool.execute(text1="hello world", text2="hello world")
        assert r.success
        assert r.data["similarity_score"] == 1.0

    def test_different(self) -> None:
        r = self.tool.execute(text1="cats and dogs", text2="apples and oranges")
        assert r.success
        assert r.data["similarity_score"] < 1.0

    def test_partial_overlap(self) -> None:
        r = self.tool.execute(text1="hello world", text2="hello there world")
        assert r.success
        assert 0 < r.data["similarity_score"] < 1.0
