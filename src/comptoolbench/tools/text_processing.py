"""Text processing tools: summarize, extract entities, sentiment, classify.

These are lightweight NLP operations. Live mode uses the LLM itself
(or simple heuristics). Simulated mode uses rule-based approaches.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)


@register_tool
class SummarizeText(BaseTool):
    """Summarize a piece of text."""

    name = "summarize_text"
    schema = ToolSchema(
        name="summarize_text",
        description="Summarize a piece of text into a shorter version, preserving key information.",
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to summarize",
            ),
            ToolParameter(
                name="max_sentences",
                type="integer",
                description="Maximum number of sentences in the summary (default 3)",
                required=False,
                default=3,
            ),
        ],
        returns="A summarized version of the text",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        text = kwargs["text"]
        max_sentences = int(kwargs.get("max_sentences", 3))

        # Simple extractive summary: take first N sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        summary_sentences = sentences[:max_sentences]
        summary = " ".join(summary_sentences)

        return {
            "original_length": len(text),
            "summary": summary,
            "summary_length": len(summary),
            "sentences_used": len(summary_sentences),
            "compression_ratio": round(len(summary) / max(len(text), 1), 2),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


@register_tool
class ExtractEntities(BaseTool):
    """Extract named entities from text."""

    name = "extract_entities"
    schema = ToolSchema(
        name="extract_entities",
        description="Extract named entities (people, places, organizations, dates, numbers) from text.",
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to extract entities from",
            ),
        ],
        returns="List of extracted entities with their types",
        returns_type="object",
    )

    # Simple regex patterns for entity extraction
    _PATTERNS: dict[str, str] = {
        "email": r'[\w.+-]+@[\w-]+\.[\w.-]+',
        "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
        "phone": r'\+?\d[\d\s\-()]{7,}\d',
        "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s*\d{4}\b',
        "number": r'\$[\d,.]+|\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',
        "capitalized_phrase": r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',
    }

    def execute_live(self, **kwargs: Any) -> Any:
        text = kwargs["text"]
        entities: list[dict[str, str]] = []

        for entity_type, pattern in self._PATTERNS.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({"text": match.strip(), "type": entity_type})

        # Deduplicate
        seen = set()
        unique_entities = []
        for e in entities:
            key = (e["text"], e["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)

        return {
            "text_length": len(text),
            "entities": unique_entities,
            "entity_count": len(unique_entities),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


@register_tool
class SentimentAnalysis(BaseTool):
    """Analyze sentiment of text."""

    name = "sentiment_analysis"
    schema = ToolSchema(
        name="sentiment_analysis",
        description="Analyze the sentiment of a piece of text, returning positive/negative/neutral classification with a confidence score.",
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to analyze sentiment for",
            ),
        ],
        returns="Sentiment classification (positive/negative/neutral) with confidence score",
        returns_type="object",
    )

    _POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "best", "happy", "beautiful", "awesome", "perfect",
        "brilliant", "outstanding", "superb", "nice", "pleasant", "enjoy",
        "recommend", "impressive", "delightful", "positive", "success",
    }

    _NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "worst", "hate",
        "poor", "disappointing", "ugly", "nasty", "disgusting", "fail",
        "boring", "annoying", "stupid", "useless", "broken", "sad",
        "angry", "frustrating", "negative", "problem", "error", "wrong",
    }

    def execute_live(self, **kwargs: Any) -> Any:
        text = kwargs["text"].lower()
        words = set(re.findall(r'\b\w+\b', text))

        pos_count = len(words & self._POSITIVE_WORDS)
        neg_count = len(words & self._NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            sentiment = "neutral"
            confidence = 0.5
        elif pos_count > neg_count:
            sentiment = "positive"
            confidence = round(0.5 + (pos_count - neg_count) / (total * 2), 2)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = round(0.5 + (neg_count - pos_count) / (total * 2), 2)
        else:
            sentiment = "neutral"
            confidence = 0.5

        return {
            "text_length": len(kwargs["text"]),
            "sentiment": sentiment,
            "confidence": min(confidence, 1.0),
            "positive_signals": pos_count,
            "negative_signals": neg_count,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


@register_tool
class ClassifyText(BaseTool):
    """Classify text into categories."""

    name = "classify_text"
    schema = ToolSchema(
        name="classify_text",
        description="Classify a piece of text into one or more predefined categories based on its content.",
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to classify",
            ),
            ToolParameter(
                name="categories",
                type="array",
                description="List of possible categories to classify into",
            ),
        ],
        returns="The most likely category with confidence scores",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        text = kwargs["text"].lower()
        categories = kwargs["categories"]

        # Simple keyword overlap scoring
        text_words = set(re.findall(r'\b\w+\b', text))
        scores: dict[str, float] = {}

        for cat in categories:
            cat_words = set(re.findall(r'\b\w+\b', cat.lower()))
            overlap = len(text_words & cat_words)
            # Also check if category appears as substring
            if cat.lower() in text:
                overlap += 3
            scores[cat] = overlap

        total = sum(scores.values()) or 1
        normalized = {k: round(v / total, 3) for k, v in scores.items()}
        best = max(normalized, key=lambda k: normalized[k])

        return {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "predicted_category": best,
            "confidence": normalized[best],
            "all_scores": normalized,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


@register_tool
class CompareTexts(BaseTool):
    """Compare two texts for similarity."""

    name = "compare_texts"
    schema = ToolSchema(
        name="compare_texts",
        description="Compare two pieces of text and return their similarity score, common keywords, and differences.",
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text1",
                type="string",
                description="First text to compare",
            ),
            ToolParameter(
                name="text2",
                type="string",
                description="Second text to compare",
            ),
        ],
        returns="Similarity score and comparison details",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        text1 = kwargs["text1"]
        text2 = kwargs["text2"]

        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))

        common = words1 & words2
        all_words = words1 | words2
        similarity = round(len(common) / max(len(all_words), 1), 3)

        only_in_1 = words1 - words2
        only_in_2 = words2 - words1

        return {
            "similarity_score": similarity,
            "common_words_count": len(common),
            "text1_unique_words": len(only_in_1),
            "text2_unique_words": len(only_in_2),
            "common_keywords": sorted(list(common))[:20],
            "text1_length": len(text1),
            "text2_length": len(text2),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)
