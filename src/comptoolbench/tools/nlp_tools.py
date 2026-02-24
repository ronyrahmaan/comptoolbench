"""NLP / AI text tools: language detection, spelling, tokenization, and more.

All tools in this module are pure-function implementations (no external APIs
required), so ``execute_live`` and ``execute_simulated`` are identical.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, ClassVar

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "it",
        "this",
        "that",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "not",
        "no",
        "so",
        "if",
        "as",
        "its",
        "my",
        "we",
        "they",
        "he",
        "she",
        "i",
        "you",
        "me",
        "him",
        "her",
        "us",
        "them",
        "our",
        "your",
        "his",
        "their",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "also",
        "am",
        "any",
        "because",
        "before",
        "below",
        "between",
        "during",
        "into",
        "out",
        "over",
        "own",
        "same",
        "then",
        "through",
        "under",
        "until",
        "up",
        "while",
    }
)

# Language detection word lists — common function words per language
_LANGUAGE_MARKERS: dict[str, tuple[str, set[str]]] = {
    "English": (
        "en",
        {
            "the",
            "and",
            "is",
            "in",
            "to",
            "of",
            "it",
            "that",
            "was",
            "for",
            "on",
            "are",
            "with",
            "as",
            "at",
            "this",
            "but",
            "not",
            "you",
            "from",
        },
    ),
    "French": (
        "fr",
        {
            "le",
            "la",
            "les",
            "de",
            "des",
            "du",
            "un",
            "une",
            "et",
            "est",
            "en",
            "que",
            "qui",
            "dans",
            "pour",
            "pas",
            "sur",
            "au",
            "avec",
            "ce",
        },
    ),
    "German": (
        "de",
        {
            "der",
            "die",
            "das",
            "und",
            "ist",
            "ein",
            "eine",
            "nicht",
            "mit",
            "auf",
            "den",
            "dem",
            "des",
            "sich",
            "von",
            "zu",
            "auch",
            "es",
            "ich",
            "aber",
        },
    ),
    "Spanish": (
        "es",
        {
            "el",
            "la",
            "los",
            "las",
            "de",
            "en",
            "que",
            "es",
            "un",
            "una",
            "por",
            "del",
            "con",
            "para",
            "como",
            "pero",
            "su",
            "al",
            "se",
            "no",
        },
    ),
    "Italian": (
        "it",
        {
            "il",
            "lo",
            "la",
            "gli",
            "le",
            "di",
            "che",
            "non",
            "un",
            "una",
            "per",
            "sono",
            "con",
            "del",
            "della",
            "dei",
            "anche",
            "alla",
            "nel",
            "ma",
        },
    ),
    "Portuguese": (
        "pt",
        {
            "os",
            "as",
            "um",
            "uma",
            "que",
            "em",
            "para",
            "com",
            "por",
            "mais",
            "mas",
            "como",
            "dos",
            "das",
            "seu",
            "sua",
            "ou",
            "quando",
            "muito",
            "nos",
        },
    ),
    "Dutch": (
        "nl",
        {
            "de",
            "het",
            "een",
            "van",
            "en",
            "in",
            "is",
            "dat",
            "op",
            "te",
            "zijn",
            "voor",
            "niet",
            "met",
            "ook",
            "maar",
            "nog",
            "aan",
            "dit",
            "bij",
        },
    ),
}

# Small hard-coded misspelling dictionary
_MISSPELLINGS: dict[str, str] = {
    "teh": "the",
    "recieve": "receive",
    "occured": "occurred",
    "seperate": "separate",
    "definately": "definitely",
    "accomodate": "accommodate",
    "occurence": "occurrence",
    "neccessary": "necessary",
    "embarass": "embarrass",
    "goverment": "government",
    "enviroment": "environment",
    "arguement": "argument",
    "begining": "beginning",
    "beleive": "believe",
    "calender": "calendar",
    "commitee": "committee",
    "concious": "conscious",
    "desparate": "desperate",
    "diffrence": "difference",
    "existance": "existence",
    "foriegn": "foreign",
    "grammer": "grammar",
    "harrass": "harass",
    "immediatly": "immediately",
    "independant": "independent",
    "jeopardize": "jeopardize",
    "knowlege": "knowledge",
    "liason": "liaison",
    "maintenence": "maintenance",
    "millenium": "millennium",
    "noticable": "noticeable",
    "persistant": "persistent",
    "publically": "publicly",
    "recomend": "recommend",
    "refered": "referred",
    "succesful": "successful",
    "tommorow": "tomorrow",
    "untill": "until",
    "wierd": "weird",
    "writting": "writing",
}

# Number-word mappings for text_to_number and number_to_text
_ONES: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS: dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_SCALES: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

_ONES_INV: dict[int, str] = {v: k for k, v in _ONES.items()}
_TENS_INV: dict[int, str] = {v: k for k, v in _TENS.items()}


# ---------------------------------------------------------------------------
# 1. detect_language
# ---------------------------------------------------------------------------


@register_tool
class DetectLanguage(BaseTool):
    """Detect the language of a given text using word-frequency heuristics."""

    name = "detect_language"
    schema = ToolSchema(
        name="detect_language",
        description=(
            "Detect the language of the input text. Uses common function-word "
            "heuristics to identify English, French, German, Spanish, Italian, "
            "Portuguese, and Dutch."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text whose language should be detected",
            ),
        ],
        returns="Detected language name, ISO code, and confidence score",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Detect language via function-word overlap."""
        text: str = kwargs["text"]
        words = set(re.findall(r"\b\w+\b", text.lower()))

        if not words:
            return {
                "language": "unknown",
                "language_code": "und",
                "confidence": 0.0,
            }

        scores: dict[str, int] = {}
        for lang, (_, marker_words) in _LANGUAGE_MARKERS.items():
            scores[lang] = len(words & marker_words)

        total = sum(scores.values()) or 1
        best_lang = max(scores, key=lambda k: scores[k])
        best_score = scores[best_lang]

        if best_score == 0:
            return {
                "language": "unknown",
                "language_code": "und",
                "confidence": 0.0,
            }

        confidence = round(best_score / total, 2)
        code = _LANGUAGE_MARKERS[best_lang][0]

        return {
            "language": best_lang,
            "language_code": code,
            "confidence": min(confidence, 1.0),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 2. spell_check
# ---------------------------------------------------------------------------


@register_tool
class SpellCheck(BaseTool):
    """Check spelling in text using a built-in misspelling dictionary."""

    name = "spell_check"
    schema = ToolSchema(
        name="spell_check",
        description=(
            "Check the spelling of words in a piece of text. Returns "
            "suggested corrections and the corrected text."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to spell-check",
            ),
        ],
        returns="List of corrections, corrected text, and error count",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Spell-check via dictionary lookup of known misspellings."""
        text: str = kwargs["text"]
        corrections: list[dict[str, str]] = []
        corrected = text

        # Iterate over words; we need to preserve case awareness
        for match in re.finditer(r"\b\w+\b", text):
            word = match.group()
            lower = word.lower()
            if lower in _MISSPELLINGS:
                replacement = _MISSPELLINGS[lower]
                # Preserve original capitalisation
                if word[0].isupper():
                    replacement = replacement.capitalize()
                if word.isupper():
                    replacement = replacement.upper()
                corrections.append(
                    {
                        "original": word,
                        "correction": replacement,
                        "position": match.start(),
                    }
                )

        # Apply corrections in reverse order to keep positions stable
        for corr in reversed(corrections):
            pos = int(corr["position"])
            end = pos + len(corr["original"])
            corrected = corrected[:pos] + corr["correction"] + corrected[end:]

        return {
            "corrections": corrections,
            "corrected_text": corrected,
            "error_count": len(corrections),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 3. tokenize_text
# ---------------------------------------------------------------------------


@register_tool
class TokenizeText(BaseTool):
    """Tokenize text into words, sentences, or characters."""

    name = "tokenize_text"
    schema = ToolSchema(
        name="tokenize_text",
        description=(
            "Tokenize a piece of text into tokens using the specified "
            "method: word, sentence, or character."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to tokenize",
            ),
            ToolParameter(
                name="method",
                type="string",
                description="Tokenization method (word, sentence, or character)",
                enum=["word", "sentence", "character"],
            ),
        ],
        returns="List of tokens and token count",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Tokenize text using lightweight rules."""
        text: str = kwargs["text"]
        method: str = kwargs["method"]

        if method == "word":
            tokens = re.findall(r"\b\w+\b", text)
        elif method == "sentence":
            tokens = [
                s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()
            ]
        elif method == "character":
            tokens = list(text)
        else:
            raise ValueError(f"Unknown tokenization method: {method}")

        return {
            "tokens": tokens,
            "count": len(tokens),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 4. text_similarity
# ---------------------------------------------------------------------------


@register_tool
class TextSimilarity(BaseTool):
    """Compute Jaccard similarity between two texts."""

    name = "text_similarity"
    schema = ToolSchema(
        name="text_similarity",
        description=(
            "Compute the similarity between two texts using Jaccard "
            "similarity of their word sets."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text1",
                type="string",
                description="First text",
            ),
            ToolParameter(
                name="text2",
                type="string",
                description="Second text",
            ),
        ],
        returns="Similarity score (0.0 - 1.0) and the method used",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Compute Jaccard similarity of word sets."""
        text1: str = kwargs["text1"]
        text2: str = kwargs["text2"]

        words1 = set(re.findall(r"\b\w+\b", text1.lower()))
        words2 = set(re.findall(r"\b\w+\b", text2.lower()))

        union = words1 | words2
        if not union:
            return {"similarity": 1.0, "method": "jaccard"}

        intersection = words1 & words2
        similarity = round(len(intersection) / len(union), 4)

        return {
            "similarity": similarity,
            "method": "jaccard",
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 5. keyword_extract
# ---------------------------------------------------------------------------


@register_tool
class KeywordExtract(BaseTool):
    """Extract keywords from text using word frequency."""

    name = "keyword_extract"
    schema = ToolSchema(
        name="keyword_extract",
        description=(
            "Extract the most important keywords from a text based on "
            "word frequency, excluding common stopwords."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to extract keywords from",
            ),
            ToolParameter(
                name="max_keywords",
                type="integer",
                description="Maximum number of keywords to return (default 5)",
                required=False,
                default=5,
            ),
        ],
        returns="List of keywords with their relative frequency scores",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Extract keywords via frequency analysis minus stopwords."""
        text: str = kwargs["text"]
        max_keywords: int = int(kwargs.get("max_keywords", 5))

        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        filtered = [w for w in words if w not in _STOPWORDS]

        if not filtered:
            return {"keywords": [], "scores": []}

        counts = Counter(filtered)
        top = counts.most_common(max_keywords)
        max_count = top[0][1] if top else 1

        keywords = [word for word, _ in top]
        scores = [round(count / max_count, 4) for _, count in top]

        return {
            "keywords": keywords,
            "scores": scores,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 6. text_to_number — helper
# ---------------------------------------------------------------------------


def _text_to_number(text: str) -> float:
    """Convert an English number phrase to its numeric value.

    Handles:
    - Pure digit strings: "42", "3.14", "-7"
    - Multiplier suffixes: "1.5 million", "3 thousand"
    - English words: "twenty-three", "one hundred forty-two"
    - Compound forms: "two million three hundred thousand"
    """
    cleaned = text.strip().lower()

    # Try pure numeric first (ints, floats, negatives, with optional commas)
    numeric_match = re.fullmatch(r"-?[\d,]+(?:\.\d+)?", cleaned.replace(" ", ""))
    if numeric_match:
        return float(cleaned.replace(",", "").replace(" ", ""))

    # Handle "<number> <scale-word>" patterns like "1.5 million"
    scale_suffix_match = re.fullmatch(
        r"(-?[\d,]+(?:\.\d+)?)\s*(thousand|million|billion)",
        cleaned,
    )
    if scale_suffix_match:
        base = float(scale_suffix_match.group(1).replace(",", ""))
        multiplier = _SCALES[scale_suffix_match.group(2)]
        return base * multiplier

    # Parse English words
    return _parse_english_number(cleaned)


def _parse_english_number(text: str) -> float:
    """Parse an English number phrase like 'two hundred forty-three thousand five hundred one'."""
    # Normalise hyphens and extra spaces
    text = text.replace("-", " ").strip()
    tokens = text.split()

    if not tokens:
        raise ValueError("Empty number text")

    # Handle negative
    negative = False
    if tokens[0] in ("minus", "negative"):
        negative = True
        tokens = tokens[1:]

    result = 0
    current = 0

    for token in tokens:
        if token == "and":
            continue
        elif token in _ONES:
            current += _ONES[token]
        elif token in _TENS:
            current += _TENS[token]
        elif token == "hundred":
            current *= 100
        elif token in ("thousand", "million", "billion"):
            if current == 0:
                current = 1
            current *= _SCALES[token]
            result += current
            current = 0
        else:
            raise ValueError(f"Unrecognised number word: '{token}'")

    result += current
    return -result if negative else result


# ---------------------------------------------------------------------------
# 6. text_to_number — tool
# ---------------------------------------------------------------------------


@register_tool
class TextToNumber(BaseTool):
    """Convert a textual number representation to a numeric value."""

    name = "text_to_number"
    schema = ToolSchema(
        name="text_to_number",
        description=(
            "Convert a number expressed in English words (e.g. 'twenty-three', "
            "'1.5 million') to its numeric value."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The textual number to convert",
            ),
        ],
        returns="Numeric value and the original text",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Convert text to number."""
        text: str = kwargs["text"]
        number = _text_to_number(text)
        # Return int when there is no fractional part
        if number == int(number):
            number = int(number)
        return {
            "number": number,
            "original": text,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 7. number_to_text — helper
# ---------------------------------------------------------------------------


def _number_to_text(n: int | float) -> str:
    """Convert a number to English words.

    Handles integers in the range 0 .. 999,999,999,999.
    Floats are split at the decimal point ('point').
    """
    if isinstance(n, float) and n != int(n):
        # Handle float by splitting at decimal point
        int_part = int(n) if n >= 0 else -int(-n)
        frac_str = str(n).split(".")[1]
        int_text = _number_to_text(int_part)
        frac_digits = " ".join(_ONES_INV[int(d)] for d in frac_str)
        return f"{int_text} point {frac_digits}"

    n = int(n)

    if n < 0:
        return "negative " + _number_to_text(-n)

    if n == 0:
        return "zero"

    parts: list[str] = []

    # Billions
    if n >= 1_000_000_000:
        billions = n // 1_000_000_000
        parts.append(_number_to_text(billions) + " billion")
        n %= 1_000_000_000

    # Millions
    if n >= 1_000_000:
        millions = n // 1_000_000
        parts.append(_number_to_text(millions) + " million")
        n %= 1_000_000

    # Thousands
    if n >= 1_000:
        thousands = n // 1_000
        parts.append(_number_to_text(thousands) + " thousand")
        n %= 1_000

    # Hundreds
    if n >= 100:
        hundreds = n // 100
        parts.append(_ONES_INV[hundreds] + " hundred")
        n %= 100

    # Tens and ones
    if n > 0:
        if n in _ONES_INV:
            parts.append(_ONES_INV[n])
        elif n in _TENS_INV:
            parts.append(_TENS_INV[n])
        else:
            tens_part = (n // 10) * 10
            ones_part = n % 10
            parts.append(f"{_TENS_INV[tens_part]}-{_ONES_INV[ones_part]}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# 7. number_to_text — tool
# ---------------------------------------------------------------------------


@register_tool
class NumberToText(BaseTool):
    """Convert a numeric value to English words."""

    name = "number_to_text"
    schema = ToolSchema(
        name="number_to_text",
        description=(
            "Convert a number to its English word representation. "
            "Handles integers up to 999,999,999 and simple decimals."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="number",
                type="number",
                description="The number to convert to words",
            ),
        ],
        returns="English text representation of the number",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Convert number to English text."""
        number = kwargs["number"]
        # Accept int or float
        if isinstance(number, str):
            number = float(number)
        text = _number_to_text(number)
        return {"text": text}

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 8. generate_summary_stats
# ---------------------------------------------------------------------------


@register_tool
class GenerateSummaryStats(BaseTool):
    """Generate descriptive statistics for a piece of text."""

    name = "generate_summary_stats"
    schema = ToolSchema(
        name="generate_summary_stats",
        description=(
            "Generate statistics about a text including word count, sentence "
            "count, average word length, estimated reading time, and a "
            "complexity score."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to analyse",
            ),
        ],
        returns=(
            "Word count, sentence count, average word length, reading time "
            "in seconds, and complexity score"
        ),
        returns_type="object",
    )

    # Average adult reading speed (words per minute)
    _READING_WPM: int = 238

    def execute_live(self, **kwargs: Any) -> Any:
        """Compute text statistics."""
        text: str = kwargs["text"]

        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        sentence_count = (
            max(len(re.split(r"[.!?]+", text.strip())) - 1, 1) if text.strip() else 0
        )

        avg_word_length = (
            round(sum(len(w) for w in words) / word_count, 2) if word_count else 0.0
        )

        reading_time_seconds = math.ceil((word_count / self._READING_WPM) * 60)

        # Simple complexity score: average word length * average sentence length
        avg_sentence_length = (
            round(word_count / sentence_count, 2) if sentence_count else 0.0
        )
        complexity_score = round(avg_word_length * avg_sentence_length * 0.1, 2)

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "reading_time_seconds": reading_time_seconds,
            "complexity_score": complexity_score,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 9. mask_pii
# ---------------------------------------------------------------------------


@register_tool
class MaskPII(BaseTool):
    """Mask personally identifiable information in text."""

    name = "mask_pii"
    schema = ToolSchema(
        name="mask_pii",
        description=(
            "Detect and mask personally identifiable information (PII) in "
            "text, including email addresses, phone numbers, and SSNs."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to scan for PII",
            ),
        ],
        returns="Masked text and list of PII items found",
        returns_type="object",
    )

    # Ordered from most specific to least specific to avoid partial matches
    _PII_PATTERNS: ClassVar[list[tuple[str, str, str]]] = [
        # (label, regex, mask)
        ("ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
        ("email", r"[\w.+-]+@[\w-]+\.[\w.-]+", "[EMAIL]"),
        ("phone", r"\+?\d[\d\s\-()]{7,}\d", "[PHONE]"),
    ]

    def execute_live(self, **kwargs: Any) -> Any:
        """Mask PII using regex patterns."""
        text: str = kwargs["text"]
        masked = text
        pii_found: list[dict[str, Any]] = []

        for label, pattern, mask in self._PII_PATTERNS:
            for match in re.finditer(pattern, masked):
                pii_found.append(
                    {
                        "type": label,
                        "value": match.group(),
                        "position": match.start(),
                    }
                )
            masked = re.sub(pattern, mask, masked)

        return {
            "masked_text": masked,
            "pii_found": pii_found,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# 10. extract_numbers
# ---------------------------------------------------------------------------


@register_tool
class ExtractNumbers(BaseTool):
    """Extract all numeric values from text."""

    name = "extract_numbers"
    schema = ToolSchema(
        name="extract_numbers",
        description=(
            "Extract all numbers from a piece of text, including integers, "
            "decimals, and negative numbers."
        ),
        category=ToolCategory.TEXT_PROCESSING,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to extract numbers from",
            ),
        ],
        returns="List of extracted numbers and the count",
        returns_type="object",
    )

    # Matches: -3.14, 1,000, 42, 0.5, etc.
    _NUMBER_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?")

    def execute_live(self, **kwargs: Any) -> Any:
        """Extract numbers via regex."""
        text: str = kwargs["text"]
        matches = self._NUMBER_RE.findall(text)

        numbers: list[float] = []
        for m in matches:
            cleaned = m.replace(",", "")
            value = float(cleaned)
            if value == int(value):
                value = int(value)
            numbers.append(value)

        return {
            "numbers": numbers,
            "count": len(numbers),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DetectLanguage",
    "ExtractNumbers",
    "GenerateSummaryStats",
    "KeywordExtract",
    "MaskPII",
    "NumberToText",
    "SpellCheck",
    "TextSimilarity",
    "TextToNumber",
    "TokenizeText",
]
