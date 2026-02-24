"""Utility / string-manipulation tools for CompToolBench.

All 15 tools in this module are pure, deterministic functions with no
external dependencies.  Both ``execute_live`` and ``execute_simulated``
delegate to the same implementation since there is nothing to simulate.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import unicodedata
from typing import Any, ClassVar
from urllib.parse import urlparse

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

# ---------------------------------------------------------------------------
# Category used for every tool in this module.
# TEXT_PROCESSING already exists in the ToolCategory enum and is the best
# semantic fit for string/utility operations.
# ---------------------------------------------------------------------------
_CATEGORY = ToolCategory.TEXT_PROCESSING


# ====================================================================
# 1. regex_match
# ====================================================================


@register_tool
class RegexMatch(BaseTool):
    """Find regex matches in text."""

    name = "regex_match"
    schema = ToolSchema(
        name="regex_match",
        description=(
            "Find all occurrences of a regular-expression pattern in the "
            "given text.  Returns the first match by default, or all matches "
            "when return_all is true."
        ),
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to search in",
            ),
            ToolParameter(
                name="pattern",
                type="string",
                description="The regular expression pattern to search for",
            ),
            ToolParameter(
                name="return_all",
                type="boolean",
                description="If true, return every match; otherwise return only the first",
                required=False,
                default=False,
            ),
        ],
        returns="List of matched strings and the total count",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        text: str = kwargs["text"]
        pattern: str = kwargs["pattern"]
        return_all: bool = kwargs.get("return_all", False)

        all_matches = re.findall(pattern, text)
        matches = all_matches if return_all else all_matches[:1]

        return {"matches": matches, "count": len(all_matches)}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 2. string_replace
# ====================================================================


@register_tool
class StringReplace(BaseTool):
    """Replace a substring in text."""

    name = "string_replace"
    schema = ToolSchema(
        name="string_replace",
        description="Replace all occurrences of a substring in the given text with a new string.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The original text",
            ),
            ToolParameter(
                name="old",
                type="string",
                description="The substring to find and replace",
            ),
            ToolParameter(
                name="new",
                type="string",
                description="The string to replace each occurrence with",
            ),
        ],
        returns="The modified text and the number of replacements made",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        text: str = kwargs["text"]
        old: str = kwargs["old"]
        new: str = kwargs["new"]

        count = text.count(old)
        result = text.replace(old, new)
        return {"result": result, "replacements": count}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 3. url_parse
# ====================================================================


@register_tool
class UrlParse(BaseTool):
    """Parse a URL into its components."""

    name = "url_parse"
    schema = ToolSchema(
        name="url_parse",
        description="Parse a URL into its constituent parts: scheme, host, path, query string, and fragment.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="The URL to parse",
            ),
        ],
        returns="An object with scheme, host, path, query, and fragment fields",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        url: str = kwargs["url"]
        parsed = urlparse(url)
        return {
            "scheme": parsed.scheme,
            "host": parsed.netloc,
            "path": parsed.path,
            "query": parsed.query,
            "fragment": parsed.fragment,
        }

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 4. json_extract
# ====================================================================


@register_tool
class JsonExtract(BaseTool):
    """Extract a value from JSON using dot-notation path."""

    name = "json_extract"
    schema = ToolSchema(
        name="json_extract",
        description=(
            "Parse a JSON string and extract a value at the given "
            "dot-notation path.  Array indices are supported via "
            "bracket notation (e.g. 'items[0].name')."
        ),
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="data",
                type="string",
                description="A JSON-encoded string",
            ),
            ToolParameter(
                name="path",
                type="string",
                description="Dot-notation path to the value (e.g. 'user.address.city' or 'items[0].name')",
            ),
        ],
        returns="The extracted value and a boolean indicating if the path was found",
        returns_type="object",
    )

    @staticmethod
    def _resolve_path(obj: Any, path: str) -> tuple[Any, bool]:
        """Walk *obj* following *path* (dot-separated, with optional [N] indices)."""
        # Split "a.b[0].c" into ["a", "b", "[0]", "c"]
        tokens: list[str] = []
        for part in path.split("."):
            # Separate "key[0]" into "key" and "[0]"
            segments = re.split(r"(\[\d+\])", part)
            tokens.extend(seg for seg in segments if seg)

        current: Any = obj
        for token in tokens:
            try:
                idx_match = re.fullmatch(r"\[(\d+)\]", token)
                if idx_match:
                    current = current[int(idx_match.group(1))]
                elif isinstance(current, dict):
                    current = current[token]
                else:
                    return None, False
            except (KeyError, IndexError, TypeError):
                return None, False
        return current, True

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        data_str: str = kwargs["data"]
        path: str = kwargs["path"]

        try:
            obj = json.loads(data_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        value, found = self._resolve_path(obj, path)
        return {"value": value, "found": found}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 5. base64_encode
# ====================================================================


@register_tool
class Base64Encode(BaseTool):
    """Encode text to Base64."""

    name = "base64_encode"
    schema = ToolSchema(
        name="base64_encode",
        description="Encode a plain-text string to its Base64 representation.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to encode",
            ),
        ],
        returns="The Base64-encoded string",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, str]:
        text: str = kwargs["text"]
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        return {"encoded": encoded}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 6. base64_decode
# ====================================================================


@register_tool
class Base64Decode(BaseTool):
    """Decode a Base64 string to plain text."""

    name = "base64_decode"
    schema = ToolSchema(
        name="base64_decode",
        description="Decode a Base64-encoded string back to plain text (UTF-8).",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="encoded",
                type="string",
                description="The Base64 string to decode",
            ),
        ],
        returns="The decoded plain-text string",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, str]:
        encoded: str = kwargs["encoded"]
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except Exception as exc:
            raise ValueError(f"Invalid Base64 input: {exc}") from exc
        return {"decoded": decoded}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 7. hash_text
# ====================================================================


@register_tool
class HashText(BaseTool):
    """Hash text with a given algorithm."""

    name = "hash_text"
    schema = ToolSchema(
        name="hash_text",
        description="Compute the cryptographic hash of a text string using md5, sha256, or sha512.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to hash",
            ),
            ToolParameter(
                name="algorithm",
                type="string",
                description="Hash algorithm to use",
                enum=["md5", "sha256", "sha512"],
            ),
        ],
        returns="The hex-digest hash and the algorithm used",
        returns_type="object",
    )

    _ALGORITHMS: ClassVar[dict[str, Any]] = {
        "md5": hashlib.md5,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512,
    }

    def _run(self, **kwargs: Any) -> dict[str, str]:
        text: str = kwargs["text"]
        algorithm: str = kwargs["algorithm"]

        if algorithm not in self._ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Choose from: {list(self._ALGORITHMS)}"
            )

        digest = self._ALGORITHMS[algorithm](text.encode("utf-8")).hexdigest()
        return {"hash": digest, "algorithm": algorithm}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 8. word_count
# ====================================================================


@register_tool
class WordCount(BaseTool):
    """Count words, characters, and sentences in text."""

    name = "word_count"
    schema = ToolSchema(
        name="word_count",
        description="Count the number of words, characters, and sentences in a piece of text.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to analyze",
            ),
        ],
        returns="Word count, character count, and sentence count",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, int]:
        text: str = kwargs["text"]
        words = len(text.split()) if text.strip() else 0
        characters = len(text)
        # Sentences end with . ! or ? (handles multiple punctuation like "...")
        sentences = len(re.findall(r"[^.!?]*[.!?]", text)) if text.strip() else 0
        return {"words": words, "characters": characters, "sentences": sentences}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 9. validate_email
# ====================================================================


@register_tool
class ValidateEmail(BaseTool):
    """Check whether an email address has a valid format."""

    name = "validate_email"
    schema = ToolSchema(
        name="validate_email",
        description="Validate whether a string is a well-formed email address and extract its domain.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="email",
                type="string",
                description="The email address to validate",
            ),
        ],
        returns="Whether the email is valid and its domain",
        returns_type="object",
    )

    # RFC-5322-ish simplified pattern (good enough for benchmark use)
    _EMAIL_RE = re.compile(
        r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        email: str = kwargs["email"].strip()
        valid = bool(self._EMAIL_RE.match(email))
        domain = email.split("@", 1)[1] if "@" in email else ""
        return {"valid": valid, "domain": domain}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 10. format_number
# ====================================================================


@register_tool
class FormatNumber(BaseTool):
    """Format a number in various styles."""

    name = "format_number"
    schema = ToolSchema(
        name="format_number",
        description="Format a number as currency (USD), percentage, scientific notation, or with comma separators.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="value",
                type="number",
                description="The number to format",
            ),
            ToolParameter(
                name="format",
                type="string",
                description="Desired output format",
                enum=["currency", "percent", "scientific", "comma"],
            ),
        ],
        returns="The formatted number string",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, str]:
        value = float(kwargs["value"])
        fmt: str = kwargs["format"]

        if fmt == "currency":
            formatted = f"${value:,.2f}"
        elif fmt == "percent":
            formatted = f"{value:.2%}"
        elif fmt == "scientific":
            formatted = f"{value:.6e}"
        elif fmt == "comma":
            formatted = f"{value:,.2f}"
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        return {"formatted": formatted}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 11. slugify
# ====================================================================


@register_tool
class Slugify(BaseTool):
    """Convert text to a URL-safe slug."""

    name = "slugify"
    schema = ToolSchema(
        name="slugify",
        description="Convert arbitrary text to a URL-safe, lowercase, hyphen-separated slug.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to slugify",
            ),
        ],
        returns="The URL-safe slug",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, str]:
        text: str = kwargs["text"]
        # Normalize unicode (e.g. accented characters -> ASCII)
        slug = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        slug = slug.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)  # remove non-word chars except hyphens
        slug = re.sub(r"[\s_]+", "-", slug)    # spaces/underscores -> hyphens
        slug = re.sub(r"-+", "-", slug)        # collapse multiple hyphens
        slug = slug.strip("-")
        return {"slug": slug}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 12. truncate_text
# ====================================================================


@register_tool
class TruncateText(BaseTool):
    """Truncate text to a maximum length."""

    name = "truncate_text"
    schema = ToolSchema(
        name="truncate_text",
        description="Truncate text to a maximum character length, appending a configurable suffix (default '...') when truncated.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to truncate",
            ),
            ToolParameter(
                name="max_length",
                type="integer",
                description="Maximum number of characters in the output (including suffix)",
            ),
            ToolParameter(
                name="suffix",
                type="string",
                description="The suffix to append when truncation occurs",
                required=False,
                default="...",
            ),
        ],
        returns="The (possibly truncated) text and whether truncation occurred",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        text: str = kwargs["text"]
        max_length: int = int(kwargs["max_length"])
        suffix: str = kwargs.get("suffix", "...")

        if max_length < 0:
            raise ValueError("max_length must be non-negative")

        if len(text) <= max_length:
            return {"truncated": text, "was_truncated": False}

        # Ensure there is room for the suffix
        cut = max(max_length - len(suffix), 0)
        return {"truncated": text[:cut] + suffix, "was_truncated": True}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 13. split_text
# ====================================================================


@register_tool
class SplitText(BaseTool):
    """Split text by a delimiter."""

    name = "split_text"
    schema = ToolSchema(
        name="split_text",
        description="Split a string into parts by the given delimiter.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to split",
            ),
            ToolParameter(
                name="delimiter",
                type="string",
                description="The delimiter string to split on (e.g. ',', ' ', '\\n')",
            ),
        ],
        returns="The list of parts and the number of parts",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        text: str = kwargs["text"]
        delimiter: str = kwargs["delimiter"]

        # Handle common escape sequences that arrive as literal strings
        escape_map = {"\\n": "\n", "\\t": "\t", "\\r": "\r"}
        delimiter = escape_map.get(delimiter, delimiter)

        parts = text.split(delimiter)
        return {"parts": parts, "count": len(parts)}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 14. join_texts
# ====================================================================


@register_tool
class JoinTexts(BaseTool):
    """Join a list of strings with a separator."""

    name = "join_texts"
    schema = ToolSchema(
        name="join_texts",
        description="Join an array of text strings into one string with a given separator.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="texts",
                type="array",
                description="List of strings to join",
            ),
            ToolParameter(
                name="separator",
                type="string",
                description="The separator to place between each text",
            ),
        ],
        returns="The joined result string",
        returns_type="object",
    )

    def _run(self, **kwargs: Any) -> dict[str, str]:
        texts: list[str] = [str(t) for t in kwargs["texts"]]
        separator: str = kwargs["separator"]

        # Handle common escape sequences
        escape_map = {"\\n": "\n", "\\t": "\t", "\\r": "\r"}
        separator = escape_map.get(separator, separator)

        result = separator.join(texts)
        return {"result": result}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ====================================================================
# 15. case_convert
# ====================================================================


@register_tool
class CaseConvert(BaseTool):
    """Convert text between cases (upper, lower, title, camel, snake)."""

    name = "case_convert"
    schema = ToolSchema(
        name="case_convert",
        description="Convert text to a different case: upper, lower, title, camelCase, or snake_case.",
        category=_CATEGORY,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to convert",
            ),
            ToolParameter(
                name="target_case",
                type="string",
                description="Target case format",
                enum=["upper", "lower", "title", "camel", "snake"],
            ),
        ],
        returns="The case-converted text",
        returns_type="object",
    )

    @staticmethod
    def _to_words(text: str) -> list[str]:
        """Split text into words, handling camelCase, snake_case, and spaces."""
        # Insert spaces before uppercase letters (for camelCase)
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
        # Replace underscores and hyphens with spaces
        spaced = re.sub(r"[_\-]+", " ", spaced)
        return [w for w in spaced.split() if w]

    def _run(self, **kwargs: Any) -> dict[str, str]:
        text: str = kwargs["text"]
        target_case: str = kwargs["target_case"]

        if target_case == "upper":
            converted = text.upper()
        elif target_case == "lower":
            converted = text.lower()
        elif target_case == "title":
            converted = text.title()
        elif target_case == "camel":
            words = self._to_words(text)
            if not words:
                converted = ""
            else:
                converted = words[0].lower() + "".join(w.capitalize() for w in words[1:])
        elif target_case == "snake":
            words = self._to_words(text)
            converted = "_".join(w.lower() for w in words)
        else:
            raise ValueError(f"Unsupported target case: {target_case}")

        return {"converted": converted}

    def execute_live(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._run(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Base64Decode",
    "Base64Encode",
    "CaseConvert",
    "FormatNumber",
    "HashText",
    "JoinTexts",
    "JsonExtract",
    "RegexMatch",
    "Slugify",
    "SplitText",
    "StringReplace",
    "TruncateText",
    "UrlParse",
    "ValidateEmail",
    "WordCount",
]
