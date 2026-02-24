"""Date and time tools: formatting, arithmetic, business-day logic.

These tools are fully local — no API keys needed. All functions use a
fixed reference time (2026-02-22T12:00:00Z) for determinism, so live
and simulated modes produce identical results.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

# Fixed reference time for deterministic outputs across all datetime tools.
_FIXED_NOW = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)

# Weekday names indexed by Monday=0 … Sunday=6 (datetime.weekday()).
_WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _parse_iso_date(date_str: str) -> datetime:
    """Parse an ISO-8601 date or datetime string.

    Accepts formats like ``2026-02-22``, ``2026-02-22T15:30:00``, and
    ``2026-02-22T15:30:00Z``.
    """
    date_str = date_str.strip()

    # Try full datetime with timezone
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    raise ValueError(
        f"Cannot parse date: '{date_str}'. Expected ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"
    )


def _is_business_day(dt: datetime) -> bool:
    """Return True if the date falls on Monday–Friday."""
    return dt.weekday() < 5


@register_tool
class FormatDate(BaseTool):
    """Format a date string into various human-readable formats."""

    name = "format_date"
    schema = ToolSchema(
        name="format_date",
        description="Format an ISO date string into a specified human-readable format. Supports short (Feb 22, 2026), long (February 22, 2026), iso (2026-02-22), us (02/22/2026), and eu (22/02/2026) formats.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="date",
                type="string",
                description="The date to format, in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
            ),
            ToolParameter(
                name="format",
                type="string",
                description="The output format",
                enum=["short", "long", "iso", "us", "eu"],
            ),
        ],
        returns="The formatted date string",
        returns_type="object",
    )

    _FORMAT_MAP: dict[str, str] = {
        "short": "%b %d, %Y",
        "long": "%B %d, %Y",
        "iso": "%Y-%m-%d",
        "us": "%m/%d/%Y",
        "eu": "%d/%m/%Y",
    }

    def execute_live(self, **kwargs: Any) -> Any:
        """Format the date according to the requested format."""
        dt = _parse_iso_date(kwargs["date"])
        fmt_key = kwargs["format"]

        if fmt_key not in self._FORMAT_MAP:
            raise ValueError(
                f"Unknown format: '{fmt_key}'. Must be one of {list(self._FORMAT_MAP)}"
            )

        formatted = dt.strftime(self._FORMAT_MAP[fmt_key])
        return {"formatted": formatted}

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Date formatting is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class AddDuration(BaseTool):
    """Add a duration (days, hours, minutes) to a date."""

    name = "add_duration"
    schema = ToolSchema(
        name="add_duration",
        description="Add a specified number of days, hours, and/or minutes to a date. Returns the resulting date in ISO format.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="date",
                type="string",
                description="The starting date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
            ),
            ToolParameter(
                name="days",
                type="integer",
                description="Number of days to add (can be negative to subtract)",
                required=False,
                default=0,
            ),
            ToolParameter(
                name="hours",
                type="integer",
                description="Number of hours to add (can be negative to subtract)",
                required=False,
                default=0,
            ),
            ToolParameter(
                name="minutes",
                type="integer",
                description="Number of minutes to add (can be negative to subtract)",
                required=False,
                default=0,
            ),
        ],
        returns="The resulting date after adding the duration",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Add days/hours/minutes to the given date."""
        dt = _parse_iso_date(kwargs["date"])
        days = int(kwargs.get("days", 0))
        hours = int(kwargs.get("hours", 0))
        minutes = int(kwargs.get("minutes", 0))

        result = dt + timedelta(days=days, hours=hours, minutes=minutes)
        return {"result_date": result.strftime("%Y-%m-%dT%H:%M:%SZ")}

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Duration addition is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class IsBusinessDay(BaseTool):
    """Check if a date is a business day (Monday–Friday)."""

    name = "is_business_day"
    schema = ToolSchema(
        name="is_business_day",
        description="Check whether a given date falls on a business day (Monday through Friday). Returns a boolean and the day of the week.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="date",
                type="string",
                description="The date to check, in ISO format (YYYY-MM-DD)",
            ),
        ],
        returns="Whether the date is a business day and the day of the week",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Check if the date is a weekday."""
        dt = _parse_iso_date(kwargs["date"])
        weekday_name = _WEEKDAY_NAMES[dt.weekday()]

        return {
            "is_business_day": _is_business_day(dt),
            "day_of_week": weekday_name,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Business-day check is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class GetWeekday(BaseTool):
    """Get the day of the week for a given date."""

    name = "get_weekday"
    schema = ToolSchema(
        name="get_weekday",
        description="Determine what day of the week a given date falls on. Returns the weekday name and its number (0=Monday, 6=Sunday).",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="date",
                type="string",
                description="The date to look up, in ISO format (YYYY-MM-DD)",
            ),
        ],
        returns="The weekday name and day number (0=Monday, 6=Sunday)",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Return the weekday name and number for the date."""
        dt = _parse_iso_date(kwargs["date"])
        day_number = dt.weekday()

        return {
            "weekday": _WEEKDAY_NAMES[day_number],
            "day_number": day_number,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Weekday lookup is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class ParseDate(BaseTool):
    """Parse various date formats into a canonical ISO representation."""

    name = "parse_date"
    schema = ToolSchema(
        name="parse_date",
        description="Parse a date string in various common formats (e.g. 'Feb 22, 2026', '02/22/2026', '22-02-2026', '2026-02-22') into a standard ISO date with extracted year, month, and day components.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="date_string",
                type="string",
                description="The date string to parse (supports ISO, US, EU, and natural-language month formats)",
            ),
        ],
        returns="The parsed date in ISO format with year, month, and day components",
        returns_type="object",
    )

    # Formats tried in order — first successful match wins.
    _FORMATS: list[str] = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d-%m-%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d",
    ]

    def execute_live(self, **kwargs: Any) -> Any:
        """Parse the date string into ISO components."""
        date_string = kwargs["date_string"].strip()

        for fmt in self._FORMATS:
            try:
                dt = datetime.strptime(date_string, fmt)
                return {
                    "iso_date": dt.strftime("%Y-%m-%d"),
                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,
                }
            except ValueError:
                continue

        raise ValueError(
            f"Cannot parse date: '{date_string}'. Supported formats include "
            "YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY, Month DD YYYY, etc."
        )

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Date parsing is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class TimeSince(BaseTool):
    """Calculate the time elapsed since a given date.

    Uses a fixed reference time (2026-02-22T12:00:00Z) so that results
    are deterministic across live and simulated modes.
    """

    name = "time_since"
    schema = ToolSchema(
        name="time_since",
        description="Calculate the time elapsed since a given date, relative to a fixed reference time of 2026-02-22T12:00:00Z. Returns the elapsed time in days and hours, plus a human-readable summary.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="date",
                type="string",
                description="The past date to measure from, in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)",
            ),
        ],
        returns="Days and hours elapsed, plus a human-readable summary",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Compute elapsed time from the given date to the fixed now."""
        dt = _parse_iso_date(kwargs["date"])
        delta = _FIXED_NOW - dt

        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            raise ValueError(
                f"Date '{kwargs['date']}' is in the future relative to "
                f"the reference time ({_FIXED_NOW.isoformat()})"
            )

        total_days = delta.days
        remaining_hours = (total_seconds % 86400) // 3600

        # Build human-readable string
        parts: list[str] = []
        years = total_days // 365
        remaining_days = total_days % 365
        months = remaining_days // 30
        days = remaining_days % 30

        if years > 0:
            parts.append(f"{years} year{'s' if years != 1 else ''}")
        if months > 0:
            parts.append(f"{months} month{'s' if months != 1 else ''}")
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if remaining_hours > 0:
            parts.append(f"{remaining_hours} hour{'s' if remaining_hours != 1 else ''}")

        human_readable = ", ".join(parts) if parts else "just now"

        return {
            "days": total_days,
            "hours": remaining_hours,
            "human_readable": human_readable,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Time-since uses a fixed now — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class NextOccurrence(BaseTool):
    """Find the next occurrence of a given weekday after a date."""

    name = "next_occurrence"
    schema = ToolSchema(
        name="next_occurrence",
        description="Find the next occurrence of a specified weekday on or after a given date. For example, find the next Friday after 2026-02-22.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="weekday",
                type="string",
                description="The target weekday (e.g. 'Monday', 'Friday')",
            ),
            ToolParameter(
                name="after_date",
                type="string",
                description="The date to start searching from, in ISO format (YYYY-MM-DD)",
            ),
        ],
        returns="The date of the next occurrence and number of days until it",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Find the next occurrence of the given weekday."""
        weekday_str = kwargs["weekday"].strip().capitalize()
        dt = _parse_iso_date(kwargs["after_date"])

        if weekday_str not in _WEEKDAY_NAMES:
            raise ValueError(
                f"Unknown weekday: '{weekday_str}'. Must be one of {_WEEKDAY_NAMES}"
            )

        target_index = _WEEKDAY_NAMES.index(weekday_str)
        current_index = dt.weekday()

        # Days until next occurrence (1–7, never 0 — always moves forward)
        days_ahead = (target_index - current_index) % 7
        if days_ahead == 0:
            days_ahead = 7

        next_date = dt + timedelta(days=days_ahead)

        return {
            "date": next_date.strftime("%Y-%m-%d"),
            "days_until": days_ahead,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Next-occurrence is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


@register_tool
class BusinessDaysBetween(BaseTool):
    """Count the number of business days between two dates."""

    name = "business_days_between"
    schema = ToolSchema(
        name="business_days_between",
        description="Count the number of business days (Monday–Friday) between two dates, exclusive of the start date and inclusive of the end date. Also returns the total calendar days.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="start_date",
                type="string",
                description="The start date in ISO format (YYYY-MM-DD)",
            ),
            ToolParameter(
                name="end_date",
                type="string",
                description="The end date in ISO format (YYYY-MM-DD)",
            ),
        ],
        returns="Number of business days and total calendar days between the dates",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        """Count business days between start_date and end_date."""
        start = _parse_iso_date(kwargs["start_date"])
        end = _parse_iso_date(kwargs["end_date"])

        if end < start:
            raise ValueError(
                f"end_date ({kwargs['end_date']}) must be on or after "
                f"start_date ({kwargs['start_date']})"
            )

        total_days = (end - start).days
        business_days = 0
        current = start + timedelta(days=1)  # Exclusive of start

        while current <= end:
            if _is_business_day(current):
                business_days += 1
            current += timedelta(days=1)

        return {
            "business_days": business_days,
            "total_days": total_days,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        """Business-days count is deterministic — same in both modes."""
        return self.execute_live(**kwargs)


__all__ = [
    "AddDuration",
    "BusinessDaysBetween",
    "FormatDate",
    "GetWeekday",
    "IsBusinessDay",
    "NextOccurrence",
    "ParseDate",
    "TimeSince",
]
