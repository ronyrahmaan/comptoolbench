"""Time and scheduling tools: current time, timezone conversion, date calculations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

_TIMEZONE_MAP: dict[str, str] = {
    "est": "America/New_York",
    "cst": "America/Chicago",
    "mst": "America/Denver",
    "pst": "America/Los_Angeles",
    "gmt": "Europe/London",
    "utc": "UTC",
    "cet": "Europe/Berlin",
    "jst": "Asia/Tokyo",
    "ist": "Asia/Kolkata",
    "aest": "Australia/Sydney",
    "kst": "Asia/Seoul",
    "cst_china": "Asia/Shanghai",
    "sgt": "Asia/Singapore",
}


def _resolve_tz(tz_str: str) -> ZoneInfo:
    """Resolve a timezone string (abbreviation or IANA name) to ZoneInfo."""
    key = tz_str.lower().strip()
    iana = _TIMEZONE_MAP.get(key, tz_str)
    return ZoneInfo(iana)


@register_tool
class GetCurrentTime(BaseTool):
    """Get the current date and time."""

    name = "get_current_time"
    schema = ToolSchema(
        name="get_current_time",
        description="Get the current date and time, optionally in a specific timezone.",
        category=ToolCategory.TIME_SCHEDULING,
        parameters=[
            ToolParameter(
                name="timezone",
                type="string",
                description="Timezone name (e.g. 'UTC', 'EST', 'Asia/Tokyo', 'Europe/London'). Default is UTC.",
                required=False,
                default="UTC",
            ),
        ],
        returns="Current date and time in the specified timezone",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        tz_str = kwargs.get("timezone", "UTC")
        tz = _resolve_tz(tz_str)
        now = datetime.now(tz)

        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": str(tz),
            "day_of_week": now.strftime("%A"),
            "unix_timestamp": int(now.timestamp()),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        # Use a fixed reference time for reproducibility
        tz_str = kwargs.get("timezone", "UTC")
        tz = _resolve_tz(tz_str)
        fixed = datetime(2026, 3, 15, 14, 30, 0, tzinfo=timezone.utc).astimezone(tz)

        return {
            "datetime": fixed.isoformat(),
            "date": fixed.strftime("%Y-%m-%d"),
            "time": fixed.strftime("%H:%M:%S"),
            "timezone": str(tz),
            "day_of_week": fixed.strftime("%A"),
            "unix_timestamp": int(fixed.timestamp()),
        }


@register_tool
class ConvertTimezone(BaseTool):
    """Convert a time between timezones."""

    name = "convert_timezone"
    schema = ToolSchema(
        name="convert_timezone",
        description="Convert a date/time from one timezone to another.",
        category=ToolCategory.TIME_SCHEDULING,
        parameters=[
            ToolParameter(
                name="datetime_str",
                type="string",
                description="The datetime to convert (ISO format or 'HH:MM' for today)",
            ),
            ToolParameter(
                name="from_timezone",
                type="string",
                description="Source timezone (e.g. 'EST', 'UTC', 'Asia/Tokyo')",
            ),
            ToolParameter(
                name="to_timezone",
                type="string",
                description="Target timezone (e.g. 'PST', 'Europe/London')",
            ),
        ],
        returns="The converted datetime in the target timezone",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        dt_str = kwargs["datetime_str"]
        from_tz = _resolve_tz(kwargs["from_timezone"])
        to_tz = _resolve_tz(kwargs["to_timezone"])

        # Parse the datetime
        if "T" in dt_str or "-" in dt_str:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=from_tz)
        else:
            # Assume HH:MM format for today
            parts = dt_str.split(":")
            now = datetime.now(from_tz)
            dt = now.replace(
                hour=int(parts[0]), minute=int(parts[1]),
                second=0, microsecond=0,
            )

        converted = dt.astimezone(to_tz)
        return {
            "original": dt.isoformat(),
            "original_timezone": str(from_tz),
            "converted": converted.isoformat(),
            "converted_timezone": str(to_tz),
            "converted_time": converted.strftime("%H:%M:%S"),
            "converted_date": converted.strftime("%Y-%m-%d"),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)


@register_tool
class CalculateDateDiff(BaseTool):
    """Calculate the difference between two dates."""

    name = "calculate_date_diff"
    schema = ToolSchema(
        name="calculate_date_diff",
        description="Calculate the difference between two dates in days, weeks, months, or years.",
        category=ToolCategory.TIME_SCHEDULING,
        parameters=[
            ToolParameter(
                name="date1",
                type="string",
                description="First date (YYYY-MM-DD format)",
            ),
            ToolParameter(
                name="date2",
                type="string",
                description="Second date (YYYY-MM-DD format)",
            ),
        ],
        returns="The difference between the two dates in various units",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        d1 = datetime.strptime(kwargs["date1"], "%Y-%m-%d")
        d2 = datetime.strptime(kwargs["date2"], "%Y-%m-%d")
        diff = abs(d2 - d1)

        return {
            "date1": kwargs["date1"],
            "date2": kwargs["date2"],
            "days": diff.days,
            "weeks": round(diff.days / 7, 1),
            "months": round(diff.days / 30.44, 1),
            "years": round(diff.days / 365.25, 2),
            "date1_is_before": d1 < d2,
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self.execute_live(**kwargs)
