"""Productivity and organisation tools: calendar, reminders, invoicing, reports.

These tools simulate real productivity actions.
Both live and simulated modes return deterministic confirmations
(no real side-effects) so benchmark runs are reproducible.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
import zlib
from typing import Any

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)


def _sim_hash(seed: str, *args: Any) -> int:
    """Deterministic hash -> integer for simulated data."""
    raw = json.dumps([seed, *args], sort_keys=True, default=str)
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16)


def _sim_hex(seed: str, *args: Any) -> str:
    """Deterministic hash -> hex string (12 chars)."""
    raw = json.dumps([seed, *args], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 1. create_calendar_event
# ---------------------------------------------------------------------------


@register_tool
class CreateCalendarEvent(BaseTool):
    """Create a calendar event."""

    name = "create_calendar_event"
    schema = ToolSchema(
        name="create_calendar_event",
        description=(
            "Create a calendar event with a title, date, duration, and "
            "optional list of attendees. Returns a confirmation with the "
            "event ID."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="title",
                type="string",
                description="Event title (e.g. 'Team Standup', 'Dentist Appointment')",
            ),
            ToolParameter(
                name="date",
                type="string",
                description="Event date and time (e.g. '2026-03-20 14:00')",
            ),
            ToolParameter(
                name="duration_minutes",
                type="integer",
                description="Duration of the event in minutes",
            ),
            ToolParameter(
                name="attendees",
                type="array",
                description="List of attendee names or email addresses",
                required=False,
            ),
        ],
        returns="Confirmation with event ID and details",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def _create(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "event_id": f"evt_{_sim_hex('cal_event', kwargs['title'], kwargs['date'])}",
            "title": kwargs["title"],
            "date": kwargs["date"],
            "duration_minutes": int(kwargs["duration_minutes"]),
            "attendees": kwargs.get("attendees", []),
            "confirmation": f"Calendar event '{kwargs['title']}' created for {kwargs['date']}.",
        }


# ---------------------------------------------------------------------------
# 2. set_reminder
# ---------------------------------------------------------------------------


@register_tool
class SetReminder(BaseTool):
    """Set a reminder."""

    name = "set_reminder"
    schema = ToolSchema(
        name="set_reminder",
        description=(
            "Set a reminder with a message and a specific date/time to be "
            "reminded. Returns a confirmation with the reminder ID."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="message",
                type="string",
                description="Reminder message (e.g. 'Call the vet')",
            ),
            ToolParameter(
                name="remind_at",
                type="string",
                description="When to remind, as a datetime string (e.g. '2026-03-20 09:00')",
            ),
        ],
        returns="Confirmation with reminder ID",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def _create(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "reminder_id": f"rem_{_sim_hex('reminder', kwargs['message'], kwargs['remind_at'])}",
            "message": kwargs["message"],
            "remind_at": kwargs["remind_at"],
            "confirmation": f"Reminder set for {kwargs['remind_at']}: '{kwargs['message']}'.",
        }


# ---------------------------------------------------------------------------
# 3. create_contact
# ---------------------------------------------------------------------------


@register_tool
class CreateContact(BaseTool):
    """Create a contact entry."""

    name = "create_contact"
    schema = ToolSchema(
        name="create_contact",
        description=(
            "Create a new contact entry with a name, email address, and "
            "optional phone number."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="name",
                type="string",
                description="Contact full name",
            ),
            ToolParameter(
                name="email",
                type="string",
                description="Contact email address",
            ),
            ToolParameter(
                name="phone",
                type="string",
                description="Contact phone number",
                required=False,
            ),
        ],
        returns="Confirmation with contact ID",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def _create(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "contact_id": f"con_{_sim_hex('contact', kwargs['name'], kwargs['email'])}",
            "name": kwargs["name"],
            "email": kwargs["email"],
            "phone": kwargs.get("phone"),
            "confirmation": f"Contact '{kwargs['name']}' created.",
        }


# ---------------------------------------------------------------------------
# 4. create_invoice
# ---------------------------------------------------------------------------


@register_tool
class CreateInvoice(BaseTool):
    """Create an invoice."""

    name = "create_invoice"
    schema = ToolSchema(
        name="create_invoice",
        description=(
            "Create an invoice for a client with a list of line items. "
            "Each item in the array should be a JSON object with 'description' "
            "(string), 'quantity' (number), and 'unit_price' (number) fields. "
            "Returns the invoice ID, computed total, and confirmation."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="client_name",
                type="string",
                description="Name of the client being invoiced",
            ),
            ToolParameter(
                name="items",
                type="array",
                description=(
                    "Line items, each with 'description', 'quantity', and "
                    "'unit_price' (e.g. [{\"description\": \"Widget\", "
                    "\"quantity\": 2, \"unit_price\": 9.99}])"
                ),
            ),
            ToolParameter(
                name="currency",
                type="string",
                description="Currency code (default 'USD')",
                required=False,
                default="USD",
            ),
        ],
        returns="Invoice ID, total amount, and confirmation",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def _create(self, **kwargs: Any) -> dict[str, Any]:
        client_name = kwargs["client_name"]
        items = kwargs["items"]
        currency = kwargs.get("currency", "USD")

        total = 0.0
        for item in items:
            qty = float(item.get("quantity", 1))
            price = float(item.get("unit_price", 0))
            total += qty * price

        return {
            "invoice_id": f"inv_{_sim_hex('invoice', client_name, json.dumps(items, default=str))}",
            "client_name": client_name,
            "items_count": len(items),
            "total": round(total, 2),
            "currency": currency,
            "confirmation": f"Invoice created for {client_name}: {total:.2f} {currency}.",
        }


# ---------------------------------------------------------------------------
# 5. generate_report
# ---------------------------------------------------------------------------


@register_tool
class GenerateReport(BaseTool):
    """Generate a summary report from data."""

    name = "generate_report"
    schema = ToolSchema(
        name="generate_report",
        description=(
            "Generate a summary report from structured data. Accepts a title, "
            "data as a JSON string, and a desired output format (text, markdown, "
            "or html)."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="title",
                type="string",
                description="Report title",
            ),
            ToolParameter(
                name="data",
                type="string",
                description=(
                    "Report data as a JSON string "
                    "(e.g. '{\"sales\": 150, \"returns\": 3}')"
                ),
            ),
            ToolParameter(
                name="format",
                type="string",
                description="Output format for the report",
                enum=["text", "markdown", "html"],
            ),
        ],
        returns="Generated report string and word count",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._generate(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._generate(**kwargs)

    def _generate(self, **kwargs: Any) -> dict[str, Any]:
        """Build a report from structured data (pure function)."""
        title = kwargs["title"]
        data_str = kwargs["data"]
        fmt = kwargs["format"]

        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, TypeError):
            data = {"raw": data_str}

        # Build report lines
        lines: list[str] = []

        if fmt == "markdown":
            lines.append(f"# {title}")
            lines.append("")
            lines.append("## Summary")
            lines.append("")
            for key, value in (data.items() if isinstance(data, dict) else [("data", data)]):
                lines.append(f"- **{key}**: {value}")
            lines.append("")
            lines.append("*Report generated at 2026-02-22T12:00:00*")
        elif fmt == "html":
            lines.append(f"<h1>{title}</h1>")
            lines.append("<h2>Summary</h2>")
            lines.append("<ul>")
            for key, value in (data.items() if isinstance(data, dict) else [("data", data)]):
                lines.append(f"  <li><strong>{key}</strong>: {value}</li>")
            lines.append("</ul>")
            lines.append("<p><em>Report generated at 2026-02-22T12:00:00</em></p>")
        else:  # text
            lines.append(title.upper())
            lines.append("=" * len(title))
            lines.append("")
            lines.append("Summary:")
            for key, value in (data.items() if isinstance(data, dict) else [("data", data)]):
                lines.append(f"  {key}: {value}")
            lines.append("")
            lines.append("Generated: 2026-02-22T12:00:00")

        report = "\n".join(lines)
        word_count = len(report.split())

        return {
            "report": report,
            "word_count": word_count,
        }


# ---------------------------------------------------------------------------
# 6. send_webhook
# ---------------------------------------------------------------------------


@register_tool
class SendWebhook(BaseTool):
    """Send a webhook notification (simulated)."""

    name = "send_webhook"
    schema = ToolSchema(
        name="send_webhook",
        description=(
            "Send a webhook HTTP POST notification to a URL with a JSON "
            "payload. Returns the simulated response status."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="Webhook endpoint URL",
            ),
            ToolParameter(
                name="payload",
                type="string",
                description="JSON payload to send (e.g. '{\"event\": \"deploy\", \"status\": \"success\"}')",
            ),
        ],
        returns="Delivery status and response code",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._send(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._send(**kwargs)

    def _send(self, **kwargs: Any) -> dict[str, Any]:
        url = kwargs["url"]
        payload = kwargs["payload"]
        h = _sim_hash("webhook", url, payload)

        # Simulate: most webhooks succeed
        codes = [200, 200, 200, 200, 201, 202, 400, 500]
        response_code = codes[h % len(codes)]
        status = "delivered" if response_code < 300 else "failed"

        return {
            "status": status,
            "response_code": response_code,
            "webhook_url": url,
            "delivery_id": f"whk_{_sim_hex('webhook_id', url, payload)}",
        }


# ---------------------------------------------------------------------------
# 7. log_event
# ---------------------------------------------------------------------------


@register_tool
class LogEvent(BaseTool):
    """Log an event."""

    name = "log_event"
    schema = ToolSchema(
        name="log_event",
        description=(
            "Log an event with a type, message, and severity level. "
            "Returns a log entry ID and timestamp."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="event_type",
                type="string",
                description="Type of event (e.g. 'user_login', 'payment', 'error')",
            ),
            ToolParameter(
                name="message",
                type="string",
                description="Log message describing the event",
            ),
            ToolParameter(
                name="severity",
                type="string",
                description="Severity level of the event",
                enum=["info", "warning", "error"],
            ),
        ],
        returns="Log entry ID and timestamp",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._log(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._log(**kwargs)

    def _log(self, **kwargs: Any) -> dict[str, Any]:
        event_type = kwargs["event_type"]
        message = kwargs["message"]
        severity = kwargs["severity"]

        return {
            "log_id": f"log_{_sim_hex('log', event_type, message)}",
            "event_type": event_type,
            "message": message,
            "severity": severity,
            "timestamp": "2026-02-22T12:00:00",
        }


# ---------------------------------------------------------------------------
# 8. create_spreadsheet
# ---------------------------------------------------------------------------


@register_tool
class CreateSpreadsheet(BaseTool):
    """Create a spreadsheet."""

    name = "create_spreadsheet"
    schema = ToolSchema(
        name="create_spreadsheet",
        description=(
            "Create a spreadsheet with a title, column headers, and data rows. "
            "Returns a spreadsheet ID and row count."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="title",
                type="string",
                description="Spreadsheet title",
            ),
            ToolParameter(
                name="headers",
                type="array",
                description="List of column header strings (e.g. ['Name', 'Age', 'City'])",
            ),
            ToolParameter(
                name="rows",
                type="array",
                description=(
                    "Array of row arrays, each containing values matching "
                    "the headers (e.g. [['Alice', '30', 'NYC'], ['Bob', '25', 'LA']])"
                ),
                items={"type": "array", "items": {"type": "string"}},
            ),
        ],
        returns="Spreadsheet ID, row count, and column count",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    def _create(self, **kwargs: Any) -> dict[str, Any]:
        title = kwargs["title"]
        headers = kwargs["headers"]
        rows = kwargs["rows"]

        return {
            "spreadsheet_id": f"sht_{_sim_hex('spreadsheet', title)}",
            "title": title,
            "columns": len(headers),
            "row_count": len(rows),
            "headers": headers,
            "confirmation": f"Spreadsheet '{title}' created with {len(rows)} rows and {len(headers)} columns.",
        }


# ---------------------------------------------------------------------------
# 9. encrypt_text
# ---------------------------------------------------------------------------


@register_tool
class EncryptText(BaseTool):
    """Encrypt text (simulated)."""

    name = "encrypt_text"
    schema = ToolSchema(
        name="encrypt_text",
        description=(
            "Encrypt a text string using a specified encryption method. "
            "Returns the simulated encrypted output as a base64 string."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The plain text to encrypt",
            ),
            ToolParameter(
                name="method",
                type="string",
                description="Encryption algorithm to use",
                enum=["aes256", "rsa"],
            ),
        ],
        returns="Encrypted text (base64) and method used",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._encrypt(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._encrypt(**kwargs)

    def _encrypt(self, **kwargs: Any) -> dict[str, Any]:
        """Simulate encryption by hashing the input deterministically."""
        text = kwargs["text"]
        method = kwargs["method"]

        # Produce a deterministic "encrypted" output via SHA-256 + base64
        raw = json.dumps({"text": text, "method": method}, sort_keys=True)
        digest = hashlib.sha256(raw.encode()).digest()
        encrypted = base64.b64encode(digest).decode()

        return {
            "encrypted": encrypted,
            "method": method,
            "original_length": len(text),
            "encrypted_length": len(encrypted),
        }


# ---------------------------------------------------------------------------
# 10. compress_data
# ---------------------------------------------------------------------------


@register_tool
class CompressData(BaseTool):
    """Compress data (simulated)."""

    name = "compress_data"
    schema = ToolSchema(
        name="compress_data",
        description=(
            "Compress a data string using a specified compression algorithm. "
            "Returns the compressed size, original size, and compression ratio."
        ),
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="data",
                type="string",
                description="The data string to compress",
            ),
            ToolParameter(
                name="algorithm",
                type="string",
                description="Compression algorithm to use",
                enum=["gzip", "zlib", "lz4"],
            ),
        ],
        returns="Compressed size, original size, and compression ratio",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._compress(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._compress(**kwargs)

    def _compress(self, **kwargs: Any) -> dict[str, Any]:
        """Compute real zlib-based compression for deterministic results."""
        data = kwargs["data"]
        algorithm = kwargs["algorithm"]

        original_size = len(data.encode("utf-8"))

        # Use zlib for all algorithms (deterministic, stdlib-only)
        # The level differs to simulate different algorithm characteristics
        levels = {"gzip": 6, "zlib": 6, "lz4": 1}
        level = levels.get(algorithm, 6)
        compressed_bytes = zlib.compress(data.encode("utf-8"), level)
        compressed_size = len(compressed_bytes)

        ratio = round(compressed_size / original_size, 4) if original_size > 0 else 1.0

        return {
            "compressed_size": compressed_size,
            "original_size": original_size,
            "ratio": ratio,
            "algorithm": algorithm,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CreateCalendarEvent",
    "SetReminder",
    "CreateContact",
    "CreateInvoice",
    "GenerateReport",
    "SendWebhook",
    "LogEvent",
    "CreateSpreadsheet",
    "EncryptText",
    "CompressData",
]
