"""Communication tools: email, messaging, notifications, tasks.

These tools simulate real communication actions.
In live mode, they log the action (not actually send).
In simulated mode, they return deterministic confirmations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

# Fixed timestamp for deterministic simulated mode
_FIXED_TIMESTAMP = "2026-02-22T12:00:00"


def _det_id(prefix: str, **kwargs: Any) -> str:
    """Generate a deterministic ID from inputs (replaces uuid.uuid4)."""
    raw = json.dumps([prefix, kwargs], sort_keys=True, default=str)
    return f"{prefix}_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


@register_tool
class SendEmail(BaseTool):
    """Compose and send an email."""

    name = "send_email"
    schema = ToolSchema(
        name="send_email",
        description="Compose and send an email to a specified recipient with a subject and body.",
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="to", type="string",
                description="Recipient email address",
            ),
            ToolParameter(
                name="subject", type="string",
                description="Email subject line",
            ),
            ToolParameter(
                name="body", type="string",
                description="Email body text",
            ),
            ToolParameter(
                name="cc", type="string",
                description="CC recipients (comma-separated)",
                required=False,
            ),
        ],
        returns="Confirmation with message ID and timestamp",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        # We don't actually send emails — we return a confirmation
        return self._make_confirmation(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._make_confirmation(**kwargs)

    def _make_confirmation(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "sent",
            "message_id": _det_id("msg", to=kwargs["to"], subject=kwargs["subject"]),
            "to": kwargs["to"],
            "subject": kwargs["subject"],
            "body_preview": kwargs["body"][:100] + ("..." if len(kwargs["body"]) > 100 else ""),
            "cc": kwargs.get("cc"),
            "timestamp": _FIXED_TIMESTAMP,
        }


@register_tool
class SendMessage(BaseTool):
    """Send a chat message."""

    name = "send_message"
    schema = ToolSchema(
        name="send_message",
        description="Send a message to a user or channel in a messaging platform.",
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="recipient", type="string",
                description="Username or channel name",
            ),
            ToolParameter(
                name="message", type="string",
                description="The message content",
            ),
        ],
        returns="Confirmation of sent message",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._confirm(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._confirm(**kwargs)

    def _confirm(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "delivered",
            "message_id": _det_id("chat", recipient=kwargs["recipient"], message=kwargs["message"]),
            "recipient": kwargs["recipient"],
            "message_preview": kwargs["message"][:80],
            "timestamp": _FIXED_TIMESTAMP,
        }


@register_tool
class CreateNotification(BaseTool):
    """Create a notification or alert."""

    name = "create_notification"
    schema = ToolSchema(
        name="create_notification",
        description="Create a notification or alert with a title and message, optionally with a priority level.",
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="title", type="string",
                description="Notification title",
            ),
            ToolParameter(
                name="message", type="string",
                description="Notification message body",
            ),
            ToolParameter(
                name="priority", type="string",
                description="Priority level",
                enum=["low", "normal", "high", "urgent"],
                default="normal",
                required=False,
            ),
        ],
        returns="Confirmation with notification ID",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._confirm(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._confirm(**kwargs)

    def _confirm(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "created",
            "notification_id": _det_id("notif", title=kwargs["title"]),
            "title": kwargs["title"],
            "priority": kwargs.get("priority", "normal"),
            "timestamp": _FIXED_TIMESTAMP,
        }


@register_tool
class CreateTask(BaseTool):
    """Create a task or to-do item."""

    name = "create_task"
    schema = ToolSchema(
        name="create_task",
        description="Create a task or to-do item with a title, description, optional due date, and priority.",
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="title", type="string",
                description="Task title",
            ),
            ToolParameter(
                name="description", type="string",
                description="Task description",
                required=False,
            ),
            ToolParameter(
                name="due_date", type="string",
                description="Due date in YYYY-MM-DD format",
                required=False,
            ),
            ToolParameter(
                name="priority", type="string",
                description="Task priority",
                enum=["low", "medium", "high"],
                default="medium",
                required=False,
            ),
        ],
        returns="Confirmation with task ID and details",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._confirm(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._confirm(**kwargs)

    def _confirm(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "created",
            "task_id": _det_id("task", title=kwargs["title"]),
            "title": kwargs["title"],
            "description": kwargs.get("description", ""),
            "due_date": kwargs.get("due_date"),
            "priority": kwargs.get("priority", "medium"),
            "created_at": _FIXED_TIMESTAMP,
        }


@register_tool
class ScheduleMeeting(BaseTool):
    """Schedule a meeting with participants."""

    name = "schedule_meeting"
    schema = ToolSchema(
        name="schedule_meeting",
        description="Schedule a meeting with a title, date/time, duration, and list of participants. Returns a confirmation with meeting ID.",
        category=ToolCategory.COMMUNICATION,
        parameters=[
            ToolParameter(
                name="title", type="string",
                description="Meeting title (e.g. 'Q1 Review', 'Team Standup')",
            ),
            ToolParameter(
                name="datetime_str", type="string",
                description="Meeting date and time (e.g. '2026-03-20 14:00' or '2026-03-20T14:00:00')",
            ),
            ToolParameter(
                name="duration_minutes", type="integer",
                description="Meeting duration in minutes (default 30)",
                required=False,
                default=30,
            ),
            ToolParameter(
                name="participants", type="array",
                description="List of participant names or email addresses",
            ),
            ToolParameter(
                name="location", type="string",
                description="Meeting location or video call link",
                required=False,
            ),
        ],
        returns="Confirmation with meeting ID and details",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._schedule(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._schedule(**kwargs)

    def _schedule(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "status": "scheduled",
            "meeting_id": _det_id("mtg", title=kwargs["title"], dt=kwargs["datetime_str"]),
            "title": kwargs["title"],
            "datetime": kwargs["datetime_str"],
            "duration_minutes": int(kwargs.get("duration_minutes", 30)),
            "participants": kwargs["participants"],
            "location": kwargs.get("location", "Virtual (link to be sent)"),
            "created_at": _FIXED_TIMESTAMP,
        }
