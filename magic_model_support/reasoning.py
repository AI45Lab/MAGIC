"""Ministral native reasoning serialization and visible-answer parsing."""

from __future__ import annotations

from typing import Any


def mistral_reasoning_api_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Encode reasoning examples using mistral-common's structured API schema."""

    encoded: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if copied.get("role") == "system" and isinstance(content, str):
            if content.count("[THINK]") == 1 and content.count("[/THINK]") == 1:
                before, remainder = content.split("[THINK]", 1)
                thinking, after = remainder.split("[/THINK]", 1)
                copied["content"] = [
                    {"type": "text", "text": before},
                    {"type": "thinking", "thinking": thinking, "closed": True},
                    {"type": "text", "text": after},
                ]
        encoded.append(copied)
    return encoded


def split_native_reasoning(text: str | None) -> tuple[str, str]:
    """Split Ministral reasoning at the last closing special-token marker."""

    content = (text or "").strip()
    if not content.startswith("[THINK]") or "[/THINK]" not in content:
        return "", content
    reasoning, answer = content[len("[THINK]") :].rsplit("[/THINK]", 1)
    return reasoning.strip(), answer.strip()


def parse_native_reasoning_response(
    text: str | None,
) -> tuple[str, str, str | None]:
    """Return reasoning, user-visible response, and a diagnostic issue.

    A native trace that never emits ``[/THINK]`` has no user-visible response.
    A non-empty direct answer remains a usable response, while its missing
    native markers are retained as diagnostics.
    """

    content = (text or "").strip()
    if not content:
        return "", "", "empty_output"
    if not content.startswith("[THINK]"):
        return "", content, "missing_native_markers"
    if "[/THINK]" not in content:
        return "", "", "missing_close_marker"

    reasoning, visible = split_native_reasoning(content)
    if not visible:
        return reasoning, "", "empty_visible_response"
    if not reasoning:
        return "", visible, "empty_reasoning"
    return reasoning, visible, None
