from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from ...logger import logger
from .. import resolved_project_root

MAX_INSIGHT_CHARS = 280
MAX_INSIGHTS = 50

_MEMORY_LOCK: asyncio.Lock | None = None


def _memory_lock() -> asyncio.Lock:
    global _MEMORY_LOCK
    if _MEMORY_LOCK is None:
        _MEMORY_LOCK = asyncio.Lock()
    return _MEMORY_LOCK


def _memory_path() -> Path:
    return resolved_project_root() / ".supporter" / "project_memory.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Insight(TypedDict):
    text: str
    source_job: str
    created_at: str


class ProjectMemory(TypedDict):
    schema_version: int
    updated_at: str
    insights: list[Insight]


def _empty_memory() -> ProjectMemory:
    return {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "insights": [],
    }


def _save_memory_sync(memory: ProjectMemory) -> None:
    path = _memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


async def _save_memory(memory: ProjectMemory) -> None:
    await asyncio.to_thread(_save_memory_sync, memory)


async def load_project_memory() -> ProjectMemory:
    """Load project memory from disk, returning empty structure on any error."""
    path = _memory_path()
    if not path.exists():
        return _empty_memory()
    try:
        with path.open(encoding="utf-8") as f:
            data: Any = json.load(f)
        if not isinstance(data, dict):
            return _empty_memory()
        # Validate shape
        if not isinstance(data.get("schema_version"), int):
            return _empty_memory()
        if not isinstance(data.get("insights"), list):
            return _empty_memory()
        insights = data.get("insights", [])
        valid_insights: list[Insight] = []
        for item in insights:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                valid_insights.append(
                    {
                        "text": str(item.get("text", "")),
                        "source_job": str(item.get("source_job", "")),
                        "created_at": str(item.get("created_at", "")),
                    }
                )
        return {
            "schema_version": data.get("schema_version", 1),
            "updated_at": str(data.get("updated_at", "")),
            "insights": valid_insights,
        }
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.debug(f"project memory unreadable, starting empty: {exc}")
        return _empty_memory()


def _normalize_text(text: str) -> str:
    """Normalize text for deduplication: strip + casefold."""
    return text.strip().casefold()


async def record_learnings(insights: list[str], source_job: str) -> None:
    """Persist insights from a completed milestone into project memory.

    - Deduplicates by normalized text (strip + casefold)
    - Truncates each insight to MAX_INSIGHT_CHARS
    - Caps total at MAX_INSIGHTS (newest kept)
    - Drops empty/whitespace-only insights
    """
    if not insights:
        return

    # Filter and truncate
    new_insights: list[Insight] = []
    for text in insights:
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        truncated = stripped[:MAX_INSIGHT_CHARS]
        new_insights.append(
            {
                "text": truncated,
                "source_job": source_job,
                "created_at": _utc_now(),
            }
        )

    if not new_insights:
        return

    memory = await load_project_memory()
    existing = memory.get("insights", [])
    seen_norm: set[str] = {_normalize_text(i["text"]) for i in existing}

    # Prepend new insights (recency-first), dedup by normalized text
    merged: list[Insight] = []
    for insight in new_insights:
        norm = _normalize_text(insight["text"])
        if norm not in seen_norm:
            merged.append(insight)
            seen_norm.add(norm)
    merged.extend(existing)

    # Cap at MAX_INSIGHTS (keep newest)
    capped = merged[:MAX_INSIGHTS]

    memory = {
        "schema_version": 1,
        "updated_at": _utc_now(),
        "insights": capped,
    }

    async with _memory_lock():
        await _save_memory(memory)


def memory_context_block(memory: ProjectMemory | None = None) -> str:
    """Render project memory as a compact block for task context.

    Returns empty string when memory has no insights.
    Bounded to ~1500 chars to avoid prompt bloat.
    """
    if memory is None:
        memory = {"schema_version": 1, "updated_at": "", "insights": []}
    insights = memory.get("insights", [])
    if not insights:
        return ""

    lines: list[str] = ["PROJECT MEMORY (learned from prior runs):"]
    total_len = len(lines[0]) + 1
    max_total = 1500

    for insight in insights:
        text = insight.get("text", "")
        if not text:
            continue
        bullet = f"• {text}"
        if total_len + len(bullet) + 1 > max_total:
            break
        lines.append(bullet)
        total_len += len(bullet) + 1

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


__all__ = [
    "MAX_INSIGHTS",
    "MAX_INSIGHT_CHARS",
    "Insight",
    "ProjectMemory",
    "load_project_memory",
    "memory_context_block",
    "record_learnings",
]
