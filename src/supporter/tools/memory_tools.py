"""Tool wrappers for the working-memory store.

These are intentionally thin: they translate LLM tool-call arguments into
:class:`supporter.memory` operations and shape the response as plain text
the orchestrator can read. The store is the source of truth; the tool
wrappers exist only to put a uniform ``str``/``dict`` interface in front
of it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from ..memory import (
    Note,
    append_note,
    list_notes,
    memory_snapshot,
    search_notes,
)
from ..memory import (
    _get_memory as _memory_singleton,
)
from ..memory import (
    clear_memory as _clear_memory,
)

__all__ = [
    "memory_clear",
    "memory_compact",
    "memory_list_kinds",
    "memory_read",
    "memory_render_block",
    "memory_search",
    "memory_status",
    "memory_write",
]

def _format_notes(notes: Iterable[Note]) -> str:
    notes_list = list(notes)
    if not notes_list:
        return "(no notes match)"
    lines: list[str] = []
    for note in notes_list:
        value_repr = json.dumps(note.value, ensure_ascii=False)
        if len(value_repr) > 240:
            value_repr = value_repr[:237] + "..."
        label_part = f" — {note.label}" if note.label else ""
        source_part = f" [source={note.source}]" if note.source else ""
        lines.append(
            f"- {note.timestamp} {note.kind}{label_part}{source_part}: {value_repr}"
        )
    return "\n".join(lines)


async def memory_write(
    kind: str,
    value_json: str,
    label: str = "",
    source: str = "",
) -> str:
    """Persist a note to the working-memory store.

    Parameters
    ----------
    kind:
        A short tag (e.g. ``"todo"``, ``"in_flight_task"``, ``"user_pref"``).
    value_json:
        A JSON object string describing what to remember.
    label, source:
        Optional human-readable metadata.
    """
    if not kind or not isinstance(kind, str):
        return "ERROR: kind must be a non-empty string"
    try:
        parsed = json.loads(value_json)
    except json.JSONDecodeError as exc:
        return f"ERROR: value_json is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return "ERROR: value_json must encode a JSON object"
    note = append_note(kind, parsed, label=label, source=source)
    if note is None:
        return "ERROR: working memory is not available"
    return f"ok: stored note (kind={kind}, label={label!r})"


async def memory_read(kind: str = "", limit: int = 20) -> str:
    """Read recent notes; optionally filtered by kind.

    Parameters
    ----------
    kind:
        If non-empty, only notes of this kind are returned.
    limit:
        Maximum number of notes to return (newest first).
    """
    notes = list_notes(kind=kind or None, limit=limit if limit > 0 else None)
    return _format_notes(notes)


async def memory_search(query: str, kind: str = "", limit: int = 20) -> str:
    """Search notes whose kind, label, source, or value contains ``query``."""
    if not query:
        return "ERROR: query must be non-empty"
    notes = search_notes(query, kind=kind or None, limit=limit if limit > 0 else 20)
    return _format_notes(notes)


async def memory_list_kinds() -> str:
    """Return one line per kind with its note count, most recent first."""
    snap = memory_snapshot()
    if not snap.get("available", False):
        return "ERROR: working memory is not available"
    kinds = snap.get("kinds", {})
    if not kinds:
        return "(no notes yet)"
    return "\n".join(
        f"- {kind}: {count}"
        for kind, count in sorted(kinds.items(), key=lambda x: -x[1])
    )


async def memory_compact() -> str:
    """Compact the store: drop the oldest half of notes to bound disk usage."""
    memory = _memory_singleton()
    if memory is None:
        return "ERROR: working memory is not available"
    removed = memory.compact()
    return f"compacted: {removed} oldest notes dropped"


async def memory_clear() -> str:
    """Wipe the working-memory store. Use with care — it cannot be undone."""
    _clear_memory()
    return "ok: working memory cleared"


async def memory_status() -> str:
    """Return a one-line summary of the memory store's state."""
    snap = memory_snapshot()
    if not snap.get("available", False):
        return "working memory: unavailable"
    return (
        f"working memory: total={snap.get('total', 0)} "
        f"path={snap.get('path', '?')} kinds={len(snap.get('kinds', {}))}"
    )


def memory_render_block(limit: int = 5) -> str:
    """Render a compact block of recent notes for prompt injection.

    Returns an empty string when no notes exist. Bounded output so it
    does not bloat prompts.
    """
    notes = list_notes(limit=limit)
    if not notes:
        return ""
    lines = ["RECENT WORKING MEMORY (most recent first):"]
    for note in notes:
        value_repr = json.dumps(note.value, ensure_ascii=False)
        if len(value_repr) > 240:
            value_repr = value_repr[:237] + "..."
        label_part = f" [{note.label}]" if note.label else ""
        lines.append(f"- {note.kind}{label_part}: {value_repr}")
    return "\n".join(lines)
