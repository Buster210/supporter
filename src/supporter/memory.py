"""Persistent working memory for the assistant.

This is the *runtime* memory the assistant reads on startup and writes to
during a session. It is separate from the long-form ``project_memory``
(which stores distilled insights from completed delegations).

The shape is intentionally tiny: append-only JSONL with one record per
write. The agent writes a *note* whenever it learns something a future
restart should know — a half-finished task, a recent URL, the user's
preferred working directory, a fingerprint of the last working build, etc.

Properties
----------

* **Crash-safe.** Writes are atomic (``tmp + rename``).
* **Append-only.** No in-place mutation; the only "delete" is compaction,
  which writes a fresh file and is itself atomic.
* **Bounded.** A per-file line cap keeps the file small and the read fast.
* **Tagged.** Each note has a *kind* (string) and a free-form *value* (JSON).
* **Searchable.** A trivial in-memory index sorts notes by kind and time so
  the assistant can ask "what did I learn about kind=X recently" in O(n).
* **No LLM in the loop.** Reads and writes are pure IO + dict.

Storage
-------

Default path: ``<project_root>/.supporter/working_memory.jsonl``. Override
with the ``WORKING_MEMORY_PATH`` env var.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import config
from .logger import logger

__all__ = [
    "DEFAULT_PATH",
    "Note",
    "WorkingMemory",
    "append_note",
    "clear_memory",
    "list_notes",
    "memory_snapshot",
    "search_notes",
]


DEFAULT_PATH = ".supporter/working_memory.jsonl"

# How many notes to keep on disk per kind (newest first). Notes are
# compacted only when total records exceed this and ``compact()`` is called.
_MAX_TOTAL_NOTES = 5000
_MAX_NOTE_VALUE_CHARS = 8000


# ---------------------------------------------------------------------------
# Note
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Note:
    timestamp: str
    kind: str
    value: dict[str, Any]
    # Optional free-text label for human inspection.
    label: str = ""
    # Optional reference to the interaction / job this note came from.
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Note | None:
        if not isinstance(raw, dict):
            return None
        kind = raw.get("kind")
        value = raw.get("value")
        if not isinstance(kind, str) or not isinstance(value, dict):
            return None
        return cls(
            timestamp=str(raw.get("timestamp", "")),
            kind=kind,
            value=value,
            label=str(raw.get("label", "")),
            source=str(raw.get("source", "")),
        )


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _memory_path() -> Path:
    raw = os.getenv("WORKING_MEMORY_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    if not config.allowed_directories:
        return Path.cwd() / DEFAULT_PATH
    root = Path(config.allowed_directories[0]).expanduser().resolve()
    return root / DEFAULT_PATH


# ---------------------------------------------------------------------------
# WorkingMemory store
# ---------------------------------------------------------------------------


@dataclass
class _MemoryState:
    path: Path
    notes: deque[Note] = field(default_factory=deque)
    # Index by kind for fast ``list(kind=...)`` queries.
    by_kind: dict[str, list[Note]] = field(default_factory=dict)


class WorkingMemory:
    """Process-wide working memory store.

    A single instance is shared across the process; all methods are
    thread-safe (a lock guards disk + in-memory updates).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path: Path = (
            Path(path).expanduser().resolve()
            if path is not None
            else _memory_path()
        )
        self._lock = threading.RLock()
        self._notes: deque[Note] = deque(maxlen=_MAX_TOTAL_NOTES)
        self._by_kind: dict[str, list[Note]] = {}
        self._load()

    # --- public API -------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    def append(
        self,
        kind: str,
        value: dict[str, Any],
        *,
        label: str = "",
        source: str = "",
    ) -> Note:
        if not kind or not isinstance(kind, str):
            raise ValueError("kind must be a non-empty string")
        if not isinstance(value, dict):
            raise ValueError("value must be a dict")
        text_value = json.dumps(value, ensure_ascii=False)
        if len(text_value) > _MAX_NOTE_VALUE_CHARS:
            raise ValueError(
                f"value too large ({len(text_value)} > {_MAX_NOTE_VALUE_CHARS} chars)"
            )
        note = Note(
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            kind=kind,
            value=value,
            label=label[:200] if label else "",
            source=source[:200] if source else "",
        )
        with self._lock:
            self._notes.appendleft(note)  # newest first
            self._by_kind.setdefault(kind, []).insert(0, note)
            self._persist_locked()
        logger.debug(f"WorkingMemory: appended kind={kind} label={label!r}")
        return note

    def list_notes(
        self,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[Note]:
        with self._lock:
            pool: list[Note]
            if kind is None:
                pool = list(self._notes)
            else:
                pool = list(self._by_kind.get(kind, []))
            if limit is not None and limit >= 0:
                pool = pool[:limit]
            return pool

    def search(
        self,
        query: str,
        *,
        kind: str | None = None,
        limit: int = 20,
    ) -> list[Note]:
        """Return notes whose label, kind, or stringified value contains
        ``query`` (case-insensitive substring match).
        """
        if not query:
            return []
        needle = query.lower()
        pattern = re.compile(re.escape(needle), re.IGNORECASE)
        with self._lock:
            candidates: list[Note] = (
                self._by_kind.get(kind, [])
                if kind
                else list(self._notes)
            )
            hits: list[Note] = []
            for note in candidates:
                haystack = " ".join(
                    [
                        note.kind,
                        note.label,
                        note.source,
                        json.dumps(note.value, ensure_ascii=False),
                    ]
                )
                if pattern.search(haystack):
                    hits.append(note)
                    if len(hits) >= limit:
                        break
            return hits

    def clear(self) -> None:
        with self._lock:
            self._notes.clear()
            self._by_kind.clear()
            self._persist_locked()

    def compact(self) -> int:
        """Drop the oldest half of records to keep the file bounded.

        Returns the number of notes removed.
        """
        with self._lock:
            current = len(self._notes)
            if current < 4:
                return 0
            keep = current // 2
            kept: list[Note] = []
            for _ in range(keep):
                kept.append(self._notes.popleft())
            self._notes.clear()
            for note in kept:
                self._notes.append(note)
            self._rebuild_index_locked()
            self._persist_locked()
            removed = current - keep
            logger.info(f"WorkingMemory: compacted; removed {removed} oldest notes")
            return removed

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counts = {k: len(v) for k, v in self._by_kind.items()}
            return {
                "path": str(self._path),
                "total": len(self._notes),
                "kinds": counts,
                "snapshot_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }

    def list(
        self,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[Note]:
        return self.list_notes(kind=kind, limit=limit)

    # --- internals --------------------------------------------------------

    def _rebuild_index_locked(self) -> None:
        self._by_kind.clear()
        for note in self._notes:
            self._by_kind.setdefault(note.kind, []).append(note)

    def _load(self) -> None:
        path = self._path
        if not path.exists():
            return
        try:
            with path.open(encoding="utf-8") as f:
                lines = f.readlines()
        except OSError as exc:
            logger.debug(f"WorkingMemory: read failed [{type(exc).__name__}]: {exc}")
            return
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("WorkingMemory: skipping malformed line")
                continue
            note = Note.from_dict(record)
            if note is None:
                continue
            self._notes.append(note)  # oldest at right
        # Reverse to newest-first
        self._notes = deque(reversed(self._notes), maxlen=_MAX_TOTAL_NOTES)
        self._rebuild_index_locked()
        logger.info(f"WorkingMemory: loaded {len(self._notes)} notes from {path}")

    def _persist_locked(self) -> None:
        path = self._path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp")
            with tmp_path.open("w", encoding="utf-8") as f:
                for note in self._notes:
                    f.write(json.dumps(note.to_dict(), ensure_ascii=False) + "\n")
            tmp_path.replace(path)
        except OSError as exc:
            logger.debug(f"WorkingMemory: persist failed [{type(exc).__name__}]: {exc}")


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------


_MEMORY_SINGLETON: WorkingMemory | None = None
_MEMORY_LOCK = threading.Lock()


def _get_memory() -> WorkingMemory | None:
    """Return the process-wide :class:`WorkingMemory`, or ``None`` if the
    config does not allow any directory (extremely unusual).
    """
    global _MEMORY_SINGLETON
    with _MEMORY_LOCK:
        if _MEMORY_SINGLETON is None:
            try:
                _MEMORY_SINGLETON = WorkingMemory()
            except Exception as exc:
                logger.debug(
                    f"WorkingMemory: init failed [{type(exc).__name__}]: {exc}"
                )
                return None
        return _MEMORY_SINGLETON


def append_note(
    kind: str,
    value: dict[str, Any],
    *,
    label: str = "",
    source: str = "",
) -> Note | None:
    """Append a note to working memory. Returns the persisted Note, or
    ``None`` if the memory store is unavailable.
    """
    memory = _get_memory()
    if memory is None:
        return None
    return memory.append(kind, value, label=label, source=source)


def list_notes(
    kind: str | None = None,
    limit: int | None = None,
) -> list[Note]:
    memory = _get_memory()
    if memory is None:
        return []
    return memory.list(kind=kind, limit=limit)


def search_notes(
    query: str,
    *,
    kind: str | None = None,
    limit: int = 20,
) -> list[Note]:
    memory = _get_memory()
    if memory is None:
        return []
    return memory.search(query, kind=kind, limit=limit)


def clear_memory() -> None:
    memory = _get_memory()
    if memory is not None:
        memory.clear()


def memory_snapshot() -> dict[str, Any]:
    memory = _get_memory()
    if memory is None:
        return {"available": False}
    snap = memory.snapshot()
    snap["available"] = True
    return snap
