"""Health-aware Gemini API key pool.

The existing :mod:`supporter.pool` already rotates keys per *request*; this
module adds the *missing* layer: long-lived, persisted health state so a key
that just hit a free-tier quota stays cold for a configurable window instead
of being retried on every request. The pool is *additive* — it can be
imported by callers that want a curated key without disturbing the existing
rotation path.

Design constraints
------------------

* **Zero-LLM health detection.** Errors are classified with regex / status-code
  matching — no model involved in deciding whether a key is sick.
* **Free-tier aware.** Gemini's free tier returns ``429 RESOURCE_EXHAUSTED`` and
  ``"Quota exceeded for ... per minute"`` style messages. Those go on a longer
  cooldown than transient 5xx; explicit ``FREE_TIER`` quota strings are
  scheduled with a *back-off* schedule so a key that has been hammered waits
  longer each time it fails.
* **Crash-safe state.** Health state is persisted as JSON next to the existing
  ``.supporter`` directory; re-loading on process start so a key that was hot
  five minutes ago is still hot now.
* **No background tasks.** Everything is synchronous / on-demand. A long-lived
  process that wants a background flusher can use the public ``flush()`` hook.
"""

from __future__ import annotations

import contextlib
import json
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import config
from .logger import logger

__all__ = [
    "KeyHealth",
    "KeyPool",
    "coerce_keys",
    "get_key_pool",
    "reset_key_pool",
]


# ---------------------------------------------------------------------------
# Free-tier / error classification
# ---------------------------------------------------------------------------

# Hard errors that mean the key itself is bad or revoked — these are essentially
# permanent; cooldown is a long time.
_REVOKED_PATTERNS = (
    re.compile(r"API key not valid", re.I),
    re.compile(r"API_KEY_INVALID", re.I),
    re.compile(r"PERMISSION_DENIED", re.I),
)

# Free-tier quota messages — Gemini's free tier returns these on per-minute /
# per-day rate limits. They are recoverable; we put the key on a *back-off*
# schedule (1m, 5m, 15m, 60m).
_FREE_TIER_PATTERNS = (
    re.compile(r"free tier", re.I),
    re.compile(r"Quota exceeded", re.I),
    re.compile(r"quota_metric", re.I),
    re.compile(r"RESOURCE_EXHAUSTED", re.I),
    re.compile(r"rate limit", re.I),
)

# Transient model / transport errors — short cooldown.
_TRANSIENT_PATTERNS = (
    re.compile(r"internal error", re.I),
    re.compile(r"unavailable", re.I),
    re.compile(r"overloaded", re.I),
    re.compile(r"\b5\d\d\b"),  # any "503"/"502"/"500" substring
)


@dataclass(frozen=True)
class KeyHealth:
    """One key's last known state."""

    key: str
    # Last unix-time (seconds) we *wrote* this entry; used for freshness checks.
    updated_at: float
    # 0 == healthy, 1+ == consecutive failures. Capped at FREE_TIER_BACKOFF steps.
    failure_streak: int = 0
    # Unix-time the key becomes healthy again. 0 == healthy now.
    cooldown_until: float = 0.0
    # Last category we classified the key under — for diagnostics only.
    last_category: str = "ok"
    # Last error snippet (truncated, secrets redacted).
    last_error: str = ""

    def is_available(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return self.cooldown_until <= now

    def seconds_to_recovery(self, now: float | None = None) -> float:
        now = now if now is not None else time.time()
        return max(0.0, self.cooldown_until - now)


# Back-off schedule (seconds) for free-tier quota failures. We start at 60s and
# quadruple, capped at 60 minutes.
_FREE_TIER_BACKOFF: tuple[int, ...] = (60, 240, 900, 3600)
_TRANSIENT_COOLDOWN_SECONDS = 30
_REVOKED_COOLDOWN_SECONDS = 24 * 60 * 60  # 1 day — effectively permanent


# ---------------------------------------------------------------------------
# KeyPool
# ---------------------------------------------------------------------------


@dataclass
class _KeyPoolState:
    """Persisted state: only the in-cooldown entries."""

    schema_version: int = 1
    updated_at: float = 0.0
    entries: dict[str, KeyHealth] = field(default_factory=dict)


def _default_state_path() -> Path:
    """State lives beside the durable history directory."""
    history_dir = Path(config.history_dir)
    if not history_dir.is_absolute():
        history_dir = (Path.cwd() / history_dir).resolve()
    return history_dir.parent / "keypool.json"


def coerce_keys(value: Any) -> list[str]:
    """Tolerate a comma-separated string, a JSON array, or an iterable of strings.

    Raises :class:`ValueError` on any unrecognised shape so misconfigurations
    surface at startup, not on the first request.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"key list is not valid JSON: {exc}") from exc
            if not isinstance(parsed, list) or not all(
                isinstance(k, str) and k.strip() for k in parsed
            ):
                raise ValueError("key list must be a JSON array of strings")
            return [k.strip() for k in parsed]
        return [k.strip() for k in text.split(",") if k.strip()]
    if isinstance(value, list):
        keys: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("key list must contain non-empty strings")
            keys.append(item.strip())
        return keys
    if isinstance(value, tuple):
        return coerce_keys(list(value))
    raise ValueError(f"unsupported key list shape: {type(value).__name__}")


def _redact_key(text: str, key: str) -> str:
    if not text or not key:
        return text
    return text.replace(key, "***")


def _classify(error: BaseException) -> str:
    """Return a category string for an exception.

    Categories:
        ``revoked`` — key is bad; long cooldown.
        ``free_tier`` — free-tier quota hit; back-off schedule.
        ``transient`` — 5xx / overload; short cooldown.
        ``unknown`` — could not classify; very short cooldown to be safe.
    """
    message = str(error)
    status = getattr(error, "status", None) or getattr(error, "code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        # 4xx (not 408 / 429) is client-error and likely the key itself.
        return "revoked"
    for pat in _REVOKED_PATTERNS:
        if pat.search(message):
            return "revoked"
    for pat in _FREE_TIER_PATTERNS:
        if pat.search(message):
            return "free_tier"
    if isinstance(status, int) and status == 429:
        return "free_tier"
    for pat in _TRANSIENT_PATTERNS:
        if pat.search(message):
            return "transient"
    return "unknown"


def _cooldown_seconds(category: str, streak: int) -> int:
    if category == "revoked":
        return _REVOKED_COOLDOWN_SECONDS
    if category == "transient":
        return _TRANSIENT_COOLDOWN_SECONDS
    if category == "unknown":
        return min(60, _TRANSIENT_COOLDOWN_SECONDS)
    # free_tier: back-off schedule
    idx = max(0, min(streak - 1, len(_FREE_TIER_BACKOFF) - 1))
    return _FREE_TIER_BACKOFF[idx]


class KeyPool:
    """A health-aware, crash-safe, on-demand key pool.

    Use :meth:`acquire` to get a healthy key (the *least-recently-used* healthy
    one), :meth:`report_failure` to mark it sick, and :meth:`report_success` to
    reset its failure streak. Health state persists to a small JSON file so a
    freshly-started process still respects cooldowns set minutes ago.
    """

    def __init__(
        self,
        keys: Iterable[str],
        *,
        state_path: Path | None = None,
    ) -> None:
        keys_list = [k for k in (k.strip() for k in keys) if k]
        if not keys_list:
            raise ValueError("KeyPool requires at least one non-empty key")
        # Preserve caller's order, but de-duplicate.
        seen: set[str] = set()
        self._keys: list[str] = []
        for k in keys_list:
            if k in seen:
                continue
            seen.add(k)
            self._keys.append(k)
        self._state_path: Path = state_path or _default_state_path()
        self._lock = threading.RLock()
        self._state: _KeyPoolState = _KeyPoolState()
        self._cursor: int = 0
        self._load_state()

    # --- public API -------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._keys)

    def health(self, key: str) -> KeyHealth:
        with self._lock:
            entry = self._state.entries.get(key)
            if entry is None:
                return KeyHealth(key=key, updated_at=time.time())
            return entry

    def all_health(self) -> list[KeyHealth]:
        with self._lock:
            return [
                self._state.entries.get(k) or KeyHealth(key=k, updated_at=0.0)
                for k in self._keys
            ]

    def acquire(self) -> str | None:
        """Return a healthy key, or ``None`` if every key is in cooldown.

        Rotates the cursor across healthy keys in a stable order so a long-
        running process spreads load across the pool.
        """
        with self._lock:
            now = time.time()
            healthy = [k for k in self._keys if self._is_available_locked(k, now)]
            if not healthy:
                logger.debug("KeyPool: no healthy key available right now")
                return None
            # Start at the cursor modulo the healthy list, then advance.
            cursor = self._cursor % len(healthy)
            chosen = healthy[cursor]
            self._cursor = (cursor + 1) % len(healthy)
            return chosen

    def report_failure(self, key: str, error: BaseException) -> None:
        with self._lock:
            now = time.time()
            prev = self._state.entries.get(key)
            streak = (prev.failure_streak if prev else 0) + 1
            category = _classify(error)
            cooldown = _cooldown_seconds(category, streak)
            snippet = _redact_key(str(error)[:200], key)
            entry = KeyHealth(
                key=key,
                updated_at=now,
                failure_streak=streak,
                cooldown_until=now + cooldown,
                last_category=category,
                last_error=snippet,
            )
            self._state.entries[key] = entry
            self._state.updated_at = now
            logger.info(
                f"KeyPool: key ...{key[-4:]} failed ({category}) "
                f"streak={streak} cooldown={cooldown}s"
            )
            self._persist_locked()

    def report_success(self, key: str) -> None:
        with self._lock:
            prev = self._state.entries.get(key)
            if prev is None or prev.failure_streak == 0:
                return
            entry = KeyHealth(
                key=key,
                updated_at=time.time(),
                failure_streak=0,
                cooldown_until=0.0,
                last_category="ok",
                last_error="",
            )
            self._state.entries[key] = entry
            self._state.updated_at = time.time()
            logger.info(f"KeyPool: key ...{key[-4:]} recovered (streak reset)")
            self._persist_locked()

    def flush(self) -> None:
        """Force-write the current state. Useful at process exit."""
        with self._lock:
            self._persist_locked()

    # --- internals --------------------------------------------------------

    def _is_available_locked(self, key: str, now: float) -> bool:
        entry = self._state.entries.get(key)
        if entry is None:
            return True
        return entry.cooldown_until <= now

    def _load_state(self) -> None:
        path = self._state_path
        if not path.exists():
            return
        try:
            with path.open(encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug(f"KeyPool: could not read state file {path}: {exc}")
            return
        if not isinstance(raw, dict):
            return
        entries_raw = raw.get("entries")
        if not isinstance(entries_raw, dict):
            return
        for key, entry_raw in entries_raw.items():
            if not isinstance(key, str) or not isinstance(entry_raw, dict):
                continue
            try:
                self._state.entries[key] = KeyHealth(
                    key=key,
                    updated_at=float(entry_raw.get("updated_at", 0.0) or 0.0),
                    failure_streak=int(entry_raw.get("failure_streak", 0) or 0),
                    cooldown_until=float(entry_raw.get("cooldown_until", 0.0) or 0.0),
                    last_category=str(entry_raw.get("last_category", "ok")),
                    last_error=str(entry_raw.get("last_error", "")),
                )
            except (TypeError, ValueError) as exc:
                logger.debug(
                    f"KeyPool: dropping malformed entry for key ...{key[-4:]}: {exc}"
                )
        logger.info(
            f"KeyPool: loaded {len(self._state.entries)} cooldowns from {path}"
        )

    def _persist_locked(self) -> None:
        path = self._state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp")
            payload = {
                "schema_version": self._state.schema_version,
                "updated_at": self._state.updated_at,
                "entries": {k: asdict(v) for k, v in self._state.entries.items()},
            }
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            tmp_path.replace(path)
        except OSError as exc:
            logger.debug(f"KeyPool: persist failed [{type(exc).__name__}]: {exc}")


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_POOL_SINGLETON: KeyPool | None = None
_POOL_LOCK = threading.Lock()


def get_key_pool() -> KeyPool | None:
    """Return the process-wide :class:`KeyPool`, or ``None`` if no keys are configured.

    The pool is built lazily on first call so tests can change ``config`` first.
    """
    global _POOL_SINGLETON
    with _POOL_LOCK:
        if _POOL_SINGLETON is not None:
            return _POOL_SINGLETON
        keys = coerce_keys(config.gemini_api_keys)
        if not keys:
            return None
        _POOL_SINGLETON = KeyPool(keys)
        return _POOL_SINGLETON


def reset_key_pool() -> None:
    """Drop the singleton — test isolation only."""
    global _POOL_SINGLETON
    with _POOL_LOCK:
        if _POOL_SINGLETON is not None:
            with contextlib.suppress(Exception):
                _POOL_SINGLETON.flush()
        _POOL_SINGLETON = None


# ---------------------------------------------------------------------------
# Liveness snapshot
# ---------------------------------------------------------------------------


def pool_snapshot(pool: KeyPool | None = None) -> dict[str, Any]:
    """Return a small JSON-serialisable view of the pool's health.

    Useful for surfacing in the TUI / decision log so users can see *why* a
    key was skipped.
    """
    pool = pool or get_key_pool()
    if pool is None:
        return {"configured": False, "keys": []}
    out_keys: list[dict[str, Any]] = []
    for health in pool.all_health():
        out_keys.append(
            {
                "key_suffix": health.key[-4:] if health.key else "????",
                "available": health.is_available(),
                "cooldown_in_s": round(health.seconds_to_recovery(), 1),
                "failure_streak": health.failure_streak,
                "category": health.last_category,
            }
        )
    return {
        "configured": True,
        "total": len(out_keys),
        "available": sum(1 for k in out_keys if k["available"]),
        "keys": out_keys,
        "snapshot_at": datetime.now().isoformat(timespec="seconds"),
    }
