from __future__ import annotations

import json
import os
import random
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final

from ...config import config
from . import humanize

_UNSET: Final = object()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"${name} must be an integer, got: {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"${name} must be a float, got: {raw!r}") from exc


ACTION_CAP: Final = 25
ACTION_CAP_JITTER: Final = 15
GAP_MIN: Final = 0.4
GAP_MAX: Final = 2.5

ACTIONS_PER_MINUTE_MAX: Final = _env_int("BROWSER_ACTIONS_PER_MIN", 36)
SESSION_IDLE_GAP_PROBABILITY: Final = 0.02
SESSION_IDLE_GAP_RANGE: Final = (
    _env_float("BROWSER_IDLE_GAP_MIN", 3.0),
    15.0,
)

FATIGUE_MAX_BONUS: Final = 0.5
FATIGUE_PER_MINUTE: Final = 0.02
TEMPO_MIN: Final = 0.8
TEMPO_MAX: Final = 1.3
TEMPO_STEP: Final = 0.1

SENSITIVE_DOMAINS: Final = frozenset(
    {
        "accounts.google.com",
        "github.com",
        "twitter.com",
        "x.com",
        "facebook.com",
        "linkedin.com",
    }
)

FAST_HOSTS: Final[frozenset[str]] = frozenset(
    {
        "google.com",
        "gemini.google.com",
    }
)

SENSITIVE_ACTION_PATTERNS: Final = (
    "submit",
    "post",
    "send",
    "pay",
    "delete",
    "buy",
    "confirm",
    "sign-in",
    "signin",
    "login",
    "log-in",
    "create-account",
    "sign-up",
    "signup",
    "register",
)

SENSITIVE_FIELD_NAMES: Final = (
    "password",
    "passwd",
    "pass",
    "email",
    "card",
    "otp",
    "cvv",
    "credit",
    "cvc",
    "secret",
)

SENSITIVE_FIELD_ROLES: Final = frozenset({"textbox", "password"})

_PASSIVE_ACTIONS: Final = frozenset(
    {
        "navigate",
        "back",
        "forward",
        "snapshot",
        "screenshot",
        "scroll",
        "hover",
        "wait",
        "extract",
        "tabs",
        "tab",
        "newtab",
        "solve_cloudflare",
    }
)

_PATTERN_GATED_ACTIONS: Final = frozenset({"click", "press", "select"})

_ALWAYS_CONFIRM_ACTIONS: Final = frozenset({"eval", "upload", "download"})


def _word_boundary_pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    alts = "|".join(re.escape(w) for w in words)
    return re.compile(rf"(?<![a-z0-9])(?:{alts})(?![a-z0-9])", re.IGNORECASE)


_ACTION_PATTERN: Final = _word_boundary_pattern(SENSITIVE_ACTION_PATTERNS)
_FIELD_NAME_PATTERN: Final = _word_boundary_pattern(SENSITIVE_FIELD_NAMES)

browse_confirmation_callback: Callable[[str, str], Awaitable[bool]] | None = None
browse_image_sink: Callable[[bytes, str], Awaitable[None]] | None = None

browse_profile_select_callback: Callable[..., Awaitable[str | None]] | None = None


def register_browse_callback(
    *,
    confirmation: Callable[[str, str], Awaitable[bool]] | None | object = _UNSET,
    image_sink: Callable[[bytes, str], Awaitable[None]] | None | object = _UNSET,
    profile_select: Callable[[list[Any]], Awaitable[str | None]]
    | None
    | object = _UNSET,
) -> None:
    global browse_confirmation_callback, browse_image_sink
    global browse_profile_select_callback
    if confirmation is not _UNSET:
        browse_confirmation_callback = confirmation  # type: ignore[assignment]
    if image_sink is not _UNSET:
        browse_image_sink = image_sink  # type: ignore[assignment]
    if profile_select is not _UNSET:
        browse_profile_select_callback = profile_select  # type: ignore[assignment]


class TrustStore:
    """Persistent per-host trust state backed by ``~/.supporter/trusted.json``.

    Tracks clean-interaction counts, promotion status, and user decisions so
    that hosts can be auto-promoted (or suppressed) across browser sessions.
    """

    _store_path: Path
    _data: dict[str, dict[str, Any]]
    _dirty: bool

    def __init__(self) -> None:
        self._store_path = Path.home() / ".supporter" / "trusted.json"
        self._data = {}
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        try:
            raw = self._store_path.read_text()
            self._data = json.loads(raw)
        except FileNotFoundError, json.JSONDecodeError:
            self._data = {}

    def _save(self) -> None:
        if not self._dirty:
            return
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._store_path.write_text(json.dumps(self._data, indent=2, default=str))
        self._dirty = False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_confirmed(self, host: str) -> bool:
        return bool(self._data.get(host, {}).get("user_confirmed", False))

    def is_suppressed(self, host: str) -> bool:
        return bool(self._data.get(host, {}).get("suppressed", False))

    def clean_success_count(self, host: str) -> int:
        return int(self._data.get(host, {}).get("clean_success_count", 0))

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def record_clean(self, host: str) -> None:
        entry = self._data.setdefault(host, {})
        entry["clean_success_count"] = entry.get("clean_success_count", 0) + 1
        self._dirty = True

    def set_confirmed(self, host: str) -> None:
        entry = self._data.setdefault(host, {})
        entry["user_confirmed"] = True
        entry["suppressed"] = False
        self._dirty = True
        self._save()

    def set_suppressed(self, host: str) -> None:
        entry = self._data.setdefault(host, {})
        entry["suppressed"] = True
        self._dirty = True
        self._save()


_trust_store: TrustStore | None = None


def _get_trust_store() -> TrustStore:
    global _trust_store
    if _trust_store is None:
        _trust_store = TrustStore()
    return _trust_store


def host_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or ""
        return host.lower().removeprefix("www.")
    except Exception:
        return ""


def host_is_fast(host: str) -> bool:
    """Return ``True`` when the host is considered "fast" (skip humanization).

    Trust sources (in order):
    1. Hardcoded ``FAST_HOSTS`` set.
    2. ``config.browser_trusted_hosts`` (comma-separated env allowlist).
    3. User-confirmed hosts in the persistent ``TrustStore``.
    """
    if not host:
        return False
    clean = host.removeprefix("www.")

    # 1. Built-in fast hosts
    if clean in FAST_HOSTS:
        return True

    # 2. Config/env allowlist
    raw = getattr(config, "browser_trusted_hosts", "")
    if raw:
        for h in raw.split(","):
            if h.strip().lower().removeprefix("www.") == clean:
                return True

    # 3. User-confirmed via dialog (persisted trust store)
    return _get_trust_store().is_confirmed(clean)


async def record_clean_interaction(host: str) -> None:
    """Record one successful non-confirmation interaction for *host*.

    After ``browser_promotion_threshold`` clean interactions a dialog fires
    asking the user whether to trust this host. Accepting persists the
    decision to ``trusted.json``; declining suppresses further re-asking.
    """
    if not host:
        return
    clean = host.removeprefix("www.")

    # Already-trusted hosts (built-in, config allowlist, or confirmed) skip promotion
    if host_is_fast(clean):
        return

    store = _get_trust_store()
    store.record_clean(clean)

    count = store.clean_success_count(clean)
    threshold = config.browser_promotion_threshold

    if count < threshold:
        store._save()
        return

    if store.is_suppressed(clean):
        store._save()
        return
    # When auto-approve is on, trust comes only from explicit config /
    # trusted.json — never as a side-effect of auto-approved interactions.
    if getattr(config, "browser_auto_approve", False):
        return

    # Record the promotion attempt timestamp
    store._data.setdefault(clean, {})["last_promoted"] = time.time()
    store._save()

    # Fire the promotion dialog
    cb = browse_confirmation_callback
    if cb is None:
        return

    accepted = await cb(
        f"Trust browser host {clean}?",
        f"After {count} clean interactions, do you want to treat {clean} "
        f"as a trusted host? This skips human-like delays and reduces "
        f"confirmation prompts on this host. You can change this later in "
        f"~/.supporter/trusted.json.",
    )

    if accepted:
        store.set_confirmed(clean)
    else:
        store.set_suppressed(clean)


def needs_confirmation(
    action: str,
    role: str,
    name: str,
    host: str,
) -> bool:
    if action in _ALWAYS_CONFIRM_ACTIONS:
        return True

    if action in _PASSIVE_ACTIONS:
        return False

    if host in SENSITIVE_DOMAINS:
        return True

    role_lower = role.lower()

    if action == "type":
        if role_lower == "password":
            return True
        if role_lower in SENSITIVE_FIELD_ROLES and _FIELD_NAME_PATTERN.search(name):
            return True

    if action in _PATTERN_GATED_ACTIONS:
        return bool(_ACTION_PATTERN.search(f"{name} {role_lower}"))

    return False


def random_gap() -> float:
    return humanize._lognormal_delay(median=0.7, sigma=0.5, lo=GAP_MIN, hi=GAP_MAX)


def action_cap() -> int:
    return ACTION_CAP + random.randint(0, ACTION_CAP_JITTER)


def rate_throttle_delay(recent_action_count: int, window_seconds: float) -> float:
    if recent_action_count <= 1 or window_seconds <= 0.0:
        return 0.0
    min_window = recent_action_count / ACTIONS_PER_MINUTE_MAX * 60.0
    return max(0.0, min_window - window_seconds)


def maybe_idle_gap() -> float:
    if random.random() < SESSION_IDLE_GAP_PROBABILITY:
        return random.uniform(*SESSION_IDLE_GAP_RANGE)
    return 0.0


def fatigue_multiplier(session_minutes: float) -> float:
    bonus = min(FATIGUE_MAX_BONUS, max(0.0, session_minutes) * FATIGUE_PER_MINUTE)
    return 1.0 + bonus


def next_tempo(tempo: float) -> float:
    stepped = tempo + random.uniform(-TEMPO_STEP, TEMPO_STEP)
    return max(TEMPO_MIN, min(TEMPO_MAX, stepped))
