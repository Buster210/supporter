from __future__ import annotations

import os
import random
import re
from collections.abc import Awaitable, Callable
from typing import Any, Final

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
GAP_MIN: Final = 0.8
GAP_MAX: Final = 4.5

ACTIONS_PER_MINUTE_MAX: Final = _env_int("BROWSER_ACTIONS_PER_MIN", 24)
SESSION_IDLE_GAP_PROBABILITY: Final = 0.04
SESSION_IDLE_GAP_RANGE: Final = (
    _env_float("BROWSER_IDLE_GAP_MIN", 5.0),
    60.0,
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


def host_from_url(url: str) -> str:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.hostname or ""
        return host.lower().removeprefix("www.")
    except Exception:
        return ""


def host_is_fast(host: str) -> bool:
    if not host:
        return False
    return host.removeprefix("www.") in FAST_HOSTS


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
    return humanize._lognormal_delay(median=1.2, sigma=0.5, lo=GAP_MIN, hi=GAP_MAX)


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
