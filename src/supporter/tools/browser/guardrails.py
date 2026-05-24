from __future__ import annotations

import random
import re
from collections.abc import Awaitable, Callable
from typing import Final

_UNSET: Final = object()

ACTION_CAP: Final = 30
GAP_MIN: Final = 0.8
GAP_MAX: Final = 2.5

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


def register_browse_callback(
    *,
    confirmation: Callable[[str, str], Awaitable[bool]] | None | object = _UNSET,
    image_sink: Callable[[bytes, str], Awaitable[None]] | None | object = _UNSET,
) -> None:
    global browse_confirmation_callback, browse_image_sink
    if confirmation is not _UNSET:
        browse_confirmation_callback = confirmation  # type: ignore[assignment]
    if image_sink is not _UNSET:
        browse_image_sink = image_sink  # type: ignore[assignment]


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
    return random.uniform(GAP_MIN, GAP_MAX)
