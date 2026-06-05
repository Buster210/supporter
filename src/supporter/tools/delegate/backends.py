"""Pluggable delegation backends.

Each delegated task names a backend that decides *who* executes it. The Gemini
sub-agent roster stays native (the brain and its QA gate); coding-class work is
meant to delegate to external non-interactive agent CLIs. A task selects its
backend via the optional ``backend`` field (default ``"gemini"``); the harness
validates it against ``KNOWN_BACKENDS`` and dispatches deterministically.
"""

GEMINI_BACKEND = "gemini"
OPENCODE_BACKEND = "opencode"

KNOWN_BACKENDS: frozenset[str] = frozenset({GEMINI_BACKEND, OPENCODE_BACKEND})
