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

# Prefix the QA gate (qa_gate.run_qa_gate) writes on a rejected task's output so a
# rejection is distinguishable from other failures. Lives here -- the delegate
# package's dependency-free constants leaf -- so the producer (qa_gate) and the
# consumer (metrics, which counts qa_rejections by matching it) share one source
# and cannot drift on a wording change.
QA_REJECTION_MARKER = "QA gate rejected"
