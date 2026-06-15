"""Deterministic JSON-to-markdown rendering for delegated task results (SPEC §8).

Parsed capsule fields (evidence, findings) are rendered to markdown here rather
than dumped as inline JSON, so the same structured result always produces the
same display output — the harness owns the rendering, never the model.
"""

from __future__ import annotations

from typing import Any

from .capsule import EVIDENCE_KEYS, preview

_EVIDENCE_LABELS = {
    "files_read": "Files read",
    "files_changed": "Files changed",
    "commands_run": "Commands run",
    "sources": "Sources",
}
_ITEM_PREVIEW_CHARS = 600  # ponytail: raised from 200 for fuller findings
_MAX_ITEMS = 50  # ponytail: raised from 20 for more evidence/findings


def _render_items(items: list[Any]) -> tuple[str, int]:
    shown = ", ".join(
        preview(str(item), _ITEM_PREVIEW_CHARS) for item in items[:_MAX_ITEMS]
    )
    return shown, max(0, len(items) - _MAX_ITEMS)


def render_evidence(evidence: Any) -> str:
    """Render an evidence dict as a markdown sub-list in stable key order.

    Keys are emitted in ``EVIDENCE_KEYS`` order — not dict-insertion order — so
    identical evidence always renders identically. Empty groups are omitted.
    """
    if not isinstance(evidence, dict):
        evidence = {}
    sections: list[str] = []
    for key in EVIDENCE_KEYS:
        items = evidence.get(key, [])
        if not isinstance(items, list) or not items:
            continue
        shown, extra = _render_items(items)
        if extra:
            shown += f" (+{extra} more)"
        sections.append(f"  - {_EVIDENCE_LABELS.get(key, key)}: {shown}")
    if not sections:
        return "- Evidence: none"
    return "- Evidence:\n" + "\n".join(sections)


def render_findings(findings: Any) -> str:
    """Render a findings list as a markdown sub-list, truncated deterministically."""
    if not isinstance(findings, list) or not findings:
        return "- Findings: none"
    lines = [
        f"  - {preview(str(item), _ITEM_PREVIEW_CHARS)}"
        for item in findings[:_MAX_ITEMS]
    ]
    extra = len(findings) - _MAX_ITEMS
    if extra > 0:
        lines.append(f"  - (+{extra} more)")
    return "- Findings:\n" + "\n".join(lines)
