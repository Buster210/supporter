from __future__ import annotations

import re

_REF_PATTERN = re.compile(r"\[ref=e\d+\]")


def filter_interactive(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    filtered = [line for line in lines if _REF_PATTERN.search(line)]
    return "\n".join(filtered) if filtered else ""


def filter_compact(text: str, max_lines: int = 40) -> str:
    filtered = filter_interactive(text)
    if not filtered:
        return ""
    lines = filtered.splitlines()
    if len(lines) <= max_lines:
        return filtered
    half = max_lines // 2
    head = lines[:half]
    tail = lines[-half:]
    head.append(f"--- {len(lines) - max_lines} lines omitted ---")
    head.extend(tail)
    return "\n".join(head)
