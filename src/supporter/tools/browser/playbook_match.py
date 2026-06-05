from __future__ import annotations

from .playbook_store import _list_playbooks_sync, _normalize_name
from .snapshot import _NAME_PATTERN, _REF_GROUP, _ROLE_PATTERN

__all__ = [
    "_find_ref",
    "_find_ref_fuzzy",
    "_fuzzy_find_goal",
    "_name_match_score",
]

_FUZZY_AMBIGUITY_MARGIN = 0.15


def _name_match_score(want: str, want_tokens: set[str], found: str) -> float:
    if want == found:
        return 1.0
    found_tokens = set(found.split())
    if want_tokens and want_tokens.issubset(found_tokens):
        return 0.6 + 0.3 * (len(want_tokens) / len(found_tokens))
    if want in found:
        return 0.3 + 0.2 * (len(want) / len(found))
    return 0.0


def _find_ref(snapshot_text: str, role: str, name: str) -> str:
    for raw in snapshot_text.splitlines():
        body = raw.lstrip().removeprefix("- ")
        role_match = _ROLE_PATTERN.match(body)
        if (role_match.group(1) if role_match else "") != role:
            continue
        name_match = _NAME_PATTERN.search(body)
        if (name_match.group(1) if name_match else "") != name:
            continue
        ref_match = _REF_GROUP.search(body)
        if ref_match:
            return ref_match.group(1)
    return ""


def _find_ref_fuzzy(snapshot_text: str, role: str, name: str) -> str:
    want = _normalize_name(name)
    if not want:
        return ""
    want_tokens = set(want.split())

    best_ref = ""
    best_score = 0.0
    runner_up = 0.0
    for raw in snapshot_text.splitlines():
        body = raw.lstrip().removeprefix("- ")
        role_match = _ROLE_PATTERN.match(body)
        if (role_match.group(1) if role_match else "") != role:
            continue
        name_match = _NAME_PATTERN.search(body)
        found = _normalize_name(name_match.group(1) if name_match else "")
        if not found:
            continue
        score = _name_match_score(want, want_tokens, found)
        if score <= 0.0:
            continue
        ref_match = _REF_GROUP.search(body)
        if ref_match is None:
            continue
        if score > best_score:
            best_score, runner_up, best_ref = score, best_score, ref_match.group(1)
        elif score > runner_up:
            runner_up = score

    if runner_up > 0.0 and best_score - runner_up < _FUZZY_AMBIGUITY_MARGIN:
        return ""
    return best_ref


def _fuzzy_find_goal(host: str, goal: str, limit: int = 5) -> list[str]:
    want = set(_normalize_name(goal).split())
    if not want:
        return []
    scored: list[tuple[float, str]] = []
    for descriptor in _list_playbooks_sync(host):
        candidate = str(descriptor["goal"])
        have = set(_normalize_name(candidate).split())
        union = want | have
        score = len(want & have) / len(union) if union else 0.0
        if score:
            scored.append((score, candidate))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored[:limit]]


def _no_playbook_message(host: str, goal: str) -> str:
    base = f"No playbook found for {goal!r} on {host}."
    candidates = _fuzzy_find_goal(host, goal)
    if not candidates:
        return base
    similar = "; ".join(repr(candidate) for candidate in candidates)
    return f"{base} Similar saved goals: {similar}. Retry with one of these."
