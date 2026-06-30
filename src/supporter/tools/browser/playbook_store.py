from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...logger import logger
from .. import resolved_project_root

_SLUG_MAX = 80
_RESULT_HEAD_MAX = 120
SCHEMA_VERSION = 2

# Phase 5: Prune thresholds
_PRUNE_TTL_DAYS = 30
_PRUNE_FAIL_FLOOR = 3

_TARGET_ACTIONS = frozenset({"click", "type", "hover", "select", "press", "scroll"})

# Phase 3: extract is ref-resolvable (Bug 3 fix)
_REF_RESOLVABLE_ACTIONS = _TARGET_ACTIONS | {"extract"}

_RECORD_PARAMS: dict[str, tuple[str, ...]] = {
    "navigate": ("url",),
    "newtab": ("url",),
    "type": ("text",),
    "select": ("value", "text"),
    "press": ("key",),
    "scroll": ("dx", "dy"),
    "wait": ("selector", "delay_ms"),
    "extract": ("html",),
}


@dataclass
class Step:
    action: str
    role: str = ""
    name: str = ""
    selector: str = ""
    url_before: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    result_head: str = ""
    variable: str = ""  # B: model-annotated variable name (Phase 3)


@dataclass
class Playbook:
    host: str
    goal: str
    created_ts: float
    steps: list[Step]
    url_template: str = "/*"
    schema_version: int = SCHEMA_VERSION  # v1->v2 additive
    variables: list[str] = field(default_factory=list)  # B: template variables
    version: int = 1
    previous_version_ref: str = ""
    success_count: int = 0  # D: replay success metric
    fail_count: int = 0  # D: replay failure metric
    last_used_ts: float = 0.0  # D: last replay timestamp
    last_outcome: str = ""  # D: "success", "drift", "error", or "repaired"


def _slug(goal: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", goal.lower()).strip("-._")
    return (slug or "task")[:_SLUG_MAX]


def _safe_host(host: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", host.lower()).strip("-._")


def _normalize_name(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _memory_root() -> Path:
    return resolved_project_root() / ".supporter" / "task_memory"


def _safe_path(host: str, goal: str) -> Path:
    root = _memory_root().resolve()
    path = (root / _safe_host(host) / f"{_slug(goal)}.json").resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"Playbook path '{path}' escapes memory root '{root}'.")
    return path


def _save_playbook_sync(playbook: Playbook) -> None:
    path = _safe_path(playbook.host, playbook.goal)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    # Stamp current schema version on write (Phase 1)
    playbook.schema_version = SCHEMA_VERSION
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(playbook), f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


async def save_playbook(playbook: Playbook) -> None:
    await asyncio.to_thread(_save_playbook_sync, playbook)




def _archive_playbook_sync(host: str, goal: str) -> str | None:
    """Archive current playbook before repair; returns archive path or None."""
    try:
        path = _safe_path(host, goal)
        if not path.is_file():
            return None
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        ver = data.get("version", 1)
        archive_path = path.parent / f"{_slug(goal)}.v{ver}.json"
        import shutil

        shutil.copy2(str(path), str(archive_path))
        return str(archive_path)
    except Exception as exc:
        logger.warning(f"Failed to archive playbook {host}/{goal}: {exc}")
        return None


async def save_playbook_version(playbook: Playbook) -> str:
    """Save a new version of a playbook, archiving the current one first.

    Returns the archive path or empty string if no archive was needed.
    """
    archive_path = await asyncio.to_thread(
        _archive_playbook_sync, playbook.host, playbook.goal
    )
    await save_playbook(playbook)
    return archive_path or ""


def load_playbook(host: str, goal: str) -> Playbook | None:
    try:
        path = _safe_path(host, goal)
        with path.open(encoding="utf-8") as f:
            data: Any = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("playbook is not a JSON object")
        # Phase 1: field-filter Step for forward-compat, .get defaults for Playbook
        known_step_fields = {f.name for f in Step.__dataclass_fields__.values()}
        steps = [
            Step(**{k: v for k, v in s.items() if k in known_step_fields})
            for s in data["steps"]
        ]
        return Playbook(
            host=str(data["host"]),
            goal=str(data["goal"]),
            created_ts=float(data["created_ts"]),
            steps=steps,
            schema_version=data.get("schema_version", 1),
            variables=data.get("variables", []),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            last_used_ts=data.get("last_used_ts", 0.0),
            last_outcome=data.get("last_outcome", ""),
            url_template=data.get("url_template", "/*"),
            version=data.get("version", 1),
            previous_version_ref=data.get("previous_version_ref", ""),
        )
    except (
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
        KeyError,
    ):
        logger.debug(f"No usable playbook for {goal!r} on {host}", exc_info=True)
        return None


def format_playbook(playbook: Playbook) -> str:
    lines = [f"Playbook for {playbook.goal!r} on {playbook.host}:"]
    for i, step in enumerate(playbook.steps, 1):
        target = step.selector or (
            f"{step.role} {step.name!r}".strip() if step.role or step.name else ""
        )
        detail = f"{step.action}" + (f" → {target}" if target else "")
        params = {k: v for k, v in step.params.items() if v not in ("", 0, False)}
        if params:
            detail += f" {params}"
        lines.append(f"  {i}. {detail}")
    return "\n".join(lines)


def build_step(
    action: str,
    *,
    role: str = "",
    name: str = "",
    selector: str = "",
    url_before: str = "",
    params: dict[str, Any] | None = None,
    result: str = "",
    variable: str = "",  # B: model-annotated variable name
) -> Step:
    return Step(
        action=action,
        role=role,
        name=name,
        selector=selector,
        url_before=url_before,
        params={k: v for k, v in (params or {}).items() if v not in ("", 0, False)},
        result_head=result[:_RESULT_HEAD_MAX],
        variable=variable,
    )


def _list_playbooks_sync(host: str) -> list[dict[str, Any]]:
    root = _memory_root().resolve()
    host_dir = root / _safe_host(host)
    if not host_dir.is_dir():
        return []
    descriptors: list[dict[str, Any]] = []
    for path in host_dir.glob("*.json"):
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                continue
            pb = Playbook(
                host=str(data.get("host", "")),
                goal=str(data.get("goal", "")),
                created_ts=float(data.get("created_ts", 0.0)),
                steps=[],
                schema_version=data.get("schema_version", 1),
                variables=data.get("variables", []),
                success_count=data.get("success_count", 0),
                fail_count=data.get("fail_count", 0),
                last_used_ts=float(data.get("last_used_ts", 0.0)),
                last_outcome=data.get("last_outcome", ""),
            )
            descriptors.append(
                {
                    "goal": pb.goal,
                    "slug": _slug(pb.goal),
                    "step_count": len(data.get("steps", [])),
                    "variables": pb.variables,
                    "url_template": data.get("url_template", "/*"),
                    "version": data.get("version", 1),
                    "success_count": pb.success_count,
                    "fail_count": pb.fail_count,
                    "last_used_ts": pb.last_used_ts,
                    "created_ts": pb.created_ts,
                }
            )
        except json.JSONDecodeError, KeyError, TypeError, ValueError, OSError:
            continue
    # Rank by recency / net success
    descriptors.sort(
        key=lambda d: (
            d["last_used_ts"] or d["created_ts"],
            d["success_count"] - d["fail_count"],
        ),
        reverse=True,
    )
    return descriptors


def _delete_playbook_sync(host: str, goal: str) -> bool:
    try:
        path = _safe_path(host, goal)
    except ValueError:
        return False
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True




def _normalize_url_path(url: str) -> str:
    """Normalize a URL path by replacing volatile segments with wildcards.

    Example: /search?q=foo&page=2 → /search?q=*&page=*
             /profile/123/edit → /profile/*/edit
    """
    from urllib.parse import parse_qs, urlparse

    parsed = urlparse(url)
    path = parsed.path
    segments = path.split("/")
    normalized_segments = []
    for seg in segments:
        if seg.isdigit():
            normalized_segments.append("*")
        else:
            normalized_segments.append(seg)
    normalized_path = "/".join(normalized_segments)
    if parsed.query:
        params = parse_qs(parsed.query)
        normalized_query = "&".join(f"{k}=*" for k in params)
    else:
        normalized_query = ""
    return (
        f"{normalized_path}?{normalized_query}" if normalized_query else normalized_path
    )


def url_pattern_match(current_url: str, template: str) -> bool:
    """Check if a current URL matches a template pattern.

    Template uses * as wildcard for path segments and query values.
    Examples:
        url_pattern_match("https://example.com/search?q=hello", "/search?q=*") → True
        url_pattern_match("https://example.com/about", "/search?q=*") → False
    """
    import re
    from urllib.parse import urlparse

    current = urlparse(current_url)
    current_path = current.path

    # Sort query params for order-independent matching
    if current.query:
        current_params = sorted(current.query.split("&"))
        current_path += "?" + "&".join(current_params)

    if "?" in template:
        tmpl_path, tmpl_query = template.split("?", 1)
    else:
        tmpl_path, tmpl_query = template, ""

    if tmpl_path == "/*":
        regex_path = "/.*"
    else:
        regex_path = re.escape(tmpl_path).replace(r"\*", "[^/]*")

    if tmpl_query:
        query_parts = sorted(tmpl_query.split("&"))
        regex_parts = []
        for part in query_parts:
            if "=" in part:
                key, val = part.split("=", 1)
                if val == "*":
                    regex_parts.append(re.escape(key) + r"=[^&]+")
                else:
                    regex_parts.append(re.escape(part))
            else:
                regex_parts.append(re.escape(part))
        regex_query = "&".join(regex_parts)
        regex = regex_path + r"\?" + regex_query
    else:
        regex = regex_path

    return bool(re.fullmatch(regex, current_path))



_HOST_INDEX: dict[str, dict[str, list[dict[str, Any]]]] = {}
_HOST_INDEX_TTL: float = 60.0
_HOST_INDEX_TS: dict[str, float] = {}


def _get_or_build_host_index(host: str) -> dict[str, list[dict[str, Any]]]:
    import time

    now = time.time()
    cached_ts = _HOST_INDEX_TS.get(host, 0.0)
    if host in _HOST_INDEX and now - cached_ts < _HOST_INDEX_TTL:
        return _HOST_INDEX[host]
    descriptors = _list_playbooks_sync(host)
    index: dict[str, list[dict[str, Any]]] = {}
    for desc in descriptors:
        pb = load_playbook(host, desc["goal"])
        tmpl = (pb.url_template or "/*") if pb is not None else "/*"
        index.setdefault(tmpl, []).append(desc)
    _HOST_INDEX[host] = index
    _HOST_INDEX_TS[host] = now
    return index


def find_cookbook_hints(url: str, max_hints: int = 3) -> list[str]:
    """Find playbook hints for a URL across all templates on the host.

    Returns formatted hint strings for the model to decide whether to replay.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = (parsed.hostname or "").removeprefix("www.")
    if not host:
        return []
    index = _get_or_build_host_index(host)
    hints: list[str] = []
    for tmpl, descs in index.items():
        if url_pattern_match(url, tmpl):
            for desc in descs[: max_hints - len(hints)]:
                stats = f"{desc['success_count']}✓/{desc['fail_count']}✗"
                hints.append(
                    f"cookbook: '{desc['goal']}' "
                    f"({desc['step_count']} steps, {stats}) — "
                    f"replay_playbook('{desc['goal']}')"
                )
                if len(hints) >= max_hints:
                    return hints
    return hints


# Phase 5: Self-prune
def prune_playbooks(host: str) -> int:
    root = _memory_root().resolve()
    host_dir = root / _safe_host(host)
    if not host_dir.is_dir():
        return 0
    deleted = 0
    now = time.time()
    ttl_seconds = _PRUNE_TTL_DAYS * 86400
    for path in host_dir.glob("*.json"):
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
            pb = Playbook(
                host=str(data.get("host", "")),
                goal=str(data.get("goal", "")),
                created_ts=float(data.get("created_ts", 0.0)),
                steps=[],
                schema_version=data.get("schema_version", 1),
                variables=data.get("variables", []),
                success_count=data.get("success_count", 0),
                fail_count=data.get("fail_count", 0),
                last_used_ts=float(data.get("last_used_ts", 0.0)),
                last_outcome=data.get("last_outcome", ""),
            )
            # Prune: stale (past TTL) OR (fail_count >= floor AND fail_count > success)
            last_used = pb.last_used_ts or pb.created_ts
            is_stale = now - last_used > ttl_seconds
            is_failing = (
                pb.fail_count >= _PRUNE_FAIL_FLOOR and pb.fail_count > pb.success_count
            )
            if is_stale or is_failing:
                path.unlink(missing_ok=True)
                deleted += 1
        except json.JSONDecodeError, KeyError, TypeError, ValueError, OSError:
            continue
    return deleted
