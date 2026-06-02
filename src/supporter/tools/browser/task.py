from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...logger import logger
from .. import resolved_project_root
from . import session

if TYPE_CHECKING:
    from .tool import BrowseRequest

RECORDABLE_ACTIONS = frozenset(
    {
        "navigate",
        "back",
        "forward",
        "newtab",
        "frame",
        "click",
        "type",
        "select",
        "press",
        "scroll",
        "hover",
        "extract",
        "snapshot",
        "wait",
        "waitnetwork",
    }
)

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
    schema_version: int = SCHEMA_VERSION  # v1->v2 additive
    variables: list[str] = field(default_factory=list)  # B: template variables
    success_count: int = 0  # D: replay success metric
    fail_count: int = 0  # D: replay failure metric
    last_used_ts: float = 0.0  # D: last replay timestamp
    last_outcome: str = ""  # D: "success", "drift", or "error"


@dataclass
class _ActiveTask:
    goal: str
    host: str = ""  # Phase 5: captured at start
    steps: list[Step] = field(default_factory=list)
    failed: bool = False
    variables: list[str] = field(default_factory=list)  # Phase 3: template variables


_ACTIVE: _ActiveTask | None = None


def start(goal: str, host: str = "", variables: list[str] | None = None) -> str:
    global _ACTIVE
    goal = goal.strip()
    if not goal:
        return "Error: a task goal is required."
    _ACTIVE = _ActiveTask(goal=goal, host=host, variables=variables or [])
    return f"Recording task: {goal!r}. Browse normally, then call finish_task."


def discard() -> None:
    global _ACTIVE
    _ACTIVE = None


def is_recording() -> bool:
    return _ACTIVE is not None


def record(step: Step) -> None:
    if _ACTIVE is None:
        return
    if step.action not in RECORDABLE_ACTIONS:
        # Non-recordable actions are skipped, not fatal (Phase 0 hotfix)
        return
    if step.result_head.startswith("Error:"):
        _ACTIVE.failed = True
    _ACTIVE.steps.append(step)


async def finish(success: bool, host: str = "") -> str:
    global _ACTIVE
    if _ACTIVE is None:
        return "No task is being recorded."
    task = _ACTIVE
    _ACTIVE = None

    if not success:
        return f"Task {task.goal!r} ended (success=False); nothing saved."
    if task.failed:
        return f"Task {task.goal!r} had an error or unsafe step; nothing saved."
    if not task.steps:
        return f"Task {task.goal!r} recorded no reusable steps; nothing saved."
    # Phase 5: finish prefers stored host if not provided (host-at-start capture)
    final_host = task.host or host
    if not final_host:
        return f"Task {task.goal!r}: could not resolve a host; nothing saved."

    # Phase 3: template variables = those declared at start + any annotated
    # on a recorded step, so replay can validate overrides against the real set.
    variables = list(task.variables)
    for step in task.steps:
        for raw in step.variable.split(","):
            var = raw.strip()
            if var and var not in variables:
                variables.append(var)

    playbook = Playbook(
        host=final_host,
        goal=task.goal,
        created_ts=time.time(),
        steps=task.steps,
        variables=variables,
    )
    await save_playbook(playbook)
    stepped_count = len(task.steps)
    return f"Saved playbook for {task.goal!r} on {final_host} ({stepped_count} steps)."


def _slug(goal: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", goal.lower()).strip("-._")
    return (slug or "task")[:_SLUG_MAX]


def _safe_host(host: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", host.lower()).strip("-._")


def _normalize_name(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for fuzzy matching."""
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


async def _record_step(req: BrowseRequest, result: str) -> None:
    if not is_recording():
        return
    try:
        from .tool import (
            _page_host,
            _record_locator,
            _resolve_role_and_name,
        )

        page = session.active_page()
        if page is None:
            return
        host = await _page_host(page)
        role = name = ""
        # Phase 3: extract now participates in role/name capture
        if req.action in _REF_RESOLVABLE_ACTIONS and (req.ref or req.selector):
            locator = _record_locator(page, req)
            if locator is not None:
                role, name = await _resolve_role_and_name(locator, req.ref)
        params = {f: getattr(req, f) for f in _RECORD_PARAMS.get(req.action, ())}
        step = build_step(
            req.action,
            role=role,
            name=name,
            selector=req.selector,
            url_before=host,
            params=params,
            result=result,
            variable=req.variable,
        )
        record(step)
    except Exception:
        logger.debug("Failed to record task step", exc_info=True)


async def start_task(goal: str, variables: list[str] | None = None) -> str:
    """Begin remembering a browser task so its steps can be reused later.

    Call this before you start a multi-step flow (e.g. "log in to X", "search
    GitHub for Y"). Browse normally afterward; each eligible step is recorded.
    Call finish_task when done to save it (on success) or discard it.

    Args:
        goal: A short, stable description of the task. The same wording lets
            query_playbook find this playbook on a future run.
        variables: Optional list of variable names for templating (Phase 3).

    Returns:
        A confirmation message.
    """
    logger.info(f"Tool: start_task — goal={goal!r}, variables={variables}")
    from .tool import _page_host

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    return start(goal, host=host, variables=variables)


async def finish_task(success: bool = True) -> str:
    """End the current task, save its steps, and resolve the browser lifecycle.

    Saves a reusable playbook only if success is True and no errored or unsafe
    step occurred; otherwise nothing is stored. Always clears the buffer. Then
    applies the user's keep-open choice: a persistent session is left untouched,
    while an atomic one prompts to close the browser now.

    Args:
        success: Whether the task completed successfully. False discards it.

    Returns:
        A message stating whether a playbook was saved and the browser's state.
    """
    logger.info(f"Tool: finish_task — success={success}")
    from .tool import _page_host

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    saved = await finish(success, host)

    lifecycle = await session.resolve_close_at_task_end()
    return f"{saved}\n{lifecycle}" if lifecycle else saved


def _find_ref(snapshot_text: str, role: str, name: str) -> str:
    from .snapshot import _NAME_PATTERN, _REF_GROUP, _ROLE_PATTERN

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


def _replay_params(
    step: Step, overrides: dict[str, str] | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(step.params)
    if step.selector:
        kwargs["selector"] = step.selector
    overrides = overrides or {}
    if not overrides or not step.variable:
        return kwargs
    variables = [v.strip() for v in step.variable.split(",") if v.strip()]
    # Mode is decided per step, not per variable. A step is "templated" when
    # any of its variables appears as a ${var} placeholder in a recorded value:
    # then every variable substitutes in place and the wholesale-swap fallback
    # is disabled, so a second variable can't clobber the param a placeholder
    # lives in. Otherwise the recorded value is a literal and the action's
    # primary param is swapped wholesale — the common record-time case, where a
    # typed field's value IS the variable (e.g. type's "text").
    templated = any(
        "${" + var + "}" in val
        for var in variables
        for val in kwargs.values()
        if isinstance(val, str)
    )
    if templated:
        for var in variables:
            if var not in overrides:
                continue
            placeholder = "${" + var + "}"
            for key, val in list(kwargs.items()):
                if isinstance(val, str) and placeholder in val:
                    kwargs[key] = val.replace(placeholder, overrides[var])
        return kwargs
    primary = _RECORD_PARAMS.get(step.action, ())
    primary_key = primary[0] if primary else ""
    if primary_key and primary_key in kwargs:
        for var in variables:
            if var in overrides:
                kwargs[primary_key] = overrides[var]
                break
    return kwargs


# Phase 4: Fuzzy ref resolution
# Two candidates whose scores are within this margin are treated as ambiguous:
# replay binds neither and hands back as drift rather than silently clicking the
# wrong one (e.g. two identically-labelled buttons).
_FUZZY_AMBIGUITY_MARGIN = 0.15


def _name_match_score(want: str, want_tokens: set[str], found: str) -> float:
    """Closeness of a live element name to the recorded one; 0.0 = no match.

    Tiered so an exact hit always outranks a partial one: exact (1.0) >
    token-subset (0.6-0.9, tighter when fewer extra tokens) > substring
    (0.3-0.5). Tiers don't overlap, so ranking is order-independent.
    """
    if want == found:
        return 1.0
    found_tokens = set(found.split())
    if want_tokens and want_tokens.issubset(found_tokens):
        return 0.6 + 0.3 * (len(want_tokens) / len(found_tokens))
    if want in found:
        return 0.3 + 0.2 * (len(want) / len(found))
    return 0.0


def _find_ref_fuzzy(snapshot_text: str, role: str, name: str) -> str:
    """Fuzzy role-exact, name-relaxed ref resolution for self-healing.

    Names are normalized (lowercased, punctuation stripped, whitespace
    collapsed) so cosmetic label drift — "Sign in" vs "Sign In!" — still
    resolves. Role must match exactly to avoid binding the wrong widget.

    Every role-matching candidate is scored and the single best one wins. When
    the top two are within _FUZZY_AMBIGUITY_MARGIN the match is ambiguous, so
    nothing is bound and replay hands back as drift instead of guessing.
    """
    from .snapshot import _NAME_PATTERN, _REF_GROUP, _ROLE_PATTERN

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
    """Rank saved goals on this host by token overlap with the query goal.

    Fallback when an exact goal lookup misses: surfaces close saved goals so
    the model can retry with a real one instead of re-discovering a known flow.
    Returns goals only — it never auto-runs; matching stays model-in-the-loop.
    """
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
    """Miss message for query/replay/delete, with fuzzy goal suggestions."""
    base = f"No playbook found for {goal!r} on {host}."
    candidates = _fuzzy_find_goal(host, goal)
    if not candidates:
        return base
    similar = "; ".join(repr(candidate) for candidate in candidates)
    return f"{base} Similar saved goals: {similar}. Retry with one of these."


async def replay_playbook(goal: str, overrides: dict[str, str] | None = None) -> str:
    """Replay a saved playbook on the current site, step by step, until one fails.

    Executes each recorded step in order, re-deriving live element refs from a
    fresh snapshot (refs are ephemeral). On the first step that can't be resolved
    or whose result is an error, it STOPS and hands control back with what ran,
    the failing step, and the current page — so you continue from there yourself.
    Far cheaper than re-discovering a known flow; falls back to you on any drift.

    Args:
        goal: The task description used when the playbook was recorded.
        overrides: Optional dict of variable name → value to substitute (Phase 3).

    Returns:
        A success summary, or a handback describing where replay stopped.
    """
    logger.info(f"Tool: replay_playbook — goal={goal!r}")
    from .tool import _live_refs_snapshot, _page_host, browse

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    if not host:
        return "No active page; navigate first, then replay a playbook."
    playbook = load_playbook(host, goal)
    if playbook is None:
        return _no_playbook_message(host, goal)

    # Phase 3: validate overrides
    if overrides:
        unknown = set(overrides) - set(playbook.variables)
        if unknown:
            declared = ", ".join(playbook.variables) or "(none declared)"
            unk = ", ".join(unknown)
            return f"Unknown override(s): {unk}. Declared variables: {declared}"

    done: list[str] = []
    for i, step in enumerate(playbook.steps, 1):
        kwargs = _replay_params(step, overrides)
        # Phase 3: extract now participates in ref resolution
        if step.action in _REF_RESOLVABLE_ACTIONS and not step.selector:
            live_page = session.active_page()
            snapshot_text = (
                await _live_refs_snapshot(live_page) if live_page is not None else ""
            )
            # Phase 4: exact first, then fuzzy
            ref = _find_ref(snapshot_text, step.role, step.name)
            if not ref and step.role:
                ref = _find_ref_fuzzy(snapshot_text, step.role, step.name)
            if not ref:
                target = f"{step.role} {step.name!r}".strip()
                # Phase 5: bump fail_count on drift
                playbook.fail_count += 1
                playbook.last_outcome = "drift"
                await save_playbook(playbook)
                return (
                    f"Replayed {len(done)}/{len(playbook.steps)} step(s): "
                    f"{'; '.join(done) or 'none'}.\n"
                    f"Stopped at step {i} ({step.action} → {target}): element not "
                    f"found on the current page. Continue from here yourself.\n"
                    f"Current page:\n{snapshot_text}"
                )
            kwargs["ref"] = ref

        result = await browse(step.action, **kwargs)
        if result.startswith("Error:"):
            # Phase 5: bump fail_count on error
            playbook.fail_count += 1
            playbook.last_outcome = "error"
            await save_playbook(playbook)
            return (
                f"Replayed {len(done)}/{len(playbook.steps)} step(s): "
                f"{'; '.join(done) or 'none'}.\n"
                f"Stopped at step {i} ({step.action}): {result}\n"
                f"Continue from here yourself."
            )
        done.append(f"{i}.{step.action}")

    # Phase 5: bump success_count on success
    playbook.success_count += 1
    playbook.last_used_ts = time.time()
    playbook.last_outcome = "success"
    await save_playbook(playbook)

    n = len(playbook.steps)
    return f"Replayed playbook {goal!r} on {host}: {n}/{n} steps succeeded."


async def query_playbook(goal: str) -> str:
    """Look up a saved playbook for this site + task to guide your next steps.

    Call before reasoning out a flow from scratch: if a playbook exists you can
    follow its steps instead of re-discovering them. The steps are advisory —
    read them and decide; nothing is executed automatically.

    Args:
        goal: The task description used when the playbook was recorded.

    Returns:
        A numbered step list, or a note that no playbook was found.
    """
    logger.info(f"Tool: query_playbook — goal={goal!r}")
    from .tool import _page_host

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    if not host:
        return "No active page; navigate first, then query a playbook."
    playbook = load_playbook(host, goal)
    if playbook is None:
        return _no_playbook_message(host, goal)
    return format_playbook(playbook)


# Phase 2: Discovery tool
def _list_playbooks_sync(host: str) -> list[dict[str, Any]]:
    """Return lightweight descriptors for playbooks on this host (never full steps)."""
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


async def list_playbooks() -> str:
    """List saved playbooks for the current host, ranked by recency and success.

    Call first to discover flows; then `query_playbook(goal)` to inspect, or
    `replay_playbook(goal, overrides={...})` to run.

    Returns:
        A compact list of playbook goals with stats, or a note to navigate first.
    """
    logger.info("Tool: list_playbooks")
    from .tool import _page_host

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    if not host:
        return "No active page; navigate first, then list playbooks."
    # Phase 5: prune lazily (sync, no await needed)
    prune_playbooks(host)
    descriptors = _list_playbooks_sync(host)
    if not descriptors:
        return f"No playbooks found for {host}. Record one with start_task/finish_task."
    lines = [f"Playbooks for {host} (ranked by recency/success):"]
    for i, d in enumerate(descriptors, 1):
        stats = f"{d['success_count']}✓/{d['fail_count']}✗"
        lines.append(f"  {i}. {d['goal']!r} ({d['step_count']} steps, {stats})")
    return "\n".join(lines)


def _delete_playbook_sync(host: str, goal: str) -> bool:
    try:
        path = _safe_path(host, goal)
    except ValueError:
        return False
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    return True


async def delete_playbook(goal: str) -> str:
    """Delete a saved playbook for this site + task when it's wrong or stale.

    Use when a replay keeps drifting or the recorded flow no longer matches the
    site, so a fresh start_task/finish_task recording can replace it. Removes
    only the playbook for the current host and the given goal.

    Args:
        goal: The task description used when the playbook was recorded.

    Returns:
        A confirmation, or a note (with close matches) that none was found.
    """
    logger.info(f"Tool: delete_playbook — goal={goal!r}")
    from .tool import _page_host

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    if not host:
        return "No active page; navigate first, then delete a playbook."
    deleted = await asyncio.to_thread(_delete_playbook_sync, host, goal)
    if deleted:
        return f"Deleted playbook {goal!r} on {host}."
    return _no_playbook_message(host, goal)


# Phase 5: Self-prune
def prune_playbooks(host: str) -> int:
    """Delete stale or repeatedly-failing playbooks. Returns count deleted."""
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
