from __future__ import annotations

import asyncio
import time
from typing import Any

from ...config import config
from ...logger import logger
from . import session
from .core import _page_host
from .playbook_match import (
    _find_ref,
    _find_ref_fuzzy,
    _no_playbook_message,
)
from .playbook_store import (
    _RECORD_PARAMS,
    _REF_RESOLVABLE_ACTIONS,
    Step,
    _delete_playbook_sync,
    _list_playbooks_sync,
    format_playbook,
    load_playbook,
    prune_playbooks,
    save_playbook,
)
from .recorder import (
    finish,
    start,
)

__all__ = [
    "_replay_params",
    "delete_playbook",
    "finish_task",
    "list_playbooks",
    "query_playbook",
    "replay_playbook",
    "start_task",
]


async def start_task(goal: str, variables: list[str] | None = None) -> str:
    """Begin remembering a browser task so its steps can be reused later.

    Call this before you start a multi-step flow (e.g. "log in to X", "search
    GitHub for Y"). Browse normally afterward; each eligible step is recorded.
    Call finish_task when done to save it (on success) or discard it.

    Args:
        goal: A short, stable description of the task. The same wording lets
            query_playbook find this playbook on a future run.
        variables: Optional list of variable names to template; tag a step's
            value with browse(..., variable=name) while recording, then
            substitute at replay via replay_playbook(overrides=...).

    Returns:
        A confirmation message.
    """
    logger.info(f"Tool: start_task — goal={goal!r}, variables={variables}")

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

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    saved = await finish(success, host)

    lifecycle = await session.resolve_close_at_task_end()
    return f"{saved}\n{lifecycle}" if lifecycle else saved


async def replay_playbook(goal: str, overrides: dict[str, str] | None = None) -> str:
    """Replay a saved playbook on the current site, step by step, until one fails.

    Executes each recorded step in order, re-deriving live element refs from a
    fresh snapshot (refs are ephemeral). On the first step that can't be resolved
    or whose result is an error, it STOPS and hands control back with what ran,
    the failing step, and the current page — so you continue from there yourself.
    Far cheaper than re-discovering a known flow; falls back to you on any drift.

    Args:
        goal: The task description used when the playbook was recorded.
        overrides: Optional dict of variable name -> value to substitute for the
            playbook's declared variables.

    Returns:
        A success summary, or a handback describing where replay stopped.
    """
    logger.info(f"Tool: replay_playbook — goal={goal!r}")
    from .support import _live_refs_snapshot
    from .tool import browse

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
    # D2: Include final page state in success return
    live_page = session.active_page()
    final_snapshot = (
        await _live_refs_snapshot(live_page) if live_page is not None else ""
    )
    char_cap = config.browse_page_chars_cap
    if len(final_snapshot) > char_cap:
        final_snapshot = (
            final_snapshot[:char_cap]
            + f"\n\n…(truncated: {len(final_snapshot) - char_cap} more chars)"
        )
    final_page = f"\n\nFinal page:\n{final_snapshot}"
    return (
        f"Replayed playbook {goal!r} on {host}: {n}/{n} steps succeeded."
        + final_page
    )


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

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    if not host:
        return "No active page; navigate first, then query a playbook."
    playbook = load_playbook(host, goal)
    if playbook is None:
        return _no_playbook_message(host, goal)
    return format_playbook(playbook)


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


async def list_playbooks() -> str:
    """List saved playbooks for the current host, ranked by recency and success.

    Call first to discover flows; then `query_playbook(goal)` to inspect, or
    `replay_playbook(goal, overrides={...})` to run.

    Returns:
        A compact list of playbook goals with stats, or a note to navigate first.
    """
    logger.info("Tool: list_playbooks")

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

    page = session.active_page()
    host = await _page_host(page) if page is not None else ""
    if not host:
        return "No active page; navigate first, then delete a playbook."
    deleted = await asyncio.to_thread(_delete_playbook_sync, host, goal)
    if deleted:
        return f"Deleted playbook {goal!r} on {host}."
    return _no_playbook_message(host, goal)
