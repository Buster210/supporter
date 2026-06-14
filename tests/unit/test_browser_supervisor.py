from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from supporter.tools.base import ToolError
from supporter.tools.browser import session
from supporter.tools.browser.supervisor import (
    SUPERVISOR_ACTIONS,
    browser_supervise,
)

_SCALAR_GLOBALS = (
    "_PWS",
    "_CONTEXT",
    "_LAUNCHING",
    "_LAUNCH_LOOP",
    "_CLONE_LOCK",
    "_KEEP_OPEN",
    "_LIFECYCLE_TASK",
    "_CLEANUP_TASK",
)
_DICT_GLOBALS = (
    "_PAGES",
    "_FRAME_SELECTORS",
    "_OWNED_PAGES",
    "_ACTION_COUNT",
    "_ACTION_CAP_CEILING",
    "_LAST_ACTION_TS",
    "_SESSION_START_TS",
    "_TEMPO",
)


@pytest.fixture(autouse=True)
def _reset_session() -> Generator[None, None, None]:
    saved = {name: getattr(session, name) for name in _SCALAR_GLOBALS}
    saved_dicts = {name: dict(getattr(session, name)) for name in _DICT_GLOBALS}
    saved_cb = session.guardrails.browse_confirmation_callback
    token = session._AGENT_ID.set("main")
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(session, name, value)
        for name, value in saved_dicts.items():
            current = getattr(session, name)
            current.clear()
            current.update(value)
        session.guardrails.browse_confirmation_callback = saved_cb
        session._AGENT_ID.reset(token)


def _fake_page(url: str = "https://example.test/") -> Any:
    page = cast("Any", type("_FakePage", (), {"url": url})())
    page.close = AsyncMock()
    page.title = AsyncMock(return_value="Fake Title")
    page.is_closed = lambda: False
    return page


def _fake_session(*, url: str = "https://example.test/", tabs: int = 1) -> None:
    pages = [_fake_page(url) for _ in range(tabs)]
    context = cast("Any", type("_Ctx", (), {"pages": pages})())
    session._PWS = cast("Any", object())
    session._CONTEXT = context
    if pages:
        session._PAGES["main"] = pages[0]
        session._OWNED_PAGES["main"] = set(pages)
    else:
        session._PAGES.pop("main", None)
    session._LAST_ACTION_TS["main"] = time.monotonic() - 5.0


@pytest.mark.parametrize(
    "action",
    sorted(SUPERVISOR_ACTIONS),
)
async def test_whitelisted_actions_are_accepted(action: str) -> None:
    _fake_session()
    try:
        result = await browser_supervise(action)
    except ToolError:
        return
    assert "not permitted" not in result


@pytest.mark.parametrize(
    "action",
    [
        "navigate",
        "click",
        "type",
        "hover",
        "scroll",
        "press",
        "select",
        "eval",
        "extract",
        "cookies",
        "storage",
        "upload",
        "download",
        "frame",
        "newtab",
        "tab",
        "wait",
        "waitnetwork",
        "diff",
        "solve_cloudflare",
        "back",
        "forward",
    ],
)
async def test_non_whitelisted_actions_are_rejected(action: str) -> None:
    result = await browser_supervise(action)
    assert "not permitted" in result
    assert action in result


async def test_rejection_lists_allowed_actions() -> None:
    result = await browser_supervise("eval")
    for name in sorted(SUPERVISOR_ACTIONS):
        assert name in result


async def test_status_returns_json_with_session_fields() -> None:
    _fake_session(url="https://foo.bar/", tabs=3)
    result = await browser_supervise("status")
    data = json.loads(result)
    assert data["active"] is True
    assert data["launching"] is False
    assert data["tabs"] == 3
    assert isinstance(data["idle_seconds"], (int, float))
    assert isinstance(data["pinned_open"], bool)


async def test_status_active_false_when_no_session() -> None:
    session._PAGES.pop("main", None)
    result = await browser_supervise("status")
    data = json.loads(result)
    assert data["active"] is False
    assert data["tabs"] == 0
    assert data["idle_seconds"] is None


async def test_status_reflects_launching_flag() -> None:
    session._LAUNCHING = True
    session._PAGES.pop("main", None)
    result = await browser_supervise("status")
    data = json.loads(result)
    assert data["launching"] is True


async def test_closenow_tears_down_active_session() -> None:
    _fake_session()
    assert session.is_active()
    result = await browser_supervise("closenow")
    assert result == "Browser closed."
    assert not session.is_active()


async def test_closenow_force_tears_down_active_session() -> None:
    _fake_session()
    assert session.is_active()
    result = await browser_supervise("closenow", force=True)
    assert result == "Browser closed."
    assert not session.is_active()


async def test_closenow_already_closed_returns_message() -> None:
    session._PAGES.pop("main", None)
    result = await browser_supervise("closenow")
    assert result == "Browser already closed."


async def test_closenow_force_on_inactive_session() -> None:
    session._PAGES.pop("main", None)
    result = await browser_supervise("closenow", force=True)
    assert result == "Browser already closed."


async def test_close_denied_by_user() -> None:
    _fake_session()

    async def _deny(title: str, detail: str) -> bool:
        return False

    session.guardrails.browse_confirmation_callback = _deny
    result = await browser_supervise("close")
    assert result == "Browser left open."
    assert session.is_active()


async def test_close_confirmed_by_user() -> None:
    _fake_session()
    assert session.is_active()

    async def _confirm(title: str, detail: str) -> bool:
        return True

    session.guardrails.browse_confirmation_callback = _confirm
    result = await browser_supervise("close")
    assert result == "Browser closed."
    assert not session.is_active()


async def test_close_already_closed() -> None:
    session._PAGES.pop("main", None)
    result = await browser_supervise("close")
    assert result == "Browser already closed."


async def test_tabs_lists_open_tabs() -> None:
    _fake_session(tabs=2)
    result = await browser_supervise("tabs")
    assert "[0]" in result
    assert "[1]" in result


async def test_closetab_rejects_out_of_range() -> None:
    _fake_session(tabs=2)
    result = await browser_supervise("closetab", index=5)
    assert "out of range" in result


def test_browser_supervise_in_orchestrator_tool_names() -> None:
    from supporter.tools.catalog import ORCHESTRATOR_TOOL_NAMES

    assert "browser_supervise" in ORCHESTRATOR_TOOL_NAMES


def test_browser_supervise_not_delegate_allowed() -> None:
    from supporter.tools.catalog import build_tool_catalog

    catalog = build_tool_catalog()
    spec = catalog["browser_supervise"]
    assert spec.delegate_allowed is False


def test_browse_still_exclusive_to_page_pilot() -> None:
    from supporter.tools.catalog import build_tool_catalog, select_delegate_tools

    catalog = build_tool_catalog()
    registry = select_delegate_tools(catalog, "all", role="test_engineer")
    assert "browse" not in registry
    pp_registry = select_delegate_tools(catalog, "all", role="page-pilot")
    assert "browse" in pp_registry


def test_sub_agents_cannot_get_supervisor_tool() -> None:
    from supporter.tools.catalog import build_tool_catalog, select_delegate_tools

    catalog = build_tool_catalog()
    for role in ("code_reviewer", "explorer", "page-pilot", "test_engineer", None):
        registry = select_delegate_tools(catalog, "all", role=role)
        assert "browser_supervise" not in registry, (
            f"role {role!r} should not get browser_supervise"
        )


def test_orchestrator_prompt_mentions_browser_supervise() -> None:
    from supporter.prompts import DEFAULT_SYSTEM_INSTRUCTION

    assert "browser_supervise" in DEFAULT_SYSTEM_INSTRUCTION


def test_orchestrator_prompt_has_recovery_protocol() -> None:
    from supporter.prompts import DEFAULT_SYSTEM_INSTRUCTION

    assert "Browser Recovery Protocol" in DEFAULT_SYSTEM_INSTRUCTION
    assert "cancel_delegation" in DEFAULT_SYSTEM_INSTRUCTION


async def test_session_status_returns_dict() -> None:
    _fake_session(url="https://test.com/", tabs=1)
    info = await session.session_status()
    assert info["active"] is True
    assert info["url"] == "https://test.com/"


async def test_session_status_inactive_session() -> None:
    session._PAGES.pop("main", None)
    session._CONTEXT = None
    info = await session.session_status()
    assert info["active"] is False
    assert info["tabs"] == 0


async def test_close_session_force_resets_globals() -> None:
    _fake_session()
    session._KEEP_OPEN = True
    session._FRAME_SELECTORS["main"] = "iframe"
    await session.close_session(force=True)
    assert not session.is_active()
    assert session._KEEP_OPEN is None
    assert session._FRAME_SELECTORS.get("main") is None


async def test_close_session_force_swallows_close_error() -> None:
    _fake_session()

    async def _raise() -> None:
        raise OSError("wedged")

    session._CONTEXT.close = _raise  # type: ignore[method-assign,union-attr]
    await session.close_session(force=True)
    assert not session.is_active()


async def test_close_session_force_bounds_hanging_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_session()
    monkeypatch.setattr(session, "_FORCE_CLOSE_TIMEOUT_S", 0.01)

    async def _hang() -> None:
        await asyncio.sleep(10)

    session._CONTEXT.close = _hang  # type: ignore[method-assign,union-attr]
    await session.close_session(force=True)
    assert not session.is_active()
