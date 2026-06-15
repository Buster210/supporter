"""G3: only the main orchestrator may close the browser.

A subagent calling close/closenow must be refused (browser left open so other
agents keep using it); the main orchestrator can still close. Reopen is already
covered by test_browser_session.py and is unchanged by the guard.
"""

from __future__ import annotations

import pytest

from supporter.config import config
from supporter.tools.browser import guardrails, handlers, session
from supporter.tools.browser.core import BrowseRequest

_REFUSAL = "Only the main orchestrator can close the browser"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    # Guard only fires when parallel pilots is on (else everyone is "main").
    monkeypatch.setattr(config, "browser_parallel_pilots", True)
    # Pretend a live session so we exercise the guard, not the already-closed path.
    monkeypatch.setattr(session, "is_session_alive", lambda: True)


@pytest.mark.asyncio
async def test_subagent_cannot_close(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    async def _record(*, force: bool = False) -> None:
        calls.append(force)

    monkeypatch.setattr(session, "close_session", _record)
    tok = session.set_agent_id("sub1")
    try:
        r1 = await handlers._handle_close(BrowseRequest(action="close"))
        r2 = await handlers._handle_closenow(BrowseRequest(action="closenow"))
    finally:
        session.reset_contextvar(tok)

    assert _REFUSAL in r1
    assert _REFUSAL in r2
    assert calls == []  # close_session never reached


@pytest.mark.asyncio
async def test_main_can_close(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    async def _record(*, force: bool = False) -> None:
        calls.append(force)

    async def _confirm(_a: str, _b: str) -> bool:
        return True

    monkeypatch.setattr(session, "close_session", _record)
    monkeypatch.setattr(guardrails, "browse_confirmation_callback", _confirm)
    # default agent id is "main"
    result = await handlers._handle_close(BrowseRequest(action="close"))

    assert result == "Browser closed."
    assert calls == [False]
