from __future__ import annotations

import pytest

from supporter.tools.browser import session
from supporter.tools.browser.tool import browse

from .conftest import FakeSession


@pytest.fixture
def close_calls(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    calls: list[bool] = []

    async def fake_close(*, force: bool = False) -> None:
        calls.append(True)

    monkeypatch.setattr(session, "close_session", fake_close)
    return calls


async def test_close_rejected_as_orchestrator_only(
    fake_session: FakeSession, close_calls: list[bool]
) -> None:
    result = await browse("close")
    assert "orchestrator-only" in result
    assert close_calls == []


async def test_closenow_rejected_as_orchestrator_only(
    fake_session: FakeSession, close_calls: list[bool]
) -> None:
    result = await browse("closenow")
    assert "orchestrator-only" in result
    assert close_calls == []
