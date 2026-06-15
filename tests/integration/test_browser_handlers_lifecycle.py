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


async def test_close_confirmed_closes_session(
    fake_session: FakeSession, close_calls: list[bool]
) -> None:
    result = await browse("close")

    assert result == "Browser closed."
    assert close_calls == [True]


async def test_close_denied_leaves_open(
    fake_session: FakeSession, close_calls: list[bool]
) -> None:
    fake_session.confirm.allow = False

    result = await browse("close")

    assert result == "Browser left open."
    assert close_calls == []


async def test_close_when_inactive_reports_already_closed(
    fake_session: FakeSession,
    close_calls: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "is_active", lambda: False)

    result = await browse("close")

    assert result == "Browser already closed."
    assert close_calls == []
    assert fake_session.confirm.calls == []


async def test_closenow_closes_without_confirm(
    fake_session: FakeSession, close_calls: list[bool]
) -> None:
    result = await browse("closenow")

    assert result == "Browser closed."
    assert close_calls == [True]
    assert fake_session.confirm.calls == []


async def test_closenow_when_inactive_reports_already_closed(
    fake_session: FakeSession,
    close_calls: list[bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "is_active", lambda: False)

    result = await browse("closenow")

    assert result == "Browser already closed."
    assert close_calls == []
