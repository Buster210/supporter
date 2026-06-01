from __future__ import annotations

import pytest

from supporter.tools.browser import guardrails, humanize, snapshot
from supporter.tools.browser.tool import browse

from .conftest import FakeSession


@pytest.fixture(autouse=True)
def _reset_snapshot_baselines(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot._LAST_SNAPSHOT.clear()

    async def _no_idle(_page: object) -> None:
        return None

    monkeypatch.setattr(humanize, "idle_flourish", _no_idle)


async def test_click_resolves_ref_and_clicks(fake_session: FakeSession) -> None:
    await browse("click", ref="e2", fast=True)

    _args, _kwargs = fake_session.log.last("click")
    assert fake_session.log.count("click") == 1


async def test_click_without_ref_errors(fake_session: FakeSession) -> None:
    result = await browse("click")

    assert result == (
        "Error: 'ref' is required for click/type. "
        "Get a snapshot first to find [ref=eN]."
    )


async def test_type_fills_text_in_fast_mode(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardrails, "host_is_fast", lambda _host: True)
    await browse("type", ref="e2", text="hello", fast=True)

    args, _kwargs = fake_session.log.last("fill")
    assert args == ("hello",)


async def test_type_without_text_errors(fake_session: FakeSession) -> None:
    result = await browse("type", ref="e2")

    assert result == "Error: 'text' is required for type action."


async def test_type_into_password_field_is_gated(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter.tools.browser import tool

    async def password_role(_locator: object, _ref: str) -> tuple[str, str]:
        return "password", "pw"

    monkeypatch.setattr(tool, "_resolve_role_and_name", password_role)
    fake_session.confirm.allow = False

    result = await browse("type", ref="e2", text="secret")

    assert result == "Error: action cancelled."
    assert fake_session.log.count("fill") == 0
    assert len(fake_session.confirm.calls) == 1


async def test_hover_resolves_and_hovers(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardrails, "host_is_fast", lambda _host: True)
    await browse("hover", ref="e2", fast=True)

    _args, _kwargs = fake_session.log.last("hover")
    assert fake_session.log.count("hover") == 1


async def test_scroll_without_target_errors(fake_session: FakeSession) -> None:
    result = await browse("scroll")

    assert result == "Error: scroll needs a 'ref' or non-zero 'dx'/'dy'."


async def test_scroll_by_delta_uses_wheel_in_fast_mode(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardrails, "host_is_fast", lambda _host: True)
    await browse("scroll", dx=0, dy=200, fast=True)

    args, _kwargs = fake_session.log.last("wheel")
    assert args == (0, 200)


async def test_press_without_key_errors(fake_session: FakeSession) -> None:
    result = await browse("press")

    assert result == "Error: 'key' is required for press (e.g. 'Enter', 'Control+a')."


async def test_press_sends_key_in_fast_mode(fake_session: FakeSession) -> None:
    await browse("press", key="Enter", fast=True)

    args, _kwargs = fake_session.log.last("press")
    assert args == ("Enter",)


async def test_select_without_ref_errors(fake_session: FakeSession) -> None:
    result = await browse("select", value="opt1")

    assert result == "Error: 'ref' is required for select. Get a snapshot first."


async def test_select_without_value_or_text_errors(
    fake_session: FakeSession,
) -> None:
    result = await browse("select", ref="e2")

    assert result == "Error: select needs 'value' or 'text' (the option to choose)."


async def test_select_by_value_chooses_option(fake_session: FakeSession) -> None:
    await browse("select", ref="e2", value="opt1", fast=True)

    _args, kwargs = fake_session.log.last("select_option")
    assert kwargs == {"value": "opt1"}
