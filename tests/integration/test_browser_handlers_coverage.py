from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest

from supporter.tools.base import ToolError
from supporter.tools.browser import guardrails, humanize, snapshot
from supporter.tools.browser.tool import browse

from .conftest import FakeSession


@pytest.fixture(autouse=True)
def _reset_snapshot_baselines() -> Generator[None]:
    snapshot._LAST_SNAPSHOT.clear()
    yield
    snapshot._LAST_SNAPSHOT.clear()


async def test_frame_drill_into_iframe(fake_session: FakeSession) -> None:
    fake_session.page.aria_text = '- contentinfo [ref=e1]:\n  - link "About" [ref=e2]'
    result = await browse("frame", selector="iframe#main")
    assert "Drilled into frame" in result
    assert "iframe#main" in result


async def test_frame_drill_error_returns_error(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_frame_locator(s: str) -> object:
        raise RuntimeError("No frame matched")

    monkeypatch.setattr(fake_session.page, "frame_locator", broken_frame_locator)

    with pytest.raises(ToolError, match="No frame matched"):
        await browse("frame", selector="iframe#missing")


async def test_click_humanized_mode(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardrails, "host_is_fast", lambda _h: False)
    slept: list[float] = []

    async def fake_sleep(s: float) -> None:
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def noop_flourish(page: object) -> None:
        return None

    monkeypatch.setattr(humanize, "idle_flourish", noop_flourish)

    await browse("click", ref="e2")
    assert fake_session.log.count("click") >= 1


async def test_type_humanized_mode(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardrails, "host_is_fast", lambda _h: False)
    monkeypatch.setattr(humanize, "human_type", humanize.human_type)

    async def noop_flourish(page: object) -> None:
        return None

    monkeypatch.setattr(humanize, "idle_flourish", noop_flourish)

    async def no_sleep(s: float) -> None:
        return None

    monkeypatch.setattr("supporter.tools.browser.humanize.asyncio.sleep", no_sleep)

    await browse("type", ref="e2", text="hello")


async def test_hover_humanized_mode(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardrails, "host_is_fast", lambda _h: False)

    async def noop_flourish(page: object) -> None:
        return None

    monkeypatch.setattr(humanize, "idle_flourish", noop_flourish)
    await browse("hover", ref="e2")
    assert fake_session.log.count("move") >= 1


async def test_scroll_with_ref_in_humanized_mode(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardrails, "host_is_fast", lambda _h: False)
    await browse("scroll", ref="e2")
    assert fake_session.log.count("scroll_into_view_if_needed") >= 1


async def test_press_with_ref(
    fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(guardrails, "host_is_fast", lambda _h: True)
    await browse("press", ref="e2", key="Enter")
    assert fake_session.log.count("press") >= 1


async def test_select_by_label_text(fake_session: FakeSession) -> None:
    await browse("select", ref="e2", text="Option A")
    _args, kwargs = fake_session.log.last("select_option")
    assert kwargs == {"label": "Option A"}


async def test_closenow(fake_session: FakeSession) -> None:
    result = await browse("closenow")
    assert "orchestrator-only" in result


async def test_cookies_with_key(fake_session: FakeSession) -> None:
    fake_session.context.cookie_list = [
        {"name": "session", "domain": ".example.com", "value": "abc123"}
    ]
    result = await browse("cookies", key="session")
    assert "session=abc123" in result


async def test_cookies_with_unknown_key(fake_session: FakeSession) -> None:
    fake_session.context.cookie_list = []
    result = await browse("cookies", key="nonexistent")
    assert "no cookie named" in result


async def test_cookies_list_empty(fake_session: FakeSession) -> None:
    fake_session.context.cookie_list = []
    result = await browse("cookies")
    assert "(no cookies)" in result


async def test_storage_list(fake_session: FakeSession) -> None:
    fake_session.page.eval_result = ["token", "theme"]
    result = await browse("storage")
    assert "2 localStorage keys" in result
    assert "token" in result


async def test_storage_list_empty(fake_session: FakeSession) -> None:
    fake_session.page.eval_result = []
    result = await browse("storage")
    assert "(empty localStorage)" in result


async def test_storage_set_value(fake_session: FakeSession) -> None:
    result = await browse("storage", key="theme", value="dark")
    assert "Set localStorage" in result


async def test_storage_get_value(fake_session: FakeSession) -> None:
    fake_session.page.eval_result = "dark"
    result = await browse("storage", key="theme")
    assert "theme=dark" in result


async def test_storage_get_missing_key(fake_session: FakeSession) -> None:
    fake_session.page.eval_result = None
    result = await browse("storage", key="missing")
    assert "no localStorage key" in result


async def test_extract_by_selector(fake_session: FakeSession) -> None:
    fake_session.page.locators = []
    result = await browse("extract", selector="#main")
    assert "visible text" in result or "(empty)" in result


async def test_extract_ref_in_frame_error(fake_session: FakeSession) -> None:
    import supporter.tools.browser.session as sm

    original = sm.active_frame_selector
    sm.active_frame_selector = lambda: "iframe#t"
    try:
        result = await browse("extract", ref="e2")
        assert "inside a frame" in result
    finally:
        sm.active_frame_selector = original


async def test_eval_success_path(fake_session: FakeSession) -> None:
    fake_session.page.eval_result = {"result": 42}
    result = await browse("eval", script="1 + 1")
    assert "eval result:" in result


async def test_screenshot_with_stamp(fake_session: FakeSession) -> None:
    from pathlib import Path

    import supporter.tools.browser.handlers as handlers_mod

    original = handlers_mod._validate_path_or_error  # type: ignore[attr-defined]

    def permissive_validate(path: str) -> tuple[object, None]:
        return Path(path), None

    handlers_mod._validate_path_or_error = permissive_validate  # type: ignore[attr-defined]
    try:
        result = await browse("screenshot", stamp="test-shot")
        assert "Screenshot saved" in result
    finally:
        handlers_mod._validate_path_or_error = original  # type: ignore[attr-defined]


async def test_upload_missing_file(fake_session: FakeSession) -> None:
    result = await browse("upload", ref="e2", path="/nonexistent/file.txt")
    assert "file not found" in result or "Error:" in result
