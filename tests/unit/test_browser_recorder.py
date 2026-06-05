from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.tools.browser import recorder
from supporter.tools.browser.core import BrowseRequest
from supporter.tools.browser.playbook_store import build_step


@pytest.fixture(autouse=True)
def _reset_recorder() -> Generator[None]:
    recorder._ACTIVE = None
    yield
    recorder._ACTIVE = None


def test_start_sets_active_task() -> None:
    result = recorder.start("Login flow", host="app.test")
    assert "Login flow" in result
    assert recorder.is_recording()


def test_start_strips_whitespace_from_goal() -> None:
    result = recorder.start("  hello  ")
    assert recorder._ACTIVE is not None
    assert recorder._ACTIVE.goal == "hello"
    assert "hello" in result


def test_start_rejects_empty_goal() -> None:
    result = recorder.start("")
    assert result.startswith("Error:")
    assert not recorder.is_recording()


def test_start_rejects_whitespace_only_goal() -> None:
    result = recorder.start("   ")
    assert result.startswith("Error:")
    assert not recorder.is_recording()


def test_start_captures_variables() -> None:
    recorder.start("task", variables=["user", "pass"])
    assert recorder._ACTIVE is not None
    assert recorder._ACTIVE.variables == ["user", "pass"]


def test_discard_clears_active() -> None:
    recorder.start("task")
    assert recorder.is_recording()
    recorder.discard()
    assert not recorder.is_recording()
    assert recorder._ACTIVE is None


def test_is_recording_false_when_no_active() -> None:
    assert not recorder.is_recording()


def test_record_appends_step_when_active() -> None:
    recorder.start("task")
    step = build_step("click", role="button", name="OK", result="clicked")
    recorder.record(step)
    assert recorder._ACTIVE is not None
    assert len(recorder._ACTIVE.steps) == 1
    assert recorder._ACTIVE.steps[0].action == "click"


def test_record_skips_when_not_active() -> None:
    step = build_step("click", result="ok")
    recorder.record(step)
    assert recorder._ACTIVE is None


def test_record_skips_non_recordable_actions() -> None:
    recorder.start("task")
    step = build_step("unknown_action", result="ok")
    recorder.record(step)
    assert recorder._ACTIVE is not None
    assert len(recorder._ACTIVE.steps) == 0


def test_record_marks_failed_on_error_result() -> None:
    recorder.start("task")
    step = build_step("click", result="Error: ref not found")
    recorder.record(step)
    assert recorder._ACTIVE is not None
    assert recorder._ACTIVE.failed is True
    assert len(recorder._ACTIVE.steps) == 1


def test_record_does_not_mark_failed_on_success() -> None:
    recorder.start("task")
    step = build_step("click", result="clicked")
    recorder.record(step)
    assert recorder._ACTIVE is not None
    assert recorder._ACTIVE.failed is False


def test_record_all_recordable_actions() -> None:
    recorder.start("task")
    for action in recorder.RECORDABLE_ACTIONS:
        recorder.record(build_step(action, result="ok"))
    assert recorder._ACTIVE is not None
    assert len(recorder._ACTIVE.steps) == len(recorder.RECORDABLE_ACTIONS)


async def test_finish_no_active_returns_message() -> None:
    result = await recorder.finish(success=True)
    assert "No task" in result


async def test_finish_false_success_returns_nothing_saved() -> None:
    recorder.start("task", host="app.test")
    recorder.record(build_step("click", result="ok"))
    result = await recorder.finish(success=False)
    assert "success=False" in result
    assert not recorder.is_recording()


async def test_finish_failed_task_returns_nothing_saved() -> None:
    recorder.start("task", host="app.test")
    recorder.record(build_step("click", result="Error: boom"))
    result = await recorder.finish(success=True)
    assert "error" in result.lower() or "nothing saved" in result


async def test_finish_no_steps_returns_nothing_saved() -> None:
    recorder.start("task", host="app.test")
    result = await recorder.finish(success=True)
    assert "no reusable steps" in result


async def test_finish_no_host_returns_error() -> None:
    recorder.start("task")
    recorder.record(build_step("click", result="ok"))
    result = await recorder.finish(success=True)
    assert "could not resolve a host" in result


async def test_finish_saves_playbook_with_host_from_start() -> None:
    mock_save = AsyncMock()
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(
        "supporter.tools.browser.playbook_store.save_playbook", mock_save
    )
    try:
        recorder.start("Login", host="app.test")
        recorder.record(build_step("click", role="button", name="OK", result="clicked"))
        recorder.record(
            build_step("type", role="textbox", name="email", result="typed")
        )
        result = await recorder.finish(success=True)
        assert "Saved playbook" in result
        assert "Login" in result
        assert "app.test" in result
        assert "2 steps" in result
    finally:
        monkeypatch_obj.undo()


async def test_finish_saves_playbook_with_host_from_finish_arg() -> None:
    mock_save = AsyncMock()
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(
        "supporter.tools.browser.playbook_store.save_playbook", mock_save
    )
    try:
        recorder.start("task2")
        recorder.record(build_step("navigate", result="ok"))
        result = await recorder.finish(success=True, host="other.test")
        assert "Saved playbook" in result
        mock_save.assert_called_once()
        playbook = mock_save.call_args[0][0]
        assert playbook.host == "other.test"
        assert playbook.goal == "task2"
    finally:
        monkeypatch_obj.undo()


async def test_finish_prefers_host_from_start_over_finish_arg() -> None:
    mock_save = AsyncMock()
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(
        "supporter.tools.browser.playbook_store.save_playbook", mock_save
    )
    try:
        recorder.start("task3", host="start.test")
        recorder.record(build_step("click", result="ok"))
        await recorder.finish(success=True, host="finish.test")
        playbook = mock_save.call_args[0][0]
        assert playbook.host == "start.test"
    finally:
        monkeypatch_obj.undo()


async def test_finish_collects_variables_from_steps() -> None:
    mock_save = AsyncMock()
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(
        "supporter.tools.browser.playbook_store.save_playbook", mock_save
    )
    try:
        recorder.start("task", host="h.test", variables=["a"])
        recorder.record(build_step("type", result="ok", variable="username"))
        recorder.record(build_step("type", result="ok", variable="username,password"))
        await recorder.finish(success=True)
        playbook = mock_save.call_args[0][0]
        assert "a" in playbook.variables
        assert "username" in playbook.variables
        assert "password" in playbook.variables
    finally:
        monkeypatch_obj.undo()


async def test_finish_deduplicates_variables() -> None:
    mock_save = AsyncMock()
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(
        "supporter.tools.browser.playbook_store.save_playbook", mock_save
    )
    try:
        recorder.start("task", host="h.test", variables=["x"])
        recorder.record(build_step("type", result="ok", variable="x"))
        await recorder.finish(success=True)
        playbook = mock_save.call_args[0][0]
        assert playbook.variables.count("x") == 1
    finally:
        monkeypatch_obj.undo()


async def test_finish_resets_active_after_saving() -> None:
    mock_save = AsyncMock()
    monkeypatch_obj = pytest.MonkeyPatch()
    monkeypatch_obj.setattr(
        "supporter.tools.browser.playbook_store.save_playbook", mock_save
    )
    try:
        recorder.start("task", host="h.test")
        recorder.record(build_step("click", result="ok"))
        await recorder.finish(success=True)
        assert not recorder.is_recording()
    finally:
        monkeypatch_obj.undo()


def test_record_locator_returns_none_when_no_frame_no_ref() -> None:
    with patch(
        "supporter.tools.browser.session.active_frame_selector",
        return_value=None,
    ):
        page = MagicMock()
        req = BrowseRequest(action="click", ref="")
        result = recorder._record_locator(page, req)
        assert result is None


def test_record_locator_returns_locator_for_ref() -> None:
    with patch(
        "supporter.tools.browser.session.active_frame_selector",
        return_value=None,
    ):
        page = MagicMock()
        fake_locator = MagicMock()
        page.locator.return_value = fake_locator
        req = BrowseRequest(action="click", ref="e3")
        result = recorder._record_locator(page, req)
        page.locator.assert_called_once_with("aria-ref=e3")
        assert result == fake_locator


def test_record_locator_returns_none_in_frame_without_selector() -> None:
    with patch(
        "supporter.tools.browser.session.active_frame_selector",
        return_value="iframe#main",
    ):
        page = MagicMock()
        req = BrowseRequest(action="click", ref="e1", selector="")
        result = recorder._record_locator(page, req)
        assert result is None


def test_record_locator_returns_frame_locator_with_selector() -> None:
    with patch(
        "supporter.tools.browser.session.active_frame_selector",
        return_value="iframe#main",
    ):
        page = MagicMock()
        fake_locator = MagicMock()
        frame_locator = MagicMock()
        frame_locator.locator.return_value.first = fake_locator
        page.frame_locator.return_value = frame_locator
        req = BrowseRequest(action="click", selector="#btn")
        result = recorder._record_locator(page, req)
        page.frame_locator.assert_called_once_with("iframe#main")
        assert result == fake_locator


async def test_record_step_does_nothing_when_not_recording() -> None:
    req = BrowseRequest(action="click", ref="e1")
    await recorder._record_step(req, "clicked")
    assert recorder._ACTIVE is None


async def test_record_step_captures_step_with_role_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder.start("task", host="app.test")

    fake_locator = MagicMock()
    page_mock = MagicMock()
    page_mock.locator.return_value = fake_locator

    monkeypatch.setattr(
        "supporter.tools.browser.session.active_page",
        lambda: page_mock,
    )
    monkeypatch.setattr(
        "supporter.tools.browser.core._page_host",
        AsyncMock(return_value="app.test"),
    )
    monkeypatch.setattr(
        "supporter.tools.browser.session.active_frame_selector",
        lambda: None,
    )
    monkeypatch.setattr(
        "supporter.tools.browser.support._resolve_role_and_name",
        AsyncMock(return_value=("button", "Submit")),
    )

    req = BrowseRequest(action="click", ref="e5")
    await recorder._record_step(req, "clicked")
    assert recorder._ACTIVE is not None
    assert len(recorder._ACTIVE.steps) == 1
    step = recorder._ACTIVE.steps[0]
    assert step.role == "button"
    assert step.name == "Submit"
    assert step.result_head == "clicked"


async def test_record_step_skips_when_page_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder.start("task", host="app.test")
    monkeypatch.setattr(
        "supporter.tools.browser.session.active_page",
        lambda: None,
    )
    req = BrowseRequest(action="click", ref="e1")
    await recorder._record_step(req, "result")
    assert recorder._ACTIVE is not None
    assert len(recorder._ACTIVE.steps) == 0


async def test_record_step_exception_does_not_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder.start("task", host="app.test")
    monkeypatch.setattr(
        "supporter.tools.browser.session.active_page",
        MagicMock(side_effect=RuntimeError("boom")),
    )
    req = BrowseRequest(action="click", ref="e1")
    await recorder._record_step(req, "result")


async def test_record_step_skips_non_ref_resolvable_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder.start("task", host="app.test")
    monkeypatch.setattr(
        "supporter.tools.browser.session.active_page",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "supporter.tools.browser.core._page_host",
        AsyncMock(return_value="app.test"),
    )
    monkeypatch.setattr(
        "supporter.tools.browser.session.active_frame_selector",
        lambda: None,
    )
    req = BrowseRequest(action="navigate", url="https://x.test/")
    await recorder._record_step(req, "ok")
    assert recorder._ACTIVE is not None
    assert len(recorder._ACTIVE.steps) == 1
    assert recorder._ACTIVE.steps[0].params == {"url": "https://x.test/"}
