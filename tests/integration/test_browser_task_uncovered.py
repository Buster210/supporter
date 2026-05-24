from __future__ import annotations

import pytest

from supporter.tools.browser import task
from supporter.tools.browser.tool import BrowseRequest

from .conftest import FakeSession


@pytest.fixture(autouse=True)
def _reset_task_state() -> None:
    task.discard()


async def test_replay_playbook_no_active_page(fake_session: FakeSession) -> None:
    # Set up the session with a URL like the existing tests
    fake_session.page.eval_result = "https://example.test/"

    # Mock active_page to return None
    import supporter.tools.browser.session as session_module

    original_active_page = session_module.active_page

    def mock_active_page() -> None:
        return None

    session_module.active_page = mock_active_page

    try:
        result = await task.replay_playbook("test goal")
        assert "No active page; navigate first, then replay a playbook." in result
    finally:
        session_module.active_page = original_active_page


async def test_start_task_empty_goal() -> None:
    result = task.start("")
    assert "Error: a task goal is required." in result


async def test_finish_task_no_active_task() -> None:
    result = await task.finish(True, "example.test")
    assert "No task is being recorded." in result


async def test_finish_task_unsuccessful() -> None:
    task.start("test goal")
    result = await task.finish(False, "example.test")
    assert "Task 'test goal' ended (success=False); nothing saved." in result


async def test_finish_task_failed_steps() -> None:
    task.start("test goal")
    # Access the active task directly
    active_task = task._ACTIVE
    if active_task is not None:
        active_task.failed = True  # Mark as having failed steps
    result = await task.finish(True, "example.test")
    assert "Task 'test goal' had an error or unsafe step; nothing saved." in result


async def test_finish_task_no_steps() -> None:
    task.start("test goal")
    # Clear steps
    active_task = task._ACTIVE
    if active_task is not None:
        active_task.steps = []
    result = await task.finish(True, "example.test")
    assert "Task 'test goal' recorded no reusable steps; nothing saved." in result


async def test_query_playbook_no_active_page(fake_session: FakeSession) -> None:
    # Mock active_page to return None
    import supporter.tools.browser.session as session_module

    original_active_page = session_module.active_page

    def mock_active_page2() -> None:
        return None

    session_module.active_page = mock_active_page2

    try:
        result = await task.query_playbook("test goal")
        assert "No active page; navigate first, then query a playbook." in result
    finally:
        session_module.active_page = original_active_page


async def test_record_step_exception_caught(fake_session: FakeSession) -> None:
    from supporter.tools.browser import task as browser_task
    from supporter.tools.browser.task import _record_step

    # Start a task so is_recording() returns True — without this, _record_step
    # bails at the early-return guard and never enters the try/except block.
    browser_task.start("test goal")

    # Patch task.record to raise; record() is only called inside _record_step's
    # try block (never during handler dispatch), so the handler runs normally
    # and the exception is caught by the except Exception guard.
    original_record = browser_task.record

    def _raising_record(step: object) -> None:
        raise ValueError("simulated record failure")

    browser_task.record = _raising_record

    try:
        req = BrowseRequest(
            action="click",
            ref="e2",
        )
        await _record_step(req, "test result")
        # No assertion needed — the test passes if _record_step swallows
        # the exception without propagating.
    finally:
        browser_task.record = original_record
        browser_task.discard()
