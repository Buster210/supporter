from __future__ import annotations

import pytest

from supporter.tools.browser import recorder
from supporter.tools.browser.core import BrowseRequest
from supporter.tools.browser.recorder import discard, finish, start
from supporter.tools.browser.task import (
    query_playbook,
    replay_playbook,
)

from .conftest import FakeSession


@pytest.fixture(autouse=True)
def _reset_task_state() -> None:
    discard()


async def test_replay_playbook_no_active_page(fake_session: FakeSession) -> None:
    fake_session.page.eval_result = "https://example.test/"

    import supporter.tools.browser.session as session_module

    original_active_page = session_module.active_page

    def mock_active_page() -> None:
        return None

    session_module.active_page = mock_active_page

    try:
        result = await replay_playbook("test goal")
        assert "No active page; navigate first, then replay a playbook." in result
    finally:
        session_module.active_page = original_active_page


async def test_start_task_empty_goal() -> None:
    result = start("")
    assert "Error: a task goal is required." in result


async def test_finish_task_no_active_task() -> None:
    result = await finish(True)
    assert "No task is being recorded." in result


async def test_finish_task_unsuccessful() -> None:
    start("test goal")
    result = await finish(False)
    assert "Task 'test goal' ended (success=False); nothing saved." in result


async def test_finish_task_failed_steps() -> None:
    start("test goal")
    active_task = recorder._ACTIVE
    if active_task is not None:
        active_task.failed = True
    result = await finish(True)
    assert "Task 'test goal' had an error or unsafe step; nothing saved." in result


async def test_finish_task_no_steps() -> None:
    start("test goal")
    active_task = recorder._ACTIVE
    if active_task is not None:
        active_task.steps = []
    result = await finish(True)
    assert "Task 'test goal' recorded no reusable steps; nothing saved." in result


async def test_query_playbook_no_active_page(fake_session: FakeSession) -> None:
    import supporter.tools.browser.session as session_module

    original_active_page = session_module.active_page

    def mock_active_page2() -> None:
        return None

    session_module.active_page = mock_active_page2

    try:
        result = await query_playbook("test goal")
        assert "No active page; navigate first, then query a playbook." in result
    finally:
        session_module.active_page = original_active_page


async def test_record_step_exception_caught(fake_session: FakeSession) -> None:
    from supporter.tools.browser.recorder import _record_step

    recorder.start("test goal")

    original_record = recorder.record

    def _raising_record(step: object) -> None:
        raise ValueError("simulated record failure")

    recorder.record = _raising_record

    try:
        req = BrowseRequest(
            action="click",
            ref="e2",
        )
        await _record_step(req, "test result")
    finally:
        recorder.record = original_record
        recorder.discard()
