from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.tools.browser import recorder
from supporter.tools.browser import session as session_module
from supporter.tools.browser.core import BrowseRequest
from supporter.tools.browser.recorder import discard, finish, start
from supporter.tools.browser.task import (
    delete_playbook,
    finish_task,
    list_playbooks,
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
    active_task = recorder._ACTIVE.get("main")
    if active_task is not None:
        active_task.failed = True
    result = await finish(True)
    assert "Task 'test goal' had an error or unsafe step; nothing saved." in result


async def test_finish_task_no_steps() -> None:
    start("test goal")
    active_task = recorder._ACTIVE.get("main")
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


async def test_finish_task_with_active_session(fake_session: FakeSession) -> None:
    """Covers task.py:80-87 — finish_task with an active page and empty lifecycle."""
    start("buy milk")

    fake_page = MagicMock()
    fake_page.url = "https://example.com/"

    with (
        patch.object(session_module, "active_page", return_value=fake_page),
        patch("supporter.tools.browser.task._page_host", return_value="example.com"),
        patch(
            "supporter.tools.browser.task.session.resolve_close_at_task_end",
            new=AsyncMock(return_value=""),
        ),
    ):
        result = await finish_task(True)

    assert result  # non-empty string returned


async def test_list_playbooks_with_active_page_no_playbooks(
    fake_session: FakeSession,
) -> None:
    """Covers task.py:262-265 — list_playbooks with no saved playbooks."""
    fake_page = MagicMock()
    fake_page.url = "https://example.com/"

    with (
        patch.object(session_module, "active_page", return_value=fake_page),
        patch("supporter.tools.browser.task._page_host", return_value="example.com"),
        patch("supporter.tools.browser.task._list_playbooks_sync", return_value=[]),
        patch("supporter.tools.browser.task.prune_playbooks"),
    ):
        result = await list_playbooks()

    assert "No playbooks found" in result


async def test_list_playbooks_with_active_page_has_playbooks(
    fake_session: FakeSession,
) -> None:
    """Covers task.py:262-270 — list_playbooks with at least one saved playbook."""
    fake_page = MagicMock()
    fake_page.url = "https://example.com/"

    playbook_entry = {
        "goal": "Sign in",
        "step_count": 3,
        "success_count": 2,
        "fail_count": 1,
    }

    with (
        patch.object(session_module, "active_page", return_value=fake_page),
        patch("supporter.tools.browser.task._page_host", return_value="example.com"),
        patch(
            "supporter.tools.browser.task._list_playbooks_sync",
            return_value=[playbook_entry],
        ),
        patch("supporter.tools.browser.task.prune_playbooks"),
    ):
        result = await list_playbooks()

    assert "Sign in" in result
    assert "2✓/1✗" in result


async def test_delete_playbook_no_active_page() -> None:
    """Covers task.py:291 — delete_playbook when no page is active."""
    with patch.object(session_module, "active_page", return_value=None):
        result = await delete_playbook("some goal")

    assert "No active page" in result


async def test_delete_playbook_success(fake_session: FakeSession) -> None:
    """Covers task.py:295 — delete_playbook when the playbook exists."""
    fake_page = MagicMock()
    fake_page.url = "https://example.com/"

    with (
        patch.object(session_module, "active_page", return_value=fake_page),
        patch("supporter.tools.browser.task._page_host", return_value="example.com"),
        patch(
            "supporter.tools.browser.task.asyncio.to_thread",
            new=AsyncMock(return_value=True),
        ),
    ):
        result = await delete_playbook("sign in")

    assert "Deleted playbook" in result
    assert "sign in" in result


async def test_query_playbook_not_found(fake_session: FakeSession) -> None:
    """Covers task.py:201 — query_playbook when no playbook is saved for goal."""
    fake_page = MagicMock()
    fake_page.url = "https://example.com/"

    with (
        patch.object(session_module, "active_page", return_value=fake_page),
        patch("supporter.tools.browser.task._page_host", return_value="example.com"),
        patch("supporter.tools.browser.task.load_playbook", return_value=None),
    ):
        result = await query_playbook("unknown goal")

    assert result  # non-empty "no playbook" message
