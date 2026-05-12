from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.containers import Vertical

from supporter.tui import SupporterApp
from supporter.tui.delegation_listener import (
    DelegationListener,
    format_completed_task_signal,
    format_delegation_progress,
)


def test_completed_task_signal_contains_only_ids() -> None:
    signal = format_completed_task_signal("job123", "get_time")

    assert signal.startswith("<br/>\n")
    assert signal.endswith("\n<br/>")
    assert "Delegation task completed" in signal
    assert "job_id: `job123`" in signal
    assert "task_id: `get_time`" in signal
    assert "Summary" not in signal
    assert "Confidence" not in signal
    assert "Evidence" not in signal
    assert "More" not in signal
    assert "query_delegation" not in signal
    assert "researcher" not in signal
    assert "Find current time in India." not in signal


@pytest.mark.asyncio
async def test_task_event_injects_explicit_completion_signal() -> None:
    bus = MagicMock()
    bus.notify_per_task = True
    bus.get_snapshot.return_value = {
        "get_time": {
            "agent_label": "researcher",
            "task_goal": "Find current time in India.",
        }
    }
    upsert_progress = AsyncMock()
    inject_message = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        upsert_progress=upsert_progress,
        drop_progress=MagicMock(),
    )
    signal = format_completed_task_signal("job123", "get_time")

    await listener._emit_task_event(
        bus,
        "job123",
        "DONE",
        "get_time",
        sys_body=signal,
    )

    upsert_progress.assert_awaited_once_with("job123", bus)
    system_call = inject_message.call_args.args[0]
    assert system_call == signal
    assert "DELEGATION_TASK_DONE:" not in system_call


def test_delegation_progress_omits_task_details_and_summaries() -> None:
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "map": {
            "status": "RUNNING",
            "agent_label": "scout",
            "task_goal": "Map root files",
            "duration": 0.0,
        },
        "review": {
            "status": "DONE",
            "agent_label": "code_reviewer",
            "task_goal": "Review findings",
            "duration": 1.25,
            "summary": "Review completed.",
        },
    }

    output = format_delegation_progress("job123", bus)

    assert "| Task | Agent | Status | Time |" in output
    assert "| map | scout | working |  |" in output
    assert "| review | code_reviewer | completed | 1.25s |" in output
    assert "Assigned task" not in output
    assert "Map root files" not in output
    assert "Review findings" not in output
    assert "Completed summaries" not in output
    assert "Review completed." not in output


@pytest.mark.asyncio
async def test_delegation_signal_renders_immediately_and_queues_for_agent() -> None:
    app = MagicMock()
    app._is_processing = True
    app._user_message_queue = []
    app.run_worker = MagicMock()
    app.query_one = MagicMock()

    signal = format_completed_task_signal("job123", "get_time")
    bound = SupporterApp._inject_system_message.__get__(app, SupporterApp)
    bound(signal)

    app.run_worker.assert_called_once()
    assert app._user_message_queue == [(signal, True)]


@pytest.mark.asyncio
async def test_render_delegation_signal_mounts_centered_label() -> None:
    app = MagicMock()
    chat_view = MagicMock(spec=Vertical)
    chat_view.mount = AsyncMock()
    chat_view.scroll_end = MagicMock()
    app.query_one = MagicMock(return_value=chat_view)
    app.active_turn = None

    bound = SupporterApp._render_delegation_signal.__get__(app, SupporterApp)
    await bound(format_completed_task_signal("job123", "get_time"))

    chat_view.mount.assert_awaited_once()
    label = chat_view.mount.call_args.args[0]
    assert label.has_class("delegation-signal")
    rendered = str(label.render())
    assert "Delegation task completed" in rendered
    assert "job_id: job123" in rendered
    assert "<br/>" not in rendered
    assert "`" not in rendered


def test_drop_delegation_progress_removes_stale_entry() -> None:
    app = MagicMock()
    app._delegation_bubbles = {"job123": object()}

    bound = SupporterApp._drop_delegation_progress.__get__(app, SupporterApp)
    bound("job123")

    assert app._delegation_bubbles == {}
