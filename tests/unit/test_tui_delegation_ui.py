from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.tui import SupporterApp


def test_completed_task_signal_contains_only_ids() -> None:
    signal = SupporterApp._format_completed_task_signal("job123", "get_time")

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
    class FakeApp:
        def __init__(self) -> None:
            self._upsert_delegation_progress = AsyncMock()
            self._inject_system_message = MagicMock()
            self._format_task_signal = SupporterApp._format_task_signal

        def _safe_call(self, callback: Any, *args: Any, **kwargs: Any) -> None:
            callback(*args, **kwargs)

    bus = MagicMock()
    bus.notify_per_task = True
    bus.get_snapshot.return_value = {
        "get_time": {
            "agent_label": "researcher",
            "task_goal": "Find current time in India.",
        }
    }
    app = FakeApp()
    signal = SupporterApp._format_completed_task_signal("job123", "get_time")

    await SupporterApp._emit_task_event(
        cast(Any, app),
        bus,
        "job123",
        "DONE",
        "get_time",
        6.94,
        "Visible summary only.",
        sys_body=signal,
    )

    app._upsert_delegation_progress.assert_awaited_once_with(job_id="job123", bus=bus)
    system_call = app._inject_system_message.call_args.args[0]
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

    output = SupporterApp._format_delegation_progress("job123", bus)

    assert "| Task | Agent | Status | Time |" in output
    assert "| map | scout | working |  |" in output
    assert "| review | code_reviewer | completed | 1.25s |" in output
    assert "Assigned task" not in output
    assert "Map root files" not in output
    assert "Review findings" not in output
    assert "Completed summaries" not in output
    assert "Review completed." not in output
