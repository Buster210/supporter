import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.containers import Vertical

from supporter.tui import SupporterApp
from supporter.tui.delegation_listener import (
    DelegationListener,
    format_completed_task_signal,
    format_delegation_progress,
)
from supporter.types import (
    MilestoneCancelled,
    MilestoneCompleted,
    MilestoneStarted,
    TaskAnomaly,
    TaskCompleted,
    TaskFailed,
    TaskOutputChunk,
    TaskSkipped,
    TaskStarted,
    TaskTimedOut,
)


class FakeDelegationBus:
    def __init__(
        self,
        events: list[Any],
        *,
        notify_per_task: bool = True,
        snapshot: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.notify_per_task = notify_per_task
        self._events = events
        self._snapshot = snapshot or {}

    def get_snapshot(self) -> dict[str, dict[str, Any]]:
        return self._snapshot

    def update_task_state(self, task_id: str, state: dict[str, Any]) -> None:
        self._snapshot[task_id] = state

    def subscribe(self) -> asyncio.Queue[Any]:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        for event in self._events:
            queue.put_nowait(event)
        queue.put_nowait(None)
        return queue


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
    assert "explorer" not in signal
    assert "Find current time in India." not in signal


@pytest.mark.asyncio
async def test_task_event_injects_explicit_completion_signal() -> None:
    bus = MagicMock()
    bus.notify_per_task = True
    bus.get_snapshot.return_value = {
        "get_time": {
            "agent_label": "explorer",
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
            "agent_label": "explorer",
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
    assert "| map | explorer | working |  |" in output
    assert "| review | code_reviewer | completed | 1.25s |" in output
    assert "Assigned task" not in output
    assert "Map root files" not in output
    assert "Review findings" not in output
    assert "Completed summaries" not in output
    assert "Review completed." not in output


def test_delegation_progress_formats_default_status_and_agent() -> None:
    bus = MagicMock()
    bus.get_snapshot.return_value = {"pending": {}}

    output = format_delegation_progress("job123", bus)

    assert "| pending | ? | pending |  |" in output


def test_delegation_progress_shows_streamed_output_tail() -> None:
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "build": {
            "status": "running",
            "agent_label": "agent-a",
            "duration": 0.5,
            "output_tail": "compiling sources\nlinking\n",
        }
    }

    output = format_delegation_progress("job123", bus)

    assert "[compiling sources]" in output


@pytest.mark.asyncio
async def test_listener_emits_anomaly_and_completed_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FakeDelegationBus(
        [
            MilestoneStarted("job123", "build", ["task1"], 1),
            TaskStarted("job123", "task1", "agent-a", 1.0, 10.0),
            TaskAnomaly("job123", "task1", "agent-a", 8.0, 10.0),
            TaskCompleted("job123", "task1", 1.25, "ok", "model-a"),
        ],
        snapshot={
            "task1": {
                "agent_label": "agent-a",
                "task_goal": "Check the thing",
                "status": "DONE",
                "duration": 1.25,
            }
        },
    )
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    inject_message = MagicMock()
    upsert_progress = AsyncMock()
    listener = DelegationListener(
        inject_message=inject_message,
        upsert_progress=upsert_progress,
        drop_progress=MagicMock(),
    )

    await listener.listen("job123")

    injected = [call.args[0] for call in inject_message.call_args_list]
    assert "AGENT ALERT: Task `task1` [agent-a]" in injected[0]
    assert injected[1] == format_completed_task_signal("job123", "task1")
    upsert_progress.assert_awaited_once_with("job123", bus)


@pytest.mark.asyncio
async def test_listener_emits_terminal_task_signals_with_inspect_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FakeDelegationBus(
        [
            TaskFailed("job123", "failed", 1.0, "boom"),
            TaskTimedOut("job123", "slow", 2.0),
            TaskSkipped("job123", "skipped", "blocked"),
        ],
        snapshot={
            "failed": {"agent_label": "agent-a", "task_goal": "Fail task"},
            "slow": {"agent_label": "agent-b", "task_goal": "Slow task"},
            "skipped": {"agent_label": "agent-c", "task_goal": "Skip task"},
        },
    )
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    inject_message = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        upsert_progress=AsyncMock(),
        drop_progress=MagicMock(),
    )

    await listener.listen("job123")

    injected = [call.args[0] for call in inject_message.call_args_list]
    assert len(injected) == 3
    assert injected[0].startswith("DELEGATION_TASK_FAIL:")
    assert injected[1].startswith("DELEGATION_TASK_TIMEOUT:")
    assert injected[2].startswith("DELEGATION_TASK_SKIP:")
    assert all('query_delegation(job_id="job123"' in msg for msg in injected)
    payload = json.loads(injected[0].split(": ", 1)[1].split("\n\nMore:", 1)[0])
    assert payload == {
        "job_id": "job123",
        "task_id": "failed",
        "agent": "agent-a",
        "assigned_task": "Fail task",
    }


@pytest.mark.asyncio
async def test_listener_suppresses_per_task_messages_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FakeDelegationBus(
        [TaskFailed("job123", "failed", 1.0, "boom")],
        notify_per_task=False,
    )
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    inject_message = MagicMock()
    upsert_progress = AsyncMock()
    listener = DelegationListener(
        inject_message=inject_message,
        upsert_progress=upsert_progress,
        drop_progress=MagicMock(),
    )

    await listener.listen("job123")

    upsert_progress.assert_awaited_once_with("job123", bus)
    inject_message.assert_not_called()


@pytest.mark.asyncio
async def test_listener_merges_output_tail_without_wiping_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FakeDelegationBus(
        [TaskOutputChunk("job123", "task1", "building module\n", 1)],
        snapshot={
            "task1": {
                "agent_label": "agent-a",
                "task_goal": "Build it",
                "status": "running",
                "duration": 0.5,
            }
        },
    )
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    upsert_progress = AsyncMock()
    listener = DelegationListener(
        inject_message=MagicMock(),
        upsert_progress=upsert_progress,
        drop_progress=MagicMock(),
    )

    await listener.listen("job123")

    state = bus.get_snapshot()["task1"]
    assert state["output_tail"] == "building module\n"
    # scheduler-owned fields survive the chunk merge
    assert state["status"] == "running"
    assert state["agent_label"] == "agent-a"
    assert state["duration"] == 0.5
    # newline boundary triggers a coalesced re-render
    upsert_progress.assert_awaited_once_with("job123", bus)


@pytest.mark.asyncio
async def test_listener_clears_tail_on_terminal_keeps_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FakeDelegationBus(
        [
            TaskOutputChunk("job123", "task1", "working\n", 1),
            TaskCompleted("job123", "task1", 1.0, "ok", "model-a"),
        ],
        snapshot={
            "task1": {
                "agent_label": "agent-a",
                "task_goal": "Build it",
                "status": "done",
                "duration": 1.0,
            }
        },
    )
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    listener = DelegationListener(
        inject_message=MagicMock(),
        upsert_progress=AsyncMock(),
        drop_progress=MagicMock(),
    )

    await listener.listen("job123")

    state = bus.get_snapshot()["task1"]
    assert "output_tail" not in state
    assert state["status"] == "done"
    assert state["agent_label"] == "agent-a"


@pytest.mark.asyncio
async def test_listener_injects_capsule_result_on_milestone_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"job_id": "job123", "status": "completed"}
    bus = FakeDelegationBus(
        [MilestoneCompleted("job123", "build", [], 3.21)],
    )
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    monkeypatch.setattr(
        "supporter.tools.delegate.api.serialize_capsule_result",
        lambda job_id: payload,
    )
    drop_progress = MagicMock()
    inject_message = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        upsert_progress=AsyncMock(),
        drop_progress=drop_progress,
    )

    await listener.listen("job123")

    message = inject_message.call_args.args[0]
    assert "DELEGATION_CAPSULE_RESULT (json):" in message
    assert '"status": "completed"' in message
    drop_progress.assert_called_once_with("job123")


@pytest.mark.asyncio
async def test_listener_falls_back_for_completed_milestone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FakeDelegationBus(
        [
            MilestoneCompleted(
                "job123",
                "build",
                [{"task_id": "failed", "status": "error"}],
                3.21,
            )
        ],
    )
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    monkeypatch.setattr(
        "supporter.tools.delegate.api.serialize_capsule_result",
        MagicMock(side_effect=RuntimeError("capsule unavailable")),
    )
    serialize_results = MagicMock(return_value={"status": "completed_with_failures"})
    monkeypatch.setattr(
        "supporter.tools.delegate.scheduler.serialize_results",
        serialize_results,
    )
    inject_message = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        upsert_progress=AsyncMock(),
        drop_progress=MagicMock(),
    )

    await listener.listen("job123")

    serialize_results.assert_called_once_with(
        "build",
        [{"task_id": "failed", "status": "error"}],
        3.21,
        "job123",
        status="completed_with_failures",
    )
    assert '"completed_with_failures"' in inject_message.call_args.args[0]


@pytest.mark.asyncio
async def test_listener_falls_back_for_cancelled_milestone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FakeDelegationBus([MilestoneCancelled("job123", "build", 3.214)])
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    monkeypatch.setattr(
        "supporter.tools.delegate.api.serialize_capsule_result",
        MagicMock(side_effect=RuntimeError("capsule unavailable")),
    )
    inject_message = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        upsert_progress=AsyncMock(),
        drop_progress=MagicMock(),
    )

    await listener.listen("job123")

    message = inject_message.call_args.args[0]
    assert '"status": "cancelled"' in message
    assert '"total_duration": 3.21' in message


@pytest.mark.asyncio
async def test_listener_logs_get_bus_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "supporter.tools.delegate.bus.get_bus",
        MagicMock(side_effect=RuntimeError("missing bus")),
    )
    logger_error = MagicMock()
    monkeypatch.setattr("supporter.tui.delegation_listener.logger.error", logger_error)
    listener = DelegationListener(
        inject_message=MagicMock(),
        upsert_progress=AsyncMock(),
        drop_progress=MagicMock(),
    )

    await listener.listen("job123")

    assert (
        "Delegation listener failed for job123: missing bus"
        in logger_error.call_args.args[0]
    )


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


@pytest.mark.asyncio
async def test_process_system_message_uses_agent_role_without_active_turn() -> None:
    app = MagicMock()
    app.active_turn = None
    app._process_message_cycle = AsyncMock()

    bound = SupporterApp._process_system_message.__get__(app, SupporterApp)
    await bound("system-only")

    app._process_message_cycle.assert_awaited_once_with(
        "system-only", mount_user=True, role="agent"
    )


def test_drop_delegation_progress_removes_stale_entry() -> None:
    app = MagicMock()
    app._delegation_bubbles = {"job123": object()}

    bound = SupporterApp._drop_delegation_progress.__get__(app, SupporterApp)
    bound("job123")

    assert app._delegation_bubbles == {}
