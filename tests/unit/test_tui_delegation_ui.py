"""Tests for delegation UI — consolidated formatter, dispatch-map event loop,
and buffer-removal (direct streaming).
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.tui import SupporterApp
from supporter.tui.chat import ChatContainer
from supporter.tui.delegation_listener import (
    DelegationListener,
    format_delegation_update,
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
        snapshot: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._events = list(events)
        self._subscribers: list[Any] = []
        self._snapshot: dict[str, dict[str, Any]] = snapshot or {}

    def subscribe(self) -> Any:
        import asyncio

        queue: asyncio.Queue[Any] = asyncio.Queue()
        for event in self._events:
            queue.put_nowait(event)
        queue.put_nowait(None)  # sentinel
        return queue

    def unsubscribe(self, _queue: Any) -> None:
        pass

    def get_snapshot(self) -> dict[str, dict[str, Any]]:
        return self._snapshot

    def update_task_state(self, task_id: str, state: dict[str, Any]) -> None:
        self._snapshot[task_id] = state


def test_task_signal_contains_status_and_agent() -> None:
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "get_time": {
            "agent_label": "explorer",
            "task_goal": "Find current time in India.",
            "status": "DONE",
        }
    }

    signal = format_delegation_update("job123", bus, task_id="get_time")

    # AC3: clean single-line format, no markup
    assert signal == "Task get_time (explorer) completed"
    assert "<br/>" not in signal
    assert "`" not in signal
    assert "Delegation task" not in signal
    assert "Find current time in India." not in signal
    assert "query_delegation" not in signal
    assert "DELEGATION_TASK_DONE:" not in signal
    assert "\n" not in signal  # Single line


@pytest.mark.asyncio
async def test_task_event_renders_signal_without_model_injection() -> None:
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "get_time": {
            "agent_label": "explorer",
            "task_goal": "Find current time in India.",
        }
    }
    inject_message = MagicMock()
    render_signal = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        drop_progress=MagicMock(),
        render_signal=render_signal,
    )

    listener._on_task_terminal(
        TaskCompleted("job123", "get_time", 1.0, "ok", "model"),
        "job123",
        bus,
        "DONE",
    )

    rendered = render_signal.call_args.args[0]
    # AC3: clean signal format
    assert rendered == "Task get_time (explorer) completed"
    inject_message.assert_not_called()


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

    output = format_delegation_update("job123", bus)

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

    output = format_delegation_update("job123", bus)

    # AC3: natural language status "waiting" for pending state
    assert "| pending | ? | waiting |  |" in output


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

    output = format_delegation_update("job123", bus)

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
    render_signal = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        drop_progress=MagicMock(),
        render_signal=render_signal,
    )

    await listener.listen("job123")

    rendered = [call.args[0] for call in render_signal.call_args_list]
    assert "AGENT ALERT: Task `task1` [agent-a]" in rendered[0]
    # AC3: clean signal format "Task <id> (<agent>) <status>"
    assert rendered[1] == "Task task1 (agent-a) completed"
    inject_message.assert_not_called()


@pytest.mark.asyncio
async def test_listener_renders_terminal_task_signals_to_ui(
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
    render_signal = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        drop_progress=MagicMock(),
        render_signal=render_signal,
    )

    await listener.listen("job123")

    rendered = [call.args[0] for call in render_signal.call_args_list]
    assert len(rendered) == 3
    # AC3: clean signal format, no markup/backticks
    assert rendered[0] == "Task failed (agent-a) failed"
    assert "Fail task" not in rendered[0]
    assert rendered[1] == "Task slow (agent-b) timed out"
    assert rendered[2] == "Task skipped (agent-c) skipped"
    assert all("query_delegation" not in msg for msg in rendered)
    assert all("DELEGATION_TASK_" not in msg for msg in rendered)
    assert all("<br/>" not in msg for msg in rendered)  # No markup
    inject_message.assert_not_called()


@pytest.mark.asyncio
async def test_listener_never_injects_per_task_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = FakeDelegationBus(
        [TaskFailed("job123", "failed", 1.0, "boom")],
        snapshot={"failed": {"agent_label": "agent-a", "task_goal": "Fail task"}},
    )
    monkeypatch.setattr("supporter.tools.delegate.bus.get_bus", lambda job_id: bus)
    inject_message = MagicMock()
    render_signal = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        drop_progress=MagicMock(),
        render_signal=render_signal,
    )

    await listener.listen("job123")

    render_signal.assert_called_once()
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
    listener = DelegationListener(
        inject_message=MagicMock(),
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
    )

    await listener.listen("job123")

    state = bus.get_snapshot()["task1"]
    assert state["output_tail"] == "building module\n"
    # scheduler-owned fields survive the chunk merge
    assert state["status"] == "running"
    assert state["agent_label"] == "agent-a"
    assert state["duration"] == 0.5


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
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
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
        drop_progress=drop_progress,
        render_signal=MagicMock(),
    )

    await listener.listen("job123")

    message = inject_message.call_args.args[0]
    assert "Delegation result" in message
    assert "status: completed" in message
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
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
    )

    await listener.listen("job123")

    serialize_results.assert_called_once_with(
        "build",
        [{"task_id": "failed", "status": "error"}],
        3.21,
        "job123",
        status="completed_with_failures",
    )
    assert "completed_with_failures" in inject_message.call_args.args[0]


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
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
    )

    await listener.listen("job123")

    message = inject_message.call_args.args[0]
    assert "status: cancelled" in message


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
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
    )

    await listener.listen("job123")

    assert (
        "Delegation listener failed for job123: missing bus"
        in logger_error.call_args.args[0]
    )


def test_inject_system_message_queues_when_processing() -> None:
    app = MagicMock()
    app._is_processing = True
    app._user_message_queue = []
    app.run_worker = MagicMock()
    app.query_one = MagicMock()

    bound = SupporterApp._inject_system_message.__get__(app, SupporterApp)
    bound("a system message")

    app.run_worker.assert_not_called()
    assert app._user_message_queue == [("a system message", True)]


@pytest.mark.asyncio
async def test_capsule_result_reaches_model_without_raw_bubble() -> None:
    app = MagicMock()
    app.active_turn = None
    app._process_message_cycle = AsyncMock()

    msg = "DELEGATION_CAPSULE_RESULT (json):\n```json\n{}\n```"
    bound = SupporterApp._process_system_message.__get__(app, SupporterApp)
    await bound(msg)

    app._process_message_cycle.assert_awaited_once_with(
        msg, mount_user=False, role="agent"
    )


@pytest.mark.asyncio
async def test_render_delegation_signal_mounts_centered_label() -> None:
    app = MagicMock()
    app._is_streaming = False
    chat_view = MagicMock(spec=ChatContainer)
    chat_view.mount = AsyncMock()
    chat_view.follow_end = MagicMock()
    app.query_one = MagicMock(return_value=chat_view)
    chat_view.query.return_value = []
    app.active_turn = None
    app._delegation_mount_target = MagicMock(return_value=chat_view)
    app._delegation_host_bubble = SupporterApp._delegation_host_bubble.__get__(
        app, SupporterApp
    )
    app._mount_delegation_widget = SupporterApp._mount_delegation_widget.__get__(
        app, SupporterApp
    )
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "get_time": {
            "agent_label": "explorer",
            "task_goal": "Find time",
            "status": "DONE",
        }
    }

    bound = SupporterApp._render_delegation_signal.__get__(app, SupporterApp)
    await bound(format_delegation_update("job123", bus, task_id="get_time"))

    chat_view.mount.assert_awaited_once()
    label = chat_view.mount.call_args.args[0]
    assert label.has_class("delegation-signal")
    rendered = str(label.render())
    # AC3: clean single-line signal, no markup
    assert "Task get_time (explorer) completed" in rendered
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
        "system-only", mount_user=False, role="agent"
    )


def test_planner_capsule_mounts_visible_bubble_and_feeds_model() -> None:
    inject_message = MagicMock()
    plan_bubble_injector = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
        plan_bubble_injector=plan_bubble_injector,
    )

    payload = {
        "job_id": "job123",
        "agent": "planner",
        "milestone": "Design the thing",
        "status": "completed",
        "tasks": [{"id": "t1", "status": "done", "summary": "lay out modules"}],
    }
    listener._inject_capsule_result(payload)

    # Visible bubble mounted with real markdown (not a raw JSON dump).
    plan_bubble_injector.assert_called_once()
    markdown = plan_bubble_injector.call_args.args[0]
    assert "## Plan: Design the thing" in markdown
    assert "lay out modules" in markdown
    assert "DELEGATION_CAPSULE_RESULT" not in markdown
    # ...and a human-readable summary still reaches the model for synthesis.
    model_msg = inject_message.call_args.args[0]
    assert "Delegation result" in model_msg
    assert "agent: planner" in model_msg


def test_worker_capsule_does_not_mount_bubble() -> None:
    inject_message = MagicMock()
    plan_bubble_injector = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
        plan_bubble_injector=plan_bubble_injector,
    )

    payload = {"job_id": "job123", "agent": "worker", "status": "completed"}
    listener._inject_capsule_result(payload)

    plan_bubble_injector.assert_not_called()
    assert "Delegation result" in inject_message.call_args.args[0]


def test_drop_delegation_progress_collapses_block() -> None:
    # G7: terminal state collapses the job's delegation block (kept for history).
    block = MagicMock()
    app = MagicMock()
    app._delegation_blocks = {"job123": block}

    bound = SupporterApp._drop_delegation_progress.__get__(app, SupporterApp)
    bound("job123")

    block.collapse_when_done.assert_called_once_with()


@pytest.mark.asyncio
async def test_process_system_message_capsule_uses_agent_role() -> None:
    app = MagicMock()
    app.active_turn = None
    app._process_message_cycle = AsyncMock()

    msg = "DELEGATION_CAPSULE_RESULT (json):\n```json\n{}\n```"
    bound = SupporterApp._process_system_message.__get__(app, SupporterApp)
    await bound(msg)

    app._process_message_cycle.assert_awaited_once_with(
        msg, mount_user=False, role="agent"
    )
