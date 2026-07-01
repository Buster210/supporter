"""G3 acceptance: no raw JSON in UI, clean formatting, live progress."""

from unittest.mock import MagicMock

from supporter.tui.delegation import DelegationBlock
from supporter.tui.delegation_listener import (
    DelegationListener,
    format_delegation_summary,
    format_delegation_update,
)


def test_task_terminal_routes_to_render_task_done() -> None:
    """Task-complete goes to the dedicated callback, not the generic signal."""
    render_task_done = MagicMock()
    render_signal = MagicMock()
    listener = DelegationListener(
        inject_message=MagicMock(),
        drop_progress=MagicMock(),
        render_signal=render_signal,
        render_task_done=render_task_done,
    )
    bus = MagicMock()
    bus.get_snapshot.return_value = {"t1": {"agent_label": "planner"}}
    event = MagicMock()
    event.task_id = "t1"

    listener._on_task_terminal(event, "job9", bus, "DONE")

    render_task_done.assert_called_once_with("job9", "Task t1 (planner) completed")
    render_signal.assert_not_called()


def test_delegation_block_signal_accumulates() -> None:
    """Multiple task completions accumulate into one ordered signal section."""
    block = DelegationBlock()
    block.set_signal("Task a completed")
    block.set_signal("Task b completed")
    assert block._signal_text == "Task a completed\nTask b completed"


def test_delegation_block_section_order() -> None:
    """compose yields progress -> signal -> result -> plan, in that order."""
    import inspect

    src = inspect.getsource(DelegationBlock.compose)
    order = [
        src.index("_progress_widget ="),
        src.index("_signal_widget ="),
        src.index("_result_widget ="),
        src.index("_plan_widget ="),
    ]
    assert order == sorted(order), order


def test_no_raw_json_in_task_signal() -> None:
    """AC3: Task signal is clean single line, no markup/backticks/pipes."""
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "build": {
            "agent_label": "explorer",
            "task_goal": "Build module",
            "status": "DONE",
        }
    }

    signal = format_delegation_update("job1", bus, task_id="build", status="DONE")

    # Clean single-line format, no markup/backticks/pipes.
    assert signal == "Task build (explorer) completed"
    assert "<br/>" not in signal
    assert "`" not in signal
    assert "|" not in signal
    assert "{" not in signal
    assert "json" not in signal.lower()
    assert "\n" not in signal  # Single line


def test_progress_table_no_raw_json() -> None:
    """AC1: Progress table renders as clean markdown, no JSON or technical terms."""
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "task1": {
            "status": "running",
            "agent_label": "explorer",
            "task_goal": "Explore files",
            "duration": 2.5,
        }
    }

    output = format_delegation_update("job1", bus)

    # Markdown table, no JSON.
    assert "| Task | Agent | Status | Time |" in output
    assert "| task1 | explorer | working | 2.50s |" in output
    assert "{" not in output
    assert '"' not in output  # No JSON quotes
    assert "json" not in output.lower()


def test_progress_signal_single_line() -> None:
    """AC3: Each signal is a single clean line, no pipe chars or raw syntax."""
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "verify": {
            "agent_label": "reviewer",
            "task_goal": "Verify code",
            "status": "FAIL",
        }
    }

    signal = format_delegation_update("job1", bus, task_id="verify", status="FAIL")

    # Clean single-line format with no markup.
    assert signal == "Task verify (reviewer) failed"
    assert "<br/>" not in signal
    assert "`" not in signal
    assert "|" not in signal
    assert "\n" not in signal  # Single line


def test_capsule_injection_feeds_model_only() -> None:
    """AC1: Capsule JSON reaches LLM (system message), rendering skipped for UI."""
    inject_message = MagicMock()
    listener = DelegationListener(
        inject_message=inject_message,
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
    )

    payload = {"job_id": "j1", "status": "completed", "tasks": []}
    listener._inject_capsule_result(payload)

    # A human-readable capsule summary is sent to the model (system message).
    model_msg = inject_message.call_args.args[0]
    assert "Delegation result" in model_msg
    assert "status: completed" in model_msg


def test_render_progress_live_mounts_bubble_not_buffered() -> None:
    """AC2: Progress renders live as updates arrive, not buffered until end."""
    render_progress_live = MagicMock()
    listener = DelegationListener(
        inject_message=MagicMock(),
        drop_progress=MagicMock(),
        render_signal=MagicMock(),
        render_progress_live=render_progress_live,
    )

    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "task1": {
            "status": "running",
            "agent_label": "explorer",
            "duration": 1.0,
        }
    }

    if listener._render_progress_live is not None:
        progress_md = format_delegation_update("job1", bus)
        listener._render_progress_live("job1", progress_md)

        render_progress_live.assert_called_once()
        rendered = render_progress_live.call_args.args[1]
        assert "| Task | Agent | Status | Time |" in rendered
        assert "task1" in rendered


def test_task_signal_labels_are_natural_language() -> None:
    """AC3: Status labels are human-friendly, not machine format codes."""
    bus = MagicMock()

    statuses = [
        ("DONE", "completed"),
        ("FAIL", "failed"),
        ("TIMEOUT", "timed out"),
        ("SKIP", "skipped"),
    ]
    for kind, expected_label in statuses:
        bus.get_snapshot.return_value = {
            "t1": {"agent_label": "agent-x", "task_goal": "Task"}
        }
        signal = format_delegation_update("job1", bus, task_id="t1", status=kind)
        assert f"Task t1 (agent-x) {expected_label}" == signal
        assert kind not in signal  # No raw code


def test_progress_output_tail_truncated_no_overflow() -> None:
    """AC3: Output tail in progress table is bounded, single-line in table cell."""
    bus = MagicMock()
    long_output = "line1\n" * 100 + "final line"
    bus.get_snapshot.return_value = {
        "compile": {
            "status": "running",
            "agent_label": "builder",
            "duration": 0.0,
            "output_tail": long_output,
        }
    }

    output = format_delegation_update("job1", bus)

    assert "[" in output and "]" in output
    lines = output.splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    assert len(table_lines) >= 3
    for row in table_lines:
        assert row.count("|") >= 4


def test_delegation_summary_compact() -> None:
    """AC4: Completion summary is compact one-liner (what done + result)."""
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "task1": {"status": "DONE"},
        "task2": {"status": "DONE"},
        "task3": {"status": "FAIL"},
    }

    summary = format_delegation_summary("job123", bus)

    assert summary == "Job job123: 2/3 tasks completed with issues"
    assert "\n" not in summary
    assert "{" not in summary
    assert "json" not in summary.lower()


def test_delegation_summary_all_completed() -> None:
    """AC4: Summary shows 'completed' when all tasks succeed."""
    bus = MagicMock()
    bus.get_snapshot.return_value = {
        "task1": {"status": "DONE"},
        "task2": {"status": "DONE"},
    }

    summary = format_delegation_summary("job123", bus)

    assert summary == "Job job123: 2/2 tasks completed"
    assert "issues" not in summary
