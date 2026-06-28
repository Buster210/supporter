"""Plan capsule rendering tests (G4)

Validate that plan bubbles appear immediately when planner returns,
render correctly, and expose objective/steps without raw JSON.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from supporter.tools.delegate.capsule_view import format_plan_capsule
from supporter.tui import SupporterApp
from supporter.tui.delegation import DelegationBlock


def _plan_payload() -> dict:
    return {
        "job_id": "j1",
        "agent": "planner",
        "milestone": "Find file",
        "status": "completed",
        "tasks": [
            {"id": "t1", "status": "done", "summary": "read src"},
            {"id": "t2", "status": "done", "summary": "grep pattern"},
        ],
    }


def test_set_plan_buffers_until_composed() -> None:
    """AC1: A plan set before compose is buffered, applied once mounted."""
    block = DelegationBlock()
    block.set_plan("## Plan: do X\n1. step A")
    # Widgets not composed yet → buffered.
    assert block._pending["plan"] == "## Plan: do X\n1. step A"


@pytest.mark.asyncio
async def test_plan_displays_immediately_when_mounted() -> None:
    """AC1: Once composed, the plan section becomes visible immediately."""
    app = SupporterApp()
    async with app.run_test(size=(80, 24)):
        block = DelegationBlock()
        await app.mount(block)
        block.set_plan("## Plan: do X\n1. step A\n2. step B")
        assert block._plan_widget is not None
        assert block._plan_widget.display is True
        assert "plan" not in block._pending


def test_inject_plan_bubble_schedules_mount() -> None:
    """AC1: _inject_plan_bubble schedules the mount (does not block)."""
    app = MagicMock()
    bound = SupporterApp._inject_plan_bubble.__get__(app, SupporterApp)

    bound("## Plan: title\n1. item")

    # Scheduling goes through _safe_call so the mount runs on the UI thread.
    app._safe_call.assert_called_once()


def test_format_plan_capsule_renders_objective_and_steps() -> None:
    """AC2: format_plan_capsule includes objective and task summaries."""
    result = format_plan_capsule(_plan_payload())
    assert "Find file" in result
    assert "read src" in result
    assert "grep pattern" in result


def test_plan_capsule_no_raw_json_in_bubble() -> None:
    """AC3: Plan markdown contains NO raw JSON, only formatted sections."""
    result = format_plan_capsule(_plan_payload())
    assert '"agent"' not in result
    assert "DELEGATION_CAPSULE_RESULT" not in result
    assert "## Plan: Find file" in result


@pytest.mark.asyncio
async def test_plan_stays_visible_during_execution() -> None:
    """AC4: Plan remains visible as progress updates arrive."""
    app = SupporterApp()
    async with app.run_test(size=(80, 24)):
        block = DelegationBlock()
        await app.mount(block)
        block.set_plan("## Plan: do X\n1. step A")
        progress = "Job `j1`\n\n| Task | Status |\n| --- | --- |\n| t1 | working |"
        block.set_progress(progress)
        assert block._plan_widget is not None
        assert block._plan_widget.display is True


def test_format_plan_capsule_with_full_content() -> None:
    """AC2: Comprehensive rendering of objective, tasks, and summary totals."""
    payload = {
        "job_id": "j2",
        "milestone": "Complete task",
        "status": "completed",
        "tasks": [
            {"id": "s1", "status": "done", "summary": "Step 1 output here"},
            {"id": "s2", "status": "done", "summary": "Step 2 output here"},
        ],
        "totals": {"completed": 2, "failed": 0},
    }
    result = format_plan_capsule(payload)

    assert "Complete task" in result
    assert "Step 1 output here" in result
    assert "Step 2 output here" in result
    assert "2 completed" in result
    # No excessive blank lines.
    assert "\n\n\n" not in result
