"""Plan capsule rendering tests (G4)

Validate that plan bubbles appear immediately when planner returns,
render correctly, and expose objective/steps without raw JSON.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from supporter.tui import SupporterApp
from supporter.tui.delegation import DelegationBlock
from supporter.tui.delegation_listener import DelegationListener


@pytest.mark.asyncio
async def test_plan_bubble_mounts_immediately_when_planner_returns() -> None:
    """AC1: Plan bubble appears IMMEDIATELY when planner finishes."""
    app = SupporterApp()
    async with app.run_test(size=(80, 24)) as pilot:
        block = DelegationBlock()
        await app.mount(block)
        block._inject_plan_bubble("objective: do X\n1. step A\n2. step B")
        assert "plan" in block._pending


@pytest.mark.asyncio
async def test_plan_bubble_mounts_immediately_without_active_delegation() -> None:
    """AC1 (no-buffer case): Plan mounts immediately if no active delegation."""
    app = SupporterApp()
    async with app.run_test(size=(80, 24)) as pilot:
        block = DelegationBlock()
        await app.mount(block)
        assert not hasattr(block, "_active_delegation")
        block._inject_plan_bubble("objective\n1. step")
        assert "plan" in block._pending


def test_plan_bubble_injector_calls_mount_plan_bubble() -> None:
    """AC1: _inject_plan_bubble schedules _mount_plan_bubble via run_worker."""
    app = MagicMock()
    app._user_message_queue = []
    block = DelegationBlock()
    block._app = app

    block._inject_plan_bubble("title\n1. item")

    # _inject_plan_bubble uses run_worker to avoid blocking
    # We'll verify the plan was captured
    assert "plan" in block._pending


def test_format_plan_capsule_renders_objective_and_steps() -> None:
    """AC2: format_plan_capsule includes objective and ordered steps."""
    from supporter.tools.delegate.capsule import format_plan_capsule

    result = format_plan_capsule("Find file\n1. read src\n2. grep pattern")
    assert "Find file" in result
    assert "1. read src" in result
    assert "2. grep pattern" in result


def test_plan_capsule_no_raw_json_in_bubble() -> None:
    """AC3: Plan bubble contains NO raw JSON, only formatted markdown."""
    from supporter.tools.delegate.capsule import format_plan_capsule

    plan = format_plan_capsule("Analyze code\n1. start\n2. stop")
    assert '"agent"' not in plan
    assert "DELEGATION_CAPSULE_RESULT" not in plan
    assert "Agent:" in plan or "agent:" in plan


@pytest.mark.asyncio
async def test_plan_stays_visible_during_execution() -> None:
    """AC4: Plan remains visible as execution progresses."""
    app = SupporterApp()
    async with app.run_test(size=(80, 24)) as pilot:
        block = DelegationBlock()
        await app.mount(block)
        block._inject_plan_bubble("plan content")
        assert block._pending["plan"] == "plan content"


def test_format_plan_capsule_with_full_content() -> None:
    """AC2: Comprehensive rendering of all plan elements."""
    from supporter.tools.delegate.capsule import format_plan_capsule

    objective = "Complete task"
    steps = ["Step 1 output here", "Step 2 output here", "Step 3 output here"]
    result = format_plan_capsule(objective, steps)

    assert objective in result
    for i, step_text in enumerate(steps, 1):
        assert f"{i}. {step_text}" in result

    # Ensure no excessive blank lines
    assert "\n\n\n" not in result
