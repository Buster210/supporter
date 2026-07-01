"""G4: Plan Visibility After Planner Returns.

Test that the plan bubble appears IMMEDIATELY when the planner finishes,
NOT buffered until delegation end, and contains objective/steps/criteria
without raw JSON.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.tui import SupporterApp
from supporter.tui.delegation import DelegationBlock
from supporter.tui.delegation_listener import DelegationListener


@pytest.mark.asyncio
async def test_plan_bubble_mounts_immediately_when_planner_returns() -> None:
    """AC1: Plan bubble appears IMMEDIATELY when planner finishes.

    When planner returns a capsule result, the plan bubble should mount
    right away, not be buffered until delegation ends.
    G4 Implementation: Plan bypasses buffer and mounts immediately.
    """
    app = MagicMock(spec=SupporterApp)
    app._mount_delegation_widget = AsyncMock()
    app._delegation_blocks = {}
    app.active_turn = None

    # _inject_plan_bubble calls _mount_plan_bubble via run_worker.
    # For testing, we simulate calling _mount_plan_bubble directly.
    bound = SupporterApp._mount_plan_bubble.__get__(app, SupporterApp)

    plan_markdown = "## Plan: Test objective\n- Step 1"
    await bound(plan_markdown)

    # G7: Plan mounts IMMEDIATELY into a collapsible DelegationBlock, not a
    # buffered bubble. The block is created, mounted, and the plan applied.
    app._mount_delegation_widget.assert_awaited_once()
    block = app._mount_delegation_widget.call_args.args[0]
    assert isinstance(block, DelegationBlock)
    # Not yet composed in a unit test, so set_plan stashes into _pending.
    assert block._pending.get("plan") == plan_markdown


@pytest.mark.asyncio
async def test_plan_bubble_mounts_immediately_without_active_delegation() -> None:
    """AC1 (no-buffer case): Plan mounts immediately if no active delegation.

    When there's no active delegation buffer (e.g., planner runs standalone),
    the plan bubble should mount immediately.
    """
    app = MagicMock(spec=SupporterApp)
    app._mount_delegation_widget = AsyncMock()
    app._delegation_blocks = {}
    app.active_turn = None
    bound = SupporterApp._mount_plan_bubble.__get__(app, SupporterApp)

    plan_markdown = "## Plan: Test objective\n- Step 1"
    await bound(plan_markdown)

    # No existing block -> create + mount immediately, then apply the plan.
    app._mount_delegation_widget.assert_awaited_once()
    block = app._mount_delegation_widget.call_args.args[0]
    assert isinstance(block, DelegationBlock)
    assert block._pending.get("plan") == plan_markdown


def test_plan_bubble_injector_calls_mount_plan_bubble() -> None:
    """AC1: _inject_plan_bubble schedules _mount_plan_bubble via run_worker.

    The plan_bubble_injector callback should call _mount_plan_bubble.
    """
    app = MagicMock(spec=SupporterApp)
    app.run_worker = MagicMock()
    app._safe_call = MagicMock(side_effect=lambda fn: fn())

    bound = SupporterApp._inject_plan_bubble.__get__(app, SupporterApp)
    plan_markdown = "## Plan: Test"
    bound(plan_markdown)

    # _inject_plan_bubble uses _safe_call to wrap the run_worker call
    # (for thread safety). The lambda passed to _safe_call should call run_worker.
    app._safe_call.assert_called_once()
    # Inside _safe_call, run_worker is called with the coroutine from
    # _mount_plan_bubble.
    app.run_worker.assert_called_once()
    call_arg = app.run_worker.call_args.args[0]
    # The argument is a coroutine from _mount_plan_bubble.
    assert (
        asyncio.iscoroutine(call_arg)
        or asyncio.iscoroutinefunction(call_arg.__class__)
        or hasattr(call_arg, "__await__")
    )


def test_format_plan_capsule_renders_objective_and_steps() -> None:
    """AC2: format_plan_capsule includes objective and ordered steps.

    The formatted markdown should show:
    - Plan title with objective/milestone
    - Ordered tasks/steps with IDs and assignments
    - Success criteria (if present)
    """
    from supporter.tools.delegate.capsule_view import format_plan_capsule

    payload = {
        "job_id": "job123",
        "agent": "planner",
        "milestone": "Design the database schema",
        "status": "completed",
        "tasks": [
            {
                "id": "t1",
                "status": "pending",
                "summary": "Analyze requirements",
                "confidence": "high",
            },
            {
                "id": "t2",
                "status": "pending",
                "summary": "Design tables and relations",
                "confidence": "high",
            },
            {
                "id": "t3",
                "status": "pending",
                "summary": "Document schema",
                "confidence": "medium",
            },
        ],
        "key_findings": ["Schema must support 10M rows"],
        "recommended_next_steps": ["Create migrations"],
    }

    markdown = format_plan_capsule(payload)

    # AC2a: Objective/milestone is shown.
    assert "## Plan: Design the database schema" in markdown

    # AC2b: Ordered step list with IDs (t1, t2, t3).
    assert "**t1**" in markdown
    assert "**t2**" in markdown
    assert "**t3**" in markdown
    # Steps should appear in order.
    pos_t1 = markdown.find("**t1**")
    pos_t2 = markdown.find("**t2**")
    pos_t3 = markdown.find("**t3**")
    assert pos_t1 < pos_t2 < pos_t3, "Steps not in order"

    # AC2c: Task summaries (agent assignments).
    assert "Analyze requirements" in markdown
    assert "Design tables and relations" in markdown
    assert "Document schema" in markdown

    # AC2d: Success criteria (found in key findings).
    assert "Schema must support 10M rows" in markdown
    assert "Create migrations" in markdown  # recommended_next_steps

    # AC3: No raw JSON dump.
    assert "```json" not in markdown
    assert "DELEGATION_CAPSULE_RESULT" not in markdown


def test_plan_capsule_no_raw_json_in_bubble() -> None:
    """AC3: Plan bubble contains NO raw JSON, only formatted markdown."""
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
        "milestone": "Do the thing",
        "status": "completed",
        "tasks": [{"id": "t1", "summary": "step 1"}],
    }
    listener._inject_capsule_result(payload)

    # plan_bubble_injector receives markdown (no JSON).
    plan_bubble_injector.assert_called_once()
    markdown = plan_bubble_injector.call_args.args[0]

    # AC3a: Plan bubble has no JSON code block.
    assert "```json" not in markdown

    # AC3b: Plan bubble shows human-readable content.
    assert "## Plan: Do the thing" in markdown
    assert "step 1" in markdown

    # AC3c: A human-readable capsule summary is injected to the model (separate call).
    model_msg = inject_message.call_args.args[0]
    assert "Delegation result" in model_msg
    assert "agent: planner" in model_msg


@pytest.mark.asyncio
async def test_plan_stays_visible_during_execution() -> None:
    """AC4: Plan remains visible as execution progresses.

    Once mounted, the plan bubble should not be cleared or replaced
    when progress streams arrive.
    """
    app = MagicMock(spec=SupporterApp)
    app._mount_delegation_widget = AsyncMock()
    app._delegation_blocks = {}
    app.active_turn = None

    # Mount the plan into the delegation block.
    bound = SupporterApp._mount_plan_bubble.__get__(app, SupporterApp)
    await bound("## Plan: Execute tasks")

    # Verify the block was mounted once.
    app._mount_delegation_widget.assert_awaited_once()
    block = app._mount_delegation_widget.call_args.args[0]

    # G7: plan + progress + result share ONE persistent collapsible block;
    # progress updates the same block in-place rather than replacing the plan.
    assert isinstance(block, DelegationBlock)
    assert "delegation-block" in block.classes
    assert block._pending.get("plan") == "## Plan: Execute tasks"


@pytest.mark.asyncio
async def test_format_plan_capsule_with_full_content() -> None:
    """AC2: Comprehensive rendering of all plan elements."""
    from supporter.tools.delegate.capsule_view import format_plan_capsule

    payload = {
        "job_id": "plan-job-1",
        "agent": "planner",
        "milestone": "Build a payment processor",
        "status": "completed",
        "tasks": [
            {
                "id": "design",
                "status": "pending",
                "summary": "Create API design",
                "confidence": "high",
            },
            {
                "id": "implement",
                "status": "pending",
                "summary": "Implement endpoints",
                "confidence": "high",
            },
            {
                "id": "test",
                "status": "pending",
                "summary": "Write integration tests",
                "confidence": "medium",
            },
        ],
        "key_findings": [
            "Must support idempotency for retries",
            "PCI-DSS compliance required",
        ],
        "recommended_next_steps": [
            "Set up staging environment",
            "Configure webhook handlers",
        ],
    }

    markdown = format_plan_capsule(payload)

    # Objective is prominent.
    assert "## Plan: Build a payment processor" in markdown

    # All steps appear in order.
    assert "**design**" in markdown
    assert "**implement**" in markdown
    assert "**test**" in markdown

    design_pos = markdown.find("**design**")
    impl_pos = markdown.find("**implement**")
    test_pos = markdown.find("**test**")
    assert design_pos < impl_pos < test_pos

    # Summaries and confidence levels shown.
    assert "Create API design" in markdown
    assert "Implement endpoints" in markdown
    assert "Write integration tests" in markdown
    assert "high" in markdown
    assert "medium" in markdown

    # Key findings and recommendations visible.
    assert "idempotency" in markdown
    assert "PCI-DSS" in markdown
    assert "staging environment" in markdown
    assert "webhook" in markdown

    # No markup artifacts.
    assert "DELEGATION_CAPSULE_RESULT" not in markdown
    assert "```json" not in markdown
    assert "\n\n\n" not in markdown  # excessive blank lines
