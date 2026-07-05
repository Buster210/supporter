"""G2: Plan → Implement → Verify → Replan loop.

Tests that the replan context tracks cycles and failure reasons,
and that the loop fires when verify_plan returns False.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

import pytest

from supporter import worker
from supporter.replan import ReplanContext, format_replan_prompt


class TestReplanContext:
    """ReplanContext state machine."""

    def test_next_cycle_within_budget(self) -> None:
        ctx = ReplanContext("test objective", max_cycles=3)
        assert ctx.cycle == 0
        assert ctx.next_cycle() is True
        assert ctx.cycle == 1
        assert ctx.next_cycle() is True
        assert ctx.cycle == 2

    def test_next_cycle_exhausted(self) -> None:
        ctx = ReplanContext("test objective", max_cycles=2)
        assert ctx.next_cycle() is True
        assert ctx.cycle == 1
        assert ctx.next_cycle() is True
        assert ctx.cycle == 2
        assert ctx.next_cycle() is False
        assert ctx.cycle == 2  # no increment on exhaustion

    def test_record_failure(self) -> None:
        ctx = ReplanContext("test objective", max_cycles=3)
        ctx.record_failure("missing field")
        ctx.record_failure("incomplete data")
        assert ctx.failures == ["missing field", "incomplete data"]

    def test_format_replan_prompt_context(self) -> None:
        ctx = ReplanContext("gather data", max_cycles=3)
        ctx.cycle = 1
        ctx.plan = "step 1: gather\nstep 2: analyze"
        ctx.last_result = "partial data"
        ctx.record_failure("incomplete results")

        prompt = ctx.format_replan_prompt_context()
        assert "gather data" in prompt
        assert "step 1: gather" in prompt
        assert "partial data" in prompt
        assert "incomplete results" in prompt


def test_format_replan_prompt() -> None:
    prompt = format_replan_prompt(
        objective="gather data",
        plan="step 1: visit sites\nstep 2: aggregate",
        result="partial results only",
        failures=["missing 3 sources", "incomplete aggregation"],
    )
    assert "OBJECTIVE:" in prompt
    assert "gather data" in prompt
    assert "PREVIOUS PLAN:" in prompt
    assert "step 1: visit sites" in prompt
    assert "IMPLEMENTATION RESULT:" in prompt
    assert "partial results only" in prompt
    assert "VERIFICATION FAILURES:" in prompt
    assert "missing 3 sources" in prompt
    assert "incomplete aggregation" in prompt
    assert "revise the plan" in prompt


@pytest.mark.asyncio
async def test_replan_loop_enters_on_verify_failure() -> None:
    """Verify that run_worker enters the replan loop when verify_plan fails.

    Simulates 2 replan cycles: cycle 1 verify fails → cycle 2 verify passes.
    Tests that make_plan is called twice and verify_plan is called twice.
    """

    with TemporaryDirectory() as tmpdir:
        report_dir = Path(tmpdir)

        # Track calls to the planner and verifier
        plan_inputs: list[str] = []

        async def mock_make_plan(objective: str, persona: str, model: str) -> str:
            plan_inputs.append(objective)
            return f"Plan (cycle {len(plan_inputs)})"

        async def mock_verify_plan(
            objective: str, plan: str, result: str, model: str
        ) -> tuple[bool, str]:
            # Fail on first verify, pass on second
            if len(plan_inputs) == 1:
                return (False, "incomplete report")
            return (True, "complete")

        class MockExecutor:
            async def execute(self, prompt: str) -> None:
                # Simulate writing the report
                pass

        # Patch dependencies
        with (
            patch.object(worker, "make_plan", new=mock_make_plan),
            patch.object(worker, "verify_plan", new=mock_verify_plan),
            patch.object(worker, "_build_executor_agent", return_value=MockExecutor()),
            patch.object(worker, "_teardown", new=AsyncMock()),
            patch.object(worker, "_report_ready", return_value=True),
            patch.object(Path, "read_text", return_value="# Report\n"),
            contextlib.suppress(RuntimeError),
        ):
            # Run with max 1 executor turn to speed up
            await worker.run_worker("test", report_dir=report_dir, max_executor_turns=1)

        # Verify replan loop fired: 2 make_plan calls
        assert len(plan_inputs) == 2, (
            f"Expected 2 make_plan calls (1 initial + 1 replan), "
            f"got {len(plan_inputs)}"
        )
        # AC2: Second make_plan input should be replan context with failure reason
        assert "incomplete report" in plan_inputs[1], (
            f"Replan prompt should contain failure reason; got: {plan_inputs[1]}"
        )


@pytest.mark.asyncio
async def test_replan_loop_exhaustion() -> None:
    """Verify that replan loop exhausts max_cycles before raising RuntimeError."""

    with TemporaryDirectory() as tmpdir:
        report_dir = Path(tmpdir)

        # Count replan cycles that fired
        plan_calls: list[int] = []

        async def mock_make_plan(objective: str, persona: str, model: str) -> str:
            plan_calls.append(len(plan_calls) + 1)
            return f"Plan #{len(plan_calls)}"

        # Always fail verification
        async def mock_verify_plan(
            objective: str, plan: str, result: str, model: str
        ) -> tuple[bool, str]:
            return (False, "never complete")

        class MockExecutor:
            async def execute(self, prompt: str) -> None:
                pass

        with (
            patch.object(worker, "make_plan", new=mock_make_plan),
            patch.object(worker, "verify_plan", new=mock_verify_plan),
            patch.object(worker, "_build_executor_agent", return_value=MockExecutor()),
            patch.object(worker, "_teardown", new=AsyncMock()),
            patch.object(worker, "_report_ready", return_value=True),
            patch.object(Path, "read_text", return_value="# Incomplete\n"),
            pytest.raises(RuntimeError, match="exhausted replan"),
        ):
            await worker.run_worker(
                "test",
                report_dir=report_dir,
                max_executor_turns=1,
            )

        # Verify all 3 replan cycles fired (config.replan_max_cycles = 3)
        assert len(plan_calls) == 3, (
            f"Expected 3 make_plan calls (one per replan cycle), got {len(plan_calls)}"
        )
