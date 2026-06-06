import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.config import config
from supporter.tools.delegate import qa_gate
from supporter.tools.delegate.bus import DelegationBus
from supporter.tools.delegate.qa_gate import _verdict_passed, run_qa_gate
from supporter.types import TaskStatus


def _base_task() -> dict[str, Any]:
    return {
        "id": "t1",
        "task": "implement feature X",
        "backend": "opencode",
        "context": "",
        "timeout": 60,
    }


def _result() -> dict[str, Any]:
    return {
        "id": "t1",
        "status": TaskStatus.COMPLETED,
        "output": "wrote code",
        "model": "google/x",
        "duration": 1.0,
        "tokens": {},
    }


def _run_gate(outputs_for: Any) -> tuple[dict[str, Any], list[str]]:
    """Run the gate with run_sub_agent mocked by a per-task-id output function.

    Forces the LLM tier-1 path (``DELEGATE_TIER1_OBJECTIVE=0``) so the
    existing orchestration assertions on tier-1/tier-2/correction call
    counts keep exercising the unchanged loop body, independent of the new
    objective tier-1 dispatch.
    """
    calls: list[str] = []

    async def fake_run(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
        calls.append(task["id"])
        return {"status": TaskStatus.COMPLETED, "output": outputs_for(task["id"])}

    bus = MagicMock(spec=DelegationBus)
    with (
        patch.object(qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)),
        patch.dict(os.environ, {"DELEGATE_TIER1_OBJECTIVE": "0"}),
    ):
        result = asyncio.run(
            run_qa_gate(_base_task(), _result(), asyncio.Semaphore(3), bus, "job1")
        )
    return result, calls


class TestVerdictParsing:
    def test_tier1_pass(self) -> None:
        assert _verdict_passed("ran tests\nQA-TIER1: PASS", "qa-tier1:", "pass")

    def test_tier1_fail(self) -> None:
        assert not _verdict_passed("QA-TIER1: FAIL", "qa-tier1:", "pass")

    def test_missing_marker_is_not_passed(self) -> None:
        assert not _verdict_passed("looks fine to me", "qa-verdict:", "approve")

    def test_tier2_approve(self) -> None:
        assert _verdict_passed("done\nQA-VERDICT: APPROVE", "qa-verdict:", "approve")


class TestQaGate:
    def test_skips_non_opencode_task(self) -> None:
        task = _base_task()
        task["backend"] = "gemini"
        bus = MagicMock(spec=DelegationBus)
        with patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as mock:
            result = asyncio.run(
                run_qa_gate(task, _result(), asyncio.Semaphore(1), bus, "job1")
            )
        mock.assert_not_awaited()
        assert result["status"] == TaskStatus.COMPLETED
        assert "QA gate" not in result["output"]

    def test_approves_on_clean_pass(self) -> None:
        def outputs(task_id: str) -> str:
            if "__tier1_" in task_id:
                return "all green\nQA-TIER1: PASS"
            return "looks good\nQA-VERDICT: APPROVE"

        result, calls = _run_gate(outputs)
        assert result["status"] == TaskStatus.COMPLETED
        assert "tier-1 + tier-2 PASSED" in result["output"]
        assert sum(1 for c in calls if "__tier1_" in c) == 1
        assert sum(1 for c in calls if "__tier2_" in c) == 3
        assert not any("__fix_" in c for c in calls)

    def test_corrects_then_passes(self) -> None:
        state = {"tier1_seen": 0}

        def outputs(task_id: str) -> str:
            if "__tier1_" in task_id:
                state["tier1_seen"] += 1
                return (
                    "QA-TIER1: FAIL" if state["tier1_seen"] == 1 else "QA-TIER1: PASS"
                )
            if "__fix_" in task_id:
                return "fixed it"
            return "QA-VERDICT: APPROVE"

        result, calls = _run_gate(outputs)
        assert result["status"] == TaskStatus.COMPLETED
        assert sum(1 for c in calls if "__fix_" in c) == 1

    def test_rejects_after_rounds(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(config, "delegate_correction_rounds", 1)

        def outputs(task_id: str) -> str:
            if "__tier1_" in task_id:
                return "QA-TIER1: PASS"
            if "__fix_" in task_id:
                return "tried"
            return "QA-VERDICT: REJECT broken logic"

        result, calls = _run_gate(outputs)
        assert result["status"] == TaskStatus.ERROR
        assert "QA gate rejected" in result["output"]
        assert sum(1 for c in calls if "__fix_" in c) == 1

    def test_aborts_when_correction_worker_fails(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(config, "delegate_correction_rounds", 3)
        calls: list[str] = []

        async def fake_run(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            calls.append(task["id"])
            if "__fix_" in task["id"]:
                return {"status": TaskStatus.ERROR, "output": "opencode crashed"}
            if "__tier1_" in task["id"]:
                return {"status": TaskStatus.COMPLETED, "output": "QA-TIER1: FAIL"}
            return {"status": TaskStatus.COMPLETED, "output": "QA-VERDICT: APPROVE"}

        bus = MagicMock(spec=DelegationBus)
        with (
            patch.object(qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)),
            patch.dict(os.environ, {"DELEGATE_TIER1_OBJECTIVE": "0"}),
        ):
            result = asyncio.run(
                run_qa_gate(_base_task(), _result(), asyncio.Semaphore(3), bus, "job1")
            )
        assert result["status"] == TaskStatus.ERROR
        assert "did not complete" in result["output"]
        assert sum(1 for c in calls if "__fix_" in c) == 1

    def test_disabled_gate_is_noop(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(config, "delegate_qa_gate_enabled", False)
        bus = MagicMock(spec=DelegationBus)
        with patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as mock:
            result = asyncio.run(
                run_qa_gate(_base_task(), _result(), asyncio.Semaphore(1), bus, "job1")
            )
        mock.assert_not_awaited()
        assert result["status"] == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_make_task_resolves_roster_role() -> None:
    base = _base_task()
    task = qa_gate._make_task(
        base, "tier2_test_engineer", "verify", backend="gemini", agent="test_engineer"
    )
    assert task["id"] == "t1__tier2_test_engineer"
    assert task["agent"] == "test_engineer"
    assert task["backend"] == "gemini"
    assert task["persona"] == config.delegate_agent_roster["test_engineer"]["persona"]
