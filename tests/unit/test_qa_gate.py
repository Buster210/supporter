import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.config import config
from supporter.tools.delegate import qa_gate
from supporter.tools.delegate.bus import DelegationBus
from supporter.tools.delegate.qa_gate import (
    _gemini_predicate_passed,
    _verdict_passed,
    run_qa_gate,
)
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


class TestGeminiPredicate:
    def test_valid_payload_high_confidence_passes(self) -> None:
        output = (
            "```json\n"
            "{\n"
            '  "summary": "Found it",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": ["url"]},\n'
            '  "findings": ["fact"],\n'
            '  "handoff": "",\n'
            '  "confidence": "high"\n'
            "}\n"
            "```"
        )
        assert _gemini_predicate_passed(output, "explorer")

    def test_valid_payload_medium_confidence_passes(self) -> None:
        output = (
            "```json\n"
            "{\n"
            '  "summary": "Found it",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": []},\n'
            '  "findings": ["fact"],\n'
            '  "handoff": "",\n'
            '  "confidence": "medium"\n'
            "}\n"
            "```"
        )
        assert _gemini_predicate_passed(output, "explorer")

    def test_invalid_payload_fails(self) -> None:
        assert not _gemini_predicate_passed("no json here", "explorer")

    def test_low_confidence_fails_when_min_is_medium(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(config, "delegate_min_confidence", "medium")
        output = (
            "```json\n"
            "{\n"
            '  "summary": "Found it",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": []},\n'
            '  "findings": [],\n'
            '  "handoff": "",\n'
            '  "confidence": "low"\n'
            "}\n"
            "```"
        )
        assert not _gemini_predicate_passed(output, "explorer")

    def test_finding_role_needs_findings_or_sources(self) -> None:
        # Explorer with no findings and no sources fails
        output = (
            "```json\n"
            "{\n"
            '  "summary": "Explored",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": []},\n'
            '  "findings": [],\n'
            '  "handoff": "",\n'
            '  "confidence": "high"\n'
            "}\n"
            "```"
        )
        assert not _gemini_predicate_passed(output, "explorer")

        # Explorer with sources passes
        output_with_sources = (
            "```json\n"
            "{\n"
            '  "summary": "Explored",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": ["https://x.com"]},\n'
            '  "findings": [],\n'
            '  "handoff": "",\n'
            '  "confidence": "high"\n'
            "}\n"
            "```"
        )
        assert _gemini_predicate_passed(output_with_sources, "explorer")

        # Explorer with findings passes
        output_with_findings = (
            "```json\n"
            "{\n"
            '  "summary": "Explored",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": []},\n'
            '  "findings": ["found it"],\n'
            '  "handoff": "",\n'
            '  "confidence": "high"\n'
            "}\n"
            "```"
        )
        assert _gemini_predicate_passed(output_with_findings, "explorer")

    def test_non_finding_role_passes_with_empty_findings(self) -> None:
        # Non-finding role (like 'test_engineer' - not in _FINDING_ROLES) should pass
        # when payload is valid and confidence is sufficient, even with empty findings
        output = (
            "```json\n"
            "{\n"
            '  "summary": "Tests written",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": []},\n'
            '  "findings": [],\n'
            '  "handoff": "",\n'
            '  "confidence": "medium"\n'
            "}\n"
            "```"
        )
        # test_engineer is in _TIER2_ROLES but NOT in _FINDING_ROLES
        assert _gemini_predicate_passed(output, "test_engineer")


class TestQaGate:
    def test_gemini_runs_predicate_path(self) -> None:
        """Gemini task runs the predicate check; good output passes immediately."""
        task = _base_task()
        task["backend"] = "gemini"
        task["agent"] = "explorer"
        bus = MagicMock(spec=DelegationBus)
        good_output = (
            "found what you need\n"
            "```json\n"
            "{\n"
            '  "summary": "Found the answer",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": ["https://example.com"]},\n'
            '  "findings": [],\n'
            '  "handoff": "",\n'
            '  "confidence": "high"\n'
            "}\n"
            "```"
        )
        with patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as mock:
            result = asyncio.run(
                run_qa_gate(
                    task,
                    {**_result(), "output": good_output},
                    asyncio.Semaphore(1),
                    bus,
                    "job1",
                )
            )
        mock.assert_not_awaited()
        assert result["status"] == TaskStatus.COMPLETED
        assert "gemini predicate PASSED" in result["output"]

    def test_gemini_high_confidence_passes_no_rerun(self) -> None:
        """High-confidence well-formed output passes without correction."""
        task = _base_task()
        task["backend"] = "gemini"
        task["agent"] = "explorer"
        bus = MagicMock(spec=DelegationBus)
        good_output = (
            "```json\n"
            "{\n"
            '  "summary": "Explored and found the module",\n'
            '  "evidence": {"files_read": ["src/main.py"],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": []},\n'
            '  "findings": ["Module implements X"],\n'
            '  "handoff": "",\n'
            '  "confidence": "high"\n'
            "}\n"
            "```"
        )
        with patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as mock:
            result = asyncio.run(
                run_qa_gate(
                    task,
                    {**_result(), "output": good_output},
                    asyncio.Semaphore(1),
                    bus,
                    "job1",
                )
            )
        mock.assert_not_awaited()
        assert result["status"] == TaskStatus.COMPLETED

    def test_gemini_unknown_confidence_triggers_correction(
        self, monkeypatch: Any
    ) -> None:
        """Unknown/low confidence triggers correction round, then escalates to ERROR."""
        monkeypatch.setattr(config, "delegate_correction_rounds", 1)
        task = _base_task()
        task["backend"] = "gemini"
        task["agent"] = "explorer"
        calls: list[str] = []

        async def fake_run(
            task_arg: dict[str, Any], *_a: Any, **_k: Any
        ) -> dict[str, Any]:
            calls.append(task_arg["id"])
            if "fix_0" in task_arg["id"]:
                return {
                    "status": TaskStatus.COMPLETED,
                    "output": "still unknown output",
                }
            # Original task - return unknown confidence
            return {"status": TaskStatus.COMPLETED, "output": "no json at all"}

        bus = MagicMock(spec=DelegationBus)
        with patch.object(
            qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)
        ):
            result = asyncio.run(
                run_qa_gate(task, _result(), asyncio.Semaphore(1), bus, "job1")
            )
        assert result["status"] == TaskStatus.ERROR
        assert "QA gate rejected" in result["output"]
        assert sum(1 for c in calls if "fix_" in c) == 1  # One correction round

    def test_gemini_low_confidence_finding_role_fails(self, monkeypatch: Any) -> None:
        """Low confidence + finding role with empty findings + no sources
        fails predicate."""
        monkeypatch.setattr(config, "delegate_min_confidence", "medium")
        monkeypatch.setattr(config, "delegate_correction_rounds", 1)
        task = _base_task()
        task["backend"] = "gemini"
        task["agent"] = "explorer"
        calls: list[str] = []

        async def fake_run(
            task_arg: dict[str, Any], *_a: Any, **_k: Any
        ) -> dict[str, Any]:
            calls.append(task_arg["id"])
            # Low confidence with no findings or sources
            return {
                "status": TaskStatus.COMPLETED,
                "output": '```json\n{"summary": "Explored",'
                ' "evidence": {"files_read": [],'
                ' "files_changed": [],'
                ' "commands_run": [],'
                ' "sources": []},'
                ' "findings": [], "handoff": "",'
                ' "confidence": "low"}\n```',
            }

        bus = MagicMock(spec=DelegationBus)
        with patch.object(
            qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)
        ):
            result = asyncio.run(
                run_qa_gate(task, _result(), asyncio.Semaphore(1), bus, "job1")
            )
        assert result["status"] == TaskStatus.ERROR
        assert sum(1 for c in calls if "fix_" in c) == 1

    def test_gemini_non_finding_role_passes_with_empty_findings(self) -> None:
        """Non-finding roles (like test_engineer) pass with valid payload
        and empty findings."""
        task = _base_task()
        task["backend"] = "gemini"
        task["agent"] = "test_engineer"  # NOT in _FINDING_ROLES
        bus = MagicMock(spec=DelegationBus)
        good_output = (
            "```json\n"
            "{\n"
            '  "summary": "Tests written",\n'
            '  "evidence": {"files_read": [],'
            ' "files_changed": [], "commands_run": [],'
            ' "sources": []},\n'
            '  "findings": [],\n'
            '  "handoff": "",\n'
            '  "confidence": "medium"\n'
            "}\n"
            "```"
        )
        with patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as mock:
            result = asyncio.run(
                run_qa_gate(
                    task,
                    {**_result(), "output": good_output},
                    asyncio.Semaphore(1),
                    bus,
                    "job1",
                )
            )
        mock.assert_not_awaited()
        assert result["status"] == TaskStatus.COMPLETED
        assert "gemini predicate PASSED" in result["output"]

    def test_gemini_persist_noncode_disabled_skips(self) -> None:
        """When delegate_persist_noncode is False, gemini tasks are skipped."""
        task = _base_task()
        task["backend"] = "gemini"
        task["agent"] = "explorer"
        bus = MagicMock(spec=DelegationBus)
        with (
            patch.object(qa_gate, "run_sub_agent", new=AsyncMock()) as mock,
            patch.object(config, "delegate_persist_noncode", False),
        ):
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

    def test_gemini_corrects_then_passes(self, monkeypatch: Any) -> None:
        """Gemini correction converges: initial fails, correction passes."""
        monkeypatch.setattr(config, "delegate_correction_rounds", 3)

        async def fake_run(
            task_arg: dict[str, Any], *_a: Any, **_k: Any
        ) -> dict[str, Any]:
            if "t1__fix_0" in task_arg["id"]:
                # Correction returns high-confidence valid payload
                return {
                    "status": TaskStatus.COMPLETED,
                    "output": (
                        "```json\n"
                        "{\n"
                        '  "summary": "Found the answer after correction",\n'
                        '  "evidence": {"files_read": [], "files_changed": [], '
                        '"commands_run": [], "sources": ["https://example.com"]},\n'
                        '  "findings": ["found it"],\n'
                        '  "handoff": "",\n'
                        '  "confidence": "high"\n'
                        "}\n"
                        "```"
                    ),
                }
            # Initial output has no findings/sources and low confidence
            return {
                "status": TaskStatus.COMPLETED,
                "output": (
                    '```json\n{"summary": "Explored", '
                    '"evidence": {"files_read": [], "files_changed": [], '
                    '"commands_run": [], "sources": []}, '
                    '"findings": [], "handoff": "", "confidence": "low"}\n```'
                ),
            }

        bus = MagicMock(spec=DelegationBus)
        task = {**_base_task(), "backend": "gemini", "agent": "explorer"}
        with patch.object(
            qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)
        ):
            result = asyncio.run(
                run_qa_gate(task, _result(), asyncio.Semaphore(1), bus, "job1")
            )
        assert result["status"] == TaskStatus.COMPLETED
        assert "gemini predicate PASSED" in result["output"]
        # Output should be the correction output + QA note
        assert "Found the answer after correction" in result["output"]

    def test_gemini_correction_worker_fails(self, monkeypatch: Any) -> None:
        """Gemini correction returns non-COMPLETED -> status ERROR."""
        monkeypatch.setattr(config, "delegate_correction_rounds", 3)
        calls: list[str] = []

        async def fake_run(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            calls.append(task["id"])
            if "t1__fix_0" in task["id"]:
                return {"status": TaskStatus.ERROR, "output": "gemini crashed"}
            # Initial output has no findings/sources and low confidence
            return {
                "status": TaskStatus.COMPLETED,
                "output": (
                    '```json\n{"summary": "Explored", '
                    '"evidence": {"files_read": [], "files_changed": [], '
                    '"commands_run": [], "sources": []}, '
                    '"findings": [], "handoff": "", "confidence": "low"}\n```'
                ),
            }

        bus = MagicMock(spec=DelegationBus)
        task = {**_base_task(), "backend": "gemini", "agent": "explorer"}
        with patch.object(
            qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)
        ):
            result = asyncio.run(
                run_qa_gate(task, _result(), asyncio.Semaphore(1), bus, "job1")
            )
        assert result["status"] == TaskStatus.ERROR
        assert "did not complete" in result["output"]
        assert sum(1 for c in calls if "__fix_" in c) == 1


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
