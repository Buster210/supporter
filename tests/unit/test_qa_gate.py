import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.config import config
from supporter.tools.delegate import qa_gate
from supporter.tools.delegate.bus import DelegationBus
from supporter.tools.delegate.qa_gate import (
    _gemini_predicate_passed,
    _verdict_passed,
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


class TestVerifyOnly:
    """Tests for run_qa_gate_verify_only — the single-shot verify path."""

    def _task(self, backend: str = "opencode", **extra: Any) -> dict[str, Any]:
        task = _base_task()
        task["backend"] = backend
        task.update(extra)
        return task

    def _result(self, status: TaskStatus = TaskStatus.COMPLETED) -> dict[str, Any]:
        r = _result()
        r["status"] = status
        return r

    def test_non_completed_returns_failed(self) -> None:
        from supporter.tools.delegate.qa_gate import run_qa_gate_verify_only

        bus = MagicMock(spec=DelegationBus)
        result = asyncio.run(
            run_qa_gate_verify_only(
                self._task(),
                self._result(TaskStatus.ERROR),
                asyncio.Semaphore(3),
                bus,
                "job1",
            )
        )
        assert not result.passed
        assert "task status" in result.reason

    def test_no_backend_returns_passed(self) -> None:
        from supporter.tools.delegate.qa_gate import run_qa_gate_verify_only

        bus = MagicMock(spec=DelegationBus)
        task = self._task()
        task["backend"] = None
        result = asyncio.run(
            run_qa_gate_verify_only(
                task,
                self._result(),
                asyncio.Semaphore(3),
                bus,
                "job1",
            )
        )
        assert result.passed

    def test_opencode_tier1_and_tier2_pass(self) -> None:
        from supporter.tools.delegate.qa_gate import run_qa_gate_verify_only

        verdict = "ran tests\nQA-TIER1: PASS\nQA-VERDICT: APPROVE"

        async def fake_run(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            return {"status": TaskStatus.COMPLETED, "output": verdict}

        bus = MagicMock(spec=DelegationBus)
        result_obj = self._result()
        with (
            patch.object(qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)),
            patch.object(qa_gate, "resolve_tier1_commands", lambda _repo: []),
        ):
            result = asyncio.run(
                run_qa_gate_verify_only(
                    self._task(),
                    result_obj,
                    asyncio.Semaphore(3),
                    bus,
                    "job1",
                )
            )
        assert result.passed
        assert result.marker == "[QA gate: tier-1 + tier-2 PASSED]"

    def test_opencode_tier1_fails(self) -> None:
        from supporter.tools.delegate.qa_gate import run_qa_gate_verify_only

        verdict = "ran tests\nQA-TIER1: FAIL"

        async def fake_run(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            return {"status": TaskStatus.COMPLETED, "output": verdict}

        bus = MagicMock(spec=DelegationBus)
        with (
            patch.object(qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)),
            patch.object(qa_gate, "resolve_tier1_commands", lambda _repo: []),
        ):
            result = asyncio.run(
                run_qa_gate_verify_only(
                    self._task(),
                    self._result(),
                    asyncio.Semaphore(3),
                    bus,
                    "job1",
                )
            )
        assert not result.passed
        assert "tier-1" in result.reason

    def test_opencode_tier2_rejects(self) -> None:
        from supporter.tools.delegate.qa_gate import run_qa_gate_verify_only

        verdict_t1 = "ran tests\nQA-TIER1: PASS"
        verdict_t2 = "review\nQA-VERDICT: REJECT"
        call_count = 0

        async def fake_run(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            out = verdict_t1 if call_count <= 1 else verdict_t2
            return {"status": TaskStatus.COMPLETED, "output": out}

        bus = MagicMock(spec=DelegationBus)
        with (
            patch.object(qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)),
            patch.object(qa_gate, "resolve_tier1_commands", lambda _repo: []),
        ):
            result = asyncio.run(
                run_qa_gate_verify_only(
                    self._task(),
                    self._result(),
                    asyncio.Semaphore(3),
                    bus,
                    "job1",
                )
            )
        assert not result.passed
        assert "tier-2" in result.reason

    def test_gemini_predicate_passes(self) -> None:
        from supporter.tools.delegate.qa_gate import run_qa_gate_verify_only

        bus = MagicMock(spec=DelegationBus)
        result_obj = self._result()
        result_obj["output"] = (
            "```json\n"
            '{"summary": "reviewed code", "findings": ["looks good"],'
            ' "confidence": "high",'
            ' "evidence": {"files_read": [], "files_changed": [],'
            ' "commands_run": [], "sources": []},'
            ' "handoff": ""}\n'
            "```"
        )
        task = self._task(backend="gemini", agent="code_reviewer")
        with patch.object(config, "delegate_persist_noncode", True):
            result = asyncio.run(
                run_qa_gate_verify_only(
                    task,
                    result_obj,
                    asyncio.Semaphore(3),
                    bus,
                    "job1",
                )
            )
        assert result.passed
        assert result.marker == "[QA gate: gemini predicate PASSED]"

    def test_gemini_predicate_fails(self) -> None:
        from supporter.tools.delegate.qa_gate import run_qa_gate_verify_only

        bus = MagicMock(spec=DelegationBus)
        result_obj = self._result()
        result_obj["output"] = (
            "```json\n"
            '{"summary": "reviewed code", "findings": [],'
            ' "confidence": "high",'
            ' "evidence": {"files_read": [], "files_changed": [],'
            ' "commands_run": [], "sources": []},'
            ' "handoff": ""}\n'
            "```"
        )
        task = self._task(backend="gemini", agent="code_reviewer")
        with patch.object(config, "delegate_persist_noncode", True):
            result = asyncio.run(
                run_qa_gate_verify_only(
                    task,
                    result_obj,
                    asyncio.Semaphore(3),
                    bus,
                    "job1",
                )
            )
        assert not result.passed
        assert "gemini" in result.reason

    def test_non_coding_role_routes_to_output_verify_not_git_diff(self) -> None:
        """A page-pilot (browser) subtask has no git diff -- it must be judged
        on its own output, never via the opencode tier-1/tier-2 git-diff path
        nor the finding-role JSON predicate (page-pilot isn't a finding role).
        """
        from supporter.tools.delegate.qa_gate import run_qa_gate_verify_only

        bus = MagicMock(spec=DelegationBus)
        result_obj = self._result()
        result_obj["output"] = "Logged into the site and downloaded the report."
        task = self._task(backend="gemini", agent="page-pilot")

        seen_instructions: list[str] = []

        async def fake_run(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            seen_instructions.append(task["task"])
            return {"status": TaskStatus.COMPLETED, "output": "QA-VERDICT: APPROVE"}

        with (
            patch.object(qa_gate, "run_sub_agent", new=AsyncMock(side_effect=fake_run)),
            patch.object(config, "delegate_persist_noncode", True),
        ):
            result = asyncio.run(
                run_qa_gate_verify_only(
                    task,
                    result_obj,
                    asyncio.Semaphore(3),
                    bus,
                    "job1",
                )
            )

        assert result.passed
        assert result.marker == "[QA gate: output verification PASSED]"
        assert "git diff --name-only" not in seen_instructions[0]
        assert "downloaded the report" in seen_instructions[0]
