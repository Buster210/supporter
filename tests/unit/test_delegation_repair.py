import asyncio
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from supporter.config import AppConfig, config, load_config
from supporter.tools.delegate import scheduler
from supporter.tools.delegate.bus import DelegationBus
from supporter.tools.delegate.capsule import validate_delegation_payload
from supporter.types import TaskStatus


def _block(payload: dict[str, Any]) -> str:
    return "work done\n\n```json\n" + json.dumps(payload) + "\n```"


_FULL: dict[str, Any] = {
    "summary": "did the thing",
    "evidence": {
        "files_read": ["a.py"],
        "files_changed": [],
        "commands_run": ["pytest"],
        "sources": [],
    },
    "findings": ["a finding"],
    "handoff": "next agent context",
    "confidence": "high",
}
VALID = _block(_FULL)
INVALID = "I finished the work but forgot the structured block."


class TestValidateDelegationPayload:
    def test_valid_full_block(self) -> None:
        assert validate_delegation_payload(VALID) is True

    def test_no_block(self) -> None:
        assert validate_delegation_payload(INVALID) is False

    def test_missing_required_key(self) -> None:
        payload = {k: v for k, v in _FULL.items() if k != "handoff"}
        assert validate_delegation_payload(_block(payload)) is False

    def test_findings_wrong_type(self) -> None:
        payload = {**_FULL, "findings": "not a list"}
        assert validate_delegation_payload(_block(payload)) is False

    def test_evidence_value_wrong_type(self) -> None:
        payload = {**_FULL, "evidence": {**_FULL["evidence"], "files_read": "nope"}}
        assert validate_delegation_payload(_block(payload)) is False

    def test_bad_confidence(self) -> None:
        payload = {**_FULL, "confidence": "great"}
        assert validate_delegation_payload(_block(payload)) is False

    def test_sparse_but_valid(self) -> None:
        payload = {
            "summary": "nothing notable",
            "evidence": {
                "files_read": [],
                "files_changed": [],
                "commands_run": [],
                "sources": [],
            },
            "findings": [],
            "handoff": "",
            "confidence": "low",
        }
        assert validate_delegation_payload(_block(payload)) is True


def _task() -> dict[str, Any]:
    return {
        "id": "t1",
        "task": "do x",
        "agent": "explorer",
        "backend": "gemini",
        "persona": "p",
        "tools": set(),
        "model": "gemini-test",
        "live": False,
        "context": "",
        "timeout": 60,
        "result_contract": True,
    }


def _result(output: str, status: Any = TaskStatus.COMPLETED) -> dict[str, Any]:
    return {"id": "t1", "status": status, "output": output, "duration": 1.0}


def _run(
    task: dict[str, Any], result: dict[str, Any], fake: Any
) -> tuple[dict[str, Any], AsyncMock]:
    bus = MagicMock(spec=DelegationBus)
    mock = AsyncMock(side_effect=fake)
    with patch.object(scheduler, "run_sub_agent", mock):
        out, _was_repaired = asyncio.run(
            scheduler._repair_or_rerequest(task, result, asyncio.Semaphore(1), bus, "j")
        )
    return out, mock  # type: ignore[return-value]


class TestRepairOrRerequest:
    def test_valid_rerequest_replaces_output(self) -> None:
        captured: dict[str, Any] = {}

        async def fake(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            captured["task"] = task
            return _result(VALID)

        out, mock = _run(_task(), _result(INVALID), fake)
        mock.assert_awaited_once()
        assert captured["task"]["id"].endswith("__repair")
        assert captured["task"]["max_retries"] == 0
        assert captured["task"]["result_contract"] is True
        assert out["output"] == VALID
        assert out["status"] == TaskStatus.COMPLETED

    def test_invalid_rerequest_keeps_original(self) -> None:
        async def fake(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            return _result("still no block")

        out, mock = _run(_task(), _result(INVALID), fake)
        mock.assert_awaited_once()
        assert out["output"] == INVALID
        assert out["status"] == TaskStatus.COMPLETED

    def test_error_rerequest_keeps_original(self) -> None:
        async def fake(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            return _result(VALID, TaskStatus.ERROR)

        out, _ = _run(_task(), _result(INVALID), fake)
        assert out["output"] == INVALID
        assert out["status"] == TaskStatus.COMPLETED

    def test_valid_original_skips_rerequest(self) -> None:
        async def fake(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            raise AssertionError("should not be called")

        out, mock = _run(_task(), _result(VALID), fake)
        mock.assert_not_awaited()
        assert out["output"] == VALID

    def test_no_result_contract_skips(self) -> None:
        task = _task()
        task["result_contract"] = False
        out, mock = _run(task, _result(INVALID), AsyncMock())
        mock.assert_not_awaited()
        assert out["output"] == INVALID

    def test_non_completed_status_skips(self) -> None:
        out, mock = _run(_task(), _result(INVALID, TaskStatus.ERROR), AsyncMock())
        mock.assert_not_awaited()
        assert out["output"] == INVALID

    def test_disabled_flag_skips(self) -> None:
        with patch.object(config, "delegate_result_repair", False):
            out, mock = _run(_task(), _result(INVALID), AsyncMock())
        mock.assert_not_awaited()
        assert out["output"] == INVALID

    def test_rerequest_exception_keeps_original(self) -> None:
        async def fake(task: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
            raise RuntimeError("boom")

        out, _ = _run(_task(), _result(INVALID), fake)
        assert out["output"] == INVALID
        assert out["status"] == TaskStatus.COMPLETED


def test_config_result_repair_defaults_true() -> None:
    assert AppConfig.delegate_result_repair is True


def test_load_config_result_repair_env() -> None:
    old_env = os.environ.copy()
    os.environ.clear()
    os.environ["GEMINI_API_KEY"] = "test-key"  # pragma: allowlist secret
    try:
        with patch("supporter.config.load_dotenv"):
            assert load_config().delegate_result_repair is True
            os.environ["DELEGATE_RESULT_REPAIR"] = "0"
            assert load_config().delegate_result_repair is False
    finally:
        os.environ.clear()
        os.environ.update(old_env)
