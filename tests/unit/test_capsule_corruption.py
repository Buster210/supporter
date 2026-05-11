from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from supporter.config import config
from supporter.tools.capsule import (
    capsule_path,
    create_capsule,
    load_capsule,
)
from supporter.tools.capsule_query import query_delegation, serialize_capsule_result


@pytest.fixture(autouse=True)
def isolate_capsules(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])


def _write_capsule(job_id: str, content: str) -> Path:
    path = capsule_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_load_capsule_raises_for_missing_capsule() -> None:
    with pytest.raises(FileNotFoundError):
        load_capsule("missing1")


def test_load_capsule_raises_for_truncated_json() -> None:
    _write_capsule("badjson1", '{"job_id": "badjson1",')

    with pytest.raises(json.JSONDecodeError):
        load_capsule("badjson1")


def test_load_capsule_raises_for_invalid_json_shape() -> None:
    _write_capsule("shape1", '["not", "an", "object"]')

    with pytest.raises(ValueError, match="not a JSON object"):
        load_capsule("shape1")


def test_serialize_capsule_result_returns_compact_missing_error() -> None:
    payload = serialize_capsule_result("missing2")

    assert payload["job_id"] == "missing2"
    assert payload["status"] == "unavailable"
    assert payload["tasks"] == []
    assert payload["totals"] == {
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "timed_out": 0,
        "tokens": 0,
    }
    assert payload["error"]["type"] == "FileNotFoundError"
    assert "Start a new delegation" in payload["error"]["action"]


def test_serialize_capsule_result_returns_compact_corrupt_error() -> None:
    _write_capsule("badjson2", '{"job_id": "badjson2"')

    payload = serialize_capsule_result("badjson2")

    assert payload["job_id"] == "badjson2"
    assert payload["status"] == "unavailable"
    assert payload["error"]["type"] == "JSONDecodeError"


def test_query_delegation_returns_actionable_corrupt_message() -> None:
    _write_capsule("badquery1", '{"job_id": "badquery1"')

    message = query_delegation(job_id="badquery1")

    assert "badquery1" in message
    assert "capsule is unavailable" in message
    assert "JSONDecodeError" in message
    assert "Start a new delegation" in message


def test_query_delegation_task_path_returns_actionable_corrupt_message() -> None:
    _write_capsule("badtask1", "42")

    message = query_delegation(job_id="badtask1", task_id="task1")

    assert "badtask1" in message
    assert "capsule is unavailable" in message
    assert "ValueError" in message


@pytest.mark.asyncio
async def test_query_delegation_list_skips_corrupt_capsules() -> None:
    await create_capsule(
        "good1",
        "Good capsule",
        [
            {
                "id": "t1",
                "task": "Do t1",
                "depends_on": [],
                "timeout": 30,
                "model": "gemini-test",
                "context": "",
            }
        ],
        1,
    )
    _write_capsule("badlist1", "{")

    message = query_delegation()

    assert "good1" in message
    assert "badlist1" not in message


@pytest.mark.asyncio
async def test_valid_capsule_serialization_still_works() -> None:
    await create_capsule(
        "valid1",
        "Valid capsule",
        [
            {
                "id": "t1",
                "task": "Do t1",
                "depends_on": [],
                "timeout": 30,
                "model": "gemini-test",
                "context": "",
            }
        ],
        1,
    )

    payload = serialize_capsule_result("valid1")

    assert payload["job_id"] == "valid1"
    assert payload["milestone"] == "Valid capsule"
    assert json.dumps(payload)
