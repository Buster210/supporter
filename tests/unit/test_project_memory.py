from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

import supporter.tools.delegate.capsule as capsule_store
import supporter.tools.delegate.project_memory as pm_module
from supporter.config import config
from supporter.tools.delegate.bus import DelegationBus
from supporter.tools.delegate.capsule import create_capsule
from supporter.tools.delegate.project_memory import (
    MAX_INSIGHT_CHARS,
    MAX_INSIGHTS,
    ProjectMemory,
    load_project_memory,
    memory_context_block,
    record_learnings,
)
from supporter.tools.delegate.scheduler import (
    _execute_dag,
    _record_milestone_learnings,
    run_milestone,
)


@pytest.fixture(autouse=True)
def isolate_memory_state(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    monkeypatch.setattr("supporter.pool.get_provider", lambda **_kwargs: object())
    pm_module._MEMORY_LOCK = None
    capsule_store._CAPSULE_LOCKS.clear()
    capsule_store._CAPSULE_CACHE.clear()
    capsule_store._CAPSULE_DIRTY_COUNT.clear()


def _task(task_id: str, *, context: str = "") -> dict[str, Any]:
    return {
        "id": task_id,
        "task": f"run {task_id}",
        "agent": "explorer",
        "tools": {"read_file"},
        "model": "gemini-test",
        "persona": "persona",
        "context": context,
        "timeout": 30,
        "max_retries": 0,
        "depends_on": [],
        "tolerate_failures": False,
    }


# --- record_learnings -------------------------------------------------------


async def test_record_learnings_deduplicates_same_text() -> None:
    """Same text (normalized) submitted twice yields a single entry."""
    await record_learnings(["  Hello World  ", "HELLO WORLD"], "job-1")
    await record_learnings(["hello world"], "job-2")

    memory = await load_project_memory()
    assert len(memory["insights"]) == 1
    assert memory["insights"][0]["text"] == "Hello World"


async def test_record_learnings_truncates_long_insights() -> None:
    """Insights longer than MAX_INSIGHT_CHARS are truncated."""
    await record_learnings(["x" * (MAX_INSIGHT_CHARS + 50)], "job-long")

    memory = await load_project_memory()
    assert len(memory["insights"]) == 1
    assert len(memory["insights"][0]["text"]) == MAX_INSIGHT_CHARS


async def test_record_learnings_caps_at_max_insights() -> None:
    """Total insights are capped at MAX_INSIGHTS, keeping the newest."""
    await record_learnings(
        [f"insight {i}" for i in range(MAX_INSIGHTS + 10)], "job-cap"
    )

    memory = await load_project_memory()
    assert len(memory["insights"]) == MAX_INSIGHTS
    assert memory["insights"][0]["text"] == "insight 0"


async def test_record_learnings_drops_empty_insights() -> None:
    """Empty, whitespace-only, and non-string insights are dropped."""
    insights = ["", "   ", "\t", "valid insight", None, 123]
    await record_learnings(insights, "job-empty")  # type: ignore[arg-type]

    memory = await load_project_memory()
    assert len(memory["insights"]) == 1
    assert memory["insights"][0]["text"] == "valid insight"


async def test_record_learnings_noop_on_empty_list() -> None:
    """An empty insight list never creates the store file."""
    await record_learnings([], "job-none")

    memory = await load_project_memory()
    assert memory["insights"] == []


# --- load_project_memory ----------------------------------------------------


async def test_load_project_memory_returns_empty_on_missing_file() -> None:
    """Missing memory file returns an empty structure."""
    memory = await load_project_memory()
    assert memory["schema_version"] == 1
    assert memory["insights"] == []


async def test_load_project_memory_returns_empty_on_corrupt_file(
    tmp_path: Any,
) -> None:
    """Corrupt JSON returns an empty structure rather than raising."""
    corrupt_path = tmp_path / ".supporter" / "project_memory.json"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text("not valid json{")

    memory = await load_project_memory()
    assert memory["insights"] == []


# --- memory_context_block ---------------------------------------------------


async def test_memory_context_block_returns_empty_when_no_insights() -> None:
    """Empty memory (and None) produces an empty string."""
    empty: ProjectMemory = {
        "schema_version": 1,
        "updated_at": "2024-01-01T00:00:00Z",
        "insights": [],
    }
    assert memory_context_block(empty) == ""
    assert memory_context_block(None) == ""


async def test_memory_context_block_renders_header_and_insights() -> None:
    """Non-empty memory renders the header plus a bullet per insight."""
    memory: ProjectMemory = {
        "schema_version": 1,
        "updated_at": "2024-01-01T00:00:00Z",
        "insights": [
            {"text": "First insight", "source_job": "j1", "created_at": "x"},
            {"text": "Second insight", "source_job": "j2", "created_at": "y"},
        ],
    }
    block = memory_context_block(memory)
    assert "PROJECT MEMORY (learned from prior runs):" in block
    assert "• First insight" in block
    assert "• Second insight" in block


async def test_memory_context_block_is_char_bounded() -> None:
    """The rendered block stays bounded to avoid prompt bloat."""
    insights = [
        {"text": "x" * 100, "source_job": "job", "created_at": "x"} for _ in range(30)
    ]
    memory: ProjectMemory = {
        "schema_version": 1,
        "updated_at": "",
        "insights": insights,  # type: ignore[typeddict-item]
    }
    assert len(memory_context_block(memory)) <= 1500


# --- scheduler injection ----------------------------------------------------


async def test_task_context_receives_memory_block_when_non_empty() -> None:
    """A non-empty store is injected into each delegated task's context."""
    memory: ProjectMemory = {
        "schema_version": 1,
        "updated_at": "2024-01-01T00:00:00Z",
        "insights": [
            {
                "text": "Prior finding about this project",
                "source_job": "job-old",
                "created_at": "",
            }
        ],
    }
    pm_module._save_memory_sync(memory)

    task = _task("t1", context="original context")
    captured: list[str] = []

    async def capture(task_arg: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
        captured.append(task_arg.get("context", ""))
        return {
            "id": task_arg["id"],
            "status": "completed",
            "output": "done",
            "duration": 0.1,
        }

    sem = asyncio.Semaphore(2)
    bus = DelegationBus("memory-test")
    with (
        patch("supporter.tools.delegate.scheduler.run_sub_agent", side_effect=capture),
        patch.object(config, "delegate_result_repair", False),
    ):
        await create_capsule("mem-test", "memory-test", [task], 2)
        await _execute_dag([task], sem, sem, bus, "mem-test")

    assert len(captured) == 1
    assert "original context" in captured[0]
    assert "PROJECT MEMORY" in captured[0]
    assert "Prior finding about this project" in captured[0]


async def test_task_context_unchanged_when_memory_empty() -> None:
    """With no memory file, task context is byte-for-byte unchanged (lossless)."""
    task = _task("t2", context="original context")
    captured: list[str] = []

    async def capture(task_arg: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
        captured.append(task_arg.get("context", ""))
        return {
            "id": task_arg["id"],
            "status": "completed",
            "output": "done",
            "duration": 0.1,
        }

    sem = asyncio.Semaphore(2)
    bus = DelegationBus("empty-memory-test")
    with (
        patch("supporter.tools.delegate.scheduler.run_sub_agent", side_effect=capture),
        patch.object(config, "delegate_result_repair", False),
    ):
        await create_capsule("mem-empty", "empty-memory-test", [task], 2)
        await _execute_dag([task], sem, sem, bus, "mem-empty")

    assert captured == ["original context"]


# --- scheduler extraction (the _record_milestone_learnings helper) ----------


async def test_record_milestone_learnings_extracts_key_findings() -> None:
    """The extraction helper records a completed capsule's key_findings."""
    fake_capsule = {"synthesis": {"key_findings": ["finding one", "finding two"]}}
    with patch(
        "supporter.tools.delegate.scheduler.load_capsule", return_value=fake_capsule
    ):
        await _record_milestone_learnings("job-x")

    memory = await load_project_memory()
    assert {i["text"] for i in memory["insights"]} == {"finding one", "finding two"}
    assert all(i["source_job"] == "job-x" for i in memory["insights"])


async def test_record_milestone_learnings_never_raises() -> None:
    """A failure loading the capsule degrades to no-op, never propagates."""
    with patch(
        "supporter.tools.delegate.scheduler.load_capsule",
        side_effect=RuntimeError("boom"),
    ):
        await _record_milestone_learnings("job-y")

    memory = await load_project_memory()
    assert memory["insights"] == []


async def test_run_milestone_records_key_findings_into_memory() -> None:
    """run_milestone wires extraction: completing it records the capsule findings."""
    task = _task("t-key")

    async def mock_run(task_arg: dict[str, Any], *_a: Any, **_k: Any) -> dict[str, Any]:
        return {
            "id": task_arg["id"],
            "status": "completed",
            "output": "done",
            "duration": 0.1,
        }

    fake_capsule = {
        "synthesis": {"key_findings": ["learned thing A", "learned thing B"]}
    }
    sem = asyncio.Semaphore(2)
    bus = DelegationBus("e2e-test")
    with (
        patch("supporter.tools.delegate.scheduler.run_sub_agent", side_effect=mock_run),
        patch(
            "supporter.tools.delegate.scheduler.load_capsule",
            return_value=fake_capsule,
        ),
        patch.object(config, "delegate_result_repair", False),
    ):
        await create_capsule("e2e-job", "e2e-test", [task], 2)
        await run_milestone("e2e-test", [task], sem, sem, bus, "e2e-job")

    memory = await load_project_memory()
    assert {i["text"] for i in memory["insights"]} == {
        "learned thing A",
        "learned thing B",
    }
