import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

import supporter.tools.delegate.capsule as dc
import supporter.tools.delegate.capsule_query as capsule_query
from supporter.config import config
from supporter.tools.delegate.api import delegate_tasks
from supporter.tools.delegate.bus import get_bus
from supporter.tools.delegate.capsule import (
    capsule_path,
    create_capsule,
    effective_status,
    extract_task_capsule_fields,
    load_capsule,
    mark_capsule_cancelled,
    mark_capsule_completed,
    mark_task_completed,
    mark_task_failed,
    mark_task_skipped,
    mark_task_started,
    mark_task_timed_out,
    save_capsule,
)
from supporter.tools.delegate.capsule_query import (
    query_delegation,
    serialize_capsule_result,
)
from supporter.types import MilestoneCompleted, TaskStatus


@pytest.fixture(autouse=True)
def isolate_capsules(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    dc._CAPSULE_LOCKS.clear()
    dc._CAPSULE_CACHE.clear()
    dc._CAPSULE_DIRTY_COUNT.clear()


def _task(task_id: str, depends_on: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": task_id,
        "task": f"Do {task_id}",
        "agent": "explorer",
        "depends_on": depends_on or [],
        "tolerate_failures": False,
        "timeout": 30,
        "model": "gemini-test",
        "context": "",
    }


def test_extract_task_capsule_fields_valid_block() -> None:
    output = """
Work completed.

DELEGATION_RESULT:
{
  "summary": "Mapped the module",
  "evidence": {
    "files_read": ["src/app.py"],
    "files_changed": [],
    "commands_run": ["pytest"],
    "sources": []
  },
  "findings": ["Found entry point"],
  "handoff": "Use the entry point next",
  "confidence": "high"
}
"""
    fields = extract_task_capsule_fields(output)
    assert fields["summary"] == "Mapped the module"
    assert fields["evidence"]["files_read"] == ["src/app.py"]
    assert fields["findings"] == ["Found entry point"]
    assert fields["handoff"] == "Use the entry point next"
    assert fields["confidence"] == "high"


def test_extract_task_capsule_fields_invalid_block_falls_back() -> None:
    fields = extract_task_capsule_fields(
        "First useful paragraph.\n\nDELEGATION_RESULT:\n{bad json"
    )
    assert fields["summary"] == "First useful paragraph."
    assert fields["evidence"]["files_read"] == []
    assert fields["confidence"] == "unknown"


@pytest.mark.asyncio
async def test_create_capsule_initial_schema() -> None:
    await create_capsule(
        "abc12345", "Compare systems", [_task("map"), _task("compare", ["map"])], 2
    )
    capsule = load_capsule("abc12345")

    assert capsule_path("abc12345").exists()
    assert capsule["schema_version"] == 1
    assert capsule["job_id"] == "abc12345"
    assert capsule["milestone"] == "Compare systems"
    assert capsule["parallel_cap"] == 2
    assert capsule["dependency_graph"] == {"map": [], "compare": ["map"]}
    assert capsule["tasks"]["map"]["status"] == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_completed_task_persists_output_and_structured_fields() -> None:
    await create_capsule("abc12345", "Done", [_task("map")], 1)
    await mark_task_started("abc12345", "map", dependency_context="dependency output")
    await mark_task_completed(
        "abc12345",
        "map",
        "Raw output.\n\nDELEGATION_RESULT:\n"
        + json.dumps(
            {
                "summary": "Mapped files",
                "evidence": {
                    "files_read": ["a.py"],
                    "files_changed": [],
                    "commands_run": [],
                    "sources": [],
                },
                "findings": ["A"],
                "handoff": "Next",
                "confidence": "medium",
            }
        ),
        1.25,
        "gemini-test",
        {"total_tokens": 42},
    )

    task = load_capsule("abc12345")["tasks"]["map"]
    assert task["status"] == TaskStatus.COMPLETED
    assert task["duration"] == 1.25
    assert task["model"] == "gemini-test"
    assert task["tokens"] == {"total_tokens": 42}
    assert task["summary"] == "Mapped files"
    assert task["evidence"]["files_read"] == ["a.py"]
    assert task["findings"] == ["A"]
    assert task["handoff"] == "Next"
    assert task["confidence"] == "medium"
    assert task["dependency_context"] == "dependency output"


@pytest.mark.asyncio
async def test_parallel_task_updates_do_not_overwrite_each_other() -> None:
    await create_capsule(
        "parallel1",
        "Parallel updates",
        [_task("a"), _task("b"), _task("c")],
        3,
    )

    await asyncio.gather(
        mark_task_completed("parallel1", "a", "A done", 0.1),
        mark_task_completed("parallel1", "b", "B done", 0.2),
        mark_task_completed("parallel1", "c", "C done", 0.3),
    )

    tasks = load_capsule("parallel1")["tasks"]
    assert tasks["a"]["output"] == "A done"
    assert tasks["b"]["output"] == "B done"
    assert tasks["c"]["output"] == "C done"


@pytest.mark.asyncio
async def test_completed_capsule_statuses() -> None:
    await create_capsule("allgood", "All good", [_task("a"), _task("b")], 2)
    await mark_task_completed("allgood", "a", "A done", 0.1)
    await mark_task_completed("allgood", "b", "B done", 0.2)
    await mark_capsule_completed("allgood")
    assert load_capsule("allgood")["status"] == "completed"

    await create_capsule(
        "mixed123",
        "Mixed",
        [_task("ok"), _task("fail"), _task("slow"), _task("skip")],
        2,
    )
    await mark_task_completed("mixed123", "ok", "ok", 0.1)
    await mark_task_failed("mixed123", "fail", "boom", 0.2)
    await mark_task_timed_out("mixed123", "slow", "too slow", 30.0)
    await mark_task_skipped("mixed123", "skip", "Dependency fail error")
    await mark_capsule_completed("mixed123")
    capsule = load_capsule("mixed123")
    assert capsule["status"] == "completed_with_failures"
    assert len(capsule["synthesis"]["failed_or_skipped_tasks"]) == 3


@pytest.mark.asyncio
async def test_cancelled_capsule_marks_unfinished_tasks() -> None:
    await create_capsule("cancel1", "Cancel", [_task("done"), _task("pending")], 1)
    await mark_task_completed("cancel1", "done", "done", 0.1)
    await mark_capsule_cancelled("cancel1")

    capsule = load_capsule("cancel1")
    assert capsule["status"] == "cancelled"
    assert capsule["tasks"]["done"]["status"] == TaskStatus.COMPLETED
    assert capsule["tasks"]["pending"]["status"] == TaskStatus.ERROR
    assert capsule["tasks"]["pending"]["error"] == "Cancelled before completion"


@pytest.mark.asyncio
async def test_serialization_and_inspection_tools() -> None:
    await create_capsule("inspect1", "Inspect", [_task("map")], 1)
    await mark_task_completed(
        "inspect1",
        "map",
        "DELEGATION_RESULT: "
        + json.dumps(
            {
                "summary": "Map summary",
                "evidence": {"files_read": ["x.py"]},
                "findings": ["finding"],
                "handoff": "handoff",
                "confidence": "high",
            }
        ),
        0.5,
        "gemini-test",
        {"total_tokens": 12},
    )
    await mark_capsule_completed("inspect1")

    payload = serialize_capsule_result("inspect1")
    assert payload["capsule_path"] == ".supporter/delegations/inspect1.json"
    assert payload["totals"]["completed"] == 1
    assert payload["tasks"] == [
        {
            "id": "map",
            "status": "completed",
            "summary": "Map summary",
            "confidence": "high",
        }
    ]
    assert "inspect1" in query_delegation()
    assert "Map summary" in query_delegation(job_id="inspect1")
    assert "handoff" in query_delegation(job_id="inspect1", detail="tasks")
    assert "x.py" in query_delegation(job_id="inspect1", task_id="map")
    assert "was not found" in query_delegation(job_id="inspect1", task_id="missing")
    assert "was not found" in query_delegation(job_id="missing")


@pytest.mark.asyncio
async def test_effective_status_detects_stale_capsule() -> None:
    await create_capsule("stale1", "Stale", [_task("done"), _task("running")], 2)
    await mark_task_completed("stale1", "done", "done output", 0.1)

    capsule = load_capsule("stale1")

    old_time = (
        (datetime.now(UTC) - timedelta(minutes=16)).isoformat().replace("+00:00", "Z")
    )
    capsule["updated_at"] = old_time
    await save_capsule(capsule)

    loaded = load_capsule("stale1")
    assert loaded["status"] == "running"

    assert effective_status(loaded) == "interrupted_by_restart"

    assert loaded["tasks"]["done"]["status"] == TaskStatus.COMPLETED
    assert loaded["tasks"]["done"]["output"] == "done output"


@pytest.mark.asyncio
async def test_effective_status_live_capsule_unaffected() -> None:
    await create_capsule("live1", "Live", [_task("t1")], 1)
    capsule = load_capsule("live1")
    assert effective_status(capsule) == "running"


@pytest.mark.asyncio
async def test_delegate_tasks_creates_capsule_at_start() -> None:
    release = asyncio.Event()

    async def slow_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        await release.wait()
        return {
            "id": "t1",
            "status": TaskStatus.COMPLETED,
            "output": "done",
            "duration": 0.1,
            "model": "gemini-test",
            "tokens": {"total_tokens": 5},
        }

    tasks_json = json.dumps([{"id": "t1", "task": "slow task"}])
    with patch(
        "supporter.tools.delegate.scheduler.run_sub_agent", side_effect=slow_run
    ):
        plan = await delegate_tasks("Capsule start", tasks_json, max_parallel=1)
        job_id = next(line for line in plan.splitlines() if "Job ID:" in line).split(
            "`"
        )[1]
        bus = get_bus(job_id)
        queue = bus.subscribe()

        assert capsule_path(job_id).exists()
        capsule = load_capsule(job_id)
        assert capsule["milestone"] == "Capsule start"
        assert capsule["tasks"]["t1"]["status"] in {
            TaskStatus.PENDING,
            TaskStatus.STARTED,
        }

        release.set()
        completed_event = None
        for _ in range(20):
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            if event is None:
                break
            if isinstance(event, MilestoneCompleted):
                completed_event = event
                break

    assert completed_event is not None
    assert load_capsule(job_id)["status"] == "completed"


def test_load_capsule_rejects_non_object_json() -> None:
    path = capsule_path("badshape")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        load_capsule("badshape")


def test_effective_status_naive_timestamp_and_invalid_timestamp() -> None:
    assert (
        effective_status(
            {"status": "running", "updated_at": "2026-01-01T10:00:00", "job_id": "x"}
        )
        == "interrupted_by_restart"
    )
    assert (
        effective_status({"status": "running", "updated_at": object(), "job_id": "x"})
        == "running"
    )


@pytest.mark.asyncio
async def test_update_capsule_mutator_exception_propagates() -> None:
    await create_capsule("muterr01", "Mutator", [_task("a")], 1)

    def bad_mutator(capsule: dict[str, Any]) -> None:
        raise RuntimeError("bad mutator")

    with pytest.raises(RuntimeError, match="bad mutator"):
        await dc.update_capsule("muterr01", bad_mutator)


@pytest.mark.asyncio
async def test_cancelled_capsule_ignores_non_dict_task_entries() -> None:
    await create_capsule("nondict1", "Cancel", [_task("a")], 1)
    capsule = load_capsule("nondict1")
    capsule["tasks"]["broken"] = "not-dict"
    await save_capsule(capsule)
    await mark_capsule_cancelled("nondict1")
    updated = load_capsule("nondict1")
    assert updated["status"] == "cancelled"


def test_build_synthesis_skips_non_dict_task_values() -> None:
    synthesis = dc.build_synthesis(
        {
            "tasks": {
                "a": {"id": "a", "status": TaskStatus.COMPLETED.value, "summary": "ok"},
                "x": "bad",
            }
        }
    )
    assert "a: ok" in synthesis["answer"]


def test_serialize_capsule_result_handles_bad_tasks_and_synthesis_types() -> None:
    with patch("supporter.tools.delegate.capsule_query.load_capsule_safe") as mock_load:
        mock_load.return_value = {
            "job_id": "j",
            "milestone": "m",
            "status": "completed",
            "tasks": [],
            "synthesis": "bad",
        }
        payload = serialize_capsule_result("j")
    assert payload["tasks"] == []
    assert payload["key_findings"] == []


def test_private_helpers_and_query_branches(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    assert dc._normalize_evidence("bad") == {
        "files_read": [],
        "files_changed": [],
        "commands_run": [],
        "sources": [],
    }
    assert dc._normalize_confidence(1) == "unknown"
    assert dc._string_or_default("", "d") == "d"
    assert dc.status_value(TaskStatus.ERROR) == "error"
    assert capsule_query.duration("bad") == 0.0
    assert dc.preview("abcdef", 3).endswith("[truncated]")


def test_safe_job_id_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        dc._safe_job_id("x/../y")


@pytest.mark.asyncio
async def test_query_and_load_error_paths(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    root = tmp_path / ".supporter" / "delegations"
    root.mkdir(parents=True, exist_ok=True)
    (root / "broken.json").write_text("{", encoding="utf-8")
    assert "No delegation capsules found." in query_delegation(limit=1)
    assert "was not found" in query_delegation(job_id="missing", task_id="x")
    await create_capsule("qd123456", "Q", [_task("a")], 1)
    assert "Unknown detail" in query_delegation(job_id="qd123456", detail="weird")


@pytest.mark.asyncio
async def test_update_capsule_non_dict_json_object_from_reader() -> None:
    path = capsule_path("nondict2")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        await dc.update_capsule("nondict2", lambda c: None)


@pytest.mark.asyncio
async def test_mark_task_started_missing_task_propagates() -> None:
    await create_capsule("miss001", "M", [_task("a")], 1)
    with pytest.raises(KeyError, match="unknown"):
        await mark_task_started("miss001", "unknown")


def test_extract_fields_with_invalid_fenced_and_marked_json() -> None:
    bad_fenced = "```json\n{bad}\n```"
    fields1 = extract_task_capsule_fields(bad_fenced)
    assert fields1["confidence"] == "unknown"

    bad_marked = "DELEGATION_RESULT:\n{bad json"
    fields2 = extract_task_capsule_fields(bad_marked)
    assert fields2["confidence"] == "unknown"


def test_extract_fields_with_valid_fenced_json_block() -> None:
    fenced = """```json
{"summary":"ok","evidence":{"files_read":["a.py"]},"findings":[],"handoff":"","confidence":"high"}
```"""
    fields = extract_task_capsule_fields(fenced)
    assert fields["summary"] == "ok"
    assert fields["confidence"] == "high"


def test_extract_fields_with_invalid_marked_json_logs_path() -> None:
    out = "DELEGATION_RESULT:\n{not:json}"
    fields = extract_task_capsule_fields(out)
    assert fields["confidence"] == "unknown"


def test_task_totals_and_list_filtering() -> None:
    totals = capsule_query.task_totals(
        {
            "a": {"status": TaskStatus.COMPLETED},
            "b": {"status": TaskStatus.ERROR},
            "c": {"status": TaskStatus.SKIPPED},
            "d": {"status": TaskStatus.TIMEOUT},
            "e": "bad",
        }
    )
    assert totals["completed"] == 1
    assert totals["failed"] == 1
    assert totals["skipped"] == 1
    assert totals["timed_out"] == 1


@pytest.mark.asyncio
async def test_query_detail_full_and_summary_shapes() -> None:
    await create_capsule("full0001", "Full", [_task("a")], 1)
    await mark_task_completed("full0001", "a", "out", 0.1)
    await mark_capsule_completed("full0001")
    full = query_delegation(job_id="full0001", detail="full")
    assert full.startswith("```json")
    summary = query_delegation(job_id="full0001", detail="summary")
    assert "Delegation `full0001`" in summary
    status_view = query_delegation(status="completed", limit=10)
    assert "full0001" in status_view


@pytest.mark.asyncio
async def test_inspect_task_includes_error_and_skip_reason() -> None:
    await create_capsule("errs0001", "Err", [_task("a"), _task("b")], 1)
    await mark_task_failed("errs0001", "a", "boom", 0.2)
    await mark_task_skipped("errs0001", "b", "skip because dep")
    out_a = query_delegation(job_id="errs0001", task_id="a")
    out_b = query_delegation(job_id="errs0001", task_id="b")
    assert "- Error: boom" in out_a
    assert "- Skip reason: skip because dep" in out_b


@pytest.mark.asyncio
async def test_load_all_capsules_warns_on_invalid_record_shape() -> None:
    root = dc.delegations_dir()
    root.mkdir(parents=True, exist_ok=True)
    (root / "badshape.json").write_text('{"foo": "bar"}', encoding="utf-8")
    rows = query_delegation(limit=20)
    assert isinstance(rows, str)


@pytest.mark.asyncio
async def test_load_or_none_warns_on_invalid_json() -> None:
    root = dc.delegations_dir()
    root.mkdir(parents=True, exist_ok=True)
    (root / "badjson01.json").write_text("{", encoding="utf-8")
    out = query_delegation(job_id="badjson01")
    assert "capsule is unavailable" in out
    assert "JSONDecodeError" in out


def test_capsule_files_when_root_missing(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    assert capsule_query.capsule_files() == []


@pytest.mark.asyncio
async def test_formatters_for_empty_and_non_dict_tasks() -> None:
    await create_capsule("fmt00001", "Fmt", [_task("a")], 1)
    capsule = load_capsule("fmt00001")
    capsule["synthesis"] = "bad"
    capsule["tasks"] = {}
    await save_capsule(capsule)
    summary = query_delegation(job_id="fmt00001", detail="summary")
    tasks_view = query_delegation(job_id="fmt00001", detail="tasks")
    assert "Answer: none" in summary
    assert "no task records" in tasks_view.lower()

    capsule["tasks"] = {"bad": "value", "ok": capsule["tasks"].get("ok", {})}
    await save_capsule(capsule)
    tasks_view2 = query_delegation(job_id="fmt00001", detail="tasks")
    assert isinstance(tasks_view2, str)


@pytest.mark.asyncio
async def test_display_capsule_truncates_output_and_dependency_context() -> None:
    await create_capsule("disp0001", "Disp", [_task("a")], 1)
    long_text = "x" * 2000
    await mark_task_started("disp0001", "a", dependency_context=long_text)
    await mark_task_completed("disp0001", "a", long_text, 0.1)
    capsule = load_capsule("disp0001")
    capsule["tasks"]["broken"] = "not-a-dict"
    await save_capsule(capsule)
    full = query_delegation(job_id="disp0001", detail="full")
    assert "[truncated]" in full


def test_jsonish_non_string_branch() -> None:
    assert capsule_query.jsonish({"a": 1}) == '{"a": 1}'


@pytest.mark.asyncio
async def test_update_capsule_buffers_below_threshold(monkeypatch: Any) -> None:
    await create_capsule("buf00001", "Buf", [_task("a")], 1)
    calls = {"n": 0}
    original = dc._save_capsule_sync

    def counting(capsule: dict[str, Any]) -> None:
        calls["n"] += 1
        original(capsule)

    monkeypatch.setattr(dc, "_save_capsule_sync", counting)
    calls["n"] = 0
    await mark_task_started("buf00001", "a")
    await mark_task_started("buf00001", "a")
    assert calls["n"] == 0
    assert dc._CAPSULE_DIRTY_COUNT["buf00001"] == 2


@pytest.mark.asyncio
async def test_update_capsule_flushes_at_threshold(monkeypatch: Any) -> None:
    await create_capsule("th000001", "Th", [_task("a")], 1)
    calls = {"n": 0}
    original = dc._save_capsule_sync

    def counting(capsule: dict[str, Any]) -> None:
        calls["n"] += 1
        original(capsule)

    monkeypatch.setattr(dc, "_save_capsule_sync", counting)
    calls["n"] = 0
    for _ in range(dc.CAPSULE_FLUSH_EVERY):
        await mark_task_started("th000001", "a")
    assert calls["n"] == 1
    assert dc._CAPSULE_DIRTY_COUNT["th000001"] == 0


@pytest.mark.asyncio
async def test_update_capsule_flushes_on_status_transition(monkeypatch: Any) -> None:
    await create_capsule("st000001", "St", [_task("a")], 1)
    await mark_task_started("st000001", "a")
    calls = {"n": 0}
    original = dc._save_capsule_sync

    def counting(capsule: dict[str, Any]) -> None:
        calls["n"] += 1
        original(capsule)

    monkeypatch.setattr(dc, "_save_capsule_sync", counting)
    calls["n"] = 0
    await mark_capsule_completed("st000001")
    assert calls["n"] >= 1
    assert "st000001" not in dc._CAPSULE_CACHE


@pytest.mark.asyncio
async def test_update_capsule_save_failure_evicts_cache(monkeypatch: Any) -> None:
    await create_capsule("fail0001", "Fail", [_task("a")], 1)
    assert "fail0001" in dc._CAPSULE_CACHE

    def boom(_capsule: dict[str, Any]) -> None:
        raise OSError("disk full")

    for _ in range(dc.CAPSULE_FLUSH_EVERY - 1):
        await mark_task_started("fail0001", "a")
    monkeypatch.setattr(dc, "_save_capsule_sync", boom)
    with pytest.raises(OSError, match="disk full"):
        await mark_task_started("fail0001", "a")
    assert "fail0001" not in dc._CAPSULE_CACHE
    assert "fail0001" not in dc._CAPSULE_DIRTY_COUNT


@pytest.mark.asyncio
async def test_cache_evicts_oldest_when_over_max(monkeypatch: Any) -> None:
    monkeypatch.setattr(dc, "CAPSULE_CACHE_MAX", 3)
    for i in range(5):
        await create_capsule(f"evict{i:03d}", "E", [_task("a")], 1)
    assert len(dc._CAPSULE_CACHE) == 3
    assert "evict000" not in dc._CAPSULE_CACHE
    assert "evict001" not in dc._CAPSULE_CACHE
    assert "evict004" in dc._CAPSULE_CACHE


@pytest.mark.asyncio
async def test_cache_eviction_flushes_dirty_entry(monkeypatch: Any) -> None:
    monkeypatch.setattr(dc, "CAPSULE_CACHE_MAX", 2)
    await create_capsule("dirty001", "D", [_task("a")], 1)
    await mark_task_started("dirty001", "a")
    assert dc._CAPSULE_DIRTY_COUNT.get("dirty001", 0) > 0
    saved_jobs: list[str] = []
    original = dc._save_capsule_sync

    def tracking(capsule: dict[str, Any]) -> None:
        saved_jobs.append(str(capsule["job_id"]))
        original(capsule)

    monkeypatch.setattr(dc, "_save_capsule_sync", tracking)
    await create_capsule("filler01", "F", [_task("a")], 1)
    await create_capsule("filler02", "F", [_task("a")], 1)
    assert "dirty001" not in dc._CAPSULE_CACHE
    assert "dirty001" in saved_jobs
    disk = load_capsule("dirty001")
    assert disk["tasks"]["a"]["status"] == TaskStatus.STARTED.value


@pytest.mark.asyncio
async def test_load_all_capsules_sorts_by_updated_at_despite_mtime(
    tmp_path: Any,
) -> None:
    import os

    older = datetime.now(UTC) - timedelta(hours=2)
    newer = datetime.now(UTC) - timedelta(minutes=5)
    await create_capsule("old00001", "O", [_task("a")], 1)
    await create_capsule("new00001", "N", [_task("a")], 1)
    old_path = capsule_path("old00001")
    new_path = capsule_path("new00001")
    old_data = load_capsule("old00001")
    new_data = load_capsule("new00001")
    old_data["updated_at"] = older.isoformat().replace("+00:00", "Z")
    new_data["updated_at"] = newer.isoformat().replace("+00:00", "Z")
    await save_capsule(old_data)
    await save_capsule(new_data)
    fresh_mtime = (datetime.now(UTC) - timedelta(seconds=1)).timestamp()
    stale_mtime = (datetime.now(UTC) - timedelta(hours=3)).timestamp()
    os.utime(old_path, (fresh_mtime, fresh_mtime))
    os.utime(new_path, (stale_mtime, stale_mtime))
    result = capsule_query.load_all_capsules(limit=1)
    assert len(result) == 1
    assert result[0]["job_id"] == "new00001"


@pytest.mark.asyncio
async def test_backend_persisted_through_capsule_round_trip() -> None:
    """Regression: opencode tasks must survive create→load with their backend."""
    task = _task("t1")
    task["backend"] = "opencode"
    await create_capsule("be000001", "Backend round-trip", [task], 1)
    loaded = load_capsule("be000001")
    assert loaded["tasks"]["t1"]["backend"] == "opencode"


def test_initial_task_record_preserves_backend() -> None:
    """Regression: _initial_task_record must persist the given backend."""
    task = _task("t1")
    task["backend"] = "opencode"
    record = dc._initial_task_record(task)
    assert record["backend"] == "opencode"


def test_initial_task_record_defaults_backend_to_gemini() -> None:
    """Legacy fallback: no backend field → gemini."""
    task = _task("t1")
    task.pop("backend", None)
    record = dc._initial_task_record(task)
    assert record["backend"] == "gemini"


def test_initial_task_record_persists_live_and_pre_approved() -> None:
    task = _task("t1")
    task["live"] = True
    task["pre_approved_commands"] = ["ls", "git status"]
    record = dc._initial_task_record(task)
    assert record["live"] is True
    assert record["pre_approved_commands"] == ["ls", "git status"]


def test_initial_task_record_defaults_live_and_pre_approved() -> None:
    record = dc._initial_task_record(_task("t1"))
    assert record["live"] is False
    assert record["pre_approved_commands"] == []


@pytest.mark.asyncio
async def test_live_and_pre_approved_survive_capsule_roundtrip() -> None:
    task = _task("t1")
    task["live"] = True
    task["pre_approved_commands"] = ["pwd"]
    await create_capsule("rf-roundtrip", "RF", [task], 1)
    loaded = load_capsule("rf-roundtrip")["tasks"]["t1"]
    assert loaded["live"] is True
    assert loaded["pre_approved_commands"] == ["pwd"]
