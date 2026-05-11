from __future__ import annotations

import json
from typing import Any

from ..types import TaskStatus
from .capsule import (
    _status_value,
    capsule_relative_path,
    effective_status,
    first_compact_paragraph,
    load_capsule,
)


def serialize_capsule_result(job_id: str) -> dict[str, Any]:
    try:
        capsule = load_capsule(job_id)
    except (
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        return capsule_error_payload(job_id, exc)

    tasks = capsule.get("tasks", {})
    if not isinstance(tasks, dict):
        tasks = {}
    synthesis = capsule.get("synthesis", {})
    if not isinstance(synthesis, dict):
        synthesis = {}
    return {
        "job_id": capsule.get("job_id", job_id),
        "milestone": capsule.get("milestone", ""),
        "status": effective_status(capsule),
        "capsule_path": capsule_relative_path(job_id),
        "totals": task_totals(tasks),
        "key_findings": synthesis.get("key_findings", []),
        "failed_or_skipped_tasks": synthesis.get("failed_or_skipped_tasks", []),
        "recommended_next_steps": synthesis.get("recommended_next_steps", []),
        "tasks": [
            {
                "id": task.get("id", task_id),
                "status": _status_value(task.get("status", "")),
                "summary": task.get("summary")
                or first_compact_paragraph(str(task.get("output", ""))),
                "confidence": task.get("confidence", "unknown"),
            }
            for task_id, task in tasks.items()
            if isinstance(task, dict)
        ],
    }


def query_delegation(
    job_id: str | None = None,
    task_id: str | None = None,
    detail: str = "summary",
    status: str | None = None,
    limit: int = 10,
) -> str:
    from .capsule_view import inspect_delegation, inspect_task, list_delegations

    if not job_id:
        return list_delegations(status=status, limit=limit)
    if task_id:
        return inspect_task(job_id, task_id)
    return inspect_delegation(job_id, detail)


def task_totals(tasks: dict[str, Any]) -> dict[str, int]:
    completed = failed = skipped = timed_out = tokens = 0
    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        task_status = _status_value(task.get("status", ""))
        if task_status == TaskStatus.COMPLETED.value:
            completed += 1
        elif task_status == TaskStatus.ERROR.value:
            failed += 1
        elif task_status == TaskStatus.SKIPPED.value:
            skipped += 1
        elif task_status == TaskStatus.TIMEOUT.value:
            timed_out += 1
        task_tokens = task.get("tokens", {})
        if isinstance(task_tokens, dict):
            tokens += int(task_tokens.get("total_tokens") or 0)
    return {
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "timed_out": timed_out,
        "tokens": tokens,
    }


def capsule_error_payload(job_id: str, exc: BaseException) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "milestone": "",
        "status": "unavailable",
        "capsule_path": capsule_relative_path(job_id),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "action": "Start a new delegation or remove the corrupt capsule file.",
        },
        "totals": {
            "completed": 0,
            "failed": 0,
            "skipped": 0,
            "timed_out": 0,
            "tokens": 0,
        },
        "key_findings": [],
        "failed_or_skipped_tasks": [],
        "recommended_next_steps": [],
        "tasks": [],
    }


__all__ = [
    "capsule_error_payload",
    "query_delegation",
    "serialize_capsule_result",
    "task_totals",
]
