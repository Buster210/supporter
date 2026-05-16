from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, cast

from ...logger import logger
from ...types import TaskStatus
from .capsule import (
    capsule_relative_path,
    delegations_dir,
    effective_status,
    first_compact_paragraph,
    load_capsule_safe,
    status_value,
)


def serialize_capsule_result(job_id: str) -> dict[str, Any]:
    capsule = load_capsule_safe(job_id)
    if capsule.get("status") == "unavailable" and "error" in capsule:
        return capsule

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
                "status": status_value(task.get("status", "")),
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
        task_status = status_value(task.get("status", ""))
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


def capsule_files() -> list[Path]:
    root = delegations_dir()
    if not root.exists():
        return []
    return list(root.glob("*.json"))


def jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def duration(value: Any) -> float:
    try:
        return float(value)
    except TypeError, ValueError:
        return 0.0


def load_all_capsules(limit: int | None = None) -> list[dict[str, Any]]:
    paths = list(capsule_files())
    with contextlib.suppress(OSError):
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if limit is not None:
        paths = paths[: max(limit * 2, limit + 10)]
    capsules: dict[str, dict[str, Any]] = {}
    for path in paths:
        try:
            with path.open(encoding="utf-8") as f:
                data: Any = json.load(f)
            if isinstance(data, dict) and "job_id" in data:
                capsules[str(data["job_id"])] = cast(dict[str, Any], data)
            else:
                logger.warning(f"Skipping invalid delegation capsule [path={path}]")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                f"Failed to load delegation capsule "
                f"[path={path}, error={type(exc).__name__}: {exc}]"
            )
    result = list(capsules.values())
    if limit is not None:
        result.sort(key=lambda c: str(c.get("updated_at", "")), reverse=True)
        result = result[:limit]
    return result
