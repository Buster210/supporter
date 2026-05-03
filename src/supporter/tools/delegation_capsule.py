from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ..logger import logger

_CAPSULE_DIR = Path.home() / ".supporter" / "delegations"
_CAPSULE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_job_id(job_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", job_id)


def _get_path(job_id: str) -> Path:
    return _CAPSULE_DIR / f"{_safe_job_id(job_id)}.json"


def _initial_task_record(t: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": t["id"],
        "task": t["task"],
        "agent": t.get("agent"),
        "status": "pending",
        "depends_on": t.get("depends_on", []),
        "output": None,
        "duration": None,
        "started_at": None,
        "completed_at": None,
    }


async def create_capsule(
    job_id: str,
    milestone: str,
    tasks: list[dict[str, Any]],
    parallel_cap: int,
) -> None:
    data = {
        "job_id": job_id,
        "milestone": milestone,
        "status": "running",
        "parallel_cap": parallel_cap,
        "created_at": datetime.now(UTC).isoformat(),
        "updated_at": datetime.now(UTC).isoformat(),
        "tasks": [_initial_task_record(t) for t in tasks],
    }
    path = _get_path(job_id)
    await asyncio.to_thread(path.write_text, json.dumps(data, indent=2))


async def update_capsule(
    job_id: str, updater: Callable[[dict[str, Any]], None]
) -> None:
    path = _get_path(job_id)
    if not path.exists():
        return

    def _atomic_update() -> None:
        try:
            data = json.loads(path.read_text())
            updater(data)
            data["updated_at"] = datetime.now(UTC).isoformat()
            path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Capsule update failed for {job_id}: {e}")

    await asyncio.to_thread(_atomic_update)


async def mark_task_started(job_id: str, task_id: str) -> None:
    def _update(data: dict[str, Any]) -> None:
        for t in data["tasks"]:
            if t["id"] == task_id:
                t["status"] = "running"
                t["started_at"] = datetime.now(UTC).isoformat()
                break

    await update_capsule(job_id, _update)


async def mark_task_completed(
    job_id: str, task_id: str, output: str, duration: float
) -> None:
    def _update(data: dict[str, Any]) -> None:
        for t in data["tasks"]:
            if t["id"] == task_id:
                t["status"] = "completed"
                t["output"] = output
                t["duration"] = round(duration, 2)
                t["completed_at"] = datetime.now(UTC).isoformat()
                break

    await update_capsule(job_id, _update)


async def mark_task_failed(
    job_id: str, task_id: str, error: str, duration: float
) -> None:
    def _update(data: dict[str, Any]) -> None:
        for t in data["tasks"]:
            if t["id"] == task_id:
                t["status"] = "failed"
                t["output"] = f"Error: {error}"
                t["duration"] = round(duration, 2)
                t["completed_at"] = datetime.now(UTC).isoformat()
                break

    await update_capsule(job_id, _update)


async def mark_task_skipped(job_id: str, task_id: str, reason: str) -> None:
    def _update(data: dict[str, Any]) -> None:
        for t in data["tasks"]:
            if t["id"] == task_id:
                t["status"] = "skipped"
                t["output"] = f"Skipped: {reason}"
                t["completed_at"] = datetime.now(UTC).isoformat()
                break

    await update_capsule(job_id, _update)


async def mark_task_timed_out(
    job_id: str, task_id: str, reason: str, duration: float
) -> None:
    def _update(data: dict[str, Any]) -> None:
        for t in data["tasks"]:
            if t["id"] == task_id:
                t["status"] = "timeout"
                t["output"] = f"Timeout: {reason}"
                t["duration"] = round(duration, 2)
                t["completed_at"] = datetime.now(UTC).isoformat()
                break

    await update_capsule(job_id, _update)


async def mark_capsule_completed(job_id: str) -> None:
    await update_capsule(job_id, lambda d: d.update({"status": "completed"}))


async def mark_capsule_cancelled(job_id: str) -> None:
    await update_capsule(job_id, lambda d: d.update({"status": "cancelled"}))


async def query_delegation(job_id: str) -> dict[str, Any] | None:
    path = _get_path(job_id)
    if not path.exists():
        return None
    try:
        return cast(dict[str, Any], json.loads(await asyncio.to_thread(path.read_text)))
    except Exception as e:
        logger.error(f"Capsule read failed for {job_id}: {e}")
        return None


def serialize_capsule(data: dict[str, Any]) -> dict[str, Any]:
    tasks = data.get("tasks", [])
    return {
        "job_id": data["job_id"],
        "milestone": data["milestone"],
        "status": data["status"],
        "totals": {
            "completed": sum(1 for t in tasks if t["status"] == "completed"),
            "failed": sum(1 for t in tasks if t["status"] in ("failed", "timeout")),
            "skipped": sum(1 for t in tasks if t["status"] == "skipped"),
        },
        "tasks": [
            {"id": t["id"], "status": t["status"], "duration": t.get("duration")}
            for t in tasks
        ],
    }
