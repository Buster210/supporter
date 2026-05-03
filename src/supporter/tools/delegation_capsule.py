from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

from ..config import config
from ..logger import logger
from ..types import TaskStatus

if TYPE_CHECKING:
    from ..agent import ChatAgent

CAPSULE_SCHEMA_VERSION = 1
ACTIVE_CAPSULE_STATUSES = {"pending", "running"}
EVIDENCE_KEYS = ("files_read", "files_changed", "commands_run", "sources")
STALE_CAPSULE_THRESHOLD_SECONDS = 900
_CAPSULE_LOCKS: dict[str, asyncio.Lock] = {}

def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

def delegations_dir() -> Path:
    return Path(config.allowed_directories[0]) / ".supporter" / "delegations"

def capsule_path(job_id: str) -> Path:
    return delegations_dir() / f"{_safe_job_id(job_id)}.json"

def capsule_relative_path(job_id: str) -> str:
    return str(Path(".supporter") / "delegations" / f"{_safe_job_id(job_id)}.json")

def _safe_job_id(job_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", job_id)

def _capsule_lock(job_id: str) -> asyncio.Lock:
    if job_id not in _CAPSULE_LOCKS:
        _CAPSULE_LOCKS[job_id] = asyncio.Lock()
    return _CAPSULE_LOCKS[job_id]

def _release_capsule_lock(job_id: str) -> None:
    _CAPSULE_LOCKS.pop(job_id, None)

async def create_capsule(
    job_id: str,
    milestone: str,
    tasks: list[dict[str, Any]],
    parallel_cap: int,
) -> dict[str, Any]:
    now = utc_now()
    capsule: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "job_id": job_id,
        "milestone": milestone,
        "status": "running",
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "parallel_cap": parallel_cap,
        "dependency_graph": {t["id"]: list(t.get("depends_on", [])) for t in tasks},
        "tasks": {t["id"]: _initial_task_record(t) for t in tasks},
        "synthesis": {
            "answer": "",
            "key_findings": [],
            "failed_or_skipped_tasks": [],
            "recommended_next_steps": [],
        },
    }
    async with _capsule_lock(job_id):
        await save_capsule(capsule)
    return capsule

def load_capsule(job_id: str) -> dict[str, Any]:
    path = capsule_path(job_id)
    with path.open(encoding="utf-8") as f:
        data: Any = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Capsule `{job_id}` is not a JSON object at {path}")
    return cast(dict[str, Any], data)

async def save_capsule(capsule: dict[str, Any]) -> None:
    job_id = str(capsule.get("job_id", "unknown"))
    path = capsule_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def _write() -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(capsule, f, indent=2, ensure_ascii=False)
            
    await asyncio.to_thread(_write)

async def update_capsule(
    job_id: str, 
    mutator: Callable[[dict[str, Any]], None]
) -> dict[str, Any] | None:
    async with _capsule_lock(job_id):
        try:
            capsule = load_capsule(job_id)
            mutator(capsule)
            capsule["updated_at"] = utc_now()
            await save_capsule(capsule)
            return capsule
        except Exception as exc:
            logger.error(f"Failed to update capsule [job={job_id}, error={exc}]")
            return None

def _initial_task_record(task_def: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task_def["id"],
        "description": task_def.get("task", task_def.get("description", "")),
        "agent": task_def.get("agent"),
        "status": TaskStatus.PENDING.value,
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "duration": None,
        "output": None,
        "summary": "",
        "evidence": _default_evidence(),
        "findings": [],
        "handoff": "",
        "confidence": "unknown",
        "attempts": 0,
    }

async def mark_task_started(job_id: str, task_id: str) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        task["status"] = TaskStatus.STARTED.value
        task["started_at"] = utc_now()
        task["attempts"] += 1
    return await update_capsule(job_id, mutate)

async def mark_task_completed(
    job_id: str, 
    task_id: str, 
    output: str, 
    duration: float
) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        fields = extract_task_capsule_fields(output)
        task.update({
            "status": TaskStatus.COMPLETED.value,
            "completed_at": utc_now(),
            "duration": duration,
            "output": output,
            **fields
        })
    return await update_capsule(job_id, mutate)

async def mark_task_failed(
    job_id: str, 
    task_id: str, 
    error: str, 
    duration: float
) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        task.update({
            "status": TaskStatus.ERROR.value,
            "completed_at": utc_now(),
            "duration": duration,
            "error": error
        })
    return await update_capsule(job_id, mutate)

async def mark_task_skipped(job_id: str, task_id: str, reason: str) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        task.update({
            "status": TaskStatus.SKIPPED.value,
            "completed_at": utc_now(),
            "skip_reason": reason
        })
    return await update_capsule(job_id, mutate)

async def mark_capsule_completed(job_id: str) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        tasks = capsule.get("tasks", {})
        statuses = {t.get("status") for t in tasks.values()}
        if statuses <= {TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value}:
            capsule["status"] = "completed"
        else:
            capsule["status"] = "completed_with_failures"
        capsule["completed_at"] = utc_now()
        capsule["synthesis"] = build_synthesis(capsule)
    
    res = await update_capsule(job_id, mutate)
    _release_capsule_lock(job_id)
    return res

async def mark_capsule_cancelled(job_id: str) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        now = utc_now()
        capsule["status"] = "cancelled"
        capsule["completed_at"] = now
        for task in capsule.get("tasks", {}).values():
            if task.get("status") in {TaskStatus.PENDING.value, TaskStatus.STARTED.value}:
                task.update({
                    "status": TaskStatus.ERROR.value,
                    "completed_at": now,
                    "error": "Cancelled by user/system"
                })
        capsule["synthesis"] = build_synthesis(capsule)
    
    res = await update_capsule(job_id, mutate)
    _release_capsule_lock(job_id)
    return res

def _get_task(capsule: dict[str, Any], task_id: str) -> dict[str, Any]:
    tasks = capsule.get("tasks", {})
    if task_id not in tasks:
        raise KeyError(f"Task `{task_id}` not found in capsule")
    return cast(dict[str, Any], tasks[task_id])

def _default_evidence() -> dict[str, list[str]]:
    return {k: [] for k in EVIDENCE_KEYS}

def effective_status(capsule: dict[str, Any]) -> str:
    raw = str(capsule.get("status", "unknown"))
    if raw not in ACTIVE_CAPSULE_STATUSES:
        return raw
    try:
        updated_dt = datetime.fromisoformat(str(capsule["updated_at"]).replace("Z", "+00:00"))
        if (datetime.now(UTC) - updated_dt).total_seconds() > STALE_CAPSULE_THRESHOLD_SECONDS:
            return "interrupted"
    except Exception:
        pass
    return raw

def build_synthesis(capsule: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], capsule.get("synthesis", {}))

def extract_task_capsule_fields(output: str) -> dict[str, Any]:
    return {
        "summary": output[:200] + "..." if len(output) > 200 else output,
        "findings": [],
        "evidence": _default_evidence()
    }

def query_delegation(
    job_id: str | None = None,
    task_id: str | None = None,
    detail: str = "summary",
    status: str | None = None,
    limit: int = 10,
) -> str:
    if not job_id:
        return _list_delegations(status=status, limit=limit)
    try:
        capsule = load_capsule(job_id)
        if task_id:
            return _inspect_task(capsule, task_id)
        return _inspect_delegation(capsule, detail)
    except Exception as exc:
        return f"Error querying delegation `{job_id}`: {exc}"

def _list_delegations(status: str | None = None, limit: int = 10) -> str:
    root = delegations_dir()
    if not root.exists():
        return "No delegations found."
    
    capsules = []
    for p in root.glob("*.json"):
        try:
            with p.open(encoding="utf-8") as f:
                c = json.load(f)
                if status and effective_status(c) != status:
                    continue
                capsules.append(c)
        except Exception:
            continue
            
    capsules.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    capsules = capsules[:limit]
    
    if not capsules:
        return "No matching delegations found."
        
    lines = ["| Job | Status | Milestone | Updated |", "|---|---|---|---|"]
    for c in capsules:
        lines.append(f"| `{c.get('job_id')}` | {effective_status(c)} | {c.get('milestone')} | {c.get('updated_at')} |")
    return "\n".join(lines)

def _inspect_delegation(capsule: dict[str, Any], detail: str) -> str:
    job_id = capsule.get("job_id")
    lines = [
        f"# Delegation `{job_id}`",
        f"- **Milestone**: {capsule.get('milestone')}",
        f"- **Status**: {effective_status(capsule)}",
        f"- **Updated**: {capsule.get('updated_at')}",
        "\n## Tasks"
    ]
    for tid, t in capsule.get("tasks", {}).items():
        lines.append(f"- [{t.get('status')}] `{tid}`: {t.get('description')}")
    
    if detail == "full":
        lines.append("\n## Synthesis")
        lines.append(json.dumps(capsule.get("synthesis"), indent=2))
        
    return "\n".join(lines)

def _inspect_task(capsule: dict[str, Any], task_id: str) -> str:
    task = _get_task(capsule, task_id)
    return json.dumps(task, indent=2, ensure_ascii=False)

def serialize_capsule_result(job_id: str) -> dict[str, Any]:
    capsule = load_capsule(job_id)
    return {
        "job_id": job_id,
        "status": effective_status(capsule),
        "milestone": capsule.get("milestone"),
        "tasks": capsule.get("tasks")
    }
