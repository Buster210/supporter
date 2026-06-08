from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ...logger import logger
from ...types import TaskStatus
from .. import resolved_project_root

_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)

CAPSULE_SCHEMA_VERSION = 1
ACTIVE_CAPSULE_STATUSES = {"pending", "running"}
_PREVIEW_LIMIT = 300
_SUMMARY_LIMIT = 500
_EVIDENCE_KEYS = ("files_read", "files_changed", "commands_run", "sources")
EVIDENCE_KEYS = _EVIDENCE_KEYS
_CONFIDENCE_VALUES = frozenset({"low", "medium", "high"})

_CAPSULE_LOCKS: dict[str, asyncio.Lock] = {}
_CAPSULE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
_CAPSULE_DIRTY_COUNT: dict[str, int] = {}

CAPSULE_FLUSH_EVERY = 5
CAPSULE_CACHE_MAX = 64
TERMINAL_CAPSULE_STATUSES = {"completed", "completed_with_failures", "cancelled"}


async def _cache_set(job_id: str, capsule: dict[str, Any]) -> None:
    _CAPSULE_CACHE[job_id] = capsule
    _CAPSULE_CACHE.move_to_end(job_id)
    while len(_CAPSULE_CACHE) > CAPSULE_CACHE_MAX:
        evict_id, evict_capsule = _CAPSULE_CACHE.popitem(last=False)
        if _CAPSULE_DIRTY_COUNT.pop(evict_id, 0) > 0:
            await asyncio.to_thread(_save_capsule_sync, evict_capsule)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def delegations_dir() -> Path:
    return resolved_project_root() / ".supporter" / "delegations"


def capsule_relative_path(job_id: str) -> str:
    return str(Path(".supporter") / "delegations" / f"{_safe_job_id(job_id)}.json")


def capsule_path(job_id: str) -> Path:
    return delegations_dir() / f"{_safe_job_id(job_id)}.json"


def _safe_job_id(job_id: str) -> str:
    if not job_id or Path(job_id).name != job_id:
        raise ValueError("Invalid delegation job id")
    return job_id


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
        await _cache_set(job_id, capsule)
    return capsule


def load_capsule(job_id: str) -> dict[str, Any]:
    cached = _CAPSULE_CACHE.get(job_id)
    if cached is not None:
        return cached
    path = capsule_path(job_id)
    with path.open(encoding="utf-8") as f:
        data: Any = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Capsule `{job_id}` is not a JSON object at {path}")
    return cast(dict[str, Any], data)


def load_capsule_safe(job_id: str) -> dict[str, Any]:
    try:
        return load_capsule(job_id)
    except (
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        return unavailable_capsule(job_id, exc)


def unavailable_capsule(job_id: str, exc: BaseException) -> dict[str, Any]:
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
        "synthesis": {},
    }


def effective_status(capsule: dict[str, Any]) -> str:
    raw = str(capsule.get("status", "unknown"))
    if raw not in ACTIVE_CAPSULE_STATUSES:
        return raw
    try:
        updated_str = str(capsule["updated_at"]).replace("Z", "+00:00")
        updated_dt = datetime.fromisoformat(updated_str)
        if updated_dt.tzinfo is None:
            updated_dt = updated_dt.replace(tzinfo=UTC)
        if (datetime.now(UTC) - updated_dt).total_seconds() > 900:
            return "interrupted_by_restart"
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            f"Delegation capsule has invalid updated_at "
            f"[job={capsule.get('job_id', 'unknown')}, "
            f"error={type(exc).__name__}: {exc}]"
        )
    return raw


def _save_capsule_sync(capsule: dict[str, Any]) -> None:
    job_id = str(capsule["job_id"])
    path = capsule_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(capsule, f, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


async def save_capsule(capsule: dict[str, Any]) -> None:
    await asyncio.to_thread(_save_capsule_sync, capsule)


async def update_capsule(
    job_id: str, mutator: Callable[[dict[str, Any]], None]
) -> dict[str, Any] | None:
    path = capsule_path(job_id)
    async with _capsule_lock(job_id):
        try:
            capsule = _CAPSULE_CACHE.get(job_id)
            if capsule is None:
                if not path.exists():
                    raise FileNotFoundError(
                        f"Delegation capsule missing during update: {path}"
                    )
                capsule = await asyncio.to_thread(_read_capsule_sync, path, job_id)
                await _cache_set(job_id, capsule)
            else:
                _CAPSULE_CACHE.move_to_end(job_id)
            pre_status = capsule.get("status")
            mutator(capsule)
            capsule["updated_at"] = utc_now()
            post_status = capsule.get("status")
            status_changed = pre_status != post_status
            dirty = _CAPSULE_DIRTY_COUNT.get(job_id, 0) + 1
            is_terminal = post_status in TERMINAL_CAPSULE_STATUSES
            if is_terminal or status_changed or dirty >= CAPSULE_FLUSH_EVERY:
                try:
                    await save_capsule(capsule)
                except Exception:
                    _CAPSULE_CACHE.pop(job_id, None)
                    _CAPSULE_DIRTY_COUNT.pop(job_id, None)
                    raise
                _CAPSULE_DIRTY_COUNT[job_id] = 0
                if is_terminal:
                    _CAPSULE_CACHE.pop(job_id, None)
            else:
                _CAPSULE_DIRTY_COUNT[job_id] = dirty
            return capsule
        except Exception as exc:
            logger.error(
                f"Delegation capsule update failed "
                f"[job={job_id}, path={path}, error={type(exc).__name__}: {exc}]"
            )
            raise


async def mark_task_started(
    job_id: str, task_id: str, dependency_context: str = ""
) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        task["status"] = TaskStatus.STARTED.value
        task["started_at"] = utc_now()
        if dependency_context:
            task["dependency_context"] = dependency_context

    return await update_capsule(job_id, mutate)


async def mark_task_completed(
    job_id: str,
    task_id: str,
    output: str,
    duration: float,
    model: str | None = None,
    tokens: dict[str, Any] | None = None,
    parsed_fields: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    parsed = parsed_fields if isinstance(parsed_fields, dict) else None
    if parsed is None:
        parsed = extract_task_capsule_fields(output)

    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        task.update(
            {
                "status": TaskStatus.COMPLETED.value,
                "completed_at": utc_now(),
                "duration": duration,
                "model": model or task.get("model", ""),
                "tokens": tokens or {},
                "output": output,
                "summary": parsed["summary"],
                "evidence": parsed["evidence"],
                "findings": parsed["findings"],
                "handoff": parsed["handoff"],
                "confidence": parsed["confidence"],
                "error": "",
                "skip_reason": "",
            }
        )

    return await update_capsule(job_id, mutate)


async def mark_task_failed(
    job_id: str,
    task_id: str,
    error: str,
    duration: float,
    output: str = "",
) -> dict[str, Any] | None:
    parsed = extract_task_capsule_fields(output or error)

    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        task.update(
            {
                "status": TaskStatus.ERROR.value,
                "completed_at": utc_now(),
                "duration": duration,
                "output": output or error,
                "summary": parsed["summary"],
                "evidence": parsed["evidence"],
                "findings": parsed["findings"],
                "handoff": parsed["handoff"],
                "confidence": parsed["confidence"],
                "error": error,
            }
        )

    return await update_capsule(job_id, mutate)


async def mark_task_timed_out(
    job_id: str,
    task_id: str,
    error: str,
    duration: float,
) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        task.update(
            {
                "status": TaskStatus.TIMEOUT.value,
                "completed_at": utc_now(),
                "duration": duration,
                "output": error,
                "summary": first_compact_paragraph(error),
                "error": error,
                "confidence": "unknown",
            }
        )

    return await update_capsule(job_id, mutate)


async def mark_task_skipped(
    job_id: str,
    task_id: str,
    skip_reason: str,
) -> dict[str, Any] | None:
    output = f"Skipped: {skip_reason}"

    def mutate(capsule: dict[str, Any]) -> None:
        task = _get_task(capsule, task_id)
        task.update(
            {
                "status": TaskStatus.SKIPPED.value,
                "completed_at": utc_now(),
                "duration": 0.0,
                "output": output,
                "summary": output,
                "skip_reason": skip_reason,
                "confidence": "unknown",
            }
        )

    return await update_capsule(job_id, mutate)


async def mark_capsule_completed(job_id: str) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        tasks = capsule.get("tasks", {})
        statuses = {
            status_value(task.get("status", "pending"))
            for task in tasks.values()
            if isinstance(task, dict)
        }
        if statuses and statuses <= {TaskStatus.COMPLETED.value}:
            capsule["status"] = "completed"
        else:
            capsule["status"] = "completed_with_failures"
        capsule["completed_at"] = utc_now()
        capsule["synthesis"] = build_synthesis(capsule)

    capsule = await update_capsule(job_id, mutate)
    if capsule is not None:
        _release_capsule_lock(job_id)
    return capsule


async def mark_capsule_cancelled(job_id: str) -> dict[str, Any] | None:
    def mutate(capsule: dict[str, Any]) -> None:
        now = utc_now()
        capsule["status"] = "cancelled"
        capsule["completed_at"] = now
        for task in capsule.get("tasks", {}).values():
            if not isinstance(task, dict):
                continue
            if status_value(task.get("status", "")) in {
                TaskStatus.PENDING.value,
                TaskStatus.STARTED.value,
            }:
                task.update(
                    {
                        "status": TaskStatus.ERROR.value,
                        "completed_at": now,
                        "error": "Cancelled before completion",
                    }
                )
        capsule["synthesis"] = build_synthesis(capsule)

    capsule = await update_capsule(job_id, mutate)
    if capsule is not None:
        _release_capsule_lock(job_id)
    return capsule


def extract_task_capsule_fields(output: str) -> dict[str, Any]:
    parsed = _parse_delegation_result(output)
    if parsed is None:
        return {
            "summary": first_compact_paragraph(output),
            "evidence": default_evidence(),
            "findings": [],
            "handoff": "",
            "confidence": "unknown",
        }

    evidence = parsed.get("evidence")
    return {
        "summary": _string_or_default(
            parsed.get("summary"), first_compact_paragraph(output)
        ),
        "evidence": _normalize_evidence(evidence),
        "findings": parsed.get("findings")
        if isinstance(parsed.get("findings"), list)
        else [],
        "handoff": _string_or_default(parsed.get("handoff"), ""),
        "confidence": _normalize_confidence(parsed.get("confidence")),
    }


def validate_delegation_payload(output: str) -> bool:
    """Hard schema check on a delegated agent's structured result block.

    True only when the JSON block parses and every contract field has the right
    type. Empty evidence/findings lists are valid -- a correct-but-sparse result
    must pass, so emptiness is never grounds for rejection.
    """
    parsed = _parse_delegation_result(output)
    if parsed is None:
        return False
    summary = parsed.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        return False
    confidence = parsed.get("confidence")
    if not isinstance(confidence, str) or confidence not in _CONFIDENCE_VALUES:
        return False
    evidence = parsed.get("evidence")
    if not isinstance(evidence, dict) or any(
        not isinstance(evidence.get(key), list) for key in EVIDENCE_KEYS
    ):
        return False
    if not isinstance(parsed.get("findings"), list):
        return False
    return isinstance(parsed.get("handoff"), str)


def build_synthesis(capsule: dict[str, Any]) -> dict[str, Any]:
    key_findings: list[Any] = []
    failed_or_skipped: list[dict[str, str]] = []
    next_steps: list[str] = []
    answer_parts: list[str] = []

    for task in capsule.get("tasks", {}).values():
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id", ""))
        status = status_value(task.get("status", ""))
        summary = str(
            task.get("summary") or first_compact_paragraph(task.get("output", ""))
        )
        if summary:
            answer_parts.append(f"{task_id}: {summary}")

        findings = task.get("findings", [])
        if isinstance(findings, list):
            key_findings.extend(findings)

        if status in {
            TaskStatus.ERROR.value,
            TaskStatus.TIMEOUT.value,
            TaskStatus.SKIPPED.value,
        }:
            reason = str(
                task.get("error") or task.get("skip_reason") or task.get("output") or ""
            )
            failed_or_skipped.append(
                {
                    "id": task_id,
                    "status": status,
                    "reason": preview(reason, 300),
                }
            )

        handoff = str(task.get("handoff") or "").strip()
        if handoff:
            next_steps.append(handoff)

    return {
        "answer": "\n".join(answer_parts),
        "key_findings": key_findings,
        "failed_or_skipped_tasks": failed_or_skipped,
        "recommended_next_steps": next_steps,
    }


def _capsule_lock(job_id: str) -> asyncio.Lock:
    safe_job_id = _safe_job_id(job_id)
    lock = _CAPSULE_LOCKS.get(safe_job_id)
    if lock is None:
        lock = asyncio.Lock()
        _CAPSULE_LOCKS[safe_job_id] = lock
    return lock


def _release_capsule_lock(job_id: str) -> None:
    _CAPSULE_LOCKS.pop(_safe_job_id(job_id), None)


def _read_capsule_sync(path: Path, job_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data: Any = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Capsule `{job_id}` is not a JSON object")
    return cast(dict[str, Any], data)


def _initial_task_record(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "goal": task["task"],
        "agent": task.get("agent") or "custom",
        "status": TaskStatus.PENDING.value,
        "depends_on": list(task.get("depends_on", [])),
        "dependency_context": "",
        "tolerate_failures": bool(task.get("tolerate_failures", False)),
        "timeout": task.get("timeout"),
        "model": task.get("model", ""),
        "started_at": None,
        "completed_at": None,
        "duration": 0.0,
        "tokens": {},
        "output": "",
        "summary": "",
        "evidence": default_evidence(),
        "findings": [],
        "handoff": "",
        "confidence": "unknown",
        "error": "",
        "skip_reason": "",
    }


def _get_task(capsule: dict[str, Any], task_id: str) -> dict[str, Any]:
    tasks = capsule.setdefault("tasks", {})
    task = tasks.get(task_id)
    if not isinstance(task, dict):
        raise KeyError(
            f"Task `{task_id}` missing from capsule `{capsule.get('job_id')}`"
        )
    return task


def _parse_delegation_result(output: str) -> dict[str, Any] | None:
    match = _JSON_FENCE_RE.search(output)
    if match:
        try:
            data: Any = json.loads(match.group(1))
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
        except json.JSONDecodeError as exc:
            logger.debug(f"Invalid fenced delegation result ignored: {exc}")

    marker = "DELEGATION_RESULT:"
    index = output.rfind(marker)
    if index >= 0:
        tail = output[index + len(marker) :]
        start = tail.find("{")
        end = tail.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(tail[start : end + 1])
                if isinstance(data, dict):
                    return cast(dict[str, Any], data)
            except json.JSONDecodeError as exc:
                logger.debug(f"Invalid marked delegation result ignored: {exc}")

    return None


def first_compact_paragraph(output: Any, limit: int = 500) -> str:
    text = str(output or "").strip()
    if not text:
        return ""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    summary = paragraphs[0] if paragraphs else text
    summary = " ".join(summary.split())
    return preview(summary, limit)


def default_evidence() -> dict[str, list[Any]]:
    return {key: [] for key in EVIDENCE_KEYS}


def _normalize_evidence(value: Any) -> dict[str, list[Any]]:
    evidence = default_evidence()
    if not isinstance(value, dict):
        return evidence
    for key in EVIDENCE_KEYS:
        item = value.get(key, [])
        evidence[key] = item if isinstance(item, list) else []
    return evidence


def _normalize_confidence(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    normalized = value.lower().strip()
    return normalized if normalized in _CONFIDENCE_VALUES else "unknown"


def _string_or_default(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def status_value(status: Any) -> str:
    if isinstance(status, TaskStatus):
        return status.value
    return str(status)


def preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "... [truncated]"
