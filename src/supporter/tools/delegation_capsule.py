from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ..config import config
from ..logger import logger
from ..types import TaskStatus

CAPSULE_SCHEMA_VERSION = 1
ACTIVE_CAPSULE_STATUSES = {"pending", "running"}
EVIDENCE_KEYS = ("files_read", "files_changed", "commands_run", "sources")
OUTPUT_PREVIEW_CHARS = 1200

_CAPSULE_LOCKS: dict[str, asyncio.Lock] = {}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def delegations_dir() -> Path:
    return Path(config.allowed_directories[0]) / ".supporter" / "delegations"


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
    return capsule


def load_capsule(job_id: str) -> dict[str, Any]:
    path = capsule_path(job_id)
    with path.open(encoding="utf-8") as f:
        data: Any = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Capsule `{job_id}` is not a JSON object at {path}")
    return cast(dict[str, Any], data)


def effective_status(capsule: dict[str, Any]) -> str:
    """Return stale running/pending capsules as interrupted without mutating disk."""
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
        json.dump(capsule, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_path.replace(path)


async def save_capsule(capsule: dict[str, Any]) -> None:
    await asyncio.to_thread(_save_capsule_sync, capsule)


async def update_capsule(
    job_id: str, mutator: Callable[[dict[str, Any]], None]
) -> dict[str, Any] | None:
    path = capsule_path(job_id)
    async with _capsule_lock(job_id):
        if not path.exists():
            logger.warning(f"Delegation capsule missing during update [job={job_id}]")
            return None
        try:
            capsule = await asyncio.to_thread(_read_capsule_sync, path, job_id)
            mutator(capsule)
            capsule["updated_at"] = utc_now()
            await save_capsule(capsule)
            return capsule
        except Exception as exc:
            logger.error(
                f"Delegation capsule update failed "
                f"[job={job_id}, path={path}, error={type(exc).__name__}: {exc}]"
            )
            return None


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
            _status_value(task.get("status", "pending"))
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
            if _status_value(task.get("status", "")) in {
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
            "evidence": _default_evidence(),
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


def build_synthesis(capsule: dict[str, Any]) -> dict[str, Any]:
    key_findings: list[Any] = []
    failed_or_skipped: list[dict[str, str]] = []
    next_steps: list[str] = []
    answer_parts: list[str] = []

    for task in capsule.get("tasks", {}).values():
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id", ""))
        status = _status_value(task.get("status", ""))
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
                    "reason": _preview(reason, 300),
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


def serialize_capsule_result(job_id: str) -> dict[str, Any]:
    capsule = load_capsule(job_id)
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
        "totals": _task_totals(tasks),
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
    """Query delegation capsule state."""
    if not job_id:
        return _list_delegations(status=status, limit=limit)
    if task_id:
        return _inspect_task(job_id, task_id)
    return _inspect_delegation(job_id, detail)


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
        "evidence": _default_evidence(),
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
    match = re.search(r"```json\s*(\{.*?\})\s*```", output, re.DOTALL)
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
    return _preview(summary, limit)


def _default_evidence() -> dict[str, list[Any]]:
    return {key: [] for key in EVIDENCE_KEYS}


def _normalize_evidence(value: Any) -> dict[str, list[Any]]:
    evidence = _default_evidence()
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
    return normalized if normalized in {"low", "medium", "high"} else "unknown"


def _string_or_default(value: Any, default: str) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def _status_value(status: Any) -> str:
    if isinstance(status, TaskStatus):
        return status.value
    return str(status)


def _task_totals(tasks: dict[str, Any]) -> dict[str, int]:
    completed = failed = skipped = timed_out = tokens = 0
    for task in tasks.values():
        if not isinstance(task, dict):
            continue
        status = _status_value(task.get("status", ""))
        if status == TaskStatus.COMPLETED.value:
            completed += 1
        elif status == TaskStatus.ERROR.value:
            failed += 1
        elif status == TaskStatus.SKIPPED.value:
            skipped += 1
        elif status == TaskStatus.TIMEOUT.value:
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


def _preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "... [truncated]"


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _duration(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _list_delegations(status: str | None = None, limit: int = 10) -> str:
    normalized_status = status.strip() if isinstance(status, str) and status else None
    limit = max(1, min(int(limit), 100))
    capsules = _load_all_capsules()
    if normalized_status:
        capsules = [c for c in capsules if effective_status(c) == normalized_status]
    capsules.sort(key=lambda c: str(c.get("updated_at", "")), reverse=True)
    capsules = capsules[:limit]
    if not capsules:
        return "No delegation capsules found."

    rows = ["| Job | Status | Tasks | Updated | Milestone |", "|---|---|---:|---|---|"]
    for capsule in capsules:
        tasks = capsule.get("tasks", {})
        totals = _task_totals(tasks if isinstance(tasks, dict) else {})
        done = totals["completed"]
        total = len(tasks) if isinstance(tasks, dict) else 0
        rows.append(
            "| `{job}` | {status} | {done}/{total} | {updated} | {milestone} |".format(
                job=capsule.get("job_id", ""),
                status=effective_status(capsule),
                done=done,
                total=total,
                updated=capsule.get("updated_at", ""),
                milestone=_escape_table(str(capsule.get("milestone", ""))),
            )
        )
    return "\n".join(rows)


def _inspect_delegation(job_id: str, detail: str = "summary") -> str:
    capsule = _load_or_none(job_id)
    if capsule is None:
        return f"Delegation `{job_id}` was not found."

    detail = detail.lower().strip()
    if detail == "summary":
        return _format_capsule_summary(capsule)
    if detail == "tasks":
        return _format_capsule_tasks(capsule)
    if detail == "full":
        return "```json\n" + json.dumps(_display_capsule(capsule), indent=2) + "\n```"
    return "Unknown detail. Use `summary`, `tasks`, or `full`."


def _inspect_task(job_id: str, task_id: str) -> str:
    capsule = _load_or_none(job_id)
    if capsule is None:
        return f"Delegation `{job_id}` was not found."
    tasks = capsule.get("tasks", {})
    task = tasks.get(task_id) if isinstance(tasks, dict) else None
    if not isinstance(task, dict):
        return f"Task `{task_id}` was not found in delegation `{job_id}`."

    evidence = task.get("evidence", _default_evidence())
    findings = task.get("findings", [])
    summary = first_compact_paragraph(str(task.get("output", "")))
    depends_on = task.get("depends_on", [])
    depends_on_text = ", ".join(depends_on) if isinstance(depends_on, list) else "none"
    lines = [
        f"**Task `{task_id}`**",
        f"- Goal: {task.get('goal', '')}",
        f"- Status: {task.get('status', '')}",
        f"- Depends on: {depends_on_text or 'none'}",
        f"- Summary: {summary}",
        f"- Duration: {_duration(task.get('duration')):.2f}s",
        f"- Model: {task.get('model', '') or 'unknown'}",
    ]
    if task.get("error"):
        lines.append(f"- Error: {task['error']}")
    if task.get("skip_reason"):
        lines.append(f"- Skip reason: {task['skip_reason']}")
    lines.append(f"- Evidence: `{json.dumps(evidence, ensure_ascii=False)}`")
    lines.append(f"- Findings: `{json.dumps(findings, ensure_ascii=False)}`")
    if task.get("output"):
        output_preview = _preview(str(task["output"]), OUTPUT_PREVIEW_CHARS)
        lines.append(f"\nOutput preview:\n\n{output_preview}")
    return "\n".join(lines)


def _load_all_capsules() -> list[dict[str, Any]]:
    capsules: dict[str, dict[str, Any]] = {}
    for path in _capsule_files():
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
    return list(capsules.values())


def _load_or_none(job_id: str) -> dict[str, Any] | None:
    try:
        return load_capsule(job_id)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            f"Failed to inspect delegation capsule "
            f"[job={job_id}, error={type(exc).__name__}: {exc}]"
        )
        return None


def _capsule_files() -> list[Path]:
    root = delegations_dir()
    if not root.exists():
        return []
    return list(root.glob("*.json"))


def _format_capsule_summary(capsule: dict[str, Any]) -> str:
    job_id = str(capsule.get("job_id", ""))
    tasks = capsule.get("tasks", {})
    totals = _task_totals(tasks if isinstance(tasks, dict) else {})
    synthesis = capsule.get("synthesis", {})
    if not isinstance(synthesis, dict):
        synthesis = {}
    return "\n".join(
        [
            f"**Delegation `{job_id}`**",
            f"- Milestone: {capsule.get('milestone', '')}",
            f"- Status: {effective_status(capsule)}",
            f"- Capsule: {capsule_relative_path(job_id)}",
            f"- Totals: {json.dumps(totals)}",
            f"- Answer: {synthesis.get('answer', '') or 'none'}",
            _format_section("Key findings", synthesis.get("key_findings", [])),
            _format_section(
                "Failed or skipped", synthesis.get("failed_or_skipped_tasks", [])
            ),
            _format_section(
                "Recommended next steps", synthesis.get("recommended_next_steps", [])
            ),
        ]
    )


def _format_capsule_tasks(capsule: dict[str, Any]) -> str:
    tasks = capsule.get("tasks", {})
    if not isinstance(tasks, dict) or not tasks:
        return "This delegation has no task records."
    rows = [
        "| Task | Status | Goal | Findings | Evidence | Handoff |",
        "|---|---|---|---:|---:|---|",
    ]
    for task_id, task in tasks.items():
        if not isinstance(task, dict):
            continue
        evidence = task.get("evidence", {})
        evidence_count = (
            sum(len(v) for v in evidence.values() if isinstance(v, list))
            if isinstance(evidence, dict)
            else 0
        )
        findings = task.get("findings", [])
        row_template = (
            "| `{task}` | {status} | {goal} | {findings} | {evidence} | {handoff} |"
        )
        rows.append(
            row_template.format(
                task=task_id,
                status=task.get("status", ""),
                goal=_escape_table(_preview(str(task.get("goal", "")), 120)),
                findings=len(findings) if isinstance(findings, list) else 0,
                evidence=evidence_count,
                handoff=_escape_table(_preview(str(task.get("handoff", "")), 120)),
            )
        )
    return "\n".join(rows)


def _format_section(title: str, value: Any) -> str:
    if not isinstance(value, list) or not value:
        return f"- {title}: none"
    rendered = "; ".join(_preview(_jsonish(item), 300) for item in value)
    return f"- {title}: {rendered}"


def _display_capsule(capsule: dict[str, Any]) -> dict[str, Any]:
    display = deepcopy(capsule)
    tasks = display.get("tasks", {})
    if isinstance(tasks, dict):
        for task in tasks.values():
            if not isinstance(task, dict):
                continue
            if task.get("output"):
                task["output"] = _preview(str(task["output"]), OUTPUT_PREVIEW_CHARS)
            if task.get("dependency_context"):
                task["dependency_context"] = _preview(
                    str(task["dependency_context"]), OUTPUT_PREVIEW_CHARS
                )
    return display


def _jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


__all__ = [
    "build_synthesis",
    "capsule_path",
    "capsule_relative_path",
    "create_capsule",
    "delegations_dir",
    "effective_status",
    "extract_task_capsule_fields",
    "first_compact_paragraph",
    "load_capsule",
    "mark_capsule_cancelled",
    "mark_capsule_completed",
    "mark_task_completed",
    "mark_task_failed",
    "mark_task_skipped",
    "mark_task_started",
    "mark_task_timed_out",
    "query_delegation",
    "save_capsule",
    "serialize_capsule_result",
    "update_capsule",
]
