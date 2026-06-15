from __future__ import annotations

import json
from typing import Any

from ...decision_log import DecisionEntry, recent_decisions
from .capsule import (
    capsule_relative_path,
    default_evidence,
    effective_status,
    load_capsule,
    preview,
)
from .capsule_query import task_totals
from .capsule_render import render_evidence, render_findings

OUTPUT_PREVIEW_CHARS = 3600  # ponytail: raised from 1200 for fuller task output
RECENT_DECISIONS_LIMIT = 20


def list_delegations(status: str | None = None, limit: int = 10) -> str:
    from .capsule_query import load_all_capsules

    normalized_status = status.strip() if isinstance(status, str) and status else None
    limit = max(1, min(int(limit), 100))
    load_limit = None if normalized_status else limit
    capsules = load_all_capsules(limit=load_limit)
    if normalized_status:
        capsules = [c for c in capsules if effective_status(c) == normalized_status]
    capsules.sort(key=lambda c: str(c.get("updated_at", "")), reverse=True)
    capsules = capsules[:limit]
    if not capsules:
        return "No delegation capsules found."

    rows = ["| Job | Status | Tasks | Updated | Milestone |", "|---|---|---:|---|---|"]
    for capsule in capsules:
        tasks = capsule.get("tasks", {})
        totals = task_totals(tasks if isinstance(tasks, dict) else {})
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


def inspect_delegation(job_id: str, detail: str = "summary") -> str:
    try:
        capsule = load_capsule(job_id)
    except (
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        return format_capsule_load_error(job_id, exc)

    detail = detail.lower().strip()
    if detail == "summary":
        return format_capsule_summary(capsule)
    if detail == "tasks":
        return format_capsule_tasks(capsule)
    if detail == "full":
        return "```json\n" + json.dumps(display_capsule(capsule), indent=2) + "\n```"
    if detail == "decisions":
        return format_recent_decisions(recent_decisions(), job_id=job_id)
    return "Unknown detail. Use `summary`, `tasks`, `decisions`, or `full`."


def inspect_task(job_id: str, task_id: str) -> str:
    try:
        capsule = load_capsule(job_id)
    except (
        FileNotFoundError,
        OSError,
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ) as exc:
        return format_capsule_load_error(job_id, exc)

    tasks = capsule.get("tasks", {})
    task = tasks.get(task_id) if isinstance(tasks, dict) else None
    if not isinstance(task, dict):
        return f"Task `{task_id}` was not found in delegation `{job_id}`."

    from .capsule_query import duration

    evidence = task.get("evidence", default_evidence())
    findings = task.get("findings", [])
    summary = task.get("summary") or preview(str(task.get("output", "")), 3000)
    depends_on = task.get("depends_on", [])
    depends_on_text = ", ".join(depends_on) if isinstance(depends_on, list) else "none"
    lines = [
        f"**Task `{task_id}`**",
        f"- Goal: {task.get('goal', '')}",
        f"- Status: {task.get('status', '')}",
        f"- Depends on: {depends_on_text or 'none'}",
        f"- Summary: {summary}",
        f"- Duration: {duration(task.get('duration')):.2f}s",
        f"- Model: {task.get('model', '') or 'unknown'}",
    ]
    if task.get("error"):
        lines.append(f"- Error: {task['error']}")
    if task.get("skip_reason"):
        lines.append(f"- Skip reason: {task['skip_reason']}")
    lines.append(render_evidence(evidence))
    lines.append(render_findings(findings))
    if task.get("output"):
        outputpreview = preview(str(task["output"]), OUTPUT_PREVIEW_CHARS)
        lines.append(f"\nOutput preview:\n\n{outputpreview}")
    return "\n".join(lines)


def format_capsule_load_error(job_id: str, exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError):
        return f"Delegation `{job_id}` was not found."
    return (
        f"Delegation `{job_id}` capsule is unavailable "
        f"({type(exc).__name__}: {exc}). "
        "Start a new delegation or remove the corrupt capsule file."
    )


def format_metrics_line(metrics: dict[str, Any]) -> str:
    """One-line reliability summary: success rate + autonomy duration."""
    success_pct = round(float(metrics.get("success_rate", 0.0)) * 100)
    steps = int(metrics.get("total_steps", 0))
    duration = float(metrics.get("milestone_duration", 0.0))
    completed = int(metrics.get("completed", 0))
    attempted = (
        completed + int(metrics.get("failed", 0)) + int(metrics.get("timed_out", 0))
    )
    parts = [
        f"Success {success_pct}% ({completed}/{attempted})",
        f"{steps} steps",
        f"{duration:.1f}s",
    ]
    cost = metrics.get("cost_usd")
    if cost is not None:
        parts.append(f"${float(cost):.4f}")
    return " · ".join(parts)


def format_recent_decisions(
    entries: list[DecisionEntry], *, job_id: str | None = None
) -> str:
    """Render the autonomous decision ring (site/chosen/reason) for inspection.

    When ``job_id`` is given, only decisions whose correlation_id matches are
    shown so a user can audit what the agent decided for that delegation.
    """
    if job_id:
        entries = [e for e in entries if e.correlation_id == job_id]
    if not entries:
        scope = f" for `{job_id}`" if job_id else ""
        return f"No recorded decisions{scope}."
    shown = entries[-RECENT_DECISIONS_LIMIT:]
    header = f"**Recent decisions** ({len(shown)} of {len(entries)})"
    rows = [header]
    for entry in shown:
        reason = f" — {entry.reason}" if entry.reason else ""
        rows.append(f"- [{entry.timestamp}] {entry.site} → {entry.chosen}{reason}")
    return "\n".join(rows)


def format_capsule_summary(capsule: dict[str, Any]) -> str:
    job_id = str(capsule.get("job_id", ""))
    tasks = capsule.get("tasks", {})
    totals = task_totals(tasks if isinstance(tasks, dict) else {})
    synthesis = capsule.get("synthesis", {})
    if not isinstance(synthesis, dict):
        synthesis = {}
    metrics = capsule.get("metrics")
    metrics_line = (
        f"- Metrics: {format_metrics_line(metrics)}"
        if isinstance(metrics, dict)
        else None
    )
    lines = [
        f"**Delegation `{job_id}`**",
        f"- Milestone: {capsule.get('milestone', '')}",
        f"- Status: {effective_status(capsule)}",
        f"- Capsule: {capsule_relative_path(job_id)}",
        f"- Totals: {json.dumps(totals)}",
    ]
    if metrics_line is not None:
        lines.append(metrics_line)
    return "\n".join(
        [
            *lines,
            f"- Answer: {synthesis.get('answer', '') or 'none'}",
            format_section("Key findings", synthesis.get("key_findings", [])),
            format_section(
                "Failed or skipped", synthesis.get("failed_or_skipped_tasks", [])
            ),
            format_section(
                "Recommended next steps", synthesis.get("recommended_next_steps", [])
            ),
        ]
    )


def format_capsule_tasks(capsule: dict[str, Any]) -> str:
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
                goal=_escape_table(preview(str(task.get("goal", "")), 120)),
                findings=len(findings) if isinstance(findings, list) else 0,
                evidence=evidence_count,
                handoff=_escape_table(preview(str(task.get("handoff", "")), 120)),
            )
        )
    return "\n".join(rows)


def format_section(title: str, value: Any) -> str:
    from .capsule_query import jsonish

    if not isinstance(value, list) or not value:
        return f"- {title}: none"
    rendered = "; ".join(preview(jsonish(item), 300) for item in value)
    return f"- {title}: {rendered}"


def display_capsule(capsule: dict[str, Any]) -> dict[str, Any]:
    display = dict(capsule)
    tasks = capsule.get("tasks", {})
    if isinstance(tasks, dict):
        display_tasks: dict[str, Any] = {}
        for task_id, task in tasks.items():
            if not isinstance(task, dict):
                display_tasks[task_id] = task
                continue
            needs_copy = bool(task.get("output")) or bool(
                task.get("dependency_context")
            )
            if not needs_copy:
                display_tasks[task_id] = task
                continue
            shallow = dict(task)
            if shallow.get("output"):
                shallow["output"] = preview(
                    str(shallow["output"]), OUTPUT_PREVIEW_CHARS
                )
            if shallow.get("dependency_context"):
                shallow["dependency_context"] = preview(
                    str(shallow["dependency_context"]), OUTPUT_PREVIEW_CHARS
                )
            display_tasks[task_id] = shallow
        display["tasks"] = display_tasks
    return display


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
