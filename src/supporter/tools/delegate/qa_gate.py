"""Two-tier QA gate over delegated tasks (SPEC §7).

After a task completes, the harness does not trust it blindly:

- **Tier 1** — a *fresh* opencode worker builds/tests/lints the change and
  reports a pass/fail verdict (ground-truth signals, not LLM judgment alone).
- **Tier 2** — native Gemini roster sub-agents (test, review, security) verify
  the change in parallel and each return an approve/reject verdict.

On any failure the gate feeds the diagnosis back and re-runs, up to
``delegate_correction_rounds`` correction rounds. Work that never passes both
tiers is returned as ERROR with a diagnosis — never as COMPLETED.

For gemini backend tasks, verification is role-aware: finder/research roles
validate the structured payload output with confidence requirements; all other
roles (page-pilot, custom, planner, ...) are judged against their actual output
and task spec via ``_output_verify`` (no git-diff assumptions). Only the
opencode backend uses the git-diff tier-1/tier-2 path above.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from ...config import config
from ...logger import logger
from ...types import SubtaskVerificationResult, TaskStatus, VerificationVerdict
from .agents import run_sub_agent
from .backends import GEMINI_BACKEND, OPENCODE_BACKEND
from .bus import DelegationBus
from .capsule import validate_delegation_payload
from .opencode_backend import _resolve_repo
from .tier1_checks import (
    Tier1ToolUnavailable,
    resolve_tier1_commands,
    run_objective_tier1,
)

_TIER1_TOKEN = "qa-tier1:"  # noqa: S105  # nosec B105 - verdict marker, not a secret
_TIER2_TOKEN = "qa-verdict:"  # noqa: S105  # nosec B105 - verdict marker, not a secret

_TIER2_ROLES = ("test_engineer", "code_reviewer", "security_auditor")

_FINDING_ROLES = ("explorer", "security_auditor", "code_reviewer")

_JSON_FENCE_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def _gemini_predicate_failure(output: str, agent: str | None) -> str | None:
    """Check gemini output quality; return failure reason or None.

    Validates:
    - Structured payload is valid (validate_delegation_payload)
    - Confidence >= delegate_min_confidence
    - For finding-roles: non-empty findings OR evidence.sources

    Note: emptiness is never grounds for rejection -- this only checks
    structural validity, not whether the answer is correct.
    """
    if not validate_delegation_payload(output):
        return "invalid payload"

    parsed = _parse_delegation_result(output)
    if parsed is None:
        return "unparseable result"

    confidence = parsed.get("confidence", "unknown")
    confidence_rank = {"low": 1, "medium": 2, "high": 3, "unknown": 0}
    min_rank = confidence_rank.get(config.delegate_min_confidence, 2)
    output_rank = confidence_rank.get(confidence, 0)
    if output_rank < min_rank:
        return (
            f"confidence '{confidence}' below minimum "
            f"'{config.delegate_min_confidence}'"
        )

    if agent in _FINDING_ROLES:
        findings = parsed.get("findings", [])
        evidence = parsed.get("evidence", {})
        sources = evidence.get("sources", [])
        if not findings and not sources:
            return f"role '{agent}' requires non-empty findings or evidence.sources"

    return None


def _gemini_predicate_passed(output: str, agent: str | None) -> bool:
    """Check if gemini output meets the quality predicate."""
    return _gemini_predicate_failure(output, agent) is None


def _parse_delegation_result(output: str) -> dict[str, Any] | None:
    """Parse the structured delegation result from output text."""
    match = _JSON_FENCE_RE.search(output)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    marker = "DELEGATION_RESULT:"
    idx = output.rfind(marker)
    if idx >= 0:
        tail = output[idx + len(marker) :]
        start = tail.find("{")
        end = tail.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(tail[start : end + 1])
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    return None


def _verdict_passed(output: str, token: str, positive: str) -> bool:
    """Parse a trailing ``<token> <verdict>`` marker.

    Defaults to *not passed* when the marker is absent or ambiguous, so the gate
    never approves red work it could not positively confirm (SPEC §7).
    """
    lowered = output.lower()
    idx = lowered.rfind(token)
    if idx == -1:
        return False
    tail = lowered[idx + len(token) : idx + len(token) + 16]
    return positive in tail


def _make_task(
    base: dict[str, Any],
    suffix: str,
    instructions: str,
    *,
    backend: str,
    agent: str | None,
) -> dict[str, Any]:
    if agent and agent in config.delegate_agent_roster:
        profile = config.delegate_agent_roster[agent]
        persona, tools = profile["persona"], set(profile["tools"])
        model, live = profile["model"], bool(profile.get("live", False))
    else:
        persona, tools = config.delegate_default_persona, set()
        model, live = config.gemini_model, False
    return {
        "id": f"{base['id']}__{suffix}",
        "task": instructions,
        "agent": agent,
        "backend": backend,
        "persona": persona,
        "tools": tools,
        "model": model,
        "live": live,
        "context": base.get("context", ""),
        "timeout": base["timeout"],
        "max_retries": 0,
        "depends_on": [],
        "pre_approved_commands": [],
        "tolerate_failures": False,
        "result_contract": False,
    }


async def _run(
    task: dict[str, Any], semaphore: asyncio.Semaphore, bus: DelegationBus, job_id: str
) -> tuple[bool, str]:
    result = await run_sub_agent(task, semaphore, bus, job_id)
    ok = result.get("status") == TaskStatus.COMPLETED
    return ok, str(result.get("output", ""))


async def _tier1_llm(
    base: dict[str, Any],
    attempt: int,
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
) -> tuple[bool, str]:
    """LLM-driven tier-1 fallback (SPEC §7, zero-regression path).

    The previous behavior: a fresh opencode worker inspects the diff and
    self-reports ``QA-TIER1: PASS|FAIL``. Preserved byte-identically so the
    objective dispatcher below can fall back to it whenever the configured
    commands are not runnable.
    """
    instructions = (
        "A previous worker changed code in this repository. Verify the change is "
        "correct WITHOUT modifying any files. Run `git diff --name-only` to see "
        "what changed, then run the project's build, type-check, lint, and tests "
        "for the affected code. Report failures verbatim. End your reply with "
        "exactly one line: `QA-TIER1: PASS` if everything is green, otherwise "
        "`QA-TIER1: FAIL`.\n\nORIGINAL TASK:\n" + base["task"]
    )
    task = _make_task(
        base, f"tier1_{attempt}", instructions, backend=OPENCODE_BACKEND, agent=None
    )
    ran, output = await _run(task, semaphore, bus, job_id)
    return ran and _verdict_passed(output, _TIER1_TOKEN, "pass"), output


async def _tier1(
    base: dict[str, Any],
    attempt: int,
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
) -> tuple[bool, str]:
    """Dispatch tier-1 to the objective harness or fall back to the LLM worker.

    Always objective-first (no bypass): resolves a list of argv from config or
    auto-detection and runs them in-process with REAL exit codes a sub-agent
    cannot spoof. There is no off-switch. The LLM self-report fallback is a
    last resort fired only when objective verification is impossible — either
    no build commands are detectable (``resolve_tier1_commands`` -> []) or a
    required tool is outside the sandbox (``Tier1ToolUnavailable``). Both cases
    log a WARNING so the (weaker, spoofable) downgrade is never silent.
    """
    repo = Path(_resolve_repo())
    commands = resolve_tier1_commands(repo)
    if commands:
        try:
            return await run_objective_tier1(repo, commands, base["timeout"])
        except Tier1ToolUnavailable as exc:
            logger.warning(f"objective tier-1 unavailable, falling back to LLM: {exc}")
    else:
        logger.warning(
            "objective tier-1 has no detectable build commands for "
            f"{repo}; falling back to spoofable LLM self-report"
        )

    return await _tier1_llm(base, attempt, semaphore, bus, job_id)


async def _tier2(
    base: dict[str, Any],
    attempt: int,
    tier1_output: str,
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
) -> tuple[bool, str]:
    instructions = (
        "Verify a delegated code change for the task below. Inspect the changed "
        "files (git diff, read_file). Approve only if the change is correct, "
        "safe, and complete for your specialty. End your reply with exactly one "
        "line: `QA-VERDICT: APPROVE` or `QA-VERDICT: REJECT`, followed by a "
        "one-line reason on rejection.\n\nORIGINAL TASK:\n"
        + base["task"]
        + "\n\nTIER-1 TEST REPORT:\n"
        + tier1_output
    )

    async def _judge(role: str) -> tuple[str, bool, str]:
        task = _make_task(
            base,
            f"tier2_{role}_{attempt}",
            instructions,
            backend=GEMINI_BACKEND,
            agent=role,
        )
        ran, output = await _run(task, semaphore, bus, job_id)
        return role, ran and _verdict_passed(output, _TIER2_TOKEN, "approve"), output

    verdicts = await asyncio.gather(*(_judge(role) for role in _TIER2_ROLES))
    approved = all(ok for _, ok, _ in verdicts)
    rejections = "; ".join(
        f"{role}: {out[-200:]}" for role, ok, out in verdicts if not ok
    )
    return approved, rejections


async def _correct(
    base: dict[str, Any],
    attempt: int,
    feedback: str,
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
) -> tuple[bool, str]:
    backend = base.get("backend", OPENCODE_BACKEND)
    if backend == GEMINI_BACKEND:
        correction_agent = base.get("agent")
        instructions = (
            base["task"]
            + "\n\nThe previous attempt FAILED verification:\n"
            + feedback
            + "\n\nImprove your findings by grounding them in concrete "
            "evidence and sources. Report honest confidence. Return ONLY "
            "the structured result block in the same format as before -- "
            "no other prose."
        )
    else:
        correction_agent = None
        instructions = (
            base["task"]
            + "\n\nThe previous attempt FAILED verification:\n"
            + feedback
            + "\n\nFix the issues with minimal, surgical changes. Do not introduce "
            "unrelated edits."
        )
    task = _make_task(
        base,
        f"fix_{attempt}",
        instructions,
        backend=backend,
        agent=correction_agent,
    )
    return await _run(task, semaphore, bus, job_id)


async def _output_verify(
    base: dict[str, Any],
    attempt: int,
    output: str,
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
) -> tuple[bool, str]:
    """Verify a non-code subtask (browser/general/no-role) against its own output.

    No git assumptions: hands the verifier the task spec plus the produced
    output text and asks whether it satisfies the task.
    """
    instructions = (
        "A previous worker completed a delegated task that does NOT produce a "
        "git diff (e.g. browser automation, research, or a general task). "
        "Judge ONLY from the task spec and the output below -- do not run "
        "`git diff` or assume any files changed. End your reply with exactly "
        "one line: `QA-VERDICT: APPROVE` if the output satisfies the task, "
        "otherwise `QA-VERDICT: REJECT`, followed by a one-line reason on "
        "rejection.\n\nORIGINAL TASK:\n" + base["task"] + "\n\nOUTPUT:\n" + output
    )
    task = _make_task(
        base,
        f"output_verify_{attempt}",
        instructions,
        backend=GEMINI_BACKEND,
        agent=None,
    )
    ran, judge_output = await _run(task, semaphore, bus, job_id)
    return ran and _verdict_passed(judge_output, _TIER2_TOKEN, "approve"), judge_output


async def run_qa_gate_verify_only(
    task: dict[str, Any],
    result: dict[str, Any],
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    attempt: int = 0,
) -> SubtaskVerificationResult:
    """Correction is deferred to post-plan verification during DAG execution."""
    task_id = task.get("id", "unknown")
    if result.get("status") != TaskStatus.COMPLETED:
        return SubtaskVerificationResult(
            task_id=task_id,
            passed=False,
            reason=f"task status: {result.get('status', 'unknown')}",
        )

    backend = task.get("backend")
    if backend is None:
        return SubtaskVerificationResult(task_id=task_id, passed=True)

    if backend == OPENCODE_BACKEND:
        tier1_ok, tier1_output = await _tier1(task, attempt, semaphore, bus, job_id)
        if not tier1_ok:
            reason = f"tier-1 tests failing:\n{tier1_output[-500:]}"
            bus.publish(
                VerificationVerdict(
                    job_id=job_id,
                    task_id=task_id,
                    scope="task",
                    passed=False,
                    reason=reason,
                    round=attempt,
                )
            )
            return SubtaskVerificationResult(
                task_id=task_id, passed=False, reason=reason
            )

        approved, rejections = await _tier2(
            task, attempt, tier1_output, semaphore, bus, job_id
        )
        if approved:
            bus.publish(
                VerificationVerdict(
                    job_id=job_id,
                    task_id=task_id,
                    scope="task",
                    passed=True,
                    round=attempt,
                )
            )
            return SubtaskVerificationResult(
                task_id=task_id, passed=True, marker="[QA gate: tier-1 + tier-2 PASSED]"
            )

        reason = f"tier-2 rejected: {rejections}"
        bus.publish(
            VerificationVerdict(
                job_id=job_id,
                task_id=task_id,
                scope="task",
                passed=False,
                reason=reason,
                round=attempt,
            )
        )
        return SubtaskVerificationResult(task_id=task_id, passed=False, reason=reason)

    agent = task.get("agent")
    output = result.get("output", "")

    if config.delegate_persist_noncode and agent in _FINDING_ROLES:
        failure = _gemini_predicate_failure(output, agent)
        if failure is None:
            return SubtaskVerificationResult(
                task_id=task_id,
                passed=True,
                marker="[QA gate: gemini predicate PASSED]",
            )
        reason = f"gemini output failed: {failure}"
        return SubtaskVerificationResult(task_id=task_id, passed=False, reason=reason)

    approved, judge_output = await _output_verify(
        task, attempt, output, semaphore, bus, job_id
    )
    if approved:
        bus.publish(
            VerificationVerdict(
                job_id=job_id, task_id=task_id, scope="task", passed=True, round=attempt
            )
        )
        return SubtaskVerificationResult(
            task_id=task_id, passed=True, marker="[QA gate: output verification PASSED]"
        )

    reason = f"output verification rejected: {judge_output[-300:]}"
    bus.publish(
        VerificationVerdict(
            job_id=job_id,
            task_id=task_id,
            scope="task",
            passed=False,
            reason=reason,
            round=attempt,
        )
    )
    return SubtaskVerificationResult(task_id=task_id, passed=False, reason=reason)
