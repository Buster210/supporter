"""Two-tier QA gate over delegated opencode coding work (SPEC §7).

After opencode writes code, the harness does not trust it blindly:

- **Tier 1** — a *fresh* opencode worker builds/tests/lints the change and
  reports a pass/fail verdict (ground-truth signals, not LLM judgment alone).
- **Tier 2** — native Gemini roster sub-agents (test, review, security) verify
  the change in parallel and each return an approve/reject verdict.

On any failure the gate feeds the diagnosis back to a fresh opencode worker and
re-runs, up to ``delegate_correction_rounds`` correction rounds. Work that never
passes both tiers is returned as ERROR with a diagnosis — never as COMPLETED.
The gate fires only for ``backend == "opencode"`` tasks and is a no-op
otherwise, so existing flows are unaffected.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from ...config import config
from ...logger import logger
from ...types import TaskStatus
from .agents import run_sub_agent
from .backends import GEMINI_BACKEND, OPENCODE_BACKEND, QA_REJECTION_MARKER
from .bus import DelegationBus
from .opencode_backend import _resolve_repo
from .tier1_checks import (
    Tier1ToolUnavailable,
    resolve_tier1_commands,
    run_objective_tier1,
)

_TIER1_TOKEN = "qa-tier1:"  # noqa: S105  # nosec B105 - verdict marker, not a secret
_TIER2_TOKEN = "qa-verdict:"  # noqa: S105  # nosec B105 - verdict marker, not a secret

_TIER2_ROLES = ("test_engineer", "code_reviewer", "security_auditor")


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

    Honors ``DELEGATE_TIER1_OBJECTIVE=0`` to force the LLM path (used in
    tests and for emergency rollback). Otherwise resolves a list of argv
    from config or auto-detection, runs them in-process with REAL exit
    codes, and falls back to the LLM worker on ``Tier1ToolUnavailable``
    (e.g. when a required tool is outside the sandbox bind-mount).
    """
    if os.getenv("DELEGATE_TIER1_OBJECTIVE", "1") == "0":
        return await _tier1_llm(base, attempt, semaphore, bus, job_id)

    repo = Path(_resolve_repo())
    commands = resolve_tier1_commands(repo)
    if commands:
        try:
            return await run_objective_tier1(repo, commands, base["timeout"])
        except Tier1ToolUnavailable as exc:
            logger.warning(f"objective tier-1 unavailable, falling back to LLM: {exc}")

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
    instructions = (
        base["task"]
        + "\n\nThe previous attempt FAILED verification:\n"
        + feedback
        + "\n\nFix the issues with minimal, surgical changes. Do not introduce "
        "unrelated edits."
    )
    task = _make_task(
        base, f"fix_{attempt}", instructions, backend=OPENCODE_BACKEND, agent=None
    )
    return await _run(task, semaphore, bus, job_id)


async def run_qa_gate(
    task: dict[str, Any],
    result: dict[str, Any],
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
) -> dict[str, Any]:
    """Gate an opencode task result through tier-1 and tier-2 verification.

    Returns the result dict: COMPLETED with a QA note appended on approval, or
    ERROR with a diagnosis if both tiers cannot be satisfied within the
    configured correction rounds.
    """
    if (
        not config.delegate_qa_gate_enabled
        or task.get("backend") != OPENCODE_BACKEND
        or result.get("status") != TaskStatus.COMPLETED
    ):
        return result

    rounds = config.delegate_correction_rounds
    task_id = task["id"]
    last_reason = "verification did not pass"

    for attempt in range(rounds + 1):
        tier1_ok, tier1_output = await _tier1(task, attempt, semaphore, bus, job_id)
        if not tier1_ok:
            last_reason = f"tier-1 tests failing:\n{tier1_output[-500:]}"
        else:
            approved, rejections = await _tier2(
                task, attempt, tier1_output, semaphore, bus, job_id
            )
            if approved:
                logger.info(f"QA gate: task '{task_id}' approved (round {attempt})")
                result["output"] += "\n\n[QA gate: tier-1 + tier-2 PASSED]"
                return result
            last_reason = f"tier-2 rejected: {rejections}"

        if attempt >= rounds:
            break
        logger.info(
            f"QA gate: task '{task_id}' correction round {attempt + 1}/{rounds}"
        )
        fixed, fix_output = await _correct(
            task, attempt, last_reason, semaphore, bus, job_id
        )
        if not fixed:
            last_reason = (
                f"correction round {attempt + 1} did not complete:\n{fix_output[-500:]}"
            )
            logger.warning(
                f"QA gate: task '{task_id}' correction round {attempt + 1} failed"
            )
            break

    logger.warning(f"QA gate: task '{task_id}' rejected after {rounds} rounds")
    result["status"] = TaskStatus.ERROR
    result["output"] = (
        f"{QA_REJECTION_MARKER} after {rounds} correction rounds. {last_reason}"
    )
    return result
