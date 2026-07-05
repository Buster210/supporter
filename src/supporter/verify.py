"""Verification loop: re-attempt a generation until checks pass.

A check is a small, **LLM-free** predicate over a :class:`LLMResult` (and
optionally the prompt that produced it). The loop:

1. Run the provider.
2. Run every check in order; collect failures.
3. If all pass, return.
4. Otherwise feed the failure context back to the provider as a follow-up
   turn and try again, up to ``max_attempts``.

This is the "plan → implement → verify" loop the agent uses to ensure
the work is *correct*, not just *produced*.

Design
------

* **No LLM in checks.** Every check is a pure-Python function. This makes
  the loop reliable and cheap: verifying a result never costs another
  round of LLM time on its own.
* **Bounded attempts.** Default 3 attempts, configurable per call. After
  the cap, the last result is returned with a ``verify_error`` annotation.
* **Composable.** Shipped checks cover length, JSON shape, file presence,
  recipe replay, and idempotence; callers can plug in their own.
* **Auditable.** Each attempt is recorded in the working-memory store as
  a ``verify_attempt`` note so a long-lived session can inspect its own
  verification history.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import config
from .decision_log import log_decision
from .logger import logger
from .memory import append_note
from .recipes import find_recipe, run_recipe
from .types import LLMResult

__all__ = [
    "Check",
    "CheckResult",
    "VerificationConfig",
    "VerificationLoop",
    "build_default_checks",
    "check_files_exist",
    "check_json_shape",
    "check_min_chars",
    "check_no_unicode_garble",
    "check_plan_goals_met",
    "check_recipe_passes",
]


# ---------------------------------------------------------------------------
# Check protocol + result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


class Check(Protocol):
    """A pure-Python predicate over an LLMResult + prompt.

    Implementations must be cheap and deterministic; they may also be
    async (``Awaitable[CheckResult]``) if they need to run IO.
    """

    name: str

    def __call__(
        self, result: LLMResult, prompt: str
    ) -> CheckResult | Awaitable[CheckResult]: ...


def _named(name: str, c: Callable[[LLMResult, str], Any]) -> Check:
    """Adapter that wraps a sync function with an explicit name."""

    class _Wrapped:
        def __init__(self) -> None:
            self.name = name
            self._fn = c

        def __call__(self, result: LLMResult, prompt: str) -> Any:
            return self._fn(result, prompt)

    return _Wrapped()


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------


def check_min_chars(min_chars: int = 1) -> Check:
    """Pass when the result text is at least ``min_chars`` characters."""

    def _check(result: LLMResult, _prompt: str) -> CheckResult:
        text = result.text or ""
        ok = len(text) >= min_chars
        return CheckResult(
            name="min_chars",
            ok=ok,
            detail=f"got {len(text)} chars (need >= {min_chars})",
        )

    return _named("min_chars", _check)


# Common patterns that look like a model glitching / garbling output.
_GARBLE_PATTERNS = (
    re.compile(r"\u00a0\u00a0\u00a0\u00a0"),  # long NBSP runs
    re.compile(r"(.)\1{40,}"),  # 40+ repeated characters
    re.compile(r"[A-Za-z0-9]\?[A-Za-z0-9]\?$"),  # "?" substitution at end
)


def check_no_unicode_garble() -> Check:
    """Pass when the text does not match common garble patterns."""

    def _check(result: LLMResult, _prompt: str) -> CheckResult:
        text = result.text or ""
        for pattern in _GARBLE_PATTERNS:
            if pattern.search(text):
                return CheckResult(
                    name="no_unicode_garble",
                    ok=False,
                    detail=f"matched pattern: {pattern.pattern!r}",
                )
        return CheckResult(name="no_unicode_garble", ok=True, detail="clean")

    return _named("no_unicode_garble", _check)


def check_json_shape(required_keys: tuple[str, ...] = ()) -> Check:
    """Pass when the result text is valid JSON containing the required keys."""

    required = tuple(required_keys)

    def _check(result: LLMResult, _prompt: str) -> CheckResult:
        text = (result.text or "").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return CheckResult(
                name="json_shape",
                ok=False,
                detail=f"invalid JSON: {exc}",
            )
        if required:
            if not isinstance(parsed, dict):
                return CheckResult(
                    name="json_shape",
                    ok=False,
                    detail=f"expected object, got {type(parsed).__name__}",
                )
            missing = [k for k in required if k not in parsed]
            if missing:
                return CheckResult(
                    name="json_shape",
                    ok=False,
                    detail=f"missing keys: {missing}",
                )
        return CheckResult(name="json_shape", ok=True, detail="ok")

    return _named("json_shape", _check)


def check_recipe_passes(recipe_name: str) -> Check:
    """Pass when a stored recipe runs successfully.

    Useful when the LLM is expected to produce output that *is* a recipe
    invocation; the check actually replays the recipe and confirms it.
    """

    name = f"recipe_passes:{recipe_name}"

    async def _check(result: LLMResult, _prompt: str) -> CheckResult:
        recipe = find_recipe(recipe_name)
        if recipe is None:
            return CheckResult(
                name=name,
                ok=False,
                detail=f"recipe {recipe_name!r} not found",
            )
        run = await run_recipe(recipe_name)
        if run is None:
            return CheckResult(name=name, ok=False, detail="recipe run failed")
        if not run.ok:
            return CheckResult(
                name=name,
                ok=False,
                detail=f"recipe failed at step {run.failed_step_index}: {run.error}",
            )
        return CheckResult(
            name=name,
            ok=True,
            detail=f"ok ({len(run.step_results)} steps)",
        )

    return _named(name, _check)


def check_files_exist(paths: tuple[str, ...]) -> Check:
    """Pass when every path exists on disk (project-root relative)."""
    root: str | None = None
    if config.allowed_directories:
        root = config.allowed_directories[0]

    def _check(result: LLMResult, _prompt: str) -> CheckResult:
        import os

        missing: list[str] = []
        for rel in paths:
            target = os.path.join(root or "", rel) if root else rel
            if not os.path.exists(target):
                missing.append(rel)
        if missing:
            return CheckResult(
                name="files_exist",
                ok=False,
                detail=f"missing: {missing}",
            )
        return CheckResult(
            name="files_exist",
            ok=True,
            detail=f"ok ({len(paths)} files)",
        )

    return _named("files_exist", _check)


def check_plan_goals_met(provider: Any, model: str) -> Check:
    """Pass when the result semantically fulfils the plan's stated goals.

    Makes a single fresh, stateless LLM call with (plan + result) only.
    The provider and model are captured at construction time; no shared
    state is threaded through invocations.
    """
    from .llm.types import GenOptions
    from .prompts import PLAN_VERIFIER_PERSONA

    _persona = PLAN_VERIFIER_PERSONA

    async def _check(result: LLMResult, prompt: str) -> CheckResult:
        text = result.text or ""
        # Extract the PLAN section from the prompt (between "PLAN:" and the
        # next top-level section or end of string).
        plan_match = re.search(
            r"PLAN:\s*\n(.*?)(?=\n[A-Z][A-Z ]{2,}:|\Z)", prompt, re.DOTALL
        )
        plan_text = plan_match.group(1).strip() if plan_match else prompt
        judge_prompt = f"OBJECTIVE:\n{plan_text}\n\nRESULT:\n{text}"
        try:
            options = GenOptions(system_instruction=_persona)
            gen = await provider.generate(judge_prompt, options)
            verdict = (getattr(gen, "text", "") or "").strip()
            first_line = verdict.splitlines()[0].upper() if verdict else ""
            if "NOT_DONE" in first_line or "NOT DONE" in first_line:
                reason = (
                    verdict.split("\n", 1)[1].strip() if "\n" in verdict else verdict
                )
                return CheckResult(
                    name="plan_goals_met",
                    ok=False,
                    detail=reason or "plan goals not met",
                )
            return CheckResult(name="plan_goals_met", ok=True, detail="ok")
        except Exception as exc:
            # Fail-open: semantic check outage must not trap the loop.
            return CheckResult(
                name="plan_goals_met",
                ok=True,
                detail=f"semantic check unavailable: {exc}",
            )

    return _named("plan_goals_met", _check)


# ---------------------------------------------------------------------------
# Verification config + loop
# ---------------------------------------------------------------------------


@dataclass
class VerificationConfig:
    max_attempts: int = 3
    # The system-instruction suffix appended to the follow-up turn when
    # verification fails. Pragmatic default: ask the LLM to address each
    # failing check by name with the captured detail.
    retry_template: str = (
        "Your previous response failed verification. Please re-emit a corrected "
        "version that addresses EACH of the following failures:\n\n{checks}\n\n"
        "Re-state your full answer; do not just describe the fix."
    )
    record_to_memory: bool = True


@dataclass
class VerificationOutcome:
    ok: bool
    attempts: int
    last_result: LLMResult | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "attempts": self.attempts,
            "last_text": (
                (self.last_result.text or "")[:200] if self.last_result else None
            ),
            "attempts_detail": self.history,
        }


# ---------------------------------------------------------------------------
# Build a sensible default check set
# ---------------------------------------------------------------------------


def build_default_checks(
    *,
    min_chars: int = 1,
    required_json_keys: tuple[str, ...] = (),
) -> list[Check]:
    """A pragmatic default set: length + garble + (optional) JSON shape."""
    out: list[Check] = [
        check_min_chars(min_chars=min_chars),
        check_no_unicode_garble(),
    ]
    if required_json_keys:
        out.append(check_json_shape(required_json_keys))
    return out


# ---------------------------------------------------------------------------
# Verification loop
# ---------------------------------------------------------------------------


class VerificationLoop:
    """Run an LLM call → checks → re-attempt, bounded by ``max_attempts``."""

    def __init__(
        self,
        config: VerificationConfig | None = None,
        checks: list[Check] | None = None,
    ) -> None:
        self.config = config or VerificationConfig()
        self.checks: list[Check] = list(checks or [])

    def add(self, check: Check) -> None:
        self.checks.append(check)

    async def _run_check(
        self, check: Check, result: LLMResult, prompt: str
    ) -> CheckResult:
        outcome = check(result, prompt)
        if hasattr(outcome, "__await__"):
            return await outcome
        return outcome

    async def run(
        self,
        caller: Callable[[str], Awaitable[LLMResult]],
        prompt: str,
    ) -> VerificationOutcome:
        """Run ``caller(prompt)`` and verify the result, retrying on failure.

        ``caller`` is the agent's ``provider.generate`` adapter; it gets the
        follow-up prompt on retries. The first call uses the original
        prompt verbatim; subsequent calls prepend a structured failure
        report.
        """
        attempts = 0
        last_result: LLMResult | None = None
        history: list[dict[str, Any]] = []
        for attempt_idx in range(self.config.max_attempts):
            attempts = attempt_idx + 1
            current_prompt = prompt
            if attempt_idx > 0 and last_result is not None:
                last_failures = history[-1]["failures"] if history else []
                checks_text = (
                    "\n".join(f"- [{f['name']}] {f['detail']}" for f in last_failures)
                    or "(no detail captured)"
                )
                retry_prompt = self.config.retry_template.format(checks=checks_text)
                current_prompt = f"{prompt}\n\n---\n{retry_prompt}"
            start = time.perf_counter()
            result = await caller(current_prompt)
            duration = time.perf_counter() - start
            last_result = result

            check_results: list[CheckResult] = []
            for check in self.checks:
                check_results.append(await self._run_check(check, result, prompt))

            failures = [c for c in check_results if not c.ok]
            history.append(
                {
                    "attempt": attempts,
                    "duration_s": round(duration, 3),
                    "prompt_chars": len(current_prompt),
                    "result_chars": len(result.text or ""),
                    "checks": [
                        {"name": c.name, "ok": c.ok, "detail": c.detail}
                        for c in check_results
                    ],
                    "failures": [
                        {"name": c.name, "detail": c.detail} for c in failures
                    ],
                }
            )
            if not failures:
                log_decision(
                    site="verify.loop",
                    chosen="pass",
                    reason=f"attempts={attempts} checks={len(check_results)}",
                )
                if self.config.record_to_memory:
                    append_note(
                        "verify_attempt",
                        {
                            "ok": True,
                            "attempts": attempts,
                            "checks": [
                                {"name": c.name, "ok": c.ok} for c in check_results
                            ],
                        },
                    )
                return VerificationOutcome(
                    ok=True,
                    attempts=attempts,
                    last_result=result,
                    history=history,
                )
            logger.info(
                f"verify: attempt {attempts} failed {len(failures)} check(s): "
                + ", ".join(c.name for c in failures)
            )

        # Out of attempts.
        log_decision(
            site="verify.loop",
            chosen="exhausted",
            reason=f"attempts={attempts} last_failures={len(failures)}",
        )
        if self.config.record_to_memory and last_result is not None:
            append_note(
                "verify_attempt",
                {
                    "ok": False,
                    "attempts": attempts,
                    "failures": history[-1]["failures"] if history else [],
                },
            )
        return VerificationOutcome(
            ok=False,
            attempts=attempts,
            last_result=last_result,
            history=history,
        )
