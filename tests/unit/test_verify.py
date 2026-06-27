"""Unit tests for the verification loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.types import LLMResult
from supporter.verify import (
    CheckResult,
    VerificationConfig,
    VerificationLoop,
    _named,
    build_default_checks,
    check_files_exist,
    check_json_shape,
    check_min_chars,
    check_no_unicode_garble,
    check_recipe_passes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(text: str = "ok", model: str = "test-model") -> LLMResult:
    return LLMResult(text=text, model=model, duration=0.01)


def _sync_check(name: str, ok: bool, detail: str = "") -> Any:
    """A check stub that always returns the same verdict."""

    class _Stub:
        def __init__(self) -> None:
            self.name = name

        def __call__(self, result: LLMResult, prompt: str) -> CheckResult:
            return CheckResult(name=name, ok=ok, detail=detail)

    return _Stub()


async def _async_check(name: str, ok: bool, detail: str = "") -> Any:
    class _Stub:
        def __init__(self) -> None:
            self.name = name

        async def __call__(self, result: LLMResult, prompt: str) -> CheckResult:
            return CheckResult(name=name, ok=ok, detail=detail)

    return _Stub()


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------


def test_check_min_chars_passes() -> None:
    c = check_min_chars(min_chars=3)
    assert c(_result("hello"), "p").ok is True  # type: ignore[union-attr]
    assert c(_result("hi"), "p").ok is False  # type: ignore[union-attr]


def test_check_no_unicode_garble_passes_normal_text() -> None:
    c = check_no_unicode_garble()
    assert c(_result("Hello world, this is clean."), "p").ok is True  # type: ignore[union-attr]


def test_check_no_unicode_garble_flags_repeat_run() -> None:
    c = check_no_unicode_garble()
    bad = "x" * 50
    assert c(_result(bad), "p").ok is False  # type: ignore[union-attr]


def test_check_no_unicode_garble_flags_nbsp_run() -> None:
    c = check_no_unicode_garble()
    assert c(_result("hello\u00a0\u00a0\u00a0\u00a0world"), "p").ok is False  # type: ignore[union-attr]


def test_check_json_shape_passes_with_required_keys() -> None:
    c = check_json_shape(required_keys=("summary", "ok"))
    payload = json.dumps({"summary": "x", "ok": True})
    assert c(_result(payload), "p").ok is True  # type: ignore[union-attr]


def test_check_json_shape_fails_on_invalid() -> None:
    c = check_json_shape()
    assert c(_result("not json"), "p").ok is False  # type: ignore[union-attr]


def test_check_json_shape_fails_on_missing_keys() -> None:
    c = check_json_shape(required_keys=("summary",))
    assert c(_result('{"x": 1}'), "p").ok is False  # type: ignore[union-attr]


def test_check_json_shape_fails_on_non_object() -> None:
    c = check_json_shape(required_keys=("x",))
    assert c(_result("[1,2,3]"), "p").ok is False  # type: ignore[union-attr]


def test_check_files_exist_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import verify as verify_mod

    monkeypatch.setattr(verify_mod.config, "allowed_directories", [str(tmp_path)])  # type: ignore[attr-defined]
    (tmp_path / "present.txt").touch()
    c = check_files_exist(paths=("present.txt", "missing.txt"))
    res = c(_result(""), "p")
    assert res.ok is False  # type: ignore[union-attr]
    assert "missing" in res.detail  # type: ignore[union-attr]


def test_check_files_exist_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import verify as verify_mod

    monkeypatch.setattr(verify_mod.config, "allowed_directories", [str(tmp_path)])  # type: ignore[attr-defined]
    (tmp_path / "present.txt").touch()
    c = check_files_exist(paths=("present.txt",))
    res = c(_result(""), "p")
    assert res.ok is True  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_check_recipe_passes_runs_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    from supporter import recipes as recipes_mod

    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    recipes_mod._STORE = None
    recipes_mod.save_recipe(
        "demo",
        "d",
        [{"kind": "emit", "value": "hi"}],
    )
    c = check_recipe_passes("demo")
    res = await c(_result(""), "p")  # type: ignore[misc]
    assert res.ok is True
    assert "steps" in res.detail


@pytest.mark.asyncio
async def test_check_recipe_passes_missing_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import recipes as recipes_mod

    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    recipes_mod._STORE = None
    c = check_recipe_passes("nope")
    res = await c(_result(""), "p")  # type: ignore[misc]
    assert res.ok is False


@pytest.mark.asyncio
async def test_check_recipe_passes_failing_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import recipes as recipes_mod

    monkeypatch.setattr(recipes_mod, "_recipe_path", lambda: tmp_path / "r.jsonl")
    recipes_mod._STORE = None
    recipes_mod.save_recipe(
        "failing",
        "d",
        [{"kind": "assert_eq", "value": "a||b"}],
    )
    c = check_recipe_passes("failing")
    res = await c(_result(""), "p")  # type: ignore[misc]
    assert res.ok is False
    assert "failed" in res.detail


# ---------------------------------------------------------------------------
# Default check builder
# ---------------------------------------------------------------------------


def test_build_default_checks_minimum() -> None:
    checks = build_default_checks()
    assert len(checks) == 2
    assert checks[0].name == "min_chars"
    assert checks[1].name == "no_unicode_garble"


def test_build_default_checks_with_json() -> None:
    checks = build_default_checks(required_json_keys=("a", "b"))
    assert len(checks) == 3


# ---------------------------------------------------------------------------
# VerificationLoop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_passes_first_try() -> None:
    loop = VerificationLoop(
        VerificationConfig(max_attempts=3),
        [_sync_check("ok", True)],
    )
    caller = AsyncMock(return_value=_result("first"))
    outcome = await loop.run(caller, "do thing")
    assert outcome.ok
    assert outcome.attempts == 1
    assert caller.await_count == 1
    assert outcome.history[0]["result_chars"] == 5  # len("first")


@pytest.mark.asyncio
async def test_loop_retries_on_failure_then_passes() -> None:
    """First call fails, second call passes — outcome should be ok=True."""
    call_count = {"n": 0}

    def _flippable(result: LLMResult, _prompt: str) -> CheckResult:
        call_count["n"] += 1
        ok = call_count["n"] >= 2
        return CheckResult(
            name="flippable",
            ok=ok,
            detail="first time" if not ok else "ok",
        )

    chk = _named("flippable", _flippable)
    loop = VerificationLoop(
        VerificationConfig(max_attempts=3),
        [chk],
    )
    caller = AsyncMock(side_effect=[_result("bad"), _result("good"), _result("ok")])
    outcome = await loop.run(caller, "do thing")
    assert outcome.ok
    assert outcome.attempts == 2
    assert outcome.history[0]["failures"][0]["name"] == "flippable"
    assert outcome.history[1]["failures"] == []


@pytest.mark.asyncio
async def test_loop_records_failure_when_exhausted() -> None:
    loop = VerificationLoop(
        VerificationConfig(max_attempts=2),
        [_sync_check("nope", False, "always fails")],
    )
    caller = AsyncMock(return_value=_result("bad"))
    outcome = await loop.run(caller, "do thing")
    assert not outcome.ok
    assert outcome.attempts == 2
    assert len(outcome.history) == 2


@pytest.mark.asyncio
async def test_loop_uses_retry_template_on_failure() -> None:
    loop = VerificationLoop(
        VerificationConfig(
            max_attempts=2,
            retry_template="RETRY: {checks}",
        ),
        [_sync_check("c1", False, "broken")],
    )
    caller = AsyncMock(side_effect=[_result("first"), _result("second")])
    await loop.run(caller, "the prompt")
    # Second call should have been prompted with the retry template.
    second_prompt = caller.await_args_list[1].args[0]
    assert "RETRY:" in second_prompt
    assert "c1" in second_prompt
    assert "broken" in second_prompt
    assert "the prompt" in second_prompt  # original prompt retained


@pytest.mark.asyncio
async def test_loop_runs_async_checks() -> None:
    chk = await _async_check("async_ok", True)
    loop = VerificationLoop(
        VerificationConfig(max_attempts=2),
        [chk],
    )
    caller = AsyncMock(return_value=_result("ok"))
    outcome = await loop.run(caller, "p")
    assert outcome.ok


@pytest.mark.asyncio
async def test_loop_add_appends() -> None:
    loop = VerificationLoop()
    loop.add(_sync_check("a", True))
    loop.add(_sync_check("b", True))
    caller = AsyncMock(return_value=_result("ok"))
    outcome = await loop.run(caller, "p")
    assert outcome.ok
    assert len(outcome.history[0]["checks"]) == 2


@pytest.mark.asyncio
async def test_loop_records_to_memory_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import memory as memory_mod

    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    memory_mod._MEMORY_SINGLETON = None
    loop = VerificationLoop(
        VerificationConfig(max_attempts=2),
        [_sync_check("c", True)],
    )
    caller = AsyncMock(return_value=_result("ok"))
    await loop.run(caller, "p")
    notes = memory_mod.list_notes(kind="verify_attempt")
    assert any(n.value.get("ok") is True for n in notes)


@pytest.mark.asyncio
async def test_loop_skips_memory_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import memory as memory_mod

    monkeypatch.setattr(memory_mod, "_memory_path", lambda: tmp_path / "wm.jsonl")
    memory_mod._MEMORY_SINGLETON = None
    loop = VerificationLoop(
        VerificationConfig(max_attempts=2, record_to_memory=False),
        [_sync_check("c", True)],
    )
    caller = AsyncMock(return_value=_result("ok"))
    await loop.run(caller, "p")
    assert memory_mod.list_notes(kind="verify_attempt") == []


# ---------------------------------------------------------------------------
# ChatAgent integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_execute_with_verification_uses_loop() -> None:
    from supporter.agent import ChatAgent

    call_count = {"n": 0}

    def _flippable(result: LLMResult, _prompt: str) -> CheckResult:
        call_count["n"] += 1
        return CheckResult(
            name="min",
            ok=call_count["n"] >= 2,
            detail="ok" if call_count["n"] >= 2 else "too short",
        )

    chk = _named("min", _flippable)

    provider = MagicMock()
    provider.get_name.return_value = "fake"
    provider.generate = AsyncMock(
        side_effect=[_result(""), _result("hi"), _result("hi again")]
    )

    agent = ChatAgent(provider=provider, system_instruction="test")
    outcome = await agent.execute_with_verification(
        "do thing",
        checks=[chk],
    )
    assert outcome.ok is True
    assert outcome.attempts == 2
    assert provider.generate.await_count == 2


@pytest.mark.asyncio
async def test_agent_execute_with_verification_exhausts() -> None:
    """Exhausted attempts still return an outcome (ok=False)."""
    from supporter.agent import ChatAgent

    provider = MagicMock()
    provider.get_name.return_value = "fake"
    provider.generate = AsyncMock(return_value=_result(""))

    agent = ChatAgent(provider=provider, system_instruction="test")
    outcome = await agent.execute_with_verification(
        "do thing",
        checks=[_sync_check("min", False, "too short")],
    )
    assert outcome.ok is False
    assert outcome.attempts == 3
    assert provider.generate.await_count == 3


@pytest.mark.asyncio
async def test_agent_execute_with_verification_records_history() -> None:
    from supporter.agent import ChatAgent

    provider = MagicMock()
    provider.get_name.return_value = "fake"
    provider.generate = AsyncMock(return_value=_result("hi"))

    agent = ChatAgent(provider=provider, system_instruction="test")
    outcome = await agent.execute_with_verification(
        "do thing",
        checks=[_sync_check("ok", True)],
    )
    assert outcome.ok
    # The agent should at minimum have appended the user message; the
    # assistant message is appended when the result has candidates /
    # history, which the mock does not provide.
    assert len(agent.history) >= 1
    assert agent.history[0].role == "user"


@pytest.mark.asyncio
async def test_agent_execute_with_verification_with_recover() -> None:
    """When ``recover`` is supplied, the provider call is wrapped."""

    from supporter.agent import ChatAgent
    from supporter.recover import AutoRecover, RecoveryStatus

    def _note_recovery(label: str = "manual", detail: str = "") -> Any:
        async def _action(*args: Any, **kwargs: Any) -> RecoveryStatus:
            return RecoveryStatus(action=f"note:{label}", healed=True, detail=detail)

        return _action

    provider = MagicMock()
    provider.get_name.return_value = "fake"
    # First call: timeout (recoverable). Second call: ok.
    provider.generate = AsyncMock(side_effect=[TimeoutError(), _result("hi")])

    recover = AutoRecover(
        name="provider.generate",
        actions=[_note_recovery("always")],
        backoff_base=0,
        backoff_cap=0,
    )

    agent = ChatAgent(provider=provider, system_instruction="test")
    outcome = await agent.execute_with_verification(
        "do thing",
        checks=[_sync_check("ok", True)],
        recover=recover,
    )
    assert outcome.ok
    # The provider was called twice: once for the failed attempt, once
    # for the recovered retry.
    assert provider.generate.await_count == 2
