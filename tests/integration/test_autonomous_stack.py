"""End-to-end smoke test for the autonomous surface.

Exercises the full chain — keypool → memory → recipe → verify → recover
— without any real LLM call. Validates that all the new pieces play
together so a real session can compose them safely.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.types import LLMResult


@pytest.fixture(autouse=True)
def _reset_singletons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[None, None, None]:
    from supporter import keypool, memory, recipes

    monkeypatch.setattr(keypool, "_default_state_path", lambda: tmp_path / "kp.json")
    monkeypatch.setattr(memory, "_memory_path", lambda: tmp_path / "wm.jsonl")
    monkeypatch.setattr(recipes, "_recipe_path", lambda: tmp_path / "r.jsonl")
    monkeypatch.setattr(recipes.config, "allowed_directories", [str(tmp_path)])
    keypool.reset_key_pool()
    memory._MEMORY_SINGLETON = None
    recipes._STORE = None
    yield
    keypool.reset_key_pool()


# ---------------------------------------------------------------------------
# Scenario: the assistant saves a recipe, replays it, and the result is
# recorded in memory.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_recipe_replay_recorded_in_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import memory
    from supporter.tools import recipe_tools

    # 1. Save a recipe that emits a value and asserts it.
    save_result = await recipe_tools.recipe_save(
        "smoke",
        "smoke test recipe",
        json.dumps(
            [
                {"kind": "emit", "value": "ok"},
                {"kind": "assert_eq", "value": "ok||ok"},
            ]
        ),
    )
    assert "ok" in save_result

    # 2. Run it.
    run_output = await recipe_tools.recipe_run("smoke")
    assert "ok=True" in run_output

    # 3. The run was recorded in working memory.
    runs = memory.list_notes(kind="recipe_run")
    assert any(n.value.get("recipe") == "smoke" for n in runs)


# ---------------------------------------------------------------------------
# Scenario: verification loop rejects a "garbled" response, retries, and
# the retry passes. The final result is the only one synced to history.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_loop_persists_only_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter.verify import (
        VerificationConfig,
        VerificationLoop,
        check_min_chars,
        check_no_unicode_garble,
    )

    # First call: garbled (50 'x' chars). Second call: clean.
    call_count = {"n": 0}

    async def _caller(prompt: str) -> LLMResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return LLMResult(text="x" * 50, model="test", duration=0.01)
        return LLMResult(text="clean response", model="test", duration=0.01)

    loop = VerificationLoop(
        VerificationConfig(max_attempts=3),
        [check_min_chars(min_chars=3), check_no_unicode_garble()],
    )
    outcome = await loop.run(_caller, "do thing")
    assert outcome.ok
    assert outcome.attempts == 2
    assert outcome.history[0]["failures"]  # first attempt had failures
    assert outcome.history[1]["failures"] == []  # second attempt clean


# ---------------------------------------------------------------------------
# Scenario: AutoRecover wraps a flaky provider and retries on transient
# failures. The retry uses the SAME arguments (no caller bookkeeping).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recover_wraps_flaky_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter.recover import AutoRecover, note_recovery

    seen_args: list[tuple[Any, ...]] = []

    async def _flaky(value: str) -> str:
        seen_args.append((value,))
        if len(seen_args) < 2:
            raise ConnectionResetError("flap")
        return "ok"

    recover = AutoRecover(
        name="flaky",
        actions=[note_recovery("manual")],
        backoff_base=0,
        backoff_cap=0,
    )
    result = await recover.call(_flaky, "the same arg")
    assert result == "ok"
    # Both calls received the same arguments.
    assert seen_args == [("the same arg",), ("the same arg",)]


# ---------------------------------------------------------------------------
# Scenario: keypool rotation, memory + recipe compose. A failing key gets
# marked sick; the next acquire() picks the other key.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keypool_failure_marks_key_sick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter import keypool

    monkeypatch.setattr(keypool.config, "gemini_api_keys", ["k1", "k2"])
    pool = keypool.get_key_pool()
    assert pool is not None
    assert pool.acquire() in {"k1", "k2"}

    pool.report_failure("k1", ConnectionResetError("flap"))

    # k1 should now be in cooldown. All subsequent acquires should
    # return k2.
    for _ in range(5):
        assert pool.acquire() == "k2"

    pool.report_success("k2")
    # k2's success doesn't bring k1 back, but the snapshot still shows
    # k1 sick.
    snap = pool.all_health()
    k1 = next(h for h in snap if h.key == "k1")
    assert not k1.is_available()


# ---------------------------------------------------------------------------
# Scenario: ChatAgent.execute_with_verification with a flippable check
# plus AutoRecover wrapping the provider survives a transient provider
# error and persists the final result.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_execute_with_verification_and_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from supporter.agent import ChatAgent
    from supporter.recover import AutoRecover, note_recovery
    from supporter.verify import _named

    call_count = {"n": 0}

    def _flippable(result: LLMResult, _prompt: str) -> Any:
        call_count["n"] += 1
        from supporter.verify import CheckResult

        return CheckResult(
            name="ok",
            ok=call_count["n"] >= 2,
            detail="ok" if call_count["n"] >= 2 else "no",
        )

    provider = MagicMock()
    provider.get_name.return_value = "fake"
    # Side effect: first call times out (recoverable), then succeed
    # twice (verify retries once).
    provider.generate = AsyncMock(
        side_effect=[
            TimeoutError(),
            LLMResult(text="hi", model="test", duration=0.01),
            LLMResult(text="hi again", model="test", duration=0.01),
        ]
    )

    recover = AutoRecover(
        name="provider.generate",
        actions=[note_recovery("always")],
        backoff_base=0,
        backoff_cap=0,
    )

    agent = ChatAgent(provider=provider, system_instruction="test")
    outcome = await agent.execute_with_verification(
        "do thing",
        checks=[_named("ok", _flippable)],
        recover=recover,
    )
    # 1 TimeoutError + 1 verify-retry + 1 verify-success = 3 provider calls.
    assert provider.generate.await_count == 3
    assert outcome.ok
    assert outcome.attempts == 2  # the verification loop ran twice
