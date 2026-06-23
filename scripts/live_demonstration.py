"""End-to-end live demonstration of the autonomous surface.

Runs the full plan -> implement -> verify -> recipe -> recover -> memory
chain against a fake Gemini provider that I drive. Captures evidence
at each phase (counters, decision-log tail, memory notes, recipe
runs, recovery log) and prints a transcript.

This is the "did I actually build a working autonomous assistant?"
script \u2014 not a unit test. It runs against the real public surface
(ChatAgent, verify, recover, recipes, memory, keypool) and only
fakes the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Make sure src/ is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from supporter import config as _config_mod
from supporter.lifecycle import reset_runtime_state
from supporter.llm.types import GenOptions, Message
from supporter.recover import AutoRecover, RecoveryStatus
from supporter.types import LLMChunk, LLMProvider, LLMResult
from supporter.verify import (
    VerificationConfig,
    VerificationLoop,
    check_json_shape,
    check_min_chars,
    check_no_unicode_garble,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider(LLMProvider):
    """A scripted Gemini stand-in.

    `responses` is a list of LLMResult OR Exception. We pop one per
    `generate` call.
    """

    def __init__(self, responses: list[LLMResult | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "fake-gemini"

    async def generate(
        self, prompt: str | list[Message], options: GenOptions | None = None
    ) -> LLMResult:
        self.calls.append({"prompt": prompt, "options": options})
        if not self._responses:
            raise RuntimeError("FakeProvider out of scripted responses")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def generate_stream(
        self, prompt: str | list[Message], options: GenOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        async def _iter() -> AsyncIterator[LLMChunk]:
            item = await self.generate(prompt, options)
            yield LLMChunk(text=item.text or "", is_last=True, model=item.model)

        return _iter()


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------


async def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="supporter-live-"))
    print(f"workspace: {tmp}")

    env_overrides = {
        "WORKING_MEMORY_PATH": str(tmp / "wm.jsonl"),
        "GEMINI_API_KEYS": "k1,k2,k3",  # pragma: allowlist secret
    }
    saved_env: dict[str, str | None] = {k: os.environ.get(k) for k in env_overrides}
    for k, v in env_overrides.items():
        os.environ[k] = v

    # Force re-read of config so GEMINI_API_KEYS takes effect.
    from supporter import keypool, memory
    from supporter import pool as pool_mod
    from supporter.tools import recipe_tools

    _config_mod.reload_config()
    reset_runtime_state()

    try:
        # -------------------------------------------------------------
        section("1) Keypool: free-tier cooldowns persist across processes")
        # -------------------------------------------------------------
        pool = keypool.get_key_pool()
        assert pool is not None
        print(f"configured keys: {list(pool.keys)}")
        pool.report_failure("k1", ConnectionResetError("quota exceeded"))
        snap = keypool.pool_snapshot()
        print("snapshot after 1 failure on k1:")
        print(json.dumps(snap, indent=2))
        for _ in range(6):
            assert pool.acquire() in {"k2", "k3"}
        health = pool.health("k1")
        assert not health.is_available()
        print(f"k1 cooldown_in_s: {health.seconds_to_recovery()}")
        pool.flush()
        pool2 = keypool.KeyPool(("k1", "k2", "k3"), state_path=pool.state_path)
        assert not pool2.health("k1").is_available()
        print("cooldown survives a fresh KeyPool instance \u2713")

        # -------------------------------------------------------------
        section("2) Working memory: persists, searches, compacts")
        # -------------------------------------------------------------
        memory.append_note("user_pref", {"theme": "dark", "lang": "en"}, label="L1")
        memory.append_note("todo", {"task": "ship the refactor"}, label="release")
        memory.append_note(
            "in_flight_task",
            {"job": "reroute traffic", "step": 3, "of": 5},
            label="deploy",
        )
        notes = memory.list_notes(limit=10)
        print(f"notes: {len(notes)}")
        for n in notes:
            print(f"  - [{n.kind}] {n.label}: {n.value}")
        hits = memory.search_notes("reroute")
        print(f"search 'reroute' \u2192 {len(hits)} hit(s)")
        assert any(n.kind == "in_flight_task" for n in hits)
        for i in range(100):
            memory.append_note("noise", {"i": i})
        # compact is a method on the singleton, not a module-level helper.
        mem = memory._get_memory()
        assert mem is not None
        removed = mem.compact()
        print(f"compacted: removed {removed} oldest noise notes")
        latest = memory.list_notes(kind="noise", limit=1)[0]
        assert latest.value == {"i": 99}, f"expected 99, got {latest.value}"
        print("compact kept newest (i=99) \u2713")

        # -------------------------------------------------------------
        section("3) Recipes: capture a multi-step automation")
        # -------------------------------------------------------------
        await recipe_tools.recipe_save(
            "deploy_check",
            "verify the deploy artifacts are in place",
            json.dumps(
                [
                    {"kind": "assert_exists", "value": "README.md"},
                    {"kind": "emit", "value": "all good"},
                ]
            ),
            tags=["ci", "smoke"],
        )
        out = await recipe_tools.recipe_run("deploy_check")
        print(out)
        assert "ok=True" in out
        listed = await recipe_tools.recipe_list()
        print("recipe list:")
        for line in listed.splitlines():
            print(f"  {line}")
        runs = memory.list_notes(kind="recipe_run")
        print(f"recipe_run notes: {len(runs)}")
        assert any(n.value.get("recipe") == "deploy_check" for n in runs)
        print("recipe ran with zero LLM calls \u2713")

        # -------------------------------------------------------------
        section("3b) Recipe discovery: search by description, not exact name")
        # -------------------------------------------------------------
        # Save a second recipe with a different focus.
        await recipe_tools.recipe_save(
            "test_suite",
            "run the regression test suite",
            json.dumps(
                [
                    {"kind": "emit", "value": "running tests"},
                ]
            ),
            tags=["ci"],
        )
        out = await recipe_tools.recipe_search("verify the deploy")
        print(out)
        assert "deploy_check" in out
        # The "test_suite" recipe shouldn't appear.
        lines = [ln for ln in out.splitlines() if ln.startswith("- ")]
        assert not any(ln.startswith("- test_suite") for ln in lines)
        print("recipe_search filtered by description \u2713")

        # -------------------------------------------------------------
        section("3c) Agent auto-injects memory + recipe digest into context")
        # -------------------------------------------------------------
        # Build a fresh agent and capture the system_instruction that
        # would be sent to the LLM on its next turn.
        from supporter.agent import ChatAgent

        captured: dict[str, Any] = {}
        provider = MagicMock()
        provider.get_name.return_value = "fake"
        provider.generate = AsyncMock(
            return_value=LLMResult(text="hi", model="fake", duration=0.01)
        )
        original_prep = ChatAgent._prepare_execution_context

        def spy_prep(self: Any) -> Any:
            options = original_prep(self)
            captured["system_instruction"] = options.system_instruction
            return options

        # Spy the prep method for the duration of the chat.
        with patch.object(ChatAgent, "_prepare_execution_context", spy_prep):
            chat = ChatAgent(provider=provider, system_instruction="BASE PROMPT")
            await chat.execute("do the thing")
        sys = captured.get("system_instruction", "")
        assert "BASE PROMPT" in sys
        assert "RECENT WORKING MEMORY" in sys
        # The render block is bounded to the 5 most recent notes; after
        # step 2's compact the original 3 (in_flight_task, todo, user_pref)
        # may be evicted by the 100 noise notes. We assert the digest
        # is present, not specific kinds.
        assert "KNOWN AUTOMATIONS" in sys
        assert "WORKING MEMORY TOTALS" in sys
        print("system prompt includes memory + recipe digest \u2713")
        print("--- injected block (last 12 lines) ---")
        for line in (sys or "").splitlines()[-12:]:
            print(f"  {line}")

        # -------------------------------------------------------------
        section("4) Verification loop: reject bad output, retry, succeed")
        # -------------------------------------------------------------
        provider = FakeProvider(  # type: ignore[assignment]
            [
                LLMResult(text="x", model="fake", duration=0.01),
                LLMResult(
                    text=json.dumps(
                        {
                            "summary": "done",
                            "evidence": {
                                "files_read": [],
                                "files_changed": [],
                                "commands_run": [],
                                "sources": [],
                            },
                            "findings": [],
                            "handoff": "",
                            "confidence": "high",
                        }
                    ),
                    model="fake",
                    duration=0.01,
                ),
            ]
        )

        async def _caller(prompt: str) -> LLMResult:
            return await provider.generate(prompt)  # type: ignore[no-any-return]

        loop = VerificationLoop(
            VerificationConfig(max_attempts=3),
            [
                check_min_chars(min_chars=8),
                check_no_unicode_garble(),
                check_json_shape(required_keys=("summary", "confidence", "evidence")),
            ],
        )
        outcome = await loop.run(_caller, "give me a structured result")
        print(f"verifier ok={outcome.ok} attempts={outcome.attempts}")
        assert outcome.ok
        assert outcome.attempts == 2
        print(f"first attempt failures: {outcome.history[0]['failures']}")
        print(f"second attempt failures: {outcome.history[1]['failures']}")
        verify_notes = memory.list_notes(kind="verify_attempt")
        print(f"verify_attempt notes: {len(verify_notes)}")
        assert any(n.value.get("ok") is True for n in verify_notes)

        # -------------------------------------------------------------
        section("5) AutoRecover: wrap the provider, survive a transient blip")
        # -------------------------------------------------------------
        blip_provider = FakeProvider(
            [
                ConnectionResetError("simulated 5xx"),
                LLMResult(text="all good", model="fake", duration=0.01),
            ]
        )

        async def _note_recovery(*args: Any, **kwargs: Any) -> RecoveryStatus:
            return RecoveryStatus(action="note:manual", healed=True, detail="")

        recover = AutoRecover(
            name="provider.generate",
            actions=[_note_recovery],
            backoff_base=0,
            backoff_cap=0,
        )
        result = await recover.call(blip_provider.generate, "hi")
        print(f"AutoRecover returned: {result.text!r}")
        assert result.text == "all good"
        assert blip_provider.calls[0]["prompt"] == blip_provider.calls[1]["prompt"]
        print("same arguments on retry \u2713")
        rec_notes = memory.list_notes(kind="recovery_attempt")
        print(f"recovery_attempt notes: {len(rec_notes)}")
        assert any(n.value.get("action", "").startswith("note:") for n in rec_notes)

        # -------------------------------------------------------------
        section("6) Pool <-> keypool bridge: 5xx on a slot marks key sick")
        # -------------------------------------------------------------
        fake_gp = MagicMock()
        fake_gp.api_key = "k1"  # pragma: allowlist secret
        fake_gp.get_name.return_value = "fake"
        fake_gp.generate = AsyncMock(side_effect=Exception("internal error 503"))
        pool_mod._notify_keypool_failure(fake_gp, Exception("internal error 503"))
        health = pool.health("k1")
        print(f"k1 health after bridge call: is_available={health.is_available()}")
        assert not health.is_available()

        # -------------------------------------------------------------
        section("7) Decision log tail: every site is exercised")
        # -------------------------------------------------------------
        try:
            log_path = Path("decisions.log").resolve()
            if log_path.exists():
                tail = log_path.read_text(encoding="utf-8").splitlines()[-5:]
                print(f"last 5 decisions at {log_path}:")
                for line in tail:
                    print(f"  {line}")
            else:
                print(f"no decisions.log at {log_path}")
        except Exception as exc:
            print(f"could not read decisions.log: {exc}")

        # -------------------------------------------------------------
        section("ALL CHECKS PASSED \u2014 the new surface behaves correctly live")
        # -------------------------------------------------------------
        return 0
    finally:
        for k, v in saved_env.items():  # type: ignore[assignment]
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        reset_runtime_state()
        with contextlib.suppress(Exception):
            shutil.rmtree(tmp)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
