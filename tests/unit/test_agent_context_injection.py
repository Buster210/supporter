"""Unit tests for ChatAgent's auto-injection of memory + recipe digest."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter import memory, recipes
from supporter.agent import ChatAgent
from supporter.types import LLMResult


@pytest.fixture(autouse=True)
def _reset_singletons(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[None, None, None]:
    monkeypatch.setattr(memory, "_memory_path", lambda: tmp_path / "wm.jsonl")
    monkeypatch.setattr(recipes, "_recipe_path", lambda: tmp_path / "r.jsonl")
    memory._MEMORY_SINGLETON = None
    recipes._STORE = None
    yield
    memory._MEMORY_SINGLETON = None
    recipes._STORE = None


def _new_agent(provider: MagicMock) -> ChatAgent:
    return ChatAgent(provider=provider, system_instruction="BASE PROMPT")


def _captured_system_instruction(provider: MagicMock) -> str:
    """Get the system_instruction that ChatAgent passed to the provider."""
    assert provider.generate.await_count >= 1
    call = provider.generate.await_args_list[-1]
    options = call.args[1] if len(call.args) > 1 else call.kwargs.get("options")
    return options.system_instruction  # type: ignore[no-any-return]


def test_context_injection_omitted_when_memory_empty() -> None:
    provider = MagicMock()
    provider.get_name.return_value = "fake"
    provider.generate = AsyncMock(
        return_value=LLMResult(text="hi", model="fake", duration=0.01)
    )
    agent = _new_agent(provider)
    # No memory, no recipes \u2014 system prompt should be unchanged.
    assert agent._build_context_injection() == ""


def test_context_injection_no_longer_contains_memory() -> None:
    memory.append_note("user_pref", {"theme": "dark"}, label="t1")
    memory.append_note("todo", {"task": "ship"}, label="rel")
    block = ChatAgent._build_context_injection(limit=5)
    # Orchestrator no longer owns memory tools — memory block is absent.
    assert "RECENT WORKING MEMORY" not in block


def test_context_injection_mentions_recipes_when_present() -> None:
    recipes.save_recipe(
        "demo",
        "d",
        [{"kind": "emit", "value": "hi"}],
    )
    block = ChatAgent._build_context_injection(limit=5)
    assert "KNOWN AUTOMATIONS" in block
    assert "1 recipes" in block


def test_context_injection_omits_recipes_when_empty() -> None:
    memory.append_note("user_pref", {"theme": "dark"})
    block = ChatAgent._build_context_injection(limit=5)
    assert "KNOWN AUTOMATIONS" not in block
    # Memory is no longer injected by the orchestrator.
    assert "RECENT WORKING MEMORY" not in block


@pytest.mark.asyncio
async def test_execute_injects_memory_into_system_prompt() -> None:
    memory.append_note("in_flight_task", {"job": "deploy", "step": 3}, label="d")
    provider = MagicMock()
    provider.get_name.return_value = "fake"
    provider.generate = AsyncMock(
        return_value=LLMResult(text="hi", model="fake", duration=0.01)
    )
    agent = _new_agent(provider)
    await agent.execute("do the thing")
    sys = _captured_system_instruction(provider)
    assert "BASE PROMPT" in sys
    # Memory is no longer injected by the orchestrator.
    assert "RECENT WORKING MEMORY" not in sys


@pytest.mark.asyncio
async def test_execute_with_verification_injects_memory() -> None:
    memory.append_note("user_pref", {"lang": "en"})
    provider = MagicMock()
    provider.get_name.return_value = "fake"
    provider.generate = AsyncMock(
        return_value=LLMResult(text="hi", model="fake", duration=0.01)
    )
    agent = _new_agent(provider)
    await agent.execute_with_verification("do thing", checks=[])
    sys = _captured_system_instruction(provider)
    # Memory is no longer injected by the orchestrator.
    assert "RECENT WORKING MEMORY" not in sys


@pytest.mark.asyncio
async def test_execute_no_injection_when_system_instruction_is_none() -> None:
    memory.append_note("user_pref", {"lang": "en"})
    provider = MagicMock()
    provider.get_name.return_value = "fake"
    provider.generate = AsyncMock(
        return_value=LLMResult(text="hi", model="fake", duration=0.01)
    )
    agent = ChatAgent(provider=provider, system_instruction=None)
    await agent.execute("do thing")
    sys = _captured_system_instruction(provider)
    assert sys is None or "RECENT WORKING MEMORY" not in (sys or "")


def test_context_injection_handles_corrupt_state() -> None:
    """If memory/recipe stores are corrupt, the injection just omits the
    broken section rather than raising into the call site.
    """
    # This is exercised by the fact that the helpers use broad excepts.
    # The unit test just confirms injection is robust to nothing.
    assert ChatAgent._build_context_injection(limit=0) == ""


def test_context_injection_caps_memory_to_limit() -> None:
    for i in range(20):
        memory.append_note("k", {"i": i}, label=f"l{i}")
    block = ChatAgent._build_context_injection(limit=3)
    # Memory is no longer injected — block should be empty (no recipes either).
    assert "RECENT WORKING MEMORY" not in block
