"""Tests for the dynamic plan classifier (needs_plan) and its integration
with the _is_substantive_task pre-filter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from supporter import worker
from supporter.tui import _is_substantive_task

# ---------------------------------------------------------------------------
# needs_plan parsing
# ---------------------------------------------------------------------------


class _FakeProvider:
    """Stub provider whose generate() returns a configurable text."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, prompt: str, options: Any) -> SimpleNamespace:
        return SimpleNamespace(text=self._response)


class _FailingProvider:
    """Provider that raises on generate()."""

    async def generate(self, prompt: str, options: Any) -> SimpleNamespace:
        raise RuntimeError("model unavailable")


def _patch_provider(provider: Any) -> Any:
    return patch(
        "supporter.worker.get_provider",
        return_value=provider,
    )


class TestNeedsPlanParsing:
    """Unit-test the response parsing inside needs_plan."""

    @pytest.mark.asyncio
    async def test_yes_returns_true(self) -> None:
        with _patch_provider(_FakeProvider("YES")):
            assert await worker.needs_plan("deploy the app", "test-model") is True

    @pytest.mark.asyncio
    async def test_yes_with_whitespace(self) -> None:
        with _patch_provider(_FakeProvider("  YES\n")):
            assert await worker.needs_plan("fix bug", "m") is True

    @pytest.mark.asyncio
    async def test_no_returns_false(self) -> None:
        with _patch_provider(_FakeProvider("NO")):
            assert await worker.needs_plan("hello", "test-model") is False

    @pytest.mark.asyncio
    async def test_no_with_newline(self) -> None:
        with _patch_provider(_FakeProvider("no\n")):
            assert await worker.needs_plan("thanks", "m") is False

    @pytest.mark.asyncio
    async def test_maybe_returns_none(self) -> None:
        with _patch_provider(_FakeProvider("maybe")):
            assert await worker.needs_plan("do something", "m") is None

    @pytest.mark.asyncio
    async def test_empty_response_returns_none(self) -> None:
        with _patch_provider(_FakeProvider("")):
            assert await worker.needs_plan("task", "m") is None

    @pytest.mark.asyncio
    async def test_provider_raises_returns_none(self) -> None:
        with _patch_provider(_FailingProvider()):
            assert await worker.needs_plan("task", "m") is None

    @pytest.mark.asyncio
    async def test_case_insensitive_yes(self) -> None:
        with _patch_provider(_FakeProvider("yes")):
            assert await worker.needs_plan("refactor auth", "m") is True


# ---------------------------------------------------------------------------
# _is_substantive_task pre-filter (unchanged behavior)
# ---------------------------------------------------------------------------


class TestIsSubstantiveTaskPreFilter:
    """Confirm the zero-cost pre-filter still short-circuits trivial input."""

    @pytest.mark.parametrize(
        "text",
        [
            "hi",
            "hello",
            "thanks",
            "ok",
            "👍",
            "",
            "   ",
            "/live",
            "DELEGATION_CAPSULE_RESULT: done",
        ],
    )
    def test_trivial_returns_false(self, text: str) -> None:
        assert _is_substantive_task(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "Implement JWT auth",
            "Fix the failing test in test_auth.py",
            "what time is it",
            "how are you today friend",
        ],
    )
    def test_non_trivial_returns_true(self, text: str) -> None:
        assert _is_substantive_task(text) is True
