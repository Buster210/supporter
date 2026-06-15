"""Tests for Feature A — plan-before-act in the interactive orchestrator."""

from __future__ import annotations

import pytest

from supporter.tui import _is_substantive_task


class TestIsSubstantiveTask:
    """Unit tests for the _is_substantive_task heuristic."""

    @pytest.mark.parametrize(
        "text",
        [
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "yes",
            "no",
            "sure",
            "cool",
            "great",
            "awesome",
            "nice",
            "got it",
            "understood",
            "perfect",
            "sounds good",
            "will do",
            "on it",
            "yep",
            "nah",
            "nope",
            "gotcha",
            "roger",
            "ack",
            "k",
            "👍",
            # With trailing punctuation
            "thanks!",
            "ok.",
            "great?",
            "hi!",
        ],
    )
    def test_trivial_greetings_return_false(self, text: str) -> None:
        assert _is_substantive_task(text) is False

    @pytest.mark.parametrize(
        "text",
        [
            "Implement JWT auth for the REST API",
            "Fix the failing test in test_auth.py",
            "Add a new feature for user preferences",
            "Refactor the database connection pool",
            "Review my changes in the auth module",
            "Deploy the application to production",
            "Write documentation for the API",
            "Investigate the memory leak in the worker",
        ],
    )
    def test_substantive_tasks_return_true(self, text: str) -> None:
        assert _is_substantive_task(text) is True

    def test_empty_string_returns_false(self) -> None:
        assert _is_substantive_task("") is False

    def test_whitespace_only_returns_false(self) -> None:
        assert _is_substantive_task("   \n\t  ") is False

    def test_slash_commands_return_false(self) -> None:
        assert _is_substantive_task("/live") is False
        assert _is_substantive_task("/agent") is False
        assert _is_substantive_task("/clear") is False
        assert _is_substantive_task("/exit") is False

    def test_delegation_re_injection_returns_false(self) -> None:
        assert (
            _is_substantive_task(
                "DELEGATION_CAPSULE_RESULT: task completed successfully"
            )
            is False
        )
        assert _is_substantive_task("MILESTONE_RESULT: all tasks done") is False

    def test_very_short_fragments_return_false(self) -> None:
        # Under 5 chars, non-trivial
        assert _is_substantive_task("ab") is False
        assert _is_substantive_task("xyz") is False
        assert _is_substantive_task("a") is False

    def test_exactly_five_chars_returns_true(self) -> None:
        # "hello" is in trivial set, but "abort" is not
        assert _is_substantive_task("abort") is True
        assert _is_substantive_task("deploy") is True

    def test_case_insensitive_trivial(self) -> None:
        assert _is_substantive_task("HI") is False
        assert _is_substantive_task("Thanks") is False
        assert _is_substantive_task("OK") is False

    def test_trailing_punctuation_stripped(self) -> None:
        assert _is_substantive_task("thanks!!!") is False
        assert _is_substantive_task("ok...") is False
        assert _is_substantive_task("great!!!") is False
