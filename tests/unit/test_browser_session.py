from __future__ import annotations

from collections.abc import Iterator

import pytest

from supporter.tools.browser import guardrails, session


@pytest.fixture(autouse=True)
def _reset_session_globals() -> Iterator[None]:
    saved_keep = session._KEEP_OPEN
    saved_page = session._PAGE
    saved_cb = guardrails.browse_confirmation_callback
    try:
        yield
    finally:
        session._KEEP_OPEN = saved_keep
        session._PAGE = saved_page
        guardrails.browse_confirmation_callback = saved_cb


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), (True, True), (False, False)],
)
def test_keep_open_defaults_to_true_unless_explicitly_false(
    value: bool | None, expected: bool
) -> None:
    session._KEEP_OPEN = value
    assert session.keep_open() is expected


def test_is_active_reflects_page_presence() -> None:
    session._PAGE = None
    assert session.is_active() is False
    session._PAGE = object()  # type: ignore[assignment]
    assert session.is_active() is True


async def test_prompt_lifecycle_is_asked_once() -> None:
    session._KEEP_OPEN = None
    calls: list[tuple[str, str]] = []

    async def cb(title: str, detail: str) -> bool:
        calls.append((title, detail))
        return False

    guardrails.browse_confirmation_callback = cb
    try:
        await session._prompt_lifecycle()
        await session._prompt_lifecycle()
    finally:
        guardrails.browse_confirmation_callback = None

    assert len(calls) == 1
    assert session._KEEP_OPEN is False


async def test_prompt_lifecycle_fails_open_when_unwired() -> None:
    session._KEEP_OPEN = None
    guardrails.browse_confirmation_callback = None
    await session._prompt_lifecycle()
    assert session._KEEP_OPEN is True
