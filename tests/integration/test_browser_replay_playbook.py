from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from supporter.config import config
from supporter.tools.browser import session, task, tool
from supporter.tools.browser.task import Playbook, Step

if TYPE_CHECKING:
    from .conftest import FakeSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_playbook(
    host: str,
    goal: str,
    steps: list[tuple[str, str | None, str | None, str | None]],
    *,
    created_ts: float = 100.0,
) -> None:
    pb = Playbook(
        host=host,
        goal=goal,
        created_ts=created_ts,
        steps=[
            Step(action=a, role=r or "", name=n or "", selector=s or "")
            for a, r, n, s in steps
        ],
    )
    task._save_playbook_sync(pb)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_memory(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    task.discard()


@pytest.fixture
def fake_host(fake_session: FakeSession) -> FakeSession:
    fake_session.page.eval_result = "https://example.test/"
    return fake_session


# ---------------------------------------------------------------------------
# Guard — no active page or no stored playbook
# ---------------------------------------------------------------------------


async def test_replay_no_active_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session, "active_page", lambda: None)

    result = await task.replay_playbook("anything")

    assert result == "No active page; navigate first, then replay a playbook."


async def test_replay_no_playbook(fake_host: FakeSession) -> None:
    result = await task.replay_playbook("never recorded")

    assert "No playbook found" in result
    assert "never recorded" in result
    assert "example.test" in result


# ---------------------------------------------------------------------------
# Happy path — all steps succeed
# ---------------------------------------------------------------------------


async def test_replay_happy_path(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook(
        "example.test",
        "my task",
        [
            ("navigate", None, None, None),
            ("click", None, None, "#btn"),
        ],
    )

    calls: list[str] = []

    async def fake_browse(action: str, **kwargs: Any) -> str:
        calls.append(action)
        return "done"

    monkeypatch.setattr(tool, "browse", fake_browse)

    result = await task.replay_playbook("my task")

    assert "2/2 steps succeeded" in result
    assert calls == ["navigate", "click"]


async def test_replay_happy_path_single_step(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook("example.test", "single", [("navigate", None, None, None)])

    async def fake_browse(action: str, **kwargs: Any) -> str:
        return "done"

    monkeypatch.setattr(tool, "browse", fake_browse)

    result = await task.replay_playbook("single")

    assert "1/1 steps succeeded" in result


# ---------------------------------------------------------------------------
# Error handback — step returns an Error
# ---------------------------------------------------------------------------


async def test_replay_stops_on_error_step(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook(
        "example.test",
        "flaky",
        [
            ("navigate", None, None, None),
            ("click", None, None, "#btn"),
            ("type", None, None, "#in"),
        ],
    )

    results = iter(["done", "Error: element not found", "done"])

    async def fake_browse(action: str, **kwargs: Any) -> str:
        return next(results)

    monkeypatch.setattr(tool, "browse", fake_browse)

    result = await task.replay_playbook("flaky")

    assert "Stopped at step 2" in result
    assert "1/3 step(s)" in result or "1/3" in result
    assert "Error: element not found" in result


# ---------------------------------------------------------------------------
# Ref resolution — target actions with role/name
# ---------------------------------------------------------------------------


async def test_replay_resolves_ref_from_aria_snapshot(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook(
        "example.test",
        "resolve",
        [("click", "button", "OK", None)],  # matches _SAMPLE_ARIA
    )

    seen: dict[str, Any] = {}

    async def fake_browse(action: str, **kwargs: Any) -> str:
        seen["action"] = action
        seen["kwargs"] = kwargs
        return "done"

    monkeypatch.setattr(tool, "browse", fake_browse)

    result = await task.replay_playbook("resolve")

    assert "1/1 steps succeeded" in result
    assert seen["action"] == "click"
    assert seen["kwargs"].get("ref") == "e2"
    # selector comes from _replay_params — empty for this step
    assert seen["kwargs"].get("selector", "") == ""


async def test_replay_uses_selector_when_present_over_ref(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook(
        "example.test",
        "select",
        [("click", "button", "OK", "#direct-btn")],
    )

    seen: dict[str, Any] = {}

    async def fake_browse(action: str, **kwargs: Any) -> str:
        seen["kwargs"] = kwargs
        return "done"

    monkeypatch.setattr(tool, "browse", fake_browse)

    result = await task.replay_playbook("select")

    assert "1/1 steps succeeded" in result
    # selector goes straight through, no ref resolution attempted
    assert seen["kwargs"].get("selector") == "#direct-btn"
    assert "ref" not in seen["kwargs"]


# ---------------------------------------------------------------------------
# Element not found — ref resolution fails
# ---------------------------------------------------------------------------


async def test_replay_stops_on_element_not_found(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook(
        "example.test",
        "missing",
        [("click", "button", "NONEXISTENT", None)],
    )

    async def fake_browse(action: str, **kwargs: Any) -> str:
        raise AssertionError("browse must not be called when element not found")

    monkeypatch.setattr(tool, "browse", fake_browse)

    result = await task.replay_playbook("missing")

    assert "Stopped at step 1" in result
    assert "element not found" in result
    assert "NONEXISTENT" in result


async def test_replay_non_target_action_skips_ref(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook(
        "example.test",
        "noref",
        [("navigate", "button", "OK", None)],
    )

    seen: dict[str, Any] = {}

    async def fake_browse(action: str, **kwargs: Any) -> str:
        seen["kwargs"] = kwargs
        return "done"

    monkeypatch.setattr(tool, "browse", fake_browse)

    result = await task.replay_playbook("noref")

    assert "1/1 steps succeeded" in result
    # navigate is not in _TARGET_ACTIONS, so no ref is injected
    assert "ref" not in seen["kwargs"]
