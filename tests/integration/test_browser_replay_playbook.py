from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from supporter.config import config
from supporter.tools.browser import session, task, tool
from supporter.tools.browser.task import Playbook, Step

if TYPE_CHECKING:
    from .conftest import FakeSession


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


@pytest.fixture(autouse=True)
def isolate_memory(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    task.discard()


@pytest.fixture
def fake_host(fake_session: FakeSession) -> FakeSession:
    fake_session.page.eval_result = "https://example.test/"
    return fake_session


async def test_replay_no_active_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(session, "active_page", lambda: None)

    result = await task.replay_playbook("anything")

    assert result == "No active page; navigate first, then replay a playbook."


async def test_replay_no_playbook(fake_host: FakeSession) -> None:
    result = await task.replay_playbook("never recorded")

    assert "No playbook found" in result
    assert "never recorded" in result
    assert "example.test" in result


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


async def test_replay_resolves_ref_from_aria_snapshot(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook(
        "example.test",
        "resolve",
        [("click", "button", "OK", None)],
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
    assert seen["kwargs"].get("selector") == "#direct-btn"
    assert "ref" not in seen["kwargs"]


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
    assert "ref" not in seen["kwargs"]


async def test_replay_sensitive_step_still_gated(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook("example.test", "sensitive", [("click", "button", "Delete", None)])

    calls: list[dict[str, Any]] = []

    async def fake_browse(action: str, **kwargs: Any) -> str:
        calls.append({"action": action, "kwargs": kwargs})
        return "done"

    monkeypatch.setattr(tool, "browse", fake_browse)

    async def fake_snapshot(page: Any) -> str:
        return '- button "Delete" [ref=e3]'

    monkeypatch.setattr(tool, "_live_refs_snapshot", fake_snapshot)

    result = await task.replay_playbook("sensitive")

    assert "1/1 steps succeeded" in result
    assert len(calls) == 1
    assert calls[0]["action"] == "click"
    assert "ref" in calls[0]["kwargs"]


async def test_replay_self_heals_fuzzy_name(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook("example.test", "fuzzy", [("click", "button", "Sign in", None)])

    seen: dict[str, Any] = {}

    async def fake_browse(action: str, **kwargs: Any) -> str:
        seen["kwargs"] = kwargs
        return "done"

    async def fake_snapshot(page: Any) -> str:
        return '- button "Sign In" [ref=e4]'

    monkeypatch.setattr(tool, "browse", fake_browse)
    monkeypatch.setattr(tool, "_live_refs_snapshot", fake_snapshot)

    result = await task.replay_playbook("fuzzy")

    assert "1/1 steps succeeded" in result
    assert seen["kwargs"].get("ref") == "e4"


async def test_replay_rejects_unknown_override(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook("example.test", "override_test", [("navigate", None, None, None)])

    result = await task.replay_playbook("override_test", overrides={"unknown_var": "x"})

    assert "Unknown override" in result
    assert "unknown_var" in result


async def test_replay_success_bumps_success_count(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook("example.test", "count_test", [("navigate", None, None, None)])

    async def fake_browse(action: str, **kwargs: Any) -> str:
        return "done"

    monkeypatch.setattr(tool, "browse", fake_browse)

    await task.replay_playbook("count_test")
    await task.replay_playbook("count_test")

    pb = task.load_playbook("example.test", "count_test")
    assert pb is not None
    assert pb.success_count >= 2


async def test_replay_drift_bumps_fail_count(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_playbook(
        "example.test",
        "drift_test",
        [("click", "button", "NONEXISTENT", None)],
    )

    async def empty_snapshot(page: Any) -> str:
        return "- heading 'Other' [ref=e1]"

    monkeypatch.setattr(tool, "_live_refs_snapshot", empty_snapshot)

    result = await task.replay_playbook("drift_test")

    assert "Stopped at step" in result

    pb = task.load_playbook("example.test", "drift_test")
    assert pb is not None
    assert pb.fail_count >= 1


async def test_start_task_captures_host(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msg = await task.start_task("capture test")
    assert "Recording task" in msg

    assert task._ACTIVE is not None
    assert task._ACTIVE.host == "example.test"

    task.discard()


async def test_replay_substitutes_recorded_variable(
    fake_host: FakeSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pb = Playbook(
        host="example.test",
        goal="login",
        created_ts=100.0,
        steps=[
            Step(
                action="type",
                selector="#user",
                params={"text": "admin"},
                variable="username",
            )
        ],
        variables=["username"],
    )
    task._save_playbook_sync(pb)

    seen: dict[str, Any] = {}

    async def fake_browse(action: str, **kwargs: Any) -> str:
        seen["kwargs"] = kwargs
        return "done"

    monkeypatch.setattr(tool, "browse", fake_browse)

    result = await task.replay_playbook("login", overrides={"username": "alice"})

    assert "1/1 steps succeeded" in result
    assert seen["kwargs"]["text"] == "alice"


async def test_query_playbook_suggests_similar_on_miss(
    fake_host: FakeSession,
) -> None:
    task._save_playbook_sync(
        Playbook(
            host="example.test",
            goal="log in to github",
            created_ts=100.0,
            steps=[Step(action="navigate")],
        )
    )
    result = await task.query_playbook("github login")
    assert "No playbook found" in result
    assert "log in to github" in result


async def test_delete_playbook_tool_removes_and_reports(
    fake_host: FakeSession,
) -> None:
    task._save_playbook_sync(
        Playbook(
            host="example.test",
            goal="stale flow",
            created_ts=100.0,
            steps=[Step(action="navigate")],
        )
    )
    msg = await task.delete_playbook("stale flow")
    assert "Deleted playbook" in msg
    assert task.load_playbook("example.test", "stale flow") is None
