from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from supporter.config import config
from supporter.tools.browser.playbook_match import (
    _fuzzy_find_goal,
)
from supporter.tools.browser.playbook_store import (
    _PRUNE_FAIL_FLOOR,
    _PRUNE_TTL_DAYS,
    Playbook,
    Step,
    _delete_playbook_sync,
    _list_playbooks_sync,
    _safe_path,
    _save_playbook_sync,
    load_playbook,
    prune_playbooks,
)
from supporter.tools.browser.recorder import discard


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
    _save_playbook_sync(pb)


@pytest.fixture(autouse=True)
def isolate_memory(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    discard()


def test_save_then_load_round_trip() -> None:
    pb = Playbook(
        host="github.com",
        goal="search repos",
        created_ts=123.0,
        steps=[
            Step(action="navigate", params={"url": "https://github.com"}),
            Step(action="click", role="button", name="Search"),
        ],
    )
    _save_playbook_sync(pb)
    loaded = load_playbook("github.com", "search repos")
    assert loaded is not None
    assert loaded.host == "github.com"
    assert loaded.goal == "search repos"
    assert [s.action for s in loaded.steps] == ["navigate", "click"]
    assert loaded.steps[1].name == "Search"
    assert (
        loaded.schema_version == Playbook.__dataclass_fields__["schema_version"].default
    )
    assert loaded.variables == []
    assert loaded.success_count == 0
    assert loaded.fail_count == 0


def test_load_missing_returns_none() -> None:
    assert load_playbook("nope.com", "never recorded") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    path = _safe_path("bad.com", "broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert load_playbook("bad.com", "broken") is None


def test_path_escape_attempt_rejected() -> None:
    from supporter.tools.browser.playbook_store import _memory_root

    path = _safe_path("../../etc", "x")
    root = _memory_root().resolve()
    assert root in path.parents


def test_v1_playbook_loads_and_upgrades(tmp_path: Path) -> None:
    from supporter.tools.browser.playbook_store import SCHEMA_VERSION

    v1_data = {
        "host": "example.com",
        "goal": "login",
        "created_ts": 100.0,
        "steps": [
            {"action": "navigate", "params": {"url": "https://example.com"}},
            {"action": "click", "role": "button", "name": "Sign in"},
        ],
    }
    path = _safe_path("example.com", "login")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(v1_data), encoding="utf-8")

    loaded = load_playbook("example.com", "login")
    assert loaded is not None
    assert loaded.schema_version == 1
    assert loaded.success_count == 0
    assert loaded.fail_count == 0
    assert loaded.steps[0].action == "navigate"
    assert loaded.steps[1].name == "Sign in"

    _save_playbook_sync(loaded)
    on_disk = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == SCHEMA_VERSION
    assert on_disk["variables"] == []
    assert on_disk["success_count"] == 0


@pytest.mark.asyncio
async def test_finish_persists_clean_run() -> None:
    from supporter.tools.browser.recorder import finish, record, start

    start("login")
    record(Step(action="navigate", params={"url": "u"}))
    record(Step(action="click", role="button", name="Sign in"))
    msg = await finish(success=True, host="example.com")
    assert "Saved playbook" in msg
    assert load_playbook("example.com", "login") is not None


@pytest.mark.asyncio
async def test_ineligible_action_is_skipped_not_fatal() -> None:
    from supporter.tools.browser.playbook_store import build_step
    from supporter.tools.browser.recorder import finish, record, start

    start("scrape")
    record(build_step("navigate", params={"url": "u"}))
    record(build_step("eval"))
    msg = await finish(success=True, host="example.com")
    assert "Saved playbook" in msg
    playbook = load_playbook("example.com", "scrape")
    assert playbook is not None
    assert len(playbook.steps) == 1
    assert playbook.steps[0].action == "navigate"


@pytest.mark.asyncio
async def test_error_result_poisons_buffer() -> None:
    from supporter.tools.browser.playbook_store import build_step
    from supporter.tools.browser.recorder import finish, record, start

    start("flaky")
    record(build_step("click", result="Error: ref not found"))
    msg = await finish(success=True, host="example.com")
    assert "error" in msg.lower()
    assert load_playbook("example.com", "flaky") is None


@pytest.mark.asyncio
async def test_finish_failure_discards() -> None:
    from supporter.tools.browser.playbook_store import build_step
    from supporter.tools.browser.recorder import finish, record, start

    start("abandon")
    record(build_step("navigate", params={"url": "u"}))
    msg = await finish(success=False, host="example.com")
    assert "nothing saved" in msg
    assert load_playbook("example.com", "abandon") is None


@pytest.mark.asyncio
async def test_list_playbooks_no_active_page() -> None:
    from supporter.tools.browser.task import list_playbooks

    msg = await list_playbooks()
    assert "No active page" in msg


def test_list_playbooks_lists_saved_for_host() -> None:
    pb1 = Playbook(
        host="example.com",
        goal="login",
        created_ts=100.0,
        steps=[Step(action="navigate", params={"url": "u"})],
    )
    pb2 = Playbook(
        host="example.com",
        goal="search",
        created_ts=200.0,
        steps=[Step(action="click", role="button")],
    )
    _save_playbook_sync(pb1)
    _save_playbook_sync(pb2)

    descriptors = _list_playbooks_sync("example.com")
    assert len(descriptors) == 2
    goals = {d["goal"] for d in descriptors}
    assert goals == {"login", "search"}
    for d in descriptors:
        assert "steps" not in d
        assert "variable" not in d
        assert "success_count" in d
        assert "step_count" in d


def test_prune_deletes_stale_and_failing(tmp_path: Path) -> None:
    import time

    stale_pb = Playbook(
        host="example.com",
        goal="old_task",
        created_ts=time.time() - (_PRUNE_TTL_DAYS + 1) * 86400,
        steps=[Step(action="navigate")],
    )
    fail_pb = Playbook(
        host="example.com",
        goal="bad_task",
        created_ts=time.time(),
        steps=[Step(action="click")],
    )
    _save_playbook_sync(stale_pb)
    _save_playbook_sync(fail_pb)

    descriptors = _list_playbooks_sync("example.com")
    assert len(descriptors) == 2

    bad_path = _safe_path("example.com", "bad_task")
    import json

    data = json.loads(bad_path.read_text())
    data["fail_count"] = _PRUNE_FAIL_FLOOR
    data["success_count"] = 0
    bad_path.write_text(json.dumps(data))

    deleted = prune_playbooks("example.com")
    assert deleted >= 1

    remaining = _list_playbooks_sync("example.com")
    goals = {d["goal"] for d in remaining}
    assert "bad_task" not in goals


@pytest.mark.asyncio
async def test_finish_derives_variables_from_step_annotations() -> None:
    from supporter.tools.browser.recorder import finish, record, start

    start("login", host="example.com", variables=["username"])
    record(
        Step(
            action="type",
            role="textbox",
            name="Password",
            params={"text": "secret"},
            variable="password",
        )
    )
    msg = await finish(success=True, host="example.com")
    assert "Saved playbook" in msg
    pb = load_playbook("example.com", "login")
    assert pb is not None
    assert set(pb.variables) == {"username", "password"}


@pytest.mark.asyncio
async def test_back_action_is_recorded() -> None:
    from supporter.tools.browser.playbook_store import build_step
    from supporter.tools.browser.recorder import finish, record, start

    start("nav")
    record(build_step("navigate", params={"url": "u"}))
    record(build_step("back"))
    await finish(success=True, host="example.com")
    pb = load_playbook("example.com", "nav")
    assert pb is not None
    assert [s.action for s in pb.steps] == ["navigate", "back"]


def test_delete_playbook_sync_removes_file() -> None:
    pb = Playbook(
        host="example.com",
        goal="kill me",
        created_ts=1.0,
        steps=[Step(action="navigate")],
    )
    _save_playbook_sync(pb)
    assert load_playbook("example.com", "kill me") is not None
    assert _delete_playbook_sync("example.com", "kill me") is True
    assert load_playbook("example.com", "kill me") is None


def test_delete_playbook_sync_missing_returns_false() -> None:
    assert _delete_playbook_sync("example.com", "never recorded") is False


def test_fuzzy_find_goal_ranks_by_token_overlap() -> None:
    for goal in ("log in to github", "search github issues", "open gmail inbox"):
        _save_playbook_sync(
            Playbook(
                host="example.com",
                goal=goal,
                created_ts=1.0,
                steps=[Step(action="navigate")],
            )
        )
    matches = _fuzzy_find_goal("example.com", "log into github")
    assert matches
    assert matches[0] == "log in to github"
    assert "open gmail inbox" not in matches
