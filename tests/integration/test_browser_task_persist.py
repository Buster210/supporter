from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from supporter.config import config
from supporter.tools.browser import task
from supporter.tools.browser.task import Playbook, Step


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
    task._save_playbook_sync(pb)
    loaded = task.load_playbook("github.com", "search repos")
    assert loaded is not None
    assert loaded.host == "github.com"
    assert loaded.goal == "search repos"
    assert [s.action for s in loaded.steps] == ["navigate", "click"]
    assert loaded.steps[1].name == "Search"
    assert loaded.schema_version == task.SCHEMA_VERSION
    assert loaded.variables == []
    assert loaded.success_count == 0
    assert loaded.fail_count == 0


def test_load_missing_returns_none() -> None:
    assert task.load_playbook("nope.com", "never recorded") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    path = task._safe_path("bad.com", "broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert task.load_playbook("bad.com", "broken") is None


def test_path_escape_attempt_rejected() -> None:
    path = task._safe_path("../../etc", "x")
    root = task._memory_root().resolve()
    assert root in path.parents


def test_v1_playbook_loads_and_upgrades(tmp_path: Path) -> None:
    v1_data = {
        "host": "example.com",
        "goal": "login",
        "created_ts": 100.0,
        "steps": [
            {"action": "navigate", "params": {"url": "https://example.com"}},
            {"action": "click", "role": "button", "name": "Sign in"},
        ],
    }
    path = task._safe_path("example.com", "login")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(v1_data), encoding="utf-8")

    loaded = task.load_playbook("example.com", "login")
    assert loaded is not None
    assert loaded.schema_version == 1
    assert loaded.success_count == 0
    assert loaded.fail_count == 0
    assert loaded.steps[0].action == "navigate"
    assert loaded.steps[1].name == "Sign in"

    task._save_playbook_sync(loaded)
    on_disk = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 2
    assert on_disk["variables"] == []
    assert on_disk["success_count"] == 0


@pytest.mark.asyncio
async def test_finish_persists_clean_run() -> None:
    task.start("login")
    task.record(task.build_step("navigate", params={"url": "u"}))
    task.record(task.build_step("click", role="button", name="Sign in"))
    msg = await task.finish(success=True, host="example.com")
    assert "Saved playbook" in msg
    assert task.load_playbook("example.com", "login") is not None


@pytest.mark.asyncio
async def test_ineligible_action_is_skipped_not_fatal() -> None:
    task.start("scrape")
    task.record(task.build_step("navigate", params={"url": "u"}))
    task.record(task.build_step("eval"))
    msg = await task.finish(success=True, host="example.com")
    assert "Saved playbook" in msg
    playbook = task.load_playbook("example.com", "scrape")
    assert playbook is not None
    assert len(playbook.steps) == 1
    assert playbook.steps[0].action == "navigate"


@pytest.mark.asyncio
async def test_error_result_poisons_buffer() -> None:
    task.start("flaky")
    task.record(task.build_step("click", result="Error: ref not found"))
    msg = await task.finish(success=True, host="example.com")
    assert "error" in msg.lower()
    assert task.load_playbook("example.com", "flaky") is None


@pytest.mark.asyncio
async def test_finish_failure_discards() -> None:
    task.start("abandon")
    task.record(task.build_step("navigate", params={"url": "u"}))
    msg = await task.finish(success=False, host="example.com")
    assert "nothing saved" in msg
    assert task.load_playbook("example.com", "abandon") is None


@pytest.mark.asyncio
async def test_list_playbooks_no_active_page() -> None:
    msg = await task.list_playbooks()
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
    task._save_playbook_sync(pb1)
    task._save_playbook_sync(pb2)

    descriptors = task._list_playbooks_sync("example.com")
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
        created_ts=time.time() - (task._PRUNE_TTL_DAYS + 1) * 86400,
        steps=[Step(action="navigate")],
    )
    fail_pb = Playbook(
        host="example.com",
        goal="bad_task",
        created_ts=time.time(),
        steps=[Step(action="click")],
    )
    task._save_playbook_sync(stale_pb)
    task._save_playbook_sync(fail_pb)

    descriptors = task._list_playbooks_sync("example.com")
    assert len(descriptors) == 2

    bad_path = task._safe_path("example.com", "bad_task")
    import json

    data = json.loads(bad_path.read_text())
    data["fail_count"] = task._PRUNE_FAIL_FLOOR
    data["success_count"] = 0
    bad_path.write_text(json.dumps(data))

    deleted = task.prune_playbooks("example.com")
    assert deleted >= 1

    remaining = task._list_playbooks_sync("example.com")
    goals = {d["goal"] for d in remaining}
    assert "bad_task" not in goals


@pytest.mark.asyncio
async def test_finish_derives_variables_from_step_annotations() -> None:
    task.start("login", host="example.com", variables=["username"])
    task.record(
        task.build_step(
            "type",
            role="textbox",
            name="Password",
            params={"text": "secret"},
            variable="password",
        )
    )
    msg = await task.finish(success=True, host="example.com")
    assert "Saved playbook" in msg
    pb = task.load_playbook("example.com", "login")
    assert pb is not None
    assert set(pb.variables) == {"username", "password"}


@pytest.mark.asyncio
async def test_back_action_is_recorded() -> None:
    task.start("nav")
    task.record(task.build_step("navigate", params={"url": "u"}))
    task.record(task.build_step("back"))
    await task.finish(success=True, host="example.com")
    pb = task.load_playbook("example.com", "nav")
    assert pb is not None
    assert [s.action for s in pb.steps] == ["navigate", "back"]


def test_delete_playbook_sync_removes_file() -> None:
    pb = Playbook(
        host="example.com",
        goal="kill me",
        created_ts=1.0,
        steps=[Step(action="navigate")],
    )
    task._save_playbook_sync(pb)
    assert task.load_playbook("example.com", "kill me") is not None
    assert task._delete_playbook_sync("example.com", "kill me") is True
    assert task.load_playbook("example.com", "kill me") is None


def test_delete_playbook_sync_missing_returns_false() -> None:
    assert task._delete_playbook_sync("example.com", "never recorded") is False


def test_fuzzy_find_goal_ranks_by_token_overlap() -> None:
    for goal in ("log in to github", "search github issues", "open gmail inbox"):
        task._save_playbook_sync(
            Playbook(
                host="example.com",
                goal=goal,
                created_ts=1.0,
                steps=[Step(action="navigate")],
            )
        )
    matches = task._fuzzy_find_goal("example.com", "log into github")
    assert matches
    assert matches[0] == "log in to github"
    assert "open gmail inbox" not in matches
