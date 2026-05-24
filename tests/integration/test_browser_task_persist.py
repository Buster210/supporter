from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from supporter.config import config
from supporter.tools.browser import task
from supporter.tools.browser.task import Playbook, Step


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


def test_load_missing_returns_none() -> None:
    assert task.load_playbook("nope.com", "never recorded") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    path = task._safe_path("bad.com", "broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert task.load_playbook("bad.com", "broken") is None


def test_path_escape_attempt_rejected() -> None:
    # A host crafted to climb out of the memory root must be sanitized, so the
    # resolved path stays inside the root rather than raising or escaping.
    path = task._safe_path("../../etc", "x")
    root = task._memory_root().resolve()
    assert root in path.parents


@pytest.mark.asyncio
async def test_finish_persists_clean_run() -> None:
    task.start("login")
    task.record(task.build_step("navigate", params={"url": "u"}))
    task.record(task.build_step("click", role="button", name="Sign in"))
    msg = await task.finish(success=True, host="example.com")
    assert "Saved playbook" in msg
    assert task.load_playbook("example.com", "login") is not None


@pytest.mark.asyncio
async def test_ineligible_action_poisons_buffer() -> None:
    task.start("scrape")
    task.record(task.build_step("navigate", params={"url": "u"}))
    task.record(task.build_step("eval"))  # excluded → poison
    msg = await task.finish(success=True, host="example.com")
    assert "unsafe step" in msg
    assert task.load_playbook("example.com", "scrape") is None


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
