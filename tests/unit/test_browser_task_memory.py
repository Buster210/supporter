from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from supporter.config import config
from supporter.tools.browser import task_memory
from supporter.tools.browser.task_memory import Playbook, Step


@pytest.fixture(autouse=True)
def isolate_memory(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    task_memory.discard()


def test_slug_sanitizes_to_safe_charset() -> None:
    assert task_memory._slug("Log in to GitHub!") == "log-in-to-github"
    assert task_memory._slug("a/../../etc/passwd") == "a-..-..-etc-passwd"
    assert task_memory._slug("") == "task"


def test_slug_is_length_bounded() -> None:
    assert len(task_memory._slug("x" * 500)) <= task_memory._SLUG_MAX


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
    task_memory._save_playbook_sync(pb)
    loaded = task_memory.load_playbook("github.com", "search repos")
    assert loaded is not None
    assert loaded.host == "github.com"
    assert loaded.goal == "search repos"
    assert [s.action for s in loaded.steps] == ["navigate", "click"]
    assert loaded.steps[1].name == "Search"


def test_load_missing_returns_none() -> None:
    assert task_memory.load_playbook("nope.com", "never recorded") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    path = task_memory._safe_path("bad.com", "broken")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert task_memory.load_playbook("bad.com", "broken") is None


def test_path_escape_attempt_rejected() -> None:
    # A host crafted to climb out of the memory root must be sanitized, so the
    # resolved path stays inside the root rather than raising or escaping.
    path = task_memory._safe_path("../../etc", "x")
    root = task_memory._memory_root().resolve()
    assert root in path.parents


def test_record_noop_without_active_task() -> None:
    task_memory.record(task_memory.build_step("click"))
    assert not task_memory.is_recording()


def test_record_appends_eligible_step() -> None:
    task_memory.start("do a thing")
    task_memory.record(task_memory.build_step("navigate", params={"url": "u"}))
    assert task_memory.is_recording()


@pytest.mark.asyncio
async def test_finish_persists_clean_run() -> None:
    task_memory.start("login")
    task_memory.record(task_memory.build_step("navigate", params={"url": "u"}))
    task_memory.record(task_memory.build_step("click", role="button", name="Sign in"))
    msg = await task_memory.finish(success=True, host="example.com")
    assert "Saved playbook" in msg
    assert task_memory.load_playbook("example.com", "login") is not None


@pytest.mark.asyncio
async def test_ineligible_action_poisons_buffer() -> None:
    task_memory.start("scrape")
    task_memory.record(task_memory.build_step("navigate", params={"url": "u"}))
    task_memory.record(task_memory.build_step("eval"))  # excluded → poison
    msg = await task_memory.finish(success=True, host="example.com")
    assert "unsafe step" in msg
    assert task_memory.load_playbook("example.com", "scrape") is None


@pytest.mark.asyncio
async def test_error_result_poisons_buffer() -> None:
    task_memory.start("flaky")
    task_memory.record(task_memory.build_step("click", result="Error: ref not found"))
    msg = await task_memory.finish(success=True, host="example.com")
    assert "error" in msg.lower()
    assert task_memory.load_playbook("example.com", "flaky") is None


@pytest.mark.asyncio
async def test_finish_failure_discards() -> None:
    task_memory.start("abandon")
    task_memory.record(task_memory.build_step("navigate", params={"url": "u"}))
    msg = await task_memory.finish(success=False, host="example.com")
    assert "nothing saved" in msg
    assert task_memory.load_playbook("example.com", "abandon") is None


@pytest.mark.asyncio
async def test_finish_without_active_task() -> None:
    msg = await task_memory.finish(success=True, host="example.com")
    assert msg == "No task is being recorded."


def test_build_step_trims_result_and_drops_empty_params() -> None:
    step = task_memory.build_step(
        "type",
        params={"text": "hi", "dx": 0, "dy": 0},
        result="x" * 500,
    )
    assert step.params == {"text": "hi"}
    assert len(step.result_head) == task_memory._RESULT_HEAD_MAX


def test_format_playbook_is_numbered() -> None:
    pb = Playbook(
        host="h.com",
        goal="g",
        created_ts=1.0,
        steps=[Step(action="click", role="button", name="OK")],
    )
    out = task_memory.format_playbook(pb)
    assert "1. click" in out
    assert "OK" in out
