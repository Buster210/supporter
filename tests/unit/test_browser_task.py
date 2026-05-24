from __future__ import annotations

import pytest

from supporter.tools.browser import task
from supporter.tools.browser.task import Playbook, Step


@pytest.fixture(autouse=True)
def _reset_active() -> None:
    task.discard()


def test_slug_sanitizes_to_safe_charset() -> None:
    assert task._slug("Log in to GitHub!") == "log-in-to-github"
    assert task._slug("a/../../etc/passwd") == "a-..-..-etc-passwd"
    assert task._slug("") == "task"


def test_slug_is_length_bounded() -> None:
    assert len(task._slug("x" * 500)) <= task._SLUG_MAX


def test_record_noop_without_active_task() -> None:
    task.record(task.build_step("click"))
    assert not task.is_recording()


def test_record_appends_eligible_step() -> None:
    task.start("do a thing")
    task.record(task.build_step("navigate", params={"url": "u"}))
    assert task.is_recording()
    # Inspect the buffered step, not just the recording flag: a recordable
    # action must actually land in the active buffer, with its fields intact and
    # the buffer left savable (failed stays False).
    active = task._ACTIVE
    assert active is not None
    assert len(active.steps) == 1
    assert active.steps[0].action == "navigate"
    assert active.steps[0].params == {"url": "u"}
    assert active.failed is False


@pytest.mark.asyncio
async def test_finish_without_active_task() -> None:
    msg = await task.finish(success=True, host="example.com")
    assert msg == "No task is being recorded."


def test_build_step_trims_result_and_drops_empty_params() -> None:
    step = task.build_step(
        "type",
        params={"text": "hi", "dx": 0, "dy": 0},
        result="x" * 500,
    )
    assert step.params == {"text": "hi"}
    assert len(step.result_head) == task._RESULT_HEAD_MAX


def test_find_ref_matches_role_and_name() -> None:
    tree = (
        '- heading "Tabs" [level=1]\n'
        '- button "Click me" [ref=e3]\n'
        '- textbox "Search" [ref=e7]'
    )
    assert task._find_ref(tree, "button", "Click me") == "e3"
    assert task._find_ref(tree, "textbox", "Search") == "e7"


def test_find_ref_first_match_in_document_order() -> None:
    tree = '- button "Go" [ref=e1]\n- button "Go" [ref=e5]'
    assert task._find_ref(tree, "button", "Go") == "e1"


def test_find_ref_returns_empty_on_no_match() -> None:
    tree = '- button "Click me" [ref=e3]'
    assert task._find_ref(tree, "button", "Submit") == ""
    assert task._find_ref(tree, "textbox", "Click me") == ""


def test_find_ref_returns_empty_on_diff_string() -> None:
    # Replay must read a FULL tree, never a diff. A diff/no-changes string carries
    # no parseable role lines, so _find_ref yields "" — guarding the bug where
    # replay fed it a baseline-diff instead of the live tree.
    assert task._find_ref("(no changes since last snapshot)", "button", "Go") == ""
    diff = 'diff vs last snapshot (x):\n+- button "Go" [ref=e3]\n-- paragraph: idle'
    assert task._find_ref(diff, "button", "Go") == ""


def test_format_playbook_is_numbered() -> None:
    pb = Playbook(
        host="h.com",
        goal="g",
        created_ts=1.0,
        steps=[Step(action="click", role="button", name="OK")],
    )
    out = task.format_playbook(pb)
    assert "1. click" in out
    assert "OK" in out
