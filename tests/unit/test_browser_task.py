from __future__ import annotations

import pytest

from supporter.tools.browser import recorder
from supporter.tools.browser.playbook_match import (
    _find_ref,
    _find_ref_fuzzy,
)
from supporter.tools.browser.playbook_store import (
    _RESULT_HEAD_MAX,
    _SLUG_MAX,
    SCHEMA_VERSION,
    Playbook,
    Step,
    _normalize_name,
    build_step,
    format_playbook,
)
from supporter.tools.browser.recorder import (
    RECORDABLE_ACTIONS,
    discard,
    is_recording,
    record,
)


@pytest.fixture(autouse=True)
def _reset_active() -> None:
    discard()


def test_slug_sanitizes_to_safe_charset() -> None:
    from supporter.tools.browser.playbook_store import _slug

    assert _slug("Log in to GitHub!") == "log-in-to-github"
    assert _slug("a/../../etc/passwd") == "a-..-..-etc-passwd"
    assert _slug("") == "task"


def test_slug_is_length_bounded() -> None:
    from supporter.tools.browser.playbook_store import _slug

    assert len(_slug("x" * 500)) <= _SLUG_MAX


def test_record_noop_without_active_task() -> None:
    record(build_step("click"))
    assert not is_recording()


def test_record_appends_eligible_step() -> None:
    from supporter.tools.browser.recorder import start

    start("do a thing")
    record(build_step("navigate", params={"url": "u"}))
    assert is_recording()
    active = recorder._ACTIVE["main"]
    assert active is not None
    assert len(active.steps) == 1
    assert active.steps[0].action == "navigate"
    assert active.steps[0].params == {"url": "u"}
    assert active.failed is False


@pytest.mark.asyncio
async def test_finish_without_active_task() -> None:
    from supporter.tools.browser.recorder import finish

    msg = await finish(success=True, host="example.com")
    assert msg == "No task is being recorded."


def test_build_step_trims_result_and_drops_empty_params() -> None:
    step = build_step(
        "type",
        params={"text": "hi", "dx": 0, "dy": 0},
        result="x" * 500,
    )
    assert step.params == {"text": "hi"}
    assert len(step.result_head) == _RESULT_HEAD_MAX


def test_find_ref_matches_role_and_name() -> None:
    tree = (
        '- heading "Tabs" [level=1]\n'
        '- button "Click me" [ref=e3]\n'
        '- textbox "Search" [ref=e7]'
    )
    assert _find_ref(tree, "button", "Click me") == "e3"
    assert _find_ref(tree, "textbox", "Search") == "e7"


def test_find_ref_first_match_in_document_order() -> None:
    tree = '- button "Go" [ref=e1]\n- button "Go" [ref=e5]'
    assert _find_ref(tree, "button", "Go") == "e1"


def test_find_ref_returns_empty_on_no_match() -> None:
    tree = '- button "Click me" [ref=e3]'
    assert _find_ref(tree, "button", "Submit") == ""
    assert _find_ref(tree, "textbox", "Click me") == ""


def test_find_ref_returns_empty_on_diff_string() -> None:
    assert _find_ref("(no changes since last snapshot)", "button", "Go") == ""
    diff = 'diff vs last snapshot (x):\n+- button "Go" [ref=e3]\n-- paragraph: idle'
    assert _find_ref(diff, "button", "Go") == ""


def test_format_playbook_is_numbered() -> None:
    pb = Playbook(
        host="h.com",
        goal="g",
        created_ts=1.0,
        steps=[Step(action="click", role="button", name="OK")],
    )
    out = format_playbook(pb)
    assert "1. click" in out
    assert "OK" in out


def test_replay_params_substitutes_overrides() -> None:
    from supporter.tools.browser.task import _replay_params

    step = Step(
        action="type",
        role="textbox",
        name="username",
        params={"text": "${username}"},
        variable="username",
    )
    kwargs = _replay_params(step, overrides={"username": "admin"})
    assert kwargs["text"] == "admin"


def test_replay_params_preserves_selector() -> None:
    from supporter.tools.browser.task import _replay_params

    step = Step(action="click", selector="#submit")
    kwargs = _replay_params(step)
    assert kwargs["selector"] == "#submit"


def test_find_ref_fuzzy_name_match() -> None:
    tree = '- button "Sign In" [ref=e1]\n- button "Submit" [ref=e2]'
    assert _find_ref_fuzzy(tree, "button", "Sign In") == "e1"
    assert _find_ref_fuzzy(tree, "button", "Sign") == "e1"


def test_find_ref_fuzzy_returns_empty_on_no_match() -> None:
    tree = '- button "Sign In" [ref=e1]'
    assert _find_ref_fuzzy(tree, "button", "Logout") == ""


def test_find_ref_fuzzy_prefers_exact_over_partial() -> None:
    tree = '- button "Sign in account" [ref=e1]\n- button "Sign in" [ref=e2]'
    assert _find_ref_fuzzy(tree, "button", "Sign in") == "e2"


def test_find_ref_fuzzy_refuses_ambiguous_duplicates() -> None:
    tree = '- button "Sign in" [ref=e1]\n- button "Sign in" [ref=e2]'
    assert _find_ref_fuzzy(tree, "button", "Sign in") == ""


def test_step_variable_defaults_to_empty() -> None:
    step = Step(action="navigate")
    assert step.variable == ""


def test_v2_playbook_defaults() -> None:
    pb = Playbook(host="h.com", goal="g", created_ts=1.0, steps=[])
    assert pb.schema_version == SCHEMA_VERSION
    assert pb.variables == []
    assert pb.success_count == 0
    assert pb.fail_count == 0
    assert pb.last_used_ts == 0.0
    assert pb.last_outcome == ""


def test_replay_params_whole_value_substitution() -> None:
    from supporter.tools.browser.task import _replay_params

    step = Step(action="type", params={"text": "admin"}, variable="username")
    kwargs = _replay_params(step, overrides={"username": "alice"})
    assert kwargs["text"] == "alice"


def test_replay_params_without_override_keeps_recorded_value() -> None:
    from supporter.tools.browser.task import _replay_params

    step = Step(action="type", params={"text": "admin"}, variable="username")
    assert _replay_params(step)["text"] == "admin"
    assert _replay_params(step, overrides={"other": "x"})["text"] == "admin"


def test_replay_params_placeholder_preserves_surrounding_text() -> None:
    from supporter.tools.browser.task import _replay_params

    step = Step(
        action="navigate",
        params={"url": "https://x.test/u/${user}"},
        variable="user",
    )
    kwargs = _replay_params(step, overrides={"user": "alice"})
    assert kwargs["url"] == "https://x.test/u/alice"


def test_replay_params_multi_var_template_substitutes_each() -> None:
    from supporter.tools.browser.task import _replay_params

    step = Step(
        action="navigate",
        params={"url": "https://${user}.x.test/${repo}"},
        variable="user,repo",
    )
    kwargs = _replay_params(step, overrides={"user": "alice", "repo": "site"})
    assert kwargs["url"] == "https://alice.x.test/site"


def test_replay_params_template_var_not_clobbered_by_sibling() -> None:
    from supporter.tools.browser.task import _replay_params

    step = Step(
        action="type",
        params={"text": "admin@${domain}"},
        variable="username,domain",
    )
    kwargs = _replay_params(step, overrides={"username": "alice", "domain": "test.com"})
    assert kwargs["text"] == "admin@test.com"


def test_normalize_name_collapses_punctuation_and_space() -> None:
    assert _normalize_name("  Sign-In!!  Now ") == "sign in now"


def test_recordable_actions_include_navigation_steps() -> None:
    for action in ("back", "forward", "newtab", "frame"):
        assert action in RECORDABLE_ACTIONS


async def _run_replay_with_snapshot(monkeypatch: pytest.MonkeyPatch, snap: str) -> str:
    """Drive replay_playbook with a one-step playbook, mocking the live page.

    Returns the success message so callers can assert the final-page payload.
    """
    import time

    from supporter.tools.browser import support, task, tool
    from supporter.tools.browser.playbook_store import Playbook, build_step

    pb = Playbook(
        host="example.com",
        goal="do thing",
        created_ts=time.time(),
        steps=[build_step("click", selector="#btn")],
    )

    async def _host(_page: object) -> str:
        return "example.com"

    async def _browse(_action: str, **_kwargs: object) -> str:
        return "step-ok"

    async def _snapshot(_page: object) -> str:
        return snap

    async def _save(_pb: object) -> None:
        return None

    monkeypatch.setattr(task.session, "active_page", lambda: object())
    monkeypatch.setattr(task, "_page_host", _host)
    monkeypatch.setattr(task, "load_playbook", lambda host, goal: pb)
    monkeypatch.setattr(task, "save_playbook", _save)
    monkeypatch.setattr(tool, "browse", _browse)
    monkeypatch.setattr(support, "_live_refs_snapshot", _snapshot)

    from supporter.tools.browser.task import replay_playbook

    return await replay_playbook("do thing")


@pytest.mark.asyncio
async def test_replay_playbook_success_return_includes_final_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D2: success return carries the collected final-page data, not just status."""
    result = await _run_replay_with_snapshot(monkeypatch, "FINAL_SNAPSHOT_XYZ")
    assert "1/1 steps succeeded" in result
    assert "Final page:" in result
    assert "FINAL_SNAPSHOT_XYZ" in result


@pytest.mark.asyncio
async def test_replay_playbook_final_page_truncates_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D2: an oversized final snapshot is capped with a visible truncation marker."""
    from supporter.config import config

    big = "z" * (config.browse_page_chars_cap + 5000)
    result = await _run_replay_with_snapshot(monkeypatch, big)
    assert "…(truncated:" in result
    assert "more chars)" in result
