from __future__ import annotations

from supporter.tools.browser.playbook_match import (
    _find_ref,
    _find_ref_fuzzy,
    _fuzzy_find_goal,
    _name_match_score,
)
from supporter.tools.browser.playbook_store import _normalize_name


def test_normalize_name() -> None:
    assert _normalize_name("  Sign-In!!  Now ") == "sign in now"
    assert _normalize_name("Hello World") == "hello world"
    assert _normalize_name("") == ""


def test_name_match_score_exact() -> None:
    assert _name_match_score("ok", {"ok"}, "ok") == 1.0


def test_name_match_score_subset_tokens() -> None:
    score = _name_match_score("sign in", {"sign", "in"}, "sign in account")
    assert score > 0.6


def test_name_match_score_contains() -> None:
    score = _name_match_score("submit", {"submit"}, "submit form")
    assert score > 0.3


def test_name_match_score_no_match() -> None:
    score = _name_match_score("login", {"login"}, "logout")
    assert score == 0.0


def test_find_ref_exact_match() -> None:
    tree = '- button "OK" [ref=e1]\n- button "Cancel" [ref=e2]'
    assert _find_ref(tree, "button", "OK") == "e1"


def test_find_ref_no_match() -> None:
    tree = '- button "OK" [ref=e1]'
    assert _find_ref(tree, "button", "Submit") == ""


def test_find_ref_role_mismatch() -> None:
    tree = '- link "OK" [ref=e1]'
    assert _find_ref(tree, "button", "OK") == ""


def test_find_ref_fuzzy_partial_match() -> None:
    tree = '- button "Sign In" [ref=e1]\n- button "Submit" [ref=e2]'
    assert _find_ref_fuzzy(tree, "button", "Sign") == "e1"


def test_find_ref_fuzzy_exact_preferred() -> None:
    tree = '- button "Sign in account" [ref=e1]\n- button "Sign in" [ref=e2]'
    assert _find_ref_fuzzy(tree, "button", "Sign in") == "e2"


def test_find_ref_fuzzy_ambiguous_returns_empty() -> None:
    tree = '- button "Sign in" [ref=e1]\n- button "Sign in" [ref=e2]'
    assert _find_ref_fuzzy(tree, "button", "Sign in") == ""


def test_find_ref_fuzzy_no_match() -> None:
    tree = '- button "OK" [ref=e1]'
    assert _find_ref_fuzzy(tree, "button", "Logout") == ""


def test_fuzzy_find_goal_empty_host(tmp_path: object) -> None:
    assert _fuzzy_find_goal("nonexistent.host", "login") == []


def test_no_playbook_message_without_suggestions() -> None:
    from supporter.tools.browser.playbook_match import _no_playbook_message

    msg = _no_playbook_message("x.test", "login")
    assert "No playbook found" in msg
    assert "login" in msg


def test_find_ref_fuzzy_requires_exact_role() -> None:
    tree = '- link "Sign in" [ref=e1]'
    assert _find_ref_fuzzy(tree, "button", "Sign in") == ""


def test_find_ref_fuzzy_strips_punctuation() -> None:
    tree = '- button "Sign in!" [ref=e1]'
    assert _find_ref_fuzzy(tree, "button", "Sign in") == "e1"
