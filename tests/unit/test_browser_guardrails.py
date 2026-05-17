from __future__ import annotations

import pytest

from supporter.tools.browser import guardrails


@pytest.mark.parametrize("action", ["navigate", "back", "snapshot", "screenshot"])
def test_read_only_actions_never_confirm(action: str) -> None:
    assert (
        guardrails.needs_confirmation(action, "button", "Submit", "example.com")
        is False
    )


def test_sensitive_domain_forces_confirm_on_any_write_action() -> None:
    assert guardrails.needs_confirmation("click", "link", "Home", "x.com") is True
    assert (
        guardrails.needs_confirmation("type", "textbox", "search", "github.com") is True
    )


def test_www_prefix_already_stripped_by_caller_is_matched() -> None:
    assert guardrails.needs_confirmation("click", "", "x", "twitter.com") is True


def test_click_submit_button_confirms() -> None:
    assert (
        guardrails.needs_confirmation("click", "button", "Submit", "example.com")
        is True
    )
    assert (
        guardrails.needs_confirmation("click", "button", "Send message", "example.com")
        is True
    )
    assert (
        guardrails.needs_confirmation("click", "button", "Sign-in", "example.com")
        is True
    )


def test_click_benign_button_does_not_confirm() -> None:
    assert (
        guardrails.needs_confirmation("click", "button", "Cancel", "example.com")
        is False
    )
    assert (
        guardrails.needs_confirmation("click", "link", "Read more", "example.com")
        is False
    )


def test_word_boundary_rejects_embedded_matches() -> None:
    # "buyer" contains "buy", "compass" contains "pass" — must NOT confirm.
    assert (
        guardrails.needs_confirmation("click", "button", "Become a buyer", "ex.com")
        is False
    )
    assert (
        guardrails.needs_confirmation("type", "textbox", "compass heading", "ex.com")
        is False
    )


def test_password_role_always_confirms_even_without_name() -> None:
    assert guardrails.needs_confirmation("type", "password", "", "example.com") is True


def test_type_into_sensitive_named_field_confirms() -> None:
    assert (
        guardrails.needs_confirmation("type", "textbox", "Email", "example.com") is True
    )
    assert (
        guardrails.needs_confirmation("type", "textbox", "card number", "example.com")
        is True
    )


def test_type_into_plain_textbox_does_not_confirm() -> None:
    assert (
        guardrails.needs_confirmation("type", "textbox", "Search", "example.com")
        is False
    )


def test_type_sensitive_name_but_non_field_role_ignored() -> None:
    # name says "password" but it is a button, not an editable field.
    assert (
        guardrails.needs_confirmation("type", "button", "password", "example.com")
        is False
    )


def test_random_gap_within_bounds() -> None:
    for _ in range(200):
        gap = guardrails.random_gap()
        assert guardrails.GAP_MIN <= gap <= guardrails.GAP_MAX


def test_host_from_url_strips_www_and_lowercases() -> None:
    assert guardrails._host_from_url("https://WWW.Example.COM/path") == "example.com"
    assert guardrails._host_from_url("not a url") == ""
