from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from supporter.tools.browser import guardrails

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator


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


def test_random_gap_is_not_uniform() -> None:
    # Log-normal clusters near the median; a flat uniform over [0.8, 2.5] would
    # put ~50% of draws above the 1.65 midpoint. We expect a clear majority below.
    samples = [guardrails.random_gap() for _ in range(2000)]
    midpoint = (guardrails.GAP_MIN + guardrails.GAP_MAX) / 2
    below = sum(1 for g in samples if g < midpoint)
    assert below > len(samples) * 0.6


def test_action_cap_within_jitter_band() -> None:
    lo = guardrails.ACTION_CAP
    hi = guardrails.ACTION_CAP + guardrails.ACTION_CAP_JITTER
    caps = {guardrails.action_cap() for _ in range(500)}
    assert all(lo <= c <= hi for c in caps)
    assert len(caps) > 1  # actually jittered, not a constant


def test_rate_throttle_zero_under_budget() -> None:
    # 5 actions over a 60s window is well under the 20/min ceiling → no throttle.
    assert guardrails.rate_throttle_delay(5, 60.0) == 0.0


def test_rate_throttle_positive_over_budget() -> None:
    # 20 actions crammed into 10s is way over budget → must inject delay.
    delay = guardrails.rate_throttle_delay(20, 10.0)
    assert delay > 0.0


def test_rate_throttle_pulls_rate_to_ceiling() -> None:
    # After adding the returned delay, the effective rate lands on the ceiling.
    count, window = 30, 5.0
    delay = guardrails.rate_throttle_delay(count, window)
    effective_rate = count / (window + delay) * 60.0
    assert effective_rate <= guardrails.ACTIONS_PER_MINUTE_MAX + 1e-9


def test_rate_throttle_nonpositive_count_is_zero() -> None:
    assert guardrails.rate_throttle_delay(0, 60.0) == 0.0


def test_rate_throttle_single_action_never_throttles() -> None:
    # One timestamp (or a zero-length window) has no measurable rate, so the
    # first action of a session must not be charged a spurious throttle.
    assert guardrails.rate_throttle_delay(1, 0.0) == 0.0
    assert guardrails.rate_throttle_delay(5, 0.0) == 0.0


def test_maybe_idle_gap_within_range_or_zero() -> None:
    lo, hi = guardrails.SESSION_IDLE_GAP_RANGE
    fired = False
    for _ in range(5000):
        g = guardrails.maybe_idle_gap()
        assert g == 0.0 or lo <= g <= hi
        fired = fired or g > 0.0
    assert fired  # at ~4% over 5000 draws it must trigger sometimes


def test_fatigue_multiplier_monotonic_and_clamped() -> None:
    prev = 0.0
    for minutes in range(0, 120, 2):
        m = guardrails.fatigue_multiplier(float(minutes))
        assert m >= prev - 1e-9  # non-decreasing
        assert 1.0 <= m <= 1.0 + guardrails.FATIGUE_MAX_BONUS + 1e-9
        prev = m
    # Negative age never dips below 1.0.
    assert guardrails.fatigue_multiplier(-10.0) == 1.0


def test_next_tempo_stays_in_band() -> None:
    t = 1.0
    for _ in range(10_000):
        t = guardrails.next_tempo(t)
        assert guardrails.TEMPO_MIN <= t <= guardrails.TEMPO_MAX


def test_host_from_url_strips_www_and_lowercases() -> None:
    assert guardrails.host_from_url("https://WWW.Example.COM/path") == "example.com"
    assert guardrails.host_from_url("not a url") == ""


def test_configured_fast_hosts_run_fast() -> None:
    # The shipped allowlist: hosts confirmed to do no fingerprinting.
    assert guardrails.host_is_fast("google.com") is True
    assert guardrails.host_is_fast("www.google.com") is True  # www stripped
    assert guardrails.host_is_fast("gemini.google.com") is True
    # An unlisted host stays humanized.
    assert guardrails.host_is_fast("twitter.com") is False


def test_sensitive_subdomain_not_fast_under_exact_match() -> None:
    # The whole point of exact match: "google.com" being fast must NOT make the
    # sensitive login subdomain fast — it stays humanized.
    assert "google.com" in guardrails.FAST_HOSTS
    assert guardrails.host_is_fast("accounts.google.com") is False


def test_host_is_fast_exact_match_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guardrails, "FAST_HOSTS", frozenset({"example.com"}))
    assert guardrails.host_is_fast("example.com") is True
    assert guardrails.host_is_fast("www.example.com") is True  # www stripped
    # Subdomains are NOT matched — exact only.
    assert guardrails.host_is_fast("app.example.com") is False
    assert guardrails.host_is_fast("accounts.example.com") is False


def test_host_is_fast_rejects_non_members_and_lookalikes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guardrails, "FAST_HOSTS", frozenset({"example.com"}))
    assert guardrails.host_is_fast("other.com") is False
    assert guardrails.host_is_fast("notexample.com") is False
    assert guardrails.host_is_fast("example.com.evil.com") is False
    assert guardrails.host_is_fast("") is False


def test_actions_per_minute_max_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    monkeypatch.setenv("BROWSER_ACTIONS_PER_MIN", "55")
    importlib.reload(guardrails)
    assert guardrails.ACTIONS_PER_MINUTE_MAX == 55


def test_actions_per_minute_max_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    monkeypatch.delenv("BROWSER_ACTIONS_PER_MIN", raising=False)
    importlib.reload(guardrails)
    assert guardrails.ACTIONS_PER_MINUTE_MAX == 40


def test_session_idle_gap_range_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    monkeypatch.setenv("BROWSER_IDLE_GAP_MIN", "7.5")
    importlib.reload(guardrails)
    assert guardrails.SESSION_IDLE_GAP_RANGE == (7.5, 60.0)


def test_session_idle_gap_range_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    monkeypatch.delenv("BROWSER_IDLE_GAP_MIN", raising=False)
    importlib.reload(guardrails)
    assert guardrails.SESSION_IDLE_GAP_RANGE == (5.0, 60.0)


# --- register_browse_callback: independent-slot update + async dispatch ----


@pytest.fixture(autouse=True)
def _restore_callbacks() -> Iterator[None]:
    saved_confirm = guardrails.browse_confirmation_callback
    saved_sink = guardrails.browse_image_sink
    try:
        yield
    finally:
        guardrails.browse_confirmation_callback = saved_confirm
        guardrails.browse_image_sink = saved_sink


async def _yes(_summary: str, _prompt: str) -> bool:
    return True


async def _sink(_data: bytes, _caption: str) -> None:
    return None


def test_register_sets_both_slots() -> None:
    guardrails.register_browse_callback(confirmation=_yes, image_sink=_sink)
    assert guardrails.browse_confirmation_callback is _yes
    assert guardrails.browse_image_sink is _sink


def test_register_confirmation_only_keeps_image_sink() -> None:
    guardrails.register_browse_callback(image_sink=_sink)
    guardrails.register_browse_callback(confirmation=_yes)
    # The second call omits image_sink (_UNSET) so it must not clobber it.
    assert guardrails.browse_confirmation_callback is _yes
    assert guardrails.browse_image_sink is _sink


def test_register_image_sink_only_keeps_confirmation() -> None:
    guardrails.register_browse_callback(confirmation=_yes)
    guardrails.register_browse_callback(image_sink=_sink)
    assert guardrails.browse_confirmation_callback is _yes
    assert guardrails.browse_image_sink is _sink


def test_register_no_args_is_noop() -> None:
    guardrails.register_browse_callback(confirmation=_yes, image_sink=_sink)
    guardrails.register_browse_callback()
    assert guardrails.browse_confirmation_callback is _yes
    assert guardrails.browse_image_sink is _sink


def test_register_explicit_none_clears_slot() -> None:
    guardrails.register_browse_callback(confirmation=_yes)
    guardrails.register_browse_callback(confirmation=None)
    assert guardrails.browse_confirmation_callback is None


async def test_registered_confirmation_is_awaitable_and_returns_bool() -> None:
    guardrails.register_browse_callback(confirmation=_yes)
    cb: Callable[[str, str], Awaitable[bool]] | None = (
        guardrails.browse_confirmation_callback
    )
    assert cb is not None
    assert await cb("summary", "prompt?") is True


async def test_registered_image_sink_is_awaitable() -> None:
    captured: list[tuple[bytes, str]] = []

    async def recorder(data: bytes, caption: str) -> None:
        captured.append((data, caption))

    guardrails.register_browse_callback(image_sink=recorder)
    sink = guardrails.browse_image_sink
    assert sink is not None
    await sink(b"png", "a shot")
    assert captured == [(b"png", "a shot")]
