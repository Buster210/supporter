from __future__ import annotations

import re

import pytest

from supporter.tools.browser import snapshot

_REF = re.compile(r"\[ref=(e\d+)\]")

_SAMPLE = """- generic [active] [ref=e1]:
  - heading "Benchmark Page" [level=1] [ref=e2]
  - generic [ref=e3]:
    - button "Button 0" [ref=e4]
    - textbox [ref=e9]
    - paragraph [ref=e10]: idle"""

_NAV = """- generic [ref=e1]:
  - link "Home" [ref=e88] [cursor=pointer]:
    - /url: /feed/home
    - generic [ref=e93]: Home
  - img [ref=e50]
  - generic [ref=e60]:
    - button "Search" [ref=e35] [cursor=pointer]"""

_FEED = """- generic [ref=e1]:
  - generic [ref=e10]:
    - link "Video A" [ref=e11]:
      - /url: /watch?v=a
    - button "More" [ref=e12]
  - generic [ref=e20]:
    - link "Video B" [ref=e21]:
      - /url: /watch?v=b
    - button "More" [ref=e22]
  - generic [ref=e30]:
    - link "Video C" [ref=e31]:
      - /url: /watch?v=c
    - button "More" [ref=e32]"""

_SCAFFOLD = """- generic [ref=e1]:
  - generic [ref=e10]:
    - link "Ad" [ref=e11]:
      - /url: /promo
    - button "Close" [ref=e12]
  - generic [ref=e20]:
    - link "Ad" [ref=e21]:
      - /url: /promo
    - button "Close" [ref=e22]
  - generic [ref=e30]:
    - link "Ad" [ref=e31]:
      - /url: /promo
    - button "Close" [ref=e32]"""


def _interactive_refs(text: str) -> set[str]:
    refs = set()
    for line in text.splitlines():
        match = _REF.search(line)
        if match and re.search(
            r"- (button|link|textbox|combobox|checkbox|tab|option|menuitem)\b", line
        ):
            refs.add(match.group(1))
    return refs


def test_filter_interactive_keeps_controls_and_bridges() -> None:
    out = snapshot.filter_interactive(_SAMPLE)
    assert "button" in out
    assert "textbox" in out
    assert "heading" not in out
    assert "paragraph" not in out


def test_filter_interactive_empty_input() -> None:
    assert snapshot.filter_interactive("") == ""


def test_filter_interactive_no_controls_returns_empty() -> None:
    text = '- heading "Title" [ref=e2]\n  - paragraph [ref=e3]: body'
    assert snapshot.filter_interactive(text) == ""


def test_filter_interactive_strips_cursor_flag() -> None:
    out = snapshot.filter_interactive(_NAV)
    assert "cursor=pointer" not in out
    assert "Search" in out


def test_clean_snapshot_preserves_every_interactive_ref() -> None:
    cleaned = snapshot.clean_snapshot(_NAV)
    assert _interactive_refs(_NAV) <= _interactive_refs(cleaned)


def test_clean_snapshot_strips_refs_from_non_interactive_nodes() -> None:
    cleaned = snapshot.clean_snapshot(_SAMPLE)
    assert "Benchmark Page" in cleaned
    heading_line = next(ln for ln in cleaned.splitlines() if "Benchmark Page" in ln)
    assert "[ref=" not in heading_line
    button_line = next(line for line in cleaned.splitlines() if "Button 0" in line)
    assert "[ref=e4]" in button_line


def test_clean_snapshot_folds_urls_inline() -> None:
    cleaned = snapshot.clean_snapshot(_NAV)
    assert "/url:/feed/home" in cleaned.replace(" ", "")
    assert not any(line.strip().startswith("- /url:") for line in cleaned.splitlines())


def test_clean_snapshot_drops_unnamed_image_and_cursor() -> None:
    cleaned = snapshot.clean_snapshot(_NAV)
    assert "cursor=pointer" not in cleaned
    assert "- img [ref=e50]" not in cleaned


def test_clean_snapshot_drops_name_echo_child() -> None:
    cleaned = snapshot.clean_snapshot(_NAV)
    assert cleaned.count("Home") == 1


_CARD = (
    '- heading "Big Title 41 minutes" [level=3] [ref=e1]:\n'
    '  - link "Big Title 41 minutes" [ref=e528]:\n'
    "    - /url: /watch?v=ID\n"
    "    - text: Big Title"
)


def test_clean_snapshot_folds_heading_onto_link() -> None:
    cleaned = snapshot.clean_snapshot(_CARD)
    lines = cleaned.splitlines()
    fold = next(ln for ln in lines if "Big Title 41 minutes" in ln)
    assert "[level=3]" in fold
    assert "[ref=e528]" in fold
    assert "/watch?v=ID" in fold.replace(" ", "")
    assert "::" not in fold
    assert fold.count("[ref=") == 1
    assert "[ref=e1]" not in fold
    assert sum("Big Title" in ln for ln in lines) == 1


def test_clean_snapshot_drops_text_prefix_echo() -> None:
    cleaned = snapshot.clean_snapshot(_CARD)
    assert "- text:" not in cleaned


def test_clean_snapshot_keeps_heading_with_distinct_link() -> None:
    text = (
        '- heading "Section A" [level=2] [ref=e1]:\n'
        '  - link "Go elsewhere" [ref=e2]:\n'
        "    - /url: /other"
    )
    cleaned = snapshot.clean_snapshot(text)
    assert "Section A" in cleaned
    assert "Go elsewhere" in cleaned
    assert "[ref=e2]" in cleaned


def test_clean_snapshot_keeps_distinct_feed_cards() -> None:
    cleaned = snapshot.clean_snapshot(_FEED)
    assert "more similar" not in cleaned
    for title in ("Video A", "Video B", "Video C"):
        assert title in cleaned
    for href in ("/watch?v=a", "/watch?v=b", "/watch?v=c"):
        assert href in cleaned.replace(" ", "")


def test_clean_snapshot_dedups_content_identical_scaffold() -> None:
    cleaned = snapshot.clean_snapshot(_SCAFFOLD)
    assert "more similar" in cleaned
    assert "Ad" in cleaned
    assert "/promo" in cleaned.replace(" ", "")


def test_clean_snapshot_flattens_bare_structural_wrappers() -> None:
    nested = (
        "- generic [ref=e1]:\n"
        "  - generic [ref=e2]:\n"
        "    - generic [ref=e3]:\n"
        '      - button "Deep" [ref=e9]'
    )
    cleaned = snapshot.clean_snapshot(nested)
    assert cleaned == '- button "Deep" [ref=e9]'


def test_clean_snapshot_trims_tracking_params_keeps_target() -> None:
    link = (
        '- link "Vid" [ref=e11]:\n'
        "  - /url: /watch?v=ABC123&pp=track&si=session&list=PL9"
    )
    cleaned = snapshot.clean_snapshot(link).replace(" ", "")
    assert "/watch?v=ABC123" in cleaned
    assert "pp=" not in cleaned and "si=" not in cleaned and "list=" not in cleaned


def test_clean_snapshot_strips_dead_trailing_colon() -> None:
    text = '- button "Guide" [ref=e9]:\n  - img [ref=e50]'
    cleaned = snapshot.clean_snapshot(text)
    assert cleaned == '- button "Guide" [ref=e9]'


def test_clean_snapshot_keeps_colon_when_children_survive() -> None:
    text = '- button "Menu" [ref=e9]:\n  - link "Item" [ref=e10]:\n    - /url: /x'
    cleaned = snapshot.clean_snapshot(text)
    menu_line = next(ln for ln in cleaned.splitlines() if "Menu" in ln)
    assert menu_line.rstrip().endswith(":")


def test_clean_snapshot_drops_root_href_keeps_link() -> None:
    text = '- link "Home" [ref=e14]:\n  - /url: /'
    cleaned = snapshot.clean_snapshot(text)
    assert "[ref=e14]" in cleaned
    assert "/url:" not in cleaned


def test_clean_snapshot_drops_ad_host_url_keeps_ref() -> None:
    text = '- link "Ad" [ref=e9]:\n  - /url: https://www.googleadservices.com/pagead/aclk?ai=huge&sig=blob'
    cleaned = snapshot.clean_snapshot(text)
    assert "[ref=e9]" in cleaned
    assert "googleadservices" not in cleaned
    assert "/url:" not in cleaned


def test_clean_snapshot_collapses_self_host_url_to_path() -> None:
    text = '- link "Watch" [ref=e9]:\n  - /url: https://www.youtube.com/watch?v=ID'
    cleaned = snapshot.clean_snapshot(text, "https://www.youtube.com/").replace(" ", "")
    assert "/url:/watch?v=ID" in cleaned
    assert "youtube.com" not in cleaned


def test_clean_snapshot_keeps_foreign_host_url_absolute() -> None:
    text = '- link "Music" [ref=e9]:\n  - /url: https://music.youtube.com/'
    cleaned = snapshot.clean_snapshot(text, "https://www.youtube.com/").replace(" ", "")
    assert "https://music.youtube.com/" in cleaned


def test_clean_snapshot_passthrough_on_unparseable() -> None:
    assert snapshot.clean_snapshot("not a tree") == "not a tree"
    assert snapshot.clean_snapshot("") == ""


def test_clean_snapshot_skips_blank_lines() -> None:
    text = '- button "OK" [ref=e2]\n\n- link "Next" [ref=e3]'
    cleaned = snapshot.clean_snapshot(text)
    assert "OK" in cleaned
    assert "Next" in cleaned


def _clear_diff_state() -> None:
    snapshot._LAST_SNAPSHOT.clear()


def test_diff_snapshot_first_call_stores_baseline() -> None:
    _clear_diff_state()
    out = snapshot.diff_snapshot("k", '- button "A" [ref=e1]')
    assert "baseline stored" in out
    assert snapshot._LAST_SNAPSHOT["k"] == '- button "A" [ref=e1]'


def test_diff_snapshot_reports_added_and_removed_lines() -> None:
    _clear_diff_state()
    snapshot.diff_snapshot("k", '- button "A" [ref=e1]\n- button "B" [ref=e2]')
    out = snapshot.diff_snapshot("k", '- button "A" [ref=e1]\n- button "C" [ref=e3]')
    assert '-- button "B" [ref=e2]' in out
    assert '+- button "C" [ref=e3]' in out
    assert "@@" not in out
    assert not any(ln.startswith(("+++", "---")) for ln in out.splitlines())


def test_diff_snapshot_no_changes() -> None:
    _clear_diff_state()
    snapshot.diff_snapshot("k", '- button "A" [ref=e1]')
    assert "no changes" in snapshot.diff_snapshot("k", '- button "A" [ref=e1]')


def test_diff_snapshot_updates_baseline_each_call() -> None:
    _clear_diff_state()
    snapshot.diff_snapshot("k", "v1")
    snapshot.diff_snapshot("k", "v2")
    assert "no changes" in snapshot.diff_snapshot("k", "v2")


def test_diff_snapshot_keys_are_independent() -> None:
    _clear_diff_state()
    snapshot.diff_snapshot("a", "page-a")
    out = snapshot.diff_snapshot("b", "page-b")
    assert "baseline stored" in out


def test_forget_snapshot_drops_baseline() -> None:
    _clear_diff_state()
    snapshot.remember_snapshot("k", "v1")
    snapshot.forget_snapshot("k")
    assert "baseline stored" in snapshot.diff_snapshot("k", "v1")


def test_has_baseline_tracks_stored_keys() -> None:
    _clear_diff_state()
    assert not snapshot.has_baseline("k")
    assert not snapshot.has_baseline("")
    snapshot.remember_snapshot("k", "v1")
    assert snapshot.has_baseline("k")
    snapshot.forget_snapshot("k")
    assert not snapshot.has_baseline("k")


def test_log_snapshot_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> object:
        raise OSError("disk full")

    monkeypatch.setattr(snapshot, "_browser_log_path", boom)
    snapshot.log_snapshot("navigate", "content")
