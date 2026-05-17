from __future__ import annotations

from supporter.tools.browser import snapshot

_SAMPLE = """- generic [active] [ref=e1]:
  - heading "Benchmark Page" [level=1] [ref=e2]
  - generic [ref=e3]:
    - button "Button 0" [ref=e4]
    - textbox [ref=e9]
    - paragraph [ref=e10]: idle"""


def test_filter_interactive_keeps_only_ref_lines() -> None:
    out = snapshot.filter_interactive(_SAMPLE)
    lines = out.splitlines()
    assert all("[ref=e" in line for line in lines)
    # the bare `generic [ref=e3]:` line also has a ref and is retained
    assert any("button" in line for line in lines)
    assert any("textbox" in line for line in lines)
    # the non-ref container line "- generic [active]..." has a ref too (e1) -> kept
    assert "heading" in out


def test_filter_interactive_empty_input() -> None:
    assert snapshot.filter_interactive("") == ""


def test_filter_interactive_no_refs_returns_empty() -> None:
    assert snapshot.filter_interactive("- generic:\n  - text only") == ""


def test_filter_compact_passthrough_under_limit() -> None:
    assert snapshot.filter_compact(_SAMPLE) == snapshot.filter_interactive(_SAMPLE)


def test_filter_compact_truncates_over_limit() -> None:
    big = "\n".join(f'- button "B{i}" [ref=e{i}]' for i in range(100))
    out = snapshot.filter_compact(big, max_lines=40)
    lines = out.splitlines()
    assert any("lines omitted" in line for line in lines)
    assert len(lines) <= 41  # 40 + the omitted marker
    assert lines[0].endswith("[ref=e0]")
    assert lines[-1].endswith("[ref=e99]")
