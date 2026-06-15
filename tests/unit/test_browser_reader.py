from __future__ import annotations

from supporter.tools.browser import reader


def test_parse_urls_single() -> None:
    assert reader._parse_urls("https://a.com/x") == ["https://a.com/x"]


def test_parse_urls_multiple_whitespace_and_newlines() -> None:
    raw = "https://a.com/x  http://b.org/y\nhttps://c.net/z"
    assert reader._parse_urls(raw) == [
        "https://a.com/x",
        "http://b.org/y",
        "https://c.net/z",
    ]


def test_parse_urls_none_when_no_http() -> None:
    assert reader._parse_urls("just some text, no link") == []
    assert reader._parse_urls("") == []
    assert reader._parse_urls(None) == []  # type: ignore[arg-type]


def test_format_read_full_block() -> None:
    data = {
        "title": "Hello World",
        "url": "https://ex.com/post",
        "siteName": "Example",
        "byline": "Jane Doe",
        "published": "2026-01-01",
        "markdown": "First paragraph.\n\nSecond paragraph.",
        "links": [{"text": "Ref", "href": "https://ref.com"}],
    }
    out = reader._format_read(data)
    assert "# Hello World" in out
    assert "Source: https://ex.com/post" in out
    assert "Site: Example" in out
    assert "By: Jane Doe" in out
    assert "Published: 2026-01-01" in out
    assert "First paragraph." in out
    assert "## Links on this page" in out
    assert "- Ref -> https://ref.com" in out


def test_format_read_truncates_at_cap() -> None:
    data = {"title": "T", "markdown": "x" * 100, "links": []}
    out = reader._format_read(data, char_cap=20)
    assert "truncated" in out
    assert out.count("x") == 20


def test_format_read_empty_markdown_placeholder() -> None:
    out = reader._format_read({"title": "T", "markdown": "", "links": []})
    assert "no readable article text" in out


def test_format_read_non_dict_is_error() -> None:
    assert reader._format_read("nope").startswith("Error")  # type: ignore[arg-type]


def test_format_read_skips_malformed_links() -> None:
    data = {
        "markdown": "body",
        "links": ["not a dict", {"href": ""}, {"href": "https://ok.com"}],
    }
    out = reader._format_read(data)
    assert "- https://ok.com -> https://ok.com" in out
    assert "not a dict" not in out


def test_format_links_lists_with_count() -> None:
    data = {
        "links": [
            {"text": "One", "href": "https://1.com"},
            {"text": "Two", "href": "https://2.com"},
        ]
    }
    out = reader._format_links(data)
    assert out.startswith("2 links:")
    assert "- One -> https://1.com" in out
    assert "- Two -> https://2.com" in out


def test_format_links_empty_placeholder() -> None:
    assert "no outbound links" in reader._format_links({"links": []})
    assert "no outbound links" in reader._format_links("nope")  # type: ignore[arg-type]


def test_format_links_falls_back_to_href_when_no_text() -> None:
    out = reader._format_links({"links": [{"href": "https://x.com"}]})
    assert "- https://x.com -> https://x.com" in out
