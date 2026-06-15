from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from supporter.tools.base import ToolError
from supporter.tools.browser import core, session, support
from supporter.tools.browser.core import BrowseRequest


def _req(**kw: Any) -> BrowseRequest:
    return BrowseRequest(action=kw.pop("action", "snapshot"), **kw)


def _wrap(action: str, body: Callable[[BrowseRequest], Awaitable[str]]) -> Any:
    return support._wrap_action_errors(action)(body)


async def test_wrap_passes_through_success() -> None:
    async def ok(_req: BrowseRequest) -> str:
        return "fine"

    assert await _wrap("noop", ok)(_req()) == "fine"


async def test_wrap_reraises_toolerror_unchanged() -> None:
    async def boom(_req: BrowseRequest) -> str:
        raise ToolError("already mapped")

    with pytest.raises(ToolError, match="already mapped"):
        await _wrap("noop", boom)(_req())


async def test_wrap_action_cap_runtimeerror_becomes_error_string() -> None:
    async def capped(_req: BrowseRequest) -> str:
        raise RuntimeError("Action cap of 30 reached")

    out = await _wrap("click", capped)(_req())
    assert out == "Error: Action cap of 30 reached"


async def test_wrap_other_runtimeerror_becomes_toolerror() -> None:
    async def other(_req: BrowseRequest) -> str:
        raise RuntimeError("nav loop stuck")

    with pytest.raises(ToolError, match="Browser action failed: nav loop stuck"):
        await _wrap("click", other)(_req())


async def test_wrap_generic_exception_names_the_action() -> None:
    async def kaboom(_req: BrowseRequest) -> str:
        raise ValueError("bad coords")

    with pytest.raises(ToolError, match="Browser action 'scroll' failed: bad coords"):
        await _wrap("scroll", kaboom)(_req())


_TREE = '- document [ref=e1]:\n  - button "OK" [ref=e2]'


def test_render_snapshot_lossless_when_not_compact() -> None:
    out = support._render_snapshot(
        _TREE, _req(compact=False), "", cleaned='- button "OK" [ref=e2]'
    )
    assert "OK" in out
    assert "interactive elements" not in out


def test_render_snapshot_compact_counts_interactive() -> None:
    out = support._render_snapshot(
        _TREE, _req(compact=True), " after click", cleaned=""
    )
    assert "interactive elements after click" in out
    assert "OK" in out


def test_render_snapshot_empty_tree_reports_empty_page() -> None:
    out = support._render_snapshot("", _req(compact=False), "", cleaned="")
    assert out == "(empty page)"


def test_page_key_empty_url_is_empty_string() -> None:
    class P:
        url = ""

    assert support._page_key(P()) == ""


def test_page_key_swallows_url_failure() -> None:
    class P:
        @property
        def url(self) -> str:
            raise RuntimeError("detached")

    assert support._page_key(P()) == ""


def test_baseline_key_is_stable_per_page() -> None:
    class P:
        pass

    page = P()
    first = support._page_baseline_key(page)
    second = support._page_baseline_key(page)
    assert first == second
    assert first.startswith("pg")


def test_baseline_key_differs_across_pages() -> None:
    class P:
        pass

    assert support._page_baseline_key(P()) != support._page_baseline_key(P())


def test_baseline_key_empty_when_weakref_unsupported() -> None:
    assert support._page_baseline_key(123) == ""


def test_diff_header_strips_scheme() -> None:
    assert (
        support._diff_header("https://x.test/page")
        == "diff vs last snapshot (x.test/page):"
    )


def test_diff_header_empty_key() -> None:
    assert support._diff_header("") == "diff vs last snapshot:"


def test_diff_header_truncates_long_tail() -> None:
    key = "https://x.test/" + "a" * 100
    header = support._diff_header(key)
    assert header.endswith("...):")
    inner = header[len("diff vs last snapshot (") : -len("):")]
    assert len(inner) == 60
    assert inner.endswith("...")


def test_render_script_result_json_roundtrip() -> None:
    assert support._render_script_result({"a": 1}) == json.dumps({"a": 1}, default=str)


def test_render_script_result_falls_back_to_str() -> None:
    class NotJson:
        def __str__(self) -> str:
            return "OPAQUE"

    assert "OPAQUE" in support._render_script_result(NotJson())


def test_render_script_result_truncates_over_config_cap() -> None:
    out = support._render_script_result("z" * 50_000)
    assert "…(truncated:" in out
    assert out.endswith(" more chars)")


def test_render_script_result_fallback_on_json_failure() -> None:
    deep: Any = []
    for _ in range(2000):
        deep = [deep]
    out = support._render_script_result(deep)
    assert isinstance(out, str)
    assert len(out) > 0


def test_validate_path_accepts_in_project_relative() -> None:
    resolved, err = support._validate_path_or_error("notes.txt")
    assert err is None
    assert resolved is not None
    assert str(resolved).endswith("notes.txt")


async def test_resolve_role_and_name_returns_empty_on_failure() -> None:
    class _BrokenLocator:
        async def evaluate(self, _script: str) -> Any:
            msg = "element detached"
            raise RuntimeError(msg)

    role, name = await support._resolve_role_and_name(_BrokenLocator(), "e5")
    assert role == ""
    assert name == ""


def test_validate_path_rejects_traversal() -> None:
    resolved, err = support._validate_path_or_error("../../etc/passwd")
    assert resolved is None
    assert err is not None
    assert err.startswith("Error: ")
    assert "outside project root" in err


async def test_page_or_error_raises_tool_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raise_get_session() -> Any:
        msg = "connection refused"
        raise RuntimeError(msg)

    monkeypatch.setattr(session, "get_session", raise_get_session)

    with pytest.raises(ToolError, match="Browser session failed"):
        await support._page_or_error()


async def test_session_parts_raises_tool_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def raise_get_session() -> Any:
        msg = "timeout"
        raise RuntimeError(msg)

    monkeypatch.setattr(session, "get_session", raise_get_session)

    with pytest.raises(ToolError, match="Browser session failed"):
        await support._session_parts()


async def test_page_host_reads_url_property() -> None:
    class _Page:
        url = "https://www.GitHub.com/foo?x=1"

    assert await core._page_host(_Page()) == "github.com"


async def test_page_host_returns_empty_on_url_failure() -> None:
    class _BrokenPage:
        @property
        def url(self) -> str:
            raise RuntimeError("page crashed")

    assert await core._page_host(_BrokenPage()) == ""
