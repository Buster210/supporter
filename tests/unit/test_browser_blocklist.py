from __future__ import annotations

from typing import Any

from supporter.tools.browser import blocklist, session


def test_host_blocked_matches_exact_and_subdomain() -> None:
    assert blocklist.host_blocked("https://doubleclick.net/x")
    assert blocklist.host_blocked("https://ads.g.doubleclick.net/pagead")
    assert blocklist.host_blocked("https://static.criteo.com/js/ld.js")


def test_host_blocked_ignores_substring_in_path() -> None:
    assert not blocklist.host_blocked("https://example.com/criteo.com/track")


def test_host_blocked_unlisted_and_empty() -> None:
    assert not blocklist.host_blocked("https://github.com/user/repo")
    assert not blocklist.host_blocked("not a url")
    assert not blocklist.host_blocked("")


def test_host_listed_subdomain_recovers_g_doubleclick() -> None:
    assert blocklist.host_listed("g.doubleclick.net")
    assert blocklist.host_listed("doubleclick.net")
    assert not blocklist.host_listed("example.com")
    assert not blocklist.host_listed("")


def test_analytics_not_blocked() -> None:
    assert not blocklist.host_blocked("https://script.hotjar.com/x.js")
    assert not blocklist.host_blocked("https://cdn.segment.com/analytics.js")


def test_should_block_resources_default_true() -> None:
    assert blocklist.should_block_resources() is True


def test_should_block_resources_disabled(monkeypatch: Any) -> None:
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("BROWSER_BLOCK_RESOURCES", value)
        assert blocklist.should_block_resources() is False


def test_blocked_types() -> None:
    assert "media" in blocklist.BLOCKED_TYPES
    assert "font" in blocklist.BLOCKED_TYPES
    assert "document" not in blocklist.BLOCKED_TYPES


class _FakeRoute:
    def __init__(self, url: str, resource_type: str) -> None:
        self.request = type("Req", (), {"url": url, "resource_type": resource_type})()
        self.actions: list[str] = []

    async def abort(self) -> None:
        self.actions.append("abort")

    async def continue_(self) -> None:
        self.actions.append("continue")


class _FakeContext:
    def __init__(self) -> None:
        self.handler: Any = None
        self.pattern: str | None = None

    async def route(self, pattern: str, handler: Any) -> None:
        self.pattern = pattern
        self.handler = handler


async def _handler_for() -> Any:
    ctx = _FakeContext()
    await session._install_block_route(ctx)
    assert ctx.pattern == "**/*"
    return ctx.handler


async def test_block_route_aborts_ad_host() -> None:
    handler = await _handler_for()
    route = _FakeRoute("https://doubleclick.net/ad.js", "script")
    await handler(route)
    assert route.actions == ["abort"]


async def test_block_route_aborts_blocked_type() -> None:
    handler = await _handler_for()
    route = _FakeRoute("https://example.com/video.mp4", "media")
    await handler(route)
    assert route.actions == ["abort"]


async def test_block_route_continues_normal_request() -> None:
    handler = await _handler_for()
    route = _FakeRoute("https://example.com/app.js", "script")
    await handler(route)
    assert route.actions == ["continue"]


class _RaisingAbortRoute(_FakeRoute):
    async def abort(self) -> None:
        raise RuntimeError("route already handled")


async def test_block_route_falls_back_to_continue_on_abort_error() -> None:
    handler = await _handler_for()
    route = _RaisingAbortRoute("https://doubleclick.net/ad.js", "script")
    await handler(route)
    assert route.actions == ["continue"]
