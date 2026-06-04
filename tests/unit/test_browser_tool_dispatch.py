from __future__ import annotations

from typing import Any

import pytest

from supporter.tools.browser import support, tool
from supporter.tools.browser.core import BrowseRequest
from supporter.tools.browser.handlers import HANDLERS


async def test_unknown_action_returns_error_with_sorted_valid_names() -> None:
    result = await tool.browse("definitely-not-an-action")

    assert result.startswith(
        "Error: Unknown action 'definitely-not-an-action'. Valid actions: "
    )
    names = result.split("Valid actions: ", 1)[1]
    listed = names.split(", ")
    assert listed == sorted(HANDLERS)
    assert set(listed) == set(HANDLERS)


async def test_unknown_action_does_not_touch_a_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(_req: BrowseRequest) -> str:
        raise AssertionError("handler must not run for an unknown action")

    poisoned = dict.fromkeys(HANDLERS, boom)
    monkeypatch.setattr(tool, "HANDLERS", poisoned)

    result = await tool.browse("still-unknown")

    assert result.startswith("Error: Unknown action 'still-unknown'.")


async def test_known_action_builds_request_routes_and_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    async def handler(req: BrowseRequest) -> str:
        seen["req"] = req
        return "handled"

    recorded: list[tuple[BrowseRequest, str]] = []

    async def record(req: BrowseRequest, result: str) -> None:
        recorded.append((req, result))

    monkeypatch.setattr(tool, "HANDLERS", {"snapshot": handler})
    monkeypatch.setattr(tool, "_record_step", record)

    result = await tool.browse("snapshot", url="https://example.test/", ref="e1")

    assert result == "handled"
    req = seen["req"]
    assert req.action == "snapshot"
    assert req.url == "https://example.test/"
    assert req.ref == "e1"
    assert recorded == [(req, "handled")]


async def test_dispatch_forwards_keyword_args_into_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, BrowseRequest] = {}

    async def handler(req: BrowseRequest) -> str:
        captured["req"] = req
        return "ok"

    async def record(_req: BrowseRequest, _result: str) -> None:
        return None

    monkeypatch.setattr(tool, "HANDLERS", {"type": handler})
    monkeypatch.setattr(tool, "_record_step", record)

    await tool.browse(
        "type",
        text="hello",
        delay_ms=250,
        key="Enter",
        value="v",
        selector="#in",
        dx=3,
        dy=4,
        script="x",
        index=2,
        html=True,
        path="p",
        stamp="s",
    )

    req = captured["req"]
    assert req.text == "hello"
    assert req.delay_ms == 250
    assert req.key == "Enter"
    assert req.value == "v"
    assert req.selector == "#in"
    assert req.dx == 3
    assert req.dy == 4
    assert req.script == "x"
    assert req.index == 2
    assert req.html is True
    assert req.path == "p"
    assert req.stamp == "s"


async def test_action_cap_runtimeerror_surfaces_as_recoverable_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def capped(_req: BrowseRequest) -> str:
        raise RuntimeError("Action cap reached (50).")

    async def record(_req: BrowseRequest, _result: str) -> None:
        return None

    wrapped = support._wrap_action_errors("snapshot")(capped)
    monkeypatch.setattr(tool, "HANDLERS", {"snapshot": wrapped})
    monkeypatch.setattr(tool, "_record_step", record)

    result = await tool.browse("snapshot")

    assert result == "Error: Action cap reached (50)."
