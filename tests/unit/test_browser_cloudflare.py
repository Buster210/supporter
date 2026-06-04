from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from supporter.tools.browser import cloudflare, humanize


class _Loc:
    def __init__(self, count: int = 0) -> None:
        self._count = count

    @property
    def first(self) -> _Loc:
        return self

    async def count(self) -> int:
        return self._count


class _Frame:
    def __init__(self, url: str, checkbox_count: int = 0) -> None:
        self.url = url
        self._checkbox_count = checkbox_count

    def locator(self, selector: str) -> _Loc:
        return _Loc(self._checkbox_count)


class _Page:
    def __init__(
        self,
        frames: list[_Frame],
        widget_count: int = 0,
        cf_response: str = "",
    ) -> None:
        self.frames = frames
        self._widget_count = widget_count
        self._cf_response = cf_response

    def locator(self, selector: str) -> _Loc:
        return _Loc(self._widget_count)

    async def evaluate(self, expression: str, *args: Any) -> str:
        return self._cf_response


def _cf_frame(checkbox_count: int = 0) -> _Frame:
    return _Frame(
        "https://challenges.cloudflare.com/turnstile/if/ov2/av0/x",
        checkbox_count=checkbox_count,
    )


async def test_detect_via_frame_url() -> None:
    page = _Page([_Frame("https://example.test/"), _cf_frame()])
    assert await cloudflare.detect_turnstile_in_page(page) is True  # type: ignore[arg-type]


async def test_detect_via_widget_locator() -> None:
    page = _Page([_Frame("https://example.test/")], widget_count=1)
    assert await cloudflare.detect_turnstile_in_page(page) is True  # type: ignore[arg-type]


async def test_detect_absent() -> None:
    page = _Page([_Frame("https://example.test/")])
    assert await cloudflare.detect_turnstile_in_page(page) is False  # type: ignore[arg-type]


async def test_solve_no_frame() -> None:
    page = _Page([_Frame("https://example.test/")])
    assert await cloudflare.solve_cloudflare(page) == (  # type: ignore[arg-type]
        "No Cloudflare Turnstile detected on the page."
    )


async def test_solve_no_checkbox_returns_manual(monkeypatch: Any) -> None:
    monkeypatch.setattr(humanize, "human_click", AsyncMock())
    page = _Page([_cf_frame(checkbox_count=0)])
    result = await cloudflare.solve_cloudflare(page)  # type: ignore[arg-type]
    assert "manual solve needed" in result


async def test_solve_clicks_and_confirms(monkeypatch: Any) -> None:
    click = AsyncMock()
    monkeypatch.setattr(humanize, "human_click", click)
    monkeypatch.setattr(cloudflare, "_SETTLE_SECONDS", 0.0)
    page = _Page([_cf_frame(checkbox_count=1)], cf_response="solved-token")
    result = await cloudflare.solve_cloudflare(page)  # type: ignore[arg-type]
    assert result == "Cloudflare Turnstile solved."
    assert click.await_count == 1
    assert click.await_args is not None
    assert "locator" in click.await_args.kwargs


async def test_solve_clicks_unconfirmed(monkeypatch: Any) -> None:
    monkeypatch.setattr(humanize, "human_click", AsyncMock())
    monkeypatch.setattr(cloudflare, "_SETTLE_SECONDS", 0.0)
    page = _Page([_cf_frame(checkbox_count=1)], cf_response="")
    result = await cloudflare.solve_cloudflare(page)  # type: ignore[arg-type]
    assert "could not confirm" in result


async def test_looks_solved_when_frame_detaches(monkeypatch: Any) -> None:
    monkeypatch.setattr(cloudflare, "_SETTLE_SECONDS", 0.0)
    frame = _cf_frame(checkbox_count=1)
    page = _Page([])  # frame no longer attached
    assert await cloudflare._looks_solved(page, frame) is True  # type: ignore[arg-type]
