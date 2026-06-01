from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

from supporter.tools.browser import tool

if TYPE_CHECKING:
    from supporter.tools.browser.tool import BrowseRequest


@dataclass
class _Req:
    fast: bool = False


def _run_effective_fast(host: str, req: _Req) -> bool:
    async def fake_host(_page: object) -> str:
        return host

    with patch.object(tool, "_page_host", fake_host):
        return asyncio.run(tool._effective_fast(object(), cast("BrowseRequest", req)))


def test_fast_host_runs_fast() -> None:
    assert _run_effective_fast("google.com", _Req(fast=False)) is True


def test_fast_flag_does_not_force_fast_off_allowlist() -> None:
    assert _run_effective_fast("example.com", _Req(fast=True)) is False


def test_fast_flag_does_not_disable_fast_on_allowlist() -> None:
    assert _run_effective_fast("gemini.google.com", _Req(fast=False)) is True


def test_non_allowlist_host_humanizes() -> None:
    assert _run_effective_fast("bank.example.com", _Req(fast=True)) is False
