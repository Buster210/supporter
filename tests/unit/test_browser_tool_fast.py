from __future__ import annotations

import asyncio
from unittest.mock import patch

from supporter.tools.browser import tool


def _run_effective_fast(host: str) -> bool:
    async def fake_host(_page: object) -> str:
        return host

    with patch.object(tool, "_page_host", fake_host):
        return asyncio.run(tool._effective_fast(object()))


def test_allowlisted_host_runs_fast() -> None:
    assert _run_effective_fast("google.com") is True


def test_allowlisted_subdomain_runs_fast() -> None:
    assert _run_effective_fast("gemini.google.com") is True


def test_non_allowlisted_host_humanizes() -> None:
    assert _run_effective_fast("example.com") is False
