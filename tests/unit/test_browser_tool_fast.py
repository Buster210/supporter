from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from supporter.tools.browser import guardrails, support


def _run_effective_fast(host: str) -> bool:
    async def fake_host(_page: object) -> str:
        return host

    mock_store = MagicMock()
    mock_store.is_confirmed.return_value = False

    with (
        patch.object(support, "_page_host", fake_host),
        patch.object(support.config, "browser_debug_overlay", False),
        patch.object(guardrails, "_get_trust_store", return_value=mock_store),
    ):
        return asyncio.run(support._effective_fast(object()))


def test_allowlisted_host_runs_fast() -> None:
    assert _run_effective_fast("google.com") is True


def test_allowlisted_subdomain_runs_fast() -> None:
    assert _run_effective_fast("gemini.google.com") is True


def test_non_allowlisted_host_humanizes() -> None:
    assert _run_effective_fast("example.com") is False
