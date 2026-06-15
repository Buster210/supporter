"""Tests for Feature C — auto-approve web interaction (files stay gated)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestConfirmBrowseAutoApprove:
    """Test that _confirm_browse auto-approves when browser_auto_approve is on."""

    async def test_auto_approve_returns_true_without_modal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When browser_auto_approve is True, _confirm_browse returns True."""
        from supporter.tui import SupporterApp

        mock_config = SimpleNamespace(browser_auto_approve=True)
        monkeypatch.setattr("supporter.tui.config", mock_config)

        app = SupporterApp.__new__(SupporterApp)
        result = await app._confirm_browse("Click button?", "Click the submit button")
        assert result is True

    async def test_auto_approve_off_falls_through_to_modal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When auto_approve is off, _confirm_browse calls push_screen_wait."""
        from supporter.tui import SupporterApp

        mock_config = SimpleNamespace(browser_auto_approve=False)
        monkeypatch.setattr("supporter.tui.config", mock_config)

        app = SupporterApp.__new__(SupporterApp)
        app.push_screen_wait = AsyncMock(return_value=False)

        result = await app._confirm_browse("Click button?", "Click the submit button")
        assert result is False
        app.push_screen_wait.assert_called_once()

    async def test_auto_approve_eval_also_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """eval actions are auto-approved too (running code on sites is intentional)."""
        from supporter.tui import SupporterApp

        mock_config = SimpleNamespace(browser_auto_approve=True)
        monkeypatch.setattr("supporter.tui.config", mock_config)

        app = SupporterApp.__new__(SupporterApp)
        result = await app._confirm_browse("Run JavaScript?", "eval: document.title")
        assert result is True


class TestRecordCleanInteractionPromotionGuard:
    """Test that record_clean_interaction skips promotion when auto-approve is on."""

    async def test_auto_approve_blocks_promotion_dialog(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When browser_auto_approve is on, promotion dialog is never fired."""
        from supporter.tools.browser import guardrails as gr

        # Set up a host that would normally be promoted
        mock_config = SimpleNamespace(
            browser_auto_approve=True,
            browser_promotion_threshold=1,
        )
        monkeypatch.setattr("supporter.tools.browser.guardrails.config", mock_config)

        # Mock the trust store
        mock_store = MagicMock()
        mock_store.is_confirmed.return_value = False
        mock_store.is_suppressed.return_value = False
        mock_store.clean_success_count.return_value = 10  # Above threshold
        monkeypatch.setattr(gr, "_get_trust_store", lambda: mock_store)

        # Mock the callback to track if it was called
        callback_called = [False]

        async def fake_callback(title: str, detail: str) -> bool:
            callback_called[0] = True
            return True

        monkeypatch.setattr(gr, "browse_confirmation_callback", fake_callback)

        # Call record_clean_interaction — should NOT trigger promotion
        await gr.record_clean_interaction("example.com")

        # The promotion dialog should never fire
        assert callback_called[0] is False
        # But record_clean should still have been called (counting is fine)
        mock_store.record_clean.assert_called_once_with("example.com")

    async def test_auto_approve_off_allows_promotion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When browser_auto_approve is off, promotion dialog fires as before."""
        from supporter.tools.browser import guardrails as gr

        mock_config = SimpleNamespace(
            browser_auto_approve=False,
            browser_promotion_threshold=1,
        )
        monkeypatch.setattr("supporter.tools.browser.guardrails.config", mock_config)

        mock_store = MagicMock()
        mock_store.is_confirmed.return_value = False
        mock_store.is_suppressed.return_value = False
        mock_store.clean_success_count.return_value = 10
        monkeypatch.setattr(gr, "_get_trust_store", lambda: mock_store)

        callback_called = [False]

        async def fake_callback(title: str, detail: str) -> bool:
            callback_called[0] = True
            return True

        monkeypatch.setattr(gr, "browse_confirmation_callback", fake_callback)

        await gr.record_clean_interaction("example.com")

        # Promotion dialog SHOULD fire
        assert callback_called[0] is True

    async def test_auto_approve_still_counts_interactions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto-approve still records clean interactions (counting is fine)."""
        from supporter.tools.browser import guardrails as gr

        mock_config = SimpleNamespace(
            browser_auto_approve=True,
            browser_promotion_threshold=100,
        )
        monkeypatch.setattr("supporter.tools.browser.guardrails.config", mock_config)

        mock_store = MagicMock()
        mock_store.is_confirmed.return_value = False
        mock_store.is_suppressed.return_value = False
        mock_store.clean_success_count.return_value = 1
        monkeypatch.setattr(gr, "_get_trust_store", lambda: mock_store)

        await gr.record_clean_interaction("example.com")

        mock_store.record_clean.assert_called_once_with("example.com")
