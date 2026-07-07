"""Delegation block — plan + live progress + result in a single collapsible widget."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Collapsible, Static


class DelegationBlock(Collapsible):
    """Single collapsible container for plan + progress + result.

    ponytail: Consolidates scattered delegation bubbles into one widget,
    collapsed when complete to keep the chat hierarchy clear. Toggling
    expands to show plan, live progress, and final result.
    """

    def __init__(self, title: str = "Delegation Details", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_class("delegation-block")
        self._plan_widget: Static | None = None
        self._progress_widget: Static | None = None
        self._result_widget: Static | None = None
        # ponytail: stash content set before compose() runs, applied on mount —
        # setters fire from the listener before the widget is in the DOM.
        self._pending: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(classes="delegation-block-content"):
            self._plan_widget = Static("", classes="delegation-plan-content")
            yield self._plan_widget

            self._progress_widget = Static("", classes="delegation-progress-content")
            yield self._progress_widget

            self._result_widget = Static("", classes="delegation-result-content")
            yield self._result_widget

        for section, markdown in self._pending.items():
            self._apply(section, markdown)
        self._pending.clear()

    def _apply(self, section: str, markdown: str) -> None:
        widget = {
            "plan": self._plan_widget,
            "progress": self._progress_widget,
            "result": self._result_widget,
        }[section]
        if widget is None:
            # Not composed yet — stash; compose() drains _pending on mount.
            self._pending[section] = markdown
            return
        widget.update(markdown)
        widget.display = bool(markdown)

    def set_plan(self, markdown: str) -> None:
        """Update the plan section with markdown content."""
        self._apply("plan", markdown)

    def set_progress(self, markdown: str) -> None:
        """Update the progress section (live updates in-place)."""
        self._apply("progress", markdown)

    def set_result(self, markdown: str) -> None:
        """Update the result/summary section."""
        self._apply("result", markdown)

    def collapse_when_done(self) -> None:
        """Collapse this block once delegation is complete."""
        self.collapsed = True
