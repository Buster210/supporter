"""Delegation block — plan + live progress + result in a single collapsible widget."""

from __future__ import annotations

from typing import Any

from rich import box
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click
from textual.widgets import Collapsible, Static

from .bubble import SectionHeader

_STATUS_STYLE = {
    "completed": ("green", "✓"),
    "working": ("yellow", "●"),
    "waiting": ("dim", "○"),
    "failed": ("red", "✗"),
    "timed out": ("red", "✗"),
    "skipped": ("dim", "-"),
}


def _style_status(label: str) -> Text:
    colour, glyph = _STATUS_STYLE.get(label.lower(), ("white", "•"))
    return Text(f"{glyph} {label}", style=colour)


def _render_progress(markdown: str) -> RenderableType:
    """Render the progress markdown as a polished full-width Rich table.

    The string is `Job \\`id\\`\\n\\n<GFM table>`; cells are already sanitised
    of ``|`` upstream, so splitting on ``|`` is safe. Rendered as an expanding
    Table with a job-id caption, coloured status column, and a simple box so it
    fills the delegation block width and reads cleanly.
    """
    lines = [ln for ln in markdown.splitlines() if ln.strip()]
    rows = [ln for ln in lines if ln.lstrip().startswith("|")]
    heading = " ".join(
        ln.replace("`", "").strip() for ln in lines if not ln.lstrip().startswith("|")
    )

    def cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    if len(rows) < 2:
        return Text(heading, style="bold")

    headers = cells(rows[0])
    data = [cells(r) for r in rows[2:]]

    keep = [
        i
        for i in range(len(headers))
        if i == 0 or any(i < len(row) and row[i] for row in data)
    ]
    headers = [headers[i] for i in keep]
    data = [[row[i] if i < len(row) else "" for i in keep] for row in data]
    status_idx = next((i for i, h in enumerate(headers) if h.lower() == "status"), -1)

    table = Table(
        expand=True,
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column(headers[0], ratio=3, overflow="fold")
    for header in headers[1:]:
        table.add_column(header, ratio=2, overflow="fold")
    for row in data:
        if 0 <= status_idx < len(row):
            row[status_idx] = _style_status(row[status_idx])  # type: ignore[call-overload]
        table.add_row(*row)

    return Group(Text(heading, style="bold cyan", justify="center"), Text(""), table)


def _render_plan(markdown: str) -> RenderableType:
    """Render the plan with a centered heading (first line) over its Markdown body."""
    lines = markdown.splitlines()
    if not lines:
        return Markdown(markdown)
    heading = lines[0].lstrip("# ").replace("`", "").strip()
    body = "\n".join(lines[1:]).strip()
    parts: list[RenderableType] = [
        Text(heading, style="bold magenta", justify="center"),
    ]
    if body:
        parts.extend((Text(""), Markdown(body)))
    return Group(*parts)


class DelegationBlock(Collapsible):
    """Single collapsible container for plan + progress + result.

    Collapsed when complete to keep the chat hierarchy clear; toggling
    expands to show plan, live progress, and final result.
    """

    def __init__(self, title: str = "Delegation Details", **kwargs: Any) -> None:
        super().__init__(title=title, **kwargs)
        self.add_class("delegation-block")
        self._plan_widget: Static | None = None
        self._progress_widget: Static | None = None
        self._signal_widget: Static | None = None
        self._result_widget: Static | None = None
        self._signal_text = ""
        self._pending: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(classes="delegation-block-content"):
            # Start hidden so empty panels don't paint a stray $boost band;
            # _apply flips display on once a section has content.
            self._progress_widget = Static("", classes="delegation-progress-content")
            self._progress_widget.display = False
            yield self._progress_widget

            # Order: progress (table) -> signal (task complete) -> result
            # (job complete) -> plan. Section position is fixed here so the
            # display order does not depend on async mount timing.
            self._signal_widget = Static("", classes="delegation-signal")
            self._signal_widget.display = False
            yield self._signal_widget

            self._result_widget = Static("", classes="delegation-result-content")
            self._result_widget.display = False
            yield self._result_widget

            self._plan_widget = Static("", classes="delegation-plan-content")
            self._plan_widget.display = False
            yield self._plan_widget

        for section, markdown in self._pending.items():
            self._apply(section, markdown)
        self._pending.clear()

    def _apply(self, section: str, markdown: str) -> None:
        widget = {
            "plan": self._plan_widget,
            "progress": self._progress_widget,
            "signal": self._signal_widget,
            "result": self._result_widget,
        }[section]
        if widget is None:
            self._pending[section] = markdown
            return
        if not markdown:
            widget.update("")
        elif section == "progress":
            widget.update(_render_progress(markdown))
        elif section == "plan":
            widget.update(_render_plan(markdown))
        elif section in ("signal", "result"):
            widget.update(Text(markdown, justify="center"))
        else:
            widget.update(Markdown(markdown))
        widget.display = bool(markdown)

    def set_plan(self, markdown: str) -> None:
        """Update the plan section with markdown content."""
        self._apply("plan", markdown)

    def set_progress(self, markdown: str) -> None:
        """Update the progress section (live updates in-place)."""
        self._apply("progress", markdown)

    def set_signal(self, text: str) -> None:
        """Append a per-task completion signal line (accumulates across tasks)."""
        self._signal_text = (
            f"{self._signal_text}\n\n{text}".strip() if self._signal_text else text
        )
        self._apply("signal", self._signal_text)

    def set_result(self, markdown: str) -> None:
        """Update the result/summary section."""
        self._apply("result", markdown)

    def collapse_when_done(self) -> None:
        """Collapse this block once delegation is complete."""
        self.collapsed = True


class _ClickAwareSectionHeader(SectionHeader):
    def on_click(self, event: Click) -> None:
        event.stop()
        block = self.parent
        while block is not None and not isinstance(block, VerificationBlock):
            block = block.parent
        if block is not None:
            block.toggle()


class VerificationBlock(Vertical):
    def __init__(self, title: str = "Verification", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_class("verification-block")
        self._header = _ClickAwareSectionHeader(
            "Verification", classes="section-header"
        )
        self._content_widget = Static("", classes="verification-content-text")
        self._overall_widget = Static("", classes="verification-overall")
        self._entries: list[str] = []
        self._passed = 0
        self._failed = 0
        self._collapsed = True
        self._overall_text = ""

    def compose(self) -> ComposeResult:
        yield self._header
        yield self._content_widget
        yield self._overall_widget

    def on_mount(self) -> None:
        self._refresh()

    def _heading(self) -> str:
        return f"Verification — {self._passed}/{self._passed + self._failed}"

    def _refresh(self) -> None:
        self._header.update_label(self._heading(), self._collapsed, True)
        self._content_widget.display = bool(self._entries) and not self._collapsed
        self._overall_widget.display = bool(self._overall_text) and not self._collapsed

    def toggle(self) -> None:
        self._collapsed = not self._collapsed
        self._refresh()

    @staticmethod
    def _styled(text: str) -> Text:
        """Red for a failure line (✗ prefix), green otherwise."""
        return Text(text, style="red" if text.startswith("✗") else "green")

    def add_entry(self, text: str) -> None:
        self._entries.append(text)
        if text.startswith("✗"):
            self._failed += 1
        else:
            self._passed += 1
        body = Text()
        for i, entry in enumerate(self._entries):
            if i:
                body.append("\n")
            body.append_text(self._styled(entry))
        self._content_widget.update(body)
        self._refresh()

    def set_overall(self, text: str) -> None:
        """Set the final overall-status line (red on failure, green on success)."""
        self._overall_text = text
        self._overall_widget.update(self._styled(text))
        self._refresh()

    def collapse_when_done(self) -> None:
        """Collapse once verification settles."""
        self._collapsed = True
        self._refresh()
