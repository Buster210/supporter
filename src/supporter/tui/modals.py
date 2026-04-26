from __future__ import annotations

from pathlib import Path

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from ..config import (
    BASH_MODAL_MAX_WIDTH,
    MODAL_MAX_WIDTH_PERCENT,
    MODAL_PADDING,
    MODAL_WIDTH_SCALE,
)


class ConfirmationModal(ModalScreen[bool]):
    def __init__(self, path: str, content: str):
        super().__init__()
        self.path = path
        self.content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            filename = Path(self.path).name
            yield Label(f"Write {filename}?", id="modal-header")
            with ScrollableContainer(id="modal-content"):
                yield Static(
                    Syntax(
                        self.content,
                        "diff",
                        theme="monokai",
                        background_color="default",
                        word_wrap=True,
                    ),
                    expand=True,
                )
            with Horizontal(id="modal-buttons"):
                yield Button("Allow", id="allow")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        lines = self.content.splitlines()
        max_line = max((len(line) for line in lines), default=40)
        width = min(
            int((max_line + MODAL_PADDING) * MODAL_WIDTH_SCALE),
            int(self.app.size.width * MODAL_MAX_WIDTH_PERCENT),
        )
        self.query_one("#modal-container").styles.width = width

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


class BashConfirmationModal(ModalScreen[bool]):
    def __init__(self, command: list[str], cwd: str):
        super().__init__()
        self.command = command
        self.cwd = cwd

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label("Authorize Bash Execution?", id="modal-header")
            with ScrollableContainer(id="modal-content"):
                cmd_str = " ".join(self.command)
                yield Label(
                    f"[bold cyan]Working Dir:[/] {self.cwd}", classes="modal-meta"
                )
                yield Static(
                    Syntax(
                        cmd_str,
                        "bash",
                        theme="monokai",
                        background_color="default",
                        word_wrap=True,
                    ),
                    expand=True,
                )
            with Horizontal(id="modal-buttons"):
                yield Button("Allow", id="allow", variant="success")
                yield Button("Deny", id="cancel", variant="error")

    def on_mount(self) -> None:
        self.query_one("#modal-container").styles.width = min(
            BASH_MODAL_MAX_WIDTH, int(self.app.size.width * MODAL_MAX_WIDTH_PERCENT)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")
