from __future__ import annotations

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from ..config import (
    MODAL_MAX_WIDTH_PERCENT,
    MODAL_PADDING,
    MODAL_WIDTH_SCALE,
)


class ConfirmationModal(ModalScreen[bool]):
    def __init__(
        self,
        title: str,
        content: str,
        language: str = "diff",
        meta: str | None = None,
    ):
        super().__init__()
        self.modal_title = title
        self.content = content
        self.language = language
        self.meta = meta

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-container"):
            yield Label(self.modal_title, id="modal-header")
            if self.meta:
                yield Label(self.meta, classes="modal-meta")

            with ScrollableContainer(id="modal-content"):
                yield Static(
                    Syntax(
                        self.content,
                        self.language,
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
        header_len = len(self.modal_title)
        meta_len = len(self.meta) if self.meta else 0

        max_line = max((len(line) for line in lines), default=40)
        max_line = max(max_line, header_len, meta_len)

        width = min(
            int((max_line + MODAL_PADDING) * MODAL_WIDTH_SCALE),
            int(self.app.size.width * MODAL_MAX_WIDTH_PERCENT),
        )
        self.query_one("#modal-container").styles.width = width

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")
