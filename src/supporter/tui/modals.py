from __future__ import annotations

from typing import Any

from rich.syntax import Syntax
from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Label, OptionList, Static

from .constants import (
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


class ProfileSelectModal(ModalScreen[str | None]):
    MAX_LABEL_WIDTH = 80

    def __init__(self, profiles: list[Any]) -> None:
        super().__init__()
        self._profiles = profiles

    def compose(self) -> ComposeResult:
        from textual.widgets import OptionList

        with Vertical(id="modal-container"):
            yield Label("Browser profile missing(BROWSER_PROFILE_NAME/BROWSER_PROFILE_PATH), select profile:", id="modal-header")
            yield OptionList(id="profile-list")
            with Horizontal(id="modal-buttons"):
                yield Button("Cancel", id="cancel")

    @staticmethod
    def _format_profile_label(i: int, p: Any) -> str:
        email = p.email or "(not signed in)"
        tag = f"[{p.dir_name}]"
        col_width = 14
        padded_tag = tag.ljust(col_width)
        padded_name = p.display_name.ljust(col_width)
        padded_email = email.ljust(col_width)
        return f"{i + 1}. {padded_tag}{padded_name}— {padded_email}"

    def _truncate_label(self, label: str, max_width: int) -> str:
        if len(label) <= max_width:
            return label
        return label[: max_width - 3] + "..."

    def on_mount(self) -> None:
        screen_width = self.app.size.width
        modal_max_width = int(screen_width * MODAL_MAX_WIDTH_PERCENT)
        available_width = min(modal_max_width - 10, self.MAX_LABEL_WIDTH)

        labels = [
            self._truncate_label(self._format_profile_label(i, p), available_width)
            for i, p in enumerate(self._profiles)
        ]

        max_label_len = max((len(label) for label in labels), default=40)
        max_len = max_label_len

        width = min(
            int((max_len + MODAL_PADDING) * MODAL_WIDTH_SCALE),
            modal_max_width,
        )
        self.query_one("#modal-container").styles.width = width

        option_list = self.query_one("#profile-list", OptionList)
        for label in labels:
            option_list.add_option(Content(label))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self._profiles[event.option_index].dir_name)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
