from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from textual.geometry import Size

from supporter.tui.bubble import MessageBubble, SectionHeader
from supporter.tui.constants import (
    COLLAPSED_SUMMARY_LEN,
    MODAL_MAX_WIDTH_PERCENT,
    MODAL_PADDING,
    MODAL_WIDTH_SCALE,
)
from supporter.tui.modals import ConfirmationModal, ProfileSelectModal


@pytest.fixture
def mock_app() -> Any:
    app = MagicMock()
    app.size = Size(120, 40)
    return app


class _Profile:
    def __init__(self, dir_name: str, display_name: str, email: str) -> None:
        self.dir_name = dir_name
        self.display_name = display_name
        self.email = email


class TestConfirmationModalOnMount:
    def test_width_scales_to_content(self, mock_app: Any) -> None:
        modal = ConfirmationModal(title="Bash", content="echo hi", meta="/work")
        container = MagicMock()
        with (
            patch.object(
                ConfirmationModal,
                "app",
                new_callable=PropertyMock,
                return_value=mock_app,
            ),
            patch.object(modal, "query_one", return_value=container),
        ):
            modal.on_mount()
        expected = int((len("echo hi") + MODAL_PADDING) * MODAL_WIDTH_SCALE)
        assert container.styles.width == expected

    def test_width_capped_at_screen_percentage(self, mock_app: Any) -> None:
        modal = ConfirmationModal(title="Bash", content="x" * 200, meta=None)
        container = MagicMock()
        with (
            patch.object(
                ConfirmationModal,
                "app",
                new_callable=PropertyMock,
                return_value=mock_app,
            ),
            patch.object(modal, "query_one", return_value=container),
        ):
            modal.on_mount()
        cap = int(mock_app.size.width * MODAL_MAX_WIDTH_PERCENT)
        assert container.styles.width == cap

    def test_width_accounts_for_long_title(self, mock_app: Any) -> None:
        modal = ConfirmationModal(title="T" * 30, content="ls", meta=None)
        container = MagicMock()
        with (
            patch.object(
                ConfirmationModal,
                "app",
                new_callable=PropertyMock,
                return_value=mock_app,
            ),
            patch.object(modal, "query_one", return_value=container),
        ):
            modal.on_mount()
        expected = int((30 + MODAL_PADDING) * MODAL_WIDTH_SCALE)
        assert container.styles.width == expected


class TestProfileSelectModalTruncateLabel:
    def test_short_label_unchanged(self) -> None:
        modal = ProfileSelectModal([])
        assert modal._truncate_label("short", 20) == "short"

    def test_label_at_boundary_unchanged(self) -> None:
        modal = ProfileSelectModal([])
        label = "x" * 20
        assert modal._truncate_label(label, 20) == label

    def test_long_label_truncated_with_ellipsis(self) -> None:
        modal = ProfileSelectModal([])
        result = modal._truncate_label("x" * 50, 20)
        assert len(result) == 20
        assert result.endswith("...")


class TestProfileSelectModalOnMount:
    def test_adds_one_option_per_profile_and_sets_width(self, mock_app: Any) -> None:
        profiles = [
            _Profile("Default", "Personal", "me@example.com"),
            _Profile("Profile 1", "Work", "work@company.com"),
        ]
        modal = ProfileSelectModal(profiles)
        container = MagicMock()
        with (
            patch.object(
                ProfileSelectModal,
                "app",
                new_callable=PropertyMock,
                return_value=mock_app,
            ),
            patch.object(modal, "query_one", return_value=container),
        ):
            modal.on_mount()
        assert container.add_option.call_count == len(profiles)
        cap = int(mock_app.size.width * MODAL_MAX_WIDTH_PERCENT)
        assert isinstance(container.styles.width, int)
        assert 0 < container.styles.width <= cap

    def test_truncates_overlong_labels(self, mock_app: Any) -> None:
        profiles = [_Profile("D" * 100, "N" * 100, "e" * 100)]
        modal = ProfileSelectModal(profiles)
        container = MagicMock()
        with (
            patch.object(
                ProfileSelectModal,
                "app",
                new_callable=PropertyMock,
                return_value=mock_app,
            ),
            patch.object(modal, "query_one", return_value=container),
        ):
            modal.on_mount()
        label = str(container.add_option.call_args[0][0])
        assert label.endswith("...")
        assert len(label) <= ProfileSelectModal.MAX_LABEL_WIDTH


class TestProfileSelectModalDismiss:
    def test_option_selected_dismisses_with_dir_name(self) -> None:
        profiles = [_Profile("Default", "Personal", ""), _Profile("P1", "Work", "")]
        modal = ProfileSelectModal(profiles)
        event = MagicMock()
        event.option_index = 1
        with patch.object(modal, "dismiss") as dismiss:
            modal.on_option_list_option_selected(event)
        dismiss.assert_called_once_with("P1")

    def test_cancel_button_dismisses_with_none(self) -> None:
        modal = ProfileSelectModal([])
        event = MagicMock()
        event.button.id = "cancel"
        with patch.object(modal, "dismiss") as dismiss:
            modal.on_button_pressed(event)
        dismiss.assert_called_once_with(None)


class TestMessageBubbleToggleSection:
    def test_toggles_first_collapsible_section(self) -> None:
        bubble = MessageBubble(role="agent", content="reply")
        bubble.elements = [
            {"type": "thought", "content": "thinking", "collapsed": False},
            {"type": "content", "content": "reply", "collapsed": False},
        ]
        header = cast("SectionHeader", object())
        bubble._elements_container = SimpleNamespace(children=[header])  # type: ignore[assignment]

        bubble.toggle_section(header)

        assert bubble.elements[0]["collapsed"] is True
        assert bubble.elements[0]["manually_interacted"] is True

    def test_toggles_section_after_a_content_element(self) -> None:
        bubble = MessageBubble(role="agent", content="reply")
        bubble.elements = [
            {"type": "content", "content": "reply", "collapsed": False},
            {"type": "thought", "content": "thinking", "collapsed": True},
        ]
        thought_header = cast("SectionHeader", object())
        bubble._elements_container = SimpleNamespace(  # type: ignore[assignment]
            children=[object(), thought_header]
        )

        bubble.toggle_section(thought_header)

        assert bubble.elements[1]["collapsed"] is False
        assert bubble.elements[1]["manually_interacted"] is True

    def test_section_not_in_container_is_a_noop(self) -> None:
        bubble = MessageBubble(role="agent", content="reply")
        bubble.elements = [
            {"type": "thought", "content": "thinking", "collapsed": False},
        ]
        bubble._elements_container = SimpleNamespace(children=[object()])  # type: ignore[assignment]

        bubble.toggle_section(cast("SectionHeader", object()))

        assert bubble.elements[0]["collapsed"] is False


class TestMessageBubbleRenderCollapsed:
    def _prime(self, bubble: MessageBubble) -> None:
        bubble._message_view = MagicMock()
        bubble._elements_container = MagicMock()
        bubble._meta_label = MagicMock()

    def test_short_content_has_no_ellipsis(self) -> None:
        bubble = MessageBubble(role="agent", content="hello world")
        self._prime(bubble)

        bubble._render_collapsed()

        rendered = bubble._message_view.update.call_args[0][0]  # type: ignore[union-attr]
        assert "hello world" in rendered
        assert "..." not in rendered
        assert bubble._message_view.display is True  # type: ignore[union-attr]
        assert bubble._meta_label.display is False  # type: ignore[union-attr]

    def test_long_content_is_truncated_with_ellipsis(self) -> None:
        bubble = MessageBubble(role="agent", content="x" * (COLLAPSED_SUMMARY_LEN + 20))
        self._prime(bubble)

        bubble._render_collapsed()

        rendered = bubble._message_view.update.call_args[0][0]  # type: ignore[union-attr]
        assert "..." in rendered
        assert "x" * COLLAPSED_SUMMARY_LEN in rendered

    def test_multiline_content_summarizes_first_line(self) -> None:
        bubble = MessageBubble(role="agent", content="first line\nsecond line")
        self._prime(bubble)

        bubble._render_collapsed()

        rendered = bubble._message_view.update.call_args[0][0]  # type: ignore[union-attr]
        assert "first line" in rendered
        assert "second line" not in rendered
        assert "..." in rendered


class TestMessageBubbleElementIsMarkdown:
    def test_explicit_markdown_flag_short_circuits(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        el = {"type": "content", "content": "plain", "is_markdown": True}
        assert bubble._element_is_markdown(el) is True

    def test_recheck_computes_and_caches_decision(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        el = {"type": "content", "content": "# heading", "_recheck_markdown": True}
        assert bubble._element_is_markdown(el) is True
        assert el["is_markdown"] is True
        assert el["_recheck_markdown"] is False

    def test_plain_text_is_not_markdown(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        el = {"type": "content", "content": "just text", "_recheck_markdown": True}
        assert bubble._element_is_markdown(el) is False
        assert el["is_markdown"] is False
