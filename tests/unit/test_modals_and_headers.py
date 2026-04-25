from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from textual.geometry import Size

from supporter.tui.widgets import BashConfirmationModal, SectionHeader


@pytest.fixture
def mock_app() -> Any:
    app = MagicMock()
    app.size = Size(120, 40)
    return app


class TestBashConfirmationModal:
    def test_init_stores_command_and_cwd(self) -> None:
        modal = BashConfirmationModal(command=["ls", "-la"], cwd="/home/user")
        assert modal.command == ["ls", "-la"]
        assert modal.cwd == "/home/user"

    def test_command_joined_for_display(self) -> None:
        modal = BashConfirmationModal(
            command=["git", "commit", "-m", "test"], cwd="/repo"
        )
        cmd_str = " ".join(modal.command)
        assert cmd_str == "git commit -m test"

    def test_modal_compose_displays_command(self, mock_app: Any) -> None:
        modal = BashConfirmationModal(command=["echo", "hello"], cwd="/fake_tmp")
        with patch.object(
            BashConfirmationModal,
            "app",
            new_callable=PropertyMock,
            return_value=mock_app,
        ):
            assert modal.command == ["echo", "hello"]
            assert modal.cwd == "/fake_tmp"

    def test_modal_compose_has_allow_button(self, mock_app: Any) -> None:
        _modal = BashConfirmationModal(command=["ls"], cwd="/")
        with patch.object(
            BashConfirmationModal,
            "app",
            new_callable=PropertyMock,
            return_value=mock_app,
        ):
            pass

    def test_modal_allow_button_resolves_correctly(self) -> None:
        modal = BashConfirmationModal(command=["ls"], cwd="/")
        allow_button = MagicMock()
        allow_button.id = "allow"
        event = MagicMock()
        event.button = allow_button
        result = None

        def capture_result(val: Any) -> None:
            nonlocal result
            result = val

        with patch.object(modal, "dismiss", side_effect=capture_result):
            modal.on_button_pressed(event)
            assert result is True

    def test_modal_deny_button_resolves_correctly(self) -> None:
        modal = BashConfirmationModal(command=["rm", "-rf"], cwd="/")
        deny_button = MagicMock()
        deny_button.id = "cancel"
        event = MagicMock()
        event.button = deny_button
        result = None

        def capture_result(val: Any) -> None:
            nonlocal result
            result = val

        with patch.object(modal, "dismiss", side_effect=capture_result):
            modal.on_button_pressed(event)
            assert result is False

    def test_modal_displays_working_directory(self, mock_app: Any) -> None:
        modal = BashConfirmationModal(command=["pwd"], cwd="/home/testuser")
        assert modal.cwd == "/home/testuser"


class TestSectionHeader:
    def test_init_stores_label(self) -> None:
        header = SectionHeader("Test Label")
        assert header.label == "Test Label"

    def test_update_label_sets_label_and_classes(self) -> None:
        header = SectionHeader("Initial")
        header.update_label("Updated", is_collapsed=True, is_collapsible=True)
        assert header.label == "Updated"
        assert "collapsed" in header.classes
        assert "collapsible" in header.classes

    def test_update_label_collapsed_false_removes_class(self) -> None:
        header = SectionHeader("Test")
        header.set_class(True, "collapsed")
        header.update_label("Test", is_collapsed=False, is_collapsible=False)
        assert "collapsed" not in header.classes

    def test_update_label_with_hint_when_collapsible(self) -> None:
        header = SectionHeader("Test")
        header.update_label("Test", is_collapsed=False, is_collapsible=True)
        content = header.render()
        assert "Click to expand/collapse" in str(content)

    def test_toggle_request_message_created(self) -> None:
        header = SectionHeader("Test")
        msg = SectionHeader.ToggleRequest(header)
        assert msg.header is header

    def test_on_click_posts_toggle_request(self, mock_app: Any) -> None:
        with patch.object(
            SectionHeader, "app", new_callable=PropertyMock, return_value=mock_app
        ):
            header = SectionHeader("Test")
            header.id = "test-header"
            with (
                patch.object(header, "_find_bubble_parent", return_value=None),
                patch.object(header, "post_message") as mock_post,
            ):
                event = MagicMock()
                header.on_click(event)
                mock_post.assert_called_once()
                msg = mock_post.call_args[0][0]
                assert isinstance(msg, SectionHeader.ToggleRequest)
                assert msg.header is header

    def test_on_click_stops_event_when_collapsible_bubble(self, mock_app: Any) -> None:
        with patch.object(
            SectionHeader, "app", new_callable=PropertyMock, return_value=mock_app
        ):
            header = SectionHeader("Test")
            header.id = "test-header"
            mock_bubble = MagicMock()
            mock_bubble.collapsible = True
            with (
                patch.object(header, "_find_bubble_parent", return_value=mock_bubble),
                patch.object(header, "post_message"),
            ):
                event = MagicMock()
                header.on_click(event)
                event.stop.assert_called_once()
                mock_bubble.toggle_section.assert_called_once_with(header.id)

    def test_on_click_posts_message_when_bubble_not_collapsible(
        self, mock_app: Any
    ) -> None:
        with patch.object(
            SectionHeader, "app", new_callable=PropertyMock, return_value=mock_app
        ):
            header = SectionHeader("Test")
            header.id = "test-header"
            mock_bubble = MagicMock()
            mock_bubble.collapsible = False
            with (
                patch.object(header, "_find_bubble_parent", return_value=mock_bubble),
                patch.object(header, "post_message") as mock_post,
            ):
                event = MagicMock()
                header.on_click(event)
                mock_post.assert_called_once()
                msg = mock_post.call_args[0][0]
                assert isinstance(msg, SectionHeader.ToggleRequest)
                event.stop.assert_not_called()

    def test_find_bubble_parent_returns_bubble(self) -> None:
        from supporter.tui.widgets import MessageBubble

        header = SectionHeader("Test")
        mock_content = MagicMock()
        mock_content.parent = header
        mock_bubble = MagicMock(spec=MessageBubble)
        mock_bubble.parent = mock_content
        header._parent = mock_bubble
        result = header._find_bubble_parent()
        assert result is mock_bubble

    def test_find_bubble_parent_returns_none_when_no_bubble(self) -> None:
        header = SectionHeader("Test")
        with patch.object(
            SectionHeader, "parent", new_callable=PropertyMock, return_value=None
        ):
            result = header._find_bubble_parent()
            assert result is None

    def test_find_bubble_parent_traverses_up_hierarchy(self) -> None:
        from supporter.tui.widgets import MessageBubble

        header = SectionHeader("Test")
        mock_wrapper = MagicMock()
        mock_wrapper.parent = None
        mock_bubble = MagicMock(spec=MessageBubble)
        mock_bubble.parent = mock_wrapper
        header._parent = mock_bubble
        result = header._find_bubble_parent()
        assert result is mock_bubble
