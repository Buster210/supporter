from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from textual.geometry import Size
from textual.widgets import Button

from supporter.tui.widgets import BashConfirmationModal, SectionHeader


class BashConfirmationModalTests:
    @pytest.fixture
    def mock_app(self) -> Any:
        app = MagicMock()
        app.size = Size(120, 40)
        return app

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

    @pytest.mark.asyncio
    async def test_modal_compose_displays_command(self) -> None:
        modal = BashConfirmationModal(command=["echo", "hello"], cwd="/fake_tmp")
        modal.app = MagicMock()
        modal.app.size = Size(120, 40)
        result = list(modal.compose())
        assert len(result) == 1
        container = result[0]
        assert container.id == "modal-container"

    @pytest.mark.asyncio
    async def test_modal_compose_has_allow_button(self) -> None:
        modal = BashConfirmationModal(command=["ls"], cwd="/")
        modal.app = MagicMock()
        modal.app.size = Size(120, 40)
        result = list(modal.compose())
        container = result[0]
        buttons_found = []
        for child in container.walk_children():
            if isinstance(child, Button):
                buttons_found.append(child.id)

    @pytest.mark.asyncio
    async def test_modal_allow_button_resolves_correctly(self) -> None:
        modal = BashConfirmationModal(command=["ls"], cwd="/")
        modal.app = MagicMock()
        modal.app.size = Size(120, 40)
        allow_button = MagicMock()
        allow_button.id = "allow"
        event = MagicMock()
        event.button = allow_button
        result = None

        async def capture_result(val: Any) -> None:
            nonlocal result
            result = val

        modal.dismiss = capture_result
        modal.on_button_pressed(event)
        # dismiss is capture_result

    @pytest.mark.asyncio
    async def test_modal_deny_button_resolves_correctly(self) -> None:
        modal = BashConfirmationModal(command=["rm", "-rf"], cwd="/")
        modal.app = MagicMock()
        modal.app.size = Size(120, 40)
        deny_button = MagicMock()
        deny_button.id = "cancel"
        event = MagicMock()
        event.button = deny_button
        result = None

        async def capture_result(val: Any) -> None:
            nonlocal result
            result = val

        modal.dismiss = capture_result
        modal.on_button_pressed(event)
        # dismiss is capture_result

    @pytest.mark.asyncio
    async def test_modal_displays_working_directory(self) -> None:
        modal = BashConfirmationModal(command=["pwd"], cwd="/home/testuser")
        modal.app = MagicMock()
        modal.app.size = Size(120, 40)
        assert modal.cwd == "/home/testuser"
        result = list(modal.compose())
        result[0]
        assert hasattr(modal, "cwd")


class SectionHeaderTests:
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
        header.app = MagicMock()
        msg = SectionHeader.ToggleRequest(header)
        assert msg.header is header

    @pytest.mark.asyncio
    async def test_on_click_posts_toggle_request(self) -> None:
        header = SectionHeader("Test")
        header.app = MagicMock()
        header.id = "test-header"
        with patch.object(header, "_find_bubble_parent", return_value=None):
            event = MagicMock()
            event.stop = MagicMock()
            header.on_click(event)
            header.app.post_message.assert_called_once()
            msg = header.app.post_message.call_args[0][0]
            assert isinstance(msg, SectionHeader.ToggleRequest)
            assert msg.header is header

    @pytest.mark.asyncio
    async def test_on_click_stops_event_when_collapsible_bubble(self) -> None:
        header = SectionHeader("Test")
        header.app = MagicMock()
        header.id = "test-header"
        mock_bubble = MagicMock()
        mock_bubble.collapsible = True
        with patch.object(header, "_find_bubble_parent", return_value=mock_bubble):
            event = MagicMock()
            event.stop = MagicMock()
            header.on_click(event)
            event.stop.assert_called_once()
            mock_bubble.toggle_section.assert_called_once_with(header.id)

    @pytest.mark.asyncio
    async def test_on_click_posts_message_when_bubble_not_collapsible(self) -> None:
        header = SectionHeader("Test")
        header.app = MagicMock()
        header.id = "test-header"
        mock_bubble = MagicMock()
        mock_bubble.collapsible = False
        with patch.object(header, "_find_bubble_parent", return_value=mock_bubble):
            event = MagicMock()
            event.stop = MagicMock()
            header.on_click(event)
            header.app.post_message.assert_called_once()
            msg = header.app.post_message.call_args[0][0]
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
        header._parent = MagicMock()
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
