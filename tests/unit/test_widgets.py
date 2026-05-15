from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from supporter.tui.bubble import MessageBubble, SectionHeader
from supporter.tui.chat import ChatTurn, ThinkingIndicator
from supporter.tui.utils import ToastManager


class TestMessageBubble:
    def test_should_use_markdown_bullet_list(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("- item 1") is True

    def test_should_use_markdown_numbered_list(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("1. first") is True

    def test_should_use_markdown_heading(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("# Heading") is True

    def test_should_use_markdown_bold(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("**bold**") is True

    def test_should_use_markdown_italic(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("*italic*") is True

    def test_should_use_markdown_code(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("`code`") is True

    def test_should_use_markdown_link(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("[text](url)") is True

    def test_should_use_markdown_quote(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("> quote") is True

    def test_should_use_markdown_plain_text(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("Just plain text") is False

    def test_should_use_markdown_multiline(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("# Title\n\nSome paragraph") is True

    def test_get_meta_text_with_model(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.model = "gemma-4-31b-it"
        bubble.duration = 1.5
        assert "gemma-4-31b-it" in bubble._get_meta_text()
        assert "1.50s" in bubble._get_meta_text()
        assert bubble._get_meta_text() == "(gemma-4-31b-it in 1.50s)"

    def test_get_meta_text_no_duration(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.model = "gemini"
        meta = bubble._get_meta_text()
        assert "gemini" in meta

    def test_styles_message_meta_has_top_spacing(self) -> None:
        styles_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "supporter"
            / "tui"
            / "styles.tcss"
        )
        styles = styles_path.read_text(encoding="utf-8")
        assert ".delegation-progress" in styles
        assert "margin-top: 1;" in styles
        assert "margin-bottom: 0;" in styles
        assert "content-align-horizontal: center;" in styles
        assert ".message-meta" in styles
        assert "margin: 0 0 0 0;" in styles

    def test_delegation_signal_uses_only_bottom_margin(self) -> None:
        styles_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "supporter"
            / "tui"
            / "styles.tcss"
        )
        styles = styles_path.read_text(encoding="utf-8")
        assert ".delegation-signal" in styles
        assert "margin: 0 0 1 0;" in styles

    def test_welcome_banner_has_small_top_margin(self) -> None:
        styles_path = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "supporter"
            / "tui"
            / "styles.tcss"
        )
        styles = styles_path.read_text(encoding="utf-8")
        start = styles.index("WelcomeBanner {")
        end = styles.index("WelcomeBanner.hidden {")
        banner_block = styles[start:end]
        assert "margin-top: 1;" in banner_block

    def test_format_tool_calls_single(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        calls = [{"name": "read_file", "args": {"path": "/test.py"}}]
        result = bubble._format_tool_calls(calls)
        assert "read_file" in result
        assert "path" in result

    def test_format_tool_calls_multiple(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        calls = [
            {"name": "read_file", "args": {"path": "/a.py"}},
            {"name": "write_file", "args": {"path": "/b.py", "content": "hello"}},
        ]
        result = bubble._format_tool_calls(calls)
        assert "read_file" in result
        assert "write_file" in result

    def test_format_tool_calls_long_args_truncated(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        calls = [{"name": "write_file", "args": {"content": "x" * 100}}]
        bubble._get_tool_line_max_width = lambda: 30  # type: ignore[method-assign]
        result = bubble._format_tool_calls(calls)
        assert "..." in result

    def test_format_tool_calls_empty_args(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        calls = [{"name": "read_file", "args": {}}]
        result = bubble._format_tool_calls(calls)
        assert "read_file" in result


class TestChatTurn:
    def test_init_defaults(self) -> None:
        user_bubble = MessageBubble(role="user", content="test user")
        turn = ChatTurn(user_bubble)
        assert turn.collapsed is False
        assert turn.manually_expanded is False

    def test_toggle_collapse_expands(self) -> None:
        user_bubble = MessageBubble(role="user", content="test user")
        turn = ChatTurn(user_bubble)
        turn.collapsed = True
        turn.manually_expanded = False
        turn.toggle_collapse()
        assert turn.collapsed is False
        assert turn.manually_expanded is True

    def test_toggle_collapse_closes(self) -> None:
        user_bubble = MessageBubble(role="user", content="test user")
        turn = ChatTurn(user_bubble)
        turn.collapsed = False
        turn.manually_expanded = True
        turn.toggle_collapse()
        assert turn.collapsed is True
        assert turn.manually_expanded is False

    def test_toggle_collapse_toggles_manually_expanded(self) -> None:
        user_bubble = MessageBubble(role="user", content="test user")
        turn = ChatTurn(user_bubble)
        turn.collapsed = False
        turn.manually_expanded = True
        turn.toggle_collapse()
        assert turn.manually_expanded is False
        turn.toggle_collapse()
        assert turn.manually_expanded is True

    def test_auto_collapse_collapses_unless_manually_expanded(self) -> None:
        user_bubble = MessageBubble(role="user", content="test user")
        turn = ChatTurn(user_bubble)
        turn.manually_expanded = False
        turn.auto_collapse()
        assert turn.collapsed is True

    def test_auto_collapse_does_not_collapse_when_manually_expanded(self) -> None:
        user_bubble = MessageBubble(role="user", content="test user")
        turn = ChatTurn(user_bubble)
        turn.manually_expanded = True
        turn.auto_collapse()
        assert turn.collapsed is False

    def test_watch_collapsed_updates_bubbles(self) -> None:
        user_bubble = MessageBubble(role="user", content="test user")
        agent_bubble = MessageBubble(role="agent", content="test agent")
        turn = ChatTurn(user_bubble)
        turn.agent_bubbles = [agent_bubble]
        turn.collapsed = True
        turn.watch_collapsed(True)
        assert user_bubble.collapsed is True
        assert agent_bubble.collapsed is True


class TestThinkingIndicator:
    def test_init_defaults(self) -> None:
        indicator = ThinkingIndicator()
        assert indicator.status_label == "Thinking"
        assert indicator.active_queries == 0
        assert indicator.is_activating_mode is False

    def test_update_display_when_inactive(self) -> None:
        indicator = ThinkingIndicator()
        indicator.status_label = "Thinking"
        indicator.active_queries = 0
        indicator.is_activating_mode = False
        indicator._spinner_idx = 0
        indicator._update_display(None)
        assert indicator.display is False


class TestToastManager:
    def test_init_defaults(self) -> None:
        manager = ToastManager()
        assert manager.timeout == 5.0
        assert manager.active_toasts == {}

    def test_init_custom_timeout(self) -> None:
        manager = ToastManager(timeout=10.0)
        assert manager.timeout == 10.0

    def test_notify_timeout_clears_old_toasts(self) -> None:
        manager = ToastManager(timeout=0.1)
        manager.last_toast_time = 0
        manager.active_toasts["test"] = "old message"

        class MockApp:
            def notify(self, *args: Any, **kwargs: Any) -> None:
                pass

        manager.notify(MockApp(), "new message", "new")
        assert "test" not in manager.active_toasts

    def test_notify_removes_old_type(self) -> None:
        manager = ToastManager()

        class MockApp:
            def notify(self, *args: Any, **kwargs: Any) -> None:
                pass

        manager.notify(MockApp(), "first", "system")
        manager.notify(MockApp(), "second", "system")
        assert len(manager.active_toasts) == 1
        assert manager.active_toasts["system"] == "second"

    def test_notify_moves_toasts_to_front(self) -> None:
        manager = ToastManager()

        class MockApp:
            def notify(self, *args: Any, **kwargs: Any) -> None:
                pass

        manager.notify(MockApp(), "a", "type_a")
        manager.notify(MockApp(), "b", "type_b")
        manager.notify(MockApp(), "c", "type_c")
        keys = list(manager.active_toasts.keys())
        assert keys[0] == "type_c"

    def test_clear_removes_all(self) -> None:
        manager = ToastManager()

        class MockApp:
            def notify(self, *args: Any, **kwargs: Any) -> None:
                pass

        manager.notify(MockApp(), "message1", "type1")
        manager.notify(MockApp(), "message2", "type2")
        manager.clear(MockApp())
        assert len(manager.active_toasts) == 0

    def test_notify_batching_multiple_types(self) -> None:
        manager = ToastManager()

        class MockApp:
            def notify(self, *args: Any, **kwargs: Any) -> None:
                pass

        manager.notify(MockApp(), "msg1", "type1")
        manager.notify(MockApp(), "msg2", "type2")
        manager.notify(MockApp(), "msg3", "type3")
        assert len(manager.active_toasts) == 3
        keys = list(manager.active_toasts.keys())
        assert keys[0] == "type3"
        assert keys[1] == "type2"
        assert keys[2] == "type1"

    def test_clear_with_clear_notifications(self) -> None:
        manager = ToastManager()

        class MockAppWithClear:
            def notify(self, *args: Any, **kwargs: Any) -> None:
                pass

            def clear_notifications(self) -> None:
                pass

        app = MockAppWithClear()
        manager.notify(app, "test", "test")
        manager.clear(app)

    def test_clear_with_screen_query(self) -> Any:
        manager = ToastManager()

        class MockQueryResult:
            def remove(self) -> None:
                pass

        class MockScreen:
            def __init__(self) -> None:
                self.query_result = MockQueryResult()

            def query(self, selector: Any) -> Any:
                return self.query_result

        class MockAppWithScreen:
            def notify(self, *args: Any, **kwargs: Any) -> None:
                pass

            @property
            def screen(self) -> Any:
                return MockScreen()

        app = MockAppWithScreen()
        manager.notify(app, "test", "test")
        manager.clear(app)


class TestMessageBubbleToggleSection:
    def test_toggle_section_with_valid_section_id(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.collapsible = True
        bubble.elements = [
            {"type": "thought", "content": "thinking...", "collapsed": False},
            {"type": "content", "content": "test", "collapsed": False},
        ]
        mock_header = MagicMock(spec=SectionHeader)
        bubble.toggle_section(mock_header)

    def test_toggle_section_with_none_section_id(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.collapsible = True
        bubble.elements = [{"type": "content", "content": "test", "collapsed": False}]
        bubble.toggle_section(None)  # type: ignore

    def test_toggle_section_toggles_collapsed_state(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.collapsible = True
        bubble.elements = [
            {"type": "thought", "content": "thinking", "collapsed": False},
            {"type": "content", "content": "response", "collapsed": False},
        ]
        assert bubble.elements[0]["collapsed"] is False
        bubble.toggle_section(None)  # type: ignore


class TestMessageBubbleAddToolCall:
    def test_add_tool_call_first_call(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        bubble.add_tool_call("read_file", {"path": "/test.py"})
        assert len(bubble.tool_calls) == 1
        assert bubble.tool_calls[0]["name"] == "read_file"
        assert bubble.tool_calls[0]["args"] == {"path": "/test.py"}

    def test_add_tool_call_appends_to_existing(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        bubble.add_tool_call("read_file", {"path": "/a.py"})
        bubble.add_tool_call("write_file", {"path": "/b.py", "content": "hello"})
        assert len(bubble.tool_calls) == 2
        assert bubble.tool_calls[0]["name"] == "read_file"
        assert bubble.tool_calls[1]["name"] == "write_file"

    def test_add_tool_call_does_not_duplicate(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        bubble.add_tool_call("read_file", {"path": "/test.py"})
        bubble.add_tool_call("read_file", {"path": "/test.py"})
        assert len(bubble.tool_calls) == 1

    def test_add_tool_call_with_none_args(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        bubble.add_tool_call("list_files", None)
        assert len(bubble.tool_calls) == 1
        assert bubble.tool_calls[0]["args"] == {}

    def test_add_tool_call_updates_elements(self) -> None:
        bubble = MessageBubble(role="agent", content="")
        bubble.add_tool_call("bash", {"command": ["ls", "-la"]})
        tool_calls_elements = [
            el for el in bubble.elements if el["type"] == "tool_calls"
        ]
        assert len(tool_calls_elements) == 1
        assert tool_calls_elements[0]["type"] == "tool_calls"
        assert "collapsed" in tool_calls_elements[0]


class TestMessageBubbleShouldUseMarkdown:
    def test_markdown_detection_code_block(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("```python\nprint('hi')\n```") is True

    def test_markdown_detection_mixed_content(self) -> None:
        bubble = MessageBubble(role="user", content="")
        text = "# Title\n- item 1\n- item 2\n\nSome **bold** text"
        assert bubble._should_use_markdown(text) is True

    def test_markdown_detection_task_list(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("- [ ] task 1") is True
        assert bubble._should_use_markdown("- [x] task 2") is True

    def test_markdown_detection_empty_string(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("") is False

    def test_markdown_detection_asterisk_pattern_matched(self) -> None:
        bubble = MessageBubble(role="user", content="")
        assert bubble._should_use_markdown("just * text") is True
