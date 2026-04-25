from typing import Any

from supporter.tui.widgets import (
    ChatTurn,
    MessageBubble,
    ThinkingIndicator,
    ToastManager,
)


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

    def test_get_meta_text_no_duration(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.model = "gemini"
        meta = bubble._get_meta_text()
        assert "gemini" in meta

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
        calls = [{"name": "write_file", "args": {"content": "x" * 50}}]
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
