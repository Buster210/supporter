from supporter.tui.widgets import MessageBubble


class TestMessageBubbleToggleSection:
    def test_toggle_section_with_valid_section_id(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.collapsible = True
        bubble.elements = [
            {"type": "thought", "content": "thinking...", "collapsed": False},
            {"type": "content", "content": "test", "collapsed": False},
        ]
        bubble.toggle_section("test-header-id")

    def test_toggle_section_with_none_section_id(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.collapsible = True
        bubble.elements = [{"type": "content", "content": "test", "collapsed": False}]
        bubble.toggle_section(None)

    def test_toggle_section_toggles_collapsed_state(self) -> None:
        bubble = MessageBubble(role="agent", content="test")
        bubble.collapsible = True
        bubble.elements = [
            {"type": "thought", "content": "thinking", "collapsed": False},
            {"type": "content", "content": "response", "collapsed": False},
        ]
        assert bubble.elements[0]["collapsed"] is False
        bubble.toggle_section(None)


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
