"""Tests for src.supporter.providers.gemini_codec — neutral ↔ Gemini codec."""

from __future__ import annotations

from unittest.mock import MagicMock

from google.genai.types import (
    Blob,
    Content,
    FunctionCall,
    FunctionResponse,
    Part,
)

from supporter.llm.types import (
    GenOptions,
    ImagePart,
    Message,
    TextPart,
    ToolCallPart,
    ToolDef,
    ToolResultPart,
)
from supporter.providers.gemini_codec import (
    afc_history_to_messages,
    content_to_message,
    gen_options_to_config,
    message_to_content,
    tooldefs_to_gemini,
)

# ---------------------------------------------------------------------------
# Round-trip: neutral → Gemini → neutral
# ---------------------------------------------------------------------------


class TestMessageToContent:
    def test_text_only(self) -> None:
        msg = Message(role="user", parts=[TextPart(text="hello")])
        c = message_to_content(msg)
        assert c.role == "user"
        assert len(c.parts) == 1
        assert c.parts[0].text == "hello"

    def test_tool_call(self) -> None:
        msg = Message(
            role="model",
            parts=[ToolCallPart(name="read_file", args={"path": "x.py"})],
        )
        c = message_to_content(msg)
        fc = c.parts[0].function_call
        assert fc.name == "read_file"
        assert fc.args == {"path": "x.py"}

    def test_tool_result(self) -> None:
        msg = Message(
            role="user",
            parts=[ToolResultPart(name="read_file", response={"content": "hi"})],
        )
        c = message_to_content(msg)
        fr = c.parts[0].function_response
        assert fr.name == "read_file"
        assert fr.response == {"content": "hi"}

    def test_image_with_data(self) -> None:
        msg = Message(
            role="user",
            parts=[ImagePart(mime_type="image/png", data=b"\x89PNG")],
        )
        c = message_to_content(msg)
        idata = c.parts[0].inline_data
        assert idata.mime_type == "image/png"
        assert idata.data == b"\x89PNG"

    def test_image_no_data_uses_text_fallback(self) -> None:
        msg = Message(
            role="user",
            parts=[ImagePart(mime_type="image/jpeg", ref="photo.jpg")],
        )
        c = message_to_content(msg)
        assert c.parts[0].text == "[image:photo.jpg]"

    def test_mixed_parts(self) -> None:
        msg = Message(
            role="model",
            parts=[
                TextPart(text="let me check"),
                ToolCallPart(name="read_file", args={"path": "a.py"}),
            ],
        )
        c = message_to_content(msg)
        assert len(c.parts) == 2
        assert c.parts[0].text == "let me check"
        assert c.parts[1].function_call.name == "read_file"


# ---------------------------------------------------------------------------
# Round-trip: Gemini → neutral → Gemini
# ---------------------------------------------------------------------------


class TestContentToMessage:
    def test_text_only(self) -> None:
        c = Content(role="user", parts=[Part(text="hello")])
        msg = content_to_message(c)
        assert msg.role == "user"
        assert len(msg.parts) == 1
        assert isinstance(msg.parts[0], TextPart)
        assert msg.parts[0].text == "hello"

    def test_function_call(self) -> None:
        c = Content(
            role="model",
            parts=[Part(function_call=FunctionCall(name="browse", args={"url": "x"}))],
        )
        msg = content_to_message(c)
        assert isinstance(msg.parts[0], ToolCallPart)
        assert msg.parts[0].name == "browse"
        assert msg.parts[0].args == {"url": "x"}

    def test_function_response(self) -> None:
        c = Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name="browse", response={"ok": True}
                    )
                )
            ],
        )
        msg = content_to_message(c)
        assert isinstance(msg.parts[0], ToolResultPart)
        assert msg.parts[0].name == "browse"

    def test_inline_data(self) -> None:
        c = Content(
            role="user",
            parts=[Part(inline_data=Blob(data=b"\x89PNG", mime_type="image/png"))],
        )
        msg = content_to_message(c)
        assert isinstance(msg.parts[0], ImagePart)
        assert msg.parts[0].data == b"\x89PNG"
        assert msg.parts[0].mime_type == "image/png"

    def test_empty_parts(self) -> None:
        c = Content(role="model", parts=[])
        msg = content_to_message(c)
        assert msg.parts == []


# ---------------------------------------------------------------------------
# Full round-trip identity
# ---------------------------------------------------------------------------


class TestRoundTripIdentity:
    def test_neutral_to_gemini_to_neutral(self) -> None:
        """message_to_content then content_to_message preserves identity."""
        fixtures = [
            Message(role="user", parts=[TextPart(text="hello world")]),
            Message(
                role="model",
                parts=[
                    TextPart(text="let me check"),
                    ToolCallPart(name="read_file", args={"path": "x.py"}),
                ],
            ),
            Message(
                role="user",
                parts=[ToolResultPart(name="read_file", response={"ok": True})],
            ),
            Message(
                role="user",
                parts=[ImagePart(mime_type="image/png", data=b"\x89PNG")],
            ),
            Message(
                role="model",
                parts=[
                    TextPart(text="analysis"),
                    ToolCallPart(name="browse", args={"url": "https://a.com"}),
                    ToolResultPart(name="browse", response={"title": "A"}),
                    ImagePart(mime_type="image/jpeg", data=b"\xff\xd8"),
                ],
            ),
        ]
        for msg in fixtures:
            c = message_to_content(msg)
            roundtripped = content_to_message(c)
            # Role preserved.
            assert roundtripped.role == msg.role
            # Same number of parts.
            assert len(roundtripped.parts) == len(msg.parts)
            # Each part type and data preserved.
            for orig, rt in zip(msg.parts, roundtripped.parts, strict=True):
                assert type(orig) is type(rt)
                if isinstance(orig, TextPart):
                    assert orig.text == rt.text
                elif isinstance(orig, ToolCallPart):
                    assert orig.name == rt.name
                    assert orig.args == rt.args
                elif isinstance(orig, ToolResultPart):
                    assert orig.name == rt.name
                    assert orig.response == rt.response
                elif isinstance(orig, ImagePart):
                    assert orig.data == rt.data
                    assert orig.mime_type == rt.mime_type

    def test_gemini_to_neutral_to_gemini(self) -> None:
        """content_to_message then message_to_content preserves Gemini structure."""
        fixtures = [
            Content(role="user", parts=[Part(text="hi")]),
            Content(
                role="model",
                parts=[
                    Part(text="thinking"),
                    Part(function_call=FunctionCall(name="fn", args={"x": 1})),
                ],
            ),
            Content(
                role="user",
                parts=[
                    Part(function_response=FunctionResponse(name="fn", response={})),
                ],
            ),
        ]
        for c in fixtures:
            msg = content_to_message(c)
            roundtripped = message_to_content(msg)
            assert roundtripped.role == c.role
            assert len(roundtripped.parts) == len(c.parts)
            for orig, rt in zip(c.parts, roundtripped.parts, strict=True):
                if orig.text:
                    assert rt.text == orig.text
                if orig.function_call:
                    assert rt.function_call.name == orig.function_call.name
                    assert rt.function_call.args == orig.function_call.args
                if orig.function_response:
                    assert rt.function_response.name == orig.function_response.name


# ---------------------------------------------------------------------------
# AFC history decode
# ---------------------------------------------------------------------------


class TestAfcHistoryDecode:
    def test_multi_turn_history(self) -> None:
        history = [
            Content(role="user", parts=[Part(text="search X")]),
            Content(
                role="model",
                parts=[
                    Part(
                        function_call=FunctionCall(
                            name="google_search", args={"q": "X"}
                        )
                    )
                ],
            ),
            Content(
                role="user",
                parts=[
                    Part(
                        function_response=FunctionResponse(
                            name="google_search", response={"result": "found"}
                        )
                    )
                ],
            ),
            Content(role="model", parts=[Part(text="X is about...")]),
        ]
        messages = afc_history_to_messages(history)
        assert len(messages) == 4
        assert messages[0].role == "user"
        assert isinstance(messages[0].parts[0], TextPart)
        assert messages[1].role == "model"
        assert isinstance(messages[1].parts[0], ToolCallPart)
        assert messages[1].parts[0].name == "google_search"
        assert messages[2].parts[0].name == "google_search"
        assert isinstance(messages[2].parts[0], ToolResultPart)
        assert messages[3].parts[0].text == "X is about..."

    def test_empty_history(self) -> None:
        assert afc_history_to_messages([]) == []


# ---------------------------------------------------------------------------
# gen_options_to_config
# ---------------------------------------------------------------------------


class TestGenOptionsToConfig:
    def test_basic_options(self) -> None:
        opts = GenOptions(
            model=None,
            system_instruction="Be helpful",
            temperature=0.7,
            top_p=0.9,
            max_output_tokens=1024,
        )
        cfg = gen_options_to_config(opts, None, default_system_instruction="Default")
        assert cfg.system_instruction == "Be helpful"
        assert cfg.temperature == 0.7
        assert cfg.top_p == 0.9
        assert cfg.max_output_tokens == 1024
        assert cfg.tools is None

    def test_extras_top_k(self) -> None:
        opts = GenOptions(extras={"top_k": 40})
        cfg = gen_options_to_config(opts, None)
        assert cfg.top_k == 40

    def test_gemma_thinking(self) -> None:
        opts = GenOptions()
        cfg = gen_options_to_config(opts, None, is_gemma=True)
        # Gemma gets ThinkingLevel.HIGH — check it's set
        assert cfg.thinking_config is not None

    def test_thinking_level_from_extras(self) -> None:
        opts = GenOptions(extras={"thinking_level": "high"})
        cfg = gen_options_to_config(opts, None)
        assert cfg.thinking_config is not None

    def test_tools_passed_through(self) -> None:
        mock_tool = MagicMock()
        opts = GenOptions()
        cfg = gen_options_to_config(opts, [mock_tool])
        assert cfg.tools == [mock_tool]
        assert cfg.automatic_function_calling is not None
        assert cfg.tool_config is not None

    def test_response_mime_and_schema(self) -> None:
        opts = GenOptions(
            extras={
                "response_mime_type": "application/json",
                "response_schema": {"type": "object"},
            }
        )
        cfg = gen_options_to_config(opts, None)
        assert cfg.response_mime_type == "application/json"
        assert cfg.response_schema == {"type": "object"}

    def test_system_instruction_fallback(self) -> None:
        opts = GenOptions(system_instruction=None)
        cfg = gen_options_to_config(opts, None, default_system_instruction="Fallback")
        assert cfg.system_instruction == "Fallback"


# ---------------------------------------------------------------------------
# tooldefs_to_gemini
# ---------------------------------------------------------------------------


class TestTooldefsToGemini:
    def test_empty(self) -> None:
        result = tooldefs_to_gemini([])
        assert result == []

    def test_callable_passed_through(self) -> None:
        def my_tool(x: str) -> str:
            return x

        td = ToolDef(
            name="my_tool", description="desc", parameters={}, callable=my_tool
        )
        result = tooldefs_to_gemini([td])
        assert result[0] is my_tool

    def test_search_tool_added(self) -> None:
        result = tooldefs_to_gemini([], use_search=True, model_name="gemini-3.1-flash")
        assert any(getattr(t, "google_search", None) is not None for t in result)

    def test_code_execution_added(self) -> None:
        result = tooldefs_to_gemini([], use_code_execution=True)
        assert any(getattr(t, "code_execution", None) is not None for t in result)

    def test_gemma_strips_code_execution(self) -> None:
        result = tooldefs_to_gemini(
            [], use_code_execution=True, model_name="gemma-4-31b-it"
        )
        assert not any(getattr(t, "code_execution", None) is not None for t in result)

    def test_registry_functions_included(self) -> None:
        def reg_tool() -> str:
            return "ok"

        registry = {"reg_tool": reg_tool}
        result = tooldefs_to_gemini([], registry=registry)
        assert reg_tool in result

    def test_registry_deduplicates_declared(self) -> None:
        def my_tool(x: str) -> str:
            return x

        td = ToolDef(name="my_tool", description="d", parameters={}, callable=my_tool)
        registry = {"my_tool": lambda: "dup"}
        result = tooldefs_to_gemini([td], registry=registry)
        # my_tool callable appears once (from tooldef), registry dup skipped.
        assert result.count(my_tool) == 1
