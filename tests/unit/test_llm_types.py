"""Tests for src.supporter.llm.types — neutral contract types."""

from __future__ import annotations

import pytest

from supporter.llm.types import (
    GenOptions,
    ImagePart,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    tool_def_from_callable,
)

# ---------------------------------------------------------------------------
# Part types
# ---------------------------------------------------------------------------


class TestTextPart:
    def test_frozen(self) -> None:
        p = TextPart(text="hello")
        assert p.text == "hello"
        with pytest.raises(AttributeError):
            p.text = "world"  # type: ignore[misc]

    def test_eq(self) -> None:
        assert TextPart(text="a") == TextPart(text="a")
        assert TextPart(text="a") != TextPart(text="b")


class TestToolCallPart:
    def test_basic(self) -> None:
        p = ToolCallPart(name="fn", args={"x": 1}, call_id="c1")
        assert p.name == "fn"
        assert p.args == {"x": 1}
        assert p.call_id == "c1"

    def test_default_call_id(self) -> None:
        p = ToolCallPart(name="fn", args={})
        assert p.call_id is None


class TestToolResultPart:
    def test_basic(self) -> None:
        p = ToolResultPart(name="fn", response={"ok": True}, call_id="c1")
        assert p.response == {"ok": True}


class TestImagePart:
    def test_with_data(self) -> None:
        p = ImagePart(mime_type="image/png", data=b"\x89PNG")
        assert p.data == b"\x89PNG"
        assert p.ref is None

    def test_with_ref(self) -> None:
        p = ImagePart(mime_type="image/jpeg", ref="images/photo.jpg")
        assert p.ref == "images/photo.jpg"


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


class TestMessage:
    def test_basic(self) -> None:
        m = Message(role="user", parts=[TextPart(text="hi")])
        assert m.role == "user"
        assert len(m.parts) == 1
        assert isinstance(m.parts[0], TextPart)

    def test_frozen(self) -> None:
        m = Message(role="model", parts=[TextPart(text="ok")])
        with pytest.raises(AttributeError):
            m.role = "user"  # type: ignore[misc]

    def test_multi_part(self) -> None:
        m = Message(
            role="model",
            parts=[
                TextPart(text="let me check"),
                ToolCallPart(name="read_file", args={"path": "x.py"}),
            ],
        )
        assert len(m.parts) == 2


# ---------------------------------------------------------------------------
# tool_def_from_callable
# ---------------------------------------------------------------------------


def _sample_tool(query: str) -> str:
    """Search the web for a query.

    Args:
        query: The search query.

    Returns:
        Search results.
    """
    return f"results for {query}"


def _tool_with_defaults(name: str, count: int = 10) -> list[str]:
    return [name] * count


def _tool_no_hints(x) -> str:  # type: ignore[no-untyped-def]
    return str(x)


class TestToolDefFromCallable:
    def test_name_from_function(self) -> None:
        td = tool_def_from_callable(_sample_tool)
        assert td.name == "_sample_tool"

    def test_description_from_docstring(self) -> None:
        td = tool_def_from_callable(_sample_tool)
        assert "Search the web" in td.description

    def test_required_params(self) -> None:
        td = tool_def_from_callable(_sample_tool)
        assert td.parameters["type"] == "object"
        assert "query" in td.parameters["properties"]
        assert "query" in td.parameters["required"]

    def test_optional_params(self) -> None:
        td = tool_def_from_callable(_tool_with_defaults)
        props = td.parameters["properties"]
        assert "name" in props
        assert "count" in props
        # count has a default, so not required
        assert "count" not in td.parameters.get("required", [])
        assert "name" in td.parameters["required"]

    def test_no_hints_defaults_to_string(self) -> None:
        td = tool_def_from_callable(_tool_no_hints)
        assert td.parameters["properties"]["x"] == {"type": "string"}

    def test_callable_stored(self) -> None:
        td = tool_def_from_callable(_sample_tool)
        assert td.callable is _sample_tool

    def test_bool_param(self) -> None:
        def flag(enable: bool = False) -> bool:
            return enable

        td = tool_def_from_callable(flag)
        assert td.parameters["properties"]["enable"] == {"type": "boolean"}

    def test_list_param(self) -> None:
        def multi(items: list[str]) -> str:
            return str(items)

        td = tool_def_from_callable(multi)
        schema = td.parameters["properties"]["items"]
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_no_docstring(self) -> None:
        def bare(x: str) -> str:
            return x

        td = tool_def_from_callable(bare)
        assert td.name == "bare"
        assert td.description == "bare"


# ---------------------------------------------------------------------------
# GenOptions
# ---------------------------------------------------------------------------


class TestGenOptions:
    def test_defaults(self) -> None:
        g = GenOptions()
        assert g.model is None
        assert g.temperature is None
        assert g.use_search is False
        assert g.extras == {}

    def test_custom(self) -> None:
        g = GenOptions(model="gemini-2.5", temperature=0.7, extras={"top_k": 5})
        assert g.model == "gemini-2.5"
        assert g.extras["top_k"] == 5
