
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from supporter.tools.catalog import (
    ToolSpec,
    build_tool_catalog,
    select_delegate_tools,
    select_tools,
)
from supporter.tools.resolver import resolve_provider_tools


async def _dummy_tool(x: int) -> str:
    return f"done-{x}"


DUMMY_SPEC = ToolSpec(
    name="dummy",
    callable=_dummy_tool,
    delegate_allowed=True,
)

DUMMY_SPEC_NO_DELEGATE = ToolSpec(
    name="dummy",
    callable=_dummy_tool,
    delegate_allowed=False,
)


def test_dummy_tool_appears_in_catalog() -> None:
    catalog = build_tool_catalog(extra_tools={"dummy": DUMMY_SPEC})
    assert "dummy" in catalog
    assert catalog["dummy"].name == "dummy"
    assert catalog["dummy"].callable is _dummy_tool


def test_dummy_tool_in_select_tools() -> None:
    catalog = build_tool_catalog(extra_tools={"dummy": DUMMY_SPEC})
    registry = select_tools(catalog, {"dummy"})
    assert registry == {"dummy": _dummy_tool}


def test_delegate_includes_when_allowed() -> None:
    catalog = build_tool_catalog(extra_tools={"dummy": DUMMY_SPEC})
    delegate_reg = select_delegate_tools(catalog, {"dummy"})
    assert delegate_reg == {"dummy": _dummy_tool}


def test_delegate_excludes_when_not_allowed() -> None:
    catalog = build_tool_catalog(extra_tools={"dummy": DUMMY_SPEC_NO_DELEGATE})
    delegate_reg = select_delegate_tools(catalog, {"dummy"})
    assert "dummy" not in delegate_reg


@pytest.mark.asyncio
async def test_selected_tool_is_directly_invokable() -> None:
    catalog = build_tool_catalog(extra_tools={"dummy": DUMMY_SPEC})
    registry = select_tools(catalog, {"dummy"})
    result = await registry["dummy"](7)
    assert result == "done-7"


def test_resolve_provider_tools_includes_dummy() -> None:
    catalog = build_tool_catalog(extra_tools={"dummy": DUMMY_SPEC})
    registry = select_tools(catalog, {"dummy"})

    mock_google_types = SimpleNamespace(
        Tool=MagicMock(side_effect=lambda **kw: kw),
        GoogleSearch=MagicMock(return_value="search"),
        ToolCodeExecution=MagicMock(return_value="code"),
    )

    result = resolve_provider_tools(
        model_name="gemini-3.1-flash",
        tools=[],
        registry=registry,
        use_search=False,
        use_code_execution=False,
        google_types=mock_google_types,
    )

    assert _dummy_tool in result
