from __future__ import annotations

import json
import re

from supporter.config import config
from supporter.prompts import DEFAULT_SYSTEM_INSTRUCTION, DELEGATE_AGENT_ROSTER
from supporter.tools.catalog import (
    ORCHESTRATOR_TOOL_NAMES,
    ToolSpec,
    build_tool_catalog,
    select_delegate_tools,
    select_tools,
)
from supporter.tools.delegate.agents import delegate_allowed_tool_names
from supporter.tools.delegate.validation import validate_tasks

BROWSER_SUITE = frozenset(
    {
        "browse",
        "start_task",
        "finish_task",
        "query_playbook",
        "replay_playbook",
        "list_playbooks",
        "delete_playbook",
    }
)


def test_page_pilot_role_grants_all_seven_browser_tools() -> None:
    catalog = build_tool_catalog()
    registry = select_delegate_tools(catalog, "all", role="page-pilot")
    assert BROWSER_SUITE.issubset(set(registry))
    assert len(BROWSER_SUITE) == 7


def test_orchestrator_tool_names_does_not_list_browser_tools() -> None:
    assert BROWSER_SUITE.isdisjoint(ORCHESTRATOR_TOOL_NAMES)


def test_orchestrator_select_tools_excludes_browser_suite() -> None:
    registry = select_tools(build_tool_catalog(), ORCHESTRATOR_TOOL_NAMES)
    assert BROWSER_SUITE.isdisjoint(registry)


def test_non_page_pilot_role_with_all_excludes_browser_suite() -> None:
    catalog = build_tool_catalog()
    for role in config.delegate_agent_roster:
        if role == "page-pilot":
            continue
        registry = select_delegate_tools(catalog, "all", role=role)
        assert BROWSER_SUITE.isdisjoint(registry), (
            f"role {role!r} should not see browser tools, got {set(registry)}"
        )


def test_custom_role_with_all_excludes_browser_suite() -> None:
    registry = select_delegate_tools(build_tool_catalog(), "all", role="custom")
    assert BROWSER_SUITE.isdisjoint(registry)


def test_two_gate_consistency_for_page_pilot() -> None:
    validation_set = delegate_allowed_tool_names(role="page-pilot")
    registry_set = set(
        select_delegate_tools(build_tool_catalog(), "all", role="page-pilot")
    )
    assert validation_set == registry_set
    assert BROWSER_SUITE.issubset(validation_set)


def test_two_gate_consistency_for_code_writer() -> None:
    validation_set = delegate_allowed_tool_names(role="code_writer")
    registry_set = set(
        select_delegate_tools(build_tool_catalog(), "all", role="code_writer")
    )
    assert validation_set == registry_set
    assert BROWSER_SUITE.isdisjoint(validation_set)


def test_validate_tasks_page_pilot_task_grants_browser_tools() -> None:
    tasks_json = json.dumps(
        [{"id": "p1", "agent": "page-pilot", "task": "drive the browser"}]
    )
    validated = validate_tasks(tasks_json)
    assert BROWSER_SUITE.issubset(validated[0]["tools"])


def test_validate_tasks_code_writer_task_excludes_browser_tools() -> None:
    tasks_json = json.dumps(
        [{"id": "c1", "agent": "code_writer", "task": "implement X"}]
    )
    validated = validate_tasks(tasks_json)
    assert BROWSER_SUITE.isdisjoint(validated[0]["tools"])


def test_back_compat_select_delegate_tools_default_role_excludes_restricted() -> None:
    registry = select_delegate_tools(build_tool_catalog(), "all")
    assert BROWSER_SUITE.isdisjoint(registry)


def test_back_compat_dummy_tool_still_granted_without_role() -> None:

    def dummy_preview() -> str:
        return "ready"

    catalog = build_tool_catalog(
        extra_tools={
            "dummy_preview": ToolSpec(
                name="dummy_preview",
                callable=dummy_preview,
                delegate_allowed=True,
            )
        }
    )
    assert "dummy_preview" in select_delegate_tools(catalog, "all")
    assert "dummy_preview" in select_delegate_tools(catalog, "all", role=None)


def test_tool_spec_supports_allowed_roles_field() -> None:
    spec = ToolSpec(
        name="x",
        callable=lambda: None,
        delegate_allowed=True,
        allowed_roles=frozenset({"page-pilot"}),
    )
    assert spec.allowed_roles == frozenset({"page-pilot"})


def test_tool_spec_allowed_roles_default_is_none() -> None:
    spec = ToolSpec(name="x", callable=lambda: None, delegate_allowed=True)
    assert spec.allowed_roles is None


def test_page_pilot_present_in_delegate_roster() -> None:
    assert "page-pilot" in DELEGATE_AGENT_ROSTER
    profile = DELEGATE_AGENT_ROSTER["page-pilot"]
    assert BROWSER_SUITE.issubset(set(profile["tools"]))
    assert profile["live"] is True


def test_orchestrator_prompt_does_not_instruct_direct_browser_use() -> None:
    forbidden_phrases = (
        "browse is your PRIMARY tool",
        "browse -- a real browser",
        "use it just for a quick self-contained fact when browse is",
        "browse normally",
    )
    for phrase in forbidden_phrases:
        assert phrase not in DEFAULT_SYSTEM_INSTRUCTION, (
            f"orchestrator prompt still contains forbidden phrase: {phrase!r}"
        )


def test_orchestrator_prompt_no_imperative_browser_calls() -> None:
    imperative_patterns = [
        r"\buse\s+browse\b",
        r"\bcall\s+browse\b",
        r"\bcall\s+start_task\b",
        r"\bcall\s+finish_task\b",
        r"\bcall\s+replay_playbook\b",
        r"\bcall\s+list_playbooks\b",
        r"\bcall\s+delete_playbook\b",
        r"\bcall\s+query_playbook\b",
    ]
    for pattern in imperative_patterns:
        match = re.search(pattern, DEFAULT_SYSTEM_INSTRUCTION, flags=re.IGNORECASE)
        assert match is None, (
            f"orchestrator prompt contains imperative browser call: "
            f"pattern={pattern!r} match={match.group(0)!r}"
        )


def test_orchestrator_prompt_mentions_page_pilot_as_priority() -> None:
    lowered = DEFAULT_SYSTEM_INSTRUCTION.lower()
    assert "page-pilot" in lowered
    assert "google_search" in DEFAULT_SYSTEM_INSTRUCTION


def test_page_pilot_with_partial_tool_set_grants_only_requested() -> None:
    registry = select_delegate_tools(
        build_tool_catalog(), {"browse"}, role="page-pilot"
    )
    assert set(registry) == {"browse"}


def test_page_pilot_requesting_non_browser_tool_only() -> None:
    registry = select_delegate_tools(
        build_tool_catalog(), {"read_file"}, role="page-pilot"
    )
    assert set(registry) == {"read_file"}


def test_non_page_pilot_role_cannot_obtain_browser_tool_by_name() -> None:
    for role in config.delegate_agent_roster:
        if role == "page-pilot":
            continue
        registry = select_delegate_tools(build_tool_catalog(), BROWSER_SUITE, role=role)
        assert not registry, (
            f"role {role!r} should not see any browser tool, got {set(registry)}"
        )


def test_validate_tasks_page_pilot_partial_tool_request() -> None:
    tasks_json = json.dumps(
        [
            {
                "id": "p1",
                "agent": "page-pilot",
                "task": "navigate",
                "tools": "browse",
            }
        ]
    )
    validated = validate_tasks(tasks_json)
    assert validated[0]["tools"] == {"browse"}


def test_validate_tasks_code_writer_explicit_browser_request_denied() -> None:
    tasks_json = json.dumps(
        [
            {
                "id": "c1",
                "agent": "code_writer",
                "task": "use the browser",
                "tools": "browse",
            }
        ]
    )
    validated = validate_tasks(tasks_json)
    assert "browse" not in validated[0]["tools"]
