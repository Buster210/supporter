import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from supporter import decision_log
from supporter.agent import ChatAgent
from supporter.decision_log import (
    DecisionEntry,
    log_decision,
    recent_decisions,
)
from supporter.types import LLMResult


@pytest.fixture(autouse=True)
def _reset_decision_log_state() -> Iterator[None]:
    decision_log._RING.clear()
    yield
    lg = decision_log._decisions_logger
    if lg is not None:
        for handler in lg.handlers[:]:
            handler.close()
            lg.removeHandler(handler)
    decision_log._decisions_logger = None
    decision_log._RING.clear()


def test_log_decision_appends_to_ring() -> None:
    log_decision("brain.tool_choice", "navigate", reason="go there", options=["a", "b"])
    entries = recent_decisions()
    assert len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, DecisionEntry)
    assert entry.site == "brain.tool_choice"
    assert entry.chosen == "navigate"
    assert entry.reason == "go there"
    assert entry.options == ("a", "b")


def test_log_decision_writes_json_line_to_sibling_file(tmp_path: Path) -> None:
    with patch("supporter.config.config.log_file", str(tmp_path / "app.log")):
        log_decision(
            "scheduler.skip",
            "skip",
            options=["run", "skip"],
            reason="Dependency 'x' failed",
            correlation_id="job1:t2",
        )
        for handler in decision_log._decisions_logger.handlers:  # type: ignore[union-attr]
            handler.flush()

        log_path = tmp_path / "decisions.log"
        assert log_path.exists()
        lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["site"] == "scheduler.skip"
        assert record["chosen"] == "skip"
        assert record["options"] == ["run", "skip"]
        assert record["correlation_id"] == "job1:t2"
        assert record["timestamp"]


def test_log_decision_redacts_google_api_key() -> None:
    leaked = "AIza" + "B" * 35
    log_decision("brain.tool_choice", "respond", reason=f"used {leaked} oops")
    entry = recent_decisions()[-1]
    assert leaked not in entry.reason
    assert "***" in entry.reason


def test_log_decision_redacts_configured_key() -> None:
    fake_key = "configured-key-material-1234"
    with patch("supporter.decision_log.config.gemini_api_keys", [fake_key]):
        log_decision("brain.tool_choice", "respond", reason=f"key is {fake_key}")
    entry = recent_decisions()[-1]
    assert fake_key not in entry.reason
    assert "***" in entry.reason


def test_ring_is_bounded() -> None:
    cap = decision_log._RING_CAPACITY
    for i in range(cap + 25):
        log_decision("site", f"choice{i}")
    entries = recent_decisions()
    assert len(entries) == cap
    assert entries[-1].chosen == f"choice{cap + 24}"


def _make_result(tool_name: str | None, thoughts: str = "") -> LLMResult:
    if tool_name is None:
        part = SimpleNamespace(function_call=None, text="hi")
    else:
        part = SimpleNamespace(function_call=SimpleNamespace(name=tool_name), text=None)
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    return LLMResult(
        text="",
        candidates=[candidate],
        interaction_id="iid-9",
        thoughts=thoughts,
    )


def test_brain_hook_records_chosen_tool() -> None:
    provider = MagicMock()
    provider.get_name.return_value = "test-model"
    with patch("supporter.agent.config") as mock_config:
        mock_config.durable_history_enabled = False
        agent = ChatAgent(provider=provider)

    agent._record_brain_decision("open example.com please", _make_result("navigate"))

    entry = recent_decisions()[-1]
    assert entry.site == "brain.tool_choice"
    assert entry.chosen == "navigate"
    assert "example.com" in entry.reason
    assert entry.correlation_id == "iid-9"


def test_brain_hook_records_text_response_when_no_tool() -> None:
    provider = MagicMock()
    provider.get_name.return_value = "test-model"
    with patch("supporter.agent.config") as mock_config:
        mock_config.durable_history_enabled = False
        agent = ChatAgent(provider=provider)

    agent._record_brain_decision("just chatting", _make_result(None))

    assert recent_decisions()[-1].chosen == "text_response"
