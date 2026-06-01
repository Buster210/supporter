from unittest.mock import MagicMock, patch

from supporter.agent import ChatAgent


def _make_provider() -> MagicMock:
    provider = MagicMock()
    provider.get_name.return_value = "test-model"
    return provider


def test_trim_history_deletes_oldest_entries_when_exceeding_cap() -> None:
    provider = _make_provider()
    agent = ChatAgent(provider=provider)
    agent.history = list(range(10))

    with patch("supporter.agent.config") as mock_config:
        mock_config.history_max_turns = 3
        agent._trim_history()

    assert agent.history == [7, 8, 9]
    assert len(agent.history) == 3


def test_agent_disabled_store_is_noop() -> None:
    provider = _make_provider()
    with patch("supporter.agent.config") as mock_config:
        mock_config.durable_history_enabled = False
        agent = ChatAgent(provider=provider)
    assert agent._store is None


def test_agent_reload_seeds_history_capped_at_max_turns() -> None:
    import tempfile

    provider = _make_provider()
    with (
        tempfile.TemporaryDirectory() as tmp,
        patch("supporter.agent.config") as mock_config,
    ):
        mock_config.durable_history_enabled = True
        mock_config.history_dir = tmp
        mock_config.history_max_turns = 3

        agent1 = ChatAgent(provider=provider, session_id="cap_test")
        from google.genai.types import Content, Part

        for i in range(5):
            agent1.history.append(Content(role="user", parts=[Part(text=f"m{i}")]))
            agent1._store.append(Content(role="user", parts=[Part(text=f"m{i}")]))
        agent1._store_prev_len = 5

        agent2 = ChatAgent(provider=provider, session_id="cap_test")
        assert len(agent2.history) == 3
        assert agent2.history[0].parts[0].text == "m2"
