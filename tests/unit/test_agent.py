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
