from unittest.mock import AsyncMock, MagicMock, patch

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


def test_afc_sync_across_compaction_writes_unique_real_turns_to_store() -> None:
    """AFC branch of _sync_history must not duplicate real turns into the
    durable store, and must not write a synthetic summary message."""
    import tempfile

    from google.genai.types import Content, Part

    provider = _make_provider()

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch("supporter.agent.config") as mock_config,
        patch(
            "supporter.history_summarizer.summarize_turns",
            new=AsyncMock(return_value=""),
        ),
    ):
        mock_config.durable_history_enabled = True
        mock_config.history_dir = tmp
        mock_config.history_max_turns = 200
        mock_config.history_compaction_enabled = True
        mock_config.history_compaction_trigger = 4
        mock_config.history_summary_keep_recent = 2

        agent = ChatAgent(provider=provider, session_id="afc_compact_test")

        # Pre-seed history AND the durable store so they agree, then bump
        # _store_prev_len so the AFC sync's `new_list[_store_prev_len:]` slice
        # covers only the genuinely new turns.
        seed_turns = [
            Content(role="user", parts=[Part(text="seed-0")]),
            Content(role="model", parts=[Part(text="seed-1")]),
            Content(role="user", parts=[Part(text="seed-2")]),
            Content(role="model", parts=[Part(text="seed-3")]),
        ]
        for t in seed_turns:
            agent.history.append(t)
            agent._store.append(t)
        agent._store_prev_len = len(agent.history)

        # First AFC sync: 3 new turns added (crosses trigger=4 -> 7).
        afc_batch_1 = [
            *agent.history,
            Content(role="user", parts=[Part(text="afc-1-u")]),
            Content(role="model", parts=[Part(text="afc-1-m")]),
            Content(role="user", parts=[Part(text="afc-1-tc")]),
        ]
        result_1 = MagicMock()
        result_1.automatic_function_calling_history = afc_batch_1
        result_1.interaction_id = "i1"
        result_1.candidates = []
        agent._sync_history(user_message=None, result=result_1)

        # Second AFC sync: 2 more turns. The afc branch must only append the
        # NEW slice (new_list[self._store_prev_len:]), not the whole list.
        afc_batch_2 = [
            *afc_batch_1,
            Content(role="model", parts=[Part(text="afc-2-m")]),
            Content(role="user", parts=[Part(text="afc-2-u")]),
        ]
        result_2 = MagicMock()
        result_2.automatic_function_calling_history = afc_batch_2
        result_2.interaction_id = "i2"
        result_2.candidates = []
        agent._sync_history(user_message=None, result=result_2)

        # Read the durable store back; assert no duplicates and no synthetic
        # summary message ever leaked into the on-disk record.
        persisted = agent._store.load()

        texts = [c.parts[0].text for c in persisted]
        assert len(texts) == len(set(texts)), f"duplicate real turns in store: {texts}"
        assert all("PREVIOUS_CONTEXT_SUMMARY" not in t for t in texts), (
            "synthetic summary message must not be persisted to the durable store"
        )
        # The new AFC turns are present exactly once.
        assert "afc-1-u" in texts
        assert "afc-2-u" in texts
        # The original real turns are still present.
        assert "seed-0" in texts
        assert "seed-3" in texts
