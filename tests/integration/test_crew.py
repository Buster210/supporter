from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from supporter.crew import crew_adapter
from supporter.crew.crew_adapter import SupporterLLM
from supporter.crew.crew_agent import CrewManager


@pytest.fixture
def mock_provider() -> Any:
    provider = MagicMock()

    async def mock_generate(*args: Any, **kwargs: Any) -> Any:
        return MagicMock(text="Mocked response")

    provider.generate = MagicMock(side_effect=mock_generate)
    result = MagicMock()
    result.text = "Mocked response"
    provider.generate.return_value = result
    return provider


def test_crew_manager_init(mock_provider: Any) -> None:
    manager = CrewManager(provider=mock_provider)
    assert isinstance(manager.llm, SupporterLLM)


@pytest.mark.asyncio
async def test_coordinate_execution(mock_provider: Any) -> None:
    manager = CrewManager(provider=mock_provider)
    with patch("supporter.crew.crew_agent.Crew") as mock_crew_cls:
        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = "Final report content"
        mock_crew.agents = [MagicMock(role="Researcher")]
        mock_crew_cls.return_value = mock_crew
        result = await manager.coordinate_execution("test topic")
        assert "Final report content" in result.text
        assert result.model == "CrewAI (Multi-Agent)"
        mock_crew.kickoff.assert_called_once()


@pytest.mark.asyncio
async def test_coordinate_execution_error(mock_provider: Any) -> None:
    manager = CrewManager(provider=mock_provider)
    with patch("supporter.crew.crew_agent.Crew") as mock_crew_cls:
        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = Exception("Orchestration error")
        mock_crew_cls.return_value = mock_crew
        result = await manager.coordinate_execution("test topic")
        assert "Error executing crew: Orchestration error" in result.text


@pytest.mark.asyncio
async def test_coordinate_execution_with_tasks_output(mock_provider: Any) -> None:
    manager = CrewManager(provider=mock_provider)
    with patch("supporter.crew.crew_agent.Crew") as mock_crew_cls:
        mock_crew = MagicMock()
        mock_result = MagicMock()
        mock_task_output = MagicMock()
        mock_task_output.agent = "Senior Researcher"
        mock_result.tasks_output = [mock_task_output]
        mock_result.__str__.return_value = "Task result"  # type: ignore[attr-defined]
        mock_crew.kickoff.return_value = mock_result
        mock_crew_cls.return_value = mock_crew
        result = await manager.coordinate_execution("test topic")
        assert "Task result" in result.text
        assert "Senior Researcher" in result.usage["agents"]


def test_supporter_llm_call(mock_provider: Any) -> Any:
    llm = SupporterLLM(provider=mock_provider)

    def mock_run_coro(coro: Any, loop: Any) -> Any:
        coro.close()
        mock_future = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "LLM response"
        mock_future.result.return_value = mock_result
        return mock_future

    with patch(
        "supporter.crew.crew_adapter.asyncio.run_coroutine_threadsafe",
        side_effect=mock_run_coro,
    ):
        response = llm.call("hello")
        assert response == "LLM response"


def test_supporter_llm_call_list_messages(mock_provider: Any) -> Any:
    llm = SupporterLLM(provider=mock_provider)

    def mock_run_coro(coro: Any, loop: Any) -> Any:
        coro.close()
        mock_future = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Result"
        mock_future.result.return_value = mock_result
        return mock_future

    with patch(
        "supporter.crew.crew_adapter.asyncio.run_coroutine_threadsafe",
        side_effect=mock_run_coro,
    ):
        response = llm.call(messages=[{"role": "user", "content": "hello"}])
        assert response == "Result"
        response = llm.call(messages="hi")
        assert response == "Result"


def test_supporter_llm_call_with_callback_and_functions(mock_provider: Any) -> Any:
    callback = MagicMock()
    llm = SupporterLLM(provider=mock_provider, status_callback=callback)

    def mock_run_coro(coro: Any, loop: Any) -> Any:
        coro.close()
        mock_future = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Result"
        mock_future.result.return_value = mock_result
        return mock_future

    with patch(
        "supporter.crew.crew_adapter.asyncio.run_coroutine_threadsafe",
        side_effect=mock_run_coro,
    ):
        mock_agent = MagicMock()
        mock_agent.role = "Tester"
        response = llm.call(
            "test", from_agent=mock_agent, available_functions={"func": lambda: None}
        )
        assert response == "Result"
        callback.assert_called_with("Tester")


def test_supporter_llm_call_exception(mock_provider: Any) -> Any:
    llm = SupporterLLM(provider=mock_provider)

    def mock_run_coro(coro: Any, loop: Any) -> Any:
        coro.close()
        mock_future = MagicMock()
        mock_future.result.side_effect = RuntimeError("Bridge failure")
        return mock_future

    with patch(
        "supporter.crew.crew_adapter.asyncio.run_coroutine_threadsafe",
        side_effect=mock_run_coro,
    ):
        response = llm.call("hello")
        assert "Error executing model: Bridge failure" in response


@pytest.mark.asyncio
async def test_supporter_llm_acall(mock_provider: Any) -> None:
    llm = SupporterLLM(provider=mock_provider)
    response = await llm.acall("hello")
    assert response == "Mocked response"
    mock_provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_supporter_llm_acall_complex(mock_provider: Any) -> None:
    llm = SupporterLLM(provider=mock_provider)
    response = await llm.acall(messages=[{"content": "async hello"}])
    assert response == "Mocked response"
    response = await llm.acall("test", available_functions={"f": lambda: None})
    assert response == "Mocked response"


def test_supporter_llm_type_and_init(mock_provider: Any) -> None:
    llm = SupporterLLM(provider=mock_provider, model="custom-model")
    assert llm._llm_type == "custom-model"
    assert llm.model == "custom-model"


def test_start_background_loop_idempotency(mock_provider: Any) -> None:
    SupporterLLM(provider=mock_provider)
    assert crew_adapter._LOOP is not None
    assert crew_adapter._LOOP_THREAD is not None
    assert crew_adapter._LOOP_THREAD.is_alive()
