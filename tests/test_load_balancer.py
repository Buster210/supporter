from unittest.mock import AsyncMock, MagicMock

import pytest

from supporter.index import LLMChunk, LLMResult, RoundRobinKeyProvider


def create_mock_provider(name):
    provider = MagicMock()
    provider.get_name.return_value = name
    provider.generate = AsyncMock(return_value=LLMResult(text=f"Response from {name}"))

    async def mock_stream(*args, **kwargs):
        yield LLMChunk(text=f"Stream from {name}", is_last=True)

    provider.generate_stream = MagicMock(side_effect=mock_stream)
    return provider


@pytest.mark.asyncio
async def test_round_robin_cycling():
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    lb = RoundRobinKeyProvider([p1, p2])

    res1 = await lb.generate("test")
    assert res1.text == "Response from P1"

    res2 = await lb.generate("test")
    assert res2.text == "Response from P2"

    res3 = await lb.generate("test")
    assert res3.text == "Response from P1"


@pytest.mark.asyncio
async def test_round_robin_streaming():
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    lb = RoundRobinKeyProvider([p1, p2])

    # First stream
    stream1 = lb.generate_stream("test")
    chunk1 = await stream1.__anext__()
    assert chunk1.text == "Stream from P1"

    # Second stream
    stream2 = lb.generate_stream("test")
    chunk2 = await stream2.__anext__()
    assert chunk2.text == "Stream from P2"


def test_load_balancer_name():
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    lb = RoundRobinKeyProvider([p1, p2])
    assert lb.get_name() == "P1 (Round Robin x2)"
