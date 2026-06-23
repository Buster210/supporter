from collections.abc import AsyncGenerator, AsyncIterator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from supporter.pool import DynamicPool, LLMChunk, LLMResult, clear_providers


@pytest.fixture(autouse=True)
async def pool_cleanup() -> AsyncGenerator[None, None]:
    yield
    await DynamicPool.shutdown_all()
    await clear_providers()


def create_mock_provider(name: str) -> MagicMock:
    provider = MagicMock()
    provider.get_name.return_value = name
    provider.generate = AsyncMock(return_value=LLMResult(text=f"Response from {name}"))

    async def mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(text=f"Stream from {name}", is_last=True)

    provider.generate_stream = MagicMock(side_effect=mock_stream)
    return provider


@pytest.mark.asyncio
async def test_round_robin_cycling() -> None:
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    with patch("supporter.pool.GeminiProvider", side_effect=[p1, p2]):
        lb = DynamicPool(["key1", "key2"], model_name="P1")
        res1 = await lb.generate("test")
        assert res1.text == "Response from P1"
        res2 = await lb.generate("test")
        assert res2.text == "Response from P2"
        res3 = await lb.generate("test")
        assert res3.text == "Response from P1"


@pytest.mark.asyncio
async def test_round_robin_streaming() -> None:
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    with patch("supporter.pool.GeminiProvider", side_effect=[p1, p2]):
        lb = DynamicPool(["key1", "key2"], model_name="P1")
        stream1 = lb.generate_stream("test")
        chunk1 = await stream1.__anext__()
        assert chunk1.text == "Stream from P1"
        stream2 = lb.generate_stream("test")
        chunk2 = await stream2.__anext__()
        assert chunk2.text == "Stream from P2"


@pytest.mark.asyncio
async def test_load_balancer_name() -> None:
    p1 = create_mock_provider("P1")
    p2 = create_mock_provider("P2")
    with patch("supporter.pool.GeminiProvider", side_effect=[p1, p2]):
        lb = DynamicPool(["key1", "key2"], model_name="P1")
        await lb.generate("test")
        await lb.generate("test")
        assert lb.get_name() == "P1 (Dynamic Pool x2)"


def test_error_categorization() -> None:
    from supporter.pool import is_model_error, is_rate_limit, should_trigger_fallback

    mock_rate_limit = MagicMock()
    mock_rate_limit.status = 429
    assert is_rate_limit(mock_rate_limit) is True
    assert is_rate_limit(Exception("Quota exceeded")) is True
    mock_503 = MagicMock()
    mock_503.status = 503
    assert is_model_error(mock_503) is True
    assert is_model_error(Exception("Service Unavailable")) is True
    assert should_trigger_fallback(mock_503) is True


def test_model_cooldown() -> None:
    from supporter.pool import (
        _is_model_in_cooldown,
        _mark_model_cooldown,
        _model_cooldowns,
    )

    model = "test-model"
    _model_cooldowns.clear()
    assert _is_model_in_cooldown(model) is False
    _mark_model_cooldown(model, minutes=1)
    assert _is_model_in_cooldown(model) is True
    with patch("supporter.pool.datetime") as mock_dt:
        from datetime import datetime, timedelta

        mock_dt.now.return_value = datetime.now() + timedelta(minutes=2)
        assert _is_model_in_cooldown(model) is False


@pytest.mark.asyncio
async def test_lazy_fallback_provider() -> None:
    from supporter.pool import LazyFallbackProvider

    primary = create_mock_provider("Primary")
    fallback = create_mock_provider("Fallback")
    lfp = LazyFallbackProvider(
        primary_factory=lambda: primary, fallback_factory=lambda: fallback
    )
    assert lfp._primary is None
    res = await lfp.generate("hi")
    assert res.text == "Response from Primary"
    assert lfp._primary is not None
    primary.generate.side_effect = Exception("quota")
    res = await lfp.generate("hi")
    assert res.text == "Response from Fallback"
    assert lfp._fallback is not None


@pytest.mark.asyncio
async def test_get_provider_registry() -> None:
    from supporter.pool import clear_providers, get_provider

    await clear_providers()
    p1 = get_provider("gemini")
    p2 = get_provider("gemini")
    assert p1 is p2
    await clear_providers()
    p3 = get_provider("gemini")
    assert p3 is not p1


@pytest.mark.asyncio
async def test_load_balancer_notifies_keypool_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 5xx on a slot notifies the keypool so future acquires skip the key."""
    from supporter import keypool
    from supporter import pool as pool_mod

    monkeypatch.setattr(keypool, "_default_state_path", lambda: tmp_path / "kp.json")
    monkeypatch.setattr(keypool.config, "gemini_api_keys", ["key-a", "key-b"])
    keypool.reset_key_pool()

    # Build a provider that fails with a 5xx, then succeeds.
    class _BoomError(Exception):
        status = 503

    call_count = {"n": 0}

    provider = MagicMock()
    provider.get_name.return_value = "P_A"
    provider.api_key = "key-a"  # pragma: allowlist secret

    async def gen(*args: Any, **kwargs: Any) -> LLMResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _BoomError("boom")
        return LLMResult(text="ok from key-a")

    provider.generate = AsyncMock(side_effect=gen)
    provider.generate_stream = MagicMock()

    class FakeGeminiProvider:
        def __init__(self, api_key: str, model_name: str) -> None:
            self._provider = provider
            self.api_key = api_key

        def get_name(self) -> str:
            return self._provider.get_name()  # type: ignore[no-any-return]

        async def generate(self, prompt: str, options: Any = None) -> LLMResult:
            return await self._provider.generate(prompt, options)  # type: ignore[no-any-return]

    monkeypatch.setattr(pool_mod, "GeminiProvider", FakeGeminiProvider)
    # Bypass model-level cooldown so we can see the key-level effect. Use
    # monkeypatch (not raw assignment) so it's restored at teardown — a raw
    # assign here leaks the stub into later tests and silently disables their
    # model-cooldown guard, sending them to the real network.
    monkeypatch.setattr(pool_mod, "_is_model_in_cooldown", lambda *_a, **_k: False)

    pool = DynamicPool(["key-a", "key-b"], model_name="P_A")
    res = await pool.generate("test")
    # The first attempt fails; the loop's `continue` re-runs.
    # We don't care which key the second call lands on — only that the
    # keypool was notified of key-a's failure.
    assert res.text.startswith("ok from key-")

    pool_snapshot = keypool.get_key_pool().all_health()  # type: ignore[union-attr]
    key_a_health = next(h for h in pool_snapshot if h.key == "key-a")
    assert not key_a_health.is_available()
    assert key_a_health.last_category == "transient"
    keypool.reset_key_pool()


@pytest.fixture()
def keypool_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Any, None, None]:
    """Set up a fresh KeyPool for each test and tear it down after."""
    from supporter import keypool

    monkeypatch.setattr(keypool, "_default_state_path", lambda: tmp_path / "kp.json")
    yield keypool
    keypool.reset_key_pool()


def _make_pool(
    keys: list[str],
    model_name: str = "test-model",
    pool_size: int = 2,
    provider_factory: Any = None,
) -> DynamicPool:
    """Create a DynamicPool with a trivial factory that tracks created keys."""
    created = []

    def _factory(key: str, model: str) -> MagicMock:
        p = MagicMock()
        p.get_name.return_value = f"P-{key[-2:]}"
        p.api_key = key
        created.append(key)
        return p

    factory = provider_factory or _factory
    pool = DynamicPool(
        keys,
        model_name=model_name,
        pool_size=pool_size,
        provider_factory=factory,
    )
    pool._created_keys = created  # type: ignore[attr-defined]
    return pool


@pytest.mark.asyncio
async def test_fill_slot_round_robin_when_keypool_unconfigured() -> None:
    """Round-robin when keypool is unconfigured (_fill_slot lossless proof)."""
    from supporter import keypool

    keypool.reset_key_pool()
    # Ensure get_key_pool() returns None (no keys configured).
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(keypool.config, "gemini_api_keys", [])
    try:
        pool = _make_pool(["k1", "k2", "k3"], pool_size=3)
        # First three fills should cycle k1 -> k2 -> k3 (exact old order).
        pool._fill_slot()
        pool._fill_slot()
        pool._fill_slot()
        assert pool._created_keys == ["k1", "k2", "k3"]  # type: ignore[attr-defined]
        # Next cycle: k1 again.
        pool._fill_slot()
        assert pool._created_keys[-1] == "k1"  # type: ignore[attr-defined]
    finally:
        monkeypatch.undo()
        keypool.reset_key_pool()


@pytest.mark.asyncio
async def test_fill_slot_skips_sick_key(
    keypool_isolated: Any,
) -> None:
    """Key A in cooldown, key B healthy -> repeated _fill_slot always picks B."""
    kp_mod = keypool_isolated
    kp_mod.reset_key_pool()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(kp_mod.config, "gemini_api_keys", ["key-a", "key-b"])
    try:
        pool = _make_pool(["key-a", "key-b"], pool_size=2)
        # Report key-a as failing with a quota error (cooldown = 60s first streak).
        err = Exception("quota exceeded: 429")
        kp = kp_mod.get_key_pool()
        assert kp is not None
        kp.report_failure("key-a", err)
        # key-a should now be in cooldown.
        health_a = kp.health("key-a")
        assert not health_a.is_available()
        # key-b should be healthy.
        health_b = kp.health("key-b")
        assert health_b.is_available()
        # Fill slots — should always get key-b.
        pool._fill_slot()
        pool._fill_slot()
        assert all(k == "key-b" for k in pool._created_keys)  # type: ignore[attr-defined]
    finally:
        monkeypatch.undo()
        kp_mod.reset_key_pool()


@pytest.mark.asyncio
async def test_fill_slot_all_keys_sick_returns_least_sick(
    keypool_isolated: Any,
) -> None:
    """All keys in cooldown -> returns least-sick key, never raises."""
    kp_mod = keypool_isolated
    kp_mod.reset_key_pool()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(kp_mod.config, "gemini_api_keys", ["key-a", "key-b"])
    try:
        pool = _make_pool(["key-a", "key-b"], pool_size=2)
        kp = kp_mod.get_key_pool()
        assert kp is not None
        # Use status=429 errors to get "free_tier" classification with
        # streak-dependent backoff: streak=1 → 60s, streak=2 → 240s.
        err = Exception("rate limit exceeded")
        err.status = 429  # type: ignore[attr-defined]
        # key-a: two failures → streak=2 → 240s cooldown.
        kp.report_failure("key-a", err)
        kp.report_failure("key-a", err)
        # key-b: one failure → streak=1 → 60s cooldown.
        kp.report_failure("key-b", err)
        # Both should be in cooldown.
        assert not kp.health("key-a").is_available()
        assert not kp.health("key-b").is_available()
        # key-b has smaller seconds_to_recovery (lower streak -> shorter backoff).
        rec_a = kp.health("key-a").seconds_to_recovery()
        rec_b = kp.health("key-b").seconds_to_recovery()
        assert rec_b < rec_a
        # _fill_slot should pick key-b (least-sick).
        pool._fill_slot()
        assert pool._created_keys[-1] == "key-b"  # type: ignore[attr-defined]
    finally:
        monkeypatch.undo()
        kp_mod.reset_key_pool()


@pytest.mark.asyncio
async def test_select_next_key_index_lossless_no_keypool() -> None:
    """Verify byte-identical round-robin when get_key_pool() returns None."""
    from supporter import keypool

    keypool.reset_key_pool()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(keypool.config, "gemini_api_keys", [])
    try:
        pool = _make_pool(["k1", "k2"], pool_size=2)
        # Simulate old behavior: start at index 0.
        pool.next_key_index = 0
        idx0 = pool._select_next_key_index()
        assert idx0 == 0
        assert pool.next_key_index == 1
        idx1 = pool._select_next_key_index()
        assert idx1 == 1
        assert pool.next_key_index == 0  # wraps
        idx2 = pool._select_next_key_index()
        assert idx2 == 0
        assert pool.next_key_index == 1
    finally:
        monkeypatch.undo()
        keypool.reset_key_pool()
