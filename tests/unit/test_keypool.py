"""Unit tests for the health-aware Gemini key pool."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from supporter import keypool
from supporter.keypool import (
    KeyPool,
    coerce_keys,
    get_key_pool,
    pool_snapshot,
    reset_key_pool,
)

# ---------------------------------------------------------------------------
# coerce_keys
# ---------------------------------------------------------------------------


def test_coerce_keys_from_csv() -> None:
    assert coerce_keys("a,b ,c") == ["a", "b", "c"]


def test_coerce_keys_from_json_array() -> None:
    assert coerce_keys('["a","b"]') == ["a", "b"]


def test_coerce_keys_from_iterable() -> None:
    assert coerce_keys(("a", "b")) == ["a", "b"]


def test_coerce_keys_empty_and_none() -> None:
    assert coerce_keys(None) == []
    assert coerce_keys("") == []
    assert coerce_keys("   ") == []


def test_coerce_keys_rejects_bad_json() -> None:
    with pytest.raises(ValueError):
        coerce_keys("[oops")


def test_coerce_keys_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        coerce_keys({"a": 1})
    with pytest.raises(ValueError):
        coerce_keys([1, 2])


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def _exc(message: str, status: int | None = None) -> Exception:
    err = Exception(message)
    if status is not None:
        err.status = status  # type: ignore[attr-defined]
    return err


def test_classify_revoked_patterns() -> None:
    assert keypool._classify(_exc("API key not valid")) == "revoked"
    assert keypool._classify(_exc("API_KEY_INVALID")) == "revoked"
    assert keypool._classify(_exc("PERMISSION_DENIED")) == "revoked"


def test_classify_free_tier_patterns() -> None:
    assert keypool._classify(_exc("Quota exceeded for metric")) == "free_tier"
    assert keypool._classify(_exc("free tier limit hit")) == "free_tier"
    assert keypool._classify(_exc("RESOURCE_EXHAUSTED")) == "free_tier"


def test_classify_transient_patterns() -> None:
    assert keypool._classify(_exc("internal error")) == "transient"
    assert keypool._classify(_exc("service unavailable")) == "transient"
    assert keypool._classify(_exc("returned 503")) == "transient"


def test_classify_status_4xx_means_revoked() -> None:
    assert keypool._classify(_exc("some failure", status=401)) == "revoked"
    assert keypool._classify(_exc("some failure", status=403)) == "revoked"
    # 408 / 429 are *not* treated as revoked.
    assert keypool._classify(_exc("timeout", status=408)) != "revoked"
    assert keypool._classify(_exc("rate limit", status=429)) == "free_tier"


def test_classify_status_5xx_means_transient() -> None:
    assert keypool._classify(_exc("boom", status=500)) == "transient"
    assert keypool._classify(_exc("boom", status=502)) == "transient"
    assert keypool._classify(_exc("boom", status=503)) == "transient"
    assert keypool._classify(_exc("boom", status=504)) == "transient"


def test_classify_unknown() -> None:
    assert keypool._classify(_exc("definitely not a recognised error")) == "unknown"


def test_cooldown_schedule_free_tier_backs_off() -> None:
    # First failure: 60s. Each subsequent: 4x, capped at 3600s.
    assert keypool._cooldown_seconds("free_tier", 1) == 60
    assert keypool._cooldown_seconds("free_tier", 2) == 240
    assert keypool._cooldown_seconds("free_tier", 3) == 900
    assert keypool._cooldown_seconds("free_tier", 4) == 3600
    # Beyond schedule: stays at the cap.
    assert keypool._cooldown_seconds("free_tier", 99) == 3600


def test_cooldown_schedule_transient_short() -> None:
    assert keypool._cooldown_seconds("transient", 1) == 30


def test_cooldown_schedule_revoked_very_long() -> None:
    assert keypool._cooldown_seconds("revoked", 1) == 24 * 60 * 60


# ---------------------------------------------------------------------------
# KeyPool
# ---------------------------------------------------------------------------


def _new_pool(tmp_path: Path, *keys: str) -> KeyPool:
    return KeyPool(keys, state_path=tmp_path / "keypool.json")


def test_pool_requires_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        KeyPool([], state_path=tmp_path / "unused.json")
    with pytest.raises(ValueError):
        KeyPool([""], state_path=tmp_path / "unused.json")


def test_pool_dedupes_preserving_order(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1", "k2", "k1", "k3")
    assert pool.keys == ("k1", "k2", "k3")


def test_acquire_returns_healthy_key(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1", "k2", "k3")
    seen = {pool.acquire() for _ in range(3)}
    assert seen == {"k1", "k2", "k3"}


def test_acquire_skips_in_cooldown_keys(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1", "k2", "k3")
    pool.report_failure("k1", _exc("Quota exceeded for metric"))
    pool.report_failure("k1", _exc("Quota exceeded for metric"))
    # k1 should now be in long cooldown; acquire should skip it.
    for _ in range(10):
        assert pool.acquire() in {"k2", "k3"}


def test_acquire_returns_none_when_all_in_cooldown(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1", "k2")
    for k in pool.keys:
        pool.report_failure(k, _exc("Quota exceeded for metric"))
    assert pool.acquire() is None


def test_success_resets_streak(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1")
    pool.report_failure("k1", _exc("Quota exceeded"))
    assert pool.health("k1").failure_streak == 1
    pool.report_success("k1")
    assert pool.health("k1").failure_streak == 0
    assert pool.health("k1").is_available()


def test_redacts_key_in_last_error(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "supersecret-1234")
    pool.report_failure("supersecret-1234", _exc("failure for supersecret-1234"))
    health = pool.health("supersecret-1234")
    assert "supersecret-1234" not in health.last_error
    assert "***" in health.last_error


def test_persists_and_reloads_cooldowns(tmp_path: Path) -> None:
    state = tmp_path / "kp.json"
    p1 = KeyPool(("k1", "k2"), state_path=state)
    p1.report_failure("k1", _exc("Quota exceeded"))

    p2 = KeyPool(("k1", "k2"), state_path=state)
    # k1 should still be in cooldown after reload.
    assert not p2.health("k1").is_available()
    # The freshly-loaded cursor is independent — acquire should still work
    # because k2 is healthy.
    assert p2.acquire() == "k2"


def test_corrupt_state_file_is_tolerated(tmp_path: Path) -> None:
    state = tmp_path / "kp.json"
    state.write_text("not json {{{", encoding="utf-8")
    pool = KeyPool(("k1",), state_path=state)
    assert pool.acquire() == "k1"


def test_unknown_entry_marks_unknown_with_short_cooldown(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1")
    pool.report_failure("k1", _exc("something new"))
    h = pool.health("k1")
    assert h.last_category == "unknown"
    assert h.failure_streak == 1
    assert h.cooldown_until > 0


def test_streak_increments_on_consecutive_failures(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1")
    pool.report_failure("k1", _exc("Quota exceeded"))
    pool.report_failure("k1", _exc("Quota exceeded"))
    assert pool.health("k1").failure_streak == 2


def test_pool_snapshot_shape(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1", "k2")
    pool.report_failure("k1", _exc("Quota exceeded"))
    snap = pool_snapshot(pool)
    assert snap["configured"] is True
    assert snap["total"] == 2
    assert snap["available"] == 1
    suffixes = {k["key_suffix"] for k in snap["keys"]}
    # Last-4 of k1/k2.
    assert suffixes == {"k1", "k2"}


def test_pool_snapshot_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_key_pool()
    monkeypatch.setattr(keypool.config, "gemini_api_keys", [])
    assert pool_snapshot() == {"configured": False, "keys": []}


def test_threadsafe_acquire(tmp_path: Path) -> None:
    pool = _new_pool(tmp_path, "k1", "k2", "k3", "k4")
    acquired: list[str | None] = []
    lock = threading.Lock()

    def worker() -> None:
        for _ in range(50):
            k = pool.acquire()
            with lock:
                acquired.append(k)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # All acquisitions should be one of the four keys (or None if everyone
    # happened to be in cooldown, which we didn't trigger).
    assert all(k in {"k1", "k2", "k3", "k4"} for k in acquired)


def test_get_key_pool_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_key_pool()
    monkeypatch.setattr(keypool.config, "gemini_api_keys", ["a", "b"])
    p1 = get_key_pool()
    p2 = get_key_pool()
    assert p1 is p2
    assert p1 is not None
    assert p1.acquire() in {"a", "b"}


def test_get_key_pool_none_when_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_key_pool()
    monkeypatch.setattr(keypool.config, "gemini_api_keys", [])
    assert get_key_pool() is None
