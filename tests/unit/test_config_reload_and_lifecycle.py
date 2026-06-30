"""Tests for reload_config() and reset_runtime_state() — FIX 1 + FIX 2.

Verifies:
1. reload_config() mutates the module-global config object in place.
2. Importers that did ``from supporter.config import config`` see the updated
   values without rebinding (same-object propagation).
3. reset_runtime_state() brings every tracked singleton back to empty/None.

Tests are hermetic: each cleans up env and runtime state in teardown.
"""

from __future__ import annotations

import pytest

import supporter.config as _config_mod
from supporter.config import reload_config
from supporter.lifecycle import reset_runtime_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prime_all_singletons() -> None:
    """Touch each singleton so it is non-None / non-empty before reset."""
    from supporter.decision_log import log_decision
    from supporter.history_summarizer import _SUMMARIZER_CACHE
    from supporter.keypool import get_key_pool
    from supporter.memory import _get_memory
    from supporter.pool import _mark_model_cooldown
    from supporter.recipes import get_recipe_store
    from supporter.recovery_metrics import record_key_rotation

    # memory
    _get_memory()  # initialises _MEMORY_SINGLETON (may be None if no dirs, skip)

    # recipes
    get_recipe_store()  # initialises _STORE (may be None, skip)

    # keypool — only if keys configured
    get_key_pool()

    # recovery counters
    record_key_rotation()

    # decision ring
    log_decision("test_site", "test_choice")

    # model cooldowns — add one entry
    _mark_model_cooldown("test-model", minutes=60)
    # summarizer cache
    _SUMMARIZER_CACHE["test_key"] = "test_summary"


# ---------------------------------------------------------------------------
# FIX 1 — reload_config mutates in place
# ---------------------------------------------------------------------------


class TestReloadConfigMutatesInPlace:
    def test_returned_object_is_same_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reload_config() returns the existing config object, not a new one."""
        before = _config_mod.config
        monkeypatch.setenv("GEMINI_MODEL", "test-model-inplace")
        returned = reload_config()
        assert returned is before, (
            "reload_config() must return the existing config object"
        )

    def test_field_reflects_new_env_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After reload, the existing config object reflects the patched env var."""
        before = _config_mod.config
        monkeypatch.setenv("GEMINI_MODEL", "test-model-updated")
        reload_config()
        assert before.gemini_model == "test-model-updated", (
            "In-place mutation must propagate env change to existing config object"
        )

    def test_teardown_restores_original(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After monkeypatch teardown, reload_config() restores the original value."""
        original = _config_mod.config.gemini_model
        monkeypatch.setenv("GEMINI_MODEL", "changed-model")
        reload_config()
        assert _config_mod.config.gemini_model == "changed-model"
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        reload_config()
        # Should revert to the default (monkeypatch has removed our override)
        assert (
            _config_mod.config.gemini_model != "changed-model"
            or original == "changed-model"
        )


# ---------------------------------------------------------------------------
# FIX 1 — importer sees update (same-object propagation)
# ---------------------------------------------------------------------------


class TestImporterSeesUpdate:
    def test_importer_bound_reference_sees_updated_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A module-level binding ``from supporter.config import config as c``
        shares the same object; mutations via reload_config() are visible through it.
        """
        # Simulate an importer binding at import time.
        from supporter.config import config as c

        monkeypatch.setenv("GEMINI_MODEL", "propagated-model")
        reload_config()
        # c IS _config_mod.config — they share identity.
        assert c is _config_mod.config
        assert c.gemini_model == "propagated-model", (
            "Importer-bound reference must see value updated by reload_config()"
        )


# ---------------------------------------------------------------------------
# FIX 2 — reset_runtime_state clears every singleton
# ---------------------------------------------------------------------------


class TestResetRuntimeState:
    @pytest.fixture(autouse=True)
    def _teardown(self, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[misc]
        """Always reset runtime state after each test to prevent pollution."""
        yield
        reset_runtime_state()

    def test_memory_singleton_cleared(self, tmp_path: pytest.TempPathFactory) -> None:
        """After reset, _MEMORY_SINGLETON is None."""
        import supporter.memory as _mem

        _mem._get_memory()  # prime
        reset_runtime_state()
        assert _mem._MEMORY_SINGLETON is None

    def test_next_memory_call_returns_fresh_instance(self) -> None:
        """After reset, _get_memory() yields a new WorkingMemory instance."""
        import supporter.memory as _mem

        first = _mem._get_memory()
        reset_runtime_state()
        second = _mem._get_memory()
        if first is not None and second is not None:
            assert second is not first, "Should be a fresh instance after reset"

    def test_recipe_store_singleton_cleared(self) -> None:
        """After reset, _STORE is None."""
        import supporter.recipes as _rcp

        _rcp.get_recipe_store()  # prime
        reset_runtime_state()
        assert _rcp._STORE is None

    def test_key_pool_singleton_cleared(self) -> None:
        """After reset, _POOL_SINGLETON is None."""
        import supporter.keypool as _kp

        # Only prime if keys are configured (may be None otherwise)
        _kp.get_key_pool()
        reset_runtime_state()
        assert _kp._POOL_SINGLETON is None

    def test_recovery_counters_zeroed(self) -> None:
        """After reset, recovery counters are back to zero."""
        from supporter.recovery_metrics import (
            record_key_rotation,
            record_re_snapshot_survived,
            recovery_snapshot,
        )

        record_key_rotation()
        record_re_snapshot_survived()
        reset_runtime_state()
        snap = recovery_snapshot()
        assert snap["key_rotations"] == 0
        assert snap["re_snapshots_survived"] == 0

    def test_decision_ring_cleared(self) -> None:
        """After reset, the decision ring is empty."""
        from supporter.decision_log import _RING, log_decision

        log_decision("site_a", "choice_a")
        assert len(_RING) > 0
        reset_runtime_state()
        assert len(_RING) == 0

    def test_model_cooldowns_cleared(self) -> None:
        """After reset, model cooldowns dict is empty."""
        from supporter.pool import _mark_model_cooldown, _model_cooldowns

        _mark_model_cooldown("some-model", minutes=10)
        assert len(_model_cooldowns) > 0
        reset_runtime_state()
        assert len(_model_cooldowns) == 0

    def test_all_singletons_primed_then_cleared(self) -> None:
        """Prime everything in one shot, then verify reset clears them all."""
        import supporter.keypool as _kp
        import supporter.memory as _mem
        import supporter.recipes as _rcp
        from supporter.decision_log import _RING, log_decision
        from supporter.history_summarizer import _SUMMARIZER_CACHE
        from supporter.pool import _mark_model_cooldown, _model_cooldowns
        from supporter.recovery_metrics import record_key_rotation, recovery_snapshot

        # prime
        _mem._get_memory()
        _rcp.get_recipe_store()
        _kp.get_key_pool()
        record_key_rotation()
        log_decision("bulk_site", "bulk_choice")
        _mark_model_cooldown("bulk-model", minutes=5)
        _SUMMARIZER_CACHE["bulk_key"] = "bulk_summary"

        reset_runtime_state()

        assert _mem._MEMORY_SINGLETON is None
        assert _rcp._STORE is None
        assert _kp._POOL_SINGLETON is None
        assert recovery_snapshot()["key_rotations"] == 0
        assert len(_RING) == 0
        assert len(_model_cooldowns) == 0
        assert len(_SUMMARIZER_CACHE) == 0


    def test_summarizer_cache_cleared(self) -> None:
        """After reset, summarizer transcript cache is empty."""
        from supporter.history_summarizer import _SUMMARIZER_CACHE

        _SUMMARIZER_CACHE["abc"] = "summary"
        assert len(_SUMMARIZER_CACHE) > 0
        reset_runtime_state()
        assert len(_SUMMARIZER_CACHE) == 0
