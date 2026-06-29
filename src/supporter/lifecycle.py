"""Central runtime-state reset for in-memory singletons.

Provides a single entry point, ``reset_runtime_state()``, that drops or clears
every sync module-level singleton so that tests and reconfiguration flows start
from a clean slate without restarting the process.

Async provider / network teardown (``pool.clear_providers()``,
``DynamicPool.shutdown_all()``) is deliberately NOT included here: those
operations do I/O and must be awaited by the caller.  Call them explicitly
before or after ``reset_runtime_state()`` when a full teardown is needed.
"""

from __future__ import annotations

__all__ = ["reset_runtime_state"]


def reset_runtime_state() -> None:
    """Reset all sync in-memory singletons.

    Resets, in order:

    * ``memory._MEMORY_SINGLETON`` — working-memory singleton
    * ``recipes._STORE`` — recipe-store singleton
    * ``keypool._POOL_SINGLETON`` — API-key pool singleton
    * ``recovery_metrics._counters`` — self-healing recovery counters
    * ``decision_log._RING`` — in-memory decision ring (deque)
    * ``pool._model_cooldowns`` — model-cooldown table (OrderedDict)
    * ``history_summarizer._SUMMARIZER_CACHE`` — summarizer transcript cache
    Imports are deferred to function scope to avoid circular-import issues.
    """
    from .decision_log import reset_decision_log
    from .history_summarizer import clear_summarizer_cache
    from .keypool import reset_key_pool
    from .memory import reset_memory
    from .pool import reset_model_cooldowns
    from .recipes import reset_recipe_store
    from .recovery_metrics import reset_recovery_counters

    reset_memory()
    reset_recipe_store()
    reset_key_pool()
    reset_recovery_counters()
    reset_decision_log()
    reset_model_cooldowns()
    clear_summarizer_cache()
