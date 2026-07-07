"""Process-wide self-healing recovery counters (SPEC §15).

Retries and reconnects are already counted at their own sites (delegation
``JobMetrics`` and the live provider's reconnect loop). The two recoveries that
were previously uncounted -- API-key rotations and browser stale-ref
re-snapshots -- live in decoupled subsystems with no shared object, so they are
tallied here behind one readable surface instead of being threaded through both
call stacks.

Counters are process-scoped on purpose: a session's total survived recoveries is
inherently global, not owned by any one request. Increments are integer ``+= 1``
with no intervening await, so they are safe under the single-threaded asyncio
loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .logger import logger


@dataclass
class RecoveryCounters:
    """Lifetime tally of self-healing events the session survived."""

    key_rotations: int = 0
    re_snapshots_survived: int = 0


_counters = RecoveryCounters()


def record_key_rotation() -> None:
    """Count one API-key rotation survived by the live provider."""
    _counters.key_rotations += 1
    logger.info(f"Recovery: key rotation #{_counters.key_rotations}")


def record_re_snapshot_survived() -> None:
    """Count one browser stale-ref auto re-snapshot recovered inline."""
    _counters.re_snapshots_survived += 1
    logger.info(f"Recovery: re-snapshot #{_counters.re_snapshots_survived}")


def recovery_snapshot() -> dict[str, int]:
    """Return a copy of the current recovery counters for inspection."""
    # ponytail: removed redundant snapshot() method; call asdict() directly
    return asdict(_counters)


def reset_recovery_counters() -> None:
    """Reset all counters to zero (test isolation)."""
    _counters.key_rotations = 0
    _counters.re_snapshots_survived = 0
