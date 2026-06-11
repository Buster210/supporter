from __future__ import annotations

import pytest

from supporter import recovery_metrics


@pytest.fixture(autouse=True)
def _reset() -> None:
    recovery_metrics.reset_recovery_counters()


def test_snapshot_starts_at_zero() -> None:
    assert recovery_metrics.recovery_snapshot() == {
        "key_rotations": 0,
        "re_snapshots_survived": 0,
    }


def test_record_key_rotation_increments() -> None:
    recovery_metrics.record_key_rotation()
    recovery_metrics.record_key_rotation()
    assert recovery_metrics.recovery_snapshot()["key_rotations"] == 2


def test_record_re_snapshot_increments() -> None:
    recovery_metrics.record_re_snapshot_survived()
    assert recovery_metrics.recovery_snapshot()["re_snapshots_survived"] == 1


def test_counters_are_independent() -> None:
    recovery_metrics.record_key_rotation()
    recovery_metrics.record_re_snapshot_survived()
    recovery_metrics.record_re_snapshot_survived()
    assert recovery_metrics.recovery_snapshot() == {
        "key_rotations": 1,
        "re_snapshots_survived": 2,
    }


def test_reset_zeroes_counters() -> None:
    recovery_metrics.record_key_rotation()
    recovery_metrics.reset_recovery_counters()
    assert recovery_metrics.recovery_snapshot()["key_rotations"] == 0


def test_snapshot_returns_copy_not_live_reference() -> None:
    snap = recovery_metrics.recovery_snapshot()
    recovery_metrics.record_key_rotation()
    assert snap["key_rotations"] == 0
