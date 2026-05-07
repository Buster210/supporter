from supporter.tools.event_bus import DelegationBus
from supporter.types import MilestoneCompleted


def test_subscribe_after_final_event_returns_completed_then_sentinel() -> None:
    bus = DelegationBus(milestone="m1")
    final = MilestoneCompleted(
        job_id="j1", milestone="m1", results=[], total_duration=1.0
    )
    bus.publish(final)

    q = bus.subscribe()
    assert q.get_nowait() is final
    assert q.get_nowait() is None
    assert len(bus._subscribers) == 0
