"""Tests for bidirectional task update channel (G13).

Tests verify:
1. send→poll round-trip delivers message
2. second poll returns nothing new (cursor advances)
3. never-polling task unaffected
4. updates gone after bus close
"""

from supporter.tools.delegate.bus import DelegationBus
from supporter.types import TaskUpdateSent


def test_send_task_update_and_poll() -> None:
    """AC1: send_task_update and poll_task_updates exist and work."""
    bus = DelegationBus(milestone="m1")

    # Send an update
    bus.send_task_update("task1", "Replan: skip step 3", job_id="job1")

    # Poll should return the update
    updates = bus.poll_task_updates("task1")
    assert len(updates) == 1
    assert updates[0] == "Replan: skip step 3"


def test_poll_advances_cursor() -> None:
    """AC2: second poll returns nothing new (cursor advances)."""
    bus = DelegationBus(milestone="m1")

    bus.send_task_update("task1", "First update", job_id="job1")
    bus.send_task_update("task1", "Second update", job_id="job1")

    # First poll gets both
    updates1 = bus.poll_task_updates("task1")
    assert len(updates1) == 2

    # Second poll gets nothing (cursor advanced)
    updates2 = bus.poll_task_updates("task1")
    assert len(updates2) == 0


def test_multiple_tasks_independent() -> None:
    """Different tasks have independent update lists and cursors."""
    bus = DelegationBus(milestone="m1")

    bus.send_task_update("task1", "Update for task1", job_id="job1")
    bus.send_task_update("task2", "Update for task2", job_id="job1")

    # task1 only sees its update
    task1_updates = bus.poll_task_updates("task1")
    assert task1_updates == ["Update for task1"]

    # task2 only sees its update
    task2_updates = bus.poll_task_updates("task2")
    assert task2_updates == ["Update for task2"]


def test_never_polling_task_unaffected() -> None:
    """AC3: tasks that never poll still work (no errors or state pollution)."""
    bus = DelegationBus(milestone="m1")

    bus.send_task_update("task1", "Some update", job_id="job1")

    # task2 never calls poll, and that's fine
    # Verify the state doesn't affect other operations
    snapshot = bus.get_snapshot()
    assert "task2" not in snapshot  # task2 just doesn't exist in snapshot


def test_updates_cleared_on_bus_close() -> None:
    """AC4: updates are discarded when bus closes (no memory leak)."""
    bus = DelegationBus(milestone="m1")

    bus.send_task_update("task1", "Update", job_id="job1")
    bus.send_task_update("task2", "Update", job_id="job1")

    # Verify updates exist
    assert len(bus._task_updates) > 0
    assert len(bus._task_update_cursors) > 0

    # Close the bus
    bus.close()

    # Updates should be cleared
    assert len(bus._task_updates) == 0
    assert len(bus._task_update_cursors) == 0


def test_send_task_update_publishes_event() -> None:
    """send_task_update publishes TaskUpdateSent event for TUI."""
    bus = DelegationBus(milestone="m1")

    # Subscribe to events
    queue = bus.subscribe()

    # Send an update with job_id so event is published
    bus.send_task_update("task1", "Test message", job_id="job1")

    # Event should be published
    event = queue.get_nowait()
    assert isinstance(event, TaskUpdateSent)
    assert event.task_id == "task1"
    assert event.message == "Test message"
    assert event.job_id == "job1"


def test_send_without_job_id_no_event() -> None:
    """send_task_update without job_id stores update but doesn't publish event."""
    import asyncio

    bus = DelegationBus(milestone="m1")

    # Subscribe to events
    queue = bus.subscribe()

    # Send an update without job_id
    bus.send_task_update("task1", "Test message")

    # Update should be stored
    updates = bus.poll_task_updates("task1")
    assert len(updates) == 1

    # No event published (queue should be empty)
    try:
        event = queue.get_nowait()
        # If we got an event, it's not TaskUpdateSent
        assert not isinstance(event, TaskUpdateSent)
    except asyncio.QueueEmpty:
        # Queue empty is fine
        pass


def test_poll_nonexistent_task_returns_empty() -> None:
    """Polling a task that never had updates returns empty list."""
    bus = DelegationBus(milestone="m1")

    updates = bus.poll_task_updates("nonexistent")
    assert updates == []


def test_sequential_updates_preserve_order() -> None:
    """Multiple updates on same task maintain order."""
    bus = DelegationBus(milestone="m1")

    messages = ["First", "Second", "Third", "Fourth"]
    for msg in messages:
        bus.send_task_update("task1", msg, job_id="job1")

    updates = bus.poll_task_updates("task1")
    assert updates == messages
