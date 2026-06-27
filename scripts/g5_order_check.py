"""G5 throwaway: plan_tool emits consulting→plan IN ORDER, before returning.

Run: uv run python scripts/g5_order_check.py
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from supporter.tools import planning

events: list[str] = []


async def main() -> None:
    planning.set_plan_signal_callback(
        lambda kind, text: events.append(f"{kind}:{text}")
    )

    async def fake_make_plan(
        objective: str, persona: str, model: str, tools_roster: str = ""
    ) -> str:
        # Plan must NOT exist yet when "consulting" fires.
        consulting = "consulting:Consulting planner sub-agent…"
        assert events == [consulting], events
        return "1. step\n2. verify"

    with patch("supporter.worker.make_plan", fake_make_plan):
        plan = await planning.plan_tool("scrape posts")

    planning.clear_plan_signal_callback()

    ok = (
        len(events) == 2
        and events[0] == "consulting:Consulting planner sub-agent…"
        and events[1] == "plan:1. step\n2. verify"
        and plan == "1. step\n2. verify"
    )
    print(f"  events={events!r}")
    print(f"{'PASS' if ok else 'FAIL'} — G5 in-order planner signals")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
