import asyncio
import functools
import time
from typing import Any

from ...agent import ChatAgent
from ...config import DELEGATE_RETRY_BACKOFF, config
from ...logger import logger
from ...types import LLMProvider, TaskRetrying, TaskStatus
from ..catalog import build_tool_catalog, select_delegate_tools
from .bus import DelegationBus


@functools.cache
def delegate_allowed_tool_names() -> set[str]:
    return set(select_delegate_tools(build_tool_catalog(), "all"))


def _create_sub_agent(
    task: dict[str, Any],
    provider: LLMProvider | None = None,
) -> tuple[ChatAgent, str]:
    from ...pool import get_provider

    registry = select_delegate_tools(build_tool_catalog(), task["tools"])
    if not provider:
        provider = get_provider(
            shared=False, model_name=task["model"], registry=registry
        )

    agent = ChatAgent(
        provider=provider,
        registry=registry,
        use_search="google_search" in task["tools"],
        system_instruction=task["persona"],
    )

    prompt = f"TASK:\n{task['task']}"
    if task["context"]:
        prompt += f"\n\nCONTEXT:\n{task['context']}"
    return agent, prompt


def _truncate_delegate_output(output: str) -> str:
    if len(output) <= config.delegate_max_output_chars:
        return output
    return output[: config.delegate_max_output_chars] + "\n\n[Output truncated...]"


async def run_sub_agent(
    task: dict[str, Any],
    semaphore: asyncio.Semaphore,
    bus: DelegationBus,
    job_id: str,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    max_retries = task.get("max_retries", 0)
    retry_delays = DELEGATE_RETRY_BACKOFF
    last_result: dict[str, Any] = {}

    for attempt in range(max_retries + 1):
        async with semaphore:
            start_time = time.perf_counter()
            task_id = task["id"]
            agent_label = task.get("agent") or "custom"

            if attempt > 0:
                delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                logger.info(
                    f"Sub-agent '{task_id}' retry {attempt}/{max_retries} "
                    f"after {delay}s backoff"
                )
                bus.publish(
                    TaskRetrying(
                        job_id=job_id,
                        task_id=task_id,
                        attempt=attempt + 1,
                        reason=last_result.get("output", "unknown error"),
                    )
                )
                await asyncio.sleep(delay)

            logger.info(
                f"Sub-agent '{task_id}' [{agent_label}] attempt {attempt + 1} started"
            )

            try:
                agent, prompt = _create_sub_agent(
                    task,
                    provider=provider,
                )
                result = await asyncio.wait_for(
                    agent.execute(prompt), timeout=task["timeout"]
                )
                duration = time.perf_counter() - start_time
                logger.info(
                    f"Sub-agent '{task_id}' completed in {duration:.2f}s "
                    f"(attempt {attempt + 1})"
                )

                output = _truncate_delegate_output(
                    result.text or "(No text output returned)"
                )

                return {
                    "id": task_id,
                    "status": TaskStatus.COMPLETED,
                    "output": output,
                    "model": result.model,
                    "duration": duration,
                    "tokens": result.usage,
                }
            except TimeoutError:
                logger.warning(
                    f"Sub-agent '{task_id}' timed out after {task['timeout']}s "
                    f"(attempt {attempt + 1}) — no retry for timeouts"
                )
                return {
                    "id": task_id,
                    "status": TaskStatus.TIMEOUT,
                    "output": (
                        f"Error: Task exceeded execution limit of {task['timeout']}s"
                    ),
                    "duration": time.perf_counter() - start_time,
                    "tokens": {},
                }
            except Exception as e:
                logger.error(
                    f"Sub-agent '{task_id}' failed (attempt {attempt + 1}): {e}"
                )
                last_result = {
                    "id": task_id,
                    "status": TaskStatus.ERROR,
                    "output": f"Error [{type(e).__name__}]: {e}",
                    "duration": time.perf_counter() - start_time,
                    "tokens": {},
                }
                if attempt < max_retries:
                    continue
                return last_result

    return last_result
