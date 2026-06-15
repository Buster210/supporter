import asyncio
import contextlib
import functools
import time
from collections.abc import Callable
from typing import Any

from ...agent import ChatAgent
from ...config import DELEGATE_RETRY_BACKOFF, config
from ...logger import logger
from ...prompts import DELEGATION_RESULT_CONTRACT
from ...types import LLMProvider, TaskOutputChunk, TaskRetrying, TaskStatus
from ..catalog import build_tool_catalog, select_delegate_tools
from .backends import OPENCODE_BACKEND
from .bus import DelegationBus
from .opencode_backend import run_opencode


class _DelegateCache:
    def __init__(self) -> None:
        self.agents: dict[tuple[str, str, bool], ChatAgent] = {}
        self.locks: dict[tuple[str, str, bool], asyncio.Lock] = {}
        self.role_offsets: dict[str, int] = {}
        self.offset_counter: int = 0

    def clear(self) -> None:
        self.agents.clear()
        self.locks.clear()
        self.role_offsets.clear()
        self.offset_counter = 0


_cache = _DelegateCache()


@functools.cache
def delegate_allowed_tool_names(role: str | None = None) -> set[str]:
    return set(select_delegate_tools(build_tool_catalog(), "all", role=role))


def _cache_key(task: dict[str, Any]) -> tuple[str, str, bool] | None:
    role = task.get("agent")
    if not role or role == "custom":
        return None
    # page-pilots run concurrently against their own browser tabs; a shared
    # cache key would serialize them under role_lock, so force a fresh
    # per-instance agent (nullcontext, no serialization) for each pilot.
    if role == "page-pilot":
        return None
    return (role, task["model"], bool(task.get("live")))


def _rotated_keys_for_role(role: str) -> list[str]:
    keys = config.gemini_api_keys
    if not keys:
        raise ValueError("GEMINI_API_KEYS is missing/empty in environment")
    if role not in _cache.role_offsets:
        _cache.role_offsets[role] = _cache.offset_counter % len(keys)
        _cache.offset_counter += 1
    offset = _cache.role_offsets[role]
    n = len(keys)
    return [keys[(offset + i) % n] for i in range(n)]


def _create_sub_agent(
    task: dict[str, Any],
    provider: LLMProvider | None = None,
) -> tuple[ChatAgent, str]:
    from ...pool import get_provider

    registry = select_delegate_tools(
        build_tool_catalog(), task["tools"], role=task.get("agent")
    )
    cache_key = _cache_key(task)
    prompt = f"TASK:\n{task['task']}"
    if task["context"]:
        prompt += f"\n\nCONTEXT:\n{task['context']}"
    if task.get("result_contract", True):
        prompt += DELEGATION_RESULT_CONTRACT

    if cache_key and cache_key in _cache.agents:
        cached = _cache.agents[cache_key]
        cached.history = []
        cached.current_interaction_id = None
        return cached, prompt

    if cache_key:
        provider = get_provider(
            shared=False,
            live=task.get("live", False),
            model_name=task["model"],
            registry=registry,
            system_instruction=task["persona"],
            pool_size=1,
            keys=_rotated_keys_for_role(task["agent"]),
        )
    elif not provider:
        provider = get_provider(
            shared=False,
            live=task.get("live", False),
            model_name=task["model"],
            registry=registry,
            system_instruction=task["persona"],
        )

    agent = ChatAgent(
        provider=provider,
        registry=registry,
        use_search="google_search" in task["tools"],
        system_instruction=task["persona"],
    )

    if cache_key:
        _cache.agents[cache_key] = agent
        _cache.locks.setdefault(cache_key, asyncio.Lock())

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

            agent: ChatAgent | None = None
            cache_key = _cache_key(task)
            # Sequence counter for TaskOutputChunk events (per-task, monotonically
            # increasing)
            seq_counter: list[int] = [0]
            try:
                if task.get("backend") == OPENCODE_BACKEND:

                    def _make_chunk_publisher(
                        job_id: str, task_id: str, seq: list[int], bus: Any
                    ) -> Callable[[str], None]:
                        def _on_chunk(chunk: str) -> None:
                            seq[0] += 1
                            bus.publish(
                                TaskOutputChunk(
                                    job_id=job_id,
                                    task_id=task_id,
                                    chunk=chunk,
                                    seq=seq[0],
                                )
                            )

                        return _on_chunk

                    text, model_name, tokens = await run_opencode(
                        task,
                        on_chunk=_make_chunk_publisher(
                            job_id, task_id, seq_counter, bus
                        ),
                    )
                    step_count = 0
                else:
                    agent, prompt = _create_sub_agent(
                        task,
                        provider=provider,
                    )
                    role_lock = _cache.locks.get(cache_key) if cache_key else None
                    async with role_lock or contextlib.nullcontext():
                        result = await asyncio.wait_for(
                            agent.execute(prompt), timeout=task["timeout"]
                        )
                    text, model_name, tokens = result.text, result.model, result.usage
                    step_count = (
                        len(result.history)
                        if result.history
                        else len(result.automatic_function_calling_history or [])
                    )
                duration = time.perf_counter() - start_time
                logger.info(
                    f"Sub-agent '{task_id}' completed in {duration:.2f}s "
                    f"(attempt {attempt + 1})"
                )

                output = _truncate_delegate_output(text or "(No text output returned)")

                return {
                    "id": task_id,
                    "status": TaskStatus.COMPLETED,
                    "output": output,
                    "model": model_name,
                    "duration": duration,
                    "tokens": tokens,
                    "step_count": step_count,
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
            finally:
                if task.get("live") and agent is not None and cache_key is None:
                    close_fn = getattr(agent.provider, "close", None)
                    if close_fn:
                        try:
                            await close_fn()
                        except Exception as close_err:
                            logger.warning(
                                f"Sub-agent '{task_id}' live session close failed: "
                                f"{close_err}"
                            )

    return last_result
