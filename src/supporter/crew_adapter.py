import asyncio
import threading
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import ConfigDict, Field, PrivateAttr

from .config import DEFAULT_AGENT_ROLE, DEFAULT_MODEL
from .logger import logger

_LOOP: asyncio.AbstractEventLoop | None = None
_LOOP_THREAD: threading.Thread | None = None


def _start_background_loop() -> asyncio.AbstractEventLoop:
    global _LOOP, _LOOP_THREAD
    if _LOOP is None:
        _LOOP = asyncio.new_event_loop()
        _LOOP_THREAD = threading.Thread(
            target=_LOOP.run_forever, name="SupporterAsyncBridge", daemon=True
        )
        _LOOP_THREAD.start()
        logger.debug("Background event loop started for sync-to-async bridging")
    return _LOOP


class SupporterLLM(BaseLLM):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str = Field(default=DEFAULT_MODEL)
    _supporter_provider: Any = PrivateAttr()
    _status_callback: Any | None = PrivateAttr(default=None)

    def __init__(
        self, provider: Any, status_callback: Any | None = None, **kwargs: Any
    ) -> None:
        super().__init__(model=kwargs.get("model", DEFAULT_MODEL), **kwargs)
        self._supporter_provider = provider
        self._status_callback = status_callback
        _start_background_loop()

    def call(
        self,
        messages: Any,
        tools: list[Any] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
        **kwargs: Any,
    ) -> str:
        prompt = ""
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list) and len(messages) > 0:
            prompt = messages[-1].get("content", "")

        if self._status_callback and from_agent:
            agent_role = getattr(from_agent, "role", DEFAULT_AGENT_ROLE)
            self._status_callback(agent_role)

        execution_options: dict[str, Any] = {
            "use_search": True,
            "use_code_execution": True,
        }
        if available_functions:
            execution_options["registry"] = available_functions

        loop = _start_background_loop()
        future = asyncio.run_coroutine_threadsafe(
            self._supporter_provider.generate(prompt, execution_options), loop
        )
        try:
            result = future.result()
            return str(result.text)
        except Exception as error:
            logger.error(f"SupporterLLM synchronous call failed: {error}")
            return f"Error executing model: {error}"

    async def acall(
        self,
        messages: Any,
        tools: list[Any] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
        **kwargs: Any,
    ) -> str:
        logger.debug("Entering SupporterLLM.acall (async)")
        prompt = (
            messages if isinstance(messages, str) else messages[-1].get("content", "")
        )

        execution_options: dict[str, Any] = {
            "use_search": True,
            "use_code_execution": True,
        }
        available_functions = kwargs.get("available_functions")
        if available_functions:
            execution_options["registry"] = available_functions

        result = await self._supporter_provider.generate(prompt, execution_options)
        logger.debug("Exiting SupporterLLM.acall")
        return str(result.text)

    @property
    def _llm_type(self) -> str:
        return DEFAULT_MODEL
