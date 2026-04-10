import asyncio
import concurrent.futures
from typing import Any

from crewai.llms.base_llm import BaseLLM
from pydantic import Field, PrivateAttr

from .config import DEFAULT_AGENT_ROLE, DEFAULT_MODEL
from .logger import logger


class SupporterLLM(BaseLLM):
    model: str = Field(default=DEFAULT_MODEL)
    _supporter_provider: Any = PrivateAttr()
    _status_callback: Any | None = PrivateAttr(default=None)

    def __init__(self, provider: Any, status_callback: Any | None = None, **kwargs):
        super().__init__(model=kwargs.get("model", DEFAULT_MODEL), **kwargs)
        self._supporter_provider = provider
        self._status_callback = status_callback

    def call(
        self,
        messages: str | list[dict[str, str]],
        tools: list[Any] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: type | None = None,
    ) -> str:
        prompt = ""
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list) and len(messages) > 0:
            prompt = messages[-1].get("content", "")
        if self._status_callback and from_agent:
            agent_role = getattr(from_agent, "role", DEFAULT_AGENT_ROLE)
            self._status_callback(agent_role)
        options = {"use_search": True, "use_code_execution": True}
        if available_functions:
            options["registry"] = available_functions
        try:
            try:
                asyncio.get_running_loop()

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        lambda: asyncio.run(
                            self._supporter_provider.generate(prompt, options)
                        )
                    ).result()
                return result.text
            except RuntimeError:
                return asyncio.run(
                    self._supporter_provider.generate(prompt, options)
                ).text
        except Exception as e:
            logger.error(f"SupporterLLM call failed: {e}")
            return f"Error executing model: {e}"

    async def acall(self, messages: str | list[dict[str, str]], **kwargs: Any) -> str:
        prompt = (
            messages if isinstance(messages, str) else messages[-1].get("content", "")
        )
        options = {"use_search": True, "use_code_execution": True}
        available_functions = kwargs.get("available_functions")
        if available_functions:
            options["registry"] = available_functions
        result = await self._supporter_provider.generate(prompt, options)
        return result.text

    @property
    def _llm_type(self) -> str:
        return DEFAULT_MODEL
