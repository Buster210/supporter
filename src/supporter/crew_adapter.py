import asyncio
import logging
from typing import Any, List, Optional, Union
from pydantic import PrivateAttr, Field
from crewai.llms.base_llm import BaseLLM
from .index import LLMProvider, LLMOptions
from .logger import logger


class SupporterLLM(BaseLLM):
    model: str = Field(default="supporter-gemini")
    _supporter_provider: Any = PrivateAttr()
    _status_callback: Optional[Any] = PrivateAttr(default=None)

    def __init__(self, provider: Any, status_callback: Optional[Any] = None, **kwargs):
        super().__init__(model=kwargs.get("model", "supporter-gemini"), **kwargs)
        self._supporter_provider = provider
        self._status_callback = status_callback

    def call(
        self,
        messages: Union[str, List[dict[str, str]]],
        tools: Optional[List[Any]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[dict[str, Any]] = None,
        from_task: Optional[Any] = None,
        from_agent: Optional[Any] = None,
        response_model: Optional[type] = None,
    ) -> str:
        prompt = ""
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, list) and len(messages) > 0:
            prompt = messages[-1].get("content", "")
        if self._status_callback and from_agent:
            agent_role = getattr(from_agent, "role", "Analyzing")
            self._status_callback(agent_role)
        try:
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        lambda: asyncio.run(self._supporter_provider.generate(prompt))
                    ).result()
                return result.text
            except RuntimeError:
                return asyncio.run(self._supporter_provider.generate(prompt)).text
        except Exception as e:
            logger.error(f"SupporterLLM call failed: {e}")
            return f"Error executing model: {e}"

    async def acall(
        self, messages: Union[str, List[dict[str, str]]], **kwargs: Any
    ) -> str:
        prompt = (
            messages if isinstance(messages, str) else messages[-1].get("content", "")
        )
        result = await self._supporter_provider.generate(prompt)
        return result.text

    @property
    def _llm_type(self) -> str:
        return "supporter-gemini"
