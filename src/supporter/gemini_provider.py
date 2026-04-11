import asyncio
import functools
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import Content, GenerateContentConfig, Part

from .config import config
from .logger import logger


class GeminiProvider:
    def __init__(self, api_key: str, system_instruction: str | None = None):
        retry_options = types.HttpRetryOptions(attempts=2)
        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(retry_options=retry_options)
        )
        self.model_name = config.gemini_model
        self.default_system_instruction = (
            system_instruction or config.default_system_instruction
        )

    def _prepare_contents(
        self, prompt: str | list[Content], history: list[Content] = None
    ) -> list[Content]:
        history = history or []
        fresh_content = (
            [Content(role="user", parts=[Part(text=prompt)])]
            if isinstance(prompt, str)
            else prompt
        )
        return history + fresh_content

    def _wrap_tool(self, name: str, func: Callable) -> Callable:

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.debug(f"Calling tool: {name}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"Tool {name} success")
                return result
            except Exception as e:
                logger.error(f"Tool {name} failed: {e!s}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger.debug(f"Calling tool: {name}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Tool {name} success")
                return result
            except Exception as e:
                logger.error(f"Tool {name} failed: {e!s}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    def _transform_tools(
        self, options: dict[str, Any] | None = None
    ) -> list[Any] | None:
        if not options:
            return None
        tools = options.get("tools", [])
        registry = options.get("registry") or {}
        use_search = options.get("use_search", False)
        use_code_execution = options.get("use_code_execution", False)
        final_tools = list(tools) if tools else []
        declared_names = set()
        for t in final_tools:
            if isinstance(t, dict) and "function_declarations" in t:
                for fd in t["function_declarations"]:
                    if isinstance(fd, dict) and "name" in fd:
                        declared_names.add(fd["name"])
            elif hasattr(t, "function_declarations"):
                for fd in t.function_declarations:
                    declared_names.add(fd.name)
        for name, func in registry.items():
            if name not in declared_names:
                final_tools.append(self._wrap_tool(name, func))
        if use_search:
            final_tools.append(types.Tool(google_search=types.GoogleSearchRetrieval()))
        if use_code_execution:
            final_tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
        return final_tools or None

    async def generate(
        self, prompt: str | list[Content], options: dict[str, Any] | None = None
    ) -> Any:
        from .index import LLMResult

        options = options or {}
        interaction_id = options.get("interaction_id")
        transformed_tools = self._transform_tools(options)
        config = GenerateContentConfig(
            system_instruction=options.get("system_instruction")
            or self.default_system_instruction,
            temperature=options.get("temperature"),
            top_p=options.get("top_p"),
            top_k=options.get("top_k"),
            max_output_tokens=options.get("max_output_tokens"),
            automatic_function_calling={"disable": False}
            if transformed_tools
            else None,
            tools=transformed_tools,
        )
        start_time = time.perf_counter()
        result = None
        if interaction_id:
            try:
                result = await self.client.aio.interactions.create(
                    model=self.model_name,
                    input=prompt if isinstance(prompt, str) else str(prompt),
                    previous_interaction_id=interaction_id,
                    config=config,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to continue interaction {interaction_id}. Falling back. Error: {e}"
                )
        if not result:
            result = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=self._prepare_contents(prompt, options.get("history")),
                config=config,
            )
        end_time = time.perf_counter()
        history = getattr(result, "automatic_function_calling_history", None)
        if not history and interaction_id:
            response = getattr(result, "response", None)
            history = getattr(
                response, "automatic_function_calling_history", None
            ) or getattr(result, "history", None)
        usage_meta = getattr(result, "usage_metadata", None)
        usage = (
            {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(usage_meta, "candidates_token_count", 0)
                or 0,
                "total_tokens": getattr(usage_meta, "total_token_count", 0) or 0,
            }
            if usage_meta
            else {}
        )
        return LLMResult(
            text=result.text or "",
            model=self.model_name,
            duration=end_time - start_time,
            interaction_id=getattr(result, "id", None),
            usage=usage,
            raw=result,
            candidates=getattr(result, "candidates", []),
            automatic_function_calling_history=history,
        )

    async def generate_stream(
        self, prompt: str | list[Content], options: dict[str, Any] | None = None
    ) -> AsyncIterator[Any]:
        from .index import LLMChunk

        options = options or {}
        transformed_tools = self._transform_tools(options)
        config = GenerateContentConfig(
            system_instruction=options.get("system_instruction")
            or self.default_system_instruction,
            tools=transformed_tools,
            automatic_function_calling={"disable": False}
            if transformed_tools
            else None,
        )
        async for chunk in self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=self._prepare_contents(prompt, options.get("history")),
            config=config,
        ):
            yield LLMChunk(text=chunk.text or "", is_last=False, raw=chunk)

    def get_name(self) -> str:
        return self.model_name
