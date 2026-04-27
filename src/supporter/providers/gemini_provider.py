import asyncio
import functools
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

from google import genai
from google.genai import types
from google.genai.types import Content, GenerateContentConfig, Part

from ..config import DEFAULT_SYSTEM_INSTRUCTION, config
from ..logger import logger
from ..types import (
    LLMChunk,
    LLMOptions,
    LLMResult,
)


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model_name: str | None = None,
    ):
        target_model = model_name or config.gemini_model
        retry_options = types.HttpRetryOptions(attempts=2)

        self.client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(retry_options=retry_options)
        )
        self.model_name = target_model

        self._tool_cache: list[Any] | None = None
        self._last_tool_key: Any = None
        logger.info(
            f"GeminiProvider initialized: model={target_model}, http_retry_attempts=2"
        )

    def _prepare_contents(
        self,
        prompt: str | list[Content],
        history: list[Content] | None = None,
    ) -> list[Content]:
        history = history or []
        fresh_content = (
            [Content(role="user", parts=[Part(text=prompt)])]
            if isinstance(prompt, str)
            else prompt
        )
        return history + fresh_content

    def _wrap_tool(self, name: str, func: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info(
                f"Tool '{name}' invoked (async): args={args!r}, kwargs={kwargs!r}"
            )
            try:
                result = await func(*args, **kwargs)
                logger.info(f"Tool '{name}' completed successfully")
                logger.debug(f"Tool '{name}' full result: {result!r}")
                return result
            except Exception as e:
                logger.error(f"Async tool '{name}' failed [{type(e).__name__}]: {e}")
                raise

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.info(
                f"Tool '{name}' invoked (sync): args={args!r}, kwargs={kwargs!r}"
            )
            try:
                result = func(*args, **kwargs)
                logger.info(f"Tool '{name}' completed successfully")
                logger.debug(f"Tool '{name}' full result: {result!r}")
                return result
            except Exception as e:
                logger.error(f"Sync tool '{name}' failed [{type(e).__name__}]: {e}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    def _transform_tools(self, options: LLMOptions | None = None) -> list[Any] | None:
        if not options:
            return None

        tools = options.get("tools", [])
        registry = options.get("registry") or {}
        use_search = options.get("use_search", False)
        use_code_execution = options.get("use_code_execution", False)

        def _get_tool_id(t: Any) -> Any:
            if hasattr(t, "__name__") and hasattr(t, "__module__"):
                return (t.__module__, t.__name__)
            return id(t)

        current_identity_key = (
            tuple(_get_tool_id(t) for t in tools),
            tuple(sorted(registry.keys())),
            use_search,
            use_code_execution,
        )

        if self._last_tool_key == current_identity_key:
            return self._tool_cache

        final_tools: list[Any] = list(tools) if tools else []
        declared_names = self._extract_declared_tool_names(final_tools)

        for name, func in registry.items():
            if name not in declared_names:
                final_tools.append(self._wrap_tool(name, func))

        if use_search:
            final_tools.append(types.Tool(google_search=types.GoogleSearch()))
        if use_code_execution:
            final_tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

        self._tool_cache = final_tools or None
        self._last_tool_key = current_identity_key
        return self._tool_cache

    def _extract_declared_tool_names(self, tools: list[Any]) -> set[str]:
        names = set()
        for tool in tools:
            declarations = getattr(tool, "function_declarations", [])
            if isinstance(tool, dict):
                declarations = tool.get("function_declarations", [])

            for decl in declarations:
                name = (
                    decl.get("name")
                    if isinstance(decl, dict)
                    else getattr(decl, "name", None)
                )
                if name:
                    names.add(name)
        return names

    async def generate(
        self,
        prompt: str | list[Content],
        options: LLMOptions | None = None,
    ) -> LLMResult:
        options = options or {}
        interaction_id = options.get("interaction_id")
        transformed_tools = self._transform_tools(options)

        generation_config = GenerateContentConfig(
            system_instruction=options.get("system_instruction")
            or DEFAULT_SYSTEM_INSTRUCTION,
            temperature=options.get("temperature"),
            top_p=options.get("top_p"),
            top_k=options.get("top_k"),
            max_output_tokens=options.get("max_output_tokens"),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            )
            if transformed_tools
            else None,
            tools=transformed_tools,
            tool_config=types.ToolConfig(include_server_side_tool_invocations=True) if transformed_tools else None,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        start_time = time.perf_counter()
        history = options.get("history") or []
        logger.info(
            f"generate(): model={self.model_name}, history_turns={len(history)}, "
            f"tools={len(transformed_tools) if transformed_tools else 0}"
        )
        sys_inst = options.get("system_instruction") or DEFAULT_SYSTEM_INSTRUCTION
        logger.debug(f"generate() system_instruction: {sys_inst}")
        logger.debug(f"generate() context options: {options!r}")
        result: Any = None

        if interaction_id:
            try:
                result = await self.client.aio.interactions.create(
                    model=self.model_name,
                    input=prompt if isinstance(prompt, str) else str(prompt),
                    previous_interaction_id=interaction_id,
                    generation_config=cast(Any, generation_config),
                )
            except Exception as e:
                logger.info(
                    f"Interaction resumption failed (id={interaction_id}, "
                    f"{type(e).__name__}: {e}) — falling back to standard generation"
                )

        if not result:
            result = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=cast(
                    Any, self._prepare_contents(prompt, options.get("history"))
                ),
                config=generation_config,
            )

        end_time = time.perf_counter()

        afc_history = getattr(result, "automatic_function_calling_history", None)
        if not afc_history and interaction_id:
            response = getattr(result, "response", None)
            afc_history = getattr(
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

        thoughts = ""
        if result.candidates and result.candidates[0].content:
            text_parts = result.candidates[0].content.parts
            if text_parts:
                thoughts = "".join(
                    [
                        p.text
                        for p in text_parts
                        if p.text and getattr(p, "thought", False)
                    ]
                )

        duration = end_time - start_time
        logger.info(
            f"generate() done: model={self.model_name}, duration={duration:.3f}s, "
            f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
            f"completion_tokens={usage.get('completion_tokens', '?')}"
        )
        if result.candidates:
            cand = result.candidates[0]
            logger.debug(f"generate() candidate[0] parts: {cand.content.parts!r}")
            meta = getattr(cand, "grounding_metadata", None)
            if meta:
                logger.debug(f"generate() grounding_metadata: {meta!r}")
        return LLMResult(
            text=result.text or "",
            thoughts=thoughts,
            model=self.model_name,
            duration=end_time - start_time,
            interaction_id=getattr(result, "id", None),
            usage=usage,
            raw=result,
            candidates=getattr(result, "candidates", []),
            automatic_function_calling_history=afc_history,
        )

    async def generate_stream(
        self,
        prompt: str | list[Content],
        options: LLMOptions | None = None,
    ) -> AsyncIterator[LLMChunk]:
        options = options or {}
        transformed_tools = self._transform_tools(options)
        generation_config = GenerateContentConfig(
            system_instruction=options.get("system_instruction")
            or DEFAULT_SYSTEM_INSTRUCTION,
            tools=transformed_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=False
            )
            if transformed_tools
            else None,
            tool_config=types.ToolConfig(include_server_side_tool_invocations=True) if transformed_tools else None,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

        history = options.get("history") or []
        logger.info(
            f"generate_stream(): model={self.model_name}, "
            f"history_turns={len(history)}, "
            f"tools={len(transformed_tools) if transformed_tools else 0}"
        )
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=cast(Any, self._prepare_contents(prompt, options.get("history"))),
            config=generation_config,
        )

        async for chunk in stream:
            if not chunk.candidates or not chunk.candidates[0].content:
                continue

            parts = chunk.candidates[0].content.parts
            if not parts:
                continue

            for part in parts:
                is_thought = getattr(part, "thought", False)
                if is_thought:
                    yield LLMChunk(
                        text=part.text or "",
                        is_thought=True,
                        is_last=False,
                        model=self.model_name,
                        raw=chunk,
                    )
                elif part.function_call:
                    yield LLMChunk(
                        text="",
                        is_thought=False,
                        is_last=False,
                        is_tool_call=True,
                        tool_name=part.function_call.name,
                        tool_args=part.function_call.args or {},
                        model=self.model_name,
                        raw=chunk,
                    )
                elif part.text:
                    yield LLMChunk(
                        text=part.text,
                        is_thought=False,
                        is_last=False,
                        model=self.model_name,
                        raw=chunk,
                    )

    def get_name(self) -> str:
        return self.model_name
