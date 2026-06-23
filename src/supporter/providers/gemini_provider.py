from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from google import genai
    from google.genai.types import Content


from ..config import config
from ..llm.types import GenOptions
from ..llm.types import Message as NeutralMessage
from ..logger import logger
from ..tools.resolver import resolve_provider_tools
from ..types import LLMChunk, LLMResult
from .gemini_codec import (
    afc_history_to_messages,
    gen_options_to_config,
    message_to_content,
)


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model_name: str | None = None,
    ):
        self.api_key = api_key
        self.model_name = model_name or config.gemini_model
        self._client: genai.Client | None = None
        self._tool_cache: list[Any] | None = None
        self._last_tool_key: Any = None
        logger.info(
            f"GeminiProvider initialized: model={self.model_name}, "
            f"http_retry_attempts={config.http_retry_attempts}"
        )

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            from google import genai
            from google.genai import types

            retry_options = types.HttpRetryOptions(attempts=config.http_retry_attempts)
            self._client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(retry_options=retry_options),
            )
        return self._client

    def _is_gemma(self) -> bool:
        return self.model_name.lower().startswith("gemma")

    def _prepare_contents(
        self,
        prompt: str | list[NeutralMessage],
    ) -> list[Content]:
        from google.genai.types import Content, Part

        if isinstance(prompt, str):
            return [Content(role="user", parts=[Part(text=prompt)])]
        return [message_to_content(m) for m in prompt]

    def _strip_gemini_only_tools(self, tools: list[Any]) -> list[Any]:
        return [t for t in tools if getattr(t, "code_execution", None) is None]

    def _prepare_request(
        self,
        prompt: str | list[NeutralMessage],
        options: GenOptions,
    ) -> tuple[list[Content], Any, list[Any] | None]:
        transformed_tools = self._transform_tools(options)
        if self._is_gemma() and transformed_tools:
            transformed_tools = self._strip_gemini_only_tools(transformed_tools) or None
        contents = self._prepare_contents(prompt)
        generation_config = gen_options_to_config(
            options,
            transformed_tools,
            is_gemma=self._is_gemma(),
            default_system_instruction=config.default_system_instruction,
        )
        return contents, generation_config, transformed_tools

    def _transform_tools(self, options: GenOptions | None = None) -> list[Any] | None:
        if not options:
            return None

        tools = options.extras.get("tools", [])
        registry = options.extras.get("registry") or {}
        use_search = options.use_search
        use_code_execution = options.extras.get("use_code_execution", False)

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

        from google.genai import types

        final_tools = resolve_provider_tools(
            model_name=self.model_name,
            tools=tools,
            registry=registry,
            use_search=use_search,
            use_code_execution=use_code_execution,
            google_types=types,
        )

        self._tool_cache = final_tools or None
        self._last_tool_key = current_identity_key
        return self._tool_cache

    async def generate(
        self,
        prompt: str | list[NeutralMessage],
        options: GenOptions | None = None,
    ) -> LLMResult:
        options = options or GenOptions()
        interaction_id = options.extras.get("interaction_id")
        contents, generation_config, transformed_tools = self._prepare_request(
            prompt, options
        )

        start_time = time.perf_counter()
        prompt_turns = len(prompt) if isinstance(prompt, list) else 1
        logger.info(
            f"generate(): model={self.model_name}, prompt_turns={prompt_turns}, "
            f"tools={len(transformed_tools) if transformed_tools else 0}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            sys_inst = options.system_instruction or config.default_system_instruction
            logger.debug(f"generate() system_instruction: {sys_inst}")
            logger.debug(f"generate() context options: {options!r}")
        result: Any = None

        if interaction_id and not self._is_gemma():
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
            from ..recover import AutoRecover, rotate_api_key

            recover = AutoRecover(
                name=f"gemini_rest.{self.model_name}",
                actions=[rotate_api_key],
                max_attempts=3,
                metrics_tag="gemini_rest",
            )
            result = await recover.call(
                self.client.aio.models.generate_content,
                model=self.model_name,
                contents=cast(Any, contents),
                config=generation_config,
            )

        end_time = time.perf_counter()

        raw_afc_history = getattr(result, "automatic_function_calling_history", None)
        if not raw_afc_history and interaction_id:
            response = getattr(result, "response", None)
            raw_afc_history = getattr(
                response, "automatic_function_calling_history", None
            ) or getattr(result, "history", None)

        if raw_afc_history:
            neutral_history = afc_history_to_messages(raw_afc_history)
        else:
            neutral_history = []

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
                    p.text
                    for p in text_parts
                    if p.text and getattr(p, "thought", False)
                )

        duration = end_time - start_time
        logger.info(
            f"generate() done: model={self.model_name}, duration={duration:.3f}s, "
            f"prompt_tokens={usage.get('prompt_tokens', '?')}, "
            f"completion_tokens={usage.get('completion_tokens', '?')}"
        )
        if result.candidates and logger.isEnabledFor(logging.DEBUG):
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
            history=neutral_history,
            usage=usage,
            raw=result,
            candidates=getattr(result, "candidates", []),
            automatic_function_calling_history=raw_afc_history,
        )

    async def generate_stream(
        self,
        prompt: str | list[NeutralMessage],
        options: GenOptions | None = None,
    ) -> AsyncIterator[LLMChunk]:
        options = options or GenOptions()
        contents, generation_config, transformed_tools = self._prepare_request(
            prompt, options
        )

        prompt_turns = len(prompt) if isinstance(prompt, list) else 1
        logger.info(
            f"generate_stream(): model={self.model_name}, "
            f"prompt_turns={prompt_turns}, "
            f"tools={len(transformed_tools) if transformed_tools else 0}"
        )
        stream = await self.client.aio.models.generate_content_stream(
            model=self.model_name,
            contents=cast(Any, contents),
            config=generation_config,
        )

        try:
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
        except Exception as e:
            logger.error(f"generate_stream() error [{type(e).__name__}]: {e}")
            raise

        yield LLMChunk(text="", is_last=True, model=self.model_name)

    def get_name(self) -> str:
        return self.model_name
