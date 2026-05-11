import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import Content

from ..config import config
from ..logger import logger
from ..tools.resolver import (
    ensure_function_search_tool,
    needs_function_search,
    resolve_live_provider_tools,
)
from ..types import (
    LLMChunk,
    LLMOptions,
    LLMResult,
    MockCandidate,
    MockRaw,
)


class GeminiLiveProvider:
    def __init__(
        self,
        api_keys: list[str],
        model_name: str | None = None,
        tools: list[Any] | None = None,
        registry: dict[str, Callable[..., Any]] | None = None,
        system_instruction: str | None = None,
        include_thoughts: bool | None = None,
    ):
        self.api_keys = api_keys
        self.model_name = model_name or config.gemini_live_model
        self._current_key_index = 0
        self.client = genai.Client(api_key=self.api_keys[0])
        self.tools = list(tools) if tools else []
        self.registry = dict(registry) if registry else {}
        self.system_instruction = (
            system_instruction or config.default_system_instruction
        )
        self.include_thoughts = (
            include_thoughts
            if include_thoughts is not None
            else config.live_thinking_level.lower() != "none"
        )

        self._session: Any = None
        self._session_manager: Any = None
        self._session_handle: str | None = None
        self._session_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._last_turn_complete = True

        ensure_function_search_tool(self.model_name, self.registry)

    def _rotate_key(self) -> None:
        self._current_key_index = (self._current_key_index + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self.api_keys[self._current_key_index])

    def _needs_function_search(self) -> bool:
        return needs_function_search(self.model_name)

    def _resolve_tools(self) -> list[Any]:
        return resolve_live_provider_tools(
            model_name=self.model_name,
            tools=self.tools,
            registry=self.registry,
            google_types=types,
        )

    def _get_session_config(self) -> types.LiveConnectConfig:
        config_kwargs: dict[str, Any] = {
            "response_modalities": [types.Modality.AUDIO],
            "system_instruction": types.Content(
                parts=[types.Part(text=self.system_instruction)]
            )
            if self.system_instruction
            else None,
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=config.voice_name,
                    )
                )
            ),
            "context_window_compression": types.ContextWindowCompressionConfig(
                trigger_tokens=config.context_trigger_tokens,
                sliding_window=types.SlidingWindow(
                    target_tokens=config.context_target_tokens
                ),
            ),
            "session_resumption": types.SessionResumptionConfig(
                handle=self._session_handle,
            ),
            "tools": self._resolve_tools(),
        }

        if self.include_thoughts:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=getattr(
                    types.ThinkingLevel,
                    config.live_thinking_level.upper(),
                    types.ThinkingLevel.MEDIUM,
                ),
            )

        return types.LiveConnectConfig(**config_kwargs)

    async def warmup(self) -> None:

        try:
            await self._ensure_session()
            logger.info(f"GeminiLiveProvider: Warmup successful for {self.model_name}")
        except Exception as e:
            logger.warning(
                f"GeminiLiveProvider: Warmup failed for {self.model_name}: {e}"
            )

    async def _ensure_session(self) -> Any:
        async with self._session_lock:
            if self._session is not None:
                return self._session

            session_config = self._get_session_config()

            for attempt in range(len(self.api_keys)):
                try:
                    logger.info(
                        f"Live session connect attempt "
                        f"{attempt + 1}/{len(self.api_keys)}: "
                        f"model={self.model_name}, "
                        f"key_index={self._current_key_index}"
                    )
                    self._session_manager = self.client.aio.live.connect(
                        model=self.model_name, config=session_config
                    )
                    self._session = await self._session_manager.__aenter__()
                    logger.info(
                        f"Live session established: model={self.model_name}, "
                        f"key_index={self._current_key_index}"
                    )
                    return self._session
                except Exception as error:
                    error_detail = str(error).lower()
                    if (
                        any(
                            code in error_detail
                            for code in config.retriable_error_strings
                        )
                        and attempt < len(self.api_keys) - 1
                    ):
                        logger.info(
                            f"Live connect retriable error (attempt {attempt + 1}): "
                            f"{type(error).__name__}: {error} — rotating key"
                        )
                        self._rotate_key()
                        continue
                    raise error

            raise RuntimeError("Failed to establish Gemini Live session")

    async def _handle_tool_call(self, session: Any, tool_call: Any) -> None:
        if not tool_call.function_calls:
            return

        function_responses = []
        for call in tool_call.function_calls:
            name, args, call_id = call.name, call.args or {}, call.id
            logger.info(f"Live tool call: '{name}' id={call_id} args={args!r}")
            if name not in self.registry:
                function_responses.append(
                    types.FunctionResponse(
                        name=name,
                        id=call_id,
                        response={"error": f"Tool {name} not found"},
                    )
                )
                continue

            try:
                func = self.registry[name]
                result = (
                    await func(**args)
                    if asyncio.iscoroutinefunction(func)
                    else func(**args)
                )
                if not isinstance(result, dict):
                    result = {"result": result}
                logger.info(f"Live tool '{name}' succeeded")
                logger.debug(f"Live tool '{name}' result payload: {result!r}")
                function_responses.append(
                    types.FunctionResponse(name=name, id=call_id, response=result)
                )
            except Exception as e:
                logger.error(f"Live tool '{name}' failed [{type(e).__name__}]: {e}")
                function_responses.append(
                    types.FunctionResponse(
                        name=name, id=call_id, response={"error": str(e)}
                    )
                )

        if function_responses:
            await session.send_tool_response(function_responses=function_responses)

    async def _drain_session(self, session: Any) -> None:
        try:
            async with asyncio.timeout(config.drain_timeout):
                async for response in session.receive():
                    if (
                        response.server_content
                        and response.server_content.turn_complete
                    ):
                        break
        except (TimeoutError, Exception):
            await self.close()

    async def _prepare_turn(self, prompt: str | list[Content]) -> Any:
        session = await self._ensure_session()
        if not self._last_turn_complete:
            await self._drain_session(session)
            session = await self._ensure_session()

        self._last_turn_complete = False
        await session.send_realtime_input(
            text=prompt if isinstance(prompt, str) else str(prompt)
        )
        return session

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        async with self._turn_lock:
            session = await self._prepare_turn(prompt)
            start_time = time.perf_counter()
            full_response, thoughts, grounding = [], [], None

            try:
                async for response in session.receive():
                    logger.debug(f"Live stream receive: {response!r}")
                    if response.tool_call:
                        await self._handle_tool_call(session, response.tool_call)
                        continue

                    if (
                        response.session_resumption_update
                        and response.session_resumption_update.new_handle
                    ):
                        self._session_handle = (
                            response.session_resumption_update.new_handle
                        )
                        continue

                    if response.go_away:
                        await self.close()
                        task = asyncio.create_task(self._ensure_session())
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
                        continue

                    content = response.server_content
                    if not content:
                        continue

                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.thought and part.text:
                                thoughts.append(part.text)
                            elif part.text:
                                full_response.append(part.text)

                    if content.grounding_metadata and not grounding:
                        grounding = content.grounding_metadata

                    if (
                        content.output_transcription
                        and content.output_transcription.text
                    ):
                        full_response.append(content.output_transcription.text)

                    if content.turn_complete or content.generation_complete:
                        self._last_turn_complete = bool(content.turn_complete)
                        break
            except Exception as e:
                logger.error(f"generate() error [{type(e).__name__}]: {e}")

            return LLMResult(
                text="".join(full_response),
                model=self.model_name,
                duration=time.perf_counter() - start_time,
                thoughts="".join(thoughts),
                usage={},
                raw=MockRaw(candidates=[MockCandidate(grounding_metadata=grounding)]),
            )

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        async with self._turn_lock:
            session = await self._prepare_turn(prompt)

            try:
                async for response in session.receive():
                    logger.debug(f"Live stream receive: {response!r}")
                    if response.tool_call:
                        for fc in response.tool_call.function_calls:
                            yield LLMChunk(
                                text="",
                                is_last=False,
                                is_tool_call=True,
                                tool_name=fc.name,
                                tool_args=fc.args or {},
                                model=self.model_name,
                            )
                        await self._handle_tool_call(session, response.tool_call)
                        continue

                    if (
                        response.session_resumption_update
                        and response.session_resumption_update.new_handle
                    ):
                        self._session_handle = (
                            response.session_resumption_update.new_handle
                        )
                        continue

                    if response.go_away:
                        await self.close()
                        task = asyncio.create_task(self._ensure_session())
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
                        continue

                    content = response.server_content
                    if not content:
                        continue

                    if content.model_turn:
                        for part in content.model_turn.parts:
                            if part.thought and part.text:
                                yield LLMChunk(
                                    text=part.text,
                                    is_last=False,
                                    is_thought=True,
                                    model=self.model_name,
                                )
                            elif part.text:
                                yield LLMChunk(
                                    text=part.text, is_last=False, model=self.model_name
                                )

                    if (
                        content.output_transcription
                        and content.output_transcription.text
                    ):
                        yield LLMChunk(
                            text=content.output_transcription.text,
                            is_last=False,
                            model=self.model_name,
                        )

                    is_finished = bool(
                        content.turn_complete
                        or content.generation_complete
                        or (
                            content.output_transcription
                            and content.output_transcription.finished
                        )
                    )
                    if is_finished:
                        self._last_turn_complete = bool(content.turn_complete)
                        yield LLMChunk(text="", is_last=True, model=self.model_name)
                        break
            except Exception as e:
                logger.error(f"generate_stream() error [{type(e).__name__}]: {e}")
                yield LLMChunk(text="", is_last=True, model=self.model_name)

    async def close(self) -> None:
        async with self._session_lock:
            if self._session_manager:
                import contextlib

                with contextlib.suppress(Exception):
                    await self._session_manager.__aexit__(None, None, None)
                self._session = None
                self._session_manager = None

    def get_name(self) -> str:
        return f"{self.model_name} (Live)"
