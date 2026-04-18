import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import Content

from ..config import config
from ..llm_types import DEFAULT_SYSTEM_INSTRUCTION, LLMChunk, LLMOptions, LLMResult
from ..logger import logger
from ..tools import google_search


class GeminiLiveProvider:
    def __init__(
        self,
        api_keys: list[str],
        model_name: str | None = None,
        system_instruction: str | None = None,
        tools: list[Any] | None = None,
        registry: dict[str, Callable[..., Any]] | None = None,
    ):
        self.api_keys = api_keys
        self.model_name = model_name or config.gemini_live_model
        self._current_key_index = 0
        self.client = genai.Client(api_key=self.api_keys[0])

        self._session: Any = None
        self._session_manager: Any = None
        self._session_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()

        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        self.tools = list(tools) if tools else []
        self.registry = dict(registry) if registry else {}

        if "3.1" in self.model_name and "google_search" not in self.registry:
            logger.info(
                "Registering enhanced 2.5-powered google_search tool for "
                f"{self.model_name}"
            )
            self.registry["google_search"] = google_search

        logger.debug(
            f"GeminiLive initialized (model: {self.model_name}, keys: "
            f"{len(self.api_keys)})"
        )

    def _rotate_key(self) -> None:
        self._current_key_index = (self._current_key_index + 1) % len(self.api_keys)
        new_key = self.api_keys[self._current_key_index]
        logger.warning(
            f"Quota issue or error detected. Rotating to key index "
            f"{self._current_key_index}"
        )
        self.client = genai.Client(api_key=new_key)

    async def _ensure_session(self) -> Any:
        async with self._session_lock:
            if self._session is not None:
                return self._session

            final_tools = list(self.tools)
            for _name, func in self.registry.items():
                final_tools.append(func)

            if (
                "2.5" in self.model_name or "fallback" in self.model_name.lower()
            ) and not any(hasattr(t, "google_search") for t in final_tools):
                final_tools.append(types.Tool(google_search=types.GoogleSearch()))

            session_config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                system_instruction=types.Content(
                    parts=[types.Part(text=self.system_instruction)]
                ),
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=config.voice_name,
                        )
                    )
                ),
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=types.ThinkingLevel.HIGH,
                ),
                tools=final_tools,
            )

            max_attempts = len(self.api_keys)
            for attempt in range(max_attempts):
                try:
                    logger.info(
                        f"Establishing Gemini Live connection (Attempt "
                        f"{attempt + 1}/{max_attempts})"
                    )
                    self._session_manager = self.client.aio.live.connect(
                        model=self.model_name, config=session_config
                    )
                    assert self._session_manager is not None
                    self._session = await self._session_manager.__aenter__()
                    return self._session

                except Exception as error:
                    error_detail = str(error).lower()
                    retriable_codes = {
                        "1011",
                        "1007",
                        "1008",
                        "429",
                        "quota",
                        "exhausted",
                    }

                    is_retriable = any(code in error_detail for code in retriable_codes)
                    if is_retriable and attempt < max_attempts - 1:
                        logger.warning(
                            f"Retriable connection failure: {error}. "
                            "Rotating credentials..."
                        )
                        self._rotate_key()
                        continue

                    logger.error(f"Fatal connection error encountered: {error}")
                    raise error

            raise RuntimeError(
                "Failed to establish Gemini Live session after multiple attempts"
            )

    async def _handle_tool_call(self, session: Any, tool_call: Any) -> None:
        function_responses = []
        for call in tool_call.function_calls:
            name = call.name
            args = call.args or {}
            call_id = call.id

            logger.info(f"Model requested tool: {name}({args})")

            if name in self.registry:
                func = self.registry[name]
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**args)
                    else:
                        result = func(**args)

                    if not isinstance(result, dict):
                        result = {"result": result}

                    function_responses.append(
                        types.FunctionResponse(name=name, id=call_id, response=result)
                    )
                except Exception as e:
                    logger.error(f"Error executing tool {name}: {e}")
                    function_responses.append(
                        types.FunctionResponse(
                            name=name, id=call_id, response={"error": str(e)}
                        )
                    )
            else:
                logger.warning(f"Tool {name} not found in registry")
                function_responses.append(
                    types.FunctionResponse(
                        name=name,
                        id=call_id,
                        response={"error": f"Tool {name} not found"},
                    )
                )

        if function_responses:
            logger.debug(
                f"Sending {len(function_responses)} tool responses back to session"
            )
            await session.send_tool_response(function_responses=function_responses)

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        async with self._turn_lock:
            session = await self._ensure_session()
            text_input = prompt if isinstance(prompt, str) else str(prompt)
            start_time = time.perf_counter()

            await session.send_realtime_input(text=text_input)

            full_response_parts = []
            thought_parts = []
            try:
                async for response in session.receive():
                    if response.tool_call:
                        await self._handle_tool_call(session, response.tool_call)
                        continue

                    server_content = response.server_content
                    if not server_content:
                        continue

                    if server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if part.thought and part.text:
                                thought_parts.append(part.text)

                    if (
                        server_content.output_transcription
                        and server_content.output_transcription.text
                    ):
                        full_response_parts.append(
                            server_content.output_transcription.text
                        )

                    if server_content.turn_complete:
                        break
            except Exception as e:
                logger.error(f"Error during realtime response retrieval: {e}")

            response_text = "".join(full_response_parts)
            return LLMResult(
                text=response_text,
                model=self.model_name,
                duration=time.perf_counter() - start_time,
                thoughts="".join(thought_parts),
                usage={},
                raw=None,
            )

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        async with self._turn_lock:
            session = await self._ensure_session()
            text_input = prompt if isinstance(prompt, str) else str(prompt)
            await session.send_realtime_input(text=text_input)

            try:
                async for response in session.receive():
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

                    server_content = response.server_content
                    if not server_content:
                        continue

                    if server_content.model_turn:
                        for part in server_content.model_turn.parts:
                            if part.thought and part.text:
                                yield LLMChunk(
                                    text=part.text,
                                    is_last=False,
                                    is_thought=True,
                                    model=self.model_name,
                                )

                    text_chunk = ""
                    if (
                        server_content.output_transcription
                        and server_content.output_transcription.text
                    ):
                        text_chunk = server_content.output_transcription.text

                    is_finished = bool(server_content.turn_complete)
                    if text_chunk or is_finished:
                        yield LLMChunk(
                            text=text_chunk,
                            is_last=is_finished,
                            model=self.model_name,
                            is_thought=False,
                        )

                    if is_finished:
                        break
            except Exception as e:
                logger.error(f"Stream interrupted: {e}")
                yield LLMChunk(text="", is_last=True, model=self.model_name)

    async def close(self) -> None:
        async with self._session_lock:
            if self._session_manager:
                logger.info("Terminating Gemini Live session")
                try:
                    await self._session_manager.__aexit__(None, None, None)
                except Exception as e:
                    logger.error(f"Error during session shutdown: {e}")

                self._session = None
                self._session_manager = None

    def get_name(self) -> str:
        return f"{self.model_name} (Live)"
