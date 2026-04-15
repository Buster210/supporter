import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types
from google.genai.types import Content

from .config import config
from .llm_types import LLMChunk, LLMOptions, LLMResult
from .logger import logger

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a high-level technical strategist and expert software architect. "
    "Provide rigorous, thorough, and architecturally sound advice. "
    "Before answering, consider all edge cases and performance implications."
)


class GeminiLiveProvider:
    def __init__(
        self,
        api_keys: list[str],
        model_name: str | None = None,
        system_instruction: str | None = None,
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

            live_config = types.LiveConnectConfig(
                response_modalities=[types.Modality.AUDIO],
                system_instruction=types.Content(
                    parts=[types.Part(text=self.system_instruction)]
                ),
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True,
                    thinking_level=types.ThinkingLevel.HIGH,
                ),
            )

            max_retries = len(self.api_keys)
            for attempt in range(max_retries):
                try:
                    logger.info(
                        f"Connecting to Gemini Live (Attempt "
                        f"{attempt + 1}/{max_retries})"
                    )
                    self._session_manager = self.client.aio.live.connect(
                        model=self.model_name, config=live_config
                    )
                    assert self._session_manager is not None
                    self._session = await self._session_manager.__aenter__()
                    return self._session
                except Exception as e:
                    err_msg = str(e).lower()
                    retriable_codes = ("1011", "1007", "1008", "429", "quota")
                    if any(code in err_msg for code in retriable_codes):
                        logger.warning(
                            f"Connection failed: {e}. Retrying with key rotation..."
                        )
                        self._rotate_key()
                        continue

                    logger.error(f"Critical connection error: {e}")
                    raise e

            raise RuntimeError(
                "Failed to establish Gemini Live session after multiple attempts"
            )

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
