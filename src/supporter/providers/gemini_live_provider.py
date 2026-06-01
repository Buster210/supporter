from __future__ import annotations

import asyncio
import collections
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
    from google.genai.types import Content

from ..config import config
from ..logger import logger
from ..tools.resolver import (
    ensure_function_search_tool,
    resolve_live_provider_tools,
)
from ..types import (
    LLMChunk,
    LLMOptions,
    LLMResult,
)

_STALE_HANDLE_ERRORS = ("session not found", "invalid session handle")


def _to_seconds(time_left: Any) -> float | None:
    if time_left is None:
        return None
    if isinstance(time_left, (int, float)):
        return float(time_left)
    try:
        return float(time_left.total_seconds())
    except AttributeError:
        return None


def _is_native_audio(model_name: str) -> bool:
    return "native-audio" in model_name.lower()


def _format_grounding_sources(grounding: Any) -> str:
    chunks = getattr(grounding, "grounding_chunks", None) or []
    lines = [
        f"- {getattr(c.web, 'title', None) or 'Search Result'}: {c.web.uri}"
        for c in chunks
        if getattr(c, "web", None) and getattr(c.web, "uri", None)
    ]
    return "\n\nSOURCES FOUND:\n" + "\n".join(lines) if lines else ""


def _summarize_live_response(response: Any) -> str | None:
    parts: list[str] = []

    sc = response.server_content
    if sc is not None:
        if sc.turn_complete:
            parts.append("turn_complete")
        if sc.generation_complete:
            parts.append("gen_complete")
        if sc.interrupted:
            parts.append("interrupted")

    if response.tool_call is not None:
        fcs = response.tool_call.function_calls or ()
        names = ",".join(fc.name for fc in fcs if fc.name) or "<no_fc>"
        parts.append(f"tool_call[{names}]")

    if response.tool_call_cancellation is not None:
        ids = response.tool_call_cancellation.ids or ()
        parts.append(f"tool_call_cancellation[n={len(ids)}]")

    sru = response.session_resumption_update
    if sru is not None:
        handle = sru.new_handle or ""
        parts.append(
            f"session_resumption[handle={handle[:8]},resumable={sru.resumable}]"
        )

    if response.go_away is not None:
        parts.append(f"go_away[time_left={response.go_away.time_left}]")

    if response.setup_complete is not None:
        parts.append("setup_complete")

    um = response.usage_metadata
    if um is not None:
        parts.append(f"usage[total={um.total_token_count}]")

    return ",".join(parts) if parts else None


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
        self._client: genai.Client | None = None
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
        self._native_audio = _is_native_audio(self.model_name)
        self._ends_turn_early = "gemini-3" in self.model_name.lower()

        self._session: Any = None
        self._session_manager: Any = None
        self._session_handle: str | None = None
        self._session_lock = asyncio.Lock()
        self._turn_lock = asyncio.Lock()
        self._last_turn_complete = True
        self._reconnect_pending = False
        self._needs_replay = False
        self._prewarm_task: asyncio.Task[Any] | None = None
        self._history: list[Content] = []
        self._recent_images: collections.deque[tuple[bytes, str]] = collections.deque(
            maxlen=config.replay_image_count
        )
        self.recovery_observer: Callable[[str, dict[str, Any]], None] | None = None
        self._reconnect_attempts = 0
        self._reconnect_attempts_max = config.reconnect_attempts_max
        self._go_away_deadline: float | None = None
        self._keepalive_task: asyncio.Task[Any] | None = None
        self._handle_resumed_pending = False

        ensure_function_search_tool(self.model_name, self.registry)

        from ..tools.browser.guardrails import register_browse_callback

        register_browse_callback(image_sink=self._inject_image)

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_keys[self._current_key_index])
        return self._client

    def _rotate_key(self) -> None:
        self._current_key_index = (self._current_key_index + 1) % len(self.api_keys)
        self._client = None

    def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        if self.recovery_observer is None:
            return
        try:
            self.recovery_observer(event, data or {})
        except Exception:
            logger.debug(f"recovery_observer raised on event={event}")

    def _resolve_tools(self) -> list[Any]:
        from google.genai import types

        return resolve_live_provider_tools(
            model_name=self.model_name,
            tools=self.tools,
            registry=self.registry,
            google_types=types,
        )

    def _get_session_config(self) -> types.LiveConnectConfig:
        from google.genai import types

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

        if self._native_audio:
            config_kwargs["output_audio_transcription"] = (
                types.AudioTranscriptionConfig()
            )

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

            key_attempt = 0
            stale_handle_dropped = False

            while key_attempt < len(self.api_keys):
                session_config = self._get_session_config()
                try:
                    logger.info(
                        f"Live session connect attempt "
                        f"{key_attempt + 1}/{len(self.api_keys)}: "
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
                    self._go_away_deadline = None
                    if self._session_handle is None:
                        self._needs_replay = True
                    elif config.empty_resume_policy == "replay":
                        self._needs_replay = True
                        self._handle_resumed_pending = False
                    else:
                        self._needs_replay = False
                        self._handle_resumed_pending = True
                    await self._start_keepalive()
                    return self._session
                except Exception as error:
                    error_detail = str(error).lower()
                    if (
                        self._session_handle is not None
                        and any(s in error_detail for s in _STALE_HANDLE_ERRORS)
                        and not stale_handle_dropped
                    ):
                        if self._history:
                            logger.warning(
                                "Live resumption handle rejected (stale or "
                                "invalid); dropping handle, reconnecting fresh "
                                "with history replay"
                            )
                        else:
                            logger.warning(
                                "Live resumption handle rejected (stale or "
                                "invalid) and no local history to replay — prior "
                                "conversation context will be lost"
                            )
                        self._session_handle = None
                        stale_handle_dropped = True
                        self._emit(
                            "handle_dropped",
                            {"had_history": bool(self._history)},
                        )
                        continue
                    if (
                        any(
                            code in error_detail
                            for code in config.retriable_error_strings
                        )
                        and key_attempt < len(self.api_keys) - 1
                    ):
                        logger.info(
                            f"Live connect retriable error "
                            f"(attempt {key_attempt + 1}): "
                            f"{type(error).__name__}: {error} — rotating key"
                        )
                        self._rotate_key()
                        key_attempt += 1
                        continue
                    raise error

            raise RuntimeError("Failed to establish Gemini Live session")

    def _replay_turns(self) -> list[Content]:
        from google.genai.types import Content, Part

        max_chars = config.replay_tool_summary_max_chars
        turns: list[Content] = []
        for content in self._history:
            role = getattr(content, "role", "user")
            parts = getattr(content, "parts", None) or []
            fragments: list[str] = []
            for p in parts:
                if getattr(p, "text", None):
                    fragments.append(p.text)
                elif getattr(p, "function_call", None):
                    fc = p.function_call
                    fc_name = getattr(fc, "name", None) or "unknown"
                    try:
                        fc_args = repr(getattr(fc, "args", None) or {})
                    except Exception:
                        fc_args = "..."
                    if len(fc_args) > max_chars:
                        fc_args = fc_args[:max_chars] + "..."
                    fragments.append(f"[called {fc_name}({fc_args})]")
                elif getattr(p, "function_response", None):
                    fr = p.function_response
                    fr_name = getattr(fr, "name", None) or "unknown"
                    try:
                        fr_resp = repr(getattr(fr, "response", None) or {})
                    except Exception:
                        fr_resp = "..."
                    if len(fr_resp) > max_chars:
                        fr_resp = fr_resp[:max_chars] + "..."
                    fragments.append(f"[tool {fr_name} -> {fr_resp}]")
                elif getattr(p, "inline_data", None):
                    pass
                else:
                    pass
            text = " ".join(fragments)
            if not text:
                continue
            if role == "model":
                text = f"(Assistant earlier said: {text})"
            turns.append(Content(role="user", parts=[Part(text=text)]))
        return turns

    async def _send_user_turn(self, session: Any, prompt: str | list[Content]) -> None:
        text = prompt if isinstance(prompt, str) else str(prompt)
        if self._needs_replay:
            self._needs_replay = False
            if self._history:
                from google.genai.types import Content, Part

                turns = self._replay_turns()
                turns.append(Content(role="user", parts=[Part(text=text)]))
                try:
                    logger.info(
                        f"Live: replaying {len(self._history)} history turns with "
                        "the prompt into fresh session (no resumption handle)"
                    )
                    self._emit("replaying", {"turns": len(self._history)})
                    await session.send_client_content(turns=turns, turn_complete=True)
                    if config.replay_image_count > 0:
                        await self._reinject_recent_images(session)
                    return
                except Exception as exc:
                    logger.warning(
                        f"Live history replay failed [{type(exc).__name__}: "
                        f"{exc}]; sending prompt without restored context"
                    )
                    self._emit("context_partial", {})
        await session.send_realtime_input(text=text)

    async def _handle_tool_call(self, session: Any, tool_call: Any) -> None:
        from google.genai import types

        if not tool_call.function_calls:
            return

        async def _invoke(call: Any) -> Any:
            name, args, call_id = call.name, call.args or {}, call.id
            logger.info(f"Live tool call: '{name}' id={call_id} args={args!r}")
            if name not in self.registry:
                return types.FunctionResponse(
                    name=name,
                    id=call_id,
                    response={"error": f"Tool {name} not found"},
                )
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
                return types.FunctionResponse(name=name, id=call_id, response=result)
            except Exception as e:
                logger.error(f"Live tool '{name}' failed [{type(e).__name__}]: {e}")
                return types.FunctionResponse(
                    name=name, id=call_id, response={"error": str(e)}
                )

        function_responses = await asyncio.gather(
            *(_invoke(call) for call in tool_call.function_calls)
        )
        if function_responses:
            await session.send_tool_response(function_responses=function_responses)

    async def _inject_image(self, data: bytes, mime_type: str) -> None:
        self._recent_images.append((data, mime_type))
        if self._session is None:
            logger.debug("Image inject skipped: no live session")
            return
        from google.genai import types

        await self._session.send_realtime_input(
            media=types.Blob(data=data, mime_type=mime_type)
        )

    async def _reinject_recent_images(self, session: Any) -> None:
        for data, mime_type in list(self._recent_images):
            try:
                from google.genai import types

                await session.send_realtime_input(
                    media=types.Blob(data=data, mime_type=mime_type)
                )
            except Exception as exc:
                logger.warning(
                    f"Image re-injection failed [{type(exc).__name__}: {exc}]; "
                    "continuing with text-only replay"
                )

    async def _drain_session(self, session: Any) -> None:
        try:
            async with asyncio.timeout(config.drain_timeout):
                async for response in session.receive():
                    if (
                        response.server_content
                        and response.server_content.turn_complete
                    ):
                        self._last_turn_complete = True
                        return
        except TimeoutError:
            logger.warning(
                "Live session drain timed out; reconnecting to recover state"
            )
            await self.close()
        except Exception as exc:
            logger.warning(
                f"Live session drain failed [{type(exc).__name__}: {exc}]; "
                "closing session"
            )
            await self.close()

    def _schedule_prewarm(self) -> None:
        if not self._reconnect_pending or self._prewarm_task is not None:
            return
        self._reconnect_pending = False
        self._emit("reconnecting", {"source": "prewarm"})
        self._prewarm_task = asyncio.create_task(self._reconnect())

    async def _reconnect(self) -> None:
        if self._reconnect_attempts >= self._reconnect_attempts_max:
            self._emit("reconnect_giving_up", {"attempts": self._reconnect_attempts})
            logger.warning(
                f"Live reconnect giving up after {self._reconnect_attempts} attempts"
            )
            return
        delay = min(
            config.reconnect_backoff_base * (2**self._reconnect_attempts),
            config.reconnect_backoff_cap,
        )
        await asyncio.sleep(delay)
        self._reconnect_attempts += 1
        await self._teardown_session()
        try:
            await self._ensure_session()
            self._reconnect_attempts = 0
        except Exception:
            self._reconnect_pending = True
            raise

    async def _consume_prewarm(self) -> None:
        task = self._prewarm_task
        if task is None:
            return
        self._prewarm_task = None
        try:
            await task
        except Exception as exc:
            logger.warning(
                f"Live prewarm reconnect failed [{type(exc).__name__}: {exc}]; "
                "reconnecting inline"
            )

    async def _cancel_prewarm(self) -> None:
        task = self._prewarm_task
        self._prewarm_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        import contextlib

        with contextlib.suppress(BaseException):
            await task

    async def _keepalive_loop(self) -> None:
        while True:
            await asyncio.sleep(config.keepalive_interval)
            if self._turn_lock.locked():
                continue
            if self._session is None:
                continue
            if (
                self._go_away_deadline is not None
                and time.monotonic() >= self._go_away_deadline
            ):
                logger.info("Keepalive: go_away deadline passed, flagging reconnect")
                self._reconnect_pending = True
                self._schedule_prewarm()

    async def _start_keepalive(self) -> None:
        if not config.keepalive_enabled or self._keepalive_task is not None:
            return
        self._keepalive_task = asyncio.create_task(self._keepalive_loop())

    async def _cancel_keepalive(self) -> None:
        task = self._keepalive_task
        self._keepalive_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        import contextlib

        with contextlib.suppress(BaseException):
            await task

    async def _prepare_turn(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> Any:
        if options and options.get("history"):
            self._history = list(options["history"])
        await self._consume_prewarm()
        if self._reconnect_pending:
            self._reconnect_pending = False
            self._go_away_deadline = None
            await self._teardown_session()
        session = await self._ensure_session()
        if not self._last_turn_complete:
            await self._drain_session(session)
            if self._session is None:
                session = await self._ensure_session()

        self._last_turn_complete = False
        await self._send_user_turn(session, prompt)
        return session

    async def generate(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> LLMResult:
        async with self._turn_lock:
            session = await self._prepare_turn(prompt, options)
            start_time = time.perf_counter()
            full_response, thoughts, grounding = [], [], None

            try:
                async for response in session.receive():
                    summary = _summarize_live_response(response)
                    if summary is not None:
                        logger.debug(f"Live receive: {summary}")
                    if response.tool_call:
                        await self._handle_tool_call(session, response.tool_call)
                        continue

                    if (
                        response.session_resumption_update
                        and response.session_resumption_update.resumable
                        and response.session_resumption_update.new_handle
                    ):
                        self._session_handle = (
                            response.session_resumption_update.new_handle
                        )
                        self._handle_resumed_pending = False
                        continue

                    if response.go_away:
                        logger.info(
                            "Live go_away received "
                            f"(time_left={response.go_away.time_left}); "
                            "reconnecting before next turn"
                        )
                        self._reconnect_pending = True
                        tl = _to_seconds(response.go_away.time_left)
                        if tl is not None:
                            self._go_away_deadline = (
                                time.monotonic() + tl - config.prewarm_safety_margin
                            )
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

                    if content.turn_complete or (
                        self._ends_turn_early and content.generation_complete
                    ):
                        self._last_turn_complete = bool(content.turn_complete)
                        break
            except Exception as e:
                logger.error(f"generate() error [{type(e).__name__}]: {e}")
                self._last_turn_complete = True
                self._reconnect_pending = True

            if self._handle_resumed_pending:
                self._emit("empty_resume_suspected", {})
                self._handle_resumed_pending = False

            self._schedule_prewarm()
            text = "".join(full_response)
            if grounding:
                text += _format_grounding_sources(grounding)

            return LLMResult(
                text=text,
                model=self.model_name,
                duration=time.perf_counter() - start_time,
                thoughts="".join(thoughts),
                usage={},
                raw=grounding,
            )

    async def generate_stream(
        self, prompt: str | list[Content], options: LLMOptions | None = None
    ) -> AsyncIterator[LLMChunk]:
        async with self._turn_lock:
            session = await self._prepare_turn(prompt, options)
            grounding: Any = None

            try:
                async for response in session.receive():
                    summary = _summarize_live_response(response)
                    if summary is not None:
                        logger.debug(f"Live receive: {summary}")
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
                        and response.session_resumption_update.resumable
                        and response.session_resumption_update.new_handle
                    ):
                        self._session_handle = (
                            response.session_resumption_update.new_handle
                        )
                        self._handle_resumed_pending = False
                        continue

                    if response.go_away:
                        logger.info(
                            "Live go_away received "
                            f"(time_left={response.go_away.time_left}); "
                            "reconnecting before next turn"
                        )
                        self._reconnect_pending = True
                        tl = _to_seconds(response.go_away.time_left)
                        if tl is not None:
                            self._go_away_deadline = (
                                time.monotonic() + tl - config.prewarm_safety_margin
                            )
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

                    if content.grounding_metadata and not grounding:
                        grounding = content.grounding_metadata

                    if (
                        content.output_transcription
                        and content.output_transcription.text
                    ):
                        yield LLMChunk(
                            text=content.output_transcription.text,
                            is_last=False,
                            model=self.model_name,
                        )

                    if content.turn_complete or (
                        self._ends_turn_early and content.generation_complete
                    ):
                        self._last_turn_complete = bool(content.turn_complete)
                        sources = (
                            _format_grounding_sources(grounding) if grounding else ""
                        )
                        if sources:
                            yield LLMChunk(
                                text=sources,
                                is_last=False,
                                model=self.model_name,
                                raw=grounding,
                            )
                        yield LLMChunk(text="", is_last=True, model=self.model_name)
                        break
            except Exception as e:
                logger.error(f"generate_stream() error [{type(e).__name__}]: {e}")
                self._last_turn_complete = True
                self._reconnect_pending = True
                yield LLMChunk(text="", is_last=True, model=self.model_name)

            if self._handle_resumed_pending:
                self._emit("empty_resume_suspected", {})
                self._handle_resumed_pending = False

            self._schedule_prewarm()

    async def _teardown_session(self) -> None:
        async with self._session_lock:
            if self._session_manager:
                import contextlib

                with contextlib.suppress(Exception):
                    await self._session_manager.__aexit__(None, None, None)
                self._session = None
                self._session_manager = None
            self._last_turn_complete = True

    async def close(self) -> None:
        await self._cancel_keepalive()
        await self._cancel_prewarm()
        await self._teardown_session()

    def get_name(self) -> str:
        return f"{self.model_name} (Live)"
