from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google import genai
    from google.genai import types
    from google.genai.types import Content

from ..config import config
from ..llm.types import GenOptions
from ..llm.types import Message as NeutralMessage
from ..logger import logger
from ..recovery_metrics import record_key_rotation
from ..tools.resolver import (
    ensure_function_search_tool,
    resolve_live_provider_tools,
)
from ..types import (
    LLMChunk,
    LLMResult,
)
from .gemini_codec import message_to_content

_STALE_HANDLE_ERRORS = ("session not found", "invalid session handle")


def _is_key_error(error: BaseException) -> bool:
    """Return True if the error is likely key-attributable (quota/auth)."""
    message = str(error).lower()
    status = getattr(error, "status", None) or getattr(error, "code", None)
    if isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
        return True
    key_patterns = (
        "api key not valid",
        "api_key_invalid",
        "permission_denied",
        "quota exceeded",
        "resource_exhausted",
        "rate limit",
        "free tier",
    )
    return any(p in message for p in key_patterns)


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
        self._monitor_task: asyncio.Task[Any] | None = None
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
        record_key_rotation()

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
                    session_manager = self.client.aio.live.connect(
                        model=self.model_name, config=session_config
                    )
                    try:
                        self._session_manager = session_manager
                        self._session = await session_manager.__aenter__()
                    except Exception:
                        # Clean up the un-entered session manager to prevent leak.
                        self._session_manager = None
                        self._session = None
                        with contextlib.suppress(Exception):
                            await session_manager.__aexit__(None, None, None)
                        raise
                    logger.info(
                        f"Live session established: model={self.model_name}, "
                        f"key_index={self._current_key_index}"
                    )
                    self._go_away_deadline = None
                    if self._session_handle is None:
                        self._needs_replay = True
                        # No handle → no resumption-pending state to clear
                        # (it's only meaningful for resumed sessions). A stale
                        # True from a prior turn that was aborted before its
                        # epilogue could otherwise trigger a false
                        # empty_resume_suspected on the next turn.
                        self._handle_resumed_pending = False
                    elif config.empty_resume_policy == "replay":
                        self._needs_replay = True
                        self._handle_resumed_pending = False
                    else:
                        self._needs_replay = False
                        self._handle_resumed_pending = True
                    # Monitor is started at turn-end (or at end of an
                    # idle-triggered reconnect), NEVER here. _ensure_session
                    # is also called from _prepare_turn right before the turn
                    # reads, so starting a monitor here would race the turn
                    # for the single recv() reader.
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

    async def _send_user_turn(
        self, session: Any, prompt: str | list[NeutralMessage] | list[Content]
    ) -> Any:
        gemini_prompt = (
            [message_to_content(m) for m in prompt]
            if isinstance(prompt, list)
            and prompt
            and isinstance(prompt[0], NeutralMessage)
            else prompt
        )
        text = gemini_prompt if isinstance(gemini_prompt, str) else str(gemini_prompt)
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
                    return session
                except Exception as exc:
                    logger.warning(
                        f"Live history replay failed [{type(exc).__name__}: "
                        f"{exc}]; reconnecting and retrying replay"
                    )
                    # Retry: tear down the broken session, reconnect, and replay
                    # with full history so context is never silently dropped.
                    try:
                        await self._teardown_session()
                        session = await self._ensure_session()
                        turns = self._replay_turns()
                        turns.append(Content(role="user", parts=[Part(text=text)]))
                        logger.info(
                            f"Live: retry-replaying {len(self._history)} history "
                            "turns after reconnect"
                        )
                        self._emit(
                            "replaying",
                            {"turns": len(self._history), "retry": True},
                        )
                        await session.send_client_content(
                            turns=turns, turn_complete=True
                        )
                        if config.replay_image_count > 0:
                            await self._reinject_recent_images(session)
                        return session
                    except Exception as retry_exc:
                        logger.error(
                            f"Live history replay retry also failed "
                            f"[{type(retry_exc).__name__}: {retry_exc}]; "
                            "cannot deliver turn with full context"
                        )
                        raise RuntimeError(
                            "Live history replay failed twice; "
                            "cannot deliver turn with full context"
                        ) from retry_exc
        await session.send_realtime_input(text=text)
        return session

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
            video=types.Blob(data=data, mime_type=mime_type)
        )

    async def _reinject_recent_images(self, session: Any) -> None:
        for data, mime_type in list(self._recent_images):
            try:
                from google.genai import types

                await session.send_realtime_input(
                    video=types.Blob(data=data, mime_type=mime_type)
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

    async def _monitor_loop(self) -> None:
        # Runs ONLY between turns. Sole reader of session.receive().
        # Start-discipline: started at turn-end (generate/generate_stream) and
        # at the end of an idle-triggered reconnect. NEVER started from
        # _ensure_session (that path is also called right before a turn reads).
        session = self._session
        if session is None:
            return
        # Proactively reconnect BEFORE the server's GoAway deadline rather than
        # waiting for a reactive go_away (which only lands on the next turn —
        # too late for idle/long turns). asyncio.timeout(None) is a transparent
        # no-op, so the deadline-less path keeps the original unbounded wait.
        receive_timeout = (
            max(self._go_away_deadline - time.monotonic(), 0)
            if self._go_away_deadline is not None
            else None
        )
        try:
            try:
                async with asyncio.timeout(receive_timeout):
                    async for response in session.receive():
                        sru = response.session_resumption_update
                        if sru and sru.resumable and sru.new_handle:
                            self._session_handle = sru.new_handle
                            self._handle_resumed_pending = False
                            continue
                        if response.go_away:
                            logger.info(
                                "Idle monitor: go_away "
                                f"(time_left={response.go_away.time_left}); "
                                "reconnecting now"
                            )
                            await self._handle_idle_reconnect("idle_monitor")
                            return
                        # ignore stray server_content during idle
            except TimeoutError:
                # Deadline reached with no go_away in hand: reconnect proactively.
                await self._handle_idle_reconnect("deadline")
                return
        except asyncio.CancelledError:
            # Turn is taking the socket; clean handoff.
            raise
        except Exception as exc:
            logger.info(
                f"Idle monitor: recv failed [{type(exc).__name__}: {exc}]; reconnecting"
            )
            await self._handle_idle_reconnect("idle_monitor_error")

    async def _handle_idle_reconnect(self, source: str) -> None:
        """Common reconnect path for the idle monitor.

        On failure, sets _reconnect_pending so the next turn reconnects inline
        (fall-back path; we never want to lose the ability to recover).
        On cancellation (a turn started while we were reconnecting), the
        running task remains referenceable in _monitor_task so _stop_monitor
        can find and cancel it cleanly. The OLD session is left intact; the
        turn will use it, and the monitor restarts on the OLD session after
        the turn (catching any subsequent go_away).

        If _reconnect() gives up (attempts exhausted), we do NOT restart the
        monitor on the dying session — that would create a tight CPU loop as
        the new monitor immediately sees the same go_away / error and calls
        _reconnect() again, which gives up again. Instead, set
        _reconnect_pending=True and let the next turn handle it.
        """
        self._emit("reconnecting", {"source": source})
        # NOTE: _monitor_task is deliberately NOT nulled here. _stop_monitor
        # must be able to find this task if a turn starts while reconnect is
        # in progress — otherwise the background task would tear down the
        # session while the turn is reading from it.
        try:
            await self._reconnect()
            # Yield control to allow any pending cancellation to be delivered
            # before we mutate _monitor_task. Without this, a cancellation
            # that was queued during _reconnect() (e.g., a turn starting)
            # could not be observed until the next event-loop iteration,
            # racing with the new monitor task creation below.
            await asyncio.sleep(0)
            # _reconnect() succeeded. If it gave up (attempts exhausted),
            # do NOT spawn a new monitor on the dying session — that would
            # loop. Fall back to inline reconnect on the next turn.
            if self._reconnect_attempts >= self._reconnect_attempts_max:
                self._reconnect_pending = True
                self._monitor_task = None
                return
            # Reconnect succeeded. The old session is gone, the new session
            # is set. Replace the current monitor task with a fresh one.
            # We CANNOT call _start_monitor() here — it would see
            # self._monitor_task is not None (this task) and early-return.
            # Inlining the create ensures the slot transitions atomically
            # (no observable window with _monitor_task == None once the
            # new task is created).
            self._monitor_task = None
            if config.idle_monitor_enabled and self._session is not None:
                self._monitor_task = asyncio.create_task(self._monitor_loop())
        except asyncio.CancelledError:
            # Turn is taking the socket; clean handoff. When cancellation
            # arrives during _reconnect's backoff sleep, the session is
            # still intact. If it arrives during _reconnect's
            # _teardown_session (after the sleep), the teardown's
            # try/finally has already nulled the session pointers; the
            # turn's _prepare_turn will see _session is None and create
            # a fresh one. Either way, the turn ends up with a usable
            # session and the monitor restarts after the turn.
            raise
        except Exception:
            self._reconnect_pending = True
            self._monitor_task = None

    async def _start_monitor(self) -> None:
        if (
            not config.idle_monitor_enabled
            or self._monitor_task is not None
            or self._session is None
        ):
            return
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _stop_monitor(self) -> None:
        task = self._monitor_task
        self._monitor_task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        import contextlib

        with contextlib.suppress(BaseException):
            await task  # MUST await — guarantees recv() unwound before turn reads

    async def _prepare_turn(
        self,
        prompt: str | list[NeutralMessage] | list[Content],
        options: GenOptions | None = None,
    ) -> Any:
        if options:
            raw_history = (
                options.extras.get("history")
                if isinstance(options, GenOptions)
                else options.get("history")
            )
            if raw_history:
                if raw_history and isinstance(raw_history[0], NeutralMessage):
                    self._history = [message_to_content(m) for m in raw_history]
                else:
                    self._history = list(raw_history)
        # Stop the idle monitor BEFORE any socket read (single-reader invariant).
        # The monitor owns the only recv() between turns; the turn must take
        # over exclusively. Awaiting the monitor here guarantees recv() has
        # fully unwound before the turn's session.receive() runs.
        await self._stop_monitor()
        await self._consume_prewarm()
        if self._reconnect_pending:
            self._reconnect_pending = False
            self._go_away_deadline = None
            await self._teardown_session()
        session = await self._ensure_session()
        # Reset the reconnect-attempt counter after a successful inline
        # reconnect. _reconnect() resets it on its own success path, but
        # the inline path (bypassing _reconnect when the monitor exhausted
        # attempts) does not — leaving the monitor permanently degraded.
        # If we just established a fresh session, the counter is stale.
        if self._reconnect_attempts > 0:
            self._reconnect_attempts = 0
        if not self._last_turn_complete:
            await self._drain_session(session)
            if self._session is None:
                session = await self._ensure_session()

        self._last_turn_complete = False
        return await self._send_user_turn(session, prompt)

    async def generate(
        self,
        prompt: str | list[NeutralMessage],
        options: GenOptions | None = None,
    ) -> LLMResult:
        gemini_prompt = (
            [message_to_content(m) for m in prompt]
            if isinstance(prompt, list)
            and prompt
            and isinstance(prompt[0], NeutralMessage)
            else prompt
        )
        async with self._turn_lock:
            session = await self._prepare_turn(gemini_prompt, options)
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
                if _is_key_error(e):
                    self._rotate_key()

            if self._handle_resumed_pending:
                self._emit("empty_resume_suspected", {})
                self._handle_resumed_pending = False

            self._schedule_prewarm()
            if self._prewarm_task is None and not self._reconnect_pending:
                await self._start_monitor()
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
                history=[],
            )

    async def generate_stream(
        self,
        prompt: str | list[NeutralMessage],
        options: GenOptions | None = None,
    ) -> AsyncIterator[LLMChunk]:
        gemini_prompt = (
            [message_to_content(m) for m in prompt]
            if isinstance(prompt, list)
            and prompt
            and isinstance(prompt[0], NeutralMessage)
            else prompt
        )
        async with self._turn_lock:
            session = await self._prepare_turn(gemini_prompt, options)
            grounding: Any = None
            self._turn_did_complete = False

            try:
                async for response in session.receive():
                    if logger.isEnabledFor(logging.DEBUG):
                        summary = _summarize_live_response(response)
                        if summary is not None:
                            logger.debug("Live receive: %s", summary)
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
                        self._turn_did_complete = True
                        break
            except Exception as e:
                logger.error(f"generate_stream() error [{type(e).__name__}]: {e}")
                self._last_turn_complete = True
                self._reconnect_pending = True
                if _is_key_error(e):
                    self._rotate_key()
                raise
            finally:
                # Track whether the receive loop completed normally.
                # If it didn't, the consumer broke out (GeneratorExit)
                # and the server is still generating — tear down so the
                # next turn gets a fresh session instead of a dirty one.
                if not self._turn_did_complete:
                    await self._teardown_session()

                # ALWAYS run the turn-end epilogue, even if a consumer
                # breaks early (async generator aclose() injects
                # GeneratorExit at the last yield, bypassing the except
                # clause). Without this, a TUI abort would leave the
                # monitor unstarted and the next turn would catch idle
                # go_away inline (the "jerking" the plan was designed
                # to prevent).
                if self._handle_resumed_pending:
                    self._emit("empty_resume_suspected", {})
                    self._handle_resumed_pending = False

                self._schedule_prewarm()
                if self._prewarm_task is None and not self._reconnect_pending:
                    await self._start_monitor()

    async def _teardown_session(self) -> None:
        async with self._session_lock:
            if self._session_manager:
                # ALWAYS null the session pointers, even on CancelledError.
                # contextlib.suppress(Exception) does NOT catch CancelledError
                # (BaseException in 3.9+). If a turn cancels the monitor
                # mid-__aexit__, the session must still be torn down cleanly
                # so the next _ensure_session does not return a zombie.
                try:
                    with contextlib.suppress(Exception):
                        await self._session_manager.__aexit__(None, None, None)
                finally:
                    self._session = None
                    self._session_manager = None
            self._last_turn_complete = True

    async def close(self) -> None:
        await self._stop_monitor()
        await self._cancel_prewarm()
        await self._teardown_session()

    def get_name(self) -> str:
        return f"{self.model_name} (Live)"
