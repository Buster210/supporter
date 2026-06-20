from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from .config import config
from .decision_log import log_decision
from .history_summarizer import summarize_turns
from .llm.types import GenOptions, Message, TextPart
from .logger import logger
from .providers.gemini_codec import content_to_message
from .types import LLMChunk, LLMProvider, LLMResult

_GOAL_PREVIEW_CHARS = 200


def _build_message(role: str, text: str) -> Message:
    return Message(role=role, parts=[TextPart(text=text)])


def _extract_assistant_message(result: LLMResult) -> Any | None:
    # Prefer neutral result.history (last model message).
    history = getattr(result, "history", None) or []
    for msg in reversed(history):
        if getattr(msg, "role", None) == "model":
            return msg

    # Fall back to candidates using duck-typing + codec.
    candidates = getattr(result, "candidates", None) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    if content is None:
        return None
    return content_to_message(content)


class ChatAgent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Any] | None = None,
        registry: dict[str, Callable[..., Any]] | None = None,
        use_search: bool = False,
        use_code_execution: bool = False,
        system_instruction: str | None = None,
        session_id: str | None = None,
    ):
        self.provider = provider
        self.history: list[Any] = []
        self.current_interaction_id: str | None = None
        self.tools = tools
        self.registry = registry
        self.use_search = use_search
        self.use_code_execution = use_code_execution
        self.system_instruction = system_instruction
        self._store: Any = None
        self._store_prev_len: int = 0
        self.last_plan: str = ""
        self.last_plan_objective: str = ""
        # WHY: Cache summary state so compaction survives AFC clobber.
        # _summary covers turns [0, _summary_turn_count) of self.history;
        # _summary_fingerprint is a structural fingerprint of that same prefix,
        # so a wholesale history replacement (different branch, same length)
        # is detected as a stale cache even when the count check would pass.
        self._summary: str = ""
        self._summary_turn_count: int = 0
        self._summary_fingerprint: str = ""
        if config.durable_history_enabled:
            from .session import HistoryStore, new_session_id

            sid = (
                session_id or os.environ.get("SUPPORTER_SESSION_ID") or new_session_id()
            )
            self._store = HistoryStore(sid, Path(config.history_dir))
            loaded = self._store.load(limit=config.history_max_turns)
            if loaded:
                self.history = loaded
                self._trim_history()
                self._store_prev_len = len(self.history)
                logger.info(
                    f"ChatAgent: reloaded {len(self.history)} turns "
                    "from durable history"
                )
        logger.info(f"ChatAgent initialized with provider: {provider.get_name()}")

    def _prepare_execution_context(self) -> GenOptions:
        history_for_send = self._build_compacted_history()
        system_instruction = self.system_instruction
        # WHY: inject recent working memory + a "known automations" hint
        # into the system prompt so the assistant's context is never
        # blank after a restart. This is the wire that turns the memory
        # + recipe stores into knowledge the model actually sees.
        if system_instruction is not None:
            injected = self._build_context_injection()
            if injected:
                system_instruction = f"{system_instruction}\n\n{injected}"
        return GenOptions(
            system_instruction=system_instruction,
            use_search=self.use_search,
            extras={
                "history": history_for_send,
                "interaction_id": self.current_interaction_id,
                "tools": self.tools or [],
                "registry": self.registry or {},
                "use_code_execution": self.use_code_execution,
            },
        )

    @staticmethod
    def _build_context_injection(limit: int = 5) -> str:
        """Compose the working-memory + recipe digest that gets prepended
        to the system prompt on every turn.

        Each source is independently best-effort: a corrupt store
        should not prevent the agent from running. Failures are logged
        at debug level.
        """
        from .logger import logger
        from .memory import memory_snapshot
        from .recipes import recipes_snapshot
        from .tools.memory_tools import memory_render_block

        def _memory_block() -> str:
            block = memory_render_block(limit=limit)
            return block or ""

        def _recipes_block() -> str:
            snap = recipes_snapshot()
            total = snap.get("total", 0) if isinstance(snap, dict) else 0
            if not total:
                return ""
            return (
                f"KNOWN AUTOMATIONS ({total} recipes available): "
                "use the recipe_list / recipe_run / recipe_find tools "
                "to replay any saved multi-step workflow without LLM tokens."
            )

        def _kinds_block() -> str:
            snap = memory_snapshot()
            kinds = snap.get("kinds", {}) if isinstance(snap, dict) else {}
            if not kinds:
                return ""
            kinds_repr = ", ".join(
                f"{k}={v}"
                for k, v in sorted(kinds.items(), key=lambda x: -x[1])[:5]
            )
            return f"WORKING MEMORY TOTALS: {kinds_repr}"

        parts: list[str] = []
        for render in (_memory_block, _recipes_block, _kinds_block):
            try:
                rendered = render()
            except Exception as exc:
                logger.debug(
                    f"ChatAgent: injection failed [{type(exc).__name__}]: {exc}"
                )
                continue
            if rendered:
                parts.append(rendered)
        return "\n\n".join(parts)

    @staticmethod
    def _entry_text(entry: Any) -> str:
        """Pull text out of a history entry — Message or genai Content.

        Both shapes expose `.role` and `.parts`; each part may expose `.text`
        (TextPart, genai Part, or anything duck-typed). Falls back to
        ``str(part)`` so non-text parts still contribute a stable token.
        """
        pieces: list[str] = []
        for part in getattr(entry, "parts", None) or []:
            text = getattr(part, "text", None)
            if text:
                pieces.append(str(text))
            else:
                pieces.append(str(part))
        return "\x1d".join(pieces)

    def _fingerprint(self, n: int) -> str:
        """Deterministic structural fingerprint of the first ``n`` entries.

        Works for both ``Message`` and ``genai.types.Content`` because both
        expose ``.role`` and ``.parts``. The fingerprint is what guards the
        cached summary against branch-swap / wholesale history replacement.
        """
        import hashlib

        parts: list[str] = []
        for entry in self.history[:n]:
            role = getattr(entry, "role", "") or ""
            text = self._entry_text(entry)
            parts.append(f"{role}\x1f{text}")
        raw = "\x1e".join(parts)
        return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()[:32]

    def _summary_is_stale(self) -> bool:
        """True iff the cached summary is no longer valid for self.history.

        Invalidation triggers:
          - no summary cached
          - _summary_turn_count >= current history length: a pure shrink to or
            below the coverage point (preserves the original guard so the
            cache-hit splice never overlaps already-summarized recent turns)
          - structural fingerprint of the first _summary_turn_count entries
            has changed (branch-swap at the same length, in-place mutation, etc.)
        """
        if not self._summary:
            return True
        if self._summary_turn_count >= len(self.history):
            return True
        return self._fingerprint(self._summary_turn_count) != self._summary_fingerprint

    def _build_compacted_history(self) -> list[Any]:
        """Build compacted history view for LLM context.

        WHY: Returns [summary turn] + recent turns instead of full history.
        This is a READ-TIME view; self.history stays full for persistence.
        The summary is cached and only regenerated when the uncovered tail grows.
        """
        if not config.history_compaction_enabled:
            return self.history

        keep_recent = config.history_summary_keep_recent

        if len(self.history) <= keep_recent:
            return self.history

        # WHY: Cached summary is valid only while the fingerprint of the first
        # _summary_turn_count history entries is unchanged. A pure shrink AND
        # a shrink-then-regrow with a DIFFERENT branch both invalidate it,
        # because we trust the structural fingerprint — not the length count.
        if self._summary_is_stale():
            self._summary = ""
            self._summary_turn_count = 0
            self._summary_fingerprint = ""
        uncovered_count = len(self.history) - self._summary_turn_count
        if uncovered_count <= keep_recent and self._summary:
            summary_text = f"[PREVIOUS_CONTEXT_SUMMARY]\n{self._summary}"
            summary_turn = Message(role="model", parts=[TextPart(text=summary_text)])
            return [summary_turn, *self.history[-keep_recent:]]

        return self.history

    async def _maybe_summarize(self) -> bool:
        """Summarize old turns if past trigger threshold.

        WHY: Called before each execution; summary must happen BEFORE hard trim
        at history_max_turns or context is lost.

        Returns True if summarization succeeded, False if fallback to trim needed.
        """
        if not config.history_compaction_enabled:
            return False

        trigger = config.history_compaction_trigger
        keep_recent = config.history_summary_keep_recent

        if len(self.history) <= trigger:
            return False

        # WHY: Invalidate stale summary so the coverage math (and resulting
        # recent-turns slice) is correct. Uses the structural fingerprint so
        # a branch-swap at the same length is also detected.
        if self._summary_is_stale():
            self._summary = ""
            self._summary_turn_count = 0
            self._summary_fingerprint = ""

        if len(self.history) > keep_recent:
            turns_to_summarize = self.history[:-keep_recent]
        else:
            turns_to_summarize = []

        try:
            summary = await summarize_turns(turns_to_summarize)
            if summary:
                self._summary = summary
                self._summary_turn_count = len(self.history) - keep_recent
                self._summary_fingerprint = self._fingerprint(
                    self._summary_turn_count
                )
                logger.info(
                    f"Summarized {len(turns_to_summarize)} history turns "
                    f"(kept {keep_recent} recent)"
                )
                return True
        except RuntimeError as e:
            logger.error(
                f"History summarization impossible (turns will be hard-dropped): {e}"
            )
        except Exception as e:
            logger.warning(f"History summarization failed: {e}")

        return False

    def _trim_history(self) -> None:
        cap = config.history_max_turns
        if cap and len(self.history) > cap:
            del self.history[: len(self.history) - cap]

    async def execute(self, prompt: str) -> LLMResult:
        logger.info(f"Agent: executing prompt — length={len(prompt)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Agent: full prompt: {prompt!r}")

        # WHY: Summarize before we might hit the hard trim cap.
        if not await self._maybe_summarize():
            self._trim_history()

        user_message = _build_message("user", prompt)
        options = self._prepare_execution_context()
        result = await self.provider.generate(prompt, options)

        self.current_interaction_id = result.interaction_id
        self._sync_history(user_message, result)
        if self._store:
            self._store.sync()
        self._record_brain_decision(prompt, result)

        duration_str = (
            f"{result.duration:.3f}s" if result.duration is not None else "unknown"
        )
        logger.info(
            f"Agent: execution complete — duration={duration_str}, "
            f"history_size={len(self.history)}"
        )
        return result

    async def execute_with_verification(
        self,
        prompt: str,
        checks: list[Any] | None = None,
        config: Any | None = None,
        recover: Any | None = None,
    ) -> Any:
        """Run the prompt through a verification loop.

        On every retry the LLM receives the *original* prompt plus a
        structured "your previous response failed verification" follow-up
        that names each failing check and its captured detail. History is
        synced only with the *final* result so the user-visible transcript
        shows the chosen answer, not the rejected intermediate ones.

        If ``recover`` is an :class:`supporter.recover.AutoRecover` it
        wraps every provider call so transient 5xx / network failures
        rotate the keypool and retry without LLM involvement.
        """
        from .verify import VerificationConfig, VerificationLoop

        if not await self._maybe_summarize():
            self._trim_history()

        cfg = config or VerificationConfig()
        loop = VerificationLoop(cfg, checks or [])

        async def _caller(text: str) -> LLMResult:
            options = self._prepare_execution_context()
            if recover is not None:
                result: LLMResult = await recover.call(
                    self.provider.generate, text, options
                )
                return result
            return await self.provider.generate(text, options)

        outcome = await loop.run(_caller, prompt)
        last = outcome.last_result
        if last is None:
            # Should not happen — at least one attempt always runs.
            return outcome

        # Sync history with the *final* result only; intermediate attempts
        # are not user-visible.
        user_message = _build_message("user", prompt)
        self.current_interaction_id = last.interaction_id
        self._sync_history(user_message, last)
        if self._store:
            self._store.sync()
        self._record_brain_decision(prompt, last)

        logger.info(
            f"Agent: verification complete — ok={outcome.ok} "
            f"attempts={outcome.attempts} history_size={len(self.history)}"
        )
        return outcome

    def _record_brain_decision(self, prompt: str, result: LLMResult) -> None:
        chosen = "text_response"
        try:
            candidate = result.candidates[0] if result.candidates else None
            content = getattr(candidate, "content", None) if candidate else None
            for part in getattr(content, "parts", None) or []:
                fc = getattr(part, "function_call", None)
                name = getattr(fc, "name", None) if fc else None
                if name:
                    chosen = name
                    break
        except Exception as exc:
            logger.debug(f"brain decision extract failed [{type(exc).__name__}]: {exc}")
        goal = " ".join(prompt.split())[:_GOAL_PREVIEW_CHARS]
        rationale = (result.thoughts or "").strip()
        reason = f"goal: {goal}" + (f" | {rationale}" if rationale else "")
        log_decision(
            site="brain.tool_choice",
            chosen=chosen,
            reason=reason,
            correlation_id=result.interaction_id,
        )

    def _sync_history(self, user_message: Any, result: LLMResult) -> None:
        # Prefer neutral result.history if available.
        history = getattr(result, "history", None)
        if history:
            logger.info("Agent: syncing history from result.history")
            new_list = list(history)
            self._commit_synced_history(new_list)
            return

        # Fall back to AFC history (convert google Content → neutral Message).
        if result.automatic_function_calling_history:
            logger.info("Agent: syncing history from automatic function calling")
            new_list_raw = result.automatic_function_calling_history
            new_list = [content_to_message(c) for c in new_list_raw]
            self._commit_synced_history(new_list)
            return

        self.history.append(user_message)
        if self._store:
            self._store.append(user_message)

        assistant_message = _extract_assistant_message(result)
        if assistant_message is None:
            self._trim_history()
            return

        self.history.append(assistant_message)
        self._commit_synced_history(None)
        logger.info(f"Agent: history synced — new size={len(self.history)}")

    def _commit_synced_history(self, new_list: list[Any] | None) -> None:
        if new_list is not None:
            if self._store and len(new_list) > self._store_prev_len:
                for msg in new_list[self._store_prev_len :]:
                    self._store.append(msg)
            self.history = new_list
            # _store_prev_len reflects the POST-trim length for the
            # replace-history branches (matches the original ordering).
            self._trim_history()
            if self._store:
                self._store_prev_len = len(self.history)
            return
        # Branch 3: user + assistant already appended to self.history; mirror
        # the assistant append to the store and bookkeep the PRE-trim length,
        # then trim (matches the original ordering for this branch).
        if self._store:
            self._store.append(self.history[-1])
            self._store_prev_len = len(self.history)
        self._trim_history()

    async def execute_stream(
        self, prompt: str, exclude_from_history: bool = False
    ) -> AsyncIterator[LLMChunk]:
        logger.info(f"Agent: executing streaming prompt — length={len(prompt)}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Agent: full streaming prompt: {prompt!r}")

        # WHY: Summarize before we might hit the hard trim cap.
        if not await self._maybe_summarize():
            self._trim_history()

        user_message = _build_message("user", prompt)
        options = self._prepare_execution_context()
        collected_parts: list[Any] = []

        async for chunk in self.provider.generate_stream(prompt, options):
            if chunk.raw is not None:
                raw_content = getattr(chunk.raw, "content", None)
                if raw_content is not None:
                    msg = content_to_message(raw_content)
                    collected_parts.extend(msg.parts)
            elif chunk.text:
                collected_parts.append(TextPart(text=chunk.text))
            yield chunk

        if not exclude_from_history:
            self.history.append(user_message)
            if collected_parts:
                model_msg = Message(role="model", parts=collected_parts)
            else:
                model_msg = _build_message("model", "")
            self.history.append(model_msg)
            if self._store:
                self._store.append(user_message)
                self._store.append(model_msg)
                self._store_prev_len = len(self.history)
                self._store.sync()
            self._trim_history()
        logger.info(f"Agent: stream complete — history_size={len(self.history)}")

    def clear_history(self) -> None:
        logger.info("Clearing agent session history")
        self.history = []
        self._summary = ""
        self._summary_turn_count = 0
        self._summary_fingerprint = ""
        self.current_interaction_id = None
        if self._store:
            self._store.rotate()
            self._store_prev_len = 0
