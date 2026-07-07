from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Input, Label, Static

from ..config import config
from ..logger import init_logger, logger, shutdown_logger
from ..tools.base import ToolError
from ..types import ModeChanged
from .bubble import MessageBubble
from .chat import (
    ChatContainer,
    ChatTurn,
    QueuedMessagesDisplay,
    SupporterHeader,
    ThinkingIndicator,
    WelcomeBanner,
)
from .delegation import DelegationBlock
from .delegation_listener import DelegationListener
from .message_processor import ChatMessageProcessor
from .modals import ConfirmationModal, ProfileSelectModal
from .mode_manager import ModeManager
from .utils import ToastManager

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.widgets import Button, Input

    from ..agent import ChatAgent


class SupporterApp(App[None]):
    CSS_PATH = "styles.tcss"

    status_label = reactive("Thinking")
    active_queries = reactive(0)
    is_activating_mode = reactive(False)
    live_mode = reactive(True)
    active_turn: reactive[ChatTurn | None] = reactive(None)

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
        Binding(
            "pageup", "scroll_chat('pageup')", "Scroll up", show=False, priority=True
        ),
        Binding(
            "pagedown",
            "scroll_chat('pagedown')",
            "Scroll down",
            show=False,
            priority=True,
        ),
        Binding("ctrl+home", "scroll_chat('home')", "Top", show=False, priority=True),
        Binding("ctrl+end", "scroll_chat('end')", "Bottom", show=False, priority=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.agent: ChatAgent | None = None
        self._mode_manager = ModeManager(self)
        self._message_processor = ChatMessageProcessor(self)
        self._is_processing = False
        self._user_message_queue: list[tuple[str, bool]] = []
        self._toast_manager = ToastManager()
        self._delegation_blocks: dict[str, DelegationBlock] = {}
        self._delegation_listener = DelegationListener(
            inject_message=self._inject_delegation_message,
            drop_progress=self._drop_delegation_progress,
            render_signal=self._render_delegation_signal_now,
            render_progress_live=self._render_delegation_progress_live,
            render_summary=self._mount_delegation_summary,
            plan_bubble_injector=self._inject_plan_bubble,
            plan_storer=self._store_pending_plan,
        )

    async def on_mode_changed(self, event: ModeChanged) -> None:
        indicator = self.query_one("#mode-indicator", Label)
        indicator.update(f"[{event.mode}]")
        status = "ENABLED" if event.enabled else "DISABLED"

        target = self.active_turn or self.query_one("#chat-view")
        await target.mount(
            MessageBubble(role="agent", content=f"Single Agent {status}")
        )

    async def on_mount(self) -> None:
        # WHY: Lazy imports — these modules spawn subprocesses, open OS handles,
        # or touch the network at import time. Deferring to on_mount avoids
        # startup overhead before the app is interactive.
        from ..tools.bash.sandbox import register_bash_callbacks
        from ..tools.browser.guardrails import register_browse_callback
        from ..tools.delegate.api import set_delegation_start_callback
        from ..tools.delegate.scheduler import resume_interrupted_jobs
        from ..tools.file_ops import register_confirmation_callback

        init_logger()
        register_confirmation_callback(self._confirm_write)
        register_bash_callbacks(
            confirmation=self._confirm_bash,
            notification=self._notify_error,
        )
        register_browse_callback(
            confirmation=self._confirm_browse,
            profile_select=self._select_profile,
        )

        set_delegation_start_callback(self._start_delegation_listener)

        from ..tools.browser.session import prewarm_clone

        logger.info("Supporter TUI dashboard active")
        self._mode_manager.start_warmup()
        try:
            # WHY: Own worker group so resume isn't cancelled by the exclusive
            # _setup_agent worker (default group).
            self.run_worker(
                resume_interrupted_jobs(), name="resume-jobs", group="resume-jobs"
            )
            self.run_worker(self._setup_agent(use_live=True), exclusive=True)
            self.run_worker(self._mode_manager.trigger_live_greeting())
            self.run_worker(prewarm_clone())
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Startup failure [{type(e).__name__}]: {e}"
            logger.error(msg)
            self._toast_manager.notify(self, msg, type="system")

    async def on_unmount(self) -> None:
        from ..tools.bash.sandbox import register_bash_callbacks
        from ..tools.browser.guardrails import register_browse_callback
        from ..tools.browser.session import close_session
        from ..tools.file_ops import register_confirmation_callback

        self.workers.cancel_all()

        register_confirmation_callback(None)
        register_bash_callbacks(confirmation=None, notification=None)
        register_browse_callback(
            confirmation=None,
            profile_select=None,
        )

        from ..tools.delegate.api import set_delegation_start_callback

        set_delegation_start_callback(None)

        await close_session()

        if (
            self.agent
            and hasattr(self.agent, "provider")
            and hasattr(self.agent.provider, "close")
        ):
            await self.agent.provider.close()

        from ..pool import DynamicPool

        await DynamicPool.shutdown_all()
        await self._mode_manager.close()
        shutdown_logger()

    async def _setup_agent(self, use_live: bool = False) -> None:
        await self._mode_manager.setup_agent(use_live=use_live)
        await self._replay_history()

    async def _replay_history(self) -> None:
        """Mount full persisted history as scrollable bubbles on startup.

        Loads ALL records from the history store (uncapped) so sessions with
        >200 turns still display everything in the UI. Does not trigger the
        thinking indicator or any LLM call.
        """
        if not self.agent or not self.agent._store:
            return

        from ..llm.types import TextPart, ToolCallPart

        records = self.agent._store.load(limit=None)
        if not records:
            return

        chat_view = self.query_one("#chat-view", ChatContainer)

        # Hide welcome banner when history is non-empty.
        for banner in self.query(WelcomeBanner):
            banner.message = ""

        current_turn: ChatTurn | None = None

        for msg in records:
            role: str = msg.role

            if role == "user":
                text = " ".join(
                    p.text for p in msg.parts if isinstance(p, TextPart) and p.text
                )
                current_turn = ChatTurn(MessageBubble(role="user", content=text))
                await chat_view.mount(current_turn)

            elif role == "model":
                bubble = MessageBubble(role="agent", content="", streaming=False)
                for part in msg.parts:
                    if isinstance(part, TextPart) and part.text:
                        bubble.append_token(part.text)
                    elif isinstance(part, ToolCallPart):
                        bubble.add_tool_call(part.name, part.args)
                bubble.finalize()
                if current_turn is not None:
                    await current_turn.mount_bubble(bubble)
                else:
                    await chat_view.mount(bubble)

        chat_view.jump_to_bottom()

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield SupporterHeader(id="supporter-header")
            with ChatContainer(id="chat-view"):
                yield WelcomeBanner(id="welcome-banner", classes="hidden")
            yield ThinkingIndicator(id="thinking-indicator")

            with Horizontal(id="scroll-btn-wrapper", classes="hidden"):
                yield Button("↓ Go to bottom", id="scroll-bottom-btn")
            yield QueuedMessagesDisplay(id="queue-display")
            with Vertical(id="input-area"), Horizontal(id="prompt-row"):
                yield Label("[LIVE]", id="mode-indicator", markup=False)
                yield Label(">", id="prompt-symbol")
                yield Input(
                    placeholder="Type a message... (/agent, /live, /clear, /exit)",
                    id="user-input",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scroll-bottom-btn":
            self.query_one("#chat-view", ChatContainer).jump_to_bottom()

    def action_scroll_chat(self, direction: str) -> None:
        chat_view = self.query_one("#chat-view", ChatContainer)
        # WHY: animate=False avoids burst of repaints that stall the event loop
        # on slow terminals; an instant jump is one repaint, not many.
        if direction == "pageup":
            chat_view.scroll_page_up(animate=False)
        elif direction == "pagedown":
            chat_view.scroll_page_down(animate=False)
        elif direction == "home":
            chat_view.scroll_home(animate=False)
        elif direction == "end":
            chat_view.jump_to_bottom()

    def action_clear_screen(self) -> None:
        chat_view = self.query_one("#chat-view")
        if not chat_view.query(ChatTurn) and (not self.agent or not self.agent.history):
            self._toast_manager.notify(self, "Session already clear", type="system")
            return

        if self.agent:
            self.agent.clear_history()
        chat_view.query(ChatTurn).remove()
        self._user_message_queue.clear()
        self.query_one("#queue-display", QueuedMessagesDisplay).update_queue([])
        self.active_turn = None

    def start_thinking(self) -> None:
        self.active_queries += 1

    def stop_thinking(self) -> None:
        self.active_queries = max(0, self.active_queries - 1)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        user_text = event.value.strip()
        if not user_text:
            return

        input_widget = self.query_one("#user-input", Input)
        input_widget.value = ""
        input_widget.focus()

        if user_text.startswith("/") and await self._handle_command(user_text):
            return

        if self._is_processing:
            self._user_message_queue.append((user_text, False))
            self.query_one("#queue-display", QueuedMessagesDisplay).update_queue(
                self._queue_display_labels()
            )
            self._toast_manager.notify(
                self, f"Message queued ({len(self._user_message_queue)})", type="queue"
            )
            return

        if not self.agent:
            self._toast_manager.notify(
                self, "Agent is still initializing... please wait.", type="system"
            )
            return

        new_turn = await self._mount_user_turn(user_text)
        self._is_processing = True
        self.run_worker(
            self._process_message_cycle(user_text, mount_user=False, target=new_turn)
        )

    async def _mount_user_turn(self, text: str, role: str = "user") -> ChatTurn:
        chat_view = self.query_one("#chat-view", ChatContainer)
        try:
            for turn in chat_view.query(ChatTurn):
                turn.auto_collapse()
        except TypeError:
            pass
        new_turn = ChatTurn(MessageBubble(role=role, content=text))
        self.active_turn = new_turn
        await chat_view.mount(new_turn)
        chat_view.jump_to_bottom()
        return new_turn

    async def _handle_command(self, command: str) -> bool:
        return await self._mode_manager.handle_command(command)

    async def set_live_mode(self, live: bool = False) -> None:
        await self._mode_manager.toggle_mode(live=live)

    async def _process_message_cycle(
        self,
        text: str,
        mount_user: bool = True,
        target: Vertical | None = None,
        exclude_from_history: bool = False,
        role: str = "user",
    ) -> None:
        self._is_processing = True
        chat_view = self.query_one("#chat-view", ChatContainer)
        if mount_user:
            await self._mount_user_turn(text, role=role)
            self.query_one("#user-input").focus()

        target_container = target or (
            self.active_turn if hasattr(self, "active_turn") else chat_view
        )
        if not isinstance(target_container, Vertical):
            raise RuntimeError("Invalid UI state: chat view container missing")

        self.status_label = "Thinking"
        self.start_thinking()
        start_time = getattr(target_container, "turn_start_time", time.perf_counter())

        try:
            if not self.agent:
                raise RuntimeError("Agent is not initialized")

            # G2: the model triages each turn — it answers directly or delegates
            # to the planner. Run one pass, then engage the verify/replan loop
            # only if a plan was actually produced (model chose the task route).
            from ..config import config
            from ..replan import ReplanContext

            # ponytail: first pass uses the caller's exclude_from_history; a task
            # turn that verifies first try therefore persists, unlike the old
            # always-excluded replan path. Acceptable — the answer belongs in
            # history. Replan re-runs below stay excluded.
            cycle_result = await self._process_streaming_execution(
                text,
                target_container,
                start_time,
                self.agent,
                exclude_from_history,
            )

            verified = False
            replan_ctx = None
            last_failure_reason = ""
            if self.agent.pending_plan_objective:
                replan_ctx = ReplanContext(
                    self.agent.pending_plan_objective, config.replan_max_cycles
                )
                replan_ctx.next_cycle()  # attempt 1 = the pass just executed

                while True:
                    # Phase 3: VERIFY
                    self.status_label = f"Verifying (attempt {replan_ctx.cycle})..."
                    objective = self.agent.pending_plan_objective
                    plan = self.agent.pending_plan_text
                    result_text = (
                        getattr(cycle_result, "content", "") if cycle_result else ""
                    )

                    verified = await self._verify_and_possibly_replan(
                        objective,
                        plan,
                        result_text,
                        target_container,
                        chat_view,
                        replan_ctx,
                    )
                    if verified:
                        self.status_label = "✓ Verification complete"
                        break
                    # Not verified; capture reason and replan if budget remains.
                    last_failure_reason = (
                        replan_ctx.failures[-1] if replan_ctx.failures else "Unknown"
                    )
                    if not replan_ctx.next_cycle():
                        break

                    # Phase 2: REPLAN + IMPLEMENT — feed failure context back.
                    self.status_label = (
                        f"Task: executing (attempt {replan_ctx.cycle})..."
                    )
                    current_prompt = replan_ctx.format_replan_prompt_context()
                    cycle_result = await self._process_streaming_execution(
                        current_prompt,
                        target_container,
                        start_time,
                        self.agent,
                        exclude_from_history=True,
                    )

            if replan_ctx and not verified:
                await target_container.mount(
                    MessageBubble(
                        role="system",
                        content=(
                            f"✗ Exhausted {config.replan_max_cycles} replan attempts. "
                            f"Last failure: {last_failure_reason}"
                        ),
                    )
                )
                chat_view.jump_to_bottom()

            # Clear so a stale plan from a failed turn never leaks.
            if self.agent:
                self.agent.pending_plan_objective = ""
                self.agent.pending_plan_text = ""
        except ToolError as e:
            await chat_view.mount(MessageBubble(role="agent", content=e.user_message))
        except Exception as e:
            logger.error(f"UI Message Cycle Error [{type(e).__name__}]: {e}")
            await chat_view.mount(
                MessageBubble(
                    role="agent",
                    content="An error occurred while processing your message. "
                    "Try again or rephrasing your request.",
                )
            )
        finally:
            self._is_processing = False
            self.stop_thinking()
            await self._flush_queued_messages()

    async def _flush_queued_messages(self) -> None:
        # Only mark busy once we know there is work; setting it before the
        # empty-queue check wedged the app permanently after the first turn
        # (flag stuck True with nothing draining).
        if not self._user_message_queue:
            return

        self._is_processing = True
        items = list(self._user_message_queue)
        self._user_message_queue.clear()
        self.query_one("#queue-display", QueuedMessagesDisplay).update_queue([])

        combined_text = "\n\n".join(msg for msg, _ in items)
        has_user_message = any(not is_sys for _, is_sys in items)

        if has_user_message:
            self.run_worker(self._process_message_cycle(combined_text, mount_user=True))
        else:
            self.run_worker(self._process_system_message(combined_text))

    async def _process_streaming_execution(
        self,
        text: str,
        target: Vertical,
        start_time: float,
        agent: ChatAgent,
        exclude_from_history: bool = False,
    ) -> Any:
        return await self._message_processor.process_streaming(
            text, target, start_time, agent, exclude_from_history
        )

    async def _verify_and_possibly_replan(
        self,
        objective: str,
        plan: str,
        result_text: str,
        target: Vertical,
        chat_view: ChatContainer,
        replan_ctx: Any,
    ) -> bool:
        """Verify result and replan on failure (G2 helper for interactive cycle).

        Returns True if verified, False if failed and out of replan budget.
        ponytail: Replan in interactive TUI reuses same verify_plan predicate
        as worker; replan message just shows failure + status; actual replanning
        (re-prompt) happens on next cycle via the message loop.
        """
        try:
            from ..prompts import DELEGATE_AGENT_ROSTER
            from ..worker import verify_plan

            model = DELEGATE_AGENT_ROSTER.get("planner", {}).get(
                "model", "gemma-4-31b-it"
            )
            is_done, reason = await verify_plan(objective, plan, result_text, model)
            if is_done:
                return True

            # Verification failed
            logger.info(
                f"TUI: planner says NOT done (cycle {replan_ctx.cycle}): {reason}"
            )
            replan_ctx.record_failure(reason)

            # Show failure warning
            await target.mount(
                MessageBubble(
                    role="system",
                    content=(f"⚠ Verification (attempt {replan_ctx.cycle}): {reason}"),
                )
            )
            chat_view.jump_to_bottom()
            return False
        except Exception as exc:
            logger.warning(f"_verify_and_possibly_replan failed: {exc}")
            return True  # Fail-open: treat verify errors as pass

    def _confirm_write(self, path: Path, content: str) -> bool:
        import threading

        event = threading.Event()
        result = [False]

        def callback(value: bool) -> None:
            result[0] = value
            event.set()

        title = f"Write {path.name}?"
        self.call_from_thread(
            self.push_screen,
            ConfirmationModal(title=title, content=content, language="diff"),
            callback,
        )
        event.wait()
        return result[0]

    def _confirm_bash(self, tokens: list[str], cwd: str) -> bool:
        import threading

        event = threading.Event()
        result = [False]

        def callback(value: bool) -> None:
            result[0] = value
            event.set()

        cmd_str = " ".join(tokens)
        self.call_from_thread(
            self.push_screen,
            ConfirmationModal(
                title="Authorize Bash Execution?",
                content=cmd_str,
                language="bash",
                meta=f"Working Dir: {cwd}",
            ),
            callback,
        )
        event.wait()
        return result[0]

    async def _confirm_browse(self, title: str, detail: str) -> bool:
        if getattr(config, "browser_auto_approve", False):
            return True
        return await self.push_screen_wait(
            ConfirmationModal(title=title, content=detail, language="text")
        )

    async def _select_profile(self, profiles: list[Any]) -> str | None:
        return await self.push_screen_wait(ProfileSelectModal(profiles))

    def _safe_call(
        self, callback: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> None:
        import threading

        if threading.current_thread() is threading.main_thread():
            callback(*args, **kwargs)
        else:
            self.call_from_thread(callback, *args, **kwargs)

    def _notify_error(self, message: str) -> None:
        self._safe_call(self._toast_manager.notify, self, message, type="error")

    def _queue_display_labels(self) -> list[str]:
        # Only user-typed messages belong in the queue badges. System messages
        # (e.g. the delegation capsule JSON) ride the same queue to serialize
        # behind a busy agent, but are not user input -- never show them.
        return [text for text, is_sys in self._user_message_queue if not is_sys]

    def _inject_system_message(self, text: str) -> None:
        if self._is_processing:
            self._user_message_queue.append((text, True))
            self.query_one("#queue-display", QueuedMessagesDisplay).update_queue(
                self._queue_display_labels()
            )
        else:
            self.run_worker(self._process_system_message(text))

    def _render_delegation_signal_now(self, text: str) -> None:
        self._safe_call(lambda: self.run_worker(self._render_delegation_signal(text)))

    def _render_delegation_progress_live(self, job_id: str, progress_md: str) -> None:
        """Render progress table live as updates arrive (bypass buffering)."""
        self._safe_call(
            lambda: self.run_worker(self._mount_live_progress(job_id, progress_md))
        )

    async def _mount_live_progress(self, job_id: str, progress_md: str) -> None:
        """Update or create collapsible delegation block with live progress."""
        # ponytail: reuse existing delegation block if it exists for this job,
        # otherwise create a new one. The block updates in-place rather than
        # creating new widgets on every progress event.
        block = self._delegation_blocks.get(job_id)
        if block is not None:
            # Update existing block in-place.
            block.set_progress(progress_md)
            return

        # Create new delegation block for this job.
        block = DelegationBlock(title=f"Delegation [{job_id}]")
        block.set_progress(progress_md)
        self._delegation_blocks[job_id] = block
        await self._mount_delegation_widget(block)

    def _delegation_mount_target(self) -> Vertical:
        # Fallback target when there is no agent bubble to host delegation UI:
        # the active turn, or the last turn, never the chat root.
        chat_view = self.query_one("#chat-view", ChatContainer)
        if self.active_turn is not None:
            return self.active_turn
        turns = list(chat_view.query(ChatTurn))
        return turns[-1] if turns else chat_view

    def _delegation_host_bubble(self) -> MessageBubble | None:
        # The delegation block belongs INSIDE the model's answer bubble (the
        # turn's last agent bubble), before its meta line -- not as a sibling
        # mounted after that bubble's metadata.
        turn = self.active_turn
        if turn is None:
            turns = list(self.query_one("#chat-view", ChatContainer).query(ChatTurn))
            turn = turns[-1] if turns else None
        bubbles = getattr(turn, "agent_bubbles", None)
        if bubbles:
            return bubbles[-1]  # type: ignore[no-any-return]
        return None

    async def _mount_delegation_widget(self, widget: Any) -> None:
        """Mount a delegation widget inside the triggering bubble, before its
        meta line. Fall back to the turn when no host bubble exists yet."""
        host = self._delegation_host_bubble()
        if host is not None and host.append_before_meta(widget):
            self.query_one("#chat-view", ChatContainer).follow_end()
            return
        target = self._delegation_mount_target()
        if isinstance(widget, MessageBubble) and hasattr(target, "mount_bubble"):
            await target.mount_bubble(widget)
        else:
            await target.mount(widget)
        self.query_one("#chat-view", ChatContainer).follow_end()

    async def _render_delegation_signal(self, text: str) -> None:
        body = text.replace("<br/>", "").replace("`", "").strip()
        # ponytail: removed buffer check — signals mount directly.
        await self._mount_delegation_widget(Static(body, classes="delegation-signal"))

    async def _process_system_message(self, text: str) -> None:
        suppress_bubble = (
            "DELEGATION_CAPSULE_RESULT (json)" in text
            or "MILESTONE_RESULT (json)" in text
        )
        # ponytail: removed _defer_agent_meta_once and flush coordination;
        # delegation updates stream directly to UI as sibling widgets.
        if self.active_turn:
            await self._process_message_cycle(text, mount_user=False)
        elif suppress_bubble:
            await self._process_message_cycle(text, mount_user=False, role="agent")
        else:
            await self._process_message_cycle(text, mount_user=True, role="agent")

    def _start_delegation_listener(self, job_id: str, plan_table: str = "") -> None:
        # Store job_id on active turn for delegation block lookup
        if self.active_turn:
            self.active_turn._delegation_job_id = job_id  # type: ignore[attr-defined]
        self.run_worker(self._delegation_listener.listen(job_id), exclusive=False)

    def _inject_delegation_message(self, message: str) -> None:
        self._safe_call(self._inject_system_message, message)

    def _inject_plan_bubble(self, markdown: str) -> None:
        """Update delegation block with plan, or mount if not yet created."""
        self._safe_call(lambda: self.run_worker(self._mount_plan_bubble(markdown)))

    async def _mount_plan_bubble(self, markdown: str) -> None:
        # G4: Plan appears IMMEDIATELY when planner finishes. If no delegation
        # block exists yet, create one. Otherwise update the existing block.
        # Use a generic job ID for the active turn's delegation
        job_id = getattr(self.active_turn, "_delegation_job_id", "current")
        block = self._delegation_blocks.get(job_id)
        if block is None:
            block = DelegationBlock(title="Delegation Details")
            self._delegation_blocks[job_id] = block
            await self._mount_delegation_widget(block)
        block.set_plan(markdown)

    def _store_pending_plan(self, objective: str, plan_text: str) -> None:
        """Store plan on agent for post-execution verification."""
        if self.agent:
            self.agent.pending_plan_objective = objective
            self.agent.pending_plan_text = plan_text

    def _mount_delegation_summary(self, job_id: str, summary: str) -> None:
        """AC4: Add completion summary to delegation block and collapse it."""
        self._safe_call(
            lambda: self.run_worker(self._do_mount_summary(job_id, summary))
        )

    async def _do_mount_summary(self, job_id: str, summary: str) -> None:
        block = self._delegation_blocks.get(job_id)
        if block is not None:
            block.set_result(summary)
            block.collapse_when_done()
        else:
            # Fallback: mount as a static widget if no block exists
            await self._mount_delegation_widget(
                Static(summary, classes="delegation-summary")
            )

    def _drop_delegation_progress(self, job_id: str) -> None:
        # ponytail: delegation blocks are cleaned up here on terminal state.
        # Block remains visible but collapsed until user scrolls past it.
        # Keep the block for history, but mark it as complete
        block = self._delegation_blocks.get(job_id)
        if block:
            block.collapse_when_done()


def main() -> None:
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
