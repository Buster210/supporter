from __future__ import annotations

import time
from collections import deque
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
    """Interactive TUI dashboard for autonomous agent with delegation monitoring.

    Combines live chat, prompt routing (direct/task), streaming responses,
    browser automation, code execution, and real-time delegation visibility.
    """

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
        self._is_streaming = False
        self._user_message_queue: list[tuple[str, bool]] = []
        self._toast_manager = ToastManager()
        self._delegation_blocks: dict[str, DelegationBlock] = {}
        self._pending_delegation_widgets: deque[tuple[Any, bool]] = deque()
        self._delegation_listener = DelegationListener(
            inject_message=self._inject_delegation_message,
            drop_progress=self._drop_delegation_progress,
            render_signal=self._render_delegation_signal_now,
            render_progress_live=self._render_delegation_progress_live,
            render_summary=self._mount_delegation_summary,
            plan_bubble_injector=self._inject_plan_bubble,
            plan_storer=self._store_pending_plan,
            render_task_done=self._mount_task_signal,
        )

    async def on_mode_changed(self, event: ModeChanged) -> None:
        """Handle agent mode toggle (live/offline). Display status bubble."""
        indicator = self.query_one("#mode-indicator", Label)
        indicator.update(f"[{event.mode}]")
        status = "ENABLED" if event.enabled else "DISABLED"

        target = self.active_turn or self.query_one("#chat-view")
        await target.mount(
            MessageBubble(role="agent", content=f"Single Agent {status}")
        )

    async def on_mount(self) -> None:
        """Initialize TUI on startup: register callbacks, warm pools, setup agent.

        Lazy imports defer subprocess/network initialization until app is visible.
        Spawns workers for agent setup, job resumption, and browser prewarm.
        """
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
            self.run_worker(
                resume_interrupted_jobs(), name="resume-jobs", group="resume-jobs"
            )
            self.run_worker(self._setup_agent(use_live=True), exclusive=True)
            greeting_worker = self.run_worker(
                self._mode_manager.trigger_live_greeting(), name="greeting"
            )
            self.run_worker(prewarm_clone())

            self.run_worker(
                self._startup_profile_select(greeting_worker),
                name="startup-profile-select",
            )
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Startup failure [{type(e).__name__}]: {e}"
            logger.error(msg)
            self._toast_manager.notify(self, msg, type="system")

    async def on_unmount(self) -> None:
        """Clean up on exit: cancel workers, deregister callbacks, close session."""
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

        for banner in self.query(WelcomeBanner):
            banner.message = ""

        current_turn: ChatTurn | None = None

        for msg in records:
            role: str = msg.role

            if role == "user":
                text = " ".join(
                    p.text for p in msg.parts if isinstance(p, TextPart) and p.text
                )
                if (
                    "DELEGATION_CAPSULE_RESULT (json)" in text
                    or "MILESTONE_RESULT (json)" in text
                ):
                    continue
                current_turn = ChatTurn(MessageBubble(role="user", content=text))
                await chat_view.mount(current_turn)

            elif role == "model":
                bubble = MessageBubble(role="agent", content="", streaming=False)
                for part in msg.parts:
                    if isinstance(part, TextPart) and part.text:
                        bubble.append_token(part.text)
                    elif isinstance(part, ToolCallPart):
                        bubble.add_tool_call(part.name, part.args)
                # Delegation-only turns persist an empty model message; mounting
                # it renders a bordered bubble with nothing inside (empty bar).
                if not bubble.elements:
                    continue
                bubble.finalize()
                if current_turn is not None:
                    await current_turn.mount_bubble(bubble)
                else:
                    await chat_view.mount(bubble)

        chat_view.jump_to_bottom()

    def compose(self) -> ComposeResult:
        """Compose the TUI layout: header, chat view, input area, mode indicator."""
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
        """Handle button press events (e.g., scroll-to-bottom button)."""
        if event.button.id == "scroll-bottom-btn":
            self.query_one("#chat-view", ChatContainer).jump_to_bottom()

    def action_scroll_chat(self, direction: str) -> None:
        """Scroll chat view in given direction. Non-animated for responsiveness."""
        chat_view = self.query_one("#chat-view", ChatContainer)
        if direction == "pageup":
            chat_view.scroll_page_up(animate=False)
        elif direction == "pagedown":
            chat_view.scroll_page_down(animate=False)
        elif direction == "home":
            chat_view.scroll_home(animate=False)
        elif direction == "end":
            chat_view.jump_to_bottom()

    def action_clear_screen(self) -> None:
        """Clear conversation history and remove all chat turns from the view."""
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
        """Increment active query counter, show thinking spinner."""
        self.active_queries += 1

    def stop_thinking(self) -> None:
        """Decrement active query counter, hide spinner when zero."""
        self.active_queries = max(0, self.active_queries - 1)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input: queue if busy, route command, or process message."""
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
        """Toggle live mode on/off via mode manager."""
        await self._mode_manager.toggle_mode(live=live)

    async def _process_message_cycle(
        self,
        text: str,
        mount_user: bool = True,
        target: Vertical | None = None,
        exclude_from_history: bool = False,
        role: str = "user",
        mount_text: str | None = None,
    ) -> None:
        self._is_processing = True
        chat_view = self.query_one("#chat-view", ChatContainer)
        if mount_user:
            shown = mount_text if mount_text is not None else text
            await self._mount_user_turn(shown, role=role)
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

            from ..config import config
            from ..replan import ReplanContext
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
                replan_ctx.next_cycle()

                while True:
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
                    last_failure_reason = (
                        replan_ctx.failures[-1] if replan_ctx.failures else "Unknown"
                    )
                    if not replan_ctx.next_cycle():
                        break

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
        if not self._user_message_queue:
            return

        self._is_processing = True
        items = list(self._user_message_queue)
        self._user_message_queue.clear()
        self.query_one("#queue-display", QueuedMessagesDisplay).update_queue([])

        combined_text = "\n\n".join(msg for msg, _ in items)
        user_text = "\n\n".join(msg for msg, is_sys in items if not is_sys)

        if user_text:
            self.run_worker(
                self._process_message_cycle(
                    combined_text, mount_user=True, mount_text=user_text
                )
            )
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
        """Verify result against plan; record failure for the next replan cycle.

        Returns True if verified. Actual re-prompting happens on the next
        cycle via the message loop — this method only records the failure reason.
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

            logger.info(
                f"TUI: planner says NOT done (cycle {replan_ctx.cycle}): {reason}"
            )
            replan_ctx.record_failure(reason)

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
            return True

    def _confirm_write(self, path: Path, content: str) -> bool:
        import threading

        event = threading.Event()
        result = [False]

        def callback(value: bool) -> None:
            """Store modal result and signal completion."""
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
            """Store modal result and signal completion."""
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

    async def _startup_profile_select(self, greeting_worker: Any) -> None:
        """Wait for greeting to finish, then prompt for profile selection."""
        from ..tools.browser.session import select_profile_at_startup

        await greeting_worker.wait()
        chosen = await select_profile_at_startup(self._select_profile)
        if chosen:
            self._toast_manager.notify(
                self, f"Browser profile: {chosen}", type="profile"
            )

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
        block = self._delegation_blocks.get(job_id)
        if block is not None:
            block.set_progress(progress_md)
            return

        block = DelegationBlock(title=f"Delegation [{job_id}]")
        block.set_progress(progress_md)
        self._delegation_blocks[job_id] = block
        await self._mount_delegation_widget(block)

    def _delegation_mount_target(self) -> Vertical:
        chat_view = self.query_one("#chat-view", ChatContainer)
        if self.active_turn is not None:
            return self.active_turn
        turns = list(chat_view.query(ChatTurn))
        return turns[-1] if turns else chat_view

    def _delegation_host_bubble(self) -> MessageBubble | None:
        turn = self.active_turn
        if turn is None:
            turns = list(self.query_one("#chat-view", ChatContainer).query(ChatTurn))
            turn = turns[-1] if turns else None
        bubbles = getattr(turn, "agent_bubbles", None)
        if bubbles:
            return bubbles[-1]
        return None

    async def _mount_delegation_widget(
        self, widget: Any, after_meta: bool = False
    ) -> None:
        """Mount widget into the current bubble. Queue only during active
        streaming so text appears first; mount directly otherwise."""
        if self._is_streaming:
            self._pending_delegation_widgets.append((widget, after_meta))
            return
        bubble = self._delegation_host_bubble()
        if bubble is not None:
            turn = self.active_turn or self._delegation_mount_target()
            start = getattr(turn, "turn_start_time", None)
            if start is not None:
                bubble._turn_start = start
            if after_meta:
                bubble.append_after_meta(widget)
            else:
                bubble.append_before_meta(widget)
        else:
            target = self._delegation_mount_target()
            await target.mount(widget)
        self.query_one("#chat-view", ChatContainer).follow_end()

    async def _flush_pending_delegation_widgets(self, bubble: Any) -> None:
        turn = self.active_turn or self._delegation_mount_target()
        start = getattr(turn, "turn_start_time", None)
        if start is not None:
            bubble._turn_start = start
        while self._pending_delegation_widgets:
            widget, after_meta = self._pending_delegation_widgets.popleft()
            if after_meta:
                bubble.append_after_meta(widget)
            else:
                bubble.append_before_meta(widget)
        self.query_one("#chat-view", ChatContainer).follow_end()

    async def _render_delegation_signal(self, text: str) -> None:
        body = text.replace("<br/>", "").replace("`", "").strip()
        await self._mount_delegation_widget(
            Static(body, classes="delegation-signal"), after_meta=True
        )

    async def _process_system_message(self, text: str) -> None:
        await self._process_message_cycle(text, mount_user=False, role="agent")

    def _start_delegation_listener(self, job_id: str, plan_table: str = "") -> None:
        if self.active_turn:
            self.active_turn._delegation_job_id = job_id
        bubble = self._delegation_host_bubble()
        if bubble is not None:
            turn = self.active_turn or self._delegation_mount_target()
            start = getattr(turn, "turn_start_time", None)
            if start is not None:
                bubble._turn_start = start
        self.run_worker(self._delegation_listener.listen(job_id), exclusive=False)

    def _inject_delegation_message(self, message: str) -> None:
        self._safe_call(self._inject_system_message, message)

    def _inject_plan_bubble(self, markdown: str) -> None:
        """Update delegation block with plan, or mount if not yet created."""
        self._safe_call(lambda: self.run_worker(self._mount_plan_bubble(markdown)))

    async def _mount_plan_bubble(self, markdown: str) -> None:
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

    def _mount_task_signal(self, job_id: str, text: str) -> None:
        """Add a task-complete signal into the job's delegation block."""
        self._safe_call(
            lambda: self.run_worker(self._do_mount_task_signal(job_id, text))
        )

    async def _do_mount_task_signal(self, job_id: str, text: str) -> None:
        body = text.replace("<br/>", "").replace("`", "").strip()
        block = self._delegation_blocks.get(job_id)
        if block is not None:
            block.set_signal(body)
        else:
            await self._mount_delegation_widget(
                Static(body, classes="delegation-signal"), after_meta=True
            )

    def _mount_delegation_summary(self, job_id: str, summary: str) -> None:
        """Add completion summary to delegation block and collapse it."""
        self._safe_call(
            lambda: self.run_worker(self._do_mount_summary(job_id, summary))
        )

    async def _do_mount_summary(self, job_id: str, summary: str) -> None:
        block = self._delegation_blocks.get(job_id)
        if block is not None:
            block.set_result(summary)
            block.collapse_when_done()
        else:
            await self._mount_delegation_widget(
                Static(summary, classes="delegation-summary"), after_meta=True
            )
        bubble = self._delegation_host_bubble()
        if bubble is not None:
            bubble.reveal_meta()

    def _drop_delegation_progress(self, job_id: str) -> None:
        block = self._delegation_blocks.get(job_id)
        if block:
            block.collapse_when_done()


def main() -> None:
    """Launch the Supporter TUI dashboard."""
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
