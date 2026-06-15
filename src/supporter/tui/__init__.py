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
from .delegation_listener import DelegationListener, format_delegation_progress
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


_TRIVIAL_RESPONSES = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "cool",
        "great",
        "awesome",
        "nice",
        "got it",
        "understood",
        "perfect",
        "sounds good",
        "will do",
        "on it",
        "yep",
        "nah",
        "nope",
        "gotcha",
        "roger",
        "ack",
        "k",
        "👍",
    }
)


def _is_substantive_task(text: str) -> bool:
    """Return True when *text* is a real task that benefits from planning."""
    stripped = text.strip()
    if not stripped:
        return False
    # System/delegation re-injections — never plan on these
    if "DELEGATION_CAPSULE_RESULT" in stripped or "MILESTONE_RESULT" in stripped:
        return False
    # Slash commands — handled separately, not tasks
    if stripped.startswith("/"):
        return False
    lower = stripped.lower().rstrip(".!?\n")
    if lower in _TRIVIAL_RESPONSES:
        return False
    # Very short non-actionable fragments
    return len(stripped) >= 5


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
        self._delegation_bubbles: dict[str, Any] = {}
        self._delegation_listener = DelegationListener(
            inject_message=self._inject_delegation_message,
            upsert_progress=self._upsert_delegation_progress,
            drop_progress=self._drop_delegation_progress,
            render_signal=self._render_delegation_signal_now,
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
        # Tool/pool imports below are intentionally lazy: each pulls in modules
        # that spawn subprocesses, open OS handles, or touch the network at
        # import time (e.g. browser prewarm). Keeping them out of module top
        # defers that work until the app is actually starting up.
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
            # Auto-resume interrupted milestones on startup. Own group so the
            # exclusive _setup_agent worker (default group) does not cancel it.
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

    def compose(self) -> ComposeResult:
        with Vertical(id="main-container"):
            yield SupporterHeader(id="supporter-header")
            with ChatContainer(id="chat-view"):
                # Inside the scroll view so the greeting scrolls away with the
                # chat history instead of staying pinned above it.
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
        # animate=False: Textual's scroll_* default to a smooth animation, which
        # is a burst of full-viewport repaints per keypress. On a real terminal
        # that can't absorb the writes fast enough they back up and stall the
        # event loop (frozen spinner, "stuck" scroll); an instant jump is one
        # repaint. Headless never feels this, so it stayed hidden.
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
                [msg for msg, _ in self._user_message_queue]
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
        chat_view = self.query_one("#chat-view", Vertical)
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

            # Feature A: plan-before-act for substantive user tasks
            exec_text = text
            if (
                role == "user"
                and getattr(config, "plan_before_act", False)
                and _is_substantive_task(text)
            ):
                exec_text = await self._run_planner_and_inject(text, chat_view)

            await self._process_streaming_execution(
                exec_text,
                target_container,
                start_time,
                self.agent,
                exclude_from_history,
            )
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
    ) -> None:
        await self._message_processor.process_streaming(
            text, target, start_time, agent, exclude_from_history
        )

    async def _run_planner_and_inject(self, text: str, chat_view: ChatContainer) -> str:
        """Run the orchestration planner and prepend the plan to *text*.

        On failure or timeout, returns *text* unchanged — planner failure
        MUST NOT block the task.
        """
        import asyncio as _asyncio

        from ..prompts import DELEGATE_AGENT_ROSTER, ORCHESTRATION_PLANNER_PERSONA
        from ..worker import make_plan

        profile = DELEGATE_AGENT_ROSTER.get("planner", {})
        model = profile.get("model", "gemma-4-31b-it")
        try:
            plan = await _asyncio.wait_for(
                make_plan(text, ORCHESTRATION_PLANNER_PERSONA, model),
                timeout=15.0,
            )
        except TimeoutError:
            logger.warning("planner timed out — proceeding without plan")
            return text
        except Exception as exc:
            logger.warning(f"planner error: {exc} — proceeding without plan")
            return text

        if not plan:
            return text

        # Show the plan to the user as a system message
        await chat_view.mount(
            MessageBubble(
                role="system",
                content=f"PLAN (from planner — route accordingly):\n{plan}",
            )
        )
        chat_view.jump_to_bottom()

        # Prepend the plan to the user's text for the orchestrator
        return (
            f"PLAN (from planner — route accordingly):\n"
            f"```\n{plan}\n```\n\n"
            f"USER REQUEST:\n{text}"
        )

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

    def _inject_system_message(self, text: str) -> None:
        if self._is_processing:
            self._user_message_queue.append((text, True))
            self.query_one("#queue-display", QueuedMessagesDisplay).update_queue(
                [msg for msg, _ in self._user_message_queue]
            )
        else:
            self.run_worker(self._process_system_message(text))

    def _render_delegation_signal_now(self, text: str) -> None:
        self._safe_call(lambda: self.run_worker(self._render_delegation_signal(text)))

    async def _render_delegation_signal(self, text: str) -> None:
        body = text.replace("<br/>", "").replace("`", "").strip()
        label = Static(body, classes="delegation-signal")
        chat_view = self.query_one("#chat-view", ChatContainer)
        target = self.active_turn or chat_view
        await target.mount(label)
        chat_view.follow_end()

    async def _process_system_message(self, text: str) -> None:
        # The aggregate delegation result must reach the model, but its raw JSON
        # is never shown as a chat bubble — the model's reply renders instead.
        suppress_bubble = (
            "DELEGATION_CAPSULE_RESULT (json)" in text
            or "MILESTONE_RESULT (json)" in text
        )
        if self.active_turn:
            await self._process_message_cycle(text, mount_user=False)
        elif suppress_bubble:
            await self._process_message_cycle(text, mount_user=False, role="agent")
        else:
            await self._process_message_cycle(text, mount_user=True, role="agent")

    def _start_delegation_listener(self, job_id: str) -> None:
        self.run_worker(self._delegation_listener.listen(job_id), exclusive=False)

    async def _upsert_delegation_progress(self, job_id: str, bus: Any) -> None:
        content = format_delegation_progress(job_id, bus)
        bubble = self._delegation_bubbles.get(job_id)
        if bubble is not None:
            bubble.elements[0]["content"] = content
            bubble._update_ui_content()
            return

        bubble = MessageBubble(role="agent", content="")
        bubble.add_class("delegation-progress")
        bubble.elements = [
            {
                "type": "subagent_result",
                "content": content,
                "collapsed": True,
                "manually_interacted": False,
            }
        ]
        bubble.collapsed = False
        self._delegation_bubbles[job_id] = bubble
        chat_view = self.query_one("#chat-view", ChatContainer)
        target = self.active_turn or chat_view
        if hasattr(target, "mount_bubble"):
            await target.mount_bubble(bubble)
        else:
            await target.mount(bubble)
        chat_view.follow_end()

    def _inject_delegation_message(self, message: str) -> None:
        self._safe_call(self._inject_system_message, message)

    def _drop_delegation_progress(self, job_id: str) -> None:
        self._delegation_bubbles.pop(job_id, None)


def main() -> None:
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
