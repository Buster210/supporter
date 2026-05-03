from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Input, Label

from .. import ChatAgent, DynamicPool
from ..logger import init_logger, logger
from ..types import ModeChanged

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.widgets import Button, Input

    from .. import ChatAgent
    from .chat import ChatTurn

CSS = (Path(__file__).parent / "styles.tcss").read_text()


class SupporterApp(App[None]):
    CSS = CSS

    status_label = reactive("Thinking")
    active_queries = reactive(0)
    is_activating_mode = reactive(False)
    live_mode = reactive(True)
    active_turn: reactive[ChatTurn | None] = reactive(None)

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("ctrl+l", "clear_screen", "Clear", show=True),
    ]

    def __init__(self) -> None:
        super().__init__()
        from .message_processor import ChatMessageProcessor
        from .mode_manager import ModeManager
        from .utils import ToastManager

        self.agent: ChatAgent | None = None
        self._mode_manager = ModeManager(self)
        self._message_processor = ChatMessageProcessor(self)
        self._is_processing = False
        self._user_message_queue: list[tuple[str, bool]] = []
        self._toast_manager = ToastManager()
        self._delegation_bubbles: dict[str, Any] = {}

    async def on_mode_changed(self, event: ModeChanged) -> None:
        indicator = self.query_one("#mode-indicator", Label)
        indicator.update(f"[{event.mode}]")
        status = "ENABLED" if event.enabled else "DISABLED"

        target = self.active_turn or self.query_one("#chat-view")
        from .bubble import MessageBubble

        await target.mount(
            MessageBubble(role="agent", content=f"Single Agent {status}")
        )

    async def on_mount(self) -> None:
        from ..tools import (
            set_bash_confirmation_callback,
            set_bash_notification_callback,
            set_confirmation_callback,
        )

        init_logger()
        set_confirmation_callback(self._confirm_write)
        set_bash_confirmation_callback(self._confirm_bash)
        set_bash_notification_callback(self._notify_error)

        from ..tools.delegate import set_delegation_start_callback

        set_delegation_start_callback(self._start_delegation_listener)

        logger.info("Supporter TUI dashboard active")
        self._mode_manager.start_warmup()
        try:
            self.run_worker(self._setup_agent(use_live=True), exclusive=True)
            self.run_worker(self._mode_manager.trigger_live_greeting())
            self.query_one("#user-input").focus()
        except Exception as e:
            msg = f"Startup failure [{type(e).__name__}]: {e}"
            logger.error(msg)
            self._toast_manager.notify(self, msg, type="system")

    async def on_unmount(self) -> None:
        from ..tools import (
            set_bash_confirmation_callback,
            set_bash_notification_callback,
            set_confirmation_callback,
        )

        set_confirmation_callback(None)
        set_bash_confirmation_callback(None)
        set_bash_notification_callback(None)

        from ..tools.delegate import set_delegation_start_callback

        set_delegation_start_callback(None)

        if (
            self.agent
            and hasattr(self.agent, "provider")
            and hasattr(self.agent.provider, "close")
        ):
            await self.agent.provider.close()

        await DynamicPool.shutdown_all()
        await self._mode_manager.close()

    async def _setup_agent(self, use_live: bool = False) -> None:
        await self._mode_manager.setup_agent(use_live=use_live)

    def compose(self) -> ComposeResult:
        from .chat import (
            ChatContainer,
            QueuedMessagesDisplay,
            SupporterHeader,
            ThinkingIndicator,
            WelcomeBanner,
        )

        with Vertical(id="main-container"):
            yield SupporterHeader(id="supporter-header")
            yield WelcomeBanner(id="welcome-banner", classes="hidden")
            with ChatContainer(id="chat-view"):
                pass
            yield QueuedMessagesDisplay(id="queue-display")
            yield ThinkingIndicator(id="thinking-indicator")

            with Horizontal(id="scroll-btn-wrapper", classes="hidden"):
                yield Button("↓ Go to bottom", id="scroll-bottom-btn")
            with Vertical(id="input-area"), Horizontal(id="prompt-row"):
                yield Label("[LIVE]", id="mode-indicator", markup=False)
                yield Label(">", id="prompt-symbol")
                yield Input(
                    placeholder="Type a message... (/agent, /live, /clear, /exit)",
                    id="user-input",
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scroll-bottom-btn":
            chat_view = self.query_one("#chat-view", Vertical)
            chat_view.scroll_end(animate=True)
            event.button.add_class("hidden")

    def action_clear_screen(self) -> None:
        chat_view = self.query_one("#chat-view")
        if not chat_view.query("*") and (not self.agent or not self.agent.history):
            self._toast_manager.notify(self, "Session already clear", type="system")
            return

        from .chat import QueuedMessagesDisplay

        if self.agent:
            self.agent.clear_history()
        chat_view.query("*").remove()
        self._user_message_queue.clear()
        self.query_one("#queue-display", QueuedMessagesDisplay).update_queue([])

    def _start_thinking(self) -> None:
        self.active_queries += 1

    def _stop_thinking(self) -> None:
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
            from .chat import QueuedMessagesDisplay

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

    async def _mount_user_turn(self, text: str) -> ChatTurn:
        from .bubble import MessageBubble
        from .chat import ChatTurn

        chat_view = self.query_one("#chat-view", Vertical)
        new_turn = ChatTurn(MessageBubble(role="user", content=text))
        if self.active_turn:
            self.active_turn.auto_collapse()
        self.active_turn = new_turn
        await chat_view.mount(new_turn)
        chat_view.scroll_end()
        return new_turn

    async def _handle_command(self, command: str) -> bool:
        return await self._mode_manager.handle_command(command)

    async def _toggle_mode(self, live: bool = False) -> None:
        await self._mode_manager.toggle_mode(live=live)

    async def _process_message_cycle(
        self,
        text: str,
        mount_user: bool = True,
        target: Vertical | None = None,
        exclude_from_history: bool = False,
    ) -> None:
        self._is_processing = True
        chat_view = self.query_one("#chat-view", Vertical)
        if mount_user:
            await self._mount_user_turn(text)
            self.query_one("#user-input").focus()

        target_container = target or (
            self.active_turn if hasattr(self, "active_turn") else chat_view
        )
        if not isinstance(target_container, Vertical):
            raise RuntimeError("Invalid UI state: chat view container missing")

        self.status_label = "Thinking"
        self._start_thinking()
        start_time = getattr(target_container, "turn_start_time", time.perf_counter())

        try:
            if not self.agent:
                raise RuntimeError("Agent is not initialized")
            await self._process_streaming_execution(
                text, target_container, start_time, self.agent, exclude_from_history
            )
        except Exception as e:
            from .bubble import MessageBubble

            logger.error(f"UI Message Cycle Error [{type(e).__name__}]: {e}")
            await chat_view.mount(MessageBubble(role="agent", content=f"Error: {e}"))
        finally:
            self._is_processing = False
            self._stop_thinking()
            await self._flush_queued_messages()

    async def _flush_queued_messages(self) -> None:
        from .chat import QueuedMessagesDisplay

        if not self._user_message_queue:
            return

        items = list(self._user_message_queue)
        self._user_message_queue.clear()
        self.query_one("#queue-display", QueuedMessagesDisplay).update_queue([])

        combined_text = "\n\n".join(msg for msg, _ in items)
        has_user_message = any(not is_sys for _, is_sys in items)
        self._is_processing = True

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

    def _confirm_write(self, path: Path, content: str) -> bool:
        import threading

        from .modals import ConfirmationModal

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

        from .modals import ConfirmationModal

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
            from .chat import QueuedMessagesDisplay

            self._user_message_queue.append((text, True))
            self.query_one("#queue-display", QueuedMessagesDisplay).update_queue(
                [msg for msg, _ in self._user_message_queue]
            )
        else:
            self.run_worker(self._process_system_message(text))

    async def _process_system_message(self, text: str) -> None:
        from .bubble import MessageBubble

        if self.active_turn:
            if (
                "DELEGATION_CAPSULE_RESULT (json)" in text
                or "MILESTONE_RESULT (json)" in text
            ):
                return
            await self.active_turn.mount_bubble(
                MessageBubble(role="user", content=text)
            )
            await self._process_message_cycle(text, mount_user=False)
        else:
            await self._process_message_cycle(text, mount_user=True)

    def _start_delegation_listener(self, job_id: str) -> None:
        self.run_worker(self._delegation_listener(job_id), exclusive=False)

    @staticmethod
    def _summarize_output(text: str, limit: int = 600) -> str:
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n\n[... truncated ...]"

    @staticmethod
    def _format_completed_task_signal(job_id: str, task_id: str) -> str:
        return (
            "<br/>\n\n"
            f"Delegation task completed — job_id: `{job_id}` | task_id: `{task_id}`\n"
            "\n<br/>"
        )

    @classmethod
    def _format_delegation_progress(cls, job_id: str, bus: Any) -> str:
        rows = []
        for task_id, state in bus.get_snapshot().items():
            status = cls._display_task_status(str(state.get("status", "PENDING")))
            agent = str(state.get("agent_label", "?"))
            duration = state.get("duration")
            duration_text = ""
            if isinstance(duration, int | float) and duration > 0:
                duration_text = f"{duration:.2f}s"
            rows.append(
                "| "
                + " | ".join(
                    [
                        task_id,
                        agent,
                        status,
                        duration_text,
                    ]
                )
                + " |"
            )

        table = [
            f"Job `{job_id}`",
            "",
            "| Task | Agent | Status | Time |",
            "| ---- | ----- | ------ | ---- |",
            *rows,
        ]
        return "\n".join(table)

    @staticmethod
    def _display_task_status(status: str) -> str:
        normalized = status.lower()
        if normalized == "running":
            return "working"
        if normalized == "done":
            return "completed"
        return normalized

    @staticmethod
    def _format_task_signal(job_id: str, kind: str, task_id: str, bus: Any) -> str:
        state = bus.get_snapshot().get(task_id, {})
        payload = {
            "job_id": job_id,
            "task_id": task_id,
            "agent": str(state.get("agent_label", "?")),
            "assigned_task": str(state.get("task_goal", "")).strip(),
        }
        status = kind.upper()
        return f"DELEGATION_TASK_{status}: {json.dumps(payload, ensure_ascii=False)}"

    async def _upsert_delegation_progress(self, job_id: str, bus: Any) -> None:
        from .bubble import MessageBubble

        content = self._format_delegation_progress(job_id, bus)
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
        chat_view = self.query_one("#chat-view", Vertical)
        target = self.active_turn or chat_view
        if hasattr(target, "mount_bubble"):
            await target.mount_bubble(bubble)
        else:
            await target.mount(bubble)
        chat_view.scroll_end()

    async def _emit_task_event(
        self,
        bus: Any,
        job_id: str,
        kind: str,
        task_id: str,
        duration: float | None,
        body: str,
        sys_extra: str = "",
        sys_body: str | None = None,
    ) -> None:
        _ = duration, body
        await self._upsert_delegation_progress(job_id=job_id, bus=bus)
        if bus.notify_per_task:
            if sys_body:
                message = sys_body
            else:
                message = self._format_task_signal(job_id, kind, task_id, bus)
                if sys_extra:
                    message += sys_extra
            self._safe_call(
                self._inject_system_message,
                message,
            )

    async def _delegation_listener(self, job_id: str) -> None:
        import json

        from ..tools.delegate import get_bus, serialize_results
        from ..tools.delegation_capsule import (
            query_delegation,
            serialize_capsule,
        )
        from ..types import (
            MilestoneCancelled,
            MilestoneCompleted,
            MilestoneStarted,
            TaskAnomaly,
            TaskCompleted,
            TaskFailed,
            TaskSkipped,
            TaskStarted,
            TaskTimedOut,
        )

        try:
            bus = get_bus(job_id)
            queue = bus.subscribe()
            while True:
                event = await queue.get()
                if event is None:
                    break

                if isinstance(event, (MilestoneStarted, TaskStarted)):
                    pass

                elif isinstance(event, TaskAnomaly):
                    msg = (
                        f"AGENT ALERT: Task `{event.task_id}` [{event.agent_label}] "
                        f"has used {event.elapsed_seconds:.0f}s of its "
                        f"{event.timeout:.0f}s limit and may be hung."
                    )
                    self._safe_call(self._inject_system_message, msg)

                elif isinstance(event, TaskCompleted):
                    sys_body = self._format_completed_task_signal(
                        job_id=job_id, task_id=event.task_id
                    )
                    await self._emit_task_event(
                        bus,
                        job_id,
                        "DONE",
                        event.task_id,
                        event.duration,
                        "",
                        sys_body=sys_body,
                    )

                elif isinstance(event, TaskFailed):
                    inspect_hint = (
                        f'\n\nMore: query_delegation(job_id="{job_id}", '
                        f'task_id="{event.task_id}")'
                    )
                    await self._emit_task_event(
                        bus,
                        job_id,
                        "FAIL",
                        event.task_id,
                        event.duration,
                        self._summarize_output(event.error, limit=400),
                        sys_extra=inspect_hint,
                    )

                elif isinstance(event, TaskTimedOut):
                    inspect_hint = (
                        f'\n\nMore: query_delegation(job_id="{job_id}", '
                        f'task_id="{event.task_id}")'
                    )
                    await self._emit_task_event(
                        bus,
                        job_id,
                        "TIMEOUT",
                        event.task_id,
                        event.duration,
                        "Execution timed out before completion.",
                        sys_extra=inspect_hint,
                    )

                elif isinstance(event, TaskSkipped):
                    inspect_hint = (
                        f'\n\nMore: query_delegation(job_id="{job_id}", '
                        f'task_id="{event.task_id}")'
                    )
                    await self._emit_task_event(
                        bus,
                        job_id,
                        "SKIP",
                        event.task_id,
                        None,
                        event.reason,
                        sys_extra=inspect_hint,
                    )

                elif isinstance(event, MilestoneCompleted):
                    try:
                        data = await query_delegation(job_id)
                        if not data:
                            raise ValueError(f"Capsule not found for {job_id}")
                        payload = serialize_capsule(data)
                    except Exception as e:
                        logger.warning(
                            "Falling back to legacy delegation result for "
                            f"{job_id} [{type(e).__name__}]: {e}"
                        )
                        has_failures = any(
                            str(result.get("status")) in {"error", "timeout", "skipped"}
                            for result in event.results
                        )
                        payload = serialize_results(
                            event.milestone,
                            event.results,
                            event.total_duration,
                            job_id,
                            status="completed_with_failures"
                            if has_failures
                            else "completed",
                        )
                    msg = (
                        f"DELEGATION_CAPSULE_RESULT (json):\n```json\n"
                        f"{json.dumps(payload, indent=2)}\n```"
                    )
                    self._safe_call(self._inject_system_message, msg)
                    break

                elif isinstance(event, MilestoneCancelled):
                    try:
                        data = await query_delegation(job_id)
                        if not data:
                            raise ValueError(f"Capsule not found for {job_id}")
                        payload = serialize_capsule(data)
                    except Exception as e:
                        logger.warning(
                            "Falling back to legacy cancellation result for "
                            f"{job_id} [{type(e).__name__}]: {e}"
                        )
                        payload = {
                            "job_id": job_id,
                            "milestone": event.milestone,
                            "status": "cancelled",
                            "total_duration": round(event.total_duration, 2),
                        }
                    msg = (
                        f"DELEGATION_CAPSULE_RESULT (json):\n```json\n"
                        f"{json.dumps(payload, indent=2)}\n```"
                    )
                    self._safe_call(self._inject_system_message, msg)
                    break

        except Exception as e:
            logger.error(f"Delegation listener failed for {job_id}: {e}")


def main() -> None:
    app = SupporterApp()
    app.run()


if __name__ == "__main__":
    main()
