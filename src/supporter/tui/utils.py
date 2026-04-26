from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from rich.text import Text

from ..config import CRYSTAL_GRADIENT_STOPS


def apply_crystal_gradient(text: str) -> Text:
    rich_text = Text(justify="center")
    char_count = len(text)
    num_stops = len(CRYSTAL_GRADIENT_STOPS) - 1

    for i, char in enumerate(text):
        progress = i / max(char_count - 1, 1)
        segment = min(int(progress * num_stops), num_stops - 1)
        local_progress = progress * num_stops - segment

        start_rgb = CRYSTAL_GRADIENT_STOPS[segment]
        end_rgb = CRYSTAL_GRADIENT_STOPS[segment + 1]

        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * local_progress)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * local_progress)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * local_progress)

        rich_text.append(char, style=f"bold rgb({r},{g},{b})")
    return rich_text


class ToastManager:
    def __init__(self, timeout: float = 5.0) -> None:
        self.active_toasts: OrderedDict[str, str] = OrderedDict()
        self.last_toast_time: float = 0
        self.timeout = timeout

    def notify(self, app: Any, message: str, type: str = "system") -> None:
        now = time.time()
        if now - self.last_toast_time > self.timeout:
            self.active_toasts.clear()
        if type in self.active_toasts:
            del self.active_toasts[type]
        self.active_toasts[type] = message
        self.active_toasts.move_to_end(type, last=False)
        self._clear_ui(app)
        self.last_toast_time = now
        content = "\n".join(self.active_toasts.values())
        app.notify(content, timeout=self.timeout)

    def clear(self, app: Any) -> None:
        self.active_toasts.clear()
        self._clear_ui(app)

    def _clear_ui(self, app: Any) -> None:
        if hasattr(app, "clear_notifications"):
            app.clear_notifications()
            return
        if hasattr(app, "screen") and app.screen:
            app.screen.query("Toast, Notification, .textual-notification").remove()
