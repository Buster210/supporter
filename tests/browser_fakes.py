from __future__ import annotations

from typing import Any

_SAMPLE_ARIA = '- document [ref=e1]:\n  - button "OK" [ref=e2]'


class CallLog:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str, tuple[Any, ...], dict[str, Any]]] = []

    def add(
        self, target: str, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self.entries.append((target, method, args, kwargs))

    def methods(self, target: str | None = None) -> list[str]:
        return [m for t, m, _a, _k in self.entries if target is None or t == target]

    def last(self, method: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        for _t, m, a, k in reversed(self.entries):
            if m == method:
                return a, k
        raise AssertionError(f"no recorded call to {method!r} in {self.methods()}")

    def count(self, method: str) -> int:
        return sum(1 for _t, m, _a, _k in self.entries if m == method)


class FakeMouse:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    async def wheel(self, dx: float, dy: float) -> None:
        self._log.add("mouse", "wheel", (dx, dy), {})

    async def move(self, x: float, y: float, **kwargs: Any) -> None:
        self._log.add("mouse", "move", (x, y), kwargs)

    async def click(self, x: float, y: float, **kwargs: Any) -> None:
        self._log.add("mouse", "click", (x, y), kwargs)

    async def down(self, **kwargs: Any) -> None:
        self._log.add("mouse", "down", (), kwargs)

    async def up(self, **kwargs: Any) -> None:
        self._log.add("mouse", "up", (), kwargs)


class FakeKeyboard:
    def __init__(self, log: CallLog) -> None:
        self._log = log

    async def press(self, key: str, **kwargs: Any) -> None:
        self._log.add("keyboard", "press", (key,), kwargs)

    async def type(self, text: str, **kwargs: Any) -> None:
        self._log.add("keyboard", "type", (text,), kwargs)

    async def down(self, key: str, **kwargs: Any) -> None:
        self._log.add("keyboard", "down", (key,), kwargs)

    async def up(self, key: str, **kwargs: Any) -> None:
        self._log.add("keyboard", "up", (key,), kwargs)


class FakeDownload:
    def __init__(self, log: CallLog, suggested_filename: str = "file.bin") -> None:
        self._log = log
        self.suggested_filename = suggested_filename

    async def save_as(self, path: str) -> None:
        self._log.add("download", "save_as", (path,), {})


class _ExpectDownload:
    def __init__(self, log: CallLog, download: FakeDownload) -> None:
        self._log = log
        self._download = download

    async def __aenter__(self) -> _ExpectDownload:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    @property
    async def value(self) -> FakeDownload:
        return self._download


class FakeLocator:
    def __init__(self, log: CallLog, selector: str = "") -> None:
        self._log = log
        self.selector = selector
        self.role_name: list[str] = ["", ""]
        self.aria_text = _SAMPLE_ARIA
        self.html = "<b>hi</b>"
        self.text = "visible text"
        self.box: dict[str, float] | None = {
            "x": 10,
            "y": 20,
            "width": 40,
            "height": 12,
        }

    def _rec(self, method: str, *args: Any, **kwargs: Any) -> None:
        self._log.add("locator", method, args, kwargs)

    @property
    def first(self) -> FakeLocator:
        return self

    def locator(self, selector: str) -> FakeLocator:
        self._rec("locator", selector)
        return self

    async def wait_for(self, **kwargs: Any) -> None:
        self._rec("wait_for", **kwargs)

    async def click(self, **kwargs: Any) -> None:
        self._rec("click", **kwargs)

    async def fill(self, text: str, **kwargs: Any) -> None:
        self._rec("fill", text, **kwargs)

    async def hover(self, **kwargs: Any) -> None:
        self._rec("hover", **kwargs)

    async def focus(self, **kwargs: Any) -> None:
        self._rec("focus", **kwargs)

    async def scroll_into_view_if_needed(self, **kwargs: Any) -> None:
        self._rec("scroll_into_view_if_needed", **kwargs)

    async def select_option(self, **kwargs: Any) -> None:
        self._rec("select_option", **kwargs)

    async def set_input_files(self, files: str, **kwargs: Any) -> None:
        self._rec("set_input_files", files, **kwargs)

    async def inner_html(self, **kwargs: Any) -> str:
        self._rec("inner_html", **kwargs)
        return self.html

    async def inner_text(self, **kwargs: Any) -> str:
        self._rec("inner_text", **kwargs)
        return self.text

    async def aria_snapshot(self, **kwargs: Any) -> str:
        self._rec("aria_snapshot", **kwargs)
        return self.aria_text

    async def bounding_box(self, **kwargs: Any) -> dict[str, float] | None:
        self._rec("bounding_box", **kwargs)
        return self.box

    async def evaluate(self, expression: str, *args: Any, **kwargs: Any) -> Any:
        self._rec("evaluate", expression, *args)
        return self.role_name


class FakePage:
    def __init__(
        self, log: CallLog | None = None, url: str = "https://example.test/"
    ) -> None:
        self.log = log if log is not None else CallLog()
        self.url = url
        self.viewport_size: dict[str, int] | None = {"width": 1280, "height": 800}
        self.aria_text = _SAMPLE_ARIA
        self.title_text = "Example"
        self.screenshot_bytes = b"\x89PNG\r\n\x1a\nFAKE"
        self.eval_result: Any = None
        self.context: FakeContext | None = None
        self.mouse = FakeMouse(self.log)
        self.keyboard = FakeKeyboard(self.log)
        self.locators: list[FakeLocator] = []
        self.download = FakeDownload(self.log)

    def _rec(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.log.add("page", method, args, kwargs)

    def _new_locator(self, selector: str) -> FakeLocator:
        loc = FakeLocator(self.log, selector)
        self.locators.append(loc)
        return loc

    def locator(self, selector: str) -> FakeLocator:
        self._rec("locator", selector)
        return self._new_locator(selector)

    def frame_locator(self, selector: str) -> FakeLocator:
        self._rec("frame_locator", selector)
        return self._new_locator(selector)

    async def goto(self, url: str, **kwargs: Any) -> None:
        self._rec("goto", url, **kwargs)
        self.url = url

    async def go_back(self, **kwargs: Any) -> None:
        self._rec("go_back", **kwargs)

    async def go_forward(self, **kwargs: Any) -> None:
        self._rec("go_forward", **kwargs)

    async def wait_for_selector(self, selector: str, **kwargs: Any) -> None:
        self._rec("wait_for_selector", selector, **kwargs)

    async def wait_for_timeout(self, ms: float) -> None:
        self._rec("wait_for_timeout", ms)

    async def wait_for_load_state(self, state: str = "load", **kwargs: Any) -> None:
        self._rec("wait_for_load_state", state, **kwargs)

    async def title(self) -> str:
        self._rec("title")
        return self.title_text

    async def bring_to_front(self) -> None:
        self._rec("bring_to_front")

    async def close(self, **kwargs: Any) -> None:
        self._rec("close", **kwargs)
        if self.context is not None:
            self.context._drop(self)

    async def screenshot(self, **kwargs: Any) -> bytes:
        self._rec("screenshot", **kwargs)
        return self.screenshot_bytes

    async def aria_snapshot(self, **kwargs: Any) -> str:
        self._rec("aria_snapshot", **kwargs)
        return self.aria_text

    async def evaluate(self, expression: str, *args: Any, **kwargs: Any) -> Any:
        self._rec("evaluate", expression, *args)
        return self.eval_result

    def expect_download(self, **kwargs: Any) -> _ExpectDownload:
        self._rec("expect_download", **kwargs)
        return _ExpectDownload(self.log, self.download)


class FakeContext:
    def __init__(self, log: CallLog | None = None) -> None:
        self.log = log if log is not None else CallLog()
        self.pages: list[FakePage] = []
        self.cookie_list: list[dict[str, Any]] = []

    def _rec(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.log.add("context", method, args, kwargs)

    def add_page(self, page: FakePage) -> FakePage:
        page.context = self
        page.log = self.log
        page.mouse = FakeMouse(self.log)
        page.keyboard = FakeKeyboard(self.log)
        self.pages.append(page)
        return page

    def _drop(self, page: FakePage) -> None:
        if page in self.pages:
            self.pages.remove(page)

    async def new_page(self, **kwargs: Any) -> FakePage:
        self._rec("new_page", **kwargs)
        return self.add_page(FakePage(self.log))

    async def cookies(self, **kwargs: Any) -> list[dict[str, Any]]:
        self._rec("cookies", **kwargs)
        return list(self.cookie_list)


def make_session(
    url: str = "https://example.test/",
) -> tuple[CallLog, FakeContext, FakePage]:
    log = CallLog()
    context = FakeContext(log)
    page = context.add_page(FakePage(log, url=url))
    return log, context, page
