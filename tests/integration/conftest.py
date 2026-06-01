from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from supporter.tools.browser import guardrails, humanize, session
from tests.browser_fakes import FakeContext, FakePage, make_session

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class ConfirmRecorder:
    allow: bool = True
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def __call__(self, summary: str, prompt: str) -> bool:
        self.calls.append((summary, prompt))
        return self.allow


@dataclass
class ImageSinkRecorder:
    images: list[tuple[bytes, str]] = field(default_factory=list)

    async def __call__(self, data: bytes, caption: str) -> None:
        self.images.append((data, caption))


@dataclass
class FakeSession:
    context: FakeContext
    page: FakePage
    confirm: ConfirmRecorder
    image_sink: ImageSinkRecorder

    @property
    def log(self) -> Any:
        return self.page.log


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeSession]:
    _log, context, page = make_session()
    confirm = ConfirmRecorder()
    image_sink = ImageSinkRecorder()
    active: list[FakePage] = [page]

    async def fake_get_session() -> tuple[Any, Any, Any]:
        return None, context, active[0]

    monkeypatch.setattr(session, "get_session", fake_get_session)
    monkeypatch.setattr(session, "is_active", lambda: True)
    monkeypatch.setattr(session, "active_page", lambda: active[0])
    monkeypatch.setattr(session, "list_pages", lambda: list(context.pages))
    monkeypatch.setattr(session, "pinned_open", lambda: False)
    monkeypatch.setattr(session, "keep_open", lambda: True)

    def fake_set_active(target: Any) -> None:
        active[0] = target

    monkeypatch.setattr(session, "set_active", fake_set_active)

    frame_box: list[str | None] = [None]
    monkeypatch.setattr(session, "active_frame_selector", lambda: frame_box[0])
    monkeypatch.setattr(session, "set_frame", lambda sel: frame_box.__setitem__(0, sel))

    monkeypatch.setattr(guardrails, "browse_confirmation_callback", confirm)
    monkeypatch.setattr(guardrails, "browse_image_sink", image_sink)

    async def no_pace() -> None:
        return None

    monkeypatch.setattr(session, "pace", no_pace)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(humanize.asyncio, "sleep", no_sleep)  # type: ignore[attr-defined]

    yield FakeSession(
        context=context, page=page, confirm=confirm, image_sink=image_sink
    )
