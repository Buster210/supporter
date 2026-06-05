from __future__ import annotations

from supporter.tools.browser.tool import browse

from .conftest import FakeSession


async def test_navigate_calls_goto_and_returns_snapshot(
    fake_session: FakeSession,
) -> None:
    result = await browse("navigate", url="https://nav.test/page")

    args, kwargs = fake_session.log.last("goto")
    assert args == ("https://nav.test/page",)
    assert kwargs["wait_until"] == "domcontentloaded"
    assert fake_session.page.url == "https://nav.test/page"
    assert 'button "OK"' in result


async def test_navigate_without_url_errors_before_goto(
    fake_session: FakeSession,
) -> None:
    result = await browse("navigate")

    assert result == "Error: 'url' is required for navigate action."
    assert fake_session.log.count("goto") == 0


async def test_back_calls_go_back(fake_session: FakeSession) -> None:
    result = await browse("back")

    _args, kwargs = fake_session.log.last("go_back")
    assert kwargs["wait_until"] == "commit"
    assert 'button "OK"' in result


async def test_forward_calls_go_forward(fake_session: FakeSession) -> None:
    result = await browse("forward")

    _args, kwargs = fake_session.log.last("go_forward")
    assert kwargs["wait_until"] == "commit"
    assert 'button "OK"' in result
