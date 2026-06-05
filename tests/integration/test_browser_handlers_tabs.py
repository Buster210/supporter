from __future__ import annotations

from supporter.tools.browser.tool import browse

from .conftest import FakeSession


async def test_tabs_lists_pages_with_active_marker(
    fake_session: FakeSession,
) -> None:
    result = await browse("tabs")

    lines = result.splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("* [0] ")
    assert fake_session.page.url in lines[0]


async def test_tabs_marks_only_the_active_page(fake_session: FakeSession) -> None:
    await browse("newtab", url="https://second.test/")

    result = await browse("tabs")

    lines = result.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("  [0] ")
    assert lines[1].startswith("* [1] ")


async def test_newtab_appends_a_page_and_navigates(
    fake_session: FakeSession,
) -> None:
    before = len(fake_session.context.pages)

    result = await browse("newtab", url="https://second.test/")

    assert len(fake_session.context.pages) == before + 1
    assert fake_session.context.pages[-1].url == "https://second.test/"
    assert 'button "OK"' in result


async def test_newtab_reuses_the_sole_blank_tab(
    fake_session: FakeSession,
) -> None:
    fake_session.page.url = "about:blank"
    before = len(fake_session.context.pages)

    result = await browse("newtab", url="https://second.test/")

    assert len(fake_session.context.pages) == before
    assert fake_session.context.pages[0] is fake_session.page
    assert fake_session.page.url == "https://second.test/"
    assert 'button "OK"' in result


async def test_tab_switch_brings_target_to_front(
    fake_session: FakeSession,
) -> None:
    await browse("newtab", url="https://second.test/")

    await browse("tab", index=0)

    assert fake_session.context.pages[0] is fake_session.page
    assert fake_session.log.count("bring_to_front") >= 1


async def test_tab_index_out_of_range_errors(fake_session: FakeSession) -> None:
    result = await browse("tab", index=5)

    assert result == "Error: tab index 5 out of range (0..0)."


async def test_closetab_closes_active_and_reports_when_none_left(
    fake_session: FakeSession,
) -> None:
    result = await browse("closetab")

    assert result == "Closed the last tab; no tabs remain open."
    assert fake_session.context.pages == []


async def test_closetab_of_background_tab_keeps_active(
    fake_session: FakeSession,
) -> None:
    await browse("newtab", url="https://second.test/")
    active = fake_session.context.pages[-1]

    result = await browse("closetab", index=0)

    assert active in fake_session.context.pages
    assert len(fake_session.context.pages) == 1
    assert result == "(no changes since last snapshot)"
