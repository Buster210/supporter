from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from supporter.config import config
from supporter.tools import _resolve_path
from supporter.tools.browser import snapshot
from supporter.tools.browser.tool import browse

from .conftest import FakeSession

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _reset_snapshot_baselines() -> None:
    snapshot._LAST_SNAPSHOT.clear()


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _resolve_path.cache_clear()
    monkeypatch.setattr(config, "allowed_directories", [str(tmp_path)])
    return tmp_path


# --- upload --------------------------------------------------------------


async def test_upload_without_ref_errors(fake_session: FakeSession) -> None:
    result = await browse("upload", path="x.txt")

    assert result == "Error: 'ref' is required for upload. Get a snapshot first."


async def test_upload_without_path_errors(fake_session: FakeSession) -> None:
    result = await browse("upload", ref="e2")

    assert result == "Error: 'path' is required for upload (the file to attach)."


async def test_upload_missing_file_errors(
    fake_session: FakeSession, project_root: Path
) -> None:
    result = await browse("upload", ref="e2", path="absent.txt")

    assert result == f"Error: file not found: {project_root / 'absent.txt'}"


async def test_upload_outside_root_is_rejected(
    fake_session: FakeSession, project_root: Path
) -> None:
    result = await browse("upload", ref="e2", path="/etc/hosts")

    # The path is resolved (so /etc → /private/etc on macOS); assert the
    # rejection contract, not the platform-specific resolved prefix.
    assert result.startswith("Error: Path ")
    assert "is outside project root" in result


async def test_upload_existing_file_sets_input(
    fake_session: FakeSession, project_root: Path
) -> None:
    src = project_root / "doc.txt"
    src.write_text("hi")

    await browse("upload", ref="e2", path="doc.txt", fast=True)

    args, _kwargs = fake_session.log.last("set_input_files")
    assert args == (str(src),)


# --- download ------------------------------------------------------------


async def test_download_without_ref_errors(fake_session: FakeSession) -> None:
    result = await browse("download", path="out")

    assert result == "Error: 'ref' is required for download. Get a snapshot first."


async def test_download_saves_to_resolved_path(
    fake_session: FakeSession, project_root: Path
) -> None:
    dest = project_root / "saved.bin"

    result = await browse("download", ref="e2", path="saved.bin", fast=True)

    assert result == f"Downloaded to {dest}"
    args, _kwargs = fake_session.log.last("save_as")
    assert args == (str(dest),)


# --- cookies -------------------------------------------------------------


async def test_cookies_empty_jar(fake_session: FakeSession) -> None:
    result = await browse("cookies")

    assert result == "(no cookies)"


async def test_cookies_lists_entries(fake_session: FakeSession) -> None:
    fake_session.context.cookie_list = [
        {"name": "sid", "domain": "example.test", "value": "abc"},
    ]

    result = await browse("cookies")

    assert result == "1 cookies:\nsid @ example.test"


async def test_cookies_get_missing_key_errors(fake_session: FakeSession) -> None:
    result = await browse("cookies", key="absent")

    assert result == "Error: no cookie named 'absent'."


async def test_cookies_get_reveals_value_after_confirm(
    fake_session: FakeSession,
) -> None:
    fake_session.context.cookie_list = [
        {"name": "sid", "domain": "example.test", "value": "abc"},
    ]

    result = await browse("cookies", key="sid")

    assert result == "sid=abc"
    # Revealing a stored cookie value is gated; the confirm must have fired.
    assert len(fake_session.confirm.calls) == 1


async def test_cookies_get_denied_does_not_reveal(
    fake_session: FakeSession,
) -> None:
    fake_session.context.cookie_list = [
        {"name": "sid", "domain": "example.test", "value": "abc"},
    ]
    fake_session.confirm.allow = False

    result = await browse("cookies", key="sid")

    assert result == "Error: action cancelled."
    # The gate was consulted and denied; the cancel string (not the value) is
    # all the cookie reveal can return.
    assert len(fake_session.confirm.calls) == 1


# --- storage -------------------------------------------------------------


async def test_storage_set_writes_value(fake_session: FakeSession) -> None:
    result = await browse("storage", key="theme", value="dark")

    assert result == "Set localStorage['theme'] (4 chars)."
    # Mutating localStorage is gated; the confirm must have fired.
    assert len(fake_session.confirm.calls) == 1


async def test_storage_set_denied_does_not_write(
    fake_session: FakeSession,
) -> None:
    fake_session.confirm.allow = False

    result = await browse("storage", key="theme", value="dark")

    assert result == "Error: action cancelled."
    # The setItem evaluate must not run once the gate denies.
    assert fake_session.log.count("evaluate") == 0


async def test_storage_get_missing_key_errors(fake_session: FakeSession) -> None:
    # The fake page.evaluate returns eval_result (None by default), which the
    # handler reads as a missing localStorage key.
    result = await browse("storage", key="absent")

    assert result == "Error: no localStorage key 'absent'."


async def test_storage_get_returns_value(fake_session: FakeSession) -> None:
    fake_session.page.eval_result = "dark"

    result = await browse("storage", key="theme")

    assert result == "theme=dark"
    # Revealing a stored value is gated; the confirm must have fired.
    assert len(fake_session.confirm.calls) == 1


async def test_storage_get_denied_does_not_read(
    fake_session: FakeSession,
) -> None:
    fake_session.page.eval_result = "dark"
    fake_session.confirm.allow = False

    result = await browse("storage", key="theme")

    assert result == "Error: action cancelled."
    # The getItem evaluate must not run once the gate denies.
    assert fake_session.log.count("evaluate") == 0
