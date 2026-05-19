from __future__ import annotations

import re
import threading
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from supporter.config import config as real_config
from supporter.tools import resolved_project_root
from supporter.tools.browser import guardrails, session
from supporter.tools.browser.tool import browse

# localStorage and document.cookie are blocked on `data:` URLs (opaque
# origin), so storage/cookie tests need a real http:// origin. A throwaway
# loopback server serves a single trivial page for exactly that.
STORAGE_BODY = b"<html><body><h1>Storage</h1></body></html>"


class _StorageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(STORAGE_BODY)))
        self.end_headers()
        self.wfile.write(STORAGE_BODY)

    def log_message(self, *_args: object) -> None:
        pass  # silence per-request stderr logging in the test run


@pytest.fixture(scope="module")
def storage_origin() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StorageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]  # host is the 127.0.0.1 bound above
        yield f"http://127.0.0.1:{port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


UPLOAD_PAGE = (
    "data:text/html,"
    "<html><body>"
    "<input id='f' type='file' aria-label='Pick file' />"
    "<p id='out'>none</p>"
    "<script>"
    "document.getElementById('f').onchange="
    "(e)=>document.getElementById('out').textContent="
    "'picked:'+(e.target.files[0]?e.target.files[0].name:'');"
    "</script>"
    "</body></html>"
)

# An anchor that, when clicked, downloads a small text file.
DOWNLOAD_PAGE = (
    "data:text/html,"
    "<html><body>"
    "<a id='dl' aria-label='Get file' download='grabbed.txt' "
    "href='data:text/plain,hello-download'>download</a>"
    "</body></html>"
)


@pytest.fixture
async def throwaway_browser(tmp_path: Path) -> AsyncIterator[Path]:
    saved_path = real_config.browser_profile_path
    saved_headless = real_config.browser_headless
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    real_config.browser_profile_path = str(profile_dir)
    real_config.browser_headless = True
    guardrails.register_browse_callback(confirmation=_always_allow)

    work = resolved_project_root() / ".tier4_tmp"
    work.mkdir(exist_ok=True)
    try:
        yield work
    finally:
        await session.close_session()
        guardrails.register_browse_callback(confirmation=None)
        real_config.browser_profile_path = saved_path
        real_config.browser_headless = saved_headless
        for f in work.glob("*"):
            f.unlink()
        work.rmdir()


async def _always_allow(_title: str, _detail: str) -> bool:
    return True


def _first_ref(snapshot: str, needle: str) -> str:
    for line in snapshot.splitlines():
        if needle in line:
            match = re.search(r"\[ref=(e\d+)\]", line)
            if match:
                return match.group(1)
    raise AssertionError(f"no [ref=eN] line containing {needle!r} in:\n{snapshot}")


@pytest.mark.asyncio
async def test_upload_attaches_file(throwaway_browser: Path) -> None:
    src = throwaway_browser / "to_upload.txt"
    src.write_text("payload")
    snap = await browse("navigate", url=UPLOAD_PAGE)
    ref = _first_ref(snap, '"Pick file"')

    after = await browse("upload", ref=ref, path=str(src))
    assert "picked:to_upload.txt" in after


@pytest.mark.asyncio
async def test_upload_rejects_path_outside_project(
    throwaway_browser: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "secret.txt"  # tmp_path is OUTSIDE the project root
    outside.write_text("nope")
    snap = await browse("navigate", url=UPLOAD_PAGE)
    ref = _first_ref(snap, '"Pick file"')

    result = await browse("upload", ref=ref, path=str(outside))
    assert result.startswith("Error:")
    assert "outside project root" in result


@pytest.mark.asyncio
async def test_upload_requires_path(throwaway_browser: Path) -> None:
    snap = await browse("navigate", url=UPLOAD_PAGE)
    ref = _first_ref(snap, '"Pick file"')
    result = await browse("upload", ref=ref)
    assert result.startswith("Error: 'path' is required")


@pytest.mark.asyncio
async def test_download_saves_into_project(throwaway_browser: Path) -> None:
    snap = await browse("navigate", url=DOWNLOAD_PAGE)
    ref = _first_ref(snap, '"Get file"')

    result = await browse("download", ref=ref, path=str(throwaway_browser))
    assert result.startswith("Downloaded to")
    saved = throwaway_browser / "grabbed.txt"
    assert saved.exists()
    assert saved.read_text() == "hello-download"


@pytest.mark.asyncio
async def test_storage_set_then_get(
    throwaway_browser: Path, storage_origin: str
) -> None:
    await browse("navigate", url=storage_origin)
    set_result = await browse("storage", key="tok", value="abc123")
    assert set_result.startswith("Set localStorage['tok']")

    got = await browse("storage", key="tok")
    assert got == "tok=abc123"


@pytest.mark.asyncio
async def test_storage_list_keys(throwaway_browser: Path, storage_origin: str) -> None:
    await browse("navigate", url=storage_origin)
    await browse("storage", key="a", value="1")
    await browse("storage", key="b", value="2")
    listing = await browse("storage")
    assert "a" in listing
    assert "b" in listing


@pytest.mark.asyncio
async def test_storage_get_missing_key(
    throwaway_browser: Path, storage_origin: str
) -> None:
    await browse("navigate", url=storage_origin)
    result = await browse("storage", key="absent")
    assert result.startswith("Error: no localStorage key")


@pytest.mark.asyncio
async def test_cookies_listing_hides_values(
    throwaway_browser: Path, storage_origin: str
) -> None:
    await browse("navigate", url=storage_origin)
    # Seed a cookie via JS (eval is gated; the fixture auto-approves).
    await browse("eval", script="document.cookie = 'sid=topsecret'")
    listing = await browse("cookies")
    assert "sid" in listing  # name present
    assert "topsecret" not in listing  # value never in a listing


@pytest.mark.asyncio
async def test_cookies_named_get_returns_value(
    throwaway_browser: Path, storage_origin: str
) -> None:
    await browse("navigate", url=storage_origin)
    await browse("eval", script="document.cookie = 'sid=topsecret'")
    got = await browse("cookies", key="sid")
    assert got == "sid=topsecret"


@pytest.mark.asyncio
async def test_storage_set_denied_blocks(tmp_path: Path, storage_origin: str) -> None:
    saved_path = real_config.browser_profile_path
    saved_headless = real_config.browser_headless
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    real_config.browser_profile_path = str(profile_dir)
    real_config.browser_headless = True

    async def deny_writes(title: str, _detail: str) -> bool:
        return "localStorage" not in title  # approve lifecycle, deny the write

    guardrails.register_browse_callback(confirmation=deny_writes)
    try:
        await browse("navigate", url=storage_origin)
        result = await browse("storage", key="tok", value="abc")
        assert result == "Error: action cancelled."
        # Confirm the value was NOT written.
        guardrails.register_browse_callback(confirmation=_always_allow)
        check = await browse("storage", key="tok")
        assert check.startswith("Error: no localStorage key")
    finally:
        await session.close_session()
        guardrails.register_browse_callback(confirmation=None)
        real_config.browser_profile_path = saved_path
        real_config.browser_headless = saved_headless
