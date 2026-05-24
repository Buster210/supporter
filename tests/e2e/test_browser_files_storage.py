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
# origin), so storage tests need a real http:// origin.
STORAGE_BODY = b"<html><body><h1>Storage</h1></body></html>"


class _StorageHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(STORAGE_BODY)))
        self.end_headers()
        self.wfile.write(STORAGE_BODY)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def storage_origin() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StorageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
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

DOWNLOAD_PAGE = (
    "data:text/html,"
    "<html><body>"
    "<a id='dl' aria-label='Get file' download='grabbed.txt' "
    "href='data:text/plain,hello-download'>download</a>"
    "</body></html>"
)


async def _always_allow(_title: str, _detail: str) -> bool:
    return True


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
async def test_storage_set_then_get(
    throwaway_browser: Path, storage_origin: str
) -> None:
    await browse("navigate", url=storage_origin)
    set_result = await browse("storage", key="tok", value="abc123")
    assert set_result.startswith("Set localStorage['tok']")
    got = await browse("storage", key="tok")
    assert got == "tok=abc123"
