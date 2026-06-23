from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from supporter.tools.browser.tool import browse

STORAGE_BODY = b"<html><body><h1>Storage</h1></body></html>"
_UPLOAD_SRC = Path(__file__).resolve().parent / "to_upload.txt"


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


def _first_ref(snapshot: str, needle: str) -> str:
    for line in snapshot.splitlines():
        if needle in line:
            match = re.search(r"\[ref=(e\d+)\]", line)
            if match:
                return match.group(1)
    raise AssertionError(f"no [ref=eN] line containing {needle!r} in:\n{snapshot}")


@pytest.mark.asyncio
async def test_upload_attaches_file(throwaway_browser: None) -> None:
    # Navigate first, then create the file just before upload to avoid
    # external file cleanup (macOS Spotlight/indexer) deleting it.
    snap = await browse("navigate", url=UPLOAD_PAGE)
    ref = _first_ref(snap, '"Pick file"')
    await asyncio.to_thread(_UPLOAD_SRC.write_text, "payload")
    try:
        after = await browse("upload", ref=ref, path=str(_UPLOAD_SRC))
        assert "picked:to_upload.txt" in after
    finally:
        _UPLOAD_SRC.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_storage_set_then_get(
    throwaway_browser: None, storage_origin: str
) -> None:
    await browse("navigate", url=storage_origin)
    set_result = await browse("storage", key="tok", value="abc123")
    assert set_result.startswith("Set localStorage['tok']")
    got = await browse("storage", key="tok")
    assert got == "tok=abc123"
