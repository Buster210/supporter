from __future__ import annotations

import difflib
import re
from datetime import datetime
from pathlib import Path

from ...config import config
from ...logger import logger
from . import blocklist

_REF_PATTERN = re.compile(r"\[ref=e\d+\]")
_NODE_PATTERN = re.compile(r"^(\s*)- (.*)$")
_ROLE_PATTERN = re.compile(r"^([a-zA-Z][a-zA-Z0-9]*)")
_NAME_PATTERN = re.compile(r'"((?:[^"\\]|\\.)*)"')
_CURSOR_PATTERN = re.compile(r"\s*\[cursor=pointer\]")
_REF_STRIP = re.compile(r"\s*\[ref=e\d+\]")
_REF_GROUP = re.compile(r"\[ref=(e\d+)\]")

INTERACTIVE_ROLES = frozenset(
    {
        "button",
        "link",
        "textbox",
        "combobox",
        "checkbox",
        "radio",
        "tab",
        "option",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "switch",
        "slider",
        "searchbox",
        "spinbutton",
        "treeitem",
    }
)
STRUCTURAL_ROLES = frozenset({"generic", "group", "none", "presentation", "section"})
_DEDUP_MIN_RUN = 3
_DEDUP_MIN_CHILDREN = 2
_TRACK_PARAMS = frozenset(
    {
        "pp",
        "si",
        "feature",
        "ab_channel",
        "list",
        "index",
        "start_radio",
        "ved",
        "usg",
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "gclid",
        "fbclid",
        "spm",
        "ref_src",
        "ref_url",
    }
)
_URL_PARAM = re.compile(r"([?&])([^=&]+)=[^&]*")
_URL_HOST = re.compile(r"^https?://([^/]+)")


def _browser_log_path() -> Path:
    return Path(config.log_file).expanduser().resolve().with_name("browser.log")


def log_snapshot(label: str, text: str) -> None:
    try:
        ts = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        sep = "=" * 72
        entry = f"{sep}\n{ts} — {label}\n{sep}\n{text}\n\n"
        path = _browser_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry)
    except Exception as exc:
        logger.debug(f"browser.log write failed [{type(exc).__name__}]: {exc}")


class _Node:
    __slots__ = (
        "aname",
        "children",
        "indent",
        "is_url",
        "named",
        "raw",
        "ref",
        "role",
        "text",
    )

    def __init__(self, indent: int, raw: str, body: str) -> None:
        self.indent = indent
        self.raw = raw
        role_match = _ROLE_PATTERN.match(body)
        self.role = role_match.group(1) if role_match else ""
        name_match = _NAME_PATTERN.search(body)
        self.aname = name_match.group(1) if name_match else ""
        self.named = name_match is not None
        self.ref = bool(_REF_PATTERN.search(body))
        self.is_url = body.startswith("/url:")
        self.text = ""
        if ":" in body:
            after = body.split(":", 1)[1].strip()
            if after and not after.startswith("-"):
                self.text = after
        self.children: list[_Node] = []


def _parse(text: str) -> _Node:
    root = _Node(-1, "", "")
    stack: list[_Node] = [root]
    for line in text.splitlines():
        match = _NODE_PATTERN.match(line)
        if not match:
            continue
        indent = len(match.group(1))
        node = _Node(indent, line, match.group(2))
        while stack[-1].indent >= indent:
            stack.pop()
        stack[-1].children.append(node)
        stack.append(node)
    return root


def _self_meaningful(node: _Node) -> bool:
    if node.is_url or node.role in INTERACTIVE_ROLES:
        return True
    if node.role == "img":
        return node.named
    return node.named or bool(node.text)


def _kept(node: _Node) -> bool:
    return _self_meaningful(node) or any(_kept(c) for c in node.children)


def _fold_urls(node: _Node) -> None:
    for child in node.children:
        _fold_urls(child)
    if node.role == "link":
        urls = [c for c in node.children if c.is_url]
        if urls:
            href = urls[0].raw.split(":", 1)[1].strip()
            node.raw = f"{node.raw.rstrip()}  /url:{href}"
            node.children = [c for c in node.children if not c.is_url]


def _fold_heading_link(node: _Node) -> None:
    for child in node.children:
        _fold_heading_link(child)
    if node.role != "heading" or not node.aname:
        return
    links = [c for c in node.children if c.role == "link" and c.aname == node.aname]
    others = [c for c in node.children if c not in links]
    if len(links) != 1 or any(_self_meaningful(c) for c in others):
        return
    link = links[0]
    ref_match = _REF_GROUP.search(link.raw)
    if ref_match:
        tail = link.raw.split("]", 1)[-1].lstrip(":") if "]" in link.raw else ""
        head = _REF_STRIP.sub("", node.raw.rstrip().rstrip(":")).rstrip()
        node.raw = f"{head} [ref={ref_match.group(1)}]:{tail}"
        node.ref = True
    node.children = link.children + others


def _name_echoed(parent_name: str, child: _Node) -> bool:
    for value in (child.aname, child.text):
        if value and (value == parent_name or value in parent_name):
            return True
    return False


def _drop_name_echo(node: _Node) -> None:
    for child in node.children:
        _drop_name_echo(child)
    if not node.aname:
        return
    kept_children: list[_Node] = []
    for child in node.children:
        echoes = (
            not child.children
            and not child.is_url
            and child.role in (STRUCTURAL_ROLES | {"text"})
            and _name_echoed(node.aname, child)
        )
        if not echoes:
            kept_children.append(child)
    node.children = kept_children


def _prune(node: _Node) -> None:
    node.children = [c for c in node.children if _kept(c)]
    for child in node.children:
        _prune(child)


def _flatten(node: _Node) -> None:
    for child in node.children:
        _flatten(child)
    new_children: list[_Node] = []
    for child in node.children:
        bare = (
            child.role in STRUCTURAL_ROLES
            and not child.named
            and not child.text
            and not child.is_url
        )
        if bare:
            new_children.extend(child.children)
        else:
            new_children.append(child)
    node.children = new_children


def _bare_host(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def _trim_urls(node: _Node, page_host: str = "") -> None:
    for child in node.children:
        _trim_urls(child, page_host)
    if "/url:" not in node.raw:
        return
    head, url = node.raw.split("/url:", 1)
    stripped = url.lstrip()

    host_match = _URL_HOST.match(stripped)
    if host_match:
        href_host = _bare_host(host_match.group(1))
        if blocklist.host_listed(href_host):
            node.raw = head.rstrip()
            return
        if page_host and href_host == _bare_host(page_host):
            stripped = stripped[host_match.end() :] or "/"

    def drop(match: re.Match[str]) -> str:
        return "" if match.group(2) in _TRACK_PARAMS else match.group(0)

    cleaned = _URL_PARAM.sub(drop, stripped)
    cleaned = re.sub(r"\?&", "?", cleaned).rstrip("?&")
    if cleaned in ("", "/"):
        node.raw = head.rstrip()
        return
    node.raw = f"{head}/url:  {cleaned}"


def _signature(node: _Node) -> tuple[object, ...]:
    url = ""
    if "/url:" in node.raw:
        url = node.raw.split("/url:", 1)[1].strip()
    return (
        node.role,
        node.aname,
        node.text,
        url,
        tuple(_signature(c) for c in node.children),
    )


def _dedup_siblings(node: _Node) -> None:
    for child in node.children:
        _dedup_siblings(child)
    children = node.children
    result: list[_Node] = []
    i = 0
    while i < len(children):
        run_end = i
        while (
            run_end + 1 < len(children)
            and _signature(children[run_end + 1]) == _signature(children[i])
            and len(children[i].children) >= _DEDUP_MIN_CHILDREN
        ):
            run_end += 1
        result.append(children[i])
        extra = run_end - i
        if extra >= _DEDUP_MIN_RUN - 1:
            refs = [
                m.group(1)
                for k in range(i + 1, run_end + 1)
                if (m := _REF_GROUP.search(children[k].raw))
            ]
            span = f" {refs[0]}..{refs[-1]}" if len(refs) >= 2 else ""
            summary = _Node(children[i].indent, f"- (x{extra} more similar{span})", "")
            summary.text = "summary"
            result.append(summary)
            i = run_end + 1
        else:
            i += 1
    node.children = result


def _emit(node: _Node, out: list[str], depth: int) -> None:
    kept_children = [c for c in node.children if c.text == "summary" or _kept(c)]
    child_depth = depth
    if node.indent >= 0 and (
        _self_meaningful(node) or node.text == "summary" or kept_children
    ):
        body = _CURSOR_PATTERN.sub("", node.raw).lstrip()
        if body.startswith("- "):
            body = body[2:]
        actionable = node.role in INTERACTIVE_ROLES or "/url:" in body
        if not actionable:
            body = _REF_STRIP.sub("", body)
        if not kept_children and "/url:" not in body and body.endswith(":"):
            body = body[:-1].rstrip()
        out.append(f"{'  ' * depth}- {body}")
        child_depth = depth + 1
    for child in kept_children:
        _emit(child, out, child_depth)


def _page_host(page_url: str) -> str:
    match = _URL_HOST.match(page_url)
    return match.group(1) if match else ""


def clean_snapshot(text: str, page_url: str = "") -> str:
    if not text or "- " not in text:
        return text
    try:
        root = _parse(text)
        _fold_urls(root)
        _fold_heading_link(root)
        _trim_urls(root, _page_host(page_url))
        _drop_name_echo(root)
        _prune(root)
        _dedup_siblings(root)
        _flatten(root)
        out: list[str] = []
        _emit(root, out, 0)
        return "\n".join(out)
    except Exception as exc:
        logger.debug(f"clean_snapshot failed [{type(exc).__name__}]: {exc}")
        return text


def filter_interactive(text: str) -> str:
    if not text:
        return text
    root = _parse(text)
    _fold_urls(root)

    def interactive_subtree(node: _Node) -> bool:
        return node.role in INTERACTIVE_ROLES or any(
            interactive_subtree(c) for c in node.children
        )

    out: list[str] = []

    def emit(node: _Node) -> None:
        keep = [c for c in node.children if interactive_subtree(c)]
        if node.indent >= 0 and (node.role in INTERACTIVE_ROLES or keep):
            out.append(_CURSOR_PATTERN.sub("", node.raw))
        for child in keep:
            emit(child)

    emit(root)
    return "\n".join(out)


_LAST_SNAPSHOT: dict[str, str] = {}


def remember_snapshot(key: str, cleaned: str) -> None:
    if key:
        _LAST_SNAPSHOT[key] = cleaned


def has_baseline(key: str) -> bool:
    return bool(key) and key in _LAST_SNAPSHOT


def forget_snapshot(key: str) -> None:
    _LAST_SNAPSHOT.pop(key, None)


def diff_snapshot(key: str, cleaned: str) -> str:
    previous = _LAST_SNAPSHOT.get(key)
    _LAST_SNAPSHOT[key] = cleaned
    if previous is None:
        return "(no previous snapshot to diff against; baseline stored)"
    if previous == cleaned:
        return "(no changes since last snapshot)"
    out: list[str] = []
    for line in difflib.unified_diff(
        previous.splitlines(), cleaned.splitlines(), n=0, lineterm=""
    ):
        if line.startswith(("+++", "---", "@@")):
            continue
        out.append(line)
    return "\n".join(out) if out else "(no changes since last snapshot)"
