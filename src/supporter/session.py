from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from .llm.types import ImagePart, Message, TextPart, ToolCallPart, ToolResultPart
from .logger import logger


def new_session_id() -> str:
    return f"session_{int(time.time() * 1000)}_{uuid4().hex[:6]}"


def _serialize_part(part: Any) -> dict[str, Any]:
    # TextPart or Gemini Part with text
    if getattr(part, "text", None):
        return {"text": part.text}

    # neutral ToolCallPart has .name and .args directly
    name: str | None = getattr(part, "name", None)
    args: Any = getattr(part, "args", None)
    if name is not None and args is not None:
        return {"function_call": {"name": name, "args": args or {}}}

    # Gemini Part with nested function_call
    fc: Any = getattr(part, "function_call", None)
    if fc is not None and getattr(fc, "name", None):
        return {
            "function_call": {
                "name": fc.name,
                "args": getattr(fc, "args", None) or {},
            }
        }

    # neutral ToolResultPart has .name and .response directly
    response: Any = getattr(part, "response", None)
    if name is not None and response is not None:
        return {"function_response": {"name": name, "response": response or {}}}

    # Gemini Part with nested function_response
    fr: Any = getattr(part, "function_response", None)
    if fr is not None and getattr(fr, "name", None):
        return {
            "function_response": {
                "name": fr.name,
                "response": getattr(fr, "response", None) or {},
            }
        }

    # neutral ImagePart has .mime_type + .data or .ref directly
    mime: str | None = getattr(part, "mime_type", None)
    img_data: Any = getattr(part, "data", None)
    img_ref: Any = getattr(part, "ref", None)
    if mime is not None and (img_data is not None or img_ref is not None):
        return {"image": {"mime_type": mime}}

    # Gemini Part with nested inline_data (Blob)
    idata: Any = getattr(part, "inline_data", None)
    if idata is not None and getattr(idata, "data", None):
        return {
            "image": {
                "mime_type": getattr(idata, "mime_type", "application/octet-stream"),
            }
        }

    return {"unsupported": True}


def _serialize_content(content: Any, images_dir: Path | None = None) -> dict[str, Any]:
    parts_raw = getattr(content, "parts", None) or []
    serialized_parts: list[dict[str, Any]] = []
    for part in parts_raw:
        sp = _serialize_part(part)
        if images_dir and "image" in sp:
            img_data = getattr(part, "data", None)
            if img_data is None:
                idata = getattr(part, "inline_data", None)
                img_data = getattr(idata, "data", None) if idata is not None else None
            if img_data:
                mime_type = sp["image"].get("mime_type", "application/octet-stream")
                ext = mime_type.split("/")[-1] or "bin"
                fname = f"img_{int(time.time() * 1000)}_{uuid4().hex[:8]}.{ext}"
                img_path = images_dir / fname
                img_path.write_bytes(img_data)
                sp["image"]["image_ref"] = str(img_path)
            else:
                logger.warning("Image part has no data; skipping image_ref")
        serialized_parts.append(sp)
    return {
        "role": getattr(content, "role", "user"),
        "parts": serialized_parts,
        "ts": time.time(),
    }


def _deserialize_content(record: dict[str, Any]) -> Message | None:
    parts: list[TextPart | ToolCallPart | ToolResultPart | ImagePart] = []
    for sp in record.get("parts", []):
        if "text" in sp:
            parts.append(TextPart(text=sp["text"]))
        elif "function_call" in sp:
            fc = sp["function_call"]
            parts.append(
                ToolCallPart(
                    name=fc["name"],
                    args=fc.get("args", {}),
                )
            )
        elif "function_response" in sp:
            fr = sp["function_response"]
            parts.append(
                ToolResultPart(
                    name=fr["name"],
                    response=fr.get("response", {}),
                )
            )
        elif "image" in sp:
            ref = sp["image"].get("image_ref")
            mime_type = sp["image"].get("mime_type", "application/octet-stream")
            if ref and Path(ref).exists():
                img_bytes = Path(ref).read_bytes()
                parts.append(
                    ImagePart(
                        mime_type=mime_type,
                        ref=ref,
                        data=img_bytes,
                    )
                )
            else:
                logger.warning(f"Image ref {ref!r} missing; inserting placeholder")
                parts.append(TextPart(text="[image unavailable]"))
        else:
            continue
    if not parts:
        return None
    return Message(role=record.get("role", "user"), parts=parts)


class HistoryStore:
    def __init__(self, session_id: str, root: Path) -> None:
        self._session_id = session_id
        self._dir = root / session_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "history.jsonl"
        self._images_dir = self._dir / "images"
        self._images_dir.mkdir(exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def session_id(self) -> str:
        return self._session_id

    def append(self, content: Any) -> None:
        record = _serialize_content(content, self._images_dir)
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def load(self, limit: int | None = None) -> list[Any]:
        if not self._path.exists():
            return []
        contents: list[Any] = []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    break
                continue
            content = _deserialize_content(record)
            if content is not None:
                contents.append(content)
        if limit and len(contents) > limit:
            contents = contents[-limit:]
        return contents

    def rotate(self) -> None:
        new_id = new_session_id()
        new_dir = self._dir.parent / new_id
        new_dir.mkdir(parents=True, exist_ok=True)
        self._session_id = new_id
        self._dir = new_dir
        self._path = new_dir / "history.jsonl"
        self._images_dir = new_dir / "images"
        self._images_dir.mkdir(exist_ok=True)

    def close(self) -> None:
        pass
