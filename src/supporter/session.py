from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4


def new_session_id() -> str:
    return f"session_{int(time.time() * 1000)}_{uuid4().hex[:6]}"


def _serialize_part(part: Any) -> dict[str, Any]:
    if getattr(part, "text", None):
        return {"text": part.text}

    fc = getattr(part, "function_call", None)
    if fc is not None and getattr(fc, "name", None):
        return {
            "function_call": {
                "name": fc.name,
                "args": getattr(fc, "args", None) or {},
            }
        }

    fr = getattr(part, "function_response", None)
    if fr is not None and getattr(fr, "name", None):
        return {
            "function_response": {
                "name": fr.name,
                "response": getattr(fr, "response", None) or {},
            }
        }

    idata = getattr(part, "inline_data", None)
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
        if images_dir and "image" in sp and getattr(part, "inline_data", None):
            idata = part.inline_data
            img_data = getattr(idata, "data", None)
            if img_data:
                ext = (getattr(idata, "mime_type", "") or "").split("/")[-1] or "bin"
                fname = f"img_{int(time.time() * 1000)}_{uuid4().hex[:8]}.{ext}"
                img_path = images_dir / fname
                img_path.write_bytes(img_data)
                sp["image"]["image_ref"] = str(img_path)
        serialized_parts.append(sp)
    return {
        "role": getattr(content, "role", "user"),
        "parts": serialized_parts,
        "ts": time.time(),
    }


def _deserialize_content(record: dict[str, Any]) -> Any:
    from google.genai.types import Blob, Content, FunctionCall, FunctionResponse, Part

    parts = []
    for sp in record.get("parts", []):
        if "text" in sp:
            parts.append(Part(text=sp["text"]))
        elif "function_call" in sp:
            fc = sp["function_call"]
            parts.append(
                Part(
                    function_call=FunctionCall(
                        name=fc["name"],
                        args=fc.get("args", {}),
                    )
                )
            )
        elif "function_response" in sp:
            fr = sp["function_response"]
            parts.append(
                Part(
                    function_response=FunctionResponse(
                        name=fr["name"],
                        response=fr.get("response", {}),
                    )
                )
            )
        elif "unsupported" in sp:
            continue
        elif "image" in sp:
            ref = sp["image"].get("image_ref")
            if ref and Path(ref).exists():
                img_bytes = Path(ref).read_bytes()
                parts.append(
                    Part(
                        inline_data=Blob(
                            data=img_bytes,
                            mime_type=sp["image"].get(
                                "mime_type", "application/octet-stream"
                            ),
                        )
                    )
                )
        else:
            continue
    if not parts:
        return None
    return Content(role=record.get("role", "user"), parts=parts)


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
