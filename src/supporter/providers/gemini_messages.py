from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import LLMResult


def build_user_message(prompt: str) -> Any:
    from google.genai.types import Content, Part

    return Content(role="user", parts=[Part(text=prompt)])


def build_assistant_message(text: str) -> Any:
    from google.genai.types import Content, Part

    return Content(role="model", parts=[Part(text=text)])


def extract_assistant_message(result: LLMResult) -> Any | None:
    if not result.candidates or not result.candidates[0].content:
        return None
    content = result.candidates[0].content
    if getattr(content, "role", None) == "model":
        return content
    from google.genai.types import Content

    return Content(role="model", parts=content.parts)


class GeminiMessageMixin:
    def build_user_message(self, prompt: str) -> Any:
        return build_user_message(prompt)

    def extract_assistant_message(self, result: LLMResult) -> Any | None:
        return extract_assistant_message(result)

    def build_assistant_message(self, text: str) -> Any:
        return build_assistant_message(text)
