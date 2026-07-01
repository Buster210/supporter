"""OpenRouter provider — REST, OpenAI-compatible chat/completions endpoint."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..llm.types import GenOptions, Message, TextPart, ToolDef, tool_def_from_callable
from ..types import LLMChunk, LLMResult

API_URL = "https://openrouter.ai/api/v1/chat/completions"
CHAT_TIMEOUT = 120.0


def _messages_to_openai(prompt: str | list[Message]) -> list[dict[str, str]]:
    """Convert neutral prompt to OpenAI messages[]."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    messages: list[dict[str, str]] = []
    for msg in prompt:
        role = "assistant" if msg.role == "model" else msg.role
        content = " ".join(p.text for p in msg.parts if isinstance(p, TextPart))
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _tools_to_openai(tools: list[Any] | None) -> list[dict[str, Any]] | None:
    """Convert tool callables to OpenAI tools[] format via ToolDef introspection."""
    if not tools:
        return None
    result = []
    for fn in tools:
        td: ToolDef = fn if isinstance(fn, ToolDef) else tool_def_from_callable(fn)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": td.name,
                    "description": td.description,
                    "parameters": td.parameters,
                },
            }
        )
    return result or None


class OpenRouterProvider:
    """LLMProvider via OpenRouter's OpenAI-compatible REST API."""

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "openai/gpt-oss-120b:free",
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name

    def get_name(self) -> str:
        return self.model_name

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/supporter",
            "X-Title": "supporter",
        }

    def _build_payload(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool = False,
        options: GenOptions | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
        }
        if options:
            if options.temperature is not None:
                payload["temperature"] = options.temperature
            if options.max_output_tokens is not None:
                payload["max_tokens"] = options.max_output_tokens
            tools = options.extras.get("tools")
            if tools:
                openai_tools = _tools_to_openai(tools)
                if openai_tools:
                    payload["tools"] = openai_tools
        return payload

    async def generate(
        self,
        prompt: str | list[Message],
        options: GenOptions | None = None,
    ) -> LLMResult:
        messages = _messages_to_openai(prompt)
        payload = self._build_payload(messages, stream=False, options=options)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=CHAT_TIMEOUT) as client:
            resp = await client.post(API_URL, headers=self._headers(), json=payload)
            resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
        return LLMResult(
            text=text,
            model=data.get("model", self.model_name),
            duration=time.monotonic() - t0,
            raw=data,
        )

    async def generate_stream(
        self,
        prompt: str | list[Message],
        options: GenOptions | None = None,
    ) -> AsyncIterator[LLMChunk]:
        messages = _messages_to_openai(prompt)
        payload = self._build_payload(messages, stream=True, options=options)
        t0 = time.monotonic()
        async with (
            httpx.AsyncClient(timeout=CHAT_TIMEOUT) as client,
            client.stream(
                "POST", API_URL, headers=self._headers(), json=payload
            ) as resp,
        ):
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[len("data: ") :]
                if raw.strip() == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield LLMChunk(
                        text=content,
                        is_last=False,
                        model=chunk_data.get("model", self.model_name),
                    )
        yield LLMChunk(
            text="",
            is_last=True,
            model=self.model_name,
            raw={"duration": time.monotonic() - t0},
        )
