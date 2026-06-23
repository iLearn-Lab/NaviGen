"""DashScope teacher-model client used by preprocessing scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from project_env import load_project_env


load_project_env(Path(__file__).resolve().parent / ".env")


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return {
        key: getattr(value, key)
        for key in ("output", "choices", "message", "content", "text", "code", "status_code", "request_id")
        if hasattr(value, key)
    }


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_response_text(response: Any) -> str:
    data = _as_mapping(response)
    output = _as_mapping(data.get("output", {}))

    choices = output.get("choices") or data.get("choices")
    if choices:
        choice = _as_mapping(choices[0])
        message = _as_mapping(choice.get("message", {}))
        text = _content_to_text(message.get("content"))
        if text:
            return text
        text = _content_to_text(choice.get("text"))
        if text:
            return text

    text = _content_to_text(output.get("text") or data.get("text"))
    if text:
        return text

    raise RuntimeError(f"DashScope response did not contain generated text: {response!r}")


def generate_text(
    messages: list[dict[str, str]],
    model_name: str,
    api_key: str | None = None,
    **kwargs: Any,
) -> str:
    """Generate text with DashScope's chat-style Generation API."""

    key = (api_key or os.getenv("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("No DashScope API key found. Set DASHSCOPE_API_KEY or DASHSCOPE_API_KEYS in .env.")

    try:
        import dashscope
    except ImportError as exc:
        raise RuntimeError("dashscope is required for teacher-model preprocessing.") from exc

    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "api_key": key,
        "result_format": "message",
    }
    request_kwargs.update(kwargs)
    response = dashscope.Generation.call(**request_kwargs)

    data = _as_mapping(response)
    status_code = data.get("status_code")
    if status_code not in (None, 200):
        code = data.get("code", "unknown")
        message = data.get("message", "no error message")
        request_id = data.get("request_id", "unknown")
        raise RuntimeError(f"DashScope request failed: status={status_code}, code={code}, request_id={request_id}, message={message}")

    return _extract_response_text(response).strip()
