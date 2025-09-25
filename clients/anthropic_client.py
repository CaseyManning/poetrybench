"""Simple Anthropics API client helper for the poetry benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

try:
    import anthropic
except ImportError:  # pragma: no cover - import guarded for environments without the SDK
    anthropic = None


@dataclass
class AnthropicConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class AnthropicClient:
    """Lightweight wrapper around the Anthropic Messages API."""

    def __init__(self, config: AnthropicConfig) -> None:
        if anthropic is None:  # pragma: no cover - makes missing dependency explicit at runtime
            raise ImportError("The anthropic package is required to use AnthropicClient")
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = anthropic.Anthropic(**client_kwargs)

    def generate_text(self, prompt: str, model: str, temperature: float = 1.0, max_tokens: int = 512) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return _join_response_text(response)

    def rate_text(self, prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return _join_response_text(response)


def _join_response_text(response: Any) -> str:
    """Concatenate user-visible text from an Anthropic response."""
    parts = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    if not parts and response.content:
        parts.extend(str(block) for block in response.content)
    return "".join(parts).strip()
