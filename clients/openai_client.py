"""Simple OpenAI API client helper for the poetry benchmark."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - import guarded for environments without the SDK
    OpenAI = None


@dataclass
class OpenAIConfig:
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class OpenAIClient:
    """Lightweight wrapper around the OpenAI Chat Completions API."""

    def __init__(self, config: OpenAIConfig) -> None:
        if OpenAI is None:  # pragma: no cover - makes missing dependency explicit at runtime
            raise ImportError("The openai package is required to use OpenAIClient")
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def generate_text(self, prompt: str, model: str, temperature: float = 1.0, max_tokens: int = 2048) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        print(response)
        return response.choices[0].message.content.strip()

    def rate_text(self, prompt: str, model: str, temperature: float = 1.0, max_tokens: int = 2048) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
