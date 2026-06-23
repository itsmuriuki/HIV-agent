"""
Direct HTTP client for any OpenAI-compatible chat completions endpoint.

Supports OpenAI, AWS Bedrock (via OPENAI_BASE_URL), and any other
provider that implements the /v1/chat/completions interface.
"""

import httpx
from dataclasses import dataclass, field


class LLMClientError(Exception):
    pass


@dataclass
class UsageStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMTarget:
    name: str
    model: str
    api_key: str
    base_url: str = "https://api.openai.com"

    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"


def _headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "Content-Type": "application/json",
    }


def _parse_usage(usage: dict | None) -> UsageStats:
    if not usage:
        return UsageStats()
    return UsageStats(
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        total_tokens=usage.get("total_tokens", 0),
    )


async def complete_chat(
    target: LLMTarget,
    messages: list[dict],
    *,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> tuple[str, UsageStats]:
    payload = {
        "model": target.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = _headers(target.api_key)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(target.chat_completions_url(), json=payload, headers=headers)
        if resp.status_code >= 400:
            raise LLMClientError(f"{target.name}: HTTP {resp.status_code} — {resp.text[:300]}")
        data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMClientError(f"Malformed response from {target.name}: {e}") from e
    usage = _parse_usage(data.get("usage"))
    return content, usage
