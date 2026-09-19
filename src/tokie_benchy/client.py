from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from time import perf_counter

import httpx

from tokie_benchy.config import Endpoint


def deep_merge(base: dict, extra: dict | None) -> dict:
    """Return base updated with extra; nested dicts merge, everything else overrides."""
    out = dict(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class RequestResult:
    index: int
    profile: str
    concurrency: int
    ok: bool = True
    error: str | None = None
    started_at: float = 0.0
    first_token_at: float | None = None
    finished_at: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    chunks: int = 0
    reasoning_chunks: int = 0
    token_times: list[float] = field(default_factory=list)
    usage_from_server: bool = False

    @property
    def has_reasoning(self) -> bool:
        return self.reasoning_tokens > 0 or self.reasoning_chunks > 0

    # --- derived metrics (None when not measurable) ---
    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.started_at) * 1000

    @property
    def e2e_ms(self) -> float | None:
        if not self.finished_at:
            return None
        return (self.finished_at - self.started_at) * 1000

    @property
    def tps(self) -> float | None:
        if self.first_token_at is None or self.output_tokens <= 0:
            return None
        window = self.finished_at - self.first_token_at
        if window <= 0:  # single-token reply: fall back to full latency
            window = self.finished_at - self.started_at
        return self.output_tokens / window if window > 0 else None

    @property
    def itl_ms(self) -> float | None:
        """Mean inter-token latency. Servers often batch several tokens per SSE chunk,
        so use the server token count when available and chunk count otherwise."""
        if self.first_token_at is None:
            return None
        tokens = self.output_tokens if self.usage_from_server else self.chunks
        if tokens < 2:
            return None
        return (self.finished_at - self.first_token_at) / (tokens - 1) * 1000

    @property
    def prefill_tps(self) -> float | None:
        ttft = self.ttft_ms
        if ttft is None or ttft <= 0 or self.prompt_tokens <= 0:
            return None
        return self.prompt_tokens / (ttft / 1000)

    def metric(self, key: str) -> float | None:
        return {
            "ttft": self.ttft_ms,
            "tps": self.tps,
            "itl": self.itl_ms,
            "e2e": self.e2e_ms,
            "prefill": self.prefill_tps,
        }.get(key)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "profile": self.profile,
            "concurrency": self.concurrency,
            "ok": self.ok,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "chunks": self.chunks,
            "reasoning_chunks": self.reasoning_chunks,
            "usage_from_server": self.usage_from_server,
            "ttft_ms": self.ttft_ms,
            "tps": self.tps,
            "itl_ms": self.itl_ms,
            "e2e_ms": self.e2e_ms,
            "prefill_tps": self.prefill_tps,
        }


def _delta_has_reasoning(delta: dict) -> bool:
    return bool(delta.get("reasoning_content") or delta.get("reasoning"))


def _delta_has_tokens(delta: dict) -> bool:
    return bool(delta.get("content")) or _delta_has_reasoning(delta)


async def stream_completion(
    client: httpx.AsyncClient,
    endpoint: Endpoint,
    *,
    prompt: str,
    max_tokens: int,
    index: int,
    profile: str,
    concurrency: int,
    temperature: float = 0.7,
    extra: dict | None = None,
) -> RequestResult:
    """Send one streaming chat completion and time every chunk.

    `extra` is deep-merged into the request body last (thinking controls, --extra-body).
    """
    body = deep_merge(
        {
            "model": endpoint.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        extra,
    )
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"

    result = RequestResult(index=index, profile=profile, concurrency=concurrency)
    result.started_at = perf_counter()
    try:
        async with client.stream("POST", endpoint.chat_url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode(errors="replace")
                result.ok = False
                result.error = f"HTTP {resp.status_code}: {text[:200]}"
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    now = perf_counter()
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if _delta_has_tokens(delta):
                            if result.first_token_at is None:
                                result.first_token_at = now
                            result.chunks += 1
                            result.token_times.append(now)
                            if _delta_has_reasoning(delta):
                                result.reasoning_chunks += 1
                    usage = obj.get("usage")
                    if usage:
                        result.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                        result.output_tokens = int(usage.get("completion_tokens") or 0)
                        details = usage.get("completion_tokens_details") or {}
                        result.reasoning_tokens = int(details.get("reasoning_tokens") or 0)
                        result.usage_from_server = True
    except (httpx.HTTPError, asyncio.TimeoutError) as exc:
        result.ok = False
        result.error = f"{type(exc).__name__}: {exc}"[:200]
    result.finished_at = perf_counter()

    if not result.usage_from_server:
        # No usage block: approximate one token per streamed chunk.
        result.output_tokens = result.chunks
    if result.ok and result.first_token_at is None:
        result.ok = False
        result.error = "no tokens received"
    return result
