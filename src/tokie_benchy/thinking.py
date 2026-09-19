"""Thinking / reasoning-budget controls across OpenAI-compatible flavors.

Every provider spells this differently. We know the common spellings and, in
`auto` mode, probe the endpoint to find one that provably changes the output.
Servers such as vLLM silently accept unknown fields, so acceptance alone is not
evidence; only an observable difference in reasoning tokens counts as verified.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable

EFFORTS = ("off", "minimal", "low", "medium", "high", "max")
DEFAULT_EFFORT = "low"

# Token budgets used by budget-style providers (Anthropic, Gemini).
BUDGET_TOKENS = {"minimal": 512, "low": 1024, "medium": 4096, "high": 16384, "max": 32768}
# Level names used by effort-style providers (OpenAI, OpenRouter, gpt-oss, Grok).
_LEVEL = {"off": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "max": "xhigh"}


@dataclass(frozen=True)
class Flavor:
    name: str
    param: str  # the request field(s) this flavor sets, for display
    granularity: str  # "levels" | "budget" | "toggle"
    build: Callable[[str], dict]
    providers: str
    probe_max_tokens: Callable[[str], int] = lambda effort: 16


def _openai(e: str) -> dict:
    return {"reasoning_effort": _LEVEL[e]}


def _openrouter(e: str) -> dict:
    return {"reasoning": {"enabled": False}} if e == "off" else {"reasoning": {"effort": _LEVEL[e]}}


def _anthropic(e: str) -> dict:
    if e == "off":
        return {"thinking": {"type": "disabled"}}
    return {"thinking": {"type": "enabled", "budget_tokens": BUDGET_TOKENS[e]}}


def _deepseek(e: str) -> dict:
    return {"thinking": {"type": "disabled" if e == "off" else "enabled"}}


def _gemini(e: str) -> dict:
    budget = 0 if e == "off" else BUDGET_TOKENS[e]
    return {"google": {"thinking_config": {"thinking_budget": budget, "include_thoughts": True}}}


def _qwen(e: str) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": e != "off"}}


def _ollama(e: str) -> dict:
    if e == "off":
        return {"think": False}
    return {"think": e if e in ("low", "medium", "high") else True}


# Probe order doubles as tie-break order among equally-verified flavors.
FLAVORS: dict[str, Flavor] = {
    "openai": Flavor("openai", "reasoning_effort", "levels", _openai, "OpenAI o-series/GPT-5, gpt-oss on vLLM, Grok, LiteLLM, DeepSeek V3.2"),
    "openrouter": Flavor("openrouter", "reasoning.effort", "levels", _openrouter, "OpenRouter"),
    "anthropic": Flavor("anthropic", "thinking.budget_tokens", "budget", _anthropic, "Anthropic Claude via proxies",
                        probe_max_tokens=lambda e: (BUDGET_TOKENS.get(e, 0) + 64)),
    "deepseek": Flavor("deepseek", "thinking.type", "toggle", _deepseek, "DeepSeek V3.1+"),
    "gemini": Flavor("gemini", "google.thinking_config.thinking_budget", "budget", _gemini, "Gemini OpenAI-compat"),
    "qwen": Flavor("qwen", "chat_template_kwargs.enable_thinking", "toggle", _qwen, "Qwen3 / any thinking chat template on vLLM, SGLang, llama.cpp"),
    "ollama": Flavor("ollama", "think", "toggle", _ollama, "Ollama"),
}
STYLES = ("auto", *FLAVORS, "none")


@dataclass
class ProbeResult:
    ok: bool
    has_reasoning: bool
    error: str | None = None


@dataclass
class ThinkingStatus:
    effort: str
    style: str
    flavor: str | None
    verdict: str  # verified | unverified | unsupported | explicit | disabled
    body: dict = field(default_factory=dict)
    probes: dict[str, str] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if self.verdict == "disabled":
            return "thinking params: none sent (style=none)"
        if self.verdict == "unsupported":
            return f"thinking effort={self.effort}: [b]unsupported[/b] — no known parameter changed the output; requests sent without one"
        fl = FLAVORS[self.flavor]  # type: ignore[index]
        note = {"toggle": " (on/off only, level ignored)", "budget": f" (budget {BUDGET_TOKENS.get(self.effort, 0)} tok)", "levels": ""}[fl.granularity]
        state = {"verified": "verified by probe", "unverified": "accepted, effect unverified", "explicit": "explicit, not probed"}[self.verdict]
        return f"thinking effort={self.effort} via {fl.param}{note} · {state}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["summary"] = self.summary
        return d


Sender = Callable[[dict, int], Awaitable[ProbeResult]]


def explicit_status(effort: str, style: str) -> ThinkingStatus:
    if style == "none":
        return ThinkingStatus(effort, style, None, "disabled")
    return ThinkingStatus(effort, style, style, "explicit", body=FLAVORS[style].build(effort))


async def probe_flavors(send: Sender, effort: str, log: Callable[[str], None]) -> ThinkingStatus:
    """Try each flavor with `effort` and with its opposite; keep the one that provably matters."""
    counter = "off" if effort != "off" else "medium"
    sem = asyncio.Semaphore(4)

    async def probe(name: str, fl: Flavor) -> tuple[str, int, str]:
        async with sem:
            a, b = await asyncio.gather(
                send(fl.build(effort), fl.probe_max_tokens(effort)),
                send(fl.build(counter), fl.probe_max_tokens(counter)),
            )
        if not a.ok:
            return name, 0, f"rejected ({a.error})"
        if not b.ok:
            return name, 2, f"accepted; could not toggle to verify ({b.error})"
        if a.has_reasoning != b.has_reasoning:
            return name, 3, f"verified: reasoning {'on' if a.has_reasoning else 'off'} at {effort}, {'on' if b.has_reasoning else 'off'} at {counter}"
        return name, 1, "accepted but no observable effect"

    outcomes = await asyncio.gather(*(probe(n, f) for n, f in FLAVORS.items()))
    probes = {name: text for name, _, text in outcomes}
    for name, _, text in outcomes:
        log(f"probe {name:<10} {FLAVORS[name].param:<38} {text}")

    best = max(outcomes, key=lambda o: o[1])  # stable: first in FLAVORS order wins ties
    name, score, _ = best
    if score >= 3:
        return ThinkingStatus(effort, "auto", name, "verified", FLAVORS[name].build(effort), probes)
    if score == 2:
        return ThinkingStatus(effort, "auto", name, "unverified", FLAVORS[name].build(effort), probes)
    return ThinkingStatus(effort, "auto", None, "unsupported", {}, probes)
