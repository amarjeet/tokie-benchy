from __future__ import annotations

import asyncio
import contextlib
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import httpx

from tokie_benchy.client import RequestResult, deep_merge, stream_completion
from tokie_benchy.config import Endpoint
from tokie_benchy.metrics import StageSummary, summarize
from tokie_benchy.profiles import Profile
from tokie_benchy.prompts import build_prompt
from tokie_benchy.thinking import DEFAULT_EFFORT, ProbeResult, ThinkingStatus, explicit_status, probe_flavors


@dataclass(frozen=True)
class RunConfig:
    endpoint: Endpoint
    stages: list[Profile]
    timeout: float = 120.0
    warmup: bool = True
    seed: int = 42
    temperature: float = 0.7
    reasoning_effort: str = DEFAULT_EFFORT
    reasoning_style: str = "auto"  # auto | <flavor> | none
    extra_body: dict = field(default_factory=dict)
    repro: str = ""  # CLI command that reproduces this run


# --- events emitted to the UI -------------------------------------------------
@dataclass(frozen=True)
class LogEvent:
    message: str
    level: str = "info"  # info | warn | error


@dataclass(frozen=True)
class ThinkingResolved:
    status: ThinkingStatus


@dataclass(frozen=True)
class StageStarted:
    profile: Profile
    stage_index: int
    stage_count: int


@dataclass(frozen=True)
class RequestFinished:
    result: RequestResult
    done: int
    total: int
    in_flight: int
    elapsed_s: float
    tokens_so_far: int


@dataclass(frozen=True)
class StageFinished:
    summary: StageSummary
    results: list[RequestResult]


@dataclass(frozen=True)
class RunFinished:
    summaries: list[StageSummary]
    results: list[RequestResult]


Event = LogEvent | ThinkingResolved | StageStarted | RequestFinished | StageFinished | RunFinished
Emit = Callable[[Event], None]


class BenchRunner:
    def __init__(self, cfg: RunConfig, emit: Emit):
        self.cfg = cfg
        self.emit = emit
        self.summaries: list[StageSummary] = []
        self.results: list[RequestResult] = []
        self.thinking: ThinkingStatus | None = None
        self.extra: dict = {}

    async def run(self) -> list[StageSummary]:
        timeout = httpx.Timeout(self.cfg.timeout, connect=15.0)
        limits = httpx.Limits(max_connections=max(p.concurrency for p in self.cfg.stages) + 4)
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            self.thinking = await self._resolve_thinking(client)
            # user --extra-body always wins over the detected thinking fields
            self.extra = deep_merge(self.thinking.body, self.cfg.extra_body)
            self.emit(ThinkingResolved(self.thinking))
            if self.cfg.warmup:
                await self._warmup(client)
            for i, profile in enumerate(self.cfg.stages):
                self.emit(StageStarted(profile, i, len(self.cfg.stages)))
                results, wall, peak = await self._run_stage(client, profile)
                summary = summarize(profile, results, wall, peak)
                self.summaries.append(summary)
                self.results.extend(results)
                if profile.exact_output and summary.ok:
                    if summary.exact_hits == summary.ok:
                        self.emit(LogEvent(f"exact output: all {summary.ok} replies hit {profile.max_tokens} tokens"))
                    else:
                        self.emit(LogEvent(
                            f"exact output NOT honoured: only {summary.exact_hits}/{summary.ok} replies hit "
                            f"{profile.max_tokens} tokens (server may ignore ignore_eos/min_tokens)", "warn"))
                self.emit(StageFinished(summary, results))
        self.emit(RunFinished(self.summaries, self.results))
        return self.summaries

    async def _resolve_thinking(self, client: httpx.AsyncClient) -> ThinkingStatus:
        cfg = self.cfg
        if cfg.reasoning_style != "auto":
            return explicit_status(cfg.reasoning_effort, cfg.reasoning_style)
        self.emit(LogEvent(f"probing thinking controls for effort={cfg.reasoning_effort}..."))

        async def send(extra: dict, max_tokens: int) -> ProbeResult:
            res = await stream_completion(
                client, cfg.endpoint,
                prompt="Reply with the single word: ok",
                max_tokens=max_tokens, index=-2, profile="probe", concurrency=1,
                temperature=cfg.temperature, extra=deep_merge(extra, cfg.extra_body),
            )
            return ProbeResult(res.ok, res.has_reasoning, res.error)

        return await probe_flavors(send, cfg.reasoning_effort, lambda m: self.emit(LogEvent(m)))

    async def _warmup(self, client: httpx.AsyncClient) -> None:
        self.emit(LogEvent("warmup request..."))
        res = await stream_completion(
            client,
            self.cfg.endpoint,
            prompt=build_prompt(32, self.cfg.seed - 1),
            max_tokens=8,
            index=-1,
            profile="warmup",
            concurrency=1,
            temperature=self.cfg.temperature,
            extra=self.extra,
        )
        if res.ok:
            self.emit(LogEvent(f"warmup ok · ttft {res.ttft_ms:.0f} ms"))
        else:
            self.emit(LogEvent(f"warmup failed: {res.error}", "warn"))

    async def _run_stage(self, client: httpx.AsyncClient, profile: Profile) -> tuple[list[RequestResult], float, int]:
        """Closed loop: hold `concurrency` requests in flight. Open loop: launch at `request_rate`."""
        gate = contextlib.nullcontext() if profile.open_loop else asyncio.Semaphore(profile.concurrency)
        extra = self.extra
        if profile.exact_output:
            extra = deep_merge(extra, {"ignore_eos": True, "min_tokens": profile.max_tokens})
        done = in_flight = peak = tokens = 0
        t0 = perf_counter()

        async def one(i: int) -> RequestResult:
            nonlocal done, in_flight, peak, tokens
            async with gate:
                in_flight += 1
                peak = max(peak, in_flight)
                res = await stream_completion(
                    client,
                    self.cfg.endpoint,
                    prompt=build_prompt(profile.prompt_tokens, self.cfg.seed + i),
                    max_tokens=profile.max_tokens,
                    index=i,
                    profile=profile.name,
                    concurrency=profile.concurrency,
                    temperature=self.cfg.temperature,
                    extra=extra,
                )
                in_flight -= 1
                done += 1
                tokens += res.output_tokens
                self.emit(RequestFinished(res, done, profile.requests, in_flight, perf_counter() - t0, tokens))
                return res

        async def at(delay: float, i: int) -> RequestResult:
            await asyncio.sleep(delay)
            return await one(i)

        if profile.open_loop:
            rate = profile.request_rate or 1.0
            rng = random.Random(self.cfg.seed)
            delays, t = [], 0.0
            for _ in range(profile.requests):
                delays.append(t)
                t += rng.expovariate(rate) if profile.arrival == "poisson" else 1.0 / rate
            tasks = [at(d, i) for i, d in enumerate(delays)]
        else:
            tasks = [one(i) for i in range(profile.requests)]

        results = await asyncio.gather(*tasks)
        return sorted(results, key=lambda r: r.index), perf_counter() - t0, peak
