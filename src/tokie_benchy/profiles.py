from __future__ import annotations

from dataclasses import dataclass, replace

MEASUREMENTS: dict[str, str] = {
    "ttft": "Time to first token (ms)",
    "tps": "Output tokens/s during decode",
    "itl": "Mean inter-token latency (ms)",
    "e2e": "End-to-end request latency (ms)",
    "prefill": "Prompt tokens/s (prompt tokens / TTFT)",
    "throughput": "Aggregate output tokens/s across concurrent requests",
}

PER_REQUEST = ("ttft", "tps", "itl", "e2e", "prefill")
UNITS = {"ttft": "ms", "tps": "tok/s", "itl": "ms", "e2e": "ms", "prefill": "tok/s", "throughput": "tok/s"}


ARRIVALS = ("constant", "poisson")


@dataclass(frozen=True)
class Profile:
    """A benchmark stage. `name` is the stage label ("M", or "M·c8" inside a sweep)."""

    name: str
    description: str
    prompt_tokens: int
    max_tokens: int
    requests: int
    concurrency: int
    measurements: tuple[str, ...]
    request_rate: float | None = None  # set => open loop: launch at this rate, no concurrency cap
    arrival: str = "constant"  # inter-arrival distribution in open-loop mode
    exact_output: bool = False  # send ignore_eos + min_tokens so every reply is exactly max_tokens

    @property
    def open_loop(self) -> bool:
        return self.request_rate is not None

    @property
    def load_label(self) -> str:
        if self.open_loop:
            return f"rate={self.request_rate:g}/s {self.arrival}"
        return f"c={self.concurrency}"

    def with_overrides(self, **changes) -> "Profile":
        """Return a copy with every non-None keyword applied."""
        return replace(self, **{k: v for k, v in changes.items() if v is not None})


PROFILES: dict[str, Profile] = {
    "L": Profile(
        name="L",
        description="Light: short prompts, sequential requests, quick sanity check",
        prompt_tokens=128,
        max_tokens=64,
        requests=6,
        concurrency=1,
        measurements=("ttft", "tps", "e2e"),
    ),
    "M": Profile(
        name="M",
        description="Medium: mid-size prompts, moderate parallelism",
        prompt_tokens=512,
        max_tokens=256,
        requests=12,
        concurrency=4,
        measurements=("ttft", "tps", "itl", "e2e", "throughput"),
    ),
    "H": Profile(
        name="H",
        description="Heavy: long prompts, high parallelism, stress test",
        prompt_tokens=2048,
        max_tokens=512,
        requests=24,
        concurrency=8,
        measurements=tuple(MEASUREMENTS),
    ),
}


def get_profile(name: str) -> Profile:
    try:
        return PROFILES[name.upper()]
    except KeyError:
        raise SystemExit(f"Unknown profile {name!r}. Choose from: {', '.join(PROFILES)}") from None


def parse_measurements(spec: str) -> tuple[str, ...]:
    keys = tuple(k.strip().lower() for k in spec.split(",") if k.strip())
    bad = [k for k in keys if k not in MEASUREMENTS]
    if bad:
        raise SystemExit(f"Unknown measurement(s): {', '.join(bad)}. Choose from: {', '.join(MEASUREMENTS)}")
    return keys
