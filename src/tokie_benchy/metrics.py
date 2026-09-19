from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev

from tokie_benchy.client import RequestResult
from tokie_benchy.profiles import PER_REQUEST, Profile


def percentile(values: list[float], pct: float) -> float:
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = (len(data) - 1) * pct / 100
    lo, hi = int(pos), min(int(pos) + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


@dataclass(frozen=True)
class Stat:
    n: int
    mean: float
    std: float
    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float

    @classmethod
    def of(cls, values: list[float]) -> "Stat | None":
        if not values:
            return None
        return cls(
            n=len(values),
            mean=mean(values),
            std=pstdev(values) if len(values) > 1 else 0.0,
            p50=percentile(values, 50),
            p90=percentile(values, 90),
            p95=percentile(values, 95),
            p99=percentile(values, 99),
            min=min(values),
            max=max(values),
        )


@dataclass
class StageSummary:
    profile: Profile
    wall_s: float
    ok: int
    failed: int
    total_prompt_tokens: int
    total_output_tokens: int
    stats: dict[str, Stat] = field(default_factory=dict)
    throughput_tps: float = 0.0
    requests_per_s: float = 0.0
    exact_hits: int = 0  # replies whose length equals max_tokens (meaningful with exact_output)
    peak_in_flight: int = 0

    def to_dict(self) -> dict:
        return {
            "profile": asdict(self.profile),
            "wall_s": self.wall_s,
            "ok": self.ok,
            "failed": self.failed,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "throughput_tps": self.throughput_tps,
            "requests_per_s": self.requests_per_s,
            "exact_hits": self.exact_hits,
            "peak_in_flight": self.peak_in_flight,
            "stats": {k: asdict(v) for k, v in self.stats.items()},
        }


def summarize(profile: Profile, results: list[RequestResult], wall_s: float, peak_in_flight: int = 0) -> StageSummary:
    good = [r for r in results if r.ok]
    summary = StageSummary(
        profile=profile,
        wall_s=wall_s,
        ok=len(good),
        failed=len(results) - len(good),
        total_prompt_tokens=sum(r.prompt_tokens for r in good),
        total_output_tokens=sum(r.output_tokens for r in good),
        exact_hits=sum(1 for r in good if r.output_tokens == profile.max_tokens),
        peak_in_flight=peak_in_flight,
    )
    for key in profile.measurements:
        if key in PER_REQUEST:
            values = [v for r in good if (v := r.metric(key)) is not None]
            stat = Stat.of(values)
            if stat:
                summary.stats[key] = stat
    if wall_s > 0:
        summary.throughput_tps = summary.total_output_tokens / wall_s
        summary.requests_per_s = len(good) / wall_s
    return summary
