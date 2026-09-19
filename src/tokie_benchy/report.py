from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rich.table import Table

from tokie_benchy.client import RequestResult
from tokie_benchy.metrics import StageSummary
from tokie_benchy.profiles import MEASUREMENTS, UNITS
from tokie_benchy.runner import RunConfig
from tokie_benchy.thinking import ThinkingStatus

COLUMNS = ("mean", "std", "p50", "p90", "p95", "p99", "min", "max")


def fmt(value: float | None, key: str = "") -> str:
    if value is None:
        return "—"
    if key in ("ttft", "itl", "e2e") or value >= 100:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def summary_rows(summary: StageSummary) -> list[dict[str, str]]:
    """One dict per measurement: stage, measure, unit, plus every COLUMNS entry."""
    rows: list[dict[str, str]] = []
    name = summary.profile.name
    for key in summary.profile.measurements:
        row = {"stage": name, "measure": key, "unit": UNITS[key], **{c: "—" for c in COLUMNS}}
        if key == "throughput":
            row["mean"] = fmt(summary.throughput_tps)
        elif (st := summary.stats.get(key)) is not None:
            row.update({c: fmt(getattr(st, c), key) for c in COLUMNS})
        rows.append(row)
    return rows


def stage_line(s: StageSummary) -> str:
    p = s.profile
    failed = f" · [red]{s.failed} failed[/]" if s.failed else ""
    load = p.load_label + (f" · peak {s.peak_in_flight} in flight" if p.open_loop else "")
    exact = ""
    if p.exact_output:
        color = "green" if s.exact_hits == s.ok else "yellow"
        exact = f" · [{color}]exact {s.exact_hits}/{s.ok}[/]"
    return (
        f"[b]{p.name}[/b] prompt≈{p.prompt_tokens} max={p.max_tokens} n={p.requests} {load} · "
        f"[green]{s.ok} ok[/]{failed}{exact} · avg {s.total_prompt_tokens // max(s.ok, 1)} in / "
        f"{s.total_output_tokens // max(s.ok, 1)} out tok · {s.requests_per_s:.2f} req/s · "
        f"[b]{s.throughput_tps:,.0f}[/b] tok/s · {s.wall_s:.1f}s wall"
    )


def summary_table(summaries: list[StageSummary]) -> Table:
    table = Table(title="tokie-benchy results", header_style="bold cyan")
    for col in ("stage", "measure", "unit", *COLUMNS):
        table.add_column(col, justify="left" if col in ("stage", "measure", "unit") else "right")
    for s in summaries:
        for row in summary_rows(s):
            table.add_row(*row.values())
        table.add_section()
    table.caption = "\n".join(stage_line(s) for s in summaries)
    table.caption_justify = "left"
    return table


def sweep_table(summaries: list[StageSummary]) -> Table | None:
    """Load vs. throughput/latency, one row per stage. None unless there are 2+ stages."""
    if len(summaries) < 2:
        return None
    t = Table(title="load sweep", header_style="bold cyan")
    for col in ("stage", "load", "ok", "tok/s", "req/s", "ttft p50 ms", "ttft p99 ms", "tps p50", "peak in flight"):
        t.add_column(col, justify="left" if col in ("stage", "load") else "right")
    for s in summaries:
        ttft, tps = s.stats.get("ttft"), s.stats.get("tps")
        t.add_row(
            s.profile.name, s.profile.load_label, str(s.ok), f"{s.throughput_tps:,.0f}", f"{s.requests_per_s:.2f}",
            fmt(ttft.p50 if ttft else None, "ttft"), fmt(ttft.p99 if ttft else None, "ttft"),
            fmt(tps.p50 if tps else None), str(s.peak_in_flight),
        )
    return t


def write_json(path: Path, cfg: RunConfig, summaries: list[StageSummary], results: list[RequestResult],
               thinking: ThinkingStatus | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repro": cfg.repro,
        "endpoint": {"base_url": cfg.endpoint.base_url, "model": cfg.endpoint.model},
        "config": {"timeout": cfg.timeout, "warmup": cfg.warmup, "seed": cfg.seed, "temperature": cfg.temperature,
                   "reasoning_effort": cfg.reasoning_effort, "reasoning_style": cfg.reasoning_style, "extra_body": cfg.extra_body},
        "thinking": thinking.to_dict() if thinking else None,
        "measurements": MEASUREMENTS,
        "stages": [s.to_dict() for s in summaries],
        "requests": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
