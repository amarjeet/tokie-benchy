from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from tokie_benchy import __version__
from tokie_benchy.config import load_env
from tokie_benchy.profiles import ARRIVALS, MEASUREMENTS, PROFILES, parse_measurements
from tokie_benchy.report import summary_table, sweep_table, write_json
from tokie_benchy.request import RunRequest
from tokie_benchy.runner import (
    BenchRunner,
    Event,
    LogEvent,
    RequestFinished,
    RunConfig,
    StageFinished,
    StageStarted,
    ThinkingResolved,
)
from tokie_benchy.thinking import DEFAULT_EFFORT, EFFORTS, STYLES

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Benchmark tokens/s and time-to-first-token of an OpenAI-compatible endpoint.",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def run(
    profile: Annotated[list[str], typer.Option("--profile", "-p", help="Profile(s): L, M, H. Repeat to run several in sequence.")] = ["M"],
    prompt_tokens: Annotated[Optional[int], typer.Option(help="Approximate prompt size in tokens.")] = None,
    max_tokens: Annotated[Optional[int], typer.Option(help="Max output tokens per request.")] = None,
    requests: Annotated[Optional[int], typer.Option("--requests", "-n", help="Total requests per profile.")] = None,
    concurrency: Annotated[Optional[str], typer.Option("--concurrency", "-c", help="Parallel in-flight requests (closed loop). A comma list like 1,2,4,8 runs a sweep, one stage per level.")] = None,
    request_rate: Annotated[Optional[str], typer.Option(help="Open loop: launch this many requests/s regardless of completions, no concurrency cap. Comma list sweeps rates.")] = None,
    arrival: Annotated[str, typer.Option(help=f"Inter-arrival distribution for --request-rate: {', '.join(ARRIVALS)}.")] = "constant",
    exact_output: Annotated[bool, typer.Option("--exact-output", help="Send ignore_eos + min_tokens so every reply is exactly --max-tokens long (vLLM/SGLang/llama.cpp).")] = False,
    measure: Annotated[Optional[str], typer.Option("--measure", "-m", help=f"Comma list of: {', '.join(MEASUREMENTS)}")] = None,
    model: Annotated[Optional[str], typer.Option(envvar="OPENAI_MODEL", show_envvar=True)] = None,
    base_url: Annotated[Optional[str], typer.Option(help="OpenAI-compatible base URL ending in /v1.")] = None,
    api_key: Annotated[Optional[str], typer.Option(envvar="OPENAI_API_KEY", show_envvar=True, show_default=False)] = None,
    timeout: Annotated[float, typer.Option(help="Per-request timeout in seconds.")] = 120.0,
    warmup: Annotated[bool, typer.Option(help="Send one small warmup request first.")] = True,
    tui: Annotated[bool, typer.Option(help="Live TUI (disable for headless/CI output).")] = True,
    configure: Annotated[bool, typer.Option("--configure", "-C", help="Open the TUI settings dialog instead of starting right away.")] = False,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Write JSON results to this file.")] = None,
    seed: Annotated[int, typer.Option(help="Prompt generation seed.")] = 42,
    temperature: Annotated[float, typer.Option()] = 0.7,
    reasoning_effort: Annotated[str, typer.Option("--reasoning-effort", "-r", help=f"Thinking effort: {', '.join(EFFORTS)}. Translated to the provider's own parameter.")] = DEFAULT_EFFORT,
    reasoning_style: Annotated[str, typer.Option(help=f"Which parameter spelling to use: {', '.join(STYLES)}. 'auto' probes the endpoint; 'none' sends nothing.")] = "auto",
    extra_body: Annotated[Optional[str], typer.Option(help='Raw JSON object merged into every request body, e.g. \'{"top_p": 0.9}\'. Overrides detected thinking fields.')] = None,
    env_file: Annotated[Optional[str], typer.Option(help="Path to .env (default: ./.env). Exported vars always win.")] = None,
) -> None:
    """Run a benchmark against the configured endpoint."""
    load_env(env_file)
    req = RunRequest.from_env()
    req.profiles = [name.upper() for name in profile]
    req.prompt_tokens, req.max_tokens, req.requests = prompt_tokens, max_tokens, requests
    req.concurrency, req.request_rate, req.arrival, req.exact_output = concurrency or "", request_rate or "", arrival.lower(), exact_output
    req.measurements = parse_measurements(measure) if measure else None
    req.model = model or req.model
    req.base_url = base_url or req.base_url
    req.api_key = api_key or req.api_key
    req.timeout, req.warmup, req.seed, req.temperature, req.output = timeout, warmup, seed, temperature, output
    req.reasoning_effort, req.reasoning_style, req.extra_body = reasoning_effort.lower(), reasoning_style.lower(), extra_body or ""

    if tui:
        from tokie_benchy.tui import BenchApp

        if not configure:  # fail fast on bad flags; in configure mode the dialog handles it
            errors = req.validate()
            if errors:
                raise SystemExit("Invalid configuration: " + "; ".join(errors))
        BenchApp(req, configure=configure).run()
        return

    errors = req.validate()
    if errors:
        raise SystemExit("Invalid configuration: " + "; ".join(errors) + ". Set env vars, .env, or CLI flags.")
    _run_headless(req.to_run_config(), output)


def _run_headless(cfg: RunConfig, output: Path | None) -> None:
    import asyncio

    console.print(f"[bold]tokie-benchy[/] {__version__} · {cfg.endpoint.model} @ {cfg.endpoint.base_url}")
    console.print(f"[dim]repro: {cfg.repro}[/]")
    progress = Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[status]}"),
        TimeElapsedColumn(),
        console=console,
    )
    task_id: dict[str, int] = {}

    def on_event(ev: Event) -> None:
        if isinstance(ev, LogEvent):
            console.log(ev.message)
        elif isinstance(ev, ThinkingResolved):
            console.print(f"[cyan]{ev.status.summary}[/]")
        elif isinstance(ev, StageStarted):
            task_id[ev.profile.name] = progress.add_task(
                f"{ev.profile.name} {ev.profile.load_label}", total=ev.profile.requests, status=""
            )
        elif isinstance(ev, RequestFinished):
            agg = ev.tokens_so_far / ev.elapsed_s if ev.elapsed_s else 0
            progress.update(
                task_id[ev.result.profile],
                completed=ev.done,
                status=f"{ev.in_flight} in flight · {agg:,.0f} tok/s agg",
            )
            if not ev.result.ok:
                console.log(f"[red]#{ev.result.index} failed: {ev.result.error}[/]")
        elif isinstance(ev, StageFinished):
            progress.update(task_id[ev.summary.profile.name], status=f"{ev.summary.throughput_tps:,.0f} tok/s · {ev.summary.wall_s:.1f}s")

    runner = BenchRunner(cfg, on_event)
    with progress:
        asyncio.run(runner.run())
    console.print(summary_table(runner.summaries))
    if (sweep := sweep_table(runner.summaries)) is not None:
        console.print(sweep)
    if output:
        console.print(f"saved [green]{write_json(output, cfg, runner.summaries, runner.results, runner.thinking)}[/]")
    if any(s.failed for s in runner.summaries):
        raise typer.Exit(code=1)


@app.command()
def profiles() -> None:
    """List benchmark profiles and measurement keys."""
    from rich.table import Table

    t = Table(title="profiles", header_style="bold cyan")
    for col in ("name", "prompt tok", "max tok", "requests", "concurrency", "measurements", "description"):
        t.add_column(col)
    for p in PROFILES.values():
        t.add_row(p.name, str(p.prompt_tokens), str(p.max_tokens), str(p.requests), str(p.concurrency), ", ".join(p.measurements), p.description)
    console.print(t)

    m = Table(title="measurements", header_style="bold cyan")
    m.add_column("key")
    m.add_column("meaning")
    for k, v in MEASUREMENTS.items():
        m.add_row(k, v)
    console.print(m)


def _version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[Optional[bool], typer.Option("--version", callback=_version, is_eager=True)] = None,
) -> None:
    pass


if __name__ == "__main__":
    app()
