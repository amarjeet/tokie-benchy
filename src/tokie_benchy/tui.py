from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, ProgressBar, RichLog, Static
from textual.worker import Worker
from textual_plotext import PlotextPlot

from tokie_benchy.client import RequestResult
from tokie_benchy.config_screen import ConfigScreen
from tokie_benchy.metrics import StageSummary
from tokie_benchy.report import summary_rows, write_json
from tokie_benchy.request import RunRequest
from tokie_benchy.runner import (
    BenchRunner,
    Event,
    LogEvent,
    RequestFinished,
    RunConfig,
    RunFinished,
    StageFinished,
    StageStarted,
    ThinkingResolved,
)
from tokie_benchy.thinking import ThinkingStatus

PROFILE_COLORS = {"L": "green", "M": "orange", "H": "red"}
PALETTE = ("green", "orange", "red", "cyan", "magenta", "blue", "yellow", "white")


_RICH_NAMES = {"orange": "dark_orange"}  # plotext and Rich disagree on this one


def rich_color(color: str) -> str:
    return _RICH_NAMES.get(color, color)


def stage_colors(names: list[str]) -> dict[str, str]:
    """Plain L/M/H keep their colors; sweep stages cycle the palette."""
    if all(n in PROFILE_COLORS for n in names):
        return {n: PROFILE_COLORS[n] for n in names}
    return {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}


class RunCommands(Provider):
    """Command-palette (ctrl+p) entries."""

    COMMANDS = (
        ("New run: configure settings and restart", "action_configure", "Open the settings dialog; starting cancels any run in progress"),
        ("Save results as JSON", "action_save", "Write per-request and per-stage results to a JSON file"),
    )

    async def discover(self) -> Hits:
        for name, action, help_text in self.COMMANDS:
            yield DiscoveryHit(name, getattr(self.app, action), help=help_text)

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        for name, action, help_text in self.COMMANDS:
            score = matcher.match(name)
            if score > 0:
                yield Hit(score, matcher.highlight(name), getattr(self.app, action), help=help_text)


class BenchApp(App[None]):
    TITLE = "tokie-benchy"
    CSS = """
    #body { height: 1fr; }
    #left { width: 68; min-width: 56; }
    #right { width: 1fr; }
    .panel { border: round $primary; padding: 0 1; }
    #config { height: auto; }
    PlotextPlot { height: 1fr; border: round $secondary; }
    #statusbar { height: 3; padding: 0 1; }
    #progress { width: 40; }
    #status { width: 1fr; padding: 0 2; content-align: left middle; }
    #log { height: 8; border: round $accent; }
    """
    BINDINGS = [
        ("n", "configure", "New run"),
        ("s", "save", "Save JSON"),
        ("d", "toggle_dark", "Dark/Light"),
        ("q", "quit", "Quit"),
    ]
    COMMANDS = App.COMMANDS | {RunCommands}

    def __init__(self, req: RunRequest, configure: bool = False):
        super().__init__()
        self.req = req
        self.cfg: RunConfig | None = None
        self.output = req.output
        self.configure_on_start = configure
        self.results: list[RequestResult] = []
        self.summaries: list[StageSummary] = []
        self.current: str = ""
        self.finished = False
        self.thinking: ThinkingStatus | None = None
        self.stage_color: dict[str, str] = {}
        self._worker: Worker | None = None

    # --- layout ---------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static(id="config", classes="panel")
                yield DataTable(id="stats", classes="panel", zebra_stripes=True, cursor_type="none")
                yield PlotextPlot(id="sweep_plot")
            with Vertical(id="right"):
                yield PlotextPlot(id="ttft_plot")
                yield PlotextPlot(id="tps_plot")
        with Horizontal(id="statusbar"):
            yield ProgressBar(id="progress", show_eta=False)
            yield Static("starting...", id="status")
        yield RichLog(id="log", max_lines=300, markup=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#stats", DataTable)
        table.add_columns("stage", "measure", "unit", "mean", "p50", "p95", "min", "max")
        for wid, title in (("#ttft_plot", "TTFT per request (ms)"), ("#tps_plot", "Decode tok/s per request"),
                           ("#sweep_plot", "per stage: tok/s (left) · TTFT p50 ms (right)")):
            self.query_one(wid, PlotextPlot).plt.title(title)
        if self.configure_on_start:
            self._status("press [b]n[/b] to configure a run · q quit")
            self.action_configure()
        else:
            self._start_run(self.req)

    def _config_text(self) -> str:
        if self.cfg is None:
            return "[dim]no run configured[/dim]"
        lines = ["[b]stages[/b]"]
        for p in self.cfg.stages:
            flags = " · exact" if p.exact_output else ""
            lines.append(
                f"[{rich_color(self.stage_color.get(p.name, 'white'))}]{p.name}[/] "
                f"prompt≈{p.prompt_tokens} max={p.max_tokens} n={p.requests} {p.load_label}{flags}"
            )
        lines.append(f"  [dim]{', '.join(self.cfg.stages[0].measurements)}[/dim]")
        lines.append(f"[dim]timeout {self.cfg.timeout:.0f}s · warmup {'on' if self.cfg.warmup else 'off'} · seed {self.cfg.seed}[/dim]")
        if self.output:
            lines.append(f"[dim]output {self.output}[/dim]")
        if self.cfg.extra_body:
            lines.append(f"[dim]extra body {self.cfg.extra_body}[/dim]")
        if self.thinking is None:
            lines.append(f"[yellow]thinking effort={self.cfg.reasoning_effort} · probing…[/yellow]")
        else:
            color = {"verified": "green", "explicit": "cyan", "unverified": "yellow", "unsupported": "red", "disabled": "dim"}[self.thinking.verdict]
            lines.append(f"[{color}]{self.thinking.summary}[/{color}]")
        lines.append(f"[dim]repro: {self.cfg.repro}[/dim]")
        return "\n".join(lines)

    # --- run lifecycle --------------------------------------------------------
    def action_configure(self) -> None:
        if isinstance(self.screen, ConfigScreen):
            return
        running = self._worker is not None and self._worker.is_running
        self.push_screen(ConfigScreen(self.req, running=running), callback=self._on_configured)

    def _on_configured(self, req: RunRequest | None) -> None:
        if req is not None:
            self._start_run(req)

    def _start_run(self, req: RunRequest) -> None:
        """Cancel any run in progress, reset the dashboard, and start `req`."""
        try:
            cfg = req.to_run_config()
        except ValueError as exc:
            self._log(f"[red]invalid configuration: {exc}[/]")
            return
        if self._worker is not None and self._worker.is_running:
            self._worker.cancel()
            self._log("[yellow]previous run cancelled[/]")
        self.req = req
        self.cfg = cfg
        self.output = req.output
        self.stage_color = stage_colors([p.name for p in cfg.stages])
        self._reset_dashboard()
        self.sub_title = f"{cfg.endpoint.model} @ {cfg.endpoint.base_url}"
        self.query_one("#config", Static).update(self._config_text())
        self._worker = self.run_worker(self._run(cfg), exclusive=True)

    def _reset_dashboard(self) -> None:
        self.results = []
        self.summaries = []
        self.current = ""
        self.finished = False
        self.thinking = None
        self.query_one("#stats", DataTable).clear()
        self.query_one("#log", RichLog).clear()
        self.query_one("#progress", ProgressBar).update(total=None, progress=0)
        self._status("starting...")
        for wid in ("#ttft_plot", "#tps_plot", "#sweep_plot"):
            widget = self.query_one(wid, PlotextPlot)
            widget.plt.clear_data()
            widget.refresh()

    # --- benchmark driver -----------------------------------------------------
    async def _run(self, cfg: RunConfig) -> None:
        try:
            await BenchRunner(cfg, self._on_event).run()
        except Exception as exc:  # surface instead of dying silently
            self._log(f"[red]fatal: {type(exc).__name__}: {exc}[/]")
            self._status("[red]failed[/] · n new run · q quit")

    def _on_event(self, ev: Event) -> None:
        if isinstance(ev, LogEvent):
            color = {"warn": "yellow", "error": "red"}.get(ev.level, "dim")
            self._log(f"[{color}]{ev.message}[/]")
        elif isinstance(ev, ThinkingResolved):
            self.thinking = ev.status
            self.query_one("#config", Static).update(self._config_text())
            color = {"verified": "green", "explicit": "cyan", "unverified": "yellow", "unsupported": "red", "disabled": "dim"}[ev.status.verdict]
            self._log(f"[{color}]{ev.status.summary}[/{color}]")
            if ev.status.verdict == "unsupported":
                self.notify("No thinking parameter had an effect on this endpoint; running without one.",
                            title="thinking unsupported", severity="warning", timeout=8)
        elif isinstance(ev, StageStarted):
            p = ev.profile
            self.current = p.name
            bar = self.query_one("#progress", ProgressBar)
            bar.update(total=p.requests, progress=0)
            self._log(
                f"[b]stage {ev.stage_index + 1}/{ev.stage_count}[/b] "
                f"[{rich_color(self.stage_color.get(p.name, 'white'))}]{p.name}[/] {p.load_label} — {p.description}"
            )
        elif isinstance(ev, RequestFinished):
            r = ev.result
            self.results.append(r)
            self.query_one("#progress", ProgressBar).update(progress=ev.done)
            agg = ev.tokens_so_far / ev.elapsed_s if ev.elapsed_s > 0 else 0
            self._status(
                f"[b]{self.current}[/b] {ev.done}/{ev.total} done · {ev.in_flight} in flight · "
                f"agg [b]{agg:,.0f}[/b] tok/s · {ev.elapsed_s:.1f}s"
            )
            if r.ok:
                self._log(
                    f"#{r.index:<3} ttft [cyan]{r.ttft_ms:>7,.0f}[/] ms · tps [magenta]{r.tps or 0:>6.1f}[/] · "
                    f"e2e {r.e2e_ms:>7,.0f} ms · out {r.output_tokens} tok · in {r.prompt_tokens} tok"
                )
            else:
                self._log(f"[red]#{r.index} failed: {r.error}[/]")
            self._redraw_charts()
        elif isinstance(ev, StageFinished):
            self.summaries.append(ev.summary)
            self._refresh_table()
            self._redraw_sweep()
            s = ev.summary
            self._log(
                f"[b]{s.profile.name} done[/b] · {s.ok} ok / {s.failed} failed · "
                f"throughput [b]{s.throughput_tps:,.0f}[/b] tok/s · {s.requests_per_s:.2f} req/s"
                + (f" · peak {s.peak_in_flight} in flight" if s.profile.open_loop else "") + f" · {s.wall_s:.1f}s"
            )
        elif isinstance(ev, RunFinished):
            self.finished = True
            self._status("[green b]done[/] · n new run · s save JSON · q quit")
            if self.output:
                self.action_save()

    # --- rendering ------------------------------------------------------------
    def _log(self, text: str) -> None:
        self.query_one("#log", RichLog).write(text)

    def _status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    def _refresh_table(self) -> None:
        table = self.query_one("#stats", DataTable)
        table.clear()
        for s in self.summaries:
            color = rich_color(self.stage_color.get(s.profile.name, "white"))
            for r in summary_rows(s):
                table.add_row(f"[{color}]{r['stage']}[/]", r["measure"], r["unit"], r["mean"], r["p50"], r["p95"], r["min"], r["max"])

    def _redraw_sweep(self) -> None:
        """Per-stage throughput (left axis) and TTFT p50 (right axis) against load."""
        widget = self.query_one("#sweep_plot", PlotextPlot)
        plt = widget.plt
        plt.clear_data()
        xs = list(range(1, len(self.summaries) + 1))
        labels = [s.profile.load_label.split()[0] for s in self.summaries]
        tps = [s.throughput_tps for s in self.summaries]
        ttft = [s.stats["ttft"].p50 if "ttft" in s.stats else 0 for s in self.summaries]
        plt.bar(xs, tps, color="cyan", width=0.4, label="tok/s")
        if len(xs) > 1:
            plt.plot(xs, ttft, yside="right", color="orange", marker="hd", label="ttft p50")
        else:
            plt.scatter(xs, ttft, yside="right", color="orange", marker="hd", label="ttft p50")
        plt.xticks(xs, labels)
        widget.refresh()

    def _redraw_charts(self) -> None:
        ok = [r for r in self.results if r.ok]
        if not ok:
            return
        ttft_w = self.query_one("#ttft_plot", PlotextPlot)
        tps_w = self.query_one("#tps_plot", PlotextPlot)
        for widget, key, kind in ((ttft_w, "ttft", "bar"), (tps_w, "tps", "scatter")):
            plt = widget.plt
            plt.clear_data()
            offset = 0
            for name in dict.fromkeys(r.profile for r in ok):  # keep stage order
                rows = [r for r in ok if r.profile == name]
                xs = [offset + i + 1 for i in range(len(rows))]
                ys = [r.metric(key) or 0 for r in rows]
                color = self.stage_color.get(name, "white")
                if kind == "bar":
                    plt.bar(xs, ys, label=name, color=color, width=0.6)
                else:
                    plt.scatter(xs, ys, label=name, color=color, marker="hd")
                    if len(xs) > 1:
                        plt.plot(xs, ys, color=color)
                offset += len(rows)
            plt.xlabel("request #")
            step = max(1, offset // 12)
            plt.xticks(list(range(1, offset + 1, step)))
            widget.refresh()

    # --- actions --------------------------------------------------------------
    def action_save(self) -> None:
        if not self.summaries or self.cfg is None:
            self._log("[yellow]nothing to save yet[/]")
            return
        path = self.output or Path("results") / f"tokie-{self.cfg.endpoint.model}-{'-'.join(s.profile.name for s in self.summaries)}.json"
        write_json(path, self.cfg, self.summaries, self.results, self.thinking)
        self._log(f"[green]saved {path}[/]")
