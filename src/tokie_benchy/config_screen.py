from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from tokie_benchy.profiles import ARRIVALS, MEASUREMENTS, PROFILES
from tokie_benchy.request import RunRequest
from tokie_benchy.thinking import EFFORTS, FLAVORS, STYLES


def _opt(value: int | float | None) -> str:
    return "" if value is None else str(value)


class ConfigScreen(ModalScreen[RunRequest | None]):
    """Modal form for every run setting. Dismisses with a RunRequest, or None on cancel."""

    BINDINGS = [Binding("escape", "cancel", "Cancel"), Binding("ctrl+s", "start", "Start")]
    DEFAULT_CSS = """
    ConfigScreen { align: center middle; }
    #dialog {
        width: 104; max-width: 96%; height: 90%; max-height: 92%;
        border: thick $primary; background: $surface; padding: 1 2;
    }
    #dialog > #title { text-style: bold; margin-bottom: 1; }
    #form { height: 1fr; }
    .stack { height: auto; }
    .stack Checkbox { width: 100%; }
    .row { height: 3; }
    .row Label { width: 18; content-align: left middle; height: 3; }
    .row Input, .row Select { width: 1fr; }
    .checks { height: 3; }
    .checks Checkbox { width: auto; }
    .section { color: $text-muted; margin-top: 1; }
    #error { color: $error; height: auto; min-height: 1; margin-top: 1; }
    #buttons { height: 3; align: right middle; margin-top: 1; }
    #buttons Button { margin-left: 2; }
    """

    def __init__(self, req: RunRequest, running: bool = False):
        super().__init__()
        self.req = req.copy()
        self.running = running

    # --- layout ---------------------------------------------------------------
    def compose(self) -> ComposeResult:
        r = self.req
        with Vertical(id="dialog"):
            yield Static("configure run" + ("  [dim](starting will cancel the current run)[/dim]" if self.running else ""), id="title")
            with VerticalScroll(id="form"):
                yield Label("profiles (run in order L → M → H)", classes="section")
                with Vertical(classes="stack"):
                    for name, p in PROFILES.items():
                        yield Checkbox(
                            f"{name}   n={p.requests}  c={p.concurrency}  prompt≈{p.prompt_tokens}  max={p.max_tokens}   [dim]{p.description}[/dim]",
                            value=name in r.profiles, id=f"prof_{name}",
                        )
                yield Label("overrides (blank = profile default, applied to every selected profile)", classes="section")
                yield from self._row("prompt tokens", Input(_opt(r.prompt_tokens), id="prompt_tokens", type="integer", placeholder="e.g. 32768"))
                yield from self._row("max tokens", Input(_opt(r.max_tokens), id="max_tokens", type="integer", placeholder="e.g. 256"))
                yield from self._row("requests (-n)", Input(_opt(r.requests), id="requests", type="integer", placeholder="e.g. 12"))
                yield from self._row("concurrency (-c)", Input(r.concurrency, id="concurrency", placeholder="e.g. 4, or a sweep 1,2,4,8"))
                yield Label("load pattern (leave rate blank for closed-loop concurrency)", classes="section")
                yield from self._row("request rate /s", Input(r.request_rate, id="request_rate", placeholder="open loop, e.g. 10 or a sweep 5,10,20"))
                yield from self._row("arrival", Select(((a, a) for a in ARRIVALS), value=r.arrival, allow_blank=False, id="arrival"))
                with Horizontal(classes="checks"):
                    yield Checkbox("exact output (ignore_eos + min_tokens = max tokens)", value=r.exact_output, id="exact_output")
                yield Label("measurements (none checked = profile default)", classes="section")
                keys = list(MEASUREMENTS)
                for chunk in (keys[:3], keys[3:]):
                    with Horizontal(classes="checks"):
                        for key in chunk:
                            yield Checkbox(key, value=bool(r.measurements and key in r.measurements), id=f"meas_{key}")
                yield Label("endpoint", classes="section")
                yield from self._row("base URL", Input(r.base_url, id="base_url", placeholder="http://host:port/v1"))
                yield from self._row("model", Input(r.model, id="model", placeholder="model id"))
                yield from self._row("api key", Input(r.api_key, id="api_key", password=True, placeholder="optional"))
                yield Label("thinking / reasoning budget", classes="section")
                yield from self._row("effort", Select(((e, e) for e in EFFORTS), value=r.reasoning_effort, allow_blank=False, id="reasoning_effort"))
                yield from self._row("param style", Select(
                    ((f"{st}  —  {FLAVORS[st].param}" if st in FLAVORS else ("auto  —  probe endpoint, pick what works" if st == "auto" else "none  —  send no thinking params"), st) for st in STYLES),
                    value=r.reasoning_style, allow_blank=False, id="reasoning_style"))
                yield from self._row("extra body JSON", Input(r.extra_body, id="extra_body", placeholder='merged into every request, e.g. {"top_p": 0.9}'))
                yield Label("run options", classes="section")
                yield from self._row("timeout (s)", Input(str(r.timeout), id="timeout", type="number"))
                yield from self._row("seed", Input(str(r.seed), id="seed", type="integer"))
                yield from self._row("temperature", Input(str(r.temperature), id="temperature", type="number"))
                yield from self._row("output JSON", Input(str(r.output) if r.output else "", id="output", placeholder="optional path, e.g. results/run.json"))
                with Horizontal(classes="checks"):
                    yield Checkbox("warmup request first", value=r.warmup, id="warmup")
            yield Static("", id="error")
            with Horizontal(id="buttons"):
                yield Button("cancel  (esc)", id="cancel")
                yield Button("start run  (ctrl+s)", variant="success", id="start")

    @staticmethod
    def _row(label: str, widget: Widget) -> ComposeResult:
        with Horizontal(classes="row"):
            yield Label(label)
            yield widget

    # --- actions --------------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start":
            self.action_start()
        else:
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def action_start(self) -> None:
        try:
            req = self._collect()
        except ValueError as exc:
            self.query_one("#error", Static).update(str(exc))
            return
        errors = req.validate()
        if errors:
            self.query_one("#error", Static).update(" · ".join(errors))
            return
        self.dismiss(req)

    def _collect(self) -> RunRequest:
        def text(wid: str) -> str:
            return self.query_one(f"#{wid}", Input).value.strip()

        def opt_int(wid: str, label: str) -> int | None:
            raw = text(wid)
            if not raw:
                return None
            try:
                return int(raw)
            except ValueError:
                raise ValueError(f"{label} must be an integer") from None

        def num(wid: str, label: str, cast):
            raw = text(wid)
            try:
                return cast(raw)
            except ValueError:
                raise ValueError(f"{label} must be a number") from None

        measurements = tuple(k for k in MEASUREMENTS if self.query_one(f"#meas_{k}", Checkbox).value)
        output = text("output")
        return RunRequest(
            profiles=[n for n in PROFILES if self.query_one(f"#prof_{n}", Checkbox).value],
            prompt_tokens=opt_int("prompt_tokens", "prompt tokens"),
            max_tokens=opt_int("max_tokens", "max tokens"),
            requests=opt_int("requests", "requests"),
            concurrency=text("concurrency"),
            request_rate=text("request_rate"),
            arrival=str(self.query_one("#arrival", Select).value),
            exact_output=self.query_one("#exact_output", Checkbox).value,
            measurements=measurements or None,
            model=text("model"),
            base_url=text("base_url"),
            api_key=text("api_key"),
            timeout=num("timeout", "timeout", float),
            warmup=self.query_one("#warmup", Checkbox).value,
            seed=num("seed", "seed", int),
            temperature=num("temperature", "temperature", float),
            output=Path(output) if output else None,
            reasoning_effort=str(self.query_one("#reasoning_effort", Select).value),
            reasoning_style=str(self.query_one("#reasoning_style", Select).value),
            extra_body=text("extra_body"),
        )
