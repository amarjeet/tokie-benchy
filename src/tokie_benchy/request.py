from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field, replace
from pathlib import Path

from tokie_benchy.config import ENV_URL_KEYS, Endpoint
from tokie_benchy.profiles import ARRIVALS, MEASUREMENTS, PROFILES, Profile, get_profile
from tokie_benchy.runner import RunConfig
from tokie_benchy.thinking import DEFAULT_EFFORT, EFFORTS, STYLES


def parse_list(spec: str | None, cast, label: str) -> tuple | None:
    """'1,2,4' -> (1, 2, 4). None/blank -> None."""
    if spec is None or not str(spec).strip():
        return None
    try:
        values = tuple(cast(x.strip()) for x in str(spec).split(",") if x.strip())
    except ValueError:
        raise ValueError(f"{label} must be a comma-separated list of numbers") from None
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{label} values must be > 0")
    return values


@dataclass
class RunRequest:
    """User-facing run settings: endpoint fields plus optional profile overrides.

    A `None` override means "use the profile's own value". Built from CLI flags
    or from the TUI settings dialog, then turned into a RunConfig.
    """

    profiles: list[str] = field(default_factory=lambda: ["M"])
    prompt_tokens: int | None = None
    max_tokens: int | None = None
    requests: int | None = None
    concurrency: str = ""  # "8" or a sweep "1,2,4,8"
    request_rate: str = ""  # open loop: "10" or a sweep "5,10,20" requests/s
    arrival: str = "constant"
    exact_output: bool = False
    measurements: tuple[str, ...] | None = None
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    timeout: float = 120.0
    warmup: bool = True
    seed: int = 42
    temperature: float = 0.7
    output: Path | None = None
    reasoning_effort: str = DEFAULT_EFFORT
    reasoning_style: str = "auto"
    extra_body: str = ""  # raw JSON object text, merged into every request body

    # --- parsing helpers ------------------------------------------------------
    def concurrencies(self) -> tuple[int, ...] | None:
        return parse_list(self.concurrency, int, "concurrency")

    def request_rates(self) -> tuple[float, ...] | None:
        return parse_list(self.request_rate, float, "request rate")

    def parsed_extra_body(self) -> dict:
        raw = self.extra_body.strip()
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra body is not valid JSON ({exc.msg} at col {exc.colno})") from None
        if not isinstance(data, dict):
            raise ValueError("extra body must be a JSON object")
        return data

    @classmethod
    def from_env(cls) -> "RunRequest":
        """Defaults with endpoint fields filled from the environment (missing ones stay empty)."""
        url = next((os.environ[k] for k in ENV_URL_KEYS if os.environ.get(k)), "")
        return cls(model=os.environ.get("OPENAI_MODEL", ""), base_url=url, api_key=os.environ.get("OPENAI_API_KEY", ""))

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.profiles:
            errors.append("select at least one profile")
        for name in self.profiles:
            if name.upper() not in PROFILES:
                errors.append(f"unknown profile {name!r}")
        for label, value in (("prompt tokens", self.prompt_tokens), ("max tokens", self.max_tokens), ("requests", self.requests)):
            if value is not None and value < 1:
                errors.append(f"{label} must be >= 1")
        for fn in (self.concurrencies, self.request_rates, self.parsed_extra_body):
            try:
                fn()
            except ValueError as exc:
                errors.append(str(exc))
        if self.arrival not in ARRIVALS:
            errors.append(f"arrival must be one of {', '.join(ARRIVALS)}")
        if self.measurements:
            bad = [m for m in self.measurements if m not in MEASUREMENTS]
            if bad:
                errors.append(f"unknown measurement(s): {', '.join(bad)}")
        if not self.base_url:
            errors.append("base URL is required (OPENAI_API_URL)")
        if not self.model:
            errors.append("model is required (OPENAI_MODEL)")
        if self.timeout <= 0:
            errors.append("timeout must be > 0")
        if self.reasoning_effort not in EFFORTS:
            errors.append(f"reasoning effort must be one of {', '.join(EFFORTS)}")
        if self.reasoning_style not in STYLES:
            errors.append(f"reasoning style must be one of {', '.join(STYLES)}")
        return errors

    # --- expansion ------------------------------------------------------------
    def stages(self) -> list[Profile]:
        """Expand profiles x sweep into ordered stages. A rate sweep wins over a concurrency sweep."""
        rates = self.request_rates()
        concs = self.concurrencies()
        out: list[Profile] = []
        for name in self.profiles:
            base = get_profile(name).with_overrides(
                prompt_tokens=self.prompt_tokens,
                max_tokens=self.max_tokens,
                requests=self.requests,
                measurements=self.measurements or None,
                arrival=self.arrival,
                exact_output=self.exact_output,
            )
            if rates:
                for r in rates:
                    label = base.name if len(rates) == 1 else f"{base.name}·r{r:g}"
                    out.append(base.with_overrides(name=label, request_rate=r))
            elif concs and len(concs) > 1:
                for c in concs:
                    out.append(base.with_overrides(name=f"{base.name}·c{c}", concurrency=c))
            else:
                out.append(base.with_overrides(concurrency=concs[0] if concs else None))
        return out

    def to_run_config(self) -> RunConfig:
        errors = self.validate()
        if errors:
            raise ValueError("; ".join(errors))
        return RunConfig(
            endpoint=Endpoint(base_url=self.base_url, api_key=self.api_key, model=self.model),
            stages=self.stages(),
            timeout=self.timeout,
            warmup=self.warmup,
            seed=self.seed,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            reasoning_style=self.reasoning_style,
            extra_body=self.parsed_extra_body(),
            repro=self.repro_command(),
        )

    def repro_command(self) -> str:
        """CLI invocation that reproduces this request (secrets and env-provided endpoint omitted)."""
        d = RunRequest()
        parts = ["tokie-benchy", "run"]
        for name in self.profiles:
            parts += ["-p", name]
        if self.prompt_tokens is not None:
            parts += ["--prompt-tokens", str(self.prompt_tokens)]
        if self.max_tokens is not None:
            parts += ["--max-tokens", str(self.max_tokens)]
        if self.requests is not None:
            parts += ["-n", str(self.requests)]
        if self.concurrency:
            parts += ["-c", self.concurrency]
        if self.request_rate:
            parts += ["--request-rate", self.request_rate]
        if self.arrival != d.arrival:
            parts += ["--arrival", self.arrival]
        if self.exact_output:
            parts.append("--exact-output")
        if self.measurements:
            parts += ["-m", ",".join(self.measurements)]
        if self.model and self.model != os.environ.get("OPENAI_MODEL"):
            parts += ["--model", self.model]
        env_url = next((os.environ[k] for k in ENV_URL_KEYS if os.environ.get(k)), None)
        if self.base_url and self.base_url != env_url:
            parts += ["--base-url", self.base_url]
        if self.timeout != d.timeout:
            parts += ["--timeout", f"{self.timeout:g}"]
        if not self.warmup:
            parts.append("--no-warmup")
        if self.seed != d.seed:
            parts += ["--seed", str(self.seed)]
        if self.temperature != d.temperature:
            parts += ["--temperature", f"{self.temperature:g}"]
        if self.reasoning_effort != d.reasoning_effort:
            parts += ["-r", self.reasoning_effort]
        if self.reasoning_style != d.reasoning_style:
            parts += ["--reasoning-style", self.reasoning_style]
        if self.extra_body.strip():
            parts += ["--extra-body", self.extra_body.strip()]
        if self.output:
            parts += ["-o", str(self.output)]
        return " ".join(shlex.quote(p) for p in parts)

    def copy(self) -> "RunRequest":
        return replace(self, profiles=list(self.profiles))
