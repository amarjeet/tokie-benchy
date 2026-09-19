from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ENV_URL_KEYS = ("OPENAI_API_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")


def load_env(env_file: str | None = None) -> Path | None:
    """Load a .env file without overriding variables already exported in the shell."""
    path = Path(env_file) if env_file else Path.cwd() / ".env"
    if path.is_file():
        load_dotenv(path, override=False)
        return path
    return None


@dataclass(frozen=True)
class Endpoint:
    base_url: str
    api_key: str
    model: str

    @property
    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def resolve_endpoint(base_url: str | None, api_key: str | None, model: str | None) -> Endpoint:
    url = base_url or next((os.environ[k] for k in ENV_URL_KEYS if os.environ.get(k)), None)
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    mdl = model or os.environ.get("OPENAI_MODEL")
    missing = [name for name, value in (("OPENAI_API_URL", url), ("OPENAI_MODEL", mdl)) if not value]
    if missing:
        raise SystemExit(
            f"Missing configuration: {', '.join(missing)}. "
            "Set env vars, put them in .env, or pass --base-url / --model."
        )
    return Endpoint(base_url=url, api_key=key, model=mdl)  # type: ignore[arg-type]
