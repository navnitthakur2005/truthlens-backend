"""Configuration. Secrets come only from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BACKEND_DIR / ".env")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _clean_key(raw: str) -> str:
    """Tolerate quotes / 'Bearer ' / stray whitespace that often sneak into .env files."""
    k = (raw or "").strip().strip("\"'").strip()
    if k.lower().startswith("bearer "):
        k = k[7:].strip()
    return k


def _model_list(raw: str) -> list[str]:
    return [m.strip() for m in (raw or "").split(",") if m.strip()]


@dataclass
class Config:
    groq_api_key: str = field(default_factory=lambda: _clean_key(os.getenv("GROQ_API_KEY", "")))
    # Tried in order when the main model is rate-limited, too big for the token limit, or rejects the request.
    groq_fallback_models: list = field(
        default_factory=lambda: _model_list(os.getenv("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,groq/compound-mini"))
    )
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip())
    groq_api_base: str = field(
        default_factory=lambda: os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1/chat/completions")
    )
    request_timeout: float = field(default_factory=lambda: _float("GROQ_TIMEOUT", 90.0))

    # input limits
    min_chars: int = 10
    max_chars: int = 8000


def get_config() -> Config:
    return Config()