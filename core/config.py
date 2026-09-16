"""Environment configuration, loaded once for the whole application.

Every module that reads an environment variable imports this one, so ``.env``
is guaranteed to have been loaded before the first read no matter which module
Python imports first.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

try:  # optional: handy locally, unnecessary when the platform injects env vars
    from pathlib import Path

    from dotenv import find_dotenv, load_dotenv

    # Look next to the project first, then from the working directory, so the
    # app picks up .env whether it is launched from the repo root or elsewhere
    # (`streamlit run /path/to/app.py`).
    _PROJECT_ENV = Path(__file__).resolve().parent.parent / ".env"
    _DOTENV_PATH = str(_PROJECT_ENV) if _PROJECT_ENV.is_file() else find_dotenv(usecwd=True)
    DOTENV_LOADED = bool(_DOTENV_PATH) and load_dotenv(_DOTENV_PATH)
except Exception:  # pragma: no cover - python-dotenv missing or unreadable file
    _DOTENV_PATH = ""
    DOTENV_LOADED = False


class MissingConfiguration(RuntimeError):
    """A required environment variable is not set."""


def env(name: str, *fallbacks: str, default: str = "") -> str:
    """First non-empty value among ``name`` and ``fallbacks``, else ``default``."""
    for key in (name, *fallbacks):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return default


def require(name: str, *fallbacks: str, hint: str = "") -> str:
    """Like :func:`env`, but raises when nothing is set.

    Raises:
        MissingConfiguration: none of the names are set.
    """
    value = env(name, *fallbacks)
    if not value:
        names = " / ".join((name, *fallbacks))
        raise MissingConfiguration(
            f"{names} is not set." + (f" {hint}" if hint else "")
        )
    return value


# --------------------------------------------------------------------------- #
# Feature requirements — used to tell the user what is missing, up front
# --------------------------------------------------------------------------- #
#   feature -> (env var names, any one of which satisfies it)
FEATURE_KEYS: Dict[str, Tuple[str, ...]] = {
    "Supabase": ("SUPABASE_URL",),
    "Supabase key": ("SUPABASE_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_ANON_KEY"),
    "Gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "Tavily (partner research)": ("TAVILY_API_KEY",),
}


def missing_settings() -> List[str]:
    """Names of features whose environment variables are not configured."""
    return [
        feature
        for feature, keys in FEATURE_KEYS.items()
        if not any((os.getenv(key) or "").strip() for key in keys)
    ]


# Convenience accessors. Read lazily so a variable set after import still works.
def gemini_model() -> str:
    return env("GEMINI_MODEL", default="gemini-3.8-flash")


def gemini_fallback_models() -> List[str]:
    """Models to try when the primary one is overloaded, in order.

    Setting the variable to an empty string disables fallbacks entirely; only an
    unset variable gets the default chain.
    """
    raw = os.getenv("GEMINI_FALLBACK_MODELS")
    if raw is None:
        raw = "gemini-3.6-flash,gemini-2.5-flash"
    return [name.strip() for name in raw.split(",") if name.strip()]


def supabase_url() -> str:
    return env("SUPABASE_URL")


def supabase_key() -> str:
    return env("SUPABASE_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_ANON_KEY")


def tavily_key() -> str:
    return env("TAVILY_API_KEY")


def gemini_key() -> str:
    return env("GEMINI_API_KEY", "GOOGLE_API_KEY")


if DOTENV_LOADED:
    logger.info("Loaded environment from %s", _DOTENV_PATH)


__all__ = [
    "env",
    "require",
    "missing_settings",
    "MissingConfiguration",
    "FEATURE_KEYS",
    "DOTENV_LOADED",
    "gemini_model",
    "gemini_fallback_models",
    "gemini_key",
    "supabase_url",
    "supabase_key",
    "tavily_key",
]
