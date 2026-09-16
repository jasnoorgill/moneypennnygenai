"""Shared Gemini plumbing for the service layer.

Client construction, retry/backoff, structured (JSON-schema) calls and
plain-text calls live here so each service only has to describe its own prompt
and schema.

Environment variables
---------------------
GEMINI_API_KEY   (required) ``GOOGLE_API_KEY`` is accepted as a fallback.
GEMINI_MODEL     (optional) default model for every service.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from core import config

logger = logging.getLogger(__name__)

Client = genai.Client

DEFAULT_MODEL = config.gemini_model()
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.5

T = TypeVar("T")


class LLMError(Exception):
    """A model call failed or returned nothing usable."""


class LLMConfigurationError(LLMError):
    """The API key or model configuration is missing/invalid."""


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def build_client() -> Client:
    """Construct a Gemini client from the environment.

    Raises:
        LLMConfigurationError: the key is missing or the client cannot be built.
    """
    api_key = config.gemini_key()
    if not api_key:
        raise LLMConfigurationError(
            "GEMINI_API_KEY is not set. Add it to your environment or .env "
            "file before running this."
        )
    try:
        return genai.Client(api_key=api_key)
    except Exception as exc:
        raise LLMConfigurationError(
            f"Could not initialise the Gemini client: {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Retry wrapper
# --------------------------------------------------------------------------- #
def _status_of(exc: Exception) -> Optional[int]:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _with_retries(send: Callable[[], T], model: str, max_retries: int) -> T:
    """Run ``send``, retrying rate limits, connection drops and 5xx responses.

    Anything that will not get better on a retry — a bad key, a model this key
    cannot use, a malformed request — is raised immediately.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return send()

        except genai_errors.ServerError as exc:
            last_error = exc
            logger.warning("Gemini server error (attempt %d): %s", attempt, exc)
        except genai_errors.ClientError as exc:
            last_error = exc
            status = _status_of(exc)
            if status in (401, 403):
                raise LLMConfigurationError(
                    f"Gemini rejected the API key ({status}). Check GEMINI_API_KEY "
                    "and that the Generative Language API is enabled for it."
                ) from exc
            if status == 404:
                raise LLMConfigurationError(
                    f"Model '{model}' is not available to this key. Set "
                    "GEMINI_MODEL to a model you have access to."
                ) from exc
            if status == 429:
                logger.warning("Gemini rate limited (attempt %d)", attempt)
            else:
                raise LLMError(
                    f"Gemini request failed ({status or 'client error'}): {exc}"
                ) from exc
        except LLMError:
            raise
        except genai_errors.APIError as exc:
            last_error = exc
            logger.warning("Gemini API error (attempt %d): %s", attempt, exc)
        except Exception as exc:
            # Transport-level failures (httpx timeouts, DNS, reset) are worth a
            # retry; anything else is a bug and should surface immediately.
            name = type(exc).__name__.lower()
            if "timeout" in name or "connect" in name or "transport" in name:
                last_error = exc
                logger.warning("Gemini call failed (attempt %d): %s", attempt, exc)
            else:
                logger.exception("Unexpected error calling Gemini")
                raise LLMError(f"Unexpected error calling the model: {exc}") from exc

        if attempt < max_retries:
            time.sleep(RETRY_BASE_DELAY * (2 ** (attempt - 1)))

    raise LLMError(
        f"Gemini was unreachable after {max_retries} attempts: {last_error}"
    )


# --------------------------------------------------------------------------- #
# JSON Schema -> Gemini schema
# --------------------------------------------------------------------------- #
# Gemini accepts an OpenAPI-3.0 subset with snake_case field names. Passing our
# JSON Schema straight through silently drops camelCase constraints such as
# minItems, so it is converted explicitly.
_SCHEMA_KEY_MAP = {
    "minItems": "min_items",
    "maxItems": "max_items",
    "minLength": "min_length",
    "maxLength": "max_length",
    "minProperties": "min_properties",
    "maxProperties": "max_properties",
    "anyOf": "any_of",
    "propertyOrdering": "property_ordering",
}
_SCHEMA_PASSTHROUGH = {
    "type", "description", "enum", "format", "nullable", "pattern",
    "minimum", "maximum", "title", "default",
}
_DROPPED_KEYS = {"additionalProperties", "$schema", "$id", "examples", "const"}


def to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a JSON Schema dict into the shape Gemini expects."""
    out: Dict[str, Any] = {}
    for key, value in (schema or {}).items():
        if key in _DROPPED_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            out["properties"] = {k: to_gemini_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out["items"] = to_gemini_schema(value)
        elif key == "required" and isinstance(value, (list, tuple)):
            out["required"] = list(value)
        elif key in _SCHEMA_KEY_MAP:
            target = _SCHEMA_KEY_MAP[key]
            out[target] = (
                [to_gemini_schema(v) for v in value]
                if target == "any_of" and isinstance(value, list)
                else value
            )
        elif key in _SCHEMA_PASSTHROUGH:
            out[key] = value
        else:
            logger.debug("Dropping unsupported schema key %r", key)
    return out


# --------------------------------------------------------------------------- #
# Structured call
# --------------------------------------------------------------------------- #
def call_tool(
    client: Client,
    model: str,
    system: str,
    user_message: str,
    tool: Dict[str, Any],
    max_tokens: int = 2048,
    max_retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """Call the model constrained to ``tool``'s schema, and return the JSON.

    ``tool`` keeps the ``{"name", "description", "input_schema"}`` shape so each
    service declares its output once; the schema is enforced by Gemini's
    structured-output mode, so there is no prose to parse.

    Raises:
        LLMConfigurationError: bad key, or a model this key cannot use.
        LLMError: the call failed, was truncated, or returned unusable JSON.
    """
    name = tool.get("name", "the result")
    try:
        response_schema = genai_types.Schema(
            **to_gemini_schema(tool["input_schema"])
        )
    except Exception as exc:
        raise LLMError(f"Invalid schema for '{name}': {exc}") from exc

    instruction = system
    if tool.get("description"):
        instruction += f"\n\nOutput contract ({name}): {tool['description']}"

    def send() -> Dict[str, Any]:
        response = client.models.generate_content(
            model=model,
            contents=user_message,
            config=genai_types.GenerateContentConfig(
                system_instruction=instruction,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
        return _parse_json_response(response, name)

    return _with_retries(send, model, max_retries)


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    return str(getattr(candidates[0], "finish_reason", "") or "")


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text:
        return text.strip()
    # Fall back to walking the parts if .text is unavailable.
    chunks: List[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "text", None):
                chunks.append(part.text)
    return "".join(chunks).strip()


def _parse_json_response(response: Any, name: str) -> Dict[str, Any]:
    reason = _finish_reason(response).upper()
    if "MAX_TOKENS" in reason:
        raise LLMError(
            "The response was cut off before it was complete. Try again with "
            "shorter input."
        )
    if "SAFETY" in reason or "BLOCK" in reason or "PROHIBITED" in reason:
        raise LLMError(
            f"Gemini blocked the response ({reason}). Rephrase the input and "
            "try again."
        )

    text = _response_text(response)
    if not text:
        raise LLMError(
            f"The model returned nothing for {name}. Try again."
        )

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return parsed

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Model returned malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError(
            f"Expected a JSON object for {name}, got {type(data).__name__}."
        )
    return data


# --------------------------------------------------------------------------- #
# Plain-text call
# --------------------------------------------------------------------------- #
def call_text(
    client: Client,
    model: str,
    system: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1500,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Call the model for a plain-text reply.

    ``messages`` uses the ``{"role": "user"|"assistant", "content": str}`` shape;
    it is translated to Gemini's ``user``/``model`` turns here so callers do not
    have to care.

    Raises:
        LLMConfigurationError: bad key, or a model this key cannot use.
        LLMError: the call failed or came back empty.
    """
    contents = [
        {
            "role": "model" if turn.get("role") == "assistant" else "user",
            "parts": [{"text": str(turn.get("content") or "")}],
        }
        for turn in messages
        if str(turn.get("content") or "").strip()
    ]
    if not contents:
        raise LLMError("There is nothing to send to the model.")

    def send() -> str:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        reason = _finish_reason(response).upper()
        if "SAFETY" in reason or "BLOCK" in reason or "PROHIBITED" in reason:
            raise LLMError(
                f"Gemini blocked the response ({reason}). Rephrase and try again."
            )
        text = _response_text(response)
        if not text:
            raise LLMError("The model returned an empty reply. Try again.")
        if "MAX_TOKENS" in reason:
            text += "\n\n_(Answer cut off — ask a narrower question.)_"
        return text

    return _with_retries(send, model, max_retries)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def truncate(text: Any, limit: int) -> str:
    """Collapse whitespace and cut ``text`` to ``limit`` characters."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def limit_sentences(text: Any, count: int = 2) -> str:
    """Trim ``text`` to at most ``count`` sentences."""
    text = " ".join(str(text).split())
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    trimmed = " ".join(parts[:count]).strip()
    if trimmed and trimmed[-1] not in ".!?":
        trimmed += "."
    return trimmed


__all__ = [
    "Client",
    "build_client",
    "call_tool",
    "call_text",
    "to_gemini_schema",
    "truncate",
    "limit_sentences",
    "LLMError",
    "LLMConfigurationError",
    "DEFAULT_MODEL",
    "MAX_RETRIES",
    "RETRY_BASE_DELAY",
]
