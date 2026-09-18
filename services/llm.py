"""Shared model plumbing for the service layer.

Client construction, retry/backoff, structured (JSON-schema) calls and
plain-text calls live here so each service only has to describe its own prompt
and schema. Two providers are supported, chosen by ``LLM_PROVIDER``:

- ``gemini`` (default) via the ``google-genai`` SDK.
- ``minimax`` via MiniMax's Anthropic-compatible Messages API.

Environment variables
---------------------
LLM_PROVIDER      (optional) "gemini" (default) or "minimax".
GEMINI_API_KEY    (required for the gemini provider) ``GOOGLE_API_KEY`` falls back.
GEMINI_MODEL      (optional) default Gemini model.
MINIMAX_API_KEY   (required for the minimax provider)
MINIMAX_BASE_URL  (optional) defaults to the outside-China endpoint.
MINIMAX_MODEL     (optional) default MiniMax model.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from core import config

logger = logging.getLogger(__name__)

GeminiClient = genai.Client

MAX_RETRIES = 4
RETRY_BASE_DELAY = 1.5
RETRY_MAX_DELAY = 8.0

ANTHROPIC_VERSION = "2023-06-01"

T = TypeVar("T")


class LLMError(Exception):
    """A model call failed or returned nothing usable."""


class LLMConfigurationError(LLMError):
    """The API key or model configuration is missing/invalid."""


class ModelOverloadedError(LLMError):
    """The model is at capacity upstream — worth retrying or falling back."""


# --------------------------------------------------------------------------- #
# MiniMax client
# --------------------------------------------------------------------------- #
class MiniMaxClient:
    """Thin holder for MiniMax's Anthropic-compatible Messages API."""

    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(timeout=60.0)

    def post_messages(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.http.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()
        return response.json()


Client = Union[GeminiClient, MiniMaxClient]

DEFAULT_MODEL = (
    config.minimax_model() if config.llm_provider() == "minimax" else config.gemini_model()
)


# --------------------------------------------------------------------------- #
# Client factory
# --------------------------------------------------------------------------- #
def _build_gemini_client() -> GeminiClient:
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


def _build_minimax_client() -> MiniMaxClient:
    api_key = config.minimax_key()
    if not api_key:
        raise LLMConfigurationError(
            "MINIMAX_API_KEY is not set. Add it to your environment or .env "
            "file before running this."
        )
    return MiniMaxClient(api_key=api_key, base_url=config.minimax_base_url())


def build_client() -> Client:
    """Construct a client for the configured provider (``LLM_PROVIDER``).

    Raises:
        LLMConfigurationError: the key is missing or the client cannot be built.
    """
    if config.llm_provider() == "minimax":
        return _build_minimax_client()
    return _build_gemini_client()


# --------------------------------------------------------------------------- #
# Generic retry / fallback wrapper
# --------------------------------------------------------------------------- #
# classify(exc) -> ("config" | "retry" | "fatal", message)
Classifier = Callable[[Exception], Tuple[str, str]]


def _with_retries(
    send: Callable[[], T], model: str, max_retries: int, classify: Classifier
) -> T:
    """Run ``send``, retrying rate limits, connection drops and 5xx responses.

    Anything that will not get better on a retry — a bad key, a model this key
    cannot use, a malformed request — is raised immediately.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return send()
        except LLMError:
            raise
        except Exception as exc:
            kind, message = classify(exc)
            if kind == "config":
                raise LLMConfigurationError(message) from exc
            if kind == "fatal":
                raise LLMError(message) from exc
            last_error = exc
            logger.warning("Model call failed (attempt %d): %s", attempt, message)

        if attempt < max_retries:
            # Exponential backoff with jitter: a capacity spike is usually
            # shared, so retrying in lockstep with everyone else does not help.
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            time.sleep(delay * (0.7 + 0.6 * random.random()))

    raise ModelOverloadedError(
        f"'{model}' did not respond after {max_retries} attempts. "
        f"Last error: {last_error}"
    )


def _model_chain(model: str, fallbacks: List[str]) -> List[str]:
    """The primary model followed by any configured fallbacks."""
    chain = [model]
    for name in fallbacks:
        if name and name not in chain:
            chain.append(name)
    return chain


def _run_with_fallback(
    make_send: Callable[[str], Callable[[], T]],
    model: str,
    max_retries: int,
    classify: Classifier,
    fallback_models: List[str],
    provider_label: str,
) -> T:
    """Try ``model``, then each fallback, when the model is overloaded.

    A 503/429 means that specific model is at capacity, not that the request
    is wrong — so the same call is retried against a less busy model before
    giving up.
    """
    chain = _model_chain(model, fallback_models)
    tried: List[str] = []
    last: Optional[Exception] = None

    for candidate in chain:
        try:
            return _with_retries(make_send(candidate), candidate, max_retries, classify)
        except ModelOverloadedError as exc:
            last = exc
            tried.append(candidate)
            if candidate != chain[-1]:
                logger.warning(
                    "%s is overloaded; falling back to the next model", candidate
                )
        except LLMConfigurationError as exc:
            # A model this key cannot use is worth skipping, but a bad key is not.
            if "not available to this key" not in str(exc) or candidate == chain[-1]:
                raise
            last = exc
            tried.append(candidate)

    raise ModelOverloadedError(
        f"{provider_label} is busy right now — "
        + ", ".join(tried)
        + " all returned no result. This is capacity on the provider's side, "
        "not a problem with your key or your input. Wait a moment and run it "
        "again, or set a different model."
    ) from last


# --------------------------------------------------------------------------- #
# Gemini: JSON Schema -> Gemini schema
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
# Gemini: error classification
# --------------------------------------------------------------------------- #
def _status_of(exc: Exception) -> Optional[int]:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


def _classify_gemini_error(model: str) -> Classifier:
    def classify(exc: Exception) -> Tuple[str, str]:
        if isinstance(exc, genai_errors.ClientError):
            status = _status_of(exc)
            if status in (401, 403):
                return (
                    "config",
                    f"Gemini rejected the API key ({status}). Check GEMINI_API_KEY "
                    "and that the Generative Language API is enabled for it.",
                )
            if status == 404:
                return (
                    "config",
                    f"Model '{model}' is not available to this key. Set "
                    "GEMINI_MODEL to a model you have access to.",
                )
            if status == 429:
                return "retry", str(exc)
            return "fatal", f"Gemini request failed ({status or 'client error'}): {exc}"
        if isinstance(exc, (genai_errors.ServerError, genai_errors.APIError)):
            return "retry", str(exc)
        name = type(exc).__name__.lower()
        if "timeout" in name or "connect" in name or "transport" in name:
            return "retry", str(exc)
        logger.exception("Unexpected error calling Gemini")
        return "fatal", f"Unexpected error calling the model: {exc}"

    return classify


# --------------------------------------------------------------------------- #
# Gemini: structured call
# --------------------------------------------------------------------------- #
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


def _parse_gemini_json_response(response: Any, name: str) -> Dict[str, Any]:
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


def _call_tool_gemini(
    client: GeminiClient,
    model: str,
    system: str,
    user_message: str,
    tool: Dict[str, Any],
    max_tokens: int,
    max_retries: int,
) -> Dict[str, Any]:
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

    def make_send(candidate: str) -> Callable[[], Dict[str, Any]]:
        def send() -> Dict[str, Any]:
            response = client.models.generate_content(
                model=candidate,
                contents=user_message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=instruction,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                ),
            )
            return _parse_gemini_json_response(response, name)

        return send

    return _run_with_fallback(
        make_send,
        model,
        max_retries,
        _classify_gemini_error(model),
        config.gemini_fallback_models(),
        "Gemini",
    )


def _call_text_gemini(
    client: GeminiClient,
    model: str,
    system: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    max_retries: int,
) -> str:
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

    def make_send(candidate: str) -> Callable[[], str]:
        def send() -> str:
            response = client.models.generate_content(
                model=candidate,
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

        return send

    return _run_with_fallback(
        make_send,
        model,
        max_retries,
        _classify_gemini_error(model),
        config.gemini_fallback_models(),
        "Gemini",
    )


# --------------------------------------------------------------------------- #
# MiniMax: error classification
# --------------------------------------------------------------------------- #
def _classify_minimax_error(model: str) -> Classifier:
    def classify(exc: Exception) -> Tuple[str, str]:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403):
                return (
                    "config",
                    f"MiniMax rejected the API key ({status}). Check MINIMAX_API_KEY.",
                )
            if status == 404:
                return (
                    "config",
                    f"Model '{model}' is not available to this key. Set "
                    "MINIMAX_MODEL to a model you have access to.",
                )
            if status == 429 or status >= 500:
                return "retry", exc.response.text or str(exc)
            return "fatal", f"MiniMax request failed ({status}): {exc.response.text}"
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return "retry", str(exc)
        logger.exception("Unexpected error calling MiniMax")
        return "fatal", f"Unexpected error calling the model: {exc}"

    return classify


# --------------------------------------------------------------------------- #
# MiniMax: structured and plain-text calls (Anthropic Messages API shape)
# --------------------------------------------------------------------------- #
def _extract_tool_input(body: Dict[str, Any], tool_name: str, label: str) -> Dict[str, Any]:
    stop_reason = str(body.get("stop_reason") or "")
    if stop_reason == "max_tokens":
        raise LLMError(
            "The response was cut off before it was complete. Try again with "
            "shorter input."
        )
    for block in body.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use" and (
            not tool_name or block.get("name") == tool_name
        ):
            result = block.get("input")
            if isinstance(result, dict):
                return result
    raise LLMError(f"The model returned nothing for {label}. Try again.")


def _extract_text(body: Dict[str, Any]) -> str:
    stop_reason = str(body.get("stop_reason") or "")
    chunks = [
        block.get("text", "")
        for block in body.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    text = "".join(chunks).strip()
    if not text:
        raise LLMError("The model returned an empty reply. Try again.")
    if stop_reason == "max_tokens":
        text += "\n\n_(Answer cut off — ask a narrower question.)_"
    return text


def _call_tool_minimax(
    client: MiniMaxClient,
    model: str,
    system: str,
    user_message: str,
    tool: Dict[str, Any],
    max_tokens: int,
    max_retries: int,
) -> Dict[str, Any]:
    name = tool.get("name", "the result")
    instruction = system
    if tool.get("description"):
        instruction += f"\n\nOutput contract ({name}): {tool['description']}"

    anthropic_tool = {
        "name": name,
        "description": tool.get("description", ""),
        "input_schema": tool["input_schema"],
    }

    def make_send(candidate: str) -> Callable[[], Dict[str, Any]]:
        def send() -> Dict[str, Any]:
            body = client.post_messages(
                {
                    "model": candidate,
                    "max_tokens": max_tokens,
                    "system": instruction,
                    "messages": [{"role": "user", "content": user_message}],
                    "tools": [anthropic_tool],
                    "tool_choice": {"type": "tool", "name": name},
                }
            )
            return _extract_tool_input(body, name, name)

        return send

    return _run_with_fallback(
        make_send,
        model,
        max_retries,
        _classify_minimax_error(model),
        config.minimax_fallback_models(),
        "MiniMax",
    )


def _call_text_minimax(
    client: MiniMaxClient,
    model: str,
    system: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    max_retries: int,
) -> str:
    turns = [
        {"role": turn.get("role"), "content": str(turn.get("content") or "")}
        for turn in messages
        if turn.get("role") in ("user", "assistant")
        and str(turn.get("content") or "").strip()
    ]
    if not turns:
        raise LLMError("There is nothing to send to the model.")

    def make_send(candidate: str) -> Callable[[], str]:
        def send() -> str:
            body = client.post_messages(
                {
                    "model": candidate,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": turns,
                }
            )
            return _extract_text(body)

        return send

    return _run_with_fallback(
        make_send,
        model,
        max_retries,
        _classify_minimax_error(model),
        config.minimax_fallback_models(),
        "MiniMax",
    )


# --------------------------------------------------------------------------- #
# Public API — dispatches to the client's provider
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
    service declares its output once; the schema is enforced by the provider's
    structured-output or tool-use mode, so there is no prose to parse.

    Raises:
        LLMConfigurationError: bad key, or a model this key cannot use.
        LLMError: the call failed, was truncated, or returned unusable JSON.
    """
    if isinstance(client, MiniMaxClient):
        return _call_tool_minimax(
            client, model, system, user_message, tool, max_tokens, max_retries
        )
    return _call_tool_gemini(
        client, model, system, user_message, tool, max_tokens, max_retries
    )


def call_text(
    client: Client,
    model: str,
    system: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 1500,
    max_retries: int = MAX_RETRIES,
) -> str:
    """Call the model for a plain-text reply.

    ``messages`` uses the ``{"role": "user"|"assistant", "content": str}`` shape.

    Raises:
        LLMConfigurationError: bad key, or a model this key cannot use.
        LLMError: the call failed or came back empty.
    """
    if isinstance(client, MiniMaxClient):
        return _call_text_minimax(client, model, system, messages, max_tokens, max_retries)
    return _call_text_gemini(client, model, system, messages, max_tokens, max_retries)


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
    "GeminiClient",
    "MiniMaxClient",
    "build_client",
    "call_tool",
    "call_text",
    "to_gemini_schema",
    "truncate",
    "limit_sentences",
    "LLMError",
    "LLMConfigurationError",
    "ModelOverloadedError",
    "DEFAULT_MODEL",
    "MAX_RETRIES",
    "RETRY_BASE_DELAY",
]
