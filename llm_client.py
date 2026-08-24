# -*- coding: utf-8 -*-
"""
Chat-completions LLM client for the curriculum-skills service.

Mirrors the backend configuration used by the future-technology-trends-identifier
service: a single Mistral model reached through an OpenAI/OpenWebUI-compatible
chat-completions endpoint.

Configuration (environment variables):
    API_URL      Base URL of the LLM backend (no trailing slash).
                   - Deployment: an OpenWebUI server, e.g. http://160.40.52.27:3000
                   - Development: a local Ollama OpenAI-compatible base, e.g.
                     http://ollama2:11434/v1
    API_TOKEN    Bearer token for the backend (only sent when non-empty; local
                 backends such as Ollama need no token).
    MODEL_NAME   Chat model name (default: mistral:latest).
    CHAT_PATH    Optional explicit chat path. Left empty it is auto-detected:
                   - base ending with /v1 (OpenAI-compatible, e.g. Ollama) ->
                     /chat/completions
                   - OpenWebUI-style base -> /api/chat/completions
    LLM_TEMPERATURE  Default sampling temperature (default: 0.0).
    LLM_TIMEOUT      Per-request timeout in seconds (default: 180).
    EXTRA_HEADERS_JSON  Optional JSON object of extra HTTP headers.

The public helper `chat_generate()` returns the assistant message content as a
plain string, so it is a drop-in replacement for the previous Ollama
`/api/generate` helpers. It raises on failure; callers may fall back to the
local Ollama native endpoint.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests


def _api_url() -> str:
    return (os.getenv("API_URL") or "http://localhost:3000").rstrip("/")


def _model_name(model: Optional[str] = None) -> str:
    return model or os.getenv("MODEL_NAME") or "mistral:latest"


def _timeout() -> int:
    try:
        return int(os.getenv("LLM_TIMEOUT", "180"))
    except (TypeError, ValueError):
        return 180


def _default_temperature() -> float:
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0.0"))
    except (TypeError, ValueError):
        return 0.0


def _chat_path() -> str:
    """Resolve the chat-completions path for the configured backend."""
    chat_path = (os.getenv("CHAT_PATH") or "").strip()
    if not chat_path:
        base = _api_url()
        chat_path = "/chat/completions" if base.endswith("/v1") else "/api/chat/completions"
    if not chat_path.startswith("/"):
        chat_path = "/" + chat_path
    return chat_path


def _chat_url() -> str:
    return f"{_api_url()}{_chat_path()}"


def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "curriculum-skills/1.0",
    }
    token = (os.getenv("API_TOKEN") or "").strip()
    if token:  # local backends need no token; only send one when configured
        headers["Authorization"] = f"Bearer {token}"
    extra = os.getenv("EXTRA_HEADERS_JSON")
    if extra:
        try:
            headers.update(dict(json.loads(extra)))
        except Exception:  # best-effort only
            pass
    return headers


def is_configured() -> bool:
    """True when a usable API_URL is configured (i.e. not the localhost default)."""
    return bool((os.getenv("API_URL") or "").strip())


def chat_generate(
    prompt: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    json_mode: bool = False,
    timeout: Optional[int] = None,
    session: Optional[requests.Session] = None,
    system: Optional[str] = None,
) -> str:
    """
    Send a single-turn chat completion and return the assistant content string.

    - Works against OpenAI-compatible (`/v1`) and OpenWebUI (`/api/...`) backends.
    - Sends a Bearer token when API_TOKEN is set.
    - An optional `system` message is prepended when provided.
    - When `json_mode` is True, requests a JSON object response
      (`response_format={"type": "json_object"}`), honoured by OpenWebUI and by
      the OpenAI-compatible endpoint.

    Raises on transport/HTTP errors or an unexpected response shape.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": _model_name(model),
        "messages": messages,
        "temperature": _default_temperature() if temperature is None else temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    poster = session.post if session is not None else requests.post
    resp = poster(
        _chat_url(),
        headers=_headers(),
        json=payload,
        timeout=timeout if timeout is not None else _timeout(),
    )
    resp.raise_for_status()

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"Unexpected chat-completions response shape: {exc}; "
            f"keys={list(data.keys()) if isinstance(data, dict) else type(data)}"
        ) from exc

    if not isinstance(content, str):
        raise ValueError("Chat model returned non-string content.")
    return content
