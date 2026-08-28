"""Thin adapter for a non-streaming OpenAI-compatible Chat Completions API.

The ordinary provider client is used only for HTTPS, authentication and wire
serialization.  This module does not use an Agent SDK and deliberately owns no
conversation history, tool execution, retry loop, or termination policy.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from coding_agent.errors import (
    ConfigurationError,
    ContextOverflow,
    PermanentModelError,
    TransientModelError,
)
from coding_agent.response_parser import normalize_chat_completion
from coding_agent.types import Message, ModelTurn

ToolSchema = Mapping[str, Any]


@runtime_checkable
class ModelClient(Protocol):
    """Minimal dependency required by the agent controller and test fakes."""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        timeout_seconds: float | None = None,
    ) -> ModelTurn:
        """Return one complete, non-streaming assistant turn.

        ``timeout_seconds`` is a per-attempt cap supplied by the controller's
        remaining wall-time budget.  Concrete adapters must not silently start
        their own retry loop behind this boundary.
        """


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    # Common API-key prefix.  A deliberately conservative minimum length avoids
    # replacing ordinary prose such as "sk-test" in diagnostic messages.
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
)


def _redact_error_text(text: str, api_key: str | None) -> str:
    """Remove likely credentials before an SDK error crosses into event logs."""

    sanitized = text
    if api_key:
        sanitized = sanitized.replace(api_key, "[REDACTED]")
    sanitized = _SECRET_PATTERNS[0].sub(r"\1[REDACTED]", sanitized)
    sanitized = _SECRET_PATTERNS[1].sub("[REDACTED]", sanitized)
    return sanitized


def _looks_like_context_overflow(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "context_length_exceeded",
            "maximum context length",
            "max context length",
            "context window",
            "too many tokens",
        )
    )


def _translate_request_error(exc: Exception, api_key: str | None) -> Exception | None:
    """Map known transport/provider errors without swallowing programming bugs.

    The mapping primarily uses HTTP status codes, which are more stable across
    compatible client versions than concrete exception classes.  Class names
    cover connection failures that have no response status and also make the
    function straightforward to test with dependency-free fake exceptions.
    Returning ``None`` means the exception is not recognized as an API/transport
    failure and should be re-raised unchanged.
    """

    class_name = type(exc).__name__
    module_name = type(exc).__module__
    status = getattr(exc, "status_code", None)
    raw_text = str(exc)
    safe_text = _redact_error_text(raw_text, api_key)
    detail = f"{class_name}: {safe_text}" if safe_text else class_name

    if isinstance(status, int):
        # Authentication/authorization is always permanent.  Status must take
        # precedence over message keywords: a 401 error saying that an account
        # cannot access a model's "context window" is not repairable by
        # truncating the prompt.
        if status in {401, 403}:
            return PermanentModelError(
                f"model request was rejected (HTTP {status}): {detail}"
            )
        if status in {400, 413, 422} and _looks_like_context_overflow(raw_text):
            return ContextOverflow(f"model rejected the context: {detail}")
        if status in {408, 409, 425, 429} or 500 <= status <= 599:
            return TransientModelError(
                f"temporary model request failure (HTTP {status}): {detail}"
            )
        if 400 <= status <= 499:
            return PermanentModelError(
                f"model request was rejected (HTTP {status}): {detail}"
            )

    if _looks_like_context_overflow(raw_text):
        return ContextOverflow(f"model rejected the context: {detail}")

    if isinstance(exc, (TimeoutError, ConnectionError)) or class_name in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
    }:
        return TransientModelError(f"temporary model request failure: {detail}")

    if class_name in {
        "AuthenticationError",
        "PermissionDeniedError",
        "BadRequestError",
        "NotFoundError",
        "ConflictError",
        "UnprocessableEntityError",
    }:
        return PermanentModelError(f"model request was rejected: {detail}")

    # Unknown exceptions from the OpenAI package still represent provider
    # failures, but classifying them as permanent is the conservative choice:
    # the controller will not burn its retry budget on an unrecognized error.
    if module_name == "openai" or module_name.startswith("openai."):
        return PermanentModelError(f"unclassified model provider failure: {detail}")
    return None


class OpenAICompatibleModelClient:
    """One tested Chat Completions adapter, with optional client injection.

    Injecting ``client`` is intended for unit tests and compatible provider
    clients.  When it is omitted, the class lazily imports and constructs the
    ordinary ``openai.OpenAI`` client.  API credentials are never read directly
    from environment variables here; the CLI/configuration layer must pass the
    selected value explicitly, keeping configuration policy out of this module.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        temperature: float | None = None,
        timeout_seconds: float = 60.0,
        extra_request_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("model must be a non-empty string")
        if temperature is not None and (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
        ):
            raise ConfigurationError("temperature must be a finite number or None")
        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ConfigurationError(
                "timeout_seconds must be a finite number greater than zero"
            )

        options = dict(extra_request_options or {})
        protected = {"model", "messages", "tools", "stream", "temperature", "timeout"}
        overlap = protected.intersection(options)
        if overlap:
            fields = ", ".join(sorted(overlap))
            raise ConfigurationError(
                f"extra_request_options cannot override managed fields: {fields}"
            )

        if client is None:
            if not isinstance(api_key, str) or not api_key.strip():
                raise ConfigurationError(
                    "api_key is required when no client is injected"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ConfigurationError(
                    "the openai package is required for the model adapter"
                ) from exc
            client_options: dict[str, Any] = {
                "api_key": api_key,
                # The controller owns the only retry policy.  Leaving SDK
                # retries enabled would multiply attempts invisibly and make
                # request counts, backoff, and wall-time limits inaccurate.
                "max_retries": 0,
            }
            if base_url is not None:
                client_options["base_url"] = base_url
            client = OpenAI(**client_options)

        self._client = client
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._timeout_seconds = float(timeout_seconds)
        self._extra_request_options = options

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        *,
        timeout_seconds: float | None = None,
    ) -> ModelTurn:
        """Make one side-effect-free model request and normalize its response."""

        if not messages:
            raise ConfigurationError("at least one message is required")
        effective_timeout = self._timeout_seconds
        if timeout_seconds is not None:
            if (
                not isinstance(timeout_seconds, (int, float))
                or isinstance(timeout_seconds, bool)
                or not math.isfinite(float(timeout_seconds))
                or timeout_seconds <= 0
            ):
                raise ConfigurationError(
                    "per-request timeout_seconds must be a finite positive number"
                )
            effective_timeout = min(effective_timeout, float(timeout_seconds))
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_api_dict() for message in messages],
            "stream": False,
            "timeout": effective_timeout,
            **self._extra_request_options,
        }
        # Some compatible gateways reject an empty tools array, so omit the
        # field entirely when the runtime has registered no tools.
        if tools:
            request["tools"] = [dict(tool) for tool in tools]
        if self._temperature is not None:
            request["temperature"] = self._temperature

        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            translated = _translate_request_error(exc, self._api_key)
            if translated is None:
                # An AttributeError from a malformed injected client or a bug
                # in request construction is an internal defect, not a model
                # outage.  Preserve its traceback instead of misclassifying it.
                raise
            raise translated from exc

        # Parsing sits outside the request exception handler.  A malformed
        # provider response is a ResponseProtocolError and follows its own
        # bounded retry policy in the controller.
        return normalize_chat_completion(response)


# Shorter spelling used by configuration/CLI code while the long class name
# remains explicit in documentation and tests.
OpenAIModelClient = OpenAICompatibleModelClient
