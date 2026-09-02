"""Thin adapter for a non-streaming OpenAI-compatible Chat Completions API.

The ordinary provider client is used only for HTTPS, authentication and wire
serialization.  This module does not use an Agent SDK and deliberately owns no
conversation history, tool execution, retry loop, or termination policy.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from coding_agent.errors import (
    ConfigurationError,
    ContextOverflow,
    PermanentModelError,
    ReasoningEffortUnsupported,
    TransientModelError,
)
from coding_agent.response_parser import normalize_chat_completion
from coding_agent.types import Message, ModelTurn

ToolSchema = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ReasoningCapability:
    """Result of a minimal native reasoning-parameter probe.

    ``status`` is one of ``supported``, ``unsupported`` or ``error``.  The
    exact request parameter/value are retained for audit reports; no secret or
    response body is stored in this object.
    """

    status: str
    requested_effort: str | None
    parameter: str | None = None
    accepted_value: object | None = None
    error_type: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"supported", "unsupported", "error"}:
            raise ValueError("reasoning capability status is invalid")

    @property
    def supported(self) -> bool:
        return self.status == "supported"

    def to_dict(self) -> dict[str, object | None]:
        return {
            "status": self.status,
            "requested_effort": self.requested_effort,
            "parameter": self.parameter,
            "accepted_value": self.accepted_value,
            "error_type": self.error_type,
            "detail": self.detail,
        }


_REASONING_LEVELS = ("max", "high", "medium", "low")
_REASONING_PARAMETER_CANDIDATES = ("reasoning_effort", "thinking")
_REASONING_UNKNOWN_MARKERS = (
    "unknown parameter",
    "unknown field",
    "unknown argument",
    "unsupported parameter",
    "unrecognized parameter",
    "unrecognised parameter",
    "unrecognized field",
    "unrecognised field",
    "unexpected keyword",
    "unexpected field",
    "invalid parameter",
    "invalid field",
    "field is not allowed",
    "field not allowed",
    "extra_forbidden",
    "extra fields not permitted",
    "extra inputs are not permitted",
    "additional properties",
    "parameter is not supported",
    "not a supported parameter",
    "does not support this parameter",
)
_REASONING_VALUE_MARKERS = (
    "invalid value",
    "invalid enum",
    "invalid effort",
    "must be one of",
    "expected one of",
    "unsupported value",
    "value is not permitted",
    "allowed values",
)
_MODEL_NOT_FOUND_MARKERS = (
    "model not found",
    "model does not exist",
    "unknown model",
    "no such model",
    "invalid model",
)


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


def _reasoning_probe_error_kind(
    exc: Exception,
    *,
    parameter: str,
) -> str:
    """Classify one failed reasoning probe as ``unsupported`` or ``error``.

    A probe is a configuration diagnostic, so its taxonomy must be narrower
    than the runtime retry taxonomy.  Only an explicit client-side response
    identifying the candidate field/value is evidence that a gateway lacks the
    native option.  Authentication, model lookup, context, transport and
    server failures are indeterminate ``error`` results and must stop probing.
    """

    if isinstance(exc, ReasoningEffortUnsupported):
        return "unsupported"
    class_name = type(exc).__name__.lower()
    text = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if not isinstance(status, int) and isinstance(response, Mapping):
            status = response.get("status_code") or response.get("status")
    if not isinstance(status, int):
        status = getattr(exc, "http_status", None)

    # These conditions can never establish capability support. Check them
    # before the broad 4xx handling below because providers often mention the
    # requested field in an authentication or model-not-found message.
    if isinstance(status, int) and (
        status in {401, 403, 404, 408, 409, 425, 429} or status >= 500
    ):
        return "error"
    if any(marker in text for marker in _MODEL_NOT_FOUND_MARKERS):
        return "error"
    if _looks_like_context_overflow(text):
        return "error"
    if any(
        marker in class_name
        for marker in (
            "timeout",
            "connection",
            "rate_limit",
            "ratelimit",
            "authentication",
            "permission",
            "notfound",
            "internalserver",
            "contextoverflow",
            "context_length",
        )
    ):
        return "error"
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "error"

    # A provider may use either an "unknown field" message or an invalid-value
    # message.  Require the candidate name (or the generic reasoning keyword)
    # to be present; a generic 400 can instead indicate malformed credentials,
    # an invalid model request, or a context problem.
    mentions_parameter = (
        parameter.lower() in text or "reasoning" in text or "thinking" in text
    )
    if not mentions_parameter:
        return "error"
    if any(marker in text for marker in _REASONING_UNKNOWN_MARKERS):
        return "unsupported"
    if any(marker in text for marker in _REASONING_VALUE_MARKERS):
        return "unsupported"
    # Some gateways return a terse 400 such as ``reasoning_effort unsupported``.
    if "unsupported" in text or "not supported" in text:
        return "unsupported"
    return "error"


def _reasoning_probe_is_value_error(exc: Exception, *, parameter: str) -> bool:
    """Return whether an unsupported probe response describes only its value.

    Unknown-field responses should advance directly to the next candidate
    field.  Only value/enum responses justify trying lower effort levels; this
    keeps a capability probe genuinely minimal while still accommodating a
    gateway whose native field accepts, for example, ``medium`` but not
    ``high``.
    """

    text = str(exc).lower()
    mentions_parameter = (
        parameter.lower() in text or "reasoning" in text or "thinking" in text
    )
    return mentions_parameter and any(
        marker in text for marker in _REASONING_VALUE_MARKERS
    )


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
        reasoning_effort: str | None = None,
        reasoning_parameter: str = "reasoning_effort",
        reasoning_value: object | None = None,
        reasoning_capability: ReasoningCapability | None = None,
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
        if reasoning_effort is not None:
            if not isinstance(
                reasoning_effort, str
            ) or reasoning_effort.strip().lower() not in {
                "low",
                "medium",
                "high",
                "max",
            }:
                raise ConfigurationError(
                    "reasoning_effort must be one of: low, medium, high, max"
                )
            reasoning_effort = reasoning_effort.strip().lower()
        if not isinstance(reasoning_parameter, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,63}", reasoning_parameter
        ):
            raise ConfigurationError("reasoning_parameter must be a field name")
        if reasoning_capability is not None and not isinstance(
            reasoning_capability, ReasoningCapability
        ):
            raise ConfigurationError("reasoning_capability is invalid")
        if reasoning_value is not None:
            try:
                # The OpenAI-compatible request must remain JSON-serializable;
                # this also preserves bool/object values selected by a probe.
                json.dumps(reasoning_value, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    "reasoning_value must be JSON-serializable"
                ) from exc

        options = dict(extra_request_options or {})
        protected = {
            "model",
            "messages",
            "tools",
            "stream",
            "temperature",
            "timeout",
            "reasoning_effort",
            "thinking",
            # ``reasoning_parameter`` may name a provider-specific field such
            # as ``reasoning`` or ``enable_thinking``.  Protect both the
            # portable aliases above and the selected dynamic field so an
            # extra option cannot silently compete with managed reasoning
            # configuration.
            reasoning_parameter,
        }
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
        self._reasoning_effort = reasoning_effort
        self._reasoning_parameter = reasoning_parameter
        self._reasoning_value = (
            reasoning_effort if reasoning_value is None else reasoning_value
        )
        self._reasoning_value_provided = reasoning_value is not None
        self._reasoning_capability = reasoning_capability

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_effort(self) -> str | None:
        return self._reasoning_effort

    @property
    def reasoning_capability(self) -> ReasoningCapability | None:
        return self._reasoning_capability

    def configure_reasoning(self, capability: ReasoningCapability) -> None:
        """Apply a previously verified native capability to this client."""

        if not isinstance(capability, ReasoningCapability):
            raise TypeError("capability must be a ReasoningCapability")
        if capability.supported:
            if (
                capability.parameter is None
                or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", capability.parameter)
                is None
            ):
                raise ConfigurationError(
                    "supported reasoning capability has an invalid parameter"
                )
            if capability.accepted_value is None:
                raise ConfigurationError(
                    "supported reasoning capability has no accepted value"
                )
            try:
                json.dumps(
                    capability.accepted_value,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    "supported reasoning capability value must be JSON-serializable"
                ) from exc
            if capability.requested_effort is not None and (
                not isinstance(capability.requested_effort, str)
                or capability.requested_effort.strip().lower()
                not in {"low", "medium", "high", "max"}
            ):
                raise ConfigurationError(
                    "supported reasoning capability has an invalid requested effort"
                )
        self._reasoning_capability = capability
        if capability.supported:
            if self._reasoning_effort is None and capability.requested_effort:
                self._reasoning_effort = capability.requested_effort
            if capability.parameter:
                self._reasoning_parameter = capability.parameter
            if capability.accepted_value is not None:
                self._reasoning_value = capability.accepted_value
                self._reasoning_value_provided = True

    def probe_reasoning_effort(
        self,
        effort: str | None = None,
        *,
        parameter_candidates: Sequence[str] = _REASONING_PARAMETER_CANDIDATES,
        timeout_seconds: float | None = None,
    ) -> ReasoningCapability:
        """Probe a gateway with a one-token request and record native support.

        The probe is intentionally explicit: callers can run it once during a
        smoke test and persist only the returned metadata.  It never falls
        back to prompt wording.  Candidate parameters are tried in order, and
        levels are tried from the requested level down to ``low`` so the
        accepted native value is unambiguous.
        """

        requested = self._reasoning_effort if effort is None else effort
        if requested is None:
            requested = "high"
        if not isinstance(requested, str) or requested.strip().lower() not in {
            "low",
            "medium",
            "high",
            "max",
        }:
            raise ConfigurationError(
                "reasoning_effort must be one of: low, medium, high, max"
            )
        requested = requested.strip().lower()
        start = _REASONING_LEVELS.index(requested)
        levels = _REASONING_LEVELS[start:]
        candidates = tuple(parameter_candidates)
        if not candidates:
            raise ConfigurationError("parameter_candidates must not be empty")
        last_unsupported: Exception | None = None
        for parameter in candidates:
            if not isinstance(parameter, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]{0,63}", parameter
            ):
                raise ConfigurationError("reasoning parameter candidate is invalid")
            # An unknown field is settled by one request.  A provider that
            # explicitly reports an invalid value may be checked at lower
            # native levels, but we never blindly send every candidate value
            # after an unrelated 400 response.
            levels_for_parameter = levels
            for level in levels_for_parameter:
                request: dict[str, Any] = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "max_tokens": 1,
                    "timeout": self._effective_timeout(timeout_seconds),
                    parameter: level,
                }
                try:
                    self._client.chat.completions.create(**request)
                except Exception as exc:  # noqa: BLE001 - provider probe boundary
                    kind = _reasoning_probe_error_kind(exc, parameter=parameter)
                    if kind == "unsupported":
                        if isinstance(exc, ReasoningEffortUnsupported):
                            # This domain error already means the provider
                            # rejected native reasoning as a whole. It carries
                            # no field-specific validation detail, so trying
                            # every alternate spelling would only turn a
                            # minimal probe into an avoidable request loop.
                            capability = ReasoningCapability(
                                status="unsupported",
                                requested_effort=requested,
                                parameter=parameter,
                                error_type=type(exc).__name__,
                                detail=_redact_error_text(str(exc), self._api_key),
                            )
                            self._reasoning_capability = capability
                            return capability
                        # Keep trying lower values and alternative native field
                        # names.  The final detail is retained only after all
                        # candidates have been explicitly rejected.
                        last_unsupported = exc
                        if not _reasoning_probe_is_value_error(
                            exc, parameter=parameter
                        ):
                            break
                        continue
                    detail = _redact_error_text(str(exc), self._api_key)
                    capability = ReasoningCapability(
                        status="error",
                        requested_effort=requested,
                        parameter=parameter,
                        error_type=type(exc).__name__,
                        detail=detail or type(exc).__name__,
                    )
                    self._reasoning_capability = capability
                    return capability
                capability = ReasoningCapability(
                    status="supported",
                    requested_effort=requested,
                    parameter=parameter,
                    accepted_value=level,
                )
                self.configure_reasoning(capability)
                return capability
        if last_unsupported is None:
            capability = ReasoningCapability(
                status="unsupported", requested_effort=requested
            )
        else:
            capability = ReasoningCapability(
                status="unsupported",
                requested_effort=requested,
                error_type=type(last_unsupported).__name__,
                detail=_redact_error_text(str(last_unsupported), self._api_key),
            )
        self._reasoning_capability = capability
        return capability

    def _effective_timeout(self, timeout_seconds: float | None) -> float:
        effective = self._timeout_seconds
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
            effective = min(effective, float(timeout_seconds))
        return effective

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
        effective_timeout = self._effective_timeout(timeout_seconds)
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
        if self._reasoning_effort is not None or self._reasoning_value_provided:
            capability = self._reasoning_capability
            # A direct constructor value is useful for known-compatible
            # gateways and preserves the simple adapter contract.  If an
            # explicit probe established ``unsupported``, fail closed instead
            # of silently downgrading or embedding a fake prompt instruction.
            if capability is not None and not capability.supported:
                raise ConfigurationError(
                    "requested reasoning_effort is unsupported by the gateway"
                )
            parameter = (
                capability.parameter
                if capability is not None and capability.parameter
                else self._reasoning_parameter
            )
            value = (
                capability.accepted_value
                if capability is not None and capability.supported
                else self._reasoning_value
            )
            request[parameter] = value

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
