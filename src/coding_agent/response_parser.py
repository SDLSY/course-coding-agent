"""Normalize Chat Completions responses and parse tool arguments.

This module is the explicit trust boundary for model output.  It supports the
ordinary OpenAI SDK object shape and equivalent dictionaries, which is useful
for deterministic tests and for a small number of compatible gateways.  It
does not attempt to guess arbitrary provider-specific response formats.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from coding_agent.errors import ResponseProtocolError, ToolRequestError
from coding_agent.types import ModelTurn, ToolCall, Usage

_MISSING = object()


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    """Read one field from either a mapping or a regular SDK/Pydantic object."""

    if isinstance(value, Mapping):
        if name in value:
            return value[name]
    elif hasattr(value, name):
        return getattr(value, name)

    if default is _MISSING:
        raise ResponseProtocolError(
            f"model response is missing required field {name!r}"
        )
    return default


def _optional_token_count(usage: Any, field_name: str) -> int | None:
    value = _field(usage, field_name, None)
    if value is None:
        return None
    # ``bool`` subclasses ``int`` in Python, but accepting true as one token
    # would conceal a malformed gateway response.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponseProtocolError(
            f"usage.{field_name} must be a non-negative integer or null"
        )
    return value


def _normalize_usage(response: Any) -> Usage | None:
    raw_usage = _field(response, "usage", None)
    if raw_usage is None:
        return None
    return Usage(
        prompt_tokens=_optional_token_count(raw_usage, "prompt_tokens"),
        completion_tokens=_optional_token_count(raw_usage, "completion_tokens"),
        total_tokens=_optional_token_count(raw_usage, "total_tokens"),
    )


def _normalize_tool_calls(message: Any) -> tuple[ToolCall, ...]:
    raw_calls = _field(message, "tool_calls", None)
    if raw_calls is None:
        return ()
    if isinstance(raw_calls, (str, bytes, Mapping)):
        raise ResponseProtocolError("message.tool_calls must be an array")
    try:
        calls = list(raw_calls)
    except TypeError as exc:
        raise ResponseProtocolError("message.tool_calls must be an array") from exc

    normalized: list[ToolCall] = []
    seen_ids: set[str] = set()
    for index, raw_call in enumerate(calls):
        call_type = _field(raw_call, "type", "function")
        if call_type != "function":
            raise ResponseProtocolError(
                f"tool call {index} has unsupported type {call_type!r}"
            )

        call_id = _field(raw_call, "id")
        function = _field(raw_call, "function")
        name = _field(function, "name")
        arguments = _field(function, "arguments")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ResponseProtocolError(f"tool call {index} has an invalid id")
        if call_id in seen_ids:
            raise ResponseProtocolError(f"duplicate tool call id {call_id!r}")
        if not isinstance(name, str) or not name.strip():
            raise ResponseProtocolError(
                f"tool call {index} has an invalid function name"
            )
        if not isinstance(arguments, str):
            raise ResponseProtocolError(
                f"tool call {call_id!r} arguments must be a JSON string"
            )

        # Do not parse JSON here.  Invalid arguments are a recoverable tool
        # request error, whereas a malformed response envelope is a provider
        # protocol error.  Keeping these categories separate is essential to
        # the controller's recovery behaviour.
        normalized.append(ToolCall(call_id, name, arguments))
        seen_ids.add(call_id)
    return tuple(normalized)


def normalize_chat_completion(response: Any) -> ModelTurn:
    """Convert a non-streaming Chat Completion into :class:`ModelTurn`.

    Only the first choice is accepted because the runtime requests one choice
    and has no policy for selecting among alternatives.  An empty choice list
    is an invalid response.  Extra choices are ignored for compatibility with
    gateways that return ``n > 1`` despite the request default.

    Empty assistant output is preserved here and rejected later by the agent's
    protocol retry policy.  Normalization should describe what the provider
    returned, not decide whether that turn ends the control loop.
    """

    raw_choices = _field(response, "choices")
    if isinstance(raw_choices, (str, bytes, Mapping)):
        raise ResponseProtocolError("model response choices must be an array")
    try:
        choices = list(raw_choices)
    except TypeError as exc:
        raise ResponseProtocolError("model response choices must be an array") from exc
    if not choices:
        raise ResponseProtocolError("model response contains no choices")

    choice = choices[0]
    message = _field(choice, "message")
    content = _field(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise ResponseProtocolError("assistant content must be a string or null")

    # ``reasoning_content`` is a de-facto OpenAI-compatible extension used by
    # reasoning providers such as DeepSeek.  With tools enabled, those providers
    # may require the exact value in subsequent assistant history.  Normalising
    # this one understood field avoids retaining an arbitrary vendor response
    # object while still producing a valid next tool round.
    reasoning_content = _field(message, "reasoning_content", None)
    if reasoning_content is not None and not isinstance(reasoning_content, str):
        raise ResponseProtocolError(
            "assistant reasoning_content must be a string or null"
        )

    finish_reason = _field(choice, "finish_reason", None)
    if finish_reason is not None and not isinstance(finish_reason, str):
        raise ResponseProtocolError("finish_reason must be a string or null")

    return ModelTurn(
        text=content,
        tool_calls=_normalize_tool_calls(message),
        finish_reason=finish_reason,
        usage=_normalize_usage(response),
        reasoning_content=reasoning_content,
    )


class _DuplicateJsonKey(ValueError):
    """Internal sentinel used to distinguish duplicate keys from syntax errors."""


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    # Python's json module accepts NaN and Infinity by default, although the
    # JSON standard and tool schemas do not.  Reject them at this boundary so a
    # handler never receives surprising floating-point values.
    raise ValueError(f"non-standard JSON constant {value}")


def parse_tool_arguments(call: ToolCall) -> dict[str, Any]:
    """Parse one raw argument string as a strict JSON object.

    Syntax errors, duplicate object keys, non-standard numeric constants and a
    non-object top level are recoverable model mistakes.  They are represented
    as ``ToolRequestError`` so the registry can return exactly one structured
    tool result and allow the model to correct its request on the next turn.
    """

    try:
        value = json.loads(
            call.arguments_json,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except _DuplicateJsonKey as exc:
        key = exc.args[0]
        raise ToolRequestError(
            f"arguments for tool {call.name!r} contain duplicate key {key!r}",
            error_code="duplicate_argument",
        ) from exc
    except json.JSONDecodeError as exc:
        raise ToolRequestError(
            f"arguments for tool {call.name!r} are invalid JSON "
            f"at line {exc.lineno}, column {exc.colno}",
            error_code="invalid_json",
        ) from exc
    except ValueError as exc:
        raise ToolRequestError(
            f"arguments for tool {call.name!r} are not strict JSON: {exc}",
            error_code="invalid_json",
        ) from exc

    if not isinstance(value, dict):
        raise ToolRequestError(
            f"arguments for tool {call.name!r} must be a JSON object",
            error_code="arguments_not_object",
        )
    return value


# A concise alias for callers that are already operating on ToolCall objects.
parse_arguments = parse_tool_arguments
