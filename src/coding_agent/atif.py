"""Project a completed run into the Harbor-compatible ATIF format.

The native JSONL event sink is intentionally a low-volume diagnostic stream.
It contains summaries and therefore cannot reconstruct a complete conversation.
This module performs a separate, pure projection from ``RunResult.history``,
which is the Runtime's canonical protocol history.  Keeping the projection
outside the controller means an export failure never changes a run's terminal
state or causes a tool to be replayed.

Only the stable ATIF-v1.7 fields used by Harbor are emitted.  The exporter does
not import Harbor, so the core project remains framework-free; Harbor is an
optional consumer of the resulting JSON document.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coding_agent import __version__
from coding_agent.errors import InvariantViolation
from coding_agent.events import redact
from coding_agent.types import Message, ToolCall, Usage

if TYPE_CHECKING:  # pragma: no cover - imported only by type checkers
    from coding_agent.agent import RunResult


ATIF_SCHEMA_VERSION = "ATIF-v1.7"
DEFAULT_AGENT_NAME = "course-coding-agent"


def export_atif(
    run: RunResult,
    *,
    task: str,
    model_name: str | None = None,
    run_id: str | None = None,
    tool_definitions: Sequence[Mapping[str, Any]] = (),
    verification: Any | None = None,
    events: Sequence[Mapping[str, Any]] = (),
    secrets: Iterable[str] = (),
    include_reasoning: bool = False,
) -> dict[str, Any]:
    """Return a deterministic ATIF dictionary for one completed Runtime run.

    ``run.history`` is validated while it is projected.  A malformed history
    must fail loudly instead of creating a plausible-looking trajectory with
    missing observations.  ``include_reasoning`` is opt-in because the
    provider field may contain private/internal reasoning; the default export
    contains only user-visible assistant text and tool observations.
    """

    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("run_id must be a non-empty string when supplied")
    if model_name is not None and not isinstance(model_name, str):
        raise TypeError("model_name must be a string or None")

    try:
        history_value = run.history
    except AttributeError as exc:
        raise InvariantViolation("run result has no canonical history") from exc
    if not isinstance(history_value, Sequence):
        raise InvariantViolation("run result history must be a sequence")
    history = tuple(history_value)
    if not history:
        raise InvariantViolation("cannot export an empty canonical history")

    steps: list[dict[str, Any]] = []
    seen_call_ids: set[str] = set()
    index = 0
    while index < len(history):
        message = history[index]
        if not isinstance(message, Message):
            raise InvariantViolation(
                "ATIF export found a non-canonical history message"
            )
        if message.role in {"system", "user"}:
            steps.append(
                {
                    "step_id": len(steps) + 1,
                    "source": message.role,
                    "message": message.content or "",
                }
            )
            index += 1
            continue

        if message.role == "tool":
            raise InvariantViolation(
                "ATIF export found a tool result without a preceding assistant call"
            )

        if message.role != "assistant":  # guarded by Message, kept defensive
            raise InvariantViolation(f"unsupported history role: {message.role!r}")

        step: dict[str, Any] = {
            "step_id": len(steps) + 1,
            "source": "agent",
            "message": message.content or "",
            # One assistant message is one accepted logical model response.
            # Transport retries are aggregate run metadata, not extra steps.
            "llm_call_count": 1,
        }
        if model_name is not None:
            step["model_name"] = model_name
        if include_reasoning and message.reasoning_content is not None:
            step["reasoning_content"] = message.reasoning_content

        if message.tool_calls:
            atif_calls: list[dict[str, Any]] = []
            observations: list[dict[str, Any]] = []
            cursor = index + 1
            for call in message.tool_calls:
                if not isinstance(call, ToolCall):
                    raise InvariantViolation(
                        "ATIF export found a non-canonical tool call"
                    )
                if call.id in seen_call_ids:
                    raise InvariantViolation(
                        f"ATIF export found duplicate tool call ID {call.id!r}"
                    )
                seen_call_ids.add(call.id)
                if cursor >= len(history) or history[cursor].role != "tool":
                    raise InvariantViolation(
                        f"assistant call {call.id!r} has no immediately following result"
                    )
                result_message = history[cursor]
                if result_message.tool_call_id != call.id:
                    raise InvariantViolation(
                        "tool result order or call ID does not match assistant call "
                        f"{call.id!r}"
                    )
                if result_message.name != call.name:
                    raise InvariantViolation(
                        f"tool result name does not match assistant call {call.name!r}"
                    )

                atif_call = _atif_tool_call(call)
                atif_calls.append(atif_call)
                observations.append(
                    {
                        "source_call_id": call.id,
                        "content": result_message.content or "",
                    }
                )
                cursor += 1

            step["tool_calls"] = atif_calls
            step["observation"] = {"results": observations}
            index = cursor
        else:
            index += 1
        steps.append(step)

    agent: dict[str, Any] = {
        "name": DEFAULT_AGENT_NAME,
        "version": __version__,
    }
    if model_name is not None:
        agent["model_name"] = model_name
    if tool_definitions:
        agent["tool_definitions"] = [dict(item) for item in tool_definitions]

    extra_coding_agent: dict[str, Any] = {
        "phase": _enum_or_value(run.phase),
        "terminal_reason": run.reason,
        "model_turns": run.model_turns,
        "model_requests": run.model_requests,
        "tool_calls": run.tool_calls,
        "elapsed_seconds": run.elapsed_seconds,
        "history_messages": len(history),
    }
    if run_id is not None:
        extra_coding_agent["run_id"] = run_id
    if verification is not None:
        extra_coding_agent["verification"] = _to_json_value(verification)
    if events:
        # Event payloads can contain source code and are not needed to replay
        # the protocol.  Preserve only event names/counts as a compact pointer.
        event_names = [
            str(item.get("event", item.get("event_type", "unknown")))
            for item in events
            if isinstance(item, Mapping)
        ]
        extra_coding_agent["event_count"] = len(event_names)
        extra_coding_agent["event_types"] = event_names

    final_metrics: dict[str, Any] = {"total_steps": len(steps)}
    usage = run.usage
    if isinstance(usage, Usage):
        _copy_usage(final_metrics, usage)

    document: dict[str, Any] = {
        "schema_version": ATIF_SCHEMA_VERSION,
        "agent": agent,
        "steps": steps,
        "final_metrics": final_metrics,
        "extra": {"coding_agent": extra_coding_agent},
    }
    if run_id is not None:
        document["session_id"] = run_id

    # Apply the same final serialization boundary as the native trace.  This
    # protects against a caller accidentally placing a credential in a custom
    # verification summary or tool metadata.
    safe_document = redact(document, secrets=secrets)
    if not isinstance(safe_document, dict):  # pragma: no cover - redact contract
        raise InvariantViolation("ATIF redaction did not return a mapping")
    _ensure_json_serializable(safe_document)
    return safe_document


def write_atif(
    path: str | os.PathLike[str],
    document: Mapping[str, Any],
    *,
    secrets: Iterable[str] = (),
) -> Path:
    """Atomically write one redacted ATIF document with owner-only permissions."""

    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_document = redact(document, secrets=secrets)
    _ensure_json_serializable(safe_document)
    encoded = json.dumps(
        safe_document,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )

    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


def _atif_tool_call(call: ToolCall) -> dict[str, Any]:
    """Convert raw arguments without ever inventing a structured object."""

    try:
        parsed = json.loads(call.arguments_json)
    except (json.JSONDecodeError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        return {
            "tool_call_id": call.id,
            "function_name": call.name,
            "arguments": parsed,
        }

    # ATIF requires ``arguments`` to be an object.  Preserve the original raw
    # value only in an extension so consumers can diagnose the malformed call
    # without accepting it as a valid structured invocation.
    return {
        "tool_call_id": call.id,
        "function_name": call.name,
        "arguments": {},
        "extra": {
            "coding_agent": {
                "raw_arguments": call.arguments_json,
                "error_code": "invalid_json_arguments",
            }
        },
    }


def _copy_usage(target: dict[str, Any], usage: Usage) -> None:
    # ATIF's FinalMetrics schema is intentionally strict.  Keep the two
    # provider/accounting fields that are not part of the standard schema in
    # its explicitly extensible ``extra`` mapping instead of emitting unknown
    # top-level keys that Harbor rejects with ``extra_forbidden``.
    extra: dict[str, Any] | None = None
    if usage.prompt_tokens is not None:
        target["total_prompt_tokens"] = usage.prompt_tokens
    if usage.completion_tokens is not None:
        target["total_completion_tokens"] = usage.completion_tokens
    if usage.cached_tokens is not None:
        target["total_cached_tokens"] = usage.cached_tokens
    if usage.total_tokens is not None:
        extra = target.setdefault("extra", {})
        extra["total_tokens"] = usage.total_tokens
    if usage.reasoning_tokens is not None:
        if extra is None:
            extra = target.setdefault("extra", {})
        extra["total_reasoning_tokens"] = usage.reasoning_tokens


def _enum_or_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _to_json_value(value: Any) -> Any:
    """Convert dataclasses/enums/mappings without exposing arbitrary reprs."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    elif hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "value"):
        return _to_json_value(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


def _ensure_json_serializable(value: Any) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("ATIF document contains a non-JSON value") from exc
