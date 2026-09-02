"""Provider-independent data types shared by the agent runtime.

Only the model adapter is allowed to know about objects returned by a vendor
SDK.  The rest of the project communicates through the dataclasses in this
module, which makes the control loop deterministic to test and keeps accidental
SDK behaviour (lazy fields, custom mappings, implicit serialization) out of the
canonical conversation history.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from coding_agent.errors import InvariantViolation

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One raw function call requested by the model.

    ``arguments_json`` is intentionally preserved byte-for-character as a
    Python string.  Parsing it here would lose useful evidence (and could make
    an invalid request impossible to report back to the model).  The tool
    registry parses and validates the JSON immediately before execution.
    """

    id: str
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("tool call id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool call name must be a non-empty string")
        if not isinstance(self.arguments_json, str):
            raise TypeError("tool call arguments_json must be a string")

    def to_api_dict(self) -> dict[str, Any]:
        """Serialize the call using the Chat Completions message shape."""

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_json,
            },
        }


@dataclass(frozen=True, slots=True)
class Message:
    """A normalized message stored in the append-only canonical history.

    Role-specific invariants are checked when the message is constructed.  The
    checks are intentionally local: relationships between adjacent messages
    (for example, whether every tool call has exactly one result) are validated
    by :class:`coding_agent.context.ContextBuilder` and the agent controller.
    """

    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    # Some OpenAI-compatible reasoning providers return this assistant field
    # and require it to be replayed while tools remain enabled.  Keeping it as
    # an explicit optional field is safer than retaining an arbitrary vendor
    # response mapping: only the compatibility extension we understand can
    # cross the next provider boundary.
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role!r}")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("message content must be a string or None")
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content, str
        ):
            raise TypeError("message reasoning_content must be a string or None")

        if self.role == "assistant":
            if self.tool_call_id is not None or self.name is not None:
                raise ValueError("assistant messages cannot identify a tool result")
            ids = [call.id for call in self.tool_calls]
            if len(ids) != len(set(ids)):
                raise ValueError(
                    "tool call ids must be unique within one assistant message"
                )
        elif self.role == "tool":
            if self.tool_calls:
                raise ValueError("tool messages cannot contain tool calls")
            if self.reasoning_content is not None:
                raise ValueError("tool messages cannot contain reasoning_content")
            if not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip():
                raise ValueError("tool messages require a non-empty tool_call_id")
            if self.name is not None and (
                not isinstance(self.name, str) or not self.name.strip()
            ):
                raise ValueError(
                    "tool message name must be a non-empty string when supplied"
                )
            if self.content is None:
                raise ValueError("tool messages require string content")
        else:
            if (
                self.tool_calls
                or self.tool_call_id is not None
                or self.name is not None
                or self.reasoning_content is not None
            ):
                raise ValueError(
                    f"{self.role} messages cannot carry tool protocol fields"
                )
            if self.content is None:
                raise ValueError(f"{self.role} messages require string content")

    @classmethod
    def assistant(
        cls,
        content: str | None,
        tool_calls: tuple[ToolCall, ...] = (),
        *,
        reasoning_content: str | None = None,
    ) -> Message:
        """Construct an assistant message without exposing role boilerplate."""

        return cls(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        )

    def to_api_dict(self) -> dict[str, Any]:
        """Return a plain dictionary accepted by Chat Completions clients.

        ``None`` assistant content is retained when tool calls exist because it
        is a legal and common provider response.  Optional fields are omitted
        instead of being sent as null, improving compatibility with gateways
        that implement only the core OpenAI message schema.
        """

        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "assistant":
            # DeepSeek-compatible thinking modes require this value from prior
            # assistant turns when tools are present.  Do not synthesize it and
            # do not send it for ordinary OpenAI turns; replay only a field the
            # provider actually returned.  Empty string remains distinct from
            # absence and is therefore intentionally retained.
            if self.reasoning_content is not None:
                payload["reasoning_content"] = self.reasoning_content
            if self.tool_calls:
                payload["tool_calls"] = [call.to_api_dict() for call in self.tool_calls]
        elif self.role == "tool":
            payload["tool_call_id"] = self.tool_call_id
            # ``name`` remains in canonical history so the runtime can assert
            # that a result matches its original call.  The current OpenAI
            # Chat Completions tool-message schema does not accept that field,
            # so it must not cross the provider boundary.
        return payload


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Uniform result of a local tool request.

    A failed command (for example, ``pytest`` returning exit code 1) can still
    be a successfully executed tool and should normally use ``ok=True`` with
    the exit code in ``metadata``.  ``ok=False`` is reserved for invalid tool
    requests or failures to perform the requested local operation.
    """

    call_id: str
    name: str
    ok: bool
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("tool result call_id must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("tool result name must be a non-empty string")
        if not isinstance(self.ok, bool):
            raise TypeError("tool result ok must be a bool")
        if not isinstance(self.content, str):
            raise TypeError("tool result content must be a string")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise TypeError("tool result error_code must be a string or None")
        if self.ok and self.error_code is not None:
            raise ValueError("a successful tool result cannot have an error_code")

    def to_message(self) -> Message:
        """Encode the result as a deterministic JSON tool message.

        Metadata is sent to the model because fields such as an exit code or a
        truncation flag affect its next decision.  ``default=str`` is avoided:
        silently stringifying arbitrary objects could hide a tool contract bug.
        Tool implementations must therefore return JSON-serializable metadata.
        """

        envelope = {
            "ok": self.ok,
            "content": self.content,
            "metadata": dict(self.metadata),
        }
        if self.error_code is not None:
            envelope["error_code"] = self.error_code
        try:
            encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise InvariantViolation(
                f"tool result metadata for {self.name!r} is not JSON-serializable"
            ) from exc
        return Message(
            role="tool",
            content=encoded,
            tool_call_id=self.call_id,
            name=self.name,
        )

    # ``as_message`` reads naturally at call sites and keeps compatibility if
    # another runtime component settled on that spelling during development.
    as_message = to_message


@dataclass(frozen=True, slots=True)
class Usage:
    """Provider token accounting, when returned by the API."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def input_tokens(self) -> int | None:
        """Alias used by benchmark/report consumers."""

        return self.prompt_tokens

    @property
    def output_tokens(self) -> int | None:
        return self.completion_tokens

    @property
    def cache_tokens(self) -> int | None:
        return self.cached_tokens


@dataclass(frozen=True, slots=True)
class ModelTurn:
    """One normalized, provider-independent assistant response."""

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: Usage | None = None
    reasoning_content: str | None = None

    def as_message(self) -> Message:
        """Convert this turn to the exact assistant message stored in history."""

        return Message.assistant(
            self.text,
            self.tool_calls,
            reasoning_content=self.reasoning_content,
        )


class RunPhase(str, Enum):
    """Observable states of the single-agent control loop."""

    CREATED = "created"
    BUILDING_CONTEXT = "building_context"
    CALLING_MODEL = "calling_model"
    PARSING_RESPONSE = "parsing_response"
    EXECUTING_TOOLS = "executing_tools"
    RECORDING_RESULTS = "recording_results"
    CHECKING_LIMITS = "checking_limits"
    COMPLETED = "completed"
    LIMIT_REACHED = "limit_reached"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            RunPhase.COMPLETED,
            RunPhase.LIMIT_REACHED,
            RunPhase.FAILED,
            RunPhase.CANCELLED,
        }


@dataclass(slots=True)
class RunState:
    """The controller's single mutable state object.

    The dataclasses representing protocol messages are immutable; only this
    aggregate owns mutable counters and append-only history.  The controller is
    responsible for legal phase transitions and for assigning exactly one
    terminal reason.
    """

    history: list[Message]
    phase: RunPhase = RunPhase.CREATED
    model_turns: int = 0
    # Physical API attempts are distinct from accepted logical turns.  Transport
    # and protocol retries consume provider capacity and matter for benchmark
    # cost even though they do not advance the conversation.
    model_requests: int = 0
    tool_calls: int = 0
    started_at: float = field(default_factory=time.monotonic)
    terminal_reason: str | None = None

    def transition(self, phase: RunPhase, *, reason: str | None = None) -> None:
        """Apply a phase change while protecting terminal-state invariants."""

        if self.phase.is_terminal:
            raise InvariantViolation(
                f"cannot transition from terminal phase {self.phase.value!r}"
            )
        if phase.is_terminal:
            if not reason or not reason.strip():
                raise InvariantViolation("terminal phases require one non-empty reason")
            if self.terminal_reason is not None:
                raise InvariantViolation("terminal reason has already been assigned")
            self.terminal_reason = reason
        elif reason is not None:
            raise InvariantViolation(
                "non-terminal phases cannot have a terminal reason"
            )
        self.phase = phase

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)
