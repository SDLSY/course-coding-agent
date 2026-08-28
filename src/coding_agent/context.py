"""Build a bounded model view from append-only canonical conversation history.

The context builder never mutates or summarizes canonical history.  It first
validates tool-call/result transactions, then retains a contiguous suffix of
complete blocks under a deterministic character budget.  Character counts are
an intentionally conservative, provider-independent approximation; they are
not advertised as exact tokenizer output.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, overload

from coding_agent.errors import ContextOverflow, InvariantViolation
from coding_agent.types import Message

DEFAULT_TRUNCATION_NOTICE = (
    "Earlier conversation blocks were omitted to fit the context budget. "
    "No summary of the omitted content is available; re-read files or rerun "
    "commands when earlier evidence is needed."
)


def _compact_json_size(value: Any, *, label: str) -> int:
    """Return deterministic serialized character size or expose a contract bug."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise InvariantViolation(f"{label} is not JSON-serializable") from exc
    return len(encoded)


def estimate_request_chars(
    messages: Sequence[Message],
    tools: Sequence[Mapping[str, Any]] = (),
    *,
    reserved_chars: int = 0,
) -> int:
    """Estimate serialized messages, tool schemas, and fixed request overhead.

    ``reserved_chars`` lets configuration reserve space for provider wrappers,
    a future assistant answer, or fields not represented here.  It is not a
    token conversion factor.  The caller should choose the total budget with a
    safety margin appropriate for its target model.
    """

    if isinstance(reserved_chars, bool) or not isinstance(reserved_chars, int):
        raise TypeError("reserved_chars must be an integer")
    if reserved_chars < 0:
        raise ValueError("reserved_chars cannot be negative")
    message_payload = [message.to_api_dict() for message in messages]
    tool_payload = [dict(tool) for tool in tools]
    return (
        _compact_json_size(message_payload, label="messages")
        + _compact_json_size(tool_payload, label="tool schemas")
        + reserved_chars
    )


@dataclass(frozen=True, slots=True)
class ContextWindow(Sequence[Message]):
    """Immutable result of one context-building pass.

    Implementing ``Sequence`` allows a window to be passed directly to
    ``ModelClient.complete`` while named fields retain useful diagnostics for
    terminal output and context-truncation events.
    """

    messages: tuple[Message, ...]
    estimated_chars: int
    truncated: bool
    omitted_blocks: int = 0

    def __len__(self) -> int:
        return len(self.messages)

    @overload
    def __getitem__(self, index: int) -> Message: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Message, ...]: ...

    def __getitem__(self, index: int | slice) -> Message | tuple[Message, ...]:
        return self.messages[index]

    def __iter__(self) -> Iterator[Message]:
        return iter(self.messages)


def _required_prefix_end(history: Sequence[Message]) -> int:
    """Locate the end of leading system instructions plus the original task."""

    if not history:
        raise InvariantViolation("canonical history is empty")
    index = 0
    while index < len(history) and history[index].role == "system":
        index += 1
    if index == 0:
        raise InvariantViolation("canonical history must start with a system message")
    if index >= len(history) or history[index].role != "user":
        raise InvariantViolation(
            "leading system messages must be followed by the original user task"
        )
    return index + 1


def _assistant_transaction(
    history: Sequence[Message],
    start: int,
    globally_seen_ids: set[str],
) -> tuple[tuple[Message, ...], int]:
    """Consume one assistant message and all of its required tool results."""

    assistant = history[start]
    if assistant.role != "assistant":
        raise AssertionError("_assistant_transaction requires an assistant message")
    if not assistant.tool_calls:
        return (assistant,), start + 1

    expected_names: dict[str, str] = {}
    for call in assistant.tool_calls:
        if call.id in globally_seen_ids:
            raise InvariantViolation(
                f"tool call id {call.id!r} is reused in canonical history"
            )
        globally_seen_ids.add(call.id)
        expected_names[call.id] = call.name

    result_messages: list[Message] = []
    seen_results: set[str] = set()
    index = start + 1
    for _ in assistant.tool_calls:
        if index >= len(history) or history[index].role != "tool":
            raise InvariantViolation(
                "assistant tool calls must be followed immediately by one result per call"
            )
        result = history[index]
        call_id = result.tool_call_id
        # Message validates that call_id is a non-empty string, but retaining
        # the explicit check keeps this relationship clear to type checkers.
        assert call_id is not None
        if call_id not in expected_names:
            raise InvariantViolation(
                f"tool result {call_id!r} does not match the preceding assistant calls"
            )
        if call_id in seen_results:
            raise InvariantViolation(f"tool call {call_id!r} has more than one result")
        if result.name is not None and result.name != expected_names[call_id]:
            raise InvariantViolation(
                f"tool result {call_id!r} names {result.name!r}, expected "
                f"{expected_names[call_id]!r}"
            )
        seen_results.add(call_id)
        result_messages.append(result)
        index += 1

    if seen_results != set(expected_names):
        missing = sorted(set(expected_names) - seen_results)
        raise InvariantViolation(f"tool calls are missing results: {missing!r}")
    return (assistant, *result_messages), index


def _transaction_blocks(
    history: Sequence[Message],
    start: int,
) -> list[tuple[Message, ...]]:
    """Partition history after the fixed prefix without splitting tool protocol."""

    blocks: list[tuple[Message, ...]] = []
    seen_ids: set[str] = set()
    index = start
    while index < len(history):
        message = history[index]
        if message.role == "tool":
            raise InvariantViolation(
                f"orphan tool result {message.tool_call_id!r} in canonical history"
            )
        if message.role == "system":
            # A late system message could carry a new mandatory constraint.
            # Treating it as optional history would be unsafe, so canonical
            # system instructions are required to remain in the fixed prefix.
            raise InvariantViolation(
                "system messages may only appear in the fixed prefix"
            )

        if message.role == "assistant":
            block, index = _assistant_transaction(history, index, seen_ids)
            blocks.append(block)
            continue

        # A later user clarification is grouped with the immediately following
        # assistant transaction when present.  This prevents retaining an
        # answer or its tool actions after dropping the question that caused it.
        if message.role == "user" and index + 1 < len(history):
            following = history[index + 1]
            if following.role == "assistant":
                assistant_block, next_index = _assistant_transaction(
                    history, index + 1, seen_ids
                )
                blocks.append((message, *assistant_block))
                index = next_index
                continue
        blocks.append((message,))
        index += 1
    return blocks


class ContextBuilder:
    """Create a budgeted model view while preserving protocol invariants."""

    def __init__(
        self,
        max_chars: int = 80_000,
        *,
        truncation_notice: str = DEFAULT_TRUNCATION_NOTICE,
    ) -> None:
        if isinstance(max_chars, bool) or not isinstance(max_chars, int):
            raise TypeError("max_chars must be an integer")
        if max_chars <= 0:
            raise ValueError("max_chars must be greater than zero")
        if not isinstance(truncation_notice, str) or not truncation_notice.strip():
            raise ValueError("truncation_notice must be a non-empty string")
        self.max_chars = max_chars
        self.truncation_notice = truncation_notice

    def build(
        self,
        canonical_history: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] = (),
        *,
        reserved_chars: int = 0,
    ) -> ContextWindow:
        """Validate history and retain the newest complete blocks that fit.

        The latest block is considered necessary for making forward progress.
        If the fixed prefix, schemas, truncation notice and latest block cannot
        fit together, ``ContextOverflow`` is raised rather than silently asking
        the model to proceed with no observation from the previous turn.
        """

        # Copy references into a tuple at entry.  Message is immutable, so this
        # provides a stable snapshot even if another component appends to the
        # canonical list after this method returns.
        history = tuple(canonical_history)
        prefix_end = _required_prefix_end(history)
        prefix = history[:prefix_end]
        blocks = _transaction_blocks(history, prefix_end)

        required_size = estimate_request_chars(
            prefix, tools, reserved_chars=reserved_chars
        )
        if required_size > self.max_chars:
            raise ContextOverflow(
                "system instructions, original task, tool schemas, and reserved "
                f"overhead require {required_size} characters, exceeding the "
                f"configured budget of {self.max_chars}"
            )

        full_size = estimate_request_chars(
            history, tools, reserved_chars=reserved_chars
        )
        if full_size <= self.max_chars:
            return ContextWindow(history, full_size, truncated=False, omitted_blocks=0)

        notice = Message(role="system", content=self.truncation_notice)
        # Keep every system message in one leading run.  Some compatible
        # gateways require system roles to precede the first user message even
        # though OpenAI accepts more flexible sequences.  ``prefix`` is known to
        # end with the original user task, so insert the derived notice directly
        # before that task without mutating canonical history.
        truncated_prefix = (*prefix[:-1], notice, prefix[-1])
        selected: list[tuple[Message, ...]] = []
        # Retain a contiguous newest suffix.  Skipping a large recent block to
        # include unrelated older blocks would satisfy the numeric budget but
        # create a misleading chronology.
        for block in reversed(blocks):
            candidate_blocks = [block, *selected]
            candidate_messages = (
                *truncated_prefix,
                *(item for candidate in candidate_blocks for item in candidate),
            )
            candidate_size = estimate_request_chars(
                candidate_messages, tools, reserved_chars=reserved_chars
            )
            if candidate_size > self.max_chars:
                break
            selected = candidate_blocks

        if blocks and not selected:
            latest_only = (*truncated_prefix, *blocks[-1])
            latest_size = estimate_request_chars(
                latest_only, tools, reserved_chars=reserved_chars
            )
            raise ContextOverflow(
                "the fixed prompt and latest complete conversation block require "
                f"{latest_size} characters, exceeding the configured budget of "
                f"{self.max_chars}; reduce tool output or increase the budget"
            )

        output = (
            *truncated_prefix,
            *(item for block in selected for item in block),
        )
        output_size = estimate_request_chars(
            output, tools, reserved_chars=reserved_chars
        )
        omitted = len(blocks) - len(selected)
        return ContextWindow(
            messages=output,
            estimated_chars=output_size,
            truncated=True,
            omitted_blocks=omitted,
        )

    def build_messages(
        self,
        canonical_history: Sequence[Message],
        tools: Sequence[Mapping[str, Any]] = (),
        *,
        reserved_chars: int = 0,
    ) -> tuple[Message, ...]:
        """Convenience wrapper for callers that need only the model messages."""

        return self.build(
            canonical_history, tools, reserved_chars=reserved_chars
        ).messages
