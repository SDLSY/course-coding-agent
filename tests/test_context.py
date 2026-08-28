"""Tests for canonical-history validation and transaction-safe truncation."""

from __future__ import annotations

import pytest

from coding_agent.context import (
    DEFAULT_TRUNCATION_NOTICE,
    ContextBuilder,
    estimate_request_chars,
)
from coding_agent.errors import ContextOverflow, InvariantViolation
from coding_agent.types import Message, ToolCall, ToolResult


def _prefix() -> list[Message]:
    return [
        Message(role="system", content="Work only inside the selected workspace."),
        Message(role="user", content="Fix the failing test."),
    ]


def _transaction(call_id: str, marker: str, *, payload_size: int = 20) -> list[Message]:
    call = ToolCall(call_id, "read_file", f'{{"path":"{marker}.py"}}')
    return [
        Message.assistant(f"Reading {marker}", (call,)),
        ToolResult(
            call_id=call_id,
            name="read_file",
            ok=True,
            content=marker * payload_size,
        ).to_message(),
    ]


def test_context_keeps_complete_history_when_it_fits() -> None:
    history = [*_prefix(), *_transaction("old", "a")]
    original_snapshot = tuple(history)

    window = ContextBuilder(max_chars=10_000).build(history)

    assert window.messages == original_snapshot
    assert not window.truncated
    assert window.omitted_blocks == 0
    assert tuple(history) == original_snapshot
    assert list(window) == history


def test_context_truncates_only_complete_tool_transactions() -> None:
    prefix = _prefix()
    old_block = _transaction("old", "o", payload_size=200)
    latest_block = _transaction("latest", "n", payload_size=30)
    notice = Message(role="system", content=DEFAULT_TRUNCATION_NOTICE)
    target_view = [*prefix[:-1], notice, prefix[-1], *latest_block]
    exact_budget = estimate_request_chars(target_view)
    history = [*prefix, *old_block, *latest_block]

    window = ContextBuilder(max_chars=exact_budget).build(history)

    assert window.truncated
    assert window.omitted_blocks == 1
    assert window.messages == tuple(target_view)
    call_ids = {
        call.id
        for message in window
        if message.role == "assistant"
        for call in message.tool_calls
    }
    result_ids = {message.tool_call_id for message in window if message.role == "tool"}
    assert call_ids == result_ids == {"latest"}
    assert window.estimated_chars <= exact_budget


def test_context_groups_later_user_message_with_its_assistant_transaction() -> None:
    prefix = _prefix()
    old_user = Message(role="user", content="Also inspect the parser.")
    # Make the old round larger than the deterministic truncation notice; this
    # guarantees that the full history exceeds the exact target-view budget.
    old_answer = Message.assistant("The parser looks correct. " + "detail " * 80)
    latest = _transaction("new", "z", payload_size=10)
    notice = Message(role="system", content=DEFAULT_TRUNCATION_NOTICE)
    target = [*prefix[:-1], notice, prefix[-1], *latest]

    window = ContextBuilder(max_chars=estimate_request_chars(target)).build(
        [*prefix, old_user, old_answer, *latest]
    )

    assert old_user not in window
    assert old_answer not in window
    assert window.omitted_blocks == 1


def test_context_overflow_does_not_drop_required_latest_block() -> None:
    prefix = _prefix()
    latest = _transaction("large", "x", payload_size=500)
    notice = Message(role="system", content=DEFAULT_TRUNCATION_NOTICE)
    budget_without_latest = estimate_request_chars([*prefix[:-1], notice, prefix[-1]])

    with pytest.raises(ContextOverflow, match="latest complete"):
        ContextBuilder(max_chars=budget_without_latest).build([*prefix, *latest])


def test_context_counts_tool_schemas_and_reserved_space() -> None:
    prefix = _prefix()
    schema = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "x" * 200,
            "parameters": {"type": "object"},
        },
    }
    prefix_without_schema = estimate_request_chars(prefix)

    with pytest.raises(ContextOverflow, match="tool schemas"):
        ContextBuilder(max_chars=prefix_without_schema + 10).build(
            prefix, [schema], reserved_chars=10
        )


def test_context_truncation_keeps_all_system_messages_before_first_user() -> None:
    prefix = [
        Message(role="system", content="Primary policy."),
        Message(role="system", content="Workspace policy."),
        Message(role="user", content="Fix the project."),
    ]
    old = _transaction("old-order", "o", payload_size=300)
    latest = _transaction("new-order", "n", payload_size=10)
    notice = Message(role="system", content=DEFAULT_TRUNCATION_NOTICE)
    target = [*prefix[:-1], notice, prefix[-1], *latest]

    window = ContextBuilder(max_chars=estimate_request_chars(target)).build(
        [*prefix, *old, *latest]
    )

    assert window.truncated
    assert [message.role for message in window.messages[:4]] == [
        "system",
        "system",
        "system",
        "user",
    ]
    first_user = next(
        index for index, message in enumerate(window.messages) if message.role == "user"
    )
    assert all(
        message.role != "system" for message in window.messages[first_user + 1 :]
    )
    # The notice is a derived model-view message; canonical input remains
    # unchanged and contains only its two original system instructions.
    assert DEFAULT_TRUNCATION_NOTICE not in {
        message.content for message in [*prefix, *old, *latest]
    }


def test_context_character_estimate_counts_reasoning_content() -> None:
    call = ToolCall("reasoning-size", "read_file", '{"path":"sample.py"}')
    without_reasoning = Message.assistant(None, (call,))
    reasoning = "inspect imports, then follow the call graph"
    with_reasoning = Message.assistant(
        None,
        (call,),
        reasoning_content=reasoning,
    )

    plain_size = estimate_request_chars([without_reasoning])
    reasoning_size = estimate_request_chars([with_reasoning])

    assert reasoning_size > plain_size
    assert reasoning in with_reasoning.to_api_dict()["reasoning_content"]


@pytest.mark.parametrize(
    "history, expected_fragment",
    [
        (
            [
                *_prefix(),
                Message(
                    role="tool",
                    content="{}",
                    tool_call_id="orphan",
                    name="read_file",
                ),
            ],
            "orphan",
        ),
        (
            [
                *_prefix(),
                Message.assistant(None, (ToolCall("missing", "read_file", "{}"),)),
            ],
            "followed immediately",
        ),
        (
            [
                *_prefix(),
                Message.assistant(None, (ToolCall("a", "read_file", "{}"),)),
                Message(
                    role="tool",
                    content="{}",
                    tool_call_id="a",
                    name="write_file",
                ),
            ],
            "expected",
        ),
    ],
)
def test_context_rejects_broken_tool_protocol(
    history: list[Message], expected_fragment: str
) -> None:
    with pytest.raises(InvariantViolation, match=expected_fragment):
        ContextBuilder().build(history)


def test_context_rejects_reused_call_id_across_transactions() -> None:
    history = [*_prefix(), *_transaction("same", "a"), *_transaction("same", "b")]
    with pytest.raises(InvariantViolation, match="reused"):
        ContextBuilder().build(history)


def test_context_requires_system_and_original_task() -> None:
    with pytest.raises(InvariantViolation, match="system"):
        ContextBuilder().build([Message(role="user", content="task")])
    with pytest.raises(InvariantViolation, match="original user task"):
        ContextBuilder().build([Message(role="system", content="rules")])
