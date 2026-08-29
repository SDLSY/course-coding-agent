"""Protocol-safety tests for ATIF trajectory projection."""

from __future__ import annotations

import pytest

from coding_agent.agent import RunResult
from coding_agent.atif import export_atif
from coding_agent.errors import InvariantViolation
from coding_agent.types import Message, RunPhase, ToolCall


def _run(history: tuple[Message, ...]) -> RunResult:
    return RunResult(
        phase=RunPhase.COMPLETED,
        reason="model returned a final response",
        final_text="done",
        model_turns=1,
        model_requests=1,
        tool_calls=sum(len(item.tool_calls) for item in history),
        elapsed_seconds=0.01,
        usage=None,
        history=history,
    )


def test_atif_rejects_tool_result_without_matching_name() -> None:
    call = ToolCall("c1", "read_file", '{"path":"x"}')
    history = (
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message.assistant(None, (call,)),
        # ``name=None`` is legal as a raw Message value but is not a complete
        # canonical ToolResult transaction and must not become ATIF evidence.
        Message(role="tool", content="ok", tool_call_id="c1"),
    )

    with pytest.raises(InvariantViolation, match="name"):
        export_atif(_run(history), task="task")


def test_atif_rejects_duplicate_call_ids_across_turns() -> None:
    first = ToolCall("same", "read_file", '{"path":"a"}')
    second = ToolCall("same", "read_file", '{"path":"b"}')
    history = (
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        Message.assistant(None, (first,)),
        Message(role="tool", content="a", tool_call_id="same", name="read_file"),
        Message.assistant(None, (second,)),
        Message(role="tool", content="b", tool_call_id="same", name="read_file"),
    )

    with pytest.raises(InvariantViolation, match="duplicate"):
        export_atif(_run(history), task="task")
