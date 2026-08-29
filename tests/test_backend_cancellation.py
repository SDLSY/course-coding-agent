"""Cancellation and execution-backend protocol boundary tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from coding_agent.agent import AgentRuntime
from coding_agent.context import ContextBuilder
from coding_agent.policy import AgentLimits
from coding_agent.types import ModelTurn, RunPhase, ToolCall, ToolResult
from tests.fakes import ScriptedModel


class RecordingBackend:
    """Small backend double that returns complete protocol results."""

    def __init__(self, on_execute) -> None:
        self.calls: list[str] = []
        self.on_execute = on_execute

    def model_schemas(self) -> Sequence[Mapping[str, Any]]:
        return ()

    def execute_call(self, call: ToolCall, *, timeout_seconds=None) -> ToolResult:
        self.calls.append(call.id)
        self.on_execute()
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=True,
            content=f"completed:{call.id}",
        )


def test_cooperative_cancel_skips_remaining_calls_but_pairs_every_result() -> None:
    cancelled = False

    def mark_executed() -> None:
        nonlocal cancelled
        cancelled = True

    backend = RecordingBackend(mark_executed)
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall("first", "remote_op", "{}"),
                    ToolCall("second", "remote_op", "{}"),
                )
            )
        ]
    )
    runtime = AgentRuntime(
        model,
        backend,
        ContextBuilder(max_chars=10_000),
        limits=AgentLimits(max_model_turns=2, max_tool_calls=4),
        cancel_check=lambda: cancelled,
    )

    result = runtime.run("run two operations")

    assert result.phase is RunPhase.CANCELLED
    assert backend.calls == ["first"]
    assert [message.role for message in result.history] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    skipped = result.history[-1]
    assert skipped.tool_call_id == "second"
    assert '"error_code":"cancelled"' in (skipped.content or "")
