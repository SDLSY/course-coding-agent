"""Deterministic tests for the explicit Agent state machine.

These tests do not patch an SDK and do not make network requests.  Each
``ScriptedModel`` response is chosen in advance, which lets the assertions
focus on controller behaviour: history ordering, retries, tool dispatch, and
terminal reasons.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from coding_agent.agent import AgentRuntime
from coding_agent.context import ContextBuilder, estimate_request_chars
from coding_agent.errors import (
    ContextOverflow,
    PermanentModelError,
    TransientModelError,
)
from coding_agent.policy import AgentLimits, RetryPolicy
from coding_agent.tools.base import Tool, ToolOutput
from coding_agent.tools.registry import ToolRegistry, build_default_registry
from coding_agent.types import ModelTurn, RunPhase, ToolCall, Usage
from tests.fakes import ScriptedModel


def _value_tool(record: list[str] | None = None) -> Tool:
    """Return a tiny deterministic tool suitable for controller tests."""

    def handler(arguments: dict[str, Any]) -> ToolOutput:
        value = arguments["value"]
        if record is not None:
            record.append(value)
        return ToolOutput(content=f"observed:{value}", metadata={"value": value})

    return Tool(
        name="record_value",
        description="Record and return one string value.",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=handler,
    )


def _runtime(
    model: Any,
    *,
    registry: ToolRegistry | None = None,
    limits: AgentLimits | None = None,
    event_sink: Any | None = None,
    sleep: Any = lambda _seconds: None,
) -> AgentRuntime:
    return AgentRuntime(
        model_client=model,
        tool_registry=registry or ToolRegistry(),
        context_builder=ContextBuilder(max_chars=20_000),
        limits=limits or AgentLimits(),
        event_sink=event_sink,
        retry_policy=RetryPolicy(jitter_ratio=0),
        sleep=sleep,
    )


def test_direct_final_response_completes_without_claiming_verification() -> None:
    model = ScriptedModel(
        [
            ModelTurn(
                text="No changes were needed.",
                finish_reason="stop",
                usage=Usage(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            )
        ]
    )

    result = _runtime(model).run("Inspect the project.")

    assert result.phase is RunPhase.COMPLETED
    assert result.final_text == "No changes were needed."
    assert result.reason == "model returned a final response"
    assert [message.role for message in result.history] == [
        "system",
        "user",
        "assistant",
    ]
    assert result.usage == Usage(10, 4, 14)


def test_tool_result_is_paired_before_the_next_model_request() -> None:
    call = ToolCall("call-1", "record_value", '{"value":"alpha"}')
    model = ScriptedModel(
        [
            ModelTurn(tool_calls=(call,), finish_reason="tool_calls"),
            ModelTurn(text="Recorded alpha.", finish_reason="stop"),
        ]
    )

    result = _runtime(model, registry=ToolRegistry([_value_tool()])).run(
        "Record one value."
    )

    assert result.phase is RunPhase.COMPLETED
    assert result.tool_calls == 1
    assert [message.role for message in model.calls[1].messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    tool_message = model.calls[1].messages[-1]
    assert tool_message.tool_call_id == "call-1"
    assert json.loads(tool_message.content or "") == {
        "ok": True,
        "content": "observed:alpha",
        "metadata": {"value": "alpha"},
    }


def test_multiple_tool_calls_execute_serially_in_model_order() -> None:
    observed: list[str] = []
    calls = (
        ToolCall("a", "record_value", '{"value":"first"}'),
        ToolCall("b", "record_value", '{"value":"second"}'),
    )
    model = ScriptedModel([ModelTurn(tool_calls=calls), ModelTurn(text="Finished.")])

    result = _runtime(
        model,
        registry=ToolRegistry([_value_tool(observed)]),
    ).run("Record values in order.")

    assert observed == ["first", "second"]
    assert result.tool_calls == 2
    assert [
        message.tool_call_id for message in result.history if message.role == "tool"
    ] == [
        "a",
        "b",
    ]


def test_invalid_json_becomes_a_recoverable_tool_result() -> None:
    bad_call = ToolCall("bad-json", "record_value", '{"value":')
    model = ScriptedModel(
        [ModelTurn(tool_calls=(bad_call,)), ModelTurn(text="I corrected the request.")]
    )

    result = _runtime(model, registry=ToolRegistry([_value_tool()])).run(
        "Handle malformed arguments."
    )

    assert result.phase is RunPhase.COMPLETED
    error_envelope = json.loads(model.calls[1].messages[-1].content or "")
    assert error_envelope["ok"] is False
    assert error_envelope["error_code"] == "invalid_json"


def test_transient_model_failure_retries_without_consuming_a_model_turn() -> None:
    delays: list[float] = []
    model = ScriptedModel(
        [TransientModelError("temporary"), ModelTurn(text="Recovered.")]
    )

    result = _runtime(model, sleep=delays.append).run("Retry a temporary failure.")

    assert result.phase is RunPhase.COMPLETED
    assert result.model_turns == 1
    assert len(model.calls) == 2
    assert delays == [0.5]


def test_permanent_model_failure_stops_without_retry() -> None:
    model = ScriptedModel([PermanentModelError("authentication rejected")])

    result = _runtime(model).run("Do not retry permanent failures.")

    assert result.phase is RunPhase.FAILED
    assert result.reason == "runtime error: PermanentModelError"
    assert len(model.calls) == 1


def test_empty_model_response_uses_bounded_protocol_retry() -> None:
    model = ScriptedModel([ModelTurn(), ModelTurn(text="Valid response.")])
    limits = AgentLimits(max_protocol_retries=1)

    result = _runtime(model, limits=limits).run("Retry an empty response.")

    assert result.phase is RunPhase.COMPLETED
    assert result.model_turns == 1
    assert len(model.calls) == 2


def test_protocol_retry_exhaustion_enters_failed_state() -> None:
    model = ScriptedModel([ModelTurn(), ModelTurn()])
    limits = AgentLimits(max_protocol_retries=1)

    result = _runtime(model, limits=limits).run("Stop after invalid responses.")

    assert result.phase is RunPhase.FAILED
    assert result.reason == "runtime error: ResponseProtocolError"
    assert result.model_turns == 0


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_incomplete_text_finish_reason_is_not_treated_as_completion(
    finish_reason: str,
) -> None:
    """A non-empty fragment is still incomplete when the provider says so."""

    model = ScriptedModel(
        [
            ModelTurn(
                text="This answer stops in the middle of",
                finish_reason=finish_reason,
            ),
            ModelTurn(
                text="Complete answer after protocol retry.", finish_reason="stop"
            ),
        ]
    )

    result = _runtime(
        model,
        limits=AgentLimits(max_protocol_retries=1),
    ).run("Do not accept a truncated final response.")

    assert result.phase is RunPhase.COMPLETED
    assert result.final_text == "Complete answer after protocol retry."
    assert result.model_requests == 2
    assert result.model_turns == 1
    assert all(
        message.content != "This answer stops in the middle of"
        for message in result.history
    )


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_incomplete_tool_call_finish_reason_never_executes_side_effects(
    finish_reason: str,
) -> None:
    observed: list[str] = []
    incomplete_call = ToolCall(
        "must-not-run",
        "record_value",
        '{"value":"side effect"}',
    )
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(incomplete_call,),
                finish_reason=finish_reason,
            ),
            ModelTurn(text="Stopped before executing the incomplete call."),
        ]
    )

    result = _runtime(
        model,
        registry=ToolRegistry([_value_tool(observed)]),
        limits=AgentLimits(max_protocol_retries=1),
    ).run("Never execute an incomplete tool request.")

    assert result.phase is RunPhase.COMPLETED
    assert observed == []
    assert result.tool_calls == 0
    assert result.model_requests == 2
    assert all(
        call.id != "must-not-run"
        for message in result.history
        if message.role == "assistant"
        for call in message.tool_calls
    )


def test_protocol_retry_accounts_for_usage_and_requests() -> None:
    model = ScriptedModel(
        [
            # The provider returned and billed this response even though its
            # empty protocol payload cannot advance canonical history.
            ModelTurn(
                usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11)
            ),
            ModelTurn(
                text="Usable response.",
                usage=Usage(prompt_tokens=12, completion_tokens=3, total_tokens=15),
            ),
        ]
    )

    result = _runtime(
        model,
        limits=AgentLimits(max_protocol_retries=1),
    ).run("Account for every provider response.")

    assert result.phase is RunPhase.COMPLETED
    assert result.model_requests == 2
    assert result.model_turns == 1
    assert result.usage == Usage(
        prompt_tokens=22,
        completion_tokens=4,
        total_tokens=26,
    )
    assert len(model.calls) == result.model_requests


def test_provider_context_overflow_retries_with_a_strictly_smaller_view() -> None:
    def payload_handler(arguments: dict[str, Any]) -> ToolOutput:
        return ToolOutput(content="x" * arguments["size"])

    payload_tool = Tool(
        name="make_payload",
        description="Return deterministic text for a context-window test.",
        parameters={
            "type": "object",
            "properties": {"size": {"type": "integer", "minimum": 1}},
            "required": ["size"],
            "additionalProperties": False,
        },
        handler=payload_handler,
    )
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(ToolCall("large-old", "make_payload", '{"size":5000}'),)
            ),
            ModelTurn(
                tool_calls=(ToolCall("small-latest", "make_payload", '{"size":1}'),)
            ),
            ContextOverflow("synthetic provider context rejection"),
            ModelTurn(text="Recovered with a smaller complete context."),
        ]
    )

    result = _runtime(
        model,
        registry=ToolRegistry([payload_tool]),
    ).run("Exercise provider context recovery.")

    assert result.phase is RunPhase.COMPLETED
    assert result.model_requests == 4
    failed_view = model.calls[2]
    retry_view = model.calls[3]
    failed_size = estimate_request_chars(failed_view.messages, failed_view.tools)
    retry_size = estimate_request_chars(retry_view.messages, retry_view.tools)
    assert retry_size < failed_size
    assert len(retry_view.messages) < len(failed_view.messages)

    # Shrinking must still retain a complete latest tool transaction rather
    # than creating an API-invalid orphan call or result.
    retry_call_ids = {
        call.id
        for message in retry_view.messages
        if message.role == "assistant"
        for call in message.tool_calls
    }
    retry_result_ids = {
        message.tool_call_id
        for message in retry_view.messages
        if message.role == "tool"
    }
    assert retry_call_ids == retry_result_ids == {"small-latest"}


def test_model_returning_after_run_deadline_enters_limit_reached() -> None:
    class SlowModel:
        def __init__(self) -> None:
            self.timeouts: list[float | None] = []

        def complete(
            self,
            messages: Any,
            tools: Any,
            *,
            timeout_seconds: float | None = None,
        ) -> ModelTurn:
            self.timeouts.append(timeout_seconds)
            # Simulate a gateway which returns just after the timeout rather
            # than raising its own timeout exception at the exact boundary.
            time.sleep(0.04)
            return ModelTurn(
                text="Too late to accept.",
                usage=Usage(prompt_tokens=2, completion_tokens=2, total_tokens=4),
            )

    model = SlowModel()
    result = _runtime(
        model,
        limits=AgentLimits(max_wall_time_seconds=0.02),
    ).run("Enforce one absolute wall-time deadline.")

    assert result.phase is RunPhase.LIMIT_REACHED
    assert result.reason == "maximum wall time reached"
    assert result.final_text is None
    assert result.model_requests == 1
    assert result.model_turns == 0
    assert result.usage == Usage(2, 2, 4)
    assert len(model.timeouts) == 1
    assert model.timeouts[0] is not None
    assert 0 < model.timeouts[0] <= 0.02


def test_event_sink_oserror_is_disabled_without_failing_the_run() -> None:
    class FailingEventSink:
        def __init__(self) -> None:
            self.calls = 0

        def emit(self, event_type: str, **data: object) -> None:
            self.calls += 1
            raise OSError("synthetic trace destination failure")

    sink = FailingEventSink()
    model = ScriptedModel([ModelTurn(text="Task still completed.")])

    result = _runtime(model, event_sink=sink).run(
        "Do not couple diagnostic I/O to agent correctness."
    )

    assert result.phase is RunPhase.COMPLETED
    assert result.final_text == "Task still completed."
    assert result.model_requests == 1
    assert len(model.calls) == 1
    # The first failure disables the sink, preventing every later event from
    # repeating the same disk or broken-pipe error.
    assert sink.calls == 1


def test_oversized_tool_batch_is_rejected_without_partial_side_effects() -> None:
    observed: list[str] = []
    calls = (
        ToolCall("a", "record_value", '{"value":"first"}'),
        ToolCall("b", "record_value", '{"value":"second"}'),
    )
    model = ScriptedModel([ModelTurn(tool_calls=calls)])
    limits = AgentLimits(max_tool_calls=1)

    result = _runtime(
        model,
        registry=ToolRegistry([_value_tool(observed)]),
        limits=limits,
    ).run("Do not execute a partial batch.")

    assert result.phase is RunPhase.LIMIT_REACHED
    assert observed == []
    assert result.tool_calls == 2
    tool_messages = [message for message in result.history if message.role == "tool"]
    assert len(tool_messages) == 2
    assert all(
        json.loads(message.content or "")["error_code"] == "tool_budget_exceeded"
        for message in tool_messages
    )


def test_final_allowed_model_turn_spent_on_tools_ends_at_limit() -> None:
    call = ToolCall("only", "record_value", '{"value":"x"}')
    model = ScriptedModel([ModelTurn(tool_calls=(call,))])
    limits = AgentLimits(max_model_turns=1)

    result = _runtime(
        model,
        registry=ToolRegistry([_value_tool()]),
        limits=limits,
    ).run("Use the last turn.")

    assert result.phase is RunPhase.LIMIT_REACHED
    assert result.reason == "maximum model turns reached after tool execution"
    assert result.model_turns == 1


def test_reused_tool_call_id_is_protocol_retried_before_history_append() -> None:
    first = ToolCall("same-id", "record_value", '{"value":"one"}')
    reused = ToolCall("same-id", "record_value", '{"value":"two"}')
    model = ScriptedModel(
        [
            ModelTurn(tool_calls=(first,)),
            ModelTurn(tool_calls=(reused,)),
            ModelTurn(text="Stopped reusing IDs."),
        ]
    )

    result = _runtime(
        model,
        registry=ToolRegistry([_value_tool()]),
        limits=AgentLimits(max_protocol_retries=1),
    ).run("Keep call IDs unique.")

    assert result.phase is RunPhase.COMPLETED
    # Only the first valid call is persisted and executed.  The reused call is
    # rejected before it can corrupt canonical history.
    assistant_calls = [
        call.id
        for message in result.history
        if message.role == "assistant"
        for call in message.tool_calls
    ]
    assert assistant_calls == ["same-id"]


def test_end_to_end_scripted_model_edits_file_and_runs_real_check(
    tmp_path: Any,
) -> None:
    """Exercise the complete local read/edit/command loop without an API.

    The model remains scripted so the test is deterministic, but every tool is
    the real production implementation bound to a temporary workspace.  This
    catches integration defects that isolated state-machine and tool tests
    cannot see, such as schema names disagreeing with the controller.
    """

    source = tmp_path / "calculator.py"
    source.write_text(
        "def add(left, right):\n    return left - right\n",
        encoding="utf-8",
    )
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "read",
                        "read_file",
                        '{"path":"calculator.py","start_line":1,"end_line":20}',
                    ),
                )
            ),
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "edit",
                        "replace_in_file",
                        '{"path":"calculator.py","old":"return left - right",'
                        '"new":"return left + right"}',
                    ),
                )
            ),
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "check",
                        "run_command",
                        '{"command":"python3 -c \\"from calculator import add; '
                        'assert add(2, 3) == 5\\"","timeout_seconds":5}',
                    ),
                )
            ),
            ModelTurn(text="Fixed addition and ran the check successfully."),
        ]
    )

    result = _runtime(
        model,
        registry=build_default_registry(tmp_path),
    ).run("Fix the add function and verify it.")

    assert result.phase is RunPhase.COMPLETED
    assert "left + right" in source.read_text(encoding="utf-8")
    check_message = next(
        message
        for message in result.history
        if message.role == "tool" and message.tool_call_id == "check"
    )
    check_result = json.loads(check_message.content or "")
    assert check_result["ok"] is True
    assert check_result["metadata"]["exit_code"] == 0
