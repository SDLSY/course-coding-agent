"""Tests for provider response normalization and the thin model adapter."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from coding_agent.errors import (
    ConfigurationError,
    ContextOverflow,
    PermanentModelError,
    ReasoningEffortUnsupported,
    ResponseProtocolError,
    ToolRequestError,
    TransientModelError,
)
from coding_agent.model import OpenAICompatibleModelClient, ReasoningCapability
from coding_agent.response_parser import normalize_chat_completion, parse_tool_arguments
from coding_agent.types import Message, ToolCall, ToolResult
from tests.fakes import FakeOpenAIClient, chat_completion, raw_tool_call


def test_normalizer_preserves_raw_arguments_and_usage() -> None:
    raw_arguments = ' { "path" : "src/main.py" } '
    response = chat_completion(
        content="I will inspect the file.",
        tool_calls=[raw_tool_call("call_1", "read_file", raw_arguments)],
        finish_reason="tool_calls",
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )

    turn = normalize_chat_completion(response)

    assert turn.text == "I will inspect the file."
    assert turn.finish_reason == "tool_calls"
    assert turn.tool_calls == (ToolCall("call_1", "read_file", raw_arguments),)
    assert turn.tool_calls[0].arguments_json == raw_arguments
    assert turn.usage is not None
    assert turn.usage.total_tokens == 14


def test_normalizer_accepts_sdk_style_objects() -> None:
    function = SimpleNamespace(name="list_files", arguments='{"path":"."}')
    raw_call = SimpleNamespace(id="sdk_call", type="function", function=function)
    message = SimpleNamespace(content=None, tool_calls=[raw_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    response = SimpleNamespace(choices=[choice], usage=None)

    turn = normalize_chat_completion(response)

    assert turn.text is None
    assert turn.tool_calls[0].id == "sdk_call"


def test_adapter_round_trips_reasoning_content_into_the_next_tool_round() -> None:
    """Preserve provider thinking state without retaining arbitrary SDK fields."""

    call = raw_tool_call("reasoning_call", "read_file", '{"path":"sample.py"}')
    reasoning = "I should inspect the target file before proposing a change."
    fake = FakeOpenAIClient(
        [
            chat_completion(
                content="",
                reasoning_content=reasoning,
                tool_calls=[call],
                finish_reason="tool_calls",
            ),
            chat_completion(content="The file has been inspected."),
        ]
    )
    adapter = OpenAICompatibleModelClient(model="reasoning-model", client=fake)
    schema = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    initial = Message(role="user", content="Inspect sample.py")

    first_turn = adapter.complete([initial], [schema])
    tool_result = ToolResult(
        call_id="reasoning_call",
        name="read_file",
        ok=True,
        content="sample contents",
    ).to_message()
    adapter.complete(
        [initial, first_turn.as_message(), tool_result],
        [schema],
    )

    assert first_turn.reasoning_content == reasoning
    replayed_assistant = fake.completions.requests[1]["messages"][1]
    assert replayed_assistant["reasoning_content"] == reasoning
    assert replayed_assistant["tool_calls"] == [call]
    assert (
        Message.assistant("", reasoning_content="").to_api_dict()["reasoning_content"]
        == ""
    )
    # Standard messages remain standard; an absent provider extension must not
    # be synthesized as null because strict gateways may reject unknown fields.
    assert "reasoning_content" not in initial.to_api_dict()


@pytest.mark.parametrize(
    "response, expected_fragment",
    [
        ({"choices": []}, "no choices"),
        ({"choices": [{"message": {"content": 42}}]}, "content"),
        (
            chat_completion(
                tool_calls=[
                    raw_tool_call("same", "first", "{}"),
                    raw_tool_call("same", "second", "{}"),
                ]
            ),
            "duplicate",
        ),
        (
            chat_completion(
                tool_calls=[{"id": "x", "function": {"name": "read_file"}}]
            ),
            "arguments",
        ),
        (
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": 42,
                        }
                    }
                ]
            },
            "reasoning_content",
        ),
    ],
)
def test_normalizer_rejects_malformed_response(
    response: dict[str, object], expected_fragment: str
) -> None:
    with pytest.raises(ResponseProtocolError, match=expected_fragment):
        normalize_chat_completion(response)


def test_parse_tool_arguments_requires_strict_json_object() -> None:
    call = ToolCall("c1", "read_file", '{"path":"a.py","start_line":1}')
    assert parse_tool_arguments(call) == {"path": "a.py", "start_line": 1}

    with pytest.raises(ToolRequestError) as invalid_json:
        parse_tool_arguments(ToolCall("c2", "read_file", "{not-json}"))
    assert invalid_json.value.error_code == "invalid_json"

    with pytest.raises(ToolRequestError) as duplicate:
        parse_tool_arguments(ToolCall("c3", "read_file", '{"path":"a","path":"b"}'))
    assert duplicate.value.error_code == "duplicate_argument"

    with pytest.raises(ToolRequestError) as wrong_shape:
        parse_tool_arguments(ToolCall("c4", "read_file", '["a.py"]'))
    assert wrong_shape.value.error_code == "arguments_not_object"

    with pytest.raises(ToolRequestError, match="strict JSON"):
        parse_tool_arguments(ToolCall("c5", "run_command", '{"x":NaN}'))


def test_tool_result_encodes_structured_tool_message() -> None:
    result = ToolResult(
        call_id="c1",
        name="run_command",
        ok=False,
        content="command could not start",
        metadata={"attempted": True},
        error_code="execution_error",
    )

    message = result.to_message()

    assert message.role == "tool"
    assert message.tool_call_id == "c1"
    assert message.name == "run_command"
    assert '"ok":false' in (message.content or "")
    assert '"error_code":"execution_error"' in (message.content or "")
    assert "name" not in message.to_api_dict()


def test_adapter_serializes_internal_messages_and_normalizes_response() -> None:
    fake = FakeOpenAIClient([chat_completion(content="done")])
    adapter = OpenAICompatibleModelClient(
        model="test-model",
        client=fake,
        temperature=0.1,
        timeout_seconds=12,
    )
    messages = [
        Message(role="system", content="You are a coding agent."),
        Message(role="user", content="Inspect the project."),
    ]

    turn = adapter.complete(messages, tools=[])

    assert turn.text == "done"
    request = fake.completions.requests[0]
    assert request["model"] == "test-model"
    assert request["messages"] == [message.to_api_dict() for message in messages]
    assert request["stream"] is False
    assert request["temperature"] == 0.1
    assert request["timeout"] == 12.0
    assert "tools" not in request


def test_adapter_uses_the_smaller_runtime_timeout_and_omits_default_temperature() -> (
    None
):
    fake = FakeOpenAIClient([chat_completion(content="done")])
    adapter = OpenAICompatibleModelClient(
        model="test-model",
        client=fake,
        timeout_seconds=12,
    )

    adapter.complete(
        [Message(role="user", content="task")],
        [],
        timeout_seconds=0.25,
    )

    request = fake.completions.requests[0]
    assert request["timeout"] == 0.25
    assert "temperature" not in request


def test_adapter_disables_hidden_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed_with: dict[str, object] = {}

    def construct_openai(**options: object) -> FakeOpenAIClient:
        constructed_with.update(options)
        return FakeOpenAIClient([])

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=construct_openai),
    )

    OpenAICompatibleModelClient(
        model="test-model",
        api_key="synthetic-credential-for-construction-test",
        base_url="https://example.invalid/v1",
    )

    assert constructed_with["max_retries"] == 0


def test_adapter_sends_tools_without_mutating_schema() -> None:
    fake = FakeOpenAIClient([chat_completion(content="done")])
    adapter = OpenAICompatibleModelClient(model="m", client=fake)
    schema = {
        "type": "function",
        "function": {
            "name": "read_file",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    adapter.complete([Message(role="user", content="task")], [schema])

    assert fake.completions.requests[0]["tools"] == [schema]
    assert schema["function"]["name"] == "read_file"


def test_reasoning_probe_accepts_high_and_forwards_native_field() -> None:
    fake = FakeOpenAIClient(
        [chat_completion(content="ok"), chat_completion(content="done")]
    )
    adapter = OpenAICompatibleModelClient(
        model="m",
        client=fake,
        api_key="EXAMPLE_CREDENTIAL_NOT_REAL",
        reasoning_effort="high",
    )

    capability = adapter.probe_reasoning_effort()
    adapter.complete([Message(role="user", content="task")], [])

    assert capability == ReasoningCapability(
        status="supported",
        requested_effort="high",
        parameter="reasoning_effort",
        accepted_value="high",
    )
    assert fake.completions.requests[0]["reasoning_effort"] == "high"
    assert fake.completions.requests[1]["reasoning_effort"] == "high"


def test_reasoning_probe_tries_alternate_parameter_after_unknown_field() -> None:
    class UnknownField(Exception):
        status_code = 422

    fake = FakeOpenAIClient(
        [
            UnknownField("unknown parameter reasoning_effort"),
            chat_completion(content="ok"),
        ]
    )
    adapter = OpenAICompatibleModelClient(
        model="m", client=fake, api_key="EXAMPLE_CREDENTIAL_NOT_REAL"
    )

    capability = adapter.probe_reasoning_effort("high")

    assert capability.status == "supported"
    assert capability.parameter == "thinking"
    assert capability.accepted_value == "high"
    assert "reasoning_effort" in fake.completions.requests[0]
    assert fake.completions.requests[1]["thinking"] == "high"


def test_reasoning_probe_records_lower_native_level_after_value_rejection() -> None:
    class InvalidValue(Exception):
        status_code = 400

    fake = FakeOpenAIClient(
        [
            InvalidValue("invalid value for reasoning_effort; allowed values: low"),
            InvalidValue("invalid value for reasoning_effort; allowed values: low"),
            chat_completion(content="ok"),
        ]
    )
    adapter = OpenAICompatibleModelClient(
        model="m", client=fake, api_key="EXAMPLE_CREDENTIAL_NOT_REAL"
    )

    capability = adapter.probe_reasoning_effort("high")

    assert capability.status == "supported"
    assert capability.accepted_value == "low"
    assert [request["reasoning_effort"] for request in fake.completions.requests] == [
        "high",
        "medium",
        "low",
    ]


def test_reasoning_probe_marks_provider_error_and_redacts_secret() -> None:
    secret = "EXAMPLE_CREDENTIAL_NOT_REAL"

    class GatewayFailure(Exception):
        status_code = 503

    fake = FakeOpenAIClient(
        [GatewayFailure(f"temporary failure Authorization: Bearer {secret}")]
    )
    adapter = OpenAICompatibleModelClient(model="m", client=fake, api_key=secret)

    capability = adapter.probe_reasoning_effort("high")

    assert capability.status == "error"
    assert capability.error_type == "GatewayFailure"
    assert secret not in (capability.detail or "")
    assert "[REDACTED]" in (capability.detail or "")
    assert len(fake.completions.requests) == 1


def test_reasoning_probe_explicit_unsupported_fails_closed_at_runtime() -> None:
    fake = FakeOpenAIClient(
        [ReasoningEffortUnsupported("native reasoning is unsupported")]
    )
    adapter = OpenAICompatibleModelClient(
        model="m",
        client=fake,
        api_key="EXAMPLE_CREDENTIAL_NOT_REAL",
        reasoning_effort="high",
    )

    capability = adapter.probe_reasoning_effort()
    assert capability.status == "unsupported"
    with pytest.raises(ConfigurationError, match="unsupported"):
        adapter.complete([Message(role="user", content="task")], [])
    # No second request can accidentally downgrade or imitate reasoning.
    assert len(fake.completions.requests) == 1


def test_reasoning_value_preserves_boolean_and_dynamic_extra_field_is_protected() -> (
    None
):
    fake = FakeOpenAIClient([chat_completion(content="done")])
    adapter = OpenAICompatibleModelClient(
        model="m",
        client=fake,
        reasoning_parameter="thinking",
        reasoning_value=False,
    )
    adapter.complete([Message(role="user", content="task")], [])
    assert fake.completions.requests[0]["thinking"] is False

    with pytest.raises(ConfigurationError, match="managed fields"):
        OpenAICompatibleModelClient(
            model="m",
            client=FakeOpenAIClient([]),
            reasoning_parameter="thinking",
            reasoning_effort="high",
            extra_request_options={"thinking": "low"},
        )


class RateLimitError(Exception):
    status_code = 429


class AuthenticationError(Exception):
    status_code = 401


class BadRequestError(Exception):
    status_code = 400


def test_adapter_classifies_transient_and_permanent_provider_errors() -> None:
    message = [Message(role="user", content="task")]

    transient = OpenAICompatibleModelClient(
        model="m", client=FakeOpenAIClient([RateLimitError("slow down")])
    )
    with pytest.raises(TransientModelError, match="429"):
        transient.complete(message, [])

    permanent = OpenAICompatibleModelClient(
        model="m", client=FakeOpenAIClient([AuthenticationError("bad key")])
    )
    with pytest.raises(PermanentModelError, match="401"):
        permanent.complete(message, [])

    # HTTP authentication semantics take precedence over incidental context
    # wording in a provider error message.
    auth_with_context_wording = OpenAICompatibleModelClient(
        model="m",
        client=FakeOpenAIClient(
            [AuthenticationError("account cannot access this context window")]
        ),
    )
    with pytest.raises(PermanentModelError, match="401"):
        auth_with_context_wording.complete(message, [])


def test_adapter_classifies_context_overflow_and_redacts_key() -> None:
    # Use an unmistakable placeholder rather than an API-key-shaped fixture so
    # repository secret scanners do not report a false positive.
    secret = "EXAMPLE_CREDENTIAL_NOT_REAL"
    error = BadRequestError(
        f"maximum context length exceeded; Authorization: Bearer {secret}"
    )
    adapter = OpenAICompatibleModelClient(
        model="m",
        client=FakeOpenAIClient([error]),
        api_key=secret,
    )

    with pytest.raises(ContextOverflow) as captured:
        adapter.complete([Message(role="user", content="task")], [])

    assert secret not in str(captured.value)
    assert "[REDACTED]" in str(captured.value)


def test_adapter_does_not_hide_unknown_programming_errors() -> None:
    adapter = OpenAICompatibleModelClient(
        model="m", client=FakeOpenAIClient([AttributeError("bad fake")])
    )
    with pytest.raises(AttributeError, match="bad fake"):
        adapter.complete([Message(role="user", content="task")], [])


def test_adapter_validates_configuration_without_constructing_real_client() -> None:
    with pytest.raises(ConfigurationError, match="model"):
        OpenAICompatibleModelClient(model="", client=FakeOpenAIClient([]))
    with pytest.raises(ConfigurationError, match="managed fields"):
        OpenAICompatibleModelClient(
            model="m",
            client=FakeOpenAIClient([]),
            extra_request_options={"messages": []},
        )
    with pytest.raises(ConfigurationError, match="finite"):
        OpenAICompatibleModelClient(
            model="m",
            client=FakeOpenAIClient([]),
            timeout_seconds=float("nan"),
        )
