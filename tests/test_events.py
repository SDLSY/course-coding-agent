from __future__ import annotations

import json
import os
from io import StringIO
from pathlib import Path

import pytest

from coding_agent.events import (
    REDACTED,
    ConsoleEventSink,
    JsonlEventSink,
    NullEventSink,
    RunEvent,
    redact,
)

FAKE_SECRET = "synthetic.test-secret-value"


def test_jsonl_sink_writes_one_stable_record_per_line(tmp_path: Path) -> None:
    trace = tmp_path / "private" / "run.jsonl"
    sink = JsonlEventSink(
        trace,
        clock=lambda: "2026-08-28T12:34:56.789Z",
    )

    sink.emit("run.started", model="test-model", attempt=1)
    sink.emit("tool.completed", ok=True, elapsed_ms=12.5)

    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records == [
        {
            "schema_version": 1,
            "timestamp": "2026-08-28T12:34:56.789Z",
            "event": "run.started",
            "data": {"model": "test-model", "attempt": 1},
        },
        {
            "schema_version": 1,
            "timestamp": "2026-08-28T12:34:56.789Z",
            "event": "tool.completed",
            "data": {"ok": True, "elapsed_ms": 12.5},
        },
    ]


def test_new_trace_file_is_private_to_current_user(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"
    JsonlEventSink(trace)

    # A restrictive umask may remove more permissions; no group/other bit may
    # be introduced by the sink's creation mode.
    assert os.stat(trace).st_mode & 0o077 == 0


def test_redaction_covers_nested_fields_headers_patterns_and_known_values(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "run.jsonl"
    sink = JsonlEventSink(
        trace,
        secrets=[FAKE_SECRET],
        clock=lambda: "2026-08-28T00:00:00.000Z",
    )
    fake_openai_style_key = "sk-" + "exampleOnly123456"
    sink.emit(
        "model.failed",
        api_key=FAKE_SECRET,
        nested={
            "Authorization": f"Bearer {FAKE_SECRET}",
            "message": f"upstream repeated {FAKE_SECRET}",
            "password": "another-value",
        },
        accidental=f"received {fake_openai_style_key} from an invalid fixture",
        url="https://user:password@example.test/v1",
    )

    raw_text = trace.read_text(encoding="utf-8")
    record = json.loads(raw_text)

    assert FAKE_SECRET not in raw_text
    assert "another-value" not in raw_text
    assert fake_openai_style_key not in raw_text
    assert "user:password@" not in raw_text
    assert record["data"]["api_key"] == REDACTED
    assert record["data"]["nested"]["Authorization"] == REDACTED
    assert REDACTED in record["data"]["nested"]["message"]


def test_safe_metadata_named_token_count_or_key_env_is_preserved() -> None:
    safe = redact(
        {
            "input_tokens": 123,
            "token_count": 456,
            "api_key_env": "DEEPSEEK_API_KEY",
            "key_env": "CUSTOM_BENCHMARK_KEY",
        }
    )

    assert safe == {
        "input_tokens": 123,
        "token_count": 456,
        "api_key_env": "DEEPSEEK_API_KEY",
        "key_env": "CUSTOM_BENCHMARK_KEY",
    }


def test_redaction_treats_camel_case_credentials_like_snake_case() -> None:
    fixture_token = "opaque-fixture-" + "token"
    safe = redact(
        {
            "clientSecret": "camel-client-secret",
            "accessToken": "camel-access-token",
            "refreshToken": "camel-refresh-token",
            "apiKey": "camel-api-key",
            "APIKey": "acronym-api-key",
            "nested": {
                "client_secret": "snake-client-secret",
                "access_token": "snake-access-token",
            },
            # Metrics and environment-variable *names* are useful diagnostics,
            # not credentials.  Their camelCase forms must retain the same
            # exceptions as the existing snake_case forms.
            "inputTokens": 123,
            "tokenCount": 456,
            "apiKeyEnv": "CUSTOM_PROVIDER_KEY",
            "message": "request used Bearer " + fixture_token,
        }
    )

    assert safe["clientSecret"] == REDACTED
    assert safe["accessToken"] == REDACTED
    assert safe["refreshToken"] == REDACTED
    assert safe["apiKey"] == REDACTED
    assert safe["APIKey"] == REDACTED
    assert safe["nested"] == {
        "client_secret": REDACTED,
        "access_token": REDACTED,
    }
    assert safe["inputTokens"] == 123
    assert safe["tokenCount"] == 456
    assert safe["apiKeyEnv"] == "CUSTOM_PROVIDER_KEY"
    assert safe["message"] == "request used Bearer [REDACTED]"


def test_redaction_handles_cycles_and_non_json_objects() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    safe = redact(
        {
            "cycle": cyclic,
            "binary": b"private bytes are not copied",
            "path": Path("relative/file.py"),
            "unknown": object(),
            "not_a_number": float("nan"),
        }
    )

    assert safe["cycle"] == {"self": {"cycle": "[CYCLE]"}}
    assert safe["binary"] == "<bytes: 28 bytes>"
    assert safe["path"] == "relative/file.py"
    assert safe["unknown"] == "<object>"
    assert safe["not_a_number"] == "NaN"


def test_console_sink_applies_same_secret_boundary() -> None:
    output = StringIO()
    sink = ConsoleEventSink(secrets=[FAKE_SECRET], stream=output)

    sink.emit("model.retried", message=f"credential={FAKE_SECRET}")

    rendered = output.getvalue()
    assert FAKE_SECRET not in rendered
    assert REDACTED in rendered
    assert rendered.startswith("[model.retried]")


def test_null_sink_validates_event_names_but_otherwise_does_nothing() -> None:
    sink = NullEventSink()
    sink.emit("run.started", arbitrary=object())

    with pytest.raises(ValueError, match="event_type"):
        sink.emit("Run Started")


def test_run_event_rejects_invalid_type_and_non_mapping_data() -> None:
    event = RunEvent("run.started", data={"api_key": FAKE_SECRET})
    assert FAKE_SECRET not in repr(event)

    with pytest.raises(ValueError, match="event_type"):
        RunEvent("INVALID")
    with pytest.raises(TypeError, match="mapping"):
        RunEvent("run.started", data=["not", "a", "mapping"])  # type: ignore[arg-type]
