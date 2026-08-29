"""Dependency-free tests for the async Harbor bridge."""

from __future__ import annotations

import asyncio

from coding_agent.agent import RunResult
from coding_agent.harbor_adapter import (
    EnvironmentExecResult,
    HarborAgentAdapter,
    HarborExecutionBackend,
    RemoteCommandBackend,
    RemoteExecutionBackend,
    sanitized_host_environment,
    write_harbor_artifacts,
)
from coding_agent.types import Message, RunPhase, ToolCall


class FakeEnvironment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float | None]] = []

    async def exec(self, command: str, *, timeout_seconds=None):
        self.calls.append((command, timeout_seconds))
        return EnvironmentExecResult(stdout="ok", exit_code=0)


def test_remote_backend_bridges_worker_call_to_async_environment() -> None:
    async def scenario() -> None:
        environment = FakeEnvironment()
        backend = RemoteCommandBackend(environment)
        backend.bind_loop(asyncio.get_running_loop())

        result = await asyncio.to_thread(
            backend.execute_call,
            ToolCall("call-1", "run_command", '{"command":"pytest -q"}'),
            timeout_seconds=5,
        )

        assert result.ok is True
        assert result.metadata["exit_code"] == 0
        assert environment.calls == [("cd /app && pytest -q", 5)]

    asyncio.run(scenario())


def test_remote_backend_accepts_timeout_named_harbor_style() -> None:
    class HarborStyleEnvironment:
        async def exec(self, command: str, *, timeout=None):
            return {"stdout": command, "exit_code": 0}

    async def scenario() -> None:
        backend = RemoteCommandBackend(HarborStyleEnvironment())
        backend.bind_loop(asyncio.get_running_loop())
        result = await asyncio.to_thread(
            backend.execute_call,
            ToolCall("call-1", "run_command", '{"command":"true"}'),
            timeout_seconds=2,
        )
        assert result.ok is True

    asyncio.run(scenario())


def test_remote_backend_shell_quotes_configured_workspace_path() -> None:
    class RecordingEnvironment:
        async def exec(self, command: str, *, timeout_seconds=None):
            self.command = command
            return {"stdout": "", "exit_code": 0}

    async def scenario() -> None:
        environment = RecordingEnvironment()
        backend = RemoteCommandBackend(environment, workspace="/app/$HOME/`marker`")
        backend.bind_loop(asyncio.get_running_loop())
        result = await asyncio.to_thread(
            backend.execute_call,
            ToolCall("quoted", "run_command", '{"command":"true"}'),
        )
        assert result.ok is True
        assert environment.command == "cd '/app/$HOME/`marker`' && true"

    asyncio.run(scenario())


def test_remote_backend_requires_loop_binding() -> None:
    backend = RemoteCommandBackend(FakeEnvironment())
    result = backend.execute_call(
        ToolCall("call-1", "run_command", '{"command":"true"}')
    )
    assert result.ok is False
    assert result.error_code == "backend_not_bound"


def test_remote_execution_backend_exposes_six_tools_and_bridges_file_ops() -> None:
    async def scenario() -> None:
        environment = FakeEnvironment()
        backend = RemoteExecutionBackend(environment)
        backend.bind_loop(asyncio.get_running_loop())
        names = [item["function"]["name"] for item in backend.model_schemas()]
        assert names == [
            "list_files",
            "search_text",
            "read_file",
            "write_file",
            "replace_in_file",
            "run_command",
        ]

        read = await asyncio.to_thread(
            backend.execute_call,
            ToolCall("read", "read_file", '{"path":"src/main.py"}'),
        )
        write = await asyncio.to_thread(
            backend.execute_call,
            ToolCall("write", "write_file", '{"path":"notes.txt","content":"hello"}'),
        )
        assert read.ok and write.ok
        assert environment.calls[0][0].startswith("cat -- /app/src/main.py")
        assert "os.replace" in environment.calls[1][0]

    asyncio.run(scenario())


def test_harbor_execution_alias_is_the_full_backend() -> None:
    assert HarborExecutionBackend is RemoteExecutionBackend


def test_harbor_artifacts_write_atif_and_redacted_summary(tmp_path) -> None:
    secret = "synthetic-harbor-secret"
    run = RunResult(
        phase=RunPhase.COMPLETED,
        reason="model returned a final response",
        final_text="done",
        model_turns=1,
        model_requests=1,
        tool_calls=0,
        elapsed_seconds=0.01,
        usage=None,
        history=(
            Message(role="system", content="system"),
            Message(role="user", content="task"),
            Message.assistant(secret),
        ),
    )
    paths = write_harbor_artifacts(
        run,
        log_dir=tmp_path,
        task="task",
        run_id="run-1",
        secrets=(secret,),
    )
    assert paths["trajectory"].name == "trajectory.json"
    assert paths["summary"].name == "run.json"
    assert secret not in paths["trajectory"].read_text(encoding="utf-8")
    assert secret not in paths["summary"].read_text(encoding="utf-8")


def test_adapter_runs_runtime_off_event_loop_and_calls_hooks() -> None:
    class Runtime:
        def run(self, task: str):
            assert task == "task"
            return "result"

    async def scenario() -> None:
        seen: list[str] = []

        async def hook(value):
            seen.append(value)

        result = await HarborAgentAdapter(
            Runtime(), trace_hook=hook, atif_hook=hook
        ).run("task")
        assert result == "result"
        assert seen == ["result", "result"]

    asyncio.run(scenario())


def test_host_environment_sanitization_removes_credentials() -> None:
    safe = sanitized_host_environment(
        {"PATH": "/bin", "SERVICE_API_KEY": "secret", "NORMAL": "value"}
    )
    assert safe == {"PATH": "/bin", "NORMAL": "value"}
