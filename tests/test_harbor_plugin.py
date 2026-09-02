"""Contract tests for the optional Harbor 0.22 agent entry point."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_agent.errors import ConfigurationError
from coding_agent.harbor_plugin import CourseCodingAgent
from coding_agent.model import ReasoningCapability
from coding_agent.types import ModelTurn, RunPhase, ToolCall


class _Context:
    def __init__(self) -> None:
        self.n_input_tokens = None
        self.n_cache_tokens = None
        self.n_output_tokens = None
        self.cost_usd = None
        self.metadata = None


class _Environment:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, dict[str, str] | None, int | None]] = []

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user=None,
    ) -> SimpleNamespace:
        self.calls.append((command, cwd, env, timeout_sec))
        return SimpleNamespace(stdout="remote-output", stderr="", return_code=0)


class _ScriptedModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.turns = [
            ModelTurn(
                tool_calls=(ToolCall("read-1", "read_file", '{"path":"README.md"}'),),
                finish_reason="tool_calls",
            ),
            ModelTurn(text="finished", finish_reason="stop"),
        ]

    def complete(self, messages, tools, *, timeout_seconds=None):
        assert tools
        return self.turns.pop(0)


def test_plugin_bridges_runtime_tools_and_writes_redacted_atif(tmp_path: Path) -> None:
    secret = "synthetic-plugin-secret"
    models: list[_ScriptedModel] = []

    def model_factory(**kwargs):
        model = _ScriptedModel(**kwargs)
        models.append(model)
        return model

    async def scenario() -> _Context:
        environment = _Environment()
        context = _Context()
        agent = CourseCodingAgent(
            logs_dir=tmp_path,
            model_name="glm-contract-model",
            provider="custom",
            base_url="http://127.0.0.1:9999/v1",
            key_env="CONTRACT_KEY",
            extra_env={"CONTRACT_KEY": secret},
            model_client_factory=model_factory,
            max_model_turns=3,
        )
        assert secret not in repr(agent._resolve_settings())
        await agent.setup(environment)
        await agent.run("inspect and finish", environment, context)
        assert len(environment.calls) == 1
        # Harbor applies ``agent.extra_env`` as a scoped container overlay;
        # the credential must therefore be absent before that boundary.
        assert "CONTRACT_KEY" not in agent.extra_env
        command, cwd, env, _timeout = environment.calls[0]
        assert command == "cat -- /app/README.md"
        assert cwd is None
        assert env is None
        return context

    context = asyncio.run(scenario())
    assert models and models[0].kwargs["api_key"] == secret
    assert context.metadata["phase"] == "completed"
    assert context.metadata["final_text"] == "finished"
    assert context.metadata["trajectory_path"] == "trajectory.json"

    trajectory = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert secret not in (tmp_path / "trajectory.json").read_text(encoding="utf-8")
    assert secret not in (tmp_path / "run.json").read_text(encoding="utf-8")
    assert secret not in (tmp_path / "events.jsonl").read_text(encoding="utf-8")


def test_plugin_requires_named_key_without_echoing_value(tmp_path: Path) -> None:
    secret = "synthetic-missing-secret"
    agent = CourseCodingAgent(
        logs_dir=tmp_path,
        model_name="model",
        provider="glm",
        extra_env={},
    )
    with pytest.raises(ConfigurationError) as caught:
        agent._resolve_settings()
    message = str(caught.value)
    assert "ZAI_API_KEY" in message
    assert secret not in message


def test_plugin_does_not_pass_extra_env_to_remote_commands(tmp_path: Path) -> None:
    secret = "synthetic-no-container-secret"

    class Model:
        def __init__(self, **kwargs) -> None:
            pass

        def complete(self, messages, tools, *, timeout_seconds=None):
            return ModelTurn(text="ok", finish_reason="stop")

    async def scenario() -> _Environment:
        environment = _Environment()
        context = _Context()
        agent = CourseCodingAgent(
            logs_dir=tmp_path,
            model_name="model",
            provider="custom",
            base_url="http://localhost:9999/v1",
            key_env="CUSTOM_SECRET",
            extra_env={"CUSTOM_SECRET": secret, "NORMAL": "safe"},
            model_client_factory=Model,
        )
        await agent.run("finish", environment, context)
        return environment

    environment = asyncio.run(scenario())
    assert all(secret not in repr(call) for call in environment.calls)


def test_plugin_propagates_async_cancellation_to_runtime(tmp_path: Path) -> None:
    started = threading.Event()
    observed_cancel = threading.Event()

    class BlockingRuntime:
        def __init__(self, *, cancel_check, **kwargs) -> None:
            self._cancel_check = cancel_check

        def run(self, task: str):
            started.set()
            while not self._cancel_check():
                time.sleep(0.002)
            observed_cancel.set()
            return SimpleNamespace(
                phase=RunPhase.CANCELLED,
                reason="cancelled",
                final_text=None,
                model_turns=0,
                model_requests=0,
                tool_calls=0,
                elapsed_seconds=0.01,
                usage=None,
                history=(),
            )

    class Model:
        def __init__(self, **kwargs) -> None:
            pass

    async def scenario() -> None:
        agent = CourseCodingAgent(
            logs_dir=tmp_path,
            model_name="model",
            provider="custom",
            base_url="http://localhost:9999/v1",
            key_env="CUSTOM_SECRET",
            extra_env={"CUSTOM_SECRET": "secret"},
            model_client_factory=Model,
            runtime_factory=BlockingRuntime,
        )
        task = asyncio.create_task(agent.run("wait", _Environment(), _Context()))
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.002)
        assert started.is_set()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    for _ in range(100):
        if observed_cancel.is_set():
            break
        time.sleep(0.002)
    assert observed_cancel.is_set()


def test_plugin_rejects_credential_value_kwargs(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="named environment variable"):
        CourseCodingAgent(logs_dir=tmp_path, api_key="do-not-store")


@pytest.mark.parametrize(
    ("status", "expected_reasoning"),
    [
        (
            "supported",
            {
                "reasoning_effort": "high",
                "reasoning_parameter": "thinking",
            },
        ),
        ("unsupported", {}),
    ],
)
def test_plugin_forwards_or_removes_reasoning_after_probe(
    tmp_path: Path,
    status: str,
    expected_reasoning: dict[str, str],
) -> None:
    models: list[_ScriptedModel] = []

    def factory(**kwargs):
        model = _ScriptedModel(**kwargs)
        models.append(model)
        return model

    async def scenario() -> None:
        agent = CourseCodingAgent(
            logs_dir=tmp_path,
            model_name="model",
            provider="custom",
            base_url="http://localhost:9999/v1",
            key_env="PLUGIN_KEY",
            reasoning_effort="high",
            reasoning_parameter="thinking",
            reasoning_capability_status=status,
            extra_env={"PLUGIN_KEY": "synthetic-plugin-key"},
            model_client_factory=factory,
        )
        await agent.run("finish", _Environment(), _Context())

    asyncio.run(scenario())
    assert models
    for name, value in expected_reasoning.items():
        assert models[0].kwargs[name] == value
    if status == "unsupported":
        assert "reasoning_effort" not in models[0].kwargs
        assert "reasoning_parameter" not in models[0].kwargs


def test_plugin_preserves_structured_reasoning_value_and_capability_metadata(
    tmp_path: Path,
) -> None:
    models: list[_ScriptedModel] = []

    class Model(_ScriptedModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.reasoning_capability = ReasoningCapability(
                status="supported",
                requested_effort="high",
                parameter="thinking",
                accepted_value=False,
            )

    def factory(**kwargs):
        model = Model(**kwargs)
        models.append(model)
        return model

    async def scenario() -> _Context:
        context = _Context()
        agent = CourseCodingAgent(
            logs_dir=tmp_path,
            model_name="model",
            provider="custom",
            base_url="http://localhost:9999/v1",
            key_env="PLUGIN_KEY",
            reasoning_parameter="thinking",
            reasoning_value='{"budget":3}',
            reasoning_capability_status="supported",
            extra_env={"PLUGIN_KEY": "synthetic-plugin-key"},
            model_client_factory=factory,
        )
        await agent.run("finish", _Environment(), context)
        return context

    context = asyncio.run(scenario())
    assert models[0].kwargs["reasoning_parameter"] == "thinking"
    assert models[0].kwargs["reasoning_value"] == {"budget": 3}
    assert context.metadata["reasoning_capability_status"] == "supported"
    assert context.metadata["reasoning_capability"]["accepted_value"] is False


def test_plugin_rejects_reasoning_probe_error_during_setup(tmp_path: Path) -> None:
    agent = CourseCodingAgent(
        logs_dir=tmp_path,
        model_name="model",
        provider="custom",
        base_url="http://localhost:9999/v1",
        key_env="PLUGIN_KEY",
        reasoning_effort="high",
        reasoning_capability_status="error",
        extra_env={"PLUGIN_KEY": "synthetic-plugin-key"},
    )

    with pytest.raises(ConfigurationError, match="probe failed"):
        asyncio.run(agent.setup(_Environment()))


def test_plugin_probes_reasoning_when_runner_did_not_supply_status(
    tmp_path: Path,
) -> None:
    secret = "synthetic-direct-probe-key"
    factory_calls: list[dict[str, object]] = []
    probe_calls: list[tuple[str, float]] = []

    class Model(_ScriptedModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            factory_calls.append(dict(kwargs))
            self.reasoning_capability = None

        def probe_reasoning_effort(self, effort, *, timeout_seconds):
            probe_calls.append((effort, timeout_seconds))
            return ReasoningCapability(
                status="supported",
                requested_effort=effort,
                parameter="thinking",
                accepted_value="high",
            )

        def configure_reasoning(self, capability):
            self.reasoning_capability = capability

    async def scenario() -> _Context:
        context = _Context()
        agent = CourseCodingAgent(
            logs_dir=tmp_path,
            model_name="model",
            provider="custom",
            base_url="http://localhost:9999/v1",
            key_env="DIRECT_PROBE_KEY",
            reasoning_effort="high",
            extra_env={"DIRECT_PROBE_KEY": secret},
            model_client_factory=Model,
        )
        environment = _Environment()
        await agent.setup(environment)
        await agent.run("finish", environment, context)
        return context

    context = asyncio.run(scenario())
    assert len(factory_calls) == 1
    assert probe_calls and probe_calls[0][0] == "high"
    assert probe_calls[0][1] <= 20.0
    assert context.metadata["reasoning_capability_status"] == "supported"
    assert context.metadata["reasoning_capability"]["parameter"] == "thinking"
    assert secret not in repr(context.metadata)


def test_plugin_unsupported_probe_rebuilds_without_reasoning_fields(
    tmp_path: Path,
) -> None:
    factory_calls: list[dict[str, object]] = []

    class Model(_ScriptedModel):
        def __init__(self, **kwargs):
            factory_calls.append(dict(kwargs))
            super().__init__(**kwargs)

        def probe_reasoning_effort(self, effort, *, timeout_seconds):
            return ReasoningCapability(
                status="unsupported",
                requested_effort=effort,
                parameter="thinking",
            )

    async def scenario() -> None:
        agent = CourseCodingAgent(
            logs_dir=tmp_path,
            model_name="model",
            provider="custom",
            base_url="http://localhost:9999/v1",
            key_env="PLUGIN_KEY",
            reasoning_effort="high",
            extra_env={"PLUGIN_KEY": "synthetic-plugin-key"},
            model_client_factory=Model,
        )
        environment = _Environment()
        await agent.setup(environment)
        await agent.run("finish", environment, _Context())

    asyncio.run(scenario())
    # One instance is used for the probe and a fresh instance is used for the
    # actual run. Neither factory call may carry a field the probe rejected.
    assert len(factory_calls) == 2
    assert all(
        "reasoning_effort" not in kwargs
        and "reasoning_parameter" not in kwargs
        and "reasoning_value" not in kwargs
        for kwargs in factory_calls
    )


@pytest.mark.parametrize("native_value", [False, 7, {"budget": 3}])
def test_plugin_forwards_nonstandard_native_reasoning_values(
    tmp_path: Path,
    native_value,
) -> None:
    models: list[_ScriptedModel] = []

    class Model(_ScriptedModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            models.append(self)
            self.reasoning_capability = None

        def probe_reasoning_effort(self, effort, *, timeout_seconds):
            return ReasoningCapability(
                status="supported",
                requested_effort=effort,
                parameter="thinking",
                accepted_value=native_value,
            )

        def configure_reasoning(self, capability):
            self.reasoning_capability = capability

    async def scenario() -> _Context:
        context = _Context()
        agent = CourseCodingAgent(
            logs_dir=tmp_path,
            model_name="model",
            provider="custom",
            base_url="http://localhost:9999/v1",
            key_env="PLUGIN_KEY",
            reasoning_effort="high",
            extra_env={"PLUGIN_KEY": "synthetic-plugin-key"},
            model_client_factory=Model,
        )
        await agent.run("finish", _Environment(), context)
        return context

    context = asyncio.run(scenario())
    assert len(models) == 1
    assert models[0].reasoning_capability.accepted_value == native_value
    assert context.metadata["reasoning_capability"]["accepted_value"] == native_value


@pytest.mark.parametrize(
    "capability",
    [
        ReasoningCapability(
            status="supported",
            requested_effort="high",
            parameter="bad-name!",
            accepted_value="high",
        ),
        ReasoningCapability(
            status="supported",
            requested_effort="high",
            parameter="thinking",
            accepted_value=float("nan"),
        ),
    ],
)
def test_plugin_rejects_invalid_native_reasoning_capability(
    tmp_path: Path,
    capability: ReasoningCapability,
) -> None:
    class Model(_ScriptedModel):
        def probe_reasoning_effort(self, effort, *, timeout_seconds):
            return capability

    agent = CourseCodingAgent(
        logs_dir=tmp_path,
        model_name="model",
        provider="custom",
        base_url="http://localhost:9999/v1",
        key_env="PLUGIN_KEY",
        reasoning_effort="high",
        extra_env={"PLUGIN_KEY": "synthetic-plugin-key"},
        model_client_factory=Model,
    )
    with pytest.raises(ConfigurationError, match="probe failed|native reasoning"):
        asyncio.run(agent.setup(_Environment()))
