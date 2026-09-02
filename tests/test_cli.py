"""Assembly-level CLI tests that do not contact a model provider."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_agent.cli as cli_module
from coding_agent.agent import RunResult
from coding_agent.cli import (
    EXIT_COMPLETED,
    EXIT_VERIFICATION_FAILED,
    _build_runtime,
    _exit_code_for_result,
    _run_verification,
    build_parser,
)
from coding_agent.config import AgentConfig, ConfigurationError
from coding_agent.events import NullEventSink
from coding_agent.model import ReasoningCapability
from coding_agent.types import RunPhase

SYNTHETIC_CREDENTIAL = "synthetic-cli-credential-for-tests-only"


def test_cli_excludes_the_selected_api_key_variable_from_run_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove the config-to-registry wiring with an arbitrary key variable name.

    Constructing the ordinary model client performs no network request.  The
    test invokes only the assembled local shell tool and checks variable
    presence without echoing the synthetic credential into captured output.
    """

    key_env = "MODEL_GATEWAY_CREDENTIAL"
    monkeypatch.setenv(key_env, SYNTHETIC_CREDENTIAL)
    config = AgentConfig.from_sources(
        task="inspect the workspace",
        workspace=tmp_path,
        provider="custom",
        model="offline-test-model",
        base_url="https://gateway.example/v1",
        key_env=key_env,
        environ=os.environ,
    )

    runtime = _build_runtime(config, NullEventSink())
    result = runtime.tool_registry.execute(
        "cli_env_test",
        "run_command",
        {
            "command": 'test -z "${MODEL_GATEWAY_CREDENTIAL+x}"',
            "timeout_seconds": 5,
        },
    )

    assert result.ok
    assert result.metadata["exit_code"] == 0
    assert SYNTHETIC_CREDENTIAL not in result.content


def test_planning_flag_is_opt_in_and_adds_only_update_plan(tmp_path: Path) -> None:
    parser = build_parser()
    assert parser.parse_args(["task"]).planning is False
    assert parser.parse_args(["--planning", "task"]).planning is True

    config = AgentConfig.from_sources(
        task="inspect the workspace",
        workspace=tmp_path,
        provider="custom",
        model="offline-test-model",
        base_url="https://gateway.example/v1",
        key_env="MODEL_GATEWAY_CREDENTIAL",
        environ={"MODEL_GATEWAY_CREDENTIAL": SYNTHETIC_CREDENTIAL},
    )
    ordinary = _build_runtime(config, NullEventSink())
    planned = _build_runtime(config, NullEventSink(), planning=True)

    assert "update_plan" not in ordinary.tool_registry.names
    assert planned.tool_registry.names[-1] == "update_plan"
    assert len(planned.tool_registry.names) == len(ordinary.tool_registry.names) + 1


def test_tui_flag_is_opt_in() -> None:
    parser = build_parser()
    assert parser.parse_args(["task"]).tui is False
    assert parser.parse_args(["--tui", "task"]).tui is True


def test_verification_commands_are_repeatable_and_alias_is_supported() -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "task",
            "--verify",
            "python -m pytest -q",
            "--verify-command",
            "ruff check .",
        ]
    )
    assert arguments.verification_commands == [
        "python -m pytest -q",
        "ruff check .",
    ]
    assert arguments.verification_timeout_seconds == 120.0


def test_cli_verification_runs_after_runtime_and_has_independent_status(
    tmp_path: Path,
) -> None:
    config = AgentConfig.from_sources(
        task="inspect the workspace",
        workspace=tmp_path,
        provider="custom",
        model="offline-test-model",
        base_url="https://gateway.example/v1",
        key_env="MODEL_GATEWAY_CREDENTIAL",
        environ={"MODEL_GATEWAY_CREDENTIAL": SYNTHETIC_CREDENTIAL},
    )
    result = _run_verification(
        config,
        [f'{os.sys.executable} -c "raise SystemExit(7)"'],
        timeout_seconds=5,
    )
    assert result.passed is False
    assert result.checks[0].exit_code == 7

    runtime_result = RunResult(
        phase=RunPhase.COMPLETED,
        reason="model returned a final response",
        final_text="done",
        model_turns=1,
        model_requests=1,
        tool_calls=0,
        elapsed_seconds=0.01,
        usage=None,
        history=(),
    )
    assert _exit_code_for_result(runtime_result) == EXIT_COMPLETED
    assert (
        _exit_code_for_result(runtime_result, verification=result)
        == EXIT_VERIFICATION_FAILED
    )


def test_main_runs_optional_verifier_once_after_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    class Runtime:
        def run(self, task: str) -> RunResult:
            calls.append(task)
            return RunResult(
                phase=RunPhase.COMPLETED,
                reason="model returned a final response",
                final_text="finished",
                model_turns=1,
                model_requests=1,
                tool_calls=0,
                elapsed_seconds=0.01,
                usage=None,
                history=(),
            )

    monkeypatch.setattr(
        cli_module, "_build_runtime", lambda config, sink, **kwargs: Runtime()
    )
    code = cli_module.main(
        [
            "--provider",
            "custom",
            "--model",
            "offline",
            "--base-url",
            "https://gateway.example/v1",
            "--key-env",
            "MODEL_GATEWAY_CREDENTIAL",
            "--verify",
            "true",
            "task",
        ],
        environ={"MODEL_GATEWAY_CREDENTIAL": SYNTHETIC_CREDENTIAL},
    )

    assert code == EXIT_COMPLETED
    assert calls == ["task"]
    captured = capsys.readouterr()
    assert "verification summary: passed=True" in captured.err
    assert "finished" in captured.out


def test_main_finishes_optional_tui_with_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Tui:
        def __init__(self, **kwargs: object) -> None:
            observed["constructor"] = kwargs

        def emit(self, event_type: str, **data: object) -> None:
            pass

        def verification_started(self, check_count: int) -> None:
            observed["verification_started"] = check_count

        def finish(self, result: object, verification: object | None) -> None:
            observed["result"] = result
            observed["verification"] = verification

        def abort(self, message: str, *, cancelled: bool = False) -> None:
            observed["abort"] = (message, cancelled)

    class Runtime:
        def run(self, task: str) -> RunResult:
            return RunResult(
                phase=RunPhase.COMPLETED,
                reason="model returned a final response",
                final_text="finished",
                model_turns=1,
                model_requests=1,
                tool_calls=0,
                elapsed_seconds=0.01,
                usage=None,
                history=(),
            )

    monkeypatch.setattr(cli_module, "RichEventSink", Tui)
    monkeypatch.setattr(
        cli_module, "_build_runtime", lambda config, sink, **kwargs: Runtime()
    )
    code = cli_module.main(
        [
            "--provider",
            "custom",
            "--model",
            "offline",
            "--base-url",
            "https://gateway.example/v1",
            "--key-env",
            "MODEL_GATEWAY_CREDENTIAL",
            "--tui",
            "--verify",
            "true",
            "task",
        ],
        environ={"MODEL_GATEWAY_CREDENTIAL": SYNTHETIC_CREDENTIAL},
    )

    assert code == EXIT_COMPLETED
    assert observed["verification_started"] == 1
    assert observed["verification"].passed is True
    constructor = observed["constructor"]
    assert isinstance(constructor, dict)
    assert constructor["task"] == "task"
    assert constructor["secrets"] == (SYNTHETIC_CREDENTIAL,)


def test_interactive_tui_preserves_history_until_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = iter(["first task", "what changed?", "/exit"])
    calls: list[tuple[str, object | None]] = []
    prior_history = ("complete-first-run-history",)

    class Tui:
        def __init__(self, **kwargs: object) -> None:
            pass

        def emit(self, event_type: str, **data: object) -> None:
            pass

        def finish(self, result: object, verification: object | None) -> None:
            pass

        def abort(self, message: str, *, cancelled: bool = False) -> None:
            pass

    class Runtime:
        def run(self, task: str, *, history: object | None = None) -> RunResult:
            calls.append((task, history))
            run_history = prior_history if history is None else (*prior_history, "next")
            return RunResult(
                phase=RunPhase.COMPLETED,
                reason="model returned a final response",
                final_text="finished",
                model_turns=1,
                model_requests=1,
                tool_calls=0,
                elapsed_seconds=0.01,
                usage=None,
                history=run_history,  # type: ignore[arg-type]
            )

    monkeypatch.setattr(
        cli_module,
        "prompt_for_task",
        lambda **kwargs: next(prompts),
    )
    monkeypatch.setattr(cli_module, "RichEventSink", Tui)
    monkeypatch.setattr(
        cli_module, "_build_runtime", lambda config, sink, **kwargs: Runtime()
    )

    code = cli_module.main(
        [
            "--provider",
            "custom",
            "--model",
            "offline",
            "--base-url",
            "https://gateway.example/v1",
            "--key-env",
            "MODEL_GATEWAY_CREDENTIAL",
            "--tui",
        ],
        environ={"MODEL_GATEWAY_CREDENTIAL": SYNTHETIC_CREDENTIAL},
    )

    assert code == EXIT_COMPLETED
    assert calls == [
        ("first task", None),
        ("what changed?", prior_history),
    ]


def test_cli_verifier_excludes_custom_api_key_environment_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_env = "MODEL_GATEWAY_CREDENTIAL"
    monkeypatch.setenv(key_env, SYNTHETIC_CREDENTIAL)
    config = AgentConfig.from_sources(
        task="inspect the workspace",
        workspace=tmp_path,
        provider="custom",
        model="offline-test-model",
        base_url="https://gateway.example/v1",
        key_env=key_env,
        environ=os.environ,
    )
    result = _run_verification(
        config,
        [
            f"{os.sys.executable} -c \"import os; print(os.getenv('{key_env}', 'missing'))\""
        ],
        timeout_seconds=5,
    )
    assert result.passed is True
    assert SYNTHETIC_CREDENTIAL not in result.checks[0].stdout
    assert "missing" in result.checks[0].stdout


def test_cli_final_output_redacts_the_configured_api_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = SimpleNamespace(
        final_text=f"The provider returned {SYNTHETIC_CREDENTIAL}.",
        reason=f"diagnostic: {SYNTHETIC_CREDENTIAL}",
        phase=RunPhase.COMPLETED,
        model_turns=1,
        model_requests=1,
        tool_calls=0,
        elapsed_seconds=0.01,
    )

    cli_module._print_result(result, secrets=(SYNTHETIC_CREDENTIAL,))

    captured = capsys.readouterr()
    assert SYNTHETIC_CREDENTIAL not in captured.out
    assert SYNTHETIC_CREDENTIAL not in captured.err
    assert captured.out.count("[REDACTED]") == 1


class _RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event_type: str, **data: object) -> None:
        self.events.append((event_type, dict(data)))


def _reasoning_config(
    tmp_path: Path, *, parameter: str = "reasoning_effort"
) -> AgentConfig:
    return AgentConfig.from_sources(
        task="inspect the workspace",
        workspace=tmp_path,
        provider="custom",
        model="offline-test-model",
        base_url="https://gateway.example/v1",
        key_env="MODEL_GATEWAY_CREDENTIAL",
        reasoning_effort="high",
        reasoning_parameter=parameter,
        environ={"MODEL_GATEWAY_CREDENTIAL": SYNTHETIC_CREDENTIAL},
    )


def test_cli_probes_custom_reasoning_field_and_applies_confirmed_value(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Model:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.reasoning_capability = None

        def probe_reasoning_effort(
            self,
            effort: str,
            *,
            parameter_candidates: tuple[str, ...],
            timeout_seconds: float,
        ) -> ReasoningCapability:
            calls.append(
                {
                    "effort": effort,
                    "parameter_candidates": parameter_candidates,
                    "timeout_seconds": timeout_seconds,
                }
            )
            return ReasoningCapability(
                status="supported",
                requested_effort=effort,
                parameter="thinking",
                accepted_value="high",
            )

        def configure_reasoning(self, capability: ReasoningCapability) -> None:
            self.reasoning_capability = capability

    sink = _RecordingEventSink()
    runtime = _build_runtime(
        _reasoning_config(tmp_path, parameter="thinking"),
        sink,
        model_client_factory=Model,
        reasoning_probe_timeout_seconds=7,
    )

    # The probe client is constructed without an unverified reasoning field;
    # the confirmed capability is applied only after the probe succeeds.
    assert "reasoning_effort" not in runtime.model_client.kwargs
    assert "reasoning_parameter" not in runtime.model_client.kwargs
    assert runtime.model_client.reasoning_capability.status == "supported"
    assert calls == [
        {
            "effort": "high",
            "parameter_candidates": ("thinking",),
            "timeout_seconds": 7.0,
        }
    ]
    assert sink.events[0][0] == "reasoning.probe"
    assert sink.events[0][1]["parameter"] == "thinking"
    assert sink.events[0][1]["accepted_value"] == "high"


def test_cli_rejects_unsupported_reasoning_before_building_runtime(
    tmp_path: Path,
) -> None:
    sink = _RecordingEventSink()

    class Model:
        def __init__(self, **kwargs: object) -> None:
            pass

        def probe_reasoning_effort(self, effort: str, **kwargs: object):
            return ReasoningCapability(
                status="unsupported",
                requested_effort=effort,
                parameter="reasoning_effort",
                detail=f"gateway rejected {SYNTHETIC_CREDENTIAL}",
            )

    with pytest.raises(ConfigurationError, match="unsupported"):
        _build_runtime(
            _reasoning_config(tmp_path),
            sink,
            model_client_factory=Model,
        )

    assert sink.events[0][1]["status"] == "unsupported"
    assert SYNTHETIC_CREDENTIAL not in repr(sink.events)


def test_cli_probe_errors_fail_closed_without_leaking_provider_detail(
    tmp_path: Path,
) -> None:
    sink = _RecordingEventSink()

    class Model:
        def __init__(self, **kwargs: object) -> None:
            pass

        def probe_reasoning_effort(self, effort: str, **kwargs: object):
            raise RuntimeError(f"gateway response included {SYNTHETIC_CREDENTIAL}")

    with pytest.raises(ConfigurationError, match="probe failed"):
        _build_runtime(
            _reasoning_config(tmp_path),
            sink,
            model_client_factory=Model,
        )

    assert sink.events[0][1]["status"] == "error"
    assert SYNTHETIC_CREDENTIAL not in repr(sink.events)
