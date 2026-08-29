"""Assembly-level CLI tests that do not contact a model provider."""

from __future__ import annotations

import os
from pathlib import Path

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
from coding_agent.config import AgentConfig
from coding_agent.events import NullEventSink
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
