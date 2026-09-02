"""Tests for the offline, version-pinned Harbor OpenCode adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.pin_opencode_images import pinned_dockerfile
from coding_agent.opencode_pinned import (
    PINNED_OPENCODE_VERSION,
    PinnedOpenCodeAgent,
    parse_opencode_version,
    validate_pinned_version,
)


class _VersionEnvironment:
    def __init__(self, *, stdout: str = "", return_code: int = 0) -> None:
        self.stdout = stdout
        self.return_code = return_code
        self.calls: list[dict[str, Any]] = []

    async def exec(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(dict(kwargs))
        return SimpleNamespace(
            stdout=self.stdout,
            stderr="",
            return_code=self.return_code,
        )


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("1.18.25\n", "1.18.25"),
        ("opencode 1.18.25\n", "1.18.25"),
        ("v1.18.25\n", "1.18.25"),
        ("", None),
        ("opencode 1.18\n", None),
        ("opencode 1.18.25.1\n", None),
    ],
)
def test_parse_opencode_version(output: str, expected: str | None) -> None:
    assert parse_opencode_version(output) == expected


def test_validate_pinned_version_never_uses_failed_output() -> None:
    assert validate_pinned_version("opencode 1.18.25", return_code=1) == (
        False,
        "opencode --version returned a non-zero status",
    )
    assert validate_pinned_version("opencode 1.17.0") == (
        False,
        f"opencode version mismatch (expected {PINNED_OPENCODE_VERSION})",
    )
    assert validate_pinned_version("opencode 1.18.25") == (
        True,
        PINNED_OPENCODE_VERSION,
    )


def test_setup_accepts_exact_version_and_runs_one_local_check(tmp_path) -> None:
    agent = PinnedOpenCodeAgent(
        logs_dir=tmp_path,
        model_name="openai/fixture",
    )
    environment = _VersionEnvironment(stdout="opencode 1.18.25\n")

    asyncio.run(agent.setup(environment))

    assert [call["command"] for call in environment.calls] == ["opencode --version"]
    assert agent.version() == PINNED_OPENCODE_VERSION


@pytest.mark.parametrize(
    ("stdout", "return_code", "needle"),
    [
        ("", 127, "non-zero status"),
        ("opencode 1.17.0", 0, "version mismatch"),
        ("not a version", 0, "missing or malformed"),
    ],
)
def test_setup_fails_closed_for_missing_or_mismatched_binary(
    tmp_path,
    stdout: str,
    return_code: int,
    needle: str,
) -> None:
    agent = PinnedOpenCodeAgent(
        logs_dir=tmp_path,
        model_name="openai/fixture",
    )
    environment = _VersionEnvironment(stdout=stdout, return_code=return_code)

    with pytest.raises(RuntimeError, match=needle):
        asyncio.run(agent.setup(environment))


def test_install_is_a_complete_no_op(tmp_path) -> None:
    agent = PinnedOpenCodeAgent(logs_dir=tmp_path, model_name="openai/fixture")
    environment = _VersionEnvironment()

    asyncio.run(agent.install(environment))

    assert environment.calls == []


class _CommandCapturingAgent(PinnedOpenCodeAgent):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.commands: list[str] = []
        self._resume = False

    @property
    def model_connection(self) -> SimpleNamespace:
        return SimpleNamespace(env={})

    def _build_register_skills_command(self) -> None:
        return None

    def _build_register_config_command(self) -> None:
        return None

    def build_cli_flags(self) -> str:
        return ""

    async def exec_as_agent(self, environment, command: str, **kwargs: Any):
        self.commands.append(command)
        return SimpleNamespace(stdout="", stderr="", return_code=0)

    def _error_messages(self) -> list[str]:
        return []


def test_run_command_has_no_runtime_install_or_nvm_steps(tmp_path) -> None:
    agent = _CommandCapturingAgent(
        logs_dir=tmp_path,
        model_name="openai/fixture",
    )
    agent._version = PINNED_OPENCODE_VERSION

    asyncio.run(agent.run("finish the task", _VersionEnvironment(), None))

    assert len(agent.commands) == 1
    command = agent.commands[0]
    assert "opencode --model=openai/fixture run" in command
    assert "--thinking" in command
    lowered = command.lower()
    assert "nvm" not in lowered
    assert "npm" not in lowered
    assert "@latest" not in lowered
    assert "install" not in lowered


def test_toolchain_dockerfile_copies_only_reference_image_paths() -> None:
    dockerfile = pinned_dockerfile(
        "example/task@sha256:" + "a" * 64,
        toolchain_image="harbor-opencode-ready@sha256:" + "b" * 64,
    )

    assert "/usr/local/bin/node" in dockerfile
    assert "/usr/local/bin/npm" in dockerfile
    assert "/usr/local/bin/opencode" in dockerfile
    assert "/usr/local/bin/npx" not in dockerfile
    assert "/usr/local/lib/node_modules/opencode-ai" not in dockerfile


def test_atif_parser_keeps_reasoning_and_token_cache_metrics() -> None:
    pytest.importorskip("harbor")
    agent = PinnedOpenCodeAgent(
        logs_dir="/tmp/opencode-pinned-test",
        model_name="openai/fixture",
    )
    agent._version = PINNED_OPENCODE_VERSION
    agent._instruction = "inspect the project"
    events = [
        {"type": "step_start", "sessionID": "session-1", "timestamp": 1000},
        {
            "type": "reasoning",
            "part": {"type": "reasoning", "text": "check the failing test"},
        },
        {
            "type": "tool_use",
            "part": {
                "type": "tool",
                "tool": "shell",
                "callID": "call-1",
                "state": {
                    "input": {"command": "pytest -q"},
                    "output": "1 passed",
                },
            },
        },
        {
            "type": "text",
            "part": {"type": "text", "text": "done"},
        },
        {
            "type": "step_finish",
            "part": {
                "tokens": {
                    "input": 10,
                    "output": 4,
                    "reasoning": 2,
                    "cache": {"read": 3, "write": 1},
                },
                "cost": 0.25,
            },
        },
    ]

    trajectory = agent._convert_events_to_trajectory(events)

    assert trajectory is not None
    assert trajectory.session_id == "session-1"
    assert trajectory.steps[0].source == "user"
    assert trajectory.steps[1].reasoning_content == "check the failing test"
    assert trajectory.steps[1].tool_calls[0].function_name == "shell"
    assert trajectory.steps[1].observation.results[0].content == "1 passed"
    assert trajectory.steps[1].metrics.prompt_tokens == 13
    assert trajectory.steps[1].metrics.completion_tokens == 4
    assert trajectory.steps[1].metrics.cached_tokens == 3
    assert trajectory.steps[1].metrics.cost_usd == 0.25
    assert trajectory.final_metrics.total_prompt_tokens == 13
    assert trajectory.final_metrics.total_completion_tokens == 4
    assert trajectory.final_metrics.total_cached_tokens == 3
    assert trajectory.final_metrics.total_cost_usd == 0.25
