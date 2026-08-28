"""Assembly-level CLI tests that do not contact a model provider."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from coding_agent.cli import _build_runtime
from coding_agent.config import AgentConfig
from coding_agent.events import NullEventSink

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
