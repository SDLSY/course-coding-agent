"""Configuration tests use synthetic credentials and never contact an API."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.config import (
    AgentConfig,
    ConfigurationError,
    Provider,
)

FAKE_KEY = "unit-test-secret-not-a-real-key"


def deepseek_environment(**overrides: str) -> dict[str, str]:
    environment = {
        "CODING_AGENT_PROVIDER": "deepseek",
        "CODING_AGENT_MODEL": "test-model",
        "DEEPSEEK_API_KEY": FAKE_KEY,
    }
    environment.update(overrides)
    return environment


def test_deepseek_preset_and_explicit_model(tmp_path: Path) -> None:
    config = AgentConfig.from_sources(
        task="fix the failing test",
        workspace=tmp_path,
        environ=deepseek_environment(),
    )

    assert config.provider is Provider.DEEPSEEK
    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "test-model"
    assert config.api_key == FAKE_KEY


def test_glm_preset_uses_official_openai_compatible_root(tmp_path: Path) -> None:
    config = AgentConfig.from_sources(
        task="inspect the project",
        workspace=tmp_path,
        provider="GLM",
        model="glm-test-model",
        environ={"ZAI_API_KEY": FAKE_KEY},
    )

    assert config.provider is Provider.GLM
    assert config.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert config.api_key_env == "ZAI_API_KEY"
    assert config.api_key == FAKE_KEY


def test_cli_values_override_environment_without_overriding_model_implicitly(
    tmp_path: Path,
) -> None:
    config = AgentConfig.from_sources(
        task="explicit task",
        workspace=tmp_path,
        provider="deepseek",
        model="explicit-model",
        base_url="https://gateway.example/v1/",
        max_model_turns=7,
        environ=deepseek_environment(
            CODING_AGENT_MODEL="environment-model",
            CODING_AGENT_BASE_URL="https://ignored.example/v1",
            CODING_AGENT_MAX_MODEL_TURNS="99",
        ),
    )

    assert config.model == "explicit-model"
    assert config.base_url == "https://gateway.example/v1"
    assert config.max_model_turns == 7


def test_model_is_required_even_when_provider_has_a_url_preset(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="model is required"):
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path,
            provider="deepseek",
            environ={"DEEPSEEK_API_KEY": FAKE_KEY},
        )


def test_custom_provider_requires_explicit_base_url(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="custom provider requires"):
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path,
            provider="custom",
            model="custom-model",
            environ={"CODING_AGENT_API_KEY": FAKE_KEY},
        )


def test_key_env_is_an_indirection_not_a_raw_key(tmp_path: Path) -> None:
    config = AgentConfig.from_sources(
        task="task",
        workspace=tmp_path,
        provider="custom",
        model="custom-model",
        base_url="https://gateway.example/v1",
        key_env="BENCHMARK_PROVIDER_KEY",
        environ={"BENCHMARK_PROVIDER_KEY": FAKE_KEY},
    )

    assert config.api_key_env == "BENCHMARK_PROVIDER_KEY"
    assert config.api_key == FAKE_KEY

    with pytest.raises(ConfigurationError, match="environment-variable name"):
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path,
            provider="custom",
            model="custom-model",
            base_url="https://gateway.example/v1",
            key_env="not a valid variable name",
            environ={},
        )


def test_missing_key_error_and_repr_never_include_secret(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as error:
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path,
            provider="deepseek",
            model="test-model",
            environ={},
        )
    assert "DEEPSEEK_API_KEY" in str(error.value)

    config = AgentConfig.from_sources(
        task="task",
        workspace=tmp_path,
        environ=deepseek_environment(),
    )
    assert FAKE_KEY not in repr(config)
    assert FAKE_KEY not in str(config.redacted_summary())
    assert config.redacted_summary()["api_key_env"] == "DEEPSEEK_API_KEY"


def test_non_string_key_environment_value_is_a_configuration_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path,
            provider="deepseek",
            model="test-model",
            environ={"DEEPSEEK_API_KEY": 123},  # type: ignore[dict-item]
        )


def test_dotenv_file_is_not_loaded_implicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text(f"DEEPSEEK_API_KEY={FAKE_KEY}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ConfigurationError, match="DEEPSEEK_API_KEY"):
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path,
            provider="deepseek",
            model="test-model",
            environ={},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_model_turns", 0, "greater than zero"),
        ("max_tool_calls", "not-an-int", "must be an integer"),
        ("max_wall_time_seconds", "nan", "greater than zero"),
        ("context_char_budget", -1, "greater than zero"),
        ("model_timeout_seconds", float("inf"), "must be finite"),
        ("model_max_retries", -1, "zero or greater"),
        ("protocol_max_retries", True, "must be an integer"),
    ],
)
def test_invalid_budgets_are_rejected_before_runtime(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path,
            environ=deepseek_environment(),
            **{field: value},
        )


def test_environment_budget_values_are_parsed(tmp_path: Path) -> None:
    config = AgentConfig.from_sources(
        task="task",
        workspace=tmp_path,
        environ=deepseek_environment(
            CODING_AGENT_MAX_MODEL_TURNS="12",
            CODING_AGENT_MAX_TOOL_CALLS="34",
            CODING_AGENT_MAX_WALL_TIME="56.5",
            CODING_AGENT_CONTEXT_CHAR_BUDGET="7890",
            CODING_AGENT_MODEL_TIMEOUT="45",
            CODING_AGENT_MODEL_MAX_RETRIES="3",
            CODING_AGENT_PROTOCOL_MAX_RETRIES="0",
        ),
    )

    assert config.max_model_turns == 12
    assert config.max_tool_calls == 34
    assert config.max_wall_time_seconds == 56.5
    assert config.context_char_budget == 7890
    assert config.model_timeout_seconds == 45.0
    assert config.model_max_retries == 3
    assert config.protocol_max_retries == 0


def test_workspace_must_exist_and_trace_path_may_be_new(tmp_path: Path) -> None:
    config = AgentConfig.from_sources(
        task="task",
        workspace=".",
        trace_path=".agent-runs/run.jsonl",
        environ=deepseek_environment(),
        cwd=tmp_path,
    )

    assert config.workspace == tmp_path.resolve()
    assert config.trace_path == (tmp_path / ".agent-runs/run.jsonl").resolve()

    with pytest.raises(ConfigurationError, match="does not exist"):
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path / "missing",
            environ=deepseek_environment(),
        )


@pytest.mark.parametrize(
    "base_url",
    [
        "not-a-url",
        "ftp://example.test/v1",
        "http://remote.example.test/v1",
        "https://user:password@example.test/v1",
        "https://example.test/v1?api_key=secret",
        "https://example.test/v1#fragment",
        "https://example.test:not-a-port/v1",
        "https://[invalid-ipv6/v1",
    ],
)
def test_unsafe_or_malformed_base_urls_are_rejected(
    tmp_path: Path, base_url: str
) -> None:
    with pytest.raises(ConfigurationError):
        AgentConfig.from_sources(
            task="task",
            workspace=tmp_path,
            provider="custom",
            model="model",
            base_url=base_url,
            environ={"CODING_AGENT_API_KEY": FAKE_KEY},
        )


def test_plain_http_is_available_only_for_a_loopback_test_server(
    tmp_path: Path,
) -> None:
    config = AgentConfig.from_sources(
        task="task",
        workspace=tmp_path,
        provider="custom",
        model="model",
        base_url="http://127.0.0.1:8080/v1/",
        environ={"CODING_AGENT_API_KEY": FAKE_KEY},
    )

    assert config.base_url == "http://127.0.0.1:8080/v1"
