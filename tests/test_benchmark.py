"""Deterministic tests for the bounded benchmark harness.

The suite never contacts a model provider.  The injected invoker tests the
runner's fixture/verifier protocol, while one tiny local subprocess checks the
CommandInvoker timeout and placeholder boundary.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from coding_agent.benchmark import (
    BENCHMARK_SCHEMA_VERSION,
    AgentExecution,
    BenchmarkAgent,
    BenchmarkBudget,
    BenchmarkError,
    BenchmarkManifest,
    BenchmarkRunner,
    BenchmarkTask,
    CommandInvoker,
    load_manifest,
    main,
)


def _manifest_dict(fixture: Path, *, repetitions: int = 1) -> dict[str, object]:
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "name": "test-suite",
        "model": {"id": "offline-model", "temperature": 0},
        "repetitions": repetitions,
        "budget": {
            "max_wall_time_seconds": 2,
            "max_model_requests": 3,
            "max_tool_calls": 4,
            "max_total_tokens": 100,
            "output_char_limit": 80,
        },
        "tasks": [
            {
                "id": "touch-marker",
                "prompt": "Create marker.txt.",
                "fixture": str(fixture),
                "checks": ["test -f marker.txt"],
            }
        ],
        "agents": [
            {"id": "fake", "command": ["unused"]},
        ],
    }


def test_manifest_resolves_fixture_relative_to_manifest(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(
        json.dumps(_manifest_dict(fixture)),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert manifest.tasks[0].fixture == fixture.resolve()
    assert manifest.repetitions == 1
    assert manifest.budget.max_total_tokens == 100
    assert manifest.agents[0].command == ("unused",)
    assert manifest.model["id"] == "offline-model"


def test_direct_task_construction_treats_a_single_check_string_as_one_command(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()

    task = BenchmarkTask(
        task_id="single-check",
        prompt="p",
        fixture=fixture,
        checks="test -f marker.txt",
    )

    assert len(task.checks) == 1
    assert task.checks[0].command == "test -f marker.txt"


def test_manifest_rejects_unknown_schema_version(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    value = _manifest_dict(fixture)
    value["schema_version"] = "coding-agent-benchmark/v99"

    with pytest.raises(BenchmarkError, match="unsupported benchmark schema"):
        BenchmarkManifest.from_dict(value, base_dir=tmp_path)


def test_manifest_serialization_never_includes_agent_environment_values(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    value = _manifest_dict(fixture)
    value["agents"] = [
        {
            "id": "fake",
            "command": ["unused", "secret-value"],
            "environment": {"CUSTOM_SETTING": "secret-value"},
        }
    ]

    manifest = BenchmarkManifest.from_dict(value, base_dir=tmp_path)

    rendered = json.dumps(manifest.to_dict())
    assert "secret-value" not in rendered
    assert "CUSTOM_SETTING" in rendered
    assert "[REDACTED]" in rendered
    assert manifest.agents[0].environment["CUSTOM_SETTING"] == "secret-value"


def test_manifest_allows_explicit_host_environment_names_without_serializing_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    monkeypatch.setenv("SYNTHETIC_PROVIDER_CREDENTIAL", "host-only-secret")
    value = _manifest_dict(fixture)
    value["agents"] = [
        {
            "id": "fake",
            "command": ["unused"],
            "environment_from_host": ["SYNTHETIC_PROVIDER_CREDENTIAL"],
        }
    ]

    manifest = BenchmarkManifest.from_dict(value, base_dir=tmp_path)

    assert manifest.agents[0].environment_from_host == (
        "SYNTHETIC_PROVIDER_CREDENTIAL",
    )
    rendered = json.dumps(manifest.to_dict())
    assert "SYNTHETIC_PROVIDER_CREDENTIAL" in rendered
    assert "host-only-secret" not in rendered


def test_manifest_direct_serialization_redacts_credential_shaped_model_values(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    value = _manifest_dict(fixture)
    value["model"] = {
        "id": "offline-model",
        "gateway_token": "opaque-model-secret",
        "max_tokens": 128,
    }

    manifest = BenchmarkManifest.from_dict(value, base_dir=tmp_path)
    rendered = json.dumps(manifest.to_dict())

    assert "opaque-model-secret" not in rendered
    assert "[REDACTED]" in rendered
    assert manifest.to_dict()["model"]["max_tokens"] == 128


def test_manifest_direct_serialization_redacts_secrets_in_prompt_and_checks(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    value = _manifest_dict(fixture)
    value["model"] = {"gateway_token": "prompt-secret-value"}
    value["tasks"][0]["prompt"] = "Use prompt-secret-value only as a fixture."
    value["tasks"][0]["checks"] = ["printf prompt-secret-value"]

    manifest = BenchmarkManifest.from_dict(value, base_dir=tmp_path)
    rendered = json.dumps(manifest.to_dict())

    assert "prompt-secret-value" not in rendered
    assert "[REDACTED]" in rendered


@pytest.mark.parametrize(
    "manifest_value",
    [
        {"metadata": [], "tasks": [], "agents": []},
        {
            "tasks": [
                {
                    "id": "t",
                    "prompt": "p",
                    "fixture": ".",
                    "checks": ["true"],
                    "metadata": [],
                }
            ],
            "agents": [{"id": "a", "command": {"bad": "mapping"}}],
        },
    ],
)
def test_malformed_manifest_fields_raise_benchmark_error(manifest_value) -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkManifest.from_dict(manifest_value)


@pytest.mark.parametrize(
    ("tasks", "agents"),
    [
        ([{}], [BenchmarkAgent("a", ("true",))]),
        ([], [{"id": "a", "command": ["true"]}]),
    ],
)
def test_direct_manifest_rejects_non_dataclass_case_specs(tasks, agents) -> None:
    with pytest.raises(BenchmarkError, match="manifest"):
        BenchmarkManifest(tasks=tasks, agents=agents)


@pytest.mark.parametrize(
    "environment_from_host",
    [
        "TOKEN",
        {"TOKEN": "value"},
        [""],
        ["A=B"],
        ["A\x00B"],
        [1],
        {1, "TOKEN"},
    ],
)
def test_environment_allow_list_rejects_ambiguous_values(environment_from_host) -> None:
    with pytest.raises(BenchmarkError, match="environment"):
        BenchmarkAgent(
            agent_id="a",
            command=("true",),
            environment_from_host=environment_from_host,
        )


def test_command_invoker_rejects_a_single_string_allow_list() -> None:
    with pytest.raises(BenchmarkError, match="environment_from_host"):
        CommandInvoker(environment_from_host="TOKEN")


def test_plan_is_side_effect_free_and_expands_repetitions(tmp_path: Path) -> None:
    fixture = tmp_path / "missing-fixture"
    manifest = BenchmarkManifest.from_dict(
        _manifest_dict(fixture, repetitions=2),
        base_dir=tmp_path,
    )
    calls: list[str] = []

    class Invoker:
        def invoke(self, case, budget):  # pragma: no cover - must not be called
            calls.append(case.task.id)
            raise AssertionError("dry-run invoked an agent")

    report = BenchmarkRunner(manifest, invoker=Invoker()).plan()

    assert report.dry_run is True
    assert len(report.runs) == 2
    assert [row.status for row in report.runs] == ["planned", "planned"]
    assert calls == []
    assert BenchmarkRunner(manifest, invoker=Invoker()).run(dry_run=True).dry_run


def test_runner_preserves_a_falsey_injected_invoker(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = BenchmarkManifest.from_dict(_manifest_dict(fixture), base_dir=tmp_path)

    class FalseyInvoker:
        def __bool__(self):
            return False

        def invoke(self, case, budget):
            (case.workspace / "marker.txt").write_text("ok", encoding="utf-8")
            return AgentExecution(status="completed")

    row = BenchmarkRunner(manifest, invoker=FalseyInvoker()).run().runs[0]

    assert row.status == "resolved"


def test_runner_copies_a_fresh_workspace_and_verifies_each_case(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "original.txt").write_text("unchanged", encoding="utf-8")
    manifest = BenchmarkManifest.from_dict(
        _manifest_dict(fixture, repetitions=2),
        base_dir=tmp_path,
    )
    seen: list[tuple[int, Path]] = []

    class Invoker:
        def invoke(self, case, budget):
            seen.append((case.repetition, case.workspace))
            (case.workspace / "marker.txt").write_text("ok", encoding="utf-8")
            return AgentExecution(
                status="completed",
                elapsed_seconds=0.01,
                model_requests=1,
                tool_calls=2,
                total_tokens=20,
            )

    workspace_root = tmp_path / "runs"
    report = BenchmarkRunner(
        manifest,
        invoker=Invoker(),
        workspace_root=workspace_root,
    ).run()

    assert [row.status for row in report.runs] == ["resolved", "resolved"]
    assert all(row.resolved for row in report.runs)
    assert [row.verification.passed for row in report.runs if row.verification] == [
        True,
        True,
    ]
    assert [item[0] for item in seen] == [1, 2]
    assert (fixture / "marker.txt").exists() is False
    assert list(workspace_root.iterdir()) == []
    summary = report.to_dict()["summary"]
    assert summary["by_agent"]["fake"]["resolved_rate"] == 1.0


def test_report_distinguishes_agent_failure_from_unresolved_code(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = BenchmarkManifest.from_dict(_manifest_dict(fixture), base_dir=tmp_path)

    class Invoker:
        def invoke(self, case, budget):
            return AgentExecution(status="failed", exit_code=7, stderr="bad")

    report = BenchmarkRunner(manifest, invoker=Invoker()).run()

    row = report.runs[0]
    assert row.status == "agent_failed"
    assert row.resolved is False
    assert row.verification is not None
    assert row.verification.passed is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_model_requests", 0),
        ("max_tool_calls", -1),
        ("max_total_tokens", 0),
        ("output_char_limit", 0),
        ("max_wall_time_seconds", float("nan")),
    ],
)
def test_budget_rejects_non_positive_or_non_finite_values(
    field: str, value: object
) -> None:
    with pytest.raises(BenchmarkError):
        BenchmarkBudget(**{field: value})


def test_budget_rejects_unknown_field_in_manifest() -> None:
    with pytest.raises(BenchmarkError, match="unknown budget"):
        BenchmarkBudget.from_mapping({"max_cats": 2})


def test_budget_rejects_conflicting_aliases() -> None:
    with pytest.raises(BenchmarkError, match="duplicate budget fields"):
        BenchmarkBudget.from_mapping({"max_time": 10, "max_wall_time_seconds": 20})


def test_check_rejects_conflicting_timeout_aliases(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    value = _manifest_dict(fixture)
    value["tasks"][0]["checks"] = [
        {
            "name": "ambiguous",
            "command": "true",
            "timeout": 1,
            "timeout_seconds": 2,
        }
    ]

    with pytest.raises(BenchmarkError, match="aliases"):
        BenchmarkManifest.from_dict(value, base_dir=tmp_path)


def test_runner_marks_reported_counter_overrun_as_budget_exceeded(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = BenchmarkManifest.from_dict(_manifest_dict(fixture), base_dir=tmp_path)

    class Invoker:
        def invoke(self, case, budget):
            return AgentExecution(
                status="completed",
                model_requests=budget.max_model_requests + 1,
            )

    row = BenchmarkRunner(manifest, invoker=Invoker()).run().runs[0]

    assert row.status == "budget_exceeded"
    assert row.error == "model_requests budget exceeded"


def test_runner_accepts_structural_runtime_result_from_in_process_adapter(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = BenchmarkManifest.from_dict(_manifest_dict(fixture), base_dir=tmp_path)

    class RuntimeResult:
        phase = "completed"
        history = ("message",)
        elapsed_seconds = 0.01
        model_requests = 1
        tool_calls = 1
        final_text = "finished"
        reason = "model returned a final response"
        usage = type("Usage", (), {"total_tokens": 10})()

    class Invoker:
        def invoke(self, case, budget):
            (case.workspace / "marker.txt").write_text("ok", encoding="utf-8")
            return RuntimeResult()

    row = BenchmarkRunner(manifest, invoker=Invoker()).run().runs[0]

    assert row.status == "resolved"
    assert row.execution is not None
    assert row.execution.total_tokens == 10


def test_command_invoker_enforces_wall_timeout_and_substitutes_task_file(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = BenchmarkManifest.from_dict(
        {
            **_manifest_dict(fixture),
            "agents": [
                {
                    "id": "slow",
                    "command": [
                        sys.executable,
                        "-c",
                        (
                            "import pathlib,time; "
                            "pathlib.Path('{workspace}/seen.txt').write_text("
                            "pathlib.Path('{task_file}').read_text()); "
                            "time.sleep(1)"
                        ),
                    ],
                }
            ],
            # Leave enough time for a cold Python interpreter to start before
            # it writes the marker; the command still sleeps well past the
            # bound, so the timeout assertion remains deterministic.
            "budget": {"max_wall_time_seconds": 0.2, "output_char_limit": 80},
        },
        base_dir=tmp_path,
    )
    task = manifest.tasks[0]
    agent = manifest.agents[0]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    case = type(
        "Case",
        (),
        {
            "agent": agent,
            "task": task,
            "workspace": workspace,
            "repetition": 1,
        },
    )()
    execution = CommandInvoker().invoke(case, manifest.budget)

    assert execution.status == "timeout"
    assert execution.exit_code is not None
    assert (workspace / "seen.txt").read_text(encoding="utf-8") == (
        "Create marker.txt.\n"
    )
    assert not list(workspace.glob(".benchmark-task-*.txt"))


def test_command_invoker_bounds_large_stdout_and_stderr(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = BenchmarkManifest.from_dict(
        {
            **_manifest_dict(fixture),
            "agents": [
                {
                    "id": "noisy",
                    "command": [
                        sys.executable,
                        "-c",
                        "import sys; sys.stdout.write('o'*1000000); sys.stderr.write('e'*1000000)",
                    ],
                }
            ],
            "budget": {"max_wall_time_seconds": 5, "output_char_limit": 80},
        },
        base_dir=tmp_path,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    case = type(
        "Case",
        (),
        {
            "agent": manifest.agents[0],
            "task": manifest.tasks[0],
            "workspace": workspace,
            "repetition": 1,
        },
    )()

    execution = CommandInvoker().invoke(case, manifest.budget)

    assert execution.status == "completed"
    assert len(execution.stdout) <= 80
    assert len(execution.stderr) <= 80
    assert "truncated" in execution.stdout
    assert "truncated" in execution.stderr


def test_command_invoker_does_not_reinterpret_markers_inside_task_prompt(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    prompt = "Keep the literal markers {agent_id} and {workspace}."
    task = BenchmarkManifest.from_dict(
        {
            **_manifest_dict(fixture),
            "tasks": [
                {
                    "id": "marker-task",
                    "prompt": prompt,
                    "fixture": str(fixture),
                    "checks": ["test -f received.txt"],
                }
            ],
            "agents": [
                {
                    "id": "marker-agent",
                    "command": [
                        sys.executable,
                        "-c",
                        "import pathlib,sys; pathlib.Path('received.txt').write_text(sys.argv[1])",
                        "{task}",
                    ],
                }
            ],
        },
        base_dir=tmp_path,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    case = type(
        "Case",
        (),
        {
            "agent": task.agents[0],
            "task": task.tasks[0],
            "workspace": workspace,
            "repetition": 1,
        },
    )()

    execution = CommandInvoker().invoke(case, task.budget)

    assert execution.status == "completed"
    assert (workspace / "received.txt").read_text(encoding="utf-8") == prompt


def test_standalone_command_invoker_redacts_explicit_agent_environment(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest = BenchmarkManifest.from_dict(
        {
            **_manifest_dict(fixture),
            "agents": [
                {
                    "id": "echo-secret",
                    "command": [
                        sys.executable,
                        "-c",
                        "import os; print(os.environ['CUSTOM_TOKEN'])",
                    ],
                    "environment": {"CUSTOM_TOKEN": "synthetic-token"},
                }
            ],
        },
        base_dir=tmp_path,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    case = type(
        "Case",
        (),
        {
            "agent": manifest.agents[0],
            "task": manifest.tasks[0],
            "workspace": workspace,
            "repetition": 1,
        },
    )()

    execution = CommandInvoker().invoke(case, manifest.budget)

    assert execution.status == "completed"
    assert "synthetic-token" not in execution.stdout
    assert "[REDACTED]" in execution.stdout


def test_command_invoker_passes_only_explicit_host_environment_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    monkeypatch.setenv("SYNTHETIC_PROVIDER_CREDENTIAL", "host-only-secret")
    monkeypatch.setenv("UNLISTED_PROVIDER_CREDENTIAL", "must-not-pass")
    agent = BenchmarkAgent(
        agent_id="host-env",
        command=[
            sys.executable,
            "-c",
            (
                "import os, pathlib; "
                "pathlib.Path('received.txt').write_text("
                "os.environ.get('SYNTHETIC_PROVIDER_CREDENTIAL', 'missing') + '|' + "
                "os.environ.get('UNLISTED_PROVIDER_CREDENTIAL', 'missing'), "
                "encoding='utf-8')"
            ),
        ],
        environment_from_host=("SYNTHETIC_PROVIDER_CREDENTIAL",),
    )
    manifest = BenchmarkManifest.from_dict(
        {
            **_manifest_dict(fixture),
            "agents": [
                {
                    "id": agent.id,
                    "command": list(agent.command),
                    "environment_from_host": list(agent.environment_from_host),
                }
            ],
        },
        base_dir=tmp_path,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    case = type(
        "Case",
        (),
        {
            "agent": manifest.agents[0],
            "task": manifest.tasks[0],
            "workspace": workspace,
            "repetition": 1,
        },
    )()

    execution = CommandInvoker().invoke(case, manifest.budget)

    assert execution.status == "completed"
    assert (workspace / "received.txt").read_text(encoding="utf-8") == (
        "host-only-secret|missing"
    )
    assert "host-only-secret" not in execution.stdout


def test_runner_excludes_forwarded_host_values_from_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    # Deliberately use a non-conventional name.  Name-based filtering alone
    # cannot identify this value; the benchmark's explicit allow-list can.
    monkeypatch.setenv("PRIVATE_GATEWAY_VALUE", "host-only-secret")
    manifest_value = _manifest_dict(fixture)
    manifest_value["agents"] = [
        {
            "id": "host-env",
            "command": [sys.executable, "-c", "pass"],
            "environment_from_host": ["PRIVATE_GATEWAY_VALUE"],
        }
    ]
    manifest_value["tasks"][0]["checks"] = ['test -z "${PRIVATE_GATEWAY_VALUE+x}"']
    manifest = BenchmarkManifest.from_dict(manifest_value, base_dir=tmp_path)

    report = BenchmarkRunner(manifest).run()

    assert report.runs[0].status == "resolved"
    assert report.runs[0].verification is not None
    assert report.runs[0].verification.passed
    assert "host-only-secret" not in report.to_json()


def test_runner_aggregates_injected_command_invoker_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    monkeypatch.setenv("INJECTED_HOST_VALUE", "host-invoker-secret")
    manifest = BenchmarkManifest.from_dict(
        _manifest_dict(fixture),
        base_dir=tmp_path,
    )

    class PassthroughInvoker(CommandInvoker):
        """Simulate an adapter that returns output without local redaction."""

        def invoke(self, case, budget):
            return AgentExecution(
                status="completed",
                stdout=(
                    self.environment["INJECTED_EXPLICIT_VALUE"]
                    + " "
                    + os.environ["INJECTED_HOST_VALUE"]
                ),
            )

    invoker = PassthroughInvoker(
        environment={"INJECTED_EXPLICIT_VALUE": "explicit-invoker-secret"},
        environment_from_host=("INJECTED_HOST_VALUE",),
    )
    report = BenchmarkRunner(manifest, invoker=invoker).run()

    assert report.runs[0].execution is not None
    assert report.runs[0].execution.stdout == "[REDACTED] [REDACTED]"
    rendered = report.to_json()
    assert "explicit-invoker-secret" not in rendered
    assert "host-invoker-secret" not in rendered


def test_runner_excludes_injected_invoker_host_values_from_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    monkeypatch.setenv("INJECTED_VERIFIER_HOST_VALUE", "verifier-host-secret")
    manifest_value = _manifest_dict(fixture)
    manifest_value["tasks"][0]["checks"] = [
        'test -z "${INJECTED_VERIFIER_HOST_VALUE+x}"'
    ]
    manifest = BenchmarkManifest.from_dict(manifest_value, base_dir=tmp_path)

    class PassthroughInvoker(CommandInvoker):
        def invoke(self, case, budget):
            (case.workspace / "marker.txt").write_text("ok", encoding="utf-8")
            return AgentExecution(status="completed")

    invoker = PassthroughInvoker(
        environment_from_host=("INJECTED_VERIFIER_HOST_VALUE",)
    )
    report = BenchmarkRunner(manifest, invoker=invoker).run()

    assert report.runs[0].status == "resolved"


def test_cli_defaults_to_dry_run_and_can_write_report(tmp_path: Path, capsys) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest_dict(fixture)), encoding="utf-8")
    output_path = tmp_path / "result.json"

    assert main(["--manifest", str(manifest_path), "--output", str(output_path)]) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert report["runs"][0]["status"] == "planned"


def test_report_redacts_explicit_secret() -> None:
    execution = AgentExecution(status="completed", stdout="token=fixture-secret")
    from coding_agent.benchmark import BenchmarkRunResult

    row = BenchmarkRunResult(
        run_id="a::t::r1",
        task_id="t",
        agent_id="a",
        repetition=1,
        status="unresolved",
        resolved=False,
        started_at="2026-01-01T00:00:00Z",
        execution=execution,
    )

    rendered = row.to_dict(secrets=["fixture-secret"])

    assert "fixture-secret" not in json.dumps(rendered)


def test_report_treats_a_single_secret_string_as_one_value() -> None:
    from coding_agent.benchmark import BenchmarkRunResult

    row = BenchmarkRunResult(
        run_id="a::t::r1",
        task_id="t",
        agent_id="a",
        repetition=1,
        status="unresolved",
        resolved=False,
        started_at="2026-01-01T00:00:00Z",
        execution=AgentExecution(status="completed", stdout="xabcx"),
    )

    rendered = row.to_dict(secrets="abc")

    assert rendered["execution"]["stdout"] == "x[REDACTED]x"


def test_runner_redacts_model_config_secret_from_adapter_output(
    tmp_path: Path,
) -> None:
    """Credential-shaped model fields must protect custom adapter output too."""

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    manifest_value = _manifest_dict(fixture)
    manifest_value["model"] = {
        "id": "offline-model",
        "gateway_token": "model-secret-value",
    }
    manifest = BenchmarkManifest.from_dict(manifest_value, base_dir=tmp_path)

    class Invoker:
        def invoke(self, case, budget):
            return AgentExecution(status="completed", stdout="model-secret-value")

    report = BenchmarkRunner(manifest, invoker=Invoker()).run()

    assert report.runs[0].execution is not None
    assert report.runs[0].execution.stdout == "[REDACTED]"
    assert "model-secret-value" not in report.to_json()
