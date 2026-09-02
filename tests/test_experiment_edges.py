"""Boundary tests for the reproducible Terminal-Bench experiment helpers."""

from __future__ import annotations

import json
import struct
import subprocess
import zipfile
from pathlib import Path

import pytest

from benchmarks.pin_opencode_images import (
    DEFAULT_SOURCE_IMAGES,
    PINNED_NODE_VERSION,
    PINNED_OPENCODE_VERSION,
    build_image_records,
    derived_image_name,
    docker_build_argv,
    pinned_dockerfile,
    prepare_pinned_dataset,
)
from benchmarks.pin_opencode_images import (
    TASK_IDS as IMAGE_TASK_IDS,
)
from benchmarks.run_terminal_bench_2_1 import (
    COURSE_AGENT_LABEL,
    MODEL_IDS,
    OPENCODE_GLM_PROVIDER,
    _aggregate_ablation_rows,
    _changed_result_files,
    _credential_values,
    _fixed_model_id_from_agent_model,
    _reasoning_kwargs_for_model,
    _result_file_snapshot,
    _scrub_generated_artifacts,
    _simple_bar_png,
    _sudo_preserving_argv,
    _summarize_result_file,
    aggregate_matrix_rows,
    choose_formal_round_strategy,
    formal_experiment_commands,
    formal_strategy_from_ablation,
    load_image_manifest,
    make_formal_plan,
    probe_model_route,
    render_matrix_markdown,
    run_ablation_experiment,
    run_formal_experiment,
    run_smoke_matrix,
    validate_matrix_rows,
    validate_smoke_report,
    write_ablation_artifacts,
)
from scripts.scan_credentials import scan_paths


def test_fixed_formal_commands_carry_the_same_budget_and_route() -> None:
    commands = formal_experiment_commands(
        dataset="terminal-bench/terminal-bench-2-1@sha256:fixture",
        environ={},
    )

    assert len(commands) == 6
    for command in commands:
        argv = command.argv
        assert "--n-concurrent" in argv
        assert argv[argv.index("--n-concurrent") + 1] == "1"
        assert argv[argv.index("--max-retries") + 1] == "0"
        assert "--allow-agent-host" in argv
        if command.agent == "coding_agent.harbor_plugin:CourseCodingAgent":
            assert "max_model_turns=20" in argv
            assert "max_tool_calls=80" in argv
            assert "max_wall_time_seconds=900" in argv


def test_glm_opencode_command_uses_chat_compatible_provider() -> None:
    commands = formal_experiment_commands(
        dataset="terminal-bench/terminal-bench-2-1@sha256:fixture",
        environ={},
    )
    glm_opencode = commands[3]
    argv_text = " ".join(glm_opencode.argv)
    assert f"--model {OPENCODE_GLM_PROVIDER}/{MODEL_IDS[1]}" in argv_text
    config_arg = next(
        glm_opencode.argv[index + 1]
        for index, value in enumerate(glm_opencode.argv)
        if value == "--agent-kwarg"
        and index + 1 < len(glm_opencode.argv)
        and glm_opencode.argv[index + 1].startswith("opencode_config=")
    )
    assert "@ai-sdk/openai-compatible" in config_arg
    assert "OPENAI_API_KEY" in config_arg
    assert "ZAI_API_KEY" not in config_arg
    assert f'"small_model":"{OPENCODE_GLM_PROVIDER}/{MODEL_IDS[1]}"' in config_arg


def test_fixed_opencode_commands_pin_title_model_to_primary_route() -> None:
    commands = formal_experiment_commands(
        dataset="terminal-bench/terminal-bench-2-1@sha256:fixture",
        environ={},
    )
    opencode_commands = commands[1::2]
    assert len(opencode_commands) == len(MODEL_IDS)
    for command in opencode_commands:
        config_arg = next(
            command.argv[index + 1]
            for index, value in enumerate(command.argv)
            if value == "--agent-kwarg"
            and index + 1 < len(command.argv)
            and command.argv[index + 1].startswith("opencode_config=")
        )
        assert '"small_model":' in config_arg
        assert "gpt-5.4-nano" not in config_arg


def test_fixed_model_id_normalizes_each_generated_opencode_provider() -> None:
    assert _fixed_model_id_from_agent_model(f"openai/{MODEL_IDS[0]}") == MODEL_IDS[0]
    assert (
        _fixed_model_id_from_agent_model(f"{OPENCODE_GLM_PROVIDER}/{MODEL_IDS[1]}")
        == MODEL_IDS[1]
    )
    assert _fixed_model_id_from_agent_model(MODEL_IDS[2]) == MODEL_IDS[2]
    with pytest.raises(ValueError, match="fixed experiment routes"):
        _fixed_model_id_from_agent_model(f"untrusted/{MODEL_IDS[1]}")


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("course_agent", "custom.CourseAgent"),
        ("opencode_agent", "opencode"),
    ),
)
def test_formal_commands_reject_unpinned_agent_overrides(
    parameter: str, value: str
) -> None:
    with pytest.raises(ValueError, match="fixed|pinned"):
        formal_experiment_commands(
            dataset="terminal-bench/terminal-bench-2-1@sha256:fixture",
            environ={},
            **{parameter: value},
        )


def test_harbor_sudo_prefix_is_explicit_and_reported() -> None:
    from benchmarks.run_terminal_bench_2_1 import build_command

    command = build_command(
        agent="agent", model="model", harbor_bin="/usr/bin/harbor", harbor_sudo=True
    )
    assert command.argv[:4] == ("sudo", "-n", "/usr/bin/harbor", "run")
    plan = make_formal_plan(harbor_bin="/usr/bin/harbor", harbor_sudo=True)
    assert plan["harbor_invocation_prefix"] == ["sudo", "-n", "/usr/bin/harbor"]
    assert all(
        argv[:3] == ["sudo", "-n", "/usr/bin/harbor"] for argv in plan["commands"]
    )


def test_sudo_execution_preserves_only_referenced_environment_names() -> None:
    argv = (
        "sudo",
        "-n",
        "/usr/bin/harbor",
        "run",
        "--agent-env",
        "OPENAI_API_KEY=${GATEWAY_SECRET}",
        "--agent-env",
        "OPENAI_BASE_URL=${CODING_AGENT_BASE_URL}",
    )
    environment = {
        "GATEWAY_SECRET": "synthetic-secret-value",
        "CODING_AGENT_BASE_URL": "https://gateway.example/v1",
        "CODING_AGENT_MODEL": "model",
        "UNRELATED": "value",
    }

    effective = _sudo_preserving_argv(argv, environment)

    assert effective[:3] == (
        "sudo",
        "-n",
        "--preserve-env=CODING_AGENT_BASE_URL,CODING_AGENT_MODEL,GATEWAY_SECRET",
    )
    assert "/usr/bin/harbor" in effective
    assert "synthetic-secret-value" not in effective
    assert "UNRELATED" not in effective[2]


def test_sudo_execution_projection_leaves_non_sudo_commands_unchanged() -> None:
    argv = ("harbor", "run", "--yes")
    assert _sudo_preserving_argv(argv, {"CODING_AGENT_MODEL": "model"}) == argv


def test_sudo_execution_projection_resolves_bare_binary_from_caller_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "harbor"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    effective = _sudo_preserving_argv(
        ("sudo", "-n", "harbor", "--version"),
        {"CODING_AGENT_MODEL": "model"},
    )

    assert effective[2] == "--preserve-env=CODING_AGENT_MODEL"
    assert effective[3] == str(binary)


def test_pinned_image_build_uses_isolated_context(tmp_path: Path) -> None:
    calls: list[tuple[list[str], list[str]]] = []
    digest = "sha256:" + "a" * 64

    def fake_runner(argv, **kwargs):
        command = list(argv)
        if "build" in command:
            context = Path(command[-1])
            calls.append((command, sorted(path.name for path in context.iterdir())))
            assert context.resolve() != Path.cwd().resolve()
            assert sorted(path.name for path in context.iterdir()) == ["Dockerfile"]
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f'["repo/image@{digest}"]|{digest}',
            stderr="",
        )

    records = build_image_records(
        images=DEFAULT_SOURCE_IMAGES,
        output_dir=tmp_path / "images",
        execute=True,
        sudo=False,
        runner=fake_runner,
    )
    assert len(records) == 8
    assert len(calls) == 8
    assert all(files == ["Dockerfile"] for _argv, files in calls)
    assert all(record.status == "built" for record in records)


def test_matrix_markdown_lists_each_token_distribution() -> None:
    summary = {
        "m::a": {
            "model": "m",
            "agent": "a",
            "n_passed": 1,
            "n_evaluable_trials": 1,
            "accuracy": 1.0,
            "accuracy_wilson_95": {"low": 0.2, "high": 1.0},
            "quartiles": {
                name: {"q1": value, "median": value + 1, "q3": value + 2}
                for name, value in {
                    "agent_elapsed_seconds": 1,
                    "total_elapsed_seconds": 2,
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "cache_tokens": 30,
                    "total_tokens": 40,
                }.items()
            },
            "median_model_requests": 3,
            "median_tool_calls": 4,
            "n_setup_failures": 0,
        }
    }
    markdown = render_matrix_markdown(summary)
    for header in (
        "input tokens median [IQR]",
        "output tokens median [IQR]",
        "cache tokens median [IQR]",
        "total tokens median [IQR]",
    ):
        assert header in markdown
    assert "11 [10-12]" in markdown
    assert "21 [20-22]" in markdown
    assert "31 [30-32]" in markdown
    assert "41 [40-42]" in markdown


def test_reasoning_kwargs_fail_closed_and_preserve_structured_values() -> None:
    supported = _reasoning_kwargs_for_model(
        MODEL_IDS[0],
        requested_effort="high",
        capabilities={
            MODEL_IDS[0]: {
                "status": "supported",
                "parameter": "thinking",
                "requested_effort": "high",
                "accepted_value": {"budget": 3},
            }
        },
    )
    assert supported["reasoning_parameter"] == "thinking"
    assert supported["reasoning_value"] == {"budget": 3}
    assert supported["reasoning_effort"] == "high"

    assert _reasoning_kwargs_for_model(
        MODEL_IDS[0],
        requested_effort="high",
        capabilities={MODEL_IDS[0]: {"status": "unsupported"}},
    ) == {"reasoning_capability_status": "unsupported"}
    assert _reasoning_kwargs_for_model(
        MODEL_IDS[0],
        requested_effort="high",
        capabilities={MODEL_IDS[0]: {"status": "error"}},
    ) == {"reasoning_capability_status": "error"}

    # A provider may use a native string outside the portable effort enum.
    # Keep the requested effort only as an audit/fallback field and forward the
    # exact native value through the JSON-aware option.
    nonstandard = _reasoning_kwargs_for_model(
        MODEL_IDS[0],
        requested_effort="high",
        capabilities={
            MODEL_IDS[0]: {
                "status": "supported",
                "parameter": "enable_thinking",
                "requested_effort": "high",
                "accepted_value": "enabled",
            }
        },
    )
    assert nonstandard["reasoning_effort"] == "high"
    assert nonstandard["reasoning_value"] == "enabled"


def test_probe_model_route_returns_safe_capability_metadata() -> None:
    class ProbeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def probe_reasoning_effort(self, effort, *, timeout_seconds):
            from coding_agent.model import ReasoningCapability

            return ReasoningCapability(
                status="supported",
                requested_effort=effort,
                parameter="thinking",
                accepted_value="high",
            )

    result = probe_model_route(
        MODEL_IDS[1],
        environ={"ZAI_API_KEY": "synthetic-probe-key"},
        client_factory=ProbeClient,
    )

    assert result["status"] == "supported"
    assert result["parameter"] == "thinking"
    assert result["key_env"] == "ZAI_API_KEY"
    assert "synthetic-probe-key" not in repr(result)


def test_probe_model_route_redacts_secret_in_capability_detail() -> None:
    secret = "synthetic-probe-detail-secret"

    class ProbeClient:
        def __init__(self, **kwargs):
            pass

        def probe_reasoning_effort(self, effort, *, timeout_seconds):
            from coding_agent.model import ReasoningCapability

            return ReasoningCapability(
                status="error",
                requested_effort=effort,
                error_type="GatewayError",
                detail=f"provider echoed {secret}",
            )

    result = probe_model_route(
        MODEL_IDS[1],
        environ={"ZAI_API_KEY": secret},
        client_factory=ProbeClient,
    )

    assert result["status"] == "error"
    assert secret not in repr(result)
    assert "[REDACTED]" in str(result["detail"])


def _matrix_rows(repetitions: int = 2) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repetition in range(1, repetitions + 1):
        rows.append(
            {
                "kind": "trial",
                "model": "m",
                "agent": COURSE_AGENT_LABEL,
                "task_name": "t",
                "repetition": repetition,
                "path": f"/trials/{repetition}/result.json",
                "passed": True,
            }
        )
    return rows


def test_matrix_validator_rejects_missing_duplicate_and_unknown_keys() -> None:
    complete = validate_matrix_rows(
        _matrix_rows(),
        repetitions=2,
        expected_models=("m",),
        expected_agents=(COURSE_AGENT_LABEL,),
        expected_tasks=("t",),
    )
    assert complete["complete"] is True

    duplicate = _matrix_rows()
    duplicate.append(dict(duplicate[0]))
    duplicate[-1]["path"] = "/trials/copy/result.json"
    duplicate_result = validate_matrix_rows(
        duplicate,
        repetitions=2,
        expected_models=("m",),
        expected_agents=(COURSE_AGENT_LABEL,),
        expected_tasks=("t",),
    )
    assert duplicate_result["complete"] is False
    assert duplicate_result["duplicate_keys"]

    unknown = _matrix_rows()
    unknown[0]["task_name"] = "other"
    unknown_result = validate_matrix_rows(
        unknown,
        repetitions=2,
        expected_models=("m",),
        expected_agents=(COURSE_AGENT_LABEL,),
        expected_tasks=("t",),
    )
    assert unknown_result["complete"] is False
    assert unknown_result["unknown_rows"]

    unresolved = _matrix_rows()
    unresolved[0].pop("passed")
    unresolved_result = validate_matrix_rows(
        unresolved,
        repetitions=2,
        expected_models=("m",),
        expected_agents=(COURSE_AGENT_LABEL,),
        expected_tasks=("t",),
    )
    assert unresolved_result["complete"] is False
    assert unresolved_result["unresolved_rows"]


def test_matrix_validator_normalizes_the_generated_zai_model_prefix() -> None:
    row = {
        "kind": "trial",
        "model": f"{OPENCODE_GLM_PROVIDER}/{MODEL_IDS[1]}",
        "agent": "opencode-pinned",
        "task_name": "fix-git",
        "repetition": 1,
        "path": "/trials/zai/result.json",
        "passed": True,
    }
    result = validate_matrix_rows(
        [row],
        repetitions=1,
        expected_models=(MODEL_IDS[1],),
        expected_agents=("opencode-pinned",),
        expected_tasks=("fix-git",),
    )

    assert result["complete"] is True
    assert result["normalised_rows"][0]["matrix_model"] == MODEL_IDS[1]


def test_ablation_summary_does_not_call_duplicate_repetitions_complete() -> None:
    rows = [
        {
            "kind": "trial",
            "model": MODEL_IDS[0],
            "ablation_model": MODEL_IDS[0],
            "ablation_strategy": "efficiency_20",
            "task_name": "build-cython-ext",
            "repetition": 1,
            "trial_name": "same",
            "path": "/one/result.json",
            "passed": True,
        },
        {
            "kind": "trial",
            "model": MODEL_IDS[0],
            "ablation_model": MODEL_IDS[0],
            "ablation_strategy": "efficiency_20",
            "task_name": "build-cython-ext",
            "repetition": 1,
            "trial_name": "same-copy",
            "path": "/two/result.json",
            "passed": True,
        },
    ]
    summary = _aggregate_ablation_rows(
        rows,
        models=(MODEL_IDS[0],),
        strategies=("efficiency_20",),
        tasks=("build-cython-ext",),
        repetitions=2,
    )
    item = summary[f"{MODEL_IDS[0]}::efficiency_20::build-cython-ext"]
    assert item["complete"] is False
    assert item["duplicate_repetitions"] == {1: 2}


def test_ablation_summary_assigns_harbor_attempt_ordinals() -> None:
    # Harbor's n_attempts result files have distinct trial paths but commonly
    # omit a repetition field.  The aggregator must treat three such results
    # as one complete three-repeat group.
    rows = [
        {
            "kind": "trial",
            "model": MODEL_IDS[0],
            "ablation_model": MODEL_IDS[0],
            "ablation_strategy": "efficiency_20",
            "task_name": "build-cython-ext",
            "path": f"/trial-{suffix}/result.json",
            "passed": bool(index % 2),
        }
        for index, suffix in enumerate(("c", "a", "b"), start=1)
    ]
    summary = _aggregate_ablation_rows(
        rows,
        models=(MODEL_IDS[0],),
        strategies=("efficiency_20",),
        tasks=("build-cython-ext",),
        repetitions=3,
    )
    item = summary[f"{MODEL_IDS[0]}::efficiency_20::build-cython-ext"]
    assert item["complete"] is True
    assert item["n_trials"] == 3
    assert item["missing_repetitions"] == []
    assert item["duplicate_repetitions"] == {}


def test_ablation_summary_requires_evaluable_and_in_range_repetitions() -> None:
    rows = [
        {
            "kind": "trial",
            "model": MODEL_IDS[0],
            "ablation_model": MODEL_IDS[0],
            "ablation_strategy": "efficiency_20",
            "task_name": "build-cython-ext",
            "repetition": 1,
            "path": "/one/result.json",
            "passed": True,
        },
        {
            "kind": "trial",
            "model": MODEL_IDS[0],
            "ablation_model": MODEL_IDS[0],
            "ablation_strategy": "efficiency_20",
            "task_name": "build-cython-ext",
            "repetition": 3,
            "path": "/three/result.json",
            # A result with no verifier verdict is unresolved, even though the
            # row count happens to match the requested repetition count.
        },
    ]
    summary = _aggregate_ablation_rows(
        rows,
        models=(MODEL_IDS[0],),
        strategies=("efficiency_20",),
        tasks=("build-cython-ext",),
        repetitions=2,
    )
    item = summary[f"{MODEL_IDS[0]}::efficiency_20::build-cython-ext"]
    assert item["complete"] is False
    assert item["n_unresolved_trials"] == 1
    assert item["invalid_repetitions"] == [3]
    assert item["missing_repetitions"] == [2]


def test_round_strategy_requires_complete_comparisons() -> None:
    summary = {}
    for task in ("build-cython-ext", "write-compressor"):
        summary[f"{MODEL_IDS[0]}::efficiency_20::{task}"] = {
            "complete": True,
            "n_passed": 1,
        }
        summary[f"{MODEL_IDS[0]}::efficiency_30::{task}"] = {
            "complete": True,
            "n_passed": 1,
        }
    decision = choose_formal_round_strategy(
        summary,
        models=(MODEL_IDS[0],),
        tasks=("build-cython-ext", "write-compressor"),
    )
    assert decision["selected_strategy"] == "efficiency_20"
    assert decision["status"] == "complete"

    summary[f"{MODEL_IDS[0]}::efficiency_30::write-compressor"]["complete"] = False
    incomplete = choose_formal_round_strategy(
        summary,
        models=(MODEL_IDS[0],),
        tasks=("build-cython-ext", "write-compressor"),
    )
    assert incomplete["selected_strategy"] == "efficiency_30"
    assert incomplete["status"] == "incomplete"


def test_formal_plan_records_and_applies_selected_round_strategy() -> None:
    plan = make_formal_plan(
        dataset="terminal-bench/terminal-bench-2-1@sha256:fixture",
        environ={},
        round_strategy="efficiency_30",
    )

    assert plan["course_round_strategy"] == "efficiency_30"
    assert plan["runtime_limits"]["course_max_model_turns"] == 30
    assert plan["runtime_limits"]["course_efficiency_mode"] is True
    course_command = plan["commands"][0]
    assert "max_model_turns=30" in course_command
    assert "efficiency_mode=True" in course_command
    assert "reserve_final_turn=True" in course_command


def test_formal_strategy_from_ablation_recomputes_canonical_values() -> None:
    decision = formal_strategy_from_ablation(
        {
            "fixed_formal_configuration": {
                "selected_strategy": "efficiency_30",
                "selected_max_model_turns": 20,
                "selected_efficiency_mode": False,
                "status": "complete",
            }
        }
    )

    assert decision["selected_strategy"] == "efficiency_30"
    assert decision["selected_max_model_turns"] == 30
    assert decision["selected_efficiency_mode"] is True
    assert decision["selected_reserve_final_turn"] is True


def test_formal_strategy_reads_report_dimensions_and_summary_alias(
    tmp_path: Path,
) -> None:
    model = "fixture-model"
    tasks = ("task-a", "task-b")
    summary: dict[str, dict[str, object]] = {}
    for task in tasks:
        for strategy in ("efficiency_20", "efficiency_30"):
            summary[f"{model}::{strategy}::{task}"] = {
                "model": model,
                "strategy": strategy,
                "task": task,
                "complete": True,
                "n_passed": 2,
            }
    artifact = tmp_path / "ablation.json"
    artifact.write_text(
        json.dumps(
            {
                "models": [model],
                "tasks": list(tasks),
                "summary": summary,
            }
        ),
        encoding="utf-8",
    )
    decision = formal_strategy_from_ablation(artifact)
    assert decision["selected_strategy"] == "efficiency_20"
    assert len(decision["comparisons"]) == 2
    assert {item["task"] for item in decision["comparisons"]} == set(tasks)


def test_formal_strategy_rejects_current_strategy_from_report() -> None:
    with pytest.raises(ValueError, match="formal efficiency strategy"):
        formal_strategy_from_ablation(
            {
                "fixed_formal_configuration": {
                    "selected_strategy": "current_20",
                    "status": "complete",
                }
            }
        )


def test_formal_strategy_rejects_contradictory_incomplete_matrix_flag() -> None:
    with pytest.raises(ValueError, match="ablation report is incomplete"):
        formal_strategy_from_ablation(
            {
                "fixed_formal_configuration": {
                    "selected_strategy": "efficiency_20",
                    "status": "complete",
                },
                "matrix_complete": False,
            }
        )


def test_ablation_artifact_keeps_dimensions_for_round_trip(tmp_path: Path) -> None:
    report = {
        "models": ["fixture-model"],
        "tasks": ["task-a"],
        "repetitions_per_task": 1,
        "ablation_summary": {
            "fixture-model::efficiency_20::task-a": {
                "model": "fixture-model",
                "strategy": "efficiency_20",
                "task": "task-a",
                "complete": True,
                "n_passed": 1,
                "n_evaluable_trials": 1,
            }
        },
        "fixed_formal_configuration": {
            "selected_strategy": "efficiency_20",
            "status": "complete",
        },
        "matrix_complete": True,
    }
    paths = write_ablation_artifacts(tmp_path, report)
    document = json.loads(Path(str(paths["json"])).read_text(encoding="utf-8"))
    assert document["models"] == ["fixture-model"]
    assert document["tasks"] == ["task-a"]
    assert formal_strategy_from_ablation(Path(str(paths["json"])))[
        "selected_strategy"
    ] == ("efficiency_20")


def test_trajectory_final_metrics_prefer_canonical_aliases(tmp_path: Path) -> None:
    trial = tmp_path / "result.json"
    trajectory = tmp_path / "trajectory.json"
    trial.write_text(
        json.dumps(
            {
                "task_name": "fix-git",
                "agent_info": {
                    "name": "course-coding-agent",
                    "model_info": {"name": "m"},
                },
                "verifier_result": {"reward": 1},
            }
        ),
        encoding="utf-8",
    )
    trajectory.write_text(
        json.dumps(
            {
                "steps": [],
                "final_metrics": {
                    "model_requests": 4,
                    "model_calls": 99,
                    "tool_calls": 3,
                    "extra": {"total_tokens": 17},
                },
            }
        ),
        encoding="utf-8",
    )
    summary = _summarize_result_file(trial)
    assert summary["model_requests"] == 4
    assert summary["tool_calls"] == 3
    assert summary["total_tokens"] == 17


def test_opencode_billing_error_is_an_infrastructure_failure(tmp_path: Path) -> None:
    trial_dir = tmp_path / "fix-git__fixture"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    result_path = trial_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "task_name": "fix-git",
                "trial_name": "fix-git__fixture",
                "agent_info": {
                    "name": "opencode-pinned",
                    "model_info": {"name": MODEL_IDS[2]},
                },
                "agent_result": {"metadata": {}},
                "verifier_result": {"reward": 0},
                "exception_info": {"exception_type": "NonZeroAgentExitCodeError"},
            }
        ),
        encoding="utf-8",
    )
    (agent_dir / "opencode.txt").write_text(
        json.dumps(
            {
                "type": "error",
                "error": {
                    "name": "APIError",
                    "data": {
                        "statusCode": 403,
                        "message": "Forbidden: Insufficient account balance",
                        "responseBody": '{"code":"INSUFFICIENT_BALANCE"}',
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = _summarize_result_file(result_path)
    group = aggregate_matrix_rows([row], expected_trials_per_group=1)[
        f"{MODEL_IDS[2]}::opencode-pinned"
    ]

    assert row["phase"] == "infrastructure"
    assert row["failure_stage"] == "model_gateway"
    assert row["infrastructure_reason"] == "model_billing"
    assert group["n_evaluable_trials"] == 0
    assert group["infrastructure_failures"] == 1
    assert group["accuracy"] is None


def test_artifact_scrubber_scans_generic_token_without_known_secret(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    token = b"sk-" + bytes("1234567890abcdef", "ascii")
    path.write_bytes(b"token=" + token)

    assert _scrub_generated_artifacts(tmp_path, secrets=()) == 1
    assert token not in path.read_bytes()


def _smoke_report_rows(*, passed: object = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in MODEL_IDS:
        for agent in ("course-coding-agent", "opencode-pinned"):
            rows.append(
                {
                    "kind": "trial",
                    "model": model,
                    "agent": agent,
                    "task_name": "fix-code-vulnerability",
                    "repetition": 1,
                    "path": f"/smoke/{model}/{agent}/result.json",
                    "passed": passed,
                }
            )
    return rows


def test_smoke_gate_allows_verifier_failure_but_blocks_setup_failure() -> None:
    report = {
        "status": "finished",
        "task": "fix-code-vulnerability",
        "expected_trials": 6,
        "result_summaries": _smoke_report_rows(),
        "reasoning_probes": [
            {"model": model, "status": "unsupported"} for model in MODEL_IDS
        ],
        "outcomes": [{"return_code": 0} for _ in range(6)],
    }
    ready = validate_smoke_report(report)
    assert ready["status"] == "ready"
    assert ready["can_proceed"] is True

    report["result_summaries"][0]["passed"] = False
    verifier = validate_smoke_report(report)
    assert verifier["status"] == "ready_with_verifier_failures"
    assert verifier["can_proceed"] is True
    assert len(verifier["verifier_failures"]) == 1

    report["result_summaries"][1]["phase"] = "setup_failure"
    blocked = validate_smoke_report(report)
    assert blocked["status"] == "blocked"
    assert blocked["can_proceed"] is False
    assert len(blocked["infrastructure_failures"]) == 1


def test_smoke_gate_rejects_incomplete_or_wrong_task(tmp_path: Path) -> None:
    report_path = tmp_path / "smoke.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "finished",
                "task": "fix-git",
                "expected_trials": 6,
                "result_summaries": _smoke_report_rows(),
            }
        ),
        encoding="utf-8",
    )
    gate = validate_smoke_report(report_path)
    assert gate["status"] == "blocked"
    assert gate["report_path"] == str(report_path.resolve())
    assert any("task" in reason for reason in gate["blocking_reasons"])


def _ready_image_manifest(output_dataset: str | None = None) -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    return {
        "schema_version": "pinned-opencode-images/v1",
        "pinned_node_version": PINNED_NODE_VERSION,
        "pinned_opencode_version": PINNED_OPENCODE_VERSION,
        "status": "ready",
        "dataset": {"output_dataset": output_dataset} if output_dataset else {},
        "records": [
            {
                "task": task,
                "source_image": "source/image@" + digest,
                "source_digest": digest,
                "derived_image": f"derived/{task}:1.18.25",
                "derived_digest": digest,
                "status": "built",
                "node_version": PINNED_NODE_VERSION,
                "opencode_version": PINNED_OPENCODE_VERSION,
            }
            for task in IMAGE_TASK_IDS
        ],
    }


def test_image_manifest_loader_validates_readiness_and_dataset(tmp_path: Path) -> None:
    manifest = _ready_image_manifest(str(tmp_path / "dataset"))
    loaded = load_image_manifest(manifest, dataset=str(tmp_path / "dataset"))
    assert loaded["ready"] is True
    assert len(loaded["records"]) == 8
    with pytest.raises(ValueError, match="does not match"):
        load_image_manifest(manifest, dataset=str(tmp_path / "other"))

    pending = _ready_image_manifest()
    pending["records"] = [dict(pending["records"][0], status="planned")]
    pending["records"] += [
        dict(item) for item in _ready_image_manifest()["records"][1:]
    ]
    assert load_image_manifest(pending)["ready"] is False
    with pytest.raises(ValueError, match="not ready"):
        load_image_manifest(pending, require_ready=True)


def test_image_manifest_schema_and_public_hash_round_trip() -> None:
    manifest = _ready_image_manifest()
    loaded = load_image_manifest(manifest)
    public = {
        "schema_version": "pinned-opencode-images/v1",
        "path": loaded["path"],
        "manifest_sha256": loaded["manifest_sha256"],
        "status": loaded["status"],
        "ready": loaded["ready"],
        "pinned_node_version": loaded["pinned_node_version"],
        "pinned_opencode_version": loaded["pinned_opencode_version"],
        "output_dataset": loaded["output_dataset"],
        "records": loaded["records"],
    }
    round_tripped = load_image_manifest(public)
    assert round_tripped["manifest_sha256"] == loaded["manifest_sha256"]

    invalid_schema = dict(manifest, schema_version="pinned-opencode-images/v99")
    with pytest.raises(ValueError, match="schema_version"):
        load_image_manifest(invalid_schema)


def test_smoke_execution_gate_retains_image_manifest_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The executed smoke gate must carry the same pinned-image identity."""

    import benchmarks.run_terminal_bench_2_1 as runner

    manifest = _ready_image_manifest()
    result_paths = [tmp_path / f"result-{index}.json" for index in range(6)]
    rows = _smoke_report_rows()
    row_index = 0

    monkeypatch.setattr(runner, "_run_one", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        runner, "_changed_result_files", lambda *args, **kwargs: result_paths
    )

    def fake_summary(path, *, secrets=(), image_manifest=None):
        nonlocal row_index
        row = dict(rows[row_index])
        row_index += 1
        assert image_manifest is not None
        return row

    monkeypatch.setattr(runner, "_summarize_result_file", fake_summary)
    report = run_smoke_matrix(
        dataset="terminal-bench/terminal-bench-2-1@sha256:fixture",
        jobs_dir=tmp_path / "runs",
        environ={
            "DEEPSEEK_API_KEY": "synthetic-deepseek-key",
            "ZAI_API_KEY": "synthetic-zai-key",
            "CODING_AGENT_API_KEY": "synthetic-relay-key",
        },
        execute=True,
        probe_reasoning=False,
        image_manifest=manifest,
    )

    loaded = load_image_manifest(manifest)
    assert report["smoke_gate"]["can_proceed"] is True
    assert report["smoke_gate"]["image_manifest_sha256"] == loaded["manifest_sha256"]


def test_short_known_credentials_are_collected_without_corrupting_gate_keys() -> None:
    values = _credential_values(
        {
            "ZAI_API_KEY": "z",
            "DEEPSEEK_API_KEY": "d",
            "CODING_AGENT_API_KEY": "c",
        }
    )
    assert set(values) == {"z", "d", "c"}
    report = {
        "status": "finished",
        "task": "fix-code-vulnerability",
        "expected_trials": 6,
        "result_summaries": _smoke_report_rows(),
        "reasoning_probes": [
            {"model": model, "status": "unsupported"} for model in MODEL_IDS
        ],
    }
    gate = validate_smoke_report(report, secrets=values)
    assert gate["can_proceed"] is True
    assert "can_proceed" in gate


def test_result_snapshot_detects_in_place_rewrite(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    result.write_text('{"value": 1}\n', encoding="utf-8")
    before = _result_file_snapshot(tmp_path)
    result.write_text('{"value": 2}\n', encoding="utf-8")
    assert _changed_result_files(tmp_path, before) == [result]


def test_smoke_gate_classifies_matching_nonzero_verifier_command() -> None:
    rows = _smoke_report_rows()
    target = rows[0]
    target["passed"] = False
    report = {
        "status": "finished",
        "task": "fix-code-vulnerability",
        "expected_trials": 6,
        "result_summaries": rows,
        "reasoning_probes": [
            {"model": model, "status": "unsupported"} for model in MODEL_IDS
        ],
        "outcomes": [
            {
                "model": target["model"],
                "agent": target["agent"],
                "return_code": 1,
            }
        ],
    }
    gate = validate_smoke_report(report)
    assert gate["can_proceed"] is True
    assert gate["verifier_command_failures"]
    assert gate["infrastructure_failures"] == []


def test_formal_execution_requires_smoke_gate_before_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched = False

    def fail_if_launched(*args: object, **kwargs: object) -> int:
        nonlocal launched
        launched = True
        return 0

    import benchmarks.run_terminal_bench_2_1 as runner

    monkeypatch.setattr(runner, "_run_one", fail_if_launched)
    with pytest.raises(ValueError, match="smoke gate blocked"):
        run_formal_experiment(
            execute=True,
            probe_reasoning=False,
            environ={
                "DEEPSEEK_API_KEY": "synthetic-deepseek-key",
                "ZAI_API_KEY": "synthetic-zai-key",
                "CODING_AGENT_API_KEY": "synthetic-relay-key",
            },
            smoke_report=None,
        )
    assert launched is False


def test_executions_require_ready_image_manifest_before_model_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No executable matrix may silently use an unpinned task image."""

    import benchmarks.run_terminal_bench_2_1 as runner

    credentials = {
        "DEEPSEEK_API_KEY": "synthetic-deepseek-key",
        "ZAI_API_KEY": "synthetic-zai-key",
        "CODING_AGENT_API_KEY": "synthetic-relay-key",
    }
    monkeypatch.setattr(
        runner,
        "probe_model_route",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model probing must not run without an image manifest")
        ),
    )

    with pytest.raises(ValueError, match="image-manifest"):
        run_smoke_matrix(
            jobs_dir=tmp_path / "smoke",
            environ=credentials,
            execute=True,
        )
    with pytest.raises(ValueError, match="image-manifest"):
        run_ablation_experiment(
            jobs_dir=tmp_path / "ablation",
            environ=credentials,
            execute=True,
        )


def test_formal_execution_requires_image_manifest_after_smoke_gate(
    tmp_path: Path,
) -> None:
    credentials = {
        "DEEPSEEK_API_KEY": "synthetic-deepseek-key",
        "ZAI_API_KEY": "synthetic-zai-key",
        "CODING_AGENT_API_KEY": "synthetic-relay-key",
    }
    smoke_report = {
        "status": "finished",
        "task": "fix-code-vulnerability",
        "expected_trials": 6,
        "result_summaries": _smoke_report_rows(),
    }
    with pytest.raises(ValueError, match="image-manifest"):
        run_formal_experiment(
            jobs_dir=tmp_path / "formal",
            environ=credentials,
            execute=True,
            probe_reasoning=False,
            smoke_report=smoke_report,
        )


def test_formal_execution_requires_completed_ablation_report_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import benchmarks.run_terminal_bench_2_1 as runner

    credentials = {
        "DEEPSEEK_API_KEY": "synthetic-deepseek-key",
        "ZAI_API_KEY": "synthetic-zai-key",
        "CODING_AGENT_API_KEY": "synthetic-relay-key",
    }
    manifest = _ready_image_manifest()
    smoke_report = {
        "status": "finished",
        "task": "fix-code-vulnerability",
        "expected_trials": 6,
        "result_summaries": _smoke_report_rows(),
        "image_manifest": {
            "manifest_sha256": load_image_manifest(manifest)["manifest_sha256"]
        },
    }
    monkeypatch.setattr(
        runner,
        "_run_one",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("formal Harbor jobs must not launch without ablation")
        ),
    )
    with pytest.raises(ValueError, match="ablation-report"):
        run_formal_experiment(
            jobs_dir=tmp_path / "formal",
            environ=credentials,
            execute=True,
            probe_reasoning=False,
            smoke_report=smoke_report,
            image_manifest=manifest,
        )


def test_formal_execution_cannot_override_incomplete_ablation_with_allow_incomplete(
    tmp_path: Path,
) -> None:
    credentials = {
        "DEEPSEEK_API_KEY": "synthetic-deepseek-key",
        "ZAI_API_KEY": "synthetic-zai-key",
        "CODING_AGENT_API_KEY": "synthetic-relay-key",
    }
    incomplete_ablation = {
        "fixed_formal_configuration": {
            "selected_strategy": "efficiency_20",
            "status": "incomplete",
        }
    }
    with pytest.raises(ValueError, match="ablation report is incomplete"):
        run_formal_experiment(
            jobs_dir=tmp_path / "formal",
            environ=credentials,
            execute=True,
            probe_reasoning=False,
            allow_incomplete=True,
            ablation_report=incomplete_ablation,
        )


def test_summary_png_contains_metric_panel_metadata(tmp_path: Path) -> None:
    path = _simple_bar_png(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        ["model-a::course", "model-b::opencode"],
        tmp_path / "summary.png",
        infrastructure_rates=[0.0, 100.0],
    )
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    text_chunks: dict[str, str] = {}
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if kind == b"tEXt":
            key, value = payload.split(b"\x00", 1)
            text_chunks[key.decode("ascii")] = value.decode("latin-1")
        if kind == b"IEND":
            break
    assert text_chunks["panel_0"] == "accuracy"
    assert text_chunks["panel_1"] == "elapsed time"
    assert text_chunks["panel_2"] == "total tokens"
    assert text_chunks["panel_3"] == "model/tool calls"
    assert "model-a::course" in text_chunks["groups"]
    assert "red bars" in text_chunks["infrastructure_marker"]
    assert "model-b::opencode=100" in text_chunks["infrastructure_rates_percent"]


def test_summary_png_handles_all_zero_panel(tmp_path: Path) -> None:
    path = _simple_bar_png(
        [[0.0, 0.0], [0.0], [0.0, 0.0], [0.0]],
        ["zero"],
        tmp_path / "zero-summary.png",
    )
    assert path.is_file()
    assert path.stat().st_size > 0


def test_pinned_opencode_docker_and_dataset_helpers(tmp_path: Path) -> None:
    assert derived_image_name("fix-git").endswith(f":{PINNED_OPENCODE_VERSION}")
    dockerfile = pinned_dockerfile("source/image:fixture")
    assert f"node:{PINNED_NODE_VERSION}-bookworm-slim" in dockerfile
    assert "opencode-ai@$OPENCODE_VERSION" in dockerfile
    assert "nvm" not in dockerfile.lower()
    argv = docker_build_argv(
        docker_bin="docker",
        source_image="source/image:fixture",
        derived_image="derived:fixture",
        dockerfile=tmp_path / "Dockerfile",
        context=tmp_path,
        sudo=False,
    )
    assert argv[:2] == ["docker", "build"]

    source = tmp_path / "source"
    output = tmp_path / "output"
    for task in IMAGE_TASK_IDS:
        task_dir = source / task
        task_dir.mkdir(parents=True)
        (task_dir / "task.toml").write_text(
            '[environment]\ndocker_image = "source/image:fixture"\n',
            encoding="utf-8",
        )
    records = []
    from benchmarks.pin_opencode_images import ImageRecord

    for task in IMAGE_TASK_IDS:
        records.append(
            ImageRecord(
                task=task,
                source_image="source/image:fixture",
                source_digest=None,
                derived_image=derived_image_name(task),
                derived_digest=None,
                node_version=PINNED_NODE_VERSION,
                opencode_version=PINNED_OPENCODE_VERSION,
                status="planned",
                dockerfile_sha256="0" * 64,
            )
        )
    report = prepare_pinned_dataset(source, output, records)
    assert report["pinned_opencode_version"] == PINNED_OPENCODE_VERSION
    assert all(
        derived_image_name(task)
        in (output / task / "task.toml").read_text(encoding="utf-8")
        for task in IMAGE_TASK_IDS
    )


@pytest.mark.parametrize("image", ['source/image:"bad"', "source/image;bad", "../bad"])
def test_pinned_image_helpers_reject_interpolation_metacharacters(image: str) -> None:
    with pytest.raises(ValueError, match="image reference"):
        pinned_dockerfile(image)


def test_image_manifest_rejects_unsafe_reference() -> None:
    records = [
        {
            "task": task,
            "source_image": "source/image:fixture",
            "derived_image": "derived/image:fixture",
            "source_digest": "sha256:" + "a" * 64,
            "derived_digest": "sha256:" + "b" * 64,
            "status": "ready",
            "node_version": PINNED_NODE_VERSION,
            "opencode_version": PINNED_OPENCODE_VERSION,
        }
        for task in IMAGE_TASK_IDS
    ]
    records[0]["derived_image"] = 'derived/image:"bad"'
    with pytest.raises(ValueError, match="safe Docker image reference"):
        load_image_manifest(
            {
                "pinned_node_version": PINNED_NODE_VERSION,
                "pinned_opencode_version": PINNED_OPENCODE_VERSION,
                "records": records,
            }
        )


def test_credential_scanner_checks_zip_members_without_printing_values(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "report.zip"
    fixture_token = "synthetic-credential-" + "value"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "trajectory.json",
            "Bearer " + fixture_token,
        )
    report = scan_paths([archive], environ={})
    assert report["clean"] is False
    assert report["finding_count"] == 1
    assert "synthetic-credential-value" not in repr(report)
