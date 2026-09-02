"""Offline tests for the experiment planner and submission gate."""

from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from benchmarks.run_terminal_bench_2_1 import (
    COURSE_AGENT,
    DATASET,
    DATASET_REFERENCE,
    OPENCODE_AGENT,
    TASK_IDS,
    _aggregate_trial_rows,
    _collect_result_files,
    _harbor_process_environment,
    _scrub_generated_artifacts,
    _summarize_result_file,
    check_credentials,
    collect_existing_results,
    make_plan,
)
from scripts import build_submission as submission
from scripts.build_submission import build_archive


def test_terminal_bench_plan_is_fixed_and_single_trial() -> None:
    plan = make_plan(model="exact-model-id")
    assert "@sha256:" in DATASET
    assert plan["dataset"] == DATASET
    assert plan["dataset_reference"] == DATASET_REFERENCE
    assert len(TASK_IDS) == 8
    assert plan["task_count"] == 8
    assert plan["repetitions_per_task"] == 1
    assert plan["agents"] == [COURSE_AGENT, OPENCODE_AGENT]
    assert plan["conditions"] == {
        "n_concurrent": 1,
        "n_attempts": 1,
        "max_retries": 0,
        "same_model": True,
    }
    assert all("exact-model-id" in " ".join(command) for command in plan["commands"])
    opencode_command = plan["commands"][1]
    assert "openai/exact-model-id" in opencode_command
    assert "OPENAI_API_KEY=${ZAI_API_KEY}" in opencode_command
    assert "OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4" in opencode_command


def test_local_terminal_bench_checkout_uses_bare_task_names() -> None:
    plan = make_plan(
        model="exact-model-id",
        dataset="/tmp/tbench21-official/terminal-bench-2-1",
    )
    assert plan["task_filters"] == list(TASK_IDS)
    assert plan["dataset_reference"] == DATASET_REFERENCE
    assert "--path" in plan["commands"][0]
    assert "--dataset" not in plan["commands"][0]


def test_registry_terminal_bench_reference_uses_dataset_option() -> None:
    plan = make_plan(model="exact-model-id")
    assert "--dataset" in plan["commands"][0]
    assert "--path" not in plan["commands"][0]
    assert plan["task_filters"] == [f"terminal-bench/{task}" for task in TASK_IDS]


def test_custom_dataset_does_not_claim_official_reference() -> None:
    plan = make_plan(model="exact-model-id", dataset="/tmp/custom-dataset")
    assert plan["dataset_reference"] is None


def test_custom_gateway_plan_uses_template_only_credentials() -> None:
    plan = make_plan(
        model="gateway-model",
        environ={
            "CODING_AGENT_PROVIDER": "custom",
            "CODING_AGENT_BASE_URL": "https://gateway.example/v1",
            "CODING_AGENT_KEY_ENV": "GATEWAY_SECRET",
            "GATEWAY_SECRET": "synthetic-secret-value",
        },
    )
    opencode_command = plan["commands"][1]
    assert "OPENAI_API_KEY=${GATEWAY_SECRET}" in opencode_command
    assert "OPENAI_BASE_URL=${CODING_AGENT_BASE_URL}" in opencode_command
    assert "synthetic-secret-value" not in repr(plan)
    assert all(command.count("--allow-agent-host") == 1 for command in plan["commands"])


def test_command_rejects_literal_sensitive_agent_environment_value() -> None:
    from benchmarks.run_terminal_bench_2_1 import build_command

    with pytest.raises(ValueError, match="templates"):
        build_command(
            agent="opencode",
            model="m",
            agent_env={"OPENAI_API_KEY": "literal-secret-value"},
        )


def test_experiment_credential_check_returns_names_only() -> None:
    assert check_credentials({"CODING_AGENT_MODEL": "m", "ZAI_API_KEY": "secret"}) == ()
    assert check_credentials({"CODING_AGENT_MODEL": "m"}) == ("ZAI_API_KEY",)


def test_harbor_process_environment_forwards_only_selected_credential() -> None:
    safe = _harbor_process_environment(
        {
            "CODING_AGENT_PROVIDER": "glm",
            "CODING_AGENT_MODEL": "model",
            "ZAI_API_KEY": "selected-secret",
            "DEEPSEEK_API_KEY": "unrelated-secret",
            "NORMAL": "value",
        }
    )
    assert safe["ZAI_API_KEY"] == "selected-secret"
    assert "DEEPSEEK_API_KEY" not in safe
    assert safe["NORMAL"] == "value"


def test_result_collection_handles_harbor_singular_result_and_summary(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job" / "result.json"
    job.parent.mkdir(parents=True)
    job.write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {
                    "n_completed_trials": 1,
                    "n_errored_trials": 0,
                    "n_cancelled_trials": 0,
                    "evals": {"course": {"metrics": [{"mean": 1.0}]}},
                },
            }
        ),
        encoding="utf-8",
    )
    trial = job.parent / "trial" / "result.json"
    trial.parent.mkdir()
    trial.write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/fix-git",
                "trial_name": "fix-git__one",
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:02Z",
                "verifier_result": {"reward": 1.0},
                "exception_info": None,
            }
        ),
        encoding="utf-8",
    )
    agent_logs = trial.parent / "agent"
    agent_logs.mkdir()
    (agent_logs / "run.json").write_text(
        json.dumps(
            {
                "model_turns": 3,
                "model_requests": 4,
                "tool_calls": 5,
                "elapsed_seconds": 1.5,
            }
        ),
        encoding="utf-8",
    )
    (agent_logs / "trajectory.json").write_text("{}", encoding="utf-8")
    paths = _collect_result_files(tmp_path)
    assert paths == [job, trial]
    job_summary = _summarize_result_file(job)
    trial_summary = _summarize_result_file(trial)
    assert job_summary["kind"] == "job"
    assert job_summary["n_completed_trials"] == 1
    assert job_summary["evals"]["course"]["mean"] == 1.0
    assert trial_summary["passed"] is True
    assert trial_summary["elapsed_seconds"] == 1.5
    assert trial_summary["model_turns"] == 3
    assert trial_summary["model_requests"] == 4
    assert trial_summary["tool_calls"] == 5
    assert trial_summary["trajectory"] == str(agent_logs / "trajectory.json")


def test_result_summary_reads_harbor_022_nested_reward_and_agent_info(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial" / "result.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/fix-git",
                "trial_name": "fix-git__nested",
                "agent_info": {
                    "name": "course-coding-agent",
                    "model_info": {"name": "exact-model-id"},
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    summary = _summarize_result_file(trial)
    assert summary["passed"] is True
    assert summary["reward"] == 1.0
    assert summary["agent"] == "course-coding-agent"
    assert summary["model"] == "exact-model-id"


def test_result_summary_marks_setup_exception_as_failed_with_reason(
    tmp_path: Path,
) -> None:
    trial = tmp_path / "trial" / "result.json"
    trial.parent.mkdir(parents=True)
    trial.write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/fix-git",
                "task_checksum": "abc123",
                "source": "terminal-bench-2-1",
                "agent_info": {"name": "opencode", "model_info": {"name": "m"}},
                "exception_info": {
                    "exception_type": "AgentSetupTimeoutError",
                    "exception_message": "Agent setup timed out after 360.0 seconds",
                },
            }
        ),
        encoding="utf-8",
    )
    summary = _summarize_result_file(trial)
    assert summary["passed"] is False
    assert summary["phase"] == "setup_timeout"
    assert "360.0" in summary["reason"]
    assert summary["task_checksum"] == "abc123"
    assert summary["source"] == "terminal-bench-2-1"


def test_result_summary_falls_back_to_atif_metrics_for_external_agent(
    tmp_path: Path,
) -> None:
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": "terminal-bench/fix-git",
                "agent_info": {
                    "name": "opencode",
                    "model_info": {"name": "m"},
                },
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    (trial_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.7",
                "steps": [
                    {"source": "user"},
                    {"source": "agent", "llm_call_count": 1, "tool_calls": [{"x": 1}]},
                    {"source": "agent", "llm_call_count": 1, "tool_calls": []},
                ],
                "final_metrics": {
                    "total_prompt_tokens": 10,
                    "total_completion_tokens": 4,
                    "total_cached_tokens": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    summary = _summarize_result_file(trial_dir / "result.json")
    assert summary["phase"] == "completed"
    assert summary["model_requests"] == 2
    assert summary["model_turns"] == 2
    assert summary["tool_calls"] == 1
    assert summary["input_tokens"] == 10
    assert summary["output_tokens"] == 4
    assert summary["cache_tokens"] == 2


def test_generated_artifact_scrubber_removes_database_cache_and_redacts_text(
    tmp_path: Path,
) -> None:
    secret = "synthetic-artifact-secret"
    (tmp_path / "events.jsonl").write_text(f"value={secret}\n", encoding="utf-8")
    (tmp_path / "opencode.db-wal").write_bytes(
        b"prefix " + secret.encode() + b" suffix"
    )
    changed = _scrub_generated_artifacts(tmp_path, secrets=(secret,))
    assert changed == 2
    assert secret not in (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert not (tmp_path / "opencode.db-wal").exists()


def test_agent_comparison_aggregates_pass_rates_and_means() -> None:
    result = _aggregate_trial_rows(
        [
            {
                "kind": "trial",
                "agent": "course",
                "passed": True,
                "elapsed_seconds": 2,
                "model_requests": 4,
                "tool_calls": 6,
            },
            {
                "kind": "trial",
                "agent": "course",
                "passed": False,
                "elapsed_seconds": 4,
                "model_requests": 8,
                "tool_calls": 10,
            },
            {"kind": "job", "agent": "course", "passed": True},
        ]
    )
    assert result["course"] == {
        "n_trials": 2,
        "n_passed": 1,
        "n_failed": 1,
        "n_unresolved": 0,
        "pass_rate": 0.5,
        "mean_elapsed_seconds": 3.0,
        "mean_model_requests": 6.0,
        "mean_tool_calls": 8.0,
    }


def test_collect_existing_results_rejects_incomplete_job(tmp_path: Path) -> None:
    job = tmp_path / "job" / "result.json"
    job.parent.mkdir(parents=True)
    job.write_text(
        json.dumps(
            {
                "n_total_trials": 8,
                "stats": {"n_running_trials": 1, "n_pending_trials": 7},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="still running"):
        collect_existing_results(
            model="m",
            dataset=str(tmp_path),
            jobs_dir=tmp_path,
            environ={},
        )


def test_custom_harbor_execution_requires_base_url() -> None:
    from benchmarks.run_terminal_bench_2_1 import run_experiment

    with pytest.raises(ValueError, match="CODING_AGENT_BASE_URL"):
        run_experiment(
            model="m",
            dataset="dataset@latest",
            jobs_dir=Path(".harbor-runs-test"),
            harbor_bin="harbor",
            execute=True,
            environ={
                "CODING_AGENT_PROVIDER": "custom",
                "CODING_AGENT_MODEL": "m",
                "CODING_AGENT_API_KEY": "secret-value",
            },
            cwd=Path.cwd(),
        )


def test_submission_archive_has_exact_two_files(tmp_path: Path) -> None:
    readme = tmp_path / "README.txt"
    video = tmp_path / "demo.mp4"
    output = tmp_path / "李上一.zip"
    readme.write_text("说明\n", encoding="utf-8")
    video.write_bytes(b"offline fixture")

    report = build_archive(
        video=video,
        readme=readme,
        output=output,
        skip_duration_check=True,
        environ={},
    )
    assert report["files"] == ["README.txt", "李上一.mp4"]
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["README.txt", "李上一.mp4"]


def test_submission_gate_rejects_known_credential(tmp_path: Path) -> None:
    readme = tmp_path / "README.txt"
    video = tmp_path / "demo.mp4"
    readme.write_text("safe", encoding="utf-8")
    video.write_bytes(b"prefix synthetic-secret suffix")
    with pytest.raises(ValueError, match="credential"):
        build_archive(
            video=video,
            readme=readme,
            output=tmp_path / "out.zip",
            skip_duration_check=True,
            environ={"ZAI_API_KEY": "synthetic-secret"},
        )


def test_submission_scanner_matches_short_configured_secret(tmp_path: Path) -> None:
    readme = tmp_path / "README.txt"
    video = tmp_path / "demo.mp4"
    readme.write_text("safe", encoding="utf-8")
    video.write_bytes(b"prefix x suffix")
    with pytest.raises(ValueError, match="credential"):
        build_archive(
            video=video,
            readme=readme,
            output=tmp_path / "out.zip",
            skip_duration_check=True,
            environ={"ZAI_API_KEY": "x"},
        )


def test_submission_gate_rejects_nonfinite_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video = tmp_path / "demo.mp4"
    video.write_bytes(b"fixture")
    fake = subprocess.CompletedProcess(
        args=["ffprobe"], returncode=0, stdout="nan\n", stderr=""
    )
    monkeypatch.setattr(submission.subprocess, "run", lambda *args, **kwargs: fake)
    with pytest.raises(ValueError, match="duration"):
        submission._check_duration(video)
