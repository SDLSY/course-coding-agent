"""Focused tests for independent acceptance verification.

Commands are deliberately tiny and local. No model or network is involved;
the tests exercise the verifier boundary and prove that it does not alter the
runtime result it wraps.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from coding_agent.verification import (
    CommandVerifier,
    VerificationCheck,
    VerificationResult,
    run_and_verify,
)


def _python(command: str) -> str:
    return f'{sys.executable} -c "{command}"'


def test_command_verifier_reports_all_checks_passed(tmp_path) -> None:
    verifier = CommandVerifier(
        tmp_path,
        [
            VerificationCheck("first", _python("print('ok')")),
            VerificationCheck("second", _python("raise SystemExit(0)")),
        ],
    )

    result = verifier.verify()

    assert result.passed is True
    assert result.success is True
    assert [item.check.name for item in result.checks] == ["first", "second"]
    assert result.checks[0].exit_code == 0
    assert result.checks[0].stdout.strip() == "ok"
    assert result.reason == "all verification checks passed"


def test_failed_check_stops_follow_up_checks(tmp_path) -> None:
    verifier = CommandVerifier(
        tmp_path,
        [
            VerificationCheck("failing", _python("print('bad'); raise SystemExit(3)")),
            VerificationCheck("unreachable", _python("raise SystemExit(0)")),
        ],
    )

    result = verifier.verify()

    assert result.passed is False
    assert len(result.checks) == 1
    assert result.checks[0].exit_code == 3
    assert result.reason == "verification check failed"


def test_timeout_is_structured_and_output_is_bounded(tmp_path) -> None:
    verifier = CommandVerifier(
        tmp_path,
        [
            VerificationCheck(
                "slow",
                _python(
                    "import sys,time; sys.stdout.write('x'*1000); sys.stdout.flush(); time.sleep(1)"
                ),
                0.05,
            )
        ],
        output_char_limit=80,
    )

    result = verifier.verify()

    check = result.checks[0]
    assert result.passed is False
    assert check.timed_out is True
    assert check.exit_code is not None
    assert len(check.stdout) <= 80
    assert "truncated" in check.stdout
    assert result.reason == "verification check timed out"


def test_run_and_verify_preserves_runtime_result_and_handles_verifier_error() -> None:
    runtime_result = SimpleNamespace(phase="completed", history=("sentinel",))

    class Runtime:
        def run(self, task: str):
            assert task == "fix it"
            return runtime_result

    class BrokenVerifier:
        def verify(self):
            raise RuntimeError("diagnostic details stay out of result")

    wrapped = run_and_verify(Runtime(), "fix it", BrokenVerifier())

    assert wrapped.run_result is runtime_result
    assert wrapped.passed is False
    assert wrapped.verification.reason == "verifier raised an exception"
    assert wrapped.verification.error == "RuntimeError"


def test_verification_result_defaults_are_immutable() -> None:
    result = VerificationResult(passed=False)
    assert result.checks == ()
    assert result.elapsed_seconds == 0.0
    assert result.success is False


def test_command_verifier_rejects_credential_like_environment_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="credential-like"):
        CommandVerifier(
            tmp_path,
            ["true"],
            environment={"SERVICE_API_KEY": "should-not-be-injected"},
        )
