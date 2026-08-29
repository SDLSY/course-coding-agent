"""Independent task verification for coding-agent runs.

The agent runtime answers a deliberately narrow question: *did the model
finish its conversation?*  A verifier answers the separate question: *does the
workspace satisfy the acceptance checks?*  Keeping these concerns separate is
important for benchmark results and for honest user-facing summaries.  A model
can return a confident final answer while tests still fail, and a failed model
request should not corrupt the runtime's protocol history merely because a
verifier is unavailable.

This module contains no model calls and no dependency on the runtime's control
loop.  ``CommandVerifier`` executes trusted, preconfigured commands (usually
``pytest`` or a project-specific checker) in a workspace.  Commands are run in
their own process group so a timeout also terminates ordinary descendants.
Output is bounded before it is returned to callers; an acceptance command must
not be able to consume unbounded memory by printing an accidental dump.
"""

from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from coding_agent.agent import RunResult
from coding_agent.tools.shell import is_sensitive_environment_name

DEFAULT_CHECK_TIMEOUT_SECONDS = 120.0
DEFAULT_OUTPUT_CHAR_LIMIT = 12_000
_TERMINATION_GRACE_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """One trusted acceptance command.

    ``name`` is a stable label used in benchmark output.  The command is
    intentionally a shell string because checks commonly use pipes or test
    runner flags; callers must treat it as configuration, never model input.
    """

    name: str
    command: str
    timeout_seconds: float = DEFAULT_CHECK_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("verification check name must be non-empty")
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("verification check command must be non-empty")
        timeout = self.timeout_seconds
        if (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("verification check timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class VerificationCheckResult:
    """Observed outcome of one :class:`VerificationCheck`."""

    check: VerificationCheck
    passed: bool
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    elapsed_seconds: float
    error: str | None = None

    @property
    def success(self) -> bool:
        """Alias for callers that use ``success`` for individual checks."""

        return self.passed


@dataclass(frozen=True, slots=True, init=False)
class VerificationResult:
    """Aggregate result returned by a :class:`Verifier`.

    ``passed`` is true only when every configured check exits with status zero.
    It is independent from ``RunResult.phase`` and therefore never changes the
    runtime's terminal semantics or conversation history.
    """

    success: bool
    checks: tuple[VerificationCheckResult, ...] = ()
    elapsed_seconds: float = 0.0
    reason: str = ""
    error: str | None = None

    def __init__(
        self,
        success: bool | None = None,
        checks: tuple[VerificationCheckResult, ...] = (),
        elapsed_seconds: float = 0.0,
        reason: str = "",
        error: str | None = None,
        *,
        passed: bool | None = None,
    ) -> None:
        """Construct a result using either ``success`` or ``passed``.

        ``success`` is the canonical stored field. ``passed`` remains accepted
        as a keyword for readability and backwards-compatible benchmark code;
        supplying both names with different values is rejected immediately.
        """

        if success is None and passed is None:
            raise TypeError("VerificationResult requires success or passed")
        if success is not None and not isinstance(success, bool):
            raise TypeError("success must be a bool")
        if passed is not None and not isinstance(passed, bool):
            raise TypeError("passed must be a bool")
        if success is not None and passed is not None and success != passed:
            raise ValueError("success and passed disagree")
        try:
            normalized_checks = tuple(checks)
        except TypeError as exc:
            raise TypeError("checks must be an iterable of check results") from exc
        if any(
            not isinstance(item, VerificationCheckResult) for item in normalized_checks
        ):
            raise TypeError("checks must contain VerificationCheckResult values")
        if (
            not isinstance(elapsed_seconds, (int, float))
            or isinstance(elapsed_seconds, bool)
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        if error is not None and not isinstance(error, str):
            raise TypeError("error must be a string or None")
        object.__setattr__(self, "success", success if success is not None else passed)
        object.__setattr__(self, "checks", normalized_checks)
        object.__setattr__(self, "elapsed_seconds", elapsed_seconds)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "error", error)

    @property
    def passed(self) -> bool:
        """Compatibility alias for code describing checks as pass/fail."""

        return self.success


class Verifier(Protocol):
    """Minimal synchronous acceptance contract."""

    def verify(self) -> VerificationResult:
        """Run configured checks without mutating agent history or budgets."""


class CommandVerifier:
    """Run a finite sequence of configured shell checks in ``workspace``."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        checks: Iterable[VerificationCheck | str],
        *,
        output_char_limit: int = DEFAULT_OUTPUT_CHAR_LIMIT,
        environment: Mapping[str, str] | None = None,
        excluded_environment_names: Iterable[str] = (),
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError("verification workspace must be an existing directory")
        normalised: list[VerificationCheck] = []
        names: set[str] = set()
        for index, item in enumerate(checks, start=1):
            if isinstance(item, str):
                item = VerificationCheck(name=f"check-{index}", command=item)
            if not isinstance(item, VerificationCheck):
                raise TypeError("checks must contain VerificationCheck or str values")
            if item.name in names:
                raise ValueError(f"duplicate verification check name: {item.name!r}")
            names.add(item.name)
            normalised.append(item)
        if not normalised:
            raise ValueError("at least one verification check is required")
        if (
            not isinstance(output_char_limit, int)
            or isinstance(output_char_limit, bool)
            or output_char_limit <= 0
        ):
            raise ValueError("output_char_limit must be a positive integer")
        self.checks = tuple(normalised)
        self.output_char_limit = output_char_limit
        # Start from the parent environment for normal tool discovery, but
        # remove conventional credential variables before checks can print
        # them. Explicit values are accepted for non-secret configuration.
        excluded = set(excluded_environment_names)
        if any(
            not isinstance(name, str) or not name or "=" in name or "\x00" in name
            for name in excluded
        ):
            raise ValueError("excluded environment names must be valid strings")
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in excluded and not is_sensitive_environment_name(key)
        }
        for key, value in (environment or {}).items():
            if not isinstance(key, str) or not key or "=" in key or "\x00" in key:
                raise ValueError("environment names must be non-empty strings")
            if key in excluded or is_sensitive_environment_name(key):
                raise ValueError(
                    "verification environment cannot contain credential-like names"
                )
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError("environment values must be strings without NUL")
            env[key] = value
        self.environment = env

    def verify(self) -> VerificationResult:
        started = time.monotonic()
        results: list[VerificationCheckResult] = []
        for check in self.checks:
            result = self._run_check(check)
            results.append(result)
            # Once a check fails, stop early. This makes a failing benchmark
            # deterministic and avoids running expensive follow-up checks.
            if not result.passed:
                break
        elapsed = time.monotonic() - started
        passed = (
            bool(results)
            and len(results) == len(self.checks)
            and all(item.passed for item in results)
        )
        if passed:
            reason = "all verification checks passed"
        elif any(item.timed_out for item in results):
            reason = "verification check timed out"
        else:
            reason = "verification check failed"
        return VerificationResult(
            passed=passed,
            checks=tuple(results),
            elapsed_seconds=elapsed,
            reason=reason,
        )

    def _run_check(self, check: VerificationCheck) -> VerificationCheckResult:
        started = time.monotonic()
        process: subprocess.Popen[bytes] | None = None
        stdout = b""
        stderr = b""
        timed_out = False
        error: str | None = None
        # File-backed streams prevent an acceptance command that prints a large
        # dump from consuming unbounded parent memory.  We read only bounded
        # head/tail slices after the process exits.
        with (
            tempfile.TemporaryFile(mode="w+b") as stdout_buffer,
            tempfile.TemporaryFile(mode="w+b") as stderr_buffer,
        ):
            try:
                process = subprocess.Popen(
                    check.command,
                    shell=True,
                    executable="/bin/sh",
                    cwd=self.workspace,
                    env=self.environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_buffer,
                    stderr=stderr_buffer,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=check.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_group(process)
            except OSError as exc:
                error = f"could not start verification command: {type(exc).__name__}"
            stdout = _read_bounded_stream(stdout_buffer, self.output_char_limit)
            stderr = _read_bounded_stream(stderr_buffer, self.output_char_limit)
        elapsed = time.monotonic() - started
        exit_code = process.returncode if process is not None else None
        passed = not timed_out and error is None and exit_code == 0
        return VerificationCheckResult(
            check=check,
            passed=passed,
            exit_code=exit_code,
            timed_out=timed_out,
            stdout=stdout,
            stderr=stderr,
            elapsed_seconds=elapsed,
            error=error,
        )


@dataclass(frozen=True, slots=True)
class VerifiedRun:
    """The unmodified runtime result paired with an independent check result."""

    run_result: RunResult
    verification: VerificationResult

    @property
    def passed(self) -> bool:
        """Convenience property for benchmark/reporting code."""

        return self.verification.passed


def run_and_verify(runtime: object, task: str, verifier: Verifier) -> VerifiedRun:
    """Run an agent, then perform independent acceptance checks.

    The verifier is invoked after ``runtime.run`` returns and receives no
    mutable runtime state.  A verifier implementation failure is represented
    as a failed verification result, while the original ``RunResult`` remains
    available for diagnosing model/runtime behaviour.
    """

    run_result = runtime.run(task)  # type: ignore[attr-defined]
    started = time.monotonic()
    try:
        verification = verifier.verify()
    except Exception as exc:  # noqa: BLE001 - verifier is an extension point
        verification = VerificationResult(
            passed=False,
            elapsed_seconds=time.monotonic() - started,
            reason="verifier raised an exception",
            error=type(exc).__name__,
        )
    return VerifiedRun(run_result=run_result, verification=verification)


def _read_bounded_stream(handle: object, limit: int) -> str:
    """Read at most ``limit`` bytes from a seekable temporary stream."""

    # TemporaryFile exposes the small seek/read interface used here. Keeping
    # the parameter structural avoids coupling this helper to a concrete IO
    # implementation and makes the bounded contract explicit.
    handle.flush()  # type: ignore[attr-defined]
    handle.seek(0, os.SEEK_END)  # type: ignore[attr-defined]
    size = handle.tell()  # type: ignore[attr-defined]
    handle.seek(0)  # type: ignore[attr-defined]
    if size <= limit:
        raw = handle.read()  # type: ignore[attr-defined]
    else:
        marker = b"\n... [verification output truncated] ...\n"
        if limit <= len(marker):
            raw = handle.read(limit)  # type: ignore[attr-defined]
        else:
            available = limit - len(marker)
            head_size = available // 2
            tail_size = available - head_size
            head = handle.read(head_size)  # type: ignore[attr-defined]
            handle.seek(-tail_size, os.SEEK_END)  # type: ignore[attr-defined]
            tail = handle.read(tail_size)  # type: ignore[attr-defined]
            raw = head + marker + tail
    return raw.decode("utf-8", errors="replace")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait()
