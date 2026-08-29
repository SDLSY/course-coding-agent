"""Bounded POSIX command execution for a trusted local workspace.

Setting ``cwd`` to the workspace is an ergonomic default, not confinement.  A
shell command can still use absolute paths, access the network, and execute any
program available to the current user.  Real isolation would require a
container, namespace, seccomp policy, or another operating-system sandbox,
which is explicitly outside this project's first version.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping
from typing import Any

from .base import Tool, ToolExecutionError, ToolOutput, ToolRequestError
from .filesystem import WorkspacePathResolver

DEFAULT_COMMAND_TIMEOUT_SECONDS = 30
MAX_COMMAND_TIMEOUT_SECONDS = 300
DEFAULT_STREAM_BYTE_LIMIT = 32_000
PROCESS_TERMINATION_GRACE_SECONDS = 0.5
_STREAM_TRUNCATION_MARKER = b"\n... [command stream truncated] ...\n"

# Names are compared case-insensitively because Windows-style or
# provider-specific casing sometimes appears even in POSIX environments.  The
# pattern intentionally errs toward withholding credentials.  The trailing
# ``(?:$|_)`` also catches compound names such as ``AWS_SECRET_ACCESS_KEY`` and
# ``PRIVATE_KEY_PATH``; a bare substring check would incorrectly classify names
# such as ``TOKENIZERS_PARALLELISM``.  It cannot infer every secret stored under
# an arbitrary name, so event logs must independently avoid dumping the entire
# environment.
_SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?:^|_)(?:API[_-]?KEY|AUTHORIZATION|ACCESS[_-]?TOKEN|"
    r"REFRESH[_-]?TOKEN|AUTH[_-]?TOKEN|TOKEN|SECRET(?:[_-]ACCESS[_-]?KEY)?|"
    r"PASSWORD|PASSWD|PRIVATE[_-]?KEY|CREDENTIALS?|"
    r"ACCESS[_-]?KEY(?:[_-]?ID)?|CLIENT[_-]?SECRET|SSH[_-]?KEY)"
    r"(?:$|_)",
    flags=re.IGNORECASE,
)


class ShellTools:
    """Implement ``run_command`` with timeout, capture, and environment policy."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        extra_env: Mapping[str, str] | None = None,
        excluded_env_names: Iterable[str] = (),
        stream_byte_limit: int = DEFAULT_STREAM_BYTE_LIMIT,
        termination_grace_seconds: float = PROCESS_TERMINATION_GRACE_SECONDS,
    ) -> None:
        if os.name != "posix":
            raise ValueError("run_command currently supports POSIX systems only")
        self.workspace = WorkspacePathResolver(workspace).root
        if (
            isinstance(stream_byte_limit, bool)
            or not isinstance(stream_byte_limit, int)
            or stream_byte_limit <= len(_STREAM_TRUNCATION_MARKER)
        ):
            raise ValueError("stream_byte_limit must exceed the truncation marker")
        if termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be positive")

        explicit: dict[str, str] = {}
        for name, value in (extra_env or {}).items():
            _validate_environment_name(name, label="explicit environment")
            if not isinstance(value, str) or "\x00" in value:
                raise ValueError(
                    "explicit environment values must be strings without NUL"
                )
            explicit[name] = value

        # Name-based exclusions close a gap that pattern matching cannot: the
        # CLI permits users to keep a provider credential in an arbitrarily
        # named variable.  Its exact name is known at assembly time and must be
        # withheld even when it does not end in API_KEY/TOKEN/SECRET.
        excluded: set[str] = set()
        for name in excluded_env_names:
            _validate_environment_name(name, label="excluded environment")
            excluded.add(name)
        overlap = excluded.intersection(explicit)
        if overlap:
            conflict = min(overlap)
            raise ValueError(
                f"environment variable {conflict!r} cannot be both explicit and excluded"
            )

        self.extra_env = explicit
        self.excluded_env_names = frozenset(excluded)
        self.stream_byte_limit = stream_byte_limit
        self.termination_grace_seconds = float(termination_grace_seconds)

    def definition(self) -> Tool:
        return Tool(
            name="run_command",
            description=(
                "Run a non-interactive POSIX shell command with the workspace as "
                "cwd. Returns stdout, stderr, exit code, duration, and timeout state."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "minLength": 1},
                    "timeout_seconds": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_COMMAND_TIMEOUT_SECONDS,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
            handler=self.run_command,
            # Shell commands may change any host data reachable by the user.
            # This flag is descriptive for UI/policy layers, not enforcement.
            modifies_workspace=True,
            timeout_argument="timeout_seconds",
        )

    def run_command(self, arguments: Mapping[str, Any]) -> ToolOutput:
        command = arguments["command"]
        if not command.strip():
            raise ToolRequestError(
                "command must contain a non-whitespace shell expression.",
                error_code="invalid_command",
            )
        timeout_seconds = arguments.get(
            "timeout_seconds", DEFAULT_COMMAND_TIMEOUT_SECONDS
        )

        environment = build_sanitized_environment(
            self.extra_env,
            excluded_names=self.excluded_env_names,
        )
        # PWD inherited from the parent may disagree with cwd.  Some build tools
        # inspect it directly, so keep this non-secret value coherent.
        environment["PWD"] = str(self.workspace)

        started = time.monotonic()
        timed_out = False
        process: subprocess.Popen[bytes] | None = None

        # Temporary files avoid ``communicate()`` buffering unbounded child
        # output in Python memory.  The command may print gigabytes, but the
        # parent retains only file-backed data and later reads bounded head/tail
        # segments for the model context.
        with (
            tempfile.TemporaryFile(mode="w+b") as stdout_buffer,
            tempfile.TemporaryFile(mode="w+b") as stderr_buffer,
        ):
            try:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    executable="/bin/sh",
                    cwd=self.workspace,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_buffer,
                    stderr=stderr_buffer,
                    # A new session creates a new process group whose ID is the
                    # shell PID.  On timeout we can signal the entire pipeline
                    # and its ordinary descendants rather than only /bin/sh.
                    start_new_session=True,
                )
            except OSError as exc:
                raise ToolExecutionError(
                    "Could not start the POSIX shell command.",
                    error_code="command_start_failed",
                ) from exc

            try:
                process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate_process_group(process)

            duration_seconds = time.monotonic() - started
            stdout_data = _read_bounded_stream(stdout_buffer, self.stream_byte_limit)
            stderr_data = _read_bounded_stream(stderr_buffer, self.stream_byte_limit)

        stdout_text = stdout_data.text
        stderr_text = stderr_data.text

        # Explicitly reintroducing a sensitive variable is allowed only because
        # the user chose it through configuration.  If the command echoes that
        # value, redact it from observations so an event trace or subsequent
        # model call does not become an accidental credential sink.  Ordinary
        # non-sensitive explicit values remain observable for debugging.
        sensitive_values = {
            value
            for name, value in self.extra_env.items()
            if is_sensitive_environment_name(name) and len(value) >= 4
        }
        for value in sorted(sensitive_values, key=len, reverse=True):
            stdout_text = stdout_text.replace(value, "[REDACTED]")
            stderr_text = stderr_text.replace(value, "[REDACTED]")

        status_note = (
            f"Command timed out after {timeout_seconds} seconds."
            if timed_out
            else f"Command exited with status {process.returncode}."
        )
        content = (
            f"{status_note}\n"
            f"stdout:\n{stdout_text if stdout_text else '<empty>'}\n"
            f"stderr:\n{stderr_text if stderr_text else '<empty>'}"
        )
        return ToolOutput(
            content=content,
            metadata={
                "exit_code": process.returncode,
                "timed_out": timed_out,
                "timeout_seconds": timeout_seconds,
                "duration_seconds": round(duration_seconds, 6),
                "stdout_original_bytes": stdout_data.original_bytes,
                "stdout_returned_bytes": stdout_data.returned_bytes,
                "stdout_truncated": stdout_data.truncated,
                "stderr_original_bytes": stderr_data.original_bytes,
                "stderr_returned_bytes": stderr_data.returned_bytes,
                "stderr_truncated": stderr_data.truncated,
                # Names help diagnose configuration without placing their values
                # in a ToolResult or eventual JSONL event.
                "explicit_environment_names": sorted(self.extra_env),
            },
        )

    def _terminate_process_group(self, process: subprocess.Popen[bytes]) -> None:
        """Terminate the timed-out shell's process group, escalating to KILL.

        Descendants can deliberately create a different session/process group
        and escape this mechanism; that is another reason timeout handling is
        not a security sandbox.  Ordinary pipelines and background children
        inherit the shell's group and are covered.
        """

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise ToolExecutionError(
                "Timed-out command could not be terminated.",
                error_code="command_termination_failed",
            ) from exc

        try:
            process.wait(timeout=self.termination_grace_seconds)
        except subprocess.TimeoutExpired:
            pass

        # Escalate even when the shell itself exited after TERM: a child may
        # still hold the process group alive.  killpg then either removes those
        # stragglers or harmlessly reports that the group no longer exists.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise ToolExecutionError(
                "Timed-out command process group could not be killed.",
                error_code="command_termination_failed",
            ) from exc

        try:
            process.wait(timeout=self.termination_grace_seconds)
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(
                "Timed-out shell did not report termination.",
                error_code="command_termination_failed",
            ) from exc


class _CapturedStream:
    """Internal immutable record for one bounded stdout/stderr stream."""

    __slots__ = ("original_bytes", "returned_bytes", "text", "truncated")

    def __init__(
        self,
        *,
        text: str,
        original_bytes: int,
        returned_bytes: int,
        truncated: bool,
    ) -> None:
        self.text = text
        self.original_bytes = original_bytes
        self.returned_bytes = returned_bytes
        self.truncated = truncated


def _read_bounded_stream(handle: Any, limit: int) -> _CapturedStream:
    """Read at most ``limit`` bytes, preserving both beginning and end."""

    handle.flush()
    handle.seek(0, os.SEEK_END)
    original_bytes = handle.tell()
    handle.seek(0)

    if original_bytes <= limit:
        raw = handle.read()
        truncated = False
    else:
        available = limit - len(_STREAM_TRUNCATION_MARKER)
        head_size = available // 2
        tail_size = available - head_size
        head = handle.read(head_size)
        handle.seek(-tail_size, os.SEEK_END)
        tail = handle.read(tail_size)
        raw = head + _STREAM_TRUNCATION_MARKER + tail
        truncated = True

    return _CapturedStream(
        # Commands can legitimately emit arbitrary bytes.  Replacement decoding
        # is preferable to declaring an otherwise useful command result failed.
        text=raw.decode("utf-8", errors="replace"),
        original_bytes=original_bytes,
        returned_bytes=len(raw),
        truncated=truncated,
    )


def is_sensitive_environment_name(name: str) -> bool:
    """Return whether a parent variable must be withheld from child commands."""

    upper = name.upper()
    return bool(
        _SENSITIVE_ENVIRONMENT_NAME.search(upper)
        or upper.startswith("BASH_FUNC_")
        or upper
        in {
            "AUTHORIZATION",
            "HTTP_AUTHORIZATION",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "ZHIPUAI_API_KEY",
            "GLM_API_KEY",
        }
    )


def _validate_environment_name(name: object, *, label: str) -> None:
    """Validate a name before it participates in child-process policy.

    POSIX itself permits unusual environment names, but ``subprocess`` rejects
    NUL and ``=`` and an empty name is never useful here.  Applying one check to
    both inclusion and exclusion avoids a configuration which appears to
    protect a credential but can never match the actual process variable.
    """

    if not isinstance(name, str) or not name or "=" in name or "\x00" in name:
        raise ValueError(f"{label} names must be valid strings")


def build_sanitized_environment(
    explicit: Mapping[str, str] | None = None,
    *,
    excluded_names: Iterable[str] = (),
) -> dict[str, str]:
    """Copy benign parent variables, then apply explicit user configuration.

    Filtering is based on names rather than values so no credential value needs
    to be inspected, copied into diagnostics, or logged.  ``excluded_names``
    carries credentials whose arbitrary variable names are known by the CLI;
    exclusion wins over both inheritance and ``explicit``.  Other explicit
    variables are applied last because the CLI may deliberately provide
    project-specific configuration.  The caller must never serialize this
    returned mapping.
    """

    excluded: set[str] = set()
    for name in excluded_names:
        _validate_environment_name(name, label="excluded environment")
        excluded.add(name)
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in excluded and not is_sensitive_environment_name(name)
    }
    environment.update(
        {
            name: value
            for name, value in (explicit or {}).items()
            if name not in excluded
        }
    )
    return environment


def build_shell_tool(
    workspace: str | os.PathLike[str],
    **options: Any,
) -> Tool:
    """Convenience factory used when assembling the default registry."""

    return ShellTools(workspace, **options).definition()
