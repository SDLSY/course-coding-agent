"""Optional Harbor-style async bridge for the synchronous coding runtime.

Harbor is intentionally not imported here.  The adapter only describes the
small async contracts needed from a task environment and runs the existing
``AgentRuntime`` in a worker thread.  Remote tool calls made by that worker
are marshalled back to the Harbor event loop with ``run_coroutine_threadsafe``.
This keeps the protocol/state-machine implementation shared by local and
containerized runs while avoiding an optional framework dependency in core
imports.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import shlex
import tempfile
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from coding_agent.atif import export_atif, write_atif
from coding_agent.errors import ToolRequestError
from coding_agent.events import redact
from coding_agent.response_parser import parse_tool_arguments
from coding_agent.tools.shell import is_sensitive_environment_name
from coding_agent.types import ToolCall, ToolResult


@dataclass(frozen=True, slots=True)
class EnvironmentExecResult:
    """Provider-neutral result returned by an async environment command."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = 0
    timed_out: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("environment exec stdout/stderr must be strings")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise TypeError("environment exec exit_code must be an integer or null")
        if not isinstance(self.timed_out, bool):
            raise TypeError("environment exec timed_out must be a bool")


@runtime_checkable
class BaseEnvironment(Protocol):
    """Subset of Harbor's environment API required by this bridge."""

    async def exec(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> EnvironmentExecResult | Mapping[str, Any]:
        """Execute a command inside the isolated task workspace."""


@runtime_checkable
class BaseAgent(Protocol):
    """Async agent contract used by Harbor-like runners."""

    async def run(self, task: str) -> Any:
        """Run one task and return a runtime result."""


class RemoteCommandBackend:
    """ExecutionBackend that exposes one command tool in a remote environment.

    The command is model-provided and therefore still subject to the model's
    normal tool schema validation before it reaches this backend.  The backend
    itself performs a second small validation and always returns a paired
    ``ToolResult``.  No host environment mapping is sent to ``BaseEnvironment``
    and no API key is accepted by this class.
    """

    def __init__(
        self, environment: BaseEnvironment, *, workspace: str = "/app"
    ) -> None:
        if not workspace or not workspace.startswith("/"):
            raise ValueError("remote workspace must be an absolute path")
        self.environment = environment
        self.workspace = workspace.rstrip("/") or "/"
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def model_schemas(self) -> Sequence[Mapping[str, Any]]:
        return (
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Run a command in the remote /app workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "minLength": 1},
                            "timeout_seconds": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 300,
                            },
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def execute_call(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        if call.name != "run_command":
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"Unknown remote tool: {call.name!r}.",
                error_code="unknown_tool",
            )
        try:
            # Keep the remote boundary on the same strict JSON contract as the
            # local registry. Duplicate keys and non-standard numeric values
            # must not change meaning when a tool moves into a container.
            arguments = parse_tool_arguments(call)
        except ToolRequestError as exc:
            return self._error(
                call,
                getattr(exc, "error_code", "invalid_tool_request"),
                str(exc),
            )
        if set(arguments) - {"command", "timeout_seconds"}:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content="run_command received an unknown argument.",
                error_code="invalid_arguments",
            )
        if (
            not isinstance(arguments.get("command"), str)
            or not arguments["command"].strip()
        ):
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content="run_command requires a non-empty command string.",
                error_code="invalid_arguments",
            )

        requested = arguments.get("timeout_seconds")
        if requested is not None and (
            not isinstance(requested, int)
            or isinstance(requested, bool)
            or requested < 1
            or requested > 300
        ):
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content="timeout_seconds must be an integer between 1 and 300.",
                error_code="invalid_arguments",
            )

        # ``workspace`` is configuration, but it still crosses a shell
        # boundary because Harbor's environment API accepts one command
        # string.  JSON quoting is not shell quoting: a path containing ``$``
        # or backticks would otherwise be expanded by ``/bin/sh`` before the
        # remote command runs.  ``shlex.quote`` emits one literal POSIX shell
        # word for every valid workspace path.
        command = f"cd {shlex.quote(self.workspace)} && {arguments['command']}"
        effective_timeout = (
            min(timeout_seconds, 300.0) if timeout_seconds is not None else None
        )
        if isinstance(requested, (int, float)) and not isinstance(requested, bool):
            effective_timeout = (
                float(requested)
                if effective_timeout is None
                else min(float(requested), effective_timeout)
            )
        if self._loop is None:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content="Remote backend is not bound to an async event loop.",
                error_code="backend_not_bound",
            )
        future: Any | None = None
        try:
            future = asyncio.run_coroutine_threadsafe(
                _environment_exec(self.environment, command, effective_timeout),
                self._loop,
            )
            raw = future.result(timeout=effective_timeout)
            result = _normalise_exec_result(raw)
        except Exception as exc:  # noqa: BLE001 - remote boundary
            # A timeout/error must not leave an orphaned coroutine continuing
            # to mutate the remote environment after the model has moved on.
            if future is not None:
                future.cancel()
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content="Remote command execution failed.",
                metadata={"error_type": type(exc).__name__},
                error_code="remote_execution_error",
            )
        content = (
            f"exit_code: {result.exit_code}\n"
            f"timed_out: {result.timed_out}\n"
            f"stdout:\n{result.stdout or '<empty>'}\n"
            f"stderr:\n{result.stderr or '<empty>'}"
        )
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=True,
            content=content,
            metadata={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
            },
        )


class RemoteExecutionBackend(RemoteCommandBackend):
    """Remote counterpart of the six built-in local filesystem tools.

    Every operation is translated to one command executed below ``/app``.
    Paths are checked lexically before quoting, so model input cannot escape
    the mounted workspace even when the remote shell has broad permissions.
    """

    _NAMES = (
        "list_files",
        "search_text",
        "read_file",
        "write_file",
        "replace_in_file",
        "run_command",
    )

    def model_schemas(self) -> Sequence[Mapping[str, Any]]:
        path = {"type": "string", "minLength": 1}
        return tuple(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Remote workspace operation: {name}.",
                    "parameters": params,
                },
            }
            for name, params in (
                (
                    "list_files",
                    {
                        "type": "object",
                        "properties": {"path": path},
                        "additionalProperties": False,
                    },
                ),
                (
                    "search_text",
                    {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "minLength": 1},
                            "path": path,
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                ),
                (
                    "read_file",
                    {
                        "type": "object",
                        "properties": {
                            "path": path,
                            "start_line": {"type": "integer", "minimum": 1},
                            "end_line": {"type": "integer", "minimum": 1},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
                (
                    "write_file",
                    {
                        "type": "object",
                        "properties": {"path": path, "content": {"type": "string"}},
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                ),
                (
                    "replace_in_file",
                    {
                        "type": "object",
                        "properties": {
                            "path": path,
                            "old": {"type": "string", "minLength": 1},
                            "new": {"type": "string"},
                        },
                        "required": ["path", "old", "new"],
                        "additionalProperties": False,
                    },
                ),
                (
                    "run_command",
                    RemoteCommandBackend.model_schemas(self)[0]["function"][
                        "parameters"
                    ],
                ),
            )
        )

    def execute_call(
        self, call: ToolCall, *, timeout_seconds: float | None = None
    ) -> ToolResult:
        if call.name == "run_command":
            return super().execute_call(call, timeout_seconds=timeout_seconds)
        try:
            args = parse_tool_arguments(call)
        except ToolRequestError as exc:
            return self._error(
                call,
                getattr(exc, "error_code", "invalid_tool_request"),
                str(exc),
            )
        try:
            self._validate_arguments(call.name, args)
            command = self._command_for(call.name, args)
        except ValueError as exc:
            return self._error(call, "invalid_arguments", str(exc))
        return self._execute_remote(
            call,
            command,
            timeout_seconds,
            None,
            success_exit_codes=(0, 1) if call.name == "search_text" else (0,),
        )

    def _command_for(self, name: str, args: Mapping[str, Any]) -> str:
        path = self._remote_path(args.get("path", "."))
        if name == "list_files":
            # Match the local list_files contract: recurse through the tree,
            # while find's default behaviour still avoids following directory
            # symlinks. The root itself is omitted by -mindepth 1.
            return f"find {shlex.quote(path)} -mindepth 1 -print"
        if name == "search_text":
            query = args.get("query")
            if not isinstance(query, str) or not query:
                raise ValueError("search_text requires a non-empty query")
            return f"grep -R -n -F -- {shlex.quote(query)} {shlex.quote(path)}"
        if name == "read_file":
            if "start_line" in args or "end_line" in args:
                start = args.get("start_line", 1)
                end = args.get("end_line", 400)
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or start < 1
                    or end < start
                ):
                    raise ValueError("read_file line range is invalid")
                return f"sed -n '{start},{end}p' -- {shlex.quote(path)}"
            return f"cat -- {shlex.quote(path)}"
        if name == "write_file":
            content = args.get("content")
            if not isinstance(content, str):
                raise ValueError("write_file requires string content")
            encoded = base64.b64encode(content.encode()).decode("ascii")
            script = (
                "import base64,os,pathlib,sys,tempfile; "
                "p=pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True); "
                "fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name+'.'); "
                "f=os.fdopen(fd,'wb'); f.write(base64.b64decode(sys.argv[2])); "
                "f.flush(); os.fsync(f.fileno()); f.close(); os.replace(tmp,p)"
            )
            return (
                "python3 -c "
                + shlex.quote(script)
                + " "
                + " ".join(shlex.quote(item) for item in (path, encoded))
            )
        if name == "replace_in_file":
            old, new = args.get("old"), args.get("new")
            if not isinstance(old, str) or not old:
                raise ValueError("replace_in_file requires non-empty old text")
            if not isinstance(new, str):
                raise ValueError("replace_in_file requires string new text")
            script = (
                "import pathlib,base64,os,sys,tempfile; "
                "p=pathlib.Path(sys.argv[1]); s=p.read_text(encoding='utf-8'); "
                "old=base64.b64decode(sys.argv[2]).decode('utf-8'); "
                "new=base64.b64decode(sys.argv[3]).decode('utf-8'); "
                "n=s.count(old); "
                "raise SystemExit(f'expected one match, got {n}') if n!=1 else None; "
                "fd,tmp=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name+'.',text=True); "
                "f=os.fdopen(fd,'w',encoding='utf-8',newline=''); f.write(s.replace(old,new,1)); "
                "f.flush(); os.fsync(f.fileno()); f.close(); os.replace(tmp,p)"
            )
            return (
                "python3 -c "
                + shlex.quote(script)
                + " "
                + " ".join(
                    shlex.quote(item)
                    for item in (
                        path,
                        base64.b64encode(old.encode()).decode(),
                        base64.b64encode(new.encode()).decode(),
                    )
                )
            )
        raise ValueError(f"Unknown remote tool: {name!r}.")

    def _remote_path(self, value: Any) -> str:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("path must be a non-empty relative string")
        if value.startswith("/") or ".." in value.split("/"):
            raise ValueError("absolute and parent-directory paths are not allowed")
        return (
            self.workspace + "/" + value.strip("/") if value != "." else self.workspace
        )

    @staticmethod
    def _validate_arguments(name: str, args: Mapping[str, Any]) -> None:
        """Apply the local six-tool argument contract at the remote boundary.

        Harbor bypasses the local ``ToolRegistry`` because handlers run in a
        different process. Repeating this small validation prevents malformed
        model JSON from becoming an arbitrary shell operation or an accidental
        write with a missing field.
        """

        contracts: dict[str, tuple[set[str], set[str]]] = {
            "list_files": ({"path"}, set()),
            "search_text": ({"query", "path"}, {"query"}),
            "read_file": ({"path", "start_line", "end_line"}, {"path"}),
            "write_file": ({"path", "content"}, {"path", "content"}),
            "replace_in_file": (
                {"path", "old", "new"},
                {"path", "old", "new"},
            ),
        }
        contract = contracts.get(name)
        if contract is None:
            raise ValueError(f"Unknown remote tool: {name!r}.")
        allowed, required = contract
        unknown = set(args) - allowed
        if unknown:
            raise ValueError(f"unknown argument: {min(unknown)}")
        missing = required - set(args)
        if missing:
            raise ValueError(f"missing argument: {min(missing)}")
        if "path" in args and (not isinstance(args["path"], str) or not args["path"]):
            raise ValueError("path must be a non-empty string")
        if name == "search_text" and (
            not isinstance(args.get("query"), str) or not args["query"]
        ):
            raise ValueError("search_text requires a non-empty query")
        if name == "read_file":
            start = args.get("start_line", 1)
            end = args.get("end_line")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or start < 1
                or (
                    end is not None
                    and (
                        not isinstance(end, int) or isinstance(end, bool) or end < start
                    )
                )
            ):
                raise ValueError("read_file line range is invalid")
        if name == "write_file" and not isinstance(args.get("content"), str):
            raise ValueError("write_file requires string content")
        if name == "replace_in_file":
            if not isinstance(args.get("old"), str) or not args["old"]:
                raise ValueError("replace_in_file requires non-empty old text")
            if not isinstance(args.get("new"), str):
                raise ValueError("replace_in_file requires string new text")

    def _execute_remote(
        self,
        call: ToolCall,
        command: str,
        timeout: float | None,
        requested: Any,
        *,
        success_exit_codes: Sequence[int] = (0,),
    ) -> ToolResult:
        if self._loop is None:
            return self._error(
                call,
                "backend_not_bound",
                "Remote backend is not bound to an async event loop.",
            )
        effective = min(timeout, 300.0) if timeout is not None else None
        if isinstance(requested, (int, float)) and not isinstance(requested, bool):
            effective = (
                float(requested)
                if effective is None
                else min(float(requested), effective)
            )
        future: Any | None = None
        try:
            future = asyncio.run_coroutine_threadsafe(
                _environment_exec(self.environment, command, effective), self._loop
            )
            raw = future.result(timeout=effective)
            result = _normalise_exec_result(raw)
        except Exception as exc:  # noqa: BLE001
            if future is not None:
                future.cancel()
            return self._error(
                call,
                "remote_execution_error",
                "Remote command execution failed.",
                {"error_type": type(exc).__name__},
            )
        stdout = _bounded(result.stdout, 32_000)
        stderr = _bounded(result.stderr, 32_000)
        succeeded = (
            not result.timed_out
            and result.exit_code in success_exit_codes
            and result.exit_code is not None
        )
        no_matches = call.name == "search_text" and result.exit_code == 1
        content = f"exit_code: {result.exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        if no_matches and not result.timed_out:
            content = "No matches found.\n" + content
        return ToolResult(
            call_id=call.id,
            name=call.name,
            # Filesystem operations report a failed remote command as a tool
            # error (matching local tool semantics); ``run_command`` keeps
            # non-zero exits as observations in its parent implementation.
            ok=succeeded,
            content=content,
            metadata={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "truncated": stdout != result.stdout or stderr != result.stderr,
                "no_matches": no_matches,
            },
            error_code=None if succeeded else "remote_command_failed",
        )

    @staticmethod
    def _error(
        call: ToolCall,
        code: str,
        content: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=False,
            content=content,
            error_code=code,
            metadata=metadata or {},
        )


class HarborAgentAdapter:
    """Run an existing AgentRuntime under an async Harbor-style harness."""

    def __init__(
        self,
        runtime: Any,
        *,
        environment: BaseEnvironment | None = None,
        backend: RemoteCommandBackend | None = None,
        trace_hook: Callable[[Any], Awaitable[None] | None] | None = None,
        atif_hook: Callable[[Any], Awaitable[None] | None] | None = None,
        artifact_dir: str | os.PathLike[str] | None = None,
        model_name: str | None = None,
        run_id: str | None = None,
        tool_definitions: Sequence[Mapping[str, Any]] = (),
        secrets: Sequence[str] = (),
        verification: Any | None = None,
        events: Sequence[Mapping[str, Any]] = (),
        events_provider: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        artifact_metadata: Mapping[str, Any] | None = None,
        cancel_event: threading.Event | None = None,
        artifact_error_hook: Callable[[Exception], Awaitable[None] | None]
        | None = None,
    ) -> None:
        self.runtime = runtime
        self.backend = backend or (
            RemoteExecutionBackend(environment) if environment is not None else None
        )
        self.trace_hook = trace_hook
        self.atif_hook = atif_hook
        self.artifact_dir = (
            Path(artifact_dir).expanduser().resolve(strict=False)
            if artifact_dir is not None
            else None
        )
        self.model_name = model_name
        self.run_id = run_id
        self.tool_definitions = tuple(tool_definitions)
        self.secrets = tuple(secrets)
        self.verification = verification
        self.events = tuple(events)
        self.events_provider = events_provider
        self.artifact_metadata = dict(artifact_metadata or {})
        self.cancel_event = cancel_event
        self.artifact_error_hook = artifact_error_hook

    async def run(self, task: str) -> Any:
        """Execute the synchronous runtime without blocking Harbor's loop."""

        loop = asyncio.get_running_loop()
        if self.backend is not None:
            self.backend.bind_loop(loop)
            # Keep the runtime's injected registry and remote backend aligned.
            # This assignment is intentionally opt-in so ordinary local runs
            # remain untouched.
            self.runtime.tool_registry = self.backend
        # Keep the worker as a Task so an outer Harbor cancellation can signal
        # the Runtime without abandoning a still-running thread object.
        worker = asyncio.create_task(asyncio.to_thread(self.runtime.run, task))
        try:
            result = await worker
        except asyncio.CancelledError:
            if self.cancel_event is not None:
                self.cancel_event.set()
            # The Runtime checks its cancellation callback at protocol
            # boundaries.  Do not block Harbor's cancellation path waiting for
            # an arbitrary provider request; the worker is left to finish under
            # its own finite timeout and cannot start a new batch after the
            # signal is observed.
            raise
        if self.artifact_dir is not None:
            try:
                artifact_events = self.events
                if self.events_provider is not None:
                    artifact_events = tuple(self.events_provider())
                write_harbor_artifacts(
                    result,
                    log_dir=self.artifact_dir,
                    task=task,
                    model_name=self.model_name,
                    run_id=self.run_id,
                    tool_definitions=self.tool_definitions,
                    verification=self.verification,
                    events=artifact_events,
                    secrets=self.secrets,
                    metadata=self.artifact_metadata,
                )
            except Exception as exc:  # noqa: BLE001 - optional diagnostics
                # Artifact persistence must never change the Runtime's terminal
                # result or cause a remote task to be retried.  Integrations that
                # need a hard failure can provide an explicit error hook.
                await _invoke_hook(self.artifact_error_hook, exc)
        await _invoke_hook(self.trace_hook, result)
        await _invoke_hook(self.atif_hook, result)
        return result


def write_harbor_artifacts(
    run: Any,
    *,
    log_dir: str | os.PathLike[str],
    task: str,
    model_name: str | None = None,
    run_id: str | None = None,
    tool_definitions: Sequence[Mapping[str, Any]] = (),
    verification: Any | None = None,
    events: Sequence[Mapping[str, Any]] = (),
    secrets: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write Harbor-friendly trajectory and summary files atomically.

    Harbor installations differ in the exact log-directory object they pass to
    an agent.  This helper only requires a filesystem path and emits the two
    conventional files ``trajectory.json`` (ATIF-v1.7) and ``run.json``.  The
    model credential is never accepted as an argument; callers may provide it
    in ``secrets`` solely so accidental custom metadata is redacted.
    """

    directory = Path(log_dir).expanduser().resolve(strict=False)
    directory.mkdir(parents=True, exist_ok=True)
    document = export_atif(
        run,
        task=task,
        model_name=model_name,
        run_id=run_id,
        tool_definitions=tool_definitions,
        verification=verification,
        events=events,
        secrets=secrets,
    )
    trajectory_path = write_atif(
        directory / "trajectory.json", document, secrets=secrets
    )
    summary = redact(
        {
            "run_id": run_id,
            "phase": getattr(getattr(run, "phase", None), "value", None),
            "reason": getattr(run, "reason", None),
            "model_turns": getattr(run, "model_turns", None),
            "model_requests": getattr(run, "model_requests", None),
            "tool_calls": getattr(run, "tool_calls", None),
            "elapsed_seconds": getattr(run, "elapsed_seconds", None),
            "input_tokens": _usage_field(getattr(run, "usage", None), "prompt_tokens"),
            "output_tokens": _usage_field(
                getattr(run, "usage", None), "completion_tokens"
            ),
            "cache_tokens": _usage_field(getattr(run, "usage", None), "cached_tokens"),
            "total_tokens": _usage_field(getattr(run, "usage", None), "total_tokens"),
            "verification": _json_safe_value(verification),
            **dict(metadata or {}),
        },
        secrets=secrets,
    )
    summary_path = _write_json_atomically(directory / "run.json", summary)
    return {"trajectory": trajectory_path, "summary": summary_path}


def _usage_field(usage: Any, name: str) -> int | float | None:
    if usage is None:
        return None
    value = getattr(usage, name, None)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(usage, Mapping):
        value = usage.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _write_json_atomically(path: Path, value: Any) -> Path:
    """Write a small diagnostic JSON object with owner-only permissions."""

    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


# Descriptive aliases make the integration boundary discoverable without
# forcing callers to depend on a Harbor package's concrete class names.
HarborExecutionBackend = RemoteExecutionBackend
HarborEnvironment = BaseEnvironment


def sanitized_host_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return non-secret environment values suitable for a task container."""

    source = os.environ if environment is None else environment
    return {
        key: value
        for key, value in source.items()
        if not is_sensitive_environment_name(key)
    }


def _normalise_exec_result(raw: Any) -> EnvironmentExecResult:
    if isinstance(raw, EnvironmentExecResult):
        return raw

    if isinstance(raw, Mapping):
        get = raw.get
    else:
        # Harbor's native ``ExecResult`` is an object with attributes rather
        # than a mapping.  Accept both spellings used by compatible wrappers.
        if raw is None:
            raise TypeError("environment exec must return a result object")

        def get(name: str, default: Any = None) -> Any:
            return getattr(raw, name, default)

    stdout = get("stdout", "")
    stderr = get("stderr", "")
    if stdout is None:
        stdout = ""
    if stderr is None:
        stderr = ""
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if isinstance(raw, Mapping):
        exit_code = raw.get(
            "exit_code",
            raw.get("return_code", raw.get("returncode", 0)),
        )
        timed_out = raw.get("timed_out", False)
    else:
        exit_code = getattr(raw, "exit_code", None)
        if exit_code is None:
            exit_code = getattr(raw, "return_code", None)
        if exit_code is None:
            exit_code = getattr(raw, "returncode", 0)
        timed_out = getattr(raw, "timed_out", False)
    return EnvironmentExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
    )


def _environment_exec(
    environment: BaseEnvironment,
    command: str,
    timeout_seconds: float | None,
) -> Awaitable[EnvironmentExecResult | Mapping[str, Any]]:
    """Construct one environment coroutine, accepting common timeout names.

    The fallback occurs while constructing the coroutine, before it is
    scheduled, so a signature mismatch cannot execute a command twice.
    """

    # Harbor 0.22 uses ``timeout_sec``; a few compatible environments use
    # ``timeout`` or the more descriptive ``timeout_seconds``.  Select the
    # supported spelling from the callable signature before invoking it.  A
    # retry based on catching ``TypeError`` is unsafe for synchronous wrappers:
    # such a wrapper may execute the command and then raise ``TypeError`` from
    # its result handling, causing a second invocation with another alias.
    method = environment.exec
    keyword: str | None = None
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        # Dynamic/C-extension callables do not expose a signature.  Omitting
        # the optional keyword is the only fail-safe choice: the worker-side
        # future still enforces the same upper bound, and no speculative retry
        # can duplicate a mutating command.
        signature = None
    if signature is not None:
        parameters = signature.parameters.values()
        if any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters
        ):
            keyword = "timeout_seconds"
        else:
            for candidate in ("timeout_seconds", "timeout", "timeout_sec"):
                parameter = signature.parameters.get(candidate)
                if (
                    parameter is not None
                    and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
                ):
                    keyword = candidate
                    break

    kwargs = {keyword: timeout_seconds} if keyword is not None else {}
    result = method(command, **kwargs)  # type: ignore[call-arg]
    if inspect.isawaitable(result):
        return result

    async def _completed(value: Any = result) -> Any:
        return value

    return _completed()


def _json_safe_value(value: Any) -> Any:
    """Project optional artifact metadata into JSON-compatible primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_safe_value(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return _json_safe_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return f"<{type(value).__name__}>"


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = "\n... [remote output truncated] ...\n"
    return text[: max(0, limit - len(marker))] + marker


async def _invoke_hook(
    hook: Callable[[Any], Awaitable[None] | None] | None,
    value: Any,
) -> None:
    if hook is None:
        return
    result = hook(value)
    if inspect.isawaitable(result):
        await result


__all__ = [
    "BaseAgent",
    "BaseEnvironment",
    "EnvironmentExecResult",
    "HarborAgentAdapter",
    "HarborEnvironment",
    "HarborExecutionBackend",
    "RemoteCommandBackend",
    "RemoteExecutionBackend",
    "sanitized_host_environment",
    "write_harbor_artifacts",
]
