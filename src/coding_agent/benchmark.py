"""A small, bounded harness for comparing coding agents.

The benchmark layer deliberately sits outside :mod:`coding_agent.agent`.  An
agent is responsible for deciding how to edit a workspace; this module only
provides the experimental envelope around that decision: fixed task fixtures,
fresh workspaces, explicit resource limits, independent acceptance checks, and
one JSON result shape for every run.

The command line defaults to a *plan* (dry run).  Starting an arbitrary agent
process therefore always requires the explicit ``--execute`` switch.  The
runner is useful with the local ``AgentRuntime`` as well as external agents:
tests and integrations can inject an ``AgentInvoker`` while the default
``CommandInvoker`` speaks a deliberately boring argv-based protocol.

This is an experiment harness, not a claim of perfect causal fairness.  The
manifest records the model/agent configuration supplied by the caller, while
the runner keeps task, fixture, repetition, and budget handling identical.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from coding_agent.events import redact
from coding_agent.tools.shell import is_sensitive_environment_name
from coding_agent.verification import (
    CommandVerifier,
    VerificationCheck,
    VerificationCheckResult,
    VerificationResult,
)

BENCHMARK_SCHEMA_VERSION = "coding-agent-benchmark/v1"
DEFAULT_MAX_WALL_TIME_SECONDS = 600.0
DEFAULT_MAX_MODEL_REQUESTS = 15
DEFAULT_MAX_TOOL_CALLS = 60
DEFAULT_MAX_TOTAL_TOKENS = 80_000
DEFAULT_OUTPUT_CHAR_LIMIT = 12_000
MAX_CASES = 1_000
_TERMINATION_GRACE_SECONDS = 0.5
_CONFIG_SECRET_NAME_RE = re.compile(
    r"(?:api[_-]?key|authorization|password|passwd|secret|credential|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|token)",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"\{(?:workspace|task_file|task|prompt|task_id|agent_id|repetition)\}"
)


class BenchmarkError(ValueError):
    """A malformed manifest or an invalid benchmark configuration."""


def _non_empty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{field_name} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise BenchmarkError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int_or_none(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BenchmarkError(f"{field_name} must be a non-negative integer or null")
    return value


def _positive_float(value: object, field_name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise BenchmarkError(f"{field_name} must be a finite positive number")
    return float(value)


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{field_name} must be an object")
    return value


def _metadata(value: object, field_name: str) -> dict[str, Any]:
    """Copy optional metadata while giving malformed input a useful error."""

    if value is None:
        return {}
    try:
        return dict(_mapping(value, field_name))
    except (TypeError, ValueError) as exc:
        raise BenchmarkError(f"{field_name} must be an object") from exc


def _normalise_secret_values(values: Iterable[str]) -> tuple[str, ...]:
    """Return deterministic, non-empty secret values for serialization guards."""

    if isinstance(values, str):
        values = (values,)
    unique = {value for value in values if isinstance(value, str) and value}
    return tuple(sorted(unique, key=lambda value: (-len(value), value)))


def _secret_sequence(values: Iterable[str]) -> tuple[str, ...]:
    """Materialize secrets without treating one string as a character list."""

    return (values,) if isinstance(values, str) else tuple(values)


def _normalise_environment_names(value: object, field_name: str) -> tuple[str, ...]:
    """Validate names that may be copied from the host environment.

    A benchmark manifest may need to give a model provider credential to an
    external Agent process.  Storing the value in JSON would make accidental
    commits too easy, so the manifest stores only an explicit allow-list of
    variable names.  The value is looked up at invocation time and is never
    part of the serialized manifest/report.
    """

    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(
        value, (list, tuple, set, frozenset)
    ):
        raise BenchmarkError(f"{field_name} must be an array of environment names")
    # Preserve the caller's order for sequence inputs, but make set-based
    # manifests deterministic so serialized plans and process environments do
    # not depend on hash iteration order.
    values = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
    names: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str) or not item or "=" in item or "\x00" in item:
            raise BenchmarkError(
                f"{field_name} values must be non-empty environment names"
            )
        if item not in seen:
            seen.add(item)
            names.append(item)
    return tuple(names)


def _host_environment_values(names: Iterable[str]) -> set[str]:
    """Collect selected host values solely for in-memory redaction guards."""

    return {os.environ[name] for name in names if os.environ.get(name)}


def _is_config_secret_name(name: object) -> bool:
    """Identify credential-shaped model/config keys without treating ``max_tokens`` as secret."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    if normalized in {"max_tokens", "token_budget", "token_count", "input_tokens"}:
        return False
    return bool(_CONFIG_SECRET_NAME_RE.search(normalized))


def _collect_config_secret_values(value: object, *, key_name: object = "") -> set[str]:
    """Collect string leaves below credential-shaped config keys.

    A manifest cannot know every provider's naming convention.  Field-name
    redaction catches common names, while this pass also protects an opaque
    value such as ``gateway_token`` when callers serialize a manifest directly
    without supplying an explicit secret list.
    """

    if isinstance(value, Mapping):
        found: set[str] = set()
        for key, child in value.items():
            found.update(
                _collect_config_secret_values(
                    child,
                    key_name=key,
                )
            )
        return found
    if isinstance(value, (list, tuple, set, frozenset)):
        found = set()
        for child in value:
            found.update(_collect_config_secret_values(child, key_name=key_name))
        return found
    if _is_config_secret_name(key_name) and isinstance(value, str):
        return {value}
    return set()


def _redacted_mapping(
    value: Mapping[str, Any], secrets: Iterable[str]
) -> dict[str, Any]:
    """Return a JSON-safe redacted mapping for user-supplied metadata."""

    safe = redact(dict(value), secrets=secrets)
    if not isinstance(safe, dict):  # pragma: no cover - ``redact`` contract
        raise BenchmarkError("metadata redaction did not return an object")
    return safe


@dataclass(frozen=True, slots=True)
class BenchmarkBudget:
    """Resource limits applied identically to every agent/task case.

    The token and model/tool counters are *accounting ceilings*.  An external
    command can report them through :class:`AgentExecution`; the harness cannot
    infer hidden provider usage.  Wall time is enforced by ``CommandInvoker``
    and can also be enforced by an injected invoker.
    """

    max_wall_time_seconds: float = DEFAULT_MAX_WALL_TIME_SECONDS
    max_model_requests: int = DEFAULT_MAX_MODEL_REQUESTS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    output_char_limit: int = DEFAULT_OUTPUT_CHAR_LIMIT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_wall_time_seconds",
            _positive_float(self.max_wall_time_seconds, "max_wall_time_seconds"),
        )
        for name in (
            "max_model_requests",
            "max_tool_calls",
            "max_total_tokens",
            "output_char_limit",
        ):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> BenchmarkBudget:
        data = {} if value is None else _mapping(value, "budget")
        aliases = {
            "max_wall_time": "max_wall_time_seconds",
            "max_wall_seconds": "max_wall_time_seconds",
            "max_time": "max_wall_time_seconds",
            "max_requests": "max_model_requests",
            "max_model_turns": "max_model_requests",
            "max_tokens": "max_total_tokens",
            "output_limit": "output_char_limit",
        }
        values: dict[str, Any] = {}
        seen_canonical: dict[str, str] = {}
        known = {
            "max_wall_time_seconds",
            "max_model_requests",
            "max_tool_calls",
            "max_total_tokens",
            "output_char_limit",
        }
        unknown: list[str] = []
        for key, item in data.items():
            raw_name = str(key)
            canonical = aliases.get(raw_name, raw_name)
            if canonical in known:
                previous = seen_canonical.get(canonical)
                if previous is not None:
                    raise BenchmarkError(
                        f"duplicate budget fields for {canonical!r}: "
                        f"{previous!r} and {raw_name!r}"
                    )
                seen_canonical[canonical] = raw_name
                values[canonical] = item
            else:
                unknown.append(raw_name)
        if unknown:
            raise BenchmarkError(
                "unknown budget field(s): " + ", ".join(sorted(unknown))
            )
        try:
            return cls(**values)
        except TypeError as exc:
            raise BenchmarkError(f"invalid budget: {exc}") from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_model_requests": self.max_model_requests,
            "max_tool_calls": self.max_tool_calls,
            "max_total_tokens": self.max_total_tokens,
            "output_char_limit": self.output_char_limit,
        }


def _normalise_checks(
    value: object, field_name: str = "checks"
) -> tuple[VerificationCheck, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise BenchmarkError(f"{field_name} must be a non-empty array")
    checks: list[VerificationCheck] = []
    names: set[str] = set()
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            try:
                check = VerificationCheck(name=f"check-{index}", command=item)
            except ValueError as exc:
                raise BenchmarkError(f"invalid {field_name}[{index}]: {exc}") from exc
        elif isinstance(item, Mapping):
            data = dict(item)
            name = data.pop("name", f"check-{index}")
            command = data.pop("command", None)
            has_timeout_seconds = "timeout_seconds" in data
            has_timeout = "timeout" in data
            if has_timeout_seconds and has_timeout:
                raise BenchmarkError(
                    f"invalid {field_name}[{index}]: timeout and "
                    "timeout_seconds are aliases; specify only one"
                )
            if has_timeout_seconds:
                timeout = data.pop("timeout_seconds")
            else:
                timeout = data.pop("timeout", 120.0)
            if data:
                unknown = ", ".join(sorted(str(key) for key in data))
                raise BenchmarkError(
                    f"unknown fields in {field_name}[{index}]: {unknown}"
                )
            try:
                check = VerificationCheck(
                    name=name,
                    command=command,
                    timeout_seconds=timeout,
                )
            except (TypeError, ValueError) as exc:
                raise BenchmarkError(f"invalid {field_name}[{index}]: {exc}") from exc
        else:
            raise BenchmarkError(
                f"{field_name}[{index}] must be a command string or object"
            )
        if check.name in names:
            raise BenchmarkError(f"duplicate verification check name: {check.name!r}")
        names.add(check.name)
        checks.append(check)
    return tuple(checks)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """One immutable task fixture and its trusted acceptance checks."""

    task_id: str
    prompt: str
    fixture: Path
    checks: tuple[VerificationCheck, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _non_empty_text(self.task_id, "task_id"))
        object.__setattr__(self, "prompt", _non_empty_text(self.prompt, "prompt"))
        if not isinstance(self.fixture, Path):
            try:
                object.__setattr__(self, "fixture", Path(self.fixture))
            except (TypeError, ValueError) as exc:
                raise BenchmarkError("fixture must be a valid path") from exc
        if not self.fixture.is_absolute():
            # Relative paths are resolved by ``BenchmarkManifest.from_dict``;
            # retaining this fallback makes direct construction predictable.
            object.__setattr__(self, "fixture", self.fixture.expanduser().resolve())
        if not self.fixture.name:
            raise BenchmarkError("fixture must name a directory")
        if isinstance(self.checks, str):
            # A direct library caller may naturally provide one command
            # string.  Treat it as one check instead of iterating over its
            # characters (the manifest JSON form remains an array).
            object.__setattr__(self, "checks", _normalise_checks([self.checks]))
        elif not isinstance(self.checks, tuple):
            try:
                object.__setattr__(self, "checks", tuple(self.checks))
            except TypeError as exc:
                raise BenchmarkError("task checks must be an iterable") from exc
        if not self.checks:
            raise BenchmarkError("at least one verification check is required")
        if not all(isinstance(item, VerificationCheck) for item in self.checks):
            object.__setattr__(self, "checks", _normalise_checks(self.checks))
        object.__setattr__(self, "metadata", _metadata(self.metadata, "task metadata"))

    @property
    def id(self) -> str:
        """Short alias useful to report consumers."""

        return self.task_id

    @property
    def name(self) -> str:
        """Human-readable alias for integrations that call tasks ``name``."""

        return self.task_id

    def to_dict(
        self,
        *,
        base_dir: Path | None = None,
        secrets: Iterable[str] = (),
    ) -> dict[str, object]:
        fixture = self.fixture
        if base_dir is not None:
            try:
                fixture_value = str(fixture.relative_to(base_dir))
            except ValueError:
                fixture_value = str(fixture)
        else:
            fixture_value = str(fixture)
        safe_metadata = _redacted_mapping(self.metadata, secrets)
        return {
            "id": self.task_id,
            "prompt": self.prompt,
            "fixture": fixture_value,
            "checks": [
                {
                    "name": item.name,
                    "command": item.command,
                    "timeout_seconds": item.timeout_seconds,
                }
                for item in self.checks
            ],
            "metadata": safe_metadata,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkAgent:
    """An external agent command represented as argv, never a shell string."""

    agent_id: str
    command: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    environment_from_host: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _non_empty_text(self.agent_id, "agent_id"))
        command = self.command
        if isinstance(command, str):
            try:
                command = tuple(shlex.split(command))
            except ValueError as exc:
                raise BenchmarkError(f"invalid command for {self.agent_id!r}") from exc
        elif isinstance(command, Mapping):
            raise BenchmarkError(
                f"command for {self.agent_id!r} must be a string or argv array"
            )
        else:
            try:
                command = tuple(command)
            except TypeError as exc:
                raise BenchmarkError(
                    f"command for {self.agent_id!r} must be non-empty argv"
                ) from exc
        if not command or any(
            not isinstance(item, str) or not item for item in command
        ):
            raise BenchmarkError(
                f"command for {self.agent_id!r} must be non-empty argv"
            )
        if any("\x00" in item for item in command):
            raise BenchmarkError("agent command arguments cannot contain NUL")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "metadata", _metadata(self.metadata, "agent metadata"))
        if self.environment is None:
            raise BenchmarkError("agent environment must be an object")
        if not isinstance(self.environment, Mapping):
            raise BenchmarkError("agent environment must be an object")
        env: dict[str, str] = {}
        for key, value in self.environment.items():
            if (
                not isinstance(key, str)
                or not key
                or "=" in key
                or "\x00" in key
                or not isinstance(value, str)
                or "\x00" in value
            ):
                raise BenchmarkError(
                    "agent environment names and values must be valid strings"
                )
            env[key] = value
        object.__setattr__(self, "environment", env)
        object.__setattr__(
            self,
            "environment_from_host",
            _normalise_environment_names(
                self.environment_from_host, "agent environment_from_host"
            ),
        )

    @property
    def id(self) -> str:
        return self.agent_id

    @property
    def name(self) -> str:
        return self.agent_id

    def to_dict(self, *, secrets: Iterable[str] = ()) -> dict[str, object]:
        secret_values = _normalise_secret_values(
            (
                *_secret_sequence(secrets),
                *self.environment.values(),
                *_host_environment_values(self.environment_from_host),
            )
        )
        safe_command = [
            _safe_text(argument, secret_values) or "" for argument in self.command
        ]
        safe_metadata = _redacted_mapping(self.metadata, secret_values)
        return {
            "id": self.agent_id,
            "command": safe_command,
            "metadata": safe_metadata,
            # Environment values may be API credentials.  Record only names;
            # callers can reproduce the contract without serializing secrets.
            "environment": {key: "[configured]" for key in sorted(self.environment)},
            # Only names are recorded.  The corresponding host values are
            # resolved immediately before a trusted benchmark command starts.
            "environment_from_host": list(self.environment_from_host),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """Validated benchmark input shared by all runs."""

    tasks: tuple[BenchmarkTask, ...]
    agents: tuple[BenchmarkAgent, ...]
    repetitions: int = 1
    budget: BenchmarkBudget = field(default_factory=BenchmarkBudget)
    name: str = "coding-agent-benchmark"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    model: Mapping[str, Any] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        """Canonical manifest schema emitted by this implementation."""

        return BENCHMARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.tasks, tuple):
            try:
                object.__setattr__(self, "tasks", tuple(self.tasks))
            except TypeError as exc:
                raise BenchmarkError("manifest tasks must be an iterable") from exc
        if not isinstance(self.agents, tuple):
            try:
                object.__setattr__(self, "agents", tuple(self.agents))
            except TypeError as exc:
                raise BenchmarkError("manifest agents must be an iterable") from exc
        if not self.tasks or not self.agents:
            raise BenchmarkError("manifest needs at least one task and one agent")
        if not all(isinstance(task, BenchmarkTask) for task in self.tasks):
            raise BenchmarkError("manifest tasks must contain BenchmarkTask values")
        if not all(isinstance(agent, BenchmarkAgent) for agent in self.agents):
            raise BenchmarkError("manifest agents must contain BenchmarkAgent values")
        task_ids = [task.task_id for task in self.tasks]
        agent_ids = [agent.agent_id for agent in self.agents]
        if len(task_ids) != len(set(task_ids)):
            raise BenchmarkError("task IDs must be unique")
        if len(agent_ids) != len(set(agent_ids)):
            raise BenchmarkError("agent IDs must be unique")
        object.__setattr__(
            self, "repetitions", _positive_int(self.repetitions, "repetitions")
        )
        if self.repetitions * len(self.tasks) * len(self.agents) > MAX_CASES:
            raise BenchmarkError(
                f"manifest expands to more than {MAX_CASES} cases; reduce repetitions/tasks/agents"
            )
        if not isinstance(self.budget, BenchmarkBudget):
            object.__setattr__(
                self, "budget", BenchmarkBudget.from_mapping(self.budget)
            )
        object.__setattr__(self, "name", _non_empty_text(self.name, "name"))
        object.__setattr__(
            self, "metadata", _metadata(self.metadata, "manifest metadata")
        )
        if self.source_path is not None and not isinstance(self.source_path, Path):
            object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "model", _metadata(self.model, "model config"))

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        base_dir: str | os.PathLike[str] | None = None,
        source_path: Path | None = None,
    ) -> BenchmarkManifest:
        data = _mapping(value, "manifest")
        root = Path(base_dir or Path.cwd()).expanduser().resolve()
        raw_tasks = data.get("tasks")
        raw_agents = data.get("agents")
        if not isinstance(raw_tasks, (list, tuple)):
            raise BenchmarkError("manifest.tasks must be an array")
        if not isinstance(raw_agents, (list, tuple)):
            raise BenchmarkError("manifest.agents must be an array")

        tasks: list[BenchmarkTask] = []
        for index, raw in enumerate(raw_tasks, start=1):
            item = _mapping(raw, f"tasks[{index}]")
            task_id = item.get("id", item.get("task_id"))
            prompt = item.get("prompt", item.get("task", item.get("description")))
            fixture_raw = item.get("fixture", item.get("workspace"))
            if fixture_raw is None:
                raise BenchmarkError(f"tasks[{index}] requires fixture")
            fixture = Path(_non_empty_text(fixture_raw, f"tasks[{index}].fixture"))
            if not fixture.is_absolute():
                fixture = (root / fixture).resolve()
            checks_raw = item.get(
                "checks", item.get("verification", item.get("verify"))
            )
            checks = _normalise_checks(checks_raw, f"tasks[{index}].checks")
            known = {
                "id",
                "task_id",
                "prompt",
                "task",
                "description",
                "fixture",
                "workspace",
                "checks",
                "verification",
                "verify",
                "metadata",
            }
            metadata = _metadata(item.get("metadata", {}), f"tasks[{index}].metadata")
            for key, extra in item.items():
                if key not in known:
                    metadata.setdefault(str(key), extra)
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    prompt=prompt,
                    fixture=fixture,
                    checks=checks,
                    metadata=metadata,
                )
            )

        agents: list[BenchmarkAgent] = []
        for index, raw in enumerate(raw_agents, start=1):
            item = _mapping(raw, f"agents[{index}]")
            agent_id = item.get("id", item.get("agent_id", item.get("name")))
            command = item.get("command", item.get("argv"))
            if command is None:
                raise BenchmarkError(f"agents[{index}] requires command")
            known = {
                "id",
                "agent_id",
                "name",
                "command",
                "argv",
                "metadata",
                "environment",
                "environment_from_host",
                "inherit_environment",
            }
            metadata = _metadata(item.get("metadata", {}), f"agents[{index}].metadata")
            for key, extra in item.items():
                if key not in known:
                    metadata.setdefault(str(key), extra)
            host_names = item.get(
                "environment_from_host",
                item.get("inherit_environment", ()),
            )
            agents.append(
                BenchmarkAgent(
                    agent_id=agent_id,
                    command=command,
                    metadata=metadata,
                    environment=item.get("environment", {}),
                    environment_from_host=_normalise_environment_names(
                        host_names, f"agents[{index}].environment_from_host"
                    ),
                )
            )

        raw_version = data.get("schema_version", BENCHMARK_SCHEMA_VERSION)
        if not isinstance(raw_version, str) or not raw_version.strip():
            raise BenchmarkError("schema_version must be a non-empty string")
        if raw_version not in {BENCHMARK_SCHEMA_VERSION, "v1"}:
            raise BenchmarkError(
                f"unsupported benchmark schema_version: {raw_version!r}"
            )
        metadata = _metadata(data.get("metadata", {}), "manifest metadata")
        model = _metadata(
            data.get("model", data.get("model_config", {})), "model config"
        )
        known_root = {
            "schema_version",
            "name",
            "tasks",
            "agents",
            "repetitions",
            "budget",
            "metadata",
            "model",
            "model_config",
        }
        for key, extra in data.items():
            if key not in known_root:
                metadata.setdefault(str(key), extra)
        try:
            repetitions = data.get("repetitions", 1)
            budget = BenchmarkBudget.from_mapping(data.get("budget"))
            return cls(
                tasks=tuple(tasks),
                agents=tuple(agents),
                repetitions=repetitions,
                budget=budget,
                name=data.get("name", "coding-agent-benchmark"),
                metadata=metadata,
                source_path=source_path,
                model=model,
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, BenchmarkError):
                raise
            raise BenchmarkError(f"invalid manifest: {exc}") from exc

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> BenchmarkManifest:
        source = Path(path).expanduser().resolve()
        try:
            with source.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except OSError as exc:
            raise BenchmarkError(
                f"could not read manifest: {type(exc).__name__}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise BenchmarkError(
                f"manifest is not valid JSON at line {exc.lineno}"
            ) from exc
        return cls.from_dict(value, base_dir=source.parent, source_path=source)

    def with_filters(
        self,
        *,
        task_ids: Iterable[str] = (),
        agent_ids: Iterable[str] = (),
    ) -> BenchmarkManifest:
        wanted_tasks = {item for item in task_ids if item}
        wanted_agents = {item for item in agent_ids if item}
        tasks = tuple(
            task for task in self.tasks if not wanted_tasks or task.id in wanted_tasks
        )
        agents = tuple(
            agent
            for agent in self.agents
            if not wanted_agents or agent.id in wanted_agents
        )
        if wanted_tasks and len(tasks) != len(wanted_tasks):
            missing = sorted(wanted_tasks - {task.id for task in tasks})
            raise BenchmarkError(f"unknown task ID(s): {', '.join(missing)}")
        if wanted_agents and len(agents) != len(wanted_agents):
            missing = sorted(wanted_agents - {agent.id for agent in agents})
            raise BenchmarkError(f"unknown agent ID(s): {', '.join(missing)}")
        return replace(self, tasks=tasks, agents=agents)

    def to_dict(
        self,
        *,
        include_source: bool = False,
        secrets: Iterable[str] = (),
    ) -> dict[str, object]:
        # Include values configured in agent environments and credential-shaped
        # model fields in the local redaction set. This keeps direct
        # ``manifest.to_dict()`` calls safe, even when no outer report supplies
        # an explicit secret list.
        safe_secrets = _normalise_secret_values(
            (
                *_secret_sequence(secrets),
                *(
                    value
                    for agent in self.agents
                    for value in agent.environment.values()
                ),
                *(
                    value
                    for agent in self.agents
                    for value in _host_environment_values(agent.environment_from_host)
                ),
                *_collect_config_secret_values(self.model),
            )
        )
        result: dict[str, object] = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "name": self.name,
            "repetitions": self.repetitions,
            "budget": self.budget.to_dict(),
            # Model metadata is useful for reproducibility, but model config
            # objects occasionally contain an accidental key/token field.
            "model": _redacted_mapping(self.model, safe_secrets),
            "tasks": [
                item.to_dict(
                    base_dir=self.source_path.parent if self.source_path else None,
                    secrets=safe_secrets,
                )
                for item in self.tasks
            ],
            "agents": [item.to_dict(secrets=safe_secrets) for item in self.agents],
            "metadata": _redacted_mapping(self.metadata, safe_secrets),
        }
        if include_source and self.source_path is not None:
            result["source_path"] = str(self.source_path)
        # ``BenchmarkManifest.to_dict`` is a public serialization boundary in
        # its own right; do not rely on ``BenchmarkReport`` to perform a later
        # pass.  Prompts, check commands, and paths can all accidentally carry
        # a credential value even when their field names look harmless.
        safe = redact(result, secrets=safe_secrets)
        if not isinstance(safe, dict):  # pragma: no cover - defensive
            raise BenchmarkError("manifest redaction did not return an object")
        _ensure_json(safe)
        return safe


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """The immutable identity supplied to an injected agent invoker."""

    agent: BenchmarkAgent
    task: BenchmarkTask
    workspace: Path
    repetition: int


@dataclass(frozen=True, slots=True)
class AgentExecution:
    """Normalized observation returned by an agent adapter."""

    status: str
    elapsed_seconds: float = 0.0
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    model_requests: int | None = None
    tool_calls: int | None = None
    total_tokens: int | None = None
    error: str | None = None

    _VALID_STATUSES = frozenset(
        {"completed", "failed", "timeout", "budget_exceeded", "cancelled", "error"}
    )

    def __post_init__(self) -> None:
        if self.status not in self._VALID_STATUSES:
            raise BenchmarkError(f"unsupported agent execution status: {self.status!r}")
        if (
            not isinstance(self.elapsed_seconds, (int, float))
            or isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise BenchmarkError("elapsed_seconds must be a finite non-negative number")
        if self.exit_code is not None and (
            not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool)
        ):
            raise BenchmarkError("exit_code must be an integer or null")
        for name in ("model_requests", "tool_calls", "total_tokens"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative_int_or_none(value, name)
        for name in ("stdout", "stderr"):
            if not isinstance(getattr(self, name), str):
                raise BenchmarkError(f"{name} must be a string")
        if self.error is not None and not isinstance(self.error, str):
            raise BenchmarkError("error must be a string or null")

    @classmethod
    def from_runtime_result(
        cls, result: object, *, output_char_limit: int = DEFAULT_OUTPUT_CHAR_LIMIT
    ) -> AgentExecution:
        """Adapt the local Runtime's ``RunResult`` without importing it eagerly."""

        phase = getattr(
            getattr(result, "phase", None), "value", getattr(result, "phase", "error")
        )
        phase_text = str(phase)
        if phase_text == "completed":
            status = "completed"
        elif phase_text == "cancelled":
            status = "cancelled"
        elif phase_text == "limit_reached":
            reason = str(getattr(result, "reason", ""))
            status = "timeout" if "time" in reason.lower() else "budget_exceeded"
        elif phase_text == "failed":
            status = "failed"
        else:
            status = "error"
        usage = getattr(result, "usage", None)
        total_tokens = getattr(usage, "total_tokens", None)
        return cls(
            status=status,
            elapsed_seconds=float(getattr(result, "elapsed_seconds", 0.0) or 0.0),
            model_requests=getattr(result, "model_requests", None),
            tool_calls=getattr(result, "tool_calls", None),
            total_tokens=total_tokens,
            stdout=_bounded_text(
                str(getattr(result, "final_text", "") or ""), output_char_limit
            ),
            error=str(getattr(result, "reason", "")) or None,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "elapsed_seconds": self.elapsed_seconds,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "model_requests": self.model_requests,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_tokens,
            "error": self.error,
        }


class AgentInvoker(Protocol):
    """Callable boundary for one agent case.

    Implementations must enforce the model/tool/token ceilings they can
    observe and return an ``AgentExecution`` rather than raising ordinary
    execution errors.  ``BenchmarkRunner`` still catches adapter bugs and
    records them as ``runner_error`` results.
    """

    def invoke(self, case: BenchmarkCase, budget: BenchmarkBudget) -> AgentExecution:
        """Run one case in ``case.workspace``."""


class CommandInvoker:
    """Invoke an agent executable with a process-group wall-time bound.

    Commands are argv arrays from the manifest.  Exact placeholders
    ``{workspace}``, ``{task_file}``, ``{task}``, ``{prompt}``, ``{task_id}``,
    ``{agent_id}``, and ``{repetition}`` are substituted without shell
    evaluation.  The task is
    also exposed through ``CODING_AGENT_BENCHMARK_*`` environment variables so
    a local adapter can choose either interface.
    """

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        environment_from_host: Iterable[str] = (),
        secrets: Iterable[str] = (),
    ) -> None:
        explicit = dict(environment or {})
        for name, value in explicit.items():
            if (
                not isinstance(name, str)
                or not name
                or "=" in name
                or "\x00" in name
                or not isinstance(value, str)
                or "\x00" in value
            ):
                raise BenchmarkError(
                    "CommandInvoker environment names and values must be valid strings"
                )
        self.environment = explicit
        self.environment_from_host = _normalise_environment_names(
            environment_from_host, "CommandInvoker environment_from_host"
        )
        self.secrets = _normalise_secret_values(secrets)

    def invoke(self, case: BenchmarkCase, budget: BenchmarkBudget) -> AgentExecution:
        started = time.monotonic()
        # Values supplied explicitly to this invoker or case may be credentials
        # even when their variable names do not match our conventional pattern.
        # Include them in the local redaction set for standalone invocations;
        # BenchmarkRunner also supplies its aggregate set at construction time.
        redaction_secrets = tuple(
            set(self.secrets)
            | {value for value in self.environment.values() if value}
            | {value for value in case.agent.environment.values() if value}
            | _host_environment_values(
                (*self.environment_from_host, *case.agent.environment_from_host)
            )
        )
        task_file: Path | None = None
        process: subprocess.Popen[bytes] | None = None
        stdout = b""
        stderr = b""
        timed_out = False
        error: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=case.workspace,
                prefix=".benchmark-task-",
                suffix=".txt",
                delete=False,
            ) as handle:
                task_file = Path(handle.name)
                handle.write(case.task.prompt)
                handle.write("\n")
            replacements = {
                "{workspace}": str(case.workspace),
                "{task_file}": str(task_file),
                "{task}": case.task.prompt,
                "{prompt}": case.task.prompt,
                "{task_id}": case.task.id,
                "{agent_id}": case.agent.id,
                "{repetition}": str(case.repetition),
            }
            command = tuple(
                _replace_placeholders(argument, replacements)
                for argument in case.agent.command
            )
            # Do not inherit conventional host credentials into an external
            # agent process.  A benchmark may explicitly opt in to a precise
            # set of names through ``environment_from_host``; values selected
            # that way are treated like explicit secrets and are redacted from
            # all returned output.
            env = {
                name: value
                for name, value in os.environ.items()
                if not is_sensitive_environment_name(name) and value not in self.secrets
            }
            selected_host_names = (
                *self.environment_from_host,
                *case.agent.environment_from_host,
            )
            env.update(
                {
                    name: os.environ[name]
                    for name in selected_host_names
                    if name in os.environ
                }
            )
            env.update(self.environment)
            env.update(case.agent.environment)
            env.update(
                {
                    "CODING_AGENT_BENCHMARK_TASK": case.task.prompt,
                    "CODING_AGENT_BENCHMARK_TASK_ID": case.task.id,
                    "CODING_AGENT_BENCHMARK_AGENT_ID": case.agent.id,
                    "CODING_AGENT_BENCHMARK_REPETITION": str(case.repetition),
                    "CODING_AGENT_BENCHMARK_WORKSPACE": str(case.workspace),
                    "CODING_AGENT_BENCHMARK_BUDGET": json.dumps(
                        budget.to_dict(), separators=(",", ":")
                    ),
                }
            )
            # File-backed streams keep a noisy agent from making the parent
            # process retain unbounded stdout/stderr in memory.  We still read
            # only a bounded head/tail after completion for the report.
            with (
                tempfile.TemporaryFile(mode="w+b") as stdout_buffer,
                tempfile.TemporaryFile(mode="w+b") as stderr_buffer,
            ):
                process = subprocess.Popen(
                    command,
                    cwd=case.workspace,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_buffer,
                    stderr=stderr_buffer,
                    start_new_session=True,
                )
                try:
                    process.wait(timeout=budget.max_wall_time_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _terminate_process_group(process)
                stdout = _read_bounded_stream(stdout_buffer, budget.output_char_limit)
                stderr = _read_bounded_stream(stderr_buffer, budget.output_char_limit)
        except OSError as exc:
            error = f"could not start agent command: {type(exc).__name__}"
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            error = f"agent adapter error: {type(exc).__name__}"
        finally:
            if task_file is not None:
                try:
                    task_file.unlink()
                except OSError:
                    pass

        elapsed = time.monotonic() - started
        if timed_out:
            status = "timeout"
        elif error is not None:
            status = "error"
        elif process is not None and process.returncode == 0:
            status = "completed"
        else:
            status = "failed"
        return AgentExecution(
            status=status,
            elapsed_seconds=elapsed,
            exit_code=process.returncode if process is not None else None,
            stdout=_bounded_text(
                _safe_text(
                    _bounded_text(stdout, budget.output_char_limit), redaction_secrets
                )
                or "",
                budget.output_char_limit,
            ),
            stderr=_bounded_text(
                _safe_text(
                    _bounded_text(stderr, budget.output_char_limit), redaction_secrets
                )
                or "",
                budget.output_char_limit,
            ),
            error=_safe_text(error, redaction_secrets),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """One row in the machine-readable benchmark report."""

    run_id: str
    task_id: str
    agent_id: str
    repetition: int
    status: str
    resolved: bool
    started_at: str
    execution: AgentExecution | None = None
    verification: VerificationResult | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "agent_id", "started_at"):
            _non_empty_text(getattr(self, name), name)
        if (
            not isinstance(self.repetition, int)
            or isinstance(self.repetition, bool)
            or self.repetition <= 0
        ):
            raise BenchmarkError("repetition must be a positive integer")
        allowed = {
            "planned",
            "resolved",
            "unresolved",
            "agent_failed",
            "timeout",
            "budget_exceeded",
            "cancelled",
            "runner_error",
        }
        if self.status not in allowed:
            raise BenchmarkError(
                f"unsupported benchmark result status: {self.status!r}"
            )
        if not isinstance(self.resolved, bool):
            raise BenchmarkError("resolved must be a bool")
        if self.resolved != (self.status == "resolved"):
            raise BenchmarkError("resolved must agree with status")
        if self.error is not None and not isinstance(self.error, str):
            raise BenchmarkError("error must be a string or null")

    def to_dict(self, *, secrets: Iterable[str] = ()) -> dict[str, object]:
        metrics: dict[str, object] = {}
        execution = self.execution
        if execution is not None:
            metrics = {
                "elapsed_seconds": execution.elapsed_seconds,
                "model_requests": execution.model_requests,
                "tool_calls": execution.tool_calls,
                "total_tokens": execution.total_tokens,
            }
        document: dict[str, object] = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "kind": "benchmark_run",
            "run_id": self.run_id,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "repetition": self.repetition,
            "status": self.status,
            "resolved": self.resolved,
            "started_at": self.started_at,
            "metrics": metrics,
            "execution": execution.to_dict() if execution is not None else None,
            "verification": _verification_to_dict(self.verification),
            "error": self.error,
        }
        safe = redact(document, secrets=_normalise_secret_values(secrets))
        if not isinstance(safe, dict):  # pragma: no cover - defensive
            raise BenchmarkError("result redaction did not return an object")
        return safe

    @property
    def success(self) -> bool:
        """Alias for callers that use success terminology."""

        return self.resolved

    @property
    def passed(self) -> bool:
        return self.resolved


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Complete report containing manifest metadata, rows, and aggregates."""

    manifest: BenchmarkManifest
    runs: tuple[BenchmarkRunResult, ...]
    dry_run: bool
    generated_at: str

    def to_dict(self, *, secrets: Iterable[str] = ()) -> dict[str, object]:
        # Reports may contain adapter-provided stdout/stderr, not just the
        # manifest projection.  Include credentials discoverable from the
        # manifest itself before redacting those observations; otherwise a
        # custom invoker could accidentally echo a model token into JSON.
        safe_secrets = _normalise_secret_values(
            (
                *_secret_sequence(secrets),
                *(
                    value
                    for agent in self.manifest.agents
                    for value in agent.environment.values()
                ),
                *(
                    value
                    for agent in self.manifest.agents
                    for value in _host_environment_values(agent.environment_from_host)
                ),
                *_collect_config_secret_values(self.manifest.model),
            )
        )
        by_agent: dict[str, dict[str, object]] = {}
        grouped: dict[str, list[BenchmarkRunResult]] = defaultdict(list)
        for run in self.runs:
            grouped[run.agent_id].append(run)
        for agent in self.manifest.agents:
            rows = grouped.get(agent.id, [])
            resolved_count = sum(item.resolved for item in rows)
            statuses = Counter(item.status for item in rows)
            by_agent[agent.id] = {
                "cases": len(rows),
                "resolved": resolved_count,
                "resolved_rate": resolved_count / len(rows) if rows else None,
                "statuses": dict(sorted(statuses.items())),
            }
        result: dict[str, object] = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "kind": "benchmark_report",
            "generated_at": self.generated_at,
            "dry_run": self.dry_run,
            "manifest": self.manifest.to_dict(
                include_source=True,
                secrets=safe_secrets,
            ),
            "summary": {
                "cases": len(self.runs),
                "resolved": sum(item.resolved for item in self.runs),
                "resolved_rate": (
                    sum(item.resolved for item in self.runs) / len(self.runs)
                    if self.runs
                    else None
                ),
                "by_agent": by_agent,
            },
            "runs": [item.to_dict(secrets=safe_secrets) for item in self.runs],
        }
        safe = redact(result, secrets=safe_secrets)
        if not isinstance(safe, dict):  # pragma: no cover - defensive
            raise BenchmarkError("report redaction did not return an object")
        _ensure_json(safe)
        return safe

    def to_json(self, *, secrets: Iterable[str] = (), indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(secrets=secrets),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=False,
        )


class BenchmarkRunner:
    """Expand a manifest into bounded, independently verifiable cases."""

    def __init__(
        self,
        manifest: BenchmarkManifest,
        *,
        invoker: AgentInvoker | None = None,
        workspace_root: str | os.PathLike[str] | None = None,
        preserve_workspaces: bool = False,
        secrets: Iterable[str] = (),
    ) -> None:
        if not isinstance(manifest, BenchmarkManifest):
            raise TypeError("manifest must be a BenchmarkManifest")
        self.manifest = manifest
        self.workspace_root = (
            Path(workspace_root).expanduser().resolve()
            if workspace_root is not None
            else None
        )
        self.preserve_workspaces = bool(preserve_workspaces)
        configured_values = {
            value
            for agent in manifest.agents
            for value in agent.environment.values()
            if value
        }
        configured_values.update(_collect_config_secret_values(manifest.model))
        host_names = {
            name for agent in manifest.agents for name in agent.environment_from_host
        }
        configured_values.update(_host_environment_values(host_names))
        self.secrets = _normalise_secret_values(
            (*_secret_sequence(secrets), *configured_values)
        )
        # Host credentials are selected per agent case.  Passing the union of
        # all names here would expose agent B's key to agent A's process and
        # make side-by-side comparisons less isolated.  Callers that need a
        # benchmark-wide non-secret variable can still construct an explicit
        # ``CommandInvoker(environment_from_host=...)``.
        selected_invoker = (
            CommandInvoker(secrets=self.secrets) if invoker is None else invoker
        )
        if isinstance(selected_invoker, CommandInvoker):
            # A caller may inject a preconfigured CommandInvoker (for example
            # to pass a private gateway setting or a host key).  Its own
            # process boundary normally redacts these values, but a subclass
            # or wrapper can return observations directly.  Aggregate the
            # values here as a second serialization guard.  Deliberately do
            # not merge its host names into another agent's environment.
            invoker_values = {
                value for value in selected_invoker.environment.values() if value
            }
            invoker_values.update(selected_invoker.secrets)
            invoker_values.update(
                _host_environment_values(selected_invoker.environment_from_host)
            )
            host_names.update(selected_invoker.environment_from_host)
            self.secrets = _normalise_secret_values((*self.secrets, *invoker_values))
        # This set is used only to keep verifier subprocesses from inheriting
        # explicitly forwarded credentials. It is intentionally separate from
        # the per-case names used by ``CommandInvoker`` so an injected
        # invoker's global allow-list cannot broaden another agent's process
        # environment.
        self.host_environment_names = frozenset(host_names)
        self.invoker = selected_invoker

    def plan(self) -> BenchmarkReport:
        """Return a report with one ``planned`` row per case, without I/O."""

        runs = tuple(
            BenchmarkRunResult(
                run_id=_run_id(agent, task, repetition),
                task_id=task.id,
                agent_id=agent.id,
                repetition=repetition,
                status="planned",
                resolved=False,
                started_at=_utc_now(),
            )
            for agent in self.manifest.agents
            for task in self.manifest.tasks
            for repetition in range(1, self.manifest.repetitions + 1)
        )
        return BenchmarkReport(
            manifest=self.manifest,
            runs=runs,
            dry_run=True,
            generated_at=_utc_now(),
        )

    def run(self, *, dry_run: bool = False) -> BenchmarkReport:
        """Execute all cases, or return a side-effect-free plan when requested."""

        if dry_run:
            return self.plan()

        runs: list[BenchmarkRunResult] = []
        for agent in self.manifest.agents:
            for task in self.manifest.tasks:
                for repetition in range(1, self.manifest.repetitions + 1):
                    runs.append(self._run_case(agent, task, repetition))
        return BenchmarkReport(
            manifest=self.manifest,
            runs=tuple(runs),
            dry_run=False,
            generated_at=_utc_now(),
        )

    def _run_case(
        self,
        agent: BenchmarkAgent,
        task: BenchmarkTask,
        repetition: int,
    ) -> BenchmarkRunResult:
        run_id = _run_id(agent, task, repetition)
        started_at = _utc_now()
        workspace: Path | None = None
        execution: AgentExecution | None = None
        verification: VerificationResult | None = None
        try:
            workspace, cleanup = self._prepare_workspace(task, run_id)
            case = BenchmarkCase(
                agent=agent,
                task=task,
                workspace=workspace,
                repetition=repetition,
            )
            execution = self._invoke(case)
            # Verification is intentionally outside the AgentRuntime and does
            # not consume its model/tool budget. It runs against the resulting
            # fresh workspace using trusted manifest commands.
            verifier = CommandVerifier(
                workspace,
                task.checks,
                output_char_limit=self.manifest.budget.output_char_limit,
                # Acceptance commands are trusted but should not inherit an
                # Agent's explicitly forwarded provider credential (or any
                # other selected host-only value).  The verifier has no reason
                # to use those variables and keeping them out avoids leaking
                # them through test output.
                excluded_environment_names=self.host_environment_names,
            )
            verification = verifier.verify()
            if execution.status == "timeout":
                status = "timeout"
            elif execution.status == "budget_exceeded":
                status = "budget_exceeded"
            elif execution.status == "cancelled":
                status = "cancelled"
            elif execution.status != "completed":
                status = "agent_failed"
            elif verification.passed:
                status = "resolved"
            else:
                status = "unresolved"
            return BenchmarkRunResult(
                run_id=run_id,
                task_id=task.id,
                agent_id=agent.id,
                repetition=repetition,
                status=status,
                resolved=status == "resolved",
                started_at=started_at,
                execution=execution,
                verification=verification,
                error=execution.error,
            )
        except Exception as exc:  # noqa: BLE001 - one row must not abort suite
            return BenchmarkRunResult(
                run_id=run_id,
                task_id=task.id,
                agent_id=agent.id,
                repetition=repetition,
                status="runner_error",
                resolved=False,
                started_at=started_at,
                execution=execution,
                verification=verification,
                error=f"{type(exc).__name__}",
            )
        finally:
            if workspace is not None and not self.preserve_workspaces:
                cleanup(workspace)

    def _invoke(self, case: BenchmarkCase) -> AgentExecution:
        invoker = self.invoker
        try:
            if hasattr(invoker, "invoke"):
                execution = invoker.invoke(case, self.manifest.budget)
            else:
                # A plain four-argument callable is convenient in tests and
                # keeps integration with small local adapters frictionless.
                execution = invoker(
                    case.agent, case.task, case.workspace, self.manifest.budget
                )  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            return AgentExecution(status="error", error=f"{type(exc).__name__}")
        if not isinstance(execution, AgentExecution):
            # Accept the local Runtime's structurally compatible RunResult so
            # a benchmark script can inject an in-process agent without a
            # bespoke adapter.  External integrations should still prefer the
            # explicit AgentExecution contract.
            if hasattr(execution, "phase") and hasattr(execution, "history"):
                execution = AgentExecution.from_runtime_result(
                    execution,
                    output_char_limit=self.manifest.budget.output_char_limit,
                )
            else:
                raise BenchmarkError("agent invoker must return AgentExecution")
        execution = self._redact_execution(execution)
        return self._apply_budget(execution)

    def _redact_execution(self, execution: AgentExecution) -> AgentExecution:
        """Keep command output safe even when callers forget report secrets."""

        if not self.secrets:
            return execution
        return replace(
            execution,
            stdout=_safe_text(execution.stdout, self.secrets),
            stderr=_safe_text(execution.stderr, self.secrets),
            error=_safe_text(execution.error, self.secrets)
            if execution.error is not None
            else None,
        )

    def _apply_budget(self, execution: AgentExecution) -> AgentExecution:
        """Convert observable counter overruns into an explicit budget status.

        A custom invoker may enforce limits internally, but the runner still
        performs this second check so an adapter cannot accidentally report an
        over-budget successful case.  Unknown (``None``) counters remain
        unknown rather than being treated as zero.
        """

        budget = self.manifest.budget
        overrun: str | None = None
        if execution.elapsed_seconds > budget.max_wall_time_seconds:
            overrun = "wall_time"
        elif (
            execution.model_requests is not None
            and execution.model_requests > budget.max_model_requests
        ):
            overrun = "model_requests"
        elif (
            execution.tool_calls is not None
            and execution.tool_calls > budget.max_tool_calls
        ):
            overrun = "tool_calls"
        elif (
            execution.total_tokens is not None
            and execution.total_tokens > budget.max_total_tokens
        ):
            overrun = "total_tokens"
        if overrun is None:
            return execution
        return replace(
            execution,
            status="timeout" if overrun == "wall_time" else "budget_exceeded",
            error=execution.error or f"{overrun} budget exceeded",
        )

    def _prepare_workspace(self, task: BenchmarkTask, run_id: str) -> tuple[Path, Any]:
        fixture = task.fixture
        if not fixture.is_dir():
            raise BenchmarkError(f"fixture is not an existing directory: {task.id}")
        if self.workspace_root is None:
            directory = Path(tempfile.mkdtemp(prefix="coding-agent-bench-"))
            target = directory / "workspace"
            try:
                shutil.copytree(fixture, target)
            except Exception:
                shutil.rmtree(directory, ignore_errors=True)
                raise
            # Remove the temporary parent as well as the copied workspace;
            # otherwise every short-lived run leaves an empty /tmp directory.
            return target, lambda _path: shutil.rmtree(directory, ignore_errors=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(
            char if char.isalnum() or char in "-_." else "_" for char in run_id
        )
        target = self.workspace_root / safe_id
        if target.exists():
            raise BenchmarkError(f"benchmark workspace already exists: {safe_id}")
        try:
            shutil.copytree(fixture, target)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        return target, lambda path: shutil.rmtree(path, ignore_errors=True)


def _run_id(agent: BenchmarkAgent, task: BenchmarkTask, repetition: int) -> str:
    return f"{agent.id}::{task.id}::r{repetition}"


def _replace_placeholders(value: str, replacements: Mapping[str, str]) -> str:
    # Resolve the template in one pass.  Sequential ``str.replace`` calls
    # would reinterpret marker-looking text inside a replacement (for example
    # a task prompt containing the literal ``{agent_id}``).
    return _PLACEHOLDER_RE.sub(lambda match: replacements[match.group(0)], value)


def _bounded_text(data: bytes | str, limit: int) -> str:
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if len(text) <= limit:
        return text
    marker = "\n... [benchmark output truncated] ...\n"
    if limit <= len(marker):
        return text[:limit]
    head = (limit - len(marker)) // 2
    tail = limit - len(marker) - head
    return text[:head] + marker + text[-tail:]


def _read_bounded_stream(stream: Any, limit: int) -> bytes:
    """Read at most ``limit`` bytes plus a bounded tail from a file stream."""

    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if size <= limit:
        stream.seek(0)
        return stream.read()
    head = max(1, limit // 2)
    tail = max(0, limit - head)
    stream.seek(0)
    prefix = stream.read(head)
    stream.seek(max(0, size - tail))
    suffix = stream.read(tail)
    return prefix + b"\n... [benchmark output truncated] ...\n" + suffix


def _safe_text(value: str | None, secrets: Iterable[str]) -> str | None:
    if value is None:
        return None
    redacted = redact(value, secrets=secrets)
    return redacted if isinstance(redacted, str) else str(redacted)


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


def _verification_to_dict(
    result: VerificationResult | None,
) -> dict[str, object] | None:
    if result is None:
        return None
    checks: list[dict[str, object]] = []
    for item in result.checks:
        if not isinstance(item, VerificationCheckResult):
            continue
        checks.append(
            {
                "name": item.check.name,
                "passed": item.passed,
                "exit_code": item.exit_code,
                "timed_out": item.timed_out,
                "stdout": item.stdout,
                "stderr": item.stderr,
                "elapsed_seconds": item.elapsed_seconds,
                "error": item.error,
            }
        )
    return {
        "passed": result.passed,
        "reason": result.reason,
        "elapsed_seconds": result.elapsed_seconds,
        "error": result.error,
        "checks": checks,
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _ensure_json(value: object) -> None:
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("benchmark result contains a non-JSON value") from exc


def write_report(
    path: str | os.PathLike[str],
    report: BenchmarkReport,
    *,
    secrets: Iterable[str] = (),
    indent: int | None = 2,
) -> Path:
    """Write one report atomically, creating the destination parent."""

    target = Path(path).expanduser().resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = report.to_json(secrets=secrets, indent=indent) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


# Short aliases keep the public surface approachable for small experiment
# scripts while the longer names remain self-documenting in the implementation.
BenchmarkConfig = BenchmarkBudget
TaskSpec = BenchmarkTask
AgentSpec = BenchmarkAgent
BenchmarkResult = BenchmarkRunResult


def load_manifest(path: str | os.PathLike[str]) -> BenchmarkManifest:
    """Convenience wrapper used by notebooks and benchmark scripts."""

    return BenchmarkManifest.load(path)


def run_benchmark(
    manifest: BenchmarkManifest,
    *,
    invoker: AgentInvoker | None = None,
    dry_run: bool = False,
    workspace_root: str | os.PathLike[str] | None = None,
    preserve_workspaces: bool = False,
    secrets: Iterable[str] = (),
) -> BenchmarkReport:
    """Run or plan a manifest with one expression.

    ``dry_run=True`` is intentionally explicit here; the CLI uses it by
    default, while library callers that ask for execution opt in by omission.
    """

    runner = BenchmarkRunner(
        manifest,
        invoker=invoker,
        workspace_root=workspace_root,
        preserve_workspaces=preserve_workspaces,
        secrets=secrets,
    )
    return runner.plan() if dry_run else runner.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent-benchmark",
        description=(
            "Plan or run a bounded, fixture-based coding-agent benchmark. "
            "Planning is the default; use --execute to launch commands."
        ),
    )
    parser.add_argument(
        "--manifest", required=True, metavar="PATH", help="JSON benchmark manifest"
    )
    parser.add_argument(
        "--output", metavar="PATH", help="write the JSON report to this path"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        action="store_true",
        help="launch agent commands (planning is the default)",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the plan without launching commands",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="retain per-run copied workspaces",
    )
    parser.add_argument(
        "--workspace-root",
        metavar="PATH",
        help="directory for retained/temporary workspaces",
    )
    parser.add_argument(
        "--task",
        dest="task_ids",
        action="append",
        help="run only this task ID (repeatable)",
    )
    parser.add_argument(
        "--agent",
        dest="agent_ids",
        action="append",
        help="run only this agent ID (repeatable)",
    )
    parser.add_argument("--repetitions", type=int, help="override manifest repetitions")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; no real agent command runs unless ``--execute`` is set."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        manifest = BenchmarkManifest.load(arguments.manifest)
        manifest = manifest.with_filters(
            task_ids=arguments.task_ids or (),
            agent_ids=arguments.agent_ids or (),
        )
        if arguments.repetitions is not None:
            manifest = replace(manifest, repetitions=arguments.repetitions)
        runner = BenchmarkRunner(
            manifest,
            workspace_root=arguments.workspace_root,
            preserve_workspaces=arguments.keep_workspaces,
        )
        report = runner.run() if arguments.execute else runner.plan()
        output = report.to_json(indent=None if arguments.compact else 2)
        if arguments.output:
            write_report(
                arguments.output, report, indent=None if arguments.compact else 2
            )
        else:
            print(output)
        # A completed benchmark is a successful *measurement* even when some
        # tasks remain unresolved; the per-row status and summary carry that
        # outcome.  Exit 2 is reserved for malformed invocation/configuration.
        return 0
    except (BenchmarkError, OSError) as exc:
        print(f"benchmark error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
