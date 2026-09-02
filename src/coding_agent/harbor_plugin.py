"""Official Harbor 0.22 entry point for the course coding agent.

The project deliberately keeps Harbor out of its core dependency graph.  This
module is the optional boundary loaded by Harbor with an import path such as
``coding_agent.harbor_plugin:CourseCodingAgent``.  When Harbor is installed,
``CourseCodingAgent`` subclasses its real ``BaseAgent``.  Importing the module
without Harbor remains safe so the normal CLI and test suite do not acquire a
large container-orchestration dependency.

Model requests are made by the host process.  The six coding tools are backed
by :class:`coding_agent.harbor_adapter.RemoteExecutionBackend`, which forwards
only validated operations to the task environment.  Credentials are resolved
from a named host environment variable and are never included in an
environment command, context metadata, or artifact document.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from coding_agent import __version__
from coding_agent.agent import AgentRuntime
from coding_agent.config import (
    PROVIDER_BASE_URLS,
    PROVIDER_KEY_ENV_NAMES,
    Provider,
    ReasoningEffort,
)
from coding_agent.context import ContextBuilder
from coding_agent.errors import ConfigurationError
from coding_agent.events import JsonlEventSink, redact
from coding_agent.harbor_adapter import (
    HarborAgentAdapter,
    RemoteExecutionBackend,
)
from coding_agent.model import OpenAICompatibleModelClient, ReasoningCapability
from coding_agent.policy import AgentLimits
from coding_agent.tools.shell import is_sensitive_environment_name

try:  # pragma: no cover - exercised by the optional-dependency test matrix
    from harbor.agents.base import BaseAgent as _HarborBaseAgent
    from harbor.environments.base import BaseEnvironment as _HarborEnvironment
    from harbor.models.agent.context import AgentContext as _HarborAgentContext

    HARBOR_AVAILABLE = True
except ImportError:  # pragma: no cover - the normal core installation path
    HARBOR_AVAILABLE = False

    class _HarborBaseAgent:
        """Small local stand-in used only when the optional package is absent."""

        def __init__(
            self,
            logs_dir: Path | str | None = None,
            model_name: str | None = None,
            logger: logging.Logger | None = None,
            *,
            extra_env: Mapping[str, str] | None = None,
            **_: Any,
        ) -> None:
            self.logs_dir = Path(logs_dir or Path.cwd())
            self.model_name = model_name
            self.logger = logger or logging.getLogger(__name__)
            self._extra_env = dict(extra_env or {})
            self.environment_logs_dir = Path("/logs/agent")
            self.session_id: str | None = None

        @property
        def extra_env(self) -> dict[str, str]:
            return dict(self._extra_env)

    _HarborEnvironment = Any
    _HarborAgentContext = Any


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DEFAULT_REMOTE_WORKSPACE = "/app"
_DEFAULT_LIMITS = {
    "max_model_turns": 20,
    "max_tool_calls": 80,
    # Keep the direct Harbor entry point aligned with the Terminal-Bench
    # experiment's explicit per-task cap.  Callers may still override this
    # through an agent kwarg or CODING_AGENT_MAX_WALL_TIME.
    "max_wall_time_seconds": 900.0,
    "context_char_budget": 120_000,
    "model_timeout_seconds": 120.0,
    "model_max_retries": 2,
    "protocol_max_retries": 1,
    "convergence_remaining_turns": 5,
    "max_repeated_tool_batches": 2,
    "max_no_progress_batches": 2,
}
_REASONING_PROBE_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class _PluginSettings:
    model: str
    base_url: str
    key_env: str
    api_key: str = field(repr=False, compare=False)
    remote_workspace: str
    max_model_turns: int
    max_tool_calls: int
    max_wall_time_seconds: float
    context_char_budget: int
    model_timeout_seconds: float
    model_max_retries: int
    protocol_max_retries: int
    temperature: float | None
    reasoning_effort: str | None
    reasoning_parameter: str
    reasoning_value: Any | None
    reasoning_capability_status: str | None
    efficiency_mode: bool
    reserve_final_turn: bool
    convergence_remaining_turns: int
    max_repeated_tool_batches: int
    max_no_progress_batches: int


class _RecordingEventSink:
    """Keep a compact, already-redacted event list for ATIF metadata."""

    def __init__(self, secrets: Sequence[str] = ()) -> None:
        self._secrets = tuple(
            item for item in secrets if isinstance(item, str) and item
        )
        self.records: list[dict[str, Any]] = []

    def emit(self, event_type: str, **data: object) -> None:
        safe = redact(data, secrets=self._secrets)
        self.records.append(
            {
                "event": event_type,
                "data": safe if isinstance(safe, Mapping) else {},
            }
        )


class CourseCodingAgent(_HarborBaseAgent):
    """Harbor 0.22 ``BaseAgent`` implementation backed by ``AgentRuntime``.

    Configuration follows the project's normal environment convention.  The
    exact model identifier is taken from Harbor's ``model_name`` argument when
    supplied, otherwise from ``CODING_AGENT_MODEL``.  No model alias is
    guessed or rewritten.  Runtime limits can be passed as agent kwargs or
    through their corresponding ``CODING_AGENT_*`` variables.

    The optional factory arguments are intentionally small dependency-injection
    seams for contract tests and local integrations; Harbor itself does not
    need to provide them.
    """

    SUPPORTS_ATIF = True
    SUPPORTS_WINDOWS = False

    def __init__(
        self,
        logs_dir: Path | str | None = None,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        *,
        provider: str | Provider | None = None,
        base_url: str | None = None,
        key_env: str | None = None,
        remote_workspace: str = _DEFAULT_REMOTE_WORKSPACE,
        max_model_turns: int | str | None = None,
        max_tool_calls: int | str | None = None,
        max_wall_time_seconds: float | str | None = None,
        context_char_budget: int | str | None = None,
        model_timeout_seconds: float | str | None = None,
        model_max_retries: int | str | None = None,
        protocol_max_retries: int | str | None = None,
        temperature: float | str | None = None,
        reasoning_effort: str | ReasoningEffort | None = None,
        reasoning_parameter: str | None = None,
        reasoning_value: Any | None = None,
        reasoning_capability_status: str | None = None,
        efficiency_mode: bool | str | None = None,
        reserve_final_turn: bool | str | None = None,
        convergence_remaining_turns: int | str | None = None,
        max_repeated_tool_batches: int | str | None = None,
        max_no_progress_batches: int | str | None = None,
        extra_env: Mapping[str, str] | None = None,
        model_client_factory: Callable[..., Any] | None = None,
        runtime_factory: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Passing a credential value as an agent kwarg is an unsafe and
        # unsupported configuration.  Fail without echoing the value.
        if "api_key" in kwargs or "api_key_value" in kwargs:
            raise ConfigurationError(
                "credentials must be supplied through a named environment variable"
            )

        self._provider = provider
        self._base_url = base_url
        self._key_env = key_env
        self._remote_workspace = remote_workspace
        # Harbor scopes ``BaseAgent.extra_env`` over every environment.exec()
        # call during an agent phase. Keep the original mapping for host-side
        # configuration resolution, but never expose credential-shaped entries
        # through the BaseAgent property that Harbor applies to the container.
        self._host_extra_env = {
            str(key): value
            for key, value in (extra_env or {}).items()
            if isinstance(key, str) and isinstance(value, str)
        }
        selected_key_env = key_env
        if selected_key_env is None and isinstance(provider, Provider):
            selected_key_env = PROVIDER_KEY_ENV_NAMES.get(provider)
        if selected_key_env is None and isinstance(provider, str):
            try:
                selected_key_env = PROVIDER_KEY_ENV_NAMES.get(
                    Provider(provider.strip().lower())
                )
            except ValueError:
                selected_key_env = None
        if selected_key_env is None:
            selected_key_env = self._host_extra_env.get("CODING_AGENT_KEY_ENV")
        safe_extra_env = {
            name: value
            for name, value in self._host_extra_env.items()
            if name != selected_key_env and not is_sensitive_environment_name(name)
        }
        self._option_values = {
            "max_model_turns": max_model_turns,
            "max_tool_calls": max_tool_calls,
            "max_wall_time_seconds": max_wall_time_seconds,
            "context_char_budget": context_char_budget,
            "model_timeout_seconds": model_timeout_seconds,
            "model_max_retries": model_max_retries,
            "protocol_max_retries": protocol_max_retries,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "reasoning_parameter": reasoning_parameter,
            "reasoning_value": reasoning_value,
            "reasoning_capability_status": reasoning_capability_status,
            "efficiency_mode": efficiency_mode,
            "reserve_final_turn": reserve_final_turn,
            "convergence_remaining_turns": convergence_remaining_turns,
            "max_repeated_tool_batches": max_repeated_tool_batches,
            "max_no_progress_batches": max_no_progress_batches,
        }
        self._model_client_factory = model_client_factory or OpenAICompatibleModelClient
        self._runtime_factory = runtime_factory or AgentRuntime
        self._last_result: Any | None = None
        self._cancel_event: threading.Event | None = None
        self._probed_reasoning_capability: ReasoningCapability | None = None
        self._probed_model_client: Any | None = None

        # Harbor's constructor accepts several optional bookkeeping kwargs.
        # Keep those intact while ensuring plugin-only options never reach a
        # stricter fake BaseAgent used by integration tests.
        super_kwargs = dict(kwargs)
        super_kwargs["extra_env"] = safe_extra_env
        resolved_logs_dir = Path(logs_dir) if logs_dir is not None else Path.cwd()
        try:
            super().__init__(
                logs_dir=resolved_logs_dir,
                model_name=model_name,
                logger=logger,
                **super_kwargs,
            )
        except TypeError:
            # A tiny compatibility fallback is useful for Harbor wrappers that
            # expose only the four common constructor fields.
            minimal = {
                key: super_kwargs[key] for key in ("extra_env",) if key in super_kwargs
            }
            super().__init__(
                logs_dir=resolved_logs_dir,
                model_name=model_name,
                logger=logger,
                **minimal,
            )

    @staticmethod
    def name() -> str:
        return "course-coding-agent"

    def version(self) -> str:
        return __version__

    async def setup(self, environment: _HarborEnvironment) -> None:
        """Prepare only the host log directory.

        Harbor task images already provide the task workspace.  Avoiding a
        setup command keeps this phase side-effect free and, importantly,
        prevents a credential-bearing host environment from crossing into the
        container before ``run`` has validated the model configuration.
        """

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        # A direct Harbor invocation may not have gone through the benchmark
        # runner's preflight probe. Resolve the host-only settings and perform
        # that bounded probe here so an unverified reasoning field can never
        # reach the gateway. Unsupported is a recorded capability result and
        # deliberately continues without the optional field.
        settings = self._resolve_settings()
        await self._ensure_reasoning_probe(settings)

    async def run(
        self,
        instruction: str,
        environment: _HarborEnvironment,
        context: _HarborAgentContext,
    ) -> None:
        """Run the shared Runtime and populate Harbor's public context."""

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        if context is None:
            raise TypeError("context is required")

        settings = self._resolve_settings()
        await self._ensure_reasoning_probe(settings)
        # The probe may have selected a provider-specific field/value or marked
        # the option unsupported, so resolve once more after applying it.
        settings = self._resolve_settings()
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._cancel_event = threading.Event()
        recorder = _RecordingEventSink((settings.api_key,))

        # The JSONL file is diagnostic only.  If a Harbor log mount is
        # read-only, retaining the in-memory sink still lets the Runtime finish
        # and the ATIF artifact writer report a useful result.
        event_sink: Any = recorder
        try:
            event_sink = _CompositeRecordingSink(
                recorder,
                JsonlEventSink(
                    self.logs_dir / "events.jsonl", secrets=(settings.api_key,)
                ),
            )
        except OSError:
            pass

        backend = RemoteExecutionBackend(
            environment,
            workspace=settings.remote_workspace,
        )
        model_values: dict[str, Any] = {
            "model": settings.model,
            "api_key": settings.api_key,
            "base_url": settings.base_url,
            "timeout_seconds": settings.model_timeout_seconds,
            "temperature": settings.temperature,
        }
        # Keep the factory call itself free of reasoning fields when the probe
        # explicitly found that the gateway does not support them.  The model
        # adapter consequently cannot accidentally add a provider-specific
        # field later in request construction.
        if (
            settings.reasoning_effort is not None
            or settings.reasoning_value is not None
        ):
            # A probe may select a provider-specific field even when its
            # accepted value is represented separately (for example a boolean
            # ``thinking`` flag). Always carry the field name alongside that
            # value so the adapter cannot fall back to ``reasoning_effort``.
            model_values["reasoning_parameter"] = settings.reasoning_parameter
        if settings.reasoning_effort is not None:
            model_values["reasoning_effort"] = settings.reasoning_effort
        if settings.reasoning_value is not None:
            model_values["reasoning_value"] = settings.reasoning_value
        model_client = self._probed_model_client
        if model_client is None:
            model_client = _invoke_factory(self._model_client_factory, model_values)
        runtime = _invoke_factory(
            self._runtime_factory,
            {
                "model_client": model_client,
                "tool_registry": backend,
                "context_builder": ContextBuilder(
                    max_chars=settings.context_char_budget
                ),
                "limits": AgentLimits(
                    max_model_turns=settings.max_model_turns,
                    max_tool_calls=settings.max_tool_calls,
                    max_wall_time_seconds=settings.max_wall_time_seconds,
                    max_model_retries=settings.model_max_retries,
                    max_protocol_retries=settings.protocol_max_retries,
                    efficiency_mode=settings.efficiency_mode,
                    reserve_final_turn=settings.reserve_final_turn,
                    convergence_remaining_turns=settings.convergence_remaining_turns,
                    max_repeated_tool_batches=settings.max_repeated_tool_batches,
                    max_no_progress_batches=settings.max_no_progress_batches,
                ),
                "event_sink": event_sink,
                "cancel_check": self._cancel_event.is_set,
            },
        )

        adapter = HarborAgentAdapter(
            runtime,
            backend=backend,
            artifact_dir=self.logs_dir,
            model_name=settings.model,
            run_id=getattr(self, "session_id", None),
            tool_definitions=backend.model_schemas(),
            events_provider=lambda: tuple(recorder.records),
            secrets=(settings.api_key,),
            cancel_event=self._cancel_event,
            artifact_metadata={
                "reasoning_effort": settings.reasoning_effort,
                "reasoning_parameter": settings.reasoning_parameter,
                "reasoning_value": settings.reasoning_value,
                "reasoning_capability_status": settings.reasoning_capability_status,
                "reasoning_capability": _json_safe_capability(
                    model_client, fallback=self._probed_reasoning_capability
                ),
                "efficiency_mode": settings.efficiency_mode,
                "max_model_turns": settings.max_model_turns,
                "max_tool_calls": settings.max_tool_calls,
                "agent_timeout_seconds": settings.max_wall_time_seconds,
            },
        )
        try:
            result = await adapter.run(instruction.strip())
        except asyncio.CancelledError:
            # AgentRuntime polls this event between protocol operations.  The
            # cancellation is re-raised so Harbor records its normal timeout /
            # cancellation outcome, while no later tool batch is admitted.
            self._cancel_event.set()
            raise
        finally:
            self._cancel_event = None

        self._last_result = result
        self._populate_context(
            context,
            result,
            model=settings.model,
            key_env=settings.key_env,
            secrets=(settings.api_key,),
            event_count=len(recorder.records),
            extra_metadata={
                "reasoning_effort": settings.reasoning_effort,
                "reasoning_parameter": settings.reasoning_parameter,
                "reasoning_value": settings.reasoning_value,
                "reasoning_capability_status": settings.reasoning_capability_status,
                "reasoning_capability": _json_safe_capability(
                    model_client, fallback=self._probed_reasoning_capability
                ),
                "efficiency_mode": settings.efficiency_mode,
                "reserve_final_turn": settings.reserve_final_turn,
            },
        )

    async def _ensure_reasoning_probe(self, settings: _PluginSettings) -> None:
        """Ensure an optional native reasoning value was verified once.

        The model client is synchronous, so the probe runs in a worker thread
        while Harbor's event loop remains responsive.  A caller that supplied
        ``reasoning_capability_status`` has already performed the probe in the
        benchmark runner and is trusted only after the status has passed the
        strict validation in :meth:`_raw_reasoning_capability_status`.
        """

        requested = settings.reasoning_effort
        if requested is None and settings.reasoning_value is None:
            return
        if settings.reasoning_capability_status is not None:
            if settings.reasoning_capability_status == "error":
                raise ConfigurationError(
                    "reasoning capability probe failed; refusing an unverified option"
                )
            return
        if self._probed_reasoning_capability is not None:
            if self._probed_reasoning_capability.status == "error":
                raise ConfigurationError(
                    "reasoning capability probe failed; refusing an unverified option"
                )
            return

        probe_values: dict[str, Any] = {
            "model": settings.model,
            "api_key": settings.api_key,
            "base_url": settings.base_url,
            "timeout_seconds": settings.model_timeout_seconds,
        }
        try:
            client = await asyncio.to_thread(
                _invoke_factory, self._model_client_factory, probe_values
            )
            probe = getattr(client, "probe_reasoning_effort", None)
            if not callable(probe):
                raise ConfigurationError(
                    "model client does not expose probe_reasoning_effort"
                )
            effort = requested or "high"
            probe_kwargs: dict[str, Any] = {
                "timeout_seconds": min(
                    float(settings.model_timeout_seconds),
                    _REASONING_PROBE_TIMEOUT_SECONDS,
                )
            }
            if settings.reasoning_parameter != "reasoning_effort":
                probe_kwargs["parameter_candidates"] = (settings.reasoning_parameter,)
            capability_raw = await asyncio.to_thread(
                _invoke_reasoning_probe, probe, effort, probe_kwargs
            )
            capability = _coerce_reasoning_capability(capability_raw, effort)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError("reasoning capability probe failed") from exc

        self._probed_reasoning_capability = capability
        self._option_values["reasoning_capability_status"] = capability.status
        if capability.status == "error":
            raise ConfigurationError(
                "reasoning capability probe failed; refusing an unverified option"
            )
        if capability.status == "unsupported":
            # ``_resolve_settings`` removes any inherited reasoning value after
            # seeing this explicit status. Do not retain the probe client: its
            # constructor may have been given a reasoning field that the
            # gateway rejected.
            self._probed_model_client = None
            return

        if not capability.parameter or capability.accepted_value is None:
            raise ConfigurationError(
                "reasoning capability probe returned no native field/value"
            )
        self._option_values["reasoning_parameter"] = capability.parameter
        if isinstance(capability.accepted_value, str) and capability.accepted_value in {
            item.value for item in ReasoningEffort
        }:
            self._option_values["reasoning_effort"] = capability.accepted_value
            # An explicit empty value suppresses a stale process environment
            # override in ``_resolve_settings`` while still parsing as absent.
            self._option_values["reasoning_value"] = ""
        else:
            self._option_values["reasoning_effort"] = (
                capability.requested_effort or requested or "high"
            )
            self._option_values["reasoning_value"] = capability.accepted_value

        configure = getattr(client, "configure_reasoning", None)
        if callable(configure):
            try:
                configure(capability)
            except Exception as exc:
                raise ConfigurationError(
                    "reasoning capability application failed"
                ) from exc
            self._probed_model_client = client
        elif (
            capability.parameter == settings.reasoning_parameter
            and capability.accepted_value == (requested or "high")
        ):
            # A narrow injected client may not expose a configure hook, but it
            # was constructed with the exact value that the probe accepted.
            self._probed_model_client = client
        else:
            # Reconstructing below with the selected scalar/JSON value is safer
            # than assuming a client without a configure hook mutated itself.
            self._probed_model_client = None

    def _resolve_settings(self) -> _PluginSettings:
        # Harbor's extra_env is an explicit higher-precedence source.  Values
        # are consulted only in this host process and are never passed to
        # RemoteExecutionBackend or environment.exec.
        # Resolve from the host-only copy first. ``extra_env`` is deliberately
        # sanitized before BaseAgent sees it, so it cannot carry a key into the
        # task container through Harbor's scoped execution environment.
        source: dict[str, str] = dict(os.environ)
        if isinstance(self._host_extra_env, Mapping):
            source.update(
                {
                    str(key): value
                    for key, value in self._host_extra_env.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
            )

        provider_value = self._provider
        if provider_value is None:
            provider_value = source.get("CODING_AGENT_PROVIDER", "glm")
        provider = _parse_provider(provider_value)

        model_value = getattr(self, "model_name", None)
        if model_value is None:
            model_value = source.get("CODING_AGENT_MODEL")
        model = _required_text(model_value, "model")

        base_value = self._base_url
        if base_value is None:
            base_value = source.get("CODING_AGENT_BASE_URL")
        if base_value is None:
            base_value = PROVIDER_BASE_URLS.get(provider)
        if base_value is None:
            raise ConfigurationError(
                "custom provider requires base_url or CODING_AGENT_BASE_URL"
            )
        base = _normalise_base_url(base_value)

        key_env_value = self._key_env
        if key_env_value is None:
            key_env_value = source.get("CODING_AGENT_KEY_ENV")
        if key_env_value is None:
            key_env_value = PROVIDER_KEY_ENV_NAMES[provider]
        key_env = _validate_env_name(key_env_value)
        api_key = source.get(key_env)
        if not isinstance(api_key, str) or not api_key.strip():
            raise ConfigurationError(
                f"API key environment variable {key_env!r} is not set"
            )

        reasoning_status = self._raw_reasoning_capability_status(source)
        if reasoning_status == "error":
            raise ConfigurationError(
                "reasoning capability probe failed; refusing an unverified option"
            )

        remote_workspace = self._remote_workspace
        if remote_workspace == _DEFAULT_REMOTE_WORKSPACE:
            remote_workspace = source.get(
                "CODING_AGENT_REMOTE_WORKSPACE", remote_workspace
            )
        if (
            not isinstance(remote_workspace, str)
            or not remote_workspace.startswith("/")
            or "\x00" in remote_workspace
        ):
            raise ConfigurationError("remote_workspace must be an absolute path")

        values: dict[str, Any] = {}
        for name, default in _DEFAULT_LIMITS.items():
            env_name = "CODING_AGENT_" + _ENV_SUFFIXES[name]
            explicit = self._option_values.get(name)
            raw = explicit if explicit is not None else source.get(env_name, default)
            values[name] = _positive_or_nonnegative(name, raw)

        raw_temperature = self._option_values.get("temperature")
        if raw_temperature is None:
            raw_temperature = source.get("CODING_AGENT_TEMPERATURE")
        temperature = _optional_finite_float(raw_temperature, "temperature")

        raw_reasoning = self._option_values.get("reasoning_effort")
        if raw_reasoning is None:
            raw_reasoning = source.get("CODING_AGENT_REASONING_EFFORT")
        reasoning_effort = _optional_reasoning_effort(raw_reasoning)
        if reasoning_status == "unsupported":
            # An environment-level default must not re-enable a field that a
            # route-specific probe explicitly rejected.
            reasoning_effort = None
        raw_reasoning_value = self._option_values.get("reasoning_value")
        if raw_reasoning_value is None:
            raw_reasoning_value = source.get("CODING_AGENT_REASONING_VALUE")
        reasoning_value = _optional_json_value(raw_reasoning_value, "reasoning_value")
        if reasoning_status == "unsupported":
            reasoning_value = None
        reasoning_parameter = self._option_values.get("reasoning_parameter")
        if reasoning_parameter is None:
            reasoning_parameter = source.get(
                "CODING_AGENT_REASONING_PARAMETER", "reasoning_effort"
            )
        if not isinstance(reasoning_parameter, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,63}", reasoning_parameter
        ):
            raise ConfigurationError("reasoning_parameter must be a field name")
        if (
            reasoning_status == "supported"
            and reasoning_effort is None
            and reasoning_value is None
        ):
            raise ConfigurationError(
                "supported reasoning capability has no native value"
            )

        raw_efficiency = self._option_values.get("efficiency_mode")
        if raw_efficiency is None:
            raw_efficiency = source.get("CODING_AGENT_EFFICIENCY")
        efficiency_mode = _optional_bool(raw_efficiency, "efficiency_mode", False)
        raw_reserve = self._option_values.get("reserve_final_turn")
        if raw_reserve is None:
            raw_reserve = source.get("CODING_AGENT_RESERVE_FINAL_TURN")
        reserve_final_turn = _optional_bool(raw_reserve, "reserve_final_turn", False)

        return _PluginSettings(
            model=model,
            base_url=base,
            key_env=key_env,
            api_key=api_key,
            remote_workspace=remote_workspace.rstrip("/") or "/",
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            reasoning_parameter=reasoning_parameter,
            reasoning_value=reasoning_value,
            reasoning_capability_status=reasoning_status,
            efficiency_mode=efficiency_mode,
            reserve_final_turn=reserve_final_turn or efficiency_mode,
            **values,
        )

    def _raw_reasoning_capability_status(
        self, source: Mapping[str, str] | None = None
    ) -> str | None:
        """Resolve and validate the runner's probe status without secrets."""

        values: Mapping[str, Any] = self._option_values
        raw: Any = values.get("reasoning_capability_status")
        if raw is None:
            effective = source
            if effective is None:
                effective = os.environ
                if isinstance(self._host_extra_env, Mapping):
                    merged = dict(effective)
                    merged.update(self._host_extra_env)
                    effective = merged
            raw = effective.get("CODING_AGENT_REASONING_CAPABILITY_STATUS")
            if raw is None:
                raw = effective.get("CODING_AGENT_REASONING_STATUS")
        if raw is None or raw == "":
            return None
        if not isinstance(raw, str) or raw.strip().lower() not in {
            "supported",
            "unsupported",
            "error",
        }:
            raise ConfigurationError(
                "reasoning_capability_status must be supported, unsupported, or error"
            )
        return raw.strip().lower()

    @staticmethod
    def _populate_context(
        context: Any,
        result: Any,
        *,
        model: str,
        key_env: str,
        secrets: Sequence[str],
        event_count: int,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        usage = getattr(result, "usage", None)
        _assign_if_supported(
            context, "n_input_tokens", _usage_value(usage, "prompt_tokens")
        )
        _assign_if_supported(
            context,
            "n_output_tokens",
            _usage_value(usage, "completion_tokens"),
        )
        _assign_if_supported(
            context,
            "n_cache_tokens",
            _usage_value(usage, "cached_tokens"),
        )
        metadata: dict[str, Any] = {}
        existing = getattr(context, "metadata", None)
        if isinstance(existing, Mapping):
            metadata.update(existing)
        metadata.update(dict(extra_metadata or {}))
        metadata.update(
            {
                "agent": CourseCodingAgent.name(),
                "version": __version__,
                "model": model,
                "credential_env": key_env,
                "phase": _enum_value(getattr(result, "phase", None)),
                "reason": getattr(result, "reason", None),
                "final_text": getattr(result, "final_text", None),
                "final_response": getattr(result, "final_text", None),
                "model_turns": getattr(result, "model_turns", None),
                "model_requests": getattr(result, "model_requests", None),
                "tool_calls": getattr(result, "tool_calls", None),
                "elapsed_seconds": getattr(result, "elapsed_seconds", None),
                "trajectory_path": "trajectory.json",
                "summary_path": "run.json",
                "event_count": event_count,
            }
        )
        safe_metadata = redact(metadata, secrets=secrets)
        _assign_if_supported(
            context,
            "metadata",
            dict(safe_metadata) if isinstance(safe_metadata, Mapping) else {},
        )

        # Harbor's AgentContext has no dedicated final-answer field.  Keep the
        # answer in metadata and point to the ATIF file written in logs_dir;
        # callers can consume both through Harbor's public context/result API.
        usage_cost = _usage_value(usage, "cost_usd")
        if usage_cost is not None:
            _assign_if_supported(context, "cost_usd", usage_cost)


class _CompositeRecordingSink:
    """Fan out events while retaining a single in-memory redacted copy."""

    def __init__(self, recorder: _RecordingEventSink, sink: Any) -> None:
        self.recorder = recorder
        self.sink = sink

    def emit(self, event_type: str, **data: object) -> None:
        self.recorder.emit(event_type, **data)
        try:
            self.sink.emit(event_type, **data)
        except OSError:
            # A diagnostic sink must not change Runtime semantics.
            pass


_ENV_SUFFIXES = {
    "max_model_turns": "MAX_MODEL_TURNS",
    "max_tool_calls": "MAX_TOOL_CALLS",
    "max_wall_time_seconds": "MAX_WALL_TIME",
    "context_char_budget": "CONTEXT_CHAR_BUDGET",
    "model_timeout_seconds": "MODEL_TIMEOUT",
    "model_max_retries": "MODEL_MAX_RETRIES",
    "protocol_max_retries": "PROTOCOL_MAX_RETRIES",
    "convergence_remaining_turns": "CONVERGENCE_REMAINING_TURNS",
    "max_repeated_tool_batches": "MAX_REPEATED_TOOL_BATCHES",
    "max_no_progress_batches": "MAX_NO_PROGRESS_BATCHES",
}


def _parse_provider(value: object) -> Provider:
    if isinstance(value, Provider):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("provider must be deepseek, glm, or custom")
    try:
        return Provider(value.strip().lower())
    except ValueError as exc:
        raise ConfigurationError("provider must be deepseek, glm, or custom") from exc


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} is required")
    return value.strip()


def _validate_env_name(value: object) -> str:
    text = _required_text(value, "key_env")
    if _ENV_NAME_RE.fullmatch(text) is None:
        raise ConfigurationError("key_env must be an environment-variable name")
    return text


def _normalise_base_url(value: object) -> str:
    text = _required_text(value, "base_url")
    try:
        parsed = urlsplit(text)
        _ = parsed.port
    except ValueError as exc:
        raise ConfigurationError("base_url is malformed") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError("base_url must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ConfigurationError("base_url must not contain query or fragment")
    if parsed.scheme == "http" and not _is_loopback(parsed.hostname or ""):
        raise ConfigurationError("base_url must use https unless it targets localhost")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _is_loopback(hostname: str) -> bool:
    if hostname.lower().rstrip(".") == "localhost":
        return True
    # Avoid importing a second URL/IP policy module in the common path; the
    # standard library parser is enough for the usual local fake endpoints.
    import ipaddress

    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _positive_or_nonnegative(name: str, value: object) -> int | float:
    if name == "max_wall_time_seconds" or name == "model_timeout_seconds":
        if isinstance(value, bool):
            raise ConfigurationError(f"{name} must be a positive number")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{name} must be a positive number") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise ConfigurationError(f"{name} must be a positive number")
        return parsed

    if isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer")
    try:
        parsed_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ConfigurationError(f"{name} must be an integer")
    if parsed_int < 0 or (
        name in {"max_model_turns", "max_tool_calls", "context_char_budget"}
        and parsed_int == 0
    ):
        raise ConfigurationError(f"{name} must be greater than zero")
    return parsed_int


def _optional_finite_float(value: object, field_name: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ConfigurationError(f"{field_name} must be a finite number")
    return parsed


def _optional_reasoning_effort(value: object) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, ReasoningEffort):
        return value.value
    if not isinstance(value, str) or value.strip().lower() not in {
        item.value for item in ReasoningEffort
    }:
        raise ConfigurationError(
            "reasoning_effort must be one of: low, medium, high, max"
        )
    return value.strip().lower()


def _optional_json_value(value: object, field_name: str) -> Any | None:
    """Decode an optional Harbor kwarg while preserving native JSON types."""

    if value is None or value == "":
        return None
    candidate: Any = value
    if isinstance(value, str):
        try:
            candidate = json.loads(value)
        except json.JSONDecodeError:
            # Plain strings such as ``medium`` are valid native values and are
            # intentionally kept as strings when they are not JSON literals.
            candidate = value
    try:
        json.dumps(candidate, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must be JSON-serializable") from exc
    return candidate


def _optional_bool(value: object, field_name: str, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigurationError(f"{field_name} must be a boolean")


def _usage_value(usage: Any, name: str) -> int | float | None:
    if usage is None:
        return None
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, Mapping):
        value = usage.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _invoke_factory(factory: Callable[..., Any], values: Mapping[str, Any]) -> Any:
    """Call an injected factory while tolerating older narrow signatures.

    The production factories accept all values.  Contract tests and small
    integrations often provide a callable that only models the fields it
    needs; filtering is done from its signature before invocation so a
    ``TypeError`` raised *inside* a factory is never mistaken for a signature
    mismatch and retried.
    """

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**dict(values))
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return factory(**dict(values))
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return factory(**{key: value for key, value in values.items() if key in accepted})


def _invoke_reasoning_probe(
    probe: Callable[..., Any],
    effort: str,
    values: Mapping[str, Any],
) -> Any:
    """Call a probe hook while tolerating older narrow integration fakes."""

    kwargs = dict(values)
    try:
        signature = inspect.signature(probe)
    except (TypeError, ValueError):
        return probe(effort, **kwargs)
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return probe(effort, **kwargs)
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return probe(
        effort, **{key: value for key, value in kwargs.items() if key in accepted}
    )


def _coerce_reasoning_capability(value: Any, requested: str) -> ReasoningCapability:
    """Normalize a probe result into the redaction-safe domain object."""

    if isinstance(value, ReasoningCapability):
        raw: Mapping[str, Any] = value.to_dict()
    elif isinstance(value, Mapping):
        raw = value
    else:
        to_dict = getattr(value, "to_dict", None)
        raw_value = to_dict() if callable(to_dict) else None
        raw = raw_value if isinstance(raw_value, Mapping) else {}
    status = raw.get("status")
    if not isinstance(status, str):
        status = "error"
    status = status.strip().lower()
    if status not in {"supported", "unsupported", "error"}:
        status = "error"
    accepted = raw.get("accepted_value")
    parameter = raw.get("parameter")
    validation_detail: str | None = None
    if parameter is not None and (
        not isinstance(parameter, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", parameter) is None
    ):
        status = "error"
        parameter = None
        validation_detail = "native reasoning parameter is invalid"
    if status == "supported" and accepted is None:
        status = "error"
        validation_detail = validation_detail or "native reasoning value is missing"
    if accepted is not None:
        try:
            json.dumps(accepted, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            status = "error"
            accepted = None
            validation_detail = (
                validation_detail or "native reasoning value is not JSON-serializable"
            )
    requested_value = raw.get("requested_effort")
    if requested_value is not None and (
        not isinstance(requested_value, str)
        or requested_value.strip().lower() not in {"low", "medium", "high", "max"}
    ):
        status = "error"
        requested_value = requested
        validation_detail = validation_detail or "requested reasoning effort is invalid"
    elif isinstance(requested_value, str):
        requested_value = requested_value.strip().lower()
    detail = raw.get("detail")
    if detail is not None:
        detail = str(detail)[:1000]
    if validation_detail is not None:
        detail = validation_detail
    return ReasoningCapability(
        status=status,
        requested_effort=requested_value
        if isinstance(requested_value, str)
        else requested,
        parameter=parameter,
        accepted_value=accepted,
        error_type=(
            raw.get("error_type") if isinstance(raw.get("error_type"), str) else None
        ),
        detail=detail,
    )


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _json_safe_capability(
    client: Any,
    *,
    fallback: ReasoningCapability | None = None,
) -> Mapping[str, Any] | None:
    capability = getattr(client, "reasoning_capability", None) or fallback
    if capability is None:
        return None
    to_dict = getattr(capability, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, Mapping) else None
    return None


def _assign_if_supported(target: Any, name: str, value: Any) -> None:
    """Set a public context field on Pydantic and lightweight fake contexts."""

    try:
        setattr(target, name, value)
        return
    except (AttributeError, TypeError, ValueError):
        pass
    if isinstance(target, dict):
        target[name] = value


__all__ = ["HARBOR_AVAILABLE", "CourseCodingAgent"]
