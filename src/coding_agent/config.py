"""Runtime configuration and validation.

This module deliberately does *not* load ``.env`` files.  A coding agent can
read and modify arbitrary project files, so implicitly interpreting a file in
the selected workspace as process configuration would create a surprising
trust boundary.  The caller must provide values through explicit CLI options
or through the environment inherited by the CLI process.

The API key is resolved indirectly: users pass the *name* of an environment
variable, never the secret itself as a command-line option.  Command-line
arguments are commonly stored in shell history and are visible in process
listings, which makes ``--api-key VALUE`` an unsafe interface.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from coding_agent.errors import ConfigurationError


class Provider(str, Enum):
    """Model-service presets understood by the first release.

    A provider selects only operational defaults such as the official base URL
    and conventional key-variable name.  It intentionally does not select a
    model.  Model aliases change independently of endpoints, and benchmark
    runs need the chosen model to remain explicit in their command line.
    """

    DEEPSEEK = "deepseek"
    GLM = "glm"
    CUSTOM = "custom"


# These are OpenAI-compatible API roots, not chat-completion endpoint URLs.
# The OpenAI client appends the resource path itself.
#
# Sources checked on 2026-08-28:
# - https://api-docs.deepseek.com/quick_start/pricing
# - https://docs.bigmodel.cn/cn/guide/develop/openai/introduction
#
# Trailing slashes are removed by ``_normalise_base_url`` so an explicit URL
# and its provider preset compare identically.
PROVIDER_BASE_URLS: Mapping[Provider, str] = {
    Provider.DEEPSEEK: "https://api.deepseek.com",
    Provider.GLM: "https://open.bigmodel.cn/api/paas/v4",
}


# These are merely conventional names.  ``--key-env`` can point at any valid
# environment-variable name, which is useful for side-by-side benchmark runs.
PROVIDER_KEY_ENV_NAMES: Mapping[Provider, str] = {
    Provider.DEEPSEEK: "DEEPSEEK_API_KEY",
    Provider.GLM: "ZAI_API_KEY",
    Provider.CUSTOM: "CODING_AGENT_API_KEY",
}


_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Validated, immutable configuration used by the runtime.

    ``api_key`` is excluded from the generated representation and equality.
    Excluding it from equality avoids secret-dependent diagnostics when config
    objects are compared in tests.  Callers must still avoid ``asdict`` or
    manually logging attributes; :meth:`redacted_summary` is the supported
    representation for traces and diagnostics.

    Paths are absolute after construction.  Relative trace paths are resolved
    against the CLI process's current directory, whereas ``workspace`` is the
    directory against which local coding tools resolve project paths.
    """

    task: str
    workspace: Path
    provider: Provider
    model: str
    base_url: str
    api_key_env: str
    api_key: str = field(repr=False, compare=False)
    trace_path: Path | None = None

    # Hard budgets bound autonomous work.  They are checked by the runtime
    # before the corresponding model/tool operation, not only afterwards.
    max_model_turns: int = 20
    max_tool_calls: int = 80
    max_wall_time_seconds: float = 600.0

    # Context and transport limits are separate: the former bounds what is sent
    # in one request, while the latter bounds how long that request may block.
    context_char_budget: int = 120_000
    model_timeout_seconds: float = 120.0
    model_max_retries: int = 2
    protocol_max_retries: int = 1

    @classmethod
    def from_sources(
        cls,
        *,
        task: str | None = None,
        workspace: str | os.PathLike[str] | None = None,
        provider: str | Provider | None = None,
        model: str | None = None,
        base_url: str | None = None,
        key_env: str | None = None,
        trace_path: str | os.PathLike[str] | None = None,
        max_model_turns: int | str | None = None,
        max_tool_calls: int | str | None = None,
        max_wall_time_seconds: float | str | None = None,
        context_char_budget: int | str | None = None,
        model_timeout_seconds: float | str | None = None,
        model_max_retries: int | str | None = None,
        protocol_max_retries: int | str | None = None,
        environ: Mapping[str, str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ) -> AgentConfig:
        """Resolve CLI overrides and environment variables, then validate.

        Each non-``None`` keyword represents an explicit CLI value and wins
        over its environment counterpart.  Empty strings are not treated as
        useful values: they reach validation and produce a focused error rather
        than silently falling through to an unrelated default.

        ``environ`` and ``cwd`` are injectable to make configuration tests
        deterministic.  Passing an empty mapping is meaningful, so this method
        must not use ``environ or os.environ``.
        """

        source_env = os.environ if environ is None else environ
        process_cwd = Path.cwd() if cwd is None else Path(cwd)

        resolved_task = _choose(task, source_env, "CODING_AGENT_TASK")
        resolved_workspace = _choose(
            workspace, source_env, "CODING_AGENT_WORKSPACE", process_cwd
        )
        resolved_provider = _parse_provider(
            _choose(provider, source_env, "CODING_AGENT_PROVIDER")
        )
        resolved_model = _require_nonempty_text(
            _choose(model, source_env, "CODING_AGENT_MODEL"), "model"
        )

        explicit_base_url = _choose(base_url, source_env, "CODING_AGENT_BASE_URL")
        if explicit_base_url is None:
            explicit_base_url = PROVIDER_BASE_URLS.get(resolved_provider)
        if explicit_base_url is None:
            raise ConfigurationError(
                "custom provider requires --base-url or CODING_AGENT_BASE_URL"
            )

        resolved_key_env = _choose(
            key_env,
            source_env,
            "CODING_AGENT_KEY_ENV",
            PROVIDER_KEY_ENV_NAMES[resolved_provider],
        )
        resolved_key_env = _validate_environment_name(resolved_key_env)
        resolved_api_key = source_env.get(resolved_key_env)
        if resolved_api_key is None or not resolved_api_key.strip():
            # Mention only the variable name.  Including a partially supplied
            # value in an exception can leak it into terminal logs or JSONL.
            raise ConfigurationError(
                f"API key environment variable {resolved_key_env!r} is not set"
            )

        resolved_trace = _choose(trace_path, source_env, "CODING_AGENT_TRACE_PATH")

        return cls(
            task=_require_nonempty_text(resolved_task, "task"),
            workspace=_validate_workspace(resolved_workspace, process_cwd),
            provider=resolved_provider,
            model=resolved_model,
            base_url=_normalise_base_url(explicit_base_url),
            api_key_env=resolved_key_env,
            api_key=resolved_api_key,
            trace_path=_resolve_optional_path(resolved_trace, process_cwd),
            max_model_turns=_positive_int(
                _choose(
                    max_model_turns,
                    source_env,
                    "CODING_AGENT_MAX_MODEL_TURNS",
                    20,
                ),
                "max_model_turns",
            ),
            max_tool_calls=_positive_int(
                _choose(
                    max_tool_calls,
                    source_env,
                    "CODING_AGENT_MAX_TOOL_CALLS",
                    80,
                ),
                "max_tool_calls",
            ),
            max_wall_time_seconds=_positive_float(
                _choose(
                    max_wall_time_seconds,
                    source_env,
                    "CODING_AGENT_MAX_WALL_TIME",
                    600.0,
                ),
                "max_wall_time_seconds",
            ),
            context_char_budget=_positive_int(
                _choose(
                    context_char_budget,
                    source_env,
                    "CODING_AGENT_CONTEXT_CHAR_BUDGET",
                    120_000,
                ),
                "context_char_budget",
            ),
            model_timeout_seconds=_positive_float(
                _choose(
                    model_timeout_seconds,
                    source_env,
                    "CODING_AGENT_MODEL_TIMEOUT",
                    120.0,
                ),
                "model_timeout_seconds",
            ),
            model_max_retries=_nonnegative_int(
                _choose(
                    model_max_retries,
                    source_env,
                    "CODING_AGENT_MODEL_MAX_RETRIES",
                    2,
                ),
                "model_max_retries",
            ),
            protocol_max_retries=_nonnegative_int(
                _choose(
                    protocol_max_retries,
                    source_env,
                    "CODING_AGENT_PROTOCOL_MAX_RETRIES",
                    1,
                ),
                "protocol_max_retries",
            ),
        )

    def redacted_summary(self) -> dict[str, object]:
        """Return log-safe configuration metadata without the credential."""

        return {
            "task": self.task,
            "workspace": str(self.workspace),
            "provider": self.provider.value,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "api_key_present": True,
            "trace_path": (
                str(self.trace_path) if self.trace_path is not None else None
            ),
            "max_model_turns": self.max_model_turns,
            "max_tool_calls": self.max_tool_calls,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "context_char_budget": self.context_char_budget,
            "model_timeout_seconds": self.model_timeout_seconds,
            "model_max_retries": self.model_max_retries,
            "protocol_max_retries": self.protocol_max_retries,
        }


def _choose(
    explicit: object | None,
    environ: Mapping[str, str],
    env_name: str,
    default: object | None = None,
) -> object | None:
    """Apply the single precedence rule used by every setting."""

    if explicit is not None:
        return explicit
    if env_name in environ:
        return environ[env_name]
    return default


def _parse_provider(value: object | None) -> Provider:
    if isinstance(value, Provider):
        return value
    if value is None:
        raise ConfigurationError(
            "provider is required; choose deepseek, glm, or custom"
        )
    try:
        return Provider(str(value).strip().lower())
    except ValueError as exc:
        choices = ", ".join(provider.value for provider in Provider)
        raise ConfigurationError(
            f"unknown provider {value!r}; expected one of: {choices}"
        ) from exc


def _require_nonempty_text(value: object | None, field_name: str) -> str:
    if value is None:
        raise ConfigurationError(f"{field_name} is required")
    text = str(value).strip()
    if not text:
        raise ConfigurationError(f"{field_name} must not be empty")
    return text


def _validate_environment_name(value: object | None) -> str:
    name = _require_nonempty_text(value, "key_env")
    if not _ENV_NAME_RE.fullmatch(name):
        raise ConfigurationError(
            "key_env must be an environment-variable name, not a key value"
        )
    return name


def _validate_workspace(value: object | None, cwd: Path) -> Path:
    text = _require_nonempty_text(value, "workspace")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ConfigurationError(f"workspace does not exist: {path}") from exc
    if not resolved.is_dir():
        raise ConfigurationError(f"workspace is not a directory: {resolved}")
    return resolved


def _resolve_optional_path(value: object | None, cwd: Path) -> Path | None:
    if value is None:
        return None
    text = _require_nonempty_text(value, "trace_path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    # ``strict=False`` is intentional: the event sink creates a new trace file.
    return path.resolve(strict=False)


def _normalise_base_url(value: object) -> str:
    text = _require_nonempty_text(value, "base_url")
    try:
        parsed = urlsplit(text)
        hostname = parsed.hostname
        # Accessing ``port`` performs urllib's range and numeric validation.
        # Without it, a URL such as https://example.test:not-a-port would pass
        # this layer and fail later inside the provider client.
        # Assigning to the throwaway name makes the intentional validation
        # access explicit; ``SplitResult.port`` raises for malformed values.
        _ = parsed.port
    except ValueError as exc:
        raise ConfigurationError("base_url is malformed") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError("base_url must be an absolute http:// or https:// URL")
    if hostname is None:
        raise ConfigurationError("base_url must contain a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigurationError("base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ConfigurationError("base_url must not contain query or fragment")
    if parsed.scheme == "http" and not _is_loopback_host(hostname):
        # Sending an API key over plaintext HTTP is never an acceptable remote
        # default.  Loopback remains available for a local fake server or model
        # gateway used during deterministic development tests.
        raise ConfigurationError(
            "base_url must use https unless it targets the local machine"
        )

    # Preserve any provider-specific path (GLM uses /api/paas/v4) while making
    # trailing-slash variants deterministic for equality and logging.
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower().rstrip(".") == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _positive_int(value: object, field_name: str) -> int:
    parsed = _parse_int(value, field_name)
    if parsed <= 0:
        raise ConfigurationError(f"{field_name} must be greater than zero")
    return parsed


def _nonnegative_int(value: object, field_name: str) -> int:
    parsed = _parse_int(value, field_name)
    if parsed < 0:
        raise ConfigurationError(f"{field_name} must be zero or greater")
    return parsed


def _parse_int(value: object, field_name: str) -> int:
    # bool is a subclass of int, but accepting True as a budget of one is an
    # accidental and confusing Python coercion.
    if isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ConfigurationError(f"{field_name} must be an integer")
    return parsed


def _positive_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{field_name} must be a number") from exc
    # NaN passes ordinary comparisons, so test positivity in this direction.
    if not parsed > 0:
        raise ConfigurationError(f"{field_name} must be greater than zero")
    if parsed == float("inf"):
        raise ConfigurationError(f"{field_name} must be finite")
    return parsed
