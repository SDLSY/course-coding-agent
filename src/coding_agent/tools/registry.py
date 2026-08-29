"""Registration, argument validation, and uniform execution of local tools."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from coding_agent.errors import (
    ToolExecutionError as RuntimeToolExecutionError,
)
from coding_agent.errors import (
    ToolRequestError as RuntimeToolRequestError,
)
from coding_agent.response_parser import parse_tool_arguments
from coding_agent.types import ToolCall, ToolResult

from .base import Tool, ToolOutput


class ToolRegistry:
    """Own the complete set of operations exposed to the model.

    The agent runtime must route every invocation through this class.  Besides
    avoiding ad-hoc dispatch, this guarantees that an unknown tool, malformed
    arguments, and a handler failure still produce exactly one ``ToolResult``
    with the original call ID.  That property is more important than raising a
    convenient Python exception: the next model request would otherwise contain
    an orphaned assistant tool call and violate the provider's message protocol.
    """

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register ``tool`` and reject ambiguous duplicate names."""

        if tool.name in self._tools:
            raise ValueError(f"tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return a registered tool without executing it."""

        return self._tools.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        """Registered names in deterministic insertion order."""

        return tuple(self._tools)

    def model_schemas(self) -> list[dict[str, Any]]:
        """Return schemas in the order in which tools were registered."""

        return [tool.as_model_schema() for tool in self._tools.values()]

    # ``schemas`` is a compact alias useful to adapters which already know the
    # values are model-facing function schemas.
    schemas = model_schemas

    def execute_call(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Parse and dispatch a raw model ``ToolCall``.

        Raw argument text stays untouched in canonical history, while this
        method creates a temporary strict-JSON dictionary for execution.  A
        syntax error still becomes exactly one result paired to ``call.id``;
        the model can then repair its request without corrupting the provider
        message sequence.
        """

        try:
            arguments = parse_tool_arguments(call)
        except RuntimeToolRequestError as exc:
            return self._error_result(
                call.id,
                call.name,
                error_code=getattr(exc, "error_code", "invalid_tool_request"),
                content=str(exc),
                metadata={"error_kind": "request"},
            )
        return self.execute(
            call.id,
            call.name,
            arguments,
            timeout_seconds=timeout_seconds,
        )

    def execute(
        self,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any] | Any,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Validate and execute one call, always returning a ``ToolResult``.

        ``arguments`` should already be decoded from the provider's JSON text by
        the response normalizer.  It remains typed as ``Any`` at this trust
        boundary because a model may return a scalar, array, or null instead of
        the required object.  Such input becomes ``invalid_arguments`` rather
        than crashing the runtime.

        ``KeyboardInterrupt`` and ``SystemExit`` deliberately remain uncaught.
        Cancellation belongs to the agent state machine and must not be
        disguised as an ordinary tool failure.
        """

        tool = self._tools.get(name)
        if tool is None:
            return self._error_result(
                call_id,
                name,
                error_code="unknown_tool",
                content=f"Unknown tool: {name!r}.",
                metadata={"error_kind": "request"},
            )

        if not isinstance(arguments, Mapping):
            return self._error_result(
                call_id,
                name,
                error_code="invalid_arguments",
                content="Tool arguments must be a JSON object.",
                metadata={"error_kind": "request"},
            )

        # Copy the mapping before validation and execution.  Apart from
        # normalising custom Mapping implementations, this prevents a caller
        # from mutating the top-level object while the handler is running.
        argument_object = dict(arguments)
        validation_error = _validate_value(
            argument_object,
            tool.parameters,
            path="arguments",
        )
        if validation_error is not None:
            return self._error_result(
                call_id,
                name,
                error_code="invalid_arguments",
                content=validation_error,
                metadata={"error_kind": "request"},
            )

        if timeout_seconds is not None:
            if (
                not isinstance(timeout_seconds, (int, float))
                or isinstance(timeout_seconds, bool)
                or not math.isfinite(float(timeout_seconds))
                or timeout_seconds <= 0
            ):
                raise ValueError("timeout_seconds must be a finite positive number")
            if tool.timeout_argument is not None:
                requested_timeout = argument_object.get(tool.timeout_argument)
                effective_timeout = float(timeout_seconds)
                if requested_timeout is not None:
                    effective_timeout = min(
                        effective_timeout,
                        float(requested_timeout),
                    )
                # Validation above applies to the model's original object.  The
                # internal float cap may be below the schema's model-facing
                # minimum of one second when the whole run has less than a
                # second remaining; subprocess accepts that precise timeout.
                argument_object[tool.timeout_argument] = effective_timeout

        try:
            output = tool.handler(argument_object)
            if not isinstance(output, ToolOutput):
                raise TypeError("tool handler did not return ToolOutput")
            # Construct inside the defensive boundary as well.  A handler which
            # returns an invalid content/metadata type is an implementation bug,
            # but it must not leave the model's call without a paired result.
            return ToolResult(
                call_id=call_id,
                name=name,
                ok=True,
                content=output.content,
                metadata=dict(output.metadata),
                error_code=None,
            )
        except RuntimeToolRequestError as exc:
            return self._error_result(
                call_id,
                name,
                error_code=getattr(exc, "error_code", "invalid_tool_request"),
                content=str(exc),
                metadata={
                    "error_kind": "request",
                    **getattr(exc, "metadata", {}),
                },
            )
        except RuntimeToolExecutionError as exc:
            return self._error_result(
                call_id,
                name,
                error_code=getattr(exc, "error_code", "tool_execution_error"),
                content=str(exc),
                metadata={
                    "error_kind": "execution",
                    **getattr(exc, "metadata", {}),
                },
            )
        except Exception as exc:  # noqa: BLE001 - deliberate trust boundary
            # Do not include ``str(exc)``.  Low-level exceptions may contain an
            # absolute host path or another value which should not be copied to
            # the model/event log.  The exception class is sufficient local
            # diagnostic information for this last-resort wrapper.
            return self._error_result(
                call_id,
                name,
                error_code="tool_execution_error",
                content=f"Tool execution failed unexpectedly ({type(exc).__name__}).",
                metadata={
                    "error_kind": "execution",
                    "exception_type": type(exc).__name__,
                },
            )

    # A semantic alias which reads naturally in callers that use "invoke" for
    # registry dispatch.  There is still only one implementation path.
    invoke = execute

    @staticmethod
    def _error_result(
        call_id: str,
        name: str,
        *,
        error_code: str,
        content: str,
        metadata: Mapping[str, Any],
    ) -> ToolResult:
        return ToolResult(
            call_id=call_id,
            name=name,
            ok=False,
            content=content,
            metadata=dict(metadata),
            error_code=error_code,
        )


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str) -> str | None:
    """Validate the deliberately small JSON-Schema subset used by our tools.

    This is not presented as a general JSON Schema implementation.  Supporting
    an incomplete standard silently would be dangerous, so unknown schema
    structures are avoided in the built-in definitions and the supported
    features are explicit here: ``type``, object properties/required/additional
    properties, array items, numeric bounds, string lengths, and enum.

    The function returns a stable, model-readable error instead of raising.
    Stable messages make both retry behaviour and unit tests deterministic.
    """

    expected = schema.get("type")
    if expected is not None and not _matches_json_type(value, expected):
        if isinstance(expected, Sequence) and not isinstance(expected, str):
            expected_text = " or ".join(str(item) for item in expected)
        else:
            expected_text = str(expected)
        return f"{path} must be of type {expected_text}."

    if "enum" in schema and value not in schema["enum"]:
        return f"{path} must be one of the allowed values."

    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                return f"{path}.{key} is required."

        if schema.get("additionalProperties", True) is False:
            extra = sorted(str(key) for key in value if key not in properties)
            if extra:
                return f"{path} contains unknown field: {extra[0]}."

        for key, child_value in value.items():
            child_schema = properties.get(key)
            if child_schema is None:
                continue
            error = _validate_value(
                child_value,
                child_schema,
                path=f"{path}.{key}",
            )
            if error is not None:
                return error

    if isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            error = _validate_value(item, schema["items"], path=f"{path}[{index}]")
            if error is not None:
                return error

    # ``bool`` is a subclass of ``int`` in Python.  The type check above has
    # already excluded it for JSON integer/number, so numeric comparisons here
    # cannot accidentally accept true as 1.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path} must be at least {schema['minimum']}."
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path} must be at most {schema['maximum']}."

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return f"{path} is shorter than the minimum allowed length."
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return f"{path} is longer than the maximum allowed length."

    return None


def _matches_json_type(value: Any, expected: Any) -> bool:
    if isinstance(expected, Sequence) and not isinstance(expected, str):
        return any(_matches_json_type(value, item) for item in expected)

    predicates = {
        "object": lambda candidate: isinstance(candidate, Mapping),
        "array": lambda candidate: isinstance(candidate, list),
        "string": lambda candidate: isinstance(candidate, str),
        "integer": lambda candidate: (
            isinstance(candidate, int) and not isinstance(candidate, bool)
        ),
        "number": lambda candidate: (
            isinstance(candidate, (int, float)) and not isinstance(candidate, bool)
        ),
        "boolean": lambda candidate: isinstance(candidate, bool),
        "null": lambda candidate: candidate is None,
    }
    predicate = predicates.get(expected)
    # A malformed built-in schema is a programmer/configuration error.  It is
    # preferable to fail registration-time tests than to treat every value as
    # valid; raising here is caught by execute's defensive wrapper.
    if predicate is None:
        raise ValueError(f"unsupported schema type: {expected!r}")
    return predicate(value)


def build_default_registry(
    workspace: str | os.PathLike[str],
    *,
    command_environment: Mapping[str, str] | None = None,
    excluded_command_environment_names: Iterable[str] = (),
) -> ToolRegistry:
    """Assemble the five file tools and one shell tool for ``workspace``.

    Imports are local to keep ``registry.py`` focused on protocol dispatch and
    to avoid making its basic unit tests instantiate filesystem components.
    ``command_environment`` is the only supported way to explicitly re-add
    project-specific variables to the otherwise sanitised command environment.
    ``excluded_command_environment_names`` carries exact credential-variable
    names known by the CLI; these names remain excluded even if a generic
    sensitive-name pattern would not recognize them.
    """

    from .filesystem import build_filesystem_tools
    from .shell import build_shell_tool

    tools = [*build_filesystem_tools(workspace)]
    tools.append(
        build_shell_tool(
            workspace,
            extra_env=command_environment,
            excluded_env_names=excluded_command_environment_names,
        )
    )
    return ToolRegistry(tools)


def build_planning_registry(
    workspace: str | os.PathLike[str],
    *,
    plan_store: Any | None = None,
    command_environment: Mapping[str, str] | None = None,
    excluded_command_environment_names: Iterable[str] = (),
) -> ToolRegistry:
    """Assemble the default tools plus the opt-in side-effect-free planner."""

    from .planning import build_update_plan_tool

    registry = build_default_registry(
        workspace,
        command_environment=command_environment,
        excluded_command_environment_names=excluded_command_environment_names,
    )
    registry.register(build_update_plan_tool(plan_store))
    return registry
