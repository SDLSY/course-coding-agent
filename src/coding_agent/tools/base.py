"""Common contracts used by every local tool.

The language model is not a trusted caller.  A tool invocation therefore has
two boundaries:

* :class:`ToolRegistry` (implemented in ``registry.py``) validates the JSON
  object supplied by the model; and
* the concrete handler validates domain rules which JSON Schema cannot express
  conveniently, such as "this path must stay below the workspace".

Handlers return :class:`ToolOutput` rather than constructing protocol-level
``ToolResult`` objects themselves.  This keeps call IDs and tool names under
the registry's control, so a buggy handler cannot accidentally break the
assistant-tool-result pairing invariant maintained by the agent runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from coding_agent.errors import (
    ToolExecutionError as RuntimeToolExecutionError,
)
from coding_agent.errors import (
    ToolRequestError as RuntimeToolRequestError,
)

JsonObject: TypeAlias = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """Successful output produced by a concrete tool handler.

    ``content`` is the human/model-readable observation.  ``metadata`` carries
    machine-readable facts such as an exit code or truncation state.  Keeping
    those facts out of prose lets the runtime and tests inspect them without
    parsing an English message.

    Command failure is deliberately representable as a successful
    ``ToolOutput``.  For example, ``pytest`` exiting with status 1 means that
    ``run_command`` worked and observed failing tests; it does *not* mean the
    tool protocol failed.
    """

    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ToolRequestError(RuntimeToolRequestError):
    """A correctable model request error with optional structured metadata.

    This extends, rather than duplicates, the runtime-wide exception from
    :mod:`coding_agent.errors`.  Consequently the registry can handle errors
    raised by both ``response_parser.parse_tool_arguments`` and concrete tools
    through one semantic branch.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "invalid_tool_request",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, error_code=error_code)
        self.metadata = dict(metadata or {})


class ToolExecutionError(RuntimeToolExecutionError):
    """A local-machine failure with a stable code and optional metadata."""

    def __init__(
        self,
        message: str,
        *,
        error_code: str = "tool_execution_error",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.metadata = dict(metadata or {})


ToolHandler: TypeAlias = Callable[[JsonObject], ToolOutput]


@dataclass(frozen=True, slots=True)
class Tool:
    """Description and implementation of one model-callable local operation.

    ``parameters`` is the JSON Schema object sent to the model and checked by
    our own small validator.  The project intentionally supports only the
    schema features needed by its six tools; using a full agent framework or a
    schema-validation framework would hide one of the assignment's key logic
    paths.

    ``modifies_workspace`` is descriptive metadata.  It does not grant
    permission and must never be treated as a security mechanism.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler
    modifies_workspace: bool = False
    # Name of an optional model argument which controls how long the handler may
    # block.  The registry can lower this value to the Agent's remaining run
    # budget without teaching the controller about a concrete tool's schema.
    timeout_argument: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError("tool name must be a non-empty Python-style identifier")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if self.parameters.get("type") != "object":
            raise ValueError("tool parameters schema must describe an object")
        if self.timeout_argument is not None:
            if not self.timeout_argument.isidentifier():
                raise ValueError("timeout_argument must be a Python-style identifier")
            properties = self.parameters.get("properties", {})
            if self.timeout_argument not in properties:
                raise ValueError("timeout_argument must name a declared tool parameter")

    def as_model_schema(self) -> dict[str, Any]:
        """Return the OpenAI-compatible function-tool representation.

        A fresh top-level dictionary is returned on each call.  The nested
        schema is intentionally not deep-copied because tool definitions are
        frozen configuration; callers must treat it as read-only.
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }
