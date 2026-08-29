"""Execution-backend contracts used by the agent controller.

The default :class:`~coding_agent.tools.registry.ToolRegistry` executes tools
in the local process.  A benchmark harness may instead need to execute the
same logical tools in a remote container.  The controller should not need to
know where a tool runs, so this module documents the deliberately tiny
structural interface shared by both implementations.

The method names intentionally match ``ToolRegistry``.  Existing registries,
test doubles, and third-party integrations therefore continue to work without
an adapter or a breaking constructor change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from coding_agent.types import ToolCall, ToolResult


@runtime_checkable
class ExecutionBackend(Protocol):
    """Minimal boundary the Runtime needs from a tool execution backend.

    ``execute_call`` must return one fully formed ``ToolResult`` for every
    input call, including malformed arguments, unknown tools, and execution
    failures.  In particular, a backend must not return ``ToolOutput`` or
    raise an ordinary handler exception after the controller has admitted a
    tool call: doing so would leave the model history without its required
    result message.

    The backend owns its workspace, permissions, environment, and actual
    execution policy.  The Runtime supplies only the remaining wall-time
    budget; it does not expose its mutable history or counters.
    """

    def model_schemas(self) -> Sequence[Mapping[str, Any]]:
        """Return the provider-facing schemas in deterministic order."""

    def execute_call(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Validate/execute one raw call and always return a paired result."""
