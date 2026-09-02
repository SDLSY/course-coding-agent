"""Domain-specific errors used by the coding-agent runtime.

The exception hierarchy is deliberately small and semantic.  The agent loop
must decide whether an operation may be retried, should be shown to the model,
or must terminate the run.  Making that decision from arbitrary exception
messages would be brittle, so adapters translate errors at their boundary and
the controller only handles the categories below.

These classes intentionally contain no retry logic.  For example,
``TransientModelError`` says that retrying *may* be safe; the retry count and
backoff remain policy decisions owned by the agent runtime.
"""

from __future__ import annotations


class CodingAgentError(Exception):
    """Base class for expected, user-facing runtime failures."""


class ConfigurationError(CodingAgentError):
    """The run cannot start because required configuration is invalid."""


class ModelError(CodingAgentError):
    """Base class for failures at the model-provider boundary."""


class TransientModelError(ModelError):
    """A model request failed in a way that can be retried without side effects."""


class PermanentModelError(ModelError):
    """A model request cannot succeed without changing configuration or input."""


class ReasoningEffortUnsupported(PermanentModelError):
    """The selected gateway rejected the requested native reasoning option.

    This is kept separate from a generic permanent model error so a preflight
    probe can report ``unsupported`` without silently retrying with a made-up
    prompt instruction or a different effort level.
    """


class ContextOverflow(ModelError):
    """Required prompt content cannot fit in the configured context budget."""


class ResponseProtocolError(ModelError):
    """A provider response cannot be represented by the internal protocol."""


class ToolRequestError(CodingAgentError):
    """The model requested a tool with invalid name, JSON, or arguments."""

    def __init__(
        self, message: str, *, error_code: str = "invalid_tool_request"
    ) -> None:
        super().__init__(message)
        self.error_code = error_code


class ToolExecutionError(CodingAgentError):
    """A valid tool request failed while interacting with the local machine."""


class InvariantViolation(CodingAgentError):
    """Internal state no longer satisfies a protocol invariant; continuing is unsafe."""


class UserCancelled(CodingAgentError):
    """The user explicitly interrupted the current run."""
