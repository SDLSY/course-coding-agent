"""Deterministic test doubles shared by runtime unit tests.

No fake in this module reads environment variables or performs network I/O.
The separation between ``ScriptedModel`` and ``FakeOpenAIClient`` mirrors the
production boundaries: the former tests the agent controller, while the latter
tests only the ordinary Chat Completions adapter.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from coding_agent.types import Message, ModelTurn


@dataclass(frozen=True, slots=True)
class RecordedModelCall:
    messages: tuple[Message, ...]
    tools: tuple[dict[str, Any], ...]


class ScriptedModel:
    """Return predefined turns/exceptions and record each normalized request."""

    def __init__(self, script: Iterable[ModelTurn | Exception]) -> None:
        self._script = deque(script)
        self.calls: list[RecordedModelCall] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Mapping[str, Any]],
        *,
        timeout_seconds: float | None = None,
    ) -> ModelTurn:
        self.calls.append(
            RecordedModelCall(
                messages=tuple(messages),
                tools=tuple(dict(tool) for tool in tools),
            )
        )
        if not self._script:
            raise AssertionError("ScriptedModel received more calls than its script")
        item = self._script.popleft()
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def remaining(self) -> int:
        return len(self._script)


class FakeCompletionsEndpoint:
    """Small stand-in for ``client.chat.completions``."""

    def __init__(self, responses: Iterable[Any | Exception]) -> None:
        self._responses = deque(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not self._responses:
            raise AssertionError("fake Chat Completions endpoint has no response left")
        item = self._responses.popleft()
        if isinstance(item, Exception):
            raise item
        return item


class FakeOpenAIClient:
    """Object with the same attribute path used by the production adapter."""

    def __init__(self, responses: Iterable[Any | Exception]) -> None:
        self.completions = FakeCompletionsEndpoint(responses)
        self.chat = SimpleNamespace(completions=self.completions)


def chat_completion(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = "stop",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build the ordinary mapping form accepted by the response normalizer."""

    message: dict[str, Any] = {"content": content, "tool_calls": tool_calls}
    # Ordinary OpenAI responses omit this provider extension.  Leaving it out
    # when unused ensures tests also catch accidental injection into standard
    # request messages.
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content

    return {
        "choices": [
            {
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def raw_tool_call(call_id: str, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
