"""Small, append-only event sinks with defence-in-depth secret redaction.

The event trace is intended for explaining control flow and diagnosing a run;
it is not a transaction journal and cannot reproduce side effects.  Each line
is an independent JSON object so an interrupted process leaves all previously
completed lines readable.

Redaction happens at the final serialization boundary.  Producers should still
avoid placing credentials in events, but central redaction protects against a
mistaken field name, a nested exception message, or an Authorization header.
The sink cannot reliably discover secrets embedded in arbitrary source files,
which is why trace files should remain local and ignored by version control.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol, TextIO

SCHEMA_VERSION = 1
REDACTED = "[REDACTED]"


_EVENT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_OPENAI_STYLE_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")
_URL_USERINFO_RE = re.compile(r"(https?://)([^/@\s:]+):([^/@\s]+)@", re.IGNORECASE)
# Split both ordinary camelCase (``clientSecret``) and an acronym followed by a
# word (``APIKey``).  Applying these before lower-casing is important: once the
# case distinction is lost, ``clientSecret`` becomes the unrecognizable
# ``clientsecret`` rather than the existing canonical name ``client_secret``.
_ACRONYM_WORD_BOUNDARY_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_LOWER_UPPER_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")


# Metrics such as ``input_tokens`` and ``token_count`` are harmless and useful,
# so generic "token" is deliberately absent.  Only names that conventionally
# contain credential material are considered sensitive.
_SENSITIVE_FIELD_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "proxy_authorization",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "credential",
    "credentials",
    "access_token",
    "refresh_token",
    "auth_token",
    "cookie",
    "set_cookie",
}


class EventSink(Protocol):
    """Minimal dependency used by the runtime.

    Keeping this to one synchronous operation makes a fake sink trivial in
    state-machine tests and avoids pulling logging-framework policy into the
    agent loop.
    """

    def emit(self, event_type: str, **data: object) -> None:
        """Record one completed runtime observation."""


@dataclass(frozen=True, slots=True)
class RunEvent:
    """Provider-independent event value before serialization."""

    event_type: str
    # Event payloads may accidentally contain a credential before the sink's
    # final redaction pass.  Excluding data from repr prevents a debugger or an
    # exception that formats this intermediate value from leaking it first.
    data: Mapping[str, object] = field(default_factory=dict, repr=False)
    timestamp: str = field(default_factory=lambda: _utc_now())

    def __post_init__(self) -> None:
        _validate_event_type(self.event_type)
        if not isinstance(self.data, Mapping):
            raise TypeError("event data must be a mapping")

    def to_record(self) -> dict[str, object]:
        """Return the stable on-disk envelope; values are not redacted yet."""

        return {
            "schema_version": SCHEMA_VERSION,
            "timestamp": self.timestamp,
            "event": self.event_type,
            "data": dict(self.data),
        }


class NullEventSink:
    """No-op sink used when no trace path was requested."""

    def emit(self, event_type: str, **data: object) -> None:
        _validate_event_type(event_type)


class JsonlEventSink:
    """Append redacted events to a local JSONL file.

    The file is opened for each event.  Agent events are low volume, and this
    design avoids a long-lived descriptor and makes every call independently
    flush its complete line.  A process-local lock prevents threads from
    interleaving lines.  Creation mode ``0600`` restricts a new trace to its
    owner because tool output can contain private project code.

    Existing file permissions are preserved.  This class does not call fsync:
    the trace is diagnostic rather than a crash-recovery log, and forcing a
    disk barrier for every tool event would add unnecessary latency.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        secrets: Iterable[str] = (),
        clock: Callable[[], str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve(strict=False)
        self._secrets = _normalise_secrets(secrets)
        self._clock = _utc_now if clock is None else clock
        self._lock = threading.Lock()

        # Fail before the autonomous loop starts if the requested trace
        # directory cannot be created.  Silently losing an explicitly requested
        # audit trace is more surprising than a focused startup failure.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file_exists()

    def emit(self, event_type: str, **data: object) -> None:
        _validate_event_type(event_type)
        event = RunEvent(event_type=event_type, data=data, timestamp=self._clock())
        record = redact(event.to_record(), secrets=self._secrets)
        line = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

        # One write call while holding the lock makes lines indivisible with
        # respect to other threads using this sink instance.
        with self._lock:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            with os.fdopen(descriptor, "a", encoding="utf-8", newline="\n") as file:
                file.write(line + "\n")

    def _ensure_file_exists(self) -> None:
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        os.close(descriptor)


class ConsoleEventSink:
    """Print compact, redacted events for a human watching the CLI."""

    def __init__(
        self,
        *,
        secrets: Iterable[str] = (),
        stream: TextIO | None = None,
    ) -> None:
        self._secrets = _normalise_secrets(secrets)
        self._stream = sys.stderr if stream is None else stream

    def emit(self, event_type: str, **data: object) -> None:
        _validate_event_type(event_type)
        safe_data = redact(data, secrets=self._secrets)
        suffix = ""
        if safe_data:
            suffix = " " + json.dumps(
                safe_data, ensure_ascii=False, separators=(",", ":")
            )
        print(f"[{event_type}]{suffix}", file=self._stream, flush=True)


class CompositeEventSink:
    """Fan out each event to sinks in deterministic order."""

    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = tuple(sinks)

    def emit(self, event_type: str, **data: object) -> None:
        for sink in self._sinks:
            sink.emit(event_type, **data)


def redact(value: object, *, secrets: Iterable[str] = ()) -> object:
    """Return a JSON-compatible, recursively redacted copy of ``value``.

    Explicit secrets are replaced wherever they occur inside strings, not only
    when the surrounding key looks sensitive.  This covers exception messages
    and accidentally formatted headers.  Unknown objects are represented by
    type name instead of arbitrary ``repr`` output because custom ``repr``
    implementations may expose credentials or perform surprising work.
    """

    known_secrets = _normalise_secrets(secrets)
    return _redact_value(value, known_secrets, depth=0, active_ids=set())


def _redact_value(
    value: object,
    secrets: tuple[str, ...],
    *,
    depth: int,
    active_ids: set[int],
) -> object:
    if depth > 20:
        return "[MAX_DEPTH]"

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # JSON's NaN/Infinity spellings are not valid portable JSON.  Preserve
        # the diagnostic meaning without asking json.dumps to emit extensions.
        if math.isnan(value):
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if isinstance(value, str):
        return _redact_text(value, secrets)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{type(value).__name__}: {len(value)} bytes>"
    if isinstance(value, Path):
        return _redact_text(str(value), secrets)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return _redact_value(
            value.value, secrets, depth=depth + 1, active_ids=active_ids
        )
    if isinstance(value, BaseException):
        return {
            "type": type(value).__name__,
            "message": _redact_text(str(value), secrets),
        }

    if isinstance(value, Mapping):
        return _redact_mapping(value, secrets, depth, active_ids)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return _redact_sequence(value, secrets, depth, active_ids)
    if isinstance(value, (set, frozenset)):
        # Sets have no stable order.  Sorting their safe string forms makes
        # traces deterministic enough for tests and human comparison.
        redacted_items = [
            _redact_value(item, secrets, depth=depth + 1, active_ids=active_ids)
            for item in value
        ]
        return sorted(redacted_items, key=lambda item: str(item))

    return f"<{type(value).__name__}>"


def _redact_mapping(
    value: Mapping[object, object],
    secrets: tuple[str, ...],
    depth: int,
    active_ids: set[int],
) -> dict[str, object]:
    object_id = id(value)
    if object_id in active_ids:
        return {"cycle": "[CYCLE]"}
    active_ids.add(object_id)
    try:
        result: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _redact_text(str(raw_key), secrets)
            if _is_sensitive_field(str(raw_key)):
                result[key] = REDACTED
            else:
                result[key] = _redact_value(
                    raw_value,
                    secrets,
                    depth=depth + 1,
                    active_ids=active_ids,
                )
        return result
    finally:
        active_ids.remove(object_id)


def _redact_sequence(
    value: Sequence[object],
    secrets: tuple[str, ...],
    depth: int,
    active_ids: set[int],
) -> list[object]:
    object_id = id(value)
    if object_id in active_ids:
        return ["[CYCLE]"]
    active_ids.add(object_id)
    try:
        return [
            _redact_value(item, secrets, depth=depth + 1, active_ids=active_ids)
            for item in value
        ]
    finally:
        active_ids.remove(object_id)


def _is_sensitive_field(name: str) -> bool:
    normalised = _normalise_field_name(name)
    # The name of the environment variable is safe and useful for diagnosing
    # configuration, even though it often contains the words "api_key".
    if normalised.endswith("_env") or normalised == "key_env":
        return False
    if normalised in _SENSITIVE_FIELD_NAMES:
        return True
    return any(
        marker in normalised
        for marker in ("api_key", "authorization", "password", "credential")
    ) or normalised.endswith("_secret")


def _normalise_field_name(name: str) -> str:
    """Convert common mapping-key styles to one snake_case comparison form.

    Event payloads may originate from Python code, JSON responses, or SDK
    objects, so the same semantic field can arrive as ``client_secret``,
    ``clientSecret``, ``ClientSecret``, ``APIKey`` or ``api-key``.  This helper
    changes only the private comparison value used by redaction; the serialized
    event retains the producer's original key for diagnostics.
    """

    with_acronym_boundaries = _ACRONYM_WORD_BOUNDARY_RE.sub(r"\1_\2", name)
    with_word_boundaries = _LOWER_UPPER_BOUNDARY_RE.sub(
        r"\1_\2", with_acronym_boundaries
    )
    return re.sub(r"[^a-z0-9]+", "_", with_word_boundaries.lower()).strip("_")


def _redact_text(value: str, secrets: tuple[str, ...]) -> str:
    safe = value
    # Longest first prevents one secret that is a prefix of another from
    # leaving a visible suffix behind.
    for secret in secrets:
        if secret:
            safe = safe.replace(secret, REDACTED)
    safe = _BEARER_RE.sub(lambda match: match.group(1) + REDACTED, safe)
    safe = _OPENAI_STYLE_KEY_RE.sub(REDACTED, safe)
    safe = _URL_USERINFO_RE.sub(r"\1[REDACTED]@", safe)
    return safe


def _normalise_secrets(secrets: Iterable[str]) -> tuple[str, ...]:
    unique = {secret for secret in secrets if isinstance(secret, str) and secret}
    return tuple(sorted(unique, key=len, reverse=True))


def _validate_event_type(event_type: str) -> None:
    if not isinstance(event_type, str) or not _EVENT_TYPE_RE.fullmatch(event_type):
        raise ValueError(
            "event_type must start with a lowercase letter and contain only "
            "lowercase letters, digits, '.', '_' or '-'"
        )


def _utc_now() -> str:
    """Return an unambiguous, sortable UTC timestamp."""

    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
