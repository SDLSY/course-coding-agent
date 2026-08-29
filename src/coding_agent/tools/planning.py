"""An opt-in, side-effect-free planning tool.

The planning tool is deliberately kept separate from the workspace tools.  A
plan is observability state, not an instruction to execute anything: updating
it never reads or writes files, runs a command, or contacts a service.  The
store validates a complete snapshot before publishing it, which means callers
can safely expose it to an untrusted model without ending up with half of an
invalid update.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal

from .base import Tool, ToolOutput, ToolRequestError

PlanStatus = Literal["pending", "in_progress", "completed"]
PLAN_STATUSES: frozenset[str] = frozenset({"pending", "in_progress", "completed"})
MAX_PLAN_STEPS = 8
MAX_STEP_LENGTH = 300
MAX_EXPLANATION_LENGTH = 2_000


class PlanValidationError(ValueError):
    """Raised when a complete plan snapshot violates the plan contract."""


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One human-readable step and its lifecycle status."""

    step: str
    status: PlanStatus

    def __post_init__(self) -> None:
        if not isinstance(self.step, str) or not self.step.strip():
            raise PlanValidationError("plan steps must have a non-empty step")
        if len(self.step) > MAX_STEP_LENGTH:
            raise PlanValidationError(
                f"plan step must be at most {MAX_STEP_LENGTH} characters"
            )
        if not isinstance(self.status, str) or self.status not in PLAN_STATUSES:
            raise PlanValidationError(
                "plan step status must be pending, in_progress, or completed"
            )

    def as_dict(self) -> dict[str, str]:
        """Return the stable JSON shape sent to the model."""

        return {"step": self.step, "status": self.status}


@dataclass(frozen=True, slots=True)
class PlanSnapshot:
    """An immutable, validated plan revision."""

    plan: tuple[PlanStep, ...]
    explanation: str | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        # Validate direct construction too; the store is not the only public
        # entry point and an immutable snapshot should always be trustworthy.
        try:
            normalized_plan = tuple(self.plan)
        except TypeError as exc:
            raise PlanValidationError(
                "plan must be a sequence of PlanStep values"
            ) from exc
        object.__setattr__(self, "plan", normalized_plan)
        _validate_plan(normalized_plan, self.explanation)
        if not isinstance(self.revision, int) or isinstance(self.revision, bool):
            raise PlanValidationError("plan revision must be an integer")
        if self.revision < 0:
            raise PlanValidationError("plan revision must not be negative")

    @property
    def completed(self) -> bool:
        """Whether every step in this snapshot is marked completed."""

        return bool(self.plan) and all(step.status == "completed" for step in self.plan)

    @property
    def steps(self) -> tuple[PlanStep, ...]:
        """Readable alias for integrations that call plan entries steps."""

        return self.plan

    def as_dict(self) -> dict[str, Any]:
        """Serialize the full snapshot, including its monotonic revision."""

        return {
            "plan": [step.as_dict() for step in self.plan],
            "explanation": self.explanation,
            "revision": self.revision,
        }

    to_dict = as_dict


class PlanStore:
    """Atomically retain the latest in-memory :class:`PlanSnapshot`."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot = PlanSnapshot(plan=())

    @property
    def snapshot(self) -> PlanSnapshot:
        """Return the current immutable snapshot."""

        with self._lock:
            return self._snapshot

    # ``current`` is a readable alias for integrations that use that name.
    current = snapshot

    @property
    def revision(self) -> int:
        """Return the current monotonic revision number."""

        return self.snapshot.revision

    def update(
        self,
        plan: Sequence[Mapping[str, Any] | PlanStep],
        explanation: str | None = None,
    ) -> PlanSnapshot:
        """Validate and publish a complete replacement snapshot.

        No state is changed if validation fails.  A replacement (rather than
        an incremental patch) also makes each tool call self-contained in the
        conversation history and easy to replay.
        """

        validated = _coerce_plan(plan)
        _validate_plan(validated, explanation)
        with self._lock:
            next_revision = self._snapshot.revision + 1
            next_snapshot = PlanSnapshot(
                plan=validated,
                explanation=explanation,
                revision=next_revision,
            )
            self._snapshot = next_snapshot
            return next_snapshot

    # Explicit naming for callers that prefer treating updates as replacement.
    replace = update


def _coerce_plan(
    plan: Sequence[Mapping[str, Any] | PlanStep],
) -> tuple[PlanStep, ...]:
    if isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence):
        raise PlanValidationError("plan must be an array of step objects")
    if not 1 <= len(plan) <= MAX_PLAN_STEPS:
        raise PlanValidationError(
            f"plan must contain between 1 and {MAX_PLAN_STEPS} steps"
        )
    steps: list[PlanStep] = []
    seen: set[str] = set()
    for index, item in enumerate(plan):
        if isinstance(item, PlanStep):
            step = item
        elif isinstance(item, Mapping):
            if set(item) - {"step", "status"}:
                raise PlanValidationError(f"plan[{index}] contains an unknown field")
            if "step" not in item or "status" not in item:
                raise PlanValidationError(f"plan[{index}] requires step and status")
            step = PlanStep(step=item["step"], status=item["status"])
        else:
            raise PlanValidationError(f"plan[{index}] must be an object")
        normalized = step.step.strip()
        if normalized in seen:
            raise PlanValidationError("plan steps must not be duplicated")
        seen.add(normalized)
        steps.append(step)
    if sum(step.status == "in_progress" for step in steps) > 1:
        raise PlanValidationError("plan may contain at most one in_progress step")
    return tuple(steps)


def _validate_plan(plan: Sequence[PlanStep], explanation: str | None) -> None:
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)):
        raise PlanValidationError("plan must be a sequence of PlanStep values")
    if len(plan) > MAX_PLAN_STEPS:
        raise PlanValidationError(f"plan must contain at most {MAX_PLAN_STEPS} steps")
    seen: set[str] = set()
    in_progress = 0
    for index, step in enumerate(plan):
        if not isinstance(step, PlanStep):
            raise PlanValidationError(f"plan[{index}] must be a PlanStep")
        normalized = step.step.strip()
        if normalized in seen:
            raise PlanValidationError("plan steps must not be duplicated")
        seen.add(normalized)
        in_progress += step.status == "in_progress"
    if in_progress > 1:
        raise PlanValidationError("plan may contain at most one in_progress step")
    if explanation is not None:
        if not isinstance(explanation, str):
            raise PlanValidationError("explanation must be a string when supplied")
        if len(explanation) > MAX_EXPLANATION_LENGTH:
            raise PlanValidationError(
                f"explanation must be at most {MAX_EXPLANATION_LENGTH} characters"
            )


def build_update_plan_tool(store: PlanStore | None = None) -> Tool:
    """Build the model-facing ``update_plan`` definition."""

    plan_store = store or PlanStore()

    def handler(arguments: Mapping[str, Any]) -> ToolOutput:
        try:
            snapshot = plan_store.update(
                arguments["plan"],
                arguments.get("explanation"),
            )
        except (KeyError, PlanValidationError) as exc:
            raise ToolRequestError(str(exc), error_code="invalid_plan") from exc
        return ToolOutput(
            content=json.dumps(snapshot.as_dict(), ensure_ascii=False, sort_keys=True),
            metadata={"revision": snapshot.revision, "completed": snapshot.completed},
        )

    return Tool(
        name="update_plan",
        description=(
            "Replace the in-memory execution plan with 1-8 ordered steps. "
            "This records intent only and has no workspace side effects."
        ),
        parameters={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PLAN_STEPS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_STEP_LENGTH,
                            },
                            "status": {
                                "type": "string",
                                "enum": sorted(PLAN_STATUSES),
                            },
                        },
                        "required": ["step", "status"],
                        "additionalProperties": False,
                    },
                },
                "explanation": {
                    "type": "string",
                    "maxLength": MAX_EXPLANATION_LENGTH,
                },
            },
            "required": ["plan"],
            "additionalProperties": False,
        },
        handler=handler,
    )


__all__ = [
    "MAX_PLAN_STEPS",
    "PlanSnapshot",
    "PlanStatus",
    "PlanStep",
    "PlanStore",
    "PlanValidationError",
    "build_update_plan_tool",
]
