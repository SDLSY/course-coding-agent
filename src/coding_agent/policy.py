"""Finite execution and retry policies for the agent control loop.

The policy objects in this module contain data and deterministic calculations;
they do not call the model or execute tools.  Keeping limits separate from the
loop matters for two reasons:

1. tests can make every boundary small without patching global constants; and
2. the CLI can expose resource budgets without gaining access to mutable
   runtime internals.

An Agent is an open-ended feedback loop by nature.  Every external operation
therefore needs a finite bound.  A maximum number of model turns alone is not
enough: one response may request many tools, and one command may run forever.
The independent limits below close those different escape paths.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EfficiencyPolicy:
    """Optional turn-management policy used by evaluation runs.

    This small value object is convenient for callers that do not want to
    encode strategy knobs in ``AgentLimits``.  ``AgentRuntime`` converts it to
    the equivalent limit fields when supplied.
    """

    enabled: bool = False
    reserve_final_turn: bool = True
    convergence_remaining_turns: int = 5
    max_repeated_tool_batches: int = 2
    max_no_progress_batches: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(
            self.reserve_final_turn, bool
        ):
            raise TypeError("efficiency policy flags must be bools")
        for name, value in (
            ("convergence_remaining_turns", self.convergence_remaining_turns),
            ("max_repeated_tool_batches", self.max_repeated_tool_batches),
            ("max_no_progress_batches", self.max_no_progress_batches),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if name == "convergence_remaining_turns" and value < 0:
                raise ValueError(f"{name} must be non-negative")
            if name != "convergence_remaining_turns" and value < 1:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class AgentLimits:
    """Hard resource limits applied to one Agent run.

    ``max_model_turns`` counts successful, normalized model responses.  A
    transport retry for the same logical turn does not consume another turn,
    but it is bounded separately by ``max_model_retries``.

    ``max_tool_calls`` counts calls requested by the model, including calls
    rejected for bad arguments.  Invalid calls still consume runtime capacity;
    otherwise a model could loop forever by repeatedly emitting malformed JSON.

    ``max_wall_time_seconds`` is checked before every model request and tool
    batch.  A command also receives its own, normally shorter timeout from the
    tool implementation.  The run-level deadline is a final containment bound,
    not a replacement for per-command timeouts.
    """

    max_model_turns: int = 20
    max_tool_calls: int = 50
    max_wall_time_seconds: float = 900.0
    max_model_retries: int = 3
    max_protocol_retries: int = 1
    # Efficiency controls are opt-in so existing callers retain the original
    # protocol exactly.  They are deliberately data-only; AgentRuntime owns
    # the corresponding state transitions.
    efficiency_mode: bool = False
    reserve_final_turn: bool = False
    convergence_remaining_turns: int = 5
    max_repeated_tool_batches: int = 2
    max_no_progress_batches: int = 2

    def __post_init__(self) -> None:
        integer_fields = {
            "max_model_turns": self.max_model_turns,
            "max_tool_calls": self.max_tool_calls,
            "max_model_retries": self.max_model_retries,
            "max_protocol_retries": self.max_protocol_retries,
            "convergence_remaining_turns": self.convergence_remaining_turns,
            "max_repeated_tool_batches": self.max_repeated_tool_batches,
            "max_no_progress_batches": self.max_no_progress_batches,
        }
        for name, value in integer_fields.items():
            # ``bool`` is an ``int`` subclass.  Accepting True as a budget of
            # one would be surprising and usually indicates a CLI/config bug.
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_model_turns == 0:
            raise ValueError("max_model_turns must be greater than zero")
        if self.max_tool_calls == 0:
            raise ValueError("max_tool_calls must be greater than zero")
        if self.convergence_remaining_turns < 0:
            raise ValueError("convergence_remaining_turns must be non-negative")
        if self.max_repeated_tool_batches < 1:
            raise ValueError("max_repeated_tool_batches must be greater than zero")
        if self.max_no_progress_batches < 1:
            raise ValueError("max_no_progress_batches must be greater than zero")
        if not isinstance(self.efficiency_mode, bool):
            raise TypeError("efficiency_mode must be a bool")
        if not isinstance(self.reserve_final_turn, bool):
            raise TypeError("reserve_final_turn must be a bool")
        if (
            not isinstance(self.max_wall_time_seconds, (int, float))
            or isinstance(self.max_wall_time_seconds, bool)
            or not math.isfinite(float(self.max_wall_time_seconds))
            or self.max_wall_time_seconds <= 0
        ):
            raise ValueError(
                "max_wall_time_seconds must be a finite number greater than zero"
            )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded exponential backoff parameters for model transport failures.

    Only a model request is eligible for automatic retry.  It has no local
    workspace side effect, whereas replaying a file write or shell command may
    duplicate an action whose first outcome is unknown.

    Jitter prevents several processes sharing the same provider limit from
    retrying at exactly the same instant.  Tests can pass ``jitter_ratio=0`` or
    inject a deterministic random source through :meth:`delay_seconds`.
    """

    initial_delay_seconds: float = 0.5
    multiplier: float = 2.0
    maximum_delay_seconds: float = 8.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        numeric_fields = {
            "initial_delay_seconds": self.initial_delay_seconds,
            "multiplier": self.multiplier,
            "maximum_delay_seconds": self.maximum_delay_seconds,
            "jitter_ratio": self.jitter_ratio,
        }
        for name, value in numeric_fields.items():
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds cannot be negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be at least one")
        if self.maximum_delay_seconds < self.initial_delay_seconds:
            raise ValueError(
                "maximum_delay_seconds cannot be below initial_delay_seconds"
            )
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between zero and one")

    def delay_seconds(
        self,
        retry_index: int,
        *,
        random_value: float | None = None,
    ) -> float:
        """Return the delay before zero-based retry ``retry_index``.

        ``random_value`` represents a sample in ``[0, 1]``.  Accepting it as an
        argument makes the calculation deterministic in tests without exposing
        a mutable random-number generator on the policy object.
        """

        if not isinstance(retry_index, int) or isinstance(retry_index, bool):
            raise TypeError("retry_index must be an integer")
        if retry_index < 0:
            raise ValueError("retry_index cannot be negative")

        base = min(
            self.maximum_delay_seconds,
            self.initial_delay_seconds * (self.multiplier**retry_index),
        )
        if base == 0 or self.jitter_ratio == 0:
            return base

        sample = random.random() if random_value is None else random_value
        if (
            not isinstance(sample, (int, float))
            or isinstance(sample, bool)
            or not math.isfinite(float(sample))
        ):
            raise ValueError("random_value must be a finite number")
        if not 0 <= sample <= 1:
            raise ValueError("random_value must be between zero and one")

        # Map [0, 1] to [-1, 1], then scale by the configured fraction.  The
        # lower clamp matters when jitter_ratio is 1 and sample is exactly 0.
        jitter = (2 * sample - 1) * self.jitter_ratio * base
        return max(0.0, base + jitter)
