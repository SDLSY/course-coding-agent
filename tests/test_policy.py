"""Tests for finite run and retry policies."""

from __future__ import annotations

import pytest

from coding_agent.policy import AgentLimits, RetryPolicy


def test_agent_limits_reject_zero_operational_budgets() -> None:
    with pytest.raises(ValueError, match="max_model_turns"):
        AgentLimits(max_model_turns=0)
    with pytest.raises(ValueError, match="max_tool_calls"):
        AgentLimits(max_tool_calls=0)
    with pytest.raises(ValueError, match="max_wall_time_seconds"):
        AgentLimits(max_wall_time_seconds=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_agent_limits_reject_non_finite_or_boolean_wall_time(value: object) -> None:
    with pytest.raises(ValueError, match="max_wall_time_seconds"):
        AgentLimits(max_wall_time_seconds=value)  # type: ignore[arg-type]


def test_retry_policy_caps_exponential_delay() -> None:
    policy = RetryPolicy(
        initial_delay_seconds=1,
        multiplier=2,
        maximum_delay_seconds=3,
        jitter_ratio=0,
    )

    assert policy.delay_seconds(0) == 1
    assert policy.delay_seconds(1) == 2
    assert policy.delay_seconds(2) == 3
    assert policy.delay_seconds(20) == 3


def test_retry_policy_jitter_is_deterministic_when_sample_is_injected() -> None:
    policy = RetryPolicy(
        initial_delay_seconds=2,
        multiplier=2,
        maximum_delay_seconds=8,
        jitter_ratio=0.25,
    )

    assert policy.delay_seconds(0, random_value=0) == 1.5
    assert policy.delay_seconds(0, random_value=0.5) == 2
    assert policy.delay_seconds(0, random_value=1) == 2.5


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_delay_seconds", float("nan")),
        ("multiplier", float("inf")),
        ("maximum_delay_seconds", True),
        ("jitter_ratio", float("nan")),
    ],
)
def test_retry_policy_rejects_non_finite_or_boolean_numbers(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        RetryPolicy(**{field: value})  # type: ignore[arg-type]
