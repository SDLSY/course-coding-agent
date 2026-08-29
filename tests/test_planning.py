"""Contract tests for the opt-in, side-effect-free planning tool."""

from __future__ import annotations

import json
from pathlib import Path

from coding_agent.agent import AgentRuntime
from coding_agent.context import ContextBuilder
from coding_agent.policy import AgentLimits
from coding_agent.tools.planning import (
    PlanStore,
    PlanValidationError,
    build_update_plan_tool,
)
from coding_agent.tools.registry import (
    ToolRegistry,
    build_default_registry,
    build_planning_registry,
)
from coding_agent.types import ModelTurn, RunPhase, ToolCall
from tests.fakes import ScriptedModel


def test_default_registry_does_not_expose_planner(tmp_path: Path) -> None:
    assert "update_plan" not in build_default_registry(tmp_path).names


def test_planning_registry_exposes_schema_and_full_snapshot(tmp_path: Path) -> None:
    store = PlanStore()
    registry = build_planning_registry(tmp_path, plan_store=store)
    assert registry.names[-1] == "update_plan"
    schema = registry.model_schemas()[-1]
    assert schema["function"]["name"] == "update_plan"
    assert schema["function"]["parameters"]["properties"]["plan"]["maxItems"] == 8

    result = registry.execute(
        "plan-1",
        "update_plan",
        {
            "plan": [
                {"step": "Inspect files", "status": "in_progress"},
                {"step": "Run checks", "status": "pending"},
            ],
            "explanation": "Start with a focused inspection.",
        },
    )
    assert result.ok
    payload = json.loads(result.content)
    assert payload["revision"] == 1
    assert payload["plan"][0]["status"] == "in_progress"
    assert result.metadata == {"revision": 1, "completed": False}


def test_falsey_plan_store_is_preserved() -> None:
    class FalseyStore(PlanStore):
        def __bool__(self) -> bool:
            return False

    store = FalseyStore()
    tool = build_update_plan_tool(store)
    result = ToolRegistry([tool]).execute(
        "falsey-store",
        "update_plan",
        {"plan": [{"step": "Keep the injected store", "status": "pending"}]},
    )

    assert result.ok
    assert store.revision == 1


def test_invalid_update_is_atomic_and_returns_structured_error() -> None:
    store = PlanStore()
    store.update([{"step": "Keep this", "status": "pending"}])
    before = store.snapshot
    registry = ToolRegistry([build_update_plan_tool(store)])

    result = registry.execute(
        "bad-plan",
        "update_plan",
        {
            "plan": [
                {"step": "Duplicate", "status": "pending"},
                {"step": "Duplicate", "status": "completed"},
            ]
        },
    )
    assert not result.ok
    assert result.error_code == "invalid_plan"
    assert store.snapshot == before
    assert store.snapshot.revision == 1


def test_store_rejects_two_in_progress_steps_without_mutating_revision() -> None:
    store = PlanStore()
    try:
        store.update(
            [
                {"step": "One", "status": "in_progress"},
                {"step": "Two", "status": "in_progress"},
            ]
        )
    except PlanValidationError as exc:
        assert "at most one in_progress" in str(exc)
    else:  # pragma: no cover - assertion keeps the contract explicit
        raise AssertionError("expected plan validation to fail")
    assert store.snapshot.revision == 0
    assert store.snapshot.plan == ()


def test_completed_plan_is_observation_only() -> None:
    store = PlanStore()
    tool = build_update_plan_tool(store)
    result = ToolRegistry([tool]).execute(
        "done",
        "update_plan",
        {"plan": [{"step": "Done", "status": "completed"}]},
    )
    assert result.ok
    assert json.loads(result.content)["plan"][0]["status"] == "completed"
    # The tool has no stop/execute callback; completion only appears in the
    # returned metadata and cannot terminate AgentRuntime by itself.
    assert result.metadata["completed"] is True


def test_completed_plan_does_not_auto_stop_agent(tmp_path: Path) -> None:
    store = PlanStore()
    model = ScriptedModel(
        [
            ModelTurn(
                tool_calls=(
                    ToolCall(
                        "plan-call",
                        "update_plan",
                        '{"plan":[{"step":"Done","status":"completed"}]}',
                    ),
                )
            ),
            ModelTurn(text="The planned work is complete.", finish_reason="stop"),
        ]
    )
    runtime = AgentRuntime(
        model,
        ToolRegistry([build_update_plan_tool(store)]),
        ContextBuilder(max_chars=10_000),
        limits=AgentLimits(max_model_turns=3, max_tool_calls=3),
    )

    result = runtime.run("Record and then report the plan.")

    assert result.phase is RunPhase.COMPLETED
    assert result.model_turns == 2
    assert result.final_text == "The planned work is complete."
