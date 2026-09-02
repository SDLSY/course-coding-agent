"""Command-line boundary for configuring and running one coding-agent task.

The CLI owns user-facing parsing, environment lookup and exit codes.  It does
not own the agent loop.  Imports of the runtime and concrete tools are delayed
until after configuration has been validated; this keeps ``--help`` cheap and
allows configuration/event unit tests to run without constructing an SDK
client or touching the network unless a native reasoning probe was explicitly
requested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from coding_agent.config import (
    AgentConfig,
    ConfigurationError,
    Provider,
    ReasoningEffort,
)
from coding_agent.errors import CodingAgentError
from coding_agent.events import (
    CompositeEventSink,
    ConsoleEventSink,
    EventSink,
    JsonlEventSink,
    redact,
)
from coding_agent.verification import (
    CommandVerifier,
    VerificationCheck,
    VerificationResult,
)

# Exit codes are intentionally small and stable so a benchmark harness can
# distinguish task/runtime outcomes from malformed invocation.  Detailed run
# semantics remain in RunResult and the optional JSONL trace.
EXIT_COMPLETED = 0
EXIT_RUNTIME_FAILED = 1
EXIT_USAGE = 2
EXIT_LIMIT_REACHED = 3
EXIT_CANCELLED = 130
# This code is used only when the opt-in ``--verify`` path is requested.  The
# default CLI path retains the historical four outcomes above.
EXIT_VERIFICATION_FAILED = 4

# A reasoning capability check is a startup diagnostic rather than part of the
# agent's turn budget. Keep it short enough that a dead gateway cannot consume
# the full per-request timeout before the CLI reports a configuration failure.
REASONING_PROBE_TIMEOUT_SECONDS = 20.0


def build_parser() -> argparse.ArgumentParser:
    """Create the parser without consulting process-global environment state."""

    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description=(
            "Run one framework-free coding-agent task in a local workspace. "
            "API credentials are read from a named environment variable."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Environment fallbacks use the CODING_AGENT_* prefix, including "
            "CODING_AGENT_PROVIDER, CODING_AGENT_MODEL, "
            "CODING_AGENT_BASE_URL and CODING_AGENT_KEY_ENV. Provider presets "
            "set only the base URL; always choose the model explicitly."
        ),
    )

    # Supporting both forms keeps an interactive invocation concise while
    # giving scripts an unambiguous named option.  ``main`` rejects supplying
    # both rather than silently choosing one.
    parser.add_argument(
        "task",
        nargs="?",
        help="natural-language coding task (or CODING_AGENT_TASK)",
    )
    parser.add_argument(
        "--task",
        dest="task_option",
        metavar="TEXT",
        help="named alternative to the positional task",
    )
    parser.add_argument(
        "--workspace",
        metavar="PATH",
        help="existing project directory (defaults to current directory)",
    )
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in Provider],
        help="base-URL preset: deepseek, glm, or custom",
    )
    parser.add_argument(
        "--model",
        help="exact provider model ID; no model is selected implicitly",
    )
    parser.add_argument(
        "--base-url",
        metavar="URL",
        help="override the provider's OpenAI-compatible API root",
    )
    parser.add_argument(
        "--key-env",
        metavar="NAME",
        help=(
            "name of the environment variable containing the API key; this "
            "option accepts a variable name, never the credential itself"
        ),
    )
    parser.add_argument(
        "--trace",
        dest="trace_path",
        metavar="PATH",
        help="append a redacted local JSONL event trace",
    )
    parser.add_argument(
        "--verify",
        "--verify-command",
        dest="verification_commands",
        action="append",
        metavar="COMMAND",
        help=(
            "trusted acceptance command to run after the agent finishes; "
            "repeat the option for multiple checks"
        ),
    )
    parser.add_argument(
        "--verify-timeout",
        dest="verification_timeout_seconds",
        type=float,
        metavar="SECONDS",
        default=120.0,
        help="timeout applied to each opt-in acceptance command",
    )
    parser.add_argument(
        "--planning",
        action="store_true",
        help="expose the side-effect-free update_plan tool",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=[item.value for item in ReasoningEffort],
        metavar="LEVEL",
        help=(
            "native provider reasoning level; sent only after a gateway "
            "capability probe confirms support"
        ),
    )
    parser.add_argument(
        "--reasoning-parameter",
        metavar="NAME",
        help=(
            "native JSON request field for the reasoning level (for example "
            "reasoning_effort or thinking)"
        ),
    )
    parser.add_argument(
        "--efficiency",
        "--efficiency-mode",
        dest="efficiency_mode",
        action="store_true",
        default=None,
        help="enable turn-efficiency reminders and a reserved final turn",
    )
    parser.add_argument(
        "--reserve-final-turn",
        dest="reserve_final_turn",
        action="store_true",
        default=None,
        help="reserve the last model turn for a tool-free final response",
    )
    parser.add_argument(
        "--convergence-reminder-turns",
        dest="convergence_remaining_turns",
        type=int,
        metavar="N",
        help="start convergence reminders with N turns remaining",
    )
    parser.add_argument(
        "--max-repeated-tool-batches",
        type=int,
        metavar="N",
        help="request a re-plan after repeated identical tool batches",
    )
    parser.add_argument(
        "--max-no-progress-batches",
        type=int,
        metavar="N",
        help="request a re-plan after repeated unchanged tool results",
    )

    budget = parser.add_argument_group("finite runtime budgets")
    budget.add_argument(
        "--max-model-turns",
        type=int,
        metavar="N",
        help="maximum successful model turns",
    )
    budget.add_argument(
        "--max-tool-calls",
        type=int,
        metavar="N",
        help="maximum requested tool calls, including rejected calls",
    )
    budget.add_argument(
        "--max-wall-time",
        dest="max_wall_time_seconds",
        type=float,
        metavar="SECONDS",
        help="wall-clock bound for the entire run",
    )
    budget.add_argument(
        "--context-chars",
        dest="context_char_budget",
        type=int,
        metavar="N",
        help="conservative per-request character budget",
    )
    budget.add_argument(
        "--model-timeout",
        dest="model_timeout_seconds",
        type=float,
        metavar="SECONDS",
        help="timeout applied to one provider request",
    )
    budget.add_argument(
        "--model-retries",
        dest="model_max_retries",
        type=int,
        metavar="N",
        help="maximum retries after transient provider failures",
    )
    budget.add_argument(
        "--protocol-retries",
        dest="protocol_max_retries",
        type=int,
        metavar="N",
        help="maximum retries after an unusable model response",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Run one task and return a process exit code.

    ``environ`` is injectable for tests.  The mapping is passed through to the
    configuration layer unchanged; no dotenv parser or implicit provider SDK
    environment lookup is involved.
    """

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.task is not None and arguments.task_option is not None:
        parser.error("provide the task either positionally or with --task, not both")

    explicit_task = (
        arguments.task_option if arguments.task_option is not None else arguments.task
    )
    try:
        config = AgentConfig.from_sources(
            task=explicit_task,
            workspace=arguments.workspace,
            provider=arguments.provider,
            model=arguments.model,
            base_url=arguments.base_url,
            key_env=arguments.key_env,
            trace_path=arguments.trace_path,
            max_model_turns=arguments.max_model_turns,
            max_tool_calls=arguments.max_tool_calls,
            max_wall_time_seconds=arguments.max_wall_time_seconds,
            context_char_budget=arguments.context_char_budget,
            model_timeout_seconds=arguments.model_timeout_seconds,
            model_max_retries=arguments.model_max_retries,
            protocol_max_retries=arguments.protocol_max_retries,
            reasoning_effort=arguments.reasoning_effort,
            reasoning_parameter=arguments.reasoning_parameter,
            efficiency_mode=arguments.efficiency_mode,
            reserve_final_turn=arguments.reserve_final_turn,
            convergence_remaining_turns=arguments.convergence_remaining_turns,
            max_repeated_tool_batches=arguments.max_repeated_tool_batches,
            max_no_progress_batches=arguments.max_no_progress_batches,
            environ=os.environ if environ is None else environ,
        )
        event_sink = _build_event_sink(config)
        runtime = _build_runtime(config, event_sink, planning=arguments.planning)
    except (ConfigurationError, OSError) as exc:
        # Configuration errors are designed not to contain a credential.  For
        # an OSError, show only class and filename; arbitrary OS messages can
        # echo a user-controlled path that happened to contain sensitive text.
        if isinstance(exc, OSError):
            location = f" ({exc.filename})" if exc.filename else ""
            detail = f"{type(exc).__name__}{location}"
        else:
            detail = str(exc)
        print(f"configuration error: {detail}", file=sys.stderr)
        return EXIT_USAGE

    try:
        result = runtime.run(config.task)
    except KeyboardInterrupt:
        # AgentRuntime normally converts cancellation into a RunResult.  This
        # boundary remains as protection for interrupts during construction or
        # terminal rendering.
        print("run cancelled by user", file=sys.stderr)
        return EXIT_CANCELLED
    except CodingAgentError as exc:
        # Expected domain failures should ordinarily become FAILED results.
        # Keeping a boundary here prevents a future adapter regression from
        # producing a Python traceback in a two-minute demonstration.
        safe_message = str(exc).replace(config.api_key, "[REDACTED]")
        print(f"runtime error: {safe_message}", file=sys.stderr)
        return EXIT_RUNTIME_FAILED

    verification: VerificationResult | None = None
    if arguments.verification_commands:
        try:
            verification = _run_verification(
                config,
                arguments.verification_commands,
                timeout_seconds=arguments.verification_timeout_seconds,
            )
        except KeyboardInterrupt:
            print("verification cancelled by user", file=sys.stderr)
            return EXIT_CANCELLED

    _print_result(result, verification=verification, secrets=(config.api_key,))
    return _exit_code_for_result(result, verification=verification)


def _build_event_sink(config: AgentConfig) -> EventSink:
    """Create terminal output and, when requested, a redacted JSONL trace."""

    console = ConsoleEventSink(secrets=(config.api_key,))
    if config.trace_path is None:
        return console
    trace = JsonlEventSink(config.trace_path, secrets=(config.api_key,))
    return CompositeEventSink(console, trace)


def _build_runtime(
    config: AgentConfig,
    event_sink: EventSink,
    *,
    planning: bool = False,
    model_client_factory: Any | None = None,
    reasoning_probe_timeout_seconds: float | None = None,
) -> Any:
    """Assemble concrete dependencies after CLI validation has succeeded.

    Imports stay local to preserve a narrow command-line boundary.  A normal
    run performs no network operation while assembling dependencies.  When a
    native reasoning level was requested, one bounded capability probe is the
    deliberate exception: the runtime is not allowed to send an unverified
    reasoning field or silently imitate it in the prompt.

    ``model_client_factory`` and ``reasoning_probe_timeout_seconds`` are small
    injection seams for offline contract tests.  They are keyword-only so the
    production CLI surface remains unchanged.
    """

    from coding_agent.agent import AgentRuntime
    from coding_agent.context import ContextBuilder
    from coding_agent.model import OpenAICompatibleModelClient
    from coding_agent.policy import AgentLimits
    from coding_agent.tools.registry import (
        build_default_registry,
        build_planning_registry,
    )

    factory = (
        OpenAICompatibleModelClient
        if model_client_factory is None
        else model_client_factory
    )
    model_client = factory(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_seconds=config.model_timeout_seconds,
    )
    if config.reasoning_effort is not None:
        _probe_cli_reasoning(
            model_client,
            config,
            event_sink,
            timeout_seconds=reasoning_probe_timeout_seconds,
        )
    # The key variable name is user-configurable and may be something generic
    # such as MODEL_GATEWAY_CREDENTIAL, which no suffix-based sanitizer can
    # infer reliably.  Pass the exact name down to the shell boundary; passing
    # the secret value itself would create an unnecessary leak surface.
    registry_builder = build_planning_registry if planning else build_default_registry
    tool_registry = registry_builder(
        config.workspace,
        excluded_command_environment_names=(config.api_key_env,),
    )
    context_builder = ContextBuilder(max_chars=config.context_char_budget)
    limits = AgentLimits(
        max_model_turns=config.max_model_turns,
        max_tool_calls=config.max_tool_calls,
        max_wall_time_seconds=config.max_wall_time_seconds,
        max_model_retries=config.model_max_retries,
        max_protocol_retries=config.protocol_max_retries,
        efficiency_mode=config.efficiency_mode,
        reserve_final_turn=(config.reserve_final_turn or config.efficiency_mode),
        convergence_remaining_turns=config.convergence_remaining_turns,
        max_repeated_tool_batches=config.max_repeated_tool_batches,
        max_no_progress_batches=config.max_no_progress_batches,
    )
    return AgentRuntime(
        model_client=model_client,
        tool_registry=tool_registry,
        context_builder=context_builder,
        limits=limits,
        event_sink=event_sink,
    )


def _probe_cli_reasoning(
    model_client: Any,
    config: AgentConfig,
    event_sink: EventSink,
    *,
    timeout_seconds: float | None = None,
) -> Any:
    """Probe and apply a native reasoning option before starting the loop.

    The model adapter owns provider-specific error classification.  This CLI
    boundary owns the startup policy: unsupported or indeterminate capability
    is a configuration failure, and only a probe-confirmed field/value may
    reach an autonomous run.  The event payload is deliberately limited to
    capability metadata and passes through the same redaction boundary as all
    other CLI events.
    """

    probe = getattr(model_client, "probe_reasoning_effort", None)
    if not callable(probe):
        capability = {
            "status": "error",
            "requested_effort": config.reasoning_effort,
            "error_type": "MissingCapabilityProbe",
            "detail": "model client does not expose probe_reasoning_effort",
        }
        _emit_cli_reasoning_event(event_sink, config, capability)
        raise ConfigurationError("reasoning capability probe failed")

    effective_timeout = _reasoning_probe_timeout(
        config.model_timeout_seconds, timeout_seconds
    )
    probe_kwargs: dict[str, Any] = {"timeout_seconds": effective_timeout}
    # The adapter can try its portable aliases when the default field is used.
    # An explicitly provider-specific field is a contract: probing another
    # spelling would make the recorded configuration ambiguous.
    if config.reasoning_parameter != "reasoning_effort":
        probe_kwargs["parameter_candidates"] = (config.reasoning_parameter,)

    try:
        capability = probe(config.reasoning_effort, **probe_kwargs)
    except Exception as exc:
        capability = {
            "status": "error",
            "requested_effort": config.reasoning_effort,
            "error_type": type(exc).__name__,
            "detail": str(exc),
        }
        _emit_cli_reasoning_event(event_sink, config, capability)
        raise ConfigurationError("reasoning capability probe failed") from exc

    record = _normalise_cli_capability(capability, config)
    _emit_cli_reasoning_event(event_sink, config, record)
    status = record["status"]
    if status != "supported":
        if status == "unsupported":
            raise ConfigurationError(
                f"native reasoning effort {config.reasoning_effort!r} "
                "is unsupported by the gateway"
            )
        raise ConfigurationError("reasoning capability probe failed")

    # ``OpenAICompatibleModelClient.probe_reasoning_effort`` applies the
    # capability itself.  Calling the optional hook again also makes the
    # boundary correct for a test/integration client whose probe only reports
    # the result and leaves application of the field to its caller.
    configure = getattr(model_client, "configure_reasoning", None)
    already_applied = getattr(model_client, "reasoning_capability", None) == capability
    if callable(configure) and not already_applied:
        try:
            configure(capability)
        except Exception as exc:
            failure = {
                "status": "error",
                "requested_effort": config.reasoning_effort,
                "parameter": record.get("parameter"),
                "accepted_value": record.get("accepted_value"),
                "error_type": type(exc).__name__,
                "detail": str(exc),
            }
            _emit_cli_reasoning_event(event_sink, config, failure)
            raise ConfigurationError("reasoning capability application failed") from exc
    return capability


def _reasoning_probe_timeout(
    model_timeout_seconds: float,
    requested_timeout_seconds: float | None,
) -> float:
    """Validate and cap the optional startup probe timeout."""

    upper_bound = min(float(model_timeout_seconds), REASONING_PROBE_TIMEOUT_SECONDS)
    if requested_timeout_seconds is None:
        return upper_bound
    if (
        isinstance(requested_timeout_seconds, bool)
        or not isinstance(requested_timeout_seconds, (int, float))
        or not math.isfinite(float(requested_timeout_seconds))
        or requested_timeout_seconds <= 0
    ):
        raise ConfigurationError(
            "reasoning_probe_timeout_seconds must be a finite positive number"
        )
    return min(upper_bound, float(requested_timeout_seconds))


def _normalise_cli_capability(capability: Any, config: AgentConfig) -> dict[str, Any]:
    """Copy only JSON-safe capability fields into a bounded event record."""

    if isinstance(capability, Mapping):
        raw = dict(capability)
    else:
        to_dict = getattr(capability, "to_dict", None)
        if callable(to_dict):
            try:
                candidate = to_dict()
            except Exception:  # noqa: BLE001 - diagnostic object boundary
                candidate = {}
            raw = dict(candidate) if isinstance(candidate, Mapping) else {}
        else:
            raw = {}
        if not raw:
            raw = {
                name: getattr(capability, name, None)
                for name in (
                    "status",
                    "requested_effort",
                    "parameter",
                    "accepted_value",
                    "error_type",
                    "detail",
                )
            }

    status = raw.get("status")
    if not isinstance(status, str) or status.strip().lower() not in {
        "supported",
        "unsupported",
        "error",
    }:
        status = "error"
    else:
        status = status.strip().lower()
    requested = raw.get("requested_effort", config.reasoning_effort)
    parameter = raw.get("parameter")
    accepted = raw.get("accepted_value")
    error_type = raw.get("error_type")
    detail = raw.get("detail")
    validation_error: str | None = None
    if parameter is not None and (
        not isinstance(parameter, str)
        or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", parameter) is None
    ):
        validation_error = "probe returned an invalid native reasoning parameter"
    if status == "supported" and accepted is None:
        validation_error = validation_error or (
            "probe returned no native reasoning value"
        )
    if accepted is not None:
        try:
            json.dumps(accepted, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            validation_error = validation_error or (
                "probe returned a non-JSON-serializable reasoning value"
            )
    if requested is not None and (
        not isinstance(requested, str)
        or requested.strip().lower() not in {"low", "medium", "high", "max"}
    ):
        validation_error = validation_error or (
            "probe returned an invalid requested reasoning effort"
        )
    if validation_error is not None:
        status = "error"
        error_type = "InvalidReasoningCapability"
        detail = validation_error
    # A supported probe is useful only when it identifies both the native field
    # and the exact accepted value.  Treat incomplete fakes/provider metadata
    # as an indeterminate setup result rather than guessing.
    if status == "supported" and (
        not isinstance(parameter, str) or not parameter or accepted is None
    ):
        status = "error"
        error_type = "InvalidReasoningCapability"
        detail = "supported probe did not return a native field and value"
    return {
        "status": status,
        "requested_effort": requested,
        "parameter": parameter,
        "accepted_value": accepted,
        "error_type": error_type,
        "detail": detail,
    }


def _emit_cli_reasoning_event(
    event_sink: EventSink,
    config: AgentConfig,
    capability: Mapping[str, Any],
) -> None:
    """Emit a value-free, redacted startup capability event."""

    payload = {
        "model": config.model,
        "base_url": config.base_url,
        "status": capability.get("status"),
        "requested_effort": capability.get("requested_effort"),
        "parameter": capability.get("parameter"),
        "accepted_value": capability.get("accepted_value"),
        "error_type": capability.get("error_type"),
        "detail": capability.get("detail"),
    }
    safe_payload = redact(payload, secrets=(config.api_key,))
    if not isinstance(safe_payload, Mapping):
        safe_payload = {"status": "error", "error_type": "InvalidEventPayload"}
    # Error bodies can be unexpectedly large. Keep the trace bounded without
    # retaining arbitrary provider response text in a local artifact.
    detail = safe_payload.get("detail")
    if isinstance(detail, str) and len(detail) > 1000:
        safe_payload = dict(safe_payload)
        safe_payload["detail"] = detail[:1000]
    try:
        event_sink.emit("reasoning.probe", **dict(safe_payload))
    except Exception:  # noqa: BLE001 - event sinks are diagnostic only
        # Runtime event sinks are diagnostic. A broken console/trace stream
        # must not turn a correctly classified configuration failure into an
        # unrelated traceback (or block a supported run).
        return


def _print_result(
    result: object,
    *,
    verification: VerificationResult | None = None,
    secrets: Iterable[str] = (),
) -> None:
    """Render final model text to stdout and deterministic metadata to stderr."""

    final_text = getattr(result, "final_text", None)
    if isinstance(final_text, str) and final_text:
        safe_text = redact(final_text, secrets=secrets)
        print(safe_text if isinstance(safe_text, str) else str(safe_text))

    fields: list[str] = [f"phase={_phase_text(result)}"]
    reason = getattr(result, "reason", None)
    if reason:
        safe_reason = redact(str(reason), secrets=secrets)
        fields.append(f"reason={safe_reason}")
    for attribute in (
        "model_turns",
        "model_requests",
        "tool_calls",
        "elapsed_seconds",
    ):
        value = getattr(result, attribute, None)
        if value is not None:
            fields.append(f"{attribute}={value}")
    print("run summary: " + " ".join(fields), file=sys.stderr)
    if verification is not None:
        check_count = len(verification.checks)
        passed_count = sum(item.passed for item in verification.checks)
        print(
            "verification summary: "
            f"passed={verification.passed} checks={passed_count}/{check_count} "
            f"reason={verification.reason} "
            f"elapsed_seconds={verification.elapsed_seconds:.6f}",
            file=sys.stderr,
        )
        for item in verification.checks:
            print(
                "verification check: "
                f"name={item.check.name} passed={item.passed} "
                f"exit_code={item.exit_code} timed_out={item.timed_out}",
                file=sys.stderr,
            )


def _exit_code_for_result(
    result: object,
    *,
    verification: VerificationResult | None = None,
) -> int:
    phase = _phase_text(result).lower()
    if phase == "completed":
        if verification is not None and not verification.passed:
            return EXIT_VERIFICATION_FAILED
        return EXIT_COMPLETED
    if phase in {"limit_reached", "limit-reached"}:
        return EXIT_LIMIT_REACHED
    if phase == "cancelled":
        return EXIT_CANCELLED
    # A runtime failure/limit/cancellation remains the primary outcome.  A
    # failed acceptance check gets its own code only after a normal completion;
    # this prevents a verifier from hiding why the agent itself stopped.
    return EXIT_RUNTIME_FAILED


def _run_verification(
    config: AgentConfig,
    commands: Sequence[str],
    *,
    timeout_seconds: float,
) -> VerificationResult:
    """Run trusted, fixed acceptance commands after the Runtime returns.

    These commands are CLI configuration, not model-generated input.  They are
    intentionally kept outside ``AgentRuntime`` so their output cannot alter
    canonical history, model/tool counters, or the meaning of ``COMPLETED``.
    Construction/validation failures become a structured failed result rather
    than a traceback at the terminal boundary.
    """

    try:
        checks = tuple(
            VerificationCheck(
                name=f"check-{index}",
                command=command,
                timeout_seconds=timeout_seconds,
            )
            for index, command in enumerate(commands, start=1)
        )
        return CommandVerifier(
            config.workspace,
            checks,
            excluded_environment_names=(config.api_key_env,),
        ).verify()
    except Exception as exc:  # noqa: BLE001 - extension/config boundary
        return VerificationResult(
            passed=False,
            reason="verification setup failed",
            error=type(exc).__name__,
        )


def _phase_text(result: object) -> str:
    phase = getattr(result, "phase", "unknown")
    value = getattr(phase, "value", phase)
    return str(value)


if __name__ == "__main__":  # pragma: no cover - covered through __main__.py
    raise SystemExit(main())
