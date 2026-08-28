"""Command-line boundary for configuring and running one coding-agent task.

The CLI owns user-facing parsing, environment lookup and exit codes.  It does
not own the agent loop.  Imports of the runtime and concrete tools are delayed
until after configuration has been validated; this keeps ``--help`` cheap and
allows configuration/event unit tests to run without constructing an SDK
client or touching the network.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from coding_agent.config import AgentConfig, ConfigurationError, Provider
from coding_agent.errors import CodingAgentError
from coding_agent.events import (
    CompositeEventSink,
    ConsoleEventSink,
    EventSink,
    JsonlEventSink,
)

# Exit codes are intentionally small and stable so a benchmark harness can
# distinguish task/runtime outcomes from malformed invocation.  Detailed run
# semantics remain in RunResult and the optional JSONL trace.
EXIT_COMPLETED = 0
EXIT_RUNTIME_FAILED = 1
EXIT_USAGE = 2
EXIT_LIMIT_REACHED = 3
EXIT_CANCELLED = 130


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
            environ=os.environ if environ is None else environ,
        )
        event_sink = _build_event_sink(config)
        runtime = _build_runtime(config, event_sink)
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

    _print_result(result)
    return _exit_code_for_result(result)


def _build_event_sink(config: AgentConfig) -> EventSink:
    """Create terminal output and, when requested, a redacted JSONL trace."""

    console = ConsoleEventSink(secrets=(config.api_key,))
    if config.trace_path is None:
        return console
    trace = JsonlEventSink(config.trace_path, secrets=(config.api_key,))
    return CompositeEventSink(console, trace)


def _build_runtime(config: AgentConfig, event_sink: EventSink) -> Any:
    """Assemble concrete dependencies after CLI validation has succeeded.

    Imports stay local to preserve a narrow command-line boundary.  No API
    request is made here: the ordinary provider client is constructed, but the
    first network operation occurs only inside ``AgentRuntime.run``.
    """

    from coding_agent.agent import AgentRuntime
    from coding_agent.context import ContextBuilder
    from coding_agent.model import OpenAICompatibleModelClient
    from coding_agent.policy import AgentLimits
    from coding_agent.tools.registry import build_default_registry

    model_client = OpenAICompatibleModelClient(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_seconds=config.model_timeout_seconds,
    )
    # The key variable name is user-configurable and may be something generic
    # such as MODEL_GATEWAY_CREDENTIAL, which no suffix-based sanitizer can
    # infer reliably.  Pass the exact name down to the shell boundary; passing
    # the secret value itself would create an unnecessary leak surface.
    tool_registry = build_default_registry(
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
    )
    return AgentRuntime(
        model_client=model_client,
        tool_registry=tool_registry,
        context_builder=context_builder,
        limits=limits,
        event_sink=event_sink,
    )


def _print_result(result: object) -> None:
    """Render final model text to stdout and deterministic metadata to stderr."""

    final_text = getattr(result, "final_text", None)
    if isinstance(final_text, str) and final_text:
        print(final_text)

    fields: list[str] = [f"phase={_phase_text(result)}"]
    reason = getattr(result, "reason", None)
    if reason:
        fields.append(f"reason={reason}")
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


def _exit_code_for_result(result: object) -> int:
    phase = _phase_text(result).lower()
    if phase == "completed":
        return EXIT_COMPLETED
    if phase in {"limit_reached", "limit-reached"}:
        return EXIT_LIMIT_REACHED
    if phase == "cancelled":
        return EXIT_CANCELLED
    return EXIT_RUNTIME_FAILED


def _phase_text(result: object) -> str:
    phase = getattr(result, "phase", "unknown")
    value = getattr(phase, "value", phase)
    return str(value)


if __name__ == "__main__":  # pragma: no cover - covered through __main__.py
    raise SystemExit(main())
