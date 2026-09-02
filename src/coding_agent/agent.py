"""The explicit, finite control loop of the coding agent.

This module is intentionally the only place that decides whether the Agent
calls the model, invokes a tool, retries, or stops.  Vendor SDKs, tools, and
context builders are passive dependencies.  Concentrating orchestration here
makes the assignment's central logic visible and lets tests drive every state
transition with a scripted model.

The loop maintains one important wire-protocol invariant:

    every assistant tool call is followed by exactly one tool-result message
    with the same call ID before the model is called again.

Even malformed JSON, unknown tools, exhausted tool budgets, cancellation, and
skipped calls therefore receive a result envelope.  Without that discipline,
the next Chat Completions request would contain an orphaned tool call and many
providers would reject the entire conversation.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from coding_agent.context import (
    ContextBuilder,
    ContextWindow,
    estimate_request_chars,
)
from coding_agent.errors import (
    CodingAgentError,
    ContextOverflow,
    InvariantViolation,
    PermanentModelError,
    ResponseProtocolError,
    TransientModelError,
)
from coding_agent.events import EventSink, NullEventSink
from coding_agent.model import ModelClient
from coding_agent.policy import AgentLimits, EfficiencyPolicy, RetryPolicy
from coding_agent.tools.backend import ExecutionBackend
from coding_agent.types import (
    Message,
    ModelTurn,
    RunPhase,
    RunState,
    ToolCall,
    ToolResult,
    Usage,
)

DEFAULT_SYSTEM_PROMPT = """You are a coding agent operating on a user-selected local workspace.

First form a short phase plan in your reasoning. Inspect the project, make
focused changes, and run appropriate checks. When several reads, searches, or
independent tests do not depend on one another, request them in one tool-call
batch. Treat every tool result as an observation, including non-zero command
exits and structured tool errors. Paths passed to file tools must be relative
to the workspace. Do not claim that a check passed unless its tool result says
so. When the task is complete or cannot be advanced responsibly, return a
concise final response describing changes, checks, and remaining limitations.
A final response stops the runtime; it is not itself proof that the code is
correct.
"""

_CONVERGENCE_REMINDER = (
    "Converge now: stop repeated exploration, prioritize the smallest required "
    "edits, run a focused verification, and return a final answer before the "
    "turn budget is exhausted."
)
_REPLAN_REMINDER = (
    "The recent tool calls made no measurable progress or repeated the same "
    "request. Re-plan from the evidence already collected, choose a different "
    "action, and avoid repeating that tool call."
)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Immutable summary returned after the Runtime enters a terminal phase.

    ``phase=COMPLETED`` means the model returned a non-empty final response
    without tool calls.  It deliberately does not expose a ``success`` or
    ``verified`` boolean: task correctness must be judged by an external test or
    acceptance command, not by the model choosing to stop.
    """

    phase: RunPhase
    reason: str
    final_text: str | None
    model_turns: int
    model_requests: int
    tool_calls: int
    elapsed_seconds: float
    usage: Usage | None
    history: tuple[Message, ...]


class _WallTimeExpired(Exception):
    """Private control signal used to enter LIMIT_REACHED, not FAILED."""


class AgentRuntime:
    """Run one model/tool feedback loop under explicit resource limits.

    Dependencies are injected rather than constructed here.  A production CLI
    supplies an OpenAI-compatible model client and real local tools; unit tests
    supply a ScriptedModel and temporary-workspace tools.  This is ordinary
    dependency inversion, not an Agent framework: all orchestration remains in
    :meth:`run` and its small private helpers below.
    """

    def __init__(
        self,
        model_client: ModelClient,
        tool_registry: ExecutionBackend,
        context_builder: ContextBuilder,
        limits: AgentLimits | None = None,
        event_sink: EventSink | None = None,
        *,
        retry_policy: RetryPolicy | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sleep: Callable[[float], None] = time.sleep,
        cancel_check: Callable[[], bool] | None = None,
        efficiency_policy: EfficiencyPolicy | None = None,
    ) -> None:
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("system_prompt must be a non-empty string")
        self.model_client = model_client
        self.tool_registry = tool_registry
        self.context_builder = context_builder
        self.limits = limits or AgentLimits()
        if efficiency_policy is not None and not isinstance(
            efficiency_policy, EfficiencyPolicy
        ):
            raise TypeError("efficiency_policy must be an EfficiencyPolicy")
        self.efficiency_policy = efficiency_policy or EfficiencyPolicy(
            enabled=self.limits.efficiency_mode,
            # Efficiency mode is a complete turn-management strategy: the
            # convergence reminders are useful only when the controller also
            # leaves one tool-free turn for the final response.  Keep the
            # explicit reserve flag independently useful for callers that do
            # not enable the other efficiency hints.
            reserve_final_turn=(
                self.limits.reserve_final_turn or self.limits.efficiency_mode
            ),
            convergence_remaining_turns=self.limits.convergence_remaining_turns,
            max_repeated_tool_batches=self.limits.max_repeated_tool_batches,
            max_no_progress_batches=self.limits.max_no_progress_batches,
        )
        self.event_sink = event_sink or NullEventSink()
        self.retry_policy = retry_policy or RetryPolicy()
        self.system_prompt = system_prompt.strip()
        self._sleep = sleep
        if cancel_check is not None and not callable(cancel_check):
            raise TypeError("cancel_check must be callable when supplied")
        self._cancel_check = cancel_check
        self._pending_efficiency_reminder: str | None = None
        self._last_tool_signature: str | None = None
        self._repeated_tool_batches = 0
        self._last_result_signature: str | None = None
        self._no_progress_batches = 0

    def run(
        self,
        task: str,
        *,
        history: Sequence[Message] | None = None,
    ) -> RunResult:
        """Execute ``task`` until a documented terminal condition is reached.

        ``history`` may contain a completed earlier run. When supplied, the new
        user task is appended to that canonical conversation instead of adding
        another system message. Runtime counters and deadlines still start
        fresh for every call.

        The outer exception boundary converts expected runtime failures and
        unexpected implementation failures into ``FAILED`` results so the CLI
        can always print a terminal summary.  It never retries a local tool.
        Tests can inspect the event and terminal reason while a developer still
        gets the exception type in the trace; raw exception messages are not
        copied into the final reason because they may contain local paths or
        provider details.
        """

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        if history is None:
            initial_history = [Message(role="system", content=self.system_prompt)]
        else:
            if isinstance(history, (str, bytes)) or not isinstance(history, Sequence):
                raise TypeError("history must be a sequence of Message objects")
            initial_history = list(history)
            if not initial_history:
                raise ValueError("history must not be empty when supplied")
            if any(not isinstance(message, Message) for message in initial_history):
                raise TypeError("history must contain only Message objects")

        initial_history.append(Message(role="user", content=task.strip()))
        state = RunState(history=initial_history)
        final_text: str | None = None
        usage_total = _UsageAccumulator()
        # One absolute deadline is shared by model requests, retry delays, and
        # local tools.  Recomputing independent relative budgets at each layer
        # would allow a run to spend the full limit several times.
        deadline = state.started_at + self.limits.max_wall_time_seconds
        self._pending_efficiency_reminder = None
        self._last_tool_signature = None
        self._repeated_tool_batches = 0
        self._last_result_signature = None
        self._no_progress_batches = 0

        self._emit(
            "run.started",
            max_model_turns=self.limits.max_model_turns,
            max_tool_calls=self.limits.max_tool_calls,
            max_wall_time_seconds=self.limits.max_wall_time_seconds,
        )

        try:
            while not state.phase.is_terminal:
                if self._cancel_requested():
                    self._terminate(
                        state, RunPhase.CANCELLED, "run cancellation requested"
                    )
                    break
                limit_reason = self._pre_model_limit_reason(state)
                if limit_reason is not None:
                    self._terminate(state, RunPhase.LIMIT_REACHED, limit_reason)
                    break

                final_response_only = self._final_response_only(state)
                turn, context_window = self._next_valid_turn(
                    state,
                    usage_total,
                    deadline,
                    allow_tools=not final_response_only,
                )
                state.model_turns += 1
                self._emit(
                    "model.completed",
                    model_turn=state.model_turns,
                    tool_calls=len(turn.tool_calls),
                    has_text=bool(turn.text and turn.text.strip()),
                    finish_reason=turn.finish_reason,
                    context_messages=len(context_window),
                    context_estimated_chars=context_window.estimated_chars,
                )

                # _next_valid_turn rejects an empty response, so the remaining
                # cases are exhaustive: tool work or a non-empty final answer.
                if turn.tool_calls:
                    if final_response_only:
                        # This should be caught by protocol validation; retain a
                        # defensive terminal branch for injected model clients.
                        self._terminate(
                            state,
                            RunPhase.FAILED,
                            "final response turn requested tools",
                        )
                        break
                    terminal = self._execute_tool_batch(state, turn, deadline)
                    if terminal is not None:
                        phase, reason = terminal
                        self._terminate(state, phase, reason)
                        break

                    state.transition(RunPhase.CHECKING_LIMITS)
                    # In efficiency mode the last turn is reserved for a
                    # tool-free final response. The old strategy keeps its
                    # historical immediate limit behavior.
                    if (
                        state.model_turns >= self.limits.max_model_turns
                        and not self._reserve_final_turn()
                    ):
                        self._terminate(
                            state,
                            RunPhase.LIMIT_REACHED,
                            "maximum model turns reached after tool execution",
                        )
                        break
                    if self._wall_time_exhausted(state):
                        self._terminate(
                            state,
                            RunPhase.LIMIT_REACHED,
                            "maximum wall time reached after tool execution",
                        )
                        break
                    continue

                # Tool calls take precedence when text and calls coexist.  This
                # branch is reached only when there are no calls, so the text is
                # the model's final response rather than intermediate narration.
                assert turn.text is not None and turn.text.strip()
                assistant_message = turn.as_message()
                state.history.append(assistant_message)
                final_text = turn.text.strip()
                self._terminate(
                    state,
                    RunPhase.COMPLETED,
                    "model returned a final response",
                )

        except KeyboardInterrupt:
            # KeyboardInterrupt outside a tool batch has no pending call IDs.
            # The batch helper handles interruption separately so it can append
            # skipped results and leave canonical history protocol-valid.
            if not state.phase.is_terminal:
                self._terminate(state, RunPhase.CANCELLED, "user interrupted the run")
        except _WallTimeExpired:
            if not state.phase.is_terminal:
                self._terminate(
                    state,
                    RunPhase.LIMIT_REACHED,
                    "maximum wall time reached",
                )
        except CodingAgentError as exc:
            if not state.phase.is_terminal:
                self._emit("run.error", error_type=type(exc).__name__)
                self._terminate(
                    state,
                    RunPhase.FAILED,
                    f"runtime error: {type(exc).__name__}",
                )
        except Exception as exc:  # noqa: BLE001 - terminal process boundary
            # Do not expose ``str(exc)`` to the terminal summary or model.  The
            # event sink records the class only; tests should make unexpected
            # errors fail visibly, while production users receive a stable exit
            # status without a credential-bearing provider message.
            if not state.phase.is_terminal:
                self._emit("run.error", error_type=type(exc).__name__)
                self._terminate(
                    state,
                    RunPhase.FAILED,
                    f"unexpected runtime error: {type(exc).__name__}",
                )

        if not state.phase.is_terminal or state.terminal_reason is None:
            raise InvariantViolation("agent loop exited without a terminal state")

        result = RunResult(
            phase=state.phase,
            reason=state.terminal_reason,
            final_text=final_text,
            model_turns=state.model_turns,
            model_requests=state.model_requests,
            tool_calls=state.tool_calls,
            elapsed_seconds=state.elapsed_seconds,
            usage=usage_total.value(),
            history=tuple(state.history),
        )
        self._emit(
            "run.ended",
            phase=result.phase.value,
            reason=result.reason,
            model_turns=result.model_turns,
            model_requests=result.model_requests,
            tool_calls=result.tool_calls,
            elapsed_seconds=round(result.elapsed_seconds, 6),
        )
        return result

    def _next_valid_turn(
        self,
        state: RunState,
        usage_total: _UsageAccumulator,
        deadline: float,
        *,
        allow_tools: bool = True,
    ) -> tuple[ModelTurn, ContextWindow]:
        """Build context and obtain one structurally usable model response.

        Transport retries and protocol retries are deliberately independent.
        A timeout may safely resend the same read-only model request; a valid
        HTTP response with malformed protocol consumes the protocol retry
        budget instead.  Neither category appends anything to canonical history
        until a usable turn is obtained.
        """

        protocol_failures = 0
        extra_reserved_chars = 0
        context_overflow_retried = False
        previous_request_chars: int | None = None

        while True:
            self._require_remaining_time(state, deadline)
            state.transition(RunPhase.BUILDING_CONTEXT)
            schemas = self.tool_registry.model_schemas() if allow_tools else ()
            reminder = self._next_efficiency_reminder(state)
            reminder_reservation = (
                _serialized_message_reservation(reminder) if reminder else 0
            )
            window = self.context_builder.build(
                state.history,
                schemas,
                reserved_chars=extra_reserved_chars + reminder_reservation,
            )
            if reminder:
                window = _inject_context_reminder(
                    window,
                    reminder,
                    tools=schemas,
                    # ``ContextBuilder`` already reserved the reminder while
                    # selecting blocks.  After the message is inserted, count
                    # the actual reminder once and retain only the provider
                    # overflow reservation; adding ``reminder_reservation``
                    # again would make the reported window exceed its budget.
                    reserved_chars=extra_reserved_chars,
                )
            request_chars = estimate_request_chars(window, schemas)
            if (
                previous_request_chars is not None
                and request_chars >= previous_request_chars
            ):
                # Retrying a provider context rejection with an identical view
                # only burns another request.  If no complete optional block can
                # be removed, fail explicitly and let the user reduce the fixed
                # prompt/tool schemas or increase the provider context window.
                raise ContextOverflow(
                    "provider context-overflow recovery could not make the "
                    "protocol-complete request smaller"
                )
            if window.truncated:
                self._emit(
                    "context.truncated",
                    omitted_blocks=window.omitted_blocks,
                    estimated_chars=window.estimated_chars,
                )

            state.transition(RunPhase.CALLING_MODEL)
            try:
                turn = self._call_model_with_retries(
                    window,
                    state,
                    usage_total,
                    deadline,
                    tools=schemas,
                )
                # Response normalization happens in the adapter.  Structural
                # and conversation-level validation below is still parsing from
                # the controller's perspective, so expose that state before an
                # invalid turn can trigger a protocol retry.
                state.transition(RunPhase.PARSING_RESPONSE)
                self._validate_turn(
                    turn,
                    state.history,
                    allow_tool_calls=allow_tools,
                )
                self._require_remaining_time(state, deadline)
                return turn, window
            except ContextOverflow:
                if context_overflow_retried:
                    raise
                # A compatible gateway may count tokens differently from our
                # provider-neutral character approximation.  Force roughly 20%
                # less payload once; repeated rejection is terminal rather than
                # an unbounded shrink/retry loop.  The reservation is derived
                # from the request that actually failed, not from the configured
                # maximum, so the next view must be observably smaller even when
                # the first request was far below that configured maximum.
                context_overflow_retried = True
                previous_request_chars = request_chars
                target_request_chars = max(1, int(request_chars * 0.8))
                extra_reserved_chars = max(
                    1,
                    self.context_builder.max_chars - target_request_chars,
                )
                self._emit(
                    "context.provider_overflow",
                    action="retry_with_smaller_view",
                    extra_reserved_chars=extra_reserved_chars,
                )
                continue
            except ResponseProtocolError as exc:
                if protocol_failures >= self.limits.max_protocol_retries:
                    raise
                protocol_failures += 1
                self._emit(
                    "model.protocol_retry",
                    retry=protocol_failures,
                    error_type=type(exc).__name__,
                )

    def _call_model_with_retries(
        self,
        messages: Sequence[Message],
        state: RunState,
        usage_total: _UsageAccumulator,
        deadline: float,
        *,
        tools: Sequence[dict[str, object]] | Sequence[object] = (),
    ) -> ModelTurn:
        """Call the model with bounded retries for transient failures only."""

        retries = 0
        while True:
            remaining_seconds = self._require_remaining_time(state, deadline)
            state.model_requests += 1
            self._emit(
                "model.requested",
                message_count=len(messages),
                retry=retries,
                model_request=state.model_requests,
                remaining_wall_seconds=round(remaining_seconds, 6),
            )
            try:
                turn = self.model_client.complete(
                    messages,
                    tools,
                    timeout_seconds=remaining_seconds,
                )
                # Usage belongs to physical API responses, including responses
                # later rejected by protocol validation.  Accumulating here is
                # essential for honest benchmark accounting.
                usage_total.add(turn.usage)
                self._require_remaining_time(state, deadline)
                return turn
            except TransientModelError as exc:
                self._require_remaining_time(state, deadline)
                if retries >= self.limits.max_model_retries:
                    raise
                delay = self.retry_policy.delay_seconds(retries)
                retries += 1
                self._emit(
                    "model.retry",
                    retry=retries,
                    delay_seconds=round(delay, 6),
                    error_type=type(exc).__name__,
                )
                # Starting a retry after its required backoff would cross the
                # run deadline.  Stop immediately instead of sleeping away the
                # remainder and issuing a request with no usable budget.
                if delay >= self._remaining_seconds(state, deadline):
                    raise _WallTimeExpired
                self._sleep(delay)
                self._require_remaining_time(state, deadline)
            except (PermanentModelError, ContextOverflow, ResponseProtocolError):
                raise

    @staticmethod
    def _validate_turn(
        turn: ModelTurn,
        history: Sequence[Message],
        *,
        allow_tool_calls: bool = True,
    ) -> None:
        """Reject empty output and call IDs already used in earlier turns."""

        if turn.finish_reason in {"length", "content_filter"}:
            # Text or JSON arguments may be only a prefix when generation hit a
            # length/content policy boundary.  Neither executing those calls nor
            # presenting the fragment as a completed task is safe.
            raise ResponseProtocolError(
                f"assistant response is incomplete ({turn.finish_reason})"
            )

        if not allow_tool_calls and turn.tool_calls:
            raise ResponseProtocolError(
                "final response turn must not contain tool calls"
            )

        if not turn.tool_calls and not (turn.text and turn.text.strip()):
            raise ResponseProtocolError(
                "assistant response contains neither tool calls nor non-empty text"
            )

        # The response normalizer normally rejects duplicates while it is
        # converting the provider object.  Keep the invariant at the
        # controller boundary too: injected model clients and future adapters
        # can return a ``ModelTurn`` directly, and a duplicate ID in one
        # assistant message would make result pairing ambiguous.
        current_ids: set[str] = set()
        for call in turn.tool_calls:
            if call.id in current_ids:
                raise ResponseProtocolError(
                    f"model returned duplicate tool call id: {call.id!r}"
                )
            current_ids.add(call.id)

        used_ids = {
            call.id
            for message in history
            if message.role == "assistant"
            for call in message.tool_calls
        }
        reused = [call.id for call in turn.tool_calls if call.id in used_ids]
        if reused:
            raise ResponseProtocolError(
                f"model reused a previous tool call id: {reused[0]!r}"
            )

    def _efficiency_enabled(self) -> bool:
        return self.efficiency_policy.enabled

    def _reserve_final_turn(self) -> bool:
        # ``--reserve-final-turn`` is useful independently of reminders (for
        # example in a controlled ablation), so do not silently discard an
        # explicit true value merely because the reminder policy is disabled.
        return self.efficiency_policy.reserve_final_turn

    def _final_response_only(self, state: RunState) -> bool:
        """Whether this request is the reserved, tool-free final turn."""

        return self._reserve_final_turn() and (
            state.model_turns == self.limits.max_model_turns - 1
        )

    def _next_efficiency_reminder(self, state: RunState) -> str | None:
        if not self._efficiency_enabled():
            return None
        if self._pending_efficiency_reminder is not None:
            reminder = self._pending_efficiency_reminder
            self._pending_efficiency_reminder = None
            self._emit(
                "efficiency.reminder",
                kind="replan" if reminder == _REPLAN_REMINDER else "convergence",
                model_turn=state.model_turns + 1,
            )
            return reminder
        remaining = self.limits.max_model_turns - state.model_turns
        if remaining <= self.efficiency_policy.convergence_remaining_turns:
            self._emit(
                "efficiency.reminder",
                kind="convergence",
                model_turn=state.model_turns + 1,
                remaining_turns=remaining,
            )
            return _CONVERGENCE_REMINDER
        return None

    def _observe_tool_batch(
        self,
        turn: ModelTurn,
        results: Sequence[ToolResult],
    ) -> None:
        """Detect repeated/no-progress batches and schedule one re-plan hint."""

        if not self._efficiency_enabled():
            return
        signature_payload = [
            (call.name, call.arguments_json) for call in turn.tool_calls
        ]
        signature = hashlib.sha256(
            json.dumps(
                signature_payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if signature == self._last_tool_signature:
            self._repeated_tool_batches += 1
        else:
            self._repeated_tool_batches = 0
        self._last_tool_signature = signature

        result_payload = [
            (item.name, item.ok, item.error_code, item.content, dict(item.metadata))
            for item in results
        ]
        result_signature = hashlib.sha256(
            json.dumps(
                result_payload, ensure_ascii=False, sort_keys=True, default=str
            ).encode("utf-8")
        ).hexdigest()
        if result_signature == self._last_result_signature:
            self._no_progress_batches += 1
        else:
            self._no_progress_batches = 0
        self._last_result_signature = result_signature

        repeated = (
            self._repeated_tool_batches
            >= self.efficiency_policy.max_repeated_tool_batches
        )
        stagnant = (
            self._no_progress_batches >= self.efficiency_policy.max_no_progress_batches
        )
        if (repeated or stagnant) and self._pending_efficiency_reminder is None:
            self._pending_efficiency_reminder = _REPLAN_REMINDER
            self._emit(
                "efficiency.replan_requested",
                repeated_batches=self._repeated_tool_batches,
                no_progress_batches=self._no_progress_batches,
            )

    def _execute_tool_batch(
        self,
        state: RunState,
        turn: ModelTurn,
        deadline: float,
    ) -> tuple[RunPhase, str] | None:
        """Execute one response's calls serially and append one complete block.

        The batch is admitted atomically with respect to the tool-call budget:
        if executing all calls would exceed the limit, none is run.  Every call
        still receives a ``tool_budget_exceeded`` result, keeping history valid
        without leaving a half-executed model plan in the workspace.
        """

        calls = turn.tool_calls
        state.transition(RunPhase.EXECUTING_TOOLS)

        if state.tool_calls + len(calls) > self.limits.max_tool_calls:
            state.tool_calls += len(calls)
            results = [
                ToolResult(
                    call_id=call.id,
                    name=call.name,
                    ok=False,
                    content="Tool batch was not executed because it exceeds the run budget.",
                    metadata={"error_kind": "limit", "executed": False},
                    error_code="tool_budget_exceeded",
                )
                for call in calls
            ]
            self._record_tool_transaction(state, turn, results)
            return RunPhase.LIMIT_REACHED, "requested tool batch exceeds tool budget"

        results: list[ToolResult] = []
        state.tool_calls += len(calls)
        for index, call in enumerate(calls):
            if self._cancel_requested():
                results.extend(
                    self._skipped_results(
                        calls[index:],
                        error_code="cancelled",
                        content="Tool was skipped after run cancellation was requested.",
                    )
                )
                self._record_tool_transaction(state, turn, results)
                return (
                    RunPhase.CANCELLED,
                    "run cancellation requested during tool batch",
                )
            if self._wall_time_exhausted(state):
                results.extend(
                    self._skipped_results(
                        calls[index:],
                        error_code="wall_time_exceeded",
                        content="Tool was skipped because the run wall-time budget expired.",
                    )
                )
                self._record_tool_transaction(state, turn, results)
                return RunPhase.LIMIT_REACHED, "wall time expired during tool batch"

            self._emit(
                "tool.requested",
                call_id=call.id,
                tool=call.name,
                position=index,
                batch_size=len(calls),
            )
            try:
                result = self._execute_one_call(
                    call,
                    timeout_seconds=self._require_remaining_time(state, deadline),
                )
            except _WallTimeExpired:
                # The run deadline may expire between the pre-tool check and
                # computing the timeout passed to a backend. Preserve the
                # assistant/tool transaction even in that narrow race by
                # recording skipped results for the current and remaining
                # calls before entering the terminal limit state.
                results.extend(
                    self._skipped_results(
                        calls[index:],
                        error_code="wall_time_exceeded",
                        content="Tool was skipped because the run wall-time budget expired.",
                    )
                )
                self._record_tool_transaction(state, turn, results)
                return RunPhase.LIMIT_REACHED, "wall time expired during tool batch"
            except Exception as exc:  # noqa: BLE001 - backend extension boundary
                # A third-party backend should honor ExecutionBackend's
                # always-return contract, but a defensive envelope here keeps
                # canonical history valid even when it raises unexpectedly.
                # The current call's outcome is unknown; remaining calls are
                # never attempted and receive explicit skipped results.
                results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        ok=False,
                        content="Tool backend raised an unexpected exception; outcome is unknown.",
                        metadata={
                            "error_kind": "backend_exception",
                            "outcome_known": False,
                            "exception_type": type(exc).__name__,
                        },
                        error_code="tool_backend_error",
                    )
                )
                results.extend(
                    self._skipped_results(
                        calls[index + 1 :],
                        error_code="tool_backend_error",
                        content="Tool was skipped after the backend raised an exception.",
                    )
                )
                self._record_tool_transaction(state, turn, results)
                return RunPhase.FAILED, "tool backend raised an exception"
            except KeyboardInterrupt:
                # The interrupted operation may already have changed local
                # state.  We report an unknown outcome and never retry it.
                results.append(
                    ToolResult(
                        call_id=call.id,
                        name=call.name,
                        ok=False,
                        content=(
                            "Tool was interrupted; its local side effects may be "
                            "partially complete and were not retried."
                        ),
                        metadata={"error_kind": "cancelled", "outcome_known": False},
                        error_code="tool_interrupted",
                    )
                )
                results.extend(
                    self._skipped_results(
                        calls[index + 1 :],
                        error_code="cancelled",
                        content="Tool was skipped after user cancellation.",
                    )
                )
                self._record_tool_transaction(state, turn, results)
                return RunPhase.CANCELLED, "user interrupted a tool batch"

            results.append(result)
            self._emit(
                "tool.completed",
                call_id=call.id,
                tool=call.name,
                ok=result.ok,
                error_code=result.error_code,
                content_chars=len(result.content),
                metadata=dict(result.metadata),
            )

        self._observe_tool_batch(turn, results)
        self._record_tool_transaction(state, turn, results)
        return None

    def _execute_one_call(
        self,
        call: ToolCall,
        *,
        timeout_seconds: float,
    ) -> ToolResult:
        """Route one untrusted raw call through the single Registry boundary.

        ``execute_call`` owns strict JSON parsing, schema validation, name
        lookup, handler invocation, and error wrapping.  Keeping all of those
        steps in the Registry prevents a future controller path from bypassing
        validation while still preserving the raw argument string in history.
        """

        return self.tool_registry.execute_call(
            call,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _skipped_results(
        calls: Sequence[ToolCall],
        *,
        error_code: str,
        content: str,
    ) -> list[ToolResult]:
        return [
            ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=content,
                metadata={"error_kind": "skipped", "executed": False},
                error_code=error_code,
            )
            for call in calls
        ]

    @staticmethod
    def _record_tool_transaction(
        state: RunState,
        turn: ModelTurn,
        results: Sequence[ToolResult],
    ) -> None:
        """Append assistant call and all results as one logical transaction."""

        if len(results) != len(turn.tool_calls):
            raise InvariantViolation(
                "tool transaction must contain one result for every call"
            )
        expected_calls = [(call.id, call.name) for call in turn.tool_calls]
        actual_calls = [(result.call_id, result.name) for result in results]
        if expected_calls != actual_calls:
            raise InvariantViolation(
                "tool results must preserve model call order, identifiers, and names"
            )

        messages = [turn.as_message(), *(result.to_message() for result in results)]
        state.transition(RunPhase.RECORDING_RESULTS)
        state.history.extend(messages)

    def _pre_model_limit_reason(self, state: RunState) -> str | None:
        if state.model_turns >= self.limits.max_model_turns:
            return "maximum model turns reached"
        if self._wall_time_exhausted(state):
            return "maximum wall time reached"
        return None

    def _wall_time_exhausted(self, state: RunState) -> bool:
        return state.elapsed_seconds >= self.limits.max_wall_time_seconds

    @staticmethod
    def _remaining_seconds(state: RunState, deadline: float) -> float:
        """Return the remaining run budget using the monotonic deadline."""

        return max(0.0, deadline - time.monotonic())

    def _require_remaining_time(self, state: RunState, deadline: float) -> float:
        """Return positive remaining time or leave the loop via a limit signal."""

        remaining = self._remaining_seconds(state, deadline)
        if remaining <= 0 or self._wall_time_exhausted(state):
            raise _WallTimeExpired
        return remaining

    def _terminate(self, state: RunState, phase: RunPhase, reason: str) -> None:
        state.transition(phase, reason=reason)
        self._emit("run.terminal", phase=phase.value, reason=reason)

    def _emit(self, event_type: str, **data: object) -> None:
        # A trace is diagnostic rather than a recovery journal.  Disk-full,
        # broken-pipe, and permission failures after startup must not corrupt
        # the model/tool protocol or turn an otherwise valid coding run into a
        # traceback.  Once a sink reports an OS-level I/O failure, disable it for
        # the rest of this Runtime instance to avoid repeated failing writes.
        if getattr(self, "_event_sink_failed", False):
            return
        try:
            self.event_sink.emit(event_type, **data)
        except OSError:
            self._event_sink_failed = True

    def _cancel_requested(self) -> bool:
        """Return whether an injected host has requested cooperative cancel.

        The callback is intentionally polled only at protocol boundaries and
        between tools.  A local tool that is already running keeps its own
        timeout/termination semantics; interrupting it from an arbitrary
        thread would risk an unknown side effect.  The default ``None`` path
        preserves the original Runtime behaviour exactly.
        """

        return bool(self._cancel_check and self._cancel_check())


def _serialized_message_reservation(text: str | None) -> int:
    if not text:
        return 0
    # Include a generous fixed envelope margin for list separators and the
    # system-role wrapper. This is a budget reservation, not token accounting.
    return len(json.dumps({"role": "system", "content": text}, ensure_ascii=False)) + 64


def _inject_context_reminder(
    window: ContextWindow,
    text: str,
    *,
    tools: Sequence[Mapping[str, object]],
    reserved_chars: int = 0,
) -> ContextWindow:
    """Add an ephemeral system hint while keeping canonical history untouched."""

    reminder = Message(role="system", content=text)
    messages = list(window.messages)
    insert_at = next(
        (index for index, message in enumerate(messages) if message.role == "user"),
        0,
    )
    messages.insert(insert_at, reminder)
    estimated = estimate_request_chars(
        tuple(messages), tools, reserved_chars=reserved_chars
    )
    return ContextWindow(
        messages=tuple(messages),
        estimated_chars=estimated,
        truncated=window.truncated,
        omitted_blocks=window.omitted_blocks,
    )


class _UsageAccumulator:
    """Add provider token counts without inventing missing measurements."""

    def __init__(self) -> None:
        self._seen_usage = False
        self._prompt = 0
        self._completion = 0
        self._total = 0
        self._cached = 0
        self._reasoning = 0
        self._prompt_complete = True
        self._completion_complete = True
        self._total_complete = True
        self._cached_complete = True
        self._reasoning_complete = True

    def add(self, usage: Usage | None) -> None:
        if usage is None:
            # A later measured response must not make a mixed run appear fully
            # measured.  Keep returning None when every provider response omitted
            # usage; if at least one provided counts, expose a Usage object whose
            # fields are None to signal that totals are incomplete.
            self._prompt_complete = False
            self._completion_complete = False
            self._total_complete = False
            self._cached_complete = False
            self._reasoning_complete = False
            return
        self._seen_usage = True
        self._prompt, self._prompt_complete = self._add_field(
            self._prompt, self._prompt_complete, usage.prompt_tokens
        )
        self._completion, self._completion_complete = self._add_field(
            self._completion,
            self._completion_complete,
            usage.completion_tokens,
        )
        self._total, self._total_complete = self._add_field(
            self._total, self._total_complete, usage.total_tokens
        )
        self._cached, self._cached_complete = self._add_field(
            self._cached, self._cached_complete, usage.cached_tokens
        )
        self._reasoning, self._reasoning_complete = self._add_field(
            self._reasoning, self._reasoning_complete, usage.reasoning_tokens
        )

    @staticmethod
    def _add_field(current: int, complete: bool, value: int | None) -> tuple[int, bool]:
        if value is None:
            return current, False
        return current + value, complete

    def value(self) -> Usage | None:
        if not self._seen_usage:
            return None
        return Usage(
            prompt_tokens=self._prompt if self._prompt_complete else None,
            completion_tokens=(self._completion if self._completion_complete else None),
            total_tokens=self._total if self._total_complete else None,
            cached_tokens=self._cached if self._cached_complete else None,
            reasoning_tokens=(self._reasoning if self._reasoning_complete else None),
        )
