"""Compact Rich dashboard for interactive coding-agent runs.

The TUI is a presentation-only event sink. It consumes the same redacted
runtime events as the plain console sink and never participates in control-flow
decisions, tool execution, or verification.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from rich.console import Console, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from coding_agent.events import REDACTED, redact


@dataclass(slots=True)
class _ToolActivity:
    call_id: str
    name: str
    status: str = "running"
    detail: str = ""


class RichEventSink:
    """Render a bounded, live dashboard from ordinary runtime events."""

    def __init__(
        self,
        *,
        task: str,
        workspace: Path,
        model: str,
        max_model_turns: int,
        max_tool_calls: int,
        max_wall_time_seconds: float,
        secrets: Iterable[str] = (),
        console: Console | None = None,
        clock: Callable[[], float] = time.monotonic,
        auto_refresh: bool = True,
        show_header: bool = True,
    ) -> None:
        self._secrets = tuple(secret for secret in secrets if secret)
        self._task = self._safe_text(task)
        self._workspace = self._safe_text(str(workspace))
        self._model = self._safe_text(model)
        self._max_model_turns = max_model_turns
        self._max_tool_calls = max_tool_calls
        self._max_wall_time_seconds = max_wall_time_seconds
        self._show_header = show_header
        self._console = console or Console(stderr=True, highlight=False)
        self._clock = clock
        self._started_at = clock()
        self._lock = threading.RLock()

        self._stage = "Ready"
        self._stage_style = "cyan"
        self._phase: str | None = None
        self._reason = ""
        self._model_turns = 0
        self._model_requests = 0
        self._tool_calls = 0
        self._final_elapsed: float | None = None
        self._final_text = ""
        self._verification_state = "Waiting"
        self._verification_style = "dim"
        self._terminal = False
        self._activities: deque[_ToolActivity] = deque(maxlen=5)
        self._activities_by_id: dict[str, _ToolActivity] = {}

        self._live = Live(
            console=self._console,
            get_renderable=self._render,
            auto_refresh=auto_refresh,
            refresh_per_second=8,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="ellipsis",
        )
        self._live_started = False
        self._closed = False

    def emit(self, event_type: str, **data: object) -> None:
        """Update dashboard state using one already-structured event."""

        if self._closed:
            return
        safe = redact(data, secrets=self._secrets)
        if not isinstance(safe, Mapping):
            safe = {}
        with self._lock:
            self._apply_event(event_type, safe)
        self._ensure_started()
        self._live.refresh()

    def verification_started(self, check_count: int) -> None:
        """Show the independent verifier as active after the agent stops."""

        if self._closed:
            return
        with self._lock:
            noun = "check" if check_count == 1 else "checks"
            self._verification_state = f"Running {check_count} {noun}"
            self._verification_style = "yellow"
            self._stage = "Verifying workspace"
            self._stage_style = "yellow"
            self._terminal = False
        self._ensure_started()
        self._live.refresh()

    def finish(self, result: object, verification: object | None = None) -> None:
        """Render the terminal runtime and verification outcome, then stop."""

        if self._closed:
            return
        with self._lock:
            self._phase = self._safe_text(_phase_text(result))
            self._reason = self._safe_text(str(getattr(result, "reason", "")))
            self._model_turns = _safe_int(
                getattr(result, "model_turns", self._model_turns), self._model_turns
            )
            self._model_requests = _safe_int(
                getattr(result, "model_requests", self._model_requests),
                self._model_requests,
            )
            self._tool_calls = _safe_int(
                getattr(result, "tool_calls", self._tool_calls), self._tool_calls
            )
            self._final_elapsed = _safe_float(getattr(result, "elapsed_seconds", None))
            final_text = getattr(result, "final_text", "")
            if isinstance(final_text, str):
                self._final_text = self._safe_text(final_text)

            phase_is_complete = self._phase.lower() == "completed"
            if verification is None:
                self._verification_state = "Not requested"
                self._verification_style = "dim"
                overall_ok = phase_is_complete
            else:
                checks = tuple(getattr(verification, "checks", ()) or ())
                passed_count = sum(
                    bool(getattr(item, "passed", False)) for item in checks
                )
                passed = bool(getattr(verification, "passed", False))
                self._verification_state = (
                    f"Passed {passed_count}/{len(checks)} checks"
                    if passed
                    else f"Failed {passed_count}/{len(checks)} checks"
                )
                self._verification_style = "green" if passed else "red"
                overall_ok = phase_is_complete and passed

            self._stage = (
                "Completed" if overall_ok else self._phase.replace("_", " ").title()
            )
            self._stage_style = "green" if overall_ok else "red"
            self._terminal = True
        self.close()

    def abort(self, message: str, *, cancelled: bool = False) -> None:
        """Leave the terminal in a readable state after a CLI boundary error."""

        if self._closed:
            return
        with self._lock:
            self._stage = "Cancelled" if cancelled else "Failed"
            self._stage_style = "yellow" if cancelled else "red"
            self._reason = self._safe_text(message)
            self._terminal = True
        self.close()

    def close(self) -> None:
        """Stop Rich's refresh thread while retaining the last dashboard."""

        if self._closed:
            return
        self._ensure_started()
        self._live.refresh()
        self._live.stop()
        self._closed = True

    def _ensure_started(self) -> None:
        if not self._live_started:
            self._live.start(refresh=True)
            self._live_started = True

    def _apply_event(self, event_type: str, data: Mapping[str, object]) -> None:
        if event_type == "reasoning.probe":
            self._stage = "Checking model capability"
            self._stage_style = "cyan"
        elif event_type == "run.started":
            self._started_at = self._clock()
            self._stage = "Starting agent"
            self._stage_style = "cyan"
        elif event_type == "model.requested":
            self._model_requests = _safe_int(
                data.get("model_request"), self._model_requests
            )
            self._stage = "Thinking"
            self._stage_style = "cyan"
        elif event_type == "model.completed":
            self._model_turns = _safe_int(data.get("model_turn"), self._model_turns)
            tool_count = _safe_int(data.get("tool_calls"), 0)
            self._stage = "Planning tool work" if tool_count else "Preparing answer"
            self._stage_style = "cyan"
        elif event_type == "tool.requested":
            call_id = self._safe_text(str(data.get("call_id", "unknown")))
            name = self._safe_text(str(data.get("tool", "unknown")))
            activity = _ToolActivity(
                call_id=call_id,
                name=name,
                detail=_batch_detail(data),
            )
            self._activities.append(activity)
            self._activities_by_id[call_id] = activity
            self._tool_calls += 1
            self._stage = f"Running {name}"
            self._stage_style = "yellow"
        elif event_type == "tool.completed":
            call_id = self._safe_text(str(data.get("call_id", "unknown")))
            activity = self._activities_by_id.pop(call_id, None)
            if activity is not None:
                ok = bool(data.get("ok", False))
                metadata = data.get("metadata")
                safe_metadata = metadata if isinstance(metadata, Mapping) else {}
                exit_code = safe_metadata.get("exit_code")
                command_warning = (
                    activity.name == "run_command"
                    and isinstance(exit_code, int)
                    and not isinstance(exit_code, bool)
                    and exit_code != 0
                )
                activity.status = (
                    "error" if not ok else "warning" if command_warning else "ok"
                )
                activity.detail = _tool_detail(
                    activity.name,
                    safe_metadata,
                    error_code=data.get("error_code"),
                )
            self._stage = "Reviewing tool result"
            self._stage_style = "cyan"
        elif event_type in {"model.retry", "model.protocol_retry"}:
            self._stage = "Retrying model request"
            self._stage_style = "yellow"
        elif event_type.startswith("context."):
            self._stage = "Trimming context"
            self._stage_style = "yellow"
        elif event_type.startswith("efficiency."):
            self._stage = "Converging on result"
            self._stage_style = "yellow"
        elif event_type == "run.error":
            self._stage = "Runtime error"
            self._stage_style = "red"
        elif event_type == "run.terminal":
            self._phase = self._safe_text(str(data.get("phase", "unknown")))
            self._reason = self._safe_text(str(data.get("reason", "")))
            self._stage = self._phase.replace("_", " ").title()
            self._stage_style = "green" if self._phase == "completed" else "red"
        elif event_type == "run.ended":
            self._phase = self._safe_text(str(data.get("phase", "unknown")))
            self._reason = self._safe_text(str(data.get("reason", "")))
            self._model_turns = _safe_int(data.get("model_turns"), self._model_turns)
            self._model_requests = _safe_int(
                data.get("model_requests"), self._model_requests
            )
            self._tool_calls = _safe_int(data.get("tool_calls"), self._tool_calls)
            self._final_elapsed = _safe_float(data.get("elapsed_seconds"))

    def _render(self) -> RenderableType:
        with self._lock:
            stage = self._stage
            stage_style = self._stage_style
            terminal = self._terminal
            activities = tuple(
                (item.status, item.name, item.detail) for item in self._activities
            )
            elapsed = (
                self._final_elapsed
                if self._final_elapsed is not None
                else max(0.0, self._clock() - self._started_at)
            )

            available_width = max(20, self._console.size.width - 2)
            content_width = min(108, available_width)
            shell = Table(
                width=content_width,
                box=None,
                padding=0,
                collapse_padding=True,
                show_header=False,
                show_footer=False,
                show_edge=False,
                pad_edge=False,
            )
            shell.add_column()

            if self._show_header:
                header = Table.grid(expand=True)
                header.add_column(ratio=1)
                header.add_column(ratio=1, justify="right")
                brand = Text("Course Coding Agent", style="bold cyan")
                brand.append(f"  {_clip(self._model, 36)}", style="dim")
                header.add_row(brand, Text(_clip(self._workspace, 48), style="dim"))
                shell.add_row(header)
                shell.add_row(Rule(style="grey35"))
                shell.add_row("")

            user = Table.grid(expand=True, padding=(0, 1))
            user.add_column(width=3, justify="center")
            user.add_column(ratio=1)
            user_text = Text("you  ", style="bold cyan")
            user_text.append(_clip(self._task, 600))
            user.add_row(Text(">", style="bold cyan"), user_text)
            shell.add_row(user)
            shell.add_row("")

            activity_table = Table.grid(expand=True, padding=(0, 1))
            activity_table.add_column(width=3, justify="center")
            activity_table.add_column(width=20, no_wrap=True, overflow="ellipsis")
            activity_table.add_column(ratio=1, overflow="ellipsis")
            for status, name, detail in activities:
                marker, style = _activity_marker(status)
                activity_table.add_row(
                    Text(marker, style=style),
                    Text(name, style="bold"),
                    Text(_clip(detail, 80), style="dim"),
                )
            stage_renderable: RenderableType
            if terminal:
                stage_renderable = Text(stage, style=f"bold {stage_style}")
            else:
                stage_renderable = Spinner(
                    "dots", text=Text(stage, style=f"bold {stage_style}")
                )
            stage_row = Table.grid(expand=True, padding=(0, 1))
            stage_row.add_column(width=3, justify="center")
            stage_row.add_column(ratio=1)
            stage_marker = "●" if terminal else "·"
            stage_row.add_row(Text(stage_marker, style=stage_style), stage_renderable)
            shell.add_row(stage_row)
            if activities:
                shell.add_row(activity_table)
            shell.add_row("")

            assistant = Table.grid(expand=True, padding=(0, 1))
            assistant.add_column(width=3, justify="center")
            # Final responses must wrap inside the bounded dashboard column.
            # Rich's default ellipsis overflow turns a long Markdown paragraph
            # into one clipped line, which hides the actual conclusion.
            assistant.add_column(ratio=1, overflow="fold")
            if self._final_text:
                assistant.add_row(
                    Text("●", style="green"),
                    Text("assistant", style="bold green"),
                )
                assistant.add_row("", Markdown(_clip(self._final_text, 1200)))
            elif self._reason:
                assistant.add_row(
                    Text("●", style=stage_style),
                    Text("assistant", style=f"bold {stage_style}"),
                )
                assistant.add_row("", Text(_clip(self._reason, 600)))
            else:
                assistant.add_row(
                    Text("○", style="dim"),
                    Text("assistant", style="bold dim"),
                )
                assistant.add_row("", Text("Working...", style="dim"))
            shell.add_row(assistant)
            shell.add_row("")

            footer = Table.grid(expand=True, padding=(0, 1))
            footer.style = "white on grey11"
            footer.add_column(ratio=1)
            footer.add_column(ratio=2, justify="right")
            status = Text(stage.upper(), style=f"bold {stage_style}")
            metrics = Text(
                f"turns {self._model_turns}/{self._max_model_turns}  "
                f"tools {self._tool_calls}/{self._max_tool_calls}  "
                f"{elapsed:.1f}s",
                style="white",
            )
            if self._verification_state != "Waiting":
                metrics.append("  verify ", style="dim")
                metrics.append(self._verification_state, style=self._verification_style)
            footer.add_row(status, metrics)
            shell.add_row(footer)
            return shell

    def _safe_text(self, value: str) -> str:
        safe = redact(value, secrets=self._secrets)
        return safe if isinstance(safe, str) else REDACTED


def prompt_for_task(
    *,
    console: Console | None = None,
    stream: TextIO | None = None,
    show_header: bool = True,
) -> str:
    """Read one non-empty task from a compact Pi-style terminal prompt."""

    console = console or Console(stderr=True, highlight=False)
    content_width = min(108, max(20, console.size.width - 2))
    if show_header:
        header = Table(
            width=content_width,
            box=None,
            padding=0,
            collapse_padding=True,
            show_header=False,
            show_footer=False,
            show_edge=False,
            pad_edge=False,
        )
        header.add_column()
        title = Text("Course Coding Agent", style="bold cyan")
        title.append("  interactive", style="dim")
        header.add_row(title)
        header.add_row(Rule(style="grey35"))
        header.add_row(Text("Persistent session  ·  /exit to quit", style="dim"))
        console.print(header)
        console.print()

    prompt = Text("> ", style="bold cyan")
    prompt.append("you  ", style="bold cyan")
    while True:
        raw_task = console.input(prompt, stream=stream)
        if stream is not None and raw_task == "":
            raise EOFError
        task = raw_task.strip()
        if task:
            console.print()
            return task


def _activity_marker(status: str) -> tuple[str, str]:
    if status == "ok":
        return "✓", "bold green"
    if status == "error":
        return "×", "bold red"
    if status == "warning":
        return "!", "bold yellow"
    return "·", "bold yellow"


def _batch_detail(data: Mapping[str, object]) -> str:
    position = _safe_int(data.get("position"), 0) + 1
    batch_size = _safe_int(data.get("batch_size"), 1)
    return f"batch {position}/{batch_size}"


def _tool_detail(
    tool: str,
    metadata: Mapping[str, object],
    *,
    error_code: object,
) -> str:
    if error_code:
        return str(error_code)
    path = metadata.get("path")
    if tool == "run_command":
        exit_code = metadata.get("exit_code")
        duration = _safe_float(metadata.get("duration_seconds"))
        detail = f"exit {exit_code}" if exit_code is not None else "finished"
        return f"{detail} · {duration:.2f}s" if duration is not None else detail
    if path is not None:
        if "replacements" in metadata:
            count = metadata["replacements"]
            noun = "replacement" if count == 1 else "replacements"
            return f"{path} · {count} exact {noun}"
        if "returned_line_count" in metadata:
            return f"{path} · {metadata['returned_line_count']} lines"
        if "match_count" in metadata:
            return f"{path} · {metadata['match_count']} matches"
        if "bytes_written" in metadata:
            return f"{path} · {metadata['bytes_written']} bytes"
        return str(path)
    if "entry_count" in metadata:
        return f"{metadata['entry_count']} entries"
    return "completed"


def _safe_int(value: object, default: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _phase_text(result: object) -> str:
    phase = getattr(result, "phase", "unknown")
    return str(getattr(phase, "value", phase))


def _single_line(value: str, limit: int) -> str:
    return _clip(" ".join(value.split()), limit)


def _clip(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."
