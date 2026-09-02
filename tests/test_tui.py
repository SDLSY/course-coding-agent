"""Focused tests for the optional Rich terminal dashboard."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from coding_agent.tui import RichEventSink, prompt_for_task
from coding_agent.types import RunPhase


def test_interactive_prompt_ignores_blank_input() -> None:
    output = StringIO()
    task = prompt_for_task(
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=90,
        ),
        stream=StringIO("\nfix the failing tests\n"),
    )

    assert task == "fix the failing tests"
    assert "Course Coding Agent" in output.getvalue()
    assert output.getvalue().count("you") == 2


def test_interactive_prompt_reports_stream_eof() -> None:
    with pytest.raises(EOFError):
        prompt_for_task(
            console=Console(
                file=StringIO(),
                force_terminal=False,
                color_system=None,
                width=80,
            ),
            stream=StringIO(""),
        )


def test_follow_up_prompt_does_not_repeat_header() -> None:
    output = StringIO()

    task = prompt_for_task(
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=80,
        ),
        stream=StringIO("what changed?\n"),
        show_header=False,
    )

    assert task == "what changed?"
    assert "Course Coding Agent" not in output.getvalue()
    assert "you" in output.getvalue()


def test_rich_sink_renders_progress_result_and_redacts_secrets(
    tmp_path: Path,
) -> None:
    secret = "synthetic-tui-credential-for-tests-only"
    output = StringIO()
    sink = RichEventSink(
        task=f"repair the fixture without exposing {secret}",
        workspace=tmp_path,
        model="offline-model",
        max_model_turns=10,
        max_tool_calls=30,
        max_wall_time_seconds=120,
        secrets=(secret,),
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=100,
        ),
        clock=lambda: 10.0,
        auto_refresh=False,
    )

    sink.emit(
        "run.started",
        max_model_turns=10,
        max_tool_calls=30,
        max_wall_time_seconds=120,
    )
    sink.emit("model.requested", model_request=1)
    sink.emit(
        "tool.requested",
        call_id="call-1",
        tool="read_file",
        position=0,
        batch_size=1,
    )
    sink.emit(
        "tool.completed",
        call_id="call-1",
        tool="read_file",
        ok=True,
        error_code=None,
        metadata={"path": "hello.py", "returned_line_count": 8},
    )
    sink.emit(
        "tool.requested",
        call_id="call-2",
        tool="replace_in_file",
        position=0,
        batch_size=1,
    )
    sink.emit(
        "tool.completed",
        call_id="call-2",
        tool="replace_in_file",
        ok=True,
        error_code=None,
        metadata={"path": "pricing.py", "replacements": 1, "bytes_written": 180},
    )
    sink.verification_started(1)
    sink.finish(
        SimpleNamespace(
            phase=RunPhase.COMPLETED,
            reason=f"finished without leaking {secret}",
            final_text=f"Fixed the project; hidden value: {secret}",
            model_turns=2,
            model_requests=2,
            tool_calls=1,
            elapsed_seconds=1.25,
        ),
        SimpleNamespace(
            passed=True,
            checks=(SimpleNamespace(passed=True),),
        ),
    )

    rendered = output.getvalue()
    assert "Course Coding Agent" in rendered
    assert "read_file" in rendered
    assert "hello.py · 8 lines" in rendered
    assert "pricing.py · 1 exact replacement" in rendered
    assert "Passed 1/1 checks" in rendered
    assert "Completed" in rendered
    assert "[REDACTED]" in rendered
    assert secret not in rendered


def test_rich_sink_shows_failed_tools_and_can_close_twice(tmp_path: Path) -> None:
    output = StringIO()
    sink = RichEventSink(
        task="run a failing command",
        workspace=tmp_path,
        model="offline-model",
        max_model_turns=3,
        max_tool_calls=5,
        max_wall_time_seconds=30,
        console=Console(
            file=output,
            force_terminal=False,
            color_system=None,
            width=90,
        ),
        clock=lambda: 5.0,
        auto_refresh=False,
    )

    sink.emit(
        "tool.requested",
        call_id="call-2",
        tool="run_command",
        position=0,
        batch_size=1,
    )
    sink.emit(
        "tool.completed",
        call_id="call-2",
        tool="run_command",
        ok=False,
        error_code="command_failed",
        metadata={"exit_code": 2},
    )
    sink.emit(
        "tool.requested",
        call_id="call-3",
        tool="run_command",
        position=0,
        batch_size=1,
    )
    sink.emit(
        "tool.completed",
        call_id="call-3",
        tool="run_command",
        ok=True,
        error_code=None,
        metadata={"exit_code": 7, "duration_seconds": 0.01},
    )
    sink.abort("demonstration failed")
    sink.close()

    rendered = output.getvalue()
    assert "×" in rendered
    assert "!" in rendered
    assert "command_failed" in rendered
    assert "exit 7 · 0.01s" in rendered
    assert "demonstration failed" in rendered
