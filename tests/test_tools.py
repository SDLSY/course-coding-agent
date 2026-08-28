"""Integration tests for model-facing filesystem and command boundaries.

Every test uses pytest's temporary directory and synthetic environment values.
No test reads a developer credential, contacts the network, or executes a
command outside its disposable workspace.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from coding_agent.tools.base import Tool, ToolOutput
from coding_agent.tools.filesystem import FileSystemTools, build_filesystem_tools
from coding_agent.tools.registry import ToolRegistry, build_default_registry
from coding_agent.tools.shell import ShellTools
from coding_agent.types import ToolCall

SYNTHETIC_SECRET = "synthetic-secret-for-tests-only"


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    return build_default_registry(tmp_path)


def invoke(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, object],
    *,
    call_id: str = "test_call",
):
    """Keep assertions focused while still exercising registry dispatch."""

    return registry.execute(call_id, name, arguments)


def test_default_registry_exposes_six_uniform_function_schemas(tmp_path: Path) -> None:
    registry = build_default_registry(tmp_path)

    assert registry.names == (
        "list_files",
        "search_text",
        "read_file",
        "write_file",
        "replace_in_file",
        "run_command",
    )
    schemas = registry.model_schemas()
    assert [schema["type"] for schema in schemas] == ["function"] * 6
    assert [schema["function"]["name"] for schema in schemas] == list(registry.names)
    assert all(
        schema["function"]["parameters"]["type"] == "object" for schema in schemas
    )


def test_registry_rejects_duplicate_names() -> None:
    tool = Tool(
        name="sample",
        description="Return a deterministic sample.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=lambda arguments: ToolOutput("ok"),
    )
    registry = ToolRegistry([tool])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_registry_wraps_a_buggy_handler_without_orphaning_the_call() -> None:
    # Concrete built-ins always return ToolOutput, but the registry is the trust
    # boundary for future plugin-like tools too.  A handler contract bug must be
    # observable as one paired failure instead of escaping as AttributeError.
    broken = Tool(
        name="broken",
        description="Synthetic invalid handler used only by this test.",
        parameters={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        handler=lambda arguments: "not-a-tool-output",  # type: ignore[arg-type]
    )

    result = ToolRegistry([broken]).execute("still_paired", "broken", {})

    assert not result.ok
    assert result.call_id == "still_paired"
    assert result.error_code == "tool_execution_error"
    assert result.metadata["exception_type"] == "TypeError"


def test_registry_caps_a_tool_timeout_without_mutating_model_arguments() -> None:
    observed: list[float] = []

    def handler(arguments: dict[str, object]) -> ToolOutput:
        observed.append(float(arguments["timeout_seconds"]))
        return ToolOutput("ok")

    tool = Tool(
        name="bounded_operation",
        description="Observe an internally capped timeout.",
        parameters={
            "type": "object",
            "properties": {
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                }
            },
            "required": ["timeout_seconds"],
            "additionalProperties": False,
        },
        handler=handler,
        timeout_argument="timeout_seconds",
    )
    registry = ToolRegistry([tool])
    model_arguments = {"timeout_seconds": 30}

    result = registry.execute(
        "bounded",
        "bounded_operation",
        model_arguments,
        timeout_seconds=0.25,
    )

    assert result.ok
    assert observed == [0.25]
    assert model_arguments == {"timeout_seconds": 30}


def test_execute_call_turns_invalid_json_into_one_paired_error(
    registry: ToolRegistry,
) -> None:
    result = registry.execute_call(ToolCall("raw_call_id", "read_file", '{"path":'))

    assert result.call_id == "raw_call_id"
    assert result.name == "read_file"
    assert not result.ok
    assert result.error_code == "invalid_json"
    assert result.metadata["error_kind"] == "request"
    # Serialising the result must preserve the provider pairing ID even on an
    # invalid request; otherwise the next API request would be malformed.
    assert result.to_message().tool_call_id == "raw_call_id"


@pytest.mark.parametrize(
    "name, arguments, expected_code",
    [
        ("does_not_exist", {}, "unknown_tool"),
        ("read_file", {}, "invalid_arguments"),
        ("read_file", {"path": 7}, "invalid_arguments"),
        ("read_file", {"path": "x", "surprise": True}, "invalid_arguments"),
        # Python considers bool an int, while JSON Schema does not.  This is a
        # common hand-written-validator bug worth locking down explicitly.
        (
            "read_file",
            {"path": "x", "start_line": True},
            "invalid_arguments",
        ),
    ],
)
def test_registry_returns_structured_request_errors(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, object],
    expected_code: str,
) -> None:
    result = invoke(registry, name, arguments, call_id="paired")

    assert not result.ok
    assert result.call_id == "paired"
    assert result.name == name
    assert result.error_code == expected_code
    assert result.metadata["error_kind"] == "request"


@pytest.mark.parametrize(
    "unsafe_path, expected_code",
    [
        ("../outside.txt", "path_traversal"),
        ("nested/../../outside.txt", "path_traversal"),
        ("/etc/passwd", "absolute_path_not_allowed"),
    ],
)
def test_file_tools_reject_absolute_and_parent_traversal_paths(
    registry: ToolRegistry,
    unsafe_path: str,
    expected_code: str,
) -> None:
    result = invoke(registry, "read_file", {"path": unsafe_path})

    assert not result.ok
    assert result.error_code == expected_code


def test_file_tools_reject_existing_symlink_escape_for_read_and_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("must remain unchanged", encoding="utf-8")
    (tmp_path / "escape.txt").symlink_to(outside)
    registry = build_default_registry(tmp_path)

    read_result = invoke(registry, "read_file", {"path": "escape.txt"})
    write_result = invoke(
        registry,
        "write_file",
        {"path": "escape.txt", "content": "changed"},
    )

    assert not read_result.ok
    assert read_result.error_code == "path_outside_workspace"
    assert not write_result.ok
    assert write_result.error_code == "path_outside_workspace"
    assert outside.read_text(encoding="utf-8") == "must remain unchanged"


def test_list_files_is_deterministic_bounded_and_skips_unsafe_symlink(
    tmp_path: Path,
) -> None:
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "inside.py").write_text("pass\n", encoding="utf-8")
    (tmp_path / "escape").symlink_to(tmp_path.parent)
    tools = FileSystemTools(tmp_path, list_limit=2)
    registry = ToolRegistry(tools.definitions())

    result = invoke(registry, "list_files", {"path": "."})

    assert result.ok
    assert result.metadata["truncated"] is True
    assert result.metadata["entry_count"] == 2
    # Directory markers make the compact text result unambiguous to a model.
    assert result.content.splitlines()[:2] == ["a/", "b.txt"]
    # The traversal stops when its display budget is filled; callers must not
    # assume skipped counts cover unvisited entries after truncation.
    assert "escape" not in result.content


def test_search_text_reports_literal_matches_and_skips_non_utf8(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text(
        "needle = 1\n# needle again\n", encoding="utf-8"
    )
    (tmp_path / "src" / "binary.dat").write_bytes(b"\xff\xfe\x00needle")
    registry = build_default_registry(tmp_path)

    result = invoke(
        registry,
        "search_text",
        {"query": "needle", "path": "src"},
    )

    assert result.ok
    assert "src/one.py:1: needle = 1" in result.content
    assert "src/one.py:2: # needle again" in result.content
    assert result.metadata["match_count"] == 2
    assert result.metadata["skipped_non_utf8_files"] == 1


def test_search_text_marks_truncation_only_when_a_match_is_actually_omitted(
    tmp_path: Path,
) -> None:
    target = tmp_path / "matches.txt"
    target.write_text("hit one\nhit two\n", encoding="utf-8")
    tools = FileSystemTools(tmp_path, search_limit=2)
    registry = ToolRegistry(tools.definitions())

    exact = invoke(registry, "search_text", {"query": "hit"})
    assert exact.metadata["match_count"] == 2
    assert exact.metadata["truncated"] is False

    target.write_text("hit one\nhit two\nhit three\n", encoding="utf-8")
    overflowing = invoke(registry, "search_text", {"query": "hit"})
    assert overflowing.metadata["match_count"] == 2
    assert overflowing.metadata["truncated"] is True


def test_read_file_uses_inclusive_ranges_and_explicit_truncation_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "many.txt").write_text(
        "".join(f"line-{number}\n" for number in range(1, 9)),
        encoding="utf-8",
    )
    tools = FileSystemTools(tmp_path, read_line_limit=2)
    registry = ToolRegistry(tools.definitions())

    result = invoke(
        registry,
        "read_file",
        {"path": "many.txt", "start_line": 3, "end_line": 7},
    )

    assert result.ok
    assert result.content == "line-3\nline-4\n"
    assert result.metadata["total_lines"] == 8
    assert result.metadata["requested_start_line"] == 3
    assert result.metadata["requested_end_line"] == 7
    assert result.metadata["returned_start_line"] == 3
    assert result.metadata["returned_end_line"] == 4
    assert result.metadata["truncated"] is True
    assert result.metadata["line_limit_reached"] is True


def test_read_file_rejects_non_utf8_and_reversed_range(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")
    (tmp_path / "text.txt").write_text("hello\n", encoding="utf-8")
    registry = build_default_registry(tmp_path)

    binary = invoke(registry, "read_file", {"path": "binary.bin"})
    reversed_range = invoke(
        registry,
        "read_file",
        {"path": "text.txt", "start_line": 5, "end_line": 2},
    )

    assert not binary.ok
    assert binary.error_code == "not_utf8_text"
    assert not reversed_range.ok
    assert reversed_range.error_code == "invalid_line_range"


def test_write_file_creates_parents_atomically_and_preserves_existing_mode(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(build_filesystem_tools(tmp_path))

    created = invoke(
        registry,
        "write_file",
        {"path": "new/deep/file.txt", "content": "first\n"},
    )
    target = tmp_path / "new" / "deep" / "file.txt"
    target.chmod(0o754)
    replaced = invoke(
        registry,
        "write_file",
        {"path": "new/deep/file.txt", "content": "second\n"},
    )

    assert created.ok and created.metadata["created"] is True
    assert replaced.ok and replaced.metadata["created"] is False
    assert target.read_text(encoding="utf-8") == "second\n"
    assert target.stat().st_mode & 0o777 == 0o754
    assert not list(target.parent.glob(".file.txt.*.tmp"))


def test_failed_atomic_replace_leaves_old_target_and_removes_temporary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "stable.txt"
    target.write_text("old", encoding="utf-8")
    registry = ToolRegistry(build_filesystem_tools(tmp_path))

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr("coding_agent.tools.filesystem.os.replace", fail_replace)
    result = invoke(
        registry,
        "write_file",
        {"path": "stable.txt", "content": "new"},
    )

    assert not result.ok
    assert result.error_code == "filesystem_error"
    assert result.metadata["error_kind"] == "execution"
    assert target.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".stable.txt.*.tmp"))


def test_replace_in_file_requires_one_exact_match_without_partial_write(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\nvalue = 1\n", encoding="utf-8")
    registry = build_default_registry(tmp_path)

    ambiguous = invoke(
        registry,
        "replace_in_file",
        {"path": "sample.py", "old": "value = 1", "new": "value = 2"},
    )
    assert not ambiguous.ok
    assert ambiguous.error_code == "non_unique_match"
    assert target.read_text(encoding="utf-8") == "value = 1\nvalue = 1\n"

    unique = invoke(
        registry,
        "replace_in_file",
        {
            "path": "sample.py",
            "old": "value = 1\nvalue = 1",
            "new": "value = 2\nvalue = 3",
        },
    )
    assert unique.ok
    assert unique.metadata["replacements"] == 1
    assert target.read_text(encoding="utf-8") == "value = 2\nvalue = 3\n"


def test_replace_in_file_preserves_unrelated_crlf_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "windows.txt"
    target.write_bytes(b"first\r\nold\r\nlast\r\n")
    registry = build_default_registry(tmp_path)

    result = invoke(
        registry,
        "replace_in_file",
        {"path": "windows.txt", "old": "old", "new": "new"},
    )

    assert result.ok
    assert target.read_bytes() == b"first\r\nnew\r\nlast\r\n"


def test_run_command_uses_workspace_and_nonzero_exit_is_normal_result(
    tmp_path: Path,
) -> None:
    registry = build_default_registry(tmp_path)
    result = invoke(
        registry,
        "run_command",
        {
            "command": "pwd; printf 'problem\\n' >&2; exit 7",
            "timeout_seconds": 5,
        },
    )

    assert result.ok
    assert result.error_code is None
    assert result.metadata["exit_code"] == 7
    assert result.metadata["timed_out"] is False
    assert str(tmp_path.resolve()) in result.content
    assert "problem" in result.content


def test_run_command_sanitizes_parent_credentials_but_keeps_benign_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", SYNTHETIC_SECRET)
    monkeypatch.setenv("SERVICE_TOKEN", SYNTHETIC_SECRET)
    monkeypatch.setenv("BENIGN_SETTING", "visible")
    registry = build_default_registry(tmp_path)

    result = invoke(
        registry,
        "run_command",
        {
            "command": (
                "printf '%s|%s|%s' \"$DEEPSEEK_API_KEY\" "
                '"$SERVICE_TOKEN" "$BENIGN_SETTING"'
            )
        },
    )

    assert result.ok
    assert "||visible" in result.content
    assert SYNTHETIC_SECRET not in result.content


def test_run_command_excludes_an_exact_custom_credential_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_name = "MODEL_GATEWAY_CREDENTIAL"
    monkeypatch.setenv(custom_name, SYNTHETIC_SECRET)
    registry = build_default_registry(
        tmp_path,
        excluded_command_environment_names=(custom_name,),
    )

    result = invoke(
        registry,
        "run_command",
        {
            # Test presence without printing the synthetic credential itself.
            # ``${name+x}`` expands to x only when the variable exists.
            "command": 'test -z "${MODEL_GATEWAY_CREDENTIAL+x}"',
        },
    )

    assert result.ok
    assert result.metadata["exit_code"] == 0
    assert SYNTHETIC_SECRET not in result.content


def test_explicit_command_environment_cannot_override_an_exclusion(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="both explicit and excluded"):
        ShellTools(
            tmp_path,
            extra_env={"CUSTOM_CREDENTIAL": SYNTHETIC_SECRET},
            excluded_env_names=("CUSTOM_CREDENTIAL",),
        )


def test_explicit_sensitive_command_environment_is_redacted_from_result(
    tmp_path: Path,
) -> None:
    shell = ShellTools(tmp_path, extra_env={"DEMO_TOKEN": SYNTHETIC_SECRET})
    registry = ToolRegistry([shell.definition()])

    result = invoke(
        registry,
        "run_command",
        {"command": "printf '%s' \"$DEMO_TOKEN\""},
    )

    assert result.ok
    assert SYNTHETIC_SECRET not in result.content
    assert "[REDACTED]" in result.content
    assert result.metadata["explicit_environment_names"] == ["DEMO_TOKEN"]
    assert SYNTHETIC_SECRET not in repr(result.metadata)


def test_run_command_bounds_stdout_and_stderr_independently(tmp_path: Path) -> None:
    shell = ShellTools(tmp_path, stream_byte_limit=200)
    registry = ToolRegistry([shell.definition()])

    result = invoke(
        registry,
        "run_command",
        {
            "command": (
                "python3 -c \"import sys; print('A'*1000); "
                "print('B'*900, file=sys.stderr)\""
            )
        },
    )

    assert result.ok
    assert result.metadata["stdout_original_bytes"] == 1001
    assert result.metadata["stderr_original_bytes"] == 901
    assert result.metadata["stdout_returned_bytes"] == 200
    assert result.metadata["stderr_returned_bytes"] == 200
    assert result.metadata["stdout_truncated"] is True
    assert result.metadata["stderr_truncated"] is True
    assert "command stream truncated" in result.content


@pytest.mark.skipif(os.name != "posix", reason="process groups require POSIX")
def test_run_command_timeout_kills_the_shell_process_group(tmp_path: Path) -> None:
    registry = build_default_registry(tmp_path)
    result = invoke(
        registry,
        "run_command",
        {
            "command": "sleep 30 & child=$!; echo $child > child.pid; wait",
            "timeout_seconds": 1,
        },
    )

    assert result.ok
    assert result.metadata["timed_out"] is True
    assert result.metadata["exit_code"] != 0
    child_pid = int((tmp_path / "child.pid").read_text(encoding="ascii"))
    proc_stat = Path(f"/proc/{child_pid}/stat")
    if proc_stat.exists():
        # A killed child can remain briefly as a zombie until PID 1 reaps it;
        # state Z is not executing and therefore satisfies the timeout promise.
        state = proc_stat.read_text(encoding="ascii").split()[2]
        assert state == "Z"
