"""Workspace-confined text-file tools.

These helpers provide a useful boundary against accidental ``../`` and
symbolic-link escapes initiated by a model.  They are *not* an operating-system
sandbox: another process can race path validation, and ``run_command`` has the
host permissions of this Python process.  The project therefore documents a
trusted-workspace threat model rather than claiming strong isolation.
"""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .base import Tool, ToolExecutionError, ToolOutput, ToolRequestError

DEFAULT_LIST_LIMIT = 300
DEFAULT_SEARCH_LIMIT = 100
DEFAULT_READ_LINE_LIMIT = 400
DEFAULT_OUTPUT_CHARACTER_LIMIT = 32_000
DEFAULT_SEARCH_FILE_SIZE_LIMIT = 2 * 1024 * 1024
DEFAULT_REPLACE_FILE_SIZE_LIMIT = 8 * 1024 * 1024
_TRUNCATION_MARKER = (
    "\n... [tool output truncated; request a narrower path or line range] ...\n"
)


class WorkspacePathResolver:
    """Resolve model-supplied relative paths beneath one canonical workspace.

    ``Path.resolve`` is important here because a lexical prefix check alone is
    insufficient: ``workspace/link/file`` may lexically look safe while
    ``link`` points to ``/etc``.  Existing links are followed to their canonical
    destination before containment is checked.  For a new write target,
    ``strict=False`` still resolves every existing parent component.

    This is a check against ordinary path traversal, not a proof against a
    hostile local process changing a directory to a symlink between this check
    and the later ``open``/``replace`` call (a TOCTOU race).
    """

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        candidate = Path(workspace).expanduser()
        try:
            root = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError("workspace does not exist or cannot be resolved") from exc
        if not root.is_dir():
            raise ValueError("workspace must be a directory")
        self.root = root

    def resolve_existing(self, user_path: str) -> Path:
        raw = self._validate_relative_path(user_path)
        try:
            resolved = (self.root / raw).resolve(strict=True)
        except FileNotFoundError as exc:
            raise ToolRequestError(
                f"Path does not exist: {user_path!r}.",
                error_code="path_not_found",
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise ToolRequestError(
                f"Path cannot be resolved: {user_path!r}.",
                error_code="invalid_path",
            ) from exc
        self._require_contained(resolved)
        return resolved

    def resolve_for_write(self, user_path: str) -> Path:
        raw = self._validate_relative_path(user_path)
        try:
            resolved = (self.root / raw).resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ToolRequestError(
                f"Path cannot be resolved: {user_path!r}.",
                error_code="invalid_path",
            ) from exc
        self._require_contained(resolved)
        return resolved

    def relative_display(self, path: Path) -> str:
        """Return a stable POSIX-style path suitable for model output."""

        try:
            relative = path.relative_to(self.root)
        except ValueError:
            # This normally indicates an internal invariant violation.  Avoid
            # leaking an absolute host path if a caller nevertheless passes it.
            return "<outside-workspace>"
        text = relative.as_posix()
        return text if text != "." else "."

    @staticmethod
    def _validate_relative_path(user_path: str) -> Path:
        if not isinstance(user_path, str) or not user_path:
            raise ToolRequestError(
                "Path must be a non-empty relative string.",
                error_code="invalid_path",
            )
        if "\x00" in user_path:
            raise ToolRequestError(
                "Path must not contain a NUL byte.",
                error_code="invalid_path",
            )

        raw = Path(user_path)
        if raw.is_absolute():
            raise ToolRequestError(
                "Absolute paths are not allowed; use a workspace-relative path.",
                error_code="absolute_path_not_allowed",
            )
        # Even a path such as ``a/../b`` which normalises inside the workspace is
        # rejected.  A simple rule is easier to audit and gives the model no
        # reason to emit traversal syntax at all.
        if ".." in raw.parts:
            raise ToolRequestError(
                "Parent-directory ('..') path components are not allowed.",
                error_code="path_traversal",
            )
        return raw

    def _require_contained(self, resolved: Path) -> None:
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolRequestError(
                "Resolved path is outside the workspace.",
                error_code="path_outside_workspace",
            )


class FileSystemTools:
    """Implement the five deterministic filesystem operations in the MVP."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        list_limit: int = DEFAULT_LIST_LIMIT,
        search_limit: int = DEFAULT_SEARCH_LIMIT,
        read_line_limit: int = DEFAULT_READ_LINE_LIMIT,
        output_character_limit: int = DEFAULT_OUTPUT_CHARACTER_LIMIT,
    ) -> None:
        self.paths = WorkspacePathResolver(workspace)
        self.list_limit = _positive_limit("list_limit", list_limit)
        self.search_limit = _positive_limit("search_limit", search_limit)
        self.read_line_limit = _positive_limit("read_line_limit", read_line_limit)
        self.output_character_limit = _positive_limit(
            "output_character_limit", output_character_limit
        )
        if self.output_character_limit <= len(_TRUNCATION_MARKER):
            raise ValueError("output_character_limit must exceed the truncation marker")

    def definitions(self) -> tuple[Tool, ...]:
        """Build tool definitions bound to this workspace instance."""

        path_property = {
            "type": "string",
            "minLength": 1,
            "description": "Workspace-relative path. Use '.' for the root.",
        }
        return (
            Tool(
                name="list_files",
                description=(
                    "Recursively list files and directories below a workspace path. "
                    "The result is bounded and does not follow directory symlinks."
                ),
                parameters={
                    "type": "object",
                    "properties": {"path": path_property},
                    "additionalProperties": False,
                },
                handler=self.list_files,
            ),
            Tool(
                name="search_text",
                description=(
                    "Find literal text in UTF-8 workspace files and return paths, "
                    "line numbers, and bounded matching lines."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "path": path_property,
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=self.search_text,
            ),
            Tool(
                name="read_file",
                description=(
                    "Read a UTF-8 text file by inclusive 1-based line range. "
                    "Use repeated narrow ranges when the result is truncated."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": path_property,
                        "start_line": {"type": "integer", "minimum": 1},
                        "end_line": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=self.read_file,
            ),
            Tool(
                name="write_file",
                description=(
                    "Create or atomically replace one UTF-8 file inside the workspace."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": path_property,
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=self.write_file,
                modifies_workspace=True,
            ),
            Tool(
                name="replace_in_file",
                description=(
                    "Atomically replace exact text only when it occurs exactly once "
                    "in a UTF-8 workspace file."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "path": path_property,
                        "old": {"type": "string", "minLength": 1},
                        "new": {"type": "string"},
                    },
                    "required": ["path", "old", "new"],
                    "additionalProperties": False,
                },
                handler=self.replace_in_file,
                modifies_workspace=True,
            ),
        )

    def list_files(self, arguments: Mapping[str, Any]) -> ToolOutput:
        start = self.paths.resolve_existing(arguments.get("path", "."))
        if start.is_file():
            display = self.paths.relative_display(start)
            return ToolOutput(
                content=display,
                metadata={
                    "path": display,
                    "entry_count": 1,
                    "truncated": False,
                    "skipped_unsafe_symlinks": 0,
                },
            )
        if not start.is_dir():
            raise ToolRequestError(
                "list_files path must refer to a regular file or directory.",
                error_code="unsupported_file_type",
            )

        entries: list[str] = []
        skipped_unsafe = 0
        truncated = False
        try:
            for lexical_path, kind, safe in self._walk_entries(start):
                if not safe:
                    skipped_unsafe += 1
                    continue
                suffix = (
                    "/" if kind == "directory" else "@" if kind == "symlink" else ""
                )
                entries.append(f"{self.paths.relative_display(lexical_path)}{suffix}")
                if len(entries) > self.list_limit:
                    entries.pop()
                    truncated = True
                    break
        except OSError as exc:
            raise ToolExecutionError(
                "Could not enumerate the requested workspace path.",
                error_code="filesystem_error",
                metadata={"operation": "list"},
            ) from exc

        content = "\n".join(entries)
        if truncated:
            content += "\n... [file list truncated; list a narrower subdirectory] ..."
        return ToolOutput(
            content=content,
            metadata={
                "path": self.paths.relative_display(start),
                "entry_count": len(entries),
                "limit": self.list_limit,
                "truncated": truncated,
                "skipped_unsafe_symlinks": skipped_unsafe,
            },
        )

    def search_text(self, arguments: Mapping[str, Any]) -> ToolOutput:
        query = arguments["query"]
        start = self.paths.resolve_existing(arguments.get("path", "."))
        if not (start.is_file() or start.is_dir()):
            raise ToolRequestError(
                "search_text path must refer to a regular file or directory.",
                error_code="unsupported_file_type",
            )

        matches: list[str] = []
        scanned_files = 0
        skipped_binary = 0
        skipped_large = 0
        skipped_unsafe = 0
        truncated = False

        for lexical_path, resolved_path, safe in self._iter_files(start):
            if not safe:
                skipped_unsafe += 1
                continue
            try:
                if resolved_path.stat().st_size > DEFAULT_SEARCH_FILE_SIZE_LIMIT:
                    skipped_large += 1
                    continue
                with resolved_path.open(
                    "r", encoding="utf-8", errors="strict", newline=""
                ) as handle:
                    scanned_files += 1
                    for line_number, line in enumerate(handle, start=1):
                        if query not in line:
                            continue
                        # A single generated/minified line can be enormous.  A
                        # bounded snippet preserves the match context without
                        # allowing it to consume the whole model context.
                        snippet = line.rstrip("\r\n")
                        if len(snippet) > 500:
                            snippet = snippet[:497] + "..."
                        if len(matches) >= self.search_limit:
                            # Detect one additional match before declaring
                            # truncation.  Merely reaching the exact limit does
                            # not prove that any result was omitted.
                            truncated = True
                            break
                        matches.append(
                            f"{self.paths.relative_display(lexical_path)}:"
                            f"{line_number}: {snippet}"
                        )
            except UnicodeDecodeError:
                skipped_binary += 1
                continue
            except OSError as exc:
                raise ToolExecutionError(
                    "Could not search a workspace file.",
                    error_code="filesystem_error",
                    metadata={"operation": "search"},
                ) from exc
            if truncated:
                break

        content = "\n".join(matches)
        if truncated:
            content += (
                "\n... [search results truncated; use a narrower path or query] ..."
            )
        elif not matches:
            content = "No matches found."
        return ToolOutput(
            content=content,
            metadata={
                "path": self.paths.relative_display(start),
                "match_count": len(matches),
                "limit": self.search_limit,
                "truncated": truncated,
                "scanned_files": scanned_files,
                "skipped_non_utf8_files": skipped_binary,
                "skipped_large_files": skipped_large,
                "skipped_unsafe_symlinks": skipped_unsafe,
            },
        )

    def read_file(self, arguments: Mapping[str, Any]) -> ToolOutput:
        path = self.paths.resolve_existing(arguments["path"])
        if not path.is_file():
            raise ToolRequestError(
                "read_file path must refer to a regular file.",
                error_code="not_a_file",
            )

        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line")
        if end_line is not None and end_line < start_line:
            raise ToolRequestError(
                "end_line must be greater than or equal to start_line.",
                error_code="invalid_line_range",
            )

        selected: list[str] = []
        total_lines = 0
        # Even if the requested end is reached, continue counting.  Accurate
        # total/range metadata tells the model whether a follow-up read is
        # useful and makes truncation explicit rather than silent.
        try:
            with path.open(
                "r", encoding="utf-8", errors="strict", newline=""
            ) as handle:
                for line_number, line in enumerate(handle, start=1):
                    total_lines = line_number
                    inside_requested_end = end_line is None or line_number <= end_line
                    if (
                        line_number >= start_line
                        and inside_requested_end
                        and len(selected) < self.read_line_limit
                    ):
                        selected.append(line)
        except UnicodeDecodeError as exc:
            raise ToolRequestError(
                "File is not valid UTF-8 text and cannot be read as source code.",
                error_code="not_utf8_text",
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                "Could not read the requested file.",
                error_code="filesystem_error",
                metadata={"operation": "read"},
            ) from exc

        actual_start = start_line
        actual_end = start_line + len(selected) - 1 if selected else start_line - 1
        last_requested_available = (
            total_lines if end_line is None else min(total_lines, end_line)
        )
        range_truncated = actual_end < last_requested_available
        complete_selected_content = "".join(selected)
        content, characters_truncated = _truncate_text(
            complete_selected_content,
            self.output_character_limit,
        )
        truncated = range_truncated or characters_truncated
        try:
            original_bytes = path.stat().st_size
        except OSError as exc:
            raise ToolExecutionError(
                "Could not inspect the requested file.",
                error_code="filesystem_error",
                metadata={"operation": "stat"},
            ) from exc

        return ToolOutput(
            content=content,
            metadata={
                "path": self.paths.relative_display(path),
                "total_lines": total_lines,
                "original_bytes": original_bytes,
                "requested_start_line": start_line,
                "requested_end_line": end_line,
                "returned_start_line": actual_start,
                "returned_end_line": actual_end,
                "returned_line_count": len(selected),
                "selected_characters": len(complete_selected_content),
                "returned_characters": len(content),
                "truncated": truncated,
                "line_limit_reached": range_truncated,
                "character_limit_reached": characters_truncated,
            },
        )

    def write_file(self, arguments: Mapping[str, Any]) -> ToolOutput:
        target = self.paths.resolve_for_write(arguments["path"])
        content = arguments["content"]
        existed = target.exists()
        if existed and not target.is_file():
            raise ToolRequestError(
                "write_file target exists but is not a regular file.",
                error_code="not_a_file",
            )

        self._atomic_write_text(target, content)
        return ToolOutput(
            content=f"Wrote {len(content.encode('utf-8'))} bytes to {self.paths.relative_display(target)}.",
            metadata={
                "path": self.paths.relative_display(target),
                "bytes_written": len(content.encode("utf-8")),
                "created": not existed,
                "atomic_replace": True,
            },
        )

    def replace_in_file(self, arguments: Mapping[str, Any]) -> ToolOutput:
        target = self.paths.resolve_existing(arguments["path"])
        if not target.is_file():
            raise ToolRequestError(
                "replace_in_file path must refer to a regular file.",
                error_code="not_a_file",
            )

        old = arguments["old"]
        new = arguments["new"]
        try:
            size = target.stat().st_size
            if size > DEFAULT_REPLACE_FILE_SIZE_LIMIT:
                raise ToolRequestError(
                    "File is too large for whole-file exact replacement.",
                    error_code="file_too_large",
                    metadata={"size_bytes": size},
                )
            # ``Path.read_text`` uses universal-newline translation.  Opening
            # with newline="" preserves CRLF and CR endings so an exact text
            # replacement cannot silently rewrite every unrelated line ending.
            with target.open(
                "r", encoding="utf-8", errors="strict", newline=""
            ) as handle:
                content = handle.read()
        except UnicodeDecodeError as exc:
            raise ToolRequestError(
                "File is not valid UTF-8 text and cannot be edited.",
                error_code="not_utf8_text",
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(
                "Could not read the file before replacement.",
                error_code="filesystem_error",
                metadata={"operation": "read_before_replace"},
            ) from exc

        occurrences = content.count(old)
        if occurrences != 1:
            detail = "not found" if occurrences == 0 else f"found {occurrences} times"
            raise ToolRequestError(
                f"Exact old text must occur once; it was {detail}. No file was changed.",
                error_code="non_unique_match",
                metadata={"occurrences": occurrences},
            )

        updated = content.replace(old, new, 1)
        self._atomic_write_text(target, updated)
        return ToolOutput(
            content=f"Replaced one exact occurrence in {self.paths.relative_display(target)}.",
            metadata={
                "path": self.paths.relative_display(target),
                "replacements": 1,
                "bytes_written": len(updated.encode("utf-8")),
                "atomic_replace": True,
            },
        )

    def _atomic_write_text(self, target: Path, content: str) -> None:
        """Write and fsync a sibling temporary file, then atomically replace.

        Placing the temporary file in the target directory is intentional:
        ``os.replace`` is only guaranteed atomic on one filesystem.  A failed
        write before ``os.replace`` leaves the old target untouched, and the
        ``finally`` block removes the temporary file.  Directory creation is a
        separate side effect and may remain if the final write fails.
        """

        try:
            encoded = content.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ToolRequestError(
                "Content cannot be encoded as valid UTF-8.",
                error_code="invalid_text_content",
            ) from exc

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Re-resolve after mkdir so an already-present parent symlink is
            # checked again immediately before opening a sibling temporary.
            target = self.paths.resolve_for_write(self.paths.relative_display(target))
            if target.exists() and target.is_dir():
                raise ToolRequestError(
                    "The write target is a directory.",
                    error_code="not_a_file",
                )

            preserved_mode = None
            if target.exists():
                preserved_mode = stat.S_IMODE(target.stat().st_mode)

            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                    # Rewriting an executable script should not silently remove
                    # its executable bit.  New files use an ordinary 0644 mode;
                    # this project is single-user and does not claim ACL/xattr
                    # preservation.
                    os.fchmod(
                        handle.fileno(),
                        preserved_mode if preserved_mode is not None else 0o644,
                    )
                os.replace(temporary, target)
            finally:
                # After a successful replace the temporary pathname no longer
                # exists, so this conditional is harmless.  Before replacement
                # it prevents orphaned partial files.
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        except ToolRequestError:
            raise
        except OSError as exc:
            raise ToolExecutionError(
                "Could not atomically write the requested file.",
                error_code="filesystem_error",
                metadata={"operation": "atomic_write"},
            ) from exc

    def _walk_entries(self, start: Path) -> Iterator[tuple[Path, str, bool]]:
        """Yield lexical paths without recursing through directory symlinks."""

        pending = [start]
        while pending:
            directory = pending.pop()
            children = sorted(directory.iterdir(), key=lambda path: path.name)
            child_directories: list[Path] = []
            for child in children:
                if child.is_symlink():
                    try:
                        resolved = child.resolve(strict=True)
                        self.paths._require_contained(resolved)
                    except (OSError, RuntimeError, ToolRequestError):
                        yield child, "symlink", False
                        continue
                    # Directory links are listed but never traversed, preventing
                    # cycles and duplicate, surprising search trees.
                    yield child, "symlink", True
                elif child.is_dir():
                    yield child, "directory", True
                    child_directories.append(child)
                elif child.is_file():
                    yield child, "file", True
            # Reverse before stack insertion so the eventual traversal remains
            # lexicographically ascending and deterministic.
            pending.extend(reversed(child_directories))

    def _iter_files(self, start: Path) -> Iterator[tuple[Path, Path, bool]]:
        if start.is_file():
            yield start, start, True
            return
        for lexical, kind, safe in self._walk_entries(start):
            if not safe:
                yield lexical, lexical, False
                continue
            if kind == "file":
                yield lexical, lexical, True
            elif kind == "symlink":
                resolved = lexical.resolve(strict=True)
                if resolved.is_file():
                    yield lexical, resolved, True


def build_filesystem_tools(
    workspace: str | os.PathLike[str],
    **limits: int,
) -> tuple[Tool, ...]:
    """Convenience factory used when assembling the default registry."""

    return FileSystemTools(workspace, **limits).definitions()


def _positive_limit(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _truncate_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    available = max(0, limit - len(_TRUNCATION_MARKER))
    head = available // 2
    tail = available - head
    suffix = text[-tail:] if tail else ""
    return text[:head] + _TRUNCATION_MARKER + suffix, True
