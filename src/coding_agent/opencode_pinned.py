"""Harbor OpenCode entry point for a pre-installed, pinned CLI.

Harbor's stock OpenCode adapter installs Node and ``opencode-ai`` during every
trial.  That is useful for general jobs but makes a comparison vulnerable to
registry drift and setup/network failures.  ``PinnedOpenCodeAgent`` assumes a
derived task image already contains the requested binary and performs one
local version check during setup.  It never invokes nvm, npm, or an unpinned
package install.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

PINNED_OPENCODE_VERSION = "1.18.25"
PINNED_NODE_VERSION = "22.22.1"
# Keep the match bounded by version separators as well as digits.  Without the
# dot checks, malformed output such as ``1.18.25.1`` would be accepted as the
# pinned ``1.18.25`` prefix and could let a different binary through setup.
_VERSION_RE = re.compile(r"(?<![\d.])(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)(?![\d.])")


def parse_opencode_version(stdout: str) -> str | None:
    """Extract a semver-like version from ``opencode --version`` output."""

    if not isinstance(stdout, str):
        return
    match = _VERSION_RE.search(stdout.strip())
    return match.group(1) if match else None


def validate_pinned_version(
    stdout: str,
    *,
    return_code: int = 0,
    expected: str = PINNED_OPENCODE_VERSION,
) -> tuple[bool, str]:
    """Return a stable setup verdict without retaining command output."""

    if return_code != 0:
        return False, "opencode --version returned a non-zero status"
    version = parse_opencode_version(stdout)
    if version is None:
        return False, "opencode version output was missing or malformed"
    if version != expected:
        return False, f"opencode version mismatch (expected {expected})"
    return True, version


try:  # pragma: no cover - exercised when the optional Harbor extra is installed
    from harbor.agents.installed.base import (
        NonZeroAgentExitCodeError,
        with_prompt_template,
    )
    from harbor.agents.installed.opencode import OpenCode as _HarborOpenCode

    _HARBOR_PINNED_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - normal core installation path
    _HARBOR_PINNED_IMPORT_ERROR = exc

    class _HarborOpenCode:  # type: ignore[no-redef]
        """Import-safe stand-in; real use requires ``.[harbor]``."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.logs_dir = kwargs.get("logs_dir")
            self.model_name = kwargs.get("model_name")

        def version(self) -> str | None:
            """Mirror Harbor's version accessor for dependency-free callers."""

            value = getattr(self, "_version", None)
            return value if isinstance(value, str) else None

    class NonZeroAgentExitCodeError(RuntimeError):  # type: ignore[no-redef]
        pass

    def with_prompt_template(function):  # type: ignore[no-redef]
        return function


class PinnedOpenCodeAgent(_HarborOpenCode):
    """OpenCode Harbor adapter requiring ``1.18.25`` in the task image."""

    # Declare the same capabilities as Harbor's OpenCode adapter explicitly.
    # Keeping these on the subclass also makes the import-safe stand-in expose
    # the contract used by Harbor's retry/resume and ATIF collectors.
    SUPPORTS_ATIF = True
    SUPPORTS_RESUME = True
    SUPPORTS_WINDOWS = False
    REQUIRED_VERSION = PINNED_OPENCODE_VERSION
    REQUIRED_NODE_VERSION = PINNED_NODE_VERSION

    @staticmethod
    def name() -> str:
        return "opencode-pinned"

    def get_version_command(self) -> str:
        # Deliberately a bare binary invocation.  Keep this string easy to
        # audit: no shell profile, nvm, npm, network install, or @latest.
        return "opencode --version"

    async def install(self, environment) -> None:
        """Keep Harbor's optional install hook idempotent and offline.

        Harbor 0.22 normally calls ``setup`` (which we override below), but a
        wrapper may invoke the inherited install lifecycle directly. A no-op is
        preferable to raising there: the subsequent explicit version check is
        the single source of truth and still fails closed when the binary is
        absent or mismatched.
        """

        return

    async def setup(self, environment) -> None:
        """Verify the image-local binary and fail closed on any mismatch."""

        logs_dir = getattr(self, "logs_dir", None)
        if logs_dir is not None:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
        result = await environment.exec(command=self.get_version_command())
        return_code = getattr(result, "return_code", None)
        if return_code is None:
            return_code = getattr(result, "exit_code", 1)
        ok, detail = validate_pinned_version(
            getattr(result, "stdout", "") or "",
            return_code=return_code,
            expected=self.REQUIRED_VERSION,
        )
        if not ok:
            # Do not include arbitrary stdout/stderr: setup output can contain
            # credentials or internal paths supplied by a wrapper.
            raise RuntimeError(f"OpenCode setup failed: {detail}")
        self._version = detail

    @with_prompt_template
    async def run(self, instruction: str, environment, context) -> None:
        """Run the inherited OpenCode protocol without installation snippets."""

        if getattr(self, "_version", None) != self.REQUIRED_VERSION:
            raise RuntimeError(
                "OpenCode setup has not verified the required pinned version"
            )
        if not isinstance(self.model_name, str) or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        self._instruction = instruction
        env = dict(self.model_connection.env)
        env["OPENCODE_FAKE_VCS"] = "git"
        env["XDG_DATA_HOME"] = "/logs/agent/opencode/xdg-data"
        env["XDG_STATE_HOME"] = "/logs/agent/opencode/xdg-state"

        skills_command = self._build_register_skills_command()
        if skills_command:
            await self.exec_as_agent(environment, command=skills_command, env=env)
        config_command = self._build_register_config_command()
        if config_command:
            await self.exec_as_agent(environment, command=config_command, env=env)

        cli_flags = self.build_cli_flags()
        cli_flags_arg = (cli_flags + " ") if cli_flags else ""
        resume_flag = "--continue " if self._resume else ""
        command = (
            f"opencode --model={shlex.quote(self.model_name)} run --format=json "
            f"{resume_flag}{cli_flags_arg}--thinking "
            f"--dangerously-skip-permissions -- {shlex.quote(instruction)} "
            "2>&1 </dev/null | stdbuf -oL tee /logs/agent/opencode.txt"
        )
        await self.exec_as_agent(environment, command=command, env=env)
        if messages := self._error_messages():
            raise NonZeroAgentExitCodeError(
                "OpenCode emitted error event(s): " + "; ".join(messages[:3])
            )


__all__ = [
    "PINNED_NODE_VERSION",
    "PINNED_OPENCODE_VERSION",
    "PinnedOpenCodeAgent",
    "parse_opencode_version",
    "validate_pinned_version",
]
