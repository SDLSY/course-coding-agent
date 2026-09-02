"""Build the exact two-file submission archive.

The command is intentionally fail-closed: it will not create a placeholder
video and it refuses a README/video that contains a known credential. Run it
after recording the real demonstration:

    python3 scripts/build_submission.py --video /path/to/demo.mp4

``ffprobe`` is required for the two-minute duration check unless
``--skip-duration-check`` is explicitly supplied for an offline fixture test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

from coding_agent.tools.shell import is_sensitive_environment_name

MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_SECONDS = 120.0
MAX_README_CHARS = 1000
README_NAME = "README.txt"
VIDEO_NAME = "李上一.mp4"
ARCHIVE_NAME = "李上一.zip"

_KEY_PATTERN = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~+/=-]{12,})"
)


def _known_secrets(environ: dict[str, str] | None = None) -> tuple[bytes, ...]:
    source = os.environ if environ is None else environ
    names = {name for name in source if is_sensitive_environment_name(name)}
    names.update(
        {
            "ZAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "CODING_AGENT_API_KEY",
        }
    )
    configured = source.get("CODING_AGENT_KEY_ENV")
    if configured:
        names.add(configured)
    values = {
        source[name].encode("utf-8")
        for name in names
        if isinstance(source.get(name), str) and source[name]
    }
    # Do not impose a production-token length heuristic here.  A short value
    # is still the exact credential configured by the caller (and is common in
    # contract/offline tests); rejecting it would leave a real value in the
    # archive scanner's blind spot.  The caller can choose not to configure a
    # variable at all when it is not in scope.
    return tuple(values)


def _contains_credential(data: bytes, secrets: Iterable[bytes]) -> bool:
    # Variable names are not credentials and may be mentioned in README
    # instructions. The scan targets actual values and conventional token
    # spellings; the caller separately supplies every value present in its
    # current environment.
    if _KEY_PATTERN.search(data):
        return True
    return any(secret in data for secret in secrets)


def _check_duration(video: Path) -> float:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise ValueError("ffprobe is required to verify the video duration") from exc
    if completed.returncode != 0:
        raise ValueError("ffprobe could not read the MP4 duration")
    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise ValueError("ffprobe returned an invalid video duration") from exc
    if not math.isfinite(duration) or duration < 0 or duration > MAX_VIDEO_SECONDS:
        raise ValueError("video duration must be at most 120 seconds")
    return duration


def build_archive(
    *,
    video: Path,
    readme: Path = Path(README_NAME),
    output: Path = Path(ARCHIVE_NAME),
    skip_duration_check: bool = False,
    force: bool = False,
    environ: dict[str, str] | None = None,
) -> dict[str, object]:
    """Validate inputs and atomically write the two-file archive."""

    video = video.expanduser().resolve(strict=False)
    readme = readme.expanduser().resolve(strict=False)
    output = output.expanduser().resolve(strict=False)
    if not readme.is_file():
        raise ValueError("README.txt does not exist")
    if not video.is_file():
        raise ValueError("video file does not exist")
    if video.suffix.lower() != ".mp4":
        raise ValueError("video must have an .mp4 extension")
    if video.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError("video must be no larger than 200 MB")

    readme_bytes = readme.read_bytes()
    try:
        readme_text = readme_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("README.txt must be UTF-8") from exc
    if len(readme_text) > MAX_README_CHARS:
        raise ValueError("README.txt must not exceed 1000 characters")

    secrets = _known_secrets(environ)
    if _contains_credential(readme_bytes, secrets):
        raise ValueError("README.txt contains a credential or credential variable")
    video_bytes = video.read_bytes()
    if _contains_credential(video_bytes, secrets):
        raise ValueError("video contains a credential pattern")
    duration: float | None = None
    if not skip_duration_check:
        duration = _check_duration(video)

    if output.exists() and not force:
        raise ValueError("output archive already exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            archive.write(readme, arcname=README_NAME)
            archive.write(video, arcname=VIDEO_NAME)
        os.chmod(temporary, 0o600)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    with zipfile.ZipFile(output, mode="r") as archive:
        names = archive.namelist()
    if names != [README_NAME, VIDEO_NAME]:
        raise ValueError("archive must contain exactly README.txt and 李上一.mp4")
    return {
        "archive": str(output),
        "files": names,
        "readme_chars": len(readme_text),
        "video_bytes": len(video_bytes),
        "video_seconds": duration,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--readme", type=Path, default=Path(README_NAME))
    parser.add_argument("--output", type=Path, default=Path(ARCHIVE_NAME))
    parser.add_argument("--skip-duration-check", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        report = build_archive(
            video=arguments.video,
            readme=arguments.readme,
            output=arguments.output,
            skip_duration_check=arguments.skip_duration_check,
            force=arguments.force,
        )
    except (OSError, ValueError) as exc:
        print(f"submission error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
