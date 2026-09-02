"""Build and describe deterministic OpenCode task images.

The script is intentionally separate from the Harbor runner.  Image creation
is a setup concern and can be audited or retried before any model request is
made.  A generated dataset copy points every task at its derived image, so the
Course agent and :class:`PinnedOpenCodeAgent` consume the same filesystem
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from coding_agent.opencode_pinned import PINNED_NODE_VERSION, PINNED_OPENCODE_VERSION

TASK_IDS: tuple[str, ...] = (
    "fix-git",
    "cancel-async-tasks",
    "kv-store-grpc",
    "polyglot-c-py",
    "headless-terminal",
    "fix-code-vulnerability",
    "build-cython-ext",
    "write-compressor",
)

# These are the published Terminal-Bench task images resolved to the manifest
# digests advertised by Docker Hub on 2026-08-30.  Keeping the digest in the
# default is important: a mutable tag would let a later image rebuild silently
# change the task environment. Callers can provide a newer, independently
# audited immutable reference through ``--image`` when the official dataset
# updates its image tag.
DEFAULT_SOURCE_IMAGES: Mapping[str, str] = {
    "fix-git": "alexgshaw/fix-git@sha256:389b9c8247610c2c5be080b1ac00429007c2c69bf57f7f26c79f0f75ba2d5c74",
    "cancel-async-tasks": "alexgshaw/cancel-async-tasks@sha256:84c7fae6b256dcc56a350790e2a9715eefc7dad662a9d8e8a472363aa71ef18d",
    "kv-store-grpc": "alexgshaw/kv-store-grpc@sha256:3399400800dcb207634daa42bc1b052e831e285cc9d221eea66c47bc0fc79791",
    "polyglot-c-py": "alexgshaw/polyglot-c-py@sha256:0f1c3b7816d70cf5551573fd6aeef76893f2ae3000be2419997b6871b5d987ed",
    "headless-terminal": "alexgshaw/headless-terminal@sha256:eb7e209672bf6cef2785fafd9e13509b10626c327bcc2b37f5bf40ca83eaf3aa",
    "fix-code-vulnerability": "alexgshaw/fix-code-vulnerability@sha256:cac325252991f823713b2d0441502972901dd782bd67f66c03d9b1e410dac5c0",
    "build-cython-ext": "alexgshaw/build-cython-ext@sha256:3612a38fadb89a96f74a1a951fb0b0af734198fd160571eeaba6401593234594",
    "write-compressor": "alexgshaw/write-compressor@sha256:88c77df05432252dabdc09126840419e814c4a27059f63d69ab430ba6fbfaf47",
}

# The image built during the initial local setup is retained as a reference in
# reports. It is not used as a source for the other seven task images.
REFERENCE_READY_IMAGE_DIGEST = (
    "sha256:746a16bd3dfde8d8c3ded4025d02e63343b88b3c5f9ae8309a44a7b46d2ec0c5"
)
REFERENCE_READY_IMAGE = f"harbor-opencode-ready@{REFERENCE_READY_IMAGE_DIGEST}"
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
# Docker image references are inserted into ``FROM`` and TOML assignments.
# This intentionally accepts the ordinary registry/name[:tag][@digest] forms
# while rejecting shell/TOML metacharacters and control characters.
_IMAGE_REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]*$")


def _validate_image_reference(value: str, field: str) -> str:
    """Validate an image reference before interpolating it into setup files."""

    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or value != value.strip()
        or _IMAGE_REFERENCE_RE.fullmatch(value) is None
        or value.startswith(("/", ".", "-"))
        or value.endswith(("/", ".", "-", ":", "@"))
        or "//" in value
    ):
        raise ValueError(f"{field} must be a safe Docker image reference")
    if value.count("@") > 1:
        raise ValueError(f"{field} must contain at most one digest")
    if "@" in value:
        _name, digest = value.split("@", 1)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError(f"{field} digest must be a sha256 digest")
    return value


@dataclass(frozen=True, slots=True)
class ImageRecord:
    task: str
    source_image: str
    source_digest: str | None
    derived_image: str
    derived_digest: str | None
    node_version: str
    opencode_version: str
    status: str
    dockerfile_sha256: str
    toolchain_image: str | None = None


def derived_image_name(task: str, *, prefix: str = "terminal-bench-opencode") -> str:
    if task not in TASK_IDS:
        raise ValueError(f"unknown task: {task}")
    safe_prefix = re.sub(r"[^a-z0-9_.-]+", "-", prefix.lower()).strip("-._")
    if not safe_prefix:
        raise ValueError("image prefix must contain a Docker-safe character")
    return f"{safe_prefix}-{task}:{PINNED_OPENCODE_VERSION}"


def pinned_dockerfile(
    source_image: str,
    *,
    node_version: str = PINNED_NODE_VERSION,
    opencode_version: str = PINNED_OPENCODE_VERSION,
    toolchain_image: str | None = None,
) -> str:
    """Render the reproducible multi-stage Dockerfile used for one task."""

    if not isinstance(source_image, str) or not source_image.strip():
        raise ValueError("source_image must be non-empty")
    if not re.fullmatch(r"\d+\.\d+\.\d+", node_version):
        raise ValueError("node_version must be a semantic version")
    if not re.fullmatch(r"\d+\.\d+\.\d+", opencode_version):
        raise ValueError("opencode_version must be a semantic version")
    _validate_image_reference(source_image, "source_image")
    if toolchain_image is not None:
        _validate_image_reference(toolchain_image, "toolchain_image")
    # Keep the source reference visible in the generated file and pin both
    # package versions. ``npm install`` is executed only while building the
    # image, never from a Harbor trial.
    if toolchain_image is None:
        return f"""# syntax=docker/dockerfile:1
# Generated by pin_opencode_images.py; do not edit manually.
FROM node:{node_version}-bookworm-slim AS pinned-node
ARG OPENCODE_VERSION={opencode_version}
FROM {source_image}
ARG OPENCODE_VERSION={opencode_version}
COPY --from=pinned-node /usr/local/bin/node /usr/local/bin/node
COPY --from=pinned-node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
ENV PATH=/usr/local/bin:$PATH
RUN ln -sf ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \\
    && ln -sf ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx \\
    && node --version | grep -F 'v{node_version}' \\
    && npm --version >/dev/null \\
    && npm install --global --no-update-notifier --no-fund opencode-ai@$OPENCODE_VERSION \\
    && opencode --version | grep -Fx "$OPENCODE_VERSION"
"""
    # Offline/air-gapped hosts can provide a separately audited image that
    # already contains the exact Node/OpenCode toolchain.  The reference image
    # used by this project intentionally contains self-contained wrappers for
    # ``node``/``npm`` and a bundled OpenCode binary, rather than an npm module
    # tree (or ``npx``).  Copy only those image-local files so this branch is
    # valid for the audited toolchain and contains no package-manager or network
    # operation.
    return f"""# syntax=docker/dockerfile:1
# Generated by pin_opencode_images.py; do not edit manually.
FROM {toolchain_image} AS pinned-toolchain
FROM {source_image}
ARG OPENCODE_VERSION={opencode_version}
COPY --from=pinned-toolchain /usr/local/bin/node /usr/local/bin/node
COPY --from=pinned-toolchain /usr/local/bin/npm /usr/local/bin/npm
COPY --from=pinned-toolchain /usr/local/bin/opencode /usr/local/bin/opencode
ENV PATH=/usr/local/bin:$PATH
RUN node --version | grep -F 'v{node_version}' \\
    && opencode --version | grep -Fx "$OPENCODE_VERSION"
"""


def docker_build_argv(
    *,
    docker_bin: str,
    source_image: str,
    derived_image: str,
    dockerfile: Path,
    context: Path,
    sudo: bool = True,
) -> list[str]:
    _validate_image_reference(source_image, "source_image")
    _validate_image_reference(derived_image, "derived_image")
    argv = [
        docker_bin,
        "build",
        "--pull=false",
        "-f",
        str(dockerfile),
        "-t",
        derived_image,
        str(context),
    ]
    return ["sudo", "-n", *argv] if sudo else argv


def docker_inspect_digest(
    image: str,
    *,
    docker_bin: str = "docker",
    sudo: bool = True,
    runner=subprocess.run,
) -> str | None:
    argv = [
        docker_bin,
        "image",
        "inspect",
        image,
        "--format",
        "{{json .RepoDigests}}|{{.Id}}",
    ]
    if sudo:
        argv = ["sudo", "-n", *argv]
    try:
        completed = runner(
            argv, check=False, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    matches = _DIGEST_RE.findall(completed.stdout or "")
    return matches[0] if matches else None


@contextmanager
def _isolated_docker_context(
    dockerfile_text: str,
    *,
    parent: Path,
) -> Iterator[tuple[Path, Path]]:
    """Yield a temporary Dockerfile and an otherwise empty build context.

    The generated Dockerfiles only use ``FROM`` and cross-stage ``COPY``
    instructions, so no checkout files are needed by the daemon.  Keeping the
    context empty is important: invoking ``docker build .`` from the repository
    would otherwise upload an ignored-but-present ``.env`` (and potentially
    other credentials) to Docker before the Dockerfile is evaluated.
    """

    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".docker-context-", dir=parent) as raw:
        context = Path(raw)
        dockerfile = context / "Dockerfile"
        dockerfile.write_text(dockerfile_text, encoding="utf-8")
        os.chmod(dockerfile, 0o600)
        yield dockerfile, context


def _source_digest_from_ref(image: str) -> str | None:
    match = _DIGEST_RE.search(image)
    return match.group(0) if match else None


def build_image_records(
    *,
    images: Mapping[str, str] = DEFAULT_SOURCE_IMAGES,
    prefix: str = "terminal-bench-opencode",
    output_dir: Path = Path(".harbor-opencode-images"),
    docker_bin: str = "docker",
    sudo: bool = True,
    execute: bool = False,
    runner=subprocess.run,
    toolchain_image: str | None = None,
) -> list[ImageRecord]:
    """Build all eight images (or emit exact commands in dry-run mode)."""

    expected_tasks = set(TASK_IDS)
    supplied_tasks = set(images)
    unknown_tasks = sorted(supplied_tasks - expected_tasks)
    if unknown_tasks:
        raise ValueError(f"unknown task image key(s): {unknown_tasks!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[ImageRecord] = []
    for task in TASK_IDS:
        source = images.get(task)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"missing source image for {task}")
        dockerfile_text = pinned_dockerfile(source, toolchain_image=toolchain_image)
        dockerfile = output_dir / f"{task}.Dockerfile"
        dockerfile.write_text(dockerfile_text, encoding="utf-8")
        digest = hashlib.sha256(dockerfile_text.encode("utf-8")).hexdigest()
        derived = derived_image_name(task, prefix=prefix)
        status = "planned"
        derived_digest: str | None = None
        source_digest = _source_digest_from_ref(source)
        if execute:
            source_digest = source_digest or docker_inspect_digest(
                source, docker_bin=docker_bin, sudo=sudo, runner=runner
            )
            if source_digest is None:
                status = "source_digest_unavailable"
            else:
                # Use an empty temporary context rather than ``Path('.')``.
                # The latter can include a local .env in the Docker upload even
                # though it is ignored by Git.  The audited Dockerfile remains
                # in ``output_dir``; only this ephemeral copy is handed to the
                # daemon.
                with _isolated_docker_context(
                    dockerfile_text,
                    parent=output_dir,
                ) as (context_dockerfile, context):
                    argv = docker_build_argv(
                        docker_bin=docker_bin,
                        source_image=source,
                        derived_image=derived,
                        dockerfile=context_dockerfile,
                        context=context,
                        sudo=sudo,
                    )
                    completed = runner(argv, check=False, timeout=1800)
                if completed.returncode != 0:
                    status = "build_failed"
                else:
                    derived_digest = docker_inspect_digest(
                        derived, docker_bin=docker_bin, sudo=sudo, runner=runner
                    )
                    status = "built" if derived_digest else "digest_unavailable"
        records.append(
            ImageRecord(
                task=task,
                source_image=source,
                source_digest=source_digest,
                derived_image=derived,
                derived_digest=derived_digest,
                node_version=PINNED_NODE_VERSION,
                opencode_version=PINNED_OPENCODE_VERSION,
                status=status,
                dockerfile_sha256=digest,
                toolchain_image=toolchain_image,
            )
        )
    return records


def _replace_environment_image(text: str, image: str) -> str:
    """Replace only ``docker_image`` in the TOML ``[environment]`` table."""

    _validate_image_reference(image, "image")
    lines = text.splitlines(keepends=True)
    section_start: int | None = None
    section_end = len(lines)
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if stripped == "[environment]":
                section_start = index
            elif section_start is not None:
                section_end = index
                break
    if section_start is None:
        raise ValueError("task.toml has no [environment] table")
    assignment = f'docker_image = "{image}"\n'
    pattern = re.compile(r"^\s*docker_image\s*=.*$", re.MULTILINE)
    section = "".join(lines[section_start:section_end])
    if pattern.search(section):
        section = pattern.sub(assignment.rstrip("\n"), section, count=1)
    else:
        if not section.endswith("\n"):
            section += "\n"
        section += assignment
    return "".join(lines[:section_start]) + section + "".join(lines[section_end:])


def prepare_pinned_dataset(
    source_dataset: Path,
    output_dataset: Path,
    records: Sequence[ImageRecord],
) -> dict[str, Any]:
    """Copy a task checkout and point every selected task at its derived image."""

    source_dataset = source_dataset.expanduser().resolve()
    output_dataset = output_dataset.expanduser().resolve()
    if not source_dataset.is_dir():
        raise ValueError("source dataset directory does not exist")
    try:
        output_dataset.relative_to(source_dataset)
    except ValueError:
        pass
    else:
        raise ValueError("output dataset must not be inside the source dataset")
    output_dataset.mkdir(parents=True, exist_ok=True)
    # Copy the complete checkout, including every task.toml.  Only the eight
    # selected task files are rewritten below; dropping non-target task.toml
    # files would turn the official 89-task dataset into a subtly different
    # package and could also make Harbor's task discovery fail.
    for source in source_dataset.rglob("*"):
        relative = source.relative_to(source_dataset)
        destination = output_dataset / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    by_task = {record.task: record for record in records}
    if set(by_task) != set(TASK_IDS) or len(by_task) != len(records):
        raise ValueError("records must contain exactly one image record per task")
    for task in TASK_IDS:
        task_dir = output_dataset / task
        if not task_dir.is_dir():
            # Official checkouts may use a namespace directory.
            matches = list(output_dataset.rglob(task))
            task_dir = matches[0] if matches else task_dir
        task_file = task_dir / "task.toml"
        if task not in by_task or not task_file.is_file():
            raise ValueError(f"task.toml missing for {task}")
        _validate_image_reference(by_task[task].source_image, "source_image")
        _validate_image_reference(by_task[task].derived_image, "derived_image")
        task_file.write_text(
            _replace_environment_image(
                task_file.read_text(encoding="utf-8"), by_task[task].derived_image
            ),
            encoding="utf-8",
        )
    return {
        "source_dataset": str(source_dataset),
        "output_dataset": str(output_dataset),
        "tasks": [asdict(record) for record in records],
        "pinned_node_version": PINNED_NODE_VERSION,
        "pinned_opencode_version": PINNED_OPENCODE_VERSION,
        "reference_ready_image_digest": REFERENCE_READY_IMAGE_DIGEST,
        "toolchain_image": next(
            (
                record.toolchain_image
                for record in records
                if record.toolchain_image is not None
            ),
            None,
        ),
    }


def write_manifest(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path)
    parser.add_argument("--output-dataset", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".harbor-opencode-images")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path(".harbor-opencode-images/manifest.json")
    )
    parser.add_argument("--image", action="append", metavar="TASK=IMAGE")
    parser.add_argument("--prefix", default="terminal-bench-opencode")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--no-sudo", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--toolchain-image",
        help=(
            "optional immutable image already containing Node 22.22.1 and "
            "OpenCode 1.18.25; skips all package-manager/network steps"
        ),
    )
    args = parser.parse_args(argv)
    images = dict(DEFAULT_SOURCE_IMAGES)
    try:
        for item in args.image or ():
            task, separator, image = item.partition("=")
            if not separator or task not in TASK_IDS or not image:
                raise ValueError(
                    "--image must be TASK=IMAGE for one of the eight tasks"
                )
            images[task] = image
        records = build_image_records(
            images=images,
            prefix=args.prefix,
            output_dir=args.output_dir,
            docker_bin=args.docker_bin,
            sudo=not args.no_sudo,
            execute=args.execute,
            toolchain_image=args.toolchain_image,
        )
        document: dict[str, Any] = {
            "schema_version": "pinned-opencode-images/v1",
            "records": [asdict(record) for record in records],
            "pinned_node_version": PINNED_NODE_VERSION,
            "pinned_opencode_version": PINNED_OPENCODE_VERSION,
            "status": (
                "ready"
                if all(
                    record.status in {"built", "ready"}
                    and record.source_digest
                    and record.derived_digest
                    for record in records
                )
                else "not_ready"
            ),
            "reference_ready_image_digest": REFERENCE_READY_IMAGE_DIGEST,
            "toolchain_image": args.toolchain_image,
        }
        if args.source_dataset is not None:
            if args.output_dataset is None:
                raise ValueError("--output-dataset is required with --source-dataset")
            document["dataset"] = prepare_pinned_dataset(
                args.source_dataset, args.output_dataset, records
            )
        write_manifest(args.manifest, document)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"image setup error: {type(exc).__name__}: {exc}")
        return 2
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
