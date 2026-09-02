"""Plan or run the bounded Terminal-Bench 2.1 comparison.

This is intentionally a thin Harbor command runner rather than a second
benchmark implementation. Harbor owns task preparation, Docker isolation,
verifiers, trial timing, and ATIF collection. The script fixes the experiment
scope and records which commands were launched so an exploratory result cannot
be mistaken for an official leaderboard submission.

The default action is a plan. ``--execute`` is an explicit opt-in and requires
the model credential to already be present in the current process environment;
the value is never printed or persisted. Captured Harbor output is redacted
before it is written to the run directory.
"""

from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from coding_agent.config import PROVIDER_BASE_URLS, PROVIDER_KEY_ENV_NAMES, Provider
from coding_agent.events import redact
from coding_agent.opencode_pinned import (
    PINNED_NODE_VERSION,
    PINNED_OPENCODE_VERSION,
    validate_pinned_version,
)
from coding_agent.tools.shell import is_sensitive_environment_name

# Harbor Hub's immutable package route for Terminal-Bench 2.1. The legacy
# ``terminal-bench@2.1`` registry name is not published in Harbor 0.22. Pin
# the package content digest so a future ``harbor run`` cannot silently pick a
# changed task set. The eight task-level checksums are retained in reports.
DATASET = (
    "terminal-bench/terminal-bench-2-1@"
    "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a"
)
# Keep the immutable registry reference in reports even when Harbor is run
# against a local checkout downloaded from that package.
DATASET_REFERENCE = DATASET
EXPERIMENT_LABEL = "8-task exploratory comparison"
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
# The smoke gate intentionally uses one task that is already part of the
# formal matrix.  Keeping this as a named constant prevents the preflight and
# formal command from drifting to different fixtures.
SMOKE_TASK = "fix-code-vulnerability"
COURSE_AGENT = "coding_agent.harbor_plugin:CourseCodingAgent"
PINNED_OPENCODE_AGENT = "coding_agent.opencode_pinned:PinnedOpenCodeAgent"
# ``OPENCODE_AGENT`` is the public experiment constant. Keep the old Harbor
# shorthand separately for callers that still need to inspect historical jobs;
# no formal command or matrix validator accepts that unpinned implementation.
OPENCODE_AGENT = PINNED_OPENCODE_AGENT
LEGACY_OPENCODE_AGENT = "opencode"
# OpenCode's built-in ``openai`` provider always selects the Responses API.
# Zhipu's official endpoint is OpenAI-compatible but exposes Chat Completions
# only, so register a private provider ID for that one route. The provider
# definition is non-secret and receives its key from ``OPENAI_API_KEY``.
OPENCODE_GLM_PROVIDER = "zai"
# Harbor records the runtime names in ``agent_info`` while the job config
# records import paths.  Matrix validation canonicalizes both spellings to
# these stable labels, so a result cannot silently move between conditions.
COURSE_AGENT_LABEL = "course-coding-agent"
PINNED_OPENCODE_AGENT_LABEL = "opencode-pinned"
MATRIX_AGENT_LABELS: tuple[str, ...] = (
    COURSE_AGENT_LABEL,
    PINNED_OPENCODE_AGENT_LABEL,
)
_AGENT_LABEL_ALIASES: Mapping[str, str] = {
    COURSE_AGENT: COURSE_AGENT_LABEL,
    COURSE_AGENT_LABEL: COURSE_AGENT_LABEL,
    "course": COURSE_AGENT_LABEL,
    PINNED_OPENCODE_AGENT: PINNED_OPENCODE_AGENT_LABEL,
    PINNED_OPENCODE_AGENT_LABEL: PINNED_OPENCODE_AGENT_LABEL,
}
FORMAL_REPETITIONS = 3
RELAY_BASE_URL = "https://spacetimeai.cc/v1"
MODEL_IDS: tuple[str, ...] = (
    "deepseek-v4-flash-vision-exp",
    "glm-5.3-flash",
    "gpt-5.6-sol",
)
MODEL_ROUTES: Mapping[str, dict[str, str]] = {
    MODEL_IDS[0]: {
        "provider": Provider.DEEPSEEK.value,
        "base_url": PROVIDER_BASE_URLS[Provider.DEEPSEEK],
        "key_env": PROVIDER_KEY_ENV_NAMES[Provider.DEEPSEEK],
    },
    MODEL_IDS[1]: {
        "provider": Provider.GLM.value,
        "base_url": PROVIDER_BASE_URLS[Provider.GLM],
        "key_env": PROVIDER_KEY_ENV_NAMES[Provider.GLM],
    },
    MODEL_IDS[2]: {
        "provider": Provider.CUSTOM.value,
        "base_url": RELAY_BASE_URL,
        "key_env": PROVIDER_KEY_ENV_NAMES[Provider.CUSTOM],
    },
}
DEFAULT_AGENT_MAX_TURNS = 20
EFFICIENCY_AGENT_MAX_TURNS = 20
LONG_AGENT_MAX_TURNS = 30
AGENT_MAX_TOOL_CALLS = 80
AGENT_TIMEOUT_SECONDS = 900
FORMAL_DEFAULT_STRATEGY = "efficiency_20"
# Keep these values in one place so the ablation and formal matrix cannot
# silently drift apart.  ``current_20`` is retained for the comparison only;
# formal runs should normally use one of the efficiency strategies selected by
# the completed ablation.
ROUND_STRATEGIES: Mapping[str, Mapping[str, int | bool]] = {
    "current_20": {
        "max_model_turns": DEFAULT_AGENT_MAX_TURNS,
        "efficiency_mode": False,
        "reserve_final_turn": False,
    },
    "efficiency_20": {
        "max_model_turns": EFFICIENCY_AGENT_MAX_TURNS,
        "efficiency_mode": True,
        "reserve_final_turn": True,
    },
    "efficiency_30": {
        "max_model_turns": LONG_AGENT_MAX_TURNS,
        "efficiency_mode": True,
        "reserve_final_turn": True,
    },
}
_ENV_TEMPLATE_RE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_ENV_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SECRET_TOKEN_RE = re.compile(
    rb"(?:sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~+/=-]{12,})"
)
_REASONING_ENV_NAMES = (
    "CODING_AGENT_REASONING_EFFORT",
    "CODING_AGENT_REASONING_PARAMETER",
    "CODING_AGENT_REASONING_VALUE",
    "CODING_AGENT_REASONING_CAPABILITY_STATUS",
    "CODING_AGENT_REASONING_STATUS",
)
_IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
# Manifest image values are later passed to Harbor through generated task
# metadata. Reject interpolation/control characters even in dry-run paths so a
# malformed manifest cannot alter the command or dataset selected for a trial.
_IMAGE_REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]*$")
_IMAGE_READY_STATUSES = frozenset({"ready", "built"})
_IMAGE_MANIFEST_SCHEMA_VERSION = "pinned-opencode-images/v1"
_MANIFEST_HASH_RE = re.compile(r"[0-9a-f]{64}")
_PUBLIC_MANIFEST_RECORD_FIELDS = frozenset(
    {
        "task",
        "source_image",
        "source_digest",
        "derived_image",
        "derived_digest",
        "status",
        "node_version",
        "opencode_version",
    }
)


class _ValidatedImageManifest(dict[str, Any]):
    """Internal mapping that retains the hash of the original source JSON."""

    __slots__ = ("source_hash",)

    source_hash: str


def _manifest_source_document(
    manifest: Mapping[str, Any] | str | os.PathLike[str],
) -> tuple[Mapping[str, Any], str | None]:
    """Load a manifest object and return its source path without raw payloads."""

    if isinstance(manifest, Mapping):
        return manifest, None
    try:
        path = Path(manifest).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError("image manifest path is invalid") from exc
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("image manifest is not readable JSON") from exc
    if not isinstance(loaded, Mapping):
        raise TypeError("image manifest must contain a JSON object")
    return loaded, str(path)


def _manifest_text(value: Any, field: str, *, max_length: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > max_length
        or any(char in value for char in "\r\n\x00")
    ):
        raise ValueError(f"image manifest {field} must be a bounded non-empty string")
    return value.strip()


def _manifest_image_reference(value: Any, field: str) -> str:
    image = _manifest_text(value, field)
    if (
        _IMAGE_REFERENCE_RE.fullmatch(image) is None
        or image.startswith(("/", ".", "-"))
        or image.endswith(("/", ".", "-", ":", "@"))
        or "//" in image
        or image.count("@") > 1
    ):
        raise ValueError(
            f"image manifest {field} must be a safe Docker image reference"
        )
    if "@" in image:
        _name, digest = image.split("@", 1)
        if _IMAGE_DIGEST_RE.fullmatch(digest) is None:
            raise ValueError(f"image manifest {field} digest must be a sha256 digest")
    return image


def _manifest_digest(value: Any, field: str, *, required: bool = False) -> str | None:
    if value is None or value == "":
        if required:
            raise ValueError(f"image manifest {field} must be a sha256 digest")
        return None
    if not isinstance(value, str) or _IMAGE_DIGEST_RE.fullmatch(value.strip()) is None:
        raise ValueError(f"image manifest {field} must be a sha256 digest")
    return value.strip()


def _manifest_version(
    document: Mapping[str, Any],
    names: Sequence[str],
    expected: str,
    label: str,
) -> str:
    values = [document.get(name) for name in names if document.get(name) is not None]
    if not values:
        raise ValueError(f"image manifest is missing {label}")
    if any(value != expected for value in values):
        raise ValueError(f"image manifest {label} must be {expected}")
    return expected


def _manifest_output_dataset(document: Mapping[str, Any]) -> str | None:
    """Read the rewritten checkout path from the pinning script's schema."""

    candidates: list[Any] = [document.get("output_dataset")]
    nested = document.get("dataset")
    if isinstance(nested, Mapping):
        candidates.append(nested.get("output_dataset"))
    elif isinstance(nested, str):
        candidates.append(nested)
    for value in candidates:
        if value is None:
            continue
        return _manifest_text(value, "output_dataset", max_length=4096)
    return None


def _manifest_dataset_matches(output_dataset: str | None, dataset: str | None) -> bool:
    if output_dataset is None or dataset is None:
        return True
    if not _is_local_dataset(dataset):
        # A manifest that names a rewritten checkout cannot be paired with an
        # immutable registry route: doing so would silently run the original
        # task images while reporting the derived-image metadata.
        return dataset.strip() == output_dataset.strip()
    try:
        return Path(output_dataset).expanduser().resolve(strict=False) == Path(
            dataset
        ).expanduser().resolve(strict=False)
    except (OSError, ValueError):
        return False


def _is_public_manifest_document(document: Mapping[str, Any]) -> bool:
    """Recognize the hash-carrying projection emitted in reports/plans."""

    if "manifest_sha256" not in document:
        return False
    records = document.get("records")
    if not isinstance(records, list):
        return False
    return all(
        isinstance(record, Mapping)
        and not (set(record) - _PUBLIC_MANIFEST_RECORD_FIELDS)
        for record in records
    )


def validate_image_manifest(
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    *,
    expected_tasks: Sequence[str] = TASK_IDS,
    dataset: str | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Validate and project a pinned-image manifest into safe metadata.

    The pinning helper writes a ``records`` list.  This function accepts a path
    or an in-memory mapping, requires exactly one record per expected task, and
    returns only fields needed by the experiment report.  ``require_ready`` is
    used immediately before an actual Harbor launch; dry plans can retain a
    useful ``not_ready`` state while setup is still pending.
    """

    document, source_path = _manifest_source_document(manifest)
    expected = tuple(str(item) for item in expected_tasks)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_tasks must contain unique task names")
    schema_version = document.get("schema_version")
    if schema_version is not None and schema_version != _IMAGE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"image manifest schema_version must be {_IMAGE_MANIFEST_SCHEMA_VERSION}"
        )
    node_version = _manifest_version(
        document,
        ("pinned_node_version", "node_version"),
        PINNED_NODE_VERSION,
        "pinned_node_version",
    )
    opencode_version = _manifest_version(
        document,
        ("pinned_opencode_version", "opencode_version"),
        PINNED_OPENCODE_VERSION,
        "pinned_opencode_version",
    )
    raw_records = document.get("records")
    if raw_records is None:
        raw_records = document.get("images")
    if not isinstance(raw_records, Sequence) or isinstance(
        raw_records, (str, bytes, bytearray)
    ):
        raise TypeError("image manifest records must be a list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise TypeError(f"image manifest record {index} must be an object")
        task_value = raw.get("task")
        if task_value is None:
            task_value = raw.get("task_name", raw.get("task_id", raw.get("name")))
        task = _manifest_text(task_value, f"records[{index}].task", max_length=256)
        if task not in expected:
            raise ValueError(f"image manifest contains unknown task {task!r}")
        if task in seen:
            raise ValueError(f"image manifest contains duplicate task {task!r}")
        seen.add(task)
        source_image = _manifest_image_reference(
            raw.get("source_image"), f"records[{index}].source_image"
        )
        derived_image = _manifest_image_reference(
            raw.get("derived_image"), f"records[{index}].derived_image"
        )
        source_digest = _manifest_digest(
            raw.get("source_digest", raw.get("source_image_digest")),
            f"records[{index}].source_digest",
        )
        if source_digest is None:
            source_match = re.search(
                r"@(?P<digest>sha256:[0-9a-f]{64})(?:$|[?#])", source_image
            )
            if source_match:
                source_digest = source_match.group("digest")
        derived_digest = _manifest_digest(
            raw.get(
                "derived_digest",
                raw.get("derived_image_digest", raw.get("image_digest")),
            ),
            f"records[{index}].derived_digest",
        )
        status_raw = raw.get(
            "status", raw.get("image_status", document.get("status", "unknown"))
        )
        status = _manifest_text(status_raw, f"records[{index}].status", max_length=64)
        record_node = raw.get("node_version", raw.get("pinned_node_version"))
        if require_ready and record_node is None:
            raise ValueError(
                f"image manifest records[{index}].node_version is required when ready"
            )
        if record_node is not None and record_node != node_version:
            raise ValueError(
                f"image manifest records[{index}].node_version must be {node_version}"
            )
        record_opencode = raw.get(
            "opencode_version", raw.get("pinned_opencode_version")
        )
        if require_ready and record_opencode is None:
            raise ValueError(
                f"image manifest records[{index}].opencode_version is required when ready"
            )
        if record_opencode is not None and record_opencode != opencode_version:
            raise ValueError(
                f"image manifest records[{index}].opencode_version must be {opencode_version}"
            )
        records.append(
            {
                "task": task,
                "source_image": source_image,
                "source_digest": source_digest,
                "derived_image": derived_image,
                "derived_digest": derived_digest,
                "status": status,
                "node_version": node_version,
                "opencode_version": opencode_version,
            }
        )
    missing = sorted(set(expected) - seen)
    if missing:
        raise ValueError(f"image manifest is missing task(s): {', '.join(missing)}")
    output_dataset = _manifest_output_dataset(document)
    if not _manifest_dataset_matches(output_dataset, dataset):
        raise ValueError("image manifest output_dataset does not match --dataset")
    top_status = document.get("status")
    if top_status is not None:
        top_status = _manifest_text(top_status, "status", max_length=64)
    ready_records = all(
        record["status"].lower() in _IMAGE_READY_STATUSES
        and _IMAGE_DIGEST_RE.fullmatch(record["source_digest"] or "") is not None
        and _IMAGE_DIGEST_RE.fullmatch(record["derived_digest"] or "") is not None
        for record in records
    )
    ready = ready_records and (
        top_status is None or top_status.lower() in _IMAGE_READY_STATUSES
    )
    if require_ready and not ready:
        raise ValueError(
            "image manifest is not ready: every task needs a built/ready status "
            "and source/derived sha256 digest"
        )
    try:
        canonical = json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("image manifest contains non-JSON values") from exc
    manifest_hash = hashlib.sha256(canonical).hexdigest()
    # A plan/report contains only the validated public projection, so its
    # canonical source payload is intentionally absent. Preserve the original
    # source hash carried by that projection while still validating every
    # exposed record above. Raw manifests always use the freshly computed hash.
    if _is_public_manifest_document(document):
        declared_hash = document.get("manifest_sha256")
        if (
            not isinstance(declared_hash, str)
            or _MANIFEST_HASH_RE.fullmatch(declared_hash) is None
        ):
            raise ValueError(
                "image manifest manifest_sha256 must be a 64-character hex digest"
            )
        manifest_hash = declared_hash
    result = _ValidatedImageManifest(
        {
            "schema_version": _IMAGE_MANIFEST_SCHEMA_VERSION,
            "path": source_path,
            "manifest_sha256": manifest_hash,
            "status": "ready" if ready else (top_status or "not_ready"),
            "ready": ready,
            "pinned_node_version": node_version,
            "pinned_opencode_version": opencode_version,
            "output_dataset": output_dataset,
            "records": records,
        }
    )
    result.source_hash = manifest_hash
    return result


def load_image_manifest(
    manifest: Mapping[str, Any] | str | os.PathLike[str],
    *,
    expected_tasks: Sequence[str] = TASK_IDS,
    dataset: str | None = None,
    require_ready: bool = False,
) -> dict[str, Any]:
    """Public path/mapping loader retained as a descriptive alias."""

    return validate_image_manifest(
        manifest,
        expected_tasks=expected_tasks,
        dataset=dataset,
        require_ready=require_ready,
    )


def _image_manifest_for_use(
    manifest: Mapping[str, Any] | str | os.PathLike[str] | None,
    *,
    dataset: str | None = None,
    require_ready: bool = False,
) -> dict[str, Any] | None:
    if manifest is None:
        return None
    # A plan may pass the already projected object back through a second
    # builder. Revalidate its public fields without requiring the original
    # source payload or path to be available.
    validated = validate_image_manifest(
        manifest,
        dataset=dataset,
        require_ready=require_ready,
    )
    if isinstance(manifest, _ValidatedImageManifest):
        # Revalidation protects against a caller mutating the mapping while
        # preserving the digest of the original source document for audit
        # continuity across plan/report builders.
        validated["manifest_sha256"] = manifest.source_hash
        validated.source_hash = manifest.source_hash
    return validated


def _require_execution_image_manifest(
    manifest: Mapping[str, Any] | str | os.PathLike[str] | None,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Require a ready pinned-image manifest immediately before a launch.

    Plans are useful before Docker preparation is complete, but an actual
    Harbor invocation must never silently fall back to the upstream task image.
    Keeping this check in one helper also ensures smoke, ablation, and formal
    runs apply identical digest/version/readiness rules.
    """

    if manifest is None:
        raise ValueError(
            "Harbor execution requires --image-manifest for the ready pinned "
            "OpenCode task images"
        )
    validated = _image_manifest_for_use(
        manifest,
        dataset=dataset,
        require_ready=True,
    )
    if validated is None:  # pragma: no cover - guarded by the check above
        raise ValueError("Harbor execution requires a pinned image manifest")
    return validated


def _image_manifest_public(manifest: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(manifest, Mapping):
        return None
    records = manifest.get("records")
    record_fields = (
        "task",
        "source_image",
        "source_digest",
        "derived_image",
        "derived_digest",
        "status",
        "node_version",
        "opencode_version",
    )
    safe_records: list[dict[str, Any]] = []
    if isinstance(records, list):
        for item in records:
            if isinstance(item, Mapping):
                safe_records.append({field: item.get(field) for field in record_fields})
    return {
        "schema_version": _IMAGE_MANIFEST_SCHEMA_VERSION,
        "path": manifest.get("path"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "status": manifest.get("status"),
        "ready": manifest.get("ready") is True,
        "pinned_node_version": manifest.get("pinned_node_version"),
        "pinned_opencode_version": manifest.get("pinned_opencode_version"),
        "output_dataset": manifest.get("output_dataset"),
        "records": safe_records,
    }


def _attach_image_manifest_row(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach only the selected task's safe image metadata to one row."""

    result = dict(row)
    if not isinstance(manifest, Mapping):
        return result
    task = _canonical_task_name(
        result.get("matrix_task") or result.get("task_name"), TASK_IDS
    )
    records = manifest.get("records")
    record = (
        next(
            (
                item
                for item in records
                if isinstance(item, Mapping) and item.get("task") == task
            ),
            None,
        )
        if isinstance(records, list) and task is not None
        else None
    )
    if isinstance(record, Mapping):
        result["image_digest"] = record.get("derived_digest")
        result["source_image_digest"] = record.get("source_digest")
        result["opencode_version"] = record.get(
            "opencode_version", manifest.get("pinned_opencode_version")
        )
        result["image_status"] = record.get("status")
        result["derived_image"] = record.get("derived_image")
        result["source_image"] = record.get("source_image")
    result["image_manifest_sha256"] = manifest.get("manifest_sha256")
    result["image_manifest_status"] = manifest.get("status")
    result["image_manifest_path"] = manifest.get("path")
    return result


@dataclass(frozen=True, slots=True)
class ExperimentCommand:
    """One fully expanded Harbor invocation, safe to display in a plan."""

    agent: str
    argv: tuple[str, ...]


def check_pinned_opencode_setup(
    stdout: str,
    *,
    return_code: int = 0,
) -> dict[str, Any]:
    """Project an image-side version check into safe setup metadata."""

    ok, detail = validate_pinned_version(
        stdout,
        return_code=return_code,
        expected=PINNED_OPENCODE_VERSION,
    )
    return {
        "status": "ready" if ok else "setup_failure",
        "required_version": PINNED_OPENCODE_VERSION,
        "observed": detail if ok else None,
        "reason": None if ok else detail,
    }


def build_command(
    *,
    agent: str,
    model: str,
    dataset: str = DATASET,
    jobs_dir: str | os.PathLike[str] = ".harbor-runs",
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    agent_env: Mapping[str, str] | None = None,
    allow_agent_hosts: Iterable[str] = (),
    repetitions: int = 1,
    agent_timeout_seconds: float | None = AGENT_TIMEOUT_SECONDS,
    agent_kwargs: Mapping[str, Any] | None = None,
    task_ids: Sequence[str] | None = None,
) -> ExperimentCommand:
    """Build one deterministic Harbor command for all eight task names."""

    if not isinstance(agent, str) or not agent.strip():
        raise ValueError("agent must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    if not isinstance(harbor_bin, str) or not harbor_bin.strip():
        raise ValueError("harbor_bin must be a non-empty string")
    if any(char in harbor_bin for char in "\x00\r\n"):
        raise ValueError("harbor_bin must not contain control characters")
    if not isinstance(harbor_sudo, bool):
        raise TypeError("harbor_sudo must be a boolean")
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("repetitions must be a positive integer")
    dataset_option = "--path" if _is_local_dataset(dataset) else "--dataset"
    argv: list[str] = [
        *harbor_invocation_prefix(harbor_bin, harbor_sudo=harbor_sudo),
        "run",
        dataset_option,
        dataset,
        "--agent",
        agent,
        "--model",
        model,
        "--n-concurrent",
        "1",
        "--n-attempts",
        str(repetitions),
        "--max-retries",
        "0",
        # Keep Harbor's task/agent/verifier timeout policy explicit and equal
        # across both agents. The task files provide the 900-second base caps.
        "--timeout-multiplier",
        "1.0",
        "--agent-timeout-multiplier",
        "1.0",
        "--verifier-timeout-multiplier",
        "1.0",
        "--environment-build-timeout-multiplier",
        "1.0",
        # The runner is deliberately non-interactive; this only confirms the
        # local Docker/host-environment prompts and does not expose a secret.
        "--yes",
        "--jobs-dir",
        str(Path(jobs_dir)),
    ]
    if agent_timeout_seconds is not None and (
        isinstance(agent_timeout_seconds, bool)
        or not isinstance(agent_timeout_seconds, (int, float))
        or not math.isfinite(float(agent_timeout_seconds))
        or agent_timeout_seconds <= 0
    ):
        raise ValueError("agent_timeout_seconds must be positive and finite")
        # ``harbor run`` is the Harbor 0.22 job command and deliberately does
        # not expose the absolute ``--agent-timeout`` option (that option is
        # available only on ``harbor trial start``).  The Terminal-Bench task
        # TOMLs pin the agent timeout at 900 seconds, and the Course agent also
        # receives the same limit through ``max_wall_time_seconds``.  Keep the
        # argument as a validated API/documentation value without emitting an
        # option that would make the job fail before its first trial.
    for host in sorted(set(allow_agent_hosts)):
        if (
            not isinstance(host, str)
            or not host
            or any(char.isspace() for char in host)
            or "/" in host
            or "\x00" in host
        ):
            raise ValueError("allow_agent_hosts must contain bare host names")
        argv.extend(("--allow-agent-host", host))
    for name, value in sorted((agent_env or {}).items()):
        if (
            not isinstance(name, str)
            or not name
            or "=" in name
            or "\x00" in name
            or not isinstance(value, str)
            or "\x00" in value
            or "\n" in value
            or "\r" in value
        ):
            raise ValueError("agent_env must contain safe KEY=VALUE strings")
        if is_sensitive_environment_name(name) and not _ENV_TEMPLATE_RE.fullmatch(
            value
        ):
            # Harbor expands templates in the host process. Requiring that
            # convention prevents literal credentials from entering argv or a
            # dry-run report when this helper is used directly.
            raise ValueError("sensitive agent_env values must be ${ENV_NAME} templates")
        argv.extend(("--agent-env", f"{name}={value}"))
    for name, value in sorted((agent_kwargs or {}).items()):
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name)
            or isinstance(value, set)
        ):
            raise ValueError("agent_kwargs must contain scalar key/value pairs")
        if isinstance(value, (dict, list, tuple)):
            # Harbor parses agent kwargs with ``json.loads``.  Preserve the
            # exact native reasoning value (which may be a JSON object/list)
            # instead of relying on Python's single-quoted repr.
            if name not in {"reasoning_value", "opencode_config"}:
                raise ValueError(
                    "structured agent kwargs are supported only for reasoning_value "
                    "and opencode_config"
                )
            try:
                encoded_value = json.dumps(
                    value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be JSON-serializable") from exc
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("agent_kwargs must contain finite scalar values")
        elif value is None:
            raise ValueError("agent_kwargs values must not be null")
        elif isinstance(value, bool) and name == "reasoning_value":
            # Lower-case JSON preserves the boolean type when Harbor parses the
            # ``key=value`` string.  Existing non-reasoning flags retain their
            # historical Python spelling for compatibility with older Harbor
            # wrappers.
            encoded_value = json.dumps(value)
        else:
            encoded_value = str(value)
        argv.extend(("--agent-kwarg", f"{name}={encoded_value}"))
    selected_tasks = list(TASK_IDS if task_ids is None else task_ids)
    unknown_tasks = [task for task in selected_tasks if task not in TASK_IDS]
    if unknown_tasks:
        raise ValueError(f"unknown task filter(s): {unknown_tasks!r}")
    filters = (
        selected_tasks
        if _is_local_dataset(dataset)
        else [
            task
            if "/" in task or not (namespace := _task_namespace(dataset))
            else f"{namespace}/{task}"
            for task in selected_tasks
        ]
    )
    for filtered_name in filters:
        argv.extend(("--include-task-name", filtered_name))
    return ExperimentCommand(agent=agent, argv=tuple(argv))


def harbor_invocation_prefix(
    harbor_bin: str = "harbor", *, harbor_sudo: bool = False
) -> tuple[str, ...]:
    """Return the safe argv prefix used to invoke Harbor.

    Keeping this projection separate from ``build_command`` lets plans record
    whether the host requires passwordless sudo without persisting an entire
    command or any environment values.
    """

    if not isinstance(harbor_bin, str) or not harbor_bin.strip():
        raise ValueError("harbor_bin must be a non-empty string")
    if any(char in harbor_bin for char in "\x00\r\n"):
        raise ValueError("harbor_bin must not contain control characters")
    if not isinstance(harbor_sudo, bool):
        raise TypeError("harbor_sudo must be a boolean")
    return ("sudo", "-n", harbor_bin) if harbor_sudo else (harbor_bin,)


def _task_namespace(dataset: str) -> str | None:
    """Return the registry namespace used by package task IDs."""

    # Package datasets use IDs such as ``terminal-bench/fix-git`` while the
    # dataset route itself is ``terminal-bench/terminal-bench-2-1``. For a
    # custom route, the first *non-filesystem* path component follows the same
    # convention.  A local absolute checkout (``/tmp/.../terminal-bench-2-1``)
    # has no registry prefix, so recognize the official directory name rather
    # than accidentally treating ``tmp`` as a task namespace.
    raw_route = dataset.split("@", 1)[0].strip()
    if raw_route.startswith("/"):
        directory_name = Path(raw_route).name
        if directory_name.startswith("terminal-bench-"):
            return "terminal-bench"
        return None
    route = raw_route.strip("/")
    if "/" not in route:
        return None
    return route.split("/", 1)[0] or None


def _is_local_dataset(dataset: str) -> bool:
    """Return whether Harbor should receive a filesystem dataset path."""

    value = dataset.strip()
    if value.startswith(("/", "./", "../")):
        return True
    # Relative paths used by callers need not exist during a dry-run, but an
    # existing directory is unambiguously a local Harbor dataset.
    try:
        return Path(value).is_dir()
    except OSError:
        return False


def _dataset_reference(dataset: str) -> str | None:
    """Return the pinned upstream reference represented by ``dataset``."""

    if dataset == DATASET_REFERENCE:
        return DATASET_REFERENCE
    if not _is_local_dataset(dataset):
        return None
    try:
        # The official package extracts to this stable directory name.  Do
        # not infer provenance from arbitrary paths with similar contents.
        if Path(dataset).resolve().name == "terminal-bench-2-1":
            return DATASET_REFERENCE
    except OSError:
        return None
    return None


def _task_filters(dataset: str) -> list[str]:
    """Return task-name filters in the form expected by Harbor's source."""

    if _is_local_dataset(dataset):
        return list(TASK_IDS)
    namespace = _task_namespace(dataset)
    return [
        task if "/" in task or not namespace else f"{namespace}/{task}"
        for task in TASK_IDS
    ]


def required_credential_names(
    environ: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return ``(model_variable, key_variable)`` without exposing values."""

    source = os.environ if environ is None else environ
    model_name = "CODING_AGENT_MODEL"
    provider_text = source.get("CODING_AGENT_PROVIDER", Provider.GLM.value)
    try:
        provider = Provider(provider_text.strip().lower())
    except (AttributeError, ValueError):
        # The actual plugin will produce the detailed configuration error. The
        # runner keeps this check conservative and never guesses a key value.
        key_name = source.get("CODING_AGENT_KEY_ENV", "ZAI_API_KEY")
    else:
        key_name = source.get("CODING_AGENT_KEY_ENV", PROVIDER_KEY_ENV_NAMES[provider])
    return model_name, key_name


def check_credentials(environ: dict[str, str] | None = None) -> tuple[str, ...]:
    """Check presence only; return missing variable *names*, never values."""

    source = os.environ if environ is None else environ
    model_name, key_name = required_credential_names(source)
    missing = tuple(
        name
        for name in (model_name, key_name)
        if not isinstance(source.get(name), str) or not source[name].strip()
    )
    return missing


def _credential_values(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Collect credential values for in-memory log redaction only."""

    _model_name, key_name = required_credential_names(dict(environ))
    names = {
        key_name,
        "ZAI_API_KEY",
        "DEEPSEEK_API_KEY",
        "CODING_AGENT_API_KEY",
    }
    names.update(
        name
        for name in environ
        if isinstance(name, str) and is_sensitive_environment_name(name)
    )
    # The four provider variables are known credential slots.  Do not impose a
    # minimum length on them: short synthetic values are common in contract
    # tests, and redaction must not depend on a token looking production-like.
    known = {
        name
        for name in names
        if name
        in {
            key_name,
            "ZAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "CODING_AGENT_API_KEY",
        }
    }
    values: set[str] = {
        value
        for name in known
        for value in (environ.get(name),)
        if isinstance(value, str) and value
    }
    # Name-based sensitive variables are also redacted.  This intentionally
    # includes short values; the caller has already opted into a credential-
    # shaped variable name, so withholding it is safer than guessing a length.
    values.update(
        value
        for name, value in environ.items()
        if isinstance(name, str)
        and is_sensitive_environment_name(name)
        and isinstance(value, str)
        and value
    )
    return tuple(sorted(values, key=len, reverse=True))


def _runner_redaction_secrets(secrets: Iterable[str]) -> tuple[str, ...]:
    """Select values safe for broad textual replacement in runner metadata.

    A one- or two-character credential can occur naturally in field names and
    prose.  The raw values are still collected and scrubbed byte-for-byte by
    ``_scrub_generated_artifacts``; this narrower set prevents broad redaction
    from changing keys such as ``can_proceed`` while preserving normal token
    replacement for realistic credentials.
    """

    return tuple(
        sorted(
            {
                secret
                for secret in secrets
                if isinstance(secret, str) and len(secret) >= 4
            },
            key=len,
            reverse=True,
        )
    )


def _redact_runner_text(value: str, secrets: Iterable[str]) -> str:
    """Redact runner text without treating a one-character key as a substring."""

    values = tuple(
        str(secret) for secret in secrets if isinstance(secret, str) and secret
    )
    safe = redact(value, secrets=_runner_redaction_secrets(values))
    if not isinstance(safe, str):
        safe = str(safe)
    for secret in values:
        if len(secret) >= 4:
            continue
        safe = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(secret)}(?![A-Za-z0-9_])",
            "[REDACTED]",
            safe,
        )
    return safe


def _harbor_process_environment(
    environ: Mapping[str, str],
    *,
    strip_reasoning: bool = False,
) -> dict[str, str]:
    """Build the minimal host environment needed by Harbor and both agents.

    Harbor resolves ``${VAR}`` templates for OpenCode in its host process, and
    the custom agent reads its selected key there as well.  Passing every
    inherited variable would unnecessarily expose unrelated credentials to
    Harbor plugins and to any diagnostic subprocess, so only the selected key
    is explicitly re-added after conventional sensitive names are filtered.
    """

    _model_name, key_name = required_credential_names(dict(environ))
    safe = {
        name: value
        for name, value in environ.items()
        if isinstance(name, str)
        and isinstance(value, str)
        and not is_sensitive_environment_name(name)
    }
    if strip_reasoning:
        for name in _REASONING_ENV_NAMES:
            safe.pop(name, None)
    selected = environ.get(key_name)
    if isinstance(selected, str) and selected:
        safe[key_name] = selected
    return safe


def _provider_for_environment(environ: Mapping[str, str]) -> Provider:
    """Resolve the configured provider without ever inspecting a key value."""

    raw = environ.get("CODING_AGENT_PROVIDER", Provider.GLM.value)
    try:
        return Provider(str(raw).strip().lower())
    except ValueError:
        # The plugin will emit the detailed configuration error. Keeping the
        # planner deterministic here lets it still produce a safe dry-run.
        return Provider.GLM


def _opencode_model(model: str) -> str:
    """Add Harbor/OpenCode's routing prefix while preserving the model ID."""

    # OpenCode requires ``provider/model``. A caller that already supplied a
    # provider-qualified identifier owns that routing choice; otherwise the
    # OpenAI-compatible provider is the neutral path for custom/GLM gateways.
    return model if "/" in model else f"openai/{model}"


def _opencode_model_for_route(model: str, environ: Mapping[str, str]) -> str:
    """Select an OpenCode provider ID compatible with a fixed model route."""

    if model == MODEL_IDS[1] and _provider_for_environment(environ) is Provider.GLM:
        return f"{OPENCODE_GLM_PROVIDER}/{model}"
    return _opencode_model(model)


def _fixed_model_id_from_agent_model(value: str) -> str:
    """Return the fixed route ID represented by a Harbor agent model value.

    OpenCode model names include their provider prefix. The formal runner must
    strip both prefixes it emits, otherwise the Zhipu command is treated as a
    custom model and Harbor does not receive ``ZAI_API_KEY``.
    """

    candidate = value.strip()
    for model in MODEL_IDS:
        if candidate == model:
            return model
        if candidate in (f"openai/{model}", f"{OPENCODE_GLM_PROVIDER}/{model}"):
            return model
    raise ValueError(
        f"agent model is not one of the fixed experiment routes: {value!r}"
    )


def _opencode_kwargs_for_route(
    model: str, environ: Mapping[str, str]
) -> dict[str, Any]:
    """Return a secret-free OpenCode config for fixed provider routes.

    OpenCode otherwise chooses its built-in ``gpt-nano`` family for the
    implicit title agent.  That auxiliary model is not guaranteed to exist on
    a private gateway (the relay used by the GPT route does not advertise it),
    so pin the small/title model to the same route as the primary model.  This
    avoids an unrelated request and keeps the complete session on the model
    being evaluated.
    """

    if model not in MODEL_ROUTES:
        return {}
    provider = _provider_for_environment(environ)
    model_ref = _opencode_model_for_route(model, environ)
    config: dict[str, Any] = {"small_model": model_ref}
    if model == MODEL_IDS[1] and provider is Provider.GLM:
        base_url = (
            environ.get("CODING_AGENT_BASE_URL") or PROVIDER_BASE_URLS[Provider.GLM]
        )
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("GLM OpenCode route requires a non-empty base URL")
        config["provider"] = {
            OPENCODE_GLM_PROVIDER: {
                "npm": "@ai-sdk/openai-compatible",
                "env": ["OPENAI_API_KEY"],
                "options": {"baseURL": base_url.strip()},
                "models": {model: {}},
            }
        }
    return {"opencode_config": config}


def _opencode_agent_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Build template-only env assignments for Harbor's OpenCode container."""

    provider = _provider_for_environment(environ)
    _model_name, key_name = required_credential_names(dict(environ))
    # Harbor resolves ${VAR} templates in the host process immediately before
    # launching OpenCode. The command therefore contains names, never values.
    result = {"OPENAI_API_KEY": f"${{{key_name}}}"}
    configured_base = environ.get("CODING_AGENT_BASE_URL")
    preset_base = PROVIDER_BASE_URLS.get(provider)
    # Fixed official routes use the public literal endpoint.  A template is
    # needed only for a custom gateway or when the caller intentionally
    # overrides a provider preset; this keeps the command self-describing and
    # avoids requiring an otherwise unnecessary host environment variable.
    if provider is Provider.CUSTOM or (
        configured_base and configured_base.rstrip("/") != (preset_base or "")
    ):
        result["OPENAI_BASE_URL"] = "${CODING_AGENT_BASE_URL}"
    else:
        # Provider presets are public URLs and do not need a secret-bearing
        # template. OpenCode still receives the same endpoint as the plugin.
        result["OPENAI_BASE_URL"] = preset_base or configured_base or ""
    return result


def route_environment(
    model: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve one of the three fixed model routes using named credentials."""

    source = dict(os.environ if environ is None else environ)
    if model not in MODEL_ROUTES:
        # Custom model IDs remain supported by the legacy one-model planner.
        return source
    route = MODEL_ROUTES[model]
    result = dict(source)
    result["CODING_AGENT_PROVIDER"] = route["provider"]
    result["CODING_AGENT_KEY_ENV"] = route["key_env"]
    if "base_url" in route:
        result["CODING_AGENT_BASE_URL"] = route["base_url"]
    result["CODING_AGENT_MODEL"] = model
    return result


def _route_without_reasoning(
    model: str,
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Resolve a fixed route while removing process-wide reasoning defaults.

    Formal and smoke jobs carry their capability decision as explicit agent
    kwargs.  Removing inherited reasoning variables here prevents a stale
    shell setting from re-enabling a field after a probe reported it as
    unsupported (or after a probe failed).
    """

    route = route_environment(model, environ)
    for name in _REASONING_ENV_NAMES:
        route.pop(name, None)
    return route


def _reasoning_kwargs_for_model(
    model: str,
    *,
    requested_effort: str | None,
    capabilities: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convert one probe result into scalar Harbor agent kwargs.

    Harbor's ``--agent-kwarg`` accepts only ``key=value`` scalars, so the
    complete capability object is retained in the report while the command
    receives only the fields needed by ``CourseCodingAgent``.  Unsupported
    capabilities intentionally receive no reasoning value at all.
    """

    capability = capabilities.get(model) if capabilities is not None else None
    if capability is None:
        return {"reasoning_effort": requested_effort} if requested_effort else {}
    status = str(capability.get("status", "")).strip().lower()
    if status == "supported":
        value = capability.get("accepted_value")
        if value is None:
            value = capability.get("requested_effort") or requested_effort
        parameter = capability.get("parameter") or "reasoning_effort"
        if (
            isinstance(parameter, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", parameter)
            and _json_reasoning_value_is_valid(value)
        ):
            # Native effort levels are normally strings.  Keep that compact
            # spelling for the common case, but carry booleans/numbers/objects
            # (and provider-specific strings such as ``enabled``) through a
            # dedicated JSON-aware kwarg so Harbor does not turn them into a
            # lossy Python repr or reject a non-standard value as a generic
            # reasoning-effort enum.
            kwargs: dict[str, Any] = {
                "reasoning_parameter": parameter,
                "reasoning_capability_status": "supported",
            }
            standard_levels = {"low", "medium", "high", "max"}
            if isinstance(value, str) and value.strip().lower() in standard_levels:
                kwargs["reasoning_effort"] = value.strip().lower()
            else:
                fallback_effort = (
                    capability.get("requested_effort") or requested_effort or "high"
                )
                if (
                    not isinstance(fallback_effort, str)
                    or fallback_effort.strip().lower() not in standard_levels
                ):
                    return {"reasoning_capability_status": "error"}
                kwargs["reasoning_effort"] = fallback_effort.strip().lower()
                kwargs["reasoning_value"] = value
            return kwargs
        # A malformed probe result must fail closed rather than silently
        # selecting an arbitrary native spelling.
        return {"reasoning_capability_status": "error"}
    if status == "unsupported":
        return {"reasoning_capability_status": "unsupported"}
    if status in {"error", "setup_failure"}:
        return {"reasoning_capability_status": "error"}
    # ``not_run`` is used only by dry plans. Keep the requested value visible
    # there, while an actual execution always probes before launching jobs.
    if status == "not_run":
        return {"reasoning_effort": requested_effort} if requested_effort else {}
    return {"reasoning_capability_status": "error"}


def _json_reasoning_value_is_valid(value: Any) -> bool:
    """Return whether a probe value can cross Harbor's JSON kwarg boundary."""

    if value is None:
        return False
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def _capability_map(
    probes: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Mapping[str, Any]]:
    """Index safe probe records by model ID for command construction."""

    result: dict[str, Mapping[str, Any]] = {}
    for probe in probes or ():
        model = probe.get("model")
        if isinstance(model, str) and model in MODEL_ROUTES:
            result[model] = probe
    return result


def _resolve_round_strategy(
    strategy: str | None,
    *,
    max_model_turns: int | None = None,
    efficiency_mode: bool | None = None,
    reserve_final_turn: bool | None = None,
) -> tuple[str, int, bool, bool]:
    """Resolve a named round strategy into the three runtime knobs.

    The explicit knobs remain accepted for backwards compatibility with the
    original planner API.  When a named strategy is supplied it is authoritative
    and conflicting overrides are rejected, preventing a report from claiming
    one strategy while launching another configuration.
    """

    if strategy is not None:
        if strategy not in ROUND_STRATEGIES:
            choices = ", ".join(ROUND_STRATEGIES)
            raise ValueError(f"unknown round strategy {strategy!r}; choose {choices}")
        selected = ROUND_STRATEGIES[strategy]
        expected_turns = int(selected["max_model_turns"])
        expected_efficiency = bool(selected["efficiency_mode"])
        expected_reserve = bool(selected["reserve_final_turn"])
        if max_model_turns is not None and max_model_turns != expected_turns:
            raise ValueError(
                f"round strategy {strategy!r} requires max_model_turns={expected_turns}"
            )
        if efficiency_mode is not None and efficiency_mode != expected_efficiency:
            raise ValueError(
                f"round strategy {strategy!r} requires efficiency_mode="
                f"{expected_efficiency}"
            )
        if reserve_final_turn is not None and reserve_final_turn != expected_reserve:
            raise ValueError(
                f"round strategy {strategy!r} requires reserve_final_turn="
                f"{expected_reserve}"
            )
        return strategy, expected_turns, expected_efficiency, expected_reserve

    turns = EFFICIENCY_AGENT_MAX_TURNS if max_model_turns is None else max_model_turns
    enabled = True if efficiency_mode is None else efficiency_mode
    reserve = enabled if reserve_final_turn is None else reserve_final_turn
    if (
        isinstance(turns, bool)
        or not isinstance(turns, int)
        or turns < 1
        or not isinstance(enabled, bool)
        or not isinstance(reserve, bool)
    ):
        raise ValueError(
            "max_model_turns must be a positive integer and efficiency/reserve "
            "flags must be booleans"
        )
    inferred = next(
        (
            name
            for name, values in ROUND_STRATEGIES.items()
            if int(values["max_model_turns"]) == turns
            and bool(values["efficiency_mode"]) == enabled
            and bool(values["reserve_final_turn"]) == reserve
        ),
        "custom",
    )
    return inferred, turns, enabled, reserve


def _invoke_probe_factory(factory: Any, values: Mapping[str, Any]) -> Any:
    """Call an injected probe factory without hiding its internal errors."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**dict(values))
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return factory(**dict(values))
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return factory(**{key: value for key, value in values.items() if key in accepted})


def fixed_model_routes(
    environ: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], ...]:
    """Return redacted route metadata in the formal experiment order."""

    routes: list[dict[str, str]] = []
    for model in MODEL_IDS:
        env = route_environment(model, environ)
        route = dict(MODEL_ROUTES[model])
        route["model"] = model
        if "base_url" not in route:
            route["base_url"] = env["CODING_AGENT_BASE_URL"]
        routes.append(route)
    return tuple(routes)


def _model_endpoint_host(environ: Mapping[str, str]) -> str | None:
    """Return the configured model host for Harbor's agent allow-list."""

    provider = _provider_for_environment(environ)
    raw_base = environ.get("CODING_AGENT_BASE_URL")
    if not isinstance(raw_base, str) or not raw_base.strip():
        raw_base = PROVIDER_BASE_URLS.get(provider)
    if not isinstance(raw_base, str):
        return None
    try:
        parsed = urlsplit(raw_base.strip())
    except ValueError:
        return None
    host = parsed.hostname
    return host.lower().rstrip(".") if host else None


def _experiment_commands(
    *,
    model: str,
    dataset: str,
    jobs_dir: str | os.PathLike[str],
    harbor_bin: str,
    environ: Mapping[str, str],
    harbor_sudo: bool = False,
    reasoning_effort: str | None = None,
    reasoning_parameter: str | None = None,
    reasoning_capability_status: str | None = None,
) -> tuple[ExperimentCommand, ExperimentCommand]:
    """Construct the two commands with equivalent model routing."""

    route = route_environment(model, environ)
    endpoint_host = _model_endpoint_host(route)
    allow_hosts = (endpoint_host,) if endpoint_host else ()
    course_kwargs: dict[str, Any] = {}
    if reasoning_effort is not None:
        course_kwargs["reasoning_effort"] = reasoning_effort
    if reasoning_parameter is not None:
        course_kwargs["reasoning_parameter"] = reasoning_parameter
    if reasoning_capability_status is not None:
        course_kwargs["reasoning_capability_status"] = reasoning_capability_status

    return (
        build_command(
            agent=COURSE_AGENT,
            model=model,
            dataset=dataset,
            jobs_dir=jobs_dir,
            harbor_bin=harbor_bin,
            harbor_sudo=harbor_sudo,
            allow_agent_hosts=allow_hosts,
            agent_kwargs=course_kwargs,
        ),
        build_command(
            agent=PINNED_OPENCODE_AGENT,
            model=_opencode_model_for_route(model, route),
            dataset=dataset,
            jobs_dir=jobs_dir,
            harbor_bin=harbor_bin,
            harbor_sudo=harbor_sudo,
            agent_env=_opencode_agent_env(route),
            allow_agent_hosts=allow_hosts,
            agent_kwargs=_opencode_kwargs_for_route(model, route),
        ),
    )


def formal_experiment_commands(
    *,
    dataset: str,
    jobs_dir: str | os.PathLike[str] = ".harbor-runs",
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: Mapping[str, str] | None = None,
    repetitions: int = FORMAL_REPETITIONS,
    course_agent: str = COURSE_AGENT,
    opencode_agent: str = PINNED_OPENCODE_AGENT,
    max_model_turns: int | None = None,
    efficiency_mode: bool | None = None,
    reserve_final_turn: bool | None = None,
    round_strategy: str | None = None,
    reasoning_effort: str | None = "high",
    reasoning_capabilities: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[ExperimentCommand, ...]:
    """Build the six fixed model/agent commands for the formal matrix.

    ``round_strategy`` is the auditable source of truth for Course-agent turn
    policy.  The older individual knobs are still available for callers that
    need a custom policy, by passing ``round_strategy=None``.
    """

    # The formal matrix is a fixed comparison.  Keep the injectable arguments
    # for callers that used the early planner API, but fail closed when they
    # would replace either condition with an unpinned implementation.  Smoke,
    # ablation, and formal reports all rely on these exact identities when
    # validating rows, so silently accepting an override would invalidate the
    # experiment while still producing plausible-looking output.
    if course_agent != COURSE_AGENT:
        raise ValueError(
            f"formal experiment requires the fixed Course agent {COURSE_AGENT!r}"
        )
    if opencode_agent != PINNED_OPENCODE_AGENT:
        raise ValueError(
            "formal experiment requires the pinned OpenCode agent "
            f"{PINNED_OPENCODE_AGENT!r}"
        )
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    _strategy_name, resolved_turns, resolved_efficiency, resolved_reserve = (
        _resolve_round_strategy(
            round_strategy,
            max_model_turns=max_model_turns,
            efficiency_mode=efficiency_mode,
            reserve_final_turn=reserve_final_turn,
        )
    )
    commands: list[ExperimentCommand] = []
    source = dict(os.environ if environ is None else environ)
    capability_map = reasoning_capabilities or {}
    for model in MODEL_IDS:
        route = _route_without_reasoning(model, source)
        endpoint_host = _model_endpoint_host(route)
        allow_hosts = (endpoint_host,) if endpoint_host else ()
        route_capability = _reasoning_kwargs_for_model(
            model,
            requested_effort=reasoning_effort,
            capabilities=capability_map,
        )
        course_kwargs: dict[str, Any] = {
            "max_model_turns": resolved_turns,
            "max_tool_calls": AGENT_MAX_TOOL_CALLS,
            "max_wall_time_seconds": AGENT_TIMEOUT_SECONDS,
            "efficiency_mode": resolved_efficiency,
            "reserve_final_turn": resolved_reserve,
        }
        course_kwargs.update(route_capability)
        commands.append(
            build_command(
                agent=course_agent,
                model=model,
                dataset=dataset,
                jobs_dir=Path(jobs_dir) / model.replace("/", "_"),
                harbor_bin=harbor_bin,
                harbor_sudo=harbor_sudo,
                allow_agent_hosts=allow_hosts,
                repetitions=repetitions,
                agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
                agent_kwargs=course_kwargs,
            )
        )
        opencode_env = _opencode_agent_env(route)
        commands.append(
            build_command(
                agent=opencode_agent,
                model=_opencode_model_for_route(model, route),
                dataset=dataset,
                jobs_dir=Path(jobs_dir) / model.replace("/", "_") / "opencode",
                harbor_bin=harbor_bin,
                harbor_sudo=harbor_sudo,
                agent_env=opencode_env,
                allow_agent_hosts=allow_hosts,
                repetitions=repetitions,
                agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
                agent_kwargs=_opencode_kwargs_for_route(model, route),
            )
        )
    return tuple(commands)


def make_formal_plan(
    *,
    dataset: str = DATASET,
    jobs_dir: str | os.PathLike[str] = ".harbor-runs",
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: Mapping[str, str] | None = None,
    repetitions: int = FORMAL_REPETITIONS,
    max_model_turns: int | None = None,
    efficiency_mode: bool | None = None,
    reserve_final_turn: bool | None = None,
    round_strategy: str | None = None,
    reasoning_effort: str | None = "high",
    reasoning_capabilities: Mapping[str, Mapping[str, Any]] | None = None,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return the complete 3-model x 2-agent x 8-task x 3-repeat plan."""

    source = os.environ if environ is None else environ
    validated_manifest = _image_manifest_for_use(image_manifest, dataset=dataset)
    strategy_name, resolved_turns, resolved_efficiency, resolved_reserve = (
        _resolve_round_strategy(
            round_strategy,
            max_model_turns=max_model_turns,
            efficiency_mode=efficiency_mode,
            reserve_final_turn=reserve_final_turn,
        )
    )
    commands = formal_experiment_commands(
        dataset=dataset,
        jobs_dir=jobs_dir,
        harbor_bin=harbor_bin,
        harbor_sudo=harbor_sudo,
        environ=source,
        repetitions=repetitions,
        max_model_turns=resolved_turns,
        efficiency_mode=resolved_efficiency,
        reserve_final_turn=resolved_reserve,
        round_strategy=strategy_name if strategy_name in ROUND_STRATEGIES else None,
        reasoning_effort=reasoning_effort,
        reasoning_capabilities=reasoning_capabilities,
    )
    capability_records = [
        dict(reasoning_capabilities[model])
        for model in MODEL_IDS
        if reasoning_capabilities is not None and model in reasoning_capabilities
    ]
    return {
        "schema_version": "terminal-bench-experiment/v2",
        "label": "8-task Terminal-Bench 2.1 exploratory experiment",
        "scope_note": "8 个 Terminal-Bench 2.1 任务、每项 3 次的探索性实验",
        "dataset": dataset,
        "dataset_reference": _dataset_reference(dataset),
        "harbor_bin": harbor_bin,
        "harbor_sudo": harbor_sudo,
        "harbor_invocation_prefix": list(
            harbor_invocation_prefix(harbor_bin, harbor_sudo=harbor_sudo)
        ),
        "task_count": len(TASK_IDS),
        "tasks": list(TASK_IDS),
        "task_filters": _task_filters(dataset),
        "models": list(MODEL_IDS),
        "agents": [COURSE_AGENT, PINNED_OPENCODE_AGENT],
        "repetitions_per_task": repetitions,
        "expected_trials": len(MODEL_IDS) * 2 * len(TASK_IDS) * repetitions,
        "routes": list(fixed_model_routes(source)),
        "reasoning_effort": reasoning_effort,
        "reasoning_probes": capability_records,
        "course_round_strategy": strategy_name,
        "runtime_limits": {
            "course_max_model_turns": resolved_turns,
            "course_max_tool_calls": AGENT_MAX_TOOL_CALLS,
            "agent_timeout_seconds": AGENT_TIMEOUT_SECONDS,
            "n_concurrent": 1,
            "max_retries": 0,
            "course_efficiency_mode": resolved_efficiency,
            "course_reserve_final_turn": resolved_reserve,
        },
        "opencode": {
            "version": PINNED_OPENCODE_VERSION,
            "agent": PINNED_OPENCODE_AGENT,
            "setup_policy": "pre-installed; version check only; no nvm/npm/@latest",
        },
        "image_manifest": _image_manifest_public(validated_manifest),
        "commands": [list(item.argv) for item in commands],
        "status": "planned",
    }


def make_smoke_plan(
    *,
    dataset: str = DATASET,
    jobs_dir: str | os.PathLike[str] = ".harbor-runs/smoke",
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: Mapping[str, str] | None = None,
    task: str = "fix-code-vulnerability",
    reasoning_effort: str | None = "high",
    reasoning_capabilities: Mapping[str, Mapping[str, Any]] | None = None,
    max_model_turns: int = EFFICIENCY_AGENT_MAX_TURNS,
    efficiency_mode: bool = True,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Plan one smoke trial for every model/agent combination."""

    if task not in TASK_IDS:
        raise ValueError(f"unknown smoke task: {task}")
    source = dict(os.environ if environ is None else environ)
    validated_manifest = _image_manifest_for_use(image_manifest, dataset=dataset)
    commands: list[list[str]] = []
    for model in MODEL_IDS:
        route = _route_without_reasoning(model, source)
        endpoint_host = _model_endpoint_host(route)
        allow_hosts = (endpoint_host,) if endpoint_host else ()
        course_kwargs: dict[str, Any] = {
            "max_model_turns": max_model_turns,
            "max_tool_calls": AGENT_MAX_TOOL_CALLS,
            "max_wall_time_seconds": AGENT_TIMEOUT_SECONDS,
            "efficiency_mode": efficiency_mode,
            "reserve_final_turn": efficiency_mode,
        }
        course_kwargs.update(
            _reasoning_kwargs_for_model(
                model,
                requested_effort=reasoning_effort,
                capabilities=reasoning_capabilities,
            )
        )
        commands.append(
            list(
                build_command(
                    agent=COURSE_AGENT,
                    model=model,
                    dataset=dataset,
                    jobs_dir=Path(jobs_dir) / model / "course",
                    harbor_bin=harbor_bin,
                    harbor_sudo=harbor_sudo,
                    allow_agent_hosts=allow_hosts,
                    task_ids=(task,),
                    repetitions=1,
                    agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
                    agent_kwargs=course_kwargs,
                ).argv
            )
        )
        commands.append(
            list(
                build_command(
                    agent=PINNED_OPENCODE_AGENT,
                    model=_opencode_model_for_route(model, route),
                    dataset=dataset,
                    jobs_dir=Path(jobs_dir) / model / "opencode",
                    harbor_bin=harbor_bin,
                    harbor_sudo=harbor_sudo,
                    agent_env=_opencode_agent_env(route),
                    allow_agent_hosts=allow_hosts,
                    task_ids=(task,),
                    repetitions=1,
                    agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
                    agent_kwargs=_opencode_kwargs_for_route(model, route),
                ).argv
            )
        )
    return {
        "schema_version": "terminal-bench-smoke/v1",
        "task": task,
        "harbor_bin": harbor_bin,
        "harbor_sudo": harbor_sudo,
        "harbor_invocation_prefix": list(
            harbor_invocation_prefix(harbor_bin, harbor_sudo=harbor_sudo)
        ),
        "models": list(MODEL_IDS),
        "agents": [COURSE_AGENT, PINNED_OPENCODE_AGENT],
        "expected_trials": len(MODEL_IDS) * 2,
        "reasoning_effort": reasoning_effort,
        "reasoning_probes": [
            dict(reasoning_capabilities[model])
            for model in MODEL_IDS
            if reasoning_capabilities is not None and model in reasoning_capabilities
        ],
        "runtime_limits": {
            "course_max_model_turns": max_model_turns,
            "course_max_tool_calls": AGENT_MAX_TOOL_CALLS,
            "agent_timeout_seconds": AGENT_TIMEOUT_SECONDS,
            "efficiency_mode": efficiency_mode,
        },
        "commands": commands,
        "image_manifest": _image_manifest_public(validated_manifest),
        "status": "planned",
    }


def make_ablation_plan(
    *,
    dataset: str = DATASET,
    jobs_dir: str | os.PathLike[str] = ".harbor-runs/ablation",
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: Mapping[str, str] | None = None,
    repetitions: int = FORMAL_REPETITIONS,
    tasks: Sequence[str] = ("build-cython-ext", "write-compressor"),
    reasoning_effort: str | None = "high",
    reasoning_capabilities: Mapping[str, Mapping[str, Any]] | None = None,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Plan the three Course-agent round strategies on the two failed tasks."""

    selected_tasks = tuple(tasks)
    if not selected_tasks or any(task not in TASK_IDS for task in selected_tasks):
        raise ValueError("ablation tasks must be selected Terminal-Bench task IDs")
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    source = dict(os.environ if environ is None else environ)
    validated_manifest = _image_manifest_for_use(image_manifest, dataset=dataset)
    capability_map = reasoning_capabilities or {}
    strategies = tuple(
        {
            "name": name,
            "max_model_turns": int(values["max_model_turns"]),
            "efficiency_mode": bool(values["efficiency_mode"]),
            "reserve_final_turn": bool(values["reserve_final_turn"]),
        }
        for name, values in ROUND_STRATEGIES.items()
    )
    commands: list[dict[str, Any]] = []
    for model in MODEL_IDS:
        route = _route_without_reasoning(model, source)
        endpoint_host = _model_endpoint_host(route)
        allow_hosts = (endpoint_host,) if endpoint_host else ()
        for strategy in strategies:
            kwargs: dict[str, Any] = {
                "max_model_turns": strategy["max_model_turns"],
                "max_tool_calls": AGENT_MAX_TOOL_CALLS,
                "max_wall_time_seconds": AGENT_TIMEOUT_SECONDS,
                "efficiency_mode": strategy["efficiency_mode"],
                "reserve_final_turn": strategy["reserve_final_turn"],
            }
            kwargs.update(
                _reasoning_kwargs_for_model(
                    model,
                    requested_effort=reasoning_effort,
                    capabilities=capability_map,
                )
            )
            command = build_command(
                agent=COURSE_AGENT,
                model=model,
                dataset=dataset,
                jobs_dir=Path(jobs_dir) / model / str(strategy["name"]),
                harbor_bin=harbor_bin,
                harbor_sudo=harbor_sudo,
                allow_agent_hosts=allow_hosts,
                repetitions=repetitions,
                task_ids=selected_tasks,
                agent_timeout_seconds=AGENT_TIMEOUT_SECONDS,
                agent_kwargs=kwargs,
            )
            commands.append(
                {
                    "model": model,
                    "strategy": strategy["name"],
                    "tasks": list(selected_tasks),
                    "argv": list(command.argv),
                }
            )
    return {
        "schema_version": "terminal-bench-ablation/v1",
        "label": "Course Agent round-budget ablation",
        "harbor_bin": harbor_bin,
        "harbor_sudo": harbor_sudo,
        "harbor_invocation_prefix": list(
            harbor_invocation_prefix(harbor_bin, harbor_sudo=harbor_sudo)
        ),
        "tasks": list(selected_tasks),
        "models": list(MODEL_IDS),
        "strategies": [str(item["name"]) for item in strategies],
        "strategy_configs": [dict(item) for item in strategies],
        "repetitions_per_task": repetitions,
        "reasoning_effort": reasoning_effort,
        "reasoning_probes": [
            dict(reasoning_capabilities[model])
            for model in MODEL_IDS
            if reasoning_capabilities is not None and model in reasoning_capabilities
        ],
        "expected_trials": len(MODEL_IDS)
        * len(strategies)
        * len(selected_tasks)
        * repetitions,
        "commands": commands,
        "image_manifest": _image_manifest_public(validated_manifest),
        "status": "planned",
    }


def _aggregate_ablation_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    models: Sequence[str] = MODEL_IDS,
    strategies: Sequence[str] = ("current_20", "efficiency_20", "efficiency_30"),
    tasks: Sequence[str] = ("build-cython-ext", "write-compressor"),
    repetitions: int = FORMAL_REPETITIONS,
) -> dict[str, dict[str, Any]]:
    """Summarize Course-agent round-strategy rows by model and task."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "trial":
            continue
        model = row.get("ablation_model") or row.get("model")
        strategy = row.get("ablation_strategy") or row.get("strategy")
        task = row.get("task_name")
        if not isinstance(model, str) or model not in models:
            continue
        if not isinstance(strategy, str) or strategy not in strategies:
            continue
        task_name = _canonical_task_name(task, tasks)
        if task_name is None:
            continue
        grouped.setdefault((model, strategy, task_name), []).append(row)

    result: dict[str, dict[str, Any]] = {}
    expected_per_group = len(tasks) * repetitions
    for model in models:
        for strategy in strategies:
            for task in tasks:
                # Harbor 0.22 expands ``n_attempts`` into separate trial
                # directories but does not persist the repetition ordinal in
                # each trial result.  Work on copies and assign the missing
                # ordinals from a stable path/name ordering so a complete run
                # is not mistaken for an incomplete matrix.  Explicit
                # ordinals remain authoritative and still surface duplicate
                # or out-of-range data below.
                group = [dict(row) for row in grouped.get((model, strategy, task), [])]
                used_repetitions = {
                    row["repetition"]
                    for row in group
                    if type(row.get("repetition")) is int
                }
                available_repetitions = [
                    number
                    for number in range(1, repetitions + 1)
                    if number not in used_repetitions
                ]
                for row in sorted(
                    (item for item in group if item.get("repetition") is None),
                    key=_matrix_sort_key,
                ):
                    if available_repetitions:
                        row["repetition"] = available_repetitions.pop(0)
                    else:
                        # Keep extras visible as an invalid ordinal instead of
                        # silently dropping them from duplicate detection.
                        row["repetition"] = repetitions + 1
                setup = [row for row in group if _is_setup_failure(row)]
                evaluable = [
                    row
                    for row in group
                    if row not in setup and type(row.get("passed")) is bool
                ]
                repetition_values = [
                    row.get("repetition")
                    for row in group
                    if type(row.get("repetition")) is int
                ]
                repetition_counts = {
                    value: repetition_values.count(value)
                    for value in set(repetition_values)
                    if repetition_values.count(value) > 1
                }
                invalid_repetitions = sorted(
                    {
                        value
                        for value in repetition_values
                        if value < 1 or value > repetitions
                    }
                )
                observed_repetitions = {
                    value for value in repetition_values if 1 <= value <= repetitions
                }
                missing_repetitions = sorted(
                    set(range(1, repetitions + 1)) - observed_repetitions
                )
                source_identities: dict[str, int] = {}
                for row in group:
                    identity = _row_matrix_identity(row)
                    if identity is not None:
                        source_identities[identity] = (
                            source_identities.get(identity, 0) + 1
                        )
                duplicate_identities = {
                    identity: count
                    for identity, count in source_identities.items()
                    if count > 1
                }
                passed = sum(row.get("passed") is True for row in evaluable)
                key = f"{model}::{strategy}::{task}"
                result[key] = {
                    "model": model,
                    "strategy": strategy,
                    "task": task,
                    "expected_trials": repetitions,
                    "n_trials": len(group),
                    "n_evaluable_trials": len(evaluable),
                    "n_passed": passed,
                    "n_failed": sum(row.get("passed") is False for row in evaluable),
                    "n_setup_failures": len(setup),
                    "n_unresolved_trials": len(group) - len(setup) - len(evaluable),
                    "accuracy": passed / len(evaluable) if evaluable else None,
                    "duplicate_repetitions": repetition_counts,
                    "invalid_repetitions": invalid_repetitions,
                    "missing_repetitions": missing_repetitions,
                    "duplicate_identities": duplicate_identities,
                    "complete": (
                        len(group) == repetitions
                        and len(evaluable) == repetitions
                        and not setup
                        and not repetition_counts
                        and not invalid_repetitions
                        and not missing_repetitions
                        and not duplicate_identities
                    ),
                }
    # Add model/strategy totals for convenient report consumption.
    for model in models:
        for strategy in strategies:
            selected = [
                item
                for item in result.values()
                if item["model"] == model and item["strategy"] == strategy
            ]
            key = f"{model}::{strategy}"
            n_trials = sum(int(item["n_trials"]) for item in selected)
            n_evaluable = sum(int(item["n_evaluable_trials"]) for item in selected)
            n_passed = sum(int(item["n_passed"]) for item in selected)
            result[key] = {
                "model": model,
                "strategy": strategy,
                "expected_trials": expected_per_group,
                "n_trials": n_trials,
                "n_evaluable_trials": n_evaluable,
                "n_passed": n_passed,
                "n_failed": sum(int(item["n_failed"]) for item in selected),
                "n_setup_failures": sum(
                    int(item["n_setup_failures"]) for item in selected
                ),
                "n_unresolved_trials": sum(
                    int(item["n_unresolved_trials"]) for item in selected
                ),
                "accuracy": n_passed / n_evaluable if n_evaluable else None,
                "duplicate_repetitions": {
                    key: value
                    for item in selected
                    for key, value in (item.get("duplicate_repetitions") or {}).items()
                },
                "invalid_repetitions": {
                    str(item["task"]): list(item.get("invalid_repetitions") or [])
                    for item in selected
                    if item.get("invalid_repetitions")
                },
                "missing_repetitions": {
                    str(item["task"]): list(item.get("missing_repetitions") or [])
                    for item in selected
                    if item.get("missing_repetitions")
                },
                "duplicate_identities": {
                    key: value
                    for item in selected
                    for key, value in (item.get("duplicate_identities") or {}).items()
                },
                "complete": all(bool(item["complete"]) for item in selected),
            }
    return result


def choose_formal_round_strategy(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    models: Sequence[str] = MODEL_IDS,
    tasks: Sequence[str] = ("build-cython-ext", "write-compressor"),
) -> dict[str, Any]:
    """Choose the formal Course strategy from the completed ablation rows.

    Efficiency-20 is retained only when every model/task comparison is
    complete and its pass count is at least the corresponding efficiency-30
    count.  Any missing or setup-tainted comparison conservatively selects the
    30-turn configuration and is marked ``incomplete``.
    """

    comparisons: list[dict[str, Any]] = []
    all_equal_or_better = True
    all_complete = True
    for model in models:
        for task in tasks:
            short_20 = summary.get(f"{model}::efficiency_20::{task}", {})
            long_30 = summary.get(f"{model}::efficiency_30::{task}", {})
            complete = (
                short_20.get("complete") is True and long_30.get("complete") is True
            )
            passed_20 = short_20.get("n_passed")
            passed_30 = long_30.get("n_passed")
            comparable = (
                complete
                and type(passed_20) is int
                and type(passed_30) is int
                and passed_20 >= 0
                and passed_30 >= 0
            )
            if not comparable:
                all_complete = False
                all_equal_or_better = False
            elif passed_20 < passed_30:
                all_equal_or_better = False
            comparisons.append(
                {
                    "model": model,
                    "task": task,
                    "efficiency_20_passed": passed_20,
                    "efficiency_30_passed": passed_30,
                    "comparable": comparable,
                    "efficiency_20_at_least_30": (
                        passed_20 >= passed_30 if comparable else None
                    ),
                }
            )
    selected = "efficiency_20" if all_equal_or_better else "efficiency_30"
    return {
        "selected_strategy": selected,
        "selected_max_model_turns": int(ROUND_STRATEGIES[selected]["max_model_turns"]),
        "selected_efficiency_mode": bool(ROUND_STRATEGIES[selected]["efficiency_mode"]),
        "status": "complete" if all_complete else "incomplete",
        "reason": (
            "efficiency_20 matched or exceeded efficiency_30 for every complete "
            "model/task comparison"
            if selected == "efficiency_20"
            else "efficiency_20 did not match efficiency_30 for every complete "
            "model/task comparison"
        ),
        "comparisons": comparisons,
    }


def formal_strategy_from_ablation(
    report: Mapping[str, Any] | str | os.PathLike[str],
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Load and validate the fixed Course strategy from an ablation report.

    Accepting both an in-memory report and a JSON path keeps the experiment
    runner composable while ensuring the formal command uses the exact choice
    made from the completed matrix.  Only the small decision object is returned;
    arbitrary report fields are never copied into a Harbor command.
    """

    document: Mapping[str, Any]
    if isinstance(report, Mapping):
        document = report
    else:
        path = Path(report).expanduser().resolve(strict=False)
        try:
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("ablation report is not readable JSON") from exc
        if not isinstance(loaded, Mapping):
            raise TypeError("ablation report must contain a JSON object")
        document = loaded

    raw_decision = document.get("fixed_formal_configuration")
    if isinstance(raw_decision, Mapping):
        decision = dict(raw_decision)
    else:
        # ``write_ablation_artifacts`` intentionally uses the shorter
        # ``summary`` key. Accept both forms so the artifact can be passed
        # directly to the formal runner without hand-editing it.
        summary = document.get("ablation_summary")
        if summary is None:
            summary = document.get("summary")
        if not isinstance(summary, Mapping):
            raise TypeError(
                "ablation report has no fixed_formal_configuration or summary"
            )
        models = _ablation_report_dimension(
            document,
            summary,
            field="models",
            item_field="model",
            fallback=MODEL_IDS,
        )
        tasks = _ablation_report_dimension(
            document,
            summary,
            field="tasks",
            item_field="task",
            fallback=("build-cython-ext", "write-compressor"),
            basename=True,
        )
        decision = choose_formal_round_strategy(
            summary,
            models=models,
            tasks=tasks,
        )

    selected = decision.get("selected_strategy")
    if selected not in {"efficiency_20", "efficiency_30"}:
        raise ValueError(
            "ablation report selected a strategy that is not a formal efficiency strategy"
        )
    complete = decision.get("status") == "complete"
    # Archived artifacts carry this independent structural flag.  Treat a
    # contradiction as incomplete rather than trusting a stale or hand-edited
    # decision object; older in-memory callers may omit the field entirely.
    declared_complete = document.get("matrix_complete")
    if declared_complete is not None:
        if not isinstance(declared_complete, bool):
            raise ValueError("ablation report matrix_complete must be a boolean")
        complete = complete and declared_complete
    if not complete and not allow_incomplete:
        raise ValueError("ablation report is incomplete; formal strategy is not fixed")
    # Return a compact, JSON-safe copy with values recomputed from the canonical
    # strategy table rather than trusting potentially stale report fields.
    values = ROUND_STRATEGIES[selected]
    return {
        "selected_strategy": selected,
        "selected_max_model_turns": int(values["max_model_turns"]),
        "selected_efficiency_mode": bool(values["efficiency_mode"]),
        "selected_reserve_final_turn": bool(values["reserve_final_turn"]),
        "status": "complete" if complete else "incomplete",
        "reason": decision.get("reason"),
        "comparisons": decision.get("comparisons", []),
    }


def _smoke_report_document(
    report: Mapping[str, Any] | str | os.PathLike[str] | None,
) -> tuple[Mapping[str, Any] | None, str | None, str | None]:
    """Load a smoke report without copying arbitrary report payloads.

    The gate only needs a small set of fields, but accepting both an in-memory
    document and a path makes it useful to callers that run the smoke matrix in
    the same process as the formal planner.  Errors are returned as data so the
    caller can include a deterministic blocked-gate record in diagnostics.
    """

    if report is None:
        return None, None, "smoke report was not supplied"
    if isinstance(report, Mapping):
        return report, None, None
    try:
        path = Path(report).expanduser().resolve(strict=False)
    except (TypeError, ValueError, OSError) as exc:
        return None, None, f"smoke report path is invalid ({type(exc).__name__})"
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            None,
            str(path),
            f"smoke report is not readable JSON ({type(exc).__name__})",
        )
    if not isinstance(loaded, Mapping):
        return None, str(path), "smoke report must contain a JSON object"
    return loaded, str(path), None


def _smoke_row_entry(index: int, row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one smoke row into a credential-free gate diagnostic."""

    key = row.get("matrix_key")
    if not isinstance(key, str) or not key:
        parts = (
            row.get("matrix_model") or row.get("model"),
            row.get("matrix_agent") or row.get("agent"),
            row.get("matrix_task") or row.get("task_name"),
            row.get("repetition"),
        )
        if all(isinstance(item, (str, int)) for item in parts):
            key = "|".join(str(item) for item in parts)
        else:
            key = None
    entry: dict[str, Any] = {"row_index": index, "key": key}
    for field in ("phase", "exception_type"):
        value = row.get(field)
        if isinstance(value, str) and value:
            entry[field] = value[:160]
    return entry


def _validate_execution_probes(
    probes: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str] = MODEL_IDS,
) -> None:
    """Fail closed before Harbor when a native reasoning probe is unusable."""

    expected = tuple(models)
    if len(probes) != len(expected):
        raise ValueError(
            "reasoning probe gate blocked: expected one probe per model "
            f"({len(expected)}), observed {len(probes)}"
        )
    seen: set[str] = set()
    invalid: list[str] = []
    for probe in probes:
        model = probe.get("model")
        status = probe.get("status")
        if not isinstance(model, str) or model not in expected:
            invalid.append("unknown-model")
            continue
        if model in seen:
            invalid.append(f"duplicate:{model}")
            continue
        seen.add(model)
        status_text = status.strip().lower() if isinstance(status, str) else status
        if status_text not in {"supported", "unsupported"}:
            invalid.append(f"{model}:{status}")
    missing = sorted(set(expected) - seen)
    if missing:
        invalid.append("missing:" + ",".join(missing))
    if invalid:
        raise ValueError(
            "reasoning probe gate blocked: " + ", ".join(str(item) for item in invalid)
        )


def validate_smoke_report(
    report: Mapping[str, Any] | str | os.PathLike[str] | None,
    *,
    task: str = SMOKE_TASK,
    expected_models: Sequence[str] = MODEL_IDS,
    expected_agents: Sequence[str] = MATRIX_AGENT_LABELS,
    secrets: Sequence[str] = (),
    require_probes: bool = True,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate the pre-formal six-trial smoke report.

    A smoke verifier failure is an evaluated observation and is therefore
    allowed to proceed (while being exposed in ``verifier_failures``).  Setup,
    configuration, infrastructure, malformed, missing, duplicate, unknown,
    and unresolved observations block the formal matrix.  The returned object
    contains only row indices, canonical keys, and bounded status metadata; it
    never copies arbitrary result text or credentials from the source report.
    """

    if task not in TASK_IDS:
        raise ValueError(f"unknown smoke task: {task}")
    if not isinstance(require_probes, bool):
        raise TypeError("require_probes must be a boolean")
    models = tuple(str(item) for item in expected_models)
    agents = tuple(str(item) for item in expected_agents)
    expected_trials = len(models) * len(agents)
    document, report_path, load_error = _smoke_report_document(report)
    blocking_reasons: list[str] = []
    if load_error:
        blocking_reasons.append(load_error)

    report_status: str | None = None
    raw_rows: object = None
    raw_probes: object = None
    raw_outcomes: object = None
    expected_manifest: Mapping[str, Any] | None = None
    if image_manifest is not None:
        expected_manifest = _image_manifest_for_use(image_manifest, dataset=None)
    if document is not None:
        report_status = (
            document.get("status") if isinstance(document.get("status"), str) else None
        )
        if report_status not in {"finished", "complete", "completed"}:
            blocking_reasons.append(
                "smoke report status must be finished, complete, or completed"
            )
        reported_task = document.get("task")
        normalised_task = _canonical_task_name(reported_task, (task,))
        if normalised_task != task:
            blocking_reasons.append(f"smoke report task must be {task}")
        reported_trials = document.get("expected_trials")
        if reported_trials is not None and (
            isinstance(reported_trials, bool)
            or not isinstance(reported_trials, int)
            or reported_trials != expected_trials
        ):
            blocking_reasons.append(
                f"smoke report expected_trials must be {expected_trials}"
            )
        raw_rows = document.get("result_summaries")
        if raw_rows is None:
            # This alias is useful for compact, manually archived reports and
            # keeps the gate independent of the runner's outer metadata.
            raw_rows = document.get("rows")
        raw_probes = document.get("reasoning_probes")
        raw_outcomes = document.get("outcomes")
        if expected_manifest is not None:
            reported_manifest = document.get("image_manifest")
            reported_hash = (
                reported_manifest.get("manifest_sha256")
                if isinstance(reported_manifest, Mapping)
                else None
            )
            if reported_hash != expected_manifest.get("manifest_sha256"):
                blocking_reasons.append(
                    "smoke image manifest does not match the formal manifest"
                )

    trial_rows: list[Mapping[str, Any]] = []
    malformed_rows: list[dict[str, Any]] = []
    if document is not None:
        if not isinstance(raw_rows, Sequence) or isinstance(
            raw_rows, (str, bytes, bytearray)
        ):
            blocking_reasons.append("smoke report has no result_summaries list")
        else:
            for index, raw_row in enumerate(raw_rows):
                if not isinstance(raw_row, Mapping):
                    malformed_rows.append(
                        {"row_index": index, "reason": "not-an-object"}
                    )
                    continue
                kind = raw_row.get("kind")
                if kind in (None, "trial"):
                    trial_rows.append(raw_row)
                elif kind == "job":
                    # Harbor emits a job-level result alongside its six trial
                    # documents. It is useful metadata but not a seventh trial.
                    continue
                else:
                    malformed_rows.append(
                        {"row_index": index, "reason": "unexpected-kind"}
                    )
            if malformed_rows:
                blocking_reasons.append(
                    f"smoke report contains {len(malformed_rows)} malformed rows"
                )

    validation: dict[str, Any] = {
        "complete": False,
        "expected_trials": expected_trials,
        "observed_trials": 0,
        "missing_keys": [],
        "duplicate_keys": {},
        "duplicate_identities": {},
        "unknown_rows": [],
        "unresolved_rows": [],
        "normalised_rows": [],
    }
    if (
        document is not None
        and isinstance(raw_rows, Sequence)
        and not isinstance(raw_rows, (str, bytes, bytearray))
    ):
        try:
            validation = validate_matrix_rows(
                trial_rows,
                repetitions=1,
                expected_models=models,
                expected_agents=agents,
                expected_tasks=(task,),
            )
        except (TypeError, ValueError) as exc:
            blocking_reasons.append(
                f"smoke matrix validation failed ({type(exc).__name__})"
            )

    structural_complete = bool(validation.get("complete")) and not malformed_rows
    if not structural_complete:
        if validation.get("missing_keys"):
            blocking_reasons.append(
                f"smoke matrix missing {len(validation['missing_keys'])} trial(s)"
            )
        if validation.get("duplicate_keys"):
            blocking_reasons.append(
                f"smoke matrix has {len(validation['duplicate_keys'])} duplicate key(s)"
            )
        if validation.get("duplicate_identities"):
            blocking_reasons.append(
                f"smoke matrix has {len(validation['duplicate_identities'])} duplicate identity(s)"
            )
        if validation.get("unknown_rows"):
            blocking_reasons.append(
                f"smoke matrix has {len(validation['unknown_rows'])} unknown trial(s)"
            )
        if validation.get("unresolved_rows"):
            blocking_reasons.append(
                f"smoke matrix has {len(validation['unresolved_rows'])} unresolved trial(s)"
            )

    normalised_rows = validation.get("normalised_rows")
    if not isinstance(normalised_rows, list):
        normalised_rows = []
    setup_failures: list[dict[str, Any]] = []
    verifier_failures: list[dict[str, Any]] = []
    unresolved_trials: list[dict[str, Any]] = []
    for index, row in enumerate(normalised_rows):
        if not isinstance(row, Mapping):
            continue
        entry = _smoke_row_entry(index, row)
        if _is_setup_failure(row):
            setup_failures.append(entry)
        elif type(row.get("passed")) is bool:
            if row.get("passed") is False:
                verifier_failures.append(entry)
        else:
            unresolved_trials.append(entry)
    if setup_failures:
        blocking_reasons.append(
            f"smoke setup/infrastructure failures: {len(setup_failures)}"
        )

    # Harbor may return a non-zero command status after a verifier reports a
    # legitimate failing answer. Attribute such a status to the matching
    # evaluated row when possible; an unattributed non-zero remains an
    # infrastructure failure and blocks the formal matrix.
    command_failures: list[dict[str, Any]] = []
    verifier_command_failures: list[dict[str, Any]] = []
    row_by_condition: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in normalised_rows:
        if not isinstance(row, Mapping):
            continue
        model = row.get("matrix_model")
        agent = row.get("matrix_agent")
        if isinstance(model, str) and isinstance(agent, str):
            row_by_condition[(model, agent)] = row
    ordered_conditions = [(model, agent) for model in models for agent in agents]
    raw_outcome_count = (
        len(raw_outcomes)
        if isinstance(raw_outcomes, Sequence)
        and not isinstance(raw_outcomes, (str, bytes, bytearray))
        else 0
    )
    if isinstance(raw_outcomes, Sequence) and not isinstance(
        raw_outcomes, (str, bytes, bytearray)
    ):
        for index, outcome in enumerate(raw_outcomes):
            if not isinstance(outcome, Mapping):
                continue
            code = outcome.get("return_code", outcome.get("exit_code"))
            if isinstance(code, bool):
                continue
            if (
                isinstance(code, (int, float))
                and math.isfinite(float(code))
                and code != 0
            ):
                entry: dict[str, Any] = {"outcome_index": index, "return_code": code}
                for field in ("model", "agent"):
                    value = outcome.get(field)
                    if isinstance(value, str) and value:
                        entry[field] = value[:160]
                model = outcome.get("model")
                agent = outcome.get("agent")
                canonical_model = _canonical_model_name(model, models)
                canonical_agent = _canonical_agent_name(agent, agents)
                matched = (
                    row_by_condition.get((canonical_model, canonical_agent))
                    if canonical_model is not None and canonical_agent is not None
                    else None
                )
                # The runner normally records model/agent on every outcome.
                # For compact archived reports that omit those fields, use the
                # deterministic command order only when the outcome list still
                # contains exactly one entry per expected condition.
                if (
                    matched is None
                    and raw_outcome_count == expected_trials
                    and index < len(ordered_conditions)
                ):
                    matched = row_by_condition.get(ordered_conditions[index])
                if (
                    structural_complete
                    and not setup_failures
                    and matched is not None
                    and _is_verifier_failure(matched)
                ):
                    verifier_command_failures.append(entry)
                else:
                    command_failures.append(entry)
    infrastructure_failures: list[dict[str, Any]] = [
        {"source": "trial", **entry} for entry in setup_failures
    ] + [{"source": "command", **entry} for entry in command_failures]
    if command_failures:
        blocking_reasons.append(
            f"smoke command infrastructure failures: {len(command_failures)}"
        )

    configuration_failures: list[dict[str, Any]] = []
    probe_statuses: dict[str, str] = {}
    if raw_probes is not None:
        if not isinstance(raw_probes, Sequence) or isinstance(
            raw_probes, (str, bytes, bytearray)
        ):
            blocking_reasons.append("smoke reasoning_probes must be a list")
        else:
            seen_probe_models: set[str] = set()
            for index, probe in enumerate(raw_probes):
                if not isinstance(probe, Mapping):
                    configuration_failures.append(
                        {"probe_index": index, "reason": "not-an-object"}
                    )
                    continue
                model = probe.get("model")
                status = probe.get("status")
                if not isinstance(model, str) or model not in models:
                    configuration_failures.append(
                        {"probe_index": index, "reason": "unknown-model"}
                    )
                    continue
                if model in seen_probe_models:
                    configuration_failures.append(
                        {"probe_index": index, "model": model, "reason": "duplicate"}
                    )
                    continue
                seen_probe_models.add(model)
                if not isinstance(status, str):
                    status = "invalid"
                status = status.strip().lower()
                probe_statuses[model] = status
                allowed_statuses = {"supported", "unsupported"}
                if not require_probes:
                    allowed_statuses.add("not_run")
                if status not in allowed_statuses:
                    entry = {"probe_index": index, "model": model, "status": status}
                    error_type = probe.get("error_type")
                    if isinstance(error_type, str) and error_type:
                        entry["error_type"] = error_type[:160]
                    configuration_failures.append(entry)
            missing_probe_models = set(models) - set(seen_probe_models)
            if require_probes and missing_probe_models:
                configuration_failures.append(
                    {
                        "reason": "missing-models",
                        "models": sorted(missing_probe_models),
                    }
                )
    if require_probes and raw_probes is None:
        configuration_failures.append({"reason": "missing-probes"})
    elif require_probes and isinstance(raw_probes, Sequence) and not raw_probes:
        configuration_failures.append({"reason": "empty-probes"})
    if configuration_failures:
        blocking_reasons.append(
            f"smoke reasoning configuration failures: {len(configuration_failures)}"
        )

    # Preserve ordering while avoiding repetitive messages when several checks
    # identify the same underlying malformed report.
    blocking_reasons = list(dict.fromkeys(blocking_reasons))
    can_proceed = not blocking_reasons
    if not can_proceed:
        gate_status = "blocked"
    elif verifier_failures:
        gate_status = "ready_with_verifier_failures"
    else:
        gate_status = "ready"
    safe_validation = {
        key: value for key, value in validation.items() if key != "normalised_rows"
    }
    gate = {
        "schema_version": "terminal-bench-smoke-gate/v1",
        "status": gate_status,
        "can_proceed": can_proceed,
        "task": task,
        "image_manifest_sha256": (
            expected_manifest.get("manifest_sha256")
            if expected_manifest is not None
            else None
        ),
        "report_path": report_path,
        "report_status": report_status,
        "expected_trials": expected_trials,
        "observed_trials": validation.get("observed_trials", 0),
        "complete": structural_complete,
        "require_probes": require_probes,
        "verifier_failures_allowed": True,
        "blocking_reasons": blocking_reasons,
        "setup_failures": setup_failures,
        "infrastructure_failures": infrastructure_failures,
        "command_failures": command_failures,
        "verifier_command_failures": verifier_command_failures,
        "configuration_failures": configuration_failures,
        "verifier_failures": verifier_failures,
        "unresolved_trials": unresolved_trials,
        "malformed_rows": malformed_rows,
        "reasoning_probe_statuses": probe_statuses,
        "matrix_validation": safe_validation,
    }
    safe_gate = redact(gate, secrets=_runner_redaction_secrets(secrets))
    return dict(safe_gate) if isinstance(safe_gate, Mapping) else gate


def _ablation_report_dimension(
    document: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    field: str,
    item_field: str,
    fallback: Sequence[str],
    basename: bool = False,
) -> tuple[str, ...]:
    """Resolve a report dimension without silently widening its matrix.

    The execution report carries explicit dimensions. Compact artifacts from
    older runs may omit them, so infer them from summary rows before falling
    back to the canonical defaults. Duplicate or malformed declarations are
    rejected rather than producing an ambiguous strategy choice.
    """

    raw = document.get(field)
    if raw is None:
        inferred: list[str] = []
        for item in summary.values():
            if not isinstance(item, Mapping):
                continue
            value = item.get(item_field)
            if not isinstance(value, str) or not value.strip():
                continue
            value = value.strip()
            if basename:
                value = value.rstrip("/").rsplit("/", 1)[-1]
            if value not in inferred:
                inferred.append(value)
        raw = inferred or fallback
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise TypeError(f"ablation report {field} must be a non-empty string list")
    values: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"ablation report {field} must contain non-empty strings")
        value = item.strip()
        if basename:
            value = value.rstrip("/").rsplit("/", 1)[-1]
        if value in values:
            raise ValueError(f"ablation report {field} contains duplicates")
        values.append(value)
    if not values:
        raise TypeError(f"ablation report {field} must not be empty")
    return tuple(values)


def run_ablation_experiment(
    *,
    dataset: str = DATASET,
    jobs_dir: Path = Path(".harbor-runs/ablation"),
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: Mapping[str, str] | None = None,
    execute: bool = False,
    probe_reasoning: bool = True,
    reasoning_effort: str = "high",
    allow_incomplete: bool = True,
    probe_client_factory: Any | None = None,
    repetitions: int = FORMAL_REPETITIONS,
    tasks: Sequence[str] = ("build-cython-ext", "write-compressor"),
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Plan or execute the 3-model x 3-strategy ablation matrix."""

    source = dict(os.environ if environ is None else environ)
    if execute:
        validated_manifest = _require_execution_image_manifest(
            image_manifest,
            dataset=dataset,
        )
        missing = formal_required_credentials(source)
        if missing:
            raise ValueError(
                "required environment variable(s) are missing: " + ", ".join(missing)
            )
    plan = make_ablation_plan(
        dataset=dataset,
        jobs_dir=jobs_dir,
        harbor_bin=harbor_bin,
        harbor_sudo=harbor_sudo,
        environ=source,
        repetitions=repetitions,
        tasks=tasks,
        reasoning_effort=reasoning_effort if probe_reasoning else None,
        image_manifest=image_manifest,
    )
    should_probe = probe_reasoning and (execute or probe_client_factory is not None)
    if should_probe:
        probes = [
            probe_model_route(
                model,
                environ=source,
                effort=reasoning_effort,
                client_factory=probe_client_factory,
            )
            for model in MODEL_IDS
        ]
    elif probe_reasoning:
        probes = [
            {
                "model": model,
                "requested_effort": reasoning_effort,
                "status": "not_run",
                "reason": "deferred until ablation execution",
            }
            for model in MODEL_IDS
        ]
    else:
        probes = []
    if execute and probe_reasoning:
        _validate_execution_probes(probes)
    plan = make_ablation_plan(
        dataset=dataset,
        jobs_dir=jobs_dir,
        harbor_bin=harbor_bin,
        harbor_sudo=harbor_sudo,
        environ=source,
        repetitions=repetitions,
        tasks=tasks,
        reasoning_effort=reasoning_effort if probe_reasoning else None,
        reasoning_capabilities=_capability_map(probes) if probe_reasoning else None,
        image_manifest=image_manifest,
    )
    plan["reasoning_probes"] = probes
    if not execute:
        return plan

    plan["status"] = "executing"
    secrets = _credential_values(source)
    outcomes: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    scrubbed_artifact_files = 0
    for index, spec in enumerate(plan["commands"]):
        if not isinstance(spec, Mapping):
            continue
        model = str(spec.get("model", ""))
        strategy = str(spec.get("strategy", ""))
        argv = tuple(str(item) for item in spec.get("argv", ()))
        command = ExperimentCommand(agent=COURSE_AGENT, argv=argv)
        strategy_dir = jobs_dir / model / strategy
        log_path = strategy_dir / f"course-coding-agent__{index}.log"
        route = _route_without_reasoning(model, source)
        # Clear/redact leftovers before the snapshot so a rerun cannot promote
        # a scrubbed historical file into this command's result set.
        scrubbed_artifact_files += _scrub_generated_artifacts(
            strategy_dir, secrets=secrets
        )
        before_results = _result_file_snapshot(strategy_dir)
        code = _run_one(
            command,
            cwd=Path.cwd(),
            environment=_harbor_process_environment(route, strip_reasoning=True),
            log_path=log_path,
            redaction_secrets=secrets,
        )
        outcomes.append(
            {
                "model": model,
                "strategy": strategy,
                "return_code": code,
                "log": str(log_path),
            }
        )
        # Scrub immediately after each Harbor invocation.  OpenCode may leave
        # a SQLite WAL or other diagnostics in the strategy directory, and the
        # result projection below must never read an unsanitized artifact.
        scrubbed_artifact_files += _scrub_generated_artifacts(
            strategy_dir, secrets=secrets
        )
        for result_path in _changed_result_files(strategy_dir, before_results):
            summary = _summarize_result_file(
                result_path,
                secrets=secrets,
                image_manifest=validated_manifest,
            )
            if summary.get("kind") == "trial":
                summary = dict(summary)
                summary["ablation_model"] = model
                summary["ablation_strategy"] = strategy
                if not summary.get("model"):
                    summary["model"] = model
                rows.append(summary)

    # Catch job-level files Harbor may place directly under the root rather than
    # inside the per-strategy directory.  The second pass is idempotent.
    scrubbed_artifact_files += _scrub_generated_artifacts(jobs_dir, secrets=secrets)
    plan["scrubbed_artifact_files"] = scrubbed_artifact_files
    plan["outcomes"] = outcomes
    plan["result_summaries"] = rows
    ablation_summary = _aggregate_ablation_rows(
        rows,
        models=MODEL_IDS,
        tasks=tasks,
        repetitions=repetitions,
    )
    plan["ablation_summary"] = ablation_summary
    decision = choose_formal_round_strategy(
        ablation_summary,
        models=MODEL_IDS,
        tasks=tasks,
    )
    plan["fixed_formal_configuration"] = decision
    plan["matrix_complete"] = decision["status"] == "complete"
    plan["status"] = "finished"
    if not allow_incomplete and not plan["matrix_complete"]:
        raise ValueError("ablation matrix incomplete; no formal strategy was fixed")
    return plan


def write_ablation_artifacts(
    output_dir: Path,
    report: Mapping[str, Any],
) -> dict[str, str | int | bool]:
    """Write compact, credential-free artifacts for an ablation report."""

    summary = report.get("ablation_summary")
    if summary is None:
        summary = report.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("ablation report has no summary to write")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ablation-summary.json"
    _write_json(
        json_path,
        {
            "schema_version": "terminal-bench-ablation-results/v1",
            # Preserve the dimensions used to choose the strategy.  Without
            # these fields a compact artifact would be forced to assume the
            # canonical three-model/two-task matrix when read later.
            "models": list(report.get("models", MODEL_IDS)),
            "tasks": list(
                report.get("tasks", ("build-cython-ext", "write-compressor"))
            ),
            "strategies": list(
                report.get(
                    "strategies", ("current_20", "efficiency_20", "efficiency_30")
                )
            ),
            "repetitions_per_task": report.get(
                "repetitions_per_task", FORMAL_REPETITIONS
            ),
            "summary": summary,
            "fixed_formal_configuration": report.get("fixed_formal_configuration"),
            "matrix_complete": bool(report.get("matrix_complete")),
            "image_manifest": _image_manifest_public(
                report.get("image_manifest")
                if isinstance(report.get("image_manifest"), Mapping)
                else None
            ),
        },
    )
    csv_path = output_dir / "ablation-summary.csv"
    rows = [dict(value) for value in summary.values() if isinstance(value, Mapping)]
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda value: (
                str(value.get("model")),
                str(value.get("strategy")),
                str(value.get("task", "")),
            ),
        ):
            writer.writerow(row)
    os.chmod(csv_path, 0o600)
    markdown_path = output_dir / "ablation-summary.md"
    lines = [
        "# Course Agent round-budget ablation",
        "",
        "效率 20 轮仅在每个模型/任务的完整比较中达到或超过效率 30 轮通过数时保留。",
        "",
        "| model | strategy | task | passed/evaluable | setup failures | complete |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in sorted(
        rows,
        key=lambda value: (
            str(value.get("model")),
            str(value.get("strategy")),
            str(value.get("task", "")),
        ),
    ):
        lines.append(
            "| {model} | {strategy} | {task} | {passed}/{evals} | {setup} | {complete} |".format(
                model=row.get("model", ""),
                strategy=row.get("strategy", ""),
                task=row.get("task", ""),
                passed=row.get("n_passed", 0),
                evals=row.get("n_evaluable_trials", 0),
                setup=row.get("n_setup_failures", 0),
                complete="yes" if row.get("complete") else "no",
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(markdown_path, 0o600)
    # A small four-panel image keeps the same visual convention as the formal
    # report while showing strategy pass counts and evaluable counts.
    total_rows = [row for row in rows if "task" not in row]
    strategy_groups = sorted(
        {f"{row.get('model')}::{row.get('strategy')}" for row in total_rows}
    )
    by_key = {f"{row.get('model')}::{row.get('strategy')}": row for row in total_rows}
    panel_data = [
        [float(by_key[key].get("n_passed", 0)) for key in strategy_groups],
        [float(by_key[key].get("n_evaluable_trials", 0)) for key in strategy_groups],
        [float(by_key[key].get("n_setup_failures", 0)) for key in strategy_groups],
        [float(by_key[key].get("n_trials", 0)) for key in strategy_groups],
    ]
    png_path = _simple_bar_png(
        panel_data, strategy_groups, output_dir / "ablation-summary.png"
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "png": str(png_path),
        "n_rows": len(rows),
        "complete": bool(report.get("matrix_complete")),
    }


def make_plan(
    *,
    model: str,
    dataset: str = DATASET,
    jobs_dir: str | os.PathLike[str] = ".harbor-runs",
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: Mapping[str, str] | None = None,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe plan shared by dry-run and execution reporting."""

    source = os.environ if environ is None else environ
    validated_manifest = _image_manifest_for_use(image_manifest, dataset=dataset)
    routed_source = route_environment(model, source)
    commands = list(
        _experiment_commands(
            model=model,
            dataset=dataset,
            jobs_dir=jobs_dir,
            harbor_bin=harbor_bin,
            harbor_sudo=harbor_sudo,
            environ=routed_source,
        )
    )
    return {
        "schema_version": "terminal-bench-experiment/v1",
        "label": EXPERIMENT_LABEL,
        "dataset": dataset,
        "dataset_reference": _dataset_reference(dataset),
        "harbor_bin": harbor_bin,
        "harbor_sudo": harbor_sudo,
        "harbor_invocation_prefix": list(
            harbor_invocation_prefix(harbor_bin, harbor_sudo=harbor_sudo)
        ),
        "task_count": len(TASK_IDS),
        "tasks": list(TASK_IDS),
        "task_filters": _task_filters(dataset),
        "repetitions_per_task": 1,
        "model": model,
        "model_routing": {
            "course_coding_agent": model,
            "opencode": _opencode_model(model),
            "provider": routed_source.get("CODING_AGENT_PROVIDER"),
            "base_url": routed_source.get("CODING_AGENT_BASE_URL"),
            "key_env": routed_source.get("CODING_AGENT_KEY_ENV"),
            "note": "PinnedOpenCodeAgent is used; the OpenCode prefix is routing only and the underlying model ID is unchanged.",
        },
        "agents": [item.agent for item in commands],
        "conditions": {
            "n_concurrent": 1,
            "n_attempts": 1,
            "max_retries": 0,
            "same_model": True,
        },
        "commands": [list(item.argv) for item in commands],
        "image_manifest": _image_manifest_public(validated_manifest),
        "status": "planned",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
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
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _run_one(
    command: ExperimentCommand,
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    redaction_secrets: tuple[str, ...] = (),
) -> int:
    """Run Harbor while writing only a redacted, owner-readable transcript."""

    # Harbor output may accidentally contain a different provider credential
    # inherited by the shell. Redact every conventionally sensitive value, not
    # only the selected model key; values are used in memory and never logged.
    secrets = tuple(
        set(redaction_secrets)
        | {
            value
            for name, value in environment.items()
            if is_sensitive_environment_name(name) and isinstance(value, str) and value
        }
    )
    redaction_secrets = _runner_redaction_secrets(secrets)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # ``sudo`` drops arbitrary environment variables by default.  Harbor needs
    # the selected key (and, for custom routes, the base URL) in its host
    # process so it can expand ``${VAR}`` agent-environment templates.  Insert
    # a name-only ``--preserve-env`` option at execution time; keeping it out of
    # the planned command preserves the public command shape while ensuring the
    # value itself never appears in argv or logs.
    runtime_argv = _sudo_preserving_argv(command.argv, environment)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        os.chmod(log_path, 0o600)
        process = subprocess.Popen(
            runtime_argv,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(_redact_runner_text(line, secrets))
            log.flush()
        return process.wait()


def _sudo_preserving_argv(
    argv: Sequence[str], environment: Mapping[str, str]
) -> tuple[str, ...]:
    """Add a least-privilege sudo environment allow-list when needed.

    Plans intentionally continue to show the stable ``sudo -n harbor`` prefix.
    This projection is applied only to the subprocess invocation because the
    preserve list depends on the selected route and may include a custom key
    name.  Only variable *names* cross the argv boundary.
    """

    values = tuple(str(item) for item in argv)
    if len(values) < 3 or values[0:2] != ("sudo", "-n"):
        return values
    # ``sudo`` commonly uses a restricted ``secure_path``. Resolve a bare
    # Harbor executable while the caller's PATH is still available so a venv
    # install can run under sudo without requiring PATH preservation. If it is
    # not discoverable, retain the original name and let the normal subprocess
    # error report the missing binary.
    executable = values[2]
    if "/" not in executable:
        resolved = shutil.which(executable)
        if resolved:
            values = (*values[:2], resolved, *values[3:])
    if any(
        item == "--preserve-env" or item.startswith("--preserve-env=")
        for item in values[:3]
    ):
        return values

    names: set[str] = set()
    # Agent-environment templates are the authoritative references for custom
    # credential names and base URLs.
    for item in values:
        names.update(_ENV_REFERENCE_RE.findall(item))
    # The Course agent reads these settings directly from the host process.
    names.update(
        name
        for name in environment
        if isinstance(name, str) and name.startswith("CODING_AGENT_")
    )
    # ``_harbor_process_environment`` re-adds only the selected credential, but
    # retain this name-based guard for callers that invoke ``_run_one`` directly.
    names.update(
        name
        for name in environment
        if isinstance(name, str) and is_sensitive_environment_name(name)
    )
    valid = sorted(
        name
        for name in names
        if name in environment and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    )
    if not valid:
        return values
    return (values[0], values[1], f"--preserve-env={','.join(valid)}", *values[2:])


def _scrub_generated_artifacts(
    root: Path,
    *,
    secrets: tuple[str, ...] = (),
) -> int:
    """Remove credential material from Harbor's generated diagnostics.

    Harbor agents may persist their own caches in a trial's artifact directory
    (OpenCode, for example, keeps a SQLite WAL). Those files are not needed for
    ATIF analysis but can contain an API key supplied through ``--agent-env``.
    Delete matching database caches and atomically redact all other generated
    files before they are included in a report or archived for inspection.
    """

    if not root.exists():
        return 0
    encoded_secrets = tuple(
        secret.encode("utf-8")
        for secret in secrets
        if isinstance(secret, str) and secret
    )
    changed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        scrubbed = data
        for secret in encoded_secrets:
            if len(secret) >= 4:
                scrubbed = scrubbed.replace(secret, b"[REDACTED]")
            else:
                scrubbed = re.sub(
                    rb"(?<![A-Za-z0-9_])" + re.escape(secret) + rb"(?![A-Za-z0-9_])",
                    b"[REDACTED]",
                    scrubbed,
                )
        scrubbed = _SECRET_TOKEN_RE.sub(b"[REDACTED]", scrubbed)
        if scrubbed == data:
            continue
        changed += 1
        # SQLite's main file/WAL pair cannot be reliably repaired by a byte
        # replacement. They are disposable agent caches, so remove the whole
        # cache when a secret was found there.
        if path.name.startswith("opencode.db"):
            try:
                path.unlink()
            except OSError as exc:
                raise OSError(f"could not remove sensitive artifact {path}") from exc
            continue
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".scrub",
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(scrubbed)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
    return changed


def _collect_result_files(jobs_dir: Path) -> list[Path]:
    """Find Harbor job and trial result documents across 0.22 layouts."""

    if not jobs_dir.exists():
        return []
    paths: set[Path] = set()
    for filename in ("result.json", "results.json"):
        paths.update(path for path in jobs_dir.rglob(filename) if path.is_file())
    return sorted(paths)


def _file_sha256(path: Path) -> str | None:
    """Return a streaming SHA-256 digest, or ``None`` if the file is unreadable."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _result_file_snapshot(
    jobs_dir: Path,
) -> dict[Path, tuple[int, int, int, str | None]]:
    """Capture result-file metadata so a rerun cannot reuse stale trials."""

    snapshot: dict[Path, tuple[int, int, int, str | None]] = {}
    for path in _collect_result_files(jobs_dir):
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[path] = (
            stat.st_mtime_ns,
            stat.st_size,
            stat.st_ino,
            _file_sha256(path),
        )
    return snapshot


def _changed_result_files(
    jobs_dir: Path,
    before: Mapping[Path, tuple[int, ...]],
) -> list[Path]:
    """Return only result files created or rewritten by the current command."""

    changed: list[Path] = []
    for path in _collect_result_files(jobs_dir):
        try:
            stat = path.stat()
        except OSError:
            continue
        current = (
            stat.st_mtime_ns,
            stat.st_size,
            stat.st_ino,
            _file_sha256(path),
        )
        previous = before.get(path)
        # Accept the old three-field shape for callers carrying an in-memory
        # snapshot from an older runner, while all new snapshots include the
        # content digest.
        if current[-1] is None or previous is None or tuple(previous) != current:
            changed.append(path)
    return changed


def _finite_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _duration_from_timestamps(started: Any, finished: Any) -> float | None:
    if not isinstance(started, str) or not isinstance(finished, str):
        return None
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(finished)
        value = (end - start).total_seconds()
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def _phase_duration(document: Mapping[str, Any], phase_name: str) -> float | None:
    """Read a Harbor phase duration (for example ``agent_execution``)."""

    phase = document.get(phase_name)
    if not isinstance(phase, Mapping):
        return None
    return _duration_from_timestamps(phase.get("started_at"), phase.get("finished_at"))


def _summary_number(
    *sources: Mapping[str, Any], names: tuple[str, ...]
) -> int | float | None:
    for source in sources:
        for name in names:
            value = _finite_number(source.get(name))
            if value is not None:
                return value
    return None


def _first_text(*sources: Mapping[str, Any], names: tuple[str, ...]) -> str | None:
    for source in sources:
        for name in names:
            value = source.get(name)
            if isinstance(value, str) and value:
                return value
    return None


def _first_present(*sources: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    """Return the first explicitly present value from ordered result sources."""

    for source in sources:
        for name in names:
            if name in source and source.get(name) is not None:
                return source.get(name)
    return None


def _nested_first_text(
    *sources: Mapping[str, Any],
    names: tuple[str, ...],
    nested_names: Sequence[str] | None = None,
) -> str | None:
    """Read a text field from direct or one-level nested result metadata.

    ``nested_names`` is intentionally configurable.  Result documents commonly
    contain several objects with a generic ``name`` field (for example the
    agent and its model); recursively searching all of them can assign the
    agent name as the model.  Callers extracting an identity should therefore
    restrict the nested roles they are willing to inspect.
    """

    direct = _first_text(*sources, names=names)
    if direct is not None:
        return direct
    roles = tuple(
        nested_names
        if nested_names is not None
        else (
            "metadata",
            "config",
            "agent",
            "agent_info",
            "model_info",
            "model",
        )
    )
    for source in sources:
        for nested_name in roles:
            nested = source.get(nested_name)
            if isinstance(nested, Mapping):
                value = _first_text(nested, names=names)
                if value is not None:
                    return value
    return None


def _verifier_reward(verifier_result: Mapping[str, Any]) -> int | float | None:
    """Read Harbor 0.22's nested reward without assuming one schema.

    Harbor serializes ``VerifierResult`` as ``{"rewards": {"reward": ...}}``
    while older wrappers and small test fixtures commonly put ``reward`` at
    the top level.  Keep the projection tolerant of both forms, but accept
    only finite numeric values so a malformed result cannot look like a pass.
    """

    candidates: list[Any] = [verifier_result.get("reward")]
    nested = verifier_result.get("rewards")
    if isinstance(nested, Mapping):
        candidates.extend(
            (
                nested.get("reward"),
                nested.get("value"),
                nested.get("score"),
            )
        )
    elif isinstance(nested, list):
        # A few Harbor-compatible runners emit a list of named reward
        # records.  This branch is deliberately narrow and does not treat
        # arbitrary strings or sequences as scores.
        for item in nested:
            if isinstance(item, Mapping):
                candidates.extend(
                    (item.get("reward"), item.get("value"), item.get("score"))
                )
            else:
                candidates.append(item)
    for candidate in candidates:
        value = _finite_number(candidate)
        if value is not None:
            return value
    return None


def _metric_number(raw: Mapping[str, Any], name: str) -> int | float | None:
    """Extract a finite metric from Harbor's direct or list-shaped fields."""

    direct = _finite_number(raw.get(name))
    if direct is not None:
        return direct
    metrics = raw.get("metrics")
    if isinstance(metrics, list):
        for item in metrics:
            if isinstance(item, Mapping):
                value = _finite_number(item.get(name))
                if value is not None:
                    return value
    return None


def _find_trial_artifact(directory: Path, filename: str) -> Path | None:
    """Locate an agent artifact across Harbor 0.22 log layouts."""

    candidates = (directory / "agent" / filename, directory / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # A few Harbor wrappers add one more role directory. Keep the fallback
    # bounded to this trial directory and deterministic for reporting.
    try:
        matches = sorted(
            candidate
            for candidate in directory.glob(f"*/{filename}")
            if candidate.is_file()
        )
    except OSError:
        return None
    return matches[0] if matches else None


def _exception_phase(exception_type: object) -> str | None:
    """Project Harbor exception classes into a stable, human-readable phase."""

    if not isinstance(exception_type, str) or not exception_type:
        return None
    lowered = exception_type.lower()
    if "setup" in lowered and "timeout" in lowered:
        return "setup_timeout"
    if "environment" in lowered and "timeout" in lowered:
        return "environment_setup_timeout"
    if "cancel" in lowered:
        return "cancelled"
    if "timeout" in lowered:
        return "timeout"
    return "error"


def _opencode_infrastructure_reason(directory: Path) -> str | None:
    """Classify a structured OpenCode API error without retaining its body."""

    log_path = _find_trial_artifact(directory, "opencode.txt")
    if log_path is None:
        return None
    try:
        with log_path.open(encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                if index >= 1000:
                    break
                if len(line) > 1024 * 1024 or '"error"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, Mapping) or event.get("type") != "error":
                    continue
                error = event.get("error")
                if not isinstance(error, Mapping):
                    continue
                data = error.get("data")
                if not isinstance(data, Mapping):
                    data = {}
                name = str(error.get("name", "")).strip().lower()
                status_value = _finite_number(data.get("statusCode"))
                status = int(status_value) if status_value is not None else None
                bounded_signals = " ".join(
                    str(data.get(field, ""))[:4096]
                    for field in ("code", "message", "responseBody")
                ).lower()
                if any(
                    marker in bounded_signals
                    for marker in (
                        "insufficient_balance",
                        "insufficient account balance",
                        "insufficient credit",
                        "billing quota",
                    )
                ):
                    return "model_billing"
                if status == 401 or "unauthorized" in bounded_signals:
                    return "model_authentication"
                if status in (402, 403):
                    return "model_authorization"
                if status == 429 or "rate limit" in bounded_signals:
                    return "model_rate_limit"
                if status in (408, 504) or "timed out" in bounded_signals:
                    return "model_gateway_timeout"
                if status is not None and status >= 500:
                    return "model_gateway"
                if status is not None and name == "apierror":
                    return "model_api_error"
                if any(
                    marker in name for marker in ("connection", "network", "timeout")
                ):
                    return "model_gateway"
    except (OSError, UnicodeError):
        return None
    return None


def _trajectory_metrics(path: Path | None) -> dict[str, int | float]:
    """Read basic call counts from an ATIF trajectory when no run summary exists."""

    if path is None:
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(document, Mapping):
        return {}
    steps = document.get("steps")
    if not isinstance(steps, list):
        return {}
    model_requests = 0
    tool_calls = 0
    token_totals: dict[str, float] = {
        "input_tokens": 0.0,
        "output_tokens": 0.0,
        "cache_tokens": 0.0,
        "total_tokens": 0.0,
    }
    token_seen = {name: False for name in token_totals}
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        raw_llm = _finite_number(step.get("llm_call_count"))
        if raw_llm is not None:
            model_requests += int(raw_llm)
        elif step.get("source") == "agent":
            # ATIF producers predating ``llm_call_count`` emit one model turn
            # per agent step. Keep this fallback deliberately conservative.
            model_requests += 1
        calls = step.get("tool_calls")
        if isinstance(calls, list):
            tool_calls += len(calls)
        metrics = step.get("metrics")
        metric_sources: tuple[Mapping[str, Any], ...]
        if isinstance(metrics, Mapping):
            metric_sources = (metrics,)
        elif isinstance(metrics, list):
            metric_sources = tuple(
                item for item in metrics if isinstance(item, Mapping)
            )
        else:
            metric_sources = ()
        # A single provider usage mapping often contains both canonical and
        # compatibility aliases (for example ``prompt_tokens`` and
        # ``input_tokens``).  Choose one value per target in priority order;
        # summing every alias would silently double-count the trial.
        aliases: Mapping[str, tuple[str, ...]] = {
            "input_tokens": (
                "prompt_tokens",
                "input_tokens",
                "total_prompt_tokens",
            ),
            "output_tokens": (
                "completion_tokens",
                "output_tokens",
                "total_completion_tokens",
            ),
            "cache_tokens": (
                "cached_tokens",
                "cache_tokens",
                "total_cached_tokens",
            ),
            "total_tokens": ("total_tokens",),
        }
        for source in metric_sources:
            for target_name, source_names in aliases.items():
                value = next(
                    (
                        _finite_number(source.get(source_name))
                        for source_name in source_names
                        if _finite_number(source.get(source_name)) is not None
                    ),
                    None,
                )
                if value is not None:
                    token_totals[target_name] += float(value)
                    token_seen[target_name] = True
    result: dict[str, int | float] = {}
    if model_requests:
        result["model_requests"] = model_requests
        result["model_turns"] = model_requests
    if tool_calls:
        result["tool_calls"] = tool_calls
    final_metrics = document.get("final_metrics")
    if isinstance(final_metrics, Mapping):
        # Non-standard aggregate metrics from our exporter live under the
        # schema-approved ``final_metrics.extra`` mapping.  Read that mapping
        # as a fallback while keeping canonical top-level fields authoritative.
        final_metric_sources: tuple[Mapping[str, Any], ...] = (final_metrics,)
        final_extra = final_metrics.get("extra")
        if isinstance(final_extra, Mapping):
            final_metric_sources += (final_extra,)
        final_aliases: Mapping[str, tuple[str, ...]] = {
            "input_tokens": (
                "total_prompt_tokens",
                "prompt_tokens",
                "input_tokens",
            ),
            "output_tokens": (
                "total_completion_tokens",
                "completion_tokens",
                "output_tokens",
            ),
            "cache_tokens": (
                "total_cached_tokens",
                "cached_tokens",
                "cache_tokens",
            ),
            "total_tokens": ("total_tokens",),
        }
        for target_name, source_names in final_aliases.items():
            value = next(
                (
                    _finite_number(source.get(source_name))
                    for source in final_metric_sources
                    for source_name in source_names
                    if _finite_number(source.get(source_name)) is not None
                ),
                None,
            )
            if value is not None:
                result[target_name] = value
        for target_name, source_names in {
            "model_requests": ("model_requests", "model_calls"),
            "tool_calls": ("tool_calls",),
        }.items():
            value = next(
                (
                    _finite_number(source.get(source_name))
                    for source in final_metric_sources
                    for source_name in source_names
                    if _finite_number(source.get(source_name)) is not None
                ),
                None,
            )
            if value is not None:
                result[target_name] = value
    for name, seen in token_seen.items():
        if seen and name not in result:
            result[name] = token_totals[name]
    if "total_tokens" not in result:
        prompt = result.get("input_tokens")
        completion = result.get("output_tokens")
        if prompt is not None or completion is not None:
            result["total_tokens"] = float(prompt or 0) + float(completion or 0)
    return result


_REPETITION_FIELDS = (
    "repetition",
    "repeat",
    "rep",
    "attempt",
    "attempt_number",
    "attempt_index",
    "repetition_number",
    "repetition_index",
    "trial_index",
)


def _coerce_repetition(value: Any, *, zero_based: bool = False) -> int | None:
    """Parse a finite positive repetition number without guessing strings."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
        parsed = int(value.strip())
    else:
        return None
    if zero_based:
        parsed += 1
    return parsed if parsed >= 1 else None


def _extract_repetition(document: Mapping[str, Any], path: Path) -> int | None:
    """Read an explicit repeat index from Harbor/wrapper result metadata.

    Harbor 0.22 expands ``n_attempts`` into independently named trials and
    does not currently serialize the expansion index.  The formal collector
    therefore assigns deterministic ordinals for rows without one; explicit
    fields from wrappers take precedence when present.
    """

    sources: list[Mapping[str, Any]] = [document]
    for key in (
        "config",
        "agent_result",
        "metadata",
        "run_summary",
        "agent_metadata",
    ):
        nested = document.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    for source in sources:
        for field in _REPETITION_FIELDS:
            if field not in source:
                continue
            value = _coerce_repetition(
                source.get(field),
                zero_based=field.endswith("_index"),
            )
            if value is not None:
                return value

    # Some wrappers encode the repeat in a human-readable trial name or
    # directory.  Restrict matching to an explicit marker so Harbor's random
    # seven-character suffix is never interpreted as a repeat number.
    candidates = [document.get("trial_name"), path.parent.name]
    marker = re.compile(
        r"(?:^|[_-])(?:rep(?:etition)?|repeat|attempt|trial)[_-]?(\d+)(?:$|[_-])",
        re.IGNORECASE,
    )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        match = marker.search(candidate)
        if match:
            return _coerce_repetition(match.group(1))
    return None


def _summarize_result_file(
    path: Path,
    *,
    secrets: tuple[str, ...] = (),
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Project a Harbor result into a compact, credential-free report row."""

    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "path": str(path),
            "kind": "unreadable",
            "error_type": type(exc).__name__,
        }
    if not isinstance(document, Mapping):
        return {"path": str(path), "kind": "invalid", "error_type": "not_object"}

    stats = document.get("stats")
    if isinstance(stats, Mapping) and "n_total_trials" in document:
        # Job-level result (Harbor 0.22 writes one of these per invocation).
        evals: dict[str, dict[str, int | float | None]] = {}
        raw_evals = stats.get("evals")
        if isinstance(raw_evals, Mapping):
            for name, raw in raw_evals.items():
                if not isinstance(raw, Mapping):
                    continue
                evals[str(name)] = {
                    metric: _metric_number(raw, metric)
                    for metric in ("mean", "std", "min", "max")
                }
        return {
            "path": str(path),
            "kind": "job",
            "n_total_trials": document.get("n_total_trials"),
            "n_completed_trials": stats.get("n_completed_trials"),
            "n_errored_trials": stats.get("n_errored_trials"),
            "n_cancelled_trials": stats.get("n_cancelled_trials"),
            "evals": evals,
        }

    trial = document
    raw_agent_result = trial.get("agent_result")
    has_agent_result = isinstance(raw_agent_result, Mapping)
    agent_result = raw_agent_result
    if not has_agent_result:
        agent_result = {}
    raw_verifier_result = trial.get("verifier_result")
    has_verifier_result = isinstance(raw_verifier_result, Mapping)
    verifier_result = raw_verifier_result
    if not has_verifier_result:
        verifier_result = {}
    # The plugin's run.json is the authoritative source for model/tool counts;
    # fall back to equivalent fields if a different agent writes them inline.
    run_summary: Mapping[str, Any] = {}
    run_path = _find_trial_artifact(path.parent, "run.json")
    if run_path is not None:
        try:
            with run_path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, Mapping):
                run_summary = loaded
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            run_summary = {}
    agent_metadata = agent_result.get("metadata")
    if not isinstance(agent_metadata, Mapping):
        agent_metadata = {}
    passed: bool | None = None
    for key in ("passed", "success", "verified"):
        if isinstance(verifier_result.get(key), bool):
            passed = verifier_result[key]
            break
    reward = _verifier_reward(verifier_result)
    if passed is None and reward is not None:
        passed = reward >= 1
    exception_info = trial.get("exception_info")
    exception_type = (
        exception_info.get("exception_type")
        if isinstance(exception_info, Mapping)
        else None
    )
    exception_message = (
        exception_info.get("exception_message")
        if isinstance(exception_info, Mapping)
        else None
    )
    # Preparation/setup failures do not have a verifier reward. They are still
    # failed observations and must remain distinguishable from an unfinished
    # trial in the final comparison report.
    if passed is None and isinstance(exception_type, str):
        passed = False
    trajectory = _find_trial_artifact(path.parent, "trajectory.json")
    trajectory_metrics = _trajectory_metrics(trajectory)
    # ``run.json`` is emitted by CourseCodingAgent and its elapsed value is
    # agent-only. Harbor's top-level timestamps cover environment, agent and
    # verifier phases, so retain them separately as total wall time.
    agent_elapsed_seconds = _summary_number(
        run_summary,
        agent_result,
        names=("elapsed_seconds", "duration_seconds"),
    )
    if agent_elapsed_seconds is None:
        agent_elapsed_seconds = _phase_duration(trial, "agent_execution")
    total_elapsed_seconds = _summary_number(
        run_summary,
        agent_metadata,
        names=("total_elapsed_seconds", "total_duration_seconds"),
    )
    if total_elapsed_seconds is None:
        total_elapsed_seconds = _duration_from_timestamps(
            trial.get("started_at"), trial.get("finished_at")
        )
    elapsed_seconds = (
        agent_elapsed_seconds
        if agent_elapsed_seconds is not None
        else total_elapsed_seconds
    )
    agent_info = trial.get("agent_info")
    if not isinstance(agent_info, Mapping):
        agent_info = {}
    config = trial.get("config")
    if not isinstance(config, Mapping):
        config = {}
    model_info = agent_info.get("model_info")
    if not isinstance(model_info, Mapping):
        model_info = {}
    config_agent = config.get("agent")
    if not isinstance(config_agent, Mapping):
        config_agent = {}
    # Keep generic ``name`` lookups scoped to the object whose role is already
    # known.  A broad recursive fallback can otherwise turn
    # ``agent_info.name`` into the model ID (or vice versa).
    agent_name = _first_text(
        agent_info,
        config_agent,
        names=("name", "agent_name", "import_path", "agent"),
    )
    if agent_name is None:
        agent_name = _nested_first_text(
            config,
            agent_metadata,
            run_summary,
            trial,
            names=("agent_name", "import_path", "agent"),
            nested_names=("metadata", "config", "agent", "agent_info"),
        )
    model_name = _first_text(
        model_info,
        names=("model", "model_name", "model_id", "name"),
    )
    if model_name is None:
        model_name = _nested_first_text(
            agent_info,
            config,
            agent_metadata,
            run_summary,
            trial,
            names=("model", "model_name", "model_id"),
            nested_names=("model_info", "model", "metadata", "config"),
        )
    task_name = _first_text(trial, names=("task_name", "task"))
    if task_name is None:
        task_name = _nested_first_text(
            config,
            run_summary,
            agent_metadata,
            names=("task_name", "task"),
            nested_names=("task", "metadata", "config", "run_summary"),
        )
    # Harbor wrappers occasionally omit identity fields from result.json but
    # preserve them in the bounded directory layout. Use exact path-component
    # matches as a conservative final fallback; never infer from arbitrary
    # substrings in a random trial ID.
    path_parts = set(path.parts)
    if model_name is None:
        model_name = next((item for item in MODEL_IDS if item in path_parts), None)
    if agent_name is None:
        agent_name = next(
            (
                item
                for item in (*MATRIX_AGENT_LABELS, COURSE_AGENT, PINNED_OPENCODE_AGENT)
                if item in path_parts
            ),
            None,
        )
    if task_name is None:
        task_name = next((item for item in TASK_IDS if item in path_parts), None)
    infrastructure_reason = _opencode_infrastructure_reason(path.parent)
    if infrastructure_reason is not None:
        phase = "infrastructure"
        reason = infrastructure_reason
    else:
        phase = (
            run_summary.get("phase")
            if isinstance(run_summary.get("phase"), str)
            else agent_metadata.get("phase")
            if isinstance(agent_metadata.get("phase"), str)
            else _exception_phase(exception_type)
            if _exception_phase(exception_type) is not None
            else "completed"
        )
        reason = (
            run_summary.get("reason")
            if isinstance(run_summary.get("reason"), str)
            else agent_metadata.get("reason")
            if isinstance(agent_metadata.get("reason"), str)
            else exception_message
            if isinstance(exception_message, str)
            else "trial completed"
        )
    summary: dict[str, Any] = {
        "path": str(path),
        "kind": "trial",
        "agent": agent_name,
        "model": model_name,
        "task_name": task_name,
        "trial_name": trial.get("trial_name")
        if isinstance(trial.get("trial_name"), str)
        else None,
        "trial_id": trial.get("id") if isinstance(trial.get("id"), str) else None,
        "repetition": _extract_repetition(
            {**trial, "run_summary": run_summary, "agent_metadata": agent_metadata},
            path,
        ),
        "task_checksum": trial.get("task_checksum")
        if isinstance(trial.get("task_checksum"), str)
        else None,
        "source": trial.get("source") if isinstance(trial.get("source"), str) else None,
        "passed": passed,
        "reward": reward,
        "has_agent_result": has_agent_result,
        "has_verifier_result": has_verifier_result,
        "exception_type": exception_type if isinstance(exception_type, str) else None,
        "phase": phase,
        "reason": reason,
        "elapsed_seconds": elapsed_seconds,
        "agent_elapsed_seconds": _summary_number(
            {"agent_elapsed_seconds": agent_elapsed_seconds}
            if agent_elapsed_seconds is not None
            else {},
            names=("agent_elapsed_seconds",),
        ),
        "total_elapsed_seconds": _summary_number(
            {"total_elapsed_seconds": total_elapsed_seconds}
            if total_elapsed_seconds is not None
            else {},
            names=("total_elapsed_seconds",),
        ),
        "model_turns": _summary_number(
            run_summary,
            agent_metadata,
            agent_result,
            trajectory_metrics,
            names=("model_turns",),
        ),
        "model_requests": _summary_number(
            run_summary,
            agent_metadata,
            agent_result,
            trajectory_metrics,
            names=("model_requests", "model_calls"),
        ),
        "tool_calls": _summary_number(
            run_summary,
            agent_metadata,
            agent_result,
            trajectory_metrics,
            names=("tool_calls",),
        ),
        "input_tokens": _summary_number(
            run_summary,
            agent_result,
            trajectory_metrics,
            names=("n_input_tokens", "prompt_tokens", "input_tokens"),
        ),
        "output_tokens": _summary_number(
            run_summary,
            agent_result,
            trajectory_metrics,
            names=("n_output_tokens", "completion_tokens", "output_tokens"),
        ),
        "cache_tokens": _summary_number(
            run_summary,
            agent_result,
            trajectory_metrics,
            names=("n_cache_tokens", "cached_tokens", "cache_tokens"),
        ),
        "total_tokens": _summary_number(
            run_summary,
            agent_result,
            trajectory_metrics,
            names=("total_tokens", "n_total_tokens"),
        ),
        "reasoning_effort": _first_text(
            run_summary, agent_metadata, config, names=("reasoning_effort",)
        ),
        "reasoning_parameter": _first_text(
            run_summary, agent_metadata, config, names=("reasoning_parameter",)
        ),
        "reasoning_value": _first_present(
            run_summary, agent_metadata, config, names=("reasoning_value",)
        ),
        "reasoning_capability_status": _first_text(
            run_summary,
            agent_metadata,
            config,
            names=(
                "reasoning_capability_status",
                "reasoning_status",
            ),
        ),
        "reasoning_capability": _first_present(
            run_summary,
            agent_metadata,
            config,
            names=("reasoning_capability",),
        ),
        "image_digest": _first_text(
            run_summary,
            agent_metadata,
            config,
            names=("image_digest", "derived_image_digest"),
        ),
        "source_image_digest": _first_text(
            run_summary, agent_metadata, config, names=("source_image_digest",)
        ),
        "failure_stage": _first_text(
            run_summary, agent_metadata, names=("failure_stage", "phase")
        )
        or ("model_gateway" if infrastructure_reason is not None else None),
        "infrastructure_reason": infrastructure_reason,
        "trajectory": str(trajectory) if trajectory is not None else None,
    }
    row_manifest: Mapping[str, Any] | None = None
    if image_manifest is not None:
        if isinstance(image_manifest, Mapping) and {
            "manifest_sha256",
            "pinned_node_version",
            "pinned_opencode_version",
        }.issubset(image_manifest):
            row_manifest = image_manifest
        else:
            row_manifest = validate_image_manifest(image_manifest)
    summary = _attach_image_manifest_row(summary, row_manifest)
    safe = redact(summary, secrets=_runner_redaction_secrets(secrets))
    return dict(safe) if isinstance(safe, Mapping) else summary


def _aggregate_trial_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a compact per-agent comparison from projected trial rows."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "trial":
            continue
        name = row.get("agent")
        key = name if isinstance(name, str) and name else "unknown"
        grouped.setdefault(key, []).append(row)

    def mean(rows_for_agent: list[Mapping[str, Any]], field: str) -> float | None:
        values = [
            float(value)
            for row in rows_for_agent
            if (value := _finite_number(row.get(field))) is not None
        ]
        if not values:
            return None
        return sum(values) / len(values)

    result: dict[str, dict[str, Any]] = {}
    for agent, agent_rows in sorted(grouped.items()):
        passed = sum(row.get("passed") is True for row in agent_rows)
        failed = sum(row.get("passed") is False for row in agent_rows)
        result[agent] = {
            "n_trials": len(agent_rows),
            "n_passed": passed,
            "n_failed": failed,
            "n_unresolved": len(agent_rows) - passed - failed,
            "pass_rate": passed / len(agent_rows) if agent_rows else None,
            "mean_elapsed_seconds": mean(agent_rows, "elapsed_seconds"),
            "mean_model_requests": mean(agent_rows, "model_requests"),
            "mean_tool_calls": mean(agent_rows, "tool_calls"),
        }
    return result


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float | None, float | None]:
    """Return a two-sided Wilson interval for a binomial proportion."""

    if trials <= 0:
        return None, None
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or successes < 0
        or successes > trials
    ):
        raise ValueError("successes must be between zero and trials")
    if not isinstance(z, (int, float)) or not math.isfinite(float(z)) or z <= 0:
        raise ValueError("z must be a positive finite number")
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def _quartiles(
    values: Iterable[Any],
) -> tuple[float | None, float | None, float | None]:
    numbers = sorted(
        float(value) for value in values if _finite_number(value) is not None
    )
    if not numbers:
        return None, None, None

    def percentile(fraction: float) -> float:
        if len(numbers) == 1:
            return numbers[0]
        position = (len(numbers) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return numbers[lower]
        return numbers[lower] + (numbers[upper] - numbers[lower]) * (position - lower)

    return percentile(0.25), percentile(0.5), percentile(0.75)


def aggregate_matrix_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_trials_per_group: int = len(TASK_IDS) * FORMAL_REPETITIONS,
) -> dict[str, dict[str, Any]]:
    """Aggregate formal rows by exact ``model`` and ``agent`` combination.

    Setup failures are reported separately and excluded from the correctness
    denominator. Other completed trials with a false verifier reward count as
    evaluated failures, while unresolved rows remain visible in the counts.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "trial":
            continue
        model = row.get("matrix_model") or row.get("model")
        agent = row.get("matrix_agent") or row.get("agent")
        key = (
            model if isinstance(model, str) and model else "unknown-model",
            agent if isinstance(agent, str) and agent else "unknown-agent",
        )
        grouped.setdefault(key, []).append(row)

    def metric_values(group: list[Mapping[str, Any]], *names: str) -> list[float]:
        values: list[float] = []
        for row in group:
            value = _summary_number(row, names=names)
            if value is not None:
                values.append(float(value))
        return values

    result: dict[str, dict[str, Any]] = {}
    for (model, agent), group in sorted(grouped.items()):
        setup_rows = [row for row in group if _is_setup_failure(row)]
        setup_ids = {id(row) for row in setup_rows}
        evaluated = [
            row
            for row in group
            if id(row) not in setup_ids and isinstance(row.get("passed"), bool)
        ]
        passed = sum(row.get("passed") is True for row in evaluated)
        low, high = wilson_interval(passed, len(evaluated))
        # Setup/image/network/gateway failures are infrastructure observations,
        # not model work. Exclude them from timing/token/call distributions
        # while retaining their count in the group summary.
        metric_group = [row for row in group if id(row) not in setup_ids]
        agent_times = metric_values(
            metric_group, "agent_elapsed_seconds", "elapsed_seconds"
        )
        total_times = metric_values(
            metric_group, "total_elapsed_seconds", "elapsed_seconds"
        )
        tokens = {
            name: metric_values(metric_group, name)
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_tokens",
                "total_tokens",
            )
        }
        quartile_data = {
            "agent_elapsed_seconds": _quartiles(agent_times),
            "total_elapsed_seconds": _quartiles(total_times),
        }
        for name, values in tokens.items():
            quartile_data[name] = _quartiles(values)
        result[f"{model}::{agent}"] = {
            "model": model,
            "agent": agent,
            "expected_trials": expected_trials_per_group,
            "n_trials": len(group),
            "n_evaluable_trials": len(evaluated),
            "n_passed": passed,
            "n_failed": sum(row.get("passed") is False for row in evaluated),
            "n_unresolved": len(group) - len(setup_rows) - len(evaluated),
            "n_setup_failures": len(setup_rows),
            "accuracy": passed / len(evaluated) if evaluated else None,
            "accuracy_wilson_95": {"low": low, "high": high},
            "quartiles": {
                name: {"q1": values[0], "median": values[1], "q3": values[2]}
                for name, values in quartile_data.items()
            },
            "median_agent_elapsed_seconds": quartile_data["agent_elapsed_seconds"][1],
            "median_total_elapsed_seconds": quartile_data["total_elapsed_seconds"][1],
            "median_input_tokens": quartile_data["input_tokens"][1],
            "median_output_tokens": quartile_data["output_tokens"][1],
            "median_cache_tokens": quartile_data["cache_tokens"][1],
            "median_total_tokens": quartile_data["total_tokens"][1],
            "median_model_requests": _quartiles(
                metric_values(metric_group, "model_requests")
            )[1],
            "median_tool_calls": _quartiles(metric_values(metric_group, "tool_calls"))[
                1
            ],
            "infrastructure_failures": len(setup_rows),
        }
    return result


def _is_setup_failure(row: Mapping[str, Any]) -> bool:
    phase = str(row.get("phase", "")).lower()
    failure_stage = str(row.get("failure_stage", "")).lower()
    exception = str(row.get("exception_type", "")).lower()
    reason = str(row.get("reason", "")).lower()
    # A verifier exception is an evaluated answer failure even when Harbor did
    # not serialize a separate ``verifier_result`` object.  Check this before
    # the generic exception-without-results fallback below.
    if "verifier" in phase or "verifier" in exception or "verifier" in reason:
        return False
    if "setup" in phase or "environment_build" in phase:
        return True
    if any(
        marker in phase or marker in failure_stage
        for marker in ("infrastructure", "model_gateway", "network")
    ):
        return True
    if "setup" in exception or "environment" in exception:
        return True
    if "setup" in reason and row.get("passed") is not True:
        return True
    # Harbor setup failures commonly surface as NetworkConnectionError or
    # Docker/image errors before either an agent or verifier result exists.
    return bool(
        exception
        and not row.get("has_agent_result")
        and not row.get("has_verifier_result")
    )


def _is_verifier_failure(row: Mapping[str, Any]) -> bool:
    return row.get("passed") is False and not _is_setup_failure(row)


def _row_total_tokens(row: Mapping[str, Any]) -> int | float | None:
    direct = _finite_number(row.get("total_tokens"))
    if direct is not None:
        return direct
    # OpenAI-compatible ``prompt_tokens``/ATIF ``total_prompt_tokens`` already
    # include the cached portion. Adding ``cache_tokens`` again would inflate
    # the total and make cross-agent comparisons misleading.
    prompt = _finite_number(row.get("input_tokens"))
    completion = _finite_number(row.get("output_tokens"))
    if prompt is not None and completion is not None:
        return float(prompt) + float(completion)
    if prompt is not None:
        return float(prompt)
    if completion is not None:
        return float(completion)
    return None


def _canonical_task_name(
    value: Any,
    expected_tasks: Sequence[str] = TASK_IDS,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip().rstrip("/").rsplit("/", 1)[-1]
    return candidate if candidate in expected_tasks else None


def _canonical_model_name(value: Any, expected_models: Sequence[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    if candidate in expected_models:
        return candidate
    # OpenCode records a provider-qualified model in the config while
    # ``agent_info`` contains the unqualified model ID. Accept exactly the two
    # provider prefixes generated by this runner, and no arbitrary aliases.
    for prefix in ("openai/", f"{OPENCODE_GLM_PROVIDER}/"):
        if candidate.startswith(prefix) and candidate[len(prefix) :] in expected_models:
            return candidate[len(prefix) :]
    return None


def _canonical_agent_name(value: Any, expected_agents: Sequence[str]) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = value.strip()
    mapped = _AGENT_LABEL_ALIASES.get(candidate, candidate)
    # Callers may supply the import paths or stable labels as their expected
    # set.  Normalize those expected values through the same alias table.
    expected_labels = {
        _AGENT_LABEL_ALIASES.get(item, item)
        for item in expected_agents
        if isinstance(item, str)
    }
    return mapped if mapped in expected_labels else None


def _row_matrix_identity(row: Mapping[str, Any]) -> str | None:
    """Return a stable source identity used to detect copied result rows."""

    # Prefer an explicit Harbor ID/URI. If wrappers omit those, the result
    # path is a stronger identity than a human-readable trial name: repeated
    # attempts can legitimately share a name while living in distinct trial
    # directories.
    for field in ("trial_id", "trial_uri", "path", "trial_name"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return f"{field}:{value.strip()}"
    return None


def _row_matrix_identities(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every available source identity for duplicate detection."""

    identities = []
    for field in ("trial_id", "trial_uri", "trial_name", "path"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            identities.append(f"{field}:{value.strip()}")
    return tuple(identities)


def _matrix_sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("trial_name") or ""),
        str(row.get("trial_id") or ""),
        str(row.get("path") or ""),
    )


def _normalise_matrix_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    repetitions: int | None = None,
    expected_models: Sequence[str] = MODEL_IDS,
    expected_agents: Sequence[str] = MATRIX_AGENT_LABELS,
    expected_tasks: Sequence[str] = TASK_IDS,
) -> list[dict[str, Any]]:
    """Copy rows and attach canonical matrix fields.

    Rows with an explicit repetition retain it.  Missing repetition values are
    assigned only within a recognized model/agent/task group, using a stable
    sort order.  This mirrors Harbor's current ``n_attempts`` expansion while
    keeping malformed rows visible to the strict validator.
    """

    if repetitions is not None and (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("repetitions must be a positive integer")
    model_set = tuple(str(item) for item in expected_models)
    agent_set = tuple(str(item) for item in expected_agents)
    task_set = tuple(
        str(item).strip().rstrip("/").rsplit("/", 1)[-1] for item in expected_tasks
    )
    output = [dict(source) for source in rows]

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in output:
        # Accept already-normalised rows and explicit compact matrix keys. This
        # makes validation idempotent when a report is loaded and re-written.
        model_raw = row.get("matrix_model")
        agent_raw = row.get("matrix_agent")
        task_raw = row.get("matrix_task")
        matrix_key = row.get("matrix_key")
        key_parts = matrix_key.split("|") if isinstance(matrix_key, str) else []
        if len(key_parts) == 4:
            model_raw = model_raw if model_raw is not None else key_parts[0]
            agent_raw = agent_raw if agent_raw is not None else key_parts[1]
            task_raw = task_raw if task_raw is not None else key_parts[2]
            if row.get("repetition") is None:
                row["repetition"] = key_parts[3]
        if model_raw is None:
            model_raw = _first_text(row, names=("model", "model_name", "model_id"))
        if agent_raw is None:
            agent_raw = _first_text(row, names=("agent", "agent_name", "import_path"))
        if task_raw is None:
            task_raw = _first_text(row, names=("task_name", "task"))
        if model_raw is None:
            model_raw = _nested_first_text(
                row,
                names=("model", "model_name", "model_id"),
                nested_names=("model_info", "model", "metadata", "config"),
            )
        if agent_raw is None:
            agent_raw = _nested_first_text(
                row,
                names=("agent_name", "import_path", "agent", "name"),
                nested_names=("agent_info", "agent", "config", "metadata"),
            )
        if task_raw is None:
            task_raw = _nested_first_text(
                row,
                names=("task_name", "task"),
                nested_names=("task", "config", "metadata"),
            )
        # Exact path-component fallbacks are useful for compact fixtures that
        # retain identity only in ``path``; keep them bounded and explicit.
        raw_path = row.get("path")
        if isinstance(raw_path, str):
            components = set(Path(raw_path).parts)
            if model_raw is None:
                model_raw = next(
                    (item for item in model_set if item in components), None
                )
            if agent_raw is None:
                agent_raw = next(
                    (
                        item
                        for item in (*agent_set, COURSE_AGENT, PINNED_OPENCODE_AGENT)
                        if item in components
                    ),
                    None,
                )
            if task_raw is None:
                task_raw = next((item for item in task_set if item in components), None)
        model = _canonical_model_name(model_raw, model_set)
        agent = _canonical_agent_name(agent_raw, agent_set)
        task = _canonical_task_name(task_raw, task_set)
        if model is None or agent is None or task is None:
            continue
        row["matrix_model"] = model
        row["matrix_agent"] = agent
        row["matrix_task"] = task
        explicit_raw = row.get("repetition")
        explicit_field = "repetition"
        if explicit_raw is None:
            for field in _REPETITION_FIELDS:
                if field in row and row.get(field) is not None:
                    explicit_raw = row.get(field)
                    explicit_field = field
                    break
        if explicit_raw is not None:
            explicit = _coerce_repetition(
                explicit_raw,
                zero_based=explicit_field.endswith("_index"),
            )
            if explicit is None:
                row["repetition"] = None
                row["_matrix_repetition_invalid"] = True
            else:
                row["repetition"] = explicit
        else:
            row["repetition"] = None
        if not row.get("_matrix_repetition_invalid"):
            groups.setdefault((model, agent, task), []).append(row)

    for group_rows in groups.values():
        used = {
            int(row["repetition"])
            for row in group_rows
            if isinstance(row.get("repetition"), int)
        }
        available = [
            number
            for number in range(1, (repetitions or FORMAL_REPETITIONS) + 1)
            if number not in used
        ]
        for row in sorted(
            (item for item in group_rows if item.get("repetition") is None),
            key=_matrix_sort_key,
        ):
            if available:
                row["repetition"] = available.pop(0)
            else:
                # Leave an out-of-range ordinal on the row so strict
                # validation reports it as an unknown/extra combination.
                row["repetition"] = (repetitions or FORMAL_REPETITIONS) + 1

    for row in output:
        total = _row_total_tokens(row)
        if total is not None:
            row["total_tokens"] = total
        row.pop("_matrix_repetition_invalid", None)
        model = row.get("matrix_model")
        agent = row.get("matrix_agent")
        task = row.get("matrix_task")
        repetition = row.get("repetition")
        if (
            isinstance(model, str)
            and isinstance(agent, str)
            and isinstance(task, str)
            and isinstance(repetition, int)
        ):
            row["matrix_key"] = f"{model}|{agent}|{task}|{repetition}"
    return output


def validate_matrix_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    repetitions: int = FORMAL_REPETITIONS,
    expected_models: Sequence[str] = MODEL_IDS,
    expected_agents: Sequence[str] = MATRIX_AGENT_LABELS,
    expected_tasks: Sequence[str] = TASK_IDS,
) -> dict[str, Any]:
    """Validate the exact model x agent x task x repetition matrix.

    The returned document is JSON-safe and intentionally contains only row
    indices/keys, never raw result payloads.  ``complete`` is true only when
    every expected key occurs exactly once and no unknown row or duplicate
    source identity is present.
    """

    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("repetitions must be a positive integer")
    models = tuple(str(item) for item in expected_models)
    agents = tuple(str(item) for item in expected_agents)
    tasks = tuple(
        str(item).strip().rstrip("/").rsplit("/", 1)[-1] for item in expected_tasks
    )
    normalised = _normalise_matrix_rows(
        rows,
        repetitions=repetitions,
        expected_models=models,
        expected_agents=agents,
        expected_tasks=tasks,
    )
    expected_keys = {
        (model, _AGENT_LABEL_ALIASES.get(agent, agent), task, repetition)
        for model in models
        for agent in agents
        for task in tasks
        for repetition in range(1, repetitions + 1)
    }
    observed: dict[tuple[str, str, str, int], list[int]] = {}
    unknown_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    identities: dict[str, list[int]] = {}
    for index, row in enumerate(normalised):
        if row.get("kind") not in (None, "trial"):
            continue
        model = row.get("matrix_model")
        agent = row.get("matrix_agent")
        task = row.get("matrix_task")
        repetition = row.get("repetition")
        key = (
            (
                model,
                agent,
                task,
                repetition,
            )
            if (
                isinstance(model, str)
                and isinstance(agent, str)
                and isinstance(task, str)
                and isinstance(repetition, int)
            )
            else None
        )
        if key is None or key not in expected_keys:
            unknown_rows.append(
                {
                    "row_index": index,
                    "key": "|".join(str(part) for part in key) if key else None,
                    "reason": "unknown-or-malformed-combination",
                }
            )
            continue
        observed.setdefault(key, []).append(index)
        if type(row.get("passed")) is not bool:
            unresolved_rows.append(
                {
                    "row_index": index,
                    "key": "|".join(str(part) for part in key),
                    "reason": "missing-boolean-verdict",
                }
            )
        identity = _row_matrix_identity(row)
        if identity is not None:
            identities.setdefault(identity, []).append(index)

    duplicate_keys = {
        "|".join(str(part) for part in key): indices
        for key, indices in observed.items()
        if len(indices) > 1
    }
    duplicate_identities = {
        identity: indices
        for identity, indices in identities.items()
        if len(indices) > 1
    }
    missing_keys = [
        "|".join(str(part) for part in key)
        for key in sorted(expected_keys)
        if key not in observed
    ]
    complete = (
        not missing_keys
        and not duplicate_keys
        and not duplicate_identities
        and not unknown_rows
        and not unresolved_rows
    )
    return {
        "complete": complete,
        "expected_trials": len(expected_keys),
        "observed_trials": sum(len(indices) for indices in observed.values()),
        "expected_keys": [
            "|".join(str(part) for part in key) for key in sorted(expected_keys)
        ],
        "observed_keys": [
            "|".join(str(part) for part in key) for key in sorted(observed)
        ],
        "missing_keys": missing_keys,
        "duplicate_keys": duplicate_keys,
        "duplicate_identities": duplicate_identities,
        "unknown_rows": unknown_rows,
        "unresolved_rows": unresolved_rows,
        "normalised_rows": normalised,
    }


def _require_complete_matrix(
    rows: Iterable[Mapping[str, Any]],
    *,
    repetitions: int = FORMAL_REPETITIONS,
    expected_models: Sequence[str] = MODEL_IDS,
    expected_agents: Sequence[str] = MATRIX_AGENT_LABELS,
    expected_tasks: Sequence[str] = TASK_IDS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation = validate_matrix_rows(
        rows,
        repetitions=repetitions,
        expected_models=expected_models,
        expected_agents=expected_agents,
        expected_tasks=expected_tasks,
    )
    if not validation["complete"]:
        problems = []
        if validation["missing_keys"]:
            problems.append(f"missing={len(validation['missing_keys'])}")
        if validation["duplicate_keys"]:
            problems.append(f"duplicate={len(validation['duplicate_keys'])}")
        if validation["duplicate_identities"]:
            problems.append(
                f"duplicate_identity={len(validation['duplicate_identities'])}"
            )
        if validation["unknown_rows"]:
            problems.append(f"unknown={len(validation['unknown_rows'])}")
        if validation.get("unresolved_rows"):
            problems.append(f"unresolved={len(validation['unresolved_rows'])}")
        detail = ", ".join(problems) or "incomplete"
        raise ValueError(
            f"formal matrix incomplete ({detail}); "
            f"observed {validation['observed_trials']} of {validation['expected_trials']} keys"
        )
    return validation["normalised_rows"], validation


def render_matrix_markdown(summary: Mapping[str, Mapping[str, Any]]) -> str:
    headers = (
        "model",
        "agent",
        "passed/evaluable",
        "accuracy (95% Wilson)",
        "agent time median [IQR]",
        "total time median [IQR]",
        "input tokens median [IQR]",
        "output tokens median [IQR]",
        "cache tokens median [IQR]",
        "total tokens median [IQR]",
        "model/tool calls median",
        "infrastructure/setup failures",
    )
    lines = [
        "# Terminal-Bench 2.1 exploratory comparison",
        "",
        "8 个 Terminal-Bench 2.1 任务、每项 3 次的探索性实验；基础设施/setup failure 不计入正确率。",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for item in sorted(
        summary.values(),
        key=lambda value: (str(value.get("model")), str(value.get("agent"))),
    ):
        interval = item.get("accuracy_wilson_95") or {}
        accuracy = item.get("accuracy")
        q = item.get("quartiles") or {}
        agent_q = q.get("agent_elapsed_seconds") or {}
        total_q = q.get("total_elapsed_seconds") or {}
        input_token_q = q.get("input_tokens") or {}
        output_token_q = q.get("output_tokens") or {}
        cache_token_q = q.get("cache_tokens") or {}
        total_token_q = q.get("total_tokens") or {}
        model_requests = item.get("median_model_requests")
        tool_calls = item.get("median_tool_calls")
        call_summary = (
            f"{model_requests if model_requests is not None else 'n/a'} / "
            f"{tool_calls if tool_calls is not None else 'n/a'}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(item.get("model", "")),
                    str(item.get("agent", "")),
                    f"{item.get('n_passed', 0)}/{item.get('n_evaluable_trials', 0)}",
                    "n/a"
                    if accuracy is None
                    else f"{accuracy:.1%} ({interval.get('low', 0):.1%}-{interval.get('high', 0):.1%})",
                    _format_stat(agent_q),
                    _format_stat(total_q),
                    _format_stat(input_token_q),
                    _format_stat(output_token_q),
                    _format_stat(cache_token_q),
                    _format_stat(total_token_q),
                    call_summary,
                    str(item.get("n_setup_failures", 0)),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _format_stat(value: Mapping[str, Any]) -> str:
    median = value.get("median")
    if median is None:
        return "n/a"
    q1 = value.get("q1")
    q3 = value.get("q3")
    if not isinstance(q1, (int, float)):
        q1 = median
    if not isinstance(q3, (int, float)):
        q3 = median
    return f"{float(median):g} [{float(q1):g}-{float(q3):g}]"


def write_matrix_csv(path: Path, summary: Mapping[str, Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for item in summary.values() for key in item})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in sorted(
            summary.values(),
            key=lambda value: (str(value.get("model")), str(value.get("agent"))),
        ):
            flat = dict(item)
            flat["accuracy_wilson_95"] = json.dumps(
                flat.get("accuracy_wilson_95"), ensure_ascii=False
            )
            flat["quartiles"] = json.dumps(flat.get("quartiles"), ensure_ascii=False)
            writer.writerow(flat)
    os.chmod(path, 0o600)
    return path


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    data = kind + payload
    return (
        struct.pack(">I", len(payload))
        + data
        + struct.pack(">I", binascii.crc32(data) & 0xFFFFFFFF)
    )


_PNG_PANEL_LABELS: tuple[str, ...] = (
    "accuracy",
    "elapsed time",
    "total tokens",
    "model/tool calls",
)


def _png_text_chunk(keyword: str, value: str) -> bytes:
    """Build a standards-compliant uncompressed PNG ``tEXt`` chunk.

    The report image is intentionally dependency-free.  Keeping the metric
    names in PNG metadata makes the four panels identifiable to image viewers
    and automated archives even though the raster renderer does not depend on
    a font library.
    """

    # PNG keywords are Latin-1 strings, 1-79 bytes, with no control characters
    # or NUL.  Internal keywords are ASCII; sanitizing the arguments keeps this
    # helper safe for callers that provide arbitrary group labels.
    safe_keyword = "".join(
        char for char in str(keyword) if 32 <= ord(char) <= 126
    ).encode("ascii", "ignore")[:79]
    if not safe_keyword:
        safe_keyword = b"Comment"
    safe_value = str(value).replace("\x00", "").replace("\r", " ").replace("\n", " ")
    value_bytes = safe_value.encode("latin-1", "replace")
    return _png_chunk(b"tEXt", safe_keyword + b"\x00" + value_bytes)


def _simple_bar_png(
    values: Sequence[float],
    labels: Sequence[str],
    path: Path,
    *,
    infrastructure_rates: Sequence[float] = (),
) -> Path:
    """Write a dependency-free RGB PNG used for the four-panel report.

    ``values`` may be the historical flat sequence (one value per panel) or a
    sequence of per-panel group sequences.  The latter is what the formal
    report uses: every panel then shows all model/agent groups instead of an
    opaque average of them.
    """

    width, height = 1200, 800
    pixels = bytearray([255, 255, 255] * width * height)

    raw_values: list[Any] = list(values)
    if (
        raw_values
        and isinstance(raw_values[0], Sequence)
        and not isinstance(raw_values[0], (str, bytes, bytearray))
    ):
        panel_values: list[list[float]] = [
            [
                float(item)
                for item in panel
                if isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
            ]
            for panel in raw_values
            if isinstance(panel, Sequence)
        ]
    else:
        flat = [
            float(item)
            for item in raw_values
            if isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
        ]
        panel_values = [[item] for item in flat]
    safe_infrastructure_rates = [
        max(0.0, min(100.0, float(item)))
        if isinstance(item, (int, float))
        and not isinstance(item, bool)
        and math.isfinite(float(item))
        else 0.0
        for item in infrastructure_rates
    ]

    def rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        for y in range(y0, y1):
            start = (y * width + x0) * 3
            pixels[start : start + (x1 - x0) * 3] = bytes(color) * (x1 - x0)

    panels = 4
    panel_w = width // panels
    colors = ((35, 114, 190), (50, 150, 90), (220, 140, 35), (170, 70, 120))
    for panel in range(panels):
        left = panel * panel_w + 35
        right = (panel + 1) * panel_w - 35
        rect(left, 70, right, 72, (40, 40, 40))
        rect(left, height - 80, right, height - 78, (40, 40, 40))
        current = panel_values[panel] if panel < len(panel_values) else []
        # Keep an all-zero panel renderable.  A zero denominator used to make
        # artifact generation fail precisely when a condition had no passes
        # or no observed failures.
        max_value = max(max(current or [0.0]), 1.0)
        if current:
            bar_w = max(4, (right - left) // len(current))
            for index, value in enumerate(current):
                scaled = int(
                    max(0.0, min(1.0, float(value) / max_value)) * (height - 180)
                )
                x0 = left + index * bar_w + 3
                x1 = min(right, x0 + bar_w - 6)
                rect(x0, height - 80 - scaled, x1, height - 80, colors[panel])
                if panel == 0 and index < len(safe_infrastructure_rates):
                    infrastructure_scaled = int(
                        safe_infrastructure_rates[index] / 100.0 * (height - 180)
                    )
                    if infrastructure_scaled:
                        marker_width = max(3, (x1 - x0) // 4)
                        rect(
                            x1 - marker_width,
                            height - 80 - infrastructure_scaled,
                            x1,
                            height - 80,
                            (190, 45, 45),
                        )
        # Panel separators make the four requested metrics visually distinct
        if panel:
            rect(panel * panel_w, 0, panel * panel_w + 2, height, (210, 210, 210))
    raw_rows = b"".join(
        b"\x00" + bytes(pixels[y * width * 3 : (y + 1) * width * 3])
        for y in range(height)
    )
    panel_metadata = [
        _png_text_chunk("Title", "Terminal-Bench 2.1 exploratory comparison"),
        *(
            _png_text_chunk(f"panel_{index}", label)
            for index, label in enumerate(_PNG_PANEL_LABELS)
        ),
        _png_text_chunk(
            "groups",
            " | ".join(str(label) for label in labels),
        ),
        _png_text_chunk(
            "infrastructure_marker",
            "red bars in the accuracy panel show infrastructure/setup failures",
        ),
        _png_text_chunk(
            "infrastructure_rates_percent",
            " | ".join(
                f"{label}={safe_infrastructure_rates[index]:g}"
                for index, label in enumerate(labels)
                if index < len(safe_infrastructure_rates)
            ),
        ),
    ]
    document = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + b"".join(panel_metadata)
        + _png_chunk(b"IDAT", __import__("zlib").compress(raw_rows, 6))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(document)
    os.chmod(path, 0o600)
    return path


def write_matrix_artifacts(
    output_dir: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    require_complete: bool = False,
    expected_trials: int | None = None,
    repetitions: int = FORMAL_REPETITIONS,
    expected_models: Sequence[str] = MODEL_IDS,
    expected_agents: Sequence[str] = MATRIX_AGENT_LABELS,
    expected_tasks: Sequence[str] = TASK_IDS,
    image_manifest: Mapping[str, Any] | None = None,
) -> dict[str, str | int | bool]:
    """Write JSON, CSV, Markdown and PNG summaries after matrix completion."""

    validation = validate_matrix_rows(
        rows,
        repetitions=repetitions,
        expected_models=expected_models,
        expected_agents=expected_agents,
        expected_tasks=expected_tasks,
    )
    normalised = validation["normalised_rows"]
    if image_manifest is not None:
        normalised = [
            _attach_image_manifest_row(row, image_manifest) for row in normalised
        ]
    expected_key_count = int(validation["expected_trials"])
    if expected_trials is not None and expected_trials != expected_key_count:
        raise ValueError(
            "expected_trials does not match the configured matrix dimensions: "
            f"{expected_trials} != {expected_key_count}"
        )
    if require_complete and not validation["complete"]:
        problems = []
        if validation["missing_keys"]:
            problems.append(f"missing={len(validation['missing_keys'])}")
        if validation["duplicate_keys"]:
            problems.append(f"duplicate={len(validation['duplicate_keys'])}")
        if validation["duplicate_identities"]:
            problems.append(
                f"duplicate_identity={len(validation['duplicate_identities'])}"
            )
        if validation["unknown_rows"]:
            problems.append(f"unknown={len(validation['unknown_rows'])}")
        if validation.get("unresolved_rows"):
            problems.append(f"unresolved={len(validation['unresolved_rows'])}")
        raise ValueError(
            "formal matrix incomplete ("
            + ", ".join(problems)
            + f"); observed {validation['observed_trials']} of {expected_key_count} keys"
        )
    summary = aggregate_matrix_rows(
        normalised,
        expected_trials_per_group=len(expected_tasks) * repetitions,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    _write_json(
        json_path,
        {
            "rows": normalised,
            "groups": summary,
            "image_manifest": _image_manifest_public(image_manifest),
        },
    )
    csv_path = write_matrix_csv(output_dir / "summary.csv", summary)
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(render_matrix_markdown(summary), encoding="utf-8")
    os.chmod(markdown_path, 0o600)
    # Four panel values are accuracy, total time, total token and call volume.
    # Keep one bar per model/agent group in every panel. Infrastructure failures
    # are excluded from accuracy and get a separate red marker in panel one.
    groups = [
        summary[key]
        for key in sorted(
            summary,
            key=lambda item: (
                str(summary[item].get("model")),
                str(summary[item].get("agent")),
            ),
        )
    ]
    group_labels = [
        f"{item.get('model', '')}::{item.get('agent', '')}" for item in groups
    ]
    panel_values = [
        [
            float(item["accuracy"]) * 100 if item.get("accuracy") is not None else 0.0
            for item in groups
        ],
        [float(item["median_total_elapsed_seconds"] or 0.0) for item in groups],
        [float(item["median_total_tokens"] or 0.0) for item in groups],
        [
            float(item["median_model_requests"] or 0.0)
            + float(item["median_tool_calls"] or 0.0)
            for item in groups
        ],
    ]
    infrastructure_rates = [
        100.0
        * float(item.get("infrastructure_failures") or 0)
        / float(item.get("expected_trials") or 1)
        for item in groups
    ]
    png_path = _simple_bar_png(
        panel_values,
        group_labels,
        output_dir / "summary.png",
        infrastructure_rates=infrastructure_rates,
    )
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "png": str(png_path),
        "n_rows": len(normalised),
        "complete": bool(validation["complete"]),
        "status": "complete" if validation["complete"] else "incomplete",
        "group_labels": group_labels,
        "matrix_validation": {
            key: value for key, value in validation.items() if key != "normalised_rows"
        },
    }


def _finish_report(
    plan: dict[str, Any],
    *,
    jobs_dir: Path,
    secrets: tuple[str, ...] = (),
    outcomes: list[dict[str, Any]] | None = None,
    image_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect sanitized Harbor files into the final comparison report."""

    plan["status"] = "finished"
    if outcomes is not None:
        plan["outcomes"] = outcomes
    plan["scrubbed_artifact_files"] = _scrub_generated_artifacts(
        jobs_dir, secrets=secrets
    )
    result_files = _collect_result_files(jobs_dir)
    plan["result_files"] = [str(path) for path in result_files]
    result_summaries = [
        _summarize_result_file(path, secrets=secrets, image_manifest=image_manifest)
        for path in result_files
    ]
    plan["result_summaries"] = result_summaries
    plan["agent_comparison"] = _aggregate_trial_rows(result_summaries)
    return plan


def collect_existing_results(
    *,
    model: str,
    dataset: str,
    jobs_dir: Path,
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: dict[str, str] | None = None,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Collect a completed Harbor job without launching another experiment."""

    source = dict(os.environ if environ is None else environ)
    job_files = []
    for path in _collect_result_files(jobs_dir):
        try:
            with path.open(encoding="utf-8") as handle:
                document = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, Mapping):
            continue
        stats = document.get("stats")
        if not isinstance(stats, Mapping) or "n_total_trials" not in document:
            continue
        job_files.append(path)
        if (
            not document.get("finished_at")
            or stats.get("n_running_trials", 0)
            or stats.get("n_pending_trials", 0)
        ):
            raise ValueError("Harbor jobs are still running; collect after completion")
    if not job_files:
        raise ValueError("no completed Harbor job result was found")
    plan = make_plan(
        model=model,
        dataset=dataset,
        jobs_dir=jobs_dir,
        harbor_bin=harbor_bin,
        harbor_sudo=harbor_sudo,
        environ=source,
        image_manifest=image_manifest,
    )
    return _finish_report(
        plan,
        jobs_dir=jobs_dir,
        secrets=_credential_values(source),
        image_manifest=_image_manifest_for_use(image_manifest, dataset=dataset),
    )


def run_experiment(
    *,
    model: str,
    dataset: str,
    jobs_dir: Path,
    harbor_bin: str,
    execute: bool,
    harbor_sudo: bool = False,
    environ: dict[str, str] | None = None,
    cwd: Path | None = None,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Create a plan and optionally launch both one-shot Harbor jobs."""

    source = dict(os.environ if environ is None else environ)
    route_source = route_environment(model, source)
    root = (Path.cwd() if cwd is None else cwd).resolve()
    plan = make_plan(
        model=model,
        dataset=dataset,
        jobs_dir=jobs_dir,
        harbor_bin=harbor_bin,
        harbor_sudo=harbor_sudo,
        environ=source,
        image_manifest=image_manifest,
    )
    if not execute:
        return plan

    validated_manifest = _image_manifest_for_use(
        image_manifest,
        dataset=dataset,
        require_ready=image_manifest is not None,
    )

    missing = check_credentials(route_source)
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"required environment variable(s) are missing: {names}")
    provider = _provider_for_environment(route_source)
    if provider is Provider.CUSTOM:
        base_url = route_source.get("CODING_AGENT_BASE_URL")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError(
                "custom provider requires CODING_AGENT_BASE_URL for Harbor runs"
            )

    plan["status"] = "executing"
    outcomes: list[dict[str, Any]] = []
    process_environment = _harbor_process_environment(route_source)
    redaction_secrets = _credential_values(route_source)
    for command in _experiment_commands(
        model=model,
        dataset=dataset,
        jobs_dir=jobs_dir,
        harbor_bin=harbor_bin,
        harbor_sudo=harbor_sudo,
        environ=route_source,
    ):
        log_path = jobs_dir / (
            "course-coding-agent.log"
            if command.agent == COURSE_AGENT
            else "opencode.log"
        )
        return_code = _run_one(
            command,
            cwd=root,
            environment=process_environment,
            log_path=log_path,
            redaction_secrets=redaction_secrets,
        )
        outcomes.append(
            {
                "agent": command.agent,
                "return_code": return_code,
                "log": str(log_path),
            }
        )
    return _finish_report(
        plan,
        jobs_dir=jobs_dir,
        secrets=redaction_secrets,
        outcomes=outcomes,
        image_manifest=validated_manifest,
    )


def formal_required_credentials(
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return missing names for all three fixed routes, without values."""

    source = dict(os.environ if environ is None else environ)
    required = [
        PROVIDER_KEY_ENV_NAMES[Provider.DEEPSEEK],
        PROVIDER_KEY_ENV_NAMES[Provider.GLM],
        PROVIDER_KEY_ENV_NAMES[Provider.CUSTOM],
    ]
    return tuple(
        name
        for name in required
        if not isinstance(source.get(name), str) or not source[name].strip()
    )


def probe_model_route(
    model: str,
    *,
    environ: Mapping[str, str] | None = None,
    effort: str = "high",
    client_factory: Any | None = None,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Run one minimal native reasoning probe and return safe metadata."""

    from coding_agent.model import OpenAICompatibleModelClient

    if model not in MODEL_ROUTES:
        raise ValueError(f"model is not one of the fixed experiment routes: {model}")
    route = route_environment(model, environ)
    route_meta = MODEL_ROUTES[model]
    key_name = route.get("CODING_AGENT_KEY_ENV") or route_meta["key_env"]
    key = route.get(key_name)
    if not isinstance(key, str) or not key.strip():
        return {
            "model": model,
            "provider": route_meta["provider"],
            "key_env": key_name,
            "requested_effort": effort,
            "status": "setup_failure",
            "reason": f"missing {key_name}",
        }
    factory = client_factory or OpenAICompatibleModelClient
    try:
        client = _invoke_probe_factory(
            factory,
            {
                "model": model,
                "api_key": key,
                "base_url": route_meta.get("base_url"),
                "timeout_seconds": timeout_seconds,
                "reasoning_effort": effort,
            },
        )
        capability = client.probe_reasoning_effort(
            effort,
            timeout_seconds=timeout_seconds,
        )
        to_dict = getattr(capability, "to_dict", None)
        if callable(to_dict):
            raw_result = to_dict()
        elif isinstance(capability, Mapping):
            raw_result = capability
        else:
            raise TypeError("reasoning probe returned an invalid capability")
        if not isinstance(raw_result, Mapping):
            raise TypeError("reasoning probe returned an invalid capability")
        # Capability details are written into the plan/report.  Redact the
        # selected key even when a provider accidentally echoes it in an error
        # detail or a custom native value; no raw probe object crosses this
        # persistence boundary.
        safe_result = redact(raw_result, secrets=(key,))
        result = dict(safe_result) if isinstance(safe_result, Mapping) else {}
        detail = result.get("detail")
        if isinstance(detail, str) and len(detail) > 1000:
            result["detail"] = detail[:1000]
    except Exception as exc:  # noqa: BLE001 - preflight boundary
        result = {
            "status": "error",
            "requested_effort": effort,
            "error_type": type(exc).__name__,
            "detail": _redact_probe_detail(exc, key),
        }
    result.update(
        {
            "model": model,
            "provider": route_meta["provider"],
            "key_env": key_name,
            "base_url": route_meta.get("base_url"),
        }
    )
    return result


def _redact_probe_detail(exc: Exception, secret: str | None) -> str:
    """Return bounded probe diagnostics without credentials or giant payloads."""

    detail = _redact_runner_text(str(exc), (secret,) if secret else ())
    # Provider exception strings occasionally embed full JSON responses. Keep
    # enough context to classify a setup failure without persisting the body.
    detail = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1[REDACTED]",
        detail,
    )
    return detail[:1000] or type(exc).__name__


def run_smoke_matrix(
    *,
    dataset: str = DATASET,
    jobs_dir: Path = Path(".harbor-runs/smoke"),
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: Mapping[str, str] | None = None,
    execute: bool = False,
    probe_reasoning: bool = True,
    reasoning_effort: str = "high",
    probe_client_factory: Any | None = None,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Plan or run the six preflight smoke trials on ``fix-code-vulnerability``."""

    source = dict(os.environ if environ is None else environ)
    plan = make_smoke_plan(
        dataset=dataset,
        jobs_dir=jobs_dir,
        harbor_bin=harbor_bin,
        harbor_sudo=harbor_sudo,
        environ=source,
        reasoning_effort=reasoning_effort if probe_reasoning else None,
        image_manifest=image_manifest,
    )
    if execute:
        validated_manifest = _require_execution_image_manifest(
            image_manifest,
            dataset=dataset,
        )
        missing = formal_required_credentials(source)
        if missing:
            raise ValueError(
                "required environment variable(s) are missing: " + ", ".join(missing)
            )
    should_probe = probe_reasoning and (execute or probe_client_factory is not None)
    if should_probe:
        probes = [
            probe_model_route(
                model,
                environ=source,
                effort=reasoning_effort,
                client_factory=probe_client_factory,
            )
            for model in MODEL_IDS
        ]
    elif probe_reasoning:
        probes = [
            {
                "model": model,
                "requested_effort": reasoning_effort,
                "status": "not_run",
                "reason": "deferred until smoke execution",
            }
            for model in MODEL_IDS
        ]
    else:
        probes = []
    if execute and probe_reasoning:
        _validate_execution_probes(probes)
    # Rebuild the command list after probing so each route receives the exact
    # native field/value (or no reasoning field at all).  The probe records are
    # copied into the plan independently and never contain credential values.
    if probe_reasoning:
        plan = make_smoke_plan(
            dataset=dataset,
            jobs_dir=jobs_dir,
            harbor_bin=harbor_bin,
            harbor_sudo=harbor_sudo,
            environ=source,
            reasoning_effort=reasoning_effort,
            reasoning_capabilities=_capability_map(probes),
            image_manifest=image_manifest,
        )
    else:
        plan = make_smoke_plan(
            dataset=dataset,
            jobs_dir=jobs_dir,
            harbor_bin=harbor_bin,
            harbor_sudo=harbor_sudo,
            environ=source,
            reasoning_effort=None,
            image_manifest=image_manifest,
        )
    plan["reasoning_probes"] = probes
    if not execute:
        return plan
    plan["status"] = "executing"
    outcomes: list[dict[str, Any]] = []
    secrets = _credential_values(source)
    # Sanitize leftovers before taking the snapshot. Otherwise redacting an
    # old file would make it look like a newly produced trial on a rerun.
    scrubbed_artifact_files = _scrub_generated_artifacts(jobs_dir, secrets=secrets)
    before_results = _result_file_snapshot(jobs_dir)
    for index, command_argv in enumerate(plan["commands"]):
        command = ExperimentCommand(
            agent=str(command_argv[command_argv.index("--agent") + 1]),
            argv=tuple(command_argv),
        )
        model = _fixed_model_id_from_agent_model(
            str(command_argv[command_argv.index("--model") + 1])
        )
        route = _route_without_reasoning(model, source)
        log_path = jobs_dir / f"{model}__{index}.log"
        code = _run_one(
            command,
            cwd=Path.cwd(),
            environment=_harbor_process_environment(route, strip_reasoning=True),
            log_path=log_path,
            redaction_secrets=secrets,
        )
        outcomes.append(
            {
                "model": model,
                "agent": command.agent,
                "return_code": code,
                "log": str(log_path),
            }
        )
    scrubbed_artifact_files += _scrub_generated_artifacts(jobs_dir, secrets=secrets)
    plan["outcomes"] = outcomes
    plan["status"] = "finished"
    result_files = _changed_result_files(jobs_dir, before_results)
    plan["scrubbed_artifact_files"] = scrubbed_artifact_files
    plan["result_files"] = [str(path) for path in result_files]
    plan["result_summaries"] = [
        _summarize_result_file(path, secrets=secrets, image_manifest=validated_manifest)
        for path in result_files
    ]
    plan["smoke_gate"] = validate_smoke_report(
        plan,
        secrets=secrets,
        require_probes=probe_reasoning,
        image_manifest=validated_manifest,
    )
    return plan


def run_formal_experiment(
    *,
    dataset: str = DATASET,
    jobs_dir: Path = Path(".harbor-runs/formal"),
    harbor_bin: str = "harbor",
    harbor_sudo: bool = False,
    environ: Mapping[str, str] | None = None,
    execute: bool = False,
    probe_reasoning: bool = True,
    reasoning_effort: str = "high",
    allow_incomplete: bool = False,
    probe_client_factory: Any | None = None,
    round_strategy: str | None = None,
    ablation_report: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    smoke_report: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    image_manifest: Mapping[str, Any] | str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run the six-command formal matrix and emit complete-report metadata.

    Offline plans may use the documented efficiency-20 default (or an explicit
    named strategy) before the ablation has been run.  An actual formal
    execution must instead be tied to a completed ablation decision so the
    round budget cannot be chosen by accident.  Executions also require a
    completed six-trial smoke report for ``fix-code-vulnerability``; verifier
    failures in that report are retained as observations, while
    setup/infrastructure failures stop the formal matrix before any Harbor job
    is launched.
    """

    source = dict(os.environ if environ is None else environ)
    ablation_decision: dict[str, Any] | None = None
    if ablation_report is not None:
        ablation_decision = formal_strategy_from_ablation(
            ablation_report,
            # ``--allow-incomplete`` is an offline artifact/plan convenience;
            # an executable matrix must never inherit an unfixed round budget.
            allow_incomplete=allow_incomplete and not execute,
        )
        report_strategy = str(ablation_decision["selected_strategy"])
        if round_strategy is not None and round_strategy != report_strategy:
            raise ValueError(
                "round_strategy conflicts with the ablation report's fixed choice"
            )
        selected_strategy = report_strategy
    else:
        selected_strategy = round_strategy or FORMAL_DEFAULT_STRATEGY
        # Validate early so a dry plan fails with a focused configuration error.
        _resolve_round_strategy(selected_strategy)

    plan = make_formal_plan(
        dataset=dataset,
        jobs_dir=jobs_dir,
        harbor_bin=harbor_bin,
        harbor_sudo=harbor_sudo,
        environ=source,
        reasoning_effort=reasoning_effort if probe_reasoning else None,
        round_strategy=selected_strategy,
        image_manifest=image_manifest,
    )
    if smoke_report is None and not execute:
        smoke_gate: dict[str, Any] = {
            "schema_version": "terminal-bench-smoke-gate/v1",
            "status": "not_checked",
            "can_proceed": None,
            "required": False,
            "task": SMOKE_TASK,
            "report_path": None,
            "blocking_reasons": [],
        }
    else:
        smoke_gate = validate_smoke_report(
            smoke_report,
            secrets=_credential_values(source),
            require_probes=probe_reasoning,
            image_manifest=image_manifest,
        )
        smoke_gate["required"] = bool(execute)
    plan["smoke_gate"] = smoke_gate
    if execute and not smoke_gate.get("can_proceed"):
        reasons = smoke_gate.get("blocking_reasons") or ["smoke gate failed"]
        detail = "; ".join(str(item) for item in reasons[:4])
        raise ValueError(f"formal smoke gate blocked: {detail}")
    if execute:
        validated_manifest = _require_execution_image_manifest(
            image_manifest,
            dataset=dataset,
        )
        if ablation_report is None:
            raise ValueError(
                "formal execution requires a completed ablation report supplied "
                "via --ablation-report"
            )
        if allow_incomplete:
            raise ValueError(
                "--allow-incomplete is only supported for offline formal plans"
            )
        missing = formal_required_credentials(source)
        if missing:
            raise ValueError(
                "required environment variable(s) are missing: " + ", ".join(missing)
            )
    should_probe = probe_reasoning and (execute or probe_client_factory is not None)
    if should_probe:
        probes = [
            probe_model_route(
                model,
                environ=source,
                effort=reasoning_effort,
                client_factory=probe_client_factory,
            )
            for model in MODEL_IDS
        ]
    elif probe_reasoning:
        probes = [
            {
                "model": model,
                "requested_effort": reasoning_effort,
                "status": "not_run",
                "reason": "deferred until formal execution",
            }
            for model in MODEL_IDS
        ]
    else:
        probes = []
    if execute and probe_reasoning:
        _validate_execution_probes(probes)
    if probe_reasoning:
        plan = make_formal_plan(
            dataset=dataset,
            jobs_dir=jobs_dir,
            harbor_bin=harbor_bin,
            harbor_sudo=harbor_sudo,
            environ=source,
            reasoning_effort=reasoning_effort,
            reasoning_capabilities=_capability_map(probes),
            round_strategy=selected_strategy,
            image_manifest=image_manifest,
        )
    else:
        plan = make_formal_plan(
            dataset=dataset,
            jobs_dir=jobs_dir,
            harbor_bin=harbor_bin,
            harbor_sudo=harbor_sudo,
            environ=source,
            reasoning_effort=None,
            round_strategy=selected_strategy,
            image_manifest=image_manifest,
        )
    plan["reasoning_probes"] = probes
    if ablation_decision is not None:
        plan["ablation_decision"] = ablation_decision
    # Reattach the gate after rebuilding the plan with probe-derived kwargs.
    # This keeps the audit record present regardless of whether probing is
    # enabled, without copying the source smoke payload into the report.
    plan["smoke_gate"] = smoke_gate
    if not execute:
        return plan
    plan["status"] = "executing"
    outcomes: list[dict[str, Any]] = []
    secrets = _credential_values(source)
    # Sanitize any prior run before taking the result snapshot. This keeps a
    # rerun from mixing historical trials into the current formal matrix.
    scrubbed_artifact_files = _scrub_generated_artifacts(jobs_dir, secrets=secrets)
    before_results = _result_file_snapshot(jobs_dir)
    for index, command_argv in enumerate(plan["commands"]):
        command = ExperimentCommand(
            agent=str(command_argv[command_argv.index("--agent") + 1]),
            argv=tuple(command_argv),
        )
        model = _fixed_model_id_from_agent_model(
            str(command_argv[command_argv.index("--model") + 1])
        )
        route = _route_without_reasoning(model, source)
        log_path = jobs_dir / f"{model}__{index}.log"
        code = _run_one(
            command,
            cwd=Path.cwd(),
            environment=_harbor_process_environment(route, strip_reasoning=True),
            log_path=log_path,
            redaction_secrets=secrets,
        )
        outcomes.append(
            {
                "model": model,
                "agent": command.agent,
                "return_code": code,
                "log": str(log_path),
            }
        )
    scrubbed_artifact_files += _scrub_generated_artifacts(jobs_dir, secrets=secrets)
    plan["outcomes"] = outcomes
    plan["status"] = "finished"
    plan["scrubbed_artifact_files"] = scrubbed_artifact_files
    result_files = _changed_result_files(jobs_dir, before_results)
    plan["result_files"] = [str(path) for path in result_files]
    summaries = [
        _summarize_result_file(path, secrets=secrets, image_manifest=validated_manifest)
        for path in result_files
    ]
    plan["result_summaries"] = summaries
    trial_rows = [row for row in summaries if row.get("kind") == "trial"]
    validation = validate_matrix_rows(
        trial_rows,
        repetitions=int(plan["repetitions_per_task"]),
        expected_models=tuple(plan["models"]),
        expected_agents=tuple(plan["agents"]),
        expected_tasks=tuple(plan["tasks"]),
    )
    plan["matrix_validation"] = {
        key: value for key, value in validation.items() if key != "normalised_rows"
    }
    # Aggregate canonicalized rows so the report's groups cannot accidentally
    # split on Harbor's import-path/runtime-name spelling differences.
    plan["matrix_summary"] = aggregate_matrix_rows(validation["normalised_rows"])
    complete = bool(validation["complete"])
    plan["matrix_complete"] = complete
    if (
        validation["duplicate_keys"]
        or validation["duplicate_identities"]
        or validation["unknown_rows"]
    ) or (not complete and not allow_incomplete):
        problems = []
        if validation["missing_keys"]:
            problems.append(f"missing={len(validation['missing_keys'])}")
        if validation["duplicate_keys"]:
            problems.append(f"duplicate={len(validation['duplicate_keys'])}")
        if validation["duplicate_identities"]:
            problems.append(
                f"duplicate_identity={len(validation['duplicate_identities'])}"
            )
        if validation["unknown_rows"]:
            problems.append(f"unknown={len(validation['unknown_rows'])}")
        raise ValueError(
            "formal matrix incomplete ("
            + ", ".join(problems)
            + f"); observed {validation['observed_trials']} of {validation['expected_trials']} keys"
        )
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run the eight-task, single-trial Terminal-Bench 2.1 "
            "exploratory comparison."
        )
    )
    parser.add_argument("--execute", action="store_true", help="launch Harbor jobs")
    parser.add_argument(
        "--formal",
        action="store_true",
        help="use the fixed 3-model x 2-agent x 3-repeat matrix",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one fix-code-vulnerability smoke trial per combination",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="plan the Course-agent 20/20-efficient/30-turn ablation",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="summarize existing Harbor jobs without launching another run",
    )
    parser.add_argument(
        "--model", help="exact model ID; defaults to CODING_AGENT_MODEL"
    )
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument(
        "--image-manifest",
        type=Path,
        help=(
            "pinned OpenCode image manifest; dry plans validate its shape, and "
            "execution requires every task image to be ready"
        ),
    )
    parser.add_argument("--jobs-dir", type=Path, default=Path(".harbor-runs"))
    parser.add_argument("--harbor-bin", default="harbor")
    parser.add_argument(
        "--harbor-sudo",
        action="store_true",
        help="invoke Harbor through passwordless sudo -n (for Docker-enabled hosts)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the redacted plan/report JSON to this path",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "max"),
        default="high",
    )
    parser.add_argument(
        "--round-strategy",
        choices=tuple(ROUND_STRATEGIES),
        help=(
            "Course-agent round policy for the formal matrix; defaults to the "
            "efficiency-20 policy unless --ablation-report selects another"
        ),
    )
    parser.add_argument(
        "--ablation-report",
        type=Path,
        help=(
            "completed ablation JSON whose fixed_formal_configuration controls "
            "the formal Course-agent round policy"
        ),
    )
    parser.add_argument(
        "--smoke-report",
        type=Path,
        help=(
            "completed fix-code-vulnerability smoke JSON required before a "
            "formal execution"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="write JSON/CSV/Markdown/PNG matrix artifacts here",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow artifact generation before all formal trials finish",
    )
    arguments = parser.parse_args(argv)
    if arguments.execute and arguments.collect_only:
        parser.error("--execute and --collect-only are mutually exclusive")
    if (
        sum(
            bool(value)
            for value in (arguments.formal, arguments.smoke, arguments.ablation)
        )
        > 1
    ):
        parser.error("--formal, --smoke and --ablation are mutually exclusive")
    if arguments.ablation_report is not None and not arguments.formal:
        parser.error("--ablation-report requires --formal")
    if arguments.smoke_report is not None and not arguments.formal:
        parser.error("--smoke-report requires --formal")
    if arguments.round_strategy is not None and not arguments.formal:
        parser.error("--round-strategy requires --formal")
    model = arguments.model or os.environ.get("CODING_AGENT_MODEL")
    if not model:
        if arguments.execute and not (
            arguments.formal or arguments.smoke or arguments.ablation
        ):
            print("missing CODING_AGENT_MODEL", file=sys.stderr)
            return 2
        model = "<CODING_AGENT_MODEL>"
    try:
        if arguments.formal:
            report = run_formal_experiment(
                dataset=arguments.dataset,
                jobs_dir=arguments.jobs_dir,
                harbor_bin=arguments.harbor_bin,
                harbor_sudo=arguments.harbor_sudo,
                execute=arguments.execute,
                reasoning_effort=arguments.reasoning_effort,
                allow_incomplete=arguments.allow_incomplete,
                round_strategy=arguments.round_strategy,
                ablation_report=arguments.ablation_report,
                smoke_report=arguments.smoke_report,
                image_manifest=arguments.image_manifest,
            )
        elif arguments.smoke:
            report = run_smoke_matrix(
                dataset=arguments.dataset,
                jobs_dir=arguments.jobs_dir,
                harbor_bin=arguments.harbor_bin,
                harbor_sudo=arguments.harbor_sudo,
                execute=arguments.execute,
                reasoning_effort=arguments.reasoning_effort,
                image_manifest=arguments.image_manifest,
            )
        elif arguments.ablation:
            report = run_ablation_experiment(
                dataset=arguments.dataset,
                jobs_dir=arguments.jobs_dir,
                harbor_bin=arguments.harbor_bin,
                harbor_sudo=arguments.harbor_sudo,
                execute=arguments.execute,
                reasoning_effort=arguments.reasoning_effort,
                allow_incomplete=arguments.allow_incomplete or not arguments.execute,
                image_manifest=arguments.image_manifest,
            )
        elif arguments.collect_only:
            report = collect_existing_results(
                model=model,
                dataset=arguments.dataset,
                jobs_dir=arguments.jobs_dir,
                harbor_bin=arguments.harbor_bin,
                harbor_sudo=arguments.harbor_sudo,
                image_manifest=arguments.image_manifest,
            )
        else:
            report = run_experiment(
                model=model,
                dataset=arguments.dataset,
                jobs_dir=arguments.jobs_dir,
                harbor_bin=arguments.harbor_bin,
                harbor_sudo=arguments.harbor_sudo,
                execute=arguments.execute,
                image_manifest=arguments.image_manifest,
            )
    except (OSError, TypeError, ValueError) as exc:
        print(f"experiment error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        if arguments.artifacts_dir is not None and report.get("result_summaries"):
            if arguments.ablation:
                report["artifacts"] = write_ablation_artifacts(
                    arguments.artifacts_dir, report
                )
            else:
                rows = [
                    row
                    for row in report["result_summaries"]
                    if row.get("kind") == "trial"
                ]
                report["artifacts"] = write_matrix_artifacts(
                    arguments.artifacts_dir,
                    rows,
                    require_complete=bool(
                        arguments.formal and not arguments.allow_incomplete
                    ),
                    image_manifest=(
                        report.get("image_manifest")
                        if isinstance(report.get("image_manifest"), Mapping)
                        else None
                    ),
                )
        if arguments.output is not None:
            _write_json(arguments.output, report)
    except (OSError, TypeError, ValueError) as exc:
        print(f"experiment error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
