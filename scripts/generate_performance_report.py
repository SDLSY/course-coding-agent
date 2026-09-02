"""Generate a scrubbed Terminal-Bench comparison package and a readable chart.

The input is the completed formal report produced by
``benchmarks/run_terminal_bench_2_1.py --formal``.  Only aggregate, non-secret
fields are copied to the output directory.  The chart is rendered with plain
SVG/JavaScript so the HTML remains editable and works without a Python plotting
stack or a network connection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_REPORT = Path(".harbor-runs/formal/formal-report.json")
DEFAULT_OUTPUT = Path("reports")

MODEL_META: dict[str, dict[str, str]] = {
    "deepseek-v4-flash-vision-exp": {
        "short": "DeepSeek",
        "display": "DeepSeek V4 Flash Vision",
    },
    "glm-5.3-flash": {"short": "GLM-5.3", "display": "GLM-5.3 Flash"},
    "gpt-5.6-sol": {"short": "GPT-5.6", "display": "GPT-5.6 Sol"},
}
AGENT_META: dict[str, dict[str, str]] = {
    "course-coding-agent": {"short": "Course", "display": "CourseCodingAgent"},
    "opencode-pinned": {"short": "OpenCode", "display": "OpenCode 1.18.25"},
}
MODEL_ORDER = list(MODEL_META)
AGENT_ORDER = list(AGENT_META)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _round(value: float | None, digits: int = 1) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 100:
        return f"{value:.0f} s"
    return f"{value:.1f} s"


def _format_tokens(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.0f}"


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _wilson_interval(passed: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = passed / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return (max(0.0, center - radius), min(1.0, center + radius))


def _iqr(summary: dict[str, Any], name: str) -> dict[str, float | None]:
    quartiles = summary.get("quartiles", {}).get(name, {})
    return {
        "q1": _number(quartiles.get("q1")),
        "median": _number(quartiles.get("median")),
        "q3": _number(quartiles.get("q3")),
    }


def _safe_route(route: dict[str, Any]) -> dict[str, Any]:
    """Keep route metadata while making it impossible to copy a key value."""

    return {
        "provider": route.get("provider"),
        "base_url": route.get("base_url"),
        "key_env": route.get("key_env"),
        "model": route.get("model"),
    }


def _public_dataset_fields(report: dict[str, Any]) -> tuple[str, str | None]:
    """Return dataset metadata without publishing a machine-local path."""

    reference = report.get("dataset_reference")
    if isinstance(reference, str) and reference.strip():
        value = reference.strip()
        return value, value
    dataset = report.get("dataset")
    if isinstance(dataset, str) and dataset.strip():
        value = dataset.strip()
        # Absolute paths (and home-relative paths) identify the producer's
        # checkout rather than a reproducible public dataset reference.
        if value.startswith(("/", "~")):
            return "local-checkout", None
        return value, None
    return "unspecified", None


def load_aggregate(report_path: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    public_dataset, dataset_reference = _public_dataset_fields(report)
    matrix = report.get("matrix_summary", {})
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for agent in AGENT_ORDER:
            key = f"{model}::{agent}"
            summary = matrix.get(key)
            if not isinstance(summary, dict):
                raise TypeError(f"missing matrix group: {key}")
            accuracy = _number(summary.get("accuracy"))
            ci = summary.get("accuracy_wilson_95", {})
            agent_iqr = _iqr(summary, "agent_elapsed_seconds")
            total_iqr = _iqr(summary, "total_elapsed_seconds")
            token_iqr = _iqr(summary, "total_tokens")
            rows.append(
                {
                    "id": key,
                    "model": model,
                    "model_short": MODEL_META[model]["short"],
                    "model_display": MODEL_META[model]["display"],
                    "agent": agent,
                    "agent_short": AGENT_META[agent]["short"],
                    "agent_display": AGENT_META[agent]["display"],
                    "label": f"{MODEL_META[model]['short']} / {AGENT_META[agent]['short']}",
                    "expected_trials": int(summary.get("expected_trials", 0)),
                    "n_trials": int(summary.get("n_trials", 0)),
                    "n_evaluable_trials": int(summary.get("n_evaluable_trials", 0)),
                    "n_passed": int(summary.get("n_passed", 0)),
                    "n_failed": int(summary.get("n_failed", 0)),
                    "n_setup_failures": int(summary.get("n_setup_failures", 0)),
                    "n_unresolved": int(summary.get("n_unresolved", 0)),
                    "accuracy": accuracy,
                    "accuracy_low": _number(ci.get("low")),
                    "accuracy_high": _number(ci.get("high")),
                    "agent_elapsed_seconds": _round(agent_iqr["median"], 3),
                    "agent_elapsed_q1": _round(agent_iqr["q1"], 3),
                    "agent_elapsed_q3": _round(agent_iqr["q3"], 3),
                    "total_elapsed_seconds": _round(total_iqr["median"], 3),
                    "total_elapsed_q1": _round(total_iqr["q1"], 3),
                    "total_elapsed_q3": _round(total_iqr["q3"], 3),
                    "input_tokens": _round(
                        _number(summary.get("median_input_tokens")), 1
                    ),
                    "output_tokens": _round(
                        _number(summary.get("median_output_tokens")), 1
                    ),
                    "cache_tokens": _round(
                        _number(summary.get("median_cache_tokens")), 1
                    ),
                    "total_tokens": _round(token_iqr["median"], 1),
                    "total_tokens_q1": _round(token_iqr["q1"], 1),
                    "total_tokens_q3": _round(token_iqr["q3"], 1),
                    "model_requests": _round(
                        _number(summary.get("median_model_requests")), 1
                    ),
                    "tool_calls": _round(_number(summary.get("median_tool_calls")), 1),
                    "infrastructure_failures": int(
                        summary.get("infrastructure_failures", 0)
                    ),
                }
            )

    trials = [
        item
        for item in report.get("result_summaries", [])
        if item.get("kind") == "trial"
        and item.get("matrix_model")
        and item.get("matrix_agent")
    ]
    task_order = list(report.get("tasks", []))
    task_matrix: list[dict[str, Any]] = []
    for task in task_order:
        task_row: dict[str, Any] = {"task": task}
        for model in MODEL_ORDER:
            for agent in AGENT_ORDER:
                group = [
                    item
                    for item in trials
                    if item.get("matrix_task") == task
                    and item.get("matrix_model") == model
                    and item.get("matrix_agent") == agent
                ]
                task_row[f"{model}::{agent}"] = {
                    "passed": sum(item.get("passed") is True for item in group),
                    "trials": len(group),
                }
        task_matrix.append(task_row)

    def aggregate(
        items: list[dict[str, Any]], key: str, value: str
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for name in MODEL_ORDER if key == "matrix_model" else AGENT_ORDER:
            selected = [item for item in items if item.get(key) == name]
            passed = sum(item.get("passed") is True for item in selected)
            trials_count = len(selected)
            low, high = _wilson_interval(passed, trials_count)
            result.append(
                {
                    value: name,
                    "n_passed": passed,
                    "n_trials": trials_count,
                    "accuracy": passed / trials_count if trials_count else None,
                    "accuracy_low": low,
                    "accuracy_high": high,
                }
            )
        return result

    model_aggregates = aggregate(trials, "matrix_model", "model")
    agent_aggregates = aggregate(trials, "matrix_agent", "agent")
    task_totals: list[dict[str, Any]] = []
    for task in task_order:
        selected = [item for item in trials if item.get("matrix_task") == task]
        passed = sum(item.get("passed") is True for item in selected)
        task_totals.append(
            {"task": task, "n_passed": passed, "n_trials": len(selected)}
        )

    trial_rows: list[dict[str, Any]] = []
    trial_fields = (
        "repetition",
        "passed",
        "reward",
        "phase",
        "reason",
        "elapsed_seconds",
        "agent_elapsed_seconds",
        "total_elapsed_seconds",
        "model_turns",
        "model_requests",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "cache_tokens",
        "total_tokens",
        "reasoning_effort",
        "reasoning_parameter",
        "reasoning_value",
        "reasoning_capability_status",
        "failure_stage",
        "infrastructure_reason",
        "opencode_version",
        "image_digest",
        "source_image_digest",
        "matrix_model",
        "matrix_agent",
        "matrix_task",
    )
    for item in sorted(
        trials,
        key=lambda row: (
            str(row.get("matrix_model")),
            str(row.get("matrix_agent")),
            str(row.get("matrix_task")),
            int(row.get("repetition") or 0),
        ),
    ):
        trial_rows.append({field: item.get(field) for field in trial_fields})

    matrix_validation = report.get("matrix_validation", {})
    if (
        not matrix_validation.get("complete")
        or int(report.get("evaluable_trials", 0)) != 144
    ):
        raise ValueError("formal report is not the complete 144-trial matrix")

    source_hash = hashlib.sha256(report_path.read_bytes()).hexdigest()
    recovery = report.get("execution_recovery", {})
    infra_records = []
    for item in report.get("infrastructure_failures", []):
        infra_records.append(
            {
                "model": item.get("model"),
                "agent": item.get("agent"),
                "stage": item.get("stage"),
                "reason": item.get("reason"),
                "planned_trials": item.get("planned_trials"),
                "recovered": bool(item.get("recovered")),
                "excluded_from_accuracy": bool(item.get("excluded_from_accuracy")),
            }
        )

    return {
        "schema_version": "coding-agent-performance-summary/v1",
        "source_report_sha256": source_hash,
        "scope_note": report.get("scope_note"),
        "dataset": public_dataset,
        "dataset_reference": dataset_reference,
        "task_count": int(report.get("task_count", len(task_order))),
        "tasks": task_order,
        "models": MODEL_ORDER,
        "agents": AGENT_ORDER,
        "repetitions_per_task": int(report.get("repetitions_per_task", 3)),
        "expected_trials": int(report.get("expected_trials", 144)),
        "evaluable_trials": int(report.get("evaluable_trials", 0)),
        "matrix_complete": bool(report.get("matrix_complete")),
        "routes": [_safe_route(route) for route in report.get("routes", [])],
        "reasoning_effort": report.get("reasoning_effort"),
        "reasoning_probes": [
            {
                "provider": probe.get("provider"),
                "model": probe.get("model"),
                "status": probe.get("status"),
                "parameter": probe.get("parameter"),
                "accepted_value": probe.get("accepted_value"),
            }
            for probe in report.get("reasoning_probes", [])
        ],
        "course_round_strategy": report.get("course_round_strategy"),
        "runtime_limits": {
            key: report.get("runtime_limits", {}).get(key)
            for key in (
                "course_max_model_turns",
                "course_max_tool_calls",
                "agent_timeout_seconds",
                "n_concurrent",
                "course_efficiency_mode",
                "course_reserve_final_turn",
            )
        },
        "opencode": {
            "version": report.get("opencode", {}).get("version"),
            "setup_policy": report.get("opencode", {}).get("setup_policy"),
        },
        "ablation_decision": {
            "selected_strategy": report.get("ablation_decision", {}).get(
                "selected_strategy"
            ),
            "selected_max_model_turns": report.get("ablation_decision", {}).get(
                "selected_max_model_turns"
            ),
            "reason": report.get("ablation_decision", {}).get("reason"),
        },
        "infrastructure_failures": infra_records,
        "recovery_count": int(recovery.get("recovery_count", 0)),
        "groups": rows,
        "task_matrix": task_matrix,
        "model_aggregates": model_aggregates,
        "agent_aggregates": agent_aggregates,
        "task_totals": task_totals,
        "trial_rows": trial_rows,
    }


def _csv_rows(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = [
        "model",
        "agent",
        "n_passed",
        "n_evaluable_trials",
        "accuracy",
        "accuracy_low",
        "accuracy_high",
        "total_elapsed_seconds",
        "total_elapsed_q1",
        "total_elapsed_q3",
        "total_tokens",
        "total_tokens_q1",
        "total_tokens_q3",
        "agent_elapsed_seconds",
        "input_tokens",
        "output_tokens",
        "cache_tokens",
        "model_requests",
        "tool_calls",
        "n_setup_failures",
    ]
    return [{field: group.get(field) for field in fields} for group in groups]


def write_csv(path: Path, groups: list[dict[str, Any]]) -> None:
    rows = _csv_rows(groups)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_trial_csv(path: Path, trial_rows: list[dict[str, Any]]) -> None:
    if not trial_rows:
        raise ValueError("formal report has no trial rows")
    fields = list(trial_rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trial_rows)


def _ci_text(group: dict[str, Any]) -> str:
    low = group.get("accuracy_low")
    high = group.get("accuracy_high")
    if low is None or high is None:
        return "n/a"
    return f"{low * 100:.1f}-{high * 100:.1f}%"


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    groups = summary["groups"]
    by_accuracy = sorted(
        groups, key=lambda item: (-item["accuracy"], item["total_elapsed_seconds"])
    )
    by_time = sorted(groups, key=lambda item: item["total_elapsed_seconds"])
    by_tokens = sorted(groups, key=lambda item: item["total_tokens"])
    lines = [
        "# Terminal-Bench coding agent 性能摘要",
        "",
        f"> {summary['scope_note']}；正式矩阵 {summary['evaluable_trials']}/{summary['expected_trials']} 条记录完整。",
        "> 正确率按 24 个可评测 trial 计算；括号为 Wilson 95% 区间。耗时和 token 为中位数，方括号为 IQR（Q1-Q3）。",
        "",
        "## 组合汇总",
        "",
        "| 模型 | Agent | 通过 | 正确率 (95% CI) | 总耗时中位数 | 总 token 中位数 | 模型请求 / 工具调用 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in groups:
        lines.append(
            "| {model_display} | {agent_display} | {n_passed}/{n_evaluable_trials} | {acc} ({ci}) | {time} [{q1}-{q3}] | {tokens} [{tq1}-{tq3}] | {requests:g} / {tools:g} |".format(
                model_display=group["model_display"],
                agent_display=group["agent_display"],
                n_passed=group["n_passed"],
                n_evaluable_trials=group["n_evaluable_trials"],
                acc=_format_percent(group["accuracy"]),
                ci=_ci_text(group),
                time=_format_seconds(group["total_elapsed_seconds"]),
                q1=_format_seconds(group["total_elapsed_q1"]).replace(" s", ""),
                q3=_format_seconds(group["total_elapsed_q3"]).replace(" s", ""),
                tokens=_format_tokens(group["total_tokens"]),
                tq1=_format_tokens(group["total_tokens_q1"]),
                tq3=_format_tokens(group["total_tokens_q3"]),
                requests=group["model_requests"],
                tools=group["tool_calls"],
            )
        )

    total_passed = sum(group["n_passed"] for group in groups)
    total_trials = sum(group["n_evaluable_trials"] for group in groups)
    lines.extend(
        [
            "",
            f"合并来看，六个组合共通过 {total_passed}/{total_trials}（{total_passed / total_trials * 100:.1f}%）。",
            "",
            "## 按模型合计",
            "",
            "| 模型（两个 Agent 合计） | 通过 | 正确率 (95% CI) |",
            "|---|---:|---:|",
        ]
    )
    for item in summary["model_aggregates"]:
        lines.append(
            f"| {MODEL_META[item['model']]['display']} | {item['n_passed']}/{item['n_trials']} | {_format_percent(item['accuracy'])} ({item['accuracy_low'] * 100:.1f}-{item['accuracy_high'] * 100:.1f}%) |"
        )
    lines.extend(
        [
            "",
            "## 按 Agent 合计",
            "",
            "| Agent（三个模型合计） | 通过 | 正确率 (95% CI) |",
            "|---|---:|---:|",
        ]
    )
    for item in summary["agent_aggregates"]:
        lines.append(
            f"| {AGENT_META[item['agent']]['display']} | {item['n_passed']}/{item['n_trials']} | {_format_percent(item['accuracy'])} ({item['accuracy_low'] * 100:.1f}-{item['accuracy_high'] * 100:.1f}%) |"
        )

    lines.extend(
        [
            "",
            "## 任务级通过数",
            "",
            "| 任务 | DeepSeek/Course | DeepSeek/OpenCode | GLM/Course | GLM/OpenCode | GPT-5.6/Course | GPT-5.6/OpenCode |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task in summary["task_matrix"]:
        cells = [task["task"]]
        for model in MODEL_ORDER:
            for agent in AGENT_ORDER:
                value = task[f"{model}::{agent}"]
                cells.append(f"{value['passed']}/{value['trials']}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## 主要结论",
            "",
            f"- 正确率最高的是 GPT-5.6 Sol + CourseCodingAgent 与 GPT-5.6 Sol + OpenCode：均为 {by_accuracy[0]['n_passed']}/{by_accuracy[0]['n_evaluable_trials']}（{_format_percent(by_accuracy[0]['accuracy'])}）。",
            f"- 总耗时中位数最低的是 {by_time[0]['model_display']} + {by_time[0]['agent_display']}（{_format_seconds(by_time[0]['total_elapsed_seconds'])}）；GLM + OpenCode 最慢（{_format_seconds(max(groups, key=lambda item: item['total_elapsed_seconds'])['total_elapsed_seconds'])}）。",
            f"- token 中位数最低的是 {by_tokens[0]['model_display']} + {by_tokens[0]['agent_display']}（{_format_tokens(by_tokens[0]['total_tokens'])}）。",
            "- GLM 与 DeepSeek 的 CourseCodingAgent 正确率分别比同模型 OpenCode 高 8.3 和 8.4 个百分点；GPT-5.6 两个 Agent 正确率相同。",
            "- 任务难度差异明显：`fix-git` 与 `fix-code-vulnerability` 各组合合计 17/18，`kv-store-grpc` 为 16/18；`write-compressor` 仅 5/18。",
            "- 结果是小样本、固定任务子集上的探索性比较，不能替代完整 Terminal-Bench leaderboard，也不能单独证明因果优劣。",
            "",
            "## 实验审计",
            "",
            f"- 模型路由的原生 `reasoning_effort=high` 探针均通过；Course agent 正式采用 `{summary['course_round_strategy']}`，OpenCode 固定为 `{summary['opencode']['version']}`。",
            f"- 最终矩阵无未决记录；历史基础设施问题共 {summary['recovery_count']} 组，均已恢复并从模型正确率中排除，不与 verifier failure 混计。",
            "- 两组历史问题分别是 GLM/OpenCode 首次未正确转发 ZAI_API_KEY，以及 GPT-5.6/OpenCode 首次网关余额/计费失败；重跑后的 GPT-5.6/OpenCode 为 21/24。",
            "- 脱敏的 144 条明细见 `terminal-bench-trials.csv`；详细原始 Harbor 日志和本机数据集不随仓库提交。本摘要和图表不包含 API key 值。",
            "",
            "## 提交材料核对",
            "",
            "按 `要求.pdf`，最终提交内容应只有一个以姓名命名的 ZIP，内部恰好包含：",
            "",
            "1. `README.txt`：仓库地址、运行方法、特色说明，1000 汉字以内。",
            "2. `李上一.mp4`：真实编程任务演示，MP4、不超过 2 分钟且不超过 200 MB。",
            "",
            "当前仓库地址为 `https://github.com/SDLSY/course-coding-agent`；不要把 `.env`、API key、评测日志或本机轨迹放入提交包。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _svg_escape(value: str) -> str:
    return html.escape(value, quote=True)


def _svg_text(
    x: float, y: float, value: str, *, anchor: str = "start", cls: str = ""
) -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{class_attr}>{_svg_escape(value)}</text>'


def _svg_line(x1: float, y1: float, x2: float, y2: float, cls: str = "grid") -> str:
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" />'


def _svg_rect(
    x: float, y: float, width: float, height: float, cls: str, extra: str = ""
) -> str:
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(0.0, width):.1f}" height="{height:.1f}"{extra} />'


def _svg_circle(cx: float, cy: float, radius: float, cls: str, extra: str = "") -> str:
    return (
        f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}"{extra} />'
    )


def _scale(value: float | None, domain_max: float, x0: float, x1: float) -> float:
    if value is None:
        return x0
    return x0 + max(0.0, min(domain_max, value)) / domain_max * (x1 - x0)


def _nice_max(value: float, step: float) -> float:
    return max(step, math.ceil(value / step) * step)


def _static_panel(
    groups: list[dict[str, Any]],
    metric: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    # Keep a dedicated value column on the right so high values never need
    # low-contrast labels drawn over the colored marks.
    left, right, top, bottom = 174.0, 132.0, 52.0, 54.0
    x0, x1 = x + left, x + width - right
    plot_top, plot_bottom = y + top, y + height - bottom
    row_height = (plot_bottom - plot_top) / len(groups)
    if metric == "accuracy":
        title, axis, domain, ticks = "正确率", "通过率 (%)", 100.0, [0, 25, 50, 75, 100]
    elif metric == "time":
        title, axis = "总耗时", "总耗时 (seconds; median, IQR)"
        domain = _nice_max(
            max(float(item["total_elapsed_q3"] or 0) for item in groups), 100.0
        )
        ticks = [domain * i / 5 for i in range(6)]
    elif metric == "tokens":
        title, axis = "总 token", "总 token (median, IQR)"
        domain = _nice_max(
            max(float(item["total_tokens_q3"] or 0) for item in groups), 100_000.0
        )
        ticks = [domain * i / 5 for i in range(6)]
    else:
        title, axis = "模型与工具调用", "次数 (median; requests / tools)"
        domain = _nice_max(
            max(
                max(float(item["model_requests"] or 0), float(item["tool_calls"] or 0))
                for item in groups
            )
            * 1.15,
            5.0,
        )
        ticks = [domain * i / 5 for i in range(6)]
    parts = [
        f'<g class="panel" data-metric="{metric}">',
        _svg_text(x + left, y + 22, title, cls="panel-title"),
    ]
    for tick in ticks:
        tx = _scale(tick, domain, x0, x1)
        parts.append(_svg_line(tx, plot_top, tx, plot_bottom))
        if metric == "tokens":
            label = (
                f"{tick / 1000:.0f}k"
                if domain < 1_000_000
                else f"{tick / 1_000_000:.1f}M"
            )
        else:
            label = f"{tick:.0f}"
        parts.append(
            _svg_text(tx, plot_bottom + 18, label, anchor="middle", cls="tick")
        )
    for index, item in enumerate(groups):
        cy = plot_top + row_height * (index + 0.5)
        parts.append(
            _svg_text(
                x + left - 10,
                cy - 2,
                item["model_short"],
                anchor="end",
                cls="row-label",
            )
        )
        parts.append(
            _svg_text(
                x + left - 10,
                cy + 12,
                item["agent_short"],
                anchor="end",
                cls="row-label secondary",
            )
        )
        color_class = f"series-{MODEL_ORDER.index(item['model']) + 1}"
        open_code = item["agent"] == "opencode-pinned"
        opacity = ' opacity="0.68"' if open_code else ""
        if metric == "accuracy":
            bar_x = _scale(0, domain, x0, x1)
            bar_w = _scale(float(item["accuracy"] or 0) * 100, domain, x0, x1) - bar_x
            parts.append(_svg_rect(bar_x, cy - 10, bar_w, 20, color_class, opacity))
            low = _scale(float(item["accuracy_low"] or 0) * 100, domain, x0, x1)
            high = _scale(float(item["accuracy_high"] or 0) * 100, domain, x0, x1)
            accuracy_x = _scale(
                float(item["accuracy"] or 0) * 100,
                domain,
                x0,
                x1,
            )
            parts.append(_svg_line(low, cy, high, cy, "interval"))
            parts.append(_svg_line(low, cy - 7, low, cy + 7, "interval"))
            parts.append(_svg_line(high, cy - 7, high, cy + 7, "interval"))
            parts.append(_svg_circle(accuracy_x, cy, 4, color_class))
            label = f"{float(item['accuracy'] or 0) * 100:.1f}% ({item['n_passed']}/{item['n_evaluable_trials']})"
        elif metric == "time":
            median = float(item["total_elapsed_seconds"] or 0)
            q1 = float(item["total_elapsed_q1"] or median)
            q3 = float(item["total_elapsed_q3"] or median)
            parts.append(
                _svg_rect(
                    x0,
                    cy - 9,
                    _scale(median, domain, x0, x1) - x0,
                    18,
                    color_class,
                    opacity,
                )
            )
            lo, hi = _scale(q1, domain, x0, x1), _scale(q3, domain, x0, x1)
            parts.append(_svg_line(lo, cy, hi, cy, "interval"))
            parts.append(
                _svg_circle(_scale(median, domain, x0, x1), cy, 4, color_class)
            )
            label = _format_seconds(median)
        elif metric == "tokens":
            median = float(item["total_tokens"] or 0)
            q1 = float(item["total_tokens_q1"] or median)
            q3 = float(item["total_tokens_q3"] or median)
            parts.append(
                _svg_rect(
                    x0,
                    cy - 9,
                    _scale(median, domain, x0, x1) - x0,
                    18,
                    color_class,
                    opacity,
                )
            )
            lo, hi = _scale(q1, domain, x0, x1), _scale(q3, domain, x0, x1)
            parts.append(_svg_line(lo, cy, hi, cy, "interval"))
            parts.append(
                _svg_circle(_scale(median, domain, x0, x1), cy, 4, color_class)
            )
            label = _format_tokens(median)
        else:
            requests = float(item["model_requests"] or 0)
            tools = float(item["tool_calls"] or 0)
            req_x = _scale(requests, domain, x0, x1)
            tool_x = _scale(tools, domain, x0, x1)
            parts.append(_svg_rect(x0, cy - 12, req_x - x0, 9, color_class, opacity))
            parts.append(_svg_rect(x0, cy + 3, tool_x - x0, 9, "tools-bar"))
            parts.append(
                _svg_text(
                    x + width - 8,
                    cy + 4,
                    f"{requests:g} / {tools:g}",
                    anchor="end",
                    cls="value",
                )
            )
            label = ""
        if label:
            parts.append(
                _svg_text(x + width - 8, cy + 4, label, anchor="end", cls="value")
            )
    parts.extend(
        [
            _svg_line(x0, plot_bottom, x1, plot_bottom, "axis"),
            f'<text class="axis-title" data-axis="x" x="{(x0 + x1) / 2:.1f}" y="{y + height - 8:.1f}" text-anchor="middle">{_svg_escape(axis)}</text>',
            f'<text class="axis-title y-axis" data-axis="y" x="{x + 15:.1f}" y="{(plot_top + plot_bottom) / 2:.1f}" transform="rotate(-90 {x + 15:.1f} {(plot_top + plot_bottom) / 2:.1f})">模型 / Agent</text>',
            "</g>",
        ]
    )
    return "".join(parts)


def write_static_svg(path: Path, summary: dict[str, Any]) -> None:
    groups = summary["groups"]
    width, height = 1500.0, 1030.0
    panel_w, panel_h = 710.0, 430.0
    panels = [
        _static_panel(groups, "accuracy", 0, 100, panel_w, panel_h),
        _static_panel(groups, "time", 790, 100, panel_w, panel_h),
        _static_panel(groups, "tokens", 0, 570, panel_w, panel_h),
        _static_panel(groups, "calls", 790, 570, panel_w, panel_h),
    ]
    legend: list[str] = []
    for index, model in enumerate(MODEL_ORDER, start=1):
        legend.append(
            f'<rect class="series-{index}" x="{30 + (index - 1) * 245}" y="62" width="14" height="14" />'
        )
        legend.append(
            _svg_text(
                51 + (index - 1) * 245, 74, MODEL_META[model]["display"], cls="legend"
            )
        )
    legend.extend(
        [
            '<line class="legend-agent" x1="760" y1="69" x2="792" y2="69" />',
            _svg_text(802, 74, "CourseCodingAgent", cls="legend"),
            '<line class="legend-agent dashed" x1="1100" y1="69" x2="1132" y2="69" />',
            _svg_text(1142, 74, "OpenCode 1.18.25", cls="legend"),
        ]
    )
    css = """
    :root { --background:#ffffff; --foreground:#17202a; --muted:#5c6770; --border:#d7dee5;
      --viz-series-1:#0f766e; --viz-series-2:#c2410c; --viz-series-3:#1d4ed8; }
    svg { background:var(--background); color:var(--foreground); font-family:Arial,'Noto Sans CJK SC',sans-serif; }
    text { fill:var(--foreground); font-size:14px; }
    .panel-title { font-size:18px; font-weight:500; }
    .legend,.axis-title { font-size:13px; }
    .tick,.secondary { fill:var(--muted); font-size:12px; }
    .row-label { font-size:13px; }
    .value { font-size:12px; font-weight:500; }
    .grid { stroke:var(--border); stroke-width:1; }
    .axis { stroke:var(--foreground); stroke-width:1.1; }
    .interval { stroke:var(--foreground); stroke-width:1.4; }
    .tools-bar { fill:var(--muted); opacity:.55; }
    .series-1 { fill:var(--viz-series-1); }
    .series-2 { fill:var(--viz-series-2); }
    .series-3 { fill:var(--viz-series-3); }
    .legend-agent { stroke:var(--foreground); stroke-width:3; }
    .dashed { stroke-dasharray:6 4; opacity:.68; }
    @media (prefers-color-scheme:dark) { :root { --background:#181818; --foreground:#edf2f7; --muted:#a9b4bf; --border:#3b4650; } }
    """
    title = "Coding agent 性能对比"
    subtitle = "8 个 Terminal-Bench 2.1 任务，每项 3 次；中位数与 Wilson 95% 区间"
    content = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-labelledby="title desc">'
        f'<style>{css}</style><title id="title">{_svg_escape(title)}</title><desc id="desc">{_svg_escape(subtitle)}</desc>'
        f"{_svg_text(30, 32, title, cls='chart-title')}{_svg_text(30, 51, subtitle, cls='subtitle')}"
        + "".join(legend)
        + "".join(panels)
        + '<text class="subtitle" x="30" y="1018">说明：横线表示 IQR；GPT-5.6 Sol 的 Course agent 未提供 cache token，故总 token 仍按记录值统计。</text></svg>'
    )
    path.write_text(content, encoding="utf-8")


HTML_TEMPLATE = r"""<section id="tb-performance-visual" class="tb-visual" aria-labelledby="tb-title">
  <style>
    #tb-performance-visual { --background: light-dark(#ffffff, #181818); --foreground: light-dark(#17202a, #edf2f7); --muted: light-dark(#5c6770, #a9b4bf); --border: light-dark(#d7dee5, #3b4650); --viz-series-1: #0f766e; --viz-series-2: #c2410c; --viz-series-3: #1d4ed8; color: var(--foreground); background: transparent; font-family: Arial, "Noto Sans CJK SC", sans-serif; color-scheme: light dark; }
    #tb-performance-visual h1 { margin: 0 0 .25rem; font-size: 1.25rem; line-height: 1.35; font-weight: 500; }
    #tb-performance-visual .tb-subtitle, #tb-performance-visual .tb-note { margin: 0; color: var(--muted); font-size: .82rem; line-height: 1.45; }
    #tb-performance-visual .tb-legend { display: flex; flex-wrap: wrap; gap: .35rem 1.1rem; margin: .75rem 0 .35rem; font-size: .78rem; color: var(--muted); }
    #tb-performance-visual .tb-legend-item { display: inline-flex; align-items: center; gap: .35rem; white-space: nowrap; }
    #tb-performance-visual .tb-swatch { width: .72rem; height: .72rem; display: inline-block; background: var(--swatch); }
    #tb-performance-visual .tb-agent-line { width: 1.3rem; display: inline-block; border-top: 2px solid var(--foreground); }
    #tb-performance-visual .tb-agent-line.dashed { border-top-style: dashed; opacity: .68; }
    #tb-performance-visual .tb-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1.2rem 1.6rem; }
    #tb-performance-visual figure { margin: 0; min-width: 0; }
    #tb-performance-visual figcaption { font-size: .9rem; font-weight: 500; margin: .3rem 0; }
    #tb-performance-visual svg { display: block; width: 100%; height: auto; overflow: visible; color: var(--foreground); }
    #tb-performance-visual svg text { fill: currentColor; font-family: inherit; font-size: 12px; }
    #tb-performance-visual svg .panel-title { font-size: 15px; font-weight: 500; }
    #tb-performance-visual svg .tick, #tb-performance-visual svg .secondary { fill: var(--muted); font-size: 11px; }
    #tb-performance-visual svg .row-label { font-size: 11px; }
    #tb-performance-visual svg .value { font-size: 11px; font-weight: 500; }
    #tb-performance-visual svg .grid { stroke: var(--border); stroke-width: 1; }
    #tb-performance-visual svg .axis { stroke: currentColor; stroke-width: 1.1; }
    #tb-performance-visual svg .interval { stroke: currentColor; stroke-width: 1.4; }
    #tb-performance-visual svg .tools-bar { fill: var(--muted); opacity: .55; }
    #tb-performance-visual svg .series-1 { fill: var(--viz-series-1); }
    #tb-performance-visual svg .series-2 { fill: var(--viz-series-2); }
    #tb-performance-visual svg .series-3 { fill: var(--viz-series-3); }
    #tb-performance-visual .tb-sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
    @media (max-width: 780px) { #tb-performance-visual .tb-grid { grid-template-columns: 1fr; gap: 1rem; } }
  </style>
  <h1 id="tb-title">Coding agent 性能对比</h1>
  <p class="tb-subtitle">8 个 Terminal-Bench 2.1 任务，每项 3 次；通过率含 Wilson 95% 区间，耗时和 token 为中位数。</p>
  <div class="tb-legend" aria-label="图例">
    <span class="tb-legend-item"><i class="tb-swatch" style="--swatch:var(--viz-series-1)"></i>DeepSeek V4 Flash Vision</span>
    <span class="tb-legend-item"><i class="tb-swatch" style="--swatch:var(--viz-series-2)"></i>GLM-5.3 Flash</span>
    <span class="tb-legend-item"><i class="tb-swatch" style="--swatch:var(--viz-series-3)"></i>GPT-5.6 Sol</span>
    <span class="tb-legend-item"><i class="tb-agent-line"></i>CourseCodingAgent</span>
    <span class="tb-legend-item"><i class="tb-agent-line dashed"></i>OpenCode 1.18.25</span>
  </div>
  <div class="tb-grid">
    <figure><svg data-metric="accuracy" role="img" aria-label="各组合正确率及 Wilson 95% 区间"></svg></figure>
    <figure><svg data-metric="time" role="img" aria-label="各组合总耗时中位数及 IQR"></svg></figure>
    <figure><svg data-metric="tokens" role="img" aria-label="各组合总 token 中位数及 IQR"></svg></figure>
    <figure><svg data-metric="calls" role="img" aria-label="各组合模型请求和工具调用中位数"></svg></figure>
  </div>
  <p class="tb-note">线段表示 IQR；调用面板中的两个数字依次为模型请求 / 工具调用。GPT-5.6 Sol 的 Course agent 未返回 cache token。</p>
  <table class="tb-sr-only"><caption>完整组合汇总</caption><thead><tr><th>模型/Agent</th><th>通过率</th><th>总耗时中位数</th><th>总 token 中位数</th></tr></thead><tbody id="tb-accessible-rows"></tbody></table>
  <script>
    (() => {
      const data = __DATA__;
      const root = document.getElementById("tb-performance-visual");
      if (!root) return;
      const ns = "http://www.w3.org/2000/svg";
      const modelColors = ["var(--viz-series-1)", "var(--viz-series-2)", "var(--viz-series-3)"];
      const modelIndex = {"deepseek-v4-flash-vision-exp": 0, "glm-5.3-flash": 1, "gpt-5.6-sol": 2};
      const shortModel = {"deepseek-v4-flash-vision-exp": "DeepSeek", "glm-5.3-flash": "GLM-5.3", "gpt-5.6-sol": "GPT-5.6"};
      const shortAgent = {"course-coding-agent": "Course", "opencode-pinned": "OpenCode"};
      const groups = data.groups;
      const make = (tag, attrs = {}, content = "") => {
        const node = document.createElementNS(ns, tag);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
        if (content) node.textContent = content;
        return node;
      };
      const formatSeconds = value => value == null ? "n/a" : (value >= 100 ? `${value.toFixed(0)} s` : `${value.toFixed(1)} s`);
      const formatTokens = value => value == null ? "n/a" : (value >= 1000000 ? `${(value / 1000000).toFixed(2)}M` : `${(value / 1000).toFixed(1)}k`);
      const niceMax = (value, step) => Math.max(step, Math.ceil(value / step) * step);
      const addText = (svg, x, y, text, cls = "", anchor = "start") => svg.appendChild(make("text", {x, y, "text-anchor": anchor, class: cls}, text));
      const scale = (value, max, x0, x1) => x0 + Math.max(0, Math.min(max, value || 0)) / max * (x1 - x0);
      const drawPanel = (svg, metric) => {
        const width = Math.max(320, svg.parentElement.getBoundingClientRect().width || 600);
        const narrow = width < 480;
        const height = narrow ? 430 : 350;
        const left = narrow ? 122 : 148;
        // Reserve a right-aligned value column for every metric.
        const right = narrow ? 102 : 112;
        const valueX = width - 6;
        const top = 42;
        const bottom = 52;
        const x0 = left;
        const x1 = width - right;
        const plotTop = top;
        const plotBottom = height - bottom;
        const rowHeight = (plotBottom - plotTop) / groups.length;
        let title, axis, domain, ticks;
        if (metric === "accuracy") { title = "正确率"; axis = "通过率 (%)"; domain = 100; ticks = [0, 25, 50, 75, 100]; }
        else if (metric === "time") { title = "总耗时"; axis = "总耗时 (seconds; median, IQR)"; domain = niceMax(Math.max(...groups.map(g => g.total_elapsed_q3 || 0)), 100); ticks = Array.from({length: 6}, (_, i) => domain * i / 5); }
        else if (metric === "tokens") { title = "总 token"; axis = "总 token (median, IQR)"; domain = niceMax(Math.max(...groups.map(g => g.total_tokens_q3 || 0)), 100000); ticks = Array.from({length: 6}, (_, i) => domain * i / 5); }
        else { title = "模型与工具调用"; axis = "次数 (median; requests / tools)"; domain = niceMax(Math.max(...groups.map(g => Math.max(g.model_requests || 0, g.tool_calls || 0))) * 1.15, 5); ticks = Array.from({length: 6}, (_, i) => domain * i / 5); }
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.setAttribute("height", height);
        const titleNode = make("title", {}, title);
        svg.appendChild(titleNode);
        addText(svg, left, 22, title, "panel-title");
        ticks.forEach(tick => {
          const tx = scale(tick, domain, x0, x1);
          svg.appendChild(make("line", {class: "grid", x1: tx, y1: plotTop, x2: tx, y2: plotBottom}));
          const label = metric === "tokens" ? (domain < 1000000 ? `${(tick / 1000).toFixed(0)}k` : `${(tick / 1000000).toFixed(1)}M`) : tick.toFixed(0);
          addText(svg, tx, plotBottom + 17, label, "tick", "middle");
        });
        groups.forEach((group, index) => {
          const cy = plotTop + rowHeight * (index + .5);
          const color = modelColors[modelIndex[group.model]];
          const opacity = group.agent === "opencode-pinned" ? .68 : 1;
          const modelLabel = shortModel[group.model];
          addText(svg, left - 8, cy - 2, modelLabel, "row-label", "end");
          addText(svg, left - 8, cy + 12, shortAgent[group.agent], "row-label secondary", "end");
          if (metric === "accuracy") {
            const x = scale((group.accuracy || 0) * 100, domain, x0, x1);
            const low = scale((group.accuracy_low || 0) * 100, domain, x0, x1);
            const high = scale((group.accuracy_high || 0) * 100, domain, x0, x1);
            svg.appendChild(make("rect", {class: `series-${modelIndex[group.model] + 1}`, x: x0, y: cy - 10, width: x - x0, height: 20, opacity}));
            svg.appendChild(make("line", {class: "interval", x1: low, y1: cy, x2: high, y2: cy}));
            svg.appendChild(make("line", {class: "interval", x1: low, y1: cy - 7, x2: low, y2: cy + 7}));
            svg.appendChild(make("line", {class: "interval", x1: high, y1: cy - 7, x2: high, y2: cy + 7}));
            svg.appendChild(make("circle", {class: `series-${modelIndex[group.model] + 1}`, cx: x, cy, r: 4}));
            const label = `${((group.accuracy || 0) * 100).toFixed(1)}% (${group.n_passed}/${group.n_evaluable_trials})`;
            addText(svg, valueX, cy + 4, label, "value", "end");
          } else if (metric === "time" || metric === "tokens") {
            const median = metric === "time" ? group.total_elapsed_seconds : group.total_tokens;
            const q1 = metric === "time" ? group.total_elapsed_q1 : group.total_tokens_q1;
            const q3 = metric === "time" ? group.total_elapsed_q3 : group.total_tokens_q3;
            const x = scale(median, domain, x0, x1);
            const lo = scale(q1, domain, x0, x1);
            const hi = scale(q3, domain, x0, x1);
            svg.appendChild(make("rect", {class: `series-${modelIndex[group.model] + 1}`, x: x0, y: cy - 9, width: x - x0, height: 18, opacity}));
            svg.appendChild(make("line", {class: "interval", x1: lo, y1: cy, x2: hi, y2: cy}));
            svg.appendChild(make("circle", {class: `series-${modelIndex[group.model] + 1}`, cx: x, cy, r: 4}));
            const label = metric === "time" ? formatSeconds(median) : formatTokens(median);
            addText(svg, valueX, cy + 4, label, "value", "end");
          } else {
            const req = scale(group.model_requests, domain, x0, x1);
            const tool = scale(group.tool_calls, domain, x0, x1);
            svg.appendChild(make("rect", {class: `series-${modelIndex[group.model] + 1}`, x: x0, y: cy - 12, width: req - x0, height: 9, opacity}));
            svg.appendChild(make("rect", {class: "tools-bar", x: x0, y: cy + 3, width: tool - x0, height: 9}));
            addText(svg, valueX, cy + 4, `${group.model_requests} / ${group.tool_calls}`, "value", "end");
          }
        });
        svg.appendChild(make("line", {class: "axis", x1: x0, y1: plotBottom, x2: x1, y2: plotBottom}));
        const xAxis = make("text", {class: "axis-title", "data-axis": "x", x: (x0 + x1) / 2, y: height - 8, "text-anchor": "middle"}, axis);
        svg.appendChild(xAxis);
        const yMid = (plotTop + plotBottom) / 2;
        const yLabel = make("text", {class: "axis-title", "data-axis": "y", x: 12, y: yMid, transform: `rotate(-90 12 ${yMid})`}, "模型 / Agent");
        svg.appendChild(yLabel);
      };
      const drawAll = () => root.querySelectorAll("svg[data-metric]").forEach(svg => drawPanel(svg, svg.dataset.metric));
      const tableBody = document.getElementById("tb-accessible-rows");
      if (tableBody) groups.forEach(group => {
        const row = document.createElement("tr");
        [ `${group.model_display} / ${group.agent_display}`, `${(group.accuracy * 100).toFixed(1)}%`, formatSeconds(group.total_elapsed_seconds), formatTokens(group.total_tokens) ].forEach(value => { const cell = document.createElement("td"); cell.textContent = value; row.appendChild(cell); });
        tableBody.appendChild(row);
      });
      drawAll();
      if (typeof ResizeObserver !== "undefined") new ResizeObserver(drawAll).observe(root);
      else window.addEventListener("resize", drawAll);
    })();
  </script>
</section>
"""


def write_html(path: Path, summary: dict[str, Any]) -> None:
    data = json.dumps(
        {"groups": summary["groups"]}, ensure_ascii=False, separators=(",", ":")
    )
    path.write_text(HTML_TEMPLATE.replace("__DATA__", data), encoding="utf-8")


def render_png(html_path: Path, png_path: Path) -> str | None:
    chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
    )
    if not chrome:
        return "browser not found; PNG was not rendered"
    with tempfile.TemporaryDirectory(prefix="coding-agent-chart-") as profile:
        command = [
            chrome,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--hide-scrollbars",
            "--window-size=2400,880",
            "--virtual-time-budget=1500",
            f"--user-data-dir={profile}",
            f"--screenshot={png_path}",
            html_path.resolve().as_uri(),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=45
        )
    if (
        completed.returncode != 0
        or not png_path.exists()
        or png_path.stat().st_size == 0
    ):
        detail = (completed.stderr or completed.stdout).strip().splitlines()[-1:]
        return "browser screenshot failed: " + (
            detail[0] if detail else "unknown error"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = load_aggregate(args.report.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "terminal-bench-performance.csv", summary["groups"])
    write_trial_csv(output / "terminal-bench-trials.csv", summary["trial_rows"])
    write_markdown(output / "terminal-bench-performance.md", summary)
    (output / "terminal-bench-performance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_html(output / "terminal-bench-performance.html", summary)
    write_static_svg(output / "terminal-bench-performance.svg", summary)
    warning = render_png(
        output / "terminal-bench-performance.html",
        output / "terminal-bench-performance.png",
    )
    result = {
        "output": str(output),
        "files": sorted(
            path.name for path in output.glob("terminal-bench-*") if path.is_file()
        ),
        "groups": len(summary["groups"]),
        "evaluable_trials": summary["evaluable_trials"],
        "png_warning": warning,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if warning is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
