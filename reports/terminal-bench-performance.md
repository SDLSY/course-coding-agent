# Terminal-Bench coding agent 性能摘要

> 8 个 Terminal-Bench 2.1 任务、每项 3 次的探索性实验；正式矩阵 144/144 条记录完整。
> 正确率按 24 个可评测 trial 计算；括号为 Wilson 95% 区间。耗时和 token 为中位数，方括号为 IQR（Q1-Q3）。

## 组合汇总

| 模型 | Agent | 通过 | 正确率 (95% CI) | 总耗时中位数 | 总 token 中位数 | 模型请求 / 工具调用 |
|---|---|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash Vision | CourseCodingAgent | 16/24 | 66.7% (46.7-82.0%) | 505 s [296-644] | 525.6k [266.0k-614.1k] | 30 / 29 |
| DeepSeek V4 Flash Vision | OpenCode 1.18.25 | 14/24 | 58.3% (38.8-75.5%) | 472 s [268-947] | 171.1k [87.3k-273.2k] | 15.5 / 15.5 |
| GLM-5.3 Flash | CourseCodingAgent | 17/24 | 70.8% (50.8-85.1%) | 347 s [154-518] | 60.2k [29.1k-86.6k] | 12 / 13 |
| GLM-5.3 Flash | OpenCode 1.18.25 | 15/24 | 62.5% (42.7-78.8%) | 761 s [225-974] | 158.4k [80.1k-294.7k] | 10 / 13 |
| GPT-5.6 Sol | CourseCodingAgent | 21/24 | 87.5% (69.0-95.7%) | 257 s [152-434] | 72.6k [28.4k-155.9k] | 10 / 17 |
| GPT-5.6 Sol | OpenCode 1.18.25 | 21/24 | 87.5% (69.0-95.7%) | 264 s [212-605] | 192.1k [167.7k-529.8k] | 13 / 21 |

合并来看，六个组合共通过 104/144（72.2%）。

## 按模型合计

| 模型（两个 Agent 合计） | 通过 | 正确率 (95% CI) |
|---|---:|---:|
| DeepSeek V4 Flash Vision | 30/48 | 62.5% (48.4-74.8%) |
| GLM-5.3 Flash | 32/48 | 66.7% (52.5-78.3%) |
| GPT-5.6 Sol | 42/48 | 87.5% (75.3-94.1%) |

## 按 Agent 合计

| Agent（三个模型合计） | 通过 | 正确率 (95% CI) |
|---|---:|---:|
| CourseCodingAgent | 54/72 | 75.0% (63.9-83.6%) |
| OpenCode 1.18.25 | 50/72 | 69.4% (58.0-78.9%) |

## 任务级通过数

| 任务 | DeepSeek/Course | DeepSeek/OpenCode | GLM/Course | GLM/OpenCode | GPT-5.6/Course | GPT-5.6/OpenCode |
|---|---:|---:|---:|---:|---:|---:|
| fix-git | 3/3 | 2/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| cancel-async-tasks | 1/3 | 1/3 | 2/3 | 0/3 | 3/3 | 2/3 |
| kv-store-grpc | 3/3 | 3/3 | 3/3 | 3/3 | 1/3 | 3/3 |
| polyglot-c-py | 3/3 | 2/3 | 2/3 | 2/3 | 3/3 | 3/3 |
| headless-terminal | 3/3 | 1/3 | 3/3 | 2/3 | 2/3 | 3/3 |
| fix-code-vulnerability | 3/3 | 3/3 | 3/3 | 2/3 | 3/3 | 3/3 |
| build-cython-ext | 0/3 | 1/3 | 1/3 | 3/3 | 3/3 | 3/3 |
| write-compressor | 0/3 | 1/3 | 0/3 | 0/3 | 3/3 | 1/3 |

## 主要结论

- 正确率最高的是 GPT-5.6 Sol + CourseCodingAgent 与 GPT-5.6 Sol + OpenCode：均为 21/24（87.5%）。
- 总耗时中位数最低的是 GPT-5.6 Sol + CourseCodingAgent（257 s）；GLM + OpenCode 最慢（761 s）。
- token 中位数最低的是 GLM-5.3 Flash + CourseCodingAgent（60.2k）。
- GLM 与 DeepSeek 的 CourseCodingAgent 正确率分别比同模型 OpenCode 高 8.3 和 8.4 个百分点；GPT-5.6 两个 Agent 正确率相同。
- 任务难度差异明显：`fix-git` 与 `fix-code-vulnerability` 各组合合计 17/18，`kv-store-grpc` 为 16/18；`write-compressor` 仅 5/18。
- 结果是小样本、固定任务子集上的探索性比较，不能替代完整 Terminal-Bench leaderboard，也不能单独证明因果优劣。

## 实验审计

- 模型路由的原生 `reasoning_effort=high` 探针均通过；Course agent 正式采用 `efficiency_30`，OpenCode 固定为 `1.18.25`。
- 最终矩阵无未决记录；历史基础设施问题共 2 组，均已恢复并从模型正确率中排除，不与 verifier failure 混计。
- 两组历史问题分别是 GLM/OpenCode 首次未正确转发 ZAI_API_KEY，以及 GPT-5.6/OpenCode 首次网关余额/计费失败；重跑后的 GPT-5.6/OpenCode 为 21/24。
- 脱敏的 144 条明细见 `terminal-bench-trials.csv`；详细原始 Harbor 日志和本机数据集不随仓库提交。本摘要和图表不包含 API key 值。

## 提交材料核对

按 `要求.pdf`，最终提交内容应只有一个以姓名命名的 ZIP，内部恰好包含：

1. `README.txt`：仓库地址、运行方法、特色说明，1000 汉字以内。
2. `李上一.mp4`：真实编程任务演示，MP4、不超过 2 分钟且不超过 200 MB。

当前仓库地址为 `https://github.com/SDLSY/course-coding-agent`；不要把 `.env`、API key、评测日志或本机轨迹放入提交包。
