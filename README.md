# Course Coding Agent

一个不依赖 Agent 框架、从普通模型 API 之上自行实现控制循环的轻量编程智能体。给定自然语言任务和本地工作区后，它会让模型选择工具，依次读取、搜索和修改文件，运行命令，并把每次观察结果加入对话历史，直到模型返回最终回答或运行预算耗尽。

本项目面向软件工程课程项目和 Agent Runtime 学习，而不是商业编程助手的完整替代品。本 README 同时记录公开的设计边界、运行方式和验收约束。

## 项目边界

项目只使用 `openai` Python 包完成 HTTP、鉴权和 Chat Completions 请求序列化，**没有使用 OpenAI Agents SDK、LangChain、AutoGen 或其他 Agent 编排框架**。原生 tool calling 只提供结构化的模型输出；以下逻辑均由本仓库实现：

- 规范对话历史与 tool call/tool result 配对；
- 基于字符预算的上下文构建与完整事务块裁剪；
- 模型响应归一化、工具参数 JSON 解析和协议校验；
- 本地工具注册、参数校验、顺序执行和错误包装；
- 模型重试、轮数/工具数/时间预算和终态转换；
- 终端事件与可选的本地 JSONL 轨迹。

`COMPLETED` 只表示模型返回了非空最终回答且没有继续请求工具，不代表修改已经被证明正确。应以测试、静态检查或其他验收命令的实际结果判断任务是否完成。

## 运行循环

```text
自然语言任务
     |
     v
构建预算内上下文 ---> 调用 OpenAI-compatible 模型
                           |
                           v
                    归一化并校验响应
                      /            \
             tool calls             final text
                 |                      |
                 v                      v
       按顺序执行本地工具             COMPLETED
                 |
                 v
       追加 assistant + tool results
                 |
                 +------> 检查预算后进入下一轮

任意阶段也可能进入 LIMIT_REACHED、FAILED 或 CANCELLED。
```

规范历史在进程内只追加；每轮发给模型的视图由 `ContextBuilder` 派生。包含多个工具调用的 assistant 消息及其全部工具结果会整体保留或整体裁剪，避免产生孤立的 tool call。

## 环境要求与安装

- Python 3.11 或更高版本；
- 首版 `run_command` 仅支持 POSIX 系统；
- 一个支持 OpenAI-compatible Chat Completions 和原生 tool calling 的模型服务。

推荐在虚拟环境中安装：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .
coding-agent --help
```

若系统 Python 报告缺少 `ensurepip`，先安装与该解释器版本匹配的系统 `venv` 包，或在已有的受管理 Python 环境中执行安装命令；这属于主机 Python 配置，不是 Agent 的运行依赖。

需要运行测试时安装测试依赖：

```bash
python3 -m pip install -e ".[test]"
```

## 模型配置

配置只从 CLI 参数和当前进程环境变量读取，CLI 参数优先。程序**不会自动加载 `.env` 文件**，也不提供 `--api-key` 明文参数；`--key-env` 接收的是保存密钥的环境变量名称。不要将真实密钥写入仓库、命令行参数、README、截图或视频。

Provider 预设只选择 base URL 和默认密钥变量名，**不会隐式选择模型**。每次运行都必须通过 `--model` 或 `CODING_AGENT_MODEL` 明确给出服务端实际模型 ID。

| Provider | 预设 base URL | 默认密钥变量 | 模型 |
|---|---|---|---|
| `deepseek` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | 必须显式填写 |
| `glm` | `https://open.bigmodel.cn/api/paas/v4` | `ZAI_API_KEY` | 必须显式填写 |
| `custom` | 必须通过参数或环境变量填写 | `CODING_AGENT_API_KEY` | 必须显式填写 |

DeepSeek 与 GLM 的 API 根地址分别依据其[官方 API 文档](https://api-docs.deepseek.com/quick_start/pricing)和[官方 OpenAI SDK 兼容文档](https://docs.bigmodel.cn/cn/guide/develop/openai/introduction)。自定义远程地址必须使用 HTTPS；仅回环地址允许 HTTP。使用 `custom` 时，API key 会发送给指定网关，请先确认网关归属和可信度。

以下全部是不可用的占位符示例。

### DeepSeek

```bash
export DEEPSEEK_API_KEY="replace-with-deepseek-api-key"

coding-agent \
  --workspace ./example-project \
  --provider deepseek \
  --model "replace-with-deepseek-model-id" \
  "修复失败测试，并运行相关测试确认结果"
```

### GLM

也可以把非秘密配置放在环境变量中：

```bash
export ZAI_API_KEY="replace-with-glm-api-key"
export CODING_AGENT_PROVIDER="glm"
export CODING_AGENT_MODEL="replace-with-glm-model-id"

coding-agent \
  --workspace ./example-project \
  --trace .coding-agent/run.jsonl \
  "找出边界条件错误，进行最小修改并执行测试"
```

### 自定义 OpenAI-compatible 网关

下面的例子显式展示所有主要预算参数。任务可以作为位置参数传入，也可以使用 `--task`，但不能同时使用两种形式。

```bash
export CODING_AGENT_API_KEY="replace-with-custom-gateway-api-key"

coding-agent \
  --task "阅读项目，修复指定问题，并报告实际运行的检查" \
  --workspace ./example-project \
  --provider custom \
  --model "replace-with-custom-model-id" \
  --base-url "https://gateway.example/v1" \
  --key-env CODING_AGENT_API_KEY \
  --trace .coding-agent/run.jsonl \
  --max-model-turns 20 \
  --max-tool-calls 80 \
  --max-wall-time 600 \
  --context-chars 120000 \
  --model-timeout 120 \
  --model-retries 2 \
  --protocol-retries 1
```

支持的环境变量如下：

| CLI 参数/用途 | 环境变量 | 未提供时的行为 |
|---|---|---|
| task | `CODING_AGENT_TASK` | 必须提供 |
| `--workspace` | `CODING_AGENT_WORKSPACE` | 当前目录 |
| `--provider` | `CODING_AGENT_PROVIDER` | 必须提供 |
| `--model` | `CODING_AGENT_MODEL` | 必须提供 |
| `--base-url` | `CODING_AGENT_BASE_URL` | 使用 provider 预设；`custom` 必须提供 |
| `--key-env` | `CODING_AGENT_KEY_ENV` | 使用上表中的默认变量名 |
| `--trace` | `CODING_AGENT_TRACE_PATH` | 不写 JSONL 文件 |
| `--max-model-turns` | `CODING_AGENT_MAX_MODEL_TURNS` | `20` |
| `--max-tool-calls` | `CODING_AGENT_MAX_TOOL_CALLS` | `80` |
| `--max-wall-time` | `CODING_AGENT_MAX_WALL_TIME` | `600` 秒 |
| `--context-chars` | `CODING_AGENT_CONTEXT_CHAR_BUDGET` | `120000` 字符 |
| `--model-timeout` | `CODING_AGENT_MODEL_TIMEOUT` | `120` 秒 |
| `--model-retries` | `CODING_AGENT_MODEL_MAX_RETRIES` | `2` |
| `--protocol-retries` | `CODING_AGENT_PROTOCOL_MAX_RETRIES` | `1` |

## 本地工具

所有调用都经过统一的 `ToolRegistry`，包括未知工具、非法 JSON、缺少参数和执行错误。普通工具失败会形成结构化结果反馈给模型，而不是直接破坏控制循环。

| 工具 | 作用 |
|---|---|
| `list_files` | 有界地列出工作区文件 |
| `search_text` | 在文本文件中搜索字符串并返回行号和片段 |
| `read_file` | 按行范围读取 UTF-8 文本文件 |
| `write_file` | 新建文件或原子替换整个文件 |
| `replace_in_file` | 仅在旧文本唯一匹配时进行原子替换 |
| `run_command` | 在工作区目录中执行非交互 POSIX shell 命令 |

### 安全边界

文件工具会拒绝绝对路径、`..` 路径逃逸和指向工作区外部的符号链接；写入使用同目录临时文件和 `os.replace`，降低半写入风险。

`run_command` 使用工作区作为 `cwd`，禁用交互输入，限制 stdout/stderr 返回量，并在超时后终止普通子进程组。当前模型密钥所在的变量会按确切名称从子进程环境中移除，其他常见凭据变量则按名称模式过滤；这种过滤仍不可能发现任意命名的所有秘密。

**`run_command` 不是安全沙箱。** 模型生成的 shell 命令仍可访问绝对路径、网络以及当前用户有权访问的主机资源，也可能执行仓库中的恶意脚本。工作区 `cwd`、环境变量过滤、输出截断和 timeout 都不等于容器或操作系统级隔离。首版只能用于用户授权且可信的本地工作区；运行前请确认工作区内容和 API 网关可信。

## 运行事件与 JSONL Trace

CLI 默认在 stderr 显示简洁事件，包括模型请求、工具调用、上下文裁剪、重试和终态。指定 `--trace PATH` 后，同一批事件还会追加到本地 JSONL 文件：

```bash
coding-agent \
  --workspace ./example-project \
  --provider deepseek \
  --model "replace-with-deepseek-model-id" \
  --trace .coding-agent/run.jsonl \
  "检查并修复失败测试"
```

每行包含 `schema_version`、UTC `timestamp`、`event` 和 `data`。新建 trace 文件权限为 `0600`；`.gitignore` 已忽略 `.coding-agent/` 和 `traces/`，建议将 trace 保存在其中。任意其他路径不会自动变成 Git ignored。

事件写入前会递归处理敏感字段，并替换当前模型 API key、Bearer 值和常见 key 模式。这是尽力而为的日志保护，不保证识别项目文件或命令输出中的所有秘密。Trace 用于审计控制流和排错，不是可重放的 checkpoint，也不能复现非确定性的模型输出和外部命令状态。

## 独立验收与可选计划

Runtime 的终态和任务正确性是两个不同问题。可以在 Agent 返回后指定一个或多个固定验收命令：

```bash
coding-agent \
  --workspace ./example-project \
  --provider deepseek \
  --model "replace-with-model-id" \
  --verify "python3 -m pytest -q" \
  "修复失败测试"
```

`--verify`（也可写作 `--verify-command`）只接受运行者预先配置的命令。命令在 Runtime 结束后独立运行，不进入模型历史，也不计入 Agent 的模型/工具预算。报告会显示每个检查的退出码和 timeout；模型正常停止但检查失败时，CLI 返回专用退出码 `4`。没有指定 `--verify` 时，原有退出码和行为保持不变。验收命令同样运行在当前用户权限下，不能视为安全沙箱。

## 提交包

真实演示录制完成后，用 `scripts/build_submission.py` 生成严格的两文件归档。脚本默认检查 UTF-8 `README.txt` 不超过 1000 字符、MP4 不超过 200 MB 且时长不超过 120 秒，并扫描常见 key/Bearer 模式和当前进程中已配置的 key 值：

```bash
python3 scripts/build_submission.py --video /path/to/demo.mp4 --output 李上一.zip
```

成功后 ZIP 内部只能有 `README.txt` 和 `李上一.mp4`。脚本不会创建占位视频；没有 `ffprobe` 时必须显式使用 `--skip-duration-check`，且不能把该离线例外当成最终验收。

`--planning` 是一个显式 opt-in 的 `update_plan` 工具。它只保存 1--8 步的内存快照，用于观察模型意图；计划全部标记为 `completed` 也不会自动终止 Runtime，更不能证明代码正确。默认仍只暴露六个 MVP 工具。

## Harbor 外部适配

核心包不依赖 Harbor。`coding_agent.harbor_adapter` 提供一个可选异步桥接：同步 `AgentRuntime` 在工作线程运行，`RemoteExecutionBackend` 将六个工具的操作通过异步环境接口发送到任务容器的 `/app` 工作区，模型请求仍在主机侧完成。适配器不向容器传递 API key，并可把 ATIF-v1.7 `trajectory.json` 和运行摘要原子写入 Harbor 日志目录。

Harbor 0.22 的官方入口是 `coding_agent.harbor_plugin:CourseCodingAgent`。它只在安装可选依赖后加载，安装方式为：

```bash
python3 -m pip install -e ".[harbor]"
```

核心运行时支持 Python 3.11+；Harbor 0.22 本身要求 Python 3.12+，因此 Harbor 入口应在 Python 3.12 或更高版本的环境中安装和执行。

插件实现 Harbor `BaseAgent` 的 `name`、`version`、`setup` 和 `run` 接口。模型请求使用宿主进程中的 `CODING_AGENT_MODEL`、`CODING_AGENT_PROVIDER`、`CODING_AGENT_BASE_URL` 和命名 key 变量（GLM 默认 `ZAI_API_KEY`）；`model_name` 参数优先且按原样发送，不猜测别名。容器只接收经过路径/参数校验的六个工具调用。运行结束后，`AgentContext.metadata` 保存脱敏终态、最终回答和统计，日志目录保存 `trajectory.json`、`run.json` 与脱敏事件流。

一个单任务冒烟命令（需已安装并运行 Docker、准备好 Harbor 数据集）如下：

```bash
export ZAI_API_KEY="<只在当前 shell 设置>"
export CODING_AGENT_MODEL="<控制台原样的 GLM model ID>"
harbor trial start \
  --agent coding_agent.harbor_plugin:CourseCodingAgent \
  --model "$CODING_AGENT_MODEL" \
  --path datasets/terminal-bench-2.1/fix-code-vulnerability
```

密钥只通过环境变量提供；不要把 `--agent-env KEY=VALUE`、命令输出或 Harbor 日志提交到 Git。

这是外部评测适配层，不是本地强隔离实现；Harbor/Docker 只在执行外部评测时需要，核心安装和默认测试不会启动容器。本仓库的实验脚本会把任务数据集、并发度、重试次数和两组 Agent 的命令固定下来，结果应标注为八任务、三次重复的探索性实验，而不是官方 leaderboard 成绩。

## 有界 Benchmark

`coding-agent-benchmark` 是一个独立的 fixture-based 实验工具。Manifest 为每个任务声明干净 fixture、固定验收命令、Agent argv 和统一预算；每次运行复制新工作区，验收在 Agent 进程退出后单独执行，结果输出为统一 JSON。默认是无副作用的计划模式，只有显式 `--execute` 才会启动 Agent 命令：

```bash
coding-agent-benchmark --manifest benchmarks/example.json --dry-run
coding-agent-benchmark --manifest benchmarks/example.json --execute --output result.json
```

它可以比较同一模型驱动的不同 Agent，但不是官方 leaderboard 成绩。Terminal-Bench 2.1 的后置实验固定为 3 个模型、2 个 Agent、8 个任务、每任务 3 次（144 个 trial）；公开套件共有 89 个任务，二者不能混称。实验脚本还提供两个失败任务的 20/20-efficient/30 轮消融，以及正式运行前的 `fix-code-vulnerability` 六组合冒烟测试。

正式执行顺序固定为：先完成两个失败任务的 `--ablation --execute`，再执行
`--smoke --execute`，最后把消融报告通过 `--ablation-report`、把六组合
冒烟报告通过 `--smoke-report` 传给 `--formal --execute`。formal 会在启动
任何 Harbor job 前拒绝缺失、未完成或 setup/config/infrastructure 失败的
冒烟报告，也会拒绝缺失或未完成的消融报告；`--allow-incomplete` 仅用于
离线计划和部分产物生成，不能放宽可执行 formal 门禁。verifier failure 会保留并单独统计。完整命令、派生镜像准备和报告归档说明见
[`benchmarks/README.md`](benchmarks/README.md)。仓库中的
[`reports/terminal-bench-performance.md`](reports/terminal-bench-performance.md)
是此前完成的脱敏 144-trial 结果快照；原始 Harbor 日志和本机数据集不随仓库提交。默认测试和计划生成均为离线操作。

需要让外部 Agent 使用模型服务时，不要把 key 写进 manifest。可以在该 Agent 项中写
`"environment_from_host": ["DEEPSEEK_API_KEY"]`，运行时只按这个明确的变量名从当前进程读取值；报告只记录变量名，命令输出仍会脱敏。未列出的敏感宿主变量不会传入子进程。

## 测试

默认测试完全使用临时工作区、Fake/Scripted Model 和伪造响应，不需要 API key，也不会调用真实模型服务：

```bash
python3 -m pytest
python3 -m ruff check src tests
python3 -m ruff format --check src tests
```

测试覆盖模型响应解析、错误分类、上下文事务裁剪、Agent 状态转换、预算终止、工具参数校验、文件路径/符号链接边界、原子写入、命令超时与进程组终止、输出截断、凭据环境过滤和事件脱敏。真实 API 冒烟测试应由开发者单独执行，不属于默认测试套件。

## 退出码

| 退出码 | 含义 |
|---|---|
| `0` | `COMPLETED`：模型返回正常最终回答 |
| `1` | Runtime 或 Agent 进入失败状态 |
| `2` | CLI 用法或启动配置错误 |
| `3` | 达到模型轮数、工具调用数或运行时间预算 |
| `4` | 仅在指定 `--verify` 且验收检查失败时返回 |
| `130` | 用户通过 Ctrl-C 等方式取消 |

最终回答写入 stdout；事件和运行摘要写入 stderr，便于脚本分别处理结果与诊断信息。

## 当前限制

- 单进程、单 Agent，多个工具调用按模型返回顺序串行执行；
- 只接入非流式 OpenAI-compatible Chat Completions，不对所有兼容网关作兼容性承诺；
- provider 预设不检查具体模型是否支持 tool calling，模型 ID 和能力由使用者确认；
- `run_command` 仅支持 POSIX，没有容器隔离、逐命令人工确认或权限降级；
- 总运行时间使用单一 deadline，模型请求和命令 timeout 会被压缩到剩余预算；进程终止宽限、系统调度和进程内文件操作仍使它不是实时系统级的绝对期限；
- 上下文预算使用确定性字符估算，不是目标模型 tokenizer 的精确 token 数；
- 不实现 streaming、多 Agent、RAG、向量数据库、长期记忆、断点恢复或执行重放；
- 不实现通用 unified diff 解析器，大范围编辑的效率有限；
- 模型输出具有非确定性，任务正确性仍需外部测试或人工验收。
