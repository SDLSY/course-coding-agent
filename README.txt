Git仓库地址：https://github.com/SDLSY/course-coding-agent

项目说明：本项目是不使用 Agent 框架的轻量编程智能体。它通过 OpenAI-compatible Chat Completions 和原生 tool calling 工作，自行实现规范历史、上下文裁剪、响应解析、工具注册与本地执行、状态循环、重试、预算和错误处理。提供文件列表、文本搜索、分段读取、原子写入、唯一替换和有界命令执行六个工具。

运行环境：Python 3.11+、POSIX 系统。执行 `python3 -m pip install -e .` 安装。API key 只放在环境变量中，模型 ID 必须显式填写。例如设置 `DEEPSEEK_API_KEY` 后运行：`coding-agent --workspace <项目目录> --provider deepseek --model <实际模型ID> "修复失败测试并验证"`。GLM 默认读取 `ZAI_API_KEY`；自定义兼容网关使用 `--provider custom --base-url <HTTPS地址> --key-env <变量名>`。

特色功能：规范历史严格保持 tool call/result 配对；按完整事务块裁剪上下文；模型请求、协议错误和上下文溢出采用独立有限恢复策略；文件路径阻止越界；命令限制时间、输出并终止普通子进程组；密钥不传给命令子进程；终端和 JSONL trace 展示状态并脱敏。可选 `--verify` 在 Agent 结束后独立运行固定验收命令，`--planning` 记录无副作用的计划快照；`coding-agent-benchmark` 支持干净 fixture、统一预算和 JSON 结果。`COMPLETED` 只表示模型正常停止，不代表代码正确。

Harbor 适配器把同步 Runtime 放入异步桥接，并将六个工具转发到容器 `/app`；核心不依赖 Harbor/Docker，也不是强沙箱。Benchmark/Terminal-Bench 仅用于后续对比。

测试：`python3 -m pytest`；静态检查：`python3 -m ruff check src tests`。
