Git仓库地址：https://github.com/SDLSY/course-coding-agent

项目说明：这是一个不使用 Agent 框架的轻量编程智能体。它通过 OpenAI-compatible Chat Completions 和原生 tool calling，自行管理对话历史、上下文裁剪、响应解析、工具注册、本地执行、重试、预算和终止状态；提供列文件、搜索、读取、写入、唯一替换和有界命令六个工具。

运行环境：Python 3.11+、POSIX。安装：`python3 -m pip install -e .`。API key 只放环境变量，模型 ID 必须显式填写：`coding-agent --workspace <目录> --provider deepseek --model <模型ID> "修复失败测试并验证"`。GLM 默认读取 `ZAI_API_KEY`；自定义网关使用 `--provider custom --base-url <HTTPS地址> --key-env <变量名>`。

特色功能：保持 tool call/result 严格配对，按完整事务块裁剪上下文；模型、协议和超时有界恢复；文件路径防越界；命令限时、限输出并过滤密钥；终端和 JSONL trace 脱敏。`--verify` 可独立验收，`--planning` 可记录计划，benchmark 输出统一 JSON。`COMPLETED` 不等于代码已正确。

Harbor 入口为 `coding_agent.harbor_plugin:CourseCodingAgent`（需 Python 3.12+ 和 Harbor 0.22）；Runtime 在主机运行，六个工具转发到容器 `/app`，日志输出脱敏 ATIF。核心不依赖 Harbor/Docker，也不是强沙箱。

测试：`python3 -m pytest`；静态检查：`python3 -m ruff check src tests`。
