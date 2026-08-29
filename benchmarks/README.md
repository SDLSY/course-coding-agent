# Benchmark fixtures

`example.json` is a self-contained offline smoke benchmark. It is safe to run
with `coding-agent-benchmark --manifest benchmarks/example.json --dry-run`; use
`--execute` only when the manifest's agent command is trusted.

The planned Terminal-Bench 2.1 experiment is intentionally not checked in as a
runnable manifest because its task fixtures and Docker/Harbor environment are
external. The current comparison scope is eight official coding-oriented task
IDs, each run once with the same model configuration:

`fix-git`, `cancel-async-tasks`, `kv-store-grpc`, `polyglot-c-py`,
`headless-terminal`, `fix-code-vulnerability`, `build-cython-ext`,
`write-compressor`.

Those rows are a post-delivery experiment, not an official Terminal-Bench
leaderboard submission. The public suite has 89 tasks, and official reporting
normally repeats each task at least five times.

For an external Agent that needs a provider credential, keep the value out of
the manifest and add an explicit allow-list such as
`"environment_from_host": ["DEEPSEEK_API_KEY"]`. The runner copies only those
named variables for that Agent case; reports contain names, not values, and
captured output is redacted.
