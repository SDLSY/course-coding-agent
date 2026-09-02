# Benchmark fixtures

`example.json` is a self-contained offline smoke benchmark. It is safe to run
with `coding-agent-benchmark --manifest benchmarks/example.json --dry-run`; use
`--execute` only when the manifest's agent command is trusted.

The Terminal-Bench 2.1 experiment is intentionally driven by a planner rather
than a checked-in Harbor manifest because its Docker environment is external.
The formal scope is eight official coding-oriented task IDs, three fixed model
routes, two agents, and three repetitions per task (144 trials):

`fix-git`, `cancel-async-tasks`, `kv-store-grpc`, `polyglot-c-py`,
`headless-terminal`, `fix-code-vulnerability`, `build-cython-ext`,
`write-compressor`.

Those rows are a post-delivery exploratory experiment, not an official
Terminal-Bench leaderboard submission. The public suite has 89 tasks; the
eight-task, three-repeat matrix must not be presented as a full-suite score.

The reproducible command planner is `run_terminal_bench_2_1.py`. Its default
dataset is pinned to a Harbor Hub content digest. The formal plan fixes the
eight task names, three attempts, `n-concurrent=1`, `max-retries=0`, and the
same 900-second agent cap for both agents. OpenCode is the local
`PinnedOpenCodeAgent` with Node `22.22.1` and OpenCode `1.18.25` pre-installed
in derived task images; a trial never runs nvm/npm or an unpinned install.

```bash
# Inspect the three plans first (these commands do not contact a model).
python3 benchmarks/run_terminal_bench_2_1.py --ablation \
  --output .harbor-runs/ablation-plan.json
python3 benchmarks/run_terminal_bench_2_1.py --smoke \
  --output .harbor-runs/smoke-plan.json
python3 benchmarks/run_terminal_bench_2_1.py --formal \
  --output .harbor-runs/formal-plan.json

# After Docker, the eight derived images, and the named key variables pass
# their preflight checks, execute in this order. Each command is opt-in.
python3 benchmarks/run_terminal_bench_2_1.py --ablation \
  --execute \
  --image-manifest .harbor-opencode-images/manifest.json \
  --jobs-dir .harbor-runs/ablation \
  --output .harbor-runs/ablation/ablation-report.json \
  --artifacts-dir .harbor-runs/ablation-artifacts
python3 benchmarks/run_terminal_bench_2_1.py --smoke \
  --execute \
  --image-manifest .harbor-opencode-images/manifest.json \
  --jobs-dir .harbor-runs/smoke \
  --output .harbor-runs/smoke/smoke-report.json
python3 benchmarks/run_terminal_bench_2_1.py --formal \
  --execute \
  --image-manifest .harbor-opencode-images/manifest.json \
  --ablation-report .harbor-runs/ablation/ablation-report.json \
  --smoke-report .harbor-runs/smoke/smoke-report.json \
  --jobs-dir .harbor-runs/formal \
  --output .harbor-runs/formal/formal-report.json \
  --artifacts-dir .harbor-runs/formal-artifacts
```

On hosts where Docker/Harbor is available only through passwordless sudo,
append `--harbor-sudo` to each planner command. This records and executes the
explicit `sudo -n <harbor-bin>` prefix; it never prompts for a password. The
default is direct `harbor` invocation, so local callers can continue to use
the planner without sudo.

`sudo` normally uses a restricted `secure_path` and removes arbitrary
environment variables. During execution the runner resolves a bare Harbor
binary from the caller's `PATH` (when possible) and adds a name-only
`--preserve-env` allow-list for the selected credential and route settings;
secret values never enter argv or reports. If Harbor is installed in a virtual
environment that is not on `PATH`, pass its absolute path explicitly, for
example `--harbor-bin /tmp/harbor-venv/bin/harbor`.

Build and rewrite the pinned task checkout before those commands with
`benchmarks/pin_opencode_images.py`. Give it the local official checkout via
`--source-dataset` and a separate `--output-dataset`; use `--execute` only after
Docker access is verified. The generated manifest records each source and
derived image digest, Node `22.22.1`, and OpenCode `1.18.25`. Pass the rewritten
checkout as `--dataset` to each runner command and pass the same manifest with
`--image-manifest`. The runner checks that the manifest's rewritten output
directory matches `--dataset` and records a manifest hash in every trial row.
If the host is air-gapped, use
an audited `--toolchain-image` that already contains the two pinned binaries;
the generated Dockerfile then contains no package-manager or network step.

The formal command refuses to launch any Harbor job unless the supplied smoke
report is complete: it must contain exactly one `fix-code-vulnerability` trial
for each of the six model/agent combinations. Setup, configuration, and other
infrastructure failures block the run; verifier failures are retained and
reported separately, so a zero reward is never silently treated as a setup
failure. It also requires `--ablation-report` from the completed 54-trial
Course-agent round-budget ablation; an incomplete report, or
`--allow-incomplete` on an executable formal command, is rejected before the
first Harbor job starts. Dry plans may omit both reports and record the gates as
`not_checked`; `--allow-incomplete` is intended only for those offline plans or
partial artifact generation.

The three fixed routes are `deepseek-v4-flash-vision-exp` with
`DEEPSEEK_API_KEY`, `glm-5.3-flash` with `ZAI_API_KEY`, and `gpt-5.6-sol`
through `https://spacetimeai.cc/v1` with `CODING_AGENT_API_KEY`. The key
values are read only from the current process; plans and reports contain the
variable names, never their values. Before the formal matrix, run the
`--smoke` plan/execution on `fix-code-vulnerability`. A probe failure is
recorded as setup/infrastructure failure; it is not silently replaced with a
different model or a prompt-level imitation of high reasoning.

The report keeps the invocation path, immutable `dataset_reference`, route and
reasoning-probe metadata. Harbor's per-trial `result.json` and
`trajectory.json` remain the source of status, timing, token/tool counts, and
ATIF evidence. Captured output is written with mode `0600` after redaction. Do
not commit `.harbor-runs/`.

The two failed Course-agent tasks are evaluated first with the `--ablation`
plan: current 20 turns, efficiency 20 turns, and efficiency 30 turns, each
for all three models and three repetitions (54 trials). The formal Course
configuration is fixed to efficiency 20 only when it matches or exceeds the
30-turn pass count for every complete model/task comparison; otherwise 30
turns is selected. The ablation archive is written with `--artifacts-dir`.

For an external Agent that needs a provider credential, keep the value out of
the manifest and add an explicit allow-list such as
`"environment_from_host": ["DEEPSEEK_API_KEY"]`. The runner copies only those
named variables for that Agent case; reports contain names, not values, and
captured output is redacted.
