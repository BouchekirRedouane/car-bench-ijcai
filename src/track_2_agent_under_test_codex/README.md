# Track 2 Codex Agent Under Test

This package is the direct Codex-backed Track 2 agent under test for CAR-bench
A2A evaluation. It preserves the same wire contract with the evaluator while
swapping the assistant decision layer to a warm Codex app-server runtime.
`gpt-5.3-codex-spark` is the default fast executor because it is served on
Cerebras infrastructure and is the practical model for the Track 2 time budget.

For the shared A2A turn contract, start with the
[development guide](../../docs/development-guide.md). For allowed advanced
harnessing boundaries and Track 2 patterns, see
[agent-under-test-harnessing.md](../../docs/agent-under-test-harnessing.md) and
[codex-harness-patterns.md](../../docs/codex-harness-patterns.md).

## What The Codex App-Server Is

The Codex reference agents do not call the OpenAI REST API directly. They also
do not run the interactive Codex chat UI. Instead, they start the Codex CLI in
app-server mode:

```bash
codex app-server --listen stdio://
```

In this mode, the Codex CLI becomes a small JSON-RPC service connected over
stdin/stdout. The Python agent owns the subprocess, sends JSON-RPC requests, and
receives app-server notifications. Authentication still comes from the local
Codex installation and `CODEX_HOME`, so local runs require an authenticated
`codex` command on `PATH`, while Docker runs require a mounted authenticated
Codex home.

This is useful for Track 2 because it gives participants subscription-backed
Codex inference, including access to `gpt-5.3-codex-spark`, while keeping the
public benchmark interface as ordinary A2A. CAR-bench still sees only text
responses and tool calls. The evaluator never talks to Codex directly.

The implementation boundary is:

| Layer | Responsibility |
| --- | --- |
| `server.py` | Starts the A2A HTTP server and warms Codex on startup. |
| `car_bench_agent.py` | Parses evaluator messages, builds Codex prompts, validates Codex output, and renders A2A responses. |
| `codex_client.py` | Owns the `codex app-server` subprocess and isolates JSON-RPC protocol details. |
| CAR-bench evaluator | Executes tool calls, simulates the user, records trajectories, and scores the task. |

## How One Benchmark Turn Uses Codex

Each A2A request from the evaluator becomes one Codex inference turn:

1. The evaluator sends the current user message or tool results, plus available
   CAR-bench tool definitions.
2. `car_bench_agent.py` updates the per-`context_id` transcript.
3. The agent builds a standalone Codex prompt containing the policy/wiki text,
   full transcript, available tools, and strict output instructions.
4. `codex_client.py` creates an ephemeral app-server thread, starts a turn, and
   waits for the final `agentMessage`.
5. The final Codex text is parsed as next-action JSON.
6. The adapter converts that JSON into the normal A2A shape:
   - user-facing speech becomes a text Part;
   - tool actions become a data Part with `{"tool_calls": [...]}`.
7. The evaluator executes any tool calls and sends the next A2A message.

The app-server process stays warm across turns, but the reference agent does
not rely on hidden Codex conversation state. Every Codex turn receives the full
benchmark transcript explicitly. That makes retries, logs, and CAR-bench
trajectory semantics easier to inspect.

Codex's coding-agent abilities are intentionally not part of the benchmark
surface. During benchmark inference, command execution, file edits, permission
requests, extra tools, and user-input requests are treated as adapter errors.
Participants may build richer internal harnesses, but the final visible output
must still be a CAR-bench text response or CAR-bench tool call.

## Design Choices

- **A2A stays the public interface.** The evaluator sends text and data Parts,
  and this agent returns either a user-facing text Part or a data Part with
  `{"tool_calls": [...]}`. In A2A SDK 1.0 these are protobuf `Part` objects,
  usually created with `new_text_part(...)` and `new_data_part(...)`.
- **CAR-bench remains the evaluator.** The agent does not execute vehicle tools.
  It only decides the next response/tool call and lets the evaluator run tools, simulate
  the user, and score rewards.
- **One warm app-server process.** The agent keeps a single `codex app-server`
  subprocess alive, initializes it during A2A server startup, and sends each
  assistant step as an ephemeral Codex thread. This avoids first-turn
  app-server startup cost while keeping each benchmark step grounded in the
  complete CAR-bench transcript.
- **No hidden Codex conversation state.** The adapter currently sends standalone
  Codex turns with the full CAR-bench transcript instead of continuing a hidden
  app-server thread. That keeps the evaluator trajectory, retries, and Codex
  context aligned. The prompt is structured with stable instructions and tool
  definitions before the dynamic transcript so OpenAI prompt caching can still
  help repeated prefixes.
- **Codex is deliberately constrained.** This harness does not expose Codex's
  normal coding-agent affordances during benchmark turns. Dynamic tools, shell
  commands, file changes, permission requests, network access, and user-input
  requests are denied by the adapter. Codex only returns the next CAR-bench
  action JSON.
- **MVP uses structured final JSON.** Codex is asked for one JSON object with
  `action`, `content`, and `tool_calls`. Tool arguments are returned as an
  `arguments_json` string because Codex structured outputs require closed JSON
  schemas; the adapter decodes that string before returning normal A2A
  `{"tool_name": "...", "arguments": {...}}` tool calls to the evaluator. Native
  dynamic tools are left for a later phase.
- **Benchmark comparability wins.** If Codex emits an unavailable CAR-bench tool
  or parameter, the adapter still passes it through as a tool call so CAR-bench
  hallucination and execution metrics can score it normally.

## Runtime Configuration

Important environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_HOME` | Codex default | Authenticated Codex home directory. Mount this into Docker. |
| `CODEX_APP_SERVER_CMD` | `codex app-server --listen stdio://` | Command used to start the warm app-server process. |
| `CODEX_MODEL` | `gpt-5.3-codex-spark` | Codex model used by the default next-action call. |
| `CODEX_REASONING_EFFORT` | `medium` | Reasoning effort passed to Codex turns. |
| `CODEX_TIMEOUT_SECONDS` | `180` | Per-turn timeout. |
| `CODEX_MALFORMED_RETRIES` | `1` | Retry budget when final JSON is malformed. |
| `CODEX_WORKDIR` | `/tmp/car-bench-codex-workdir` | Read-only sandbox working directory. |
| `CODEX_USAGE_LIMIT_RETRY_BUFFER_SECONDS` | `60` | Extra cushion after a Codex usage-limit reset time before retrying. |
| `CODEX_USAGE_LIMIT_MAX_WAIT_SECONDS` | unset | Optional cap; if the reset wait exceeds this, fail instead of sleeping. |
| `CAR_BENCH_RATE_LIMIT_REPORT_DIR` | `/tmp/car-bench-rate-limit-reports` | Directory for JSON snapshots written when Codex reports a usage limit. |
| `CAR_BENCH_A2A_TIMEOUT_SECONDS` | `86400` | HTTP timeout for long benchmark calls, including quota-reset waits. |

## App-Server Stability

`codex app-server` is marked experimental by the Codex CLI. The reference
adapter reduces that risk by using a tiny stable-protocol subset:

- It does not opt in to `capabilities.experimentalApi`.
- It follows the documented initialize handshake, including the `initialized`
  notification.
- It uses only `thread/start`, `turn/start`, item notifications,
  `thread/tokenUsage/updated`, and `turn/completed`.
- It keeps all app-server JSON-RPC details isolated in `codex_client.py`.
- Dockerfiles pin `@openai/codex@0.130.0` by default so local image builds do
  not drift during the competition.

The adapter maps app-server token usage into the standard CAR-bench
`turn_metrics` metadata when Codex emits `thread/tokenUsage/updated`.
`inputTokens`, `outputTokens`, and `reasoningOutputTokens` become
`prompt_tokens`, `completion_tokens`, and `thinking_tokens`. Per-turn cost is
not exposed reliably by app-server, so the reference agent reports `cost: 0.0`.
When Codex reports a temporary usage limit with a reset time, the adapter waits
and retries the same turn. That sleep is reported as `quota_wait_time_ms` and is
subtracted from benchmark latency and wall-clock summaries while raw timings
remain available for audit. The adapter also writes a temporary JSON report to
`CAR_BENCH_RATE_LIMIT_REPORT_DIR` containing the elapsed wall time before the
limit, reset time, wait duration, successful Codex call counts, and aggregate
token usage overall and by model. If another usage limit is hit after a reset,
the next report also includes `previous_retry_at` and
`wall_time_since_previous_retry_at_seconds`. The actual wake-up target is
reported as `retry_with_buffer_at`.

For a three-month run, publish and evaluate with pinned images. If you update
Codex CLI, regenerate the app-server schema with the new CLI and run the smoke
scenarios before accepting the update.

## Local Run

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_codex/local_smoke.toml --show-logs
```

This expects Codex CLI to be available on `PATH` and already authenticated. For
local setup, install Codex CLI with the official
[OpenAI Codex CLI instructions](https://developers.openai.com/codex/cli), then
log in with a dedicated Codex home:

```bash
mkdir -p "$HOME/.codex-car-bench"
CODEX_HOME="$HOME/.codex-car-bench" codex login
CODEX_HOME="$HOME/.codex-car-bench" codex login status
```

You can override the app-server command explicitly in `.env`:

```bash
CODEX_APP_SERVER_CMD="/usr/local/bin/codex app-server --listen stdio://"
```

Change the model by editing `CODEX_MODEL` in `.env` or by passing
`--model <model-id>` to `server.py`. Spark is the recommended practical default
for Track 2 time-budgeted runs, but advanced harnesses can use larger Codex
models for selected planner, verifier, or condenser steps if the complete agent
stays within the official time budget.

## Docker Run

Build the Codex agent-under-test image with the included Dockerfile, then mount a writable
authenticated Codex home:

```bash
uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_codex/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_2_agent_under_test_codex/docker-compose.yml up --abort-on-container-exit
```

Set `CODEX_HOME_HOST` in `.env` to an absolute host path containing Codex auth,
for example `/Users/alice/.codex`.
Prefer creating a dedicated benchmark home with
`CODEX_HOME="$HOME/.codex-car-bench" codex login` instead of mounting your
everyday Codex desktop/app state.

To intentionally test a newer or older CLI in Docker, override the build arg:

```bash
docker build \
  --build-arg CODEX_NPM_PACKAGE='@openai/codex@0.130.0' \
  -f src/track_2_agent_under_test_codex/Dockerfile.track-2-agent-under-test-codex .
```

The Dockerfile installs Codex in a Node build stage and recreates the runtime
`codex` launcher as a symlink into that global package so npm-managed optional
dependencies resolve normally.

There is no active auto-publishing workflow in this repository. To publish an
agent image to GHCR, either use `docker build` / `docker push` manually, or copy
the disabled template `.github/workflows/publish-ghcr.yml.disabled` to a real
workflow file and update it for your own image. The template is manual-only.

## Extending The Harness

Participants can add planner, critic, reranker, validator, memory, or
sub-agent-style components around the Codex call if those components only use
benchmark-allowed inputs and still return one final A2A response to the evaluator. Do
not execute vehicle tools, inspect hidden CAR-bench state, add private
capability tools, or let Codex perform file/shell/network side effects during
benchmark inference.

See [`../../docs/codex-harness-patterns.md`](../../docs/codex-harness-patterns.md)
for model-selection guidance and starter templates for Spark-only,
planner-plus-executor, Python-call DSL, ensemble-plus-condenser, and
budget-gated harnesses.
