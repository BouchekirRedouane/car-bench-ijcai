# Codex Purple Agent

This package is the Codex-backed purple agent for CAR-bench A2A evaluation. It
preserves the same wire contract with the green evaluator while swapping the
assistant decision layer to Codex app-server.

## Design Choices

- **A2A stays the public interface.** Green sends `TextPart` and `DataPart`
  messages, and this agent returns either a user-facing `TextPart` or a
  `DataPart` with `{"tool_calls": [...]}`.
- **CAR-bench remains the evaluator.** The agent does not execute vehicle tools.
  It only decides the next response/tool call and lets green run tools, simulate
  the user, and score rewards.
- **One warm app-server process.** The agent keeps a single `codex app-server`
  subprocess alive and sends each assistant step as an ephemeral Codex thread.
  This avoids per-request `codex exec` startup cost while keeping each benchmark
  step grounded in the complete CAR-bench transcript.
- **Codex is deliberately constrained.** This harness does not expose Codex's
  normal coding-agent affordances during benchmark turns. Dynamic tools, shell
  commands, file changes, permission requests, network access, and user-input
  requests are denied by the adapter. Codex only returns the next CAR-bench
  action JSON.
- **MVP uses structured final JSON.** Codex is asked for one JSON object with
  `action`, `content`, and `tool_calls`. Tool arguments are returned as an
  `arguments_json` string because Codex structured outputs require closed JSON
  schemas; the adapter decodes that string before returning normal A2A
  `{"tool_name": "...", "arguments": {...}}` tool calls to green. Native
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

## App-Server Stability

`codex app-server` is marked experimental by the Codex CLI. The reference
adapter reduces that risk by using a tiny stable-protocol subset:

- It does not opt in to `capabilities.experimentalApi`.
- It follows the documented initialize handshake, including the `initialized`
  notification.
- It uses only `thread/start`, `turn/start`, item notifications, and
  `turn/completed`.
- It keeps all app-server JSON-RPC details isolated in `codex_client.py`.
- Dockerfiles pin `@openai/codex@0.130.0` by default so local image builds do
  not drift during the competition.

For a three-month run, publish and evaluate with pinned images. If you update
Codex CLI, regenerate the app-server schema with the new CLI and run the smoke
scenarios before accepting the update.

## Local Run

```bash
uv run agentbeats-run scenarios/scenario-test-codex.toml --show-logs
```

This expects Codex CLI to be available on `PATH` and already authenticated. You
can override the command explicitly in `.env`:

```bash
CODEX_APP_SERVER_CMD="/usr/local/bin/codex app-server --listen stdio://"
```

Change the model by editing `CODEX_MODEL` in `.env` or by passing
`--model <model-id>` to `server.py`. Spark is the recommended practical default
for time-budgeted benchmark runs, but advanced harnesses can use larger models
for selected planner or condenser steps.

## Docker Run

Build the Codex purple image with the included Dockerfile, then mount a writable
authenticated Codex home:

```bash
uv run python generate_compose.py --scenario scenarios/scenario-docker-codex.toml
mkdir -p output
docker compose up --abort-on-container-exit
```

Set `CODEX_HOME_HOST` in `.env` to an absolute host path containing Codex auth,
for example `/Users/alice/.codex`.

To intentionally test a newer or older CLI in Docker, override the build arg:

```bash
docker build \
  --build-arg CODEX_NPM_PACKAGE='@openai/codex@0.130.0' \
  -f src/purple_car_bench_agent_codex/Dockerfile.car-bench-agent-codex .
```

## Extending The Harness

Participants can add planner, critic, reranker, validator, memory, or
sub-agent-style components around the Codex call if those components only use
benchmark-allowed inputs and still return one final A2A response to green. Do
not execute vehicle tools, inspect hidden CAR-bench state, add private
capability tools, or let Codex perform file/shell/network side effects during
benchmark inference.

See [`../../docs/codex-harness-patterns.md`](../../docs/codex-harness-patterns.md)
for model-selection guidance and starter templates for Spark-only,
planner-plus-executor, Python-call DSL, ensemble-plus-condenser, and
budget-gated harnesses.
