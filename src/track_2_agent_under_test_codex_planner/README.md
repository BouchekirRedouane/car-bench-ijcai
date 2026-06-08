# Track 2 Codex Planner/Executor Agent Under Test

This package is a Track 2 reference implementation for a plan-on-user-turn
Codex Pro / Cerebras Spark harness:

1. A private planner call uses `gpt-5.5` to emit a compact
   `planning_tool`-shaped plan after each user message.
2. One or more Spark executor calls use `gpt-5.3-codex-spark`, served on
   Cerebras infrastructure, to return normal benchmark action JSON.
3. Tool-result turns reuse the active private plan until the executor responds
   to the user, then the plan is cleared.

The planner output is internal reasoning. It is not sent to the evaluator as a
CAR-bench tool call, and the evaluator remains the only component that executes
vehicle tools.

This README only describes what is special about the planner/executor variant.
Read the [base Codex agent README](../track_2_agent_under_test_codex/README.md)
first for the shared technical setup: what `codex app-server` is, how Codex is
started and authenticated, how A2A messages are converted into Codex turns, and
which benchmark boundaries the adapter enforces.

Like the direct Codex agent, this variant warms the shared `codex app-server`
process during A2A server startup, then uses explicit transcript-based Codex
turns. The difference is that a user message triggers one private planner turn
before the executor starts returning benchmark-visible actions.

For the shared A2A turn contract, start with the
[development guide](../../docs/development-guide.md). For allowed advanced
harnessing boundaries and additional Track 2 patterns, see
[agent-under-test-harnessing.md](../../docs/agent-under-test-harnessing.md) and
[codex-harness-patterns.md](../../docs/codex-harness-patterns.md).

## Model Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_PLANNER_MODEL` | `gpt-5.5` | Private planner model. |
| `CODEX_EXECUTOR_MODEL` | `gpt-5.3-codex-spark` | Final next-action executor model. |
| `CODEX_PLANNER_REASONING_EFFORT` | `medium` | Planner reasoning effort. |
| `CODEX_EXECUTOR_REASONING_EFFORT` | `medium` | Executor reasoning effort. |
| `CODEX_TIMEOUT_SECONDS` | `180` | Per-Codex-turn timeout. |
| `CODEX_MALFORMED_RETRIES` | `1` | Retry budget for malformed JSON. |
| `CODEX_USAGE_LIMIT_RETRY_BUFFER_SECONDS` | `60` | Extra cushion after a Codex usage-limit reset time before retrying. |
| `CODEX_USAGE_LIMIT_MAX_WAIT_SECONDS` | unset | Optional cap; if the reset wait exceeds this, fail instead of sleeping. |
| `CAR_BENCH_RATE_LIMIT_REPORT_DIR` | `/tmp/car-bench-rate-limit-reports` | Directory for JSON snapshots written when Codex reports a usage limit. |
| `CAR_BENCH_A2A_TIMEOUT_SECONDS` | `86400` | HTTP timeout for long benchmark calls, including quota-reset waits. |

Participants can replace the private planning shape with their own planning
tool, planning mode, or sub-agent-style component. The important boundary is
that internal planning only uses benchmark-visible inputs and the final response
still conforms to the A2A contract. Larger Codex planner/verifier models are
allowed inside Track 2 harnesses if the complete agent stays within the official
time budget.

The intended loop is:

```text
user message -> planner -> executor -> tool call
tool result  -> executor -> tool call
tool result  -> executor -> response
```

If the executor needs extra reasoning and CAR-bench exposes `planning_tool`, it
may still return `planning_tool` as a normal benchmark-visible tool call.

## Local Run

```bash
uv run car-bench-run scenarios/track_2_agent_under_test_codex_planner/local_smoke.toml --show-logs
```

## Docker Run

```bash
uv run python generate_compose.py --scenario scenarios/track_2_agent_under_test_codex_planner/local_docker_smoke.toml
docker compose --env-file .env -f scenarios/track_2_agent_under_test_codex_planner/docker-compose.yml up --abort-on-container-exit
```

Set `CODEX_HOME_HOST` in `.env` to an absolute host path containing Codex auth.
Prefer creating a dedicated benchmark home with
`CODEX_HOME="$HOME/.codex-car-bench" codex login` instead of mounting your
everyday Codex desktop/app state.
