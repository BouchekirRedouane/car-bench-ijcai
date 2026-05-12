# Codex Planner/Executor Purple Agent

This package is a reference implementation for a plan-on-user-turn Codex
harness:

1. A private planner call uses `gpt-5.5` to emit a compact
   `planning_tool`-shaped plan after each user message.
2. One or more Spark executor calls use `gpt-5.3-codex-spark` to return normal
   benchmark action JSON.
3. Tool-result turns reuse the active private plan until the executor responds
   to the user, then the plan is cleared.

The planner output is internal reasoning. It is not sent to green as a
CAR-bench tool call, and green remains the only component that executes vehicle
tools.

## Model Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `CODEX_PLANNER_MODEL` | `gpt-5.5` | Private planner model. |
| `CODEX_EXECUTOR_MODEL` | `gpt-5.3-codex-spark` | Final next-action executor model. |
| `CODEX_PLANNER_REASONING_EFFORT` | `medium` | Planner reasoning effort. |
| `CODEX_EXECUTOR_REASONING_EFFORT` | `medium` | Executor reasoning effort. |
| `CODEX_TIMEOUT_SECONDS` | `180` | Per-Codex-turn timeout. |
| `CODEX_MALFORMED_RETRIES` | `1` | Retry budget for malformed JSON. |

Participants can replace the private planning shape with their own planning
tool, planning mode, or sub-agent-style component. The important boundary is
that internal planning only uses benchmark-visible inputs and the final response
still conforms to the A2A contract.

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
uv run agentbeats-run scenarios/scenario-test-codex-planner.toml --show-logs
```

## Docker Run

```bash
uv run python generate_compose.py --scenario scenarios/scenario-docker-codex-planner.toml
mkdir -p output
docker compose up --abort-on-container-exit
```

Set `CODEX_HOME_HOST` in `.env` to an absolute host path containing Codex auth.
