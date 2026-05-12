# Building Sophisticated Purple Agent Harnesses

This repository evaluates a **purple agent** through A2A. The green agent owns
CAR-bench: task loading, tool filtering, tool execution, simulated user turns,
trajectory recording, and rewards. A purple agent only needs to decide the next
assistant step and return it in the expected A2A shape.

## A2A Contract

Green sends one A2A message per assistant step:

- First turn: a `TextPart` containing `System: <wiki>\n\nUser: <request>` plus a
  `DataPart` containing `{"tools": [...]}` in OpenAI function-calling format.
- Tool-result turn: a `DataPart` containing `{"tool_results": [...]}`.
- User follow-up turn: a `TextPart` containing the simulated user's next message.

Purple returns one A2A message:

- User-facing response: `TextPart("...")`.
- Tool call response: `DataPart({"tool_calls": [{"tool_name": "...", "arguments": {...}}]})`.
- Optional debug reasoning: `DataPart({"reasoning_content": "..."})`.

The green wrapper converts these A2A parts back into the OpenAI-style assistant
message format expected by CAR-bench core. Do not execute vehicle tools inside
purple; doing so bypasses the benchmark.

## Harness Pattern

A robust purple harness usually has four layers:

1. **A2A parser**: Reads `TextPart` and `DataPart`, extracts the system prompt,
   user text, tool definitions, and tool results.
2. **Conversation store**: Maintains per-`context_id` history. This prevents one
   benchmark task from leaking into another.
3. **Inference adapter**: Calls your model/runtime and asks for one next action:
   either tool calls or a user-facing response.
4. **A2A renderer**: Converts the model output into `TextPart`/`DataPart` while
   attaching optional turn metrics.

The Codex implementation in `src/purple_car_bench_agent_codex/` follows this
shape and keeps Codex-specific app-server details behind `codex_client.py`.
For Codex-specific model selection and multi-pass templates, see
`docs/codex-harness-patterns.md`. A concrete planner/executor reference agent
lives in `src/purple_car_bench_agent_codex_planner/`, and a Python-call DSL
reference agent lives in `src/purple_car_bench_agent_codex_python/`.

Reference packages:

| Package | Purpose |
|---------|---------|
| `src/purple_car_bench_agent/` | Minimal LiteLLM-compatible template agent. |
| `src/purple_car_bench_agent_codex/` | Codex app-server agent returning next-action JSON. |
| `src/purple_car_bench_agent_codex_planner/` | Private planner plus Spark executor. |
| `src/purple_car_bench_agent_codex_python/` | Python-call DSL parser inspired by programmatic tool calling. |

## Important Design Rules

- Preserve tool names, parameter names, and result text exactly as green provides
  them.
- Do not add convenience tools, hidden vehicle state reads, shell commands, file
  reads, or network tools to the benchmark decision loop.
- Pass through invalid CAR-bench tool calls rather than silently repairing them
  if you want hallucination and tool-execution metrics to remain comparable.
- Keep user interaction natural text. The simulated user is not a purple-side
  tool.
- Attach latency/cost metadata only when you can measure it reliably. It is
  acceptable to report token and cost fields as zero for subscription-backed
  runtimes that do not expose usage.

## Agentic Harness Boundaries

Participants may build sophisticated purple-side harnesses, but the benchmark
boundary is the A2A exchange with green. Your harness can:

- Run multiple internal model calls before choosing the next action.
- Add a planner, critic, reranker, validator, memory layer, or policy-check pass.
- Use sub-agent-style code inside your own participant container if each internal
  component only sees benchmark-allowed inputs: the system prompt, transcript,
  tool definitions, and tool results already sent by green.
- Swap Codex for another model/runtime while preserving the same A2A output
  contract.

Your harness must not:

- Execute CAR-bench vehicle tools directly; only green executes tools.
- Inspect CAR-bench files, hidden mock data, answer keys, task definitions, or
  evaluator internals to decide the next action.
- Add private vehicle-state tools, shell commands, file reads, browser/network
  tools, or simulated-user tools to the decision loop.
- Hide tool calls from green or convert unavailable tools into available ones in
  a way that prevents hallucination metrics from scoring the behavior.
- Let an external runtime perform uncontrolled side effects that change the
  benchmark state outside the recorded A2A trajectory.

## Codex Harness

The Codex purple agent uses a warm `codex app-server` process and asks for
schema-constrained final JSON:

```json
{"action": "respond", "content": "Sure, I can help with that.", "tool_calls": []}
```

or:

```json
{
  "action": "tool_calls",
  "content": "",
  "tool_calls": [
    {"tool_name": "get_weather", "arguments_json": "{\"location\":\"Munich\"}"}
  ]
}
```

Each Codex step gets the full CAR-bench transcript and the task-filtered tool
definitions. The app-server process stays warm, but each step uses an ephemeral
Codex thread so the model-visible context is explicit and reproducible.
`arguments_json` is decoded by the adapter before returning normal A2A
`{"tool_name": "...", "arguments": {...}}` payloads to green.

This Codex harness intentionally does not expose Codex's normal coding-agent
affordances to the benchmark turn. Dynamic tools, shell commands, file changes,
permission requests, network access, and user-input requests are denied by the
adapter. Codex is used here as a constrained next-action reasoning layer, not as
an unconstrained coding workspace or hidden multi-agent system.

## Extension Ideas

- Add pre-validation that warns on unknown tool names while still passing them
  through for benchmark scoring.
- Add a reranker or policy-check pass before returning the final A2A response.
- Use a budget-gated planner/executor or ensemble/condenser pattern, reserving
  larger models for risky turns and Spark-like models for the common case.
- Use CAR-bench's `planning_tool` shape, or your own planning tool/mode, as
  private internal reasoning. Keep it private unless you intentionally want
  green to execute and record `planning_tool` as a normal benchmark tool call.
- Use a parsed Python-call DSL as an alternative action representation. It can
  be extracted from a fenced code block in model chat text, but generated code
  must be parsed rather than executed and converted back into normal A2A output.
- Swap the inference adapter for another runtime while reusing the parser,
  conversation store, renderer, and metrics code.
- Add native dynamic tools only after the JSON-output MVP is stable; if you do,
  mirror every dynamic tool call back into the `tool_calls` DataPart shape so
  CAR-bench trajectories remain comparable.
