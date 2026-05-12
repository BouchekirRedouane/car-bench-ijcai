# Building a Custom Purple Agent

This guide explains how to build your own **purple agent** (the agent under test) for CAR-bench evaluation. The purple agent communicates with the green agent (CAR-bench evaluator) via the **A2A (Agent-to-Agent) protocol**.

> **Reference implementation:** All concepts described here are demonstrated in the baseline agent code in this folder:
> - [`car_bench_agent.py`](car_bench_agent.py) — Main agent logic (`CARBenchAgentExecutor`)
> - [`tool_call_types.py`](tool_call_types.py) — `ToolCall` and `ToolCallsData` Pydantic models
> - [`server.py`](server.py) — HTTP server setup and `AgentCard` configuration
>
> For a more sophisticated external-runtime harness, see [`../../docs/purple-agent-harnessing.md`](../../docs/purple-agent-harnessing.md), [`../../docs/codex-harness-patterns.md`](../../docs/codex-harness-patterns.md), the Codex-backed package in [`../purple_car_bench_agent_codex/`](../purple_car_bench_agent_codex/), the planner/executor reference in [`../purple_car_bench_agent_codex_planner/`](../purple_car_bench_agent_codex_planner/), and the Python-call DSL reference in [`../purple_car_bench_agent_codex_python/`](../purple_car_bench_agent_codex_python/).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [A2A Message Protocol](#a2a-message-protocol)
3. [Inbound Messages — What Your Agent Receives](#inbound-messages--what-your-agent-receives)
4. [Outbound Messages — What Your Agent Should Return](#outbound-messages--what-your-agent-should-return)
5. [Conversation Lifecycle](#conversation-lifecycle)
6. [Agent Executor Contract](#agent-executor-contract)
7. [Server Setup](#server-setup)
8. [Testing Locally](#testing-locally)
9. [Key Considerations](#key-considerations)

---

## Architecture Overview

```
┌─────────────────────┐        A2A Messages         ┌─────────────────────┐
│   Green Agent       │ ◄──────────────────────────► │   Purple Agent      │
│   (CAR-bench        │    TextPart + DataPart       │   (Your Agent)      │
│    Evaluator)       │                              │                     │
└─────────────────────┘                              └─────────────────────┘
```

The green agent wraps the CAR-bench environment. It sends system prompt, available tools, user messages, and tool execution results to your purple agent. Your agent decides what to do — call tools, respond with text, or both — and sends a response back.

---

## A2A Message Protocol

All messages are exchanged as a list of **Parts**. Each Part is one of:

| Part Type    | Purpose                              | Examples                                    |
|-------------|--------------------------------------|---------------------------------------------|
| **TextPart** | Natural language content             | System prompt, user message, text responses |
| **DataPart** | Structured/machine-readable data     | Tool definitions, tool calls, reasoning     |

A single message can contain **multiple Parts** of different types. For example, a response can have a `TextPart` (explanation) and a `DataPart` (tool calls) simultaneously.

---

## Inbound Messages — What Your Agent Receives

### First Message (Task Initialization)

The first message in a conversation contains **two Parts**:

| Part | Type | Content |
|------|------|---------|
| 1    | `TextPart` | Combined system prompt and user message, formatted as: `"System: <policies and instructions>\n\nUser: <initial task>"` |
| 2    | `DataPart` | Tool definitions in `{"tools": [...]}` format (OpenAI function calling schema) |

**What each part contains:**

- **TextPart** — The `System:` section includes all 19 CAR-bench policies the agent must follow (e.g., check weather before opening sunroof, validate addresses). The `User:` section is the initial user request (e.g., "Navigate to Munich city center").

- **DataPart** — A dictionary with a `"tools"` key containing a list of tool definitions. Each tool follows the OpenAI function calling format:
  ```json
  {
    "type": "function",
    "function": {
      "name": "get_current_location",
      "description": "Get the current GPS location...",
      "parameters": { "type": "object", "properties": {...} }
    }
  }
  ```

See how the baseline agent parses this in `car_bench_agent.py`, inside the `execute()` method — it splits on `"System:"` / `"\n\nUser:"` to extract the system prompt and user message, and reads tools from the DataPart's `data["tools"]`.

### Subsequent Messages

After the first turn, each message usually contains one Part. The content depends on what happened in the previous turn:

#### Alternative A: Tool Execution Results

If your agent called tools in its previous response, the green agent executes them against the CAR-bench environment and returns the results as a **`DataPart`** with structured tool results:

```json
{
  "tool_results": [
    {
      "tool_name": "get_current_location",
      "tool_call_id": "call_abc123",
      "content": "{\"latitude\": 48.1351, \"longitude\": 11.5820, \"city\": \"Munich\"}"
    },
    {
      "tool_name": "get_weather",
      "tool_call_id": "call_def456",
      "content": "{\"temperature\": 15, \"condition\": \"sunny\"}"
    }
  ]
}
```

Each entry in `tool_results` includes the `tool_name` and `content` (the execution result), allowing your agent to match each result to the corresponding tool call from its previous response. The baseline agent matches results by `tool_name` against the previous turn's tool calls.

#### Alternative B: User Follow-up

If your agent responded with text only (no tool calls), the green agent advances the conversation and sends the next user utterance as plain text. For example:

```
Yes, please navigate there.
```

#### Edge Case: Empty Messages

Occasionally, the message may be empty or whitespace-only. The green agent replaces these with `"none"` before sending. Your agent should handle this gracefully.

---

## Outbound Messages — What Your Agent Should Return

Your agent sends its response as a list of Parts via `new_agent_parts_message()`. There are several valid response shapes:

### Option 1: Text Response Only

Return a single `TextPart` with your response text. Use this when your agent is responding directly to the user without needing to call any tools.

See the baseline agent's `execute()` method — when the LLM returns content but no tool calls, it creates a `TextPart` with the content text.

### Option 2: Tool Call(s) Only

Return a single `DataPart` containing the tool calls. Use the `ToolCallsData` model from `tool_call_types.py` to structure the data:

The DataPart's `data` field should be the `.model_dump()` of a `ToolCallsData` instance, which produces:
```json
{
  "tool_calls": [
    {"tool_name": "get_current_location", "arguments": {}},
    {"tool_name": "get_weather", "arguments": {}}
  ]
}
```

You can call **multiple tools** in a single response by adding multiple `ToolCall` entries to the list.

### Option 3: Text + Tool Call(s)

Return both a `TextPart` and a `DataPart`. The text serves as a natural language explanation of what the agent is doing, while the DataPart contains the actual tool calls.

This is the most common pattern in the baseline agent — see how it constructs the parts list in `car_bench_agent.py`.

### Optional: Reasoning Content

If your LLM produces reasoning/thinking output (e.g., Claude extended thinking), you can include it as an additional `DataPart` with `{"reasoning_content": "..."}`. The green agent will capture it for debugging but it doesn't affect evaluation.

---

## Conversation Lifecycle

```
Turn 1:  Green → Purple:  TextPart(System + User) + DataPart(tools)
         Purple → Green:  TextPart(text) + DataPart(tool_calls)

Turn 2:  Green → Purple:  DataPart(tool results)
         Purple → Green:  TextPart(text) + DataPart(tool_calls)

Turn 3:  Green → Purple:  DataPart(tool results)
         Purple → Green:  TextPart(final answer)      ← no tool calls = done

Turn 4:  Green → Purple:  TextPart(next user utterance)
         Purple → Green:  ...
```

Key points:
- The conversation continues until the environment/task is complete (managed by the green agent).
- Each `context_id` represents one independent conversation (one CAR-bench task).
- Your agent should maintain conversation state per `context_id` (see `ctx_id_to_messages` in the baseline).
- Clean up state when `cancel()` is called.

---

## Agent Executor Contract

Your agent must implement the `AgentExecutor` interface from `a2a.server.agent_execution`:

```python
class AgentExecutor:
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Process an incoming message and enqueue a response."""
        ...

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation — clean up conversation state."""
        ...
```

**Key objects:**
- `context.message` — The inbound `Message` with `.parts` (list of `Part` objects)
- `context.context_id` — Unique conversation identifier
- `event_queue.enqueue_event(response)` — Send your response back
- `new_agent_parts_message(parts=..., context_id=...)` — Helper to build the response message

See `car_bench_agent.py` for the full implementation.

---

## Server Setup

Your agent needs an HTTP server to expose it via A2A. The server setup involves:

1. **AgentCard** — Metadata describing your agent (name, skills, URL). See `prepare_agent_card()` in `server.py`.
2. **RequestHandler** — Wraps your executor. Use `DefaultRequestHandler` from `a2a.server.request_handlers`.
3. **A2AStarletteApplication** — The ASGI app. Takes the agent card and request handler.
4. **uvicorn** — Runs the ASGI app.

The server also accepts CLI arguments and environment variables for LLM configuration. See `server.py` for the full setup, including support for `--agent-llm`, `--temperature`, `--thinking`, and `--reasoning-effort` flags.

---

## Testing Locally

1. **Start your agent:**
   ```bash
   python server.py --host localhost --port 8080 --agent-llm "gemini/gemini-2.5-flash"
   ```

2. **Configure the scenario** (`scenarios/scenario.toml`):
   ```toml
   [[participants]]
   name = "agent"
   url = "http://localhost:8080"
   env = { GEMINI_API_KEY = "${GEMINI_API_KEY}" }
   ```

3. **Run evaluation** (in another terminal):
   ```bash
   uv run agentbeats-run scenarios/scenario.toml --show-logs
   ```

4. **Check results** — The green agent will report per-task pass/fail and overall metrics.

---

## Key Considerations

### Policy Compliance
The system prompt in the first message includes all 19 CAR-bench policies. Your agent must follow them to pass evaluation. Examples:
- Check weather before opening the sunroof
- Validate addresses before navigating
- Confirm actions with the user when required

You can perform prompt optimization on the system prompt, however the original policies are used for code-based and LLM-as-a-Judge evaluation (so changing the rules/logic will likely result in error).

### Tool Calling Format
- Tools are provided in **OpenAI function calling format** (see DataPart in first message)
- Return tool calls using the `ToolCallsData` model from `tool_call_types.py`
- Arguments must match the tool's parameter schema

You can edit the tool descriptions and parameter descriptions, but NOT the tool name, tool structure, and tool parameter name, tool parameter type, tool parameter structure.

### Conversation State
- Maintain message history per `context_id`
- The baseline agent uses `ctx_id_to_messages` and `ctx_id_to_tools` dicts
- Clean up in `cancel()` to avoid memory leaks

### Error Handling
- Handle missing or malformed message parts gracefully
- Return error messages as `TextPart` if something fails
- The baseline agent has a fallback using `context.get_user_input()` if part parsing fails

### LLM Flexibility
You are **not** limited to the baseline approach. You can use:
- Any LLM provider (OpenAI, Anthropic, Google, local models) or finetuned LLM
- Any framework (LangChain, LlamaIndex, etc.)
- Rule-based logic, retrieval-augmented generation, or hybrid approaches
- The only requirement is conforming to the A2A message protocol described above

Advanced harnesses may also use internal planning, validation, reranking, memory,
or sub-agent-style components. These components must stay inside the benchmark
boundary: use only the prompt, transcript, tool definitions, and tool results
sent by the green agent, then return one benchmark-compatible A2A response.
Do not execute CAR-bench tools directly, inspect hidden task/evaluator state, add
private vehicle tools, or give your runtime shell/file/network abilities that can
bypass the recorded A2A trajectory.
