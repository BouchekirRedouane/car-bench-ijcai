"""
CAR-bench Agent - Agent under test that solves CAR-bench tasks.

The A2A boundary (parse inbound parts, render outbound parts, maintain per-context
state, report turn_metrics) is unchanged. The single-pass LLM call has been
replaced by the v4 reliability harness (Grounded Chain-of-Verification): see
`harness/` and `harness/__init__.py`. Every harness layer is env-flag gated and
fail-safe, so with the harness disabled this behaves as the original baseline.
"""
import argparse
import json
import os
import time
from pathlib import Path
import sys
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.helpers.proto_helpers import new_message, new_text_part, new_data_part, new_task_from_user_message
from a2a.types import Role, TaskState
from google.protobuf.json_format import MessageToDict
import litellm
litellm.drop_params = True  # drop unsupported params (e.g. reasoning_effort on Nemotron)
litellm.suppress_debug_info = True  # silence cosmetic "Provider List"/bedrock log spam
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))
from logging_utils import configure_logger
from tool_call_types import ToolCall, ToolCallsData
from turn_metrics import TURN_METRICS_KEY, PROMPT_TOKENS, COMPLETION_TOKENS, COST, MODEL, THINKING_TOKENS, NUM_LLM_CALLS, AVG_LLM_CALL_TIME_MS, NUM_PASSES
# fortiss public-Ollama auth patch — install BEFORE the harness imports
# `from litellm import completion`, so the bound reference is the patched one.
import fortiss_ollama
fortiss_ollama.install()
# manual (human-as-LLM) round-trip — patch AFTER fortiss so it is outermost.
# No-op passthrough unless MANUAL_LLM is set.
import manual_llm
manual_llm.install()
sys.path.pop(0)

# Reliability harness (same directory, on sys.path via server.py).
from harness import HarnessConfig, CoVeOrchestrator, ContextState
from harness.prompts import HARNESS_SYSTEM_SUFFIX

logger = configure_logger(role="agent_under_test", context="-")

# Surface the harness's stdlib logs (policy compile, CoVe rounds, findings) so we
# can see what the teacher actually did. Level via HARNESS_LOG_LEVEL (default INFO).
import logging as _stdlog
_harness_logger = _stdlog.getLogger("harness")
if not _harness_logger.handlers:
    _h = _stdlog.StreamHandler(sys.stdout)
    _h.setFormatter(_stdlog.Formatter("HARNESS | %(name)s | %(message)s"))
    _harness_logger.addHandler(_h)
    _harness_logger.setLevel(os.getenv("HARNESS_LOG_LEVEL", "INFO").upper())
    _harness_logger.propagate = False


class CARBenchAgentExecutor(AgentExecutor):
    """Executor for the CAR-bench agent under test using native tool calling,
    wrapped in the Grounded Chain-of-Verification reliability harness."""

    def __init__(self, model: str, temperature: float = 0.0, thinking: bool = False, reasoning_effort: str = "medium", interleaved_thinking: bool = False):
        self.model = model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.interleaved_thinking = interleaved_thinking

        # Build the harness configuration from CLI args + env flags.
        self.cfg = HarnessConfig.from_env(
            model=model,
            temperature=temperature,
            thinking=thinking,
            reasoning_effort=reasoning_effort,
            interleaved_thinking=interleaved_thinking,
        )
        self.orchestrator = CoVeOrchestrator(self.cfg)
        logger.info("Harness configured", **self.cfg.summary())

        # Per-context conversation + harness state.
        self.ctx_id_to_messages: dict[str, list[dict]] = {}
        self.ctx_id_to_tools: dict[str, list[dict]] = {}
        self.ctx_id_to_turn_metrics: dict[str, dict] = {}
        self.ctx_id_to_state: dict[str, ContextState] = {}






    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        # ---- (debug) request context ------------------------------------- #
        task_id = getattr(context, 'task_id', 'N/A')
        context_id = getattr(context, 'context_id', 'N/A')
        current_task = getattr(context, 'current_task', None)
        
        try:
            user_input = context.get_user_input()
        except Exception:
            user_input = "No text payload"
        # print(
        #     f"\n\n [RED - car_bench_agent.py]  --- Starting execution loop...\n"
        #     f"--- [REQUEST CONTEXT] ---\n"
        #     f"• Task ID:     {task_id}\n"
        #     f"-------------------------\n"
        # )

        inbound_message = context.message
        ctx_logger = logger.bind(role="agent_under_test", context=f"ctx:{context.context_id[:8]}")

        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []

        messages = self.ctx_id_to_messages[context.context_id]
        tools = self.ctx_id_to_tools.get(context.context_id, [])

        # ---- parse the inbound A2A message (protobuf parts) -------------- #
        user_message_text = None
        incoming_tool_results = None

        try:
            for part in inbound_message.parts:
                content_type = part.WhichOneof("content")
                if content_type == "text":
                    text = part.text
                    if "System:" in text and "\n\nUser:" in text:
                        # First message: "System: <policy>\n\nUser: <request>"
                        parts_split = text.split("\n\nUser:", 1)
                        system_prompt = parts_split[0].replace("System:", "").strip()
                        user_message_text = parts_split[1].strip()
                        if not messages:  # add system prompt once
                            sys_content = system_prompt
                            if self.cfg.enable_system_prompt:
                                sys_content = system_prompt + "\n\n" + HARNESS_SYSTEM_SUFFIX
                            messages.append({"role": "system", "content": sys_content})
                    else:
                        user_message_text = text

                elif content_type == "data":
                    data = MessageToDict(part.data)
                    if "tools" in data:
                        tools = data["tools"]
                        self.ctx_id_to_tools[context.context_id] = tools
                    elif "tool_results" in data:
                        incoming_tool_results = data["tool_results"]

            if not user_message_text and not incoming_tool_results:
                user_message_text = context.get_user_input()

            ctx_logger.info(
                "Received user message",
                context_id=context.context_id[:8],
                turn=len(messages) + 1,
                message_preview=(user_message_text[:100] if user_message_text else
                                 f"[{len(incoming_tool_results)} tool results]" if incoming_tool_results else "")
            )
            ctx_logger.debug(
                "Message details",
                context_id=context.context_id[:8],
                message=user_message_text,
                num_parts=len(inbound_message.parts),
                has_tools=bool(tools),
                num_tools=len(tools) if tools else 0,
                has_tool_results=bool(incoming_tool_results),
                num_tool_results=len(incoming_tool_results) if incoming_tool_results else 0
            )
        except Exception as e:
            logger.warning(f"Failed to parse message parts: {e}, using fallback")
            user_message_text = context.get_user_input()

        # ---- append tool results or the user message to history ---------- #
        if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
            prev_tool_calls = messages[-1]["tool_calls"]

            if incoming_tool_results:
                tool_call_by_name = {}
                for tc in prev_tool_calls:
                    name = tc["function"]["name"]
                    tool_call_by_name.setdefault(name, []).append(tc)

                tool_results = []
                for tr in incoming_tool_results:
                    tr_name = tr.get("tool_name", "") if isinstance(tr, dict) else tr.get("toolName", "")
                    matching_calls = tool_call_by_name.get(tr_name, [])
                    if matching_calls:
                        matched_tc = matching_calls.pop(0)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": matched_tc["id"],
                            "name": tr_name,
                            "content": tr.get("content", ""),
                        })
                    else:
                        ctx_logger.warning("No matching tool_call_id for tool result", tool_name=tr_name)
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_call_id", tr.get("toolCallId", f"unknown_{tr_name}")),
                            "name": tr_name,
                            "content": tr.get("content", ""),
                        })
            else:
                tool_results = []
                for tc in prev_tool_calls:
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "content": user_message_text or "",
                    })

            messages.extend(tool_results)
            ctx_logger.debug(
                "Formatted tool results",
                num_tools=len(tool_results),
                tool_call_ids=[tr["tool_call_id"] for tr in tool_results]
            )
        else:
            messages.append({"role": "user", "content": user_message_text})

        # ---- prompt caching hints --------------------------------------- #
        # Only for Anthropic/Claude, where ephemeral caching helps and is free to
        # set. On Gemini, litellm maps cache_control to context caching, whose
        # free-tier storage quota (TotalCachedContentStorageTokensPerModelFreeTier)
        # is tiny and 429s immediately. Override with ENABLE_PROMPT_CACHE=1/0.
        _pc = os.getenv("ENABLE_PROMPT_CACHE")
        _use_cache = (_pc.strip().lower() in ("1", "true", "yes", "on")) if _pc else ("claude" in self.model.lower())
        if _use_cache:
            try:
                if tools:
                    tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                if messages:
                    messages[0]["cache_control"] = {"type": "ephemeral"}
            except Exception:
                pass

        # ---- turn metrics sink (records every internal LLM call) --------- #
        if context.context_id not in self.ctx_id_to_turn_metrics:
            self.ctx_id_to_turn_metrics[context.context_id] = {
                PROMPT_TOKENS: 0,
                COMPLETION_TOKENS: 0,
                THINKING_TOKENS: 0,
                COST: 0.0,
                NUM_LLM_CALLS: 0,
                "_total_llm_time_ms": 0.0,
            }
        turn_m = self.ctx_id_to_turn_metrics[context.context_id]

        def _record(prompt_tokens, completion_tokens, thinking_tokens, cost, elapsed_ms):
            turn_m[PROMPT_TOKENS] += prompt_tokens
            turn_m[COMPLETION_TOKENS] += completion_tokens
            turn_m[THINKING_TOKENS] += thinking_tokens
            turn_m[COST] += cost
            turn_m[NUM_LLM_CALLS] += 1
            turn_m["_total_llm_time_ms"] += elapsed_ms

        # ---- per-context harness state ---------------------------------- #
        ctx_state = self.ctx_id_to_state.get(context.context_id)
        if ctx_state is None:
            ctx_state = ContextState()
            ctx_state._context_id = context.context_id  # for trace correlation
            self.ctx_id_to_state[context.context_id] = ctx_state

        # ---- run the reliability harness for this turn ------------------ #
        num_passes = 1
        try:
            assistant_content = self.orchestrator.run_turn(
                messages, tools if tools else None, ctx_state, record=_record
            )
            num_passes = assistant_content.pop("num_passes", 1)

            tool_calls = assistant_content.get("tool_calls")
            ctx_logger.info(
                f"Harness turn complete | num_passes={num_passes} | "
                f"tool_calls={len(tool_calls) if tool_calls else 0} | "
                f"content_len={len(assistant_content.get('content') or '')}"
            )

            # Build A2A Message Parts (protobuf)
            parts = []
            if assistant_content.get("content"):
                parts.append(new_text_part(assistant_content["content"]))
            if assistant_content.get("tool_calls"):
                tool_calls_list = []
                for tc in assistant_content["tool_calls"]:
                    raw_args = tc["function"]["arguments"]
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    tool_calls_list.append(ToolCall(tool_name=tc["function"]["name"], arguments=args))
                tool_calls_data = ToolCallsData(tool_calls=tool_calls_list)
                parts.append(new_data_part(tool_calls_data.model_dump()))
            if assistant_content.get("reasoning_content"):
                parts.append(new_data_part({"reasoning_content": assistant_content["reasoning_content"]}))
            if not parts:
                parts.append(new_text_part(assistant_content.get("content", "") or ""))

        except Exception as e:
            logger.error(f"Harness/LLM error: {e}")
            parts = [new_text_part(f"Error processing request: {str(e)}")]
            assistant_content = {"content": f"Error processing request: {str(e)}"}

        # ---- commit assistant message to history ------------------------ #
        assistant_message_for_history = {
            "role": "assistant",
            "content": assistant_content.get("content"),
        }
        if assistant_content.get("tool_calls"):
            assistant_message_for_history["tool_calls"] = assistant_content["tool_calls"]
        if assistant_content.get("thinking_blocks"):
            assistant_message_for_history["thinking_blocks"] = assistant_content["thinking_blocks"]
        if assistant_content.get("reasoning_content"):
            assistant_message_for_history["reasoning_content"] = assistant_content["reasoning_content"]
        messages.append(assistant_message_for_history)

        # ---- build outbound A2A message --------------------------------- #
        response_message = new_message(
            parts=parts,
            context_id=context.context_id,
            role=Role.ROLE_AGENT,
        )

        # Attach turn_metrics on the final response of an assistant step
        # (no tool calls => the turn is complete).
        has_tool_calls = bool(assistant_content.get("tool_calls"))
        if not has_tool_calls and context.context_id in self.ctx_id_to_turn_metrics:
            turn_m = self.ctx_id_to_turn_metrics.pop(context.context_id)
            num_calls = turn_m[NUM_LLM_CALLS]
            avg_time = (turn_m["_total_llm_time_ms"] / num_calls) if num_calls > 0 else 0.0
            metrics_data = {
                PROMPT_TOKENS: turn_m[PROMPT_TOKENS],
                COMPLETION_TOKENS: turn_m[COMPLETION_TOKENS],
                COST: turn_m[COST],
                MODEL: self.model,
                THINKING_TOKENS: turn_m[THINKING_TOKENS],
                NUM_LLM_CALLS: num_calls,
                AVG_LLM_CALL_TIME_MS: round(avg_time, 1),
                NUM_PASSES: num_passes,
            }
            response_message.metadata.update({TURN_METRICS_KEY: metrics_data})
            ctx_logger.info(
                "Attached turn_metrics to final response",
                num_llm_calls=num_calls,
                num_passes=num_passes,
                avg_llm_call_time_ms=round(avg_time, 1),
                prompt_tokens=turn_m[PROMPT_TOKENS],
                completion_tokens=turn_m[COMPLETION_TOKENS],
            )

        await event_queue.enqueue_event(response_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the current execution and drop per-context state."""
        logger.bind(role="agent_under_test", context=f"ctx:{context.context_id[:8]}").info(
            "Canceling context", context_id=context.context_id[:8]
        )
        for store in (self.ctx_id_to_messages, self.ctx_id_to_tools,
                      self.ctx_id_to_turn_metrics, self.ctx_id_to_state):
            store.pop(context.context_id, None)
