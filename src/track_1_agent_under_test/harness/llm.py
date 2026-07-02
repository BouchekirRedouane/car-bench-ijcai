"""Thin LiteLLM wrapper used by every layer of the harness.

* Builds completion kwargs (incl. thinking/reasoning config) consistently.
* Records token/cost/latency for every internal call into a caller-supplied
  callback, so all CoVe passes are reflected in turn_metrics.
* Returns the assistant message as a plain dict (litellm `model_dump`).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Optional

from litellm import completion

# record(prompt_tokens, completion_tokens, thinking_tokens, cost, elapsed_ms)
Recorder = Callable[[int, int, int, float, float], None]


def build_completion_kwargs(
    model: str,
    tools: Optional[list],
    temperature: Optional[float],
    thinking: bool,
    reasoning_effort: str,
    interleaved_thinking: bool,
) -> dict:
    kwargs: dict[str, Any] = {"model": model, "tools": tools if tools else None}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if thinking:
        if model == "claude-opus-4-6":
            kwargs["thinking"] = {"type": "adaptive"}
        elif reasoning_effort in ("none", "disable", "low", "medium", "high"):
            kwargs["reasoning_effort"] = reasoning_effort
        else:
            try:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(reasoning_effort)}
            except (TypeError, ValueError):
                pass
        if interleaved_thinking:
            kwargs["extra_headers"] = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
    return kwargs


def call_llm(
    messages: list,
    tools: Optional[list],
    *,
    model: str,
    temperature: Optional[float] = 0.0,
    thinking: bool = False,
    reasoning_effort: str = "medium",
    interleaved_thinking: bool = False,
    json_mode: bool = False,
    record: Optional[Recorder] = None,
) -> dict:
    """Make one completion call and return the assistant message dict."""
    kwargs = build_completion_kwargs(
        model, tools, temperature, thinking, reasoning_effort, interleaved_thinking
    )
    if json_mode:
        # JSON tasks (policy compile, teacher critic) never use tools and ask for
        # a JSON object response when the provider supports it.
        kwargs["tools"] = None
        kwargs.pop("thinking", None)
        kwargs.pop("reasoning_effort", None)
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    response = completion(messages=messages, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if record is not None:
        usage = getattr(response, "usage", None)
        pt = ct = tt = 0
        if usage:
            pt = getattr(usage, "prompt_tokens", 0) or 0
            ct = getattr(usage, "completion_tokens", 0) or 0
            details = getattr(usage, "completion_tokens_details", None)
            if details:
                tt = getattr(details, "reasoning_tokens", 0) or 0
        cost = getattr(response, "_hidden_params", {}).get("response_cost", 0.0) or 0.0
        record(pt, ct, tt, cost, elapsed_ms)

    return response.choices[0].message.model_dump(exclude_unset=True)


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_json_object(text: Optional[str]) -> dict:
    """Best-effort JSON parse that tolerates markdown fences and surrounding prose."""
    if not text:
        return {}
    text = text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the outermost {...} span.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return {}
    return {}
