"""Python-call DSL variant of the Codex CAR-bench agent under test.

Spark can answer in its natural chat shape, including a short explanation and a
fenced Python block such as ``open_close_sunshade(percentage=50)``. The adapter
extracts that block, parses it with ``ast``, and converts it into the same
next-action dictionary used by the direct JSON Codex agent. The generated code
is never executed.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from track_2_agent_under_test_codex.car_bench_agent import (
    AgentInferenceResult,
    CARBenchAgentExecutor as CodexNextActionExecutor,
    build_codex_prompt,
)
from track_2_agent_under_test_codex.codex_client import (
    CodexAppServerError,
    CodexMalformedResponseError,
    CodexTokenUsage,
    add_token_usage,
)
sys.path.pop(0)


class PythonCallCARBenchAgentExecutor(CodexNextActionExecutor):
    """A2A executor that asks Codex for Python-call DSL next actions."""

    def _call_codex_with_retries(
        self,
        *,
        context_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        ctx_logger,
    ) -> AgentInferenceResult:
        last_error: Exception | None = None
        correction = None
        total_duration_ms = 0.0
        total_quota_wait_ms = 0.0
        total_token_usage: CodexTokenUsage | None = None
        internal_calls = 0
        for attempt in range(self.malformed_retries + 1):
            prompt = build_python_call_prompt(
                messages=messages,
                tools=tools,
                correction=correction,
            )
            ctx_logger.debug(
                "Calling Codex Python-call agent",
                attempt=attempt + 1,
                model=self.model or "<app-server-default>",
                reasoning_effort=self.reasoning_effort,
                num_messages=len(messages),
                num_tools=len(tools),
                prompt_chars=len(prompt),
                tool_names=[
                    tool.get("function", {}).get("name", "<unknown>")
                    for tool in tools[:10]
                ],
            )
            try:
                result = self.client.generate(
                    prompt=prompt,
                    output_schema=None,
                    developer_instructions=PYTHON_CALL_DEVELOPER_INSTRUCTIONS,
                )
                internal_calls += 1
                total_duration_ms += result.duration_ms
                total_quota_wait_ms += result.quota_wait_ms
                total_token_usage = add_token_usage(
                    total_token_usage,
                    result.token_usage,
                )
                parsed = parse_python_next_action_output(result.text)
                ctx_logger.debug(
                    "Parsed Codex Python-call next action",
                    raw_preview=result.text[:500],
                    parsed=parsed,
                )
                ctx_logger.info(
                    "Codex Python-call response received",
                    action=parsed["action"],
                    num_tool_calls=len(parsed.get("tool_calls") or []),
                    model=result.model or self.model or "<app-server-default>",
                    reasoning_effort=result.reasoning_effort or self.reasoning_effort,
                    inference_ms=round(result.duration_ms, 1),
                    total_inference_ms=round(total_duration_ms, 1),
                    quota_wait_ms=round(total_quota_wait_ms, 1),
                    input_tokens=(
                        total_token_usage.input_tokens
                        if total_token_usage is not None
                        else 0
                    ),
                    cached_input_tokens=(
                        total_token_usage.cached_input_tokens
                        if total_token_usage is not None
                        else 0
                    ),
                    output_tokens=(
                        total_token_usage.output_tokens
                        if total_token_usage is not None
                        else 0
                    ),
                    reasoning_tokens=(
                        total_token_usage.reasoning_output_tokens
                        if total_token_usage is not None
                        else 0
                    ),
                    attempt=attempt + 1,
                )
                return AgentInferenceResult(
                    next_action=parsed,
                    elapsed_ms=total_duration_ms,
                    token_usage=total_token_usage,
                    internal_calls=max(internal_calls, 1),
                    quota_wait_ms=total_quota_wait_ms,
                )
            except (
                CodexMalformedResponseError,
                SyntaxError,
                ValueError,
                json.JSONDecodeError,
            ) as e:
                last_error = e
                correction = (
                    "The previous Codex output was invalid. Return any brief "
                    "private note you need, then exactly one fenced ```python "
                    "code block containing only top-level direct calls. "
                    f"Error: {e}"
                )
                ctx_logger.warning(
                    "Malformed Codex Python-call response",
                    attempt=attempt + 1,
                    retrying=attempt < self.malformed_retries,
                    error=str(e),
                )
            except CodexAppServerError:
                raise

        raise CodexMalformedResponseError(
            f"Codex did not produce valid Python-call next action: {last_error}"
        )


def build_python_call_prompt(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    correction: str | None = None,
) -> str:
    payload = json.loads(
        build_codex_prompt(messages=messages, tools=tools, correction=correction)
    )
    payload["task"] = (
        "Choose exactly one next assistant action for this CAR-bench turn and "
        "represent it as a tiny Python-call program."
    )
    payload["output_contract"] = {
        "format": (
            "Return ordinary assistant text with exactly one fenced code block "
            "labeled python. The adapter parses only that code block."
        ),
        "respond": (
            'To speak to the user, the code block must contain one call: '
            'respond("short user-facing message"). Use this only when no tool '
            "call is needed right now, or after the transcript already contains "
            "the relevant tool result."
        ),
        "tool_calls": (
            "To call tools, the code block must contain one or more top-level "
            "calls named exactly like supplied CAR-bench tools, for example "
            "open_close_sunshade(percentage=50). Do not include respond(...) in "
            "the same code block; evaluator will send the tool result back, and you "
            "can respond on the following assistant turn."
        ),
    }
    payload["python_call_rules"] = [
        "You may include a very brief private note before the code block if helpful.",
        "End with exactly one fenced ```python code block.",
        "The code block is the only benchmark action; prose outside it is ignored.",
        "Generate only direct function calls inside the code block.",
        "Choose either respond(...) or tool calls, never both in the same code block.",
        "Use respond(...) for a user-facing response only when no tool call is needed now.",
        "If you call any tool, stop after the tool calls and wait for the tool result.",
        "Use CAR-bench tool names as function names for tool calls.",
        "Use keyword arguments only for tool calls.",
        "Arguments must be Python literals: strings, numbers, booleans, None, lists, or dicts.",
        "Do not use imports, variables, assignments, attributes, loops, conditionals, "
        "comprehensions, or helper functions.",
        "Do not call tools yourself; the adapter parses these calls and evaluator executes tools.",
        "Do not use JSON for the final action.",
    ]
    payload["python_call_examples"] = [
        '```python\nrespond("Sure, what percentage should I set it to?")\n```',
        "```python\nopen_close_sunshade(percentage=50)\n```",
        (
            "```python\n"
            'get_user_preferences(preference_categories={"vehicle_settings": {"vehicle_settings": True}})\n'
            "open_close_sunshade(percentage=50)\n"
            "```"
        ),
    ]
    payload["python_call_invalid_examples"] = [
        (
            "Do not combine a tool call with a final response before the tool "
            "result arrives: "
            '```python\nopen_close_sunshade(percentage=50)\nrespond("Done.")\n```'
        )
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_python_next_action_output(text: str) -> dict[str, Any]:
    python_code = _load_legacy_json_python_code(text)
    if python_code is None:
        python_code = extract_python_call_code(text)
    return parse_python_calls(python_code)


def extract_python_call_code(text: str) -> str:
    """Extract the single Python-call action block from Codex chat text.

    Codex app-server returns markdown proposals as ordinary agentMessage text.
    The harness treats prose outside the code block as private commentary and
    parses only one final Python block back into A2A actions.
    """

    if not isinstance(text, str) or not text.strip():
        raise CodexMalformedResponseError("Codex output must be non-empty text")

    fenced_blocks = _find_fenced_code_blocks(text)
    python_blocks = [
        block
        for block in fenced_blocks
        if block["language"] in {"python", "py"}
    ]
    if len(python_blocks) > 1:
        raise CodexMalformedResponseError(
            "Codex output must contain exactly one fenced Python action block"
        )
    if python_blocks:
        return python_blocks[0]["code"].strip()

    unlabeled_blocks = [block for block in fenced_blocks if not block["language"]]
    if len(unlabeled_blocks) == 1 and len(fenced_blocks) == 1:
        return unlabeled_blocks[0]["code"].strip()
    if fenced_blocks:
        raise CodexMalformedResponseError(
            "Codex output must contain a fenced Python action block"
        )

    return text.strip()


def parse_python_calls(python_code: str) -> dict[str, Any]:
    try:
        module = ast.parse(python_code.strip(), mode="exec")
    except SyntaxError as exc:
        raise CodexMalformedResponseError(
            f"python_code is not valid Python: {exc}"
        ) from exc

    if not module.body:
        raise CodexMalformedResponseError("python_code must contain at least one call")

    calls = []
    for node in module.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            raise CodexMalformedResponseError(
                "python_code may contain only top-level direct function calls"
            )
        calls.append(_parse_call(node.value))

    respond_calls = [call for call in calls if call["kind"] == "respond"]
    tool_calls = [call for call in calls if call["kind"] == "tool"]
    if respond_calls and tool_calls:
        # Codex sometimes writes the right tool call and a premature user-facing
        # "done" response in the same block. CAR-bench needs the tool action now;
        # evaluator will call us again after the tool result so we can respond then.
        respond_calls = []
    if len(respond_calls) > 1:
        raise CodexMalformedResponseError("python_code may contain only one respond call")
    if respond_calls:
        return {"action": "respond", "content": respond_calls[0]["content"]}
    if not tool_calls:
        raise CodexMalformedResponseError("python_code did not contain a valid action")
    return {
        "action": "tool_calls",
        "tool_calls": [
            {
                "tool_name": call["tool_name"],
                "arguments": call["arguments"],
            }
            for call in tool_calls
        ],
    }


def _parse_call(call: ast.Call) -> dict[str, Any]:
    if not isinstance(call.func, ast.Name):
        raise CodexMalformedResponseError("calls must use simple function names")
    name = call.func.id
    if name == "respond":
        if len(call.args) != 1 or call.keywords:
            raise CodexMalformedResponseError("respond requires exactly one string argument")
        content = _literal_from_node(call.args[0])
        if not isinstance(content, str):
            raise CodexMalformedResponseError("respond argument must be a string")
        return {"kind": "respond", "content": content}

    if call.args:
        raise CodexMalformedResponseError("tool calls must use keyword arguments only")
    arguments: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise CodexMalformedResponseError("starred keyword arguments are not allowed")
        arguments[keyword.arg] = _literal_from_node(keyword.value)
    return {"kind": "tool", "tool_name": name, "arguments": arguments}


def _literal_from_node(node: ast.AST) -> Any:
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError) as exc:
        raise CodexMalformedResponseError(
            "arguments must be Python literals, not expressions"
        ) from exc
    _validate_json_like_literal(value)
    return value


def _validate_json_like_literal(value: Any) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_like_literal(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CodexMalformedResponseError("dict argument keys must be strings")
            _validate_json_like_literal(item)
        return
    raise CodexMalformedResponseError(
        f"unsupported literal type in argument: {type(value).__name__}"
    )


def _load_legacy_json_python_code(text: str) -> str | None:
    """Accept the older structured JSON envelope for backwards compatibility."""

    candidates = [text.strip()]
    candidates.extend(
        block["code"].strip()
        for block in _find_fenced_code_blocks(text)
        if block["language"] == "json"
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "python_code" in payload:
            python_code = payload["python_code"]
            if not isinstance(python_code, str) or not python_code.strip():
                raise CodexMalformedResponseError(
                    "python_code must be a non-empty string"
                )
            return python_code
    return None


def _find_fenced_code_blocks(text: str) -> list[dict[str, str]]:
    blocks = []
    for match in FENCED_CODE_BLOCK_RE.finditer(text):
        info = match.group("info").strip()
        language = info.split(None, 1)[0].lower() if info else ""
        blocks.append({"language": language, "code": match.group("code")})
    return blocks


PYTHON_CALL_DEVELOPER_INSTRUCTIONS = """You are an in-car assistant reasoning layer for CAR-bench.
Never inspect files, run shell commands, edit files, browse the network, or mention Codex.
Return ordinary chat text plus exactly one final fenced ```python code block.
The fenced code block must contain only top-level direct Python function calls.
Use respond("...") for a user-facing response.
Use only supplied CAR-bench tool names as function names for tool calls.
Use keyword arguments only for tool calls.
Arguments must be Python literals: strings, numbers, booleans, None, lists, or dicts.
Never use imports, variables, assignments, attributes, loops, conditionals, comprehensions, or helper functions.
Do not execute tools yourself; evaluator executes any parsed tool calls.
Respect confirmation and disambiguation policy from the wiki/system prompt."""


FENCED_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*(?P<info>[^\n`]*)\n(?P<code>.*?)```",
    re.DOTALL,
)
