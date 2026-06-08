"""Minimal Codex app-server client used by the CAR-bench agent under test.

The app-server protocol is intentionally kept behind this small wrapper so the
rest of the A2A agent only deals with "give me one next action" semantics.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


DEFAULT_USAGE_LIMIT_RETRY_BUFFER_SECONDS = 60.0
USAGE_LIMIT_CLOCK_ROLLOVER_GRACE = timedelta(minutes=5)


class CodexAppServerError(RuntimeError):
    """Raised when Codex app-server cannot complete a request."""


class CodexMalformedResponseError(CodexAppServerError):
    """Raised when Codex returns text that is not a valid next-action object."""


class CodexUsageLimitError(CodexAppServerError):
    """Raised when Codex reports a temporary model usage limit."""

    def __init__(
        self,
        message: str,
        retry_at: datetime,
        *,
        raw_error: Any | None = None,
        raw_error_source: str | None = None,
        raw_payload: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_at = retry_at
        self.raw_error = raw_error
        self.raw_error_source = raw_error_source
        self.raw_payload = raw_payload


@dataclass
class CodexTokenUsage:
    """Token usage reported by Codex app-server for one or more turns."""

    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_app_server(
        cls,
        token_usage: dict[str, Any] | None,
    ) -> "CodexTokenUsage | None":
        """Parse a `thread/tokenUsage/updated` payload.

        The app-server notification contains both `last` and `total` usage. The
        CAR-bench turn metrics should count just this Codex call, so we use
        `last` and leave aggregation to the A2A adapter.
        """

        if not isinstance(token_usage, dict):
            return None
        last = token_usage.get("last")
        if not isinstance(last, dict):
            return None
        return cls(
            input_tokens=_safe_int(last.get("inputTokens")),
            cached_input_tokens=_safe_int(last.get("cachedInputTokens")),
            output_tokens=_safe_int(last.get("outputTokens")),
            reasoning_output_tokens=_safe_int(last.get("reasoningOutputTokens")),
            total_tokens=_safe_int(last.get("totalTokens")),
        )

    def __bool__(self) -> bool:
        return any(
            (
                self.input_tokens,
                self.cached_input_tokens,
                self.output_tokens,
                self.reasoning_output_tokens,
                self.total_tokens,
            )
        )


def add_token_usage(
    left: CodexTokenUsage | None,
    right: CodexTokenUsage | None,
) -> CodexTokenUsage | None:
    """Return the sum of two optional Codex token usage records."""

    if left is None:
        return right
    if right is None:
        return left
    return CodexTokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        cached_input_tokens=left.cached_input_tokens + right.cached_input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_output_tokens=(
            left.reasoning_output_tokens + right.reasoning_output_tokens
        ),
        total_tokens=left.total_tokens + right.total_tokens,
    )


@dataclass
class CodexTurnResult:
    """Final assistant text, duration, and optional token usage for one Codex turn."""

    text: str
    duration_ms: float
    model: str | None = None
    reasoning_effort: str | None = None
    token_usage: CodexTokenUsage | None = None
    quota_wait_ms: float = 0.0


class CodexAppServerClient:
    """JSON-RPC-over-stdio client for `codex app-server`.

    The client serializes turns through a single process. That is deliberate for
    the benchmark wrapper: it keeps Codex warm while avoiding app-server protocol
    races until we have characterized quota and concurrency behavior.
    """

    def __init__(
        self,
        *,
        command: list[str] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float = 180.0,
        logger: Any | None = None,
    ) -> None:
        self.command = command or _default_app_server_command()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.cwd = str(Path(cwd or "/tmp/car-bench-codex-workdir").resolve())
        self.timeout_seconds = timeout_seconds
        self.logger = logger
        self.usage_limit_retry_buffer_seconds = _env_float(
            "CODEX_USAGE_LIMIT_RETRY_BUFFER_SECONDS",
            DEFAULT_USAGE_LIMIT_RETRY_BUFFER_SECONDS,
        )
        self.usage_limit_max_wait_seconds = _optional_env_float(
            "CODEX_USAGE_LIMIT_MAX_WAIT_SECONDS"
        )
        self.rate_limit_report_dir = Path(
            os.getenv(
                "CAR_BENCH_RATE_LIMIT_REPORT_DIR",
                "/tmp/car-bench-rate-limit-reports",
            )
        )

        self._process: subprocess.Popen[str] | None = None
        self._request_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._notifications: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._initialized = False
        self._session_started_at = datetime.now().astimezone()
        self._session_started_monotonic = time.perf_counter()
        self._total_token_usage = CodexTokenUsage()
        self._token_usage_by_model: dict[str, CodexTokenUsage] = {}
        self._successful_turns = 0
        self._successful_turns_by_model: dict[str, int] = {}
        self._previous_usage_limit_retry_at: datetime | None = None

        atexit.register(self.close)

    def generate(
        self,
        *,
        prompt: str,
        output_schema: dict[str, Any] | None,
        developer_instructions: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> CodexTurnResult:
        """Run one Codex turn in a fresh ephemeral thread and return final text."""

        with self._request_lock:
            effective_model = model if model is not None else self.model
            effective_effort = (
                reasoning_effort
                if reasoning_effort is not None
                else self.reasoning_effort
            )
            quota_wait_ms = 0.0
            quota_retries = 0

            while True:
                self._ensure_started()
                try:
                    thread = self._start_thread(
                        developer_instructions,
                        model=effective_model,
                        reasoning_effort=effective_effort,
                    )
                    thread_id = thread["id"]
                    start = time.perf_counter()
                    turn = self._start_turn(
                        thread_id=thread_id,
                        prompt=prompt,
                        output_schema=output_schema,
                        model=effective_model,
                        reasoning_effort=effective_effort,
                    )
                    completed_turn, token_usage = self._wait_for_turn_completed(
                        thread_id=thread_id,
                        turn_id=turn["id"],
                    )
                except CodexUsageLimitError as exc:
                    wait_seconds = self._usage_limit_wait_seconds(exc.retry_at)
                    max_wait = self.usage_limit_max_wait_seconds
                    quota_retries += 1
                    report_path = self._write_usage_limit_report(
                        error_message=str(exc),
                        raw_error=exc.raw_error,
                        raw_error_source=exc.raw_error_source,
                        raw_payload=exc.raw_payload,
                        retry_at=exc.retry_at,
                        wait_seconds=wait_seconds,
                        model=effective_model,
                        reasoning_effort=effective_effort,
                        prompt=prompt,
                        output_schema=output_schema,
                        quota_retries=quota_retries,
                    )
                    if max_wait is not None and wait_seconds > max_wait:
                        raise CodexAppServerError(
                            "Codex usage limit wait exceeds "
                            f"CODEX_USAGE_LIMIT_MAX_WAIT_SECONDS={max_wait}: "
                            f"{wait_seconds:.1f}s until {exc.retry_at.isoformat()}"
                        ) from exc
                    quota_wait_ms += wait_seconds * 1000.0
                    if self.logger:
                        report_path_text = str(report_path) if report_path else "<not written>"
                        self.logger.warning(
                            "Codex usage limit reached; waiting for reset. "
                            f"Report: {report_path_text}",
                            model=effective_model or "<app-server-default>",
                            reasoning_effort=effective_effort,
                            retry_at=exc.retry_at.isoformat(),
                            wait_seconds=round(wait_seconds, 1),
                            quota_retries=quota_retries,
                            report_path=str(report_path) if report_path else None,
                            codex_app_server_payload_json=(
                                _compact_json(exc.raw_payload)
                            ),
                        )
                    time.sleep(wait_seconds)
                    continue

                duration_ms = (time.perf_counter() - start) * 1000.0
                text = _extract_final_agent_message(completed_turn)
                if not text:
                    raise CodexAppServerError(
                        "Codex completed without an assistant message. "
                        f"Turn summary: {_summarize_turn_items(completed_turn)}"
                    )
                self._record_successful_turn(
                    model=effective_model,
                    token_usage=token_usage,
                )
                return CodexTurnResult(
                    text=text,
                    duration_ms=duration_ms,
                    model=effective_model,
                    reasoning_effort=effective_effort,
                    token_usage=token_usage,
                    quota_wait_ms=quota_wait_ms,
                )

    def warmup(self) -> None:
        """Start and initialize the Codex app-server process without inference."""

        with self._request_lock:
            start = time.perf_counter()
            self._ensure_started()
            if self.logger:
                self.logger.debug(
                    "Codex app-server warmup complete",
                    duration_ms=round((time.perf_counter() - start) * 1000.0, 1),
                )

    def close(self) -> None:
        proc = self._process
        self._process = None
        self._initialized = False
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        Path(self.cwd).mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        try:
            proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=env,
            )
        except FileNotFoundError as exc:
            raise CodexAppServerError(
                "Codex CLI executable was not found. Install Codex CLI in this "
                "terminal, or set CODEX_APP_SERVER_CMD to an absolute command "
                "such as '/usr/local/bin/codex app-server --listen stdio://'."
            ) from exc
        self._process = proc
        self._initialized = False
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        self._initialize()

    def _initialize(self) -> None:
        if self._initialized:
            return
        result = self._request(
            "initialize",
            {
                "clientInfo": {
                    "name": "car-bench-codex-agent-under-test",
                    "version": "0.1.0",
                },
                "capabilities": {},
            },
        )
        self._write_json({"method": "initialized"})
        self._initialized = True
        if self.logger:
            self.logger.debug(
                "Initialized Codex app-server",
                user_agent=result.get("userAgent"),
                codex_home=result.get("codexHome"),
            )

    def _start_thread(
        self,
        developer_instructions: str,
        *,
        model: str | None,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "approvalPolicy": "never",
            "baseInstructions": CODEX_BASE_INSTRUCTIONS,
            "developerInstructions": developer_instructions,
            "cwd": self.cwd,
            "sandbox": "read-only",
            "ephemeral": True,
            "personality": "none",
        }
        if model:
            params["model"] = model
        if reasoning_effort and reasoning_effort != "none":
            params["config"] = {
                "model_reasoning_effort": reasoning_effort,
            }
        result = self._request("thread/start", params)
        try:
            return result["thread"]
        except KeyError as exc:
            raise CodexAppServerError(f"Malformed thread/start response: {result}") from exc

    def _start_turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        output_schema: dict[str, Any] | None,
        model: str | None,
        reasoning_effort: str | None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": prompt}],
            "approvalPolicy": "never",
            "sandboxPolicy": {"type": "readOnly", "networkAccess": False},
            "summary": "none",
        }
        if output_schema is not None:
            params["outputSchema"] = output_schema
        if model:
            params["model"] = model
        if reasoning_effort and reasoning_effort != "none":
            params["effort"] = reasoning_effort

        result = self._request("turn/start", params)
        try:
            return result["turn"]
        except KeyError as exc:
            raise CodexAppServerError(f"Malformed turn/start response: {result}") from exc

    def _wait_for_turn_completed(
        self,
        *,
        thread_id: str,
        turn_id: str,
    ) -> tuple[dict[str, Any], CodexTokenUsage | None]:
        deadline = time.monotonic() + self.timeout_seconds
        completed_items: list[dict[str, Any]] = []
        token_usage: CodexTokenUsage | None = None
        while time.monotonic() < deadline:
            timeout = max(0.1, min(1.0, deadline - time.monotonic()))
            try:
                notification = self._notifications.get(timeout=timeout)
            except queue.Empty:
                self._raise_if_process_exited()
                continue

            method = notification.get("method")
            params = notification.get("params") or {}

            if method == "item/completed":
                if params.get("threadId") != thread_id or params.get("turnId") != turn_id:
                    continue
                item = params.get("item") or {}
                completed_items.append(item)
                if self.logger:
                    self.logger.debug(
                        "Codex item completed",
                        item_type=item.get("type"),
                        phase=item.get("phase"),
                        status=item.get("status"),
                        text_preview=(item.get("text") or "")[:200],
                )
                continue

            if method == "thread/tokenUsage/updated":
                if params.get("threadId") != thread_id or params.get("turnId") != turn_id:
                    continue
                parsed_usage = CodexTokenUsage.from_app_server(
                    params.get("tokenUsage")
                )
                if parsed_usage is not None:
                    token_usage = parsed_usage
                    if self.logger:
                        self.logger.debug(
                            "Codex token usage updated",
                            input_tokens=parsed_usage.input_tokens,
                            cached_input_tokens=parsed_usage.cached_input_tokens,
                            output_tokens=parsed_usage.output_tokens,
                            reasoning_output_tokens=(
                                parsed_usage.reasoning_output_tokens
                            ),
                            total_tokens=parsed_usage.total_tokens,
                        )
                continue

            if method != "turn/completed":
                continue
            raw_notification = _json_safe(notification)
            turn = params.get("turn") or {}
            if params.get("threadId") == thread_id and turn.get("id") == turn_id:
                turn["_completed_items"] = completed_items
                if self.logger:
                    self.logger.debug(
                        "Codex turn completed",
                        status=turn.get("status"),
                        turn_items=len(turn.get("items") or []),
                        completed_items=len(completed_items),
                        duration_ms=turn.get("durationMs"),
                        item_summary=_summarize_turn_items(turn),
                    )
                if turn.get("status") == "failed":
                    error = turn.get("error") or {}
                    error_message = (
                        error.get("message")
                        if isinstance(error, dict)
                        else str(error)
                    )
                    retry_at = _parse_usage_limit_retry_at(error_message)
                    if retry_at is not None:
                        raise CodexUsageLimitError(
                            error_message,
                            retry_at,
                            raw_error=error,
                            raw_error_source="turn.error",
                            raw_payload=raw_notification,
                        )
                    raise CodexAppServerError(
                        f"Codex turn failed: {error_message or error}"
                    )
                return turn, token_usage

        raise CodexAppServerError(
            f"Timed out waiting for Codex turn {turn_id} after {self.timeout_seconds}s."
        )

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._raise_if_process_exited()
        request_id = uuid.uuid4().hex
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        self._pending[request_id] = response_queue
        self._write_json({"id": request_id, "method": method, "params": params or {}})
        try:
            response = response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._pending.pop(request_id, None)
            raise CodexAppServerError(
                f"Timed out waiting for Codex app-server response to {method}."
            ) from exc

        if "error" in response:
            error = response["error"]
            error_message = (
                error.get("message")
                if isinstance(error, dict)
                else str(error)
            )
            retry_at = _parse_usage_limit_retry_at(error_message)
            if retry_at is not None:
                raise CodexUsageLimitError(
                    error_message,
                    retry_at,
                    raw_error=error,
                    raw_error_source="jsonrpc.error",
                    raw_payload=response,
                )
            raise CodexAppServerError(f"Codex app-server {method} error: {error}")
        return response.get("result") or {}

    def _usage_limit_wait_seconds(self, retry_at: datetime) -> float:
        target = retry_at + timedelta(seconds=self.usage_limit_retry_buffer_seconds)
        return max(0.0, (target - datetime.now().astimezone()).total_seconds())

    def _record_successful_turn(
        self,
        *,
        model: str | None,
        token_usage: CodexTokenUsage | None,
    ) -> None:
        model_key = model or "<app-server-default>"
        self._successful_turns += 1
        self._successful_turns_by_model[model_key] = (
            self._successful_turns_by_model.get(model_key, 0) + 1
        )
        if token_usage is None:
            return
        self._total_token_usage = add_token_usage(
            self._total_token_usage,
            token_usage,
        ) or CodexTokenUsage()
        self._token_usage_by_model[model_key] = add_token_usage(
            self._token_usage_by_model.get(model_key),
            token_usage,
        ) or CodexTokenUsage()

    def _write_usage_limit_report(
        self,
        *,
        error_message: str,
        raw_error: Any | None = None,
        raw_error_source: str | None = None,
        raw_payload: Any | None = None,
        retry_at: datetime,
        wait_seconds: float,
        model: str | None,
        reasoning_effort: str | None,
        prompt: str,
        output_schema: dict[str, Any] | None,
        quota_retries: int,
    ) -> Path | None:
        created_at = datetime.now().astimezone()
        model_key = model or "<app-server-default>"
        previous_retry_at = self._previous_usage_limit_retry_at
        retry_with_buffer_at = created_at + timedelta(seconds=wait_seconds)
        wall_time_since_previous_retry_at = (
            max(0.0, (created_at - previous_retry_at).total_seconds())
            if previous_retry_at is not None
            else None
        )
        payload = {
            "schema_version": 1,
            "event": "codex_usage_limit",
            "created_at": created_at.isoformat(),
            "session_started_at": self._session_started_at.isoformat(),
            "wall_time_until_rate_limit_seconds": round(
                time.perf_counter() - self._session_started_monotonic,
                3,
            ),
            "previous_retry_at": (
                previous_retry_at.isoformat()
                if previous_retry_at is not None
                else None
            ),
            "wall_time_since_previous_retry_at_seconds": (
                round(wall_time_since_previous_retry_at, 3)
                if wall_time_since_previous_retry_at is not None
                else None
            ),
            "retry_at": retry_at.isoformat(),
            "retry_with_buffer_at": retry_with_buffer_at.isoformat(),
            "wait_seconds": round(wait_seconds, 3),
            "model": model_key,
            "reasoning_effort": reasoning_effort,
            "quota_retries_in_current_call": quota_retries,
            "successful_codex_calls": self._successful_turns,
            "successful_codex_calls_by_model": dict(self._successful_turns_by_model),
            "tokens_consumed": _token_usage_to_dict(self._total_token_usage),
            "tokens_consumed_by_model": {
                key: _token_usage_to_dict(value)
                for key, value in sorted(self._token_usage_by_model.items())
            },
            "current_call": {
                "prompt_chars": len(prompt),
                "has_output_schema": output_schema is not None,
                "output_schema_name": (
                    output_schema.get("name")
                    if isinstance(output_schema, dict)
                    else None
                ),
            },
            "error_message": error_message,
            "raw_error_source": raw_error_source,
            "raw_error": _json_safe(raw_error),
            "raw_payload": _json_safe(raw_payload),
        }

        timestamp = created_at.strftime("%Y%m%d-%H%M%S")
        filename = (
            f"codex-rate-limit-{timestamp}-"
            f"{uuid.uuid4().hex[:8]}.json"
        )
        try:
            self.rate_limit_report_dir.mkdir(parents=True, exist_ok=True)
            path = self.rate_limit_report_dir / filename
            path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            self._previous_usage_limit_retry_at = retry_at
            return path
        except OSError as exc:
            if self.logger:
                self.logger.warning(
                    "Failed to write Codex usage-limit report",
                    report_dir=str(self.rate_limit_report_dir),
                    error=str(exc),
                )
            self._previous_usage_limit_retry_at = retry_at
            return None

    def _write_json(self, payload: dict[str, Any]) -> None:
        proc = self._process
        if proc is None or proc.stdin is None:
            raise CodexAppServerError("Codex app-server is not running.")
        line = json.dumps(payload, separators=(",", ":"))
        with self._write_lock:
            proc.stdin.write(line + "\n")
            proc.stdin.flush()

    def _read_stdout(self) -> None:
        proc = self._process
        if proc is None or proc.stdout is None:
            return
        try:
            for raw_line in proc.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    if self.logger:
                        self.logger.debug("Ignoring non-JSON app-server stdout", line=line[:200])
                    continue

                if "id" in message and "method" not in message:
                    pending = self._pending.pop(str(message["id"]), None)
                    if pending is not None:
                        pending.put(message)
                    continue

                if "id" in message and "method" in message:
                    self._handle_server_request(message)
                    continue

                if "method" in message:
                    self._notifications.put(message)
        finally:
            message = self._process_exit_message()
            for pending in list(self._pending.values()):
                pending.put(
                    {
                        "error": {
                            "code": -32000,
                            "message": message,
                        }
                    }
                )
            self._pending.clear()

    def _drain_stderr(self) -> None:
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        for raw_line in proc.stderr:
            line = raw_line.strip()
            if not line:
                continue
            self._stderr_tail.append(line)
            if self.logger:
                self.logger.debug("Codex app-server stderr", line=line[:500])

    def _handle_server_request(self, request: dict[str, Any]) -> None:
        method = request.get("method")
        request_id = request.get("id")
        if method == "item/tool/call":
            result = {
                "success": False,
                "contentItems": [
                    {
                        "type": "inputText",
                        "text": "Dynamic tools are disabled for CAR-bench MVP runs.",
                    }
                ],
            }
        elif method == "item/tool/requestUserInput":
            result = {"answers": {}}
        elif method in {
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
        }:
            result = {"decision": "decline"}
        elif method == "item/permissions/requestApproval":
            result = {
                "permissions": {
                    "fileSystem": None,
                    "network": {"enabled": False},
                },
                "scope": "turn",
            }
        elif method in {"execCommandApproval", "applyPatchApproval"}:
            result = {"decision": "denied"}
        else:
            self._write_json(
                {
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unsupported app-server request: {method}",
                    },
                }
            )
            return
        self._write_json({"id": request_id, "result": result})

    def _raise_if_process_exited(self) -> None:
        proc = self._process
        if proc is not None and proc.poll() is not None:
            raise CodexAppServerError(self._process_exit_message())

    def _process_exit_message(self) -> str:
        proc = self._process
        status = proc.poll() if proc is not None else "unknown"
        tail = "\n".join(self._stderr_tail)
        message = f"Codex app-server exited with status {status}."
        if tail:
            message += f" Recent stderr:\n{tail}"
        return message


def _default_app_server_command() -> list[str]:
    raw = os.getenv("CODEX_APP_SERVER_CMD")
    if raw:
        return shlex.split(raw)
    return ["codex", "app-server", "--listen", "stdio://"]


def _extract_final_agent_message(turn: dict[str, Any]) -> str:
    items = (turn.get("items") or []) + (turn.get("_completed_items") or [])
    agent_messages = [
        item
        for item in items
        if item.get("type") == "agentMessage" and isinstance(item.get("text"), str)
    ]
    for item in reversed(agent_messages):
        if item.get("phase") == "final_answer":
            return item["text"].strip()
    if agent_messages:
        return agent_messages[-1]["text"].strip()
    return ""


def _parse_usage_limit_retry_at(
    message: str | None,
    *,
    now: datetime | None = None,
) -> datetime | None:
    if not message:
        return None
    lower = message.lower()
    if "usage limit" not in lower and "rate limit" not in lower:
        return None

    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        current = current.astimezone()

    relative_match = re.search(
        r"try again in\s+(\d+(?:\.\d+)?)\s*(second|minute|hour)s?\b",
        message,
        flags=re.IGNORECASE,
    )
    if relative_match:
        amount = float(relative_match.group(1))
        unit = relative_match.group(2).lower()
        if unit == "second":
            return current + timedelta(seconds=amount)
        if unit == "minute":
            return current + timedelta(minutes=amount)
        if unit == "hour":
            return current + timedelta(hours=amount)

    clock_match = re.search(
        r"try again at\s+(\d{1,2})(?::(\d{2}))?\s*(AM|PM)\b",
        message,
        flags=re.IGNORECASE,
    )
    if not clock_match:
        return None

    hour = int(clock_match.group(1))
    minute = int(clock_match.group(2) or 0)
    meridiem = clock_match.group(3).upper()
    if hour < 1 or hour > 12 or minute > 59:
        return None
    if meridiem == "PM" and hour != 12:
        hour += 12
    elif meridiem == "AM" and hour == 12:
        hour = 0

    retry_at = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if retry_at <= current:
        if current - retry_at <= USAGE_LIMIT_CLOCK_ROLLOVER_GRACE:
            return retry_at
        retry_at += timedelta(days=1)
    return retry_at


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _optional_env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _token_usage_to_dict(usage: CodexTokenUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "cached_input_tokens": usage.cached_input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_output_tokens": usage.reasoning_output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return repr(value)


def _compact_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(
        _json_safe(value),
        separators=(",", ":"),
        sort_keys=True,
    )


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _summarize_turn_items(turn: dict[str, Any]) -> list[dict[str, Any]]:
    items = (turn.get("items") or []) + (turn.get("_completed_items") or [])
    summary = []
    for item in items:
        text = item.get("text")
        summary.append(
            {
                "type": item.get("type"),
                "phase": item.get("phase"),
                "status": item.get("status"),
                "text_preview": text[:160] if isinstance(text, str) else None,
            }
        )
    return summary


CODEX_BASE_INSTRUCTIONS = """You are an in-car assistant reasoning layer.
You are not a coding agent for this task. Never inspect files, run shell
commands, edit files, browse the network, or mention Codex. Use only the
supplied CAR-bench tool definitions. Follow the requested output contract
exactly."""
