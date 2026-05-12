"""Minimal Codex app-server client used by the CAR-bench purple agent.

The app-server protocol is intentionally kept behind this small wrapper so the
rest of the A2A agent only deals with "give me one next action" semantics.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import shlex
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CodexAppServerError(RuntimeError):
    """Raised when Codex app-server cannot complete a request."""


class CodexMalformedResponseError(CodexAppServerError):
    """Raised when Codex returns text that is not a valid next-action object."""


@dataclass
class CodexTurnResult:
    """Final assistant text and measured wall-clock duration for one Codex turn."""

    text: str
    duration_ms: float
    model: str | None = None
    reasoning_effort: str | None = None


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

        self._process: subprocess.Popen[str] | None = None
        self._request_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._notifications: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._initialized = False

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
            self._ensure_started()
            effective_model = model if model is not None else self.model
            effective_effort = (
                reasoning_effort
                if reasoning_effort is not None
                else self.reasoning_effort
            )
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
            completed_turn = self._wait_for_turn_completed(
                thread_id=thread_id,
                turn_id=turn["id"],
            )
            duration_ms = (time.perf_counter() - start) * 1000.0
            text = _extract_final_agent_message(completed_turn)
            if not text:
                raise CodexAppServerError(
                    "Codex completed without an assistant message. "
                    f"Turn summary: {_summarize_turn_items(completed_turn)}"
                )
            return CodexTurnResult(
                text=text,
                duration_ms=duration_ms,
                model=effective_model,
                reasoning_effort=effective_effort,
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
                    "name": "car-bench-codex-purple-agent",
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

    def _wait_for_turn_completed(self, *, thread_id: str, turn_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        completed_items: list[dict[str, Any]] = []
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

            if method != "turn/completed":
                continue
            turn = params.get("turn") or {}
            if params.get("threadId") == thread_id and turn.get("id") == turn_id:
                turn["_completed_items"] = completed_items
                if self.logger:
                    self.logger.debug(
                        "Codex turn completed",
                        status=turn.get("status"),
                        turn_items=len(turn.get("items") or []),
                        completed_items=len(completed_items),
                        item_summary=_summarize_turn_items(turn),
                    )
                if turn.get("status") == "failed":
                    error = turn.get("error") or {}
                    raise CodexAppServerError(
                        f"Codex turn failed: {error.get('message') or error}"
                    )
                return turn

        raise CodexAppServerError(
            f"Timed out waiting for Codex turn {turn_id} after {self.timeout_seconds}s."
        )

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
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
            raise CodexAppServerError(f"Codex app-server {method} error: {response['error']}")
        return response.get("result") or {}

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
