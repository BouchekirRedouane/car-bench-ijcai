"""Structured trace recorder.

Every assistant turn (student draft, per-gate findings, each revision, final
action, pass count) is appended as one JSON line so runs can be analysed offline
without scraping stdout. Compiled policy rules are dumped once per distinct
policy.

Location: $HARNESS_TRACE_DIR, else <cwd>/output/harness_traces/ (git-ignored).
Disable with HARNESS_TRACE=0. Fully fail-safe — tracing never breaks a turn.
"""
from __future__ import annotations

import json
import os
import threading
import time

_lock = threading.Lock()
_RUN_TAG = None  # stable-ish per process (no Date dependency issues here)


def enabled() -> bool:
    return (os.getenv("HARNESS_TRACE", "1").strip().lower() not in ("0", "false", "no", "off"))


def trace_dir() -> str:
    d = os.getenv("HARNESS_TRACE_DIR") or os.path.join(os.getcwd(), "output", "harness_traces")
    os.makedirs(d, exist_ok=True)
    return d


def _run_tag() -> str:
    global _RUN_TAG
    if _RUN_TAG is None:
        _RUN_TAG = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    return _RUN_TAG


def record_rules(rules: list) -> None:
    if not enabled():
        return
    try:
        path = os.path.join(trace_dir(), f"policy_rules_{_run_tag()}.json")
        with _lock, open(path, "w", encoding="utf-8") as f:
            json.dump({"count": len(rules), "rules": rules}, f, indent=2, default=str)
    except Exception:
        pass


def record_turn(record: dict) -> None:
    if not enabled():
        return
    try:
        record = {"ts": time.time(), "run": _run_tag(), **record}
        path = os.path.join(trace_dir(), f"turns_{_run_tag()}.jsonl")
        with _lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
